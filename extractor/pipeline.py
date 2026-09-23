"""
Document extraction pipeline.

1. Classify each page as native text, scanned, garbled, or empty.
2. Native pages: PyMuPDF dict extraction (font + bbox metadata).
3. Scanned/garbled pages: Tesseract OCR at modest DPI, full page,
   up to 45 seconds per page (at most 3 OCR pages).
4. Always extract the first PRIMARY_PAGE_COUNT pages. If that top-of-page
   text is too thin, extract up to MAX_SEARCH_PAGES, then make one Gemini
   call. Do not call Gemini twice for the same PDF.
5. Rotate Gemini API keys across documents. The layout heuristic runs
   only after every key is out of daily quota.
"""

from __future__ import annotations

import time
from typing import Any

from extractor.gemini_title import (
    SPARSE_FRONT_CHARS,
    apply_gemini_layer,
    front_text_char_count,
    gemini_is_configured,
)
from extractor.heuristic import detect_title
from extractor.ocr import OcrUnavailableError, ocr_page, ocr_status
from extractor.pdf_utils import (
    classify_page,
    extract_page_spans,
    extract_usable_native_spans,
    open_pdf,
    reconstruct_text,
    spans_to_lines,
)

PRIMARY_PAGE_COUNT = 4
MAX_SEARCH_PAGES = 7
OCR_BUDGET_SEC = 45.0
MAX_OCR_PAGES = 3
MIN_OCR_TIMEOUT_SEC = 20.0


def extract_document(
    pdf_path: str,
    max_pages: int | None = None,
    filename: str | None = None,
    use_ocr: bool = True,
    primary_pages: int = PRIMARY_PAGE_COUNT,
    search_pages: int = MAX_SEARCH_PAGES,
    use_gemini: bool = True,
) -> dict[str, Any]:
    doc = open_pdf(pdf_path)
    try:
        page_reports: list[dict[str, Any]] = []
        spans: list[dict[str, Any]] = []
        warnings: list[str] = []
        page_geom: dict[int, tuple[float, float]] = {}
        metadata = doc.metadata or {}
        document_page_count = doc.page_count
        limit = document_page_count if max_pages is None else min(max_pages, document_page_count)
        first_pass = min(max(primary_pages, 1), limit)
        search_limit = min(max(search_pages, first_pass), limit)

        ocr_ctx = {
            "budget": OCR_BUDGET_SEC,
            "used": 0.0,
            "pages": 0,
            "max_pages": MAX_OCR_PAGES,
        }
        _extract_page_range(
            doc,
            range(first_pass),
            use_ocr,
            spans,
            page_reports,
            page_geom,
            warnings,
            ocr_ctx,
        )

        lines = _lines_with_geom(spans, page_geom)
        pages_extended = False
        if (
            use_gemini
            and search_limit > first_pass
            and front_text_char_count(lines) < SPARSE_FRONT_CHARS
        ):
            _extract_page_range(
                doc,
                range(first_pass, search_limit),
                use_ocr,
                spans,
                page_reports,
                page_geom,
                warnings,
                ocr_ctx,
            )
            pages_extended = True
            lines = _lines_with_geom(spans, page_geom)
            warnings.append(
                f"First {first_pass} page(s) had little title-area text; "
                f"included pages {first_pass + 1}–{len(page_reports)} in one Gemini call."
            )

        title_info = _blank_pipeline_title(page_reports)
        if use_gemini:
            title_info = apply_gemini_layer(lines, warnings, page_reports)
            if title_info.get("quota_exhausted") or _gemini_blocked_by_quota(title_info):
                title_info = _heuristic_quota_fallback(
                    title_info, lines, metadata, filename, warnings
                )
        pdf_type = _document_type(page_reports)
        ocr_info = ocr_status()
        reason = title_info.get("reason") or ""
        if pages_extended:
            reason = (
                f"{reason} Used pages 1–{len(page_reports)} in a single Gemini call."
            ).strip()
        elif document_page_count > len(page_reports):
            reason = (
                f"{reason} Used the first {len(page_reports)} page(s); later pages were not needed."
            ).strip()

        return {
            "pdf_type": pdf_type,
            "page_count": len(page_reports),
            "document_page_count": document_page_count,
            "pages_extended": pages_extended,
            "span_count": len(spans),
            "line_count": len(lines),
            "ocr_available": ocr_info["available"],
            "ocr_reason": ocr_info["reason"],
            "pages": page_reports,
            "page_char_counts": title_info.get("page_char_counts") or _page_char_counts(page_reports),
            "front_text_chars": title_info.get("front_text_chars")
            if title_info.get("front_text_chars") is not None
            else sum(item["char_count"] for item in page_reports),
            "lines": lines,
            "spans": spans,
            "text": reconstruct_text(lines),
            "warnings": warnings,
            "metadata_title": (metadata.get("title") or "").strip() or None,
            "title": title_info.get("title"),
            "title_source": title_info.get("source"),
            "title_confidence": title_info.get("confidence"),
            "title_score": title_info.get("score"),
            "title_page": title_info.get("page"),
            "title_reason": reason,
            "title_signals": title_info.get("signals") or {},
            "title_alternatives": title_info.get("alternatives") or [],
            "rejected_metadata": title_info.get("rejected_metadata"),
            "title_confidence_01": title_info.get("title_confidence_01"),
            "gemini_used": bool(title_info.get("gemini_used")),
            "gemini_mode": title_info.get("gemini_mode") or "skip",
            "gemini_configured": gemini_is_configured(),
            "gemini_input_text": title_info.get("gemini_input_text"),
            "gemini_input_chars": title_info.get("gemini_input_chars") or 0,
            "gemini_candidate": title_info.get("gemini_candidate"),
            "gemini_title": title_info.get("gemini_title"),
            "gemini_is_correct": title_info.get("gemini_is_correct"),
            "gemini_stats": title_info.get("gemini_stats") or {},
            "gemini_key_count": title_info.get("gemini_key_count") or 0,
            "gemini_keys_remaining": title_info.get("gemini_keys_remaining") or 0,
            "gemini_key_used": title_info.get("gemini_key_used"),
            "quota_exhausted": bool(title_info.get("quota_exhausted")),
        }
    finally:
        doc.close()


def _gemini_blocked_by_quota(title_info: dict[str, Any]) -> bool:
    reason = (title_info.get("reason") or "").lower()
    if title_info.get("gemini_mode") not in {"unavailable", "error", "quota_exhausted"}:
        return False
    return any(
        marker in reason
        for marker in ("quota", "rate limit", "resource exhausted", "resource_exhausted")
    )


def _heuristic_quota_fallback(
    gemini_info: dict[str, Any],
    lines: list[dict[str, Any]],
    metadata: dict[str, Any],
    filename: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Last resort: layout heuristic, only after every Gemini key is out of quota."""
    warnings.append("All Gemini API keys are out of quota; using heuristic fallback.")
    heuristic = detect_title(
        lines,
        metadata_title=metadata.get("title"),
        filename=filename,
        metadata_author=metadata.get("author"),
    )
    merged = dict(gemini_info)
    merged["title"] = heuristic.get("title")
    merged["source"] = heuristic.get("source") or "heuristic"
    merged["confidence"] = heuristic.get("confidence")
    merged["score"] = heuristic.get("score")
    merged["page"] = heuristic.get("page")
    merged["signals"] = heuristic.get("signals") or {}
    merged["alternatives"] = heuristic.get("alternatives") or []
    merged["rejected_metadata"] = heuristic.get("rejected_metadata")
    merged["title_confidence_01"] = heuristic.get("title_confidence_01")
    extra = "All Gemini keys were quota-exhausted; used heuristic as the last fallback."
    heuristic_reason = (heuristic.get("reason") or "").strip()
    merged["reason"] = f"{extra} {heuristic_reason}".strip()
    merged["gemini_mode"] = "quota_exhausted"
    merged["gemini_used"] = False
    merged["quota_exhausted"] = True
    return merged


def _has_gemini_title(title_info: dict[str, Any] | None) -> bool:
    if not title_info:
        return False
    title = (title_info.get("title") or title_info.get("gemini_title") or "").strip()
    if not title:
        return False
    return title_info.get("source") == "gemini" or bool(title_info.get("gemini_used"))


def _page_char_counts(page_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "page": item["page"],
            "char_count": item["char_count"],
            "kind": item["kind"],
            "method": item["method"],
        }
        for item in page_reports
    ]


def _blank_pipeline_title(page_reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _page_char_counts(page_reports)
    return {
        "title": None,
        "source": None,
        "confidence": "low",
        "score": None,
        "page": None,
        "reason": "Gemini was not requested.",
        "signals": {},
        "alternatives": [],
        "rejected_metadata": None,
        "page_char_counts": counts,
        "front_text_chars": sum(item["char_count"] for item in counts),
        "gemini_input_chars": 0,
    }


def _extract_page_range(
    doc: Any,
    indexes: range,
    use_ocr: bool,
    spans: list[dict[str, Any]],
    page_reports: list[dict[str, Any]],
    page_geom: dict[int, tuple[float, float]],
    warnings: list[str],
    ocr_ctx: dict[str, Any],
) -> None:
    for page_index in indexes:
        _extract_page(
            doc[page_index],
            page_index + 1,
            use_ocr,
            spans,
            page_reports,
            page_geom,
            warnings,
            ocr_ctx,
        )


def _run_budgeted_ocr(
    page: Any,
    page_number: int,
    warnings: list[str],
    ocr_ctx: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    ctx = ocr_ctx or {"budget": OCR_BUDGET_SEC, "used": 0.0, "pages": 0, "max_pages": MAX_OCR_PAGES}
    remaining = float(ctx["budget"]) - float(ctx["used"])
    if int(ctx["pages"]) >= int(ctx["max_pages"]):
        return [], "ocr_skipped", "OCR page limit reached"
    # Each scanned page gets a full Tesseract window. Do not shrink it to a second or two.
    timeout = max(MIN_OCR_TIMEOUT_SEC, min(OCR_BUDGET_SEC, remaining if remaining > 0 else OCR_BUDGET_SEC))
    started = time.monotonic()
    try:
        spans = ocr_page(page, page_number, timeout=timeout)
        method = "ocr"
        error = None
    except OcrUnavailableError as exc:
        spans = []
        method = "ocr_unavailable"
        error = str(exc)
        warnings.append(f"Page {page_number}: {error}")
    except Exception as exc:
        spans = []
        method = "ocr_failed"
        error = f"OCR failed: {exc}"
        warnings.append(f"Page {page_number}: {error}")
    ctx["used"] = float(ctx["used"]) + (time.monotonic() - started)
    ctx["pages"] = int(ctx["pages"]) + 1
    return spans, method, error


def _extract_page(
    page: Any,
    page_number: int,
    use_ocr: bool,
    spans: list[dict[str, Any]],
    page_reports: list[dict[str, Any]],
    page_geom: dict[int, tuple[float, float]],
    warnings: list[str],
    ocr_ctx: dict[str, Any] | None = None,
) -> None:
    page_geom[page_number] = (float(page.rect.width), float(page.rect.height))
    kind = classify_page(page)
    method = "skipped"
    error = None
    page_spans: list[dict[str, Any]] = []

    if kind == "native":
        page_spans = extract_page_spans(page, page_number)
        method = "native"
    elif kind in {"scanned", "garbled"}:
        native_spans = extract_usable_native_spans(page, page_number)
        if native_spans:
            page_spans = native_spans
            method = "native"
        elif not use_ocr:
            error = "OCR skipped"
            method = "ocr_skipped"
        else:
            page_spans, method, error = _run_budgeted_ocr(page, page_number, warnings, ocr_ctx)
    else:
        method = "empty"

    spans.extend(page_spans)
    page_reports.append(
        {
            "page": page_number,
            "kind": kind,
            "method": method,
            "span_count": len(page_spans),
            "char_count": sum(len(span["text"]) for span in page_spans),
            "error": error,
        }
    )


def _lines_with_geom(
    spans: list[dict[str, Any]],
    page_geom: dict[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    lines = spans_to_lines(spans)
    for line in lines:
        width, height = page_geom.get(line["page"], (612.0, 792.0))
        line["page_width"] = round(width, 2)
        line["page_height"] = round(height, 2)
    return lines


def _document_type(pages: list[dict[str, Any]]) -> str:
    kinds = {page["kind"] for page in pages if page["kind"] != "empty"}
    if not kinds:
        return "empty"
    if kinds == {"native"}:
        return "native"
    if kinds <= {"scanned", "garbled"}:
        return "scanned"
    return "mixed"
