"""
Document extraction pipeline.

1. Classify each page as native text, scanned, garbled, or empty.
2. Native pages: PyMuPDF dict extraction (font + bbox metadata).
3. Scanned/garbled pages: Tesseract OCR on the top title band only,
   capped at ~10 seconds total.

Page budget: read pages until the title is high-confidence, capped at
PRIMARY_PAGE_COUNT, then keep going only if it still is not. Scanned
pages are the slow path, so extra OCR after a locked title is skipped.
Gemini is a last fallback after native metadata + layout heuristics.
"""

from __future__ import annotations

import time
from typing import Any

from extractor.gemini_title import apply_gemini_layer, gemini_is_configured
from extractor.heuristic import detect_title, is_high_confidence_title
from extractor.ocr import OcrUnavailableError, ocr_page, ocr_status
from extractor.pdf_utils import (
    classify_page,
    extract_page_spans,
    extract_usable_native_spans,
    open_pdf,
    reconstruct_text,
    spans_to_lines,
)

PRIMARY_PAGE_COUNT = 7
OCR_BUDGET_SEC = 10.0
MAX_OCR_PAGES = 2
MIN_OCR_TIMEOUT_SEC = 1.5


def extract_document(
    pdf_path: str,
    max_pages: int | None = None,
    filename: str | None = None,
    use_ocr: bool = True,
    primary_pages: int = PRIMARY_PAGE_COUNT,
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

        ocr_ctx = {
            "budget": OCR_BUDGET_SEC,
            "used": 0.0,
            "pages": 0,
            "max_pages": MAX_OCR_PAGES,
        }
        title_info: dict[str, Any] | None = None
        locked = False
        for page_index in range(first_pass):
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
            title_info = _detect_from_spans(spans, page_geom, metadata, filename or pdf_path)
            locked = is_high_confidence_title(title_info)
            if locked:
                break

        extended = False
        next_index = first_pass
        while not locked and next_index < limit:
            extended = True
            _extract_page(
                doc[next_index],
                next_index + 1,
                use_ocr,
                spans,
                page_reports,
                page_geom,
                warnings,
                ocr_ctx,
            )
            next_index += 1
            title_info = _detect_from_spans(spans, page_geom, metadata, filename or pdf_path)
            locked = is_high_confidence_title(title_info)

        if title_info is None:
            title_info = _detect_from_spans(spans, page_geom, metadata, filename or pdf_path)

        lines = _lines_with_geom(spans, page_geom)
        if use_gemini:
            title_info = apply_gemini_layer(title_info, lines, warnings)
        pdf_type = _document_type(page_reports)
        ocr_info = ocr_status()
        reason = title_info.get("reason") or ""
        if extended:
            reason = (
                f"{reason} Searched pages {first_pass + 1}–{len(page_reports)} "
                f"after the first {first_pass} were not high-confidence."
            ).strip()
        elif document_page_count > len(page_reports):
            reason = (
                f"{reason} Used the first {len(page_reports)} page(s); later pages were not needed."
            ).strip()

        return {
            "pdf_type": pdf_type,
            "page_count": len(page_reports),
            "document_page_count": document_page_count,
            "pages_extended": extended,
            "span_count": len(spans),
            "line_count": len(lines),
            "ocr_available": ocr_info["available"],
            "ocr_reason": ocr_info["reason"],
            "pages": page_reports,
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
            "gemini_candidate": title_info.get("gemini_candidate"),
            "gemini_title": title_info.get("gemini_title"),
            "gemini_is_correct": title_info.get("gemini_is_correct"),
            "gemini_stats": title_info.get("gemini_stats") or {},
        }
    finally:
        doc.close()


def _run_budgeted_ocr(
    page: Any,
    page_number: int,
    warnings: list[str],
    ocr_ctx: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    ctx = ocr_ctx or {"budget": OCR_BUDGET_SEC, "used": 0.0, "pages": 0, "max_pages": MAX_OCR_PAGES}
    remaining = float(ctx["budget"]) - float(ctx["used"])
    if int(ctx["pages"]) >= int(ctx["max_pages"]) or remaining < MIN_OCR_TIMEOUT_SEC:
        return [], "ocr_skipped", "OCR budget exhausted"
    timeout = max(MIN_OCR_TIMEOUT_SEC, min(OCR_BUDGET_SEC, remaining))
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


def _detect_from_spans(
    spans: list[dict[str, Any]],
    page_geom: dict[int, tuple[float, float]],
    metadata: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    lines = _lines_with_geom(spans, page_geom)
    return detect_title(
        lines,
        metadata_title=metadata.get("title"),
        filename=filename,
        metadata_author=metadata.get("author"),
    )


def _document_type(pages: list[dict[str, Any]]) -> str:
    kinds = {page["kind"] for page in pages if page["kind"] != "empty"}
    if not kinds:
        return "empty"
    if kinds == {"native"}:
        return "native"
    if kinds <= {"scanned", "garbled"}:
        return "scanned"
    return "mixed"
