"""
Native PDF text extraction with layout metadata.

Uses PyMuPDF (fitz) `page.get_text("dict")` so each span keeps font size,
font name, bold/italic flags, bounding box, and page number — not just a
flat string of words.
"""

from __future__ import annotations

import re
from typing import Any

import fitz

# Pages with fewer than this many extractable characters are treated as
# having no usable text layer (typical of scanned pages that only have a
# page number or running header buried in the image).
MIN_NATIVE_CHARS = 40

_CID_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_BOLD_HINTS = ("bold", "black", "heavy", "demibold", "semibold", "extrabold")
_ITALIC_HINTS = ("italic", "oblique")


def open_pdf(pdf_path: str) -> fitz.Document:
    return fitz.open(pdf_path)


def has_text_layer(pdf_path: str, max_pages: int | None = 3) -> bool:
    """True if any inspected page has a usable native text layer."""
    doc = open_pdf(pdf_path)
    try:
        for page_index, page in enumerate(doc):
            if max_pages is not None and page_index >= max_pages:
                break
            if classify_page(page) == "native":
                return True
        return False
    finally:
        doc.close()


def classify_page(page: fitz.Page) -> str:
    """
    Return one of: "native", "scanned", "garbled", "empty".

    Native academic PDFs (IJMET-style) almost always have a real text
    layer, so OCR is skipped unless get_text() is empty or unreadable.
    """
    raw = page.get_text("text") or ""
    stripped = raw.strip()

    if not stripped:
        return "scanned" if _page_has_images(page) else "empty"

    if _looks_garbled(stripped):
        return "garbled"

    if len(stripped) < MIN_NATIVE_CHARS and _page_has_images(page):
        # A handful of characters plus a full-page image is almost always
        # a scan with a tiny hidden/OCR-less text layer.
        return "scanned"

    return "native"


def extract_spans(pdf_path: str, max_pages: int | None = None) -> list[dict[str, Any]]:
    """Extract native-text spans only (no OCR). Kept for later title scoring."""
    doc = open_pdf(pdf_path)
    try:
        spans: list[dict[str, Any]] = []
        for page_index, page in enumerate(doc):
            if max_pages is not None and page_index >= max_pages:
                break
            spans.extend(extract_page_spans(page, page_number=page_index + 1))
        return spans
    finally:
        doc.close()


def extract_page_spans(page: fitz.Page, page_number: int) -> list[dict[str, Any]]:
    """Pull every text span on a page along with layout/style metadata."""
    dict_spans = _extract_dict_spans(page, page_number)
    dict_text = " ".join(span["text"] for span in dict_spans)
    plain = page.get_text("text") or ""

    # Some fonts decode correctly in get_text("text") but produce mojibake in
    # dict mode (AIP author-manuscript PDFs are a common case).
    if dict_spans and _looks_garbled(dict_text) and not _looks_garbled(plain):
        return _extract_block_spans(page, page_number)

    clean = [span for span in dict_spans if not _looks_garbled(span["text"])]
    if clean and (
        len(clean) >= max(1, int(len(dict_spans) * 0.35))
        or _readable_letter_count(clean) >= 40
    ):
        return clean
    return dict_spans


def extract_usable_native_spans(page: fitz.Page, page_number: int) -> list[dict[str, Any]]:
    """
    Native spans worth keeping when a page was classified scanned/garbled.

    Nature-style PDFs often have a real title layer plus images or CID junk.
    Skipping native text just because OCR is missing drops the first-page title.
    URL-only overlays are not enough to count as usable.
    """
    spans = extract_page_spans(page, page_number)
    clean = [
        span
        for span in spans
        if (span.get("text") or "").strip() and not _looks_garbled(span["text"])
    ]
    if _readable_letter_count(clean) >= 40:
        return clean
    if _readable_letter_count(spans) >= 40:
        return spans
    return []


def _readable_letter_count(spans: list[dict[str, Any]]) -> int:
    parts: list[str] = []
    for span in spans:
        text = span.get("text") or ""
        if re.search(r"https?://|\bwww\.", text, re.I):
            continue
        parts.append(text)
    return sum(ch.isalpha() for ch in " ".join(parts))


def _extract_dict_spans(page: fitz.Page, page_number: int) -> list[dict[str, Any]]:
    payload = page.get_text("dict") or {}
    spans: list[dict[str, Any]] = []
    for block_index, block in enumerate(payload.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                text = span.get("text") or ""
                if not text.strip():
                    continue
                font_name = span.get("font") or ""
                flags = int(span.get("flags") or 0)
                bbox = _normalize_bbox(span.get("bbox"))
                spans.append(
                    {
                        "text": text,
                        "page": page_number,
                        "font_name": font_name,
                        "font_size": round(float(span.get("size") or 0.0), 2),
                        "bold": _is_bold(font_name, flags),
                        "italic": _is_italic(font_name, flags),
                        "bbox": bbox,
                        "block": block_index,
                        "line": line_index,
                        "source": "native",
                        "confidence": None,
                    }
                )
    return spans


def _extract_block_spans(page: fitz.Page, page_number: int) -> list[dict[str, Any]]:
    """Fallback when dict extraction is mojibake but block text is readable."""
    spans: list[dict[str, Any]] = []
    for block_index, block in enumerate(page.get_text("blocks") or []):
        if len(block) < 7 or block[6] != 0:
            continue
        x0, y0, x1, y1, text, *_ = block
        lines = [part.strip() for part in str(text).splitlines() if part.strip()]
        if not lines:
            continue
        height = max(float(y1) - float(y0), 1.0)
        line_height = height / max(len(lines), 1)
        for line_index, line_text in enumerate(lines):
            top = float(y0) + line_index * line_height
            spans.append(
                {
                    "text": line_text,
                    "page": page_number,
                    "font_name": "",
                    "font_size": round(line_height * 0.75, 2),
                    "bold": False,
                    "italic": False,
                    "bbox": _normalize_bbox((x0, top, x1, top + line_height)),
                    "block": block_index,
                    "line": line_index,
                    "source": "native",
                    "confidence": None,
                }
            )
    return spans


def spans_to_lines(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group spans into visual lines by vertical overlap, not PDF line indexes."""
    lines: list[dict[str, Any]] = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        by_page.setdefault(int(span["page"]), []).append(span)

    for page_number, page_spans in by_page.items():
        clusters: list[list[dict[str, Any]]] = []
        for span in sorted(page_spans, key=lambda item: (item["bbox"][1], item["bbox"][0])):
            placed = False
            for cluster in clusters:
                if _same_visual_line(span, cluster):
                    cluster.append(span)
                    placed = True
                    break
            if not placed:
                clusters.append([span])

        for line_index, cluster in enumerate(clusters):
            line_spans = sorted(cluster, key=lambda item: (item["bbox"][0], item["bbox"][1]))
            text = _join_span_texts(line_spans)
            if not text.strip():
                continue
            sizes = [item["font_size"] for item in line_spans if item["font_size"]]
            dominant = max(line_spans, key=lambda item: (len(item["text"]), item["font_size"]))
            lines.append(
                {
                    "text": text,
                    "page": page_number,
                    "font_name": dominant.get("font_name") or "",
                    "font_size": round(max(sizes) if sizes else 0.0, 2),
                    "bold": any(item.get("bold") for item in line_spans),
                    "italic": any(item.get("italic") for item in line_spans),
                    "bbox": _union_bbox([item["bbox"] for item in line_spans]),
                    "block": dominant.get("block", 0),
                    "line": line_index,
                    "source": dominant.get("source", "native"),
                    "span_count": len(line_spans),
                }
            )
    lines = _merge_ocr_baseline_lines(lines)
    lines.sort(key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]))
    _mark_ocr_prominence(lines)
    return lines


def reconstruct_text(lines: list[dict[str, Any]]) -> str:
    """Readable document text, page-separated, from layout-ordered lines."""
    if not lines:
        return ""

    pages: dict[int, list[str]] = {}
    for line in lines:
        pages.setdefault(line["page"], []).append(line["text"])

    chunks = []
    for page_number in sorted(pages):
        body = "\n".join(pages[page_number]).strip()
        chunks.append(f"--- page {page_number} ---\n{body}")
    return "\n\n".join(chunks)


def _page_has_images(page: fitz.Page) -> bool:
    try:
        if page.get_images(full=True):
            return True
    except Exception:
        pass
    payload = page.get_text("dict") or {}
    return any(block.get("type") == 1 for block in payload.get("blocks", []))


def is_garbled_text(text: str) -> bool:
    return _looks_garbled(text)


def _looks_garbled(text: str) -> bool:
    sample = text[:5000]
    controls = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\t\n\r")
    if controls >= 8 or (controls >= 2 and controls / max(len(sample), 1) > 0.015):
        return True
    if sample.count("\x03") >= 2 or sample.count("\x00") >= 2:
        return True
    if re.search(r"^,[A-Z]{6,}", sample):
        return True

    compact = re.sub(r"\s+", "", sample)
    if not compact:
        return True

    replacement_ratio = sample.count("\ufffd") / max(len(sample), 1)
    if replacement_ratio > 0.05:
        return True

    cid_hits = len(_CID_RE.findall(sample))
    if cid_hits >= 8:
        return True

    letters = sum(ch.isalpha() for ch in compact)
    if letters / len(compact) < 0.25:
        return True

    return False


def _same_visual_line(span: dict[str, Any], cluster: list[dict[str, Any]]) -> bool:
    sx0, sy0, sx1, sy1 = span["bbox"]
    span_mid = (sy0 + sy1) / 2.0
    span_h = max(sy1 - sy0, float(span.get("font_size") or 1.0), 1.0)
    cy0 = min(item["bbox"][1] for item in cluster)
    cy1 = max(item["bbox"][3] for item in cluster)
    cluster_mid = (cy0 + cy1) / 2.0
    cluster_h = max(cy1 - cy0, max(float(item.get("font_size") or 1.0) for item in cluster), 1.0)
    overlap = min(sy1, cy1) - max(sy0, cy0)
    same_band = overlap > 0.25 * min(span_h, cluster_h) or abs(span_mid - cluster_mid) < 0.4 * max(span_h, cluster_h)
    if not same_band:
        return False
    cx0 = min(item["bbox"][0] for item in cluster)
    cx1 = max(item["bbox"][2] for item in cluster)
    gutter = max(0.0, sx0 - cx1, cx0 - sx1)
    size_span = float(span.get("font_size") or 0.0)
    size_cluster = max(float(item.get("font_size") or 0.0) for item in cluster)
    if size_span and size_cluster:
        size_gap = abs(size_span - size_cluster) / max(size_span, size_cluster)
        # 16pt article title vs 7pt IOP "You may also like" sidebar.
        if size_gap > 0.30 and gutter > 8:
            return False
    ocr = span.get("source") == "ocr" or any(item.get("source") == "ocr" for item in cluster)
    page_width = max(cx1, sx1, 1.0)
    gutter_limit = max(90.0, 0.16 * page_width) if ocr else 20.0
    if gutter > gutter_limit:
        return False
    return True


def _merge_ocr_baseline_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join OCR title halves that sit on one baseline with a gap Tesseract treated as two lines."""
    if len(lines) < 2:
        return lines
    ordered = sorted(lines, key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]))
    merged: list[dict[str, Any]] = []
    for line in ordered:
        if merged and _ocr_split_title_pair(merged[-1], line):
            merged[-1] = _combine_visual_lines(merged[-1], line)
        else:
            merged.append(line)
    return merged


def _ocr_split_title_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page") or 0) != int(right.get("page") or 0):
        return False
    if left.get("source") != "ocr" and right.get("source") != "ocr":
        return False
    ly0, ly1 = left["bbox"][1], left["bbox"][3]
    ry0, ry1 = right["bbox"][1], right["bbox"][3]
    left_h = max(ly1 - ly0, float(left.get("font_size") or 1.0), 1.0)
    right_h = max(ry1 - ry0, float(right.get("font_size") or 1.0), 1.0)
    overlap = min(ly1, ry1) - max(ly0, ry0)
    if overlap <= 0.45 * min(left_h, right_h):
        return False
    size_a = float(left.get("font_size") or 0.0)
    size_b = float(right.get("font_size") or 0.0)
    if size_a and size_b and abs(size_a - size_b) / max(size_a, size_b) > 0.18:
        return False
    gutter = max(0.0, right["bbox"][0] - left["bbox"][2], left["bbox"][0] - right["bbox"][2])
    width = max(left["bbox"][2], right["bbox"][2], 1.0)
    return gutter <= max(90.0, 0.18 * width)


def _combine_visual_lines(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    combined = dict(left)
    left_text = (left.get("text") or "").strip()
    right_text = (right.get("text") or "").strip()
    if left_text and right_text:
        combined["text"] = f"{left_text} {right_text}"
    else:
        combined["text"] = left_text or right_text
    combined["bbox"] = _union_bbox([left["bbox"], right["bbox"]])
    combined["font_size"] = max(float(left.get("font_size") or 0.0), float(right.get("font_size") or 0.0))
    combined["bold"] = bool(left.get("bold") or right.get("bold"))
    combined["italic"] = bool(left.get("italic") or right.get("italic"))
    combined["span_count"] = int(left.get("span_count") or 1) + int(right.get("span_count") or 1)
    combined["source"] = "ocr" if "ocr" in {left.get("source"), right.get("source")} else left.get("source", "native")
    return combined


def _mark_ocr_prominence(lines: list[dict[str, Any]]) -> None:
    """OCR has no bold flag; treat the largest cover lines as visually prominent."""
    ocr_lines = [line for line in lines if line.get("source") == "ocr"]
    if not ocr_lines:
        return
    peak = max(float(line.get("font_size") or 0.0) for line in ocr_lines)
    if peak <= 0:
        return
    for line in ocr_lines:
        if float(line.get("font_size") or 0.0) >= peak * 0.97:
            line["bold"] = True


def _is_bold(font_name: str, flags: int) -> bool:
    # PyMuPDF flag bit 4 = bold
    if flags & 2**4:
        return True
    lowered = font_name.lower()
    return any(hint in lowered for hint in _BOLD_HINTS)


def _is_italic(font_name: str, flags: int) -> bool:
    # PyMuPDF flag bit 1 = italic
    if flags & 2**1:
        return True
    lowered = font_name.lower()
    return any(hint in lowered for hint in _ITALIC_HINTS)


def _normalize_bbox(bbox: Any) -> list[float]:
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [round(float(value), 2) for value in bbox]


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def _join_span_texts(spans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    previous_x1: float | None = None
    for span in spans:
        text = span["text"]
        x0, _, x1, _ = span["bbox"]
        if (
            previous_x1 is not None
            and x0 - previous_x1 > 1.2
            and parts
            and not parts[-1].endswith(" ")
            and not text.startswith(" ")
        ):
            parts.append(" ")
        parts.append(text)
        previous_x1 = x1
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()
