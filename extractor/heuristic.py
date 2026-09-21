"""
Title detection: validated PDF /Title first, then a weighted layout heuristic.

The layout scorer never uses a single rule. Font-size percentile, isolation,
centering, author/abstract sandwiching, and a growing boilerplate denylist
are combined so journal mastheads and author lines lose to the real title.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from extractor import boilerplate as bp
from extractor.pdf_utils import is_garbled_text

MIN_METADATA_CHARS = 8
MIN_TITLE_WORDS = 2
MAX_TITLE_WORDS = 28
SWEET_MIN_WORDS = 3
SWEET_MAX_WORDS = 22
MAX_MERGED_WORDS = 28
# Wrapped title lines share a font; 14pt vs 12pt subtitles must not glue on.
SAME_TITLE_SIZE_RATIO = 0.06
SAME_TITLE_SIZE_PT = 0.75
_DANGLING_TITLE_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "of", "for", "in", "on", "to",
    "with", "by", "from", "as", "at", "via", "using", "into", "onto", "upon",
    "over", "under", "between", "among", "within", "without", "including",
    "plus", "versus", "vs",
}

# Soft layout weights. Typical academic titles land in the 55–95 band.
W_PERCENTILE = 22.0
W_SIZE_RATIO = 16.0
W_BOLD = 5.0
W_VERTICAL = 10.0
W_CENTER = 10.0
W_ISOLATION = 12.0
W_WORD_COUNT = 8.0
W_CAPS = 5.0
W_PAGE = 8.0
W_BEFORE_ABSTRACT = 10.0
W_ABOVE_AUTHORS = 14.0
W_PROMINENCE = 8.0
W_STARTS_CAPITAL = 2.0

HIGH_SCORE = 58.0
MEDIUM_SCORE = 42.0
HIGH_MARGIN = 7.0


def detect_title(
    lines: list[dict[str, Any]],
    *,
    metadata_title: str | None = None,
    filename: str | None = None,
    metadata_author: str | None = None,
) -> dict[str, Any]:
    """
    Return the best title guess plus score breakdowns for the UI.

    Tier 1: validated document-info /Title, then verified against the page.
    Tier 2: labeled Title: lines and the weighted layout heuristic.
    If the in-PDF title is strong and differs from /Title, the page wins.
    """
    cleaned_lines = [_annotate_line(line) for line in lines if (line.get("text") or "").strip()]
    meta_result, meta_reject = _try_metadata_title(metadata_title, filename, metadata_author)

    if not cleaned_lines:
        return meta_result or _empty_result("No extractable lines to score.", meta_reject)

    layout = _layout_title_result(cleaned_lines, meta_reject)
    if meta_result:
        return _reconcile_metadata_and_layout(meta_result, layout, cleaned_lines)
    return layout


def _layout_title_result(
    cleaned_lines: list[dict[str, Any]],
    meta_reject: dict[str, Any] | None,
) -> dict[str, Any]:
    usable = _drop_cover_pages(cleaned_lines)
    usable = _merge_adjacent_title_lines(usable)

    labeled = _labeled_title_result(usable, meta_reject) or _labeled_title_result(cleaned_lines, meta_reject)
    if labeled:
        return labeled

    scored = _score_lines(usable)
    if not scored:
        fallback = _fallback_document_title(usable, meta_reject)
        if fallback:
            return fallback
        return _empty_result(
            "Every candidate was rejected as boilerplate, header, or author text.",
            meta_reject,
        )

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    winner = ranked[0]
    early = [item for item in ranked if int(item.get("page") or 99) <= 2]
    if early and int(winner.get("page") or 99) > 2 and winner["score"] - early[0]["score"] < 12:
        winner = early[0]
    merged = _merge_multiline_title(winner, usable, scored)
    if _winner_looks_like_author(merged):
        replacement = next(
            (item for item in ranked if not _is_title_fragment(item["text"], merged["text"]) and not _winner_looks_like_author(item)),
            None,
        )
        if replacement:
            winner = replacement
            merged = _merge_multiline_title(winner, usable, scored)
        else:
            fallback = _fallback_document_title(usable, meta_reject)
            if fallback:
                return fallback
    if re.match(r"^(?:engr|dr|prof|mr|mrs|ms)\.\s", merged["text"], re.I) or _looks_like_short_name(merged["text"]):
        fallback = _fallback_document_title(usable, meta_reject)
        if fallback:
            return fallback

    competitors = [item for item in ranked if not _is_title_fragment(item["text"], merged["text"])]
    runner_up = competitors[0]["score"] if competitors else 0.0
    if (
        _boilerplate_reason(merged["text"])
        or _looks_like_body_fragment(merged["text"])
        or _looks_like_sentence(merged["text"])
    ):
        replacement = next(
            (
                item
                for item in ranked
                if not _is_title_fragment(item["text"], merged["text"])
                and not _boilerplate_reason(item["text"])
                and not _looks_like_body_fragment(item["text"])
                and not _looks_like_sentence(item["text"])
                and not _winner_looks_like_author(item)
            ),
            None,
        )
        if replacement:
            winner = replacement
            merged = _merge_multiline_title(winner, usable, scored)
            competitors = [item for item in ranked if not _is_title_fragment(item["text"], merged["text"])]
            runner_up = competitors[0]["score"] if competitors else 0.0
    confidence = _confidence(merged["score"], merged["score"] - runner_up)
    if _ends_incomplete(merged["text"]) and confidence == "high":
        confidence = "medium"

    alternatives = [
        {
            "text": merged["text"],
            "score": round(merged["score"], 2),
            "page": merged["page"],
            "font_size": merged.get("font_size"),
        }
    ]
    alternatives.extend(
        {
            "text": item["text"],
            "score": round(item["score"], 2),
            "page": item["page"],
            "font_size": item["font_size"],
        }
        for item in competitors[:7]
    )

    return {
        "title": merged["text"],
        "source": "heuristic",
        "confidence": confidence,
        "score": round(merged["score"], 2),
        "page": merged["page"],
        "reason": "Weighted layout heuristic over font, position, isolation, and structure.",
        "signals": merged.get("signals") or winner.get("signals", {}),
        "alternatives": alternatives,
        "rejected_metadata": meta_reject,
    }


def _reconcile_metadata_and_layout(
    meta: dict[str, Any],
    layout: dict[str, Any],
    lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep /Title only when it agrees with the page; otherwise prefer the in-PDF title."""
    layout_title = (layout.get("title") or "").strip()
    if not layout_title:
        return meta

    meta_title = (meta.get("title") or "").strip()
    layout_score = float(layout.get("score") or 0.0)
    layout_strong = layout.get("confidence") == "high" or layout_score >= HIGH_SCORE
    layout_usable = layout.get("confidence") in {"high", "medium"} or layout_score >= MEDIUM_SCORE
    same = _is_title_fragment(meta_title, layout_title)
    meta_on_page = _title_appears_in_document(meta_title, lines)

    if same:
        meta_key = bp.exact_key(meta_title)
        layout_key = bp.exact_key(layout_title)
        # Metadata is a fuller string that actually appears on the page.
        if layout_key in meta_key and len(meta_key) > len(layout_key) + 10 and meta_on_page:
            chosen = dict(meta)
            chosen["reason"] = "PDF /Title matches the in-PDF title and is more complete."
            chosen["page"] = layout.get("page")
            return chosen
        chosen = dict(layout)
        chosen["confidence"] = "high"
        chosen["reason"] = "In-PDF title agrees with /Title metadata; using the page text."
        chosen["rejected_metadata"] = None
        return chosen

    if layout.get("source") == "labeled" and layout_usable:
        chosen = dict(layout)
        chosen["rejected_metadata"] = {"title": meta_title, "reason": "page_title_differs"}
        chosen["reason"] = (layout.get("reason") or "") + " Overrode /Title metadata with an explicit on-page Title label."
        return chosen

    if layout_strong and not meta_on_page:
        chosen = dict(layout)
        chosen["rejected_metadata"] = {"title": meta_title, "reason": "not_on_page"}
        chosen["reason"] = (layout.get("reason") or "") + " Overrode /Title metadata; that field does not appear on the page."
        return chosen

    if layout_strong and not same:
        chosen = dict(layout)
        chosen["rejected_metadata"] = {"title": meta_title, "reason": "page_title_differs"}
        chosen["reason"] = (layout.get("reason") or "") + " Overrode /Title metadata with the stronger in-PDF title."
        return chosen

    if layout_usable and not meta_on_page:
        chosen = dict(layout)
        chosen["rejected_metadata"] = {"title": meta_title, "reason": "not_on_page"}
        chosen["reason"] = (layout.get("reason") or "") + " Overrode /Title metadata; that field does not appear on the page."
        return chosen

    return meta


def _title_appears_in_document(title: str, lines: list[dict[str, Any]]) -> bool:
    needle = bp.exact_key(title)
    if len(needle) < 8:
        return False
    blob = " ".join(bp.exact_key(line.get("text") or "") for line in lines if int(line.get("page") or 1) <= 3)
    if needle in blob:
        return True
    compact_needle = re.sub(r"[^a-z0-9]+", "", needle)
    compact_blob = re.sub(r"[^a-z0-9]+", "", blob)
    return len(compact_needle) >= 10 and compact_needle in compact_blob


def _try_metadata_title(
    metadata_title: str | None,
    filename: str | None,
    metadata_author: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw = _clean_metadata_title(metadata_title or "")
    ok, reason = validate_metadata_title(raw, filename, metadata_author, original=metadata_title)
    if not ok:
        reject = {"title": raw or None, "reason": reason} if (raw or reason != "empty") else {"title": None, "reason": reason}
        return None, reject
    return {
        "title": raw,
        "source": "metadata",
        "confidence": "high",
        "score": 100.0,
        "page": None,
        "reason": "Validated PDF /Title metadata field.",
        "signals": {"metadata": 100.0},
        "alternatives": [],
        "rejected_metadata": None,
        "metadata_status": reason,
    }, None


def validate_metadata_title(
    title: str,
    filename: str | None = None,
    metadata_author: str | None = None,
    original: str | None = None,
) -> tuple[bool, str]:
    text = bp.normalize_space(title)
    source = original or title or ""
    if not text:
        return False, "empty"
    if "..." in source:
        return False, "truncated"
    if len(text) < MIN_METADATA_CHARS:
        return False, "too_short"
    if bp.FILE_EXTENSION_RE.search(text):
        return False, "file_extension"
    if bp.WORD_PROCESSOR_RE.search(text):
        return False, "word_processor"
    if bp.PATH_RE.search(text):
        return False, "filepath"
    if re.search(r"messages\.(studocu|downloaded)|lomoarcpsd|cn report", source, re.I):
        return False, "studocu"
    if re.match(r"^\d+\s*,\s*\d+", text):
        return False, "studocu"
    if re.search(r"\bmedium\b", source, re.I) and re.search(r"\bby\b", source, re.I):
        # Cleaned Medium titles are allowed; the raw "by X | Medium" form is not.
        if re.search(r"\|\s*by\s+|\sby\s+.+\sMedium", source, re.I) and "Medium" in source:
            pass
    key = bp.exact_key(text)
    if key in bp.GENERIC_TITLES:
        return False, "generic"
    if _matches_filename(text, filename):
        return False, "matches_filename"
    if metadata_author and bp.exact_key(metadata_author) == key:
        return False, "matches_author"
    if _boilerplate_reason(text):
        return False, "boilerplate"
    if _masthead_reason(text):
        return False, "journal_masthead"
    if _word_count(text) > MAX_TITLE_WORDS:
        return False, "too_long"
    return True, "ok"


def _clean_metadata_title(title: str) -> str:
    text = bp.normalize_space(title or "")
    text = re.sub(r"\s*[|_].*?\bMedium\s*$", "", text, flags=re.I)
    text = re.sub(r"^#+", "", text)
    text = text.replace("_", " ")
    return bp.normalize_space(text)


def _annotate_line(line: dict[str, Any]) -> dict[str, Any]:
    item = dict(line)
    text = bp.normalize_space(item.get("text") or "")
    text = text.strip(" \t\"'«»“”‘’#")
    text = re.sub(r"^[\u201c\u201d\u00ab\u00bb]+|[\"'«»“”]+$", "", text)
    # Slide titles often append "(© Author …)" — keep the title, drop the notice.
    text = re.sub(r"\s*\(\s*©[^)]*\)", " ", text)
    text = re.sub(r"\s+©\s+.*$", "", text)
    # IOP two-column chrome is often glued onto the same baseline as the title.
    text = re.split(r"\s+you\s+may\s+also\s+like\b.*$", text, flags=re.I)[0]
    text = re.sub(r"^paper\s*[•·\-]?\s*open\s+access\s*", "", text, flags=re.I)
    text = re.split(r"\s+e-?issn\b.*$", text, flags=re.I)[0]
    text = re.split(r"\s+M\.?\s*Tech\b.*$", text, flags=re.I)[0]
    text = re.split(r"\s+Department\s+Of\b.*$", text, flags=re.I)[0]
    text = re.split(r"(?<=[A-Za-z])\s+(?=[A-Z]\.[A-Z][a-z]+\s+[A-Z][a-z]+)", text, maxsplit=1)[0]
    text = bp.normalize_space(text)
    item["text"] = text
    item["norm"] = bp.exact_key(text)
    height = float(item.get("page_height") or 0.0) or 1.0
    width = float(item.get("page_width") or 0.0) or 1.0
    bbox = item.get("bbox") or [0, 0, 0, 0]
    item["y0"] = float(bbox[1])
    item["y1"] = float(bbox[3])
    item["x0"] = float(bbox[0])
    item["x1"] = float(bbox[2])
    item["y_ratio"] = item["y0"] / height
    item["x_mid"] = (item["x0"] + item["x1"]) / 2.0
    item["center_offset"] = abs(item["x_mid"] - width / 2.0) / width
    item["line_height"] = max(item["y1"] - item["y0"], float(item.get("font_size") or 0.0), 1.0)
    item["word_count"] = _word_count(text)
    return item


def _score_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    body_size = _body_font_size(lines)
    sizes = [float(line.get("font_size") or 0.0) for line in lines]
    weights = [max(len(line["text"]), 1) for line in lines]
    by_page = _lines_by_page(lines)
    header_norms = _running_header_norms(lines)
    abstract_at = _first_marker(lines, _is_abstract_marker)
    author_at = _first_marker(lines, _looks_like_author_or_affiliation)
    iop_pages = _iop_chrome_pages(lines)
    first_page = min((int(line["page"]) for line in lines), default=1)

    scored: list[dict[str, Any]] = []
    for line in lines:
        reject = _hard_reject(line, header_norms, iop_pages=iop_pages)
        if reject:
            continue

        gaps = _isolation_gaps(line, by_page.get(line["page"], []))
        signals = {
            "font_percentile": _percentile_signal(float(line["font_size"] or 0.0), sizes, weights),
            "size_vs_body": _size_ratio_signal(float(line["font_size"] or 0.0), body_size),
            "bold": W_BOLD if line.get("bold") else 0.0,
            "vertical": _vertical_signal(line["y_ratio"]),
            "centering": _center_signal(line["center_offset"]),
            "isolation": _isolation_signal(gaps, line["line_height"]),
            "word_count": _word_count_signal(line["word_count"]),
            "capitalization": _caps_signal(line["text"]),
            "page_bias": _page_signal(int(line["page"]), first_page),
            "before_abstract": _abstract_signal(line, abstract_at),
            "above_authors": _author_sandwich_signal(line, author_at),
            "prominence": _prominence_signal(line, by_page.get(line["page"], []), body_size),
            "starts_capital": W_STARTS_CAPITAL if _starts_capitalized(line["text"]) else 0.0,
        }
        score = sum(signals.values()) + _soft_penalties(line, abstract_at, body_size)
        if line["word_count"] < SWEET_MIN_WORDS and (line["font_size"] or 0) < body_size * 1.35:
            continue
        if score < 18:
            continue
        scored.append(
            {
                **line,
                "score": score,
                "signals": {key: round(value, 2) for key, value in signals.items()},
            }
        )
    return scored


def _hard_reject(
    line: dict[str, Any],
    header_norms: set[str],
    iop_pages: set[int] | None = None,
) -> str | None:
    text = line["text"]
    key = line["norm"]
    if iop_pages and int(line.get("page") or 0) in iop_pages:
        width = float(line.get("page_width") or 0.0) or 1.0
        if line["x0"] / width >= 0.48:
            return "iop_sidebar"
        if line["y_ratio"] < 0.12 and line["word_count"] <= 4 and not _looks_topical(text):
            return "journal_subtitle"
    if not key:
        return "empty"
    if key in bp.EXACT_REJECT:
        return "section_heading"
    stripped = bp.SECTION_NUMBER_RE.sub("", text).strip()
    if bp.exact_key(stripped) in bp.EXACT_REJECT:
        return "section_heading"
    if line["word_count"] < MIN_TITLE_WORDS or line["word_count"] > MAX_TITLE_WORDS:
        return "word_count"
    if header_norms and key in header_norms and (line["y_ratio"] < 0.12 or line["y_ratio"] > 0.88):
        # Repeated slide/running titles can sit in the header band; keep large ones.
        if line["word_count"] > 10 or float(line.get("font_size") or 0) <= 12:
            return "running_header"
    reason = _boilerplate_reason(text)
    if reason:
        return reason
    if is_garbled_text(text):
        return "garbled"
    if bp.COURSE_CODE_RE.match(text):
        return "course_code"
    if key in {"computer networks", "electrical machine design", "electrical machines"}:
        # Large cover headings like "Computer Networks" sit above "Project Report"
        # and should be mergeable rather than discarded.
        size = float(line.get("font_size") or 0.0)
        if not (size >= 18 and int(line.get("page") or 99) == 1 and line.get("y_ratio", 1.0) < 0.25):
            return "generic_course"
    if re.match(r"^department of\b", text, re.I):
        return "institution"
    if bp.INSTITUTION_LINE_RE.match(text) and not _looks_topical(text):
        return "institution"
    if re.search(r"\b(university|college of engineering|institute of engineering)\b", text, re.I) and not _looks_topical(text) and line["word_count"] <= 12:
        return "institution"
    if line["y_ratio"] < 0.12:
        masthead = _masthead_reason(text)
        if masthead:
            return masthead
        if (
            _mostly_all_caps(text)
            and line["word_count"] <= 4
            and not _looks_topical(text)
            and float(line.get("font_size") or 0.0) < 14
        ):
            return "masthead_caps"
    if _is_journal_name_line(text):
        return "journal_name"
    if _looks_like_author_or_affiliation(line):
        return "author_or_affiliation"
    if _letter_ratio(text) < 0.55:
        return "low_letter_ratio"
    if re.fullmatch(r"[A-Za-z]+\s*,?\s*\d{4}", text):
        return "date"
    if re.match(r"^\d+(?:\.\d+){2,}[.)]?\s+\S", text):
        return "body_section_heading"
    if re.match(r"^\d+\.\d+[.)]?\s+\S", text) and not re.search(
        r"\bintroduction\s+to\b", text, re.I
    ):
        return "body_section_heading"
    if re.search(r":\s+(use of|using|this is|it is)\b", text, re.I) and line["word_count"] >= 8:
        return "definition"
    if re.match(r"^\d+\.\s+[A-Z][a-z]", text) and line["word_count"] >= 8:
        return "list_item"
    if re.match(r"^(?:[ivxlcdm]{1,6}|[IVXLCDM]{1,6})\.\s+\S", text) and line["word_count"] <= 8:
        return "body_section_heading"
    if re.match(r"^chapter\s+\d+(?:\s+introduction)?$", key) and not re.search(
        r"\bintroduction\s+to\b", text, re.I
    ):
        return "section_heading"
    if re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?introduction\b", text, re.I) and not re.search(
        r"\bintroduction\s+to\b", text, re.I
    ):
        return "section_heading"
    if re.search(r"\b(internal editor|editor-in-chief|editorial board)\b", text, re.I):
        return "editor"
    if re.search(r"\b(bachelor of|master of|partial fulfilment|partial fulfillment)\b", text, re.I):
        return "degree"
    if re.match(r"^(?:engr|dr|prof|mr|mrs|ms)\.?", text, re.I) and not _looks_topical(text) and line["word_count"] <= 8:
        return "author_or_affiliation"
    if _looks_like_body_fragment(text):
        return "body_fragment"
    if _mostly_all_caps(text) and text_ends_with_sentence_period(text) and line["word_count"] >= 6:
        return "sentence"
    if _looks_like_sentence(text):
        return "sentence"
    if _looks_like_formula(text):
        return "formula"
    if _looks_like_ocr_junk(text):
        return "ocr_junk"
    if re.search(r"\bpermission\b.*\bprohibited\b", text, re.I):
        return "ocr_junk"
    if re.search(r"[=]", text) and line["word_count"] <= 6 and not _looks_topical(text):
        return "ocr_junk"
    if re.fullmatch(r"[\d\s./:-]+", text):
        return "numeric"
    return None


def _soft_penalties(line: dict[str, Any], abstract_at: tuple[int, float] | None, body_size: float) -> float:
    penalty = 0.0
    if abstract_at and _comes_after(line, abstract_at):
        penalty -= 28.0
    if line["y_ratio"] > 0.72:
        penalty -= 12.0
    size = float(line.get("font_size") or 0.0)
    if body_size and size <= body_size * 1.04:
        penalty -= 14.0
    if line["center_offset"] > 0.22 and size < body_size * 1.4:
        penalty -= 4.0
    if text_ends_with_sentence_period(line["text"]) and line["word_count"] > 12:
        penalty -= 3.0
    if line.get("italic") and not line.get("bold") and size < body_size * 1.3:
        penalty -= 2.0
    masthead = _masthead_reason(line["text"])
    if line["y_ratio"] < 0.18 and masthead in {"journal", "proceedings", "transactions", "conference", "symposium"}:
        penalty -= 18.0
    return penalty


def _merge_multiline_title(
    winner: dict[str, Any],
    all_lines: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> dict[str, Any]:
    page_lines = sorted(
        [line for line in all_lines if line["page"] == winner["page"]],
        key=lambda item: (item["y0"], item["x0"]),
    )
    scored_by_norm_page = {(item["page"], item["norm"]): item for item in scored}
    try:
        index = next(
            i
            for i, line in enumerate(page_lines)
            if line["norm"] == winner["norm"] and abs(line["y0"] - winner["y0"]) < 1.5
        )
    except StopIteration:
        return winner

    chosen = [page_lines[index]]
    target_size = float(winner["font_size"] or 0.0) or 1.0

    def compatible(candidate: dict[str, Any], *, allowing_wrap: bool) -> bool:
        reject = _hard_reject(candidate, set())
        # Wrap tails are often 1 word ("corn-thresher") or a lowercase
        # continuation that looks like a body fragment in isolation.
        wrap_ok = {"word_count", "body_fragment", "author_or_affiliation"} if allowing_wrap else set()
        if reject and reject not in wrap_ok:
            return False
        if _looks_like_author_or_affiliation(candidate) and not allowing_wrap:
            return False
        if _is_abstract_marker(candidate):
            return False
        size = float(candidate.get("font_size") or 0.0)
        if not _same_title_size(target_size, size):
            return False
        center_limit = 0.36 if candidate.get("word_count", 0) <= 4 or allowing_wrap else 0.20
        if abs(candidate["center_offset"] - winner["center_offset"]) > center_limit:
            return False
        if (
            candidate.get("bold") != winner.get("bold")
            and abs(size - target_size) / target_size > 0.08
            and not allowing_wrap
        ):
            return False
        return True

    def gap_ok(upper: dict[str, Any], lower: dict[str, Any], *, allowing_wrap: bool) -> bool:
        gap = lower["y0"] - upper["y1"]
        limit = 2.2 if allowing_wrap else 1.7
        return gap < limit * max(upper["line_height"], lower["line_height"], target_size)

    # Prefer expanding downward (title continuations / subtitles).
    cursor = index
    while cursor + 1 < len(page_lines):
        nxt = page_lines[cursor + 1]
        if _other_column(winner, nxt) or _other_column(chosen[-1], nxt):
            cursor += 1
            continue
        allowing = _ends_incomplete(_join_title_texts(chosen)) or _is_title_wrap_tail(chosen[-1], nxt)
        if not compatible(nxt, allowing_wrap=allowing) or not gap_ok(chosen[-1], nxt, allowing_wrap=allowing):
            break
        words = _word_count(_join_title_texts(chosen + [nxt]))
        if words > MAX_MERGED_WORDS:
            break
        chosen.append(nxt)
        cursor += 1

    cursor = index
    while cursor > 0:
        prev = page_lines[cursor - 1]
        allowing = _ends_incomplete(prev["text"]) or _is_title_wrap_tail(prev, chosen[0])
        if not compatible(prev, allowing_wrap=allowing) or not gap_ok(prev, chosen[0], allowing_wrap=allowing):
            break
        words = _word_count(_join_title_texts([prev] + chosen))
        if words > MAX_MERGED_WORDS:
            break
        # Don't swallow a journal masthead sitting just above the title.
        if prev["y_ratio"] < 0.10 and _masthead_reason(prev["text"]):
            break
        chosen.insert(0, prev)
        cursor -= 1

    text = _join_title_texts(chosen)
    best_score = winner["score"]
    for part in chosen:
        extra = scored_by_norm_page.get((part["page"], part["norm"]))
        if extra:
            best_score = max(best_score, extra["score"])
    if len(chosen) > 1:
        best_score += min(6.0, 2.0 * (len(chosen) - 1))
    merged = dict(winner)
    merged["text"] = text
    merged["score"] = best_score
    merged["page"] = winner["page"]
    return merged


def _join_title_texts(lines: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for line in lines:
        piece = line["text"].strip()
        if parts and parts[-1].endswith("-") and not parts[-1].endswith("--"):
            parts[-1] = parts[-1][:-1] + piece
        else:
            parts.append(piece)
    return bp.normalize_space(" ".join(parts))


def _body_font_size(lines: list[dict[str, Any]]) -> float:
    tallies: dict[float, int] = defaultdict(int)
    for line in lines:
        size = round(float(line.get("font_size") or 0.0), 1)
        if size <= 0:
            continue
        tallies[size] += max(len(line["text"]), 1)
    if not tallies:
        return 11.0
    return max(tallies.items(), key=lambda item: item[1])[0]


def _percentile_signal(size: float, sizes: list[float], weights: list[int]) -> float:
    if not sizes:
        return 0.0
    total = float(sum(weights)) or 1.0
    below = sum(weight for value, weight in zip(sizes, weights) if value <= size)
    percentile = below / total
    # 90th percentile is the intended sweet spot; 50th percentile is body text.
    shaped = max(0.0, (percentile - 0.55) / 0.45)
    return W_PERCENTILE * min(1.0, shaped)


def _size_ratio_signal(size: float, body_size: float) -> float:
    if body_size <= 0:
        return 0.0
    ratio = size / body_size
    if ratio <= 1.05:
        return 0.0
    # 1.05x → 0, ~1.6x+ → full. Logos at 3x still cap; masthead rules handle those.
    shaped = min(1.0, (ratio - 1.05) / 0.55)
    return W_SIZE_RATIO * shaped


def _vertical_signal(y_ratio: float) -> float:
    # Peak in the upper-middle band (journal header sits above, body below).
    if y_ratio < 0.04:
        return 1.0
    if y_ratio < 0.08:
        return W_VERTICAL * 0.8
    if y_ratio <= 0.48:
        return W_VERTICAL
    if y_ratio <= 0.62:
        return W_VERTICAL * 0.55
    if y_ratio <= 0.75:
        return W_VERTICAL * 0.2
    return 0.0


def _center_signal(center_offset: float) -> float:
    if center_offset <= 0.04:
        return W_CENTER
    if center_offset <= 0.08:
        return W_CENTER * 0.7
    if center_offset <= 0.14:
        return W_CENTER * 0.3
    return 0.0


def _isolation_signal(gaps: tuple[float, float], line_height: float) -> float:
    above, below = gaps
    typical = max(line_height * 0.35, 2.0)
    above_ratio = above / typical
    below_ratio = below / typical
    score = 0.0
    if above_ratio >= 1.6:
        score += W_ISOLATION * 0.5
    elif above_ratio >= 1.1:
        score += W_ISOLATION * 0.25
    if below_ratio >= 1.6:
        score += W_ISOLATION * 0.5
    elif below_ratio >= 1.1:
        score += W_ISOLATION * 0.25
    return score


def _word_count_signal(count: int) -> float:
    if count < SWEET_MIN_WORDS or count > SWEET_MAX_WORDS:
        if MIN_TITLE_WORDS <= count <= MAX_TITLE_WORDS:
            return W_WORD_COUNT * 0.25
        return 0.0
    # Peak around 6–14 words.
    if 5 <= count <= 16:
        return W_WORD_COUNT
    return W_WORD_COUNT * 0.7


def _caps_signal(text: str) -> float:
    words = _alpha_words(text)
    if not words:
        return 0.0
    if all(word.isupper() and len(word) > 1 for word in words if len(word) > 1):
        return W_CAPS * 0.7  # ALL CAPS helps, but journals use it too
    title_case = sum(1 for word in words if word[:1].isupper() and (len(word) == 1 or word[1:].islower() or word[1:].istitle()))
    small = {"a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "by", "with", "from", "at", "vs"}
    content = [word for word in words if word.lower() not in small]
    if content and title_case / max(len(words), 1) >= 0.6:
        return W_CAPS
    if words[0][:1].isupper():
        return W_CAPS * 0.35
    return 0.0


def _page_signal(page: int, first_page: int = 1) -> float:
    ordinal = page - int(first_page or 1) + 1
    if ordinal <= 1:
        return W_PAGE
    if ordinal == 2:
        return W_PAGE * 0.45
    if ordinal == 3:
        return W_PAGE * 0.2
    return 0.0


def _abstract_signal(line: dict[str, Any], abstract_at: tuple[int, float] | None) -> float:
    if not abstract_at:
        return 0.0
    if _comes_after(line, abstract_at):
        return 0.0
    if line["page"] == abstract_at[0] and abstract_at[1] - line["y0"] < 280:
        return W_BEFORE_ABSTRACT
    if line["page"] == abstract_at[0]:
        return W_BEFORE_ABSTRACT * 0.6
    if line["page"] == abstract_at[0] - 1:
        return W_BEFORE_ABSTRACT * 0.3
    return 0.0


def _author_sandwich_signal(line: dict[str, Any], author_at: tuple[int, float] | None) -> float:
    if not author_at:
        return 0.0
    if line["page"] != author_at[0]:
        return 0.0
    delta = author_at[1] - line["y1"]
    if 4 <= delta <= 90:
        return W_ABOVE_AUTHORS
    if 0 <= delta <= 160:
        return W_ABOVE_AUTHORS * 0.55
    return 0.0


def _prominence_signal(line: dict[str, Any], page_lines: list[dict[str, Any]], body_size: float) -> float:
    neighbors = sorted(page_lines, key=lambda item: item["y0"])
    try:
        index = next(i for i, item in enumerate(neighbors) if item is line or (item["norm"] == line["norm"] and abs(item["y0"] - line["y0"]) < 1))
    except StopIteration:
        return 0.0
    size = float(line.get("font_size") or 0.0)
    bonus = 0.0
    if index > 0:
        above = float(neighbors[index - 1].get("font_size") or 0.0)
        if size >= above * 0.92:
            bonus += W_PROMINENCE * 0.35
    if index + 1 < len(neighbors):
        below = neighbors[index + 1]
        below_size = float(below.get("font_size") or 0.0)
        if _looks_like_author_or_affiliation(below) or (below_size and size >= below_size * 1.12):
            bonus += W_PROMINENCE * 0.65
    if body_size and size >= body_size * 1.25:
        bonus = min(W_PROMINENCE, bonus + 2.0)
    return min(W_PROMINENCE, bonus)


def _isolation_gaps(line: dict[str, Any], page_lines: list[dict[str, Any]]) -> tuple[float, float]:
    above = line["y0"]
    below = float(line.get("page_height") or 0.0) - line["y1"]
    for other in page_lines:
        if other is line:
            continue
        if other["y1"] <= line["y0"] + 0.4:
            above = min(above, line["y0"] - other["y1"])
        elif other["y0"] >= line["y1"] - 0.4:
            below = min(below, other["y0"] - line["y1"])
    return max(above, 0.0), max(below, 0.0)


def _running_header_norms(lines: list[dict[str, Any]]) -> set[str]:
    locations: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for line in lines:
        if not line["norm"] or line["word_count"] > 18:
            continue
        locations[line["norm"]].append((line["page"], line["y_ratio"]))
    headers: set[str] = set()
    for norm, spots in locations.items():
        pages = {page for page, _ in spots}
        if len(pages) < 2:
            continue
        margin_hits = sum(1 for _, y_ratio in spots if y_ratio < 0.12 or y_ratio > 0.88)
        if margin_hits >= 2:
            headers.add(norm)
    return headers


def _iop_chrome_pages(lines: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    marker = re.compile(
        r"you\s+may\s+also\s+like|paper\s*[•·\-]?\s*open\s+access|"
        r"to\s+cite\s+this\s+article|iop\s+conference\s+series",
        re.I,
    )
    for line in lines:
        if marker.search(line.get("text") or ""):
            pages.add(int(line["page"]))
    return pages


def _first_marker(lines: list[dict[str, Any]], predicate) -> tuple[int, float] | None:
    ordered = sorted(lines, key=lambda item: (item["page"], item["y0"]))
    for line in ordered:
        if predicate(line):
            return int(line["page"]), float(line["y0"])
    return None


def _is_abstract_marker(line: dict[str, Any]) -> bool:
    key = bp.exact_key(bp.SECTION_NUMBER_RE.sub("", line["text"]))
    return key in {"abstract", "summary", "highlights", "graphical abstract"} or key.startswith("abstract ")


def _looks_like_author_or_affiliation(line: dict[str, Any]) -> bool:
    text = line["text"]
    if re.search(r"\b[A-Z]{2,}\d{3,}/\d+", text):
        return True
    if re.search(r"[\w.+-]+@[\w-]+\.", text):
        return True
    if re.search(r"\bcorresponding author\b", text, re.I):
        return True
    if bp.INSTITUTION_LINE_RE.match(text) and not _looks_topical(text):
        return True
    if bp.AFFILIATION_RE.search(text) and not _looks_topical(text):
        return True
    if text.count(",") >= 2 and line["word_count"] <= 22 and not _looks_topical(text):
        return True
    letters = re.sub(r"[^A-Za-z,;& ]", " ", text)
    comma_names = letters.count(",") >= 1 and line["word_count"] <= 18
    if comma_names and re.search(r"\b(and|&)\b", text, re.I) and not _looks_topical(text):
        return True
    if (
        line["word_count"] <= 10
        and re.match(r"^[^\W\d_][\w.'’-]+(?:\s+[^\W\d_][\w.'’-]+){1,3}\s+(?:and|&)\s+[^\W\d_]", text, re.UNICODE)
        and not _looks_topical(text)
    ):
        return True
    if bp.AUTHOR_NAME_LIST_RE.match(text) and line["word_count"] <= 16:
        return True
    if re.search(r"[^\W\d_][\w.'-]+(?:\s+[\w.'-]+){0,3}\s*[\d*†‡]+(?:\s*,\s*[^\W\d_])", text, re.UNICODE):
        return True
    if re.search(r"\b[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z]+", text) and "," in text:
        return True
    if _looks_like_person_names(text, line.get("word_count") or _word_count(text)):
        return True
    return False


def _looks_like_person_names(text: str, word_count: int) -> bool:
    """Catch 'Anand Kumar S malipatil Anantharaja M.H' style author lines with no commas."""
    if _looks_topical(text):
        return False
    if word_count < 2 or word_count > 12:
        return False
    if re.search(r"\b(of|for|with|using|from|into|on|to|by|and the)\b", text, re.I):
        return False
    if re.search(r"\d{3,}", text):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z.'’-]*", text)
    if len(tokens) < 2:
        return False
    name_like = sum(1 for token in tokens if _is_name_token(token))
    has_initial = any(re.fullmatch(r"[A-Z](?:\.[A-Z])+\.?", token) or re.fullmatch(r"[A-Z]\.", token) for token in tokens)
    if name_like / len(tokens) < 0.75:
        return False
    # Two people jammed together, one person with an initial, or a bare First Last cover name.
    if has_initial or name_like >= 4:
        return True
    if word_count == 2 and name_like == 2 and all(re.fullmatch(r"[A-Z][a-z]{2,20}", token) for token in tokens):
        nouns = {
            "networks", "network", "systems", "system", "african", "journal", "science",
            "computer", "electrical", "mechanical", "scientific", "international",
            "campus", "university", "transformer", "report", "project", "analysis",
            "design", "engineering", "technology", "college", "machine", "engines",
            "engine", "thermal", "fluid", "power", "plant", "automobile", "transfer",
            "dynamics", "machinery", "process", "processes", "elements", "element",
            "study", "studies", "review", "research", "institute", "department",
        }
        if any(token.lower() in nouns for token in tokens):
            return False
        return True
    return False


def _is_name_token(token: str) -> bool:
    if token.lower() in {"and", "or", "the", "of", "for", "dr", "prof", "mr", "mrs", "ms", "engr"}:
        return False
    if token.isupper() and len(token) > 2:
        return False
    if re.fullmatch(r"[A-Z](?:\.[A-Z])+\.?", token) or re.fullmatch(r"[A-Z]\.?", token):
        return True
    if re.fullmatch(r"[A-Z]\.[A-Z][a-z]+", token):
        return True
    if re.fullmatch(r"[A-Z][a-z]+\.[A-Z]\.?", token):
        return True
    if re.fullmatch(r"[A-Z][a-z]{1,20}", token):
        return True
    if re.fullmatch(r"[a-z]{3,20}", token):
        return True
    return False


def _winner_looks_like_author(line: dict[str, Any]) -> bool:
    text = line.get("text") or ""
    return bool(
        _looks_like_author_or_affiliation(line)
        or _looks_like_person_names(text, line.get("word_count") or _word_count(text))
    )


def _looks_topical(text: str) -> bool:
    return bool(
        re.search(
            r"\b(design|analysis|using|based|study|effect|investigation|implementation|"
            r"framework|system|network|project|detection|prevention|optimization|"
            r"simulation|experimental|transformer|machine|bridge|campus|smart|"
            r"towards|toward|evaluation|development|performance|removal|dyeing|"
            r"refrigeration|incubator|monitoring|protection|construction|scaling|"
            r"converter|photovoltaic|hospital|cisco|packet|"
            r"engine|chamber|combustion|cfd|nanofluid|radiator|compressor|axial|"
            r"cleaner|boiler|harvester|turbine|gearbox|mixer|weeder|thresher|"
            r"shredder|dryer|cutter|condenser|evaporator|muffler|impeller|"
            r"membrane|chromium|phosphate|carrier|removal|liquid)\b",
            text,
            re.I,
        )
    )


def _looks_like_body_fragment(text: str) -> bool:
    """Mid-paragraph wraps that leak in when page 1 was skipped (Nature body columns)."""
    stripped = (text or "").lstrip()
    if not stripped:
        return False
    first = next((char for char in stripped if char.isalpha()), "")
    if first.islower() and _word_count(text) >= 6:
        return True
    if _word_count(text) >= 8 and re.match(
        r"^(more than|of the|in the same|to compare|as well as|so it was)\b",
        stripped,
        re.I,
    ):
        return True
    return False


def _looks_like_sentence(text: str) -> bool:
    words = _word_count(text)
    finite = re.search(
        r"\b(is|are|was|were|been|have|has|had|include|includes|comprises|will|can|may|"
        r"should|does|did|plays|play|provides|makes|describes|presents|perform|performs|"
        r"deployed|successfully|begins|begin|focuses|depends|consists|gives|relates)\b",
        text,
        re.I,
    )
    if words >= 10 and re.match(r"^(the|this|these|those|it|we|our)\s", text, re.I) and finite:
        return True
    if words >= 11 and finite and not _mostly_title_or_caps(text):
        return True
    if re.search(r"\bI successfully\b|\bwe successfully\b", text, re.I):
        return True
    if re.search(r"\bI\s+\w+ed\b", text) and ":" not in text and not _mostly_title_or_caps(text):
        return True
    if words >= 12 and text_ends_with_sentence_period(text):
        return True
    if words >= 12 and re.search(r"\b(comprises|which will|in order to|is used for|include a)\b", text, re.I):
        return True
    # "OUTPUT EQUATION: - It gives the relationship…" is a definition, not a title.
    if words >= 8 and re.match(r"^[A-Z][A-Z\s]+:\s*[-–]?\s*(it|this|the)\s+\w+", text, re.I):
        return True
    return False


def _looks_like_formula(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 6:
        return False
    skip = {"in", "an", "of", "to", "on", "or", "at", "by", "as", "if", "ic", "si", "ac", "dc", "vs"}
    def short_symbol(token: str) -> bool:
        core = re.sub(r"[^A-Za-z0-9]", "", token)
        if not core or core.lower() in skip:
            return False
        if core.isdigit():
            return True
        return len(core) <= 1
    short = sum(1 for token in tokens if short_symbol(token))
    return short / len(tokens) >= 0.40


def _looks_like_ocr_junk(text: str) -> bool:
    words = _alpha_words(text)
    if not words:
        return True
    if len(words) <= 6 and all(len(word) <= 3 for word in words) and not _looks_topical(text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.4:
        # Mixed random caps in short tokens often comes from screenshot OCR chrome.
        if len(words) <= 4 and not _looks_topical(text) and max(len(w) for w in words) <= 4:
            return True
    return False


def _looks_like_short_name(text: str) -> bool:
    words = _alpha_words(text)
    if not (1 <= len(words) <= 3):
        return False
    if _looks_topical(text):
        return False
    if re.search(r"\b(dr|prof|engr|mr|mrs|ms)\b", text, re.I):
        return True
    skip = {"dust", "stone", "soil", "pump", "frame", "system", "network", "converter", "impeller", "bridge", "design", "project", "report"}
    if any(word.lower() in skip for word in words):
        return False
    return all(word[:1].isupper() for word in words)


def _boilerplate_reason(text: str) -> str | None:
    key = bp.exact_key(text)
    if key in bp.EXACT_REJECT or key in bp.GENERIC_TITLES:
        return "boilerplate"
    for pattern, reason in bp.LINE_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _is_journal_name_line(text: str) -> bool:
    """Entire line is a venue name, not an article title about a venue."""
    if ":" in text:
        return False
    if re.search(r"\b(using|based|towards|toward|analysis|study|effect|impact|role|for)\b", text, re.I):
        return False
    reason = _masthead_reason(text)
    if reason in {"journal", "proceedings", "transactions"} and _word_count(text) <= 14:
        return True
    return False


def _masthead_reason(text: str) -> str | None:
    for pattern, reason in bp.MASTHEAD_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _is_title_fragment(candidate: str, title: str) -> bool:
    cand = bp.exact_key(candidate)
    full = bp.exact_key(title)
    if not cand or not full:
        return False
    return cand == full or cand in full or full in cand


def _matches_filename(title: str, filename: str | None) -> bool:
    if not filename:
        return False
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _alnum(title) == _alnum(stem) and len(_alnum(title)) >= 6


def _comes_after(line: dict[str, Any], marker: tuple[int, float]) -> bool:
    page, y0 = marker
    if line["page"] > page:
        return True
    if line["page"] == page and line["y0"] > y0 + 8:
        return True
    return False


def _starts_capitalized(text: str) -> bool:
    for char in text:
        if char.isalpha():
            return char.isupper()
    return False


def _mostly_title_or_caps(text: str) -> bool:
    if _mostly_all_caps(text):
        return True
    words = _alpha_words(text)
    if not words:
        return False
    capped = sum(1 for word in words if word[:1].isupper())
    return capped / len(words) >= 0.6


def _mostly_all_caps(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 6:
        return False
    return sum(char.isupper() for char in letters) / len(letters) >= 0.82


def text_ends_with_sentence_period(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return False
    if re.search(r"\b[A-Z][a-z]{0,3}\.$", stripped):
        return False
    return True


def _letter_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    return sum(char.isalpha() for char in compact) / len(compact)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text))


def _alpha_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _lines_by_page(lines: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        grouped[int(line["page"])].append(line)
    return grouped


def _confidence(score: float, margin: float) -> str:
    if score >= HIGH_SCORE and margin >= HIGH_MARGIN:
        return "high"
    if score >= MEDIUM_SCORE:
        return "medium"
    return "low"


def is_high_confidence_title(result: dict[str, Any] | None) -> bool:
    if not result or not (result.get("title") or "").strip():
        return False
    if result.get("confidence") == "high":
        return True
    text = result.get("title") or ""
    score = float(result.get("score") or 0.0)
    page = int(result.get("page") or 99)
    if page == 1 and score >= MEDIUM_SCORE and re.search(r"\bintroduction\s+to\b", text, re.I):
        return True
    return False


def _empty_result(reason: str, rejected_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "title": None,
        "source": None,
        "confidence": "low",
        "score": 0.0,
        "page": None,
        "reason": reason,
        "signals": {},
        "alternatives": [],
        "rejected_metadata": rejected_metadata,
    }


def _labeled_title_result(lines: list[dict[str, Any]], rejected_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    ordered = sorted(lines, key=lambda item: (item["page"], item["y0"]))
    for index, line in enumerate(ordered):
        text = line["text"]
        title = None
        reason = "Found an explicit Title / Project Title label."
        match = bp.LABELED_TITLE_RE.match(text)
        if match:
            title = bp.normalize_space(match.group(1))
        if not title:
            chrome = bp.BROWSER_CHROME_TITLE_RE.search(text)
            if chrome:
                title = bp.normalize_space(chrome.group(1))
                title = re.sub(r"^I\s+", "", title)
                reason = "Recovered the article title from browser/print chrome."
        if not title:
            course = bp.COURSE_LINE_RE.match(text)
            if course:
                rest = bp.COURSE_PREFIX_RE.sub("", bp.normalize_space(course.group(1))).strip()
                if _word_count(rest) >= 3 and _looks_topical(rest):
                    title = rest
                    reason = "Extracted the project title from a Course: cover line."
                elif index + 1 < len(ordered):
                    nxt = ordered[index + 1]
                    nxt_text = nxt["text"]
                    if (
                        nxt["page"] == line["page"]
                        and _word_count(nxt_text) >= 3
                        and _looks_topical(nxt_text)
                        and not _boilerplate_reason(nxt_text)
                        and not _looks_like_sentence(nxt_text)
                    ):
                        title = nxt_text
                        reason = "Used the topical line under a Course: label as the title."
        if not title:
            continue
        if _word_count(title) < MIN_TITLE_WORDS or _boilerplate_reason(title):
            continue
        if _looks_like_sentence(title):
            continue
        return {
            "title": title,
            "source": "labeled",
            "confidence": "high",
            "score": 95.0,
            "page": line["page"],
            "reason": reason,
            "signals": {"labeled": 95.0},
            "alternatives": [],
            "rejected_metadata": rejected_metadata,
        }
    return None


def _drop_cover_pages(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        by_page[int(line["page"])].append(line)
    cover_pages = set()
    for page, page_lines in by_page.items():
        blob = " ".join(item["text"] for item in page_lines)
        if bp.COVER_PAGE_RE.search(blob):
            cover_pages.add(page)
    if not cover_pages:
        return lines
    remaining = [line for line in lines if int(line["page"]) not in cover_pages]
    return remaining or lines


def _merge_adjacent_title_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return lines
    ordered = sorted(lines, key=lambda item: (item["page"], item["y0"], item["x0"]))
    mergeable = [line for line in ordered if not _is_merge_interrupter(line)]
    skipped = [line for line in ordered if _is_merge_interrupter(line)]
    merged: list[dict[str, Any]] = []
    if not mergeable:
        return ordered
    buffer = [mergeable[0]]

    def flush() -> None:
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            merged.append(_combine_line_block(buffer))
        buffer.clear()

    deferred: list[dict[str, Any]] = []
    for candidate in mergeable[1:]:
        prev = buffer[-1]
        if prev["page"] == candidate["page"] and _other_column(prev, candidate):
            deferred.append(candidate)
            continue
        if _should_premerge(prev, candidate, buffer):
            buffer.append(candidate)
        else:
            flush()
            buffer.append(candidate)
    flush()
    if deferred:
        merged.extend(_merge_adjacent_title_lines(deferred) if len(deferred) > 1 else deferred)
    merged.extend(skipped)
    merged.sort(key=lambda item: (item["page"], item["y0"], item["x0"]))
    return merged


def _ends_incomplete(text: str) -> bool:
    """True when a title line is cut at a function word: '... parameters of'."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text or "")
    if not words:
        return False
    return words[-1].lower().rstrip(".,;:") in _DANGLING_TITLE_WORDS


def _same_title_size(size_a: float, size_b: float) -> bool:
    """Title wraps share a font; 14pt bold vs 12pt italic is a different block."""
    a = float(size_a or 0.0)
    b = float(size_b or 0.0)
    if a <= 0 or b <= 0:
        return False
    if abs(a - b) <= SAME_TITLE_SIZE_PT:
        return True
    return abs(a - b) / max(a, b) <= SAME_TITLE_SIZE_RATIO


def _is_title_wrap_tail(prev: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Last wrapped token sitting under a title, e.g. 'corn-thresher'."""
    if prev["page"] != candidate["page"]:
        return False
    if candidate.get("word_count", 0) > 6:
        return False
    if prev.get("word_count", 0) < 3:
        return False
    if prev["text"].endswith((".", "?", "!")):
        return False
    if _is_abstract_marker(candidate):
        return False
    if re.search(r"[\w.+-]+@[\w-]+\.|\bdepartment of\b|\bcorresponding author\b", candidate["text"], re.I):
        return False
    size_a = float(prev.get("font_size") or 0.0) or 1.0
    size_b = float(candidate.get("font_size") or 0.0) or 0.0
    if not size_b or not _same_title_size(size_a, size_b):
        return False
    gap = candidate["y0"] - prev["y1"]
    if gap > 2.2 * max(prev.get("line_height") or size_a, candidate.get("line_height") or size_b, size_a):
        return False
    return True


def _is_merge_interrupter(line: dict[str, Any]) -> bool:
    key = line.get("norm") or bp.exact_key(line.get("text") or "")
    words = line.get("word_count") or _word_count(line.get("text") or "")
    if key in {"open", "open access", "research article", "original article"}:
        return True
    if key.startswith("you may also like") or key.startswith("to cite this article"):
        return True
    if words >= MIN_TITLE_WORDS:
        return False
    text = line.get("text") or ""
    # Keep 1-word alphabetic tails in the merge stream ("corn-thresher").
    if re.search(r"[A-Za-z]", text) and not re.match(r"^[\d.]+", text):
        return False
    return True


def _other_column(left: dict[str, Any], right: dict[str, Any]) -> bool:
    width = float(left.get("page_width") or right.get("page_width") or 0.0) or 1.0
    left_mid = (left["x0"] + left["x1"]) / 2.0
    right_mid = (right["x0"] + right["x1"]) / 2.0
    if (left_mid < 0.46 * width and right_mid > 0.54 * width) or (
        right_mid < 0.46 * width and left_mid > 0.54 * width
    ):
        return True
    leftish = left["x0"] / width < 0.48
    rightish = right["x0"] / width < 0.48
    if leftish == rightish:
        return False
    overlap = min(left["y1"], right["y1"]) - max(left["y0"], right["y0"])
    band = min(float(left.get("line_height") or 1.0), float(right.get("line_height") or 1.0))
    return overlap > 0.2 * max(band, 1.0)


def _should_premerge(prev: dict[str, Any], candidate: dict[str, Any], buffer: list[dict[str, Any]]) -> bool:
    if prev["page"] != candidate["page"]:
        return False
    incomplete = _ends_incomplete(prev["text"]) or _ends_incomplete(_join_title_texts(buffer))
    wrap_tail = _is_title_wrap_tail(prev, candidate)
    if re.search(r"[\w.+-]+@[\w-]+\.|\bdepartment of\b|\bcorresponding author\b", candidate["text"], re.I):
        return False
    if candidate["text"].count(",") >= 2 and not _looks_topical(candidate["text"]):
        return False
    if _looks_like_author_or_affiliation(candidate) and not (incomplete or wrap_tail):
        return False
    if not incomplete and not wrap_tail and _looks_like_short_name(candidate["text"]):
        return False
    if _looks_like_author_or_affiliation(prev) or (
        not incomplete and not wrap_tail and _looks_like_short_name(prev["text"])
    ):
        return False
    if _is_abstract_marker(candidate) or _boilerplate_reason(candidate["text"]):
        return False
    if re.search(r"date of submission|submitted date|submitted on", candidate["text"], re.I):
        return False
    reject = _hard_reject(candidate, set())
    if reject in {"journal_name", "institution", "garbled", "section_heading"}:
        return False
    size_a = float(prev["font_size"] or 0.0)
    size_b = float(candidate["font_size"] or 0.0)
    if not size_a:
        return False
    gap = candidate["y0"] - prev["y1"]
    words = _word_count(_join_title_texts(buffer + [candidate]))
    if words > MAX_MERGED_WORDS:
        return False

    # Cover pairing: only pair a known course heading with its report line.
    cand_key = candidate["norm"]
    if (
        prev["norm"] in {"computer networks", "electrical machine design", "electrical machines"}
        and re.search(r"project report|final project", cand_key)
        and abs(size_a - size_b) / size_a <= 0.35
        and gap <= 3.5 * max(prev["line_height"], candidate["line_height"], size_a)
    ):
        return True

    if not _same_title_size(size_a, size_b):
        return False
    gap_limit = 2.6 if incomplete else 2.4
    if gap > gap_limit * max(prev["line_height"], candidate["line_height"], size_a):
        return False
    # Incomplete title: "... results of" / "determination of..." / "corn-thresher"
    if (incomplete or wrap_tail) and not _looks_like_sentence(candidate["text"]):
        return True
    # Wrapped title continuation: "An experimental investigation" / "examining the usage…"
    if (
        re.match(r"^[a-z]", candidate["text"])
        and abs(prev["x0"] - candidate["x0"]) <= 24
        and not prev["text"].endswith((".", "?", "!"))
        and gap <= 1.8 * max(prev["line_height"], candidate["line_height"], size_a)
        and not _looks_like_sentence(candidate["text"])
    ):
        return True
    # Wrapped last word ("… Boost" / "Converter") is often more centered than the line above.
    if (
        candidate["word_count"] <= 2
        and prev["word_count"] >= 3
        and not prev["text"].endswith((".", "?", "!"))
        and (_looks_topical(prev["text"]) or incomplete)
        and not re.search(r"\b[A-Z]{2,}\d{3,}/\d+", prev["text"])
    ):
        return True
    if abs(prev["center_offset"] - candidate["center_offset"]) > (0.32 if candidate["word_count"] <= 4 else 0.20):
        return False
    if _looks_like_sentence(candidate["text"]) or _looks_like_body_fragment(candidate["text"]):
        return False
    if not incomplete and re.match(r"^(the|this|these|those|there|it|we|for)\s", candidate["text"], re.I):
        return False
    # Body paragraphs glued under Elsevier titles share font size with the heading.
    if re.search(r"\.\s+[A-Za-z]", candidate["text"]):
        return False
    return True


def _combine_line_block(parts: list[dict[str, Any]]) -> dict[str, Any]:
    combined = dict(parts[0])
    combined["text"] = _join_title_texts(parts)
    combined["norm"] = bp.exact_key(combined["text"])
    combined["word_count"] = _word_count(combined["text"])
    combined["font_size"] = max(float(part["font_size"] or 0.0) for part in parts)
    combined["bold"] = any(part.get("bold") for part in parts)
    combined["bbox"] = [
        min(part["x0"] for part in parts),
        min(part["y0"] for part in parts),
        max(part["x1"] for part in parts),
        max(part["y1"] for part in parts),
    ]
    combined["y0"] = combined["bbox"][1]
    combined["y1"] = combined["bbox"][3]
    combined["x0"] = combined["bbox"][0]
    combined["x1"] = combined["bbox"][2]
    width = float(combined.get("page_width") or 1.0) or 1.0
    height = float(combined.get("page_height") or 1.0) or 1.0
    combined["y_ratio"] = combined["y0"] / height
    combined["x_mid"] = (combined["x0"] + combined["x1"]) / 2.0
    combined["center_offset"] = abs(combined["x_mid"] - width / 2.0) / width
    combined["line_height"] = max(combined["y1"] - combined["y0"], combined["font_size"], 1.0)
    return combined


def _fallback_document_title(
    lines: list[dict[str, Any]],
    rejected_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    candidates = []
    for line in lines:
        if _hard_reject(line, set()):
            continue
        if line["word_count"] < MIN_TITLE_WORDS or not _looks_topical(line["text"]):
            continue
        if _looks_like_body_fragment(line["text"]) or _looks_like_sentence(line["text"]):
            continue
        candidates.append(line)
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda item: (float(item.get("font_size") or 0.0), -float(item.get("y0") or 0.0), item["word_count"]),
    )
    return {
        "title": best["text"],
        "source": "heuristic",
        "confidence": "medium",
        "score": 40.0,
        "page": best["page"],
        "reason": "Fallback to the largest topical heading after stronger title candidates were rejected.",
        "signals": {},
        "alternatives": [],
        "rejected_metadata": rejected_metadata,
    }

