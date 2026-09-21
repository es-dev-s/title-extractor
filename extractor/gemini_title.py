"""
Gemini fallback for title verification / extraction.

Native metadata + layout heuristics always run first. This module is
invoked only after that step, and is skipped when the heuristic
confidence is already high and the candidate is not a known-bad heading.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from extractor import boilerplate as bp

logger = logging.getLogger("extractor.gemini")


class GeminiUnavailableError(RuntimeError):
    """SDK missing, auth failed, or no usable model."""

MODEL_NAME = "gemini-flash-latest"
# Verified against genai.list_models() generateContent flash IDs (2026-09-21).
FALLBACK_MODELS = (
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
)
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.4
MAX_PAGE_CHARS = 800
PLACEHOLDER_KEYS = {"", "YOUR_API_KEY_HERE", "your_api_key_here"}
REQUEST_TIMEOUT_SEC = 25.0
TRANSIENT_RETRIES = 2
BACKOFF_SEC = 0.7

_ENV_PATHS = (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
)
_ENV_MTIMES: dict[str, float] = {}
_CACHED_API_KEY: str | None = None
_KEY_CACHE_WARM = False
_STATS: Counter[str] = Counter()

_BAD_TITLE_RE = re.compile(
    r"^(?:table of contents|contents|abstract|chapter(?:\s+\d+)?|references|"
    r"bibliography|acknowledg(?:e)?ments?|appendix|"
    r"(?:chapter\s+\d+\s+)?introduction|conclusion|"
    r"list of (?:figures?|tables?|abbreviations|contents)|"
    r"(?:examiner'?s?\s+)?certificate(?:\s+of\s+approval)?|"
    r"declaration(?:\s+of\s+the\s+(?:student|candidate))?|"
    r"page\s*\d+|\d+)$",
    re.I,
)
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d+(?:\s*/\s*\d+)?$", re.I)
_INTRO_TO_RE = re.compile(r"\bintroduction\s+to\s+", re.I)
_TITLE_CUT_RE = re.compile(
    r"(?<=[A-Za-z0-9'’])\s+(?:The|This|These|Those|That|It|We|Given)\s+",
    re.I,
)
_TITLE_RULES = (
    "- Copy a title verbatim from the page text. Do not invent or paraphrase.\n"
    '- Bare section labels are NOT titles: "Introduction", "Chapter 1", '
    '"Chapter 1 Introduction", "1.1 Introduction", "Abstract", "References".\n'
    '- "Introduction" and "Introduction to …" are different. '
    'Phrases such as "Introduction to Harvester" ARE valid titles if they appear in the block.\n'
    '- Numbered body headings such as "5.2.2.2 MACHINE THRESHING" followed by a sentence '
    "are NOT the document title.\n"
    "- If the candidate is wrong, still return a corrected title copied from the block. "
    "Only use null/empty if the block has no title-like phrase at all.\n"
)


def apply_gemini_layer(
    title_info: dict[str, Any],
    lines: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """
    Optionally verify or replace the heuristic title with Gemini.

    Never mutates heuristic scoring. Returns a new/updated title_info dict.
    """
    result = dict(title_info or {})
    candidate = (result.get("title") or "").strip() or None
    confidence = _confidence_01(result)
    page_text = page1_top_text(lines, title_page=result.get("page"))
    known_bad = bool(candidate and is_known_bad_title(candidate))

    result["title_confidence_01"] = round(confidence, 3)
    result["gemini_mode"] = "skip"
    result["gemini_used"] = False
    result["gemini_input_text"] = None
    result["gemini_candidate"] = None
    result["gemini_title"] = None
    result["gemini_stats"] = dict(_STATS)

    incomplete = bool(candidate and _title_looks_incomplete(candidate))
    if confidence >= HIGH_CONFIDENCE and candidate and not known_bad and not incomplete:
        _record("skip")
        result["gemini_stats"] = dict(_STATS)
        return result

    api_key = load_gemini_api_key()
    if not api_key:
        result["gemini_mode"] = "unavailable"
        _record("unavailable", reason="missing_key")
        warnings.append("Gemini skipped: set GEMINI_API_KEY in extractor/.env")
        result["gemini_stats"] = dict(_STATS)
        return result

    if not page_text.strip():
        result["gemini_mode"] = "unavailable"
        _record("unavailable", reason="empty_page_text")
        warnings.append("Gemini skipped: no page text to send.")
        result["gemini_stats"] = dict(_STATS)
        return result

    result["gemini_input_text"] = page_text
    result["gemini_candidate"] = candidate
    try:
        if candidate and confidence >= MEDIUM_CONFIDENCE:
            updated = _verify(candidate, page_text, result, api_key)
            if not _gemini_supplied_title(updated):
                updated = _fill_missing_gemini_title(updated, page_text, api_key, candidate)
        else:
            updated = _extract(page_text, result, api_key)
            if not _gemini_supplied_title(updated):
                updated = _fill_missing_gemini_title(updated, page_text, api_key, candidate)
    except GeminiUnavailableError as exc:
        warnings.append(f"Gemini unavailable: {exc}")
        result["gemini_mode"] = "unavailable"
        _record("unavailable", error=type(exc).__name__)
        result["gemini_stats"] = dict(_STATS)
        return result
    except Exception as exc:
        warnings.append(f"Gemini failed: {exc}")
        result["gemini_mode"] = "error"
        _record("error", error=type(exc).__name__)
        result["gemini_stats"] = dict(_STATS)
        return result

    mode = updated.get("gemini_mode") or "error"
    success = _gemini_supplied_title(updated)
    _record(str(mode), success=success)
    if success:
        _record(f"{mode}_ok")
    updated["gemini_stats"] = dict(_STATS)
    return updated


def load_gemini_api_key() -> str | None:
    """Read .env files when they change; otherwise reuse the in-memory key."""
    global _CACHED_API_KEY, _KEY_CACHE_WARM
    changed = _load_env_files()
    if _KEY_CACHE_WARM and not changed:
        return _CACHED_API_KEY
    key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if key in PLACEHOLDER_KEYS:
        key = ""
    _CACHED_API_KEY = key or None
    _KEY_CACHE_WARM = True
    return _CACHED_API_KEY


def gemini_is_configured() -> bool:
    return load_gemini_api_key() is not None


def gemini_counters() -> dict[str, int]:
    return dict(_STATS)


def page1_top_text(
    lines: list[dict[str, Any]],
    max_chars: int = MAX_PAGE_CHARS,
    title_page: int | None = None,
) -> str:
    if not lines:
        return ""
    usable = _content_lines(lines)
    wanted = int(title_page or 0)
    page_lines = [line for line in usable if int(line.get("page") or 0) == wanted] if wanted else []
    if not page_lines:
        earliest = min(int(line.get("page") or 99) for line in usable)
        page_lines = [line for line in usable if int(line.get("page") or 0) == earliest]
    if not page_lines:
        return ""
    text = _join_gemini_lines(_top_heading_lines(page_lines, y_limit=0.72 if _page_is_ocr(page_lines) else 0.30), max_chars)
    if text and len(text) >= 40:
        return text
    extra = _join_gemini_lines(_largest_heading_lines(page_lines), max_chars)
    if not text:
        return extra
    if extra and extra not in text:
        return (text + "\n" + extra)[:max_chars]
    return text


def _page_is_ocr(page_lines: list[dict[str, Any]]) -> bool:
    return any(line.get("source") == "ocr" for line in page_lines)


def _join_gemini_lines(chosen: list[dict[str, Any]], max_chars: int) -> str:
    texts = []
    for line in chosen:
        text = (line.get("text") or "").strip()
        if not text or _skip_gemini_line(text):
            continue
        texts.append(text)
    return bp.normalize_space("\n".join(texts))[:max_chars]


def _top_heading_lines(page_lines: list[dict[str, Any]], y_limit: float) -> list[dict[str, Any]]:
    ordered = sorted(
        page_lines,
        key=lambda item: (float(item.get("y0") or _bbox_y0(item)), float(item.get("x0") or 0.0)),
    )
    top = [line for line in ordered if _y_ratio(line) <= y_limit]
    if top:
        return top
    cutoff = max(1, int(len(ordered) * y_limit))
    return ordered[:cutoff]


def _largest_heading_lines(page_lines: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(
        page_lines,
        key=lambda item: (float(item.get("font_size") or 0.0), -_y_ratio(item)),
        reverse=True,
    )
    picked = []
    for line in ranked:
        text = (line.get("text") or "").strip()
        if not text or _skip_gemini_line(text):
            continue
        if re.match(r"^[a-z]", text) and len(text.split()) >= 8:
            continue
        picked.append(line)
        if len(picked) >= limit:
            break
    picked.sort(key=lambda item: (float(item.get("y0") or _bbox_y0(item)), float(item.get("x0") or 0.0)))
    return picked


def _skip_gemini_line(text: str) -> bool:
    if re.search(r"https?://|\bwww\.", text, re.I):
        return True
    first = next((char for char in text if char.isalpha()), "")
    return bool(first.islower() and len(text.split()) >= 8)


def is_known_bad_title(text: str) -> bool:
    key = bp.exact_key(text)
    if not key:
        return True
    if re.search(r"\bintroduction\s+to\b", text, re.I):
        return False
    if key in bp.EXACT_REJECT or key in bp.GENERIC_TITLES:
        return True
    if _BAD_TITLE_RE.match(key) or _PAGE_NUMBER_RE.match(key):
        return True
    if re.fullmatch(r"[\d\s./:-]+", text.strip()):
        return True
    if re.match(r"^chapter(?:\s+\d+)?$", text.strip(), re.I):
        return True
    for pattern, _reason in bp.LINE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _gemini_supplied_title(result: dict[str, Any]) -> bool:
    title = (result.get("gemini_title") or "").strip()
    if not title or is_known_bad_title(title):
        return False
    if result.get("gemini_is_correct"):
        return True
    return result.get("source") == "gemini"


def _fill_missing_gemini_title(
    result: dict[str, Any],
    page_text: str,
    api_key: str,
    candidate: str | None,
) -> dict[str, Any]:
    updated = dict(result)
    updated["gemini_candidate"] = candidate
    updated["gemini_input_text"] = page_text

    local = _fallback_title_from_block(page_text)
    if local:
        return _apply_recovered_title(
            updated,
            local,
            page_text,
            "Recovered a verbatim 'Introduction to …' title from the page-1 block after Gemini returned none.",
        )

    if updated.get("gemini_mode") == "extract":
        return updated

    extracted = _extract(page_text, updated, api_key)
    extracted["gemini_candidate"] = candidate
    extracted["gemini_mode"] = "verify+extract"
    extracted["gemini_is_correct"] = False
    if _gemini_supplied_title(extracted):
        return extracted

    local = _fallback_title_from_block(page_text)
    if local:
        return _apply_recovered_title(
            extracted,
            local,
            page_text,
            "Recovered a verbatim title from the page-1 block after Gemini returned none.",
        )
    return extracted


def _apply_recovered_title(
    result: dict[str, Any],
    title: str,
    page_text: str,
    reason: str,
) -> dict[str, Any]:
    if not title or not _appears_in_block(title, page_text) or is_known_bad_title(title):
        return result
    result["title"] = title
    result["source"] = "gemini"
    result["confidence"] = "high" if result.get("gemini_mode") in {"verify", "verify+extract"} else "medium"
    result["gemini_title"] = title
    result["gemini_used"] = True
    result["reason"] = _append_reason(result.get("reason"), reason)
    return result


def _fallback_title_from_block(page_text: str) -> str | None:
    """Prefer an 'Introduction to …' span over a bare Introduction heading."""
    match = _INTRO_TO_RE.search(page_text or "")
    if not match:
        return None
    rest = (page_text or "")[match.end() :]
    rest = re.split(r"[.?!;\n]", rest, maxsplit=1)[0]
    rest = _TITLE_CUT_RE.split(rest, maxsplit=1)[0]
    rest = bp.normalize_space(rest).rstrip(".,;:")
    words = rest.split()
    if not words:
        return None
    title = bp.normalize_space(f"{match.group(0)} {' '.join(words[:10])}")
    if is_known_bad_title(title) or not _appears_in_block(title, page_text):
        return None
    return title


def _confidence_01(title_info: dict[str, Any]) -> float:
    """Map existing high/medium/low labels onto 0-1 without changing heuristic weights."""
    title = (title_info.get("title") or "").strip()
    if not title:
        return 0.0
    source = title_info.get("source")
    if source in {"metadata", "labeled"}:
        return 1.0
    label = title_info.get("confidence") or "low"
    if label == "high":
        return 0.90
    if label == "medium":
        return 0.62
    return 0.20


def _verify(candidate: str, page_text: str, title_info: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = _generate(
        api_key,
        (
            "You are checking a PDF title candidate from a layout heuristic.\n\n"
            f"Candidate title:\n{candidate}\n\n"
            f"Text from the top of page 1 (only use this block):\n{page_text}\n\n"
            "Return JSON only with this shape:\n"
            '{"is_correct": false, "corrected_title": "verbatim title from the block"}\n\n'
            "Rules:\n"
            "- is_correct is true if the candidate is the document title or a faithful substring of it.\n"
            "- If false, corrected_title MUST be copied verbatim from the page text. "
            "Do not paraphrase, translate, or invent.\n"
            f"{_TITLE_RULES}"
        ),
    )
    result = dict(title_info)
    result["gemini_used"] = True
    result["gemini_mode"] = "verify"
    result["gemini_input_text"] = page_text
    result["gemini_candidate"] = candidate
    is_correct = bool(payload.get("is_correct")) and not is_known_bad_title(candidate)
    corrected = _clean_model_title(payload.get("corrected_title"))
    result["gemini_title"] = candidate if is_correct else corrected
    result["gemini_is_correct"] = is_correct
    if is_correct:
        result["reason"] = _append_reason(result.get("reason"), "Gemini verified the heuristic title.")
        return result
    if corrected and _appears_in_block(corrected, page_text) and not is_known_bad_title(corrected):
        result["title"] = corrected
        result["source"] = "gemini"
        result["confidence"] = "high"
        result["reason"] = _append_reason(result.get("reason"), "Gemini replaced the heuristic title with a verbatim page-1 span.")
        return result
    result["reason"] = _append_reason(result.get("reason"), "Gemini rejected the candidate but offered no usable verbatim correction.")
    return result


def _extract(page_text: str, title_info: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = _generate(
        api_key,
        (
            "Extract the document title from this page-1 text.\n\n"
            f"Text (only use this block):\n{page_text}\n\n"
            "Return JSON only with this shape:\n"
            '{"title": "exact substring from the block"}\n\n'
            "Rules:\n"
            "- Copy the title verbatim from the block. Never invent or paraphrase a title that is not present.\n"
            f"{_TITLE_RULES}"
            '- If no title-like phrase is present, return {"title": ""}.\n'
        ),
    )
    result = dict(title_info)
    result["gemini_used"] = True
    result["gemini_mode"] = "extract"
    result["gemini_input_text"] = page_text
    result["gemini_candidate"] = title_info.get("gemini_candidate")
    extracted = _clean_model_title(payload.get("title"))
    result["gemini_title"] = extracted
    if extracted and _appears_in_block(extracted, page_text) and not is_known_bad_title(extracted):
        result["title"] = extracted
        result["source"] = "gemini"
        result["confidence"] = "medium"
        result["reason"] = _append_reason(result.get("reason"), "Gemini extracted a verbatim title from the top of page 1.")
    else:
        result["reason"] = _append_reason(
            result.get("reason"),
            "Gemini extract returned nothing that literally appears in the page-1 block.",
        )
    return result


def _generate(api_key: str, prompt: str) -> dict[str, Any]:
    try:
        import google.generativeai as genai
        from google.api_core import exceptions as gexc
    except ImportError as exc:
        raise GeminiUnavailableError(
            "google-generativeai is not installed in this Python. Use .venv."
        ) from exc

    genai.configure(api_key=api_key)
    last_error: Exception | None = None
    for index, model_name in enumerate(_model_candidates()):
        try:
            payload = _generate_with_retries(genai, model_name, prompt)
            if index:
                _record("model_fallback", model=model_name)
            return payload
        except _permanent_model_errors(gexc) as exc:
            last_error = exc
            _record("model_missing", model=model_name)
            continue
        except _fatal_auth_errors(gexc) as exc:
            raise GeminiUnavailableError(f"Gemini auth failed ({type(exc).__name__})") from exc
        except Exception as exc:
            last_error = exc
            raise
    if last_error:
        raise GeminiUnavailableError(
            f"No usable Gemini model ({type(last_error).__name__})"
        ) from last_error
    return {}


def _generate_with_retries(genai: Any, model_name: str, prompt: str) -> dict[str, Any]:
    from google.api_core import exceptions as gexc

    last_error: Exception | None = None
    attempts = TRANSIENT_RETRIES + 1
    for attempt in range(attempts):
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            response = model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT_SEC},
            )
            raw = (getattr(response, "text", None) or "").strip()
            return _parse_json(raw)
        except _transient_errors(gexc) as exc:
            last_error = exc
            _record("retry", model=model_name, attempt=attempt + 1)
            if attempt + 1 >= attempts:
                break
            time.sleep(BACKOFF_SEC * (2 ** attempt))
        except _permanent_model_errors(gexc):
            raise
        except _fatal_auth_errors(gexc):
            raise
    assert last_error is not None
    raise last_error


def _transient_errors(gexc: Any) -> tuple[type[BaseException], ...]:
    return (
        gexc.TooManyRequests,
        gexc.ResourceExhausted,
        gexc.ServiceUnavailable,
        gexc.DeadlineExceeded,
        gexc.InternalServerError,
        gexc.Aborted,
        TimeoutError,
        ConnectionError,
    )


def _permanent_model_errors(gexc: Any) -> tuple[type[BaseException], ...]:
    return (gexc.NotFound,)


def _fatal_auth_errors(gexc: Any) -> tuple[type[BaseException], ...]:
    return (gexc.PermissionDenied, gexc.Unauthenticated, gexc.Forbidden)


def _model_candidates() -> list[str]:
    preferred = (os.environ.get("GEMINI_MODEL") or "").strip() or MODEL_NAME
    ordered: list[str] = []
    for name in (preferred, *FALLBACK_MODELS):
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        data = json.loads(match.group(0))
    return data if isinstance(data, dict) else {}


def _clean_model_title(value: Any) -> str | None:
    if value is None:
        return None
    text = bp.normalize_space(str(value))
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def _appears_in_block(title: str, block: str) -> bool:
    needle = re.sub(r"\s+", " ", title).strip().lower()
    haystack = re.sub(r"\s+", " ", block).strip().lower()
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    compact_title = re.sub(r"[^a-z0-9]+", "", needle)
    compact_block = re.sub(r"[^a-z0-9]+", "", haystack)
    return len(compact_title) >= 8 and compact_title in compact_block


def _append_reason(existing: str | None, extra: str) -> str:
    existing = (existing or "").strip()
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}"


def _y_ratio(line: dict[str, Any]) -> float:
    if line.get("y_ratio") is not None:
        return float(line["y_ratio"])
    height = float(line.get("page_height") or 0.0) or 1.0
    return _bbox_y0(line) / height


def _bbox_y0(line: dict[str, Any]) -> float:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    try:
        return float(bbox[1])
    except (TypeError, IndexError, ValueError):
        return 0.0


def _load_env_files() -> bool:
    """Load GEMINI_* from .env files only when mtime changes. Returns True if reloaded."""
    changed = False
    for path in _ENV_PATHS:
        key = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            if key in _ENV_MTIMES:
                _ENV_MTIMES.pop(key, None)
                changed = True
            continue
        if _ENV_MTIMES.get(key) == mtime:
            continue
        _ENV_MTIMES[key] = mtime
        changed = True
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name:
                os.environ[name] = value
    return changed


def _content_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        by_page.setdefault(int(line.get("page") or 0), []).append(line)
    kept: list[dict[str, Any]] = []
    for page in sorted(by_page):
        blob = " ".join(item.get("text") or "" for item in by_page[page])
        if bp.COVER_PAGE_RE.search(blob):
            continue
        kept.extend(by_page[page])
    return kept or list(lines)


def _title_looks_incomplete(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text or "")
    if not words:
        return True
    dangling = {
        "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
        "by", "from", "as", "at", "using",
    }
    return words[-1].lower().rstrip(".,;:") in dangling


def _record(event: str, **info: Any) -> None:
    _STATS[event] += 1
    extras = " ".join(f"{key}={value}" for key, value in info.items() if value is not None)
    logger.info("gemini %s %s", event, extras)
