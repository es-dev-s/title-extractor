"""
Gemini title extraction.

The pipeline extracts native/OCR text from the first few pages, then this
module sends that text block to Gemini. Keys rotate on quota errors.
Heuristic scoring is not used here; the pipeline may call it only after
every Gemini key is out of quota.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("extractor.gemini")


class GeminiUnavailableError(RuntimeError):
    """SDK missing, auth failed, or no usable model."""


class GeminiQuotaExhaustedError(GeminiUnavailableError):
    """Every configured Gemini API key has hit quota."""


class _KeyQuotaError(RuntimeError):
    """This specific API key is out of quota."""


MODEL_NAME = "gemini-3.6-flash"
# Retired 2.0/2.5 Flash IDs 404. Stay on the current Flash SKU Google returns.
MODEL_FALLBACKS = (
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
)
_RETIRED_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
}
MAX_FRONT_CHARS = 1800
MAX_CHARS_PER_PAGE = 500
# Titles sit near the top; send the upper 40% of each extracted page, not the body.
TOP_PAGE_FRACTION = 0.40
PLACEHOLDER_KEYS = {"", "YOUR_API_KEY_HERE", "your_api_key_here"}
GEMINI_BUDGET_SEC = 60.0
REQUEST_TIMEOUT_SEC = 45.0
MIN_CALL_SEC = 1.0
# Free-tier Flash is ~10–15 RPM. Space calls so one PDF cannot dump 4–6 requests.
MIN_CALL_INTERVAL_SEC = 6.5
RPM_COOLDOWN_SEC = 60.0
MAX_OUTPUT_TOKENS = 512
# Gemini 3 Flash spends the output budget on hidden thinking. 0 keeps the title JSON as the only output.
SPARSE_FRONT_CHARS = 120

_ENV_PATHS = (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
)
_ENV_MTIMES: dict[str, float] = {}
_CACHED_KEYS: list[tuple[str, str]] = []
_KEY_CACHE_WARM = False
_EXHAUSTED_KEYS: set[str] = set()
_RPM_COOLDOWN_UNTIL: dict[str, float] = {}
_RR_INDEX = 0
_LAST_CALL_AT = 0.0
_STATS: Counter[str] = Counter()
_KEY_ENV_RE = re.compile(r"^GEMINI_API_KEY(?:_(\d+))?$")

SYSTEM_PROMPT = (
    "Extract the document's official title from the text. "
    "Return JSON only: {\"title\":\"verbatim title\",\"page\":1,\"reason\":\"short\"}. "
    "Copy the title from the text; do not invent or paraphrase. "
    "Join wrapped title lines with single spaces. "
    "Not a title by themselves: journal names, authors, affiliations, Abstract, Contents, "
    "Certificate, Declaration, page numbers, DOIs, Research Article, "
    "bare Introduction, or a bare 'Chapter 1' with no topic after it. "
    "\"Introduction to …\" is a valid title if that is the work's name. "
    "If a cover title appears before Abstract or Contents, use that and ignore later chapter headings. "
    "If there is no cover title, a named heading is the title. "
    "From 'CHAPTER 1: TRANSFORMER CONSTRUCTION' or a later line 'TRANSFORMER CONSTRUCTION', "
    "return TRANSFORMER CONSTRUCTION. Drop only the 'Chapter N:' prefix; keep the topic. "
    "Do not return an empty title when a topic heading like that is in the text. "
    "If none, return {\"title\":\"\",\"page\":null,\"reason\":\"no title found\"}."
)


def apply_gemini_layer(
    lines: list[dict[str, Any]],
    warnings: list[str],
    page_reports: list[dict[str, Any]] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Send extracted front-page text to Gemini and return a title_info dict."""
    page_counts = _page_char_counts(page_reports, lines)
    page_text, sent_chars, sent_by_page = _front_text_block(lines, max_chars=MAX_FRONT_CHARS)
    for item in page_counts:
        item["sent_char_count"] = int(sent_by_page.get(int(item.get("page") or 0), 0))
    result = _blank_title_info(page_counts, page_text, sent_chars)

    if not page_text.strip():
        result["gemini_mode"] = "unavailable"
        _record("unavailable", reason="empty_page_text")
        warnings.append("Gemini skipped: no extracted text to send.")
        result["gemini_stats"] = dict(_STATS)
        result["reason"] = "No extracted text from the first pages."
        return result

    keys = load_gemini_api_keys()
    result["gemini_key_count"] = len(keys)
    result["gemini_keys_remaining"] = _live_key_count(keys)
    if not keys:
        result["gemini_mode"] = "unavailable"
        _record("unavailable", reason="missing_key")
        warnings.append("Gemini skipped: set GEMINI_API_KEY in extractor/.env")
        result["gemini_stats"] = dict(_STATS)
        result["reason"] = "Gemini API key is not configured."
        return result

    _respect_rpm_gap()
    deadline = time.monotonic() + GEMINI_BUDGET_SEC
    prompt = page_text
    try:
        payload, key_name = _generate(prompt, deadline)
    except GeminiQuotaExhaustedError as exc:
        warnings.append(str(exc))
        result["gemini_mode"] = "quota_exhausted"
        result["quota_exhausted"] = True
        result["gemini_keys_remaining"] = 0
        _record("quota_exhausted")
        result["gemini_stats"] = dict(_STATS)
        result["reason"] = str(exc)
        return result
    except GeminiUnavailableError as exc:
        warnings.append(f"Gemini unavailable: {exc}")
        result["gemini_mode"] = "unavailable"
        _record("unavailable", error=type(exc).__name__)
        result["gemini_stats"] = dict(_STATS)
        result["reason"] = f"Gemini unavailable: {exc}"
        return result
    except Exception as exc:
        warnings.append(f"Gemini failed: {exc}")
        result["gemini_mode"] = "error"
        _record("error", error=type(exc).__name__)
        result["gemini_stats"] = dict(_STATS)
        result["reason"] = f"Gemini failed: {exc}"
        return result

    title = _clean_model_title(payload.get("title"))
    page = _clean_page(payload.get("page"))
    model_reason = _clean_model_title(payload.get("reason")) or ""

    result["gemini_used"] = True
    result["gemini_mode"] = "extract"
    result["gemini_title"] = title
    result["gemini_input_text"] = page_text
    result["gemini_key_used"] = key_name
    result["gemini_keys_remaining"] = _live_key_count(load_gemini_api_keys())

    if title:
        result["title"] = title
        result["source"] = "gemini"
        result["confidence"] = "high"
        result["page"] = page
        result["reason"] = model_reason or "Gemini extracted the title from the top 40% of the first pages."
        _record("extract_ok")
    else:
        result["reason"] = model_reason or "Gemini did not find a title in the extracted text."
        _record("extract")
    result["gemini_stats"] = dict(_STATS)
    return result


def load_gemini_api_keys() -> list[tuple[str, str]]:
    """Return (env_name, key) pairs from GEMINI_API_KEY, GEMINI_API_KEY_2, ..."""
    global _CACHED_KEYS, _KEY_CACHE_WARM
    changed = _load_env_files()
    if _KEY_CACHE_WARM and not changed:
        return list(_CACHED_KEYS)

    found: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for name, raw in os.environ.items():
        match = _KEY_ENV_RE.match(name)
        if not match:
            continue
        value = _clean_key(raw)
        if not value or value in seen:
            continue
        suffix = match.group(1)
        order = 1 if suffix is None else int(suffix)
        found.append((order, name, value))
        seen.add(value)

    extra = os.environ.get("GEMINI_API_KEYS") or ""
    for index, part in enumerate(_split_key_list(extra), start=100):
        value = _clean_key(part)
        if not value or value in seen:
            continue
        found.append((index, f"GEMINI_API_KEYS_{index}", value))
        seen.add(value)

    found.sort(key=lambda item: (item[0], item[1]))
    _CACHED_KEYS = [(name, value) for _, name, value in found]
    _KEY_CACHE_WARM = True
    stale = {key for key in _EXHAUSTED_KEYS if key not in seen}
    _EXHAUSTED_KEYS.difference_update(stale)
    return list(_CACHED_KEYS)


def load_gemini_api_key() -> str | None:
    keys = load_gemini_api_keys()
    for _, value in keys:
        if value not in _EXHAUSTED_KEYS:
            return value
    return keys[0][1] if keys else None


def gemini_is_configured() -> bool:
    return bool(load_gemini_api_keys())


def gemini_deadline(budget_sec: float = GEMINI_BUDGET_SEC) -> float:
    return time.monotonic() + budget_sec


def gemini_budget_remaining(deadline: float | None) -> float:
    if deadline is None:
        return GEMINI_BUDGET_SEC
    return max(0.0, deadline - time.monotonic())


def front_text_char_count(lines: list[dict[str, Any]]) -> int:
    _, sent_chars, _ = _front_text_block(lines)
    return sent_chars


def gemini_counters() -> dict[str, int]:
    return dict(_STATS)


def _blank_title_info(
    page_counts: list[dict[str, Any]],
    page_text: str,
    sent_chars: int,
) -> dict[str, Any]:
    return {
        "title": None,
        "source": None,
        "confidence": "low",
        "score": None,
        "page": None,
        "reason": "",
        "signals": {},
        "alternatives": [],
        "rejected_metadata": None,
        "title_confidence_01": None,
        "gemini_mode": "skip",
        "gemini_used": False,
        "gemini_input_text": page_text or None,
        "gemini_input_chars": sent_chars,
        "gemini_candidate": None,
        "gemini_title": None,
        "gemini_is_correct": None,
        "gemini_stats": dict(_STATS),
        "gemini_key_count": 0,
        "gemini_keys_remaining": 0,
        "gemini_key_used": None,
        "quota_exhausted": False,
        "page_char_counts": page_counts,
        "front_text_chars": sum(int(item.get("char_count") or 0) for item in page_counts),
    }


def _page_char_counts(
    page_reports: list[dict[str, Any]] | None,
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if page_reports:
        return [
            {
                "page": int(item.get("page") or 0),
                "char_count": int(item.get("char_count") or 0),
                "kind": item.get("kind"),
                "method": item.get("method"),
            }
            for item in page_reports
        ]
    counts: dict[int, int] = {}
    for line in lines:
        page = int(line.get("page") or 0)
        counts[page] = counts.get(page, 0) + len(line.get("text") or "")
    return [
        {"page": page, "char_count": counts[page], "kind": None, "method": None}
        for page in sorted(counts)
    ]


def _front_text_block(
    lines: list[dict[str, Any]],
    max_chars: int = MAX_FRONT_CHARS,
    top_fraction: float = TOP_PAGE_FRACTION,
) -> tuple[str, int, dict[int, int]]:
    """Keep only lines whose top edge sits in the upper fraction of the page."""
    if not lines:
        return "", 0, {}

    by_page: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        by_page.setdefault(int(line.get("page") or 0), []).append(line)

    chunks: list[str] = []
    sent_by_page: dict[int, int] = {}
    used = 0
    for page_number in sorted(by_page):
        page_lines = sorted(
            by_page[page_number],
            key=lambda item: (_line_y_ratio(item), _line_x0(item)),
        )
        top_lines = [item for item in page_lines if _line_y_ratio(item) <= top_fraction]
        if not top_lines:
            top_lines = page_lines[: max(1, (len(page_lines) + 1) // 2)]
        body = "\n".join((item.get("text") or "").strip() for item in top_lines).strip()
        if not body:
            sent_by_page[page_number] = 0
            continue
        header = f"--- page {page_number} (top {int(top_fraction * 100)}%) ---\n"
        remaining = max_chars - used
        if remaining <= len(header):
            break
        snippet = body[: min(len(body), remaining - len(header), MAX_CHARS_PER_PAGE)]
        chunk = header + snippet
        chunks.append(chunk)
        sent_by_page[page_number] = len(snippet)
        used += len(chunk)
        if used >= max_chars:
            break
    text = "\n\n".join(chunks).strip()
    return text, len(text), sent_by_page


def _line_y_ratio(line: dict[str, Any]) -> float:
    height = float(line.get("page_height") or 0.0) or 1.0
    bbox = line.get("bbox") or [0, 0, 0, 0]
    try:
        y0 = float(bbox[1])
    except (TypeError, IndexError, ValueError):
        y0 = 0.0
    return y0 / height


def _line_x0(line: dict[str, Any]) -> float:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    try:
        return float(bbox[0])
    except (TypeError, IndexError, ValueError):
        return 0.0


def _generate(prompt: str, deadline: float) -> tuple[dict[str, Any], str]:
    try:
        import google.generativeai as genai
        from google.api_core import exceptions as gexc
    except ImportError as exc:
        raise GeminiUnavailableError(
            "google-generativeai is not installed in this Python. Use .venv."
        ) from exc

    chosen_keys = _ordered_live_keys()
    if not chosen_keys:
        raise GeminiQuotaExhaustedError(
            "All Gemini API keys are out of quota or cooling down from rate limits."
        )
    last_block: Exception | None = None
    blocked = 0
    for key_name, api_key in chosen_keys:
        genai.configure(api_key=api_key)
        last_missing: Exception | None = None
        try:
            for index, model_name in enumerate(_model_candidates()):
                try:
                    payload = _generate_once(api_key, gexc, model_name, prompt, deadline)
                    if index:
                        _record("model_fallback", model=model_name)
                    _record("key_ok", key=key_name, model=model_name)
                    return payload, key_name
                except Exception as exc:
                    if _is_missing_model(exc, gexc):
                        last_missing = exc
                        _record("model_missing", model=model_name)
                        continue
                    raise
            raise GeminiUnavailableError(
                "No current Gemini Flash model is available. Set GEMINI_MODEL in extractor/.env."
            ) from last_missing
        except _KeyQuotaError as exc:
            last_block = exc
            blocked += 1
            if _is_daily_quota(exc):
                _EXHAUSTED_KEYS.add(api_key)
                _record("key_quota", key=key_name)
            else:
                _RPM_COOLDOWN_UNTIL[api_key] = time.monotonic() + RPM_COOLDOWN_SEC
                _record("key_rpm", key=key_name)
            continue
        except _fatal_auth_errors(gexc) as exc:
            _record("key_auth", key=key_name)
            raise GeminiUnavailableError(f"Gemini auth failed on {key_name}") from exc
    if blocked and (_live_key_count(load_gemini_api_keys()) == 0 or blocked == len(chosen_keys)):
        raise GeminiQuotaExhaustedError(
            "All Gemini API keys are out of quota."
        ) from last_block
    raise GeminiUnavailableError("Gemini failed on every remaining key.") from last_block


def _generate_once(api_key: str, gexc: Any, model_name: str, prompt: str, deadline: float) -> dict[str, Any]:
    timeout = _call_timeout(deadline)
    if timeout is None:
        raise TimeoutError("Gemini budget exhausted")
    _mark_call_started()
    try:
        payload = _post_generate(api_key, model_name, prompt, timeout, thinking_budget=0)
    except Exception as exc:
        if not _thinking_rejected(exc):
            _raise_api_error(exc, gexc)
        payload = _post_without_thinking(api_key, model_name, prompt, timeout, gexc)
    text, reason = _candidate_text(payload)
    if text:
        return _parse_json(text)
    if reason in {"MAX_TOKENS", "2"}:
        payload = _post_without_thinking(api_key, model_name, prompt, timeout, gexc, max_tokens=2048)
        text, reason = _candidate_text(payload)
        if text:
            return _parse_json(text)
    raise GeminiUnavailableError(
        f"Gemini returned no title text (finish_reason={reason or 'empty'})."
    )


def _post_without_thinking(
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: float,
    gexc: Any,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    try:
        return _post_generate(
            api_key,
            model_name,
            prompt,
            timeout,
            thinking_budget=None,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _raise_api_error(exc, gexc)
        raise


def _thinking_rejected(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "thinking" in text and ("400" in text or "invalid" in text)


def _post_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: float,
    *,
    thinking_budget: int | None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "temperature": 0,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if thinking_budget is not None:
        generation["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {detail}") from exc
    data = json.loads(raw) if raw else {}
    return data if isinstance(data, dict) else {}


def _candidate_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason") or "empty"
        return "", str(reason)
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    reason = candidate.get("finishReason")
    parts = ((candidate.get("content") or {}).get("parts")) or []
    texts = [
        str(part.get("text"))
        for part in parts
        if isinstance(part, dict) and part.get("text") and not part.get("thought")
    ]
    return "\n".join(texts).strip(), None if reason is None else str(reason)


def _raise_api_error(exc: BaseException, gexc: Any) -> None:
    if _is_missing_model(exc, gexc):
        raise exc
    if _is_rpm_error(exc) or _is_quota_error(exc, gexc):
        raise _KeyQuotaError(str(exc)) from exc
    if isinstance(exc, _fatal_auth_errors(gexc)):
        raise exc
    text = str(exc).lower()
    if " 401 " in f" {text} " or " 403 " in f" {text} " or "permission denied" in text or "unauthenticated" in text:
        raise GeminiUnavailableError(f"Gemini auth failed ({exc})") from exc
    raise exc


def _primary_model() -> str:
    name = _normalize_model((os.environ.get("GEMINI_MODEL") or "").strip() or MODEL_NAME)
    if name in _RETIRED_MODELS:
        return MODEL_NAME
    return name


def _model_candidates() -> list[str]:
    extra = _normalize_model((os.environ.get("GEMINI_MODEL_FALLBACK") or "").strip())
    ordered: list[str] = []
    for name in (_primary_model(), extra, *MODEL_FALLBACKS):
        name = _normalize_model(name)
        if not name or name in _RETIRED_MODELS or name in ordered:
            continue
        ordered.append(name)
    return ordered or [MODEL_NAME]


def _normalize_model(name: str) -> str:
    text = (name or "").strip()
    if text.lower().startswith("models/"):
        text = text[7:]
    return text


def _is_missing_model(exc: BaseException, gexc: Any) -> bool:
    if isinstance(exc, _permanent_model_errors(gexc)):
        return True
    text = str(exc).lower()
    return "no longer available" in text or (
        "404" in text and "model" in text and "429" not in text
    )


def _respect_rpm_gap() -> None:
    if _LAST_CALL_AT <= 0:
        return
    wait = MIN_CALL_INTERVAL_SEC - (time.monotonic() - _LAST_CALL_AT)
    if wait > 0:
        _record("rpm_wait", sec=round(wait, 2))
        time.sleep(wait)


def _mark_call_started() -> None:
    global _LAST_CALL_AT
    _LAST_CALL_AT = time.monotonic()


def _ordered_live_keys() -> list[tuple[str, str]]:
    global _RR_INDEX
    now = time.monotonic()
    live: list[tuple[str, str]] = []
    for name, value in load_gemini_api_keys():
        if value in _EXHAUSTED_KEYS:
            continue
        if _RPM_COOLDOWN_UNTIL.get(value, 0.0) > now:
            continue
        live.append((name, value))
    if not live:
        return []
    start = _RR_INDEX % len(live)
    _RR_INDEX = (_RR_INDEX + 1) % len(live)
    return live[start:] + live[:start]


def _pick_key() -> tuple[str, str] | None:
    ordered = _ordered_live_keys()
    return ordered[0] if ordered else None


def _call_timeout(deadline: float) -> float | None:
    remaining = deadline - time.monotonic()
    if remaining < MIN_CALL_SEC:
        return None
    return min(REQUEST_TIMEOUT_SEC, remaining)


def _is_daily_quota(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "quota",
            "resource_exhausted",
            "resource exhausted",
            "exceeded your current",
        )
    )


def _is_rpm_error(exc: BaseException) -> bool:
    if _is_daily_quota(exc):
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("rate limit", "rate-limit", "too many requests", "requests per minute")
    )


def _is_quota_error(exc: BaseException, gexc: Any) -> bool:
    if _is_daily_quota(exc):
        return True
    quota_types = []
    for name in ("ResourceExhausted",):
        kind = getattr(gexc, name, None)
        if isinstance(kind, type):
            quota_types.append(kind)
    if quota_types and isinstance(exc, tuple(quota_types)):
        return True
    return False


def _live_key_count(keys: list[tuple[str, str]]) -> int:
    return sum(1 for _, value in keys if value not in _EXHAUSTED_KEYS)


def _clean_key(raw: str) -> str:
    value = (raw or "").strip().strip('"').strip("'")
    return "" if value in PLACEHOLDER_KEYS else value


def _split_key_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]


def _permanent_model_errors(gexc: Any) -> tuple[type[BaseException], ...]:
    return (gexc.NotFound,)


def _fatal_auth_errors(gexc: Any) -> tuple[type[BaseException], ...]:
    kinds: list[type[BaseException]] = []
    for name in ("PermissionDenied", "Unauthenticated", "Forbidden"):
        kind = getattr(gexc, name, None)
        if isinstance(kind, type):
            kinds.append(kind)
    return tuple(kinds)


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
    text = re.sub(r"\s+", " ", str(value)).strip().strip('"').strip("'")
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def _clean_page(value: Any) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


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


def _record(event: str, **info: Any) -> None:
    _STATS[event] += 1
    extras = " ".join(f"{key}={value}" for key, value in info.items() if value is not None)
    logger.info("gemini %s %s", event, extras)
