"""
Tesseract OCR fallback for scanned / garbled PDF pages.

Title extraction OCRs a full page at modest DPI in grayscale so Tesseract
stays fast without clipping a cover title that sits mid-page.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import fitz

pytesseract = None
Output = None
Image = None

DEFAULT_DPI = 180
TITLE_BAND = 1.0
MAX_OCR_WIDTH = 1400
PSM_NO_OSD = "3"
TESSERACT_OEM = "1"
OCR_TIMEOUT_SEC = 45
_TESSERACT_CMD: str | None = None


class OcrUnavailableError(RuntimeError):
    """Tesseract or its Python bindings are missing."""


def _load_bindings() -> bool:
    global pytesseract, Output, Image
    if pytesseract is not None:
        return True
    try:
        import pytesseract as _pt
        from pytesseract import Output as _Output
        from PIL import Image as _Image
    except ImportError:
        return False
    pytesseract = _pt
    Output = _Output
    Image = _Image
    return True


def resolve_tesseract_cmd() -> str | None:
    global _TESSERACT_CMD
    if _TESSERACT_CMD and os.path.isfile(_TESSERACT_CMD):
        return _TESSERACT_CMD

    explicit = os.environ.get("TESSERACT_CMD")
    if explicit and os.path.isfile(explicit):
        _TESSERACT_CMD = explicit
        return explicit

    on_path = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if on_path:
        _TESSERACT_CMD = on_path
        return on_path

    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.EXE",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\ASUS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(candidate):
            _TESSERACT_CMD = candidate
            return candidate
    return None


def ocr_is_available() -> bool:
    return ocr_status()["available"]


def ocr_status() -> dict[str, Any]:
    extras = _load_bindings()
    cmd = resolve_tesseract_cmd()
    if not extras:
        return {
            "available": False,
            "cmd": cmd,
            "reason": "pytesseract/Pillow missing in this Python",
        }
    if not cmd:
        return {
            "available": False,
            "cmd": None,
            "reason": "tesseract.exe not on PATH",
        }
    return {"available": True, "cmd": cmd, "reason": "ready"}


def ocr_page(
    page: fitz.Page,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    timeout: float = OCR_TIMEOUT_SEC,
    band: float = TITLE_BAND,
) -> list[dict[str, Any]]:
    """OCR one page (full page, modest DPI) and return native-shaped spans."""
    _ensure_tesseract()

    rect = page.rect
    clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * min(max(band, 0.35), 1.0))
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csGRAY,
        clip=clip,
    )
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    if image.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(image.width)
        image = image.resize(
            (MAX_OCR_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )

    data = pytesseract.image_to_data(
        image,
        config=f"--psm {PSM_NO_OSD} --oem {TESSERACT_OEM}",
        output_type=Output.DICT,
        timeout=max(1, int(timeout)),
    )

    x_scale = clip.width / max(image.width, 1)
    y_scale = clip.height / max(image.height, 1)

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    n_items = len(data.get("text", []))
    for index in range(n_items):
        text = (data["text"][index] or "").strip()
        conf = _safe_confidence(data["conf"][index])
        if not text or conf < 0:
            continue

        x, y, w, h = (
            data["left"][index],
            data["top"][index],
            data["width"][index],
            data["height"][index],
        )
        bbox = [
            round(clip.x0 + x * x_scale, 2),
            round(clip.y0 + y * y_scale, 2),
            round(clip.x0 + (x + w) * x_scale, 2),
            round(clip.y0 + (y + h) * y_scale, 2),
        ]
        font_size = round(max(h * y_scale, 0.0), 2)
        key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": font_size,
                "bold": False,
                "italic": False,
                "bbox": bbox,
                "block": key[0],
                "line": key[2],
                "source": "ocr",
                "confidence": conf,
            }
        )

    spans: list[dict[str, Any]] = []
    for key in sorted(grouped):
        words = sorted(grouped[key], key=lambda item: (item["bbox"][0], item["bbox"][1]))
        median_height = _median([word["font_size"] for word in words])
        text = " ".join(word["text"] for word in words if word["text"]).strip()
        if not text:
            continue
        boxes = [word["bbox"] for word in words]
        confs = [float(word.get("confidence") or 0.0) for word in words]
        spans.append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": median_height,
                "bold": False,
                "italic": False,
                "bbox": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "block": key[0],
                "line": key[2],
                "source": "ocr",
                "confidence": round(sum(confs) / max(len(confs), 1), 1),
            }
        )
    return spans


def _ensure_tesseract() -> None:
    if not _load_bindings():
        raise OcrUnavailableError(
            "OCR extras are not installed. Run: pip install pytesseract Pillow"
        )
    cmd = resolve_tesseract_cmd()
    if not cmd:
        raise OcrUnavailableError(
            "Tesseract is not installed or not on PATH. Install it from "
            "https://github.com/tesseract-ocr/tesseract and restart the app. "
            "On Windows you can also set the TESSERACT_CMD environment variable "
            "to tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = cmd


def _safe_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 2)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 2)
