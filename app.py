"""
app.py

Flask entry point. Upload a PDF -> inspect extracted text and layout metadata.

Extraction order:
    1. Detect whether each page has a usable native text layer
    2. Native pages: PyMuPDF dict extraction (font, bbox, page)
    3. Scanned/garbled pages: Tesseract OCR fallback (--psm 1)
"""

import os
import sys
import uuid
from pathlib import Path

# `python app.py` often uses the system interpreter. Prefer the project venv
# so pytesseract and google-generativeai resolve the same way as eval scripts.
_ROOT = Path(__file__).resolve().parent
_VENV_SITE = _ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

from flask import Flask, request, render_template, jsonify

from extractor.pipeline import extract_document

app = Flask(__name__)

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    uploaded = request.files.get("pdf")
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "No PDF uploaded"}), 400

    temp_name = f"{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    uploaded.save(temp_path)

    try:
        result = extract_document(temp_path, filename=uploaded.filename)
        result["filename"] = uploaded.filename
        _log_preview(result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Could not read PDF: {exc}"}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _safe_print(text: str) -> None:
    """Windows consoles default to cp1252 and crash on Greek/math glyphs like φ."""
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    line = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(line)


def _log_preview(result: dict) -> None:
    try:
        _safe_print(
            f"[extract] {result.get('filename')} | type={result['pdf_type']} | "
            f"title={result.get('title')!r} | source={result.get('title_source')} | "
            f"confidence={result.get('title_confidence')} | score={result.get('title_score')} | "
            f"gemini={result.get('gemini_mode')} | "
            f"pages={result.get('page_count')}/{result.get('document_page_count')}"
            f"{' extended' if result.get('pages_extended') else ''}"
        )
        preview = (result.get("text") or "")[:800]
        if preview:
            _safe_print("--- extracted text preview ---")
            _safe_print(preview)
            _safe_print("------------------------------")
    except Exception:
        # Console encoding must never fail a successful extract.
        return


if __name__ == "__main__":
    app.run(debug=True, port=5000)
