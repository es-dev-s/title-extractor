# Titlextractor

Reads a PDF and returns its printed title. Text comes from the PDF itself
or from Tesseract. Gemini reads that text and names the title. The layout
heuristic runs only after every Gemini key is out of quota.

## Setup

```bash
pip install -r requirements.txt
```

Tesseract must be on `PATH` for scanned or garbled pages. Put one or more
Gemini keys in `extractor/.env`:

```
GEMINI_API_KEY=...
GEMINI_API_KEY_2=...
```

## Run

```bash
python app.py
```

Open http://localhost:5000, upload a PDF, and the page shows the title plus
the extracted layout. `GET /health` returns `{"ok": true}`.

## Production

From this folder, with Gemini keys in `extractor/.env`:

```bash
docker compose up -d --build
```

The container listens on port 5000 and runs Gunicorn, not the Flask debug server. Tesseract is installed in the image. Set `WEB_CONCURRENCY` to change how many PDFs are titled at once.

## How a title is chosen

`extractor/pipeline.py` does this once per PDF:

1. Classify each page as native text, scanned, garbled, or empty.
2. Native pages: PyMuPDF reads the text with font and position (`extractor/pdf_utils.py`).
3. Scanned or garbled pages: Tesseract OCRs the full page (`extractor/ocr.py`). At most 3 pages, about 45 seconds each.
4. Always read the first 4 pages. If the title area is too thin, read through page 7, still as one pass.
5. Send that front-matter text to Gemini once (`extractor/gemini_title.py`). Keys rotate when one hits quota. Gemini is not called a second time for the same PDF.
6. If every key is out of quota, score lines by font size, position, and boilerplate rejection (`extractor/heuristic.py`) and use that heading instead.

## What the app calls

The PDF app posts each file to `POST /extract?title_only=1` with the form field `pdf` (or `file`). That response is only the heading:

```json
{"ok": true, "title": "...", "title_source": "gemini", "filename": "paper.pdf", "method": "gemini"}
```

`POST /extract` without `title_only` returns the same title plus pages, lines, and the text that was sent to Gemini. A scanned PDF with no Tesseract install comes back as `{"ok": false, "message": "No OCR"}`.
