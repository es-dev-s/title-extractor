# Titlextractor

Extracts the title from a PDF, even when it's not the biggest font on
page 1. Two-tier pipeline: GROBID first (accurate, scholarly-trained),
heuristic scorer as fallback (font size + position + boilerplate
rejection).

## Setup

```bash
pip install -r requirements.txt
```

## (Optional but recommended) Run GROBID

GROBID is what gives you near-100% accuracy on academic PDFs. It runs as
a separate Docker container — your Flask app just calls it over HTTP.

```bash
docker run -t --rm -p 8070:8070 grobid/grobid:0.8.0
```

If you skip this step, the app still works — it just falls back straight
to the heuristic scorer (still decent, not as strong on tricky layouts).

## Run the app

```bash
python app.py
```

Visit http://localhost:5000, upload a PDF, get the title back.

## How it decides the title

1. **GROBID** (`extractor/grobid_client.py`) — sends the PDF to the
   running GROBID container, parses the `<title>` out of its TEI-XML
   response. Trained specifically on scholarly document structure, so it
   correctly ignores journal names/DOIs/ISSN lines even when they're
   visually similar to the real title.
2. **Heuristic fallback** (`extractor/heuristic.py`) — only runs if
   GROBID is unreachable or returns nothing. Extracts every text line
   with its font size, boldness, page, and position (`extractor/pdf_utils.py`),
   scores each one, rejects boilerplate patterns and repeated
   headers/footers, and merges multi-line titles.

## Next steps worth adding

- **OCR tier**: if `has_text_layer()` returns `False`, run PaddleOCR or
  Tesseract before either tier above — the current app just reports an
  error in that case.
- **LLM tier**: for the cases where GROBID fails AND the heuristic score
  is low, send the first page's raw text to an LLM asking it to return
  the title verbatim, then verify the response is an exact substring of
  the source text before trusting it (prevents hallucinated titles).
- Batch mode: loop `run_pipeline()` over a folder of PDFs instead of one
  upload at a time.
