"""PDF text extraction via PyMuPDF, with a no-text-layer detector.

OCR is deliberately out of scope for this pass — a PDF with no text
layer is marked low-confidence and non-quotable rather than silently
producing garbage or requiring a Tesseract dependency the team hasn't
signed off on yet. Extractor/grounding both respect is_quotable=False.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from app.core.logging import get_logger

log = get_logger()

MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 20


def extract_pdf(pdf_bytes: bytes) -> tuple[str, float, bool]:
    """Returns (text, extraction_confidence, is_quotable).

    is_quotable=False when the PDF has no real text layer (scanned,
    pre-OCR pipeline not yet built) — the extractor may still use such
    text for a rough statement, but grounding will never accept a
    verbatim_quote sourced from it.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        log.warning("pdf_open_failed", error=str(e))
        return "", 0.0, False

    pages_text: list[str] = []
    pages_with_text = 0

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
        if len(text.strip()) >= MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER:
            pages_with_text += 1

    doc.close()

    full_text = "\n".join(pages_text)
    total_pages = max(1, len(pages_text))
    text_layer_ratio = pages_with_text / total_pages

    if text_layer_ratio < 0.5:
        # Most pages have no real text layer — likely a scan.
        return full_text, 0.4, False

    if text_layer_ratio < 0.9:
        return full_text, 0.75, True

    return full_text, 0.95, True
