"""Text normalization — NFKC, whitespace collapse, de-hyphenation.

Offsets recorded by the grounding validator are computed against exactly
this normalized text, so this function must be deterministic and never
change behavior between runs — otherwise stored quote_start/quote_end
values drift and can no longer be re-located.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Repeated short lines across many pages of a PDF are almost always
# headers/footers/page numbers, not content.
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)


def normalize_text(raw: str) -> str:
    """Canonicalize extracted text for storage and quote-offset matching."""
    text = unicodedata.normalize("NFKC", raw)

    # De-hyphenate line-break-split words: "insol-\nvency" -> "insolvency"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse all whitespace runs (including newlines) to single spaces.
    # This is a deliberate simplification — paragraph structure is not
    # preserved in v1; pinpoint citations rely on the LLM's own
    # "para N" / "s. X" text, not on our paragraph segmentation.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip lines that are just page numbers
    lines = [ln for ln in text.split("\n") if not _PAGE_NUMBER_RE.match(ln.strip())]
    text = "\n".join(lines)

    # Strip repeated leading/trailing whitespace per line, collapse blank runs
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def content_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def chunk_text(text: str, max_chars: int = 25_000, overlap: int = 800) -> list[str]:
    """Split long documents on paragraph boundaries, preserving overlap
    so quotes near a chunk edge aren't lost. Offsets returned by the
    grounding step are always computed against the FULL text, not the
    chunk — chunking only bounds what's sent to the extractor LLM."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Back up to the nearest paragraph boundary
            boundary = text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start, end - overlap)

    return chunks
