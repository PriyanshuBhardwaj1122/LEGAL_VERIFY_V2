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


# A boundary is only worth backing up to if it doesn't shrink the chunk
# too much — otherwise one early newline near the start of a window
# would produce a long tail of tiny chunks.
_MIN_BOUNDARY_FRACTION = 0.6

# Sentence end followed by a capital/quote/digit. Deliberately simple:
# this only decides where to cut for the LLM's benefit, so a missed
# abbreviation costs nothing.
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?\s")


def _split_point(text: str, start: int, end: int) -> int:
    """Best place to end a chunk that begins at `start`, searching back
    from `end`. Tries paragraph break, then line break, then sentence
    end, then gives up and cuts at `end`.

    The paragraph-only version of this silently degraded to a blind cut:
    HTML extracted by trafilatura contains no blank lines at all (0
    occurrences of "\\n\\n" across every stored judgment), so every
    split landed mid-sentence at exactly max_chars.
    """
    floor = start + int((end - start) * _MIN_BOUNDARY_FRACTION)

    for sep in ("\n\n", "\n"):
        boundary = text.rfind(sep, start, end)
        if boundary > floor:
            return boundary

    # Last sentence end inside the window.
    best = -1
    for m in _SENTENCE_END_RE.finditer(text, floor, end):
        best = m.end()
    if best > floor:
        return best

    return end


def chunk_text(text: str, max_chars: int = 25_000, overlap: int = 800) -> list[str]:
    """Split long documents on the cleanest available boundary,
    preserving overlap so quotes near a chunk edge aren't lost.

    Offsets returned by the grounding step are always computed against
    the FULL text, not the chunk — chunking only bounds what's sent to
    the extractor LLM, so changing where chunks fall never invalidates
    stored quote_start/quote_end values.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _split_point(text, start, end)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start, end - overlap)

    return chunks
