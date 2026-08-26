"""HTML text extraction — trafilatura first, BeautifulSoup fallback."""

from __future__ import annotations

import trafilatura
from bs4 import BeautifulSoup

from app.core.logging import get_logger

log = get_logger()

MIN_TRAFILATURA_CHARS = 400

# Per-domain fallback selectors for sites trafilatura tends to under-extract.
_DOMAIN_SELECTORS: dict[str, str] = {
    "indiankanoon.org": "div.judgments",
    "sebi.gov.in": "div.article-content, div.content",
    "indiacode.nic.in": "div.content, body",
}


def extract_html(html: str, url: str) -> tuple[str, str]:
    """Returns (text, extraction_method)."""
    extracted = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )

    if extracted and len(extracted) >= MIN_TRAFILATURA_CHARS:
        return extracted, "trafilatura"

    # Fallback: BeautifulSoup with a domain-aware selector, or a plain
    # get_text() sweep if we don't have one.
    soup = BeautifulSoup(html, "html.parser")

    selector = None
    for domain, sel in _DOMAIN_SELECTORS.items():
        if domain in url:
            selector = sel
            break

    if selector:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(separator="\n", strip=True)
            if len(text) >= MIN_TRAFILATURA_CHARS:
                return text, "bs4"

    # Last resort: strip script/style, grab all text
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    if extracted and len(extracted) > len(text):
        # trafilatura's short extraction still beat a noisy full-page dump
        return extracted, "trafilatura"

    return text, "bs4"
