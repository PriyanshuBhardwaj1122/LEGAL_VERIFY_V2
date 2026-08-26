"""URL canonicalization and dedup utilities."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Params to strip during canonicalization
_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "source", "mc_cid", "mc_eid",
}

# Known mirror host mappings
_MIRROR_HOSTS: dict[str, str] = {
    "www.indiankanoon.org": "indiankanoon.org",
    "www.indiacode.nic.in": "indiacode.nic.in",
    "www.sci.gov.in": "sci.gov.in",
}

# Domains whose content is subscription-gated — never fetch from these
BLOCKED_DOMAINS = frozenset({
    "scconline.com", "www.scconline.com",
    "manupatra.com", "www.manupatra.com",
    "westlawindia.com", "www.westlawindia.com",
    "lexisnexis.com", "www.lexisnexis.com",
})

# Content-farm / junk domains — drop at zero cost
JUNK_DOMAINS = frozenset({
    "toprankers.com", "byjus.com", "vedantu.com", "unacademy.com",
    "geeksforgeeks.org", "javatpoint.com", "tutorialspoint.com",
    "legalbites.in", "lawctopus.com",
    "cleartax.in", "taxguru.in",
})


def canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    - Lowercase scheme + host
    - Strip www. prefix
    - Remove tracking params
    - Remove fragment
    - Remove trailing slash
    - Resolve known mirror hosts
    """
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
    host = host.lower()

    # Strip www.
    if host.startswith("www."):
        host = host[4:]

    # Known mirrors
    host = _MIRROR_HOSTS.get(host, host)

    # Reconstruct path
    path = parsed.path.rstrip("/") or ""

    # Filter query params
    params = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {k: v for k, v in params.items() if k.lower() not in _STRIP_PARAMS}
    query = urlencode(filtered, doseq=True) if filtered else ""

    # No fragment
    canonical = urlunparse((scheme, host, path, "", query, ""))
    return canonical


def is_blocked_domain(url: str) -> bool:
    """Check if URL belongs to a subscription-gated domain."""
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in BLOCKED_DOMAINS


def is_junk_domain(url: str) -> bool:
    """Check if URL belongs to a known content-farm domain."""
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in JUNK_DOMAINS
