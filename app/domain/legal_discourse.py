"""Speaker and discourse markers for common-law judgments.

Two jobs, both jurisdiction-general:

  * telling the COURT's voice from a PARTY's voice. A judgment recites
    what each side argued before deciding; a sentence reporting counsel's
    contention is not the court's holding, however confidently it reads.
  * telling substantive passages from front matter, so passage selection
    can spend its budget on reasoning rather than on counsel lists.

Deliberately independent of app/domain/authority.py and
citation_parser.py: those encode Indian specifics, these are the phrases
common-law courts share. Both US ("we affirm", "we reverse") and
Commonwealth ("the appeal is allowed") disposition phrasings are
included — that is coverage, not coupling.

Civil-law jurisdictions phrase reasoning differently and will need these
sets extended; they are module-level constants so an operator can do
that without touching the nodes that consume them.
"""

from __future__ import annotations

import re

# A party's voice — counsel submitting, contending, urging. Text matching
# these is advocacy, not ratio, even where the court later agrees with it.
ATTRIBUTION_MARKERS = re.compile(
    r"\b("
    r"submitted|submits|contended|contends|argued|argues|urged|urges|canvassed|"
    r"it was (?:submitted|argued|urged|contended)|"
    r"on behalf of the (?:appellant|respondent|petitioner|plaintiff|defendant|state)|"
    r"counsel for|appearing for|learned counsel|"
    r"according to (?:him|her|them)|"
    r"(?:their|his|her) (?:submission|contention)"
    r")\b",
    re.IGNORECASE,
)

# The court speaking in its own voice — reasoning and disposition.
COURT_VOICE_MARKERS = re.compile(
    r"\b("
    r"we (?:hold|are of the (?:view|opinion)|conclude|agree|disagree|therefore|find)|"
    r"it is held|held that|it is settled|in our (?:view|opinion|judgment|judgement)|"
    r"for the (?:foregoing|above) reasons|"
    r"we (?:affirm|reverse|remand|vacate)|"                 # US
    r"the (?:appeal|petition)s? (?:is|are) (?:allowed|dismissed)|"  # Commonwealth
    r"it is (?:so )?ordered|accordingly, the|we therefore"
    r")\b",
    re.IGNORECASE,
)

# Separate opinions carry real argumentative weight and are worth keeping.
DISSENT_MARKERS = re.compile(
    r"\b(dissent(?:ing|s)?|respectfully dissent|concurring|I would)\b",
    re.IGNORECASE,
)

# Front matter: counsel appearance lists, bench composition, signature
# blocks. Dense at the top of a judgment and almost never citable.
FRONT_MATTER_MARKERS = re.compile(
    r"\b("
    r"bench:|author:|coram|"
    r"for the (?:appellant|respondent|petitioner)s?\s*:|"
    r"advocates?(?:\s+for)?|senior advocate|"
    r"digitally signed|signature not verified|reportable|"
    r"equivalent citations?"
    r")\b",
    re.IGNORECASE,
)

# A quoted instrument being set out — statutory text, regulations.
INSTRUMENT_MARKERS = re.compile(
    r"(\bSection\s+\d|\bArticle\s+\d|§\s*\d|\bs\.\s*\d|\breg(?:ulation)?\.?\s*\d|\bcl\.\s*\d)",
    re.IGNORECASE,
)


def counts(text: str) -> dict[str, int]:
    """Marker counts for one passage — the raw signal for scoring."""
    return {
        "attribution": len(ATTRIBUTION_MARKERS.findall(text)),
        "court_voice": len(COURT_VOICE_MARKERS.findall(text)),
        "dissent": len(DISSENT_MARKERS.findall(text)),
        "front_matter": len(FRONT_MATTER_MARKERS.findall(text)),
        "instrument": len(INSTRUMENT_MARKERS.findall(text)),
    }
