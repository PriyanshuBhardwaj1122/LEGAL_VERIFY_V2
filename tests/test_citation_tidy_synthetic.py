"""Citation strings must be fit to print.

Scraped legal sources carry two defects that reach the page and read as
machine-assembled: a dump of every parallel citation for one judgment,
and a trailing ellipsis where an index truncated a cause title. Both
appeared in shipped articles.

Run: PYTHONPATH=. python -m pytest tests/test_citation_tidy_synthetic.py -q
"""

from __future__ import annotations

import pytest

from app.domain.citation_render import tidy_citation_string


def test_parallel_citation_dump_reduced_to_one_reporter():
    """A lawyer cites one reporter, not all six."""
    dump = "1994 AIR 988 1993 SCR (3) 128 1993 SCC (3) 499 JT 1993 (3) 15 1993 SCALE (2)506"
    out = tidy_citation_string(dump)
    assert out == "1993 SCC (3) 499"
    for noise in ("SCALE", "JT", "SCR"):
        assert noise not in out


def test_scc_preferred_over_air():
    """Supreme Court Cases outranks AIR when both are present."""
    assert "SCC" in tidy_citation_string("1994 AIR 988 1993 SCC (3) 499")


@pytest.mark.parametrize(
    "already_clean",
    ["(1993) 3 SCC 499", "1983 (1) S.C.C. 71", "2023 INSC 709", "(1999) 4 SCC 727"],
)
def test_single_citations_pass_through_unchanged(already_clean):
    assert tidy_citation_string(already_clean) == already_clean


def test_truncation_ellipsis_removed():
    assert tidy_citation_string("Union Of India And Ors vs Hindustan Development Corpn. ...") == (
        "Union Of India And Ors v Hindustan Development Corpn."
    )
    assert tidy_citation_string("Some Long Cause Title…") == "Some Long Cause Title"


def test_vs_normalised_to_v():
    assert " v " in tidy_citation_string("A vs B")
    assert " vs " not in tidy_citation_string("A vs B")


def test_empty_input_is_safe():
    assert tidy_citation_string("") == ""
    assert tidy_citation_string(None) == ""


def test_case_number_not_mangled():
    """Not every citation is a reporter reference."""
    assert tidy_citation_string("CIVIL APPEAL NO. 5254 OF 2010") == "CIVIL APPEAL NO. 5254 OF 2010"
