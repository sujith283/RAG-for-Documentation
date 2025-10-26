# tests/test_citations.py
from app.utils import build_inline_citations, insert_citation_tags

def test_citation_numbers():
    ctxs = [
        {"text":"A1", "source":"S1", "title":"T", "section":"Intro", "position":0},
        {"text":"A2", "source":"S2", "title":"T", "section":"Intro", "position":0},
        {"text":"A3", "source":"S1", "title":"T", "section":"Intro", "position":0}, # same as 1
    ]
    _, uniq = build_inline_citations(ctxs)
    assert len(uniq) == 2  # de-dup
    # Add numbers and format
    for i, s in enumerate(uniq, start=1):
        pass
