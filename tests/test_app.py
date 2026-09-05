"""Tests for the logic the Week-4 UI runs on.

The Streamlit app (app.py) is a thin shell — it draws whatever `rank_candidates`
returns. So we test that function, not the UI: get the ranking right and the UI
is right by construction. No Streamlit import here on purpose, so the suite runs
without the UI dependency installed.
"""
from __future__ import annotations

import pytest

from src.train import rank_candidates, train


@pytest.fixture(scope="module")
def model():
    """One trained (synthetic) model shared across the tests in this file."""
    m, _is_real = train(seed=42)
    return m


def test_returns_all_nonblank_titles(model):
    titles = ["How to Use Python", "Is Claude Worth It?", "AI News Today"]
    ranked = rank_candidates(model, titles)
    assert len(ranked) == 3
    assert {t for t, _ in ranked} == set(titles)


def test_sorted_best_first(model):
    ranked = rank_candidates(model, ["A quiet title", "How to 5 X", "Why This?"])
    scores = [ctr for _, ctr in ranked]
    assert scores == sorted(scores, reverse=True), "must be ranked high-to-low CTR"


def test_blank_lines_are_dropped(model):
    # The UI hands us raw text split on newlines — empty/whitespace rows and
    # a trailing blank must not become phantom candidates.
    ranked = rank_candidates(model, ["Real title", "", "   ", "\t"])
    assert [t for t, _ in ranked] == ["Real title"]


def test_empty_input_gives_empty_list(model):
    assert rank_candidates(model, []) == []
    assert rank_candidates(model, ["", "  "]) == []


def test_each_score_matches_single_title_score(model):
    # rank_candidates must agree with scoring one title at a time — no drift.
    from src.train import score

    titles = ["Ranked together A", "Ranked together B"]
    ranked = dict(rank_candidates(model, titles))
    for t in titles:
        assert ranked[t] == pytest.approx(score(model, t))
