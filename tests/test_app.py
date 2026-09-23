"""Tests for the logic the Week-4 UI runs on, plus the UI's own error handling.

The Streamlit app (app.py) is a thin shell — it draws whatever `rank_candidates`
returns. So most of this file tests that function, not the UI: get the ranking
right and the UI is right by construction. Those tests import no Streamlit on
purpose, so they run without the UI dependency installed.

The one thing that lives only in app.py — catching BadVideosCsv so a broken
data/videos.csv shows a clean banner instead of a raw traceback — can't be
tested that way, since there's no ranking logic to call directly. For that one
case we run the real app.py through Streamlit's own AppTest harness (bundled
with streamlit, see requirements.txt), which needs the UI dependency.
"""
from __future__ import annotations

import os

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


# --- app.py's own error handling: a broken data/videos.csv must not crash ---

def test_app_shows_clean_banner_for_bad_csv(tmp_path):
    """A malformed data/videos.csv must surface as one st.error banner, not a
    raw traceback. Never touches the real data/videos.csv: dataset.DATA_CSV is
    pointed at a throwaway file in tmp_path before app.py runs.
    """
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import src.dataset as dataset

    bad_csv = tmp_path / "videos.csv"
    bad_csv.write_text("title,ctr\n", encoding="utf-8")  # 0 usable rows -> BadVideosCsv

    # AppTest.from_string runs the script in THIS process, re-importing
    # src.dataset only if it isn't already in sys.modules — so the patch line
    # inside the script mutates the same module object other tests import.
    # Save/restore DATA_CSV so this test can't leak state into test_dataset.py.
    original_data_csv = dataset.DATA_CSV
    try:
        script = (
            "import src.dataset as dataset\n"
            f"dataset.DATA_CSV = {str(bad_csv)!r}\n"
            "import app\n"
        )
        at = AppTest.from_string(script)
        at.run(timeout=30)
    finally:
        dataset.DATA_CSV = original_data_csv

    assert at.exception == [], "BadVideosCsv reached Streamlit as a raw exception, not a caught one"
    messages = [e.value for e in at.error]
    assert any("only has 0 usable row(s)" in m for m in messages), messages
    assert any("data/README.md" in m for m in messages), "banner should point at the fix"


def test_app_runs_clean_on_synthetic_data():
    """No data/videos.csv (the normal day-one case) must not hit the new
    except branch at all — the model loads and no exception reaches the page.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)

    assert at.exception == []
