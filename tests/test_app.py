"""Tests for the logic the Week-4 UI runs on, plus the UI's own error handling.

The Streamlit app (app.py) is a thin shell — it draws whatever `rank_candidates`
returns. So most of this file tests that function, not the UI: get the ranking
right and the UI is right by construction. Those tests import no Streamlit on
purpose, so they run without the UI dependency installed.

Two things live only in app.py and can't be tested that way, since there's no
standalone function to call directly: catching BadVideosCsv so a broken
data/videos.csv shows a clean banner instead of a raw traceback, and the trust
panel that renders src/evaluate.py's cross-validation verdict on the page. For
those we run the real app.py through Streamlit's own AppTest harness (bundled
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


# --- app.py's trust panel: must show evaluate.summarize()'s own numbers -----

def test_trust_panel_matches_evaluate_summarize():
    """The three metrics in the trust panel (R^2, model MAE, baseline MAE) must
    be the exact numbers `evaluate.summarize()` computes for the same seeded
    synthetic data — the app must not drift from the harness it's supposed to
    be surfacing. This is the app.py-side companion to test_evaluate.py, which
    already checks summarize() itself is correct.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.dataset import load
    from src.evaluate import build_matrix, cross_validate, summarize

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert at.exception == []

    # app.py's get_trust_summary() uses seed=42, k=5 by default -- match them.
    df, is_real = load(seed=42)
    assert is_real is False, "this test assumes the no-CSV synthetic path"
    X = build_matrix(df["title"].tolist())
    y = df["ctr"].to_numpy(dtype=float)
    expected = summarize(cross_validate(X, y, k=5, seed=42))

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Model R² (avg across folds)"] == f"{expected['model_r2_mean']:.3f}"
    assert metrics["Model MAE"] == (
        f"{expected['model_mae_mean']:.2f} ± {expected['model_mae_std']:.2f}"
    )
    assert metrics["Baseline MAE (guess the mean)"] == f"{expected['base_mae_mean']:.2f}"

    # The verdict banner must agree with beats_baseline, not just the numbers.
    banner_text = " ".join(e.value for e in list(at.success) + list(at.warning))
    if expected["beats_baseline"]:
        assert "Beats the baseline" in banner_text
    else:
        assert "Does NOT beat the baseline" in banner_text


def test_trust_panel_warns_loud_on_synthetic_data():
    """On synthetic data the trust caption must say the verdict is about the
    FAKE rule, not YouTube -- the honesty rule applies to this panel too.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)

    caption_text = " ".join(c.value for c in at.caption)
    assert "SYNTHETIC" in caption_text
    assert "fake" in caption_text.lower()
    assert "says nothing" in caption_text or "nothing about YouTube" in caption_text
