"""Real assertions on the evaluation harness — not smoke tests.

These check the honesty machinery itself: the baseline is computed the way we
claim, cross-validation returns exactly k independent folds, and on the seeded
synthetic data (where a real signal is baked in) the model genuinely beats the
mean-guess baseline. If any of these break, the verdict the tool prints can't be
trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset import make_synthetic
from src.evaluate import (
    baseline_mae,
    build_matrix,
    cross_validate,
    summarize,
)


def _synthetic_xy(n: int = 300, seed: int = 42):
    df = make_synthetic(n=n, seed=seed)
    X = build_matrix(df["title"].tolist())
    y = df["ctr"].to_numpy(dtype=float)
    return X, y


def test_baseline_mae_matches_hand_calc():
    # Train mean is 3.0; MAE against test = mean(|4-3|, |2-3|) = 1.0.
    y_train = np.array([2.0, 4.0])
    y_test = np.array([4.0, 2.0])
    assert baseline_mae(y_train, y_test) == pytest.approx(1.0)


def test_baseline_uses_train_mean_not_test_mean():
    # If it peeked at the test mean it would score a perfect 0.0 here; it must not.
    y_train = np.array([10.0, 10.0, 10.0])
    y_test = np.array([2.0, 4.0])
    # Guess 10 for both -> MAE = mean(8, 6) = 7.0.
    assert baseline_mae(y_train, y_test) == pytest.approx(7.0)


def test_cross_validate_returns_k_folds_numbered():
    X, y = _synthetic_xy()
    folds = cross_validate(X, y, k=5, seed=42)
    assert len(folds) == 5
    assert [f["fold"] for f in folds] == [1, 2, 3, 4, 5]
    for f in folds:
        assert {"fold", "model_r2", "model_mae", "base_mae"} <= set(f)


def test_cross_validate_rejects_too_few_rows():
    X, y = _synthetic_xy(n=3)
    with pytest.raises(ValueError):
        cross_validate(X, y, k=5)


def test_cross_validate_is_reproducible():
    X, y = _synthetic_xy()
    a = summarize(cross_validate(X, y, k=5, seed=7))
    b = summarize(cross_validate(X, y, k=5, seed=7))
    assert a["model_mae_mean"] == pytest.approx(b["model_mae_mean"])


def test_model_beats_baseline_on_synthetic_signal():
    # The synthetic CTR is a real (if fake) function of the features + noise, so a
    # forest MUST beat the mean guess. If it doesn't, the pipeline is broken.
    X, y = _synthetic_xy(n=400)
    s = summarize(cross_validate(X, y, k=5, seed=42))
    assert s["beats_baseline"] is True
    assert s["mae_improvement"] > 0.0
    assert s["model_r2_mean"] > 0.0


def test_summarize_reports_spread():
    X, y = _synthetic_xy()
    s = summarize(cross_validate(X, y, k=5, seed=42))
    assert s["k"] == 5
    assert s["model_r2_std"] >= 0.0
    assert s["model_mae_std"] >= 0.0
