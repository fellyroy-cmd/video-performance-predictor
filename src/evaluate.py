"""Is the model actually any good? Honest evaluation, before we trust it.

    python -m src.evaluate

The plain `train` command fits ONE model on ONE train/test split and prints its
R^2. On a small channel that single number can lie — you might land a lucky (or
unlucky) split and read far too much into it. This module exists to keep us
honest, and it does two things `train` doesn't:

1. BASELINE — the dumbest honest predictor: ignore the title entirely and always
   guess the average CTR from the training rows. If our fancy model can't beat
   that, the title features carry no real signal and we should say so out loud.
   This is the bar every model has to clear. "Beats a coin flip" is the wrong
   test; "beats guessing the mean" is the right one.

2. CROSS-VALIDATION — instead of one split, cut the data into k folds. Train on
   k-1 folds, test on the held-out one, k times, so every row gets to be test
   data exactly once. Report the AVERAGE across folds and the SPREAD (std). A
   big spread means the model is fragile / we don't have enough data to trust it
   yet — which, on a small channel, is exactly the trap we're watching for.

Everything runs on whatever `load()` returns. On synthetic data this still only
checks plumbing + that the forest recovers the fake rule better than a mean
guess. On your real Week-2 export, THIS is the module that tells you whether the
thing genuinely works — or whether you're fooling yourself.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .dataset import load
from .features import feature_vector


def build_matrix(titles) -> np.ndarray:
    """Stack titles into a 2D feature matrix (one row per title)."""
    return np.array([feature_vector(t) for t in titles], dtype=float)


def _new_model(seed: int) -> RandomForestRegressor:
    """One place that defines the model, so train and evaluate stay identical."""
    return RandomForestRegressor(n_estimators=200, random_state=seed)


def baseline_mae(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """MAE of the mean predictor: guess the training-mean CTR for every test row.

    This is the floor. The model has to do better than this to have earned its
    keep. Uses the TRAIN mean (never the test mean) — peeking at the test mean
    would be cheating, the same way it would be for the real model.
    """
    guess = float(np.mean(y_train))
    preds = np.full(len(y_test), guess, dtype=float)
    return float(mean_absolute_error(y_test, preds))


def cross_validate(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> list[dict]:
    """k-fold CV. Returns one dict per fold with the model's and baseline's scores.

    Shuffled folds (seeded, so it's reproducible). For each fold we fit a fresh
    model on the other folds and score it on the held-out one, and we compute the
    mean-guess baseline on the same split — an apples-to-apples comparison every
    single fold.
    """
    if len(y) < k:
        raise ValueError(f"Need at least k={k} rows to run {k}-fold CV, got {len(y)}.")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        model = _new_model(seed)
        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])
        rows.append(
            {
                "fold": fold,
                "model_r2": float(r2_score(y[te], preds)),
                "model_mae": float(mean_absolute_error(y[te], preds)),
                "base_mae": baseline_mae(y[tr], y[te]),
            }
        )
    return rows


def summarize(folds: list[dict]) -> dict:
    """Collapse per-fold rows into averages, spreads, and the headline verdict."""
    model_r2 = np.array([f["model_r2"] for f in folds], dtype=float)
    model_mae = np.array([f["model_mae"] for f in folds], dtype=float)
    base_mae = np.array([f["base_mae"] for f in folds], dtype=float)
    return {
        "k": len(folds),
        "model_r2_mean": float(model_r2.mean()),
        "model_r2_std": float(model_r2.std()),
        "model_mae_mean": float(model_mae.mean()),
        "model_mae_std": float(model_mae.std()),
        "base_mae_mean": float(base_mae.mean()),
        # How much lower (better) the model's error is vs just guessing the mean.
        # Positive = the features are pulling their weight. <= 0 = they aren't.
        "mae_improvement": float(base_mae.mean() - model_mae.mean()),
        "beats_baseline": bool(model_mae.mean() < base_mae.mean()),
    }


def evaluate(k: int = 5, seed: int = 42) -> dict:
    """Load data, run k-fold CV, print an honest report, return the summary."""
    df, is_real = load(seed=seed)
    X = build_matrix(df["title"].tolist())
    y = df["ctr"].to_numpy(dtype=float)

    folds = cross_validate(X, y, k=k, seed=seed)
    s = summarize(folds)

    print("=" * 66)
    if is_real:
        print("DATA: real (data/videos.csv) — this verdict is about YOUR channel.")
    else:
        print("DATA: SYNTHETIC (no data/videos.csv found).")
        print("  [!] This only proves the forest can recover the FAKE rule better")
        print("      than a mean guess. It says nothing real about YouTube yet.")
        print("      Drop your export at data/videos.csv to make it mean something.")
    print("=" * 66)
    print(f"{s['k']}-fold cross-validation on {len(df)} rows (seed {seed})\n")

    print("Per fold:")
    print(f"  {'fold':<6}{'model R^2':>12}{'model MAE':>12}{'baseline MAE':>14}")
    for f in folds:
        print(
            f"  {f['fold']:<6}{f['model_r2']:>12.3f}"
            f"{f['model_mae']:>12.3f}{f['base_mae']:>14.3f}"
        )
    print()

    print("Averages across folds (mean +/- std):")
    print(f"  model  R^2 : {s['model_r2_mean']:6.3f} +/- {s['model_r2_std']:.3f}")
    print(f"  model  MAE : {s['model_mae_mean']:6.3f} +/- {s['model_mae_std']:.3f} CTR pts")
    print(f"  baseline MAE: {s['base_mae_mean']:6.3f} CTR pts  (guess the mean)")
    print()

    if s["beats_baseline"]:
        print(
            f"  VERDICT: model beats the mean-guess baseline by "
            f"{s['mae_improvement']:.3f} CTR pts of error. The title features carry signal."
        )
    else:
        print(
            "  VERDICT: model does NOT beat guessing the mean. On this data the title "
            "features aren't helping — that's an honest finding, not a bug."
        )
    print("=" * 66)
    return s


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
