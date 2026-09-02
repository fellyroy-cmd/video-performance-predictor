"""Train the model and score title candidates. One command, end to end.

    python -m src.train

What happens:
1. Load data (real data/videos.csv if present, else synthetic — it tells you which).
2. Turn every title into features.
3. Split train/test, fit a RandomForest regressor (predict CTR from title features).
4. Print how well it did (R^2 on held-out data) and which features mattered most.
5. Score a few example title candidates so you see the payoff immediately.

RandomForest, not linear regression, on purpose: title effects aren't linear
(a number helps, but ten numbers doesn't help ten times) and a forest handles
that without us hand-tuning anything. On a small real channel it also won't melt
down the way a big model would.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .dataset import load
from .features import FEATURE_NAMES, feature_vector


def build_matrix(titles) -> np.ndarray:
    """Stack a list of titles into a 2D feature matrix (one row per title)."""
    return np.array([feature_vector(t) for t in titles], dtype=float)


def train(seed: int = 42):
    """Train on whatever load() gives us. Returns (model, is_real)."""
    df, is_real = load(seed=seed)
    X = build_matrix(df["title"].tolist())
    y = df["ctr"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    model = RandomForestRegressor(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("=" * 66)
    if is_real:
        print("DATA: real (data/videos.csv) — these numbers are about YOUR channel.")
    else:
        print("DATA: SYNTHETIC (no data/videos.csv found).")
        print("  [!] The model has learned NOTHING real about YouTube yet - only")
        print("      the fake scoring rule in dataset.py. This mode proves the")
        print("      pipeline works. Drop your real export at data/videos.csv to")
        print("      make it mean something (that's Week 2).")
    print("=" * 66)
    print(f"Rows: {len(df)}   |   Test R^2: {r2:0.3f}   |   Test MAE: {mae:0.3f} CTR pts")
    print()

    print("Which title features moved the prediction most:")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, imp in importances:
        bar = "#" * int(round(imp * 40))
        print(f"  {name:<20} {imp:0.3f} {bar}")
    print()

    return model, is_real


def score(model, title: str) -> float:
    """Predicted CTR for a single new title candidate."""
    return float(model.predict(build_matrix([title]))[0])


def _demo(model) -> None:
    print("Scoring a few example candidates (predicted CTR):")
    candidates = [
        "5 AI Tools That Feel Illegal to Know",
        "How to Automate Your Newsletter With Python",
        "My Honest Take on AI Agents in 2026",
        "A Detailed Overview of Artificial Intelligence Frameworks and Their Uses",
        "Is Claude Worth It?",
    ]
    ranked = sorted(candidates, key=lambda t: score(model, t), reverse=True)
    for t in ranked:
        print(f"  {score(model, t):5.2f}  {t}")


def main() -> None:
    model, _ = train()
    _demo(model)


if __name__ == "__main__":
    main()
