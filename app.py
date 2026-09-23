"""Paste a title, get a predicted CTR. The Week-4 UI over the trained model.

    streamlit run app.py

This file is deliberately a thin shell. Every hard part already lives in a
tested module and this just calls it and draws the result:

- ``src/features.py`` turns a title into numbers (the model can only see numbers)
- ``src/train.py``    trains the model and scores a title (``rank_candidates``)
- ``src/dataset.py``  decides real-vs-synthetic data and shouts which one
- ``src/evaluate.py`` k-fold cross-validation vs a mean-guess baseline, so the
  trust panel below can say whether a score is worth anything at all

The same honesty rule as the CLI holds here: with no real ``data/videos.csv`` the
banner turns red and says the numbers are synthetic and mean nothing about
YouTube yet. When your Week-2 export lands, the exact same UI turns green and
starts talking about your real channel — no code change.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dataset import BadVideosCsv, load
from src.evaluate import build_matrix, cross_validate, summarize
from src.features import FEATURE_NAMES, extract
from src.train import rank_candidates, score, train

st.set_page_config(
    page_title="Video Performance Predictor",
    page_icon="\U0001F4C8",  # 📈
    layout="centered",
)


@st.cache_resource(show_spinner="Training the model (once per session)...")
def get_model(seed: int = 42):
    """Train once and cache for the whole session.

    ``@st.cache_resource`` means the RandomForest is fit a single time, not on
    every keystroke — the model is a shared resource, not per-user data. Returns
    ``(model, is_real)`` straight from ``train()``.
    """
    return train(seed=seed)


@st.cache_resource(show_spinner="Cross-validating (checking the model earns its keep)...")
def get_trust_summary(seed: int = 42, k: int = 5):
    """Run the same k-fold CV as ``python -m src.evaluate`` and cache the verdict.

    Reuses ``src/evaluate.py``'s own functions end to end (``load`` ->
    ``build_matrix`` -> ``cross_validate`` -> ``summarize``) instead of
    re-implementing any of the model or scoring logic here. Cached with
    ``@st.cache_resource`` for the same reason as ``get_model``: CV fits 5
    fresh RandomForests, so it must run once per session, not on every rerun
    the moment someone clicks "Score them". Returns ``(summary_dict, is_real)``.
    """
    df, is_real = load(seed=seed)
    X = build_matrix(df["title"].tolist())
    y = df["ctr"].to_numpy(dtype=float)
    folds = cross_validate(X, y, k=k, seed=seed)
    return summarize(folds), is_real


# A malformed data/videos.csv makes train() raise BadVideosCsv (see
# src/dataset.py). Without this, Streamlit's default behaviour is to render
# the raw Python traceback in the browser — technically correct, but exactly
# the wrong moment to hand someone a stack trace: this fires the instant a
# real export lands and something about it is slightly off. Catch it here,
# once, and show the same plain-English message as a stoppable error banner
# instead of letting the rest of the page try (and fail) to render.
try:
    model, is_real = get_model()
except BadVideosCsv as e:
    st.error(f"**Can't load your data/videos.csv:**\n\n{e}")
    st.stop()

st.title("\U0001F4C8 Video Performance Predictor")
st.caption("Paste a title idea, get a predicted click-through rate before I publish.")

# --- The honesty banner: synthetic vs real, impossible to miss --------------
if is_real:
    st.success(
        "**Real data** — trained on `data/videos.csv`. These scores are about "
        "**your channel**."
    )
else:
    st.error(
        "**SYNTHETIC data** — no `data/videos.csv` found. The model has learned "
        "**nothing real about YouTube yet**, only the invented scoring rule in "
        "`src/dataset.py`. Drop your real analytics export at `data/videos.csv` "
        "and reload — the same UI will start scoring your actual channel (Week 2)."
    )

# --- Trust panel: is any score above actually worth trusting? ---------------
# Every title score on this page comes from a RandomForest. A number with no
# context looks trustworthy just because it's a number. This panel runs the
# same honest check as `python -m src.evaluate` (k-fold CV vs a mean-guess
# baseline) so nobody reads meaning into a score before checking whether the
# model beats simply guessing the average.
st.subheader("Can you trust these scores?")
trust, trust_is_real = get_trust_summary()

if trust_is_real:
    st.caption("Cross-validated on `data/videos.csv` — this verdict is about YOUR channel.")
else:
    st.caption(
        "Cross-validated on **SYNTHETIC** data. This verdict only says whether the "
        "model can recover the **fake** rule in `src/dataset.py` — it says nothing "
        "about YouTube. The real test happens once your Week-2 export lands."
    )

col1, col2, col3 = st.columns(3)
col1.metric(
    "Model R² (avg across folds)",
    f"{trust['model_r2_mean']:.3f}",
    help="How much of the variation in CTR the model explains, averaged over "
    f"{trust['k']} folds (std {trust['model_r2_std']:.3f}). Closer to 1 is better; "
    "0 means no better than guessing the mean; negative means worse.",
)
col2.metric(
    "Model MAE",
    f"{trust['model_mae_mean']:.2f} ± {trust['model_mae_std']:.2f}",
    help="Average error in predicted CTR points, mean ± std across folds. Lower is better.",
)
col3.metric(
    "Baseline MAE (guess the mean)",
    f"{trust['base_mae_mean']:.2f}",
    help="Error you'd get by ignoring the title entirely and always guessing the "
    "average CTR. The model has to beat this to be worth anything.",
)

if trust["beats_baseline"]:
    st.success(
        f"**Beats the baseline** by {trust['mae_improvement']:.2f} CTR pts of error "
        "(lower MAE than guessing the mean). The title features carry real signal "
        "on this data — scores below are worth reading."
    )
else:
    st.warning(
        "**Does NOT beat the baseline.** On this data, the title features aren't "
        "helping — the model isn't doing better than just guessing the average CTR "
        "every time. Treat every score below as noise until this flips."
    )

# --- Score title candidates -------------------------------------------------
st.subheader("Score your title ideas")
st.write("One title per line. They come back ranked best-first.")

default_titles = "\n".join(
    [
        "5 AI Tools That Feel Illegal to Know",
        "How to Automate Your Newsletter With Python",
        "My Honest Take on AI Agents in 2026",
        "A Detailed Overview of Artificial Intelligence Frameworks",
        "Is Claude Worth It?",
    ]
)
raw = st.text_area("Title candidates", value=default_titles, height=160)

if st.button("Score them", type="primary"):
    ranked = rank_candidates(model, raw.splitlines())
    if not ranked:
        st.warning("Type at least one title above.")
    else:
        table = pd.DataFrame(
            [
                {"Rank": i, "Predicted CTR (%)": round(ctr, 2), "Title": title}
                for i, (title, ctr) in enumerate(ranked, start=1)
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        # Quick visual: bar height = predicted CTR, best idea on top.
        st.bar_chart(
            table.set_index("Title")["Predicted CTR (%)"], horizontal=True
        )

# --- Why did a title score that way? Feature breakdown ----------------------
st.subheader("Break down one title")
st.write("See exactly which features the model read out of a single title.")
one = st.text_input("A title to inspect", value="5 AI Tools That Feel Illegal to Know")
if one.strip():
    st.metric("Predicted CTR", f"{score(model, one):.2f}%")
    feats = extract(one)
    breakdown = pd.DataFrame(
        {"Feature": list(feats.keys()), "Value": list(feats.values())}
    )
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

# --- What the model learned overall -----------------------------------------
with st.expander("What the model weighs most (feature importances)"):
    st.write(
        "Across all titles, how much each feature moved the model's predictions. "
        "On synthetic data this just reflects the fake rule; on real data it's "
        "the interesting part — what actually earns clicks on my channel."
    )
    importances = pd.DataFrame(
        {"Feature": FEATURE_NAMES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    st.bar_chart(importances.set_index("Feature")["Importance"], horizontal=True)

st.divider()
st.caption(
    "Model: RandomForest on title features. Not a crystal ball — a second "
    "opinion. Project 2 of my build-in-public run."
)
