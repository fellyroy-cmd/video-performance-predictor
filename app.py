"""Paste a title, get a predicted CTR. The Week-4 UI over the trained model.

    streamlit run app.py

This file is deliberately a thin shell. Every hard part already lives in a
tested module and this just calls it and draws the result:

- ``src/features.py`` turns a title into numbers (the model can only see numbers)
- ``src/train.py``    trains the model and scores a title (``rank_candidates``)
- ``src/dataset.py``  decides real-vs-synthetic data and shouts which one

The same honesty rule as the CLI holds here: with no real ``data/videos.csv`` the
banner turns red and says the numbers are synthetic and mean nothing about
YouTube yet. When your Week-2 export lands, the exact same UI turns green and
starts talking about your real channel — no code change.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

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


model, is_real = get_model()

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
