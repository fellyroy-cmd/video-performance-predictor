"""Where the training data comes from.

Two modes, one clean seam:

1. REAL: if `data/videos.csv` exists, load it. That's your YouTube Studio export,
   trimmed to the columns we need: `title` and `ctr` (click-through rate as a
   percentage, e.g. 4.8). This is the mode that matters — the model only learns
   anything true about YouTube here. That's Week 2's job.

2. SYNTHETIC: if there's no CSV, generate fake titles from templates using a
   KNOWN scoring rule + noise. This exists so the whole pipeline (features ->
   train -> score) runs end-to-end on day one, before any real export. The model
   trained on this has learned NOTHING about YouTube — only about the fake rule
   below. We keep the rule honest and documented so, once real data lands, you
   can literally compare "what the generator baked in" vs "what my channel
   actually rewards." That comparison is a great video moment.

The generator is seeded, so runs and tests are reproducible.
"""

from __future__ import annotations

import csv
import os
import random

import pandas as pd

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "videos.csv")

# ---- pieces to assemble fake titles from -----------------------------------
_TOPICS = [
    "AI Tools", "ChatGPT", "Claude", "Python", "Automation", "Prompt Engineering",
    "AI Agents", "n8n", "Midjourney", "AI News", "Machine Learning", "RAG",
]
_HOOKS = [
    "{n} {topic} That Feel Illegal to Know",
    "How to Use {topic} Like a Pro",
    "I Tried {topic} for 30 Days",
    "Why {topic} Is Changing Everything",
    "The Truth About {topic}",
    "{topic}: Everything You Need to Know",
    "Stop Using {topic} Wrong",
    "{n} {topic} Mistakes Beginners Make",
    "This {topic} Trick Saved Me Hours",
    "My Honest Take on {topic}",
    "{topic} Explained in {n} Minutes",
    "Is {topic} Worth It in 2026?",
]


def _synthetic_ctr(title: str, rng: random.Random) -> float:
    """The fake, KNOWN rule the synthetic model is really learning.

    Deliberately rewards things creators believe help — numbers, questions,
    'how to', first person, moderate length — and penalises very long titles.
    Plus noise, so it's not trivially perfect. Base CTR ~4%, clamped sane.

    IMPORTANT: this is invented. Its whole point is to be a fixed target we can
    later contrast against what real data says. Do not read insight into it.
    """
    from .features import extract  # local import avoids a cycle at module load

    f = extract(title)
    ctr = 4.0
    ctr += 0.8 * f["has_number"]
    ctr += 0.6 * f["is_question"]
    ctr += 0.7 * f["is_how_to"]
    ctr += 0.4 * f["first_person_count"]
    ctr += 0.25 * f["power_word_count"]
    ctr += 0.15 * f["sentiment_score"]
    ctr -= 0.03 * max(0, f["char_length"] - 55)  # long titles get truncated -> hurt
    ctr += rng.gauss(0, 0.6)                      # noise: reality is messy
    return round(max(0.5, min(15.0, ctr)), 2)


def make_synthetic(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate `n` fake (title, ctr) rows with a seeded RNG (reproducible)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        hook = rng.choice(_HOOKS)
        title = hook.format(topic=rng.choice(_TOPICS), n=rng.choice([3, 5, 7, 10]))
        rows.append({"title": title, "ctr": _synthetic_ctr(title, rng)})
    return pd.DataFrame(rows)


class BadVideosCsv(ValueError):
    """data/videos.csv exists but isn't usable. Message always points at the fix.

    A dedicated exception (not a bare ValueError) so callers — and tests — can
    tell "your export needs a fix" apart from other, unrelated ValueErrors
    instead of pattern-matching on a message string.
    """


_HELP = "See data/README.md for the exact export steps."


def load(seed: int = 42) -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_real).

    `is_real` is True only when a real `data/videos.csv` was found and loaded, so
    callers (train.py) can shout the honest warning when it's False.

    Every failure mode here raises `BadVideosCsv` with a plain-language message
    instead of letting a raw pandas/sklearn error (EmptyDataError,
    UnicodeDecodeError, "could not convert string to float", "n_samples=0"...)
    surface three function calls away from the actual cause. The goal: whatever
    is wrong with the export, Dara sees ONE message that says what to fix.
    """
    if not os.path.exists(DATA_CSV):
        return make_synthetic(seed=seed), False

    try:
        df = pd.read_csv(DATA_CSV)
    except pd.errors.EmptyDataError as e:
        raise BadVideosCsv(f"{DATA_CSV} is empty (no header, no rows). {_HELP}") from e
    except UnicodeDecodeError as e:
        raise BadVideosCsv(
            f"{DATA_CSV} isn't readable as text (found non-UTF-8 bytes). "
            f"Re-export and save it as plain CSV (UTF-8). {_HELP}"
        ) from e
    except pd.errors.ParserError as e:
        raise BadVideosCsv(f"{DATA_CSV} isn't valid CSV: {e} {_HELP}") from e

    missing = {"title", "ctr"} - set(df.columns)
    if missing:
        raise BadVideosCsv(
            f"{DATA_CSV} is missing required column(s): {sorted(missing)}. "
            f"Expected at least 'title' and 'ctr'. {_HELP}"
        )

    # A common real-export quirk: CTR saved as "4.8%" instead of 4.8. Strip a
    # trailing '%' before the numeric check below so that case gets the same
    # clear message as any other bad value, not a raw sklearn crash later.
    if df["ctr"].dtype == object:
        df["ctr"] = df["ctr"].astype(str).str.rstrip("%")

    ctr_numeric = pd.to_numeric(df["ctr"], errors="coerce")
    bad_ctr = ctr_numeric.isna() & df["ctr"].notna()
    if bad_ctr.any():
        examples = df.loc[bad_ctr, "ctr"].astype(str).unique()[:3]
        raise BadVideosCsv(
            f"{DATA_CSV} has non-numeric 'ctr' value(s), e.g. {list(examples)}. "
            f"'ctr' must be a plain number like 4.8. {_HELP}"
        )
    df["ctr"] = ctr_numeric

    df = df.dropna(subset=["title", "ctr"])
    df = df[df["title"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise BadVideosCsv(
            f"{DATA_CSV} only has {len(df)} usable row(s) after cleaning "
            f"(need at least 10 to train/test-split anything meaningful). "
            f"Export more videos, or check that 'title' and 'ctr' are both filled in. {_HELP}"
        )

    return df, True


def write_sample_csv(path: str = DATA_CSV, n: int = 400, seed: int = 42) -> None:
    """Helper: dump a synthetic CSV to the real path, matching the export schema.

    Handy for testing the real-data code path before your export exists. NOT run
    automatically — that would fake real data. Call it by hand if you want to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = make_synthetic(n=n, seed=seed)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
