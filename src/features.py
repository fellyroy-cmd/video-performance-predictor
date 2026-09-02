"""Turn a YouTube title (a string) into numbers a model can learn from.

This is the heart of the whole project: a model can't read "5 AI Tools That
Feel Illegal to Know" — it can only learn from numbers. So every title gets
turned into the same fixed set of features (length, has a number, is it a
question, etc.). Every video, real or synthetic, goes through THIS function,
so the model always sees titles the same way.

Design choices worth knowing:
- Everything is a plain int/float. No external NLP library — a small channel
  doesn't have enough data to justify heavy models, and keeping it simple keeps
  the repo cloneable from the YouTube walkthrough with no surprises.
- "sentiment" here is a deliberately tiny lexicon score, NOT real NLP. It's a
  v1 placeholder, clearly named, good enough to see if emotional words matter.
  If Week 3 shows it's useless, delete it — that's an honest finding, not a bug.
"""

from __future__ import annotations

import re

# Words creators lean on to earn the click. Not science — a starting hypothesis
# we can actually test once real data is loaded (do these move CTR on MY channel?).
POWER_WORDS = {
    "secret", "secrets", "illegal", "insane", "crazy", "shocking", "ultimate",
    "best", "worst", "never", "always", "proven", "easy", "fast", "free",
    "new", "now", "stop", "why", "how", "you", "your", "instantly", "hack",
    "hacks", "mistake", "mistakes", "nobody", "everyone", "actually", "real",
}

# Tiny sentiment lexicon. Positive minus negative, that's it. A placeholder for
# "does emotional charge in the title matter?" — replace with something real
# only if the data says it's worth it.
POSITIVE_WORDS = {
    "best", "great", "amazing", "easy", "free", "win", "love", "smart",
    "powerful", "proven", "ultimate", "insane", "genius", "perfect", "fast",
}
NEGATIVE_WORDS = {
    "worst", "bad", "hard", "fail", "failed", "mistake", "mistakes", "stop",
    "wrong", "never", "illegal", "shocking", "avoid", "scam", "dead", "broke",
}

QUESTION_WORDS = {"how", "what", "why", "when", "where", "which", "who", "should", "can", "is", "are", "do"}

# The exact feature order the model sees. Keep this list and extract() in sync —
# the tests enforce that they match, so you can't silently drift.
FEATURE_NAMES = [
    "char_length",
    "word_count",
    "number_count",
    "has_number",
    "is_question",
    "is_how_to",
    "first_person_count",
    "allcaps_word_count",
    "exclamation_count",
    "power_word_count",
    "sentiment_score",
]


def _words(title: str) -> list[str]:
    """Lowercased word tokens, punctuation stripped. 'How-To' -> ['how', 'to']."""
    return re.findall(r"[a-zA-Z']+", title.lower())


def extract(title: str) -> dict[str, float]:
    """One title -> a dict of named features. See FEATURE_NAMES for the full set.

    Kept as a dict (not a bare list) so it's readable and self-labelling; the
    model layer turns it into a row via feature_vector().
    """
    if title is None:
        title = ""
    words = _words(title)
    word_set = set(words)

    numbers = re.findall(r"\d+", title)

    # ALL-CAPS words of length >= 2 ("AI", "STOP") — a real emphasis signal in titles.
    allcaps = re.findall(r"\b[A-Z]{2,}\b", title)

    first_person = sum(1 for w in words if w in {"i", "i'm", "my", "me", "i've"})

    is_question = 1 if ("?" in title or (words and words[0] in QUESTION_WORDS)) else 0
    # "how to" as adjacent tokens — catches "How to", "How-To", extra spaces;
    # does NOT fire on "How I Automate" (how, but not followed by to).
    is_how_to = 1 if any(
        words[i] == "how" and words[i + 1] == "to" for i in range(len(words) - 1)
    ) else 0

    sentiment = sum(w in POSITIVE_WORDS for w in word_set) - sum(w in NEGATIVE_WORDS for w in word_set)

    return {
        "char_length": len(title),
        "word_count": len(words),
        "number_count": len(numbers),
        "has_number": 1 if numbers else 0,
        "is_question": is_question,
        "is_how_to": is_how_to,
        "first_person_count": first_person,
        "allcaps_word_count": len(allcaps),
        "exclamation_count": title.count("!"),
        "power_word_count": sum(1 for w in words if w in POWER_WORDS),
        "sentiment_score": float(sentiment),
    }


def feature_vector(title: str) -> list[float]:
    """Same features as extract(), but as a plain ordered row for the model."""
    f = extract(title)
    return [float(f[name]) for name in FEATURE_NAMES]
