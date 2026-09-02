"""Tests for title feature extraction — the one part that must stay correct,
because every video (real or synthetic) is seen by the model through it.

These are real assertions on hand-picked titles, not smoke tests: if someone
changes the tokeniser or a rule, one of these should go red.
"""

from src.features import FEATURE_NAMES, extract, feature_vector


def test_all_feature_names_present_and_ordered():
    # extract() and feature_vector() must agree on exactly these keys, in order.
    f = extract("Hello World")
    assert set(f.keys()) == set(FEATURE_NAMES)
    assert len(feature_vector("Hello World")) == len(FEATURE_NAMES)


def test_numbers_are_counted():
    f = extract("5 AI Tools and 3 Tricks")
    assert f["has_number"] == 1
    assert f["number_count"] == 2


def test_no_number():
    f = extract("Just Some Words Here")
    assert f["has_number"] == 0
    assert f["number_count"] == 0


def test_question_detected_by_mark_and_by_leading_word():
    assert extract("Is Claude Worth It?")["is_question"] == 1
    assert extract("How Do Agents Work")["is_question"] == 1   # leading question word
    assert extract("Agents Are Cool")["is_question"] == 0


def test_how_to_framing():
    assert extract("How to Automate Your Newsletter")["is_how_to"] == 1
    assert extract("How I Automate My Newsletter")["is_how_to"] == 0  # 'how' but not 'how to'
    assert extract("Automating Newsletters")["is_how_to"] == 0


def test_first_person_counts_i_and_my():
    f = extract("I Tried This and My Life Changed")
    assert f["first_person_count"] == 2


def test_allcaps_words():
    f = extract("STOP Using AI Wrong")
    assert f["allcaps_word_count"] == 2  # STOP and AI


def test_single_letter_caps_not_counted_as_allcaps():
    # 'A' is a single capital letter, not an emphasis word — must not count.
    assert extract("A Simple Guide")["allcaps_word_count"] == 0


def test_exclamations():
    assert extract("Wow! This Works!")["exclamation_count"] == 2


def test_power_and_sentiment():
    f = extract("The Best Free Secret Nobody Tells You")
    assert f["power_word_count"] >= 3          # best, free, secret, nobody, you
    assert f["sentiment_score"] >= 1           # 'best' + 'free' positive


def test_empty_and_none_are_safe():
    for bad in ("", None):
        f = extract(bad)
        assert f["char_length"] == 0
        assert f["word_count"] == 0
        assert len(feature_vector(bad)) == len(FEATURE_NAMES)


def test_feature_vector_matches_extract_order():
    title = "5 Ways to Win With AI in 2026"
    f = extract(title)
    vec = feature_vector(title)
    assert vec == [float(f[name]) for name in FEATURE_NAMES]
