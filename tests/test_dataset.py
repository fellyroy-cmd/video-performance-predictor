"""Tests for load() — the real-vs-synthetic seam and its error handling.

data/videos.csv comes from a human export (YouTube Studio, then hand-trimmed),
so it WILL be malformed sometimes: empty, wrong encoding, a stray '%' on CTR,
too few rows. Before this test file, those cases surfaced as raw pandas/sklearn
tracebacks three calls away from the real problem (EmptyDataError,
UnicodeDecodeError, "could not convert string to float", "n_samples=0"...).
These tests pin the fix: every bad-CSV shape raises one clear BadVideosCsv
telling Dara exactly what to fix, and every good shape still loads cleanly.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.dataset import DATA_CSV, BadVideosCsv, load, make_synthetic, write_sample_csv


@pytest.fixture(autouse=True)
def _clean_data_csv():
    """Every test gets a real_csv-free slate, and cleans up after itself.

    Uses the real DATA_CSV path (not a temp file) because `load()` hardcodes
    it, but always removes the file afterwards so this suite never leaves
    behind something that looks like a real export.
    """
    if os.path.exists(DATA_CSV):
        os.remove(DATA_CSV)
    yield
    if os.path.exists(DATA_CSV):
        os.remove(DATA_CSV)


def _write(text: str, mode: str = "w", **kwargs) -> None:
    with open(DATA_CSV, mode, **kwargs) as f:
        f.write(text)


def test_no_csv_falls_back_to_synthetic():
    df, is_real = load(seed=1)
    assert is_real is False
    assert len(df) > 0


def test_valid_csv_loads_as_real():
    make_synthetic(n=50, seed=1).to_csv(DATA_CSV, index=False)
    df, is_real = load()
    assert is_real is True
    assert len(df) == 50


def test_empty_file_raises_clear_error():
    _write("")
    with pytest.raises(BadVideosCsv, match="empty"):
        load()


def test_missing_columns_raises_clear_error():
    _write("headline,clicks\nHello,5\n" * 15)
    with pytest.raises(BadVideosCsv, match="missing required column"):
        load()


def test_non_numeric_ctr_raises_clear_error_and_names_the_value():
    rows = "".join(f"title {i},4.0\n" for i in range(12))
    _write(f"title,ctr\n{rows}Weird Row,not-a-number\n")
    with pytest.raises(BadVideosCsv, match="not-a-number"):
        load()


def test_percent_sign_ctr_is_tolerated():
    # "4.8%" is a plausible real export quirk (a percent column pasted as text).
    # This should NOT raise — it should be parsed as 4.8.
    rows = "".join(f"title {i},{4.0 + i * 0.1}%\n" for i in range(12))
    _write(f"title,ctr\n{rows}")
    df, is_real = load()
    assert is_real is True
    assert df["ctr"].dtype.kind == "f"
    assert df["ctr"].iloc[0] == pytest.approx(4.0)


def test_too_few_rows_after_cleaning_raises_clear_error():
    # Only 2 rows survive dropna; below the 10-row floor.
    _write("title,ctr\nGood Title,4.8\nAnother Good One,5.1\n,\nBlank CTR,\n")
    with pytest.raises(BadVideosCsv, match="usable row"):
        load()


def test_blank_titles_are_dropped_not_kept_as_empty_strings():
    rows = "".join(f"title {i},4.0\n" for i in range(12))
    _write(f"title,ctr\n{rows}   ,5.0\n")
    df, is_real = load()
    assert "" not in df["title"].tolist()
    assert not any(t.strip() == "" for t in df["title"])


def test_non_utf8_bytes_raise_clear_error():
    _write(b"title,ctr\n\xffbroken,4.8\n", mode="wb")
    with pytest.raises(BadVideosCsv, match="non-UTF-8|readable as text"):
        load()


def test_bad_videos_csv_is_a_value_error_subclass():
    # Callers that only catch ValueError (older code, ad-hoc scripts) still work.
    assert issubclass(BadVideosCsv, ValueError)


# --- write_sample_csv(): the "rehearse the real-data path" helper -----------
# data/README.md tells Dara to call this by hand to test load()'s real-data
# branch before a real export exists. It had no test coverage at all — these
# pin down the two things that matter: it writes a file `load()` accepts as
# real, and it never touches anything outside the `path` it's given.

def test_write_sample_csv_produces_a_file_load_accepts_as_real(tmp_path):
    out = tmp_path / "sample.csv"
    write_sample_csv(path=str(out), n=50, seed=1)

    assert out.exists()

    # write_sample_csv() only writes a file; it doesn't touch DATA_CSV, so point
    # load() at this exact path the same way app.py/train.py do in production.
    import src.dataset as dataset_module

    original_data_csv = dataset_module.DATA_CSV
    try:
        dataset_module.DATA_CSV = str(out)
        df, is_real = load()
    finally:
        dataset_module.DATA_CSV = original_data_csv

    assert is_real is True
    assert len(df) == 50
    assert set(df.columns) >= {"title", "ctr"}


def test_write_sample_csv_is_reproducible_for_a_given_seed(tmp_path):
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    write_sample_csv(path=str(out_a), n=20, seed=7)
    write_sample_csv(path=str(out_b), n=20, seed=7)
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_write_sample_csv_creates_missing_parent_directories(tmp_path):
    # data/README.md's rehearsal snippet calls this with the default path, whose
    # parent (data/) may not exist yet on a fresh clone -- must not crash on that.
    nested = tmp_path / "nested" / "subdir" / "videos.csv"
    write_sample_csv(path=str(nested), n=15, seed=1)
    assert nested.exists()
