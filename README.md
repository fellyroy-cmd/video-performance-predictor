# Video Performance Predictor

Learn what drives clicks on **my own** YouTube channel, then score new title ideas
before I publish. Project 2 of my 9-month "build an AI product every month" run
([Project Alpha](https://github.com/fellyroy-cmd)).

> **Status: Week 1 — early scaffold, on purpose.** Right now it trains on
> *synthetic* data (fake titles + a made-up scoring rule) so the whole pipeline
> runs end to end today. **It has not learned anything real about YouTube yet.**
> That starts when I load my actual analytics export (Week 2). The code shouts
> this at you when it's running on synthetic data — by design.

## The idea
I pick titles and thumbnails on vibes. But I have the data to do better — every
video's title and its click-through rate sit in YouTube Studio. So:

1. **Turn each title into numbers** — length, has a number, is it a question,
   "how-to" framing, first person, power words, a rough sentiment score
   (`src/features.py`).
2. **Train a model** to predict CTR from those numbers (`src/train.py`).
3. **Score new candidates** — paste a title, get a predicted CTR and see which
   features helped, in a small Streamlit app (`app.py`).

## Run it
```bash
pip install -r requirements.txt
python -m src.train
```
With no `data/videos.csv`, it generates synthetic data, trains, prints how well
it did (held-out R²) and which title features mattered, then scores a few example
titles. Drop a real export at `data/videos.csv` (see [`data/README.md`](data/README.md))
and the exact same command retrains on real numbers.

### Is the model actually any good?
```bash
python -m src.evaluate
```
`train` fits one model on one split — on a small channel that single R² can lie.
`evaluate` is the honesty check: it runs **k-fold cross-validation** (every row
gets to be test data once) and compares the model against a **baseline** that
ignores the title and just guesses the average CTR. If the model can't beat that
baseline, the title features carry no real signal — and the tool says so out
loud. This is what tells me whether the thing works once real data lands.

### The app — paste a title, get a score
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens a browser tab: paste your title candidates (one per line), get them ranked
by predicted CTR, break down which features the model read out of any single
title, and see what the model weighs most overall. Same honesty rule as the CLI
— a big red banner until a real `data/videos.csv` is loaded, green once it is.

Right below that banner is a **"Can you trust these scores?"** panel — the same
k-fold cross-validation from `src/evaluate.py`, run once per session and shown
as R², MAE, and the beats-baseline verdict, so nobody reads meaning into a
ranked title before checking whether the model beats guessing the average.

The UI is a thin shell over the tested `rank_candidates()` logic in `src/train.py`
and the tested CV logic in `src/evaluate.py`.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
The feature extractor is the one part that has to stay correct — every video is
seen by the model through it — so that's what the tests pin down.

## Roadmap (this month)
- [x] **Week 1** — scaffold, feature extraction + tests, synthetic pipeline that runs
- [ ] **Week 2** — load my real YouTube export, retrain, compare to synthetic
- [x] **Week 3 (started early)** — evaluation harness: k-fold cross-validation + mean-guess baseline (`src/evaluate.py`). Built ahead on synthetic data; drops onto the real export unchanged.
- [x] **Week 4 (UI built early)** — Streamlit "paste a title, get a score" app (`app.py`): ranked candidates, per-title feature breakdown, feature-importance chart. Built ahead on synthetic data; turns real the moment the Week-2 export lands. Demo GIF + ship still to come.

## Honesty note
The synthetic scoring rule in `src/dataset.py` is invented. Its only job is to
give the plumbing something to chew on and to be a fixed thing I can later
contrast against what my real channel actually rewards. Don't read insight into
the synthetic numbers — the insight is Week 2's job.

MIT licensed. Built in public, ugly early commits included.
