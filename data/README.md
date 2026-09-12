# data/

Drop your real YouTube analytics export here as **`videos.csv`**.

Minimum columns the code needs:

| column  | what it is                                   | example                                   |
|---------|----------------------------------------------|-------------------------------------------|
| `title` | the video title, exactly as published        | `5 AI Tools That Feel Illegal to Know`    |
| `ctr`   | impressions click-through rate, as a percent | `4.8`                                     |

Extra columns are fine — they're ignored for now.

**How to get it:** YouTube Studio → Analytics → Advanced mode → Content tab →
add the "Impressions click-through rate" column → Export → clean it down to
`title` and `ctr` → save as `data/videos.csv`.

Until this file exists, the pipeline runs on **synthetic** data so you can test
everything. `videos.csv` is git-ignored — it's your data, not the tool.

Want to rehearse the real-data path before your export exists? Run:

```python
from src.dataset import write_sample_csv; write_sample_csv()
```

That writes a synthetic `videos.csv` in the right schema (delete it afterwards).

## If `videos.csv` exists but the app won't load it
`load()` checks the file before training and raises one clear error instead of
a raw crash. It will tell you exactly what to fix if:
- the file is empty, not valid CSV, or saved with the wrong encoding — re-export
  and save as plain UTF-8 CSV
- `title` or `ctr` columns are missing — check your Advanced-mode export has
  both, then trim to just those two
- `ctr` has a non-numeric value — a stray `%` sign (`"4.8%"`) is handled
  automatically, but text or blanks in that column are not; fix the cell
- fewer than 10 usable rows remain after removing blanks — export more videos,
  or check that most rows actually have both `title` and `ctr` filled in

The error message always names the exact problem — read it, fix the CSV, rerun.
