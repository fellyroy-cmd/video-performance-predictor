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
