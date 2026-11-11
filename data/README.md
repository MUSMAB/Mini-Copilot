# Data

`dataset.json` lists the 10 Python projects we train on, pinned to exact commits. See [docs/DATASET.md](../docs/DATASET.md) for why each was picked.

```
python scripts/download_dataset.py   # fills data/raw/ with the .py files
python scripts/make_splits.py        # writes data/processed/splits.json (70/30 + 5 folds)
```

`raw/` and `processed/` are git-ignored, so everyone regenerates them locally with the same seed.
