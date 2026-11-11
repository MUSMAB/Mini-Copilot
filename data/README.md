# Data

`dataset.json` lists the 10 Python projects we train on, pinned to exact commits.

```
python3 scripts/download_dataset.py   # fills data/raw/ with the .py files
python3 scripts/make_splits.py        # only if you need to rebuild the split
```

`raw/` is git-ignored: everyone downloads the same pinned commits.

`processed/splits.json` is committed so the whole team uses the exact same
train/test split and 5 folds. Don't regenerate and commit it unless the team
agrees, because it changes everyone's results.
