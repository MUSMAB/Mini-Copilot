"""Load source files and cross-validation folds from data/."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_splits() -> dict:
    """Read data/processed/splits.json (made by scripts/make_splits.py)."""
    return json.loads((DATA_DIR / "processed" / "splits.json").read_text())


def read_source(rel_path: str) -> str:
    return (DATA_DIR / "raw" / rel_path).read_text(encoding="utf-8", errors="replace")


def fold_split(folds: list[list[str]], k: int) -> tuple[list[str], list[str]]:
    """Fold k is the test set, the other folds are training."""
    test = folds[k]
    train = [f for i, fold in enumerate(folds) if i != k for f in fold]
    return train, test
