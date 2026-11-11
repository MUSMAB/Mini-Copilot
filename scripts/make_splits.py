"""Write the 70/30 train/test split and the 5 cross-validation folds.

Files are shuffled with a fixed seed inside each project, so every project
shows up in every split. TheAlgorithms/Python is about 10x bigger than the
others, so --max-files-per-project lets you cap it to keep the data balanced.

Usage: python scripts/make_splits.py --max-files-per-project 300
Output: data/processed/splits.json
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=329)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--max-files-per-project", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train, test = [], []
    folds = [[] for _ in range(args.folds)]

    for project in sorted(p for p in RAW.iterdir() if p.is_dir()):
        files = sorted(str(f.relative_to(RAW)) for f in project.rglob("*.py"))
        rng.shuffle(files)
        if args.max_files_per_project:
            files = files[: args.max_files_per_project]
        cut = int(len(files) * (1 - args.test_ratio))
        train += files[:cut]
        test += files[cut:]
        for i, f in enumerate(files):
            folds[i % args.folds].append(f)

    out = ROOT / "data" / "processed" / "splits.json"
    out.write_text(json.dumps({"seed": args.seed, "train": train, "test": test, "folds": folds}, indent=1))
    print(f"train={len(train)} test={len(test)} folds={[len(f) for f in folds]} -> {out}")


if __name__ == "__main__":
    main()
