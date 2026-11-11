"""Write the 70/30 train/test split and the 5 cross-validation folds.

Files that are empty (mostly blank __init__.py files) or that Python cannot
parse are skipped, because there is nothing useful to predict in them.

Files are shuffled with a fixed seed inside each project, so every project
shows up in every split. TheAlgorithms/Python is about 10x bigger than the
others, so each project is capped at --max-files-per-project files (default
300) to keep the data balanced. Use 0 for no cap.

The result is committed to git, so everyone uses the same split. Only rerun
this if the team agrees to change the split. (Which files count as
"unparsable" can depend on your Python version, so rerunning on a different
version may give a slightly different split.)

Usage: python3 scripts/make_splits.py
Output: data/processed/splits.json
"""

import argparse
import ast
import json
import random
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def check_file(rel: str) -> str:
    """Return 'ok', 'empty', or 'unparsable' for one file under data/raw/."""
    text = (RAW / rel).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return "empty"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old files trigger harmless escape-sequence warnings
            ast.parse(text)
    except (SyntaxError, ValueError):
        return "unparsable"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=329)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--max-files-per-project", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train, test = [], []
    folds = [[] for _ in range(args.folds)]
    skipped_empty = 0
    skipped_unparsable = []

    for project in sorted(p for p in RAW.iterdir() if p.is_dir()):
        # as_posix() keeps forward slashes so Windows and Mac get the same file
        all_files = sorted(f.relative_to(RAW).as_posix() for f in project.rglob("*.py"))
        files = []
        for rel in all_files:
            status = check_file(rel)
            if status == "ok":
                files.append(rel)
            elif status == "empty":
                skipped_empty += 1
            else:
                skipped_unparsable.append(rel)

        rng.shuffle(files)
        if args.max_files_per_project:
            files = files[: args.max_files_per_project]
        cut = int(len(files) * (1 - args.test_ratio))
        train += files[:cut]
        test += files[cut:]
        for i, f in enumerate(files):
            folds[i % args.folds].append(f)

    out = ROOT / "data" / "processed" / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_ratio": args.test_ratio,
                "max_files_per_project": args.max_files_per_project,
                "skipped_empty": skipped_empty,
                "skipped_unparsable": skipped_unparsable,
                "train": train,
                "test": test,
                "folds": folds,
            },
            indent=1,
        )
    )
    print(f"train={len(train)} test={len(test)} folds={[len(f) for f in folds]} -> {out}")
    print(f"skipped: {skipped_empty} empty, {len(skipped_unparsable)} unparsable")


if __name__ == "__main__":
    main()
