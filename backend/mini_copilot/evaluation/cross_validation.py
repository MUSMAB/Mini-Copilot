"""Run 5-fold CV for every model and report mean and std for each metric."""

import json
from statistics import mean, pstdev

from ..data.loader import DATA_DIR, fold_split, read_source
from ..parsing.tolerant_parser import completes_to_valid_python
from .metrics import normalized_edit_distance, top1_accuracy, topk_accuracy
from .simulate import make_samples

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"


def run_cv(model_factory, folds: list[list[str]], max_samples: int = 20, save: bool = True) -> dict:
    """Train a fresh model on 4 folds, test on the 5th, repeat for every fold.

    model_factory is called with no arguments and returns an untrained Predictor.
    """
    per_fold: dict[str, list[float]] = {"top1": [], "top5": [], "edit_distance": [], "valid": []}
    name = "model"
    for k in range(len(folds)):
        train_files, test_files = fold_split(folds, k)
        model = model_factory()
        name = model.name
        model.train([read_source(f) for f in train_files])

        contexts, predictions, targets = [], [], []
        for f in test_files:
            for context, word in make_samples(read_source(f), max_samples):
                contexts.append(context)
                predictions.append(model.predict(context, 5))
                targets.append(word)

        per_fold["top1"].append(top1_accuracy(predictions, targets))
        per_fold["top5"].append(topk_accuracy(predictions, targets, 5))
        per_fold["edit_distance"].append(
            mean(normalized_edit_distance(p[0] if p else "", t) for p, t in zip(predictions, targets))
        )
        # Does the top suggestion, spliced in and with the missing syntax filled in, still parse?
        valid = [completes_to_valid_python(c, p[0]) for c, p in zip(contexts, predictions) if p]
        per_fold["valid"].append(mean(valid) if valid else 0.0)
        print(f"{name} fold {k}: " + "  ".join(f"{m}={v[-1]:.4f}" for m, v in per_fold.items()), flush=True)

    summary = {m: {"mean": mean(v), "std": pstdev(v), "folds": v} for m, v in per_fold.items()}
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{name}_cv.json").write_text(json.dumps(summary, indent=1))
    return summary
