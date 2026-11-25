# Runs 5-fold cross-validation for one n-gram model and prints the mean and std of each metric.
# Uses data/loader.py, evaluation/cross_validation.py and models/ngram.py; saves the summary in experiments/results.

import argparse
from functools import partial

from mini_copilot.data.loader import load_splits
from mini_copilot.evaluation.cross_validation import run_cv
from mini_copilot.models.ngram import NGramModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate one n-gram model.")
    parser.add_argument("--n", type=int, default=3, help="order of the model, so n-1 tokens of context (default 3)")
    parser.add_argument("--level", choices=["word", "char"], default="word", help="count words or characters")
    parser.add_argument("--smoothing", choices=["add-k", "backoff"], default="add-k", help="how unseen contexts work")
    parser.add_argument("--k", type=float, default=1.0, help="the k in add-k smoothing")
    args = parser.parse_args()

    summary = run_cv(partial(NGramModel, args.n, args.level, args.smoothing, args.k), load_splits()["folds"])
    print()
    for metric, stats in summary.items():
        print(f"{metric:14} {stats['mean']:.4f} ± {stats['std']:.4f}")
