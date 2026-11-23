"""5-fold cross-validation for the AST model.

Usage (from the repo root, after `pip install -e backend`):
    python3 scripts/run_ast_cv.py
Writes experiments/results/ast_cv.json.
"""

from mini_copilot.data.loader import load_splits
from mini_copilot.evaluation.cross_validation import run_cv
from mini_copilot.models.ast_model import ASTModel

if __name__ == "__main__":
    summary = run_cv(ASTModel, load_splits()["folds"])
    print()
    for metric, stats in summary.items():
        print(f"{metric:14} {stats['mean']:.4f} ± {stats['std']:.4f}")
