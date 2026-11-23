# Runs 5-fold cross-validation for the AST model and prints the mean and std of each metric.
# Uses data/loader.py, evaluation/cross_validation.py and models/ast_model.py; saves experiments/results/ast_cv.json.

from mini_copilot.data.loader import load_splits
from mini_copilot.evaluation.cross_validation import run_cv
from mini_copilot.models.ast_model import ASTModel

if __name__ == "__main__":
    summary = run_cv(ASTModel, load_splits()["folds"])
    print()
    for metric, stats in summary.items():
        print(f"{metric:14} {stats['mean']:.4f} ± {stats['std']:.4f}")
