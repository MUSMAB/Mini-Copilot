"""Run 5-fold CV for every model and report mean and std for each metric."""


def run_cv(model_factory, folds: list[list[str]]) -> dict:
    raise NotImplementedError
