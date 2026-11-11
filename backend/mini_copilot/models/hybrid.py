"""Hybrid model: best n-gram model, re-ranked or filtered by AST context."""

from .base import Predictor


class HybridModel(Predictor):
    name = "hybrid"

    def __init__(self, ngram: Predictor, ast_model: Predictor):
        self.ngram, self.ast_model = ngram, ast_model

    def train(self, sources: list[str]) -> None:
        self.ngram.train(sources)
        self.ast_model.train(sources)

    def predict(self, context: str, k: int = 5) -> list[str]:
        raise NotImplementedError
