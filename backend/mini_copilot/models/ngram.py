"""Character and word n-gram models (n = 1..10) with smoothing.

The dataset is small, so unseen n-grams are common. Use a smoothing method
(add-k, Kneser-Ney, or backoff to shorter n) so they don't get zero probability.
"""

from collections import Counter, defaultdict

from .base import Predictor


class NGramModel(Predictor):
    def __init__(self, n: int, level: str = "word", smoothing: str = "add-k", k: float = 1.0):
        assert level in ("word", "char")
        self.n, self.level, self.smoothing, self.k = n, level, smoothing, k
        self.name = f"{level}-{n}gram"
        self.counts: dict[tuple, Counter] = defaultdict(Counter)

    def train(self, sources: list[str]) -> None:
        raise NotImplementedError

    def predict(self, context: str, k: int = 5) -> list[str]:
        # For the char model, keep generating characters until a word boundary
        # so it returns whole-word suggestions like the word model.
        raise NotImplementedError
