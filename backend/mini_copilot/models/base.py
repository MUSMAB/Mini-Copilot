"""Shared interface so the evaluator and the editor can swap models freely."""

from abc import ABC, abstractmethod


class Predictor(ABC):
    name = "base"

    @abstractmethod
    def train(self, sources: list[str]) -> None: ...

    @abstractmethod
    def predict(self, context: str, k: int = 5) -> list[str]:
        """Top k next-word suggestions for the code typed so far."""
