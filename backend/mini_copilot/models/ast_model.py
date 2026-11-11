"""AST-based predictor: suggests names and keywords that fit the syntax at the cursor."""

from .base import Predictor


class ASTModel(Predictor):
    name = "ast"

    def train(self, sources: list[str]) -> None:
        """Learn which tokens follow which AST node types."""
        raise NotImplementedError

    def predict(self, context: str, k: int = 5) -> list[str]:
        raise NotImplementedError
