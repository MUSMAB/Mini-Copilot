"""Simulate typing: cut a test file at a word boundary and ask the model for the next word.

Steps for each sample: take the full file, truncate it right before a token,
predict, then (for AST/hybrid) splice the suggestion in, let the tolerant
parser fill in missing syntax, and check the result still parses.
"""


def make_samples(source: str, max_samples: int = 50, seed: int = 329) -> list[tuple[str, str]]:
    """Return (context, expected_next_word) pairs from one file."""
    raise NotImplementedError
