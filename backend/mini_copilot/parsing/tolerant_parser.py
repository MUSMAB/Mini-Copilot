"""Error-tolerant parsing for half-written code.

Python's ast module fails on any syntax error, which is every keystroke while
someone is typing. Tree-sitter keeps going and marks the broken parts with
ERROR / MISSING nodes, so we use it for the prediction context and use ast
only to check that a completed suggestion is valid Python.
"""

import ast


def parse_partial(source: str):
    """Parse possibly broken code with tree-sitter and return the tree."""
    raise NotImplementedError("wire up tree-sitter-python here")


def is_valid_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False
