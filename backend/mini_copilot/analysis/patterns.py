"""AST rules for common beginner patterns (long if/elif chains, nested loops, etc.)."""

import ast
from dataclasses import dataclass


@dataclass
class Finding:
    rule: str
    line: int
    message: str
    suggestion: str


def find_patterns(source: str) -> list[Finding]:
    """Run all AST rules over the source."""
    raise NotImplementedError


def count_if_chain(node: ast.If) -> int:
    """Length of an if/elif/elif... chain starting at node."""
    length, cur = 1, node
    while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
        cur = cur.orelse[0]
        length += 1
    return length
