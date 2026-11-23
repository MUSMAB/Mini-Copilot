"""Error-tolerant parsing for half-written code.

Python's ast module fails on any syntax error, which is every keystroke while
someone is typing. Tree-sitter keeps going and marks the broken parts with
ERROR / MISSING nodes, so we use it for the prediction context and use ast
only to check that a completed suggestion is valid Python.

Tree-sitter's recovery is shallow, though: `x = foo(` comes back as one flat
ERROR node with no `call` / `argument_list` inside it. So before parsing we
repair the prefix (close open brackets, add a placeholder operand, add a `pass`
body after a block header). The repaired text parses cleanly, and the nodes
around the last real token then tell us which syntactic slot the cursor is in.
"""

import ast
import threading
import warnings
from typing import NamedTuple

_local = threading.local()  # one parser per thread; a tree-sitter Parser is not thread-safe

PLACEHOLDER = "_"
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = frozenset(_OPEN.values())
_BLOCK_KEYWORDS = frozenset(
    {"def", "class", "if", "elif", "else", "while", "for", "try", "except", "finally", "with", "async"}
)
# After one of these tokens the code is incomplete until an operand follows.
_NEEDS_OPERAND = frozenset(
    {"=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", ">>=", "<<=", "&=", "|=", "^=", "@=", ":=",
     "+", "-", "*", "/", "//", "%", "**", "@", "&", "|", "^", "~", "<<", ">>",
     "<", ">", "<=", ">=", "==", "!=", "<>", "->", ".", ",", "(", "[", "{",
     "not", "and", "or", "in", "is", "if", "elif", "while", "assert", "raise", "del",
     "return", "yield", "await", "import", "from", "as", "with", "global", "nonlocal",
     "class", "def", "else", "except", "for"}
)  # fmt: skip


class Leaf(NamedTuple):
    """One token of a parse tree, plus the tree context around it."""

    type: str  # keywords and punctuation are their own type: "def", "(", ":"
    text: str  # "" for strings (we never need their contents)
    named: bool  # False for keywords and punctuation
    row: int  # start row / column (column in bytes)
    col: int
    end_row: int
    end_byte: int
    up: tuple  # nearest ancestor types, without the ones the cursor can't know about yet
    stmts: tuple  # ((statement type, owner of its block, column), ...) outermost first
    depth: int  # number of brackets still open after this token
    field: str  # the role its branch plays in up[0] ("left", "right", "name", "condition", ...), if any


class PartialContext(NamedTuple):
    text: str  # the source with the missing syntax filled in
    leaves: list  # only the leaves that were really typed (not the filler)
    in_string_or_comment: bool


def _get_parser():
    parser = getattr(_local, "parser", None)
    if parser is None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        parser = _local.parser = Parser(Language(tspython.language()))
    return parser


def parse_partial(source: str):
    """Parse possibly broken code with tree-sitter and return the tree."""
    return _get_parser().parse(source.encode("utf-8"))


def leaves(tree, source: bytes) -> list[Leaf]:
    """All tokens of the tree in order, each with the context a predictor needs.

    `up` is the list of enclosing node types *as they looked before the next
    token arrived*. A node whose first child is the token's own branch (the
    `attribute` in `a.b`, the `call` in `f(x)`, the `binary_operator` in `a + b`)
    only exists because of tokens that come later, so it is left out. Without
    that, a model trained on whole files would see node types at training time
    that cannot exist yet while someone is still typing.
    """
    out: list[Leaf] = []
    depth = 0
    stack = [(tree.root_node, (), (), "", "")]  # node, up, stmts, parent type, field
    while stack:
        node, up, stmts, parent_type, field = stack.pop()
        t = node.type
        if t == "comment" or node.is_missing or node.start_byte == node.end_byte:
            continue  # comments, and the zero-width nodes tree-sitter invents for missing pieces
        if node.child_count == 0 or t == "string":
            if t == "ERROR" or t == "line_continuation":
                continue
            if t in _OPEN:
                depth += 1
            elif t in _CLOSE and depth:
                depth -= 1
            text = "" if t == "string" else source[node.start_byte : node.end_byte].decode("utf-8", "replace")
            (row, col), (end_row, _) = node.start_point, node.end_point
            out.append(Leaf(t, text, node.is_named, row, col, end_row, node.end_byte, up, stmts, depth, field))
            continue
        holds_statements = t == "block" or t == "module"
        owner = (parent_type if t == "block" else "module") if holds_statements else ""
        chain = (t,) + up
        children = node.children
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            # Keep this parent unless it only exists because of tokens still to come.
            # A block or module is fixed by its header, whatever its first statement is.
            kept = i or holds_statements or not child.is_named
            child_up = chain[:4] if kept else up
            child_stmts = stmts
            if holds_statements and child.type != "comment":
                child_stmts = stmts + ((child.type, owner, child.start_point[1]),)
            # A field only counts if the parent it belongs to is kept (same rule as `up`).
            child_field = (node.field_name_for_child(i) or "") if kept else field
            stack.append((child, child_up, child_stmts, t, child_field))
    return out


def analyze_context(source: str) -> PartialContext:
    """Repair the code typed so far, parse it, and return the tokens that were really typed."""
    raw = source.encode("utf-8")
    tree = parse_partial(source)
    first_pass = leaves(tree, raw)

    # A string or comment that is still open: any word we suggest would land inside it.
    if any(lf.type == "string_start" for lf in first_pass) or _ends_in_comment(tree, source, raw):
        return PartialContext(source, [], True)

    fixed = source + _missing_syntax(first_pass)
    fixed_bytes = fixed.encode("utf-8")
    typed = [lf for lf in leaves(parse_partial(fixed), fixed_bytes) if lf.end_byte <= len(raw)]
    return PartialContext(fixed, typed, False)


def repair_partial(source: str) -> str:
    """The source with whatever is needed to make it parse cleanly appended."""
    return analyze_context(source).text


def _ends_in_comment(tree, source: str, raw: bytes) -> bool:
    if not raw or source.endswith("\n"):
        return False
    node = tree.root_node.descendant_for_byte_range(len(raw) - 1, len(raw))
    return node is not None and node.type == "comment"


def _missing_syntax(toks: list[Leaf]) -> str:
    """Text that finishes whatever the last line left open."""
    if not toks:
        return ""
    stack: list[str] = []
    open_tries: list[int] = []  # columns of `try` statements that have no except/finally yet
    line_first, line_col = "", 0
    colon_at_top = paren_seen = has_def = False
    pending_for = pending_import = from_dots_only = False
    prev_end_row = -1
    for lf in toks:
        if not stack and lf.row > prev_end_row:  # a new logical line starts here
            line_first, line_col = lf.type, lf.col
            colon_at_top = paren_seen = has_def = pending_for = pending_import = False
            if lf.type == "try":
                open_tries.append(lf.col)
            elif lf.type in ("except", "finally"):
                while open_tries and open_tries[-1] >= lf.col:
                    open_tries.pop()
            elif lf.type == "from":
                pending_import, from_dots_only = True, True
        t = lf.type
        if t in _OPEN:
            stack.append(t)
            paren_seen = paren_seen or t == "("
        elif t in _CLOSE:
            if stack:
                stack.pop()
        elif t == ":" and not stack:
            colon_at_top = True
        elif t == "for":
            pending_for = True
        elif t == "in":
            pending_for = False
        elif t == "import":
            pending_import = False
        elif t == "def":
            has_def = True
        if t != "from" and t != ".":
            from_dots_only = False
        prev_end_row = lf.end_row
    last = toks[-1].type

    if last == "async":
        return " def _(): pass"
    out = ""
    if last == ":" and not stack and line_first in _BLOCK_KEYWORDS:
        out = " pass"  # the block header is complete, it only needs a body
    else:
        needs_operand = last in _NEEDS_OPERAND and not (last == line_first and last in {"else", "except"})
        if last == "." and from_dots_only:
            needs_operand = False  # `from .` / `from ..` is a complete relative import path
        if last == "," and stack:
            needs_operand = False  # a trailing comma is fine inside brackets: f(a=1,) is valid
        if last == "*" and len(toks) > 1 and toks[-2].type == "import":
            needs_operand = False  # `from x import *`
        if last == ":" or needs_operand:
            out = " " + PLACEHOLDER
        if pending_for:
            out += f" in {PLACEHOLDER}"
        if pending_import:
            out += f" import {PLACEHOLDER}"
        out += "".join(_OPEN[b] for b in reversed(stack))
        if line_first in _BLOCK_KEYWORDS and not colon_at_top:
            if has_def and not paren_seen:
                out += "()"
            out += ": pass"
    if line_first == "@":
        out += "\n" + " " * line_col + "def _(): pass"  # a decorator must be followed by a def or class
    # Python will not accept a `try` without a handler, so give each open one a bare `except`.
    for col in reversed(open_tries):
        out += "\n" + " " * col + "except: pass"
    return out


def is_valid_python(source: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old code triggers harmless invalid-escape warnings
            ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def splice(context: str, suggestion: str) -> str:
    """Put a suggested word after the code typed so far."""
    if context and suggestion and _is_word_char(context[-1]) and _is_word_char(suggestion[0]):
        return context + " " + suggestion  # two words in a row need a space between them
    return context + suggestion


def completes_to_valid_python(context: str, suggestion: str) -> bool:
    """Splice the suggestion in, let the repair fill in missing syntax, and check it still parses."""
    return is_valid_python(repair_partial(splice(context, suggestion)))
