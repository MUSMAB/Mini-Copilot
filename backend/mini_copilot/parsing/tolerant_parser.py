# Repairs half-typed Python code and parses it with tree-sitter so the model can read the syntax at the cursor.
# Used by models/ast_model.py and evaluation/cross_validation.py, and imported by tests/test_tokenizer.py.

import ast
import threading
import warnings
from typing import NamedTuple

_local = threading.local()

PLACEHOLDER = "_"
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = frozenset(_OPEN.values())
_BLOCK_KEYWORDS = frozenset(
    {"def", "class", "if", "elif", "else", "while", "for", "try", "except", "finally", "with", "async"}
)
_OPERAND_TOKENS = (
    "= += -= *= /= //= %= **= >>= <<= &= |= ^= @= := "
    "+ - * / // % ** @ & | ^ ~ << >> "
    "< > <= >= == != <> -> . , ( [ { "
    "not and or in is if elif while assert raise del "
    "return yield await import from as with global nonlocal "
    "class def else except for"
)
_NEEDS_OPERAND = frozenset(_OPERAND_TOKENS.split())


# One token of a parse tree plus the context around it.
# up = enclosing node types, stmts = enclosing statements, depth = open brackets, field = its role in its parent.
class Leaf(NamedTuple):
    type: str
    text: str
    named: bool
    row: int
    col: int
    end_row: int
    end_byte: int
    up: tuple
    stmts: tuple
    depth: int
    field: str


# The repaired source, the tokens that were really typed, and whether the cursor is inside a string or comment.
class PartialContext(NamedTuple):
    text: str
    leaves: list
    in_string_or_comment: bool


# Returns this thread's tree-sitter parser, creating it on first use.
# A parser is not thread-safe, so each thread keeps its own.
def _get_parser():
    parser = getattr(_local, "parser", None)
    if parser is None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        parser = _local.parser = Parser(Language(tspython.language()))
    return parser


# Parses possibly broken code with tree-sitter and returns the tree.
# Unlike ast.parse it does not fail on errors, it marks the broken parts as ERROR nodes.
def parse_partial(source: str):
    return _get_parser().parse(source.encode("utf-8"))


# Walks the tree and returns every token in order, each with its context (up, stmts, depth, field).
# Parents that only exist because of later tokens (attribute, call, binary_operator) are left out of up.
def leaves(tree, source: bytes) -> list[Leaf]:
    out: list[Leaf] = []
    depth = 0
    stack = [(tree.root_node, (), (), "", "")]
    while stack:
        node, up, stmts, parent_type, field = stack.pop()
        t = node.type
        if t == "comment" or node.is_missing or node.start_byte == node.end_byte:
            continue
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
            kept = i or holds_statements or not child.is_named
            child_up = chain[:4] if kept else up
            child_stmts = stmts
            if holds_statements and child.type != "comment":
                child_stmts = stmts + ((child.type, owner, child.start_point[1]),)
            child_field = (node.field_name_for_child(i) or "") if kept else field
            stack.append((child, child_up, child_stmts, t, child_field))
    return out


# Repairs the typed code, parses it, and returns the repaired text with only the tokens that were really typed.
# If the cursor is inside an open string or comment it returns no tokens and sets in_string_or_comment.
def analyze_context(source: str) -> PartialContext:
    raw = source.encode("utf-8")
    tree = parse_partial(source)
    first_pass = leaves(tree, raw)

    if any(lf.type == "string_start" for lf in first_pass) or _ends_in_comment(tree, source, raw):
        return PartialContext(source, [], True)

    fixed = source + _missing_syntax(first_pass)
    fixed_bytes = fixed.encode("utf-8")
    typed = [lf for lf in leaves(parse_partial(fixed), fixed_bytes) if lf.end_byte <= len(raw)]
    return PartialContext(fixed, typed, False)


# Returns the source with whatever text is needed to make it parse cleanly appended.
def repair_partial(source: str) -> str:
    return analyze_context(source).text


# Checks whether the source ends in the middle of a comment.
# A trailing newline ends the comment, so that case returns False.
def _ends_in_comment(tree, source: str, raw: bytes) -> bool:
    if not raw or source.endswith("\n"):
        return False
    node = tree.root_node.descendant_for_byte_range(len(raw) - 1, len(raw))
    return node is not None and node.type == "comment"


# Builds the text that closes whatever the last line left open (brackets, operands, block bodies, try handlers).
# The placeholder "_" stands in for a missing operand and "pass" for a missing block body.
def _missing_syntax(toks: list[Leaf]) -> str:
    if not toks:
        return ""
    stack: list[str] = []
    open_tries: list[int] = []
    line_first, line_col = "", 0
    colon_at_top = paren_seen = has_def = False
    pending_for = pending_import = from_dots_only = False
    prev_end_row = -1
    for lf in toks:
        if not stack and lf.row > prev_end_row:
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
        out = " pass"
    else:
        needs_operand = last in _NEEDS_OPERAND and not (last == line_first and last in {"else", "except"})
        if last == "." and from_dots_only:
            needs_operand = False
        if last == "," and stack:
            needs_operand = False
        if last == "*" and len(toks) > 1 and toks[-2].type == "import":
            needs_operand = False
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
        out += "\n" + " " * line_col + "def _(): pass"
    for col in reversed(open_tries):
        out += "\n" + " " * col + "except: pass"
    return out


# Returns True if Python's own ast.parse accepts the source.
# Warnings are silenced because old code triggers harmless invalid-escape warnings.
def is_valid_python(source: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


# Returns True for a letter, digit or underscore.
def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# Appends a suggested word to the typed code, adding a space when two words would run together.
def splice(context: str, suggestion: str) -> str:
    if context and suggestion and _is_word_char(context[-1]) and _is_word_char(suggestion[0]):
        return context + " " + suggestion
    return context + suggestion


# Checks that the typed code plus a suggestion still parses once the missing syntax is filled in.
# cross_validation.py uses this for the valid score.
def completes_to_valid_python(context: str, suggestion: str) -> bool:
    return is_valid_python(repair_partial(splice(context, suggestion)))
