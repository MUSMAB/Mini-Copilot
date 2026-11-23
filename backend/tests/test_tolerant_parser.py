# Tests the tolerant parser: repairing half-written code, token context, splicing and validity checks.
# Covers parsing/tolerant_parser.py.

import pytest
from mini_copilot.parsing.tolerant_parser import (
    analyze_context,
    completes_to_valid_python,
    is_valid_python,
    leaves,
    parse_partial,
    repair_partial,
    splice,
)

HALF_WRITTEN = [
    "x = foo(",
    "def f(a,",
    "def f",
    "if x",
    "if x:",
    "for i in",
    "for y ",
    "class A(B",
    "x = [1, 2",
    "from os import",
    "from rich ",
    "from .",
    "try:\n    x = 1\n",
    "d = {'a': 1, 'b':",
    "with open(f) as",
    "@property\n",
    "while True:\n    x = foo(a)[",
    "x = [i for i in y if",
]


# Checks that each snippet really is incomplete and that it parses after repair.
# pytest.mark.parametrize runs it once for every snippet in HALF_WRITTEN.
@pytest.mark.parametrize("code", HALF_WRITTEN)
def test_repair_makes_half_written_code_parse(code):
    assert not is_valid_python(code) or code.endswith("\n")
    assert is_valid_python(repair_partial(code))


# Checks that code that is already complete comes back unchanged.
def test_repair_leaves_complete_code_alone():
    assert repair_partial("x = 1\n") == "x = 1\n"


# Checks that the filler added by the repair is not returned as typed tokens.
def test_only_typed_tokens_are_returned():
    typed = analyze_context("x = foo(").leaves
    assert [t.type for t in typed] == ["identifier", "=", "identifier", "("]


# Checks that an unfinished string or comment sets in_string_or_comment and a finished one does not.
def test_open_string_or_comment_is_flagged():
    assert analyze_context('x = "abc').in_string_or_comment
    assert analyze_context("x = 1  # todo").in_string_or_comment
    assert not analyze_context("x = 1  # todo\n").in_string_or_comment
    assert not analyze_context('x = "abc"').in_string_or_comment


# Checks that a token gets the same up context in the full file and in a prefix of it.
# attribute and call only exist once later tokens arrive, so they must not appear in up.
def test_parents_that_depend_on_later_tokens_are_ignored():
    source = "x = a.b(c)\n"
    full = next(t for t in leaves(parse_partial(source), source.encode()) if t.text == "a")
    typed = analyze_context("x = a").leaves[-1]
    assert full.up == typed.up
    assert "attribute" not in full.up and "call" not in full.up


# Checks that stmts lists the enclosing statements from outermost to innermost.
# Each entry is (statement type, owner of its block, column).
def test_statement_context_tracks_blocks():
    source = "def f(x):\n    if x:\n        return 1\n    "
    last = analyze_context(source).leaves[-1]
    kinds = [kind for kind, _, _ in last.stmts]
    assert kinds == ["function_definition", "if_statement", "return_statement"]


# Checks that splice adds a space only when two words would run together.
def test_splice():
    assert splice("def", "foo") == "def foo"
    assert splice("foo(", "x") == "foo(x"
    assert splice("x ", "=") == "x ="


# Checks that sensible suggestions still parse after repair and a wrong one does not.
def test_completes_to_valid_python():
    assert completes_to_valid_python("for i ", "in")
    assert completes_to_valid_python("x = foo(", "a")
    assert not completes_to_valid_python("x = ", ")")


# Checks that empty and junk input does not raise.
def test_empty_and_garbage_input_do_not_crash():
    for source in ["", "   \n", ")))((({{", "\x00", "def (:\n\t\t"]:
        analyze_context(source)
