# Tests the tokenizer: word tokens, line start indentation, f-strings, half-typed code and characters.
# Covers parsing/tokenizer.py and the is_valid_python check in parsing/tolerant_parser.py.

from mini_copilot.parsing.tokenizer import char_tokens, word_tokens
from mini_copilot.parsing.tolerant_parser import is_valid_python


# Checks that a simple line becomes its words and a final newline token.
def test_word_tokens():
    assert word_tokens("x = 1\n") == ["x", "=", "1", "\n"]


# Checks that code cut off in the middle of a line keeps the tokens read so far.
def test_truncated_code_does_not_crash():
    assert word_tokens("def f(a,")[:3] == ["def", "f", "("]


# Checks that every line start is one token holding its newline and indentation.
def test_line_starts_hold_the_indentation():
    assert word_tokens("if a:\n    b = 1\n") == ["if", "a", ":", "\n    ", "b", "=", "1", "\n"]


# Checks that comments and blank lines add no tokens.
def test_comments_and_blank_lines_are_dropped():
    assert word_tokens("x = 1  # note\n\n\n# alone\ny = 2\n") == ["x", "=", "1", "\n", "y", "=", "2", "\n"]


# Checks that indentation typed on a new line at the end of the code becomes the last token.
def test_typed_indentation_at_the_end_is_kept():
    assert word_tokens("def f():\n    ")[-1] == "\n    "
    assert word_tokens("x = 1")[-1] == "1"


# Checks that a whole f-string is one token on every Python version.
def test_an_fstring_is_one_token():
    assert word_tokens('x = f"{a}!"\n') == ["x", "=", 'f"{a}!"', "\n"]


# Checks that junk and empty input do not raise.
def test_junk_input_does_not_crash():
    for junk in ["", "   ", "\x00", ")))((({{", "def (:\n\t\t", "x = '"]:
        assert isinstance(word_tokens(junk), list)


# Checks that characters come back one by one, spaces and newlines included.
def test_char_tokens():
    assert char_tokens("a b\n") == ["a", " ", "b", "\n"]


# Checks that Python's own parser accepts complete code and rejects broken code.
def test_is_valid_python():
    assert is_valid_python("x = 1")
    assert not is_valid_python("def f(:")
