from mini_copilot.parsing.tokenizer import word_tokens
from mini_copilot.parsing.tolerant_parser import is_valid_python


def test_word_tokens():
    assert word_tokens("x = 1\n") == ["x", "=", "1", "\n"]


def test_truncated_code_does_not_crash():
    assert word_tokens("def f(a,")[:3] == ["def", "f", "("]


def test_is_valid_python():
    assert is_valid_python("x = 1")
    assert not is_valid_python("def f(:")
