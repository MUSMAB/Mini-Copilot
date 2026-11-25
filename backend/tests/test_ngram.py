# Tests the n-gram model: word and char levels, both smoothings, indentation, half-typed words, retraining and junk input.
# Covers models/ngram.py and models/base.py.

import pytest
from mini_copilot.models.base import Predictor
from mini_copilot.models.ngram import NGramModel

PROGRAM_A = '''\
import os


class Box:
    def __init__(self, size):
        self.size = size
        self.label = "box"

    def grow(self, amount):
        if amount > 0:
            self.size += amount
        else:
            self.size -= amount
        return self.size


def main():
    try:
        box = Box(3)
        box.grow(2)
    except ValueError:
        print("bad")
    for i in range(3):
        print(i)
'''

PROGRAM_B = '''\
from os import path


def total(values):
    count = 0
    for value in values:
        if value > 0:
            count += value
        else:
            count -= value
    return count


def hello(name):
    print("hello", name)


def show(items):
    for item in items:
        print(item)
    try:
        print(total(items))
    except TypeError:
        print("not numbers")
'''

# A tiny file where the same word, 1, is followed by a different word depending only on the indentation.
INDENT_PROGRAM = "if a:\n    b = 1\n    c = 1\nd = 1\n"


# Trains one word model on the two sample programs and shares it with the tests in this file.
# scope="module" makes pytest build it only once.
@pytest.fixture(scope="module")
def word_model():
    m = NGramModel(3, "word")
    m.train([PROGRAM_A, PROGRAM_B] * 3)
    return m


# Trains one char model on the two sample programs and shares it with the tests in this file.
@pytest.fixture(scope="module")
def char_model():
    m = NGramModel(5, "char")
    m.train([PROGRAM_A, PROGRAM_B] * 3)
    return m


# Checks that NGramModel is a Predictor and that its name shows the level, the order and the smoothing.
def test_is_a_predictor():
    assert isinstance(NGramModel(3), Predictor)
    assert NGramModel(3, "word").name == "word-3gram-add-k"
    assert NGramModel(5, "char", "backoff").name == "char-5gram-backoff"


# Checks that bad settings are rejected right away.
def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        NGramModel(0)
    with pytest.raises(ValueError):
        NGramModel(3, "sentence")
    with pytest.raises(ValueError):
        NGramModel(3, "word", "magic")


# Checks that predict returns at most k strings at both levels.
def test_returns_at_most_k_strings(word_model, char_model):
    for model in (word_model, char_model):
        for k in (1, 3, 5):
            out = model.predict("def f(x):\n    ", k)
            assert len(out) <= k and all(isinstance(w, str) for w in out)


# Checks that "(" is the top suggestion after a function name like print, for both smoothings.
@pytest.mark.parametrize("smoothing", ["add-k", "backoff"])
def test_call_paren_after_function_name(smoothing):
    m = NGramModel(3, "word", smoothing)
    m.train([PROGRAM_A, PROGRAM_B] * 3)
    assert m.predict("def f(x):\n    print", 1) == ["("]


# Checks that "import" is the top suggestion after "from" and a module name.
def test_from_module_is_followed_by_import(word_model):
    assert word_model.predict("from os ", 1) == ["import"]


# Checks that an empty file gets a file start suggestion like import or from.
def test_empty_file_suggests_how_files_start(word_model):
    assert {"import", "from"} & set(word_model.predict("", 3))


# Checks that the indentation on the new line decides what comes next.
def test_indentation_is_part_of_the_context():
    m = NGramModel(3, "word")
    m.train([INDENT_PROGRAM] * 3)
    assert m.predict("if a:\n    b = 1\n    ", 1) == ["c"]
    assert m.predict("if a:\n    b = 1\n    c = 1\n", 1) == ["d"]


# Checks that line start tokens are used as context but never returned as suggestions.
def test_line_starts_are_never_suggested(word_model):
    for context in ["def f(x):\n    ", "x = 1", "def f(x):\n    return 1\n\n"]:
        assert not any(w.startswith("\n") for w in word_model.predict(context, 10))


# Checks that a half-typed word like "pri" suggests "print" at both levels.
def test_finishes_a_half_typed_word(word_model, char_model):
    assert "print" in word_model.predict("def f(x):\n    pri", 3)
    assert "print" in char_model.predict("def f(x):\n    pri", 3)


# Checks that a finished word like "print" is not replaced and "(" is suggested next at both levels.
def test_complete_word_is_not_replaced(word_model, char_model):
    assert word_model.predict("def f(x):\n    print", 1) == ["("]
    assert char_model.predict("def f(x):\n    print", 1) == ["("]


# Checks that the char level returns whole words and not pieces of words.
def test_char_level_returns_whole_words(char_model):
    assert char_model.predict("from os ", 1) == ["import"]
    assert "size" in char_model.predict("class P:\n    def get(self):\n        return self.", 3)


# Checks that the char level keeps operators like == and whole strings together.
def test_char_level_keeps_operators_and_strings_whole():
    m = NGramModel(5, "char")
    m.train(['if x == 1:\n    name = "box"\n'] * 3)
    assert m.predict("if x ", 1) == ["=="]
    assert m.predict("if x == 1:\n    name = ", 1) == ['"box"']


# Checks that training twice gives the same predictions as training once, at both levels.
@pytest.mark.parametrize("level", ["word", "char"])
def test_retraining_replaces_old_statistics(level):
    m = NGramModel(3, level)
    m.train([PROGRAM_A])
    first = m.predict("def f(x):\n    if x:\n        y = 1\n    ", 5)
    m.train([PROGRAM_A])
    assert m.predict("def f(x):\n    if x:\n        y = 1\n    ", 5) == first


# Checks that an untrained model returns an empty list and junk input does not raise, at both levels.
@pytest.mark.parametrize("level", ["word", "char"])
def test_untrained_model_and_junk_input_do_not_crash(level):
    assert NGramModel(3, level).predict("x = ", 5) == []
    m = NGramModel(3, level)
    m.train([PROGRAM_A])
    for junk in ["", "   ", ")))((({{", "\x00", "def (:\n\t\t", "é = 1\nprint(é"]:
        m.predict(junk, 5)


# Checks that a model of order 1 only looks at word counts and still works.
def test_order_one_uses_no_context():
    m = NGramModel(1, "word")
    m.train([PROGRAM_A, PROGRAM_B])
    assert m.predict("def f(x):\n    ", 3) == m.predict("class A:\n    ", 3)
