import pytest

from mini_copilot.models.ast_model import ASTModel
from mini_copilot.models.base import Predictor

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


@pytest.fixture(scope="module")
def model():
    m = ASTModel()
    m.train([PROGRAM_A, PROGRAM_B] * 3)
    return m


def test_is_a_predictor():
    assert isinstance(ASTModel(), Predictor)
    assert ASTModel().name == "ast"


def test_returns_at_most_k_strings(model):
    for k in (1, 3, 5):
        out = model.predict("def f(x):\n    ", k)
        assert len(out) <= k and all(isinstance(w, str) for w in out)


def test_else_follows_an_if_body(model):
    source = "def f(x):\n    if x:\n        y = 1\n    "
    assert {"else", "elif"} & set(model.predict(source, 3))


def test_except_follows_a_try_body(model):
    assert "except" in model.predict("def f(x):\n    try:\n        y = 1\n    ", 3)


def test_for_target_is_followed_by_in(model):
    assert model.predict("for y ", 1) == ["in"]


def test_from_module_is_followed_by_import(model):
    assert model.predict("from os ", 1) == ["import"]


def test_call_paren_after_function_name(model):
    assert model.predict("def f(x):\n    print", 1) == ["("]


def test_suggests_names_from_the_current_file(model):
    source = "def scale(readings, factor):\n    return readings * "
    assert "factor" in model.predict(source, 5)


def test_attribute_after_dot_comes_from_the_file(model):
    source = "class P:\n    def __init__(self):\n        self.weight = 1\n\n    def get(self):\n        return self."
    assert "weight" in model.predict(source, 3)


def test_finishes_a_half_typed_word(model):
    assert "print" in model.predict("def f(x):\n    pri", 3)


def test_complete_word_is_not_replaced(model):
    assert model.predict("def f(x):\n    print", 1) == ["("]  # `print` is done, what comes next is "("


def test_nothing_is_suggested_inside_a_string_or_comment(model):
    assert model.predict('x = "hello wor', 5) == []
    assert model.predict("x = 1  # todo fi", 5) == []


def test_retraining_replaces_old_statistics():
    m = ASTModel()
    m.train([PROGRAM_A])
    first = m.predict("def f(x):\n    if x:\n        y = 1\n    ", 5)
    m.train([PROGRAM_A])
    assert m.predict("def f(x):\n    if x:\n        y = 1\n    ", 5) == first


def test_untrained_model_and_junk_input_do_not_crash(model):
    assert ASTModel().predict("x = ", 5) == []
    for junk in ["", "   ", ")))((({{", "\x00", "def (:\n\t\t", "é = 1\nprint(é"]:
        model.predict(junk, 5)
