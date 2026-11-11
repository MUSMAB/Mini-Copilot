from mini_copilot.evaluation.metrics import (
    levenshtein,
    normalized_edit_distance,
    top1_accuracy,
    topk_accuracy,
)


def test_levenshtein():
    assert levenshtein("print", "prnt") == 1
    assert levenshtein("", "abc") == 3


def test_accuracy():
    preds = [["for", "if"], ["x", "range"]]
    assert top1_accuracy(preds, ["for", "range"]) == 0.5
    assert topk_accuracy(preds, ["for", "range"], k=5) == 1.0


def test_normalized_edit_distance():
    assert normalized_edit_distance("range", "range") == 0.0
    assert normalized_edit_distance("abc", "xyz") == 1.0
