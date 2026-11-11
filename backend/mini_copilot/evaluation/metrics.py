"""Accuracy and edit distance for next-word predictions."""


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def top1_accuracy(predictions: list[list[str]], targets: list[str]) -> float:
    hits = sum(1 for p, t in zip(predictions, targets) if p and p[0] == t)
    return hits / len(targets) if targets else 0.0


def topk_accuracy(predictions: list[list[str]], targets: list[str], k: int = 5) -> float:
    hits = sum(1 for p, t in zip(predictions, targets) if t in p[:k])
    return hits / len(targets) if targets else 0.0


def normalized_edit_distance(prediction: str, target: str) -> float:
    """0 means identical, 1 means nothing in common."""
    longest = max(len(prediction), len(target))
    return levenshtein(prediction, target) / longest if longest else 0.0
