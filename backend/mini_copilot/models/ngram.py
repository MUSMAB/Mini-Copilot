# N-gram predictor: suggests the next word from the last n-1 words (or characters) using counts from the training files.
# Uses parsing/tokenizer.py and models/base.py; run by evaluation/cross_validation.py and scripts/run_ngram_cv.py.

import re
from bisect import bisect_left
from collections import defaultdict
from heapq import nlargest
from token import EXACT_TOKEN_TYPES

from ..parsing.tokenizer import word_tokens
from .base import Predictor

_START = "\x02"
_SMOOTHINGS = ("add-k", "backoff")
_WORD_AT_END = re.compile(r"[^\W\d]\w*$")
_OPERATORS = frozenset(EXACT_TOKEN_TYPES)
_OPERATOR_STARTS = frozenset(op[:size] for op in _OPERATORS for size in range(1, len(op)))


# Returns True for the line start token ("\n" plus indentation), which is used as context but never suggested.
def _is_layout(token: str) -> bool:
    return token.startswith("\n")


# Returns True for a letter, digit or underscore.
def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# Reads the first word in text (skipping leading spaces) and says whether that word is surely finished.
# A word is a run of letters, digits and underscores, a whole string literal, or one operator such as == or (.
def _first_word(text: str) -> tuple[str, bool]:
    text = text.lstrip()
    if not text:
        return "", False
    first = text[0]
    if _is_word_char(first):
        end = 1
        while end < len(text) and (_is_word_char(text[end]) or (text[end] == "." and first.isdigit())):
            end += 1
        return text[:end], end < len(text)
    if first in "\"'":
        end = 1
        while end < len(text):
            if text[end] == "\\":
                end += 2
            elif text[end] == first:
                return text[: end + 1], True
            else:
                end += 1
        return text, False
    op = first
    for size in (2, 3):
        if text[:size] in _OPERATORS:
            op = text[:size]
    return op, len(text) > len(op) or op not in _OPERATOR_STARTS


# Predicts the next word from counts of what followed the last n-1 words (or characters) in the training files.
# level "char" builds each suggestion one character at a time, so it also finishes half-typed words.
class NGramModel(Predictor):
    # Stores the settings and starts with empty tables.
    # n is the order, level is "word" or "char", smoothing is "add-k" or "backoff", and k is the add-k amount.
    # backoff is the penalty per step down to a shorter context, max_next caps the successors scored per context,
    # beam is how many partial words the char level keeps, and max_steps is how many characters it may add.
    def __init__(
        self,
        n: int,
        level: str = "word",
        smoothing: str = "add-k",
        k: float = 1.0,
        backoff: float = 0.4,
        max_next: int = 50,
        beam: int = 12,
        max_steps: int = 30,
    ):
        if n < 1:
            raise ValueError("n must be at least 1")
        if level not in ("word", "char"):
            raise ValueError("level must be 'word' or 'char'")
        if smoothing not in _SMOOTHINGS:
            raise ValueError(f"smoothing must be one of {_SMOOTHINGS}")
        self.n, self.level, self.smoothing, self.k = n, level, smoothing, k
        self.backoff = backoff
        self.max_next = max_next
        self.beam = beam
        self.max_steps = max_steps
        self.name = f"{level}-{n}gram-{smoothing}"
        self._root = () if level == "word" else ""
        self._reset()

    # Empties everything learned so each training run starts fresh.
    # counts[m][context][token] is how often token followed a context of m tokens.
    def _reset(self) -> None:
        self.counts: list = [defaultdict(dict) for _ in range(self.n)]
        self.known: set[str] = set()
        self._big: list[dict] = [{} for _ in range(self.n)]
        self._vocab_size = 0
        self._words: list[str] = []
        self._word_freq: dict[str, float] = {}

    # Turns text into the tokens this model counts, with a start marker in front so file starts have a context too.
    # Words give a tuple of word tokens and characters give the text itself, so slices work as dictionary keys.
    def _tokens(self, text: str):
        if self.level == "word":
            return (_START,) + tuple(word_tokens(text))
        return _START + text

    # Returns the last n-1 tokens, which is all the context the model looks at.
    def _tail(self, tokens):
        return tokens[max(0, len(tokens) - self.n + 1) :]

    # Counts which token follows every context of 0 to n-1 tokens in every training file.
    def train(self, sources: list[str]) -> None:
        self._reset()
        for source in sources:
            tokens = self._tokens(source)
            for i in range(1, len(tokens)):
                for m in range(min(self.n - 1, i) + 1):
                    successors = self.counts[m][tokens[i - m : i]]
                    successors[tokens[i]] = successors.get(tokens[i], 0) + 1
        self._finish()

    # Freezes the tables and summarises contexts with many successors so scoring them stays fast.
    # Also builds the sorted word list used to complete half-typed words at the word level.
    def _finish(self) -> None:
        self.counts = [dict(table) for table in self.counts]
        for m, table in enumerate(self.counts):
            for context, successors in table.items():
                if len(successors) > self.max_next:
                    best = nlargest(self.max_next, successors, key=successors.get)
                    self._big[m][context] = (sum(successors.values()), best)
        unigrams = self.counts[0].get(self._root, {})
        self._vocab_size = len(unigrams)
        total = sum(unigrams.values()) or 1
        if self.level == "word":
            self._word_freq = {w: c / total for w, c in unigrams.items() if w.isidentifier()}
            self._words = sorted(self._word_freq)
            self.known = set(self._words)

    # Returns the top k next-word suggestions for the code typed so far.
    # The word level mixes in completions of a half-typed last word, the char level does that on its own.
    def predict(self, context: str, k: int = 5) -> list[str]:
        if self.level == "char":
            scores = self._char_scores(context)
        else:
            scores = self._word_scores(context)
            word = _WORD_AT_END.search(context)
            if word:
                scores = self._allow_unfinished_word(context, word, scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[:k]]

    # Returns (total count, tokens worth scoring) for one context that was seen in training.
    # Big contexts were summarised in _finish, small ones are cheap to add up on the spot.
    def _summary(self, m: int, context, successors: dict) -> tuple[int, object]:
        return self._big[m].get(context) or (sum(successors.values()), successors)

    # Scores the likely next tokens after history, using add-k or backoff over shorter and shorter contexts.
    # add-k pretends it saw k extra examples guessed by the shorter context, backoff takes a penalty per step down.
    def _score(self, history) -> dict:
        levels = []
        candidates: set = set()
        for m in range(len(history) + 1):
            context = history[len(history) - m :]
            successors = self.counts[m].get(context)
            if successors is None:
                break
            total, best = self._summary(m, context, successors)
            levels.append((successors, total))
            candidates.update(best)
        if not levels:
            return {}
        if self.smoothing == "add-k":
            successors, total = levels[0]
            scores = {t: (successors.get(t, 0) + self.k) / (total + self.k * self._vocab_size) for t in candidates}
            for successors, total in levels[1:]:
                for t in candidates:
                    scores[t] = (successors.get(t, 0) + self.k * scores[t]) / (total + self.k)
            return scores
        scores = {}
        penalty = self.backoff ** (len(history) + 1 - len(levels))
        for successors, total in reversed(levels):
            for t in candidates:
                if t not in scores and t in successors:
                    scores[t] = penalty * successors[t] / total
            penalty *= self.backoff
        return scores

    # Scores every word that could come next, assuming the context ends on a word boundary.
    # Line start tokens are dropped and the rest is scaled to add up to 1, so mixing with other scores is fair.
    def _word_scores(self, context: str) -> dict[str, float]:
        scores = self._score(self._tail(self._tokens(context)))
        words = {t: s for t, s in scores.items() if not _is_layout(t)}
        total = sum(words.values())
        return {t: s / total for t, s in words.items()} if total else {}

    # Mixes in words that would finish the last typed word (for example "pri" gives "print").
    # The unfinished share is 0.2 if the word is known or already in the file, and 0.6 if it is not.
    def _allow_unfinished_word(self, context: str, word: re.Match, after: dict[str, float]) -> dict[str, float]:
        text = word.group()
        before = self._word_scores(context[: word.start()])
        endings = {w: s for w, s in before.items() if w.startswith(text) and w != text}
        seen_before = re.search(rf"\b{re.escape(text)}\b", context[: word.start()]) is not None
        is_known = text in self.known or seen_before
        if not is_known:
            i = bisect_left(self._words, text)
            while i < len(self._words) and self._words[i].startswith(text):
                candidate = self._words[i]
                endings.setdefault(candidate, self._word_freq[candidate])
                i += 1
        total = sum(endings.values())
        if not total:
            return after
        unfinished = 0.2 if is_known else 0.6
        mixed = {w: s * (1 - unfinished) for w, s in after.items()}
        for w, s in endings.items():
            mixed[w] = mixed.get(w, 0.0) + unfinished * s / total
        return mixed

    # Returns the most likely next characters after history, best first.
    def _next_chars(self, history: str) -> list[tuple[str, float]]:
        ranked = sorted(self._score(history).items(), key=lambda item: (-item[1], item[0]))
        return ranked[: self.beam]

    # Builds whole-word suggestions by adding one character at a time until the word ends.
    # A beam search keeps the likeliest partial words, and a word's score adds up every way of writing it.
    def _char_scores(self, context: str) -> dict[str, float]:
        match = _WORD_AT_END.search(context)
        partial = match.group() if match else ""
        tail = self._tail(self._tokens(context))
        words: dict[str, float] = {}
        beam = [(1.0, "")]
        for _ in range(self.max_steps):
            grown = []
            for prob, text in beam:
                for char, p in self._next_chars(self._tail(tail + text)):
                    new_text = text + char
                    continues_word = bool(partial) and _is_word_char(new_text[0])
                    word, done = _first_word(partial + new_text if continues_word else new_text)
                    if done:
                        words[word] = words.get(word, 0.0) + prob * p
                    else:
                        grown.append((prob * p, new_text))
            if not grown:
                break
            beam = sorted(grown, reverse=True)[: self.beam]
        return words
