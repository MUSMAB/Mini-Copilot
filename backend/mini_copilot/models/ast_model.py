# AST predictor: suggests the next word from the syntax slot the cursor sits in, using tree-sitter parse trees.
# Uses parsing/tolerant_parser.py and models/base.py; run by evaluation/cross_validation.py and scripts/run_ast_cv.py.

import builtins
import re
from bisect import bisect_left
from collections import Counter, defaultdict

from ..parsing.tolerant_parser import Leaf, analyze_context, leaves, parse_partial
from .base import Predictor

NAME, NUMBER, STRING = "<name>", "<number>", "<string>"
_COMMON_NAMES = frozenset(dir(builtins)) | {"self", "cls"}
_NO_FILE_NAMES = frozenset({"def", "class", "import", "from", "as"})
_WORD_AT_END = re.compile(r"[^\W\d]\w*$")
_ANYWHERE = ("ANY",)


# Turns a token into the label we count as the next token: keywords and punctuation as text, names and literals as classes.
# For example any identifier becomes <name> and any integer becomes <number>.
def _label(tok: Leaf) -> str:
    if not tok.named:
        return tok.type
    if tok.type in ("identifier", "keyword_identifier"):
        return NAME
    if tok.type in ("true", "false", "none"):
        return tok.text
    if tok.type in ("integer", "float"):
        return NUMBER
    if tok.type == "string":
        return STRING
    return f"<{tok.type}>"


# Builds the context keys for the token after prev, from most specific to most general.
# L = inside a line, B = first line of a new block, N = line start that lines up with an earlier statement.
def _keys(prev: Leaf | None, new_line: bool, col: int) -> list[tuple]:
    if prev is None:
        return [("START",)]
    if new_line and prev.depth == 0:
        for stmt_type, owner, stmt_col in reversed(prev.stmts):
            if stmt_col == col:
                inner = prev.stmts[-1][1]
                return [("N", stmt_type, owner, inner), ("N", stmt_type, owner), ("N", stmt_type), ("N",)]
        tag = "B"
    else:
        tag = "L"
    p1 = prev.up[0] if prev.up else ""
    p2 = prev.up[1] if len(prev.up) > 1 else ""
    keys = []
    if prev.text in _COMMON_NAMES:
        keys.append((tag, prev.text, prev.field, p1, p2))
    keys += [
        (tag, prev.type, prev.field, p1, p2),
        (tag, prev.type, prev.field, p1),
        (tag, prev.type, p1),
        (tag, prev.type),
    ]
    return keys


# Predicts the next word from counts of AST contexts plus the names already used in the file.
class ASTModel(Predictor):
    name = "ast"

    # Stores the settings used to rank names and starts with empty tables.
    # half_life is in tokens, slot_bonus boosts names that followed the same kind of token, floor is the minimum weight.
    def __init__(
        self,
        name_weight: float = 0.6,
        half_life: float = 80.0,
        slot_bonus: float = 5.0,
        floor: float = 0.1,
        max_names: int = 100,
        min_name_count: int = 2,
    ):
        self.name_weight = name_weight
        self.half_life = half_life
        self.slot_bonus = slot_bonus
        self.floor = floor
        self.max_names = max_names
        self.min_name_count = min_name_count
        self._reset()

    # Empties everything learned so each training run starts fresh.
    def _reset(self) -> None:
        self.table: dict[tuple, Counter] = defaultdict(Counter)
        self.names: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: Counter = Counter()
        self.global_names: Counter = Counter()
        self.known: set[str] = set()
        self._words: list[str] = []
        self._word_freq: dict[str, float] = {}
        self._totals: dict[tuple, int] = {}

    # Parses every training file and counts which token follows each context key.
    # Name text is counted under the medium key, and one-off names are dropped later in _finish.
    def train(self, sources: list[str]) -> None:
        self._reset()
        for source in sources:
            toks = leaves(parse_partial(source), source.encode("utf-8"))
            if not toks:
                continue
            prev = toks[0]
            self.table[("START",)][_label(prev)] += 1
            for tok in toks[1:]:
                keys = _keys(prev, tok.row > prev.end_row, tok.col)
                label = _label(tok)
                for key in keys:
                    self.table[key][label] += 1
                self.table[_ANYWHERE][label] += 1
                if label == NAME:
                    self.names[keys[-2]][tok.text] += 1
                    self.vocab[tok.text] += 1
                prev = tok
        self._finish()

    # Drops names seen fewer than min_name_count times and freezes the tables for prediction.
    # Also builds the sorted word list used to complete half-typed words.
    def _finish(self) -> None:
        for key, counts in list(self.names.items()):
            kept = [(n, c) for n, c in counts.most_common(self.max_names) if c >= self.min_name_count]
            if kept:
                self.names[key] = Counter(dict(kept))
            else:
                del self.names[key]
        self.vocab = Counter({n: c for n, c in self.vocab.items() if c >= self.min_name_count})
        self.global_names = Counter(dict(self.vocab.most_common(self.max_names)))
        self.table = dict(self.table)
        self.names = dict(self.names)
        self._totals = {key: sum(counts.values()) for key, counts in self.table.items()}
        everywhere = self.table.get(_ANYWHERE, {})
        total = sum(everywhere.values()) or 1
        self._word_freq = {name: count / total for name, count in self.vocab.items()}
        self._word_freq.update({lbl: c / total for lbl, c in everywhere.items() if lbl.isidentifier()})
        self._words = sorted(self._word_freq)
        self.known = set(self._words)

    # Returns the top k next-word suggestions for the code typed so far.
    # If the context ends in a word it might be unfinished, so completions of it are mixed in.
    def predict(self, context: str, k: int = 5) -> list[str]:
        scores = self._scores(context)
        word = _WORD_AT_END.search(context)
        if word:
            scores = self._allow_unfinished_word(context, word, scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[:k]]

    # Mixes in words that would finish the last typed word (for example "pri" gives "print").
    # The unfinished share is 0.2 if the word is known or already in the file, and 0.6 if it is not.
    def _allow_unfinished_word(self, context: str, word: re.Match, after: dict[str, float]) -> dict[str, float]:
        text = word.group()
        before = self._scores(context[: word.start()])
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

    # Scores every candidate next token, assuming the context ends on a token boundary.
    # The probability left for names is split between names in this file and names seen in training.
    def _scores(self, context: str) -> dict[str, float]:
        parsed = analyze_context(context)
        if parsed.in_string_or_comment:
            return {}
        typed = parsed.leaves
        prev = typed[-1] if typed else None
        new_line = prev is not None and context.count("\n") > prev.end_row
        col = len(context.rsplit("\n", 1)[-1].encode("utf-8"))
        keys = _keys(prev, new_line, col)

        dist = self._blend(keys + [_ANYWHERE])
        name_mass = dist.pop(NAME, 0.0)
        scores = {label: p for label, p in dist.items() if not label.startswith("<")}
        if name_mass:
            in_file = self._names_in_file(typed, prev)
            in_training = self.names.get(keys[-2 if len(keys) > 1 else -1]) or self.global_names
            share = self.name_weight if in_file else 0.0
            for names, weight in ((in_file, share), (in_training, 1 - share)):
                total = sum(names.values())
                for name, count in names.items():
                    scores[name] = scores.get(name, 0.0) + name_mass * weight * count / total
        return scores

    # Blends counts from the most general key up to the most specific key into one probability table.
    # Witten-Bell weighting: a key that has seen many different next tokens leans more on the looser keys.
    def _blend(self, keys: list[tuple]) -> dict[str, float]:
        dist: dict[str, float] = {}
        for key in reversed(keys):
            counts = self.table.get(key)
            if not counts:
                continue
            n = self._totals[key]
            lam = n / (n + len(counts))
            blended = {label: (1 - lam) * p for label, p in dist.items()}
            for label, count in counts.items():
                blended[label] = blended.get(label, 0.0) + lam * count / n
            dist = blended
        return dist

    # Ranks the names already used in this file, recent and frequent names first.
    # A weight halves every half_life tokens; after a dot only attribute names count, elsewhere they are skipped.
    def _names_in_file(self, typed: list[Leaf], prev: Leaf | None) -> dict[str, float]:
        if prev is not None and prev.type in _NO_FILE_NAMES:
            return {}
        after_dot = prev is not None and prev.type == "."
        decay = 0.5 ** (1.0 / self.half_life)
        n = len(typed)
        pool: dict[str, float] = {}
        for i, tok in enumerate(typed):
            if tok.type != "identifier":
                continue
            before = typed[i - 1].type if i else ""
            if after_dot:
                if before not in (".", "def"):
                    continue
            elif before == ".":
                continue
            weight = self.floor + decay ** (n - i)
            if prev is not None and before == prev.type:
                weight *= 1.0 + self.slot_bonus
            pool[tok.text] = pool.get(tok.text, 0.0) + weight
        return pool
