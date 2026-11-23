"""AST-based predictor: suggests names and keywords that fit the syntax at the cursor.

How it works
------------
Training parses every file with tree-sitter and, for each token, records which
token came next given the *syntactic slot* the previous token sits in:

  * the previous token's node type (identifier, "(", "def", ...), the role its
    branch plays in the parent node (`left` or `right` of a for loop, the
    `name` of a def, ...) and the types of the nodes around it (argument_list,
    parameters, if_statement, ...);
  * at the start of a line, the kind of statement just above (an `if` is
    followed by `else`/`elif`, a `try` by `except`) and which block we are in.

These contexts go from specific to general and are blended with Witten-Bell
backoff, so a slot never seen in training falls back to a looser one, and in
the end to plain token frequencies.

Names are the one thing the grammar cannot tell us: all it knows is "a name
goes here". For that slot the model ranks the names already used in the current
file (recent and frequent first, restricted to attribute names after a ".") and
mixes in the names training saw most often in that kind of slot (self, len, ...).

At prediction time the code typed so far is repaired and parsed by
parsing/tolerant_parser.py, and the slot is read off the last real token.
"""

import builtins
import re
from bisect import bisect_left
from collections import Counter, defaultdict

from ..parsing.tolerant_parser import Leaf, analyze_context, leaves, parse_partial
from .base import Predictor

NAME, NUMBER, STRING = "<name>", "<number>", "<string>"  # classes we never suggest as literal text
_COMMON_NAMES = frozenset(dir(builtins)) | {"self", "cls"}  # names whose exact text predicts what follows
_NO_FILE_NAMES = frozenset({"def", "class", "import", "from", "as"})  # a *new* name is being introduced here
_WORD_AT_END = re.compile(r"[^\W\d]\w*$")
_ANYWHERE = ("ANY",)  # final fallback: how often each token occurs at all


def _label(tok: Leaf) -> str:
    """What we count as the 'next token': keywords and punctuation as text, names and literals as classes."""
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


def _keys(prev: Leaf | None, new_line: bool, col: int) -> list[tuple]:
    """Context keys for the token after `prev`, most specific first.

    The last two keys are always the 'medium' and 'general' ones; the name
    statistics are stored under the medium one. (`_ANYWHERE` is added after these.)
    """
    if prev is None:
        return [("START",)]
    if new_line and prev.depth == 0:
        # The cursor starts a line. If it lines up with an earlier statement, that
        # statement (and the block it lives in) decides what can come next.
        for stmt_type, owner, stmt_col in reversed(prev.stmts):
            if stmt_col == col:
                inner = prev.stmts[-1][1]  # e.g. else_clause: the `if` already has its else
                return [("N", stmt_type, owner, inner), ("N", stmt_type, owner), ("N", stmt_type), ("N",)]
        tag = "B"  # first line of a new block, or a line that lines up with nothing
    else:
        tag = "L"
    p1 = prev.up[0] if prev.up else ""
    p2 = prev.up[1] if len(prev.up) > 1 else ""
    keys = []
    if prev.text in _COMMON_NAMES:
        keys.append((tag, prev.text, prev.field, p1, p2))
    keys += [
        (tag, prev.type, prev.field, p1, p2),  # e.g. the `left` identifier of a for loop vs. its `right` one
        (tag, prev.type, prev.field, p1),
        (tag, prev.type, p1),
        (tag, prev.type),
    ]
    return keys


class ASTModel(Predictor):
    name = "ast"

    def __init__(
        self,
        name_weight: float = 0.6,
        half_life: float = 80.0,
        slot_bonus: float = 5.0,
        floor: float = 0.1,
        max_names: int = 100,
        min_name_count: int = 2,
    ):
        self.name_weight = name_weight  # share of a name slot given to names from the current file
        self.half_life = half_life  # in tokens: how fast a name in the file stops counting as 'recent'
        self.slot_bonus = slot_bonus  # extra weight for a name that followed the same kind of token before
        self.floor = floor  # weight a name keeps no matter how long ago it appeared
        self.max_names = max_names
        self.min_name_count = min_name_count
        self._reset()

    def _reset(self) -> None:
        self.table: dict[tuple, Counter] = defaultdict(Counter)  # context key -> next-token label counts
        self.names: dict[tuple, Counter] = defaultdict(Counter)  # medium key -> next-name text counts
        self.vocab: Counter = Counter()
        self.global_names: Counter = Counter()
        self.known: set[str] = set()
        self._words: list[str] = []  # every word we have seen, sorted, to look up completions
        self._word_freq: dict[str, float] = {}
        self._totals: dict[tuple, int] = {}

    # ------------------------------------------------------------------ training

    def train(self, sources: list[str]) -> None:
        """Learn which tokens follow which AST node types."""
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

    def _finish(self) -> None:
        """Drop one-off names (almost all of them) and freeze the tables."""
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

    # ---------------------------------------------------------------- prediction

    def predict(self, context: str, k: int = 5) -> list[str]:
        """Top k next-word suggestions for the code typed so far."""
        scores = self._scores(context)
        word = _WORD_AT_END.search(context)
        if word:
            scores = self._allow_unfinished_word(context, word, scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[:k]]

    def _allow_unfinished_word(self, context: str, word: re.Match, after: dict[str, float]) -> dict[str, float]:
        """The last word might be a complete token or half of one ("pri" -> "print").

        `after` assumes it is complete. Here we also score the words that
        would finish it, using the slot the word started in, and mix the two.
        """
        text = word.group()
        before = self._scores(context[: word.start()])
        endings = {w: s for w, s in before.items() if w.startswith(text) and w != text}
        seen_before = re.search(rf"\b{re.escape(text)}\b", context[: word.start()]) is not None
        is_known = text in self.known or seen_before
        if not is_known:
            # Not a word we know, so it is probably unfinished: offer any known word that
            # starts with it, even one this slot would not normally pick.
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

    def _scores(self, context: str) -> dict[str, float]:
        """Probability of every candidate next token, assuming the context ends on a token boundary."""
        parsed = analyze_context(context)
        if parsed.in_string_or_comment:
            return {}
        typed = parsed.leaves
        prev = typed[-1] if typed else None
        new_line = prev is not None and context.count("\n") > prev.end_row
        col = len(context.rsplit("\n", 1)[-1].encode("utf-8"))
        keys = _keys(prev, new_line, col)

        dist = self._blend([*keys, _ANYWHERE])
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

    def _blend(self, keys: list[tuple]) -> dict[str, float]:
        """Witten-Bell interpolation from the most general context to the most specific."""
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

    def _names_in_file(self, typed: list[Leaf], prev: Leaf | None) -> dict[str, float]:
        """Names already used in this file, weighted by how recently they appeared."""
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
                    continue  # after a dot only attribute names fit (methods count: they are def names)
            elif before == ".":
                continue  # a bare name slot: skip names that only ever appeared as attributes
            weight = self.floor + decay ** (n - i)
            if prev is not None and before == prev.type:
                weight *= 1.0 + self.slot_bonus
            pool[tok.text] = pool.get(tok.text, 0.0) + weight
        return pool
