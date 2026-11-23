"""Simulate typing: cut a test file at a word boundary and ask the model for the next word.

Steps for each sample: take the full file, truncate it right before a token,
predict, then (for AST/hybrid) splice the suggestion in, let the tolerant
parser fill in missing syntax, and check the result still parses.
"""

import io
import random
import tokenize

# Layout tokens: nobody types these as "the next word".
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
         tokenize.ENDMARKER, tokenize.ENCODING}  # fmt: skip
# Python 3.12+ splits an f-string into several tokens; older versions give one STRING token.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def make_samples(source: str, max_samples: int = 50, seed: int = 329) -> list[tuple[str, str]]:
    """Return (context, expected_next_word) pairs from one file.

    An f-string counts as one word on every Python version.
    """
    line_starts = [0]
    for line in io.StringIO(source).readlines():  # split lines exactly like tokenize does
        line_starts.append(line_starts[-1] + len(line))

    def offset(pos: tuple[int, int]) -> int:
        return line_starts[pos[0] - 1] + pos[1]

    cuts: list[tuple[int, str]] = []
    fstring_start = None
    fstring_depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == _FSTRING_START:
                if fstring_depth == 0:
                    fstring_start = offset(tok.start)
                fstring_depth += 1
            elif tok.type == _FSTRING_END:
                fstring_depth -= 1
                if fstring_depth == 0:
                    cuts.append((fstring_start, source[fstring_start : offset(tok.end)]))
            elif fstring_depth == 0 and tok.type not in _SKIP and tok.string:
                cuts.append((offset(tok.start), tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # keep the tokens we got before the error
    rng = random.Random(f"{seed}:{len(source)}")
    picked = rng.sample(cuts[1:], min(max_samples, len(cuts) - 1)) if len(cuts) > 1 else []
    return [(source[:start], word) for start, word in sorted(picked)]
