# Builds test samples by cutting a source file right before a token and recording the word that follows.
# Used by evaluation/cross_validation.py and tests/test_simulate.py.

import io
import random
import tokenize

_SKIP = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
}
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


# Returns (context, expected_next_word) pairs cut at random token boundaries of one file.
# An f-string counts as one word on every Python version, and the seed makes the picks repeatable.
def make_samples(source: str, max_samples: int = 50, seed: int = 329) -> list[tuple[str, str]]:
    line_starts = [0]
    for line in io.StringIO(source).readlines():
        line_starts.append(line_starts[-1] + len(line))

    # Converts a (line, column) token position into a character offset in the source.
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
        pass
    rng = random.Random(f"{seed}:{len(source)}")
    picked = rng.sample(cuts[1:], min(max_samples, len(cuts) - 1)) if len(cuts) > 1 else []
    return [(source[:start], word) for start, word in sorted(picked)]
