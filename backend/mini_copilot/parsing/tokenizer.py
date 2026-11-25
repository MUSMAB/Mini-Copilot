# Splits Python source into word tokens or single characters so the n-gram models can count them.
# Used by models/ngram.py and tests/test_tokenizer.py; half-typed code is fine, it keeps the tokens it managed to read.

import io
import tokenize

_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


# Returns the source as a list of tokens with comments and blank lines dropped.
# Every line start becomes one token, "\n" plus that line's indentation, and an f-string counts as one token.
def word_tokens(source: str) -> list[str]:
    line_starts = [0]
    for line in io.StringIO(source).readlines():
        line_starts.append(line_starts[-1] + len(line))

    # Converts a (line, column) token position into a character offset in the source.
    def offset(pos: tuple[int, int]) -> int:
        return line_starts[pos[0] - 1] + pos[1]

    out: list[str] = []
    new_line = False
    fstring_start = 0
    fstring_depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.NEWLINE:
                new_line = new_line or bool(tok.string)
                continue
            if tok.type in _SKIP or (tok.type == tokenize.ERRORTOKEN and not tok.string.strip()):
                continue
            if new_line and fstring_depth == 0:
                out.append("\n" + tok.line[: tok.start[1]])
                new_line = False
            if tok.type == _FSTRING_START:
                if fstring_depth == 0:
                    fstring_start = offset(tok.start)
                fstring_depth += 1
            elif tok.type == _FSTRING_END:
                fstring_depth -= 1
                if fstring_depth == 0:
                    out.append(source[fstring_start : offset(tok.end)])
            elif fstring_depth == 0:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # half-typed code: keep what we got so far
    if new_line:
        typed = source[source.rfind("\n") + 1 :]
        out.append("\n" + (typed if not typed.strip() else ""))
    return out


# Returns the source as a list of single characters.
def char_tokens(source: str) -> list[str]:
    return list(source)
