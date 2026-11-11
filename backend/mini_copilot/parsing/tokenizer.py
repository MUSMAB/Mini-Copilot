"""Turn Python source into word tokens (for the word n-gram) or characters."""

import io
import tokenize


def word_tokens(source: str) -> list[str]:
    """Python tokens with comments and blank-line markers dropped."""
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type not in skip:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # truncated code: keep what we got so far
    return out


def char_tokens(source: str) -> list[str]:
    return list(source)
