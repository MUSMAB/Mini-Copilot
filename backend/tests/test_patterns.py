import ast

from mini_copilot.analysis.patterns import count_if_chain


def test_count_if_chain():
    src = "if a:\n  pass\nelif b:\n  pass\nelif c:\n  pass\n"
    assert count_if_chain(ast.parse(src).body[0]) == 3
