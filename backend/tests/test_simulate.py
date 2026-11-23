from mini_copilot.evaluation.simulate import make_samples

SOURCE = 'import os\n\n\ndef f(x):\n    # note\n    return f"{x}!" + os.sep\n'


def test_samples_are_cut_right_before_the_expected_word():
    samples = make_samples(SOURCE, max_samples=100)
    assert samples
    for context, word in samples:
        assert SOURCE.startswith(context)
        assert SOURCE[len(context):].startswith(word)


def test_layout_and_comments_are_never_targets():
    words = {word for _, word in make_samples(SOURCE, max_samples=100)}
    assert not words & {"\n", "", "# note"} and not any(w.isspace() for w in words)


def test_an_fstring_is_one_word():
    words = [word for _, word in make_samples(SOURCE, max_samples=100)]
    assert 'f"{x}!"' in words


def test_samples_are_reproducible_and_capped():
    assert make_samples(SOURCE, 5) == make_samples(SOURCE, 5)
    assert len(make_samples(SOURCE, 5)) == 5


def test_broken_and_empty_files_do_not_crash():
    assert make_samples("") == []
    make_samples("def f(:\n  x = (")
