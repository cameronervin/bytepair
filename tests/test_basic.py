import pytest

from bytepair.base import get_stats, merge
from bytepair.basic import BasicTokenizer

SAMPLES = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉",
    "Ünïcödé and\ttabs\nand newlines\r\n",
    "日本語のテキスト",
]


def test_get_stats():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
    assert merge([1, 1, 1], (1, 1), 2) == [2, 1]


def test_wikipedia_example():
    # https://en.wikipedia.org/wiki/Byte_pair_encoding
    # "aaabdaaabac" -> "ZabdZabac" -> "ZYdZYac" -> "XdXac"
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", 256 + 3)
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tok.decode(tok.encode("aaabdaaabac")) == "aaabdaaabac"


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(text):
    tok = BasicTokenizer()
    tok.train("hello world, this is some training text. " * 20, 300)
    assert tok.decode(tok.encode(text)) == text


def test_save_load(tmp_path):
    tok = BasicTokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 10, 280)
    tok.save(tmp_path / "basic")
    loaded = BasicTokenizer()
    loaded.load(tmp_path / "basic.model")
    assert loaded.merges == tok.merges
    text = "the lazy fox"
    assert loaded.encode(text) == tok.encode(text)
    assert (tmp_path / "basic.vocab").exists()


def test_vocab_size_too_small():
    with pytest.raises(ValueError):
        BasicTokenizer().train("abc", 100)


def test_render_token():
    from bytepair.base import render_token

    assert render_token(b"hi\n") == "hi\\u000a"
    assert render_token("😉".encode()[:2]) == "<f0><9f>"
