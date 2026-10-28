import pytest

tiktoken = pytest.importorskip("tiktoken")

from bytepair.gpt4 import GPT4Tokenizer  # noqa: E402

SAMPLES = [
    "",
    "hello world",
    "hello world!!!? (안녕하세요!) lol123 😉",
    "It's 2024. We'll see 1234567 numbers, they've said.",
    "def f(x):\n    return x ** 2\n\n\n",
    "   leading and trailing spaces   ",
    "Ünïcödé 日本語 العربية",
]


@pytest.fixture(scope="module")
def pair():
    return GPT4Tokenizer(), tiktoken.get_encoding("cl100k_base")


@pytest.mark.parametrize("text", SAMPLES)
def test_matches_tiktoken(pair, text):
    ours, theirs = pair
    ids = ours.encode(text)
    assert ids == theirs.encode(text)
    assert ours.decode(ids) == text


def test_special_tokens(pair):
    ours, theirs = pair
    text = "<|endoftext|>hello<|fim_prefix|>"
    assert ours.encode(text, "all") == theirs.encode(text, allowed_special="all")
