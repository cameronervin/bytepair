import pytest

from bytepair.basic import BasicTokenizer
from bytepair.regex import GPT2_SPLIT_PATTERN, RegexTokenizer

TRAIN_TEXT = (
    "The dog ran. The dog sat! Was it the dog's ball? It's 2024 and 12345 dogs ran.\n"
) * 20
SAMPLES = ["", "dog", "hello world!!!? (안녕하세요!) lol123 😉", "a\n\n  b\r\n", "日本語"]


@pytest.fixture
def tok():
    t = RegexTokenizer()
    t.train(TRAIN_TEXT, 320)
    return t


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_split_pattern():
    tok = RegexTokenizer()
    chunks = tok.compiled_pattern.findall("Hello world's 12345!!\n\n")
    assert chunks == ["Hello", " world", "'s", " ", "123", "45", "!!\n\n"]


def test_merges_stay_inside_chunks(tok):
    # No learned token may contain a letter followed by a space.
    for token in tok.vocab.values():
        assert b"g " not in token and b". " not in token


def test_weighted_training_matches_naive():
    # Training on unique chunks with weights must match merging every chunk.
    text = "aaa bbb aaa aaa ccc aaab "
    tok = RegexTokenizer(GPT2_SPLIT_PATTERN)
    tok.train(text, 262)
    naive_merges = {}
    chunks = [list(c.encode()) for c in tok.compiled_pattern.findall(text)]
    from bytepair.base import get_stats, merge

    for i in range(6):
        stats = {}
        for c in chunks:
            get_stats(c, stats)
        pair = max(stats, key=stats.get)
        chunks = [merge(c, pair, 256 + i) for c in chunks]
        naive_merges[pair] = 256 + i
    assert tok.merges == naive_merges


def test_special_tokens(tok):
    tok.register_special_tokens({"<|endoftext|>": 1000})
    text = "the dog<|endoftext|>the cat"
    ids = tok.encode(text, allowed_special="all")
    assert 1000 in ids
    assert tok.decode(ids) == text
    assert 1000 not in tok.encode(text, allowed_special="none")
    assert tok.decode(tok.encode(text, allowed_special="none")) == text
    with pytest.raises(ValueError):
        tok.encode(text)
    assert 1000 in tok.encode(text, allowed_special={"<|endoftext|>"})


def test_save_load(tok, tmp_path):
    tok.register_special_tokens({"<|endoftext|>": 1000})
    tok.save(tmp_path / "regex")
    loaded = RegexTokenizer()
    loaded.load(tmp_path / "regex.model")
    text = "the dog<|endoftext|>ran"
    assert loaded.encode(text, "all") == tok.encode(text, "all")
    assert loaded.pattern == tok.pattern


def test_basic_and_regex_agree_on_single_chunk():
    # One chunk with no pattern splits behaves like BasicTokenizer.
    b, r = BasicTokenizer(), RegexTokenizer()
    b.train("aaabdaaabac", 259)
    r.train("aaabdaaabac", 259)
    assert b.merges == r.merges
