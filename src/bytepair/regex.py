"""RegexTokenizer: split text with a regex, then run BPE inside each chunk.

Splitting stops merges from crossing category boundaries. For example,
"dog." and "dog!" share the token "dog" instead of learning "dog." and
"dog!" as separate tokens. It also adds special tokens such as
<|endoftext|>, which are never produced by BPE.
"""

from collections import Counter
from pathlib import Path

import regex as re

from .base import Tokenizer, merge

# Split patterns from openai/tiktoken.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens: dict[int, str] = {}
        self._cache: dict[bytes, list[int]] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        num_merges = vocab_size - 256

        # Many chunks repeat (" the", " and"). Train on each unique chunk once
        # and weight its pair counts by how often it appears. This gives the
        # same merges as processing every chunk, with much less work.
        chunk_counts = Counter(self.compiled_pattern.findall(text))
        chunks = [list(c.encode("utf-8")) for c in chunk_counts]
        weights = list(chunk_counts.values())

        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for ids, w in zip(chunks, weights, strict=True):
                for pair in zip(ids, ids[1:], strict=False):
                    stats[pair] = stats.get(pair, 0) + w
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(ids, pair, idx) for ids in chunks]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) x{stats[pair]}"
                )

        self.merges = merges
        self.vocab = vocab
        self._cache = {}

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Add special tokens, e.g. {"<|endoftext|>": 100257}."""
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()

    def load(self, model_file: str | Path) -> None:
        super().load(model_file)
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self._cache = {}

    def decode(self, ids: list[int]) -> str:
        parts = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        if chunk not in self._cache:
            self._cache[chunk] = self._encode_bytes(chunk)
        return self._cache[chunk]

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text and treat any special token strings as plain text."""
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise") -> list[int]:
        """Encode text, handling special tokens.

        allowed_special works like tiktoken:
        - "all": every registered special token is encoded as its id.
        - "none": special strings are encoded as plain text.
        - "none_raise": raise if the text contains a special string (default).
        - a set of strings: only those are encoded as special tokens.
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for s in self.special_tokens:
                if s in text:
                    raise ValueError(f"text contains special token {s!r}")
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"invalid allowed_special: {allowed_special!r}")

        if not special:
            return self.encode_ordinary(text)

        # Split on the special tokens, keeping them (the capture group does that).
        special_pattern = "(" + "|".join(re.escape(s) for s in special) + ")"
        ids = []
        for part in re.split(special_pattern, text):
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
