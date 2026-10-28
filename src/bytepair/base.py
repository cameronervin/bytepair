"""Shared BPE helpers and the Tokenizer base class.

A tokenizer is defined by:
- merges: an ordered dict mapping a pair of token ids to the new token id.
- special_tokens: a dict mapping a special string to its token id.
- pattern: the regex used to split text before BPE (empty for BasicTokenizer).
"""

import unicodedata
from pathlib import Path

MODEL_VERSION = "bytepair v1"


def get_stats(ids: list[int], counts: dict | None = None) -> dict[tuple[int, int], int]:
    """Count how often each consecutive pair appears in ids.

    Pass an existing counts dict to accumulate across several chunks.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:], strict=False):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of pair in ids with the single token idx."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def replace_control_characters(s: str) -> str:
    """Escape control characters (like newline) so a token prints on one line."""
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(t: bytes) -> str:
    """Decode token bytes for display. Partial UTF-8 sequences show as U+FFFD."""
    return replace_control_characters(t.decode("utf-8", errors="replace"))


class Tokenizer:
    """Base class. Subclasses implement train, encode and decode."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.pattern = ""
        self.special_tokens: dict[str, int] = {}
        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        """Apply merges in the order they were learned.

        At each step, merge the pair that was learned earliest. Later merges
        can depend on earlier ones, so order matters.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def _build_vocab(self) -> dict[int, bytes]:
        """Rebuild the id -> bytes table from the 256 base bytes plus merges."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, prefix: str | Path) -> None:
        """Write prefix.model (loadable) and prefix.vocab (for reading only)."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        lines = [MODEL_VERSION, self.pattern, str(len(self.special_tokens))]
        lines += [f"{special} {idx}" for special, idx in self.special_tokens.items()]
        lines += [f"{p0} {p1}" for p0, p1 in self.merges]
        Path(f"{prefix}.model").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Each merged token is shown with the two tokens it was built from.
        children = {idx: pair for pair, idx in self.merges.items()}
        out = []
        for idx, token in self.vocab.items():
            s = render_token(token)
            if idx in children:
                c0, c1 = children[idx]
                s0, s1 = render_token(self.vocab[c0]), render_token(self.vocab[c1])
                out.append(f"[{s0}][{s1}] -> [{s}] {idx}")
            else:
                out.append(f"[{s}] {idx}")
        Path(f"{prefix}.vocab").write_text("\n".join(out) + "\n", encoding="utf-8")

    def load(self, model_file: str | Path) -> None:
        """Load a .model file written by save."""
        lines = Path(model_file).read_text(encoding="utf-8").split("\n")
        if lines[0] != MODEL_VERSION:
            raise ValueError(f"unknown model version: {lines[0]!r}")
        self.pattern = lines[1]
        n_special = int(lines[2])
        self.special_tokens = {}
        for line in lines[3 : 3 + n_special]:
            special, idx = line.rsplit(" ", 1)
            self.special_tokens[special] = int(idx)
        self.merges = {}
        idx = 256
        for line in lines[3 + n_special :]:
            if not line:
                continue
            p0, p1 = map(int, line.split())
            self.merges[(p0, p1)] = idx
            idx += 1
        self.vocab = self._build_vocab()
