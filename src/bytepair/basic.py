"""BasicTokenizer: BPE on the raw UTF-8 bytes of the whole text.

It does not split text first, so merges can cross word and punctuation
boundaries. It also has no special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) x{stats[pair]}"
                )

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        return self._encode_bytes(text.encode("utf-8"))

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
