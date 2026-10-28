"""GPT4Tokenizer: RegexTokenizer loaded with OpenAI's cl100k_base merges.

tiktoken stores a vocab as {token bytes: rank}, not as a list of merges. We
recover the merges by running BPE on each token's bytes, using only the
merges ranked below it. The two parts that merge into the token are its pair.

tiktoken also gives the 256 single bytes ranks that are not equal to the
byte values. We keep that as a byte shuffle: bytes are permuted before BPE
and permuted back after decoding.

Needs tiktoken: pip install bytepair[gpt4]
"""

from .regex import GPT4_SPLIT_PATTERN, RegexTokenizer

GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """Split token into parts using only merges with rank below max_rank."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i, pair in enumerate(zip(parts[:-1], parts[1:], strict=True)):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None or min_rank >= max_rank:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def recover_merges(mergeable_ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        merges[(mergeable_ranks[pair[0]], mergeable_ranks[pair[1]])] = rank
    return merges


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self) -> None:
        import tiktoken

        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.vocab = self._build_vocab()
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        return super()._encode_bytes(bytes(self.byte_shuffle[b] for b in text_bytes))

    def decode(self, ids: list[int]) -> str:
        parts = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                parts.append(bytes(self.inverse_byte_shuffle[b] for b in self.vocab[idx]))
        return b"".join(parts).decode("utf-8", errors="replace")

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError("GPT4Tokenizer is pretrained")

    def save(self, prefix) -> None:
        raise NotImplementedError("GPT4Tokenizer merges come from tiktoken")

    def load(self, model_file) -> None:
        raise NotImplementedError("GPT4Tokenizer merges come from tiktoken")
