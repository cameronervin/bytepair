"""Train the showcase tokenizer on Pride and Prejudice and save it to models/."""

import time
from pathlib import Path

from bytepair.regex import RegexTokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "pride_and_prejudice.txt"
VOCAB_SIZE = 1024


def main() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    tok = RegexTokenizer()
    t0 = time.perf_counter()
    tok.train(text, VOCAB_SIZE)
    print(f"trained {VOCAB_SIZE} tokens in {time.perf_counter() - t0:.1f}s")
    tok.register_special_tokens({"<|endoftext|>": VOCAB_SIZE})
    tok.save(ROOT / "models" / "pride")

    ids = tok.encode_ordinary(text)
    n_bytes = len(text.encode("utf-8"))
    print(f"{n_bytes:,} bytes -> {len(ids):,} tokens ({n_bytes / len(ids):.2f} bytes/token)")
    assert tok.decode(ids) == text


if __name__ == "__main__":
    main()
