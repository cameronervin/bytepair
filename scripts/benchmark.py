"""Compare bytepair with tiktoken on Pride and Prejudice.

Reports compression (bytes per token) and encode speed. tiktoken is written
in Rust, so the speed gap is expected. bytepair caches each unique chunk,
which is why encoding the whole book takes well under a second.
"""

import time
from pathlib import Path

import tiktoken

from bytepair.regex import RegexTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "data" / "pride_and_prejudice.txt").read_text(encoding="utf-8")
N_BYTES = len(TEXT.encode("utf-8"))


def timed(fn):
    t0 = time.perf_counter()
    ids = fn()
    return ids, time.perf_counter() - t0


def main() -> None:
    rows = []

    tok = RegexTokenizer()
    tok.load(ROOT / "models" / "pride.model")
    ids, secs = timed(lambda: tok.encode_ordinary(TEXT))
    rows.append(("bytepair Regex, 1k vocab (trained on this book)", len(ids), secs))

    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        ids, secs = timed(lambda enc=enc: enc.encode_ordinary(TEXT))
        rows.append((f"tiktoken {name} ({enc.n_vocab // 1000}k vocab)", len(ids), secs))

    print(f"Corpus: Pride and Prejudice, {N_BYTES:,} bytes\n")
    print("| Tokenizer | Tokens | Bytes/token | Encode time |")
    print("|---|---:|---:|---:|")
    for name, n, secs in rows:
        print(f"| {name} | {n:,} | {N_BYTES / n:.2f} | {secs:.2f}s |")


if __name__ == "__main__":
    main()
