# bytepair

[![CI](https://github.com/cameronervin/bytepair/actions/workflows/ci.yml/badge.svg)](https://github.com/cameronervin/bytepair/actions/workflows/ci.yml)

A byte-level BPE tokenizer written from scratch in pure Python. It includes a tokenizer that reproduces
GPT-4's `cl100k_base` output exactly.

- **[Live visualizer](https://cameronervin.github.io/bytepair/)**: type text and see it split into tokens. The Python code runs in the browser with Pyodide.
- **[Walkthrough notebook](notebooks/walkthrough.ipynb)**: builds the tokenizer step by step, from UTF-8 bytes to GPT-4 parity.

## Tokenizers

| Class | File | What it does |
|---|---|---|
| `BasicTokenizer` | [basic.py](src/bytepair/basic.py) | BPE on the raw bytes of the whole text. |
| `RegexTokenizer` | [regex.py](src/bytepair/regex.py) | Splits text with the GPT-4 regex first, so merges stay inside words, numbers or punctuation. Supports special tokens. |
| `GPT4Tokenizer` | [gpt4.py](src/bytepair/gpt4.py) | `RegexTokenizer` loaded with merges rebuilt from tiktoken's `cl100k_base`. Output matches tiktoken. |

Shared helpers and save/load live in [base.py](src/bytepair/base.py).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cameronervin/bytepair && cd bytepair
uv sync
```

Python:

```python
from bytepair.regex import RegexTokenizer

tok = RegexTokenizer()
tok.train(open("some_text.txt").read(), vocab_size=1024)
ids = tok.encode("hello world")
tok.decode(ids)  # "hello world"
tok.save("models/mine")  # writes mine.model and a readable mine.vocab
```

CLI:

```bash
uv run bytepair train some_text.txt --vocab-size 1024 --out models/mine
uv run bytepair encode "Elizabeth Bennet met Mr. Darcy." --model models/pride.model --ids
uv run bytepair decode 400 531 --model models/pride.model
```

`encode` prints the text with a different background color for each token.

## Showcase model

[`models/pride.model`](models/pride.model) is a 1,024-token `RegexTokenizer` trained on *Pride and Prejudice*
from Project Gutenberg. Training takes about 15 seconds on a laptop CPU. [`pride.vocab`](models/pride.vocab)
lists every merge. The main character names become single tokens: `Elizabeth` is token 400, ` Darcy` is 472.

To reproduce:

```bash
uv run python scripts/fetch_corpus.py
uv run python scripts/train_showcase.py
```

## Benchmark

`scripts/benchmark.py` encodes the full book (737,914 bytes) with each tokenizer on an Apple Silicon laptop.

| Tokenizer | Tokens | Bytes/token | Encode time |
|---|---:|---:|---:|
| bytepair Regex, 1k vocab (trained on this book) | 253,534 | 2.91 | 0.28s |
| tiktoken gpt2 (50k vocab) | 191,608 | 3.85 | 0.06s |
| tiktoken cl100k_base (100k vocab) | 171,510 | 4.30 | 0.06s |
| tiktoken o200k_base (200k vocab) | 170,258 | 4.33 | 0.04s |

With 1% of GPT-4's vocabulary, the showcase model reaches about two thirds of its compression on this book.
tiktoken is written in Rust, so it is faster. bytepair stays under a second because it encodes each unique
regex chunk once and caches the result.

`GPT4Tokenizer` produces the same 171,510 ids as tiktoken `cl100k_base` on the full book.

## Implementation notes

- **Weighted training.** `RegexTokenizer.train` counts unique chunks and weights pair counts by chunk frequency.
  The merges are the same as when processing every chunk, and training runs much faster.
- **Merge order on encode.** Encoding repeatedly merges the present pair with the lowest merge id. Later merges
  are built from earlier ones, so this order is required.
- **Special tokens.** `encode` raises by default if the text contains a special token string. This follows
  tiktoken and stops user text from injecting structure tokens. Pass `allowed_special="all"` to allow them.
- **GPT-4 byte shuffle.** tiktoken gives the 256 single bytes ranks that differ from their byte values.
  `GPT4Tokenizer` permutes bytes before BPE and reverses the permutation when decoding.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## What I learned

- Tokens are bytes, not characters. An emoji is 4 bytes, and a small tokenizer shows it as 4 tokens.
- The training data sets the tokenizer's strengths. The Austen model compresses English prose well and handles
  code and Korean at close to one byte per token.
- Many LLM quirks come from tokenization: weak arithmetic on numbers split into chunks, higher cost for
  non-English text, and sensitivity to trailing whitespace.

## Credits

Based on Andrej Karpathy's video [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
and his [minbpe](https://github.com/karpathy/minbpe) repo. This code is a separate implementation written for
learning. Split patterns and the `cl100k_base` vocabulary come from [tiktoken](https://github.com/openai/tiktoken).

## License

MIT
