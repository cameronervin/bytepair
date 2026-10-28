"""bytepair command-line interface.

bytepair train corpus.txt --vocab-size 1024 --out models/mine
bytepair encode "some text" --model models/mine.model
bytepair decode 72 101 108 --model models/mine.model
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.text import Text

from .base import render_token
from .basic import BasicTokenizer
from .regex import RegexTokenizer

# Background colors cycled across tokens so boundaries are visible.
COLORS = ["#4c6ef5", "#f76707", "#2f9e44", "#e64980", "#7048e8", "#1098ad"]


def load_tokenizer(model: str) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.load(model)
    return tok


def colorize(tok: RegexTokenizer, ids: list[int]) -> Text:
    out = Text()
    for i, idx in enumerate(ids):
        piece = render_token(tok.vocab[idx]) if idx in tok.vocab else f"<{idx}>"
        out.append(piece, style=f"white on {COLORS[i % len(COLORS)]}")
    return out


def cmd_train(args: argparse.Namespace) -> None:
    text = Path(args.corpus).read_text(encoding="utf-8")
    tok = BasicTokenizer() if args.basic else RegexTokenizer()
    t0 = time.perf_counter()
    tok.train(text, args.vocab_size, verbose=args.verbose)
    tok.save(args.out)
    print(
        f"trained {args.vocab_size} tokens in {time.perf_counter() - t0:.1f}s -> {args.out}.model"
    )


def cmd_encode(args: argparse.Namespace) -> None:
    tok = load_tokenizer(args.model)
    text = args.text if args.text is not None else sys.stdin.read()
    ids = tok.encode(text, allowed_special="all")
    console = Console()
    console.print(colorize(tok, ids))
    n_bytes = len(text.encode("utf-8"))
    console.print(f"[dim]{len(ids)} tokens, {n_bytes} bytes[/dim]")
    if args.ids:
        print(" ".join(map(str, ids)))


def cmd_decode(args: argparse.Namespace) -> None:
    tok = load_tokenizer(args.model)
    print(tok.decode(args.ids))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bytepair", description="Byte-level BPE tokenizer.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a tokenizer on a text file")
    p.add_argument("corpus")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--out", required=True, help="output prefix, e.g. models/mine")
    p.add_argument("--basic", action="store_true", help="use BasicTokenizer (no regex split)")
    p.add_argument("-v", "--verbose", action="store_true", help="print each merge")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("encode", help="show how text splits into tokens")
    p.add_argument("text", nargs="?", help="text to encode (reads stdin if omitted)")
    p.add_argument("--model", required=True)
    p.add_argument("--ids", action="store_true", help="also print token ids")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("decode", help="turn token ids back into text")
    p.add_argument("ids", nargs="+", type=int)
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_decode)

    args = parser.parse_args(argv)
    args.func(args)
