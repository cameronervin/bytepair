"""Download Pride and Prejudice from Project Gutenberg into data/.

Strips the Gutenberg license header and footer so only the book remains.
"""

import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "pride_and_prejudice.txt"


def main() -> None:
    if OUT.exists():
        print(f"already have {OUT}")
        return
    raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8-sig")
    start = raw.index("*** START OF")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF")
    text = raw[start:end].strip().replace("\r\n", "\n") + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
