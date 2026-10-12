"""Download the TinyStories corpus.

The .txt files are large (valid 18.5 MB, train 1.8 GB), so they are gitignored
and fetched on demand rather than committed.  Start with valid.txt - it is
already ~18x TinyShakespeare and plenty for getting a pipeline working.

    uv run data/tinystories/download.py            # valid.txt only
    uv run data/tinystories/download.py --train    # also the 1.8 GB train set
"""

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"valid.txt": "TinyStories-valid.txt", "train.txt": "TinyStories-train.txt"}


def report(done: int, block: int, total: int) -> None:
    """Print progress every 10%, so piped output stays readable."""
    if total <= 0:
        return
    step = max(1, total // (10 * block))
    if done % step == 0:
        pct = min(100.0, done * block * 100 / total)
        print(f"  {pct:5.1f}%  of {total / 1048576:.1f} MB", flush=True)


def fetch(local: str) -> None:
    here = Path(__file__).parent
    out = here / local
    if out.exists():
        print(f"{local}: already present ({out.stat().st_size / 1048576:.1f} MB)")
        return
    print(f"{local}: downloading")
    urllib.request.urlretrieve(BASE + FILES[local], out, reporthook=report)
    print(f"{local}: {out.stat().st_size / 1048576:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="also fetch the 1.8 GB train set")
    args = parser.parse_args()

    fetch("valid.txt")
    if args.train:
        fetch("train.txt")


if __name__ == "__main__":
    main()
