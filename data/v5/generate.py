"""Generate data/v5: pure associative recall (no grammar costume).

v4 was solvable by POSITION-addressed retrieval: multiplicative position
binding plus an MLP can memorise the nine structural layouts and unbind the
right slot.  v5 removes that option.  Each line is a list of key-value pairs in
random order, then a query:

    a 7 c 12 b 3 ? c 12
    ^^^^^^^^^^^^   ^ ^^
    the store      |  the answer: the value bound to c
                   the query key

Keys are distinct within a line and appear in random order, so the position of
any given key is re-randomised every example.  There is no function from
(query key, sequence shape) to a position, which is exactly what a
position-binding model needs.  Answering requires comparing the query against
what is at each position - a content-addressed lookup.

The answer is the second-to-last token, so `i == len(token_ids) - 2` still
marks the probe slot, as in v4.
"""

import json
import random
from pathlib import Path

KEYS = [chr(ord("a") + i) for i in range(20)]  # a..t
VALUES = [str(i) for i in range(20)]  # 0..19
QUERY = "?"
MIN_PAIRS, MAX_PAIRS = 3, 8


def line(rng, hard=False):
    n = MAX_PAIRS if hard else rng.randint(MIN_PAIRS, MAX_PAIRS)
    keys = rng.sample(KEYS, n)  # distinct -> the query is unambiguous
    values = [rng.choice(VALUES) for _ in range(n)]  # may repeat

    # hard: query a key from the first half, so the answer is far from the query
    q = rng.randrange(n // 2) if hard else rng.randrange(n)

    tokens = []
    for k, v in zip(keys, values):
        tokens += [k, v]
    tokens += [QUERY, keys[q], values[q]]
    return " ".join(tokens)


def main():
    here = Path(__file__).parent
    rng = random.Random(0)

    train = [line(rng) for _ in range(10000)]
    train_set = set(train)

    def fresh(n, hard):
        out = []
        while len(out) < n:
            s = line(rng, hard)
            if s not in train_set:
                out.append(s)
        return out

    held_out = fresh(3000, hard=False)
    held_out_hard = fresh(1500, hard=True)

    vocab = {"<bos>": 0, "<eos>": 1, QUERY: 2}
    for w in KEYS + VALUES:
        vocab.setdefault(w, len(vocab))

    (here / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    for name, rows in [
        ("train", train),
        ("held_out", held_out),
        ("held_out_hard", held_out_hard),
    ]:
        (here / f"{name}.txt").write_text("\n".join(rows) + "\n")

    lens = sorted({len(s.split()) for s in train})
    print(
        f"vocab {len(vocab)} | train {len(train)} | held_out {len(held_out)} "
        f"| hard {len(held_out_hard)}"
    )
    print(f"line lengths: {lens} -> max tokens with <bos>/<eos>: {max(lens) + 2}")


if __name__ == "__main__":
    main()
