"""Generate data/v3: subject-verb agreement with VARIABLE sentence structure.

v2 had a single fixed template, so the subject was always at index 1 and the
verb always at index 5.  A model that weights positions can solve that without
ever looking at content.  v3 varies both the sentence-initial material and the
number of attractor phrases, so the subject's position is 1-4 and the verb's is
2-9.  Finding the subject now requires content-dependent routing.

    [adverbial]  the N1  [PREP the N2]*  VERB  the N3
                 ^subject                ^agrees with N1, not with any N2
"""
import json
import random
from pathlib import Path

SING = ["cat", "dog", "bird", "fish", "horse", "mouse"]
PLUR = ["cats", "dogs", "birds", "fishes", "horses", "mice"]
VERB_S = ["chases", "sees", "likes"]
VERB_P = ["chase", "see", "like"]
PREPS = ["near", "by"]
PLACES = ["garden", "river", "lake", "park"]
# sentence-initial adverbials of length 0, 1 and 3 -> subject index varies
ADVERBIALS = [[], ["today"], ["here"], ["then"]] + [
    [p, "the", n] for p in ("in", "at") for n in PLACES
]

def noun(plural, rng):
    return rng.choice(PLUR if plural else SING)

def sentence(rng, hard=False):
    subj_plural = rng.random() < 0.5
    n_mods = rng.choice([1, 2]) if hard else rng.choice([0, 1, 2])

    words = list(rng.choice(ADVERBIALS))
    words += ["the", noun(subj_plural, rng)]
    for _ in range(n_mods):
        # hard: every attractor disagrees in number with the subject
        attr_plural = (not subj_plural) if hard else rng.random() < 0.5
        words += [rng.choice(PREPS), "the", noun(attr_plural, rng)]
    words += [rng.choice(VERB_P if subj_plural else VERB_S)]
    words += ["the", noun(rng.random() < 0.5, rng)]
    return " ".join(words)

def build(n, rng, hard=False):
    seen, out = set(), []
    while len(out) < n:
        s = sentence(rng, hard)
        out.append(s)
        seen.add(s)
    return out, seen

def main():
    here = Path(__file__).parent
    rng = random.Random(0)

    train, train_set = build(10000, rng, hard=False)
    # held-out sentences must not appear in train
    def fresh(n, hard):
        out = []
        while len(out) < n:
            s = sentence(rng, hard)
            if s not in train_set:
                out.append(s)
        return out
    held_out = fresh(3000, hard=False)
    held_out_hard = fresh(1500, hard=True)

    words = {w for s in train + held_out + held_out_hard for w in s.split()}
    vocab = {"<bos>": 0, "<eos>": 1}
    for w in ["the"] + sorted(words - {"the"}):
        vocab.setdefault(w, len(vocab))

    (here / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    for name, rows in [("train", train), ("held_out", held_out),
                       ("held_out_hard", held_out_hard)]:
        (here / f"{name}.txt").write_text("\n".join(rows) + "\n")

    lens = sorted({len(s.split()) for s in train})
    print(f"vocab {len(vocab)} | train {len(train)} | held_out {len(held_out)} "
          f"| hard {len(held_out_hard)}")
    print(f"sentence lengths: {lens}  -> max tokens with <bos>/<eos>: {max(lens)+2}")

if __name__ == "__main__":
    main()
