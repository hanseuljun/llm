"""Generate data/v4: subject-verb agreement PLUS a cued copy (associative recall).

v3 only tested agreement, which a position-weighted bag-of-words already solves
(~95%).  v4 adds a slot that such a model structurally cannot do: the object
noun is an exact COPY of an earlier noun, and *which* earlier noun is selected
by the verb.

    [adverbial]  the N1  [PREP the N2]*  VERB  the N3
                 ^subject      ^attractor(s)   ^== N1 or == last N2,
                                                  chosen by the VERB

    "sees" / "likes"     -> copy the SUBJECT
    "chases" / "watches" -> copy the last ATTRACTOR

Both source positions vary (the adverbial is 0, 1 or 3 tokens; there are 0-2
attractors), and place nouns sit in noun-like slots, so the copy source cannot
be found by position alone.  Retrieving it requires content-addressed lookup -
exactly what an attention head does and a masked mean cannot.

Agreement is still tested at the verb slot, so v3's probe carries over.
"""

import json
import random
from pathlib import Path

SING = ["cat", "dog", "bird", "fish", "horse", "mouse"]
PLUR = ["cats", "dogs", "birds", "fishes", "horses", "mice"]
# verb lemma -> (singular form, plural form); cue says which noun the object copies
SUBJECT_CUE = [("sees", "see"), ("likes", "like")]
ATTRACTOR_CUE = [("chases", "chase"), ("watches", "watch")]
PREPS = ["near", "by"]
PLACES = ["garden", "river", "lake", "park"]
ADVERBIALS = [[], ["today"], ["here"], ["then"]] + [
    [p, "the", n] for p in ("in", "at") for n in PLACES
]


def noun(plural, rng):
    return rng.choice(PLUR if plural else SING)


def sentence(rng, hard=False):
    subj_plural = rng.random() < 0.5
    n_mods = 2 if hard else rng.choice([0, 1, 2])

    words = list(rng.choice(ADVERBIALS))
    subject = noun(subj_plural, rng)
    words += ["the", subject]

    last_attractor = None
    for _ in range(n_mods):
        # hard: every attractor disagrees in number with the subject
        attr_plural = (not subj_plural) if hard else rng.random() < 0.5
        last_attractor = noun(attr_plural, rng)
        words += [rng.choice(PREPS), "the", last_attractor]

    # with no attractor there is nothing for an attractor-cue verb to point at
    cue_subject = True if last_attractor is None else rng.random() < 0.5
    lemma = rng.choice(SUBJECT_CUE if cue_subject else ATTRACTOR_CUE)
    words += [lemma[1] if subj_plural else lemma[0]]
    words += ["the", subject if cue_subject else last_attractor]
    return " ".join(words)


def main():
    here = Path(__file__).parent
    rng = random.Random(0)

    train = [sentence(rng) for _ in range(10000)]
    train_set = set(train)

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
    print(f"sentence lengths: {lens} -> max tokens with <bos>/<eos>: {max(lens) + 2}")


if __name__ == "__main__":
    main()
