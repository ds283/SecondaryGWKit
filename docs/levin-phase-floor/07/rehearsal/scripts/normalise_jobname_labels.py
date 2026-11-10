"""Write a copy of store_value_diff's JSON (R26 against R27) with the job-name prefix of every
`label` normalised, so that 05's classify_non_qsi.py can be run on it unchanged.

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.3 (S0, R27 against R26); 06m rehearsal agent's script with the two prefixes and this docstring changed.

Why: R26 was built with `--job-name levin-phase-floor-06m` and R27 with
`--job-name levin-phase-floor-07` (the brief's job name). main.py puts the job name at the head
of every label, so every label differs in its prefix as well as in its trailing
`-YYYY-MM-DDTHH:MM:SS`. 05's classifier calls a label difference a `label` only if the two are
equal once the stamp is removed, so it reports these as UNCLASSIFIED. This script replaces, in
side `a` of every `label` entry of every class, the leading `levin-phase-floor-06m-` by
`levin-phase-floor-07-`, asserting that `a` starts with the first and `b` with the second.
Nothing else in the JSON is touched; the counts of replacements per class are printed.

    ./venv/bin/python -I normalise_jobname_labels.py <in.json> <out.json>
"""

import hashlib
import json
import sys
from collections import Counter

A_PREFIX = "levin-phase-floor-06m-"
B_PREFIX = "levin-phase-floor-07-"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src, dst = sys.argv[1:3]
    print(f"# script {__file__} sha256 {sha256(__file__)}")
    print(f"# input {src} sha256 {sha256(src)}")
    r = json.load(open(src))
    n = Counter()
    other_cols = Counter()
    for cls, c in r["classes"].items():
        for row in c["rows"]["rows"]:
            e = row["columns"].get("label")
            if e is None:
                continue
            assert e["a"].startswith(A_PREFIX), (cls, e["a"])
            assert e["b"].startswith(B_PREFIX), (cls, e["b"])
            e["a"] = B_PREFIX + e["a"][len(A_PREFIX):]
            n[cls] += 1
    for cls, k in sorted(n.items()):
        print(f"{cls}: {k} labels normalised")
    print(f"total: {sum(n.values())}")
    with open(dst, "w") as f:
        json.dump(r, f)
    print(f"# output {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
