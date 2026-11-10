"""S0 corroboration from store_value_diff's JSON (R26 against R27).

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.3 (S0). New. Reads the tool's raw JSON
only (no store, no tree code) and reports:

1. Every `label` difference, in every class: whether the two labels are equal once (i) the job-name
   prefix `levin-phase-floor-06m-` (a) / `levin-phase-floor-07-` (b) is removed and (ii) the trailing
   stamp `-YYYY-MM-DDTHH:MM:SS` (with optional fraction) is removed. This is what the normalisation
   changes, and the claim that nothing else in a label differs.
2. QuadSourceIntegral as the tool reports it: rows differing, only-in-a/b, unmatched, the columns
   it reports differing with their kinds, and every `metadata` key it reports differing, folded over
   indices, with counts.

    ./venv/bin/python -I -B s0_labels_and_qsi_tool_report.py <store_value_diff json>
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

A_PREFIX = "levin-phase-floor-06m-"
B_PREFIX = "levin-phase-floor-07-"
STAMP = re.compile(r"-\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src = sys.argv[1]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    print(f"# input {src} sha256 {sha256(src)}")
    r = json.load(open(src))
    print(f"# a {r['a']}")
    print(f"# b {r['b']}")
    n = Counter()
    bad = []
    stamp_missing = Counter()
    for cls, c in r["classes"].items():
        for row in c["rows"]["rows"]:
            e = row["columns"].get("label")
            if e is None:
                continue
            a, b = e["a"], e["b"]
            ok_prefix = a.startswith(A_PREFIX) and b.startswith(B_PREFIX)
            ra, rb = a[len(A_PREFIX):], b[len(B_PREFIX):]
            if not STAMP.search(ra) or not STAMP.search(rb):
                stamp_missing[cls] += 1
            ra, rb = STAMP.sub("", ra), STAMP.sub("", rb)
            n[cls] += 1
            if not (ok_prefix and ra == rb):
                bad.append((cls, a, b))
    print("1. label differences, by class (all must differ only by the job-name prefix and stamp):")
    for cls, k in sorted(n.items()):
        print(f"   {cls}: {k}")
    print(f"   total {sum(n.values())}; differing in anything else: {len(bad)}; labels without a stamp: {dict(stamp_missing)}")
    for x in bad[:20]:
        print("   NOT ONLY PREFIX/STAMP:", x)
    q = r["classes"]["QuadSourceIntegral"]["rows"]
    print("2. QuadSourceIntegral as the tool reports it:")
    print(f"   identical {q['identical']}, differ {q['differ']}, only_in_a {q['only_in_a']}, "
          f"only_in_b {q['only_in_b']}, unmatched {q['unmatched']}, columns_only_in_a "
          f"{q['columns_only_in_a']}, columns_only_in_b {q['columns_only_in_b']}")
    cols = Counter()
    kinds = Counter()
    mkeys = Counter()
    for row in q["rows"]:
        for col, e in row["columns"].items():
            cols[col] += 1
            kinds[(col, e.get("kind"))] += 1
            if col == "metadata":
                for k in e.get("keys", []):
                    mkeys[re.sub(r"\[\d+\]", "[i]", k)] += 1
    print("   columns reported differing (column, kind): rows")
    for (col, kind), k in sorted(kinds.items()):
        print(f"     {col} ({kind}): {k}")
    print("   metadata keys reported differing (indices folded): row-key count")
    for k, v in sorted(mkeys.items()):
        print(f"     {k}: {v}")
    leaves = Counter(re.split(r"[.\]]", k)[-1] for k in mkeys.elements())
    print(f"   by leaf name: {dict(leaves)}")
    print(f"VERDICT labels: {'only prefix and stamp' if not bad else 'OTHER LABEL DIFFERENCES'}; "
          f"QSI metadata leaves other than elapsed/compute_time: "
          f"{sorted(set(leaves) - {'elapsed', 'compute_time'})}")


if __name__ == "__main__":
    main()
