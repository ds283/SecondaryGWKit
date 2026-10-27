"""levin-phase-floor prompt 06 §5.4: a per-class summary of store_value_diff's JSON, R13 against R14,
split at the five classes §0.3 expects to change (GkWKBIntegration, TkWKBIntegration, GkSource,
GkSourcePolicyData, QuadSourceIntegral). Reads only the JSON.

It is 05a's inline summary (05a/rehearsal/out/commands.log), with three additions: the columns
present in one store only (the tool's prompt-06 keys), the split at the five classes, and a check
that, outside them, every value table is identical, no table has a column in one store only, and
no row is in one store only or unmatched.

    ./venv/bin/python -I summarise_diff.py <store_value_diff.json>
"""

import hashlib
import json
import sys
from pathlib import Path

FIVE = {"GkWKBIntegration", "TkWKBIntegration", "GkSource", "GkSourcePolicyData", "QuadSourceIntegral"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    me = Path(__file__).resolve()
    print(f"# {me.name} sha256 {sha256(me)}")
    print(f"# input {sys.argv[1]} sha256 {sha256(sys.argv[1])}")
    d = json.load(open(sys.argv[1]))
    print("a:", d["a"]); print("b:", d["b"])
    print("not_compared:", d["not_compared"], "identical:", d["identical"])
    nv = nrows = 0
    vbad = []
    outside_problems = []
    for cls, c in d["classes"].items():
        r = c["rows"]
        side = "INSIDE" if cls in FIVE else "outside"
        print(f'{cls} [{side}]: identical={c["identical"]} rows identical={r["identical"]} differ={r["differ"]} '
              f'onlyA={len(r["only_in_a"])} onlyB={len(r["only_in_b"])} unmatched={len(r["unmatched"])} '
              f'cols_onlyA={r["columns_only_in_a"]} cols_onlyB={r["columns_only_in_b"]} '
              f'unresolvable={c["unresolvable_columns"]}')
        for col, e in sorted(r["columns"].items()):
            print("    col", col, json.dumps(e)[:600])
        if cls not in FIVE and (r["only_in_a"] or r["only_in_b"] or r["unmatched"] or r["columns_only_in_a"]
                                or r["columns_only_in_b"] or c["unresolvable_columns"]):
            outside_problems.append((cls, "rows"))
        for vt, v in c["values"].items():
            nv += 1
            nrows += v["identical"] + v["differ"]
            ok = (v["differ"] == 0 and not v["only_in_a"] and not v["only_in_b"] and not v["unmatched"]
                  and not v["columns_only_in_a"] and not v["columns_only_in_b"])
            if not ok:
                vbad.append((cls, vt))
                if cls not in FIVE:
                    outside_problems.append((cls, vt))
            print(f'    value table {vt}: identical={v["identical"]} differ={v["differ"]} '
                  f'onlyA={len(v["only_in_a"])} onlyB={len(v["only_in_b"])} unmatched={len(v["unmatched"])} '
                  f'cols_onlyA={v["columns_only_in_a"]} cols_onlyB={v["columns_only_in_b"]}')
            for col, e in sorted(v["columns"].items()):
                print("        col", col, json.dumps(e)[:400])
    print(f"value tables: {nv}, rows {nrows}, not bit-identical: {vbad}")
    print("classes not identical:", [k for k, c in d["classes"].items() if not c["identical"]])
    print("outside the five classes, value tables not identical or structural problems:",
          outside_problems or "none")
    outside_vt = [(cls, vt, v["identical"]) for cls, c in d["classes"].items() if cls not in FIVE
                  for vt, v in c["values"].items()]
    print("outside the five classes, value tables (rows identical):", outside_vt)


if __name__ == "__main__":
    main()
