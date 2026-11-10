"""Classify every difference store_value_diff reports outside QuadSourceIntegral, R9 against R10.

Reads only the tool's JSON. Each differing column of each non-QuadSourceIntegral class is put in
one of these classes, and anything that fits none is printed as UNCLASSIFIED:

- `timing`   a wall-clock column: its name ends in `compute_time` or `RHS_time`;
- `label`    the `label` column, and the two labels are equal once their trailing
             `-YYYY-MM-DDTHH:MM:SS` (main.py's `datetime.now()`) is removed;
- `parents_digest`  GkSource's key column, the SHA-256 of its parents' *serials*
             (`ObjectFactories/GkSource.py`, `GkSource_parents_digest`);
- `rho_reuse` GkWKBIntegration's `metadata` with only the keys `rho_evals` and `rho_reused`
             differing, and `stage_1_RHS_evaluations` on the same rows (the per-process residual
             memo of `ComputeTargets.phase_residual`; GkSTkRemedial prompt 14).

It also checks that every value table of every non-QuadSourceIntegral class is identical, and that
no class has a row in one store only or a row that cannot be matched.

    ./venv/bin/python -I classify_non_qsi.py <store_value_diff.json>
"""

import json
import re
import sys
from collections import Counter, defaultdict

STAMP = re.compile(r"-\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")


def classify(cls, col, entry, row_cols):
    if col.endswith("compute_time") or col.endswith("RHS_time"):
        return "timing"
    if col == "label":
        a, b = entry["a"], entry["b"]
        return "label" if STAMP.sub("", a) == STAMP.sub("", b) and a != b else None
    if cls == "GkSource" and col == "parents_digest":
        return "parents_digest"
    if cls == "GkWKBIntegration" and col == "metadata":
        return "rho_reuse" if set(entry["keys"]) <= {"rho_evals", "rho_reused"} else None
    if cls == "GkWKBIntegration" and col == "stage_1_RHS_evaluations":
        return "rho_reuse" if "metadata" in row_cols else None
    return None


def main():
    report = json.load(open(sys.argv[1]))
    bad = 0
    print(f"not_compared: {report['not_compared']}")
    for cls, c in report["classes"].items():
        if cls == "QuadSourceIntegral":
            continue
        t = c["rows"]
        structural = (len(t["only_in_a"]), len(t["only_in_b"]), len(t["unmatched"]),
                      len(c["unresolvable_columns"]))
        vt_bad = {name: (v["differ"], len(v["only_in_a"]), len(v["only_in_b"]), len(v["unmatched"]))
                  for name, v in c["values"].items()
                  if v["differ"] or v["only_in_a"] or v["only_in_b"] or v["unmatched"]}
        counts = defaultdict(Counter)
        for row in t["rows"]:
            for col, entry in row["columns"].items():
                kind = classify(cls, col, entry, row["columns"])
                counts[col][kind or "UNCLASSIFIED"] += 1
                if kind is None:
                    bad += 1
        vt_rows = sum(v["identical"] for v in c["values"].values())
        print(f"{cls}: rows identical {t['identical']}, differ {t['differ']}; "
              f"only-A/only-B/unmatched/unresolvable {structural}; "
              f"value-table rows identical {vt_rows}; value tables not identical {vt_bad or 'none'}")
        for col, cnt in sorted(counts.items()):
            print(f"    {col}: {dict(cnt)}")
        if any(structural) or vt_bad:
            bad += 1
    print()
    print(f"UNCLASSIFIED differences or structural problems outside QuadSourceIntegral: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
