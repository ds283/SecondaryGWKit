"""Orchestrator's read of a store_value_diff JSON: every class's value tables, rows in one store only,
and (with --labels A B) whether every label difference is only the job-name prefix A -> B plus stamp."""
import json, re, sys, hashlib
p = sys.argv[1]
print(f"# vdiff_values.py sha256 {hashlib.sha256(open(__file__,'rb').read()).hexdigest()}; json {p} sha256 {hashlib.sha256(open(p,'rb').read()).hexdigest()}")
d = json.load(open(p))
for cls, c in sorted(d["classes"].items()):
    r = c["rows"]
    one = sum(len(r.get(x) or []) for x in ("only_in_a", "only_in_b", "unmatched"))
    vals = c.get("values") or {}
    bad = [t for t, v in vals.items() if not v.get("identical", False)]
    nrows = sum((v.get("rows") or {}).get("identical", 0) if isinstance(v.get("rows"), dict) else 0 for v in vals.values())
    print(f"{cls}: value tables {len(vals)} not identical {bad}; one-store/unmatched rows {one}; unresolvable {len(c.get('unresolvable_columns') or [])}; cols-only {r.get('columns_only_in_a')}{r.get('columns_only_in_b')}")
if len(sys.argv) > 3 and sys.argv[2] == "--labels":
    a, b = sys.argv[3], sys.argv[4]
    n = bad = 0
    stamp = re.compile(r"\d{8}T?\d{6}|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?")
    for cls, c in d["classes"].items():
        for row in c["rows"].get("rows") or []:
            col = (row.get("columns") or {}).get("label")
            if col is None:
                continue
            n += 1
            la, lb = str(col.get("a")), str(col.get("b"))
            if stamp.sub("S", la.replace(a, b, 1)) != stamp.sub("S", lb):
                bad += 1
                if bad <= 5:
                    print("LABEL NOT PREFIX-ONLY", cls, la, lb)
    print(f"label differences {n}; not prefix-and-stamp only {bad}")
