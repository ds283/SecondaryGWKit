"""Read-only look at store_value_diff's JSON (levin-phase-floor 06 §5.4): for each class, the
value tables' counts and columns present in one store only, and one example differing row of each
table that differs, truncated. Reads only the JSON.

    ./venv/bin/python -I peek_structure.py <store_value_diff.json>
"""
import json
import sys

d = json.load(open(sys.argv[1]))
print("a:", d["a"]); print("b:", d["b"])
print("top-level keys:", sorted(d))
for cls, c in d["classes"].items():
    tabs = [("rows", c["rows"])] + list(c["values"].items())
    for name, t in tabs:
        if t["differ"] or t["columns_only_in_a"] or t["columns_only_in_b"] or t["only_in_a"] or t["only_in_b"] or t["unmatched"]:
            print(f"{cls}/{name}: identical {t['identical']} differ {t['differ']} onlyA {len(t['only_in_a'])} "
                  f"onlyB {len(t['only_in_b'])} unmatched {len(t['unmatched'])} "
                  f"cols_only_a {t['columns_only_in_a']} cols_only_b {t['columns_only_in_b']}")
            print("   column summaries:", json.dumps(t["columns"])[:1500])
            if t["rows"]:
                print("   example row:", json.dumps(t["rows"][0])[:1500])
