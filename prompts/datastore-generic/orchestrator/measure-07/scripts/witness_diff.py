"""Compare two schema witnesses: is the only difference the record keys added?

usage: python witness_diff.py <old.json> <new.json>
"""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if k != "classes" and a.get(k) != b.get(k):
        print("top-level differs:", k)
assert list(a["classes"]) == list(b["classes"])
added = {}
for cls in a["classes"]:
    ca, cb = a["classes"][cls], b["classes"][cls]
    if ca["table"] != cb["table"]:
        print("TABLE DIFFERS:", cls)
    if ca["has_table"] != cb["has_table"]:
        print("has_table differs:", cls)
    ra, rb = ca["record"], cb["record"]
    for key in sorted(set(ra) | set(rb)):
        if key not in rb:
            print("record key removed:", cls, key)
        elif key not in ra:
            added.setdefault(key, {})[cls] = rb[key]
        elif ra[key] != rb[key]:
            print("record key changed:", cls, key)
for key, per in added.items():
    nondefault = {c: v for c, v in per.items() if v not in (None, [])}
    print(f"added record key {key!r} on {len(per)} classes; non-default: {nondefault}")
