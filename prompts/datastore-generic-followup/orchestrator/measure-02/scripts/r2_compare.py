"""Compare two r2_resolve.py outputs: every differing entry, grouped. Usage: r2_compare.py a.json b.json"""

import json, sys
from collections import Counter

p, r = (json.load(open(f)) for f in sys.argv[1:3])
diffs = []


def walk(a, b, path):
    if type(a) != type(b) or not isinstance(a, (dict, list)):
        if a != b:
            diffs.append((path, a, b))
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path, "keys", sorted(set(a) ^ set(b))))
            return
        for k in a:
            walk(a[k], b[k], path + [k])
    else:
        if len(a) != len(b):
            diffs.append((path, "len", (len(a), len(b))))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i])


walk(p, r, [])
print(len(diffs), "entries differ")
for k, n in sorted(
    Counter(
        (d[0][0], d[0][1], d[0][-1], json.dumps(d[1]), json.dumps(d[2])) for d in diffs
    ).items()
):
    print(n, k)
