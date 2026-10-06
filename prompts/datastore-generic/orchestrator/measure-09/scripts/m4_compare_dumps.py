"""M4(a): compare two dumps from m4_sqlite_dump.py, file by file. Usage: python m4_compare_dumps.py a.json b.json"""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
print("files", len(a), len(b), "same names:", sorted(a) == sorted(b))
nt = nr = 0; diffs = []
for f in sorted(set(a) | set(b)):
    if f not in a or f not in b:
        diffs.append(f"{f}: only in {'a' if f in a else 'b'}"); continue
    if a[f]["master"] != b[f]["master"]:
        sa, sb = set(map(tuple, a[f]["master"])), set(map(tuple, b[f]["master"]))
        diffs.append(f"{f}: sqlite_master differs: -{sorted(sa - sb)} +{sorted(sb - sa)}")
    for t in sorted(set(a[f]["tables"]) | set(b[f]["tables"])):
        ta, tb = a[f]["tables"].get(t), b[f]["tables"].get(t)
        if ta != tb:
            diffs.append(f"{f}:{t}: rows/columns differ ({ta and len(ta['rows'])} vs {tb and len(tb['rows'])})")
        else:
            nt += 1; nr += len(ta["rows"])
print("identical tables", nt, "identical rows", nr, "differences", len(diffs))
for d in diffs: print("  ", d)
