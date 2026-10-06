"""M4(c): compare the open records of two trees, with the store directory and timestamps in the
printed lines normalised. Usage: python m4c_compare.py a.json b.json"""
import json, re, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def norm(lines):
    return [re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "<time>", re.sub(r"/stores/c_[^/]+/", "/stores/<copy>/", l)) for l in lines]
for k in sorted(set(a) | set(b)):
    if k in ("tree",): continue
    va, vb = (norm(x[k]) if k == "printed" else x.get(k) for x in (a, b))
    print(k, "SAME" if va == vb else "DIFFERS")
