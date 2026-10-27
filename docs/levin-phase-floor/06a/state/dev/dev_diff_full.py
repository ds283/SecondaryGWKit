# dev only: which keys of the full result summary differ between W1 and W2 (C0 (d) diagnosis)
import json, sys
sys.path.insert(0, "scripts")
import c0
W1, _ = c0.read_emit("state/C0/W1/emit.jsonl"); W2, _ = c0.read_emit("state/C0/W2/emit.jsonl")
uid = "h-1e-7-2-38302"
a = c0.load_census_record("state/C0/W1", W1[uid])["result_full"]
b = c0.load_census_record("state/C0/W2", W2[uid])["result_full"]
def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            walk(x.get(k), y.get(k), path + [k])
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, path + [i])
    elif x != y:
        print("/".join(map(str, path)), repr(x)[:80], repr(y)[:80])
walk(a, b, [])
