"""Orchestrator's own reader, 07a Part C stop 2. Independent of the census agent's compare scripts.

mode p  A_UNITS_JSONL CK07_LINES  : the orchestrator's parent re-runs (made in h with the parent's module by the agent's
                                    run_k0.py) against 07's census checkpoint lines for the same rows, unit record and
                                    region record (pickle at its offset, checked against pickle_record_sha256).
mode h  RUN1_UNITS_JSONL RUN2_UNITS_JSONL : two runs of one unit at h, unit and region record.
Every leaf is flattened (floats as float.hex); differing paths are printed grouped by their key names with indices
dropped, so each kind can be read off and classified by hand. Nothing is written but stdout.
"""

import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fl(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items():
            fl(v, path + (str(k),), out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            fl(v, path + (i,), out)
    else:
        if isinstance(x, float):
            out[path] = x.hex()
        elif hasattr(x, "tolist"):
            fl(x.tolist(), path, out)
        else:
            out[path] = repr(x)


def region(rec, base):
    p = Path(rec["pickle"])
    if not p.is_absolute():
        p = base / p if (base / p).exists() else base / rec["worker"] / p
    with open(p, "rb") as f:
        f.seek(rec["pickle_offset"])
        b = f.read(rec["pickle_length"])
    assert hashlib.sha256(b).hexdigest() == rec["pickle_record_sha256"], ("pickle sha", p)
    return pickle.loads(b)


def shape(path):
    return "/".join(s for s in path if isinstance(s, str))


def compare(a, b, label):
    fa, fb = {}, {}
    fl(a, (), fa)
    fl(b, (), fb)
    diff = [p for p in set(fa) | set(fb) if fa.get(p) != fb.get(p)]
    groups = Counter(shape(p) for p in diff)
    only_a = Counter(shape(p) for p in diff if p not in fb)
    only_b = Counter(shape(p) for p in diff if p not in fa)
    print(f"== {label}: {len(fa)} / {len(fb)} leaves, {len(diff)} differ")
    for g, n in sorted(groups.items()):
        print(f"   {n:6d}  {g}   (only first {only_a.get(g,0)}, only second {only_b.get(g,0)})")
    return groups


def find_pickle_base(units_jsonl):
    return Path(units_jsonl).parent


def main():
    mode = sys.argv[1]
    print("script", sha(__file__))
    for p in sys.argv[2:]:
        print("input", sha(p), p)
    if mode == "p":
        mine = {json.loads(l)["data"]["row"] + "@" + json.loads(l)["data"]["rtol"]: json.loads(l) for l in open(sys.argv[2])}
        theirs = {json.loads(l)["data"]["row"] + "@" + json.loads(l)["data"]["rtol"]: json.loads(l) for l in open(sys.argv[3])}
        assert set(mine) == set(theirs), (set(mine), set(theirs))
        run07 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-census-20261108T171137")
        for k in sorted(mine):
            a, b = mine[k]["data"], theirs[k]["data"]
            print(f"\n##### row {k}: mine {a['unit']} status {a['status']}, 07's {b['unit']} status {b['status']}")
            compare(a, b, "unit record (mine first, 07 second)")
            ra = region(a, find_pickle_base(sys.argv[2]))
            rb = region(b, run07)
            compare(ra, rb, "region record (mine first, 07 second)")
    else:
        r1 = [json.loads(l) for l in open(sys.argv[2])]
        r2 = [json.loads(l) for l in open(sys.argv[3])]
        assert len(r1) == len(r2) == 1
        a, b = r1[0]["data"], r2[0]["data"]
        print(f"unit {a['unit']} / {b['unit']}, status {a['status']} / {b['status']}")
        compare(a, b, "unit record (run1, run2)")
        compare(region(a, find_pickle_base(sys.argv[2])), region(b, find_pickle_base(sys.argv[3])), "region record (run1, run2)")


if __name__ == "__main__":
    main()
