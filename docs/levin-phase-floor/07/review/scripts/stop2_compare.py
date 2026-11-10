"""Orchestrator's own comparison at planned stop 2 (levin-phase-floor 07 Part B).

Usage:
  python -B stop2_compare.py vs06m RUN_DIR      # each unit of RUN_DIR/units.jsonl against 06m's census
  python -B stop2_compare.py det RUN_DIR1 RUN_DIR2

06m's record for a unit is read from 06m's checkpoint.jsonl at the offset the census agent's index
gives; the line is accepted only if its SHA-256 equals the index's AND the parsed line names the same
unit (so the index is checked, not trusted). Its region record is read from 06m's pickle at the
pointer the line itself carries, accepted only if the bytes' SHA-256 equals the line's own
`pickle_record_sha256`. Every leaf of the emitted record and of the unpickled region record is
compared: floats by float.hex, arrays by dtype, shape and bytes. Every differing path is printed,
grouped by its last key, so the reader sees what differs and can judge whether it is timing,
provenance or a new 07 key.
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

RUN06M = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06m-census-20261106T102153")
INDEX = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/index_06m.json")


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, (list, tuple)):
        if not x:
            yield p, ("emptyseq", type(x).__name__)
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(x, np.ndarray):
        yield p, ("nd", str(x.dtype), x.shape, x.tobytes())
    elif isinstance(x, float):
        yield p, ("f", x.hex())
    elif isinstance(x, np.floating):
        yield p, ("f", float(x).hex(), str(type(x)))
    else:
        yield p, (type(x).__name__, repr(x))


def diff(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    return sorted(k for k in set(la) | set(lb) if la.get(k, "<absent>") != lb.get(k, "<absent>"))


def lastkey(p):
    import re
    return re.sub(r"\[\d+\]", "[i]", p).split(".")[-1]


def read_units(run_dir):
    out = {}
    for line in open(Path(run_dir) / "units.jsonl"):
        r = json.loads(line)
        out[r["unit"]] = r
    return out


def region(run_dir, worker, data):
    pk = Path(run_dir) / worker / data["pickle"] if (Path(run_dir) / worker).is_dir() else None
    if pk is None or not pk.exists():
        cands = list(Path(run_dir).rglob(data["pickle"]))
        assert len(cands) == 1, (data["pickle"], cands)
        pk = cands[0]
    with open(pk, "rb") as fh:
        fh.seek(data["pickle_offset"])
        raw = fh.read(data["pickle_length"])
    assert hashlib.sha256(raw).hexdigest() == data["pickle_record_sha256"], "pickle record sha mismatch"
    return pickle.loads(raw)


def summarise(paths):
    from collections import Counter
    c = Counter(lastkey(p) for p in paths)
    return dict(sorted(c.items(), key=lambda t: -t[1]))


NEW07 = ("guard_07", "scan_07", "scan_07_error", "window_07", "guarded_07", "guard_at_depth_max_07",
         "stationary_points", "stationary_windows", "unlocated_minima", "num_window_pieces",
         "num_guarded_regions", "num_guard_at_depth_max", "location_skipped_phases", "window_pieces",
         "guarded_regions", "guard_at_depth_max")
PROV = ("unit", "tree", "tree_head", "tree_status_short", "worker", "worker_pid", "pickle", "pickle_offset",
        "pickle_length", "pickle_record_sha256", "scripts_sha256", "modules", "py")


def classify(p):
    import re
    parts = re.split(r"[.\[\]]+", p)
    if any(x in NEW07 for x in parts):
        return "new07"
    if any(x.endswith("_s") or x in ("seconds", "elapsed", "compute_time") for x in parts):
        return "timing"
    if parts[0] in PROV or any(x in ("scripts_sha256", "modules") for x in parts):
        return "provenance"
    if parts[0] == "cache_built":
        return "cache_built"
    return "UNCLASSIFIED"


def vs06m(run_dir):
    idx = json.load(open(INDEX))["units"]
    mine = read_units(run_dir)
    with open(RUN06M / "checkpoint.jsonl", "rb") as ck:
        for unit, r in sorted(mine.items()):
            uid = "g-" + unit.split("-", 1)[1]
            e = idx[uid]
            ck.seek(e["offset"])
            raw = ck.read(e["length"])
            assert hashlib.sha256(raw).hexdigest() == e["line_sha256"], "06m line sha"
            line = json.loads(raw)
            data06 = line.get("data", line)
            assert (line.get("unit") or data06.get("unit")) == uid, ("06m line names", line.get("unit"), uid)
            pfile, poff, plen, psha = e["pickle"]
            assert (data06["pickle"], data06["pickle_offset"], data06["pickle_length"], data06["pickle_record_sha256"]) == (pfile, poff, plen, psha), "index pickle pointer != line's own"
            with open(RUN06M / pfile, "rb") as fh:
                fh.seek(poff)
                praw = fh.read(plen)
            assert hashlib.sha256(praw).hexdigest() == data06["pickle_record_sha256"], "06m pickle sha"
            reg06 = pickle.loads(praw)
            reg = region(run_dir, r["worker"], r["data"])
            d_rec = diff(data06, r["data"])
            d_reg = diff(reg06, reg)
            print(f"== {unit} vs 06m {uid}: record leaves differing {len(d_rec)}; region leaves differing {len(d_reg)}")
            print("   record, by last key:", summarise(d_rec))
            print("   region, by last key:", summarise(d_reg))
            for name, paths in (("record", d_rec), ("region", d_reg)):
                cls = {}
                for p in paths:
                    cls.setdefault(classify(p), []).append(p)
                print(f"   {name}: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(cls.items())))
                for p in cls.get("UNCLASSIFIED", []):
                    print(f"   {name.upper()} UNCLASSIFIED PATH", p)
                for p in cls.get("provenance", [])[:0]:
                    pass


def det(r1, r2):
    a, b = read_units(r1), read_units(r2)
    assert set(a) == set(b)
    for unit in sorted(a):
        d_rec = diff(a[unit]["data"], b[unit]["data"])
        d_reg = diff(region(r1, a[unit]["worker"], a[unit]["data"]), region(r2, b[unit]["worker"], b[unit]["data"]))
        print(f"== {unit}: record leaves differing {len(d_rec)}; region leaves differing {len(d_reg)}")
        print("   record, by last key:", summarise(d_rec))
        print("   region, by last key:", summarise(d_reg))
        g = [c.get("guard_07", {}).get("n_points_by_kind") for c in a[unit]["data"]["calls"]]
        print("   located per group call:", g)


if __name__ == "__main__":
    print("# stop2_compare.py sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    {"vs06m": vs06m, "det": det}[sys.argv[1]](*sys.argv[2:])
