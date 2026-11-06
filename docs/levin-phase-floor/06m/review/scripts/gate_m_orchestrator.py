"""Orchestrator's own comparison of census unit records at m against route B's census (06m §5).

Usage:
  python -B gate_m_orchestrator.py units  ROUTEB_RUN_DIR OUT_DIR   OUT_JSON   # run_k0.py output dir
  python -B gate_m_orchestrator.py census ROUTEB_RUN_DIR M_RUN_DIR OUT_JSON   # a census run dir

Written independently of the census agent's compare_06m.py. For every unit at m, route B's record is
found by unit id (one pass over route B's checkpoint building byte offsets), and both the unit's
emitted record (`data`) and its region-pickle record (bytes at pickle_offset/pickle_length, SHA-256
checked against pickle_record_sha256) are compared recursively in full:
  - floats (and numpy floating scalars) by float.hex (NaN by repr); numpy arrays by dtype, shape and
    bytes; other numpy scalars by dtype and bytes; dicts key by key; lists and tuples position by
    position (so calls and regions must match in order, kind, index, label and span); the rest by
    type and equality.
  - Excluded, each counted apart (present / differing): TIMING (keys named below, and every key
    ending `_s`), INSTRUMENT (route_06k, route_06l), PROVENANCE (tree, worker, pickle-pointer, module
    and script records, and cache_built, which records where a worker's producer cache was built and
    so depends on which units share a worker).
Also checks, per unit at m, that `modules` names the code worktree's levin_quadrature.py and no
variant. Exits 1 if any unit is missing, not done, or differs outside the exclusions.
"""

import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TIMING = {"elapsed", "compute_time", "seconds"}
INSTRUMENT = {"route_06k", "route_06l"}
PROVENANCE = {
    "tree_head",
    "tree_status_short",
    "scripts_sha256",
    "worker",
    "worker_pid",
    "worker_tree",
    "pickle",
    "pickle_offset",
    "pickle_length",
    "pickle_record_sha256",
    "modules",
    "rows_sha256",
    "w2_capture_sha256",
    "cache_built",
}
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-38f6a82"


def kind_of(k):
    if not isinstance(k, str):
        return None
    if k in TIMING or k.endswith("_s"):
        return "timing"
    if k in INSTRUMENT:
        return "instrument"
    if k in PROVENANCE:
        return "provenance"
    return None


def fk(x):
    return repr(x) if x != x else float(x).hex()


def cmp(a, b, path, diffs, excl, exk):
    """Record differences; exk is the exclusion class inherited from an ancestor key, or None."""

    def note(why):
        if exk:
            excl[(exk, path)] += 1
        else:
            diffs.append((path, why))

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.tobytes() == b.tobytes()
        ):
            note("array")
        return
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if type(a) is not type(b) or fk(a) != fk(b):
            note(f"{a!r} != {b!r}")
        return
    if isinstance(a, np.generic) or isinstance(b, np.generic):
        if type(a) is not type(b) or np.asarray(a).tobytes() != np.asarray(b).tobytes():
            note(f"{a!r} != {b!r}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            kk = exk or kind_of(k)
            p = f"{path}.{k}"
            if k not in a or k not in b:
                if kk:
                    excl[(kk, p)] += 1
                else:
                    diffs.append((p, "key in one only"))
                continue
            if kk and not exk:
                excl[(kk, "present:" + generic(p))] += 0
            cmp(a[k], b[k], p, diffs, excl, kk)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            note(f"sequence {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, f"{path}[{i}]", diffs, excl, exk)
        return
    if type(a) is not type(b) or a != b:
        note(f"{str(a)[:60]!r} != {str(b)[:60]!r}")


def generic(p):
    import re

    return re.sub(r"\[\d+\]", "[]", p)


def uid_of(line):
    i = line.find('"unit": "')
    j = line.find('"', i + 9)
    return line[i + 9 : j]


def routeb_index(run_dir):
    idx = {}
    with open(run_dir / "checkpoint.jsonl", "rb") as f:
        off = 0
        for line in f:
            u = uid_of(line[:200].decode())
            assert u not in idx, u
            idx[u] = (off, len(line))
            off += len(line)
    return idx


def load_record(pkl_path, data):
    with open(pkl_path, "rb") as f:
        f.seek(data["pickle_offset"])
        raw = f.read(data["pickle_length"])
    assert hashlib.sha256(raw).hexdigest() == data["pickle_record_sha256"], (pkl_path, data["unit"])
    return pickle.loads(raw)


def m_units(mode, m_dir):
    if mode == "units":
        for line in open(m_dir / "units.jsonl"):
            e = json.loads(line)
            yield e["unit"], e["data"], m_dir / e["worker"] / e["data"]["pickle"]
    else:
        for line in open(m_dir / "checkpoint.jsonl"):
            d = json.loads(line)["data"]
            yield d["unit"], d, m_dir / d["pickle"]


def module_evidence(d):
    mods = json.dumps(d.get("modules"))
    return (f"{TREE}/AdaptiveLevin/levin_quadrature.py" in mods) and ("variant" not in mods.lower())


def main():
    mode, rb_dir, m_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    idx = routeb_index(rb_dir)
    rbf = open(rb_dir / "checkpoint.jsonl", "rb")
    n_units = n_calls = n_regions_recs = n_diff_units = 0
    not_done, missing, bad_evidence = [], [], []
    first = []
    diff_paths = Counter()
    excl = Counter()
    for uid, d, pkl in m_units(mode, m_dir):
        n_units += 1
        if d.get("status") != "done":
            not_done.append(uid)
        if not module_evidence(d):
            bad_evidence.append(uid)
        if uid not in idx:
            missing.append(uid)
            continue
        off, ln = idx[uid]
        rbf.seek(off)
        rb = json.loads(rbf.read(ln))["data"]
        diffs = []
        cmp(rb, d, "data", diffs, excl, None)
        rec_b = load_record(rb_dir / rb["pickle"], rb)
        rec_m = load_record(pkl, d)
        cmp(rec_b, rec_m, "pickle", diffs, excl, None)
        n_calls += len(d.get("calls") or []) + len(d.get("other_calls") or [])
        n_regions_recs += 1
        if diffs:
            n_diff_units += 1
            for p, why in diffs:
                diff_paths[generic(p)] += 1
            if len(first) < 20:
                first.append({"unit": uid, "diffs": diffs[:20]})
    exs = Counter()
    for (k, p), n in excl.items():
        exs[(k, generic(p.replace("present:", "")) if p.startswith("present:") else generic(p))] += n
    res = {
        "script_sha256": me,
        "mode": mode,
        "routeB": str(rb_dir),
        "m": str(m_dir),
        "units_compared": n_units,
        "calls_compared": n_calls,
        "unit_pickle_records_compared": n_regions_recs,
        "units_differing": n_diff_units,
        "units_not_done": not_done,
        "units_missing_in_routeB": missing,
        "units_without_tree_module_evidence": bad_evidence,
        "differing_paths": dict(diff_paths),
        "first_differing_units": first,
        "excluded_differences_by_class_and_path": {f"{k} {p}": n for (k, p), n in sorted(exs.items()) if n},
    }
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"# gate_m_orchestrator.py sha256 {me}")
    for k in (
        "units_compared",
        "calls_compared",
        "unit_pickle_records_compared",
        "units_differing",
    ):
        print(k, res[k])
    print("not done", len(not_done), "missing", len(missing), "no tree-module evidence", len(bad_evidence))
    print("differing paths (not excluded):", dict(diff_paths.most_common(20)))
    print("excluded differences by class:", dict(Counter(k for (k, p), n in exs.items() if n for _ in [0])))
    for (k, p), n in sorted(exs.items()):
        if n:
            print(f"  excluded {k:10s} {n:8d}  {p}")
    ok = not (n_diff_units or not_done or missing or bad_evidence)
    print("VERDICT", "ALL EQUAL" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
