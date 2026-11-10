"""Orchestrator's own comparison at planned stop 3 (levin-phase-floor 07 Part B). Read only.

Usage:  python -I -B stop3_compare.py CENSUS_RUN_DIR OUT_JSON [--processes N]

For every unit of the census run's checkpoint.jsonl, 06m's record is read from 06m's checkpoint.jsonl at the offset the
census agent's index gives, accepted only if its SHA-256 equals the index's and the line names the same unit; its
region record is read from 06m's pickle at the pointer the line itself carries, accepted only if the bytes' SHA-256
equals the line's own pickle_record_sha256. The census unit's region record is checked against its own sha likewise.

A call is QUIET when, from its own guard_07 summary and its region arrays: no stationary point located, no window, no
window piece, no guarded leaf, no guard at depth max, and every returned leaf's window_07 / guarded_07 /
guard_at_depth_max_07 flag 0. Every other call is ACTIVE. Rule A's skipped phases are recorded, not a criterion.

Both records are first re-keyed so that a call's entries cannot shift positionally when another call's lists change
length: data.calls / data.other_calls by (index, notify_label); every pickle list whose entries carry `index` by that
index (grouped); the pickle's positional `scan` list by the matching group call's index; the data's `scan`,
`deep_lines`, `deep_lines_06e.lines` entries by `index` where they carry one. Then every leaf is compared (floats by
float.hex, numpy arrays by dtype, shape and bytes, other scalars by type and repr). Each differing path is classified
by stop2_compare.classify (new07, timing, provenance, cache_built) and anything else is attributed to the call whose
key it sits under, or to UNIT level.

Verdict: a unit is FAIL if any unclassified difference is attributed to a QUIET call, or if a unit with no ACTIVE call
has any unclassified difference at all (unit level included). Unit-level differences in units with an ACTIVE call are
tabulated by key, not failed (their totals sum the active calls). Exits 1 on any FAIL or any evidence failure.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stop2_compare import INDEX, RUN06M, classify  # noqa: E402


def leaves(x, p=""):
    if isinstance(x, dict):
        if not x:
            yield p, ("emptydict",)
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


def rekey_list(lst, keyf):
    out = {}
    for e in lst:
        out.setdefault(keyf(e), []).append(e)
    return out


def ckey(c):
    return f"CALL<{c['index']}>"


def rekey_data(d):
    d = dict(d)
    for name in ("calls", "other_calls"):
        if isinstance(d.get(name), list):
            d[name] = rekey_list(d[name], ckey)
    for name in ("scan", "deep_lines"):
        v = d.get(name)
        if isinstance(v, list) and all(isinstance(e, dict) and "index" in e for e in v):
            d[name] = rekey_list(v, ckey)
    dl = d.get("deep_lines_06e")
    if isinstance(dl, dict) and isinstance(dl.get("lines"), list) and all(
        isinstance(e, dict) and "index" in e for e in dl["lines"]
    ):
        dl = dict(dl)
        dl["lines"] = rekey_list(dl["lines"], ckey)
        d["deep_lines_06e"] = dl
    return d


def rekey_pickle(p, group_indices):
    p = dict(p)
    for name, v in list(p.items()):
        if name == "scan" and isinstance(v, list):
            if len(v) == len(group_indices):
                p[name] = {f"CALL<{i}>": [e] for i, e in zip(group_indices, v)}
            else:
                p[name] = {"SCAN_LEN_MISMATCH": v}
        elif isinstance(v, list) and v and all(isinstance(e, dict) and "index" in e for e in v):
            p[name] = rekey_list(v, ckey)
    return p


def attribute(path):
    m = re.search(r"CALL<(-?\d+)>", path)
    return int(m.group(1)) if m else None


def call_is_quiet(c, regs):
    g = c.get("guard_07") or {}
    if not g.get("present"):
        return False, "guard_07 absent"
    ret = g.get("returned") or {}
    quiet = (
        g.get("n_points", 0) == 0
        and g.get("n_windows", 0) == 0
        and g.get("num_window_pieces", 0) == 0
        and g.get("num_guarded_regions", 0) == 0
        and g.get("num_guard_at_depth_max", 0) == 0
        and ret.get("n_window", 0) == 0
        and ret.get("n_guarded", 0) == 0
        and ret.get("n_guard_at_depth_max", 0) == 0
    )
    for r in regs:
        for f in ("window_07", "guarded_07", "guard_at_depth_max_07"):
            a = r.get(f)
            if a is not None and np.asarray(a).size and np.any(np.asarray(a) != 0):
                quiet = False
    return quiet, None


def load_bytes(path, off, length, sha):
    with open(path, "rb") as fh:
        fh.seek(off)
        raw = fh.read(length)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise RuntimeError(f"sha mismatch {path}@{off}")
    return raw


def one(args):
    line, run_dir, ent = args
    rec = json.loads(line)
    d = rec["data"]
    uid = d["unit"]
    row = {"unit": uid, "status": d["status"]}
    raw06 = load_bytes(RUN06M / "checkpoint.jsonl", ent["offset"], ent["length"], ent["line_sha256"])
    l06 = json.loads(raw06)
    g = l06["data"]
    if g.get("unit") != uid:
        raise RuntimeError(f"06m line names {g.get('unit')} not {uid}")
    if (g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]) != tuple(ent["pickle"]):
        raise RuntimeError(f"index pickle pointer differs from 06m line's own for {uid}")
    p06 = pickle.loads(load_bytes(RUN06M / g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]))
    pm = pickle.loads(load_bytes(Path(run_dir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]))
    row["status_06m"] = g["status"]
    # call sets must match by (index, notify_label, x_span)
    def sig(dd):
        return sorted((k, c["index"], c["notify_label"], json.dumps(c["x_span"], sort_keys=True))
                      for k in ("calls", "other_calls") for c in dd.get(k) or [])
    row["calls_match"] = sig(d) == sig(g)
    # quiet / active, from census record and its region arrays
    regs_by = {}
    for name in ("regions", "other_regions"):
        for r in pm.get(name) or []:
            regs_by.setdefault(r["index"], []).append(r)
    quiet, active, skipped = set(), set(), {}
    for c in (d.get("calls") or []) + (d.get("other_calls") or []):
        q, why = call_is_quiet(c, regs_by.get(c["index"], []))
        (quiet if q else active).add(c["index"])
        sk = (c.get("guard_07") or {}).get("location_skipped_phases")
        if sk:
            skipped[c["index"]] = sk
    gi_m = [c["index"] for c in d.get("calls") or []]
    gi_6 = [c["index"] for c in g.get("calls") or []]
    dr = diff(rekey_data(g), rekey_data(d))
    pr = diff(rekey_pickle(p06, gi_6), rekey_pickle(pm, gi_m))
    cls = Counter()
    bad_quiet, unit_level, active_paths = [], [], Counter()
    for name, paths in (("data", dr), ("pickle", pr)):
        for p in paths:
            c = classify(p)
            cls[c] += 1
            if c != "UNCLASSIFIED":
                continue
            i = attribute(p)
            if i is None:
                unit_level.append(f"{name}:{p}")
            elif i in quiet:
                bad_quiet.append(f"{name}:{p}")
            else:
                active_paths[re.sub(r"\[\d+\]", "[i]", re.sub(r"CALL<-?\d+>", "CALL", f"{name}:{p}"))] += 1
    row.update(
        n_calls=len(quiet) + len(active),
        n_quiet=len(quiet),
        n_active=len(active),
        active=sorted(active),
        skipped=skipped,
        classes=dict(cls),
        n_bad_quiet=len(bad_quiet),
        bad_quiet=bad_quiet[:100],
        n_unit_level=len(unit_level),
        unit_level=unit_level[:100],
        active_path_kinds=dict(active_paths),
        quiet_calls_differing=sorted({attribute(p.split(":", 1)[1]) for p in bad_quiet}),
    )
    row["fail"] = (
        not row["calls_match"]
        or d["status"] != "done"
        or g["status"] != "done"
        or bool(bad_quiet)
        or (not active and bool(unit_level))
    )
    if not active:
        row["total_equal_hex"] = (
            json.dumps(d["result"]["total"], sort_keys=True) == json.dumps(g["result"]["total"], sort_keys=True)
            and json.dumps(d["result"]["total_abserr"], sort_keys=True)
            == json.dumps(g["result"]["total_abserr"], sort_keys=True)
        )
        row["fail"] = row["fail"] or not row["total_equal_hex"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    me = Path(__file__).read_bytes()
    idx = json.load(open(INDEX))["units"]
    jobs = []
    with open(Path(a.run_dir) / "checkpoint.jsonl", "rb") as fh:
        ck_sha = hashlib.sha256(fh.read()).hexdigest()
    seen = set()
    for line in open(Path(a.run_dir) / "checkpoint.jsonl"):
        u = json.loads(line)["unit"]
        if u in seen:
            raise SystemExit(f"duplicate unit {u}")
        seen.add(u)
        jobs.append((line, a.run_dir, idx[u]))
    with mp.Pool(a.processes) as pool:
        rows = pool.map(one, jobs, chunksize=20)
    tot = Counter()
    unit_level_keys = Counter()
    active_kinds = Counter()
    for r in rows:
        tot["units"] += 1
        tot["calls"] += r["n_calls"]
        tot["quiet_calls"] += r["n_quiet"]
        tot["active_calls"] += r["n_active"]
        tot["units_with_active"] += bool(r["n_active"])
        tot["units_fail"] += r["fail"]
        tot["quiet_calls_differing"] += len(r["quiet_calls_differing"])
        tot["calls_with_skipped_phases"] += len(r["skipped"])
        tot["calls_mismatch_units"] += not r["calls_match"]
        for p in r["unit_level"]:
            unit_level_keys[re.sub(r"\[\d+\]", "[i]", re.sub(r"CALL<-?\d+>", "CALL", p))] += 1
        for k, v in r["active_path_kinds"].items():
            active_kinds[k] += v
    summary = {
        "script_sha256": hashlib.sha256(me).hexdigest(),
        "run_dir": a.run_dir,
        "checkpoint_sha256": ck_sha,
        "totals": dict(tot),
        "unit_level_keys_in_active_units": dict(unit_level_keys.most_common()),
        "active_call_path_kinds": dict(active_kinds.most_common(80)),
        "failing_units": [r for r in rows if r["fail"]],
        "rows": rows,
    }
    Path(a.out_json).write_text(json.dumps(summary, indent=1, default=str))
    print("# stop3_compare.py sha256", summary["script_sha256"])
    print("# checkpoint sha256", ck_sha)
    print(json.dumps(summary["totals"], indent=1))
    print("unit-level keys in units with an active call (top 30):")
    for k, v in list(unit_level_keys.most_common(30)):
        print(f"  {v:6d}  {k}")
    print("failing units:", len(summary["failing_units"]))
    for r in summary["failing_units"][:20]:
        print("  ", r["unit"], "calls_match", r["calls_match"], "bad_quiet", r["bad_quiet"][:5], "unit_level", r["unit_level"][:5])
    sys.exit(1 if summary["failing_units"] else 0)


if __name__ == "__main__":
    main()
