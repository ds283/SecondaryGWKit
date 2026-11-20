"""levin-phase-floor prompt 06a section 3.1, gate C0 clauses (c), (d) and (e); and, with ``c4``,
prediction C4 after the run. New in 06a.

``run OUT_DIR``: three fresh census workers (worker.py, tree h, the producer cache), each its own
process, at most three at once, no Ray:
- W1: ``low-G+r-1`` at 1e-7 first, then the other seven rows of its (k, z_response) block at
  1e-7, then ``low-G+r-1`` at 1e-8;
- W2: the same seven rows at 1e-7, then ``low-G+r-1`` at 1e-7 last;
- W3: ``ctrl-low-1`` at 1e-7.
Every 1e-7 unit carries the scan, as in the run. Budget 300 s, as in the run.

``check OUT_DIR``:
- (c) W1's ``low-G+r-1`` at 1e-7 and at 1e-8, and W3's ``ctrl-low-1``, against 06's ``s`` units
  ``s-1e-7-low-G+r-1``, ``s-1e-8-low-G+r-1`` and ``s-1e-7-ctrl-low-1`` in run
  levin-phase-floor-06-wider-20261026T223634, bit for bit: total, total_abserr, every call's
  value and abserr (and its label, regime and group, by index), and the region arrays (06's five:
  start, end, depth, type, abserr), as bytes with their dtypes;
- (d) W1's ``low-G+r-1`` at 1e-7 (run first, in a fresh worker) against W2's (run last, after the
  block's other seven rows): every field of the census record that is a computed value (the
  checkpoint's calls without ``elapsed``, the slim result without ``compute_time``, the region
  arrays in full, the scan without its timing) bit for bit. The other seven rows, run in both W1
  and W2 at different positions, are compared the same way (descriptive);
- (e) no session directory appeared under /tmp/ray-sgk (the listing before and after), and every
  worker reported ``ray_initialized`` false.

``c4 RUN_DIR``: prediction C4 on the census run: every 1e-7 unit of 05's 39 rows, and every 1e-8
unit of its 31 depth-20 rows, against 06's ``s`` unit, as (c).

Read-only outside OUT_DIR. Usage: python -B c0.py run|check OUT_DIR ; python -B c0.py c4 RUN_DIR
(run with PYTHONPATH the tree h, since the worker is started from here; check and c4 import
nothing from the tree).
"""

import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, RUN06, TREES, WIDER, header, sha256  # noqa: E402

BUDGET_S = 300.0
ROWS_JSON = WIDER / "state" / "rows.json"
REGION_FIELDS_06 = ("start", "end", "depth", "type", "abserr")


def load_rows():
    doc = json.loads(ROWS_JSON.read_text())
    by_name05 = {}
    for r in doc["rows"]:
        for n in r["names05"] or []:
            by_name05[n] = r
    return doc, by_name05


def unit(r, rtol):
    return {
        "uid": f"h-{rtol}-{r['shard']}-{r['serial']}",
        "tree": "h",
        "rtol": rtol,
        "row": r["name"],
        "names05": r["names05"],
        "shard": r["shard"],
        "serial": r["serial"],
        "k_key": r["k_key"],
        "budget": BUDGET_S,
        "scan": rtol == "1e-7",
    }


def tmp_listing():
    try:
        return sorted(os.listdir("/tmp/ray-sgk"))
    except FileNotFoundError:
        return []


def cmd_run(out_dir):
    out_dir = Path(out_dir)
    doc, by05 = load_rows()
    R = by05["low-G+r-1"]
    block = [
        r for r in doc["rows"]
        if r["k_key"] == R["k_key"] and r["z_response_serial"] == R["z_response_serial"]
    ]
    others = [r for r in block if r["name"] != R["name"]]
    C = by05["ctrl-low-1"]
    specs = {
        "W1": [unit(R, "1e-7")] + [unit(r, "1e-7") for r in others] + [unit(R, "1e-8")],
        "W2": [unit(r, "1e-7") for r in others] + [unit(R, "1e-7")],
        "W3": [unit(C, "1e-7")],
    }
    before = tmp_listing()
    (out_dir).mkdir(parents=True, exist_ok=False)
    (out_dir / "tmp_ray_sgk_before.json").write_text(json.dumps(before))
    procs = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES["h"][0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    for name, units in specs.items():
        wdir = out_dir / name
        wdir.mkdir()
        spec = {
            "tree": "h",
            "name": name,
            "units": units,
            "out_dir": str(wdir),
            "emit_fd": None,
            "emit_file": str(wdir / "emit.jsonl"),
            "rows_sha256": sha256(ROWS_JSON),
        }
        sp = wdir / "spec.json"
        sp.write_text(json.dumps(spec, indent=1))
        log = open(wdir / "worker.log", "ab")
        procs[name] = subprocess.Popen(
            [str(PYTHON), "-B", str(HERE / "worker.py"), str(sp)],
            cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        print(f"{name}: pid {procs[name].pid}, units {[u['uid'] for u in units]}", flush=True)
    rc = {name: p.wait() for name, p in procs.items()}
    after = tmp_listing()
    (out_dir / "tmp_ray_sgk_after.json").write_text(json.dumps(after))
    print(f"exit codes {rc}")
    print(f"/tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {sorted(set(after) - set(before))}")
    return 0 if all(v == 0 for v in rc.values()) else 1


# --------------------------------------------------------------------------------------------- reading


def read_emit(path):
    units, bye = {}, None
    with open(path) as f:
        for line in f:
            m = json.loads(line)
            if m["kind"] == "unit":
                units[m["data"]["unit"]] = m["data"]
            elif m["kind"] == "bye":
                bye = m
    return units, bye


def read_ledger_records(run_dir):
    recs = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs[r["unit"]] = r["data"]
    return recs


def load_census_record(dir_, data):
    with open(Path(dir_) / data["pickle"], "rb") as f:
        f.seek(data["pickle_offset"])
        blob = f.read(data["pickle_length"])
    import hashlib

    if hashlib.sha256(blob).hexdigest() != data["pickle_record_sha256"]:
        raise RuntimeError(f"{data['unit']}: pickle record sha256 differs")
    return pickle.loads(blob)


def load_06(uid):
    recs = read_ledger_records(RUN06)
    data = recs[uid]
    with open(RUN06 / data["pickle"], "rb") as f:
        rec = pickle.load(f)
    return data, rec


def hx(d):
    return None if d is None else d["hex"]


def compare_to_06(c_data, c_rec, s_data, s_rec):
    """(c): total, total_abserr, every call's value and abserr (and label, regime, group), and
    06's five region arrays, bit for bit. Returns (holds, lines)."""
    lines = []
    ok = True

    def chk(name, a, b):
        nonlocal ok
        same = a == b
        ok &= same
        lines.append(f"     {'==' if same else '!='} {name}: {a!r} | 06: {b!r}" if not same else f"     == {name}")
        return same

    cr, sr = c_data.get("result"), s_data.get("result")
    if cr is None or sr is None:
        lines.append(f"     !! a result is missing: census {c_data['status']}, 06 {s_data['status']}")
        return False, lines
    chk("total", hx(cr["total"]), hx(sr["total"]))
    chk("total_abserr", hx(cr["total_abserr"]), hx(sr["total_abserr"]))
    chk("total_converged", cr["total_converged"], sr["total_converged"])
    chk("number of group calls", len(c_data["calls"]), len(s_data["calls"]))
    n_vals = 0
    for a, b in zip(c_data["calls"], s_data["calls"]):
        for key in ("index", "notify_label", "regime", "group"):
            if a[key] != b[key]:
                chk(f"call {a['index']} {key}", a[key], b[key])
        for key in ("value", "abserr"):
            n_vals += 1
            if hx(a[key]) != hx(b[key]):
                chk(f"call {a['index']} {key}", hx(a[key]), hx(b[key]))
    lines.append(f"     calls: {len(c_data['calls'])}; {n_vals} call values/abserrs compared")
    chk("number of region-array records", len(c_rec["regions"]), len(s_rec["regions"]))
    n_regions = 0
    for a, b in zip(c_rec["regions"], s_rec["regions"]):
        for key in ("index", "regime", "group"):
            if a[key] != b[key]:
                chk(f"regions {a['index']} {key}", a[key], b[key])
        if tuple(map(float.hex, map(float, a["x_span"]))) != tuple(map(float.hex, map(float, b["x_span"]))):
            chk(f"regions {a['index']} x_span", a["x_span"], b["x_span"])
        for f in REGION_FIELDS_06:
            x, y = a[f], b[f]
            if not (x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()):
                chk(f"regions {a['index']} {f}", (x.dtype.str, x.shape), (y.dtype.str, y.shape))
        n_regions += len(a["start"])
    lines.append(f"     regions: {n_regions} regions x {len(REGION_FIELDS_06)} arrays compared as bytes")
    return ok, lines


def _strip_calls(calls):
    return [{k: v for k, v in c.items() if k != "elapsed"} for c in calls]


def _strip_result(r):
    return None if r is None else {k: v for k, v in r.items() if k != "compute_time"}


def _strip_scan(scan):
    if scan is None:
        return None
    return [None if s is None else {k: v for k, v in s.items() if k != "seconds"} for s in scan]


def _arr_bytes(rec):
    out = []
    for a in rec["regions"]:
        d = {}
        for k, v in a.items():
            d[k] = (v.dtype.str, v.shape, v.tobytes()) if isinstance(v, np.ndarray) else v
        out.append(d)
    return out


TIMING_KEYS = ("compute_time", "elapsed")


def _drop_timings(x):
    """Every key named compute_time or elapsed removed, at any depth (timings are not values)."""
    if isinstance(x, dict):
        return {k: _drop_timings(v) for k, v in x.items() if k not in TIMING_KEYS}
    if isinstance(x, list):
        return [_drop_timings(v) for v in x]
    return x


def _full_result_values(rec):
    r = rec.get("result_full")
    if r is None:
        return None
    return _drop_timings(r)


def compare_census(a_data, a_rec, b_data, b_rec):
    """(d): every computed value of two census units. Returns (holds, differing keys)."""
    diffs = []
    if _strip_calls(a_data["calls"]) != _strip_calls(b_data["calls"]):
        diffs.append("calls")
    if _strip_result(a_data.get("result")) != _strip_result(b_data.get("result")):
        diffs.append("result")
    if a_data.get("GkPolicy") != b_data.get("GkPolicy"):
        diffs.append("GkPolicy")
    if (a_data.get("scan") is None) != (b_data.get("scan") is None) or json.dumps(a_data.get("scan")) != json.dumps(b_data.get("scan")):
        diffs.append("scan (checkpoint)")
    if _arr_bytes(a_rec) != _arr_bytes(b_rec):
        diffs.append("region arrays")
    if pickle.dumps(_strip_scan(a_rec["scan"])) != pickle.dumps(_strip_scan(b_rec["scan"])):
        diffs.append("scan (pickle)")
    if json.dumps(_full_result_values(a_rec), sort_keys=True) != json.dumps(_full_result_values(b_rec), sort_keys=True):
        diffs.append("full result summary")
    return len(diffs) == 0, diffs


def cmd_check(out_dir):
    out_dir = Path(out_dir)
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: gate C0 clauses (c)-(e)",
                    [ROWS_JSON, RUN06 / "checkpoint.jsonl"]):
        print(l)
    doc, by05 = load_rows()
    W = {}
    byes = {}
    for name in ("W1", "W2", "W3"):
        W[name], byes[name] = read_emit(out_dir / name / "emit.jsonl")
    R = by05["low-G+r-1"]
    C = by05["ctrl-low-1"]
    uidR7 = f"h-1e-7-{R['shard']}-{R['serial']}"
    uidR8 = f"h-1e-8-{R['shard']}-{R['serial']}"
    uidC7 = f"h-1e-7-{C['shard']}-{C['serial']}"
    all_ok = True
    print("\n== (c) the census path is 06's path: census units against 06's s units, bit for bit")
    for w, uid, s_uid in (("W1", uidR7, "s-1e-7-low-G+r-1"), ("W1", uidR8, "s-1e-8-low-G+r-1"),
                          ("W3", uidC7, "s-1e-7-ctrl-low-1")):
        cd = W[w][uid]
        crec = load_census_record(out_dir / w, cd)
        sd, srec = load_06(s_uid)
        ok, lines = compare_to_06(cd, crec, sd, srec)
        all_ok &= ok
        print(f"   {w} {uid} vs 06 {s_uid}: {'IDENTICAL' if ok else 'DIFFERS'} "
              f"(census status {cd['status']}, total {cd['result']['total']['repr'] if cd.get('result') else None})")
        for l in lines:
            print(l)
    c_ok = all_ok
    print(f"   (c) {'holds' if c_ok else 'FAILS'}")

    print("\n== (d) order does not matter: low-G+r-1 at 1e-7 first in a fresh worker (W1) and last after "
          "its block's other rows (W2)")
    pos1 = [u for u in W["W1"]].index(uidR7)
    pos2 = [u for u in W["W2"]].index(uidR7)
    a, b = W["W1"][uidR7], W["W2"][uidR7]
    ok, diffs = compare_census(a, load_census_record(out_dir / "W1", a), b,
                               load_census_record(out_dir / "W2", b))
    print(f"   position in W1 {pos1 + 1} of {len(W['W1'])}; in W2 {pos2 + 1} of {len(W['W2'])}")
    print(f"   W1 cache_built {a['cache_built']}")
    print(f"   W2 cache_built {b['cache_built']}")
    print(f"   {uidR7}: {'IDENTICAL in every computed value' if ok else 'DIFFERS in ' + str(diffs)}")
    d_ok = ok
    print("   descriptive, the block's other rows, run in both workers at different positions:")
    for uid in W["W2"]:
        if uid == uidR7 or uid not in W["W1"]:
            continue
        a, b = W["W1"][uid], W["W2"][uid]
        o, df = compare_census(a, load_census_record(out_dir / "W1", a), b,
                               load_census_record(out_dir / "W2", b))
        print(f"     {uid}: {'identical' if o else 'DIFFERS ' + str(df)} "
              f"(positions {list(W['W1']).index(uid) + 1} and {list(W['W2']).index(uid) + 1})")
        d_ok &= o
    print(f"   (d) {'holds' if d_ok else 'FAILS'}")

    print("\n== (e) no Ray")
    before = json.loads((out_dir / "tmp_ray_sgk_before.json").read_text())
    after = json.loads((out_dir / "tmp_ray_sgk_after.json").read_text())
    new = sorted(set(after) - set(before))
    ray_flags = {n: (None if byes[n] is None else byes[n]["ray_initialized"]) for n in byes}
    e_ok = len(new) == 0 and all(v is False for v in ray_flags.values())
    print(f"   /tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {new}")
    print(f"   workers' ray_initialized at exit: {ray_flags}")
    statuses = {n: sorted({d['status'] for d in W[n].values()}) for n in W}
    print(f"   unit statuses by worker: {statuses}")
    print(f"   (e) {'holds' if e_ok else 'FAILS'}")
    ok = c_ok and d_ok and e_ok
    print(f"\n== C0 (c), (d), (e): {'HOLD' if ok else 'DO NOT HOLD'}")
    return 0 if ok else 1


def cmd_c4(run_dir):
    run_dir = Path(run_dir)
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: prediction C4",
                    [ROWS_JSON, RUN06 / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]):
        print(l)
    doc, by05 = load_rows()
    recs = read_ledger_records(run_dir)
    n = 0
    n_ok = 0
    fails = []
    for r in doc["rows"]:
        if not r["names05"]:
            continue
        name05 = r["names05"][0]
        for rtol in ("1e-7", "1e-8"):
            if rtol == "1e-8" and r["role05"] != "d20":
                continue
            uid = f"h-{rtol}-{r['shard']}-{r['serial']}"
            s_uid = f"s-{rtol}-{name05}"
            n += 1
            cd = recs.get(uid)
            if cd is None or cd.get("pickle") is None:
                fails.append((uid, s_uid, "census unit missing or without a record"))
                print(f"   {uid} ({name05}) vs {s_uid}: MISSING")
                continue
            crec = load_census_record(run_dir, cd)
            sd, srec = load_06(s_uid)
            ok, lines = compare_to_06(cd, crec, sd, srec)
            n_ok += ok
            print(f"   {uid} ({','.join(r['names05'])}) vs {s_uid}: {'IDENTICAL' if ok else 'DIFFERS'}")
            if not ok:
                fails.append((uid, s_uid, [l for l in lines if '!=' in l]))
                for l in lines:
                    print(l)
    print(f"\n== C4: {n_ok} of {n} units bit-identical to 06's s units; "
          f"{'HOLDS' if n_ok == n else 'FAILS'}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    sys.exit({"run": cmd_run, "check": cmd_check, "c4": cmd_c4}[cmd](arg))
