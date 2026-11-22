"""levin-phase-floor prompt 06b section 5, Gate W0's second and third clauses (and the import list
its first clause reads). New in 06b; adapted from 06a's c0.py (its ``run`` and ``check``).

``run OUT_DIR``: five fresh census workers (worker.py, the producer cache), each its own process,
all at once (five, within the six-worker rule), no Ray:
- at the parent ``p`` (``da5c24d``), one worker per unit: P1 ``low-G+r-1`` at 1e-7, P2
  ``ctrl-low-1`` at 1e-7, P3 06a's ``0-1971`` at 1e-8;
- at ``b`` (``92bc62f``), B1 and B2: the same unit twice, ``b-1e-7-1-34599``, in two fresh
  workers. The row is fixed by rule: of the 1e-7 units in state/w2_capture.json (rows with a
  call phase-limited in 06a), the one with the most captured calls, ties broken by rows.json's
  order.
Every 1e-7 unit carries the scan and W3's evidence, and its capture list if it has one, as in the
run. Budget 300 s, as in the run. Each worker's ``bye`` lists every module it loaded from its
tree.

``check OUT_DIR``:
- **clause 2, the harness reproduces 06a:** each P unit against 06a's ``h`` unit of the same
  (rtol, shard, serial) in run levin-phase-floor-06a-census-20261027T023143, bit for bit in every
  computed value of the census record, as 06a's C0 (d) defined it (c0.compare_census): the
  checkpoint's calls (06a's keys; without ``elapsed``), the slim result without ``compute_time``,
  ``GkPolicy``, the checkpoint's scan, the region arrays in full (all eight, as bytes with their
  dtypes), the pickle's scan without its timing, and the full result summary without timings;
  and 06a's C0 (c) comparison (total, total_abserr, every call's value and abserr, the arrays) is
  printed with it. Work counters (the dispatcher's) are printed, and are not part of the verdict;
- **clause 3, the new code is deterministic:** B1's unit against B2's, on the same terms, and also
  the other calls (the analytic blocks: value, abserr, regions, depth, premise counts), their
  region arrays, W3's numbers and each captured call's value and abserr;
- **no Ray:** every worker reported ``ray_initialized`` false at exit. (/tmp/ray-sgk is listed
  before and after, descriptively: R15 and R16 run Ray there in the same period.)

Read-only outside OUT_DIR. Usage: python -B w0.py run|check OUT_DIR. ``check`` runs with
``PYTHONPATH`` the tree b: the captured calls' records hold functions pickled by reference to b's
modules, and are unpickled to read their value and abserr (out/cmd/13, the first check, stopped
there without it).
"""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, RUN06A, TREES, WIDER, header, sha256  # noqa: E402

BUDGET_S = 300.0
ROWS_JSON = WIDER / "state" / "rows.json"
CAPTURE_JSON = WIDER / "state" / "w2_capture.json"
DET_UNIT_KEY = "1e-7-1-34599"  # the rule in the docstring; checked in cmd_run
REGION_FIELDS_06 = ("start", "end", "depth", "type", "abserr")
NEW_CALL_KEYS = ("n_phase_limited_regions", "premise_checks", "premise_failed")


def load_rows():
    doc = json.loads(ROWS_JSON.read_text())
    by_name05 = {}
    by_name = {}
    for r in doc["rows"]:
        by_name[r["name"]] = r
        for n in r["names05"] or []:
            by_name05[n] = r
    return doc, by_name05, by_name


def det_unit_key(doc):
    cap = json.loads(CAPTURE_JSON.read_text())["capture"]
    order = {f"1e-7-{r['shard']}-{r['serial']}": i for i, r in enumerate(doc["rows"])}
    return sorted(cap.items(), key=lambda kv: (-len(kv[1]), order[kv[0]]))[0][0]


def unit(r, rtol, tree, cap):
    key = f"{rtol}-{r['shard']}-{r['serial']}"
    return {
        "uid": f"{tree}-{key}",
        "tree": tree,
        "rtol": rtol,
        "row": r["name"],
        "names05": r["names05"],
        "shard": r["shard"],
        "serial": r["serial"],
        "k_key": r["k_key"],
        "budget": BUDGET_S,
        "scan": rtol == "1e-7",
        "w3": rtol == "1e-7",
        "capture": cap.get(key),
    }


def tmp_listing():
    try:
        return sorted(os.listdir("/tmp/ray-sgk"))
    except FileNotFoundError:
        return []


def cmd_run(out_dir):
    out_dir = Path(out_dir)
    doc, by05, by_name = load_rows()
    cap = json.loads(CAPTURE_JSON.read_text())["capture"]
    key = det_unit_key(doc)
    if key != DET_UNIT_KEY:
        raise SystemExit(f"the determinism row by rule is {key}, not {DET_UNIT_KEY}")
    shard, serial = map(int, key.split("-")[2:])
    D = next(r for r in doc["rows"] if r["shard"] == shard and r["serial"] == serial)
    specs = {
        "P1": ("p", [unit(by05["low-G+r-1"], "1e-7", "p", cap)]),
        "P2": ("p", [unit(by05["ctrl-low-1"], "1e-7", "p", cap)]),
        "P3": ("p", [unit(by_name["0-1971"], "1e-8", "p", cap)]),
        "B1": ("b", [unit(D, "1e-7", "b", cap)]),
        "B2": ("b", [unit(D, "1e-7", "b", cap)]),
    }
    before = tmp_listing()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "tmp_ray_sgk_before.json").write_text(json.dumps(before))
    procs = {}
    for name, (tree, units) in specs.items():
        env = dict(os.environ)
        env["PYTHONPATH"] = str(TREES[tree][0])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        wdir = out_dir / name
        wdir.mkdir()
        spec = {
            "tree": tree,
            "name": name,
            "units": units,
            "out_dir": str(wdir),
            "emit_fd": None,
            "emit_file": str(wdir / "emit.jsonl"),
            "rows_sha256": sha256(ROWS_JSON),
            "w2_capture_sha256": sha256(CAPTURE_JSON),
        }
        sp = wdir / "spec.json"
        sp.write_text(json.dumps(spec, indent=1))
        log = open(wdir / "worker.log", "ab")
        procs[name] = subprocess.Popen(
            [str(PYTHON), "-B", str(HERE / "worker.py"), str(sp)],
            cwd=str(WIDER),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        print(f"{name}: tree {tree}, pid {procs[name].pid}, units {[u['uid'] for u in units]}", flush=True)
    rc = {name: p.wait() for name, p in procs.items()}
    after = tmp_listing()
    (out_dir / "tmp_ray_sgk_after.json").write_text(json.dumps(after))
    print(f"exit codes {rc}")
    print(f"/tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {sorted(set(after) - set(before))}")
    return 0 if all(v == 0 for v in rc.values()) else 1


# --------------------------------------------------------------------------------------------- reading


def read_emit(path):
    units, bye, hello = {}, None, None
    with open(path) as f:
        for line in f:
            m = json.loads(line)
            if m["kind"] == "unit":
                units[m["data"]["unit"]] = m["data"]
            elif m["kind"] == "bye":
                bye = m
            elif m["kind"] == "hello":
                hello = m
    return units, bye, hello


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
    if hashlib.sha256(blob).hexdigest() != data["pickle_record_sha256"]:
        raise RuntimeError(f"{data['unit']}: pickle record sha256 differs")
    return pickle.loads(blob)


def load_input_record(dir_, ent):
    with open(Path(dir_) / ent["pickle"], "rb") as f:
        f.seek(ent["offset"])
        blob = f.read(ent["length"])
    if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
        raise RuntimeError(f"input record sha256 differs: {ent}")
    return blob


def hx(d):
    return None if d is None else d["hex"]


TIMING_KEYS = ("compute_time", "elapsed", "seconds")


def _drop_timings(x):
    if isinstance(x, dict):
        return {k: _drop_timings(v) for k, v in x.items() if k not in TIMING_KEYS}
    if isinstance(x, list):
        return [_drop_timings(v) for v in x]
    return x


def _strip_calls(calls, drop_new=False):
    drop = ("elapsed",) + (NEW_CALL_KEYS if drop_new else ())
    return [{k: v for k, v in c.items() if k not in drop} for c in calls]


def _strip_result(r):
    return None if r is None else {k: v for k, v in r.items() if k != "compute_time"}


def _strip_scan(scan):
    if scan is None:
        return None
    return [None if s is None else {k: v for k, v in s.items() if k != "seconds"} for s in scan]


def _arr_bytes(arrs):
    out = []
    for a in arrs:
        d = {}
        for k, v in a.items():
            d[k] = (v.dtype.str, v.shape, v.tobytes()) if isinstance(v, np.ndarray) else v
        out.append(d)
    return out


def compare_census(a_data, a_rec, b_data, b_rec, drop_new=False):
    """Every computed value of two census units, as 06a's c0.compare_census (drop_new: compare
    only 06a's per-call keys). Returns (holds, differing keys)."""
    diffs = []
    if _strip_calls(a_data["calls"], drop_new) != _strip_calls(b_data["calls"], drop_new):
        diffs.append("calls")
    if _strip_result(a_data.get("result")) != _strip_result(b_data.get("result")):
        diffs.append("result")
    if a_data.get("GkPolicy") != b_data.get("GkPolicy"):
        diffs.append("GkPolicy")
    if json.dumps(a_data.get("scan")) != json.dumps(b_data.get("scan")):
        diffs.append("scan (checkpoint)")
    if _arr_bytes(a_rec["regions"]) != _arr_bytes(b_rec["regions"]):
        diffs.append("region arrays")
    if pickle.dumps(_strip_scan(a_rec["scan"])) != pickle.dumps(_strip_scan(b_rec["scan"])):
        diffs.append("scan (pickle)")
    if json.dumps(_drop_timings(a_rec.get("result_full")), sort_keys=True) != json.dumps(
        _drop_timings(b_rec.get("result_full")), sort_keys=True
    ):
        diffs.append("full result summary")
    return len(diffs) == 0, diffs


def compare_c(c_data, c_rec, s_data, s_rec):
    """06a's C0 (c) comparison: total, total_abserr, every call's value and abserr (and label,
    regime, group), and 06's five region arrays. Returns (holds, n call values, n regions)."""
    cr, sr = c_data.get("result"), s_data.get("result")
    if cr is None or sr is None:
        return False, 0, 0
    ok = hx(cr["total"]) == hx(sr["total"]) and hx(cr["total_abserr"]) == hx(sr["total_abserr"])
    ok &= len(c_data["calls"]) == len(s_data["calls"])
    n_vals = 0
    for a, b in zip(c_data["calls"], s_data["calls"]):
        for key in ("index", "notify_label", "regime", "group"):
            ok &= a[key] == b[key]
        for key in ("value", "abserr"):
            n_vals += 1
            ok &= hx(a[key]) == hx(b[key])
    ok &= len(c_rec["regions"]) == len(s_rec["regions"])
    n_regions = 0
    for a, b in zip(c_rec["regions"], s_rec["regions"]):
        for f in REGION_FIELDS_06:
            x, y = a[f], b[f]
            ok &= x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        n_regions += len(a["start"])
    return bool(ok), n_vals, n_regions


def cmd_check(out_dir):
    out_dir = Path(out_dir)
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06b: Gate W0 clauses 2 and 3",
        [ROWS_JSON, CAPTURE_JSON, RUN06A / "checkpoint.jsonl"],
    ):
        print(l)
    W, byes, hellos = {}, {}, {}
    for name in ("P1", "P2", "P3", "B1", "B2"):
        W[name], byes[name], hellos[name] = read_emit(out_dir / name / "emit.jsonl")
    recs06a = read_ledger_records(RUN06A)

    print("\n== tree evidence (each worker's hello) and each unit's module roots")
    for name in W:
        ev = hellos[name]["evidence"]
        print(
            f"   {name}: tree {ev['tree']} head {ev['head']} status {'empty' if not ev['status_short'].strip() else repr(ev['status_short'])}; "
            f"all_in_tree {ev['all_in_tree']}; levin_quadrature {ev['files']['AdaptiveLevin.levin_quadrature']}; "
            f"premise check {ev['levin_has_roundoff_exit_premise_holds']} ratio {ev['levin_roundoff_exit_min_floor_ratio']}"
        )
        for uid, d in W[name].items():
            m = d["modules"]
            print(f"      {uid}: status {d['status']}; tree_head {d['tree_head']}; modules by root {m['counts_by_root']}; foreign {m['foreign']}")

    all_ok = True
    print("\n== clause 2: the harness at the parent reproduces 06a's units bit for bit")
    for name in ("P1", "P2", "P3"):
        for uid, d in W[name].items():
            h_uid = "h-" + uid.split("-", 1)[1]
            hd = recs06a[h_uid]
            hrec = load_census_record(RUN06A, hd)
            prec = load_census_record(out_dir / name, d)
            ok, diffs = compare_census(d, prec, hd, hrec, drop_new=True)
            okc, nv, nr = compare_c(d, prec, hd, hrec)
            all_ok &= ok and okc
            r = d.get("result") or {}
            print(
                f"   {name} {uid} vs 06a {h_uid} ({d['row']}): "
                f"{'IDENTICAL in every computed value' if ok else 'DIFFERS in ' + str(diffs)}; "
                f"C0 (c) terms {'identical' if okc else 'DIFFER'} ({nv} call values/abserrs, {nr} regions x 5 arrays)"
            )
            print(
                f"      total {hx(r.get('total'))} abserr {hx(r.get('total_abserr'))} | 06a "
                f"{hx(hd['result']['total'])} {hx(hd['result']['total_abserr'])}; "
                f"group calls {len(d['calls'])}, phase-limited {[c['index'] for c in d['calls'] if c['phase_limited']]}; "
                f"region arrays {len(prec['regions'])} calls x {sorted(prec['regions'][0].keys()) if prec['regions'] else []}"
            )
            print(
                f"      (descriptive) dispatcher counts {d['dispatcher_counts']} | 06a {hd['dispatcher_counts']}: "
                f"{'equal' if d['dispatcher_counts'] == hd['dispatcher_counts'] else 'differ'}; "
                f"W3 at p: {d.get('w3')}"
            )
    c2_ok = all_ok
    print(f"   clause 2 {'holds' if c2_ok else 'FAILS'}")

    print("\n== clause 3: the new code is deterministic (one b unit, two fresh workers)")
    (uid,) = list(W["B1"])
    a, b = W["B1"][uid], W["B2"][uid]
    arec = load_census_record(out_dir / "B1", a)
    brec = load_census_record(out_dir / "B2", b)
    ok, diffs = compare_census(a, arec, b, brec, drop_new=False)
    if _strip_calls(a["other_calls"]) != _strip_calls(b["other_calls"]):
        diffs.append("other calls")
    if _arr_bytes(arec["other_regions"]) != _arr_bytes(brec["other_regions"]):
        diffs.append("other calls' region arrays")
    if _drop_timings(a["w3"]) != _drop_timings(b["w3"]) or arec["w3"] != brec["w3"]:
        diffs.append("W3")
    vals = []
    for ea, eb in zip(a["w2_inputs"] or [], b["w2_inputs"] or []):
        ra = pickle.loads(load_input_record(out_dir / "B1", ea))
        rb = pickle.loads(load_input_record(out_dir / "B2", eb))
        same = (ra["index"], ra["value"], ra["abserr"], ra["num_regions"]) == (
            rb["index"], rb["value"], rb["abserr"], rb["num_regions"]
        )
        vals.append((ra["index"], ra["value"], ra["abserr"], same, ea["sha256"] == eb["sha256"], ea["length"]))
        if not same:
            diffs.append(f"captured call {ra['index']}")
    if len(a["w2_inputs"] or []) != len(b["w2_inputs"] or []):
        diffs.append("number of captured calls")
    c3_ok = len(diffs) == 0
    hd = recs06a["h-" + uid.split("-", 1)[1]]
    print(f"   {uid} ({a['row']}): B1 vs B2 {'IDENTICAL in every computed value' if c3_ok else 'DIFFERS in ' + str(diffs)}")
    print(f"      total {hx(a['result']['total'])} abserr {hx(a['result']['total_abserr'])}; 06a {hx(hd['result']['total'])} {hx(hd['result']['total_abserr'])}")
    for ca, ch in zip(a["calls"], hd["calls"]):
        print(
            f"      call {ca['index']} {ca['group']}: b value {hx(ca['value'])} abserr {hx(ca['abserr'])} regions {ca['num_regions']} "
            f"depth {ca['max_depth']} phase-limited regions {ca['n_phase_limited_regions']} premise checks {ca['premise_checks']} "
            f"failed {ca['premise_failed']} | 06a value {hx(ch['value'])} abserr {hx(ch['abserr'])} regions {ch['num_regions']} "
            f"depth {ch['max_depth']} phase_limited {ch['phase_limited']}"
        )
    print(f"      other calls: {[(c['notify_label'], c['num_regions'], c['max_depth'], c['n_phase_limited_regions'], c['premise_checks'], c['premise_failed']) for c in a['other_calls']]}")
    print(f"      W3 {a['w3']}")
    print(f"      captured calls (index, value, abserr, equal in B1 and B2, blob sha256 equal, blob bytes): {vals}")
    print(f"      wall {a['wall_s']:.2f} s and {b['wall_s']:.2f} s (timings, not compared)")
    print(f"   clause 3 {'holds' if c3_ok else 'FAILS'}")

    print("\n== no Ray")
    before = json.loads((out_dir / "tmp_ray_sgk_before.json").read_text())
    after = json.loads((out_dir / "tmp_ray_sgk_after.json").read_text())
    ray_flags = {n: (None if byes[n] is None else byes[n]["ray_initialized"]) for n in byes}
    r_ok = all(v is False for v in ray_flags.values())
    print(f"   workers' ray_initialized at exit: {ray_flags}: {'none initialised Ray' if r_ok else 'RAY INITIALISED'}")
    print(f"   (descriptive; R15/R16 use /tmp/ray-sgk in this period) /tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {sorted(set(after) - set(before))}")

    # the import list for clause 1, from the parent's workers
    own = {}
    for name in ("P1", "P2", "P3"):
        for mname, f in byes[name]["modules_own_tree"]:
            own[mname] = f
    out = out_dir / "p_tree_modules.json"
    out.write_text(json.dumps(dict(sorted(own.items())), indent=1))
    print(f"\n== the parent tree's modules imported by the harness (P1-P3, union): {len(own)}; written to {out}")
    tests = sorted(m for m in own if ".tests." in m or m.endswith(".tests"))
    print(f"   of them under a tests package: {tests}")
    ok = c2_ok and c3_ok and r_ok
    print(f"\n== W0 clauses 2 and 3 (and no Ray): {'HOLD' if ok else 'DO NOT HOLD'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    sys.exit({"run": cmd_run, "check": cmd_check}[cmd](arg))
