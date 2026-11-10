"""levin-phase-floor prompt 06e section 5, Gate Z0's second and third clauses (and the import list
its first clause reads). Copied from 06c's x0.py (kept unchanged in state/scripts_06c_reference/)
and changed, each change marked "prompt 06e":
- **the trees and the "before":** P1-P3 run at the parent ``p`` (50277f3, whose code is 06c's census
  tree e5764a6's) and are compared with 06c's census units ``g-<rtol>-<shard>-<serial>`` of run
  levin-phase-floor-06c-census-20261029T130214 (prov.RUN06C), on 06c's own keys: every key 06c's
  worker recorded, now **including 06c's deep lines** (checkpoint and pickle), which 06b's record,
  X0's "before", did not have; 06c's keys are a subset of what this worker records, each unchanged;
- **the determinism unit** is 06c's X0 unit, ``g-1e-7-0-14164``, run at ``g`` (6d81fad) in two fresh
  workers, G1 and G2, compared on every key the worker records, 06e's new ones included
  (``cc_floor`` per call, ``deep_lines_06e``, ``cc_floor_terms``), timings excepted;
- **no capture:** no unit captures inputs (Z3's inputs are depth20-tiling's captures), so the
  captured-call comparison of 06c's clause 3 is not made;
- the five workers run at once, as in 06c (five, within the six-core rule), only when the
  orchestrator has released the cores.
Usage: python -B z0.py run|check OUT_DIR. ``check`` runs with ``PYTHONPATH`` the tree g.

(06c's text follows, unchanged; read its 06b as 06c, its X0 as Z0, its g as 06e's g.)

levin-phase-floor prompt 06c section 5, Gate X0's second and third clauses (and the import list
its first clause reads). New in 06c; adapted from 06b's w0.py (its ``run`` and ``check``; a record,
kept unchanged in state/scripts_06b_reference/).

``run OUT_DIR``: five fresh census workers (worker.py, the producer cache), each its own process,
all at once (five, within the six-worker rule), no Ray:
- at the parent ``p`` (``fef401c``, whose code is 06b's census's), one worker per unit: P1
  ``low-G+r-1`` at 1e-7, P2 ``ctrl-low-1`` at 1e-7, P3 ``0-1971`` at 1e-8 (prompt 06c section 5);
- at ``g`` (``e5764a6``), G1 and G2: the same unit twice, ``g-1e-7-0-14164``, in two fresh workers.
  The row is fixed by rule (explore06b_pool.py, out/cmd/07): of the 1e-7 rows outside 06a's 720
  stationary-point rows, the one whose 06b unit has the most X2-type calls (06b unconverged, no
  region at depth_max, no phase-limited region, finite abserr; group and analytic), ties broken by
  rows.json's order. Each G worker also captures the inputs of that unit's first X2-type group call
  (lowest index), so that the capture path X1 will use is exercised and compared.
Every 1e-7 unit carries the scan and 06b's W3 evidence, as in the run; no P unit captures. Budget
300 s, as in the run. Each worker's ``bye`` lists every module it loaded from its tree.

``check OUT_DIR``:
- **clause 2, the harness reproduces 06b:** each P unit against 06b's ``b`` unit of the same
  (rtol, shard, serial) in run levin-phase-floor-06b-census-20261028T161212, bit for bit in every
  value 06b's census recorded, restricted to 06b's own keys (the worker records more under 06c):
  the checkpoint's group calls and analytic-block calls (every 06b key but ``elapsed``), the slim
  result without ``compute_time``, ``GkPolicy``, the checkpoint's scan, the group calls' and the
  analytic blocks' region arrays (06b's eight arrays, as bytes with their dtypes, and each entry's
  index, labels and span), the pickle's scan without its timing, the full result summary without
  timings, and 06b's W3 numbers (checkpoint summary without its timing, and the pickle's per-region
  numbers). 06a's C0 (c) terms are printed with it. The dispatcher's counts are printed, and are
  not part of the verdict;
- **clause 3, the new code is deterministic:** G1's unit against G2's, on every key the worker
  records (calls and analytic-block calls with every 06c key, the budget counters and the solves by
  purpose among them; every region array; W3; the deep lines; the captured call's value, abserr,
  regions, depth and converged), timings excepted;
- **no Ray:** every worker reported ``ray_initialized`` false at exit. (/tmp/ray-sgk is listed
  before and after, descriptively: R17 and R18 run Ray there in the same period.)

Read-only outside OUT_DIR. Usage: python -B x0.py run|check OUT_DIR. ``check`` runs with
``PYTHONPATH`` the tree g: the captured call's record holds functions pickled by reference to g's
modules, and is unpickled to read its value and abserr (06b's out/cmd/13 stopped there without it).
"""

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, RUN06A, RUN06B, RUN06C, TREES, WIDER, header, sha256  # noqa: E402  prompt 06e: RUN06C

BUDGET_S = 300.0
ROWS_JSON = WIDER / "state" / "rows.json"
DET_UNIT_KEY = "1e-7-0-14164"  # the rule in the docstring (out/cmd/07); checked in cmd_run
DEPTH_MAX = 20
REGION_FIELDS_06 = ("start", "end", "depth", "type", "abserr")


def load_rows():
    doc = json.loads(ROWS_JSON.read_text())
    by_name05 = {}
    by_name = {}
    for r in doc["rows"]:
        by_name[r["name"]] = r
        for n in r["names05"] or []:
            by_name05[n] = r
    return doc, by_name05, by_name


def read_one(run_dir, uid):
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{uid}"' in line:
                r = json.loads(line)
                if r["unit"] == uid:
                    return r["data"]
    raise KeyError(uid)


def x2_type(c):
    return (
        not c["converged"]
        and c["max_depth"] < DEPTH_MAX
        and c["n_phase_limited_regions"] == 0
        and math.isfinite(float.fromhex(c["abserr"]["hex"]))
    )


def unit(r, rtol, tree, capture=None):
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
        "capture": capture,
    }


def tmp_listing():
    try:
        return sorted(os.listdir("/tmp/ray-sgk"))
    except FileNotFoundError:
        return []


def cmd_run(out_dir):
    out_dir = Path(out_dir)
    doc, by05, by_name = load_rows()
    shard, serial = map(int, DET_UNIT_KEY.split("-")[2:])
    D = next(r for r in doc["rows"] if r["shard"] == shard and r["serial"] == serial)
    cap = None  # prompt 06e: no capture
    print(f"determinism unit g-{DET_UNIT_KEY} ({D['name']}) (06c's X0 unit); no capture")
    specs = {
        "P1": ("p", [unit(by05["low-G+r-1"], "1e-7", "p")]),
        "P2": ("p", [unit(by05["ctrl-low-1"], "1e-7", "p")]),
        "P3": ("p", [unit(by_name["0-1971"], "1e-8", "p")]),
        "G1": ("g", [unit(D, "1e-7", "g", cap)]),
        "G2": ("g", [unit(D, "1e-7", "g", cap)]),
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
            "w2_capture_sha256": None,
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
    print(f"/tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {sorted(set(after) - set(before))} (descriptive)")
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


def _calls_on_keys(calls, keys=None):
    """Each call dict without ``elapsed``; restricted to ``keys`` (06b's) if given."""
    out = []
    for c in calls:
        out.append({k: v for k, v in c.items() if k != "elapsed" and (keys is None or k in keys)})
    return out


def _arr_bytes(arrs, keys=None):
    out = []
    for a in arrs:
        d = {}
        for k, v in a.items():
            if keys is not None and k not in keys:
                continue
            d[k] = (v.dtype.str, v.shape, v.tobytes()) if isinstance(v, np.ndarray) else v
        out.append(d)
    return out


def _strip_result(r):
    return None if r is None else {k: v for k, v in r.items() if k != "compute_time"}


def _strip_scan(scan):
    if scan is None:
        return None
    return [None if s is None else {k: v for k, v in s.items() if k != "seconds"} for s in scan]


def _keys_of(lst):
    ks = set()
    for d in lst:
        ks |= set(d.keys())
    return ks


def compare_units(a_data, a_rec, b_data, b_rec, restrict_to_b=False):
    """Every computed value two census units recorded; with restrict_to_b, only on the keys
    ``b`` (06b's record) holds. Returns the list of differing items (empty: identical)."""
    diffs = []
    ck = _keys_of(b_data["calls"]) if restrict_to_b else None
    ok_ = _keys_of(b_data["other_calls"]) if restrict_to_b else None
    if _calls_on_keys(a_data["calls"], ck) != _calls_on_keys(b_data["calls"], ck):
        diffs.append("group calls")
    if _calls_on_keys(a_data["other_calls"], ok_) != _calls_on_keys(b_data["other_calls"], ok_):
        diffs.append("analytic-block calls")
    if _strip_result(a_data.get("result")) != _strip_result(b_data.get("result")):
        diffs.append("result")
    if a_data.get("GkPolicy") != b_data.get("GkPolicy"):
        diffs.append("GkPolicy")
    if json.dumps(a_data.get("scan")) != json.dumps(b_data.get("scan")):
        diffs.append("scan (checkpoint)")
    rk = _keys_of(b_rec["regions"]) if restrict_to_b else None
    if _arr_bytes(a_rec["regions"], rk) != _arr_bytes(b_rec["regions"], rk):
        diffs.append("group calls' region arrays")
    ork = _keys_of(b_rec["other_regions"]) if restrict_to_b else None
    if _arr_bytes(a_rec["other_regions"], ork) != _arr_bytes(b_rec["other_regions"], ork):
        diffs.append("analytic blocks' region arrays")
    if pickle.dumps(_strip_scan(a_rec["scan"])) != pickle.dumps(_strip_scan(b_rec["scan"])):
        diffs.append("scan (pickle)")
    if json.dumps(_drop_timings(a_rec.get("result_full")), sort_keys=True) != json.dumps(
        _drop_timings(b_rec.get("result_full")), sort_keys=True
    ):
        diffs.append("full result summary")
    if _drop_timings(a_data.get("w3")) != _drop_timings(b_data.get("w3")):
        diffs.append("W3 (checkpoint)")
    if json.dumps(a_rec.get("w3"), sort_keys=True) != json.dumps(b_rec.get("w3"), sort_keys=True):
        diffs.append("W3 (pickle)")
    # prompt 06e: 06c recorded deep lines, so they are compared in both modes
    if json.dumps(_drop_timings(a_data.get("deep_lines")), sort_keys=True) != json.dumps(
        _drop_timings(b_data.get("deep_lines")), sort_keys=True
    ):
        diffs.append("deep lines (checkpoint)")
    if json.dumps(a_rec.get("deep_lines"), sort_keys=True) != json.dumps(b_rec.get("deep_lines"), sort_keys=True):
        diffs.append("deep lines (pickle)")
    if not restrict_to_b:
        # prompt 06e: the new keys (per call cc_floor is inside the calls' dicts, compared above)
        if json.dumps(_drop_timings(a_data.get("deep_lines_06e")), sort_keys=True) != json.dumps(
            _drop_timings(b_data.get("deep_lines_06e")), sort_keys=True
        ):
            diffs.append("deep lines past depth 12 (checkpoint)")
        if json.dumps(a_rec.get("deep_lines_06e"), sort_keys=True) != json.dumps(b_rec.get("deep_lines_06e"), sort_keys=True):
            diffs.append("deep lines past depth 12 (pickle)")
        if _cc_terms_bytes(a_rec.get("cc_floor_terms")) != _cc_terms_bytes(b_rec.get("cc_floor_terms")):
            diffs.append("Clenshaw-Curtis floor terms (pickle)")
        ka = set(a_data) - {"wall_s", "rebuild_s", "scan_s", "cc_floor_s", "worker", "worker_pid", "pickle_offset", "pickle_record_sha256", "pickle_length", "pickle", "unit", "policy_warnings"}
        kb = set(b_data) - {"wall_s", "rebuild_s", "scan_s", "cc_floor_s", "worker", "worker_pid", "pickle_offset", "pickle_record_sha256", "pickle_length", "pickle", "unit", "policy_warnings"}
        if ka != kb:
            diffs.append(f"checkpoint keys differ: {sorted(ka ^ kb)}")
    return diffs


def _cc_terms_bytes(lst):
    """prompt 06e: the pickle's cc_floor_terms, arrays as (dtype, shape, bytes)."""
    if lst is None:
        return None
    return [_arr_bytes([{k: v for k, v in d.items() if k != "errors"}])[0] | {"errors": json.dumps(d.get("errors"))} for d in lst]


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
        "levin-phase-floor 06e: Gate Z0 clauses 2 and 3",  # prompt 06e
        [ROWS_JSON, RUN06C / "checkpoint.jsonl"],
    ):
        print(l)
    W, byes, hellos = {}, {}, {}
    for name in ("P1", "P2", "P3", "G1", "G2"):
        W[name], byes[name], hellos[name] = read_emit(out_dir / name / "emit.jsonl")

    print("\n== tree evidence (each worker's hello) and each unit's module roots")
    for name in W:
        ev = hellos[name]["evidence"]
        print(
            f"   {name}: tree {ev['tree']} head {ev['head']} status {'empty' if not ev['status_short'].strip() else repr(ev['status_short'])}; "
            f"all_in_tree {ev['all_in_tree']}; levin_quadrature {ev['files']['AdaptiveLevin.levin_quadrature']}; "
            f"premise check {ev['levin_has_roundoff_exit_premise_holds']}; trust check {ev['levin_has_trust_check']}; "
            f"CC floor {ev['levin_has_cc_roundoff_floor']}; _local_atol {ev['levin_has_local_atol']}; default depth_max {ev['levin_default_max_depth']}"
        )
        for uid, d in W[name].items():
            m = d["modules"]
            print(f"      {uid}: status {d['status']}; wall {d['wall_s']:.2f} s; tree_head {d['tree_head']}; modules by root {m['counts_by_root']}; foreign {m['foreign']}")

    all_ok = True
    print("\n== clause 2: the harness at the parent reproduces 06c's units bit for bit")  # prompt 06e
    for name in ("P1", "P2", "P3"):
        for uid, d in W[name].items():
            b_uid = "g-" + uid.split("-", 1)[1]  # prompt 06e: 06c's census unit
            bd = read_one(RUN06C, b_uid)
            brec = load_census_record(RUN06C, bd)
            prec = load_census_record(out_dir / name, d)
            diffs = compare_units(d, prec, bd, brec, restrict_to_b=True)
            okc, nv, nr = compare_c(d, prec, bd, brec)
            ok = not diffs
            all_ok &= ok and okc
            r = d.get("result") or {}
            n_oregions = sum(len(o["start"]) for o in prec["other_regions"])
            print(
                f"   {name} {uid} vs 06c {b_uid} ({d['row']}): "
                f"{'IDENTICAL in every value 06c recorded' if ok else 'DIFFERS in ' + str(diffs)}; "
                f"C0 (c) terms {'identical' if okc else 'DIFFER'} ({nv} group-call values/abserrs, {nr} group-call regions)"
            )
            print(
                f"      total {hx(r.get('total'))} abserr {hx(r.get('total_abserr'))} | 06c "
                f"{hx(bd['result']['total'])} {hx(bd['result']['total_abserr'])}; "
                f"group calls {len(d['calls'])}, analytic-block calls {len(d['other_calls'])} ({n_oregions} regions); "
                f"region arrays compared on 06c's keys {sorted(_keys_of(brec['regions']))}; "
                f"06c deep lines {(bd.get('deep_lines') or {}).get('n_calls')}, here {(d.get('deep_lines') or {}).get('n_calls')}"
            )
            print(
                f"      (descriptive) dispatcher counts {d['dispatcher_counts']} | 06c {bd['dispatcher_counts']}: "
                f"{'equal' if d['dispatcher_counts'] == bd['dispatcher_counts'] else 'differ'}; W3 {_drop_timings(d.get('w3'))}"
            )
    c2_ok = all_ok
    print(f"   clause 2 {'holds' if c2_ok else 'FAILS'}")

    print("\n== clause 3: the new code is deterministic (one g unit, two fresh workers)")
    (uid,) = list(W["G1"])
    a, b = W["G1"][uid], W["G2"][uid]
    arec = load_census_record(out_dir / "G1", a)
    brec = load_census_record(out_dir / "G2", b)
    diffs = compare_units(a, arec, b, brec, restrict_to_b=False)
    vals = "none (prompt 06e: no capture)"
    c3_ok = len(diffs) == 0
    bd = read_one(RUN06C, "g-" + uid.split("-", 1)[1])  # prompt 06e: 06c's census unit, descriptive
    print(f"   {uid} ({a['row']}): G1 vs G2 {'IDENTICAL in every computed value' if c3_ok else 'DIFFERS in ' + str(diffs)}")
    print(f"      total {hx(a['result']['total'])} abserr {hx(a['result']['total_abserr'])} converged {a['result']['total_converged']}; "
          f"06c {hx(bd['result']['total'])} {hx(bd['result']['total_abserr'])} converged {bd['result']['total_converged']}")
    cb = {c["index"]: c for c in bd["calls"]}
    for ca in a["calls"]:
        ch = cb[ca["index"]]
        print(
            f"      call {ca['index']} {ca['group']}: g value {hx(ca['value'])} abserr {hx(ca['abserr'])} conv {ca['converged']} regions {ca['num_regions']} "
            f"depth {ca['max_depth']} untrusted {ca['num_untrusted_regions']} leaves {ca['leaves']} E_share-largest {ca['E_largest_share']} "
            f"budget {ca['budget']} solves {ca['solves_by_purpose']} cc_floor {ca.get('cc_floor')} | 06c value {hx(ch['value'])} abserr {hx(ch['abserr'])} conv {ch['converged']} "
            f"regions {ch['num_regions']} depth {ch['max_depth']}"
        )
    ob = {(c["index"]): c for c in bd["other_calls"]}
    print("      analytic-block calls (index, label, g: conv, regions, depth, untrusted, CC leaves reproduced/n; 06c: conv, regions, depth):")
    for ca in a["other_calls"]:
        ch = ob.get(ca["index"])
        ccf = ca.get("cc_floor") or {}
        print(f"         {ca['index']} {ca['notify_label']}: {ca['converged']} {ca['num_regions']} {ca['max_depth']} {ca['num_untrusted_regions']} "
              f"{ccf.get('n_reproduced')}/{ccf.get('n_cc_leaves')} | "
              f"{None if ch is None else (ch['converged'], ch['num_regions'], ch['max_depth'])}")
    print(f"      W3 {_drop_timings(a['w3'])}; deep lines {_drop_timings({k: v for k, v in (a['deep_lines'] or {}).items() if k != 'lines'})}")
    print(f"      (prompt 06e) deep lines past depth 12 {_drop_timings({k: v for k, v in (a['deep_lines_06e'] or {}).items() if k != 'lines'})}; "
          f"CC floor terms: {sum(1 for d_ in (arec.get('cc_floor_terms') or []))} calls with fallback leaves, "
          f"{sum(int(d_['reproduces'].sum()) for d_ in (arec.get('cc_floor_terms') or []))} of "
          f"{sum(len(d_['reproduces']) for d_ in (arec.get('cc_floor_terms') or []))} leaves reproduced; "
          f"tree evidence g: {[(k, v) for k, v in hellos['G1']['evidence'].items() if k.startswith('levin_has_c') or k == 'levin_quadrature_sha256']}")
    print(f"      captured calls: {vals}")
    print(f"      wall {a['wall_s']:.2f} s and {b['wall_s']:.2f} s (timings, not compared)")
    print(f"   clause 3 {'holds' if c3_ok else 'FAILS'}")

    print("\n== no Ray")
    before = json.loads((out_dir / "tmp_ray_sgk_before.json").read_text())
    after = json.loads((out_dir / "tmp_ray_sgk_after.json").read_text())
    ray_flags = {n: (None if byes[n] is None else byes[n]["ray_initialized"]) for n in byes}
    r_ok = all(v is False for v in ray_flags.values())
    print(f"   workers' ray_initialized at exit: {ray_flags}: {'none initialised Ray' if r_ok else 'RAY INITIALISED'}")
    print(f"   (descriptive) /tmp/ray-sgk entries before {len(before)}, after {len(after)}, new {sorted(set(after) - set(before))}")

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
    print(f"\n== Z0 clauses 2 and 3 (and no Ray): {'HOLD' if ok else 'DO NOT HOLD'}")  # prompt 06e
    return 0 if ok else 1


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    sys.exit({"run": cmd_run, "check": cmd_check}[cmd](arg))
