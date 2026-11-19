"""levin-phase-floor prompt 06j, Gate J0: replay one captured call at the tree, in this process, and compare it
with its 06g census record bit for bit. New in 06j (after 06f's replay06f.py and 06g's worker.py).

The capture (state/inputs/<sid>.pkl, sliced from x1_capture.py's worker pickle; loaded with ray.cloudpickle,
this directory on PYTHONPATH after the tree for ``harness._rebuild_lru_cache``) holds the call's span, f, theta
and keyword arguments as the census worker saw them. The replay is
``AdaptiveLevin.levin_quadrature.adaptive_levin_sincos(x_span, f, theta, **kw)``, with one pass-through
installed for the call only and restored in a finally: ``used_interval.__init__`` (the class attribute; it
calls the original and keeps the region), exactly as 06g's census worker kept every region it made. No
other wrapper; nothing in the tree is changed.

Compared with the census (the 06g census run, read only; its checkpoint call record and its region pickle
record, SHA-256 checked):
- value and abserr by hex, num_regions, max_depth (over the returned regions) and converged;
- the region arrays (worker.py's ``_region_arrays_06g``, imported unchanged) against the census record's
  ``regions`` entry for the call, every key: arrays by dtype, shape and bytes;
- the checked leaves (worker.py's ``_checked_leaves_06g``) against the census record's
  ``checked_leaves_06g`` entry, every key.

It also writes the per-leaf table of every checked leaf (returned or bisected) with each float by hex, and
its SHA-256 (the determinism clause: two replays in fresh processes must give the same digest), and the
returned leaves' (start, end, depth, value, r) in full.

Usage: PYTHONPATH=<tree>:<this directory> python -B replay_06j.py SID OUT_JSON
"""

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402
from check_captures import same  # noqa: E402


def load_selection(sid):
    sel = [e for e in json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text()) if e["sid"] == sid]
    if len(sel) != 1:
        raise SystemExit(f"{sid}: {len(sel)} selection entries")
    return sel[0]


def census_pickle(e):
    run06g = RC.census_run_06g()
    name, off, ln, sha = e["g_pickle"]
    with open(run06g / name, "rb") as f:
        f.seek(off)
        blob = f.read(ln)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit(f"{name}@{off}: sha256 differs")
    return pickle.loads(blob)


def leaf_table(rec):
    """Every checked leaf, each float by hex, in the order the driver made them."""
    keys = sorted(k for k, v in rec.items() if isinstance(v, np.ndarray))
    n = len(rec["depth"])
    rows = []
    for i in range(n):
        row = {}
        for k in keys:
            v = rec[k][i]
            if rec[k].dtype.kind == "f":
                row[k] = float(v).hex()
            elif rec[k].dtype.kind == "b":
                row[k] = bool(v)
            else:
                row[k] = int(v)
        rows.append(row)
    return keys, rows


def main():
    sid, out_p = sys.argv[1], Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    import ray
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ
    import worker as WK  # the census worker's recorders, unchanged

    e = load_selection(sid)
    pkl = RC.ROOT / "state" / "inputs" / f"{sid}.pkl"
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06j: J0 replay of {sid} ({e['unit']} #{e['index']})",
                 [pkl, RC.ROOT / "out" / "sample" / "selection.json", Path(LQ.__file__), Path(WK.__file__)])
    rec = cp.loads(pkl.read_bytes())
    if rec["unit"] != e["unit"] or rec["index"] != e["index"] or rec["notify_label"] != e["notify_label"]:
        raise SystemExit(f"{sid}: the capture is not the selection's call")
    made = []
    UI = LQ.used_interval
    orig_init = UI.__dict__["__init__"]

    def w_init(self, *a, **k):
        orig_init(self, *a, **k)
        made.append(self)

    UI.__init__ = w_init
    t0 = time.perf_counter()
    try:
        result = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **rec["kw"])
    finally:
        UI.__init__ = orig_init
    wall = time.perf_counter() - t0
    call = {"index": rec["index"], "notify_label": rec["notify_label"], "result": result, "made": made}
    arrays = WK._region_arrays_06g(call)
    chk, _summary = WK._checked_leaves_06g(call)
    prec = census_pickle(e)
    gr = [x for x in prec["regions"] if x["index"] == e["index"]]
    gc = [x for x in prec["checked_leaves_06g"] if x is not None and x["index"] == e["index"]]
    problems = []
    g = e["g_record"]
    got = {"value": float(result["value"]).hex(), "abserr": float(result["abserr"]).hex(),
           "num_regions": int(result["num_regions"]), "max_depth": int(max(r.depth for r in result["regions"])),
           "converged": bool(result["converged"])}
    want = {"value": g["value"]["hex"], "abserr": g["abserr"]["hex"], "num_regions": g["num_regions"],
            "max_depth": g["max_depth"], "converged": bool(g["converged"])}
    for k in got:
        if got[k] != want[k]:
            problems.append(f"{k}: replay {got[k]} census {want[k]}")
    region_diff, checked_diff = None, None
    if len(gr) != 1:
        problems.append(f"census region entries for the call: {len(gr)}")
    else:
        region_diff = sorted(k for k in set(arrays) | set(gr[0]) if k not in ("index", "regime", "group", "x_span")
                             and (k not in arrays or k not in gr[0] or not same(arrays[k], gr[0][k])))
        if region_diff:
            problems.append(f"region arrays differ: {region_diff}")
    if len(gc) != 1:
        problems.append(f"census checked-leaf entries for the call: {len(gc)}")
    else:
        checked_diff = sorted(k for k in set(chk) | set(gc[0]) if k not in chk or k not in gc[0] or not same(chk[k], gc[0][k]))
        if checked_diff:
            problems.append(f"checked leaves differ: {checked_diff}")
    keys, rows = leaf_table(chk)
    table_bytes = json.dumps({"keys": keys, "rows": rows}, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(table_bytes).hexdigest()
    ret = [{"start": float(r.start).hex(), "end": float(r.end).hex(), "depth": int(r.depth),
            "value": float(r.value).hex(), "value_own": float(r.value_own).hex(),
            "r": None if r.convergence_ratio is None else float(r.convergence_ratio).hex(),
            "trusted": r.trusted} for r in result["regions"]]
    ret_digest = hashlib.sha256(json.dumps(ret, sort_keys=True).encode()).hexdigest()
    doc = {"header": hdr, "sid": sid, "unit": e["unit"], "index": e["index"], "replay": got, "census": want,
           "region_keys_differing": region_diff, "checked_keys_differing": checked_diff,
           "n_made": len(made), "n_checked": len(rows), "checked_table_sha256": digest,
           "returned_leaves": ret, "returned_leaves_sha256": ret_digest, "wall_s": wall,
           "problems": problems, "reproduces_census": not problems,
           "ray_initialized": ray.is_initialized()}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(f"{sid} {e['unit']} #{e['index']}: {'REPRODUCES the census bit for bit' if not problems else 'DIFFERS: ' + '; '.join(problems)}"
          f"; {len(rows)} checked leaves, table sha256 {digest}; returned {len(ret)} sha256 {ret_digest}; {wall:.1f} s;"
          f" ray initialised {ray.is_initialized()}")
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
