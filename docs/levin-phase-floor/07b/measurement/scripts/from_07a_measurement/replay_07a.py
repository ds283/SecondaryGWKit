"""levin-phase-floor prompt 07a Part A, sections 2.1-2.3: replay one captured call, in this process, at the tree's
module or at one variant of the window's stop, and compare it with 07's census record. New in 07a (after 06j's
replay_06j.py, 07's census worker and 06k's variant_loader.py).

``VARIANT`` is ``tree`` (the worktree's own ``AdaptiveLevin/levin_quadrature.py``, imported as production imports
it: the replay of section 2.1's rule) or one of a-g (scripts/variants/levin_quadrature_07a_<v>.py, installed by
07's ``variant_loader.install`` before anything imports ``AdaptiveLevin``, SHA-256 checked against
out/variants/variants.json). The capture (state/calls_07a.json: path, offset, length, SHA-256) is loaded with
ray.cloudpickle, this directory on the path for ``harness._rebuild_lru_cache``; for the 396, a capture taken at
1e-7 gets the 1e-8 call's ``rtol`` and ``atol`` (``override``). The replay is
``adaptive_levin_sincos(x_span, f, theta, **kw)``; with ``--instrument`` (a variant only) the variant's
``_MEASURE_07A`` list is set for the call and its records are written.

Compared (each by hex for floats; arrays by dtype, shape and bytes):
- with 07's census record (state/census_refs.pkl): value, abserr, num_regions, max_depth, converged; the region
  arrays (07's worker's ``_region_arrays_06g`` and ``_region_arrays_07``, imported unchanged) against the census
  record's ``regions`` entry, every key; section 1.5's result entries (``stationary_points``,
  ``stationary_windows``, ...; 07's ``_guard_record_07``) against the census's ``guard_07`` entry, every key. For
  the no-window calls, the scalars only (their regions were checked by check_captures.py at the capture);
- ``reproduces_census`` is true iff all of these agree. At ``tree`` and ``a`` a false is section 8's stop.

Writes OUT_DIR/<variant>/<sid>.json (the summary, the comparison, the returned leaves) and, with --instrument,
OUT_DIR/<variant>/<sid>.instrument.json; refuses to overwrite. Each carries the SHA-256 of this script, the module
run and the capture.

Usage: PYTHONPATH=<tree> python -B replay_07a.py SID VARIANT OUT_DIR [--instrument]
"""

import hashlib
import json
import math
import os
import pickle
import socket
import sys
import time
import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
TREE_LQ = TREE / "AdaptiveLevin" / "levin_quadrature.py"
TREE_LQ_SHA = "4e5da371f51e6e3dc20e73579eb45e0e985e2a3b18571ee29eb86b3d3e9c93fc"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def H(x):
    if x is None:
        return None
    return float(x).hex()


def jsonable(x):
    """Every float as hex (exact); numpy scalars made Python; tuples as lists."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a.hex() == b.hex() if (a == a or b == b) else True
    return a == b


def main():
    sid, variant, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    instrument = "--instrument" in sys.argv[4:]
    vdir = out_dir / variant
    out_p = vdir / f"{sid}.json"
    inst_p = vdir / f"{sid}.instrument.json"
    if out_p.exists() or (instrument and inst_p.exists()):
        raise SystemExit(f"{out_p} exists; not overwritten")
    if os.environ.get("PYTHONPATH", "").split(":")[0] != str(TREE):
        raise SystemExit("PYTHONPATH must start with the tree")
    sys.path.insert(1, str(HERE))
    if sha256(TREE_LQ) != TREE_LQ_SHA:
        raise SystemExit("the tree's levin_quadrature.py has another SHA-256")
    manifest = json.loads((ROOT / "state" / "calls_07a.json").read_text())
    c = manifest["calls"][sid]
    if variant == "tree":
        if instrument:
            raise SystemExit("--instrument needs a variant")
        import AdaptiveLevin.levin_quadrature as LQ

        module_path, module_sha = TREE_LQ, TREE_LQ_SHA
    else:
        import variant_loader as VL

        vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
        LQ = VL.install(vj["path"], vj["sha256"], f"07a_{variant}")
        module_path, module_sha = Path(vj["path"]), vj["sha256"]
        if LQ._VARIANT_07A != variant:
            raise SystemExit("the variant's constant is not its name")
    import ray
    import ray.cloudpickle as cp

    import worker as WK  # 07's census worker's recorders, unchanged

    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit(f"{sid}: capture blob has another SHA-256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c["override"]:
        kw["rtol"] = c["override"]["rtol"]
        kw["atol"] = c["override"]["atol"]
    import AdaptiveLevin

    if AdaptiveLevin.levin_quadrature is not LQ or Path(LQ.__file__).resolve() != Path(module_path).resolve():
        raise SystemExit("the module the replay runs is not the one named")
    if instrument:
        LQ._MEASURE_07A = []
    t0 = time.perf_counter()
    try:
        result = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **kw)
    finally:
        records = LQ._MEASURE_07A if instrument else None
        if instrument:
            LQ._MEASURE_07A = None
    wall = time.perf_counter() - t0
    regs = result["regions"]
    call = {"index": rec["index"], "notify_label": rec["notify_label"], "result": result, "kw": kw,
            "default_depth_max": getattr(LQ, "DEFAULT_LEVIN_MAX_DEPTH", None)}
    arrays = {**WK._region_arrays_06g(call), **WK._region_arrays_07(call)}
    guard = WK._guard_record_07(call)
    got = {"value": H(result["value"]), "abserr": H(result["abserr"]), "num_regions": int(result["num_regions"]),
           "max_depth": int(max(r.depth for r in regs)), "converged": bool(result["converged"])}
    problems = []
    region_diff = guard_diff = None
    if c["set"] == "nowindow":
        want = {"value": c["census_value"], "abserr": c["census_abserr"], "num_regions": c["census_num_regions"],
                "max_depth": c["census_max_depth"], "converged": c["census_converged"]}
    else:
        refs = pickle.loads((ROOT / "state" / "census_refs.pkl").read_bytes())["calls"][c["census"]]
        cr = refs["call"]
        want = {"value": cr["value"]["hex"], "abserr": cr["abserr"]["hex"], "num_regions": cr["num_regions"],
                "max_depth": cr["max_depth"], "converged": bool(cr["converged"])}
        if rec["notify_label"] != cr["notify_label"]:
            problems.append("notify_label differs from the census call's")
        gr = refs["regions"]
        region_diff = sorted(k for k in set(arrays) | set(gr) if k not in ("index", "regime", "group", "x_span", "notify_label")
                             and (k not in arrays or k not in gr or not same(arrays[k], gr[k])))
        if region_diff:
            problems.append(f"region arrays differ: {region_diff}")
        g7 = refs["guard_07"]
        guard_diff = sorted(k for k in set(guard) | set(g7) if k not in ("index", "notify_label")
                            and (k not in guard or k not in g7 or not same(guard[k], g7[k])))
        if guard_diff:
            problems.append(f"section 1.5 entries differ: {guard_diff}")
    for k in got:
        if got[k] != want[k]:
            problems.append(f"{k}: replay {got[k]} census {want[k]}")
    wins = result.get("stationary_windows") or []
    final = None
    if records:
        finals = [r for r in records if r["kind"] == "final"]
        final = finals[-1] if finals else None
    leaves = []
    for i, r in enumerate(regs):
        leaves.append({"start": H(r.start), "end": H(r.end), "depth": int(r.depth), "type": r.type,
                       "value": H(r.value), "value_own": H(r.value_own), "budget_err": H(r.budget_err),
                       "total_err": H(r.total_err), "abserr": H(r.abserr), "phase_err": H(r.phase_err),
                       "node_error_signed": H(r.node_error_signed), "window": bool(r.window),
                       "phase_limited": bool(r.phase_limited), "trusted": r.trusted,
                       "window_index": None if final is None else final["leaves"][i]["window_index"],
                       "fell": None if final is None else final["leaves"][i]["fell"],
                       "raw": None if final is None else H(final["leaves"][i]["raw"])})
    if final is not None:
        for i, r in enumerate(regs):
            if H(final["leaves"][i]["start"]) != H(r.start) or H(final["leaves"][i]["value"]) != H(r.value):
                problems.append("the instrument's final leaves are not the returned regions")
                break
    hdr = {"title": f"levin-phase-floor 07a Part A: replay of {sid} at {variant}",
           "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "module": str(module_path), "module_sha256": module_sha, "module_sha256_now": sha256(module_path),
           "worker_sha256": sha256(WK.__file__), "capture": c, "calls_manifest_sha256": sha256(ROOT / "state" / "calls_07a.json"),
           "variant_evidence": None if variant == "tree" else jsonable(VL.evidence()), "instrument": instrument,
           "python": sys.version}
    doc = {"header": hdr, "sid": sid, "set": c["set"], "variant": variant, "replay": got, "census": want,
           "region_keys_differing": region_diff, "guard_keys_differing": guard_diff, "problems": problems,
           "reproduces_census": not problems, "wall_s": wall, "atol": kw.get("atol"), "rtol": kw.get("rtol"),
           "x_span": [H(rec["x_span"][0]), H(rec["x_span"][1])],
           "n_windows": len(wins), "n_windows_stopped": sum(1 for w in wins if w.get("stopped")),
           "leaves_falling_at_stop": sum(int(w.get("leaves_falling_at_stop", 0) or 0) for w in wins if w.get("stopped")),
           "windows": jsonable([{k: v for k, v in w.items() if k != "cuts"} | {"n_cuts": len(w.get("cuts") or [])} for w in wins]),
           "num_window_pieces": int(result.get("num_window_pieces", 0)),
           "num_guarded_regions": int(result.get("num_guarded_regions", 0)),
           "num_untrusted_regions": result.get("num_untrusted_regions"),
           "ier2_windows": None if final is None else final["ier2_windows"],
           "window_counts": None if final is None else final["window_counts"],
           "shadow_counts_07": None if final is None else final["shadow_counts_07"],
           "num_splits": None if final is None else final["num_splits"],
           "regions_sha256": hashlib.sha256(pickle.dumps({k: arrays[k] for k in sorted(arrays)}, protocol=5)).hexdigest(),
           "leaves": leaves, "ray_initialized": ray.is_initialized()}
    vdir.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(jsonable(doc) | {"wall_s": wall}, indent=1))
    if instrument:
        inst_p.write_text(json.dumps({"header": hdr, "sid": sid, "variant": variant, "records": jsonable(records)}))
    print(f"{sid} {variant}: {'REPRODUCES' if not problems else 'DIFFERS: ' + '; '.join(problems)}; value {got['value']} "
          f"abserr {got['abserr']} conv {got['converged']} regions {got['num_regions']} depth {got['max_depth']} "
          f"stopped {doc['n_windows_stopped']} falling {doc['leaves_falling_at_stop']} {wall:.2f} s")
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
