"""levin-phase-floor prompt 07b Part A, sections 2.1-2.3: replay one captured call, in this process, at the tree's module
(07a's code: variant (a)), at 07's module (``07``, section 2.1 item 3), or at a variant (scripts/variants/), and compare it
with its census record. Copied from 07a Part A's replay_07a.py (a record), changed where marked "prompt 07b":
- the tree is this worktree (436ed5d4, module 4ea1948b...); the manifest is state/calls_07b.json and the census records
  state/census_refs_07b.pkl (07a's census, or 07's for ``i3-34599-07``);
- the recorders are 07a's census worker's ``_region_arrays_06g``, ``_region_arrays_07`` and ``_guard_record_07``;
- 07a's instrument (``_MEASURE_07A``) is gone; every returned leaf is recorded from its ``used_interval`` (value, own
  value, e', abserr, floor, the trust check's delta1, delta2, r and declared resolution, trusted, value kind, window,
  guarded), with the index of the window whose edge it touches (``touches_window``: the windows of
  ``stationary_windows`` whose ``lo`` or ``hi`` equals one of its ends);
- a variant's module constant ``_VARIANT_07B`` must equal its name; a variant may expose ``_MEASURE_07B`` (a list set
  for the call, its records written to <sid>.instrument.json).

Compared (each by hex for floats; arrays by dtype, shape and bytes): value, abserr, num_regions, max_depth, converged; the
region arrays against the census record's ``regions`` entry, every key; section 1.5's entries against ``guard_07``.
``reproduces_census`` is true iff all agree. At ``tree`` (and ``07`` on ``i3-34599-07``) a false is section 8's stop.

Writes OUT_DIR/<variant>/<sid>.json; refuses to overwrite. Usage:
PYTHONPATH=<tree> python -B replay_07b.py SID VARIANT OUT_DIR [--instrument]
"""

import datetime
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")  # prompt 07b
TREE_LQ = TREE / "AdaptiveLevin" / "levin_quadrature.py"
TREE_LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"  # prompt 07b: 07a's module
V07 = HERE / "variants" / "levin_quadrature_07.py"  # prompt 07b
V07_SHA = "4e5da371f51e6e3dc20e73579eb45e0e985e2a3b18571ee29eb86b3d3e9c93fc"  # prompt 07b: 07's module


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def H(x):
    if x is None:
        return None
    return float(x).hex()


def jsonable(x):
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


def load_capture(c):
    import ray.cloudpickle as cp

    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c["override"]:
        kw["rtol"] = c["override"]["rtol"]
        kw["atol"] = c["override"]["atol"]
    return rec, kw


def leaf_records(result):
    regs = result["regions"]
    wins = result.get("stationary_windows") or []
    edges = {}
    for wi, w in enumerate(wins):
        edges.setdefault(float(w["lo"]), []).append(wi)
        edges.setdefault(float(w["hi"]), []).append(wi)
    out = []
    for r in regs:
        touch = sorted(set(edges.get(float(r.start), []) + edges.get(float(r.end), [])))
        out.append({"start": H(r.start), "end": H(r.end), "depth": int(r.depth), "type": r.type, "value": H(r.value),
                    "value_own": H(r.value_own), "value_kind": r.value_kind, "budget_err": H(r.budget_err),
                    "total_err": H(r.total_err), "own_err": H(getattr(r, "own_err", None)), "abserr": H(r.abserr), "phase_err": H(r.phase_err),
                    "abserr_truncation": H(r.abserr_truncation), "node_error_signed": H(r.node_error_signed),
                    "delta1": H(r.delta1), "delta2": H(r.delta2), "convergence_ratio": H(r.convergence_ratio),
                    "declared_resolution": H(r.declared_resolution), "trusted": r.trusted, "window": bool(r.window),
                    "guarded": bool(r.guarded), "phase_limited": bool(r.phase_limited), "touches_window": touch,
                    "edge_lo_is": [("hi" if float(r.start) == float(w["hi"]) else "lo") for w in wins if float(r.start) in (float(w["lo"]), float(w["hi"]))],
                    "edge_hi_is": [("lo" if float(r.end) == float(w["lo"]) else "hi") for w in wins if float(r.end) in (float(w["lo"]), float(w["hi"]))]})
    return out


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
    manifest = json.loads((ROOT / "state" / "calls_07b.json").read_text())
    c = manifest["calls"][sid]
    vl_evidence = None
    if variant == "tree":
        if instrument:
            raise SystemExit("--instrument needs a variant")
        import AdaptiveLevin.levin_quadrature as LQ

        module_path, module_sha = TREE_LQ, TREE_LQ_SHA
    else:
        import variant_loader as VL

        if variant == "07":
            vpath, vsha = V07, V07_SHA
        else:
            vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
            vpath, vsha = Path(vj["path"]), vj["sha256"]
        LQ = VL.install(vpath, vsha, f"07b_{variant}")
        module_path, module_sha = Path(vpath), vsha
        if variant != "07" and getattr(LQ, "_VARIANT_07B", None) != variant:
            raise SystemExit("the variant's constant is not its name")
    import ray

    import worker as WK  # 07a's census worker's recorders, unchanged

    rec, kw = load_capture(c)
    import AdaptiveLevin

    if AdaptiveLevin.levin_quadrature is not LQ or Path(LQ.__file__).resolve() != Path(module_path).resolve():
        raise SystemExit("the module the replay runs is not the one named")
    if instrument:
        LQ._MEASURE_07B = []
    t0 = time.perf_counter()
    try:
        result = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **kw)
    finally:
        records = getattr(LQ, "_MEASURE_07B", None) if instrument else None
        if instrument:
            LQ._MEASURE_07B = None
    wall = time.perf_counter() - t0
    regs = result["regions"]
    call = {"index": rec["index"], "notify_label": rec["notify_label"], "result": result, "kw": kw,
            "default_depth_max": getattr(LQ, "DEFAULT_LEVIN_MAX_DEPTH", None)}
    arrays = {**WK._region_arrays_06g(call), **WK._region_arrays_07(call)}
    guard = WK._guard_record_07(call)
    got = {"value": H(result["value"]), "abserr": H(result["abserr"]), "num_regions": int(result["num_regions"]),
           "max_depth": int(max(r.depth for r in regs)), "converged": bool(result["converged"])}
    problems = []
    refs = pickle.loads((ROOT / "state" / "census_refs_07b.pkl").read_bytes())["calls"][c["census"]]
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
    hdr = {"title": f"levin-phase-floor 07b Part A: replay of {sid} at {variant}",
           "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "module": str(module_path), "module_sha256": module_sha, "module_sha256_now": sha256(module_path),
           "worker_sha256": sha256(WK.__file__), "capture": c, "calls_manifest_sha256": sha256(ROOT / "state" / "calls_07b.json"),
           "census_refs_sha256": sha256(ROOT / "state" / "census_refs_07b.pkl"),
           "variant_evidence": None if variant == "tree" else jsonable(VL.evidence()), "instrument": instrument,
           "python": sys.version}
    doc = {"header": hdr, "sid": sid, "set": c["set"], "variant": variant, "replay": got, "census": want,
           "census_which": refs["which"], "census_unit": refs["unit"],
           "region_keys_differing": region_diff, "guard_keys_differing": guard_diff, "problems": problems,
           "reproduces_census": not problems, "wall_s": wall, "atol": kw.get("atol"), "rtol": kw.get("rtol"),
           "x_span": [H(rec["x_span"][0]), H(rec["x_span"][1])],
           "n_windows": len(wins), "n_windows_stopped": sum(1 for w in wins if w.get("stopped")),
           "windows": jsonable([{k: v for k, v in w.items() if k != "cuts"} | {"n_cuts": len(w.get("cuts") or [])} for w in wins]),
           "stationary_points": jsonable(result.get("stationary_points") or []),
           "num_window_pieces": int(result.get("num_window_pieces", 0)),
           "num_guarded_regions": int(result.get("num_guarded_regions", 0)),
           "num_untrusted_regions": result.get("num_untrusted_regions"),
           "num_subregion_solves": result.get("num_subregion_solves"),
           "regions_sha256": hashlib.sha256(pickle.dumps({k: arrays[k] for k in sorted(arrays)}, protocol=5)).hexdigest(),
           "leaves": leaf_records(result), "ray_initialized": ray.is_initialized()}
    vdir.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(jsonable(doc) | {"wall_s": wall}, indent=1))
    if instrument:
        inst_p.write_text(json.dumps({"header": hdr, "sid": sid, "variant": variant, "records": jsonable(records)}))
    print(f"{sid} {variant}: {'REPRODUCES' if not problems else 'DIFFERS: ' + '; '.join(problems)}; value {got['value']} "
          f"abserr {got['abserr']} conv {got['converged']} regions {got['num_regions']} depth {got['max_depth']} "
          f"windows {doc['n_windows']} stopped {doc['n_windows_stopped']} {wall:.2f} s", flush=True)
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
