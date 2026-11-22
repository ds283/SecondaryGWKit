"""levin-phase-floor prompt 07d Part A (sections 2.1, 2.5, 2.7): replay captured calls, in this process, at the tree's
module (07c's code) or at a variant (scripts/variants/, found in out/variants/variants.json), and compare each with its
07c census record. New in 07d, after 07c Part A's replay_07c.py (scripts/from_07c_measurement/, a record), changed in:
- the tree is this worktree (49ce9642, module d141a80b..., 07c's code); manifests state/calls_07d*.json; census records
  state/census_refs_07d*.pkl (07c's census, by census sid ``<unit>|<g|a><index>``);
- **many calls per process** (the module is loaded once; the census ran many calls per worker process too);
- compared (floats by hex, arrays by dtype, shape and bytes): value, abserr, num_regions, max_depth, converged, the two
  refusal counts and **the chase stop's three counts** (num_chase_stops, num_chase_stopped_leaves,
  num_chase_noise_bisections); the region arrays against the census record's ``regions`` entry (every key, which holds
  the three counts again); section 1.5's entries against ``guard_07``. ``reproduces_census`` iff all agree;
- an entry with ``override.atol`` None takes the atol of its census record (the counterpart at the other rtol);
- ``R:r31`` (R31's record) is compared with R32's record of the call (value, abserr, converged, num_regions), from 07c's
  rehearsal agent's refcap.json; it has no region arrays;
- **the stopped leaves:** 07c's census worker's instrument, as a pass-through: ``_chase_at_noise_level`` is wrapped at
  the module attribute, and on its first call in a Levin call the ``_adaptive_levin`` frame's ``_stop_chase_leaf`` is
  replaced by a pass-through that records each leaf it stops (span, route, trust before). Each returned leaf records
  ``chase_stopped``; nothing the pass-throughs return differs from the original's;
- a variant's constant ``_VARIANT_07D`` must equal its name; every result key starting ``num_07d_`` is recorded.
Writes OUT_DIR/<variant>/<sid with ':' and '|' replaced>.json; refuses to overwrite.
Usage: PYTHONPATH=<tree> python -B replay_07d.py VARIANT OUT_DIR (--sids-file F | SID ...)
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
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964")
TREE_LQ = TREE / "AdaptiveLevin" / "levin_quadrature.py"
TREE_LQ_SHA = "d141a80bf0484058a0d7f625926b2e283a4bb64114bc84ccfd81339a6f249be6"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def H(x):
    return None if x is None else float(x).hex()


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


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
    return cp.loads(blob)


def leaf_records(result, stopped):
    regs = result["regions"]
    wins = result.get("stationary_windows") or []
    edges = {}
    for wi, w in enumerate(wins):
        edges.setdefault(float(w["lo"]), []).append(wi)
        edges.setdefault(float(w["hi"]), []).append(wi)
    stopped_spans = {(s["start"], s["end"]) for s in stopped if s}
    out = []
    for r in regs:
        touch = sorted(set(edges.get(float(r.start), []) + edges.get(float(r.end), [])))
        out.append({"start": H(r.start), "end": H(r.end), "depth": int(r.depth), "type": r.type, "value": H(r.value),
                    "value_own": H(r.value_own), "value_kind": r.value_kind, "budget_err": H(r.budget_err),
                    "total_err": H(r.total_err), "own_err": H(r.own_err), "abserr": H(r.abserr), "phase_err": H(r.phase_err),
                    "phase_err_less_node": H(r.phase_err_less_node), "abserr_truncation": H(r.abserr_truncation),
                    "node_error_signed": H(r.node_error_signed), "resasc": H(r.resasc),
                    "delta1": H(r.delta1), "delta2": H(r.delta2), "convergence_ratio": H(r.convergence_ratio),
                    "declared_resolution": H(r.declared_resolution), "trusted": r.trusted, "window": bool(r.window),
                    "guarded": bool(r.guarded), "guard_at_depth_max": bool(r.guard_at_depth_max),
                    "phase_limited": bool(r.phase_limited), "touches_window": touch,
                    "chase_stopped": (float(r.start), float(r.end)) in stopped_spans})
    return out


class StopInstrument:
    """07c's census worker's stop instrument, as pass-throughs (see the module docstring)."""

    def __init__(self, LQ):
        self.LQ = LQ
        self.orig = getattr(LQ, "_chase_at_noise_level", None)
        self.state = None

    def begin(self):
        self.state = {"frame": None, "fail": None, "recs": []}

    def install(self):
        if self.orig is None:
            return
        inst = self
        LQ = self.LQ
        lq_file = LQ.__file__

        def install_stop(st):
            fr = sys._getframe(2)
            while fr is not None and not (fr.f_code.co_name == "_adaptive_levin" and fr.f_code.co_filename == lq_file):
                fr = fr.f_back
            if fr is None:
                st["fail"] = "no _adaptive_levin frame"
                return
            loc = fr.f_locals
            orig_stop = loc["_stop_chase_leaf"]
            recs = st["recs"]

            def w_stop(leaf):
                try:
                    reg = leaf.region
                    before = (bool(reg._phase_limited), reg._trusted)
                except Exception:
                    before = None
                r = orig_stop(leaf)
                try:
                    if before is None:
                        recs.append(None)
                    elif not before[0] and reg._phase_limited:
                        recs.append({"start": float(leaf.start), "end": float(leaf.end), "depth": int(leaf.depth),
                                     "type": reg.type, "trusted_before": before[1]})
                except Exception:
                    recs.append(None)
                return r

            loc["_stop_chase_leaf"] = w_stop
            st["frame"] = True

        def w_chase(*a, **k):
            r = inst.orig(*a, **k)
            st = inst.state
            if st is not None and st["frame"] is None and st["fail"] is None:
                try:
                    install_stop(st)
                except Exception as e:  # never let the record change the computation
                    st["fail"] = repr(e)
            return r

        LQ._chase_at_noise_level = w_chase


def manifest():
    man = {}
    for p in sorted((ROOT / "state").glob("calls_07d*.json")):
        for sid, c in json.loads(p.read_text())["calls"].items():
            if sid in man:
                raise SystemExit(f"{sid} in two manifests")
            man[sid] = dict(c, _manifest=str(p), _manifest_sha256=sha256(p))
    return man


def census_records():
    out = {}
    for p in sorted((ROOT / "state").glob("census_refs_07d*.pkl")):
        out.update({k: dict(v, _pkl=str(p)) for k, v in pickle.loads(p.read_bytes())["calls"].items()})
    return out


def main():
    variant, out_dir = sys.argv[1], Path(sys.argv[2])
    rest = sys.argv[3:]
    if rest and rest[0] == "--sids-file":
        sids = [s for s in Path(rest[1]).read_text().split() if s]
    else:
        sids = rest
    if os.environ.get("PYTHONPATH", "").split(":")[0] != str(TREE):
        raise SystemExit("PYTHONPATH must start with the tree")
    sys.path.insert(1, str(HERE))
    if sha256(TREE_LQ) != TREE_LQ_SHA:
        raise SystemExit("the tree's levin_quadrature.py has another SHA-256")
    vdir = out_dir / variant
    for sid in sids:
        if (vdir / f"{fname(sid)}.json").exists():
            raise SystemExit(f"{sid}: output exists; not overwritten")
    man = manifest()
    if variant == "tree":
        import AdaptiveLevin.levin_quadrature as LQ

        module_path, module_sha = TREE_LQ, TREE_LQ_SHA
        VL = None
    else:
        import variant_loader as VL

        vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
        LQ = VL.install(Path(vj["path"]), vj["sha256"], f"07d_{variant}")
        module_path, module_sha = Path(vj["path"]), vj["sha256"]
        if getattr(LQ, "_VARIANT_07D", None) != variant:
            raise SystemExit("the variant's constant is not its name")
    import ray

    import worker as WK  # 07c's census worker's recorders, unchanged

    import AdaptiveLevin

    if AdaptiveLevin.levin_quadrature is not LQ or Path(LQ.__file__).resolve() != Path(module_path).resolve():
        raise SystemExit("the module the replay runs is not the one named")
    census = census_records()
    inst = StopInstrument(LQ)
    inst.install()
    vdir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for sid in sids:
        c = man[sid]
        rec = load_capture(c)
        kw = dict(rec["kw"])
        cen = None if c["census"].startswith("r32:") else census[c["census"]]
        if c["override"]:
            kw["rtol"] = c["override"]["rtol"]
            kw["atol"] = c["override"]["atol"] if c["override"].get("atol") is not None else cen["call"]["atol"]
        inst.begin()
        t0 = time.perf_counter()
        result = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **kw)
        wall = time.perf_counter() - t0
        st = inst.state
        inst.state = None
        regs = result["regions"]
        call = {"index": rec["index"], "notify_label": rec["notify_label"], "result": result, "kw": kw,
                "default_depth_max": getattr(LQ, "DEFAULT_LEVIN_MAX_DEPTH", None)}
        arrays = {**WK._region_arrays_06g(call), **WK._region_arrays_07(call)}
        guard = WK._guard_record_07(call)
        got = {"value": H(result["value"]), "abserr": H(result["abserr"]), "num_regions": int(result["num_regions"]),
               "max_depth": int(max(r.depth for r in regs)), "converged": bool(result["converged"]),
               "num_edge_rule_refusals": int(result.get("num_edge_rule_refusals", -1)),
               "num_range_rule_refusals": int(result.get("num_range_rule_refusals", -1)),
               "num_chase_stops": int(result.get("num_chase_stops", -1)),
               "num_chase_stopped_leaves": int(result.get("num_chase_stopped_leaves", -1)),
               "num_chase_noise_bisections": int(result.get("num_chase_noise_bisections", -1))}
        problems = []
        if cen is None:
            r32 = c["r32_record"]
            want = {"value": r32["value"], "abserr": r32["abserr"], "num_regions": r32["regions"],
                    "converged": bool(r32["converged"]),
                    **{k: int(r32[k]) for k in ("num_edge_rule_refusals", "num_range_rule_refusals", "num_chase_stops",
                                                "num_chase_stopped_leaves", "num_chase_noise_bisections")}}
            for k in want:
                if got[k] != want[k]:
                    problems.append(f"{k}: replay {got[k]} R32 {want[k]}")
            region_diff, guard_diff, which, unit = [], [], "R32 (refcap.json)", c["census"]
        else:
            cr = cen["call"]
            want = {"value": cr["value"]["hex"], "abserr": cr["abserr"]["hex"], "num_regions": cr["num_regions"],
                    "max_depth": cr["max_depth"], "converged": bool(cr["converged"]),
                    "num_edge_rule_refusals": int(cr["num_edge_rule_refusals"]),
                    "num_range_rule_refusals": int(cr["num_range_rule_refusals"]),
                    "num_chase_stops": int(cr["num_chase_stops"]),
                    "num_chase_stopped_leaves": int(cr["num_chase_stopped_leaves"]),
                    "num_chase_noise_bisections": int(cr["num_chase_noise_bisections"])}
            if rec["notify_label"] != cr["notify_label"]:
                problems.append("notify_label differs from the census call's")
            gr = cen["regions"]
            region_diff = sorted(k for k in set(arrays) | set(gr) if k not in ("index", "regime", "group", "x_span", "notify_label")
                                 and (k not in arrays or k not in gr or not same(arrays[k], gr[k])))
            if region_diff:
                problems.append(f"region arrays differ: {region_diff}")
            g7 = cen["guard_07"]
            guard_diff = sorted(k for k in set(guard) | set(g7) if k not in ("index", "notify_label")
                                and (k not in guard or k not in g7 or not same(guard[k], g7[k])))
            if guard_diff:
                problems.append(f"section 1.5 entries differ: {guard_diff}")
            for k in want:
                if got[k] != want[k]:
                    problems.append(f"{k}: replay {got[k]} census {want[k]}")
            which, unit = "07c census", cen["unit"]
        stopped = st["recs"] if st else []
        wins = result.get("stationary_windows") or []
        hdr = {"title": f"levin-phase-floor 07d Part A: replay of {sid} at {variant}",
               "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
               "variant_name": variant, "module": str(module_path), "module_sha256": module_sha,
               "module_sha256_now": sha256(module_path), "worker_sha256": sha256(WK.__file__),
               "capture": {k: v for k, v in c.items() if not k.startswith("_")}, "manifest": c["_manifest"],
               "manifest_sha256": c["_manifest_sha256"], "census_pkl": None if cen is None else cen["_pkl"],
               "variant_evidence": None if VL is None else jsonable(VL.evidence()), "python": sys.version}
        doc = {"header": hdr, "sid": sid, "set": c["set"], "variant": variant, "replay": got, "census": want,
               "census_which": which, "census_unit": unit, "region_keys_differing": region_diff,
               "guard_keys_differing": guard_diff, "problems": problems, "reproduces_census": not problems,
               "wall_s": wall, "atol": kw.get("atol"), "rtol": kw.get("rtol"), "chebyshev_order": kw.get("chebyshev_order"),
               "notify_label": rec["notify_label"], "x_span": [H(rec["x_span"][0]), H(rec["x_span"][1])],
               "n_windows": len(wins), "n_windows_stopped": sum(1 for w in wins if w.get("stopped")),
               "windows": jsonable([{k: v for k, v in w.items() if k != "cuts"} | {"n_cuts": len(w.get("cuts") or [])} for w in wins]),
               "num_window_pieces": int(result.get("num_window_pieces", 0)),
               "num_guarded_regions": int(result.get("num_guarded_regions", 0)),
               "num_guard_at_depth_max": int(result.get("num_guard_at_depth_max", 0)),
               "num_untrusted_regions": result.get("num_untrusted_regions"),
               "num_subregion_solves": result.get("num_subregion_solves"),
               "extra_07d": {k: v for k, v in result.items() if k.startswith("num_07d_")},
               "stop_instrument": {"fail": None if st is None else st["fail"], "n_recorded": len(stopped),
                                   "n_unrecorded": sum(1 for s in stopped if s is None)},
               "stopped_leaves": [s for s in stopped if s],
               "regions_sha256": hashlib.sha256(pickle.dumps({k: arrays[k] for k in sorted(arrays)}, protocol=5)).hexdigest(),
               "leaves": leaf_records(result, stopped), "ray_initialized": ray.is_initialized()}
        (vdir / f"{fname(sid)}.json").write_text(json.dumps(jsonable(doc) | {"wall_s": wall}))
        n_ok += not problems
        print(f"{sid} {variant}: {'REPRODUCES' if not problems else 'DIFFERS: ' + '; '.join(problems)}; value {got['value']} "
              f"abserr {got['abserr']} conv {got['converged']} regions {got['num_regions']} depth {got['max_depth']} "
              f"chase {got['num_chase_stops']}/{got['num_chase_stopped_leaves']}/{got['num_chase_noise_bisections']} "
              f"stopped recorded {len(stopped)} {wall:.2f} s", flush=True)
        if ray.is_initialized():
            raise SystemExit("Ray was initialised")
    print(f"{n_ok} of {len(sids)} reproduce their census record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
