"""levin-phase-floor prompt 07d Part A' (section 9.2): replay Part A's captured calls in this process, at the tree or at a
variant of out/variants/variants.json, and compare each with **Part A's tree replay of the same call**
(Part A's out/replays/tree/<sid>.json, which reproduced 07c's census record bit for bit on all 4,220 calls). New in Part A';
its helpers (``StopInstrument``, ``leaf_records``, ``H``, ``jsonable``, ``load_capture``, ``fname``) are Part A's
replay_07d.py, copied byte for byte into scripts/ and imported, not edited.

Compared (strings of float.hex): Part A's ``replay`` dict (value, abserr, num_regions, max_depth, converged, the two
refusal counts, the chase stop's three counts) and its ``leaves`` list (every returned region's span, depth, type, value,
value_own, value_kind, budget_err, total_err, own_err, abserr, phase_err, phase_err_less_node, truncation, node error,
resasc, delta1, delta2, r, declared resolution, trusted, window, guarded, guard at depth_max, phase_limited, touches_window,
chase_stopped). ``equals_tree`` iff both are equal. For the tree and I6 every call must be equal.

For **I6** (the instrumented variant) the module dict ``_A2_RECORDS`` is set before each call and its records written:
``made`` (seq, a, b, depth, is_direct, phase_span, window, guarded, guard_at_depth_max, value_own, abserr, phase_err,
truncation, own_err, total_err at making), ``checks`` (seq, a, b, depth, is_direct, forced, V_P, V_L + V_R, Q, delta1,
delta2, r, nu_B, trusted, refused by a rate rule, rule, residual, phase_err, the six sub-solves' is_direct) and
``removed`` (seq, trusted, total_err, depth at bisection).

Writes OUT_DIR/<variant>/<sid>.json; refuses to overwrite.
Usage: PYTHONPATH=<tree>:<scripts> python -B replay_a2.py VARIANT OUT_DIR --sids-file F
"""

import datetime
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PARTA, ROOT, TREE, TREE_LQ, TREE_LQ_SHA, sha256  # noqa: E402


def main():
    variant, out_dir = sys.argv[1], Path(sys.argv[2])
    sids = [s for s in Path(sys.argv[4]).read_text().split() if s]
    vdir = out_dir / variant
    for sid in sids:
        from replay_07d import fname as _fn  # noqa: E402  (import-time: constants only)

        if (vdir / f"{_fn(sid)}.json").exists():
            raise SystemExit(f"{sid}: output exists; not overwritten")
    if sha256(TREE_LQ) != TREE_LQ_SHA:
        raise SystemExit("the tree's levin_quadrature.py has another SHA-256")
    if variant == "tree":
        import AdaptiveLevin.levin_quadrature as LQ

        module_path, module_sha, VL = TREE_LQ, TREE_LQ_SHA, None
    else:
        import variant_loader as VL

        vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
        LQ = VL.install(Path(vj["path"]), vj["sha256"], f"a2_{variant}")
        module_path, module_sha = Path(vj["path"]), vj["sha256"]
        if getattr(LQ, "_VARIANT_A2", None) != variant:
            raise SystemExit("the variant's constant is not its name")
    import ray

    import AdaptiveLevin
    from replay_07d import H, StopInstrument, fname, jsonable, leaf_records, load_capture

    if AdaptiveLevin.levin_quadrature is not LQ or Path(LQ.__file__).resolve() != Path(module_path).resolve():
        raise SystemExit("the module the replay runs is not the one named")
    if Path(AdaptiveLevin.__file__).resolve().parent.parent != TREE:
        raise SystemExit("AdaptiveLevin is not the tree's")
    man = {}
    for p in sorted((PARTA / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    census = {}
    import pickle

    for p in sorted((PARTA / "state").glob("census_refs_07d*.pkl")):
        census.update(pickle.loads(p.read_bytes())["calls"])
    inst = StopInstrument(LQ)
    inst.install()
    vdir.mkdir(parents=True, exist_ok=True)
    n_eq = 0
    for sid in sids:
        c = man[sid]
        rec = load_capture(c)
        kw = dict(rec["kw"])
        if c["override"]:
            cen = None if c["census"].startswith("r32:") else census[c["census"]]
            kw["rtol"] = c["override"]["rtol"]
            kw["atol"] = c["override"]["atol"] if c["override"].get("atol") is not None else cen["call"]["atol"]
        tree_p = PARTA / "out" / "replays" / "tree" / f"{fname(sid)}.json"
        tree_doc = json.loads(tree_p.read_text())
        records = None
        if variant == "I6":
            records = {"made": [], "checks": [], "removed": []}
            LQ._A2_RECORDS = records
        # (K) and (G): the harness supplies the call's knots, of the variant's kinds (variants.json "knot_kinds"), from
        # leaf_a2.typed_knots (production_edges' walk); None where it does not apply (analytic blocks), which bisect and
        # gate as the tree does
        n_knots_supplied = None
        if getattr(LQ, "_A2_KNOTS_ON", False) or getattr(LQ, "_A2_GATE_KNOTS", None) is not None:
            import numpy as np
            from leaf_a2 import typed_knots

            try:
                tk = typed_knots(rec)
                lo_, hi_ = min(rec["x_span"]), max(rec["x_span"])
                ks = sorted({x for k in vj["knot_kinds"] for x in tk[k] if lo_ < x < hi_})
                LQ._A2_KNOTS = np.asarray(ks, dtype=float)
                n_knots_supplied = len(ks)
            except Exception:  # noqa: BLE001
                LQ._A2_KNOTS = None
        inst.begin()
        t0 = time.perf_counter()
        try:
            result = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **kw)
        finally:
            if variant == "I6":
                LQ._A2_RECORDS = None
            if hasattr(LQ, "_A2_KNOTS"):
                LQ._A2_KNOTS = None
        wall = time.perf_counter() - t0
        st = inst.state
        inst.state = None
        regs = result["regions"]
        got = {"value": H(result["value"]), "abserr": H(result["abserr"]), "num_regions": int(result["num_regions"]),
               "max_depth": int(max(r.depth for r in regs)), "converged": bool(result["converged"]),
               "num_edge_rule_refusals": int(result.get("num_edge_rule_refusals", -1)),
               "num_range_rule_refusals": int(result.get("num_range_rule_refusals", -1)),
               "num_chase_stops": int(result.get("num_chase_stops", -1)),
               "num_chase_stopped_leaves": int(result.get("num_chase_stopped_leaves", -1)),
               "num_chase_noise_bisections": int(result.get("num_chase_noise_bisections", -1))}
        stopped = st["recs"] if st else []
        leaves = jsonable(leaf_records(result, stopped))
        diff_keys = sorted(k for k in got if got[k] != tree_doc["replay"].get(k))
        leaves_equal = leaves == tree_doc["leaves"]
        equals_tree = not diff_keys and leaves_equal
        doc = {"header": {"title": f"levin-phase-floor 07d Part A': replay of {sid} at {variant}",
                          "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "host": socket.gethostname(), "script": str(Path(__file__).resolve()),
                          "script_sha256": sha256(__file__), "replay_07d_sha256": sha256(HERE / "replay_07d.py"),
                          "variant_name": variant, "module": str(module_path), "module_sha256": module_sha,
                          "module_sha256_now": sha256(module_path), "tree_replay": str(tree_p),
                          "tree_replay_sha256": sha256(tree_p), "capture": c,
                          "variant_evidence": None if VL is None else jsonable(VL.evidence())},
               "sid": sid, "set": c["set"], "variant": variant, "replay": got, "tree_replay": tree_doc["replay"],
               "census": tree_doc.get("census"), "keys_differing_from_tree": diff_keys, "leaves_equal_tree": leaves_equal,
               "equals_tree": equals_tree, "wall_s": wall, "tree_wall_s": tree_doc.get("wall_s"),
               "rtol": kw.get("rtol"), "atol": kw.get("atol"), "chebyshev_order": kw.get("chebyshev_order"),
               "notify_label": rec["notify_label"], "x_span": [H(rec["x_span"][0]), H(rec["x_span"][1])],
               "num_subregion_solves": result.get("num_subregion_solves"),
               "num_untrusted_regions": result.get("num_untrusted_regions"),
               "extra_a2": {k: v for k, v in result.items() if str(k).startswith("num_a2_")},
               "n_knots_supplied": n_knots_supplied,
               "stop_instrument": {"fail": None if st is None else st["fail"], "n_recorded": len(stopped)},
               "leaves": leaves, "a2_records": jsonable(records), "ray_initialized": ray.is_initialized()}
        (vdir / f"{fname(sid)}.json").write_text(json.dumps(jsonable(doc) | {"wall_s": wall}))
        n_eq += equals_tree
        print(f"{sid} {variant}: {'EQUALS TREE' if equals_tree else 'DIFFERS ' + str(diff_keys) + ' leaves ' + str(leaves_equal)}"
              f" {wall:.2f} s" + (f" made {len(records['made'])} checks {len(records['checks'])}" if records else ""),
              flush=True)
        if ray.is_initialized():
            raise SystemExit("Ray was initialised")
    print(f"{n_eq} of {len(sids)} equal Part A's tree replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
