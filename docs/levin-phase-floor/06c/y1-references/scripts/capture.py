"""Y1 references, step 2: re-run each Y1 row's QuadSourceIntegral in-process in one tree, on that
tree's store copy, and capture every adaptive_levin_sincos call.

Run with PYTHONPATH=<tree>:<this scripts dir>; the tree decides the store (fef401c -> R17's copy,
e5764a6 -> R18's copy). Route as the census's: harness.standin_ray() (Ray never initialised; the
pool's actors, ray.get, ray.put, ray.wait and compute_QuadSource_integral.remote stood in) and a
read-only datastorekit ShardedPool on the copy; rebuild.py (the census's, rehearsal scope) builds
main.py's context and each row's work object and compute payload, with every check against the
stored row. The compute is ``work.compute(payload=...)`` then ``ray.get``, as the census worker.

The wrapper on ``ComputeTargets.QuadSourceIntegral.adaptive_levin_sincos`` (the name the call site
uses) records, per call, in order: the span (hex), the notify_label, the keyword arguments, and the
result's value, abserr (hex), num_regions, max_depth, converged, phase_limited and every returned
region (start, end, depth, type, abserr, phase_err, phase_limited, total_err if the code has it).
Per row it checks that the recomputed total and total_abserr equal the store's bit for bit, and
that every group's (1 + z_response) * value and * abserr equal the store's metadata bit for bit.

With --pickle SELECTION_JSON (a list of {"row", "index"}), the listed calls' inputs (span, f,
theta dict, kwargs) and results are pickled with harness._GroupPickler, one file per call, into
OUT_DIR/inputs/, for replay.py.

Usage: PYTHONPATH=<tree>:<scripts> python -B capture.py ROWS_JSON OUT_JSON [--rows Y01,Y02,..]
       [--pickle SELECTION_JSON --pickle-dir DIR]
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yprov import header, which_tree  # noqa: E402


def hx(x):
    return None if x is None else float(x).hex()


def region_summary(r):
    d = {"start": hx(r.start), "end": hx(r.end), "depth": int(r.depth), "type": str(r.type),
         "abserr": hx(r.abserr), "phase_err": hx(r.phase_err), "phase_limited": bool(r.phase_limited)}
    try:
        d["total_err"] = hx(r.total_err)
    except Exception:
        d["total_err"] = None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_json")
    ap.add_argument("out_json")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--pickle", default=None)
    ap.add_argument("--pickle-dir", default=None)
    a = ap.parse_args()

    tree = which_tree()
    if tree is None:
        raise SystemExit("PYTHONPATH must name one of the two trees")
    inputs = [Path(a.rows_json)] + ([Path(a.pickle)] if a.pickle else [])
    hdr = header(Path(__file__).resolve(), f"Y1 references: capture in tree {tree}", inputs)
    for l in hdr:
        print(l, flush=True)

    import ray
    import rebuild
    from harness import QSI_MOD, _GroupPickler, close_pool, open_readonly_pool, standin_ray
    import AdaptiveLevin.levin_quadrature as LQ

    print(f"# AdaptiveLevin.levin_quadrature {LQ.__file__}")
    print(f"# ComputeTargets.QuadSourceIntegral {QSI_MOD.__file__}")
    print(f"# rebuild {rebuild.__file__}; K sample {rebuild.A3V2_K_SAMPLE}; zend {rebuild.A3V2_ZEND}")

    rows = json.loads(Path(a.rows_json).read_text())
    if a.rows:
        want = set(a.rows.split(","))
        rows = [r for r in rows if r["id"] in want]
    to_pickle = set()
    if a.pickle:
        to_pickle = {(e["row"], e["index"]) for e in json.loads(Path(a.pickle).read_text())}
        pdir = Path(a.pickle_dir)
        pdir.mkdir(parents=True, exist_ok=True)

    primary = rebuild.use_store(tree)
    print(f"# store copy {primary}")
    orig = QSI_MOD.adaptive_levin_sincos
    holder = {"calls": None}

    def w_levin(*args, **kw):
        result = orig(*args, **kw)
        calls = holder["calls"]
        if calls is not None:
            x_span = args[0] if len(args) > 0 else kw["x_span"]
            f_ = args[1] if len(args) > 1 else kw["f"]
            theta = args[2] if len(args) > 2 else kw["theta"]
            calls.append({"x_span": tuple(x_span), "f": f_, "theta": theta,
                          "kw": {k: v for k, v in kw.items() if k not in ("x_span", "f", "theta")},
                          "result": result})
        return result

    out = {"header": hdr, "tree": tree, "store": str(primary), "rows": []}
    with standin_ray() as cluster:
        pool, printed = open_readonly_pool(primary, cluster)
        print(printed.strip(), flush=True)
        try:
            t0 = time.perf_counter()
            ctx = rebuild.build_context(pool, log=lambda s: print(s, flush=True))
            print(f"# context built in {time.perf_counter() - t0:.1f} s; checks {ctx['checks']}", flush=True)
            for R in rows:
                rid = R["id"]
                S = R[tree]
                rec = {"row": rid, "shard": S["shard"], "serial": S["serial"]}
                try:
                    t1 = time.perf_counter()
                    rb = rebuild.rebuild_row(ctx, pool, S["shard"], S["serial"], log=lambda s: print(s, flush=True))
                    rec["rebuild_checks"] = [[n, ok, d] for n, ok, d in rb["checks"]]
                    calls = []
                    holder["calls"] = calls
                    QSI_MOD.adaptive_levin_sincos = w_levin
                    try:
                        res = ray.get(rb["work"].compute(payload=rb["compute_payload"]))
                    finally:
                        QSI_MOD.adaptive_levin_sincos = orig
                        holder["calls"] = None
                    rec["seconds"] = time.perf_counter() - t1
                    zr = float(rb["z_response"].z)
                    scale = 1.0 + zr
                    rec["z_response"] = hx(zr)
                    rec["total"] = hx(res["total"])
                    rec["total_abserr"] = hx(res["total_abserr"])
                    rec["numeric_quad"] = hx(res["numeric_quad"])
                    rec["WKB_Levin"] = hx(res["WKB_Levin"])
                    rec["total_equals_store"] = (hx(res["total"]), hx(res["total_abserr"])) == (S["total"], S["total_abserr"])
                    # groups, in the metadata's order, against the calls in call order
                    # the analytic oracle's Bessel calls (_three_bessel_Levin) go through the same name;
                    # the phase-group calls are those whose notify_label names a group
                    rec["other_calls"] = len([c for c in calls if ", group " not in (c["kw"].get("notify_label") or "")])
                    calls = [c for c in calls if ", group " in (c["kw"].get("notify_label") or "")]
                    md_groups = [(i, g) for i, s in enumerate(res["metadata"]["WKB_Levin"]["subintervals"]) for g in s["groups"]]
                    st_groups = [(i, g) for i, s in enumerate(S["subintervals"]) for g in s["groups"]]
                    if len(md_groups) != len(calls) or len(st_groups) != len(calls):
                        raise RuntimeError(f"{len(calls)} calls, {len(md_groups)} groups recomputed, {len(st_groups)} stored")
                    crecs = []
                    for idx, (c, (si, g), (si2, gs)) in enumerate(zip(calls, md_groups, st_groups)):
                        r_ = c["result"]
                        lab = c["kw"].get("notify_label") or ""
                        if not lab.endswith(f"group {g['label']}") or g["label"] != gs["label"] or si != si2:
                            raise RuntimeError(f"call {idx}: label {lab!r} vs group {g['label']} / stored {gs['label']}")
                        sv, sa = scale * r_["value"], scale * r_["abserr"]
                        cr = {
                            "index": idx, "sub": si, "label": g["label"], "notify_label": lab,
                            "x_span": [hx(c["x_span"][0]), hx(c["x_span"][1])],
                            "kw": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in c["kw"].items()},
                            "atol": hx(c["kw"].get("atol")), "rtol": hx(c["kw"].get("rtol")),
                            "value": hx(r_["value"]), "abserr": hx(r_["abserr"]),
                            "num_regions": int(r_["num_regions"]), "max_depth": int(r_["max_depth"]),
                            "converged": bool(r_["converged"]), "phase_limited": bool(r_["phase_limited"]),
                            "scaled_value": hx(sv), "scaled_abserr": hx(sa),
                            "scaled_equals_store": (hx(sv), hx(sa)) == (gs["value"], gs["abserr"]),
                            "regions_equal_store": int(r_["num_regions"]) == int(gs["regions"]),
                            "regions": [region_summary(x) for x in r_["regions"]],
                        }
                        if (rid, idx) in to_pickle:
                            buf = io.BytesIO()
                            _GroupPickler(buf, protocol=5).dump({
                                "row": rid, "tree": tree, "index": idx, "label": g["label"], "notify_label": lab,
                                "x_span": c["x_span"], "f": c["f"], "theta": c["theta"], "kw": c["kw"],
                                "value": hx(r_["value"]), "abserr": hx(r_["abserr"]),
                                "num_regions": int(r_["num_regions"]), "max_depth": int(r_["max_depth"]),
                                "converged": bool(r_["converged"]), "scale": hx(scale),
                                "store_scaled_value": gs["value"], "store_scaled_abserr": gs["abserr"],
                            })
                            blob = buf.getvalue()
                            pth = pdir / f"{rid}-{tree}-{idx:02d}.pkl"
                            pth.write_bytes(blob)
                            cr["pickle"] = str(pth)
                            cr["pickle_sha256"] = hashlib.sha256(blob).hexdigest()
                            cr["pickle_bytes"] = len(blob)
                        crecs.append(cr)
                    rec["calls"] = crecs
                    rec["all_calls_equal_store"] = all(c["scaled_equals_store"] and c["regions_equal_store"] for c in crecs)
                    rec["error"] = None
                except Exception:
                    rec["error"] = traceback.format_exc()
                    QSI_MOD.adaptive_levin_sincos = orig
                assert QSI_MOD.adaptive_levin_sincos is orig
                out["rows"].append(rec)
                if rec["error"]:
                    print(f"{rid}: ERROR\n{rec['error']}", flush=True)
                else:
                    print(f"{rid}: {len(rec['calls'])} group calls ({rec['other_calls']} others), {rec['seconds']:.1f} s; total {rec['total']} "
                          f"abserr {rec['total_abserr']} equal to store {rec['total_equals_store']}; every call equal "
                          f"to the store's metadata {rec['all_calls_equal_store']}", flush=True)
                    for c in rec["calls"]:
                        print(f"    {c['index']:2d} sub {c['sub']} {c['label']:<6} value {c['value']} abserr {c['abserr']} "
                              f"regions {c['num_regions']} depth {c['max_depth']} conv {int(c['converged'])} "
                              f"store-equal {c['scaled_equals_store']}{' pickled' if 'pickle' in c else ''}", flush=True)
        finally:
            print(close_pool(pool, cluster).strip())
    print(f"# ray initialised at the end: {ray.is_initialized()}")
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    ok = all(r["error"] is None and r["total_equals_store"] and r["all_calls_equal_store"] for r in out["rows"])
    print(f"rows {len(out['rows'])}; every row and call reproduces the store bit for bit: {ok}")
    print(f"wrote {a.out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
