"""levin-phase-floor prompt 07b extension, section 2.7.2 (g): 07's family (R) at rtol 1e-9, at 07's harness settings,
under the tree (07a's code, (a)) and the extension's variants. New in the extension.

**07's harness** (docs/levin-phase-floor/07/calibration/scripts/calibrate.py ``do_unit``, a record, read only): x in
[0, 1], f = [exp(-x), cos(3x)], the phase {"theta", "theta_deriv"} of ``calibrate._phase`` (theta = omega x +
(c/2)(x - x_s)^2), atol = 1e-14 |ref|, rtol 1e-9, depth_max 20, the module's other defaults (chebyshev_order 16), each
call in its own process. The ten cases are ``reference.cases()``'s family (R) (07's exact doubles); the reference is
07's closed form ``reference.closed_R`` (07's run 2 found it and the contour form to agree to 2.2e-37 relative).
Copied in substance, not imported, from calibrate.py: ``_phase`` and the request max(atol, rtol |value|)
(analysis07.request_of). 07 set the module's ``_LEVIN_WINDOW_HALF_PHASE`` per rung; here it is the variant's own (2 pi;
(g) adapts it). 07's Recorder and section 2.5 accounts are not run.

**Recorded per (variant, case):** value, abserr, |value - ref|/abserr (honest if at most 1), converged, abserr/request,
regions, maximum depth, the windows (lo, hi, pieces, stopped), and for (g) the instrument: each window's half-width
reached and, at each final edge, the rate ratio across the first spacing, spacing/l and whether the test still fails.

Usage: PYTHONPATH=<tree>:<scripts> python -B family_r_ext.py run OUT_JSON VARIANT [VARIANT ...]   (one process per unit,
at most six at a time); ``unit SPEC_JSON`` is the per-process entry.
"""

import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
CAL = TREE / "docs" / "levin-phase-floor" / "07" / "calibration" / "scripts"
TREE_LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
RTOL = 1.0e-9
ATOL_FACTOR = 1.0e-14
DEPTH_MAX = 20


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phase(case):
    c, x_s, omega = case["c"], case["x_s"], case["omega"]
    if omega == 0.0:
        return {"theta": lambda x: 0.5 * c * (x - x_s) ** 2, "theta_deriv": lambda x: c * (x - x_s)}
    return {"theta": lambda x: omega * x + 0.5 * c * (x - x_s) ** 2, "theta_deriv": lambda x: omega + c * (x - x_s)}


def unit(spec):
    import numpy as np

    sys.path.insert(1, str(HERE.parent))
    v = spec["variant"]
    if v == "tree":
        import AdaptiveLevin.levin_quadrature as LQ

        msha = sha256(LQ.__file__)
        if msha != TREE_LQ_SHA:
            raise SystemExit("the tree's module has another SHA-256")
    else:
        import variant_loader as VL

        vx = json.loads((ROOT / "out" / "ext" / "variants" / "variants.json").read_text())["variants"]
        vj = vx[v] if v in vx else json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][v]
        LQ = VL.install(vj["path"], vj["sha256"], f"07b_{v}")
        msha = vj["sha256"]
        if getattr(LQ, "_VARIANT_07B", None) != v:
            raise SystemExit("the variant's constant is not its name")
    if hasattr(LQ, "_MEASURE_07B"):
        LQ._MEASURE_07B = []
    case, ref = spec["case"], float(spec["ref"])
    atol = ATOL_FACTOR * abs(ref)
    t0 = time.perf_counter()
    res = LQ.adaptive_levin_sincos((0.0, 1.0), [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)], phase(case),
                                   atol=atol, rtol=RTOL, depth_max=DEPTH_MAX)
    wall = time.perf_counter() - t0
    import mpmath as mp

    err = float(abs(mp.mpf(res["value"]) - mp.mpf(spec["ref"])))
    request = max(atol, RTOL * abs(res["value"]))
    regs = res["regions"]
    wins = res.get("stationary_windows") or []
    rec = {"variant": v, "module_sha256": msha, "case": case, "ref": spec["ref"], "value": res["value"],
           "value_hex": float(res["value"]).hex(), "abserr": res["abserr"], "abserr_hex": float(res["abserr"]).hex(),
           "true_error": err, "err_over_abserr": err / res["abserr"], "honest": err <= res["abserr"],
           "converged": bool(res["converged"]), "abserr_over_request": res["abserr"] / request,
           "regions": int(res["num_regions"]), "max_depth": int(max(r.depth for r in regs)),
           "windows": [{"lo": w["lo"], "hi": w["hi"], "pieces": len(w.get("cuts") or []) - 1, "stopped": bool(w.get("stopped"))}
                       for w in wins],
           "instrument": [x for x in (getattr(LQ, "_MEASURE_07B", None) or []) if x.get("kind") == "g"], "wall_s": wall}
    print("UNIT " + json.dumps(rec, default=float))


def run(out_p, variants):
    sys.path.insert(0, str(CAL))
    import mpmath as mp
    import reference as R

    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    cases = [c for c in R.cases() if c["family"] == "R"]
    specs = []
    for case in cases:
        with mp.workdps(R.WORKING_DPS):
            ref = mp.nstr(R.closed_R(case["c"], case["x_s"], case["omega"]), R.REPORT_DPS)
        for v in variants:
            specs.append({"variant": v, "case": case, "ref": ref})
    env_pp = f"{TREE}:{HERE.parent}"

    def one(spec):
        p = subprocess.run([PY, "-B", str(Path(__file__).resolve()), "unit", json.dumps(spec)], capture_output=True,
                           text=True, cwd=str(TREE), env={"PYTHONPATH": env_pp, "PYTHONDONTWRITEBYTECODE": "1",
                                                          "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("UNIT ")]
        if p.returncode != 0 or not lines:
            return {"spec": spec, "error": p.stderr[-3000:]}
        return json.loads(lines[-1][5:])

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        recs = list(pool.map(one, specs))
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "reference_py_sha256": sha256(CAL / "reference.py"), "calibrate_py_sha256": sha256(CAL / "calibrate.py"),
                      "rtol": RTOL, "atol_factor": ATOL_FACTOR, "depth_max": DEPTH_MAX, "variants": variants,
                      "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": time.time() - t0},
           "records": recs}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=float))
    md = ["| variant | case | true error | abserr | err/abserr | honest | converged | abserr/request | regions | max depth | windows (pieces) | half-width reached / pi | final edges: rate ratio, spacing/l, fails |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    import math

    for r in recs:
        if "error" in r:
            md.append(f"| {r['spec']['variant']} | {r['spec']['case']} | ERROR | | | | | | | | | | |")
            continue
        c = r["case"]
        ins = r["instrument"][0] if r["instrument"] else None
        halfw = ", ".join(f"{w['half_max'] / math.pi:g}" for w in ins["windows"]) if ins else "-"
        edges = "; ".join(f"{e['rate_ratio']:.3g}, {e['spacing_over_l']:.3g}, {'F' if e['fails'] else 'ok'}" for e in ins["edges"]) if ins else "-"
        md.append(f"| {r['variant']} | c={c['c']:.0e} x_s={c['x_s']:.4f} omega={c['omega']:g} | {r['true_error']:.3g} | {r['abserr']:.3g} | "
                  f"{r['err_over_abserr']:.2g} | {r['honest']} | {r['converged']} | {r['abserr_over_request']:.3g} | {r['regions']} | "
                  f"{r['max_depth']} | {len(r['windows'])} ({', '.join(str(w['pieces']) for w in r['windows'])}) | {halfw} | {edges} |")
    out_p.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out_p, sha256(out_p), out_p.with_suffix(".md"), sha256(out_p.with_suffix(".md")))


if __name__ == "__main__":
    if sys.argv[1] == "unit":
        unit(json.loads(sys.argv[2]))
    else:
        run(Path(sys.argv[2]), sys.argv[3:])
