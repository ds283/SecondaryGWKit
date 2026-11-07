"""levin-phase-floor 07 section 2: Phi_w and Lambda_c, measured by the rules the prompt fixes.

The two families, on x in [0, 1], f_sin = exp(-x), f_cos = cos(3x), the phase handed to
``adaptive_levin_sincos`` as {"theta": theta, "theta_deriv": theta'} (so that the location runs):
- (R) theta = omega x + (c/2)(x - x_s)^2, root at x_s - omega/c; c in {1e4, 1e6, 1e8},
  x_s in {0.3819660112501051, 0.7236067977499790}, every case at omega = 0, and four also at
  omega = 1 (prompt section 9 point 3; ``reference.OMEGA_ONE_CASES``). For each (R) case the
  location's point is compared with the root x_s - omega/c and with x_s;
- (C) theta = a(x - x_0) + (b/3)(x - x_0)^3, x_0 = 0.3819660112501051, b in {1e4, 1e8}, a such
  that lambda = a^(3/2)/b^(1/2) is in {0.25, 1, 4, 16, 64, 256, 1024}.
Every parameter is the exact double of ``reference.cases()``, shared with the reference.

**Every run:** rtol 1e-10, atol 1e-14 |ref|, the module's defaults otherwise (chebyshev_order 16,
depth_max 20). "Honest": |value - ref| <= abserr. "Converged": the call reports converged.

**Phi_w** (family R): each rung of {2, 8, 32, 128, 512} pi is set as
``_LEVIN_WINDOW_HALF_PHASE`` and every (R) case run. The rule: Phi_w is the smallest rung on which
every case of (R) is honest and converged. Recorded for every case and rung: the true error,
abserr, regions, window pieces, maximum depth, theta'^2/|theta''| at the window's edges (from
theta' at the edges and the exact theta'' = c), and the location's points.

**Lambda_c** (family C), at that Phi_w:
- every case run with no minimum located, as if Lambda_c = 0 (``_LEVIN_COALESCENCE_LAMBDA`` = 0);
- the rule: Lambda_c is the smallest rung L of {4, 16, 64, 256, 1024} such that every case with
  lambda >= L, at both b, is honest;
- the check: the cases with lambda < L re-run with their minimum located
  (``_LEVIN_COALESCENCE_LAMBDA`` = L); they must be honest and converged.
If no rung of Phi_w qualifies, (C) is not run, and the script says so.

**For context only, not part of either rule:** every case is also run once at the parent's
code (``git show ad63e8a:AdaptiveLevin/levin_quadrature.py``, loaded as a module of its own),
with the same arguments.

**The reference** is ``reference.py``'s, as the user ruled (prompt section 9 point 2): (R) by the
closed form in erf and by contour integration, both computed here, the closed form used and their
relative disagreement recorded per case (a disagreement is a stop); (C) by contour integration.
Each at 40 working digits, validated against the prompt's reference where that is practical
(``validate_reference.py``).

Writes ``out/calibrate.json`` and ``out/calibrate_tables.md``.
"""

import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import reference as R  # noqa: E402
from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

PHI_LADDER = tuple(k * math.pi for k in (2.0, 8.0, 32.0, 128.0, 512.0))
LAMBDA_LADDER = (4.0, 16.0, 64.0, 256.0, 1024.0)
RTOL = 1.0e-10
ATOL_FACTOR = 1.0e-14
PARENT_SHA = "ad63e8a"


def load_parent():
    source = subprocess.run(
        ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
        cwd=MAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    module = types.ModuleType("levin_quadrature_parent")
    exec(compile(source, "levin_quadrature_parent.py", "exec"), module.__dict__)
    return module


def amplitudes():
    return [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]


def phase(case):
    if case["family"] == "R":
        c, x_s, omega = case["c"], case["x_s"], case["omega"]
        if omega == 0.0:
            return {
                "theta": lambda x: 0.5 * c * (x - x_s) ** 2,
                "theta_deriv": lambda x: c * (x - x_s),
            }
        return {
            "theta": lambda x: omega * x + 0.5 * c * (x - x_s) ** 2,
            "theta_deriv": lambda x: omega + c * (x - x_s),
        }
    a, b, x_0 = case["a"], case["b"], case["x_0"]
    return {
        "theta": lambda x: a * (x - x_0) + b / 3.0 * (x - x_0) ** 3,
        "theta_deriv": lambda x: a + b * (x - x_0) ** 2,
    }


def theta_second(case, x):
    if case["family"] == "R":
        return case["c"]
    return 2.0 * case["b"] * (x - case["x_0"])


def label(case):
    if case["family"] == "R":
        return f"R c={case['c']:.0e} x_s={case['x_s']:.4f} omega={case['omega']:g}"
    return f"C b={case['b']:.0e} lambda={case['lambda']:g}"


def run(module, case, ref):
    atol = ATOL_FACTOR * abs(ref)
    t0 = time.perf_counter()
    res = module.adaptive_levin_sincos(
        (0.0, 1.0), amplitudes(), phase(case), atol=atol, rtol=RTOL
    )
    wall = time.perf_counter() - t0
    err = abs(res["value"] - ref)
    rec = {
        "value": res["value"],
        "value_hex": float(res["value"]).hex(),
        "abserr": res["abserr"],
        "true_error": err,
        "error_over_abserr": err / res["abserr"] if res["abserr"] > 0 else None,
        "honest": bool(err <= res["abserr"]),
        "converged": bool(res["converged"]),
        "regions": res["num_regions"],
        "simple_regions": res["num_simple_regions"],
        "max_depth": res["max_depth"],
        "num_phase_limited_regions": res["num_phase_limited_regions"],
        "abserr_resolution": res["abserr_resolution"],
        "abserr_fallback": res["abserr_fallback"],
        "abserr_roundoff": res["abserr_roundoff"],
        "atol": atol,
        "wall_s": wall,
    }
    if "stationary_points" in res:
        rec["window_pieces"] = res["num_window_pieces"]
        rec["guarded_regions"] = res["num_guarded_regions"]
        rec["guard_at_depth_max"] = res["num_guard_at_depth_max"]
        rec["window_phase_limited"] = sum(
            1 for r in res["regions"] if r.type == "window" and r.phase_limited
        )
        points = []
        for p in res["stationary_points"]:
            edges = []
            for e in p["own_window_edges"]:
                second = theta_second(case, e["x"])
                edges.append(
                    {
                        "x": e["x"],
                        "theta_minus_theta_point": e["theta_minus_theta_point"],
                        "theta_deriv": e["theta_deriv"],
                        "theta_deriv_sq_over_second_exact": (
                            e["theta_deriv"] ** 2 / abs(second)
                            if second != 0
                            else float("inf")
                        ),
                        "theta_deriv_sq_over_second_difference": e[
                            "theta_deriv_sq_over_second"
                        ],
                    }
                )
            points.append(
                {
                    "x": p["x"],
                    "kind": p["kind"],
                    "found_as": p["found_as"],
                    "theta_deriv": p["theta_deriv"],
                    "lambda": p.get("lambda"),
                    "window": p["window"],
                    "window_pieces": p["window_pieces"],
                    "own_window_edges": edges,
                }
            )
        rec["stationary_points"] = points
        if case["family"] == "R":
            rec["point_minus_root"] = [p["x"] - case["x_root"] for p in points]
            rec["point_minus_x_s"] = [p["x"] - case["x_s"] for p in points]
        rec["unlocated_minima_lambda"] = [m["lambda"] for m in res["unlocated_minima"]]
    return rec


def main():
    log = lambda s: print(s, flush=True)  # noqa: E731
    parent = load_parent()
    out = {
        "header": header(Path(__file__), "07 section 2: Phi_w and Lambda_c"),
        "phi_ladder": PHI_LADDER,
        "lambda_ladder": LAMBDA_LADDER,
        "rtol": RTOL,
        "atol_factor": ATOL_FACTOR,
        "parent_sha": PARENT_SHA,
        "module_defaults_before": {
            "_LEVIN_WINDOW_HALF_PHASE": lq._LEVIN_WINDOW_HALF_PHASE,
            "_LEVIN_COALESCENCE_LAMBDA": lq._LEVIN_COALESCENCE_LAMBDA,
            "_LEVIN_WINDOW_PIECE_PHASE": lq._LEVIN_WINDOW_PIECE_PHASE,
            "_LEVIN_STATIONARY_SCAN_POINTS": lq._LEVIN_STATIONARY_SCAN_POINTS,
        },
    }
    saved = (lq._LEVIN_WINDOW_HALF_PHASE, lq._LEVIN_COALESCENCE_LAMBDA)

    cases = R.cases()
    refs = {}
    for case in cases:
        if case["family"] == "R":
            value = R.closed_R(case["c"], case["x_s"], case["omega"])
            contour = R.contour_R(case["c"], case["x_s"], case["omega"])
            with mp.workdps(R.WORKING_DPS):
                case["ref_contour"] = mp.nstr(contour, R.REPORT_DPS)
                case["ref_closed_minus_contour_relative"] = float(
                    abs((value - contour) / value)
                )
        else:
            value, _ = R.contour_C(case["b"], case["a"])
        case["ref"] = mp.nstr(value, R.REPORT_DPS)
        refs[label(case)] = float(value)
    out["cases"] = cases
    out["R_closed_minus_contour_largest_relative"] = max(
        c["ref_closed_minus_contour_relative"] for c in cases if c["family"] == "R"
    )

    out["parent"] = {}
    for case in cases:
        out["parent"][label(case)] = run(parent, case, refs[label(case)])
        log(f"parent {label(case)}: {out['parent'][label(case)]['honest']}")

    # Phi_w on family (R); Lambda_c is irrelevant there (theta' is linear: no interior minimum)
    out["R"] = {}
    phi_w = None
    for phi in PHI_LADDER:
        lq._LEVIN_WINDOW_HALF_PHASE = phi
        rung = {}
        for case in cases:
            if case["family"] != "R":
                continue
            rung[label(case)] = run(lq, case, refs[label(case)])
            r = rung[label(case)]
            log(
                f"Phi_w {phi / math.pi:g} pi {label(case)}: err {r['true_error']:.3g} "
                f"abserr {r['abserr']:.3g} honest {r['honest']} converged {r['converged']}"
            )
        out["R"][f"{phi / math.pi:g}pi"] = rung
        if phi_w is None and all(r["honest"] and r["converged"] for r in rung.values()):
            phi_w = phi
    out["phi_w_by_rule"] = phi_w
    log(f"Phi_w by the rule: {None if phi_w is None else f'{phi_w / math.pi:g} pi'}")

    out["C"] = None
    out["lambda_c_by_rule"] = None
    out["C_check"] = None
    if phi_w is not None:
        lq._LEVIN_WINDOW_HALF_PHASE = phi_w
        lq._LEVIN_COALESCENCE_LAMBDA = 0.0
        unlocated = {}
        for case in cases:
            if case["family"] != "C":
                continue
            unlocated[label(case)] = run(lq, case, refs[label(case)])
            r = unlocated[label(case)]
            log(
                f"C, Lambda_c = 0, {label(case)}: err {r['true_error']:.3g} "
                f"abserr {r['abserr']:.3g} honest {r['honest']} converged {r['converged']}"
            )
        out["C"] = unlocated
        lambda_c = None
        for L in LAMBDA_LADDER:
            if all(
                unlocated[label(c)]["honest"]
                for c in cases
                if c["family"] == "C" and c["lambda"] >= L
            ):
                lambda_c = L
                break
        out["lambda_c_by_rule"] = lambda_c
        log(f"Lambda_c by the rule: {lambda_c}")
        if lambda_c is not None:
            lq._LEVIN_COALESCENCE_LAMBDA = lambda_c
            check = {}
            for case in cases:
                if case["family"] == "C" and case["lambda"] < lambda_c:
                    check[label(case)] = run(lq, case, refs[label(case)])
                    r = check[label(case)]
                    log(
                        f"check, Lambda_c = {lambda_c:g}, {label(case)}: err "
                        f"{r['true_error']:.3g} abserr {r['abserr']:.3g} honest {r['honest']} "
                        f"converged {r['converged']} located "
                        f"{[p['kind'] for p in r['stationary_points']]}"
                    )
            out["C_check"] = check
            out["C_check_holds"] = all(
                r["honest"] and r["converged"] for r in check.values()
            )

    lq._LEVIN_WINDOW_HALF_PHASE, lq._LEVIN_COALESCENCE_LAMBDA = saved

    path = OUT / "calibrate.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {path}")
    (OUT / "calibrate_tables.md").write_text(tables(out))
    log(f"wrote {OUT / 'calibrate_tables.md'}")


def _fmt(x, spec=".2e"):
    return "--" if x is None else format(x, spec)


def tables(out):
    lines = [
        "# 07 section 2 tables",
        "",
        f"script sha256 `{out['header']['script_sha256']}`, levin_quadrature.py sha256 "
        f"`{out['header']['levin_quadrature_sha256']}`, HEAD `{out['header']['git_head']}`, "
        f"tree dirty: {out['header']['tree_dirty']}",
        "",
        "## Phi_w on family (R)",
        "",
        f"The (R) reference by erf and by contour: largest relative disagreement "
        f"{out['R_closed_minus_contour_largest_relative']:.2e}.",
        "",
        "| Phi_w | case | true error | abserr | err/abserr | honest | converged | regions | "
        "window pieces | max depth | theta'^2/theta'' at edges | point - (x_s - omega/c) | "
        "point - x_s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rung, rows in out["R"].items():
        for name, r in rows.items():
            case = next(c for c in out["cases"] if label(c) == name)
            edges = [
                e["theta_deriv_sq_over_second_exact"]
                for p in r["stationary_points"]
                for e in p["own_window_edges"]
            ]
            dx = r["point_minus_root"]
            dxs = r["point_minus_x_s"]
            lines.append(
                f"| {rung} | {name} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | "
                f"{_fmt(r['error_over_abserr'], '.2g')} | {r['honest']} | {r['converged']} | "
                f"{r['regions']} | {r['window_pieces']} | {r['max_depth']} | "
                f"{', '.join(format(e, '.4g') for e in edges)} | "
                f"{', '.join(format(d, '.1e') for d in dx)} | "
                f"{', '.join(format(d, '.1e') for d in dxs)} |"
            )
    lines += [
        "",
        f"**Phi_w by the rule:** "
        f"{'none' if out['phi_w_by_rule'] is None else format(out['phi_w_by_rule'] / math.pi, 'g') + ' pi'}",
        "",
    ]
    if out["C"] is not None:
        lines += [
            "## Lambda_c on family (C), no minimum located (Lambda_c = 0)",
            "",
            "| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | "
            "regions | max depth | minima found, lambda |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, r in out["C"].items():
            case = next(c for c in out["cases"] if label(c) == name)
            lines.append(
                f"| {name} | {case['lambda_actual']:.6g} | {_fmt(r['true_error'])} | "
                f"{_fmt(r['abserr'])} | {_fmt(r['error_over_abserr'], '.2g')} | "
                f"{r['honest']} | {r['converged']} | {r['regions']} | {r['max_depth']} | "
                f"{', '.join(format(v, '.4g') for v in r['unlocated_minima_lambda'])} |"
            )
        lines += ["", f"**Lambda_c by the rule:** {out['lambda_c_by_rule']}", ""]
    if out.get("C_check"):
        lines += [
            f"## The check: lambda < {out['lambda_c_by_rule']:g}, minimum located",
            "",
            "| case | true error | abserr | err/abserr | honest | converged | regions | "
            "window pieces | max depth | located (kind, lambda) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, r in out["C_check"].items():
            lines.append(
                f"| {name} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | "
                f"{_fmt(r['error_over_abserr'], '.2g')} | {r['honest']} | {r['converged']} | "
                f"{r['regions']} | {r['window_pieces']} | {r['max_depth']} | "
                f"{', '.join(p['kind'] + ' ' + format(p['lambda'] or 0, '.4g') for p in r['stationary_points'])} |"
            )
        lines += ["", f"**The check holds:** {out['C_check_holds']}", ""]
    lines += [
        "## For context: the parent's code (ad63e8a, 06c) on every case (not part of either rule)",
        "",
        "| case | true error | abserr | err/abserr | honest | converged | regions | max depth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in out["parent"].items():
        lines.append(
            f"| {name} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | "
            f"{_fmt(r['error_over_abserr'], '.2g')} | {r['honest']} | {r['converged']} | "
            f"{r['regions']} | {r['max_depth']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
