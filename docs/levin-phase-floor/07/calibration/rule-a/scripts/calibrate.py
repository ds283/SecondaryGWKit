"""levin-phase-floor 07 section 2.2 (and 2.5 on its cases): Phi_w and Lambda_c, by the rules the prompt fixes.

Built afresh for the rewritten prompt (2026-11-07); its families, references and table layout follow the
set-aside work's calibrate.py (var/levin-phase-floor-07/wip/07-part-a-rebased-on-06d-untracked.tar.gz,
docs/levin-phase-floor/07/calibration/scripts/calibrate.py), and its references are reference.py's.

The two families (prompt section 2.1), on x in [0, 1], f_sin = exp(-x), f_cos = cos(3x), the phase handed to
``adaptive_levin_sincos`` as {"theta": theta, "theta_deriv": theta'} (so that the location runs):
- (R) theta = omega x + (c/2)(x - x_s)^2, root at x_s - omega/c; c in {1e4, 1e6, 1e8}, x_s in
  {0.3819660112501051, 0.7236067977499790}, every case at omega = 0, and four also at omega = 1, the agent's
  choice (``reference.OMEGA_ONE_CASES``: c = 1e4, 1e6, 1e8 at x_s = 0.38..., and c = 1e8 at x_s = 0.72...);
- (C) theta = a(x - x_0) + (b/3)(x - x_0)^3, x_0 = 0.3819660112501051, b in {1e4, 1e8}, a the double nearest
  (lambda^2 b)^(1/3), lambda in {0.25, 1, 4, 16, 64, 256, 1024}.
Every parameter is the exact double of ``reference.cases()``, shared with the references.

**Every run:** rtol 1e-10 in the first run, 1e-9 (and, if a rule finds no rung there, 1e-8) after the section 2
stop (section 9 point 1; ``--rtol``, outputs named by it), atol 1e-14 |ref|, the module's defaults otherwise (chebyshev_order 16, depth_max 20).
"Honest": |value - ref| <= abserr. "Converged": the call reports converged.

**The references** (section 2.1): (R) by the closed form in erf (``reference.closed_R``) and by contour
integration (``reference.contour_R``), both computed here, the closed form used and their relative disagreement
recorded per case (a disagreement at the precision the rules need is a stop); (C) by contour integration
(``reference.contour_C``). Their validation against mpmath.quad is validate_reference.py's.

**Phi_w** (family R): each rung of {2, 8, 32, 128, 512} pi is set as the module's
``_LEVIN_WINDOW_HALF_PHASE`` and every (R) case run. The rule: Phi_w is the smallest rung on which every case of
(R) is honest and converged, those at omega = 1 included. Recorded for every case and rung: the true error,
abserr, regions, window pieces, maximum depth, theta'^2/|theta''| at the window's edges (theta' at the edge and
the exact theta'' = c), whether the call ended node_limited, and the located point against x_s - omega/c and x_s.

**Lambda_c** (family C), at that Phi_w:
- every case with no minimum located, as if Lambda_c = 0 (``_LEVIN_COALESCENCE_LAMBDA`` = 0);
- the rule: Lambda_c is the smallest rung L of {4, 16, 64, 256, 1024} such that every case with lambda >= L, at
  both b, is honest;
- the check: the cases with lambda < L re-run with their minimum located (``_LEVIN_COALESCENCE_LAMBDA`` = L);
  they must be honest and converged.
**If no rung of Phi_w qualifies**, (C) cannot be run under its rule, which needs a Phi_w. Then, labelled
"supplementary, not the rule", (C)'s procedure is run at every rung of Phi_w, so that the user can see it.

**For context, not part of either rule:** every case at the parent's code (``prov.PARENT_SHA``'s
AdaptiveLevin/levin_quadrature.py, read with git show and loaded as a module of its own), same arguments.

**Section 2.5 on these cases:** every call that does not converge or reaches depth_max carries
``analysis07.section_2_5``'s account. And each such call whose window leaves reach depth_max is re-run once at
the same rung by the "bar lifted" variant: the tree's module with the bar on the floor-limited exit for window and
guarded leaves removed (one condition, replaced by text substitution and checked to occur once), loaded as a
module of its own, to say whether today's exit would have fired on them if it were not barred.

**A call that does not finish** within the unit limit is recorded as such, and re-run at bounded depth_max
(6, 8, 10 and 12), each with section 2.5's account, so that where its error sits and whether it falls with h can
be read; and once with the bar lifted at depth_max 20.

**Mechanics.** Each call runs in its own process (``calibrate.py unit``), so that a call that does not finish
cannot hold the others back: ``run`` executes the units in a pool of ``--processes`` workers, each unit with a
wall-clock limit ``--unit-timeout`` (a unit over it is recorded as "did not finish", with its wall time, and
counts as neither honest nor converged). Writes ``out/calibrate.json`` and ``out/calibrate_tables.md``; each
unit's output line is also appended, as it arrives, to ``out/calibrate_units.jsonl``.
"""

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import prov  # noqa: E402
import reference as R  # noqa: E402

PHI_LADDER_PI = (2.0, 8.0, 32.0, 128.0, 512.0)
LAMBDA_LADDER = (4.0, 16.0, 64.0, 256.0, 1024.0)
RTOL = 1.0e-10  # the first run's; after the section 2 stop the run's rtol is --rtol (section 9 point 1)
ATOL_FACTOR = 1.0e-14
DEPTH_MAX = 20
BOUNDED_DEPTHS = (6, 8, 10, 12)

BAR_TEXT = (
    "        if (\n"
    "            not region.window\n"
    "            and not leaf.guarded\n"
    "            and np.isfinite(phase_err)\n"
)
BAR_LIFTED_TEXT = "        if (\n            np.isfinite(phase_err)\n"


def label(case):
    if case["family"] == "R":
        return f"R c={case['c']:.0e} x_s={case['x_s']:.4f} omega={case['omega']:g}"
    return f"C b={case['b']:.0e} lambda={case['lambda']:g}"


# ------------------------------------------------------------------------------------------------
# one unit, in its own process


def _load(mode):
    sys.path.insert(0, str(prov.REPO))
    if mode == "commit":
        import AdaptiveLevin.levin_quadrature as lq

        return lq, prov.sha256(prov.LEVIN)
    if mode == "parent":
        source = prov.git(
            "show", f"{prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"
        )
        return prov.load_module_from_source(source, "levin_quadrature_parent_bd21bf2")
    if mode == "barlifted":
        source = prov.LEVIN.read_text()
        if source.count(BAR_TEXT) != 1:
            raise RuntimeError("calibrate: the bar's text is not found exactly once")
        return prov.load_module_from_source(
            source.replace(BAR_TEXT, BAR_LIFTED_TEXT), "levin_quadrature_bar_lifted"
        )
    raise ValueError(mode)


def _phase(case):
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


def _theta_second(case, x):
    if case["family"] == "R":
        return case["c"]
    return 2.0 * case["b"] * (x - case["x_0"])


def do_unit(spec):
    import numpy as np

    import analysis07 as A

    module, module_sha = _load(spec["mode"])
    case = spec["case"]
    ref = spec["ref"]
    if spec["mode"] != "parent":
        module._LEVIN_WINDOW_HALF_PHASE = spec["phi_w"]
        module._LEVIN_COALESCENCE_LAMBDA = spec["lambda_c"]
    recorder = A.Recorder(module)
    recorder.install()
    atol = ATOL_FACTOR * abs(ref)
    t0 = time.perf_counter()
    result = module.adaptive_levin_sincos(
        (0.0, 1.0),
        [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)],
        _phase(case),
        atol=atol,
        rtol=spec.get("rtol", RTOL),
        depth_max=spec.get("depth_max", DEPTH_MAX),
    )
    wall = time.perf_counter() - t0
    recorder.remove()
    depth_max = spec.get("depth_max", DEPTH_MAX)
    rec = A.summary(result, ref=ref, depth_max=depth_max)
    rec["depth_max"] = depth_max
    rec["wall_s"] = wall
    rec["atol"] = atol
    rec["request"] = A.request_of(result, atol, spec.get("rtol", RTOL))
    rec["abserr_over_request"] = result["abserr"] / rec["request"]
    rec["module_sha256"] = module_sha
    if "stationary_points" in rec:
        for p in rec["stationary_points"]:
            for e in p["own_window_edges"]:
                second = _theta_second(case, e["x"])
                e["theta_deriv_sq_over_second_exact"] = (
                    e["theta_deriv"] ** 2 / abs(second) if second != 0 else math.inf
                )
        if case["family"] == "R":
            rec["point_minus_root"] = [
                p["x"] - case["x_root"] for p in rec["stationary_points"]
            ]
            rec["point_minus_x_s"] = [
                p["x"] - case["x_s"] for p in rec["stationary_points"]
            ]
        rec["window_leaves_at_depth_max"] = sum(
            1 for r in result["regions"] if r.window and r.depth >= depth_max
        )
    if A.needs_section_2_5(result, depth_max):
        rec["section_2_5"] = A.section_2_5(
            result,
            recorder.made,
            rec["request"],
            A.n_seed_leaves(result) if "stationary_windows" in result else 1,
            depth_max,
        )
    return rec


# ------------------------------------------------------------------------------------------------
# the run


def _refs(cases):
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
            value, omitted = R.contour_C(case["b"], case["a"])
            case["ref_contour_omitted_bound"] = mp.nstr(omitted, 3)
        case["ref"] = mp.nstr(value, R.REPORT_DPS)
        case["ref_float"] = float(value)
    return cases


def _execute(specs, processes, timeout, units_path, log):
    def one(spec):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__)), "unit", json.dumps(spec)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(HERE),
            )
        except subprocess.TimeoutExpired:
            return {
                "spec": spec,
                "did_not_finish": True,
                "wall_s": time.perf_counter() - t0,
            }
        if proc.returncode != 0:
            return {
                "spec": spec,
                "error": proc.stderr[-4000:],
                "wall_s": time.perf_counter() - t0,
            }
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("UNIT ")]
        return {"spec": spec, "record": json.loads(lines[-1][5:])}

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as pool:
        futures = [pool.submit(one, s) for s in specs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            out.append(res)
            with open(units_path, "a") as handle:
                handle.write(json.dumps(res, default=str) + "\n")
            s = res["spec"]
            r = res.get("record")
            if r is None:
                log(
                    f"  {s['mode']} {s.get('phi_w_pi')} {s['label']}: "
                    f"{'DID NOT FINISH' if res.get('did_not_finish') else 'ERROR'} "
                    f"after {res['wall_s']:.0f} s"
                )
            else:
                log(
                    f"  {s['mode']} phi {s.get('phi_w_pi')} L {s.get('lambda_c')} {s['label']}: err "
                    f"{r['true_error']:.3g} abserr {r['abserr']:.3g} honest {r['honest']} "
                    f"converged {r['converged']} node_limited {r['node_limited']} depth "
                    f"{r['max_depth']} regions {r['regions']} wall {r['wall_s']:.1f} s"
                )
    return out


def _ok(res, need_converged=True):
    r = res.get("record")
    if r is None:
        return False
    return r["honest"] and (r["converged"] or not need_converged)


def do_run(processes, timeout, rtol):
    tag = f"rtol{rtol:g}"
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = {
        "header": prov.json_header(
            __file__,
            "section 2.2, Phi_w and Lambda_c (run)",
            (HERE / "reference.py", HERE / "analysis07.py"),
        ),
        "phi_ladder_pi": PHI_LADDER_PI,
        "lambda_ladder": LAMBDA_LADDER,
        "rtol": rtol,
        "atol_factor": ATOL_FACTOR,
        "depth_max": DEPTH_MAX,
        "parent_sha": prov.PARENT_SHA,
        "processes": processes,
        "unit_timeout_s": timeout,
    }
    units_path = prov.OUT / f"calibrate_units_{tag}.jsonl"
    units_path.write_text("")
    t_start = time.perf_counter()

    cases = _refs(R.cases())
    out["cases"] = cases
    out["R_closed_minus_contour_largest_relative"] = max(
        c["ref_closed_minus_contour_relative"] for c in cases if c["family"] == "R"
    )
    log(
        f"references done; (R) erf vs contour, largest relative {out['R_closed_minus_contour_largest_relative']:.3g}"
    )

    def spec(mode, case, phi_pi=None, lambda_c=None, part=None):
        return {
            "mode": mode,
            "case": case,
            "label": label(case),
            "ref": case["ref_float"],
            "phi_w_pi": phi_pi,
            "phi_w": None if phi_pi is None else phi_pi * math.pi,
            "lambda_c": lambda_c,
            "part": part,
            "rtol": rtol,
        }

    # stage 1: the parent on every case, and (R) at every rung (Lambda_c is immaterial on (R): theta' is
    # linear, with no interior minimum); and, so that a no-rung outcome costs no second stage, (C) with no
    # minimum located at every rung too (stage 2 picks the rung's rows)
    specs = [spec("parent", c, part="parent") for c in cases]
    for phi in PHI_LADDER_PI:
        specs += [
            spec("commit", c, phi, 64.0, "R") for c in cases if c["family"] == "R"
        ]
        specs += [
            spec("commit", c, phi, 0.0, "C0") for c in cases if c["family"] == "C"
        ]
    log(f"stage 1: {len(specs)} units")
    results = _execute(specs, processes, timeout, units_path, log)

    def pick(part, phi=None, lambda_c=None):
        return {
            r["spec"]["label"]: r
            for r in results
            if r["spec"]["part"] == part
            and (phi is None or r["spec"]["phi_w_pi"] == phi)
            and (lambda_c is None or r["spec"]["lambda_c"] == lambda_c)
        }

    out["parent"] = pick("parent")
    out["R"] = {f"{phi:g}pi": pick("R", phi) for phi in PHI_LADDER_PI}
    phi_w = None
    for phi in PHI_LADDER_PI:
        if all(_ok(res) for res in out["R"][f"{phi:g}pi"].values()):
            phi_w = phi
            break
    out["phi_w_by_rule_pi"] = phi_w
    log(f"Phi_w by the rule: {'none' if phi_w is None else f'{phi_w:g} pi'}")

    def lambda_rule(rows):
        for L in LAMBDA_LADDER:
            if all(
                _ok(rows[label(c)], need_converged=False)
                for c in cases
                if c["family"] == "C" and c["lambda"] >= L
            ):
                return L
        return None

    rungs = [phi_w] if phi_w is not None else list(PHI_LADDER_PI)
    out["C_is_supplementary"] = phi_w is None
    out["C"] = {}
    check_specs = []
    for phi in rungs:
        rows = pick("C0", phi)
        L = lambda_rule(rows)
        out["C"][f"{phi:g}pi"] = {"no_minimum_located": rows, "lambda_c_by_rule": L}
        if L is not None:
            check_specs += [
                spec("commit", c, phi, L, "Ccheck")
                for c in cases
                if c["family"] == "C" and c["lambda"] < L
            ]
    log(f"stage 2 (the check): {len(check_specs)} units")
    results += _execute(check_specs, processes, timeout, units_path, log)
    for phi in rungs:
        L = out["C"][f"{phi:g}pi"]["lambda_c_by_rule"]
        if L is None:
            continue
        check = pick("Ccheck", phi, L)
        out["C"][f"{phi:g}pi"]["check"] = check
        out["C"][f"{phi:g}pi"]["check_holds"] = all(_ok(r) for r in check.values())

    # stage 3: the bar lifted, where window leaves reached depth_max
    bar_specs = []
    for r in results:
        rec = r.get("record")
        if rec and rec.get("window_leaves_at_depth_max"):
            s = dict(r["spec"])
            s["mode"] = "barlifted"
            s["part"] = "barlifted:" + r["spec"]["part"]
            bar_specs.append(s)
    log(f"stage 3 (bar lifted): {len(bar_specs)} units")
    results += _execute(bar_specs, processes, timeout, units_path, log)
    out["bar_lifted"] = [
        r for r in results if str(r["spec"]["part"]).startswith("barlifted:")
    ]

    # stage 4: a call that did not finish within the unit's limit is re-run at bounded depth_max (6, 8, 10,
    # 12), so that section 2.5 can say where its error sits and whether it falls with h, and once with the bar
    # lifted at depth_max 20
    bounded_specs = []
    for r in results:
        if r.get("did_not_finish") and r["spec"]["mode"] == "commit":
            for dm in BOUNDED_DEPTHS:
                s = dict(r["spec"])
                s["depth_max"] = dm
                s["part"] = f"bounded:{r['spec']['part']}"
                bounded_specs.append(s)
            s = dict(r["spec"])
            s["mode"] = "barlifted"
            s["part"] = "barlifted:" + r["spec"]["part"]
            bounded_specs.append(s)
    log(
        f"stage 4 (did not finish: bounded depth, and the bar lifted): {len(bounded_specs)} units"
    )
    results += _execute(bounded_specs, processes, timeout, units_path, log)
    out["bounded"] = [
        r for r in results if str(r["spec"]["part"]).startswith("bounded:")
    ]
    out["bar_lifted"] = [
        r for r in results if str(r["spec"]["part"]).startswith("barlifted:")
    ]
    out["did_not_finish"] = [r["spec"] for r in results if r.get("did_not_finish")]
    out["errors"] = [
        {"spec": r["spec"], "error": r["error"]} for r in results if r.get("error")
    ]
    out["wall_s"] = time.perf_counter() - t_start
    out["uptime_at_end"] = prov.uptime()
    path = prov.OUT / f"calibrate_{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {path}")
    (prov.OUT / f"calibrate_tables_{tag}.md").write_text(tables(out))
    log(f"wrote {prov.OUT / f'calibrate_tables_{tag}.md'}")


# ------------------------------------------------------------------------------------------------
# tables


def _fmt(x, spec=".2e"):
    if x is None:
        return "--"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return format(x, spec)


def _row_R(rung, name, res):
    r = res.get("record")
    if r is None:
        return (
            f"| {rung} | {name} | did not finish ({res.get('wall_s', 0):.0f} s) |"
            + " |" * 11
        )
    edges = [
        e["theta_deriv_sq_over_second_exact"]
        for p in r.get("stationary_points", [])
        for e in p["own_window_edges"]
    ]
    return (
        f"| {rung} | {name} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | "
        f"{_fmt(r['error_over_abserr'], '.2g')} | {_fmt(r['abserr_over_request'], '.3g')} | "
        f"{r['honest']} | {r['converged']} | {r['node_limited']} | {r['regions']} | "
        f"{r.get('window_pieces', '--')} | {r.get('windows_stopped', '--')}, "
        f"{r.get('windows_stopped_with_falling_leaves', '--')} | {r['max_depth']} | "
        f"{', '.join(_fmt(e, '.4g') for e in edges)} | "
        f"{', '.join(format(d, '.1e') for d in r.get('point_minus_root', []))} | "
        f"{', '.join(format(d, '.1e') for d in r.get('point_minus_x_s', []))} |"
    )


def _row_C(name, res, case):
    r = res.get("record")
    if r is None:
        return f"| {name} | did not finish ({res.get('wall_s', 0):.0f} s) |" + " |" * 9
    return (
        f"| {name} | {case['lambda_actual']:.6g} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | "
        f"{_fmt(r['error_over_abserr'], '.2g')} | {r['honest']} | {r['converged']} | "
        f"{r['node_limited']} | {r['regions']} | {r.get('window_pieces', '--')} | {r['max_depth']} | "
        f"{', '.join(p['kind'] + ' ' + _fmt(p.get('lambda'), '.4g') for p in r.get('stationary_points', []))} | "
        f"{', '.join(_fmt(m['lambda'], '.4g') for m in r.get('unlocated_minima', []))} |"
    )


def tables(out):
    h = out["header"]
    lines = ["# 07 section 2.2 tables", ""] + [f"    {x}" for x in h] + [""]
    lines += [
        "## Phi_w on family (R)",
        "",
        f"(R)'s two references, by erf and by contour: largest relative disagreement "
        f"{out['R_closed_minus_contour_largest_relative']:.2e}.",
        "",
        "| Phi_w | case | true error | abserr | err/abserr | abserr/request | honest | converged | "
        "node_limited | regions | window pieces | windows stopped, of them with a leaf falling | "
        "max depth | theta'^2/theta'' at edges | point - (x_s - omega/c) | point - x_s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rung, rows in out["R"].items():
        for name in sorted(rows):
            lines.append(_row_R(rung, name, rows[name]))
    phi = out["phi_w_by_rule_pi"]
    lines += [
        "",
        f"**Phi_w by the rule:** {'none' if phi is None else f'{phi:g} pi'}",
        "",
    ]
    case_by = {label(c): c for c in out["cases"]}
    for rung, block in out["C"].items():
        lines += [
            f"## Lambda_c on family (C) at Phi_w = {rung}"
            + (
                " -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)"
                if out["C_is_supplementary"]
                else ""
            ),
            "",
            "No minimum located (Lambda_c = 0):",
            "",
            "| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | "
            "regions | window pieces | max depth | located | minima found, not located (lambda) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in sorted(
            block["no_minimum_located"],
            key=lambda n: (case_by[n]["b"], case_by[n]["lambda"]),
        ):
            lines.append(_row_C(name, block["no_minimum_located"][name], case_by[name]))
        lines += ["", f"**Lambda_c by the rule:** {block['lambda_c_by_rule']}", ""]
        if block.get("check") is not None:
            lines += [
                f"The check, lambda < {block['lambda_c_by_rule']:g}, minimum located:",
                "",
                "| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | "
                "node_limited | regions | window pieces | max depth | located | minima not located |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for name in sorted(
                block["check"], key=lambda n: (case_by[n]["b"], case_by[n]["lambda"])
            ):
                lines.append(_row_C(name, block["check"][name], case_by[name]))
            lines += ["", f"**The check holds:** {block['check_holds']}", ""]
    lines += [
        "## Section 2.5 on these cases: calls that do not converge or reach depth_max",
        "",
        "| part | Phi_w | Lambda_c | case | converged | node_limited | max depth | abserr/request | "
        "leaves by class: count, sum e', at depth_max | QUADPACK iroff1, iroff2, ier=2 at bisection |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    every = []
    for rung, rows in out["R"].items():
        every += list(rows.values())
    for block in out["C"].values():
        every += list(block["no_minimum_located"].values())
        every += list((block.get("check") or {}).values())
    every += list(out["parent"].values())
    every += list(out.get("bounded", []))
    for res in every:
        r = res.get("record")
        if r is None or "section_2_5" not in r:
            continue
        s = res["spec"]
        a = r["section_2_5"]
        cls = "; ".join(
            f"{k} {v['count']}, {v['sum_budget_err']:.2e}, {v['at_depth_max']}"
            for k, v in sorted(a["classes"].items())
        )
        q = a["quadpack_global"]
        lines.append(
            f"| {s['part']} (depth_max {r.get('depth_max', DEPTH_MAX)}) | {s.get('phi_w_pi')} | "
            f"{s.get('lambda_c')} | {s['label']} | {r['converged']} | "
            f"{r['node_limited']} | {r['max_depth']} | {_fmt(r['abserr_over_request'], '.3g')} | {cls} | "
            f"{q['iroff1']}, {q['iroff2']}, {q['ier2_reached_at_bisection']} |"
        )
    lines += [
        "",
        "Windows stopped by section 1.3's rule while one of their leaves' errors fell at its last "
        "bisection (a finding for the user, not a stop; section 9 point 4):",
        "",
    ]
    for res in every:
        r = res.get("record")
        if r is None or not r.get("windows_stopped_with_falling_leaves"):
            continue
        s = res["spec"]
        for w in r.get("stationary_windows", []):
            if w.get("stopped") and w.get("leaves_falling_at_stop"):
                lines.append(
                    f"- {s['part']} {s['mode']} Phi_w {s.get('phi_w_pi')} pi, {s['label']}: window "
                    f"[{w['lo']:.10g}, {w['hi']:.10g}], {w['leaves_falling_at_stop']} of "
                    f"{w['leaves_at_stop']} leaves falling, stopped after split "
                    f"{w['stopped_after_split']}, deepest {w['max_depth_at_stop']}"
                )
    lines += ["", "Units that did not finish within the unit limit:", ""]
    for s in out.get("did_not_finish", []):
        lines.append(
            f"- {s['part']} {s['mode']} Phi_w {s.get('phi_w_pi')} pi, Lambda_c {s.get('lambda_c')}, "
            f"{s['label']}, depth_max {s.get('depth_max', DEPTH_MAX)}"
        )
    lines += ["", "Units that ended in an error:", ""]
    for e in out.get("errors", []):
        lines.append(f"- {e['spec']['part']} {e['spec']['label']}: {e['error'][-300:]}")
    lines += [
        "",
        "### The bar lifted (window and guarded leaves allowed the floor-limited exit), where window leaves reached depth_max",
        "",
        "| part | Phi_w | case | window leaves floor-limited | converged | node_limited | max depth | "
        "regions | err/abserr | abserr/request | honest |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in out["bar_lifted"]:
        r = res.get("record")
        s = res["spec"]
        if r is None:
            lines.append(
                f"| {s['part']} | {s.get('phi_w_pi')} | {s['label']} | did not finish |"
                + " |" * 7
            )
            continue
        lines.append(
            f"| {s['part']} | {s.get('phi_w_pi')} | {s['label']} | {r.get('window_leaves_phase_limited')} | "
            f"{r['converged']} | {r['node_limited']} | {r['max_depth']} | {r['regions']} | "
            f"{_fmt(r['error_over_abserr'], '.2g')} | {_fmt(r['abserr_over_request'], '.3g')} | {r['honest']} |"
        )
    lines += [
        "",
        f"## For context: the parent's code ({out['parent_sha'][:8]}, 06m) on every case (not part of either rule)",
        "",
        "| case | true error | abserr | err/abserr | abserr/request | honest | converged | node_limited | "
        "regions | max depth |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(out["parent"]):
        r = out["parent"][name].get("record")
        if r is None:
            lines.append(f"| {name} | did not finish |" + " |" * 8)
            continue
        lines.append(
            f"| {name} | {_fmt(r['true_error'])} | {_fmt(r['abserr'])} | {_fmt(r['error_over_abserr'], '.2g')} | "
            f"{_fmt(r['abserr_over_request'], '.3g')} | {r['honest']} | {r['converged']} | {r['node_limited']} | "
            f"{r['regions']} | {r['max_depth']} |"
        )
    lines += [
        "",
        f"Wall time {out['wall_s']:.0f} s; uptime at end: {out['uptime_at_end']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "unit", "tables"))
    parser.add_argument("spec", nargs="?")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--unit-timeout", type=float, default=4 * 3600.0)
    parser.add_argument(
        "--rtol", type=float, default=RTOL
    )  # after the section 2 stop (section 9 point 1)
    args = parser.parse_args()
    if args.action == "unit":
        rec = do_unit(json.loads(args.spec))
        print("UNIT " + json.dumps(rec, default=str), flush=True)
    elif args.action == "run":
        do_run(args.processes, args.unit_timeout, args.rtol)
    else:
        tag = f"rtol{args.rtol:g}"
        out = json.loads((prov.OUT / f"calibrate_{tag}.json").read_text())
        (prov.OUT / f"calibrate_tables_{tag}.md").write_text(tables(out))


if __name__ == "__main__":
    main()
