"""levin-phase-floor 07 section 2, supplementary: family (C) at every rung of the Phi_w ladder.

**Not the prompt's rule.** Section 2 runs (C) "with that Phi_w", the rung its Phi_w rule picks.
On ``calibrate.py``'s recorded run no rung qualified -- every (R) case is honest at every rung, and
each rung leaves at least one (R) case unconverged -- so section 2 stops there, and ``calibrate.py``
did not run (C). This script supplies, for the user's decision only, what section 2's Lambda_c
rule and its check would give at each rung in turn: everything is exactly ``calibrate.py``'s
procedure for (C) (its ``run``, ``phase`` and ``label``, its rtol and atol, its references),
repeated with ``_LEVIN_WINDOW_HALF_PHASE`` set to each rung.

Writes ``out/calibrate_supplement.json`` and ``out/calibrate_supplement.md``.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import calibrate as K  # noqa: E402
import reference as R  # noqa: E402
from prov import OUT, header  # noqa: E402

lq = K.lq


def main():
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = {
        "header": header(
            Path(__file__), "07 section 2, supplementary: (C) at every rung"
        ),
        "rungs": {},
    }
    saved = (lq._LEVIN_WINDOW_HALF_PHASE, lq._LEVIN_COALESCENCE_LAMBDA)
    cases = [c for c in R.cases() if c["family"] == "C"]
    refs = {}
    for case in cases:
        value, _ = R.contour_C(case["b"], case["a"])
        case["ref"] = mp.nstr(value, R.REPORT_DPS)
        refs[K.label(case)] = float(value)
    out["cases"] = cases
    try:
        for phi in K.PHI_LADDER:
            rung = f"{phi / math.pi:g}pi"
            lq._LEVIN_WINDOW_HALF_PHASE = phi
            lq._LEVIN_COALESCENCE_LAMBDA = 0.0
            unlocated = {K.label(c): K.run(lq, c, refs[K.label(c)]) for c in cases}
            lambda_c = None
            for L in K.LAMBDA_LADDER:
                if all(
                    unlocated[K.label(c)]["honest"] for c in cases if c["lambda"] >= L
                ):
                    lambda_c = L
                    break
            check = None
            if lambda_c is not None:
                lq._LEVIN_COALESCENCE_LAMBDA = lambda_c
                check = {
                    K.label(c): K.run(lq, c, refs[K.label(c)])
                    for c in cases
                    if c["lambda"] < lambda_c
                }
            out["rungs"][rung] = {
                "unlocated": unlocated,
                "lambda_c_by_rule": lambda_c,
                "check": check,
                "check_holds": (
                    None
                    if check is None
                    else all(r["honest"] and r["converged"] for r in check.values())
                ),
            }
            log(
                f"Phi_w {rung}: Lambda_c {lambda_c}, check holds "
                f"{out['rungs'][rung]['check_holds']}"
            )
    finally:
        lq._LEVIN_WINDOW_HALF_PHASE, lq._LEVIN_COALESCENCE_LAMBDA = saved

    (OUT / "calibrate_supplement.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    (OUT / "calibrate_supplement.md").write_text(tables(out))
    log(f"wrote {OUT / 'calibrate_supplement.json'} and .md")


def tables(out):
    f = K._fmt
    lines = [
        "# 07 section 2, supplementary: (C) at every rung (not the prompt's rule)",
        "",
        f"script sha256 `{out['header']['script_sha256']}`, levin_quadrature.py sha256 "
        f"`{out['header']['levin_quadrature_sha256']}`, HEAD `{out['header']['git_head']}`, "
        f"tree dirty: {out['header']['tree_dirty']}",
        "",
        "With Lambda_c = 0 nothing is located on (C), so its runs are the same at every rung; "
        "they are tabulated once, from the first rung.",
        "",
        "## (C), no minimum located (Lambda_c = 0)",
        "",
        "| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | "
        "regions | max depth | minima found, lambda |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    first = next(iter(out["rungs"].values()))
    for name, r in first["unlocated"].items():
        case = next(c for c in out["cases"] if K.label(c) == name)
        lines.append(
            f"| {name} | {case['lambda_actual']:.6g} | {f(r['true_error'])} | "
            f"{f(r['abserr'])} | {f(r['error_over_abserr'], '.2g')} | {r['honest']} | "
            f"{r['converged']} | {r['regions']} | {r['max_depth']} | "
            f"{', '.join(format(v, '.4g') for v in r['unlocated_minima_lambda'])} |"
        )
    for rung, data in out["rungs"].items():
        lines += [
            "",
            f"## Phi_w = {rung}: Lambda_c by the rule = {data['lambda_c_by_rule']}; "
            f"the check holds: {data['check_holds']}",
            "",
        ]
        if data["check"]:
            lines += [
                "| case | true error | abserr | err/abserr | honest | converged | regions | "
                "window pieces | max depth | located (kind, lambda) |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for name, r in data["check"].items():
                lines.append(
                    f"| {name} | {f(r['true_error'])} | {f(r['abserr'])} | "
                    f"{f(r['error_over_abserr'], '.2g')} | {r['honest']} | "
                    f"{r['converged']} | {r['regions']} | {r['window_pieces']} | "
                    f"{r['max_depth']} | "
                    f"{', '.join(p['kind'] + ' ' + format(p['lambda'] or 0, '.4g') for p in r['stationary_points'])} |"
                )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
