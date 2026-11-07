"""levin-phase-floor 07 section 2.5, beside its cases: the ComputeTargets test that does not finish under section 1's
code. New in 07.

On section 1's code (without section 1.3) the ComputeTargets root did not finish: it ran 31 minutes, against 291 s
at the dispatch commit, inside ComputeTargets/tests/test_quadsource_integral.py's
TestDirectQuadOracle.test_short_range, and was stopped. The call it was in is a three-Bessel block of
QuadSourceIntegral._three_bessel_Levin ("analytic J4", rtol 1e-10, atol 1e-25, chebyshev_order 24, x_span =
(-7.457061423316192, -4.749011222213982)) whose group phase has a root of theta' inside the span; at 32 pi the
window covers the whole span.

This script runs that test in-process with the driver wrapped at the module attribute adaptive_levin_sincos()
calls it through (AdaptiveLevin.levin_quadrature._adaptive_levin), and at the first call that locates a point
and has rtol 1e-10 and that span, it stops the test and keeps the call's arguments in memory. Then, on that call:
- section 1's code at depth_max 0, 2, 4, 6, 8, 10 and 12, each with section 2.5's account (analysis07.py): the
  window leaves' summed error against the request, depth by depth;
- the parent's code (the dispatch commit's module, git show) at the call's own depth_max (20);
- the "bar lifted" variant (calibrate.py's) at depth_max 20.
Writes out/ct_stuck_call.json. Nothing else is written; no Ray, no store.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))

SPAN = (-7.457061423316192, -4.749011222213982)
DEPTHS = (0, 2, 4, 6, 8, 10, 12)


class _Captured(Exception):
    pass


def capture():
    import unittest

    import AdaptiveLevin.levin_quadrature as lq

    original = lq._adaptive_levin
    box = {}

    def wrapped(x_span, f, basis, **kw):
        if (
            kw.get("rtol") == 1e-10
            and abs(x_span[0] - SPAN[0]) < 1e-12
            and abs(x_span[1] - SPAN[1]) < 1e-12
        ):
            rates = basis.independent_phase_rates()
            if rates is not None:
                loc = lq._locate_stationary_points(
                    x_span,
                    basis.independent_phases(),
                    rates,
                    {},
                    lq._LEVIN_WINDOW_HALF_PHASE,
                    lq._LEVIN_COALESCENCE_LAMBDA,
                    lq._LEVIN_WINDOW_PIECE_PHASE,
                )
                if loc["points"]:
                    box.update(x_span=x_span, f=f, basis=basis, kw=dict(kw))
                    raise _Captured()
        return original(x_span, f, basis, **kw)

    lq._adaptive_levin = wrapped
    import ComputeTargets.tests.test_quadsource_integral as T

    case = T.TestDirectQuadOracle("test_short_range")
    result = unittest.TestResult()
    case.run(result)
    lq._adaptive_levin = original
    if not box:
        raise SystemExit(
            f"the call was not captured: {result.errors or result.failures}"
        )
    return lq, box


def run(module, box, depth_max, label):
    import analysis07 as A

    recorder = A.Recorder(module)
    recorder.install()
    kw = dict(box["kw"])
    kw["depth_max"] = depth_max
    t0 = time.perf_counter()
    result = module._adaptive_levin(box["x_span"], box["f"], box["basis"], **kw)
    wall = time.perf_counter() - t0
    recorder.remove()
    rec = A.summary(result, depth_max=depth_max)
    rec["label"] = label
    rec["depth_max"] = depth_max
    rec["wall_s"] = wall
    rec["request"] = A.request_of(result, kw["atol"], kw["rtol"])
    rec["abserr_over_request"] = result["abserr"] / rec["request"]
    win = [r for r in result["regions"] if getattr(r, "window", False)]
    rec["window_leaves"] = len(win)
    rec["window_sum_budget_err"] = sum(r.budget_err for r in win)
    rec["window_sum_phase_err_less_node"] = sum(
        (r.phase_err_less_node or 0.0) for r in win
    )
    rec["window_sum_abserr_fallback"] = sum((r.abserr or 0.0) for r in win)
    rec["other_sum_budget_err"] = sum(
        r.budget_err for r in result["regions"] if not getattr(r, "window", False)
    )
    if A.needs_section_2_5(result, depth_max):
        rec["section_2_5"] = A.section_2_5(
            result,
            recorder.made,
            rec["request"],
            A.n_seed_leaves(result) if "stationary_windows" in result else 1,
            depth_max,
            max_listed=6,
        )
    return rec


def main():
    import calibrate

    lq, box = capture()
    out = {
        "header": prov.json_header(
            __file__,
            "the ComputeTargets test that does not finish: its call",
            (
                HERE / "analysis07.py",
                HERE / "calibrate.py",
                prov.REPO / "ComputeTargets" / "tests" / "test_quadsource_integral.py",
                prov.REPO / "ComputeTargets" / "QuadSourceIntegral.py",
            ),
        ),
        "kw": {k: v for k, v in box["kw"].items()},
        "x_span": [float(box["x_span"][0]).hex(), float(box["x_span"][1]).hex()],
        "window_half_phase": lq._LEVIN_WINDOW_HALF_PHASE,
        "coalescence_lambda": lq._LEVIN_COALESCENCE_LAMBDA,
        "runs": [],
    }
    for dm in DEPTHS:
        rec = run(lq, box, dm, "commit")
        out["runs"].append(rec)
        print(
            f"commit depth_max {dm}: abserr {rec['abserr']:.4g} request {rec['request']:.4g} window leaves "
            f"{rec['window_leaves']} window sum e' {rec['window_sum_budget_err']:.4g} (floor less node "
            f"{rec['window_sum_phase_err_less_node']:.4g}, nested pair {rec['window_sum_abserr_fallback']:.4g}) "
            f"converged {rec['converged']} wall {rec['wall_s']:.1f} s",
            flush=True,
        )
    parent, parent_sha = prov.load_module_from_source(
        prov.git("show", f"{prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"),
        "levin_quadrature_parent_bd21bf2",
    )
    rec = run(parent, box, box["kw"].get("depth_max", 20), "parent")
    rec["module_sha256"] = parent_sha
    out["runs"].append(rec)
    print(
        f"parent: value {rec['value']!r} abserr {rec['abserr']:.4g} converged {rec['converged']} "
        f"regions {rec['regions']} phase-limited {rec['num_phase_limited_regions']} wall {rec['wall_s']:.1f} s",
        flush=True,
    )
    bar, bar_sha = calibrate._load("barlifted")
    bar._LEVIN_WINDOW_HALF_PHASE = lq._LEVIN_WINDOW_HALF_PHASE
    bar._LEVIN_COALESCENCE_LAMBDA = lq._LEVIN_COALESCENCE_LAMBDA
    rec = run(bar, box, box["kw"].get("depth_max", 20), "barlifted")
    rec["module_sha256"] = bar_sha
    out["runs"].append(rec)
    print(
        f"bar lifted: value {rec['value']!r} abserr {rec['abserr']:.4g} converged {rec['converged']} "
        f"regions {rec['regions']} window leaves floor-limited {rec.get('window_leaves_phase_limited')} "
        f"wall {rec['wall_s']:.1f} s",
        flush=True,
    )
    (prov.OUT / "ct_stuck_call.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {prov.OUT / 'ct_stuck_call.json'}")


if __name__ == "__main__":
    main()
