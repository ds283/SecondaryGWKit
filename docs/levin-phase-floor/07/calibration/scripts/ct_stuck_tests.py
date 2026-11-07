"""levin-phase-floor 07 section 2.5, beside its cases: the ComputeTargets tests that do not finish under section 1's
code, and the calls that hold them. New in 07.

On section 1's code (without section 1.3) three ComputeTargets tests did not finish: TestDirectQuadOracle.test_short_range
(test_quadsource_integral.py; the root ran 31 minutes inside it, against 291 s for the whole root at the dispatch
commit) and TestNormalisation.test_N_is_constant_over_the_nine_b_02_fixture_cases (test_domenech_oracle.py; 18
minutes inside it when the root was re-run without the first); and a third,
test_kohri_terada_oracle.py's TestPipeline.test_total_is_minus_nine_eighths_of_I_RD, over 600 s when the root was
re-run without those two. Each test is run here, in this process, with the
driver wrapped at its module attribute (AdaptiveLevin.levin_quadrature._adaptive_levin):
- a call that locates no point runs as it stands;
- a call that locates a point runs at depth_max = min(its own, ``CAP``), so that the test can finish, and is
  recorded with its window leaves' summed error against the request and section 2.5's account (analysis07.py); a
  call that reaches ``CAP`` with window leaves there is re-run (a) by the parent's code at its own depth_max, and
  (b) by the "bar lifted" variant (calibrate.py's) at its own depth_max, under a SIGALRM limit of ``BAR_LIMIT_S``.
The test's outcome under the cap is recorded; it is not the test's outcome on section 1's code (that is: did not
finish). Writes out/ct_stuck_tests.json and out/ct_stuck_tests.txt. No Ray, no store.
"""

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))

TESTS = (
    "ComputeTargets.tests.test_quadsource_integral.TestDirectQuadOracle.test_short_range",
    "ComputeTargets.tests.test_domenech_oracle.TestNormalisation.test_N_is_constant_over_the_nine_b_02_fixture_cases",
    "ComputeTargets.tests.test_kohri_terada_oracle.TestPipeline.test_total_is_minus_nine_eighths_of_I_RD",
)
CAP = 10
BAR_LIMIT_S = 300


class _Alarm(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Alarm()


def main():
    import analysis07 as A
    import calibrate

    import AdaptiveLevin.levin_quadrature as lq

    parent, parent_sha = prov.load_module_from_source(
        prov.git("show", f"{prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"),
        "levin_quadrature_parent_bd21bf2",
    )
    bar, bar_sha = calibrate._load("barlifted")
    original = lq._adaptive_levin
    signal.signal(signal.SIGALRM, _on_alarm)
    out = {
        "header": prov.json_header(
            __file__,
            "section 2.5 on the ComputeTargets tests that do not finish",
            (
                HERE / "analysis07.py",
                HERE / "calibrate.py",
                prov.REPO / "ComputeTargets" / "tests" / "test_quadsource_integral.py",
                prov.REPO / "ComputeTargets" / "tests" / "test_domenech_oracle.py",
                prov.REPO / "ComputeTargets" / "QuadSourceIntegral.py",
            ),
        ),
        "cap": CAP,
        "bar_limit_s": BAR_LIMIT_S,
        "window_half_phase": lq._LEVIN_WINDOW_HALF_PHASE,
        "coalescence_lambda": lq._LEVIN_COALESCENCE_LAMBDA,
        "parent_sha256": parent_sha,
        "bar_lifted_sha256": bar_sha,
        "tests": [],
    }
    txt = [f"# {x}" for x in out["header"]] + [""]
    for test_id in TESTS:
        calls = []

        def wrapped(x_span, f, basis, **kw):
            rates = (
                basis.independent_phase_rates()
                if hasattr(basis, "independent_phase_rates")
                else None
            )
            loc = None
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
            own_depth = kw.get("depth_max", lq.DEFAULT_LEVIN_MAX_DEPTH)
            rec = {
                "index": len(calls),
                "notify_label": kw.get("notify_label"),
                "x_span": [float(x_span[0]), float(x_span[1])],
                "atol": kw.get("atol"),
                "rtol": kw.get("rtol"),
                "own_depth_max": own_depth,
                "located": bool(loc and loc["points"]),
            }
            calls.append(rec)
            if not rec["located"]:
                return original(x_span, f, basis, **kw)
            capped = dict(kw)
            capped["depth_max"] = min(own_depth, CAP)
            recorder = A.Recorder(lq)
            recorder.install()
            t0 = time.perf_counter()
            result = original(x_span, f, basis, **capped)
            rec["wall_s"] = time.perf_counter() - t0
            recorder.remove()
            request = A.request_of(result, kw["atol"], kw["rtol"])
            rec.update(
                A.summary(result, depth_max=capped["depth_max"]),
                request=request,
                abserr_over_request=result["abserr"] / request,
            )
            win = [r for r in result["regions"] if r.window]
            rec["window_leaves_at_cap"] = sum(
                1 for r in win if r.depth >= capped["depth_max"]
            )
            rec["window_untrusted"] = sum(1 for r in win if r.trusted is False)
            rec["window_sum_budget_err"] = sum(r.budget_err for r in win)
            rec["other_sum_budget_err"] = sum(
                r.budget_err for r in result["regions"] if not r.window
            )
            if rec["window_leaves_at_cap"]:
                t0 = time.perf_counter()
                pres = parent._adaptive_levin(x_span, f, basis, **kw)
                rec["parent"] = {
                    "value": pres["value"],
                    "abserr": pres["abserr"],
                    "converged": pres["converged"],
                    "regions": pres["num_regions"],
                    "max_depth": pres["max_depth"],
                    "phase_limited": pres["num_phase_limited_regions"],
                    "wall_s": time.perf_counter() - t0,
                }
                bar._LEVIN_WINDOW_HALF_PHASE = lq._LEVIN_WINDOW_HALF_PHASE
                bar._LEVIN_COALESCENCE_LAMBDA = lq._LEVIN_COALESCENCE_LAMBDA
                t0 = time.perf_counter()
                signal.alarm(BAR_LIMIT_S)
                try:
                    bres = bar._adaptive_levin(x_span, f, basis, **kw)
                    signal.alarm(0)
                    rec["bar_lifted"] = {
                        "value": bres["value"],
                        "abserr": bres["abserr"],
                        "converged": bres["converged"],
                        "node_limited": bres["node_limited"],
                        "regions": bres["num_regions"],
                        "max_depth": bres["max_depth"],
                        "window_leaves_floor_limited": sum(
                            1 for r in bres["regions"] if r.window and r.phase_limited
                        ),
                        "window_untrusted": sum(
                            1
                            for r in bres["regions"]
                            if r.window and r.trusted is False
                        ),
                        "wall_s": time.perf_counter() - t0,
                    }
                except _Alarm:
                    rec["bar_lifted"] = {
                        "did_not_finish": True,
                        "wall_s": time.perf_counter() - t0,
                    }
            return result

        lq._adaptive_levin = wrapped
        cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp(prefix="lpf07-ctstuck-"))
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
            result = unittest.TestResult()
            t0 = time.perf_counter()
            suite.run(result)
            wall = time.perf_counter() - t0
        finally:
            os.chdir(cwd)
            lq._adaptive_levin = original
        outcome = "error" if result.errors else "fail" if result.failures else "pass"
        detail = (result.errors or result.failures or [[None, ""]])[0][1][-2000:]
        located = [c for c in calls if c["located"]]
        out["tests"].append(
            {
                "test": test_id,
                "outcome_under_the_cap": outcome,
                "detail": detail,
                "wall_s": wall,
                "n_calls": len(calls),
                "calls_located": located,
            }
        )
        txt.append(
            f"{test_id}: {len(calls)} calls, {len(located)} locating; outcome under the cap {outcome}; {wall:.1f} s"
        )
        for c in located:
            line = (
                f"   #{c['index']} {c['notify_label']} span {c['x_span']} rtol {c['rtol']} depth "
                f"{c['max_depth']}/{CAP} window leaves {c['window_pieces']} at cap {c['window_leaves_at_cap']} "
                f"untrusted window {c['window_untrusted']} e' window {c['window_sum_budget_err']:.3g} other "
                f"{c['other_sum_budget_err']:.3g} request {c['request']:.3g} converged {c['converged']} "
                f"wall {c['wall_s']:.1f} s"
            )
            if "parent" in c:
                p, b = c["parent"], c["bar_lifted"]
                line += (
                    f"\n      parent: converged {p['converged']} abserr {p['abserr']:.3g} regions {p['regions']} "
                    f"depth {p['max_depth']} floor-limited {p['phase_limited']} wall {p['wall_s']:.1f} s"
                    + (
                        "\n      bar lifted: did not finish"
                        if b.get("did_not_finish")
                        else f"\n      bar lifted: converged {b['converged']} abserr {b['abserr']:.3g} regions "
                        f"{b['regions']} depth {b['max_depth']} window floor-limited "
                        f"{b['window_leaves_floor_limited']} untrusted window {b['window_untrusted']} "
                        f"wall {b['wall_s']:.1f} s"
                    )
                )
            txt.append(line)
        print("\n".join(txt[-6:]), flush=True)
    (prov.OUT / "ct_stuck_tests.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    (prov.OUT / "ct_stuck_tests.txt").write_text("\n".join(txt) + "\n")
    print(f"wrote {prov.OUT / 'ct_stuck_tests.json'}")


if __name__ == "__main__":
    main()
