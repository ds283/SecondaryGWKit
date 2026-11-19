"""
levin-phase-floor prompt 07c, Part B: the code compared with Part A's variant (S) (section 1.1, section 9.5 Y8) and with
the parent, on the problems this Part runs on the laptop: every call of section 3's tests (a)-(f) in
AdaptiveLevin/tests/test_levin_chase_stop.py; every cell of the existing tests in which the stop acts or whose
assertions it changes (test_levin_trust_yardstick test 2, test_levin_leaf_values test 1, test_levin_roundoff_exit
test 1, test_levin_global_budget tests 1 and 3); and 07b's five calls of test_levin_rate_rules (b)-(e). The
LiouvilleGreen draws are compared by lg_draws_07c.py.

Three modules are loaded under names of their own: the commit's AdaptiveLevin/levin_quadrature.py; (S),
docs/levin-phase-floor/07c/measurement/scripts/variants/levin_quadrature_07c_S.py, its SHA-256 checked; and the
parent, the dispatch commit cff4237f's module read with `git show`. Each call is run under all three and compared:
every result key but "elapsed" and the stop's counts, and every returned region field by field (each float by its
hex). The commit against (S) must be identical on every call, with (S)'s three counts (num_s07c_stops,
num_s07c_stopped_leaves, num_s07c_counted) equal to the commit's (num_chase_stops, num_chase_stopped_leaves,
num_chase_noise_bisections). Against the parent: every call in which no root stops must be identical; the others are
reported.

The tests run the parent's code path as the commit with _LEVIN_CHASE_STOP_LIMIT patched to infinity. Each call they
run that way is run so here too, and compared with the parent's module field by field; and the values the tests quote
as the parent's are printed from the parent's module.

Run from the repository root:  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/07c/code/compare_s_07c.py
Writes docs/levin-phase-floor/07c/code/out/compare_s_07c.txt and .json. No Ray and no datastore.
"""

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
S_PATH = (
    REPO
    / "docs/levin-phase-floor/07c/measurement/scripts/variants/levin_quadrature_07c_S.py"
)
S_SHA256 = "09088683c6da28fc7e003b31a999d1120b902c221ce9f59b755673c5a7e5a6fb"
PARENT_SHA = "cff4237f"
CHASE = ("num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections")
S_COUNTS = ("num_s07c_stops", "num_s07c_stopped_leaves", "num_s07c_counted")


def _exec(name, source, label):
    module = types.ModuleType(name)
    module.__file__ = label
    sys.modules[name] = module
    exec(compile(source, label, "exec"), module.__dict__)
    return module


def canon(x):
    import numpy as np

    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.ndarray):
        return [canon(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {
            str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if x is None or isinstance(x, str):
        return x
    if hasattr(x, "__dict__"):
        return canon(vars(x))
    return repr(x)


def record(data):
    return canon(
        {k: v for k, v in data.items() if k not in ("elapsed",) + CHASE + S_COUNTS}
    )


def main():
    import AdaptiveLevin.levin_quadrature as commit
    import AdaptiveLevin.tests.test_levin_chase_stop as cs
    import AdaptiveLevin.tests.test_levin_rate_rules as rr
    import AdaptiveLevin.tests.test_levin_trust_yardstick as ty
    import AdaptiveLevin.tests.test_levin_leaf_values as lv
    import AdaptiveLevin.tests.test_levin_roundoff_exit as rx
    import AdaptiveLevin.tests.test_levin_global_budget as gb

    raw = S_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == S_SHA256
    smod = _exec("levin_quadrature_07c_S", raw.decode(), str(S_PATH))
    source = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parent = _exec("levin_quadrature_parent_cff4237f", source, PARENT_SHA)

    def sincos(span, f, theta, **kw):
        return lambda m: m.adaptive_levin_sincos(span, f, theta, **kw)

    def driver(span, f, theta, **kw):
        return lambda m: m._adaptive_levin(span, f, m._Basis_SinCos(theta), **kw)

    fb, thb = cs.problem(cs.B_A, cs.B_KAPPA, cs.ETA)
    fc3, thc3 = cs.problem(cs.C3_A, cs.C3_KAPPA, cs.ETA)
    ff2, thf2 = cs.problem(cs.F2_A, cs.F2_KAPPA, 0.0)
    calls = [
        (
            "07c (b), (c1), (c2), (d), (e)",
            sincos(cs.SPAN, fb, thb, atol=cs.B_ATOL, rtol=cs.B_RTOL),
        ),
        (
            "07c (e) at depth_max 8",
            sincos(
                cs.SPAN,
                fb,
                thb,
                atol=cs.B_ATOL,
                rtol=cs.B_RTOL,
                depth_max=cs.E_DEPTH_MAX,
            ),
        ),
        ("07c (c3)", sincos(cs.SPAN, fc3, thc3, atol=cs.B_ATOL, rtol=cs.B_RTOL)),
        (
            "07c (f1)",
            sincos((0.0, 1.0), cs.F1_F, cs.F1_THETA, atol=cs.F1_ATOL, rtol=cs.F1_RTOL),
        ),
        ("07c (f2)", sincos(cs.SPAN, ff2, thf2, atol=cs.F2_ATOL, rtol=cs.F2_RTOL)),
        (
            "07b (b)",
            sincos(
                (0.0, 1.0),
                rr.Q_F,
                rr.Q_THETA,
                atol=rr.B_ATOL,
                rtol=rr.B_RTOL,
                chebyshev_order=rr.Q_ORDER,
            ),
        ),
        (
            "07b (c)",
            sincos(
                rr.C_SPAN,
                rr.Q_F,
                rr.Q_THETA,
                atol=rr.C_ATOL,
                rtol=rr.C_RTOL,
                chebyshev_order=rr.Q_ORDER,
            ),
        ),
        (
            "07b (d)",
            sincos(
                rr.D_SPAN,
                rr.D_F,
                rr.D_THETA,
                atol=rr.D_ATOL,
                rtol=rr.D_RTOL,
                depth_max=rr.D_DEPTH_MAX,
            ),
        ),
        (
            "07b (e1)",
            sincos((0.0, 1.0), rr.E1_F, rr.E1_THETA, atol=rr.E_ATOL, rtol=rr.E_RTOL),
        ),
        (
            "07b (e2)",
            sincos((0.0, 6.0), rr.F_23, rr.E2_THETA, atol=rr.E_ATOL, rtol=rr.E_RTOL),
        ),
    ]
    for name, span, f, theta, kwargs in ty.test2_cells():
        calls.append((f"yardstick test 2 {name}", sincos(span, f, theta, **kwargs)))
    for name, span, f, theta, kwargs, _, _ in lv.test1_cells():
        calls.append((f"leaf values test 1 {name}", driver(span, f, theta, **kwargs)))
    f, theta = rx._test1_problem()
    calls.append(
        (
            "roundoff exit test 1",
            driver(rx.TEST1_SPAN, f, theta, atol=rx.TEST1_ATOL, rtol=rx.TEST1_RTOL),
        )
    )
    for label, (f, theta, kwargs, _), span in (
        ("global budget test 1", gb.test1_problem(), gb.TEST1_SPAN),
        ("global budget test 3", gb.test3_problem(), gb.TEST3_SPAN),
    ):
        calls.append((label, driver(span, f, theta, **kwargs)))

    lines, rows = [], []
    for name, call in calls:
        c, s, p = call(commit), call(smod), call(parent)
        with mock.patch.object(commit, "_LEVIN_CHASE_STOP_LIMIT", math.inf):
            cp = call(commit)
        counts = tuple(c[k] for k in CHASE)
        row = {
            "call": name,
            "counts": counts,
            "counts_equal_S": counts == tuple(s[k] for k in S_COUNTS),
            "identical_to_S": record(c) == record(s),
            "identical_to_parent": record(c) == record(p),
            "parent_path_identical_to_parent": record(cp) == record(p),
            "parent_path_stops": cp["num_chase_stops"],
            "value_hex": float(c["value"]).hex(),
            "abserr_hex": float(c["abserr"]).hex(),
            "converged": bool(c["converged"]),
            "regions": c["num_regions"],
            "parent_value_hex": float(p["value"]).hex(),
            "parent_abserr_hex": float(p["abserr"]).hex(),
            "parent_converged": bool(p["converged"]),
            "parent_regions": p["num_regions"],
            "parent_max_depth": p["max_depth"],
        }
        rows.append(row)
        lines.append(
            f"{name}: stops/stopped/counted {counts}; equal to (S) {row['identical_to_S']} (counts "
            f"{row['counts_equal_S']}); identical to the parent {row['identical_to_parent']}; parent path (limit inf) "
            f"identical to the parent {row['parent_path_identical_to_parent']}; regions {row['parent_regions']} -> "
            f"{row['regions']}; converged {row['parent_converged']} -> {row['converged']}; value "
            f"{row['value_hex']} abserr {row['abserr_hex']}; parent value {row['parent_value_hex']} abserr "
            f"{row['parent_abserr_hex']}"
        )
    acts = [r for r in rows if r["counts"][0] > 0]
    quiet = [r for r in rows if r["counts"][0] == 0]
    summary = (
        f"{len(rows)} calls; identical to (S) with equal counts: "
        f"{sum(r['identical_to_S'] and r['counts_equal_S'] for r in rows)}; the stop acts in {len(acts)}; of the "
        f"{len(quiet)} where it does not, identical to the parent: {sum(r['identical_to_parent'] for r in quiet)}; "
        f"the parent's code path (limit inf) identical to the parent's module: "
        f"{sum(r['parent_path_identical_to_parent'] for r in rows)} of {len(rows)}"
    )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    provenance = [
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"git HEAD {head}; commit module sha256 {hashlib.sha256(Path(commit.__file__).read_bytes()).hexdigest()}; "
        f"test module sha256 {hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest()}",
        f"(S) {S_PATH.relative_to(REPO)} sha256 {S_SHA256}; parent {PARENT_SHA}",
    ]
    text = "\n".join(provenance + [summary, ""] + lines) + "\n"
    (HERE / "out" / "compare_s_07c.txt").write_text(text)
    (HERE / "out" / "compare_s_07c.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
