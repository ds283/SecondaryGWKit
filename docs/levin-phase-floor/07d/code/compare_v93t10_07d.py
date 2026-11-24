"""
levin-phase-floor prompt 07d, Part B: the code compared with Part A''s variant V93T10 (section 1.1, section 9.4.5 Z8)
and with the parent, on the problems this Part runs on the laptop:
- every call of section 3's tests (c)-(f) in AdaptiveLevin/tests/test_levin_cc_estimate.py that runs the driver, and
  every single Clenshaw-Curtis solve of its tests (a) and (b), each key of the solve's dict compared;
- the cells and calls 07c's compare_s_07c.py ran: the nine yardstick cells (test_levin_trust_yardstick test 2), the
  seven leaf-value cells (test_levin_leaf_values test 1), the round-off exit's test 1, the global budget's tests 1
  and 3, 07b's five rate-rule calls (b)-(e2) and 07c's chase-stop calls (b), (c3), (f1) and (f2).

Three modules are loaded under names of their own: the commit's AdaptiveLevin/levin_quadrature.py; V93T10,
docs/levin-phase-floor/07d/measurement-a2/scripts/variants/levin_quadrature_a2_V93T10.py, its SHA-256 checked; and the
parent, the dispatch commit ca1d8cfa's module read with `git show`. Each call is run under all three, and under the
commit on 07c's code path (cc_rule_07c(), the parent's Clenshaw-Curtis rule and the gate at 6 pi), and compared:
every result key but "elapsed" and the counts, and every returned region field by field (each float by its hex).
- The commit against V93T10 must be identical on every call, with V93T10's counts (read from its module dict _A2_C,
  cleared before each call) equal to the commit's: num_a2_cc93 to num_cc_solves, num_a2_route_changed to
  num_cc_gate_route_changes, num_a2_resampled to num_cc_resampled.
- Against the parent: every call with num_cc_solves 0 must be identical; the others are reported.
- The commit on 07c's code path must be identical to the parent on every call.

Run from the repository root:  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/07d/code/compare_v93t10_07d.py
Writes docs/levin-phase-floor/07d/code/out/compare_v93t10_07d.txt and .json. No Ray and no datastore.
"""

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
V_PATH = (
    REPO
    / "docs/levin-phase-floor/07d/measurement-a2/scripts/variants/levin_quadrature_a2_V93T10.py"
)
V_SHA256 = "f85976fc6bf6d5e00ff3d10dc7d363cd1449a9b81ab2a751685b3409d5963b9d"
PARENT_SHA = "ca1d8cfa"
COUNTS = (
    "num_cc_solves",
    "num_cc_gate_route_changes",
    "num_cc_noise_set",
    "num_cc_resampled",
)
V_COUNTS = ("num_a2_cc93", "num_a2_route_changed", "num_a2_resampled")
SOLVE_ONLY = ("cc_noise_set", "cc_num_resampled", "cc_gate_route_changed")


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
        {
            k: v
            for k, v in data.items()
            if k not in ("elapsed",) + COUNTS + V_COUNTS + SOLVE_ONLY
        }
    )


def main():
    import numpy as np

    import AdaptiveLevin.levin_quadrature as commit
    import AdaptiveLevin.tests.test_levin_cc_estimate as ce
    import AdaptiveLevin.tests.test_levin_chase_stop as cs
    import AdaptiveLevin.tests.test_levin_rate_rules as rr
    import AdaptiveLevin.tests.test_levin_trust_yardstick as ty
    import AdaptiveLevin.tests.test_levin_leaf_values as lv
    import AdaptiveLevin.tests.test_levin_roundoff_exit as rx
    import AdaptiveLevin.tests.test_levin_global_budget as gb

    raw = V_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V_SHA256
    vmod = _exec("levin_quadrature_07d_V93T10", raw.decode(), str(V_PATH))
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
    parent = _exec("levin_quadrature_parent_ca1d8cfa", source, PARENT_SHA)

    def sincos(span, f, theta, **kw):
        return lambda m: m.adaptive_levin_sincos(span, f, theta, **kw)

    def driver(span, f, theta, **kw):
        return lambda m: m._adaptive_levin(span, f, m._Basis_SinCos(theta), **kw)

    calls = []
    for factor in (8.0, 9.99, 5.0):
        calls.append(
            (
                f"07d (c2) V = {factor} pi",
                sincos(
                    ce.SPAN,
                    ce.F_C,
                    ce.linear_theta(factor * np.pi),
                    atol=1e-16,
                    rtol=1e-10,
                ),
            )
        )
    A = 16.0 * np.pi
    calls.append(
        (
            "07d (c3)",
            sincos(
                ce.SPAN,
                ce.F_C,
                {
                    "theta": lambda x: A * (x - 0.5) ** 2,
                    "theta_deriv": lambda x: 2.0 * A * (x - 0.5),
                },
                atol=1e-16,
                rtol=1e-10,
            ),
        )
    )
    calls.append(
        ("07d (d)", sincos((0.0, 1.0), ce.F_D, ce.THETA_D, atol=1e-14, rtol=1e-7))
    )
    for eta in ce.ETA_E:
        calls.append(
            (
                f"07d (e), (f) eta = {eta}",
                sincos(ce.SPAN, ce.f_e(eta), ce.THETA_E, **ce.KWARGS_E),
            )
        )
    fb, thb = cs.problem(cs.B_A, cs.B_KAPPA, cs.ETA)
    fc3, thc3 = cs.problem(cs.C3_A, cs.C3_KAPPA, cs.ETA)
    ff2, thf2 = cs.problem(cs.F2_A, cs.F2_KAPPA, 0.0)
    calls += [
        ("07c (b)", sincos(cs.SPAN, fb, thb, atol=cs.B_ATOL, rtol=cs.B_RTOL)),
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

    def run_variant(call):
        vmod._A2_C.clear()
        data = call(vmod)
        counts = tuple(int(vmod._A2_C.get(k, 0)) for k in V_COUNTS)
        return data, counts

    lines, rows = [], []
    for name, call in calls:
        c = call(commit)
        v, vcounts = run_variant(call)
        p = call(parent)
        with ce.cc_rule_07c():
            cp = call(commit)
        counts = tuple(c[k] for k in COUNTS)
        row = {
            "call": name,
            "counts": counts,
            "V93T10_counts": vcounts,
            "counts_equal_V93T10": (counts[0], counts[1], counts[3]) == vcounts,
            "identical_to_V93T10": record(c) == record(v),
            "identical_to_parent": record(c) == record(p),
            "path_07c_identical_to_parent": record(cp) == record(p),
            "value_hex": float(c["value"]).hex(),
            "abserr_hex": float(c["abserr"]).hex(),
            "converged": bool(c["converged"]),
            "regions": c["num_regions"],
            "max_depth": c["max_depth"],
            "parent_value_hex": float(p["value"]).hex(),
            "parent_abserr_hex": float(p["abserr"]).hex(),
            "parent_converged": bool(p["converged"]),
            "parent_regions": p["num_regions"],
            "parent_max_depth": p["max_depth"],
        }
        rows.append(row)
        lines.append(
            f"{name}: counts {counts} (V93T10 {vcounts}, equal {row['counts_equal_V93T10']}); identical to V93T10 "
            f"{row['identical_to_V93T10']}; identical to the parent {row['identical_to_parent']}; 07c path identical "
            f"to the parent {row['path_07c_identical_to_parent']}; regions {row['parent_regions']} -> "
            f"{row['regions']}; depth {row['parent_max_depth']} -> {row['max_depth']}; converged "
            f"{row['parent_converged']} -> {row['converged']}; value {row['value_hex']} abserr {row['abserr_hex']}; "
            f"parent value {row['parent_value_hex']} abserr {row['parent_abserr_hex']}"
        )

    # the single Clenshaw-Curtis solves of tests (a) and (b): every key of the commit's dict but the 07d flags against
    # V93T10's _adaptive_levin_subregion_cc(), on the same arguments
    solves = []
    for order in ce.ORDERS:
        for name, (f, _, _, _) in ce.a_cases(order).items():
            solves.append((order, name, f, ce.THETA_A))
        solves.append((order, "poly", [lambda x: 0.0 * x, ce.poly], ce.THETA_ZERO))
        for eta in ce.ETA_B:
            solves.append(
                (
                    order,
                    f"scatter {eta}",
                    [
                        lambda x, e=eta: 0.0 * x,
                        lambda x, e=eta: np.exp(-x) + e * ce.noise(x),
                    ],
                    ce.THETA_ZERO,
                )
            )
    solve_lines = []
    solve_same = 0
    for order, name, f, theta in solves:
        out = {}
        for label, m in (("commit", commit), ("V93T10", vmod)):
            basis = m._Basis_SinCos(theta)
            grid, D = m.chebyshev_matrices(ce.SPAN, order)
            _, tp, _, _, V = basis.build_Levin_data(
                grid, D, need_AmatT=False, vectorize_cache={}
            )
            out[label] = m._adaptive_levin_subregion_cc(
                ce.SPAN, f, basis, order, V, tp, None, None, vectorize_cache={}
            )
        same = record(out["commit"]) == record(out["V93T10"])
        solve_same += same
        solve_lines.append(
            f"solve N={order} {name}: identical to V93T10 {same}; value {out['commit']['value'].hex()} abserr_direct "
            f"{out['commit']['abserr_direct'].hex()} phase_err {out['commit']['phase_err'].hex()}"
        )

    acts = [r for r in rows if r["counts"][0] > 0]
    quiet = [r for r in rows if r["counts"][0] == 0]
    summary = (
        f"{len(rows)} driver calls; identical to V93T10 with equal counts: "
        f"{sum(r['identical_to_V93T10'] and r['counts_equal_V93T10'] for r in rows)}; the fix acts "
        f"(num_cc_solves > 0) in {len(acts)}; of the {len(quiet)} where it does not, identical to the parent: "
        f"{sum(r['identical_to_parent'] for r in quiet)}; the commit on 07c's code path identical to the parent: "
        f"{sum(r['path_07c_identical_to_parent'] for r in rows)} of {len(rows)}; converged changes (parent -> "
        f"commit): {sum(r['converged'] != r['parent_converged'] for r in rows)}. {len(solves)} single solves of "
        f"tests (a) and (b): identical to V93T10's _adaptive_levin_subregion_cc() in every key: {solve_same}"
    )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    provenance = [
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"git HEAD {head}; commit module sha256 {hashlib.sha256(Path(commit.__file__).read_bytes()).hexdigest()}; "
        f"test module sha256 {hashlib.sha256(Path(ce.__file__).read_bytes()).hexdigest()}",
        f"V93T10 {V_PATH.relative_to(REPO)} sha256 {V_SHA256}; parent {PARENT_SHA}",
    ]
    text = "\n".join(provenance + [summary, ""] + lines + [""] + solve_lines) + "\n"
    (HERE / "out" / "compare_v93t10_07d.txt").write_text(text)
    (HERE / "out" / "compare_v93t10_07d.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "solves": solve_lines}, indent=1)
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
