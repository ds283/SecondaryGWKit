"""levin-phase-floor prompt 06c, section 2: the parent's numbers for the new tests, and test 11.

Runs every problem of AdaptiveLevin/tests/test_levin_global_budget.py (tests 1-7) and every problem
test_levin_quadrature.py drives _adaptive_levin() with (test 11, restated in
test_levin_roundoff_exit.levin_quadrature_problems()) two ways:

- the parent: the parent commit's own AdaptiveLevin/levin_quadrature.py (prov.PARENT_SHA, read with
  `git show` into a temporary directory and imported under another name);
- the new code: the working tree's module.

For tests 1-7 it records value, abserr, converged, regions (and how many are fallback), depth,
solves and phase-limited regions under both, the true error against the test's own reference, and
sum_i |v_i| over each call's leaves (each re-solved on its own interval, by its own module). It
checks the closed-form references of tests 1, 5 and 6 against mpmath.quad, subdivided.

For test 11 (descriptive; cost is not an argument, README section 5 rule 4) it records the same
under both codes, and counts, from the driver's DEBUG summary, any call in which the loop's running
stop test and the returned "converged" flag disagree.

Writes ../out/parent_demo.txt. No Ray and no datastore.
"""

import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))

import mpmath  # noqa: E402

import AdaptiveLevin.levin_quadrature as NEW  # noqa: E402
from AdaptiveLevin.tests import test_levin_global_budget as T  # noqa: E402
from AdaptiveLevin.tests.test_levin_roundoff_exit import (  # noqa: E402
    TEST2A_SPAN,
    TEST2B_SPAN,
    _test2a_problem,
    _test2b_problem,
    levin_quadrature_problems,
)

OUT = prov.OUT / "parent_demo.txt"
INPUTS = (
    prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_global_budget.py",
    prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_roundoff_exit.py",
)


def run(module, span, f, theta, basis=None, **kwargs):
    if basis is None:
        basis = module._Basis_SinCos(theta)
    return module._adaptive_levin(span, f, basis, **kwargs)


def sum_abs_values(module, data, f, theta, rtol, order):
    basis = module._Basis_SinCos(theta)
    return sum(
        abs(
            module._adaptive_levin_subregion(
                (r.start, r.end), f, basis, None, chebyshev_order=order, rtol=rtol
            )["value"]
        )
        for r in data["regions"]
    )


def describe(data, ref=None, rtol=None, atol=None):
    s = (
        f"value {data['value']!r} abserr {data['abserr']!r} converged {data['converged']}"
        f" regions {data['num_regions']} (fallback {data['num_simple_regions']}) depth"
        f" {data['max_depth']} solves {data['evaluations']} phase-limited"
        f" {data['num_phase_limited_regions']}"
    )
    if ref is not None:
        true = abs(data["value"] - ref)
        s += f" | true error {true:.4e} honest {true <= data['abserr']}"
        if rtol is not None:
            req = max(atol, rtol * abs(ref))
            s += (
                f" | true/request {true / req:.4f} abserr/request"
                f" {data['abserr'] / req:.4f}"
            )
    return s


def quad_check(lines, label, closed, integrand, a, b, pieces):
    with mpmath.workdps(30):
        nodes = mpmath.linspace(mpmath.mpf(a), mpmath.mpf(b), pieces + 1)
        q = mpmath.quad(integrand, nodes)
        c = closed()
        lines.append(
            f"   {label}: closed form {mpmath.nstr(c, 25)}; mpmath.quad over {pieces} pieces"
            f" {mpmath.nstr(q, 25)}; difference {mpmath.nstr(abs(c - q), 3)}"
        )


def main():
    start = time.perf_counter()
    PARENT, parent_sha = prov.load_parent()
    lines = prov.header(
        __file__, "section 2: the parent's numbers, and test 11", INPUTS
    )
    lines.append(
        f"# parent: {prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py, sha256 {parent_sha}"
    )
    lines.append("")

    lines.append(
        "== References: closed forms against mpmath.quad (30 digits), subdivided"
    )
    _, _, _, e1 = T.test1_problem()
    quad_check(
        lines,
        "test 1",
        e1,
        lambda x: -mpmath.cos(3 * x)
        * mpmath.sin(mpmath.mpf(T.TEST1_C) / 2 * (x - mpmath.mpf(T.TEST1_XS)) ** 2)
        + mpmath.cos(3 * x)
        * mpmath.cos(mpmath.mpf(T.TEST1_C) / 2 * (x - mpmath.mpf(T.TEST1_XS)) ** 2),
        0,
        1,
        400,
    )
    _, _, _, e5 = T.test5_problem()
    with mpmath.workdps(30):
        k = mpmath.mpf(1) / 3
        q5 = mpmath.quad(
            lambda x: abs(x - k) * mpmath.sin(100 * x),
            list(mpmath.linspace(0, k, 50)) + list(mpmath.linspace(k, 1, 100))[1:],
        )
        lines.append(
            f"   test 5: closed form {mpmath.nstr(e5(), 25)}; mpmath.quad split at 1/3"
            f" {mpmath.nstr(q5, 25)}; difference {mpmath.nstr(abs(e5() - q5), 3)}"
        )
    _, _, _, e6 = T.test6_problem()
    quad_check(
        lines,
        "test 6",
        e6,
        lambda x: mpmath.exp(-x) * mpmath.sin(mpmath.mpf(T.TEST6_C) * (x - x * x)),
        0,
        1,
        400,
    )
    lines.append("")

    problems = [
        ("test 1", T.test1_problem(), T.TEST1_SPAN, 16),
        ("test 2", T.test2_problem(), T.TEST2_SPAN, T.TEST2_ORDER),
        ("test 3", T.test3_problem(), T.TEST3_SPAN, 16),
        ("test 5", T.test5_problem(), T.TEST5_SPAN, 16),
        ("test 6", T.test6_problem(), T.TEST6_SPAN, 16),
    ]
    lines.append("== Tests 1-7: the parent and the new code")
    for label, (f, theta, kwargs, expression), span, order in problems:
        ref, ref_err = T._reference(expression)
        rtol, atol = kwargs["rtol"], kwargs["atol"]
        lines.append(
            f"-- {label}: span {span}, kwargs {kwargs}; reference {ref!r} (own error"
            f" {ref_err:.1e}); rtol |I| {rtol * abs(ref):.4e}"
        )
        for name, module in (("parent", PARENT), ("new", NEW)):
            data = run(module, span, f, theta, **kwargs)
            sv = sum_abs_values(module, data, f, theta, rtol, order)
            lines.append(f"   {name:6s} {describe(data, ref, rtol, atol)}")
            lines.append(
                f"          sum_i |v_i| / |I| = {sv / abs(ref):.1f}; Levin leaves"
                f" {sum(1 for r in data['regions'] if r.type == 'Levin')}; leaves with an"
                f" infinite error {sum(1 for r in data['regions'] if not math.isfinite(r.total_err))}"
            )
            if label in ("test 3", "test 6") or data["num_regions"] <= 4:
                for r in data["regions"]:
                    lines.append(f"            {r}")
            if name == "parent":
                missed = (
                    not data["converged"]
                    and abs(data["value"] - ref) <= rtol * abs(ref)
                ) or (data["converged"] and abs(data["value"] - ref) > rtol * abs(ref))
                lines.append(f"          the parent's flag missed: {missed}")
    for label, problem, span in (
        ("test 4 (06b 2a)", _test2a_problem(), TEST2A_SPAN),
        ("test 4 (06b 2b)", _test2b_problem(), TEST2B_SPAN),
    ):
        f, theta, kwargs, _, expression = problem
        ref, _ = T._reference(expression)
        lines.append(f"-- {label}: span {span}, kwargs {kwargs}")
        for name, module in (("parent", PARENT), ("new", NEW)):
            data = run(module, span, f, theta, **kwargs)
            lines.append(f"   {name:6s} {describe(data, ref)}")
    f, theta, kwargs = T.test7_problem()
    lines.append(f"-- test 7: span {T.TEST7_SPAN}, kwargs {kwargs}; reference 0")
    for name, module in (("parent", PARENT), ("new", NEW)):
        data = run(module, T.TEST7_SPAN, f, theta, **kwargs)
        lines.append(f"   {name:6s} {describe(data, 0.0)}")
    lines.append("")

    lines.append("== Test 12: the trust check one level down (06c section 8)")
    for label, (f, theta, kwargs, expression), span in (
        ("12a bump", T.test12a_problem(), T.TEST12A_SPAN),
        ("12b SincIntegral", T.test12b_problem(), T.TEST12B_SPAN),
        ("12c family (R) x_s 0.382 c 1e5", T.test12c_problem(), T.TEST12C_SPAN),
        ("12d spike at the cap", T.test12d_problem(), T.TEST12D_SPAN),
    ):
        ref, ref_err = T._reference(expression)
        rtol = kwargs.get("rtol", NEW.DEFAULT_LEVIN_RELTOL)
        atol = kwargs["atol"]
        lines.append(
            f"-- {label}: span {span}, kwargs {kwargs}; reference {ref!r} (own error"
            f" {ref_err:.1e}); request {max(atol, rtol * abs(ref)):.4e}"
        )
        for name, module in (("parent", PARENT), ("new", NEW)):
            data = run(module, span, f, theta, **kwargs)
            lines.append(
                f"   {name:6s} {describe(data, ref, rtol, atol)} untrusted"
                f" {data.get('num_untrusted_regions')}"
            )
            if name == "new" and data["num_regions"] <= 12:
                for r in data["regions"]:
                    lines.append(
                        f"            {r} | delta1 {r.delta1!r} delta2 {r.delta2!r}"
                        f" declared {r.declared_resolution!r} total_err {r.total_err!r}"
                    )
    # 12b: r along SincIntegral's leftmost line, from its own solves at each depth
    f, theta, kwargs, _ = T.test12b_problem()
    basis = NEW._Basis_SinCos(theta)

    def solve(span):
        return NEW._adaptive_levin_subregion(
            span, f, basis, None, chebyshev_order=12, rtol=kwargs["rtol"]
        )

    a, b = T.TEST12B_SPAN
    ratios = []
    for depth in range(10):
        c = (b + a) / 2.0
        cl = (c + a) / 2.0
        cr = (b + c) / 2.0
        vp = solve((a, b))["value"]
        vh = solve((a, c))["value"] + solve((c, b))["value"]
        vq = (solve((a, cl))["value"] + solve((cl, c))["value"]) + (
            solve((c, cr))["value"] + solve((cr, b))["value"]
        )
        d1, d2 = abs(vp - vh), abs(vh - vq)
        ratios.append(
            f"d{depth} [{a:.6g}, {b:.6g}] r={d2 / d1 if d1 else float('inf'):.4g}"
        )
        b = c
    lines.append("   12b r along the leftmost line: " + "; ".join(ratios))
    lines.append("")

    lines.append(
        "== Test 11 (descriptive) and test 12e: every problem test_levin_quadrature.py drives,"
        " parent -> new"
    )
    lines.append(
        "   problem: converged; regions; depth; solves; phase-limited | value; abserr |"
        " honest (true error <= abserr against the test module's reference) | trust check"
        " (new): checks, found untrusted, returned untrusted | running-test/returned-flag"
        " disagreements and running-test rejections (new)"
    )
    references = T.levin_quadrature_references()
    totals = {
        "disagree": 0,
        "reject": 0,
        "parent_conv": 0,
        "new_conv": 0,
        "calls": 0,
        "honest_to_dishonest": [],
    }
    for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
        if raises is not None:
            outcome = []
            for module in (PARENT, NEW):
                try:
                    run(module, span, f, theta, basis=basis, **kwargs)
                    outcome.append("returned")
                except raises:
                    outcome.append(f"raised {raises.__name__}")
            lines.append(f"   {name}: parent {outcome[0]}, new {outcome[1]}")
            continue
        t0 = time.perf_counter()
        p = run(PARENT, span, f, theta, basis=basis, **kwargs)
        t1 = time.perf_counter()
        with prov.DebugCapture() as cap:
            n = run(NEW, span, f, theta, basis=basis, **kwargs)
        t2 = time.perf_counter()
        totals["calls"] += 1
        totals["parent_conv"] += p["converged"]
        totals["new_conv"] += n["converged"]
        totals["disagree"] += cap.disagreements()
        totals["reject"] += cap.resum_rejections()
        ref, _ = T._reference(references[name])
        honest_p = abs(p["value"] - ref) <= p["abserr"]
        honest_n = abs(n["value"] - ref) <= n["abserr"]
        if honest_p and not honest_n:
            totals["honest_to_dishonest"].append(name)
        checks, _, found, returned, _ = cap.trust_counts()
        lines.append(
            f"   {name}: {p['converged']} -> {n['converged']}; {p['num_regions']} ->"
            f" {n['num_regions']}; {p['max_depth']} -> {n['max_depth']}; {p['evaluations']} ->"
            f" {n['evaluations']}; {p['num_phase_limited_regions']} ->"
            f" {n['num_phase_limited_regions']} | {p['value']!r} -> {n['value']!r};"
            f" {p['abserr']:.4e} -> {n['abserr']:.4e} | honest {honest_p} -> {honest_n}"
            f" (true {abs(p['value'] - ref):.3e} -> {abs(n['value'] - ref):.3e}) |"
            f" {checks} {found} {returned} | {cap.disagreements()}"
            f" {cap.resum_rejections()} | {t1 - t0:.2f} s -> {t2 - t1:.2f} s"
        )
    lines.append(
        f"   totals: {totals['calls']} driven calls; converged parent {totals['parent_conv']},"
        f" new {totals['new_conv']}; running-test/flag disagreements {totals['disagree']};"
        f" running-test rejections {totals['reject']}; honest at the parent and dishonest"
        f" under the new code: {totals['honest_to_dishonest'] or 'none'}"
    )
    lines.append("")
    lines.append(
        f"# wall time {time.perf_counter() - start:.1f} s; uptime at end: {prov.uptime()}"
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
