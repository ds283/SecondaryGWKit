"""levin-phase-floor prompt 06d (a round-off floor for the Clenshaw-Curtis rule): the numbers its
tests and log quote, at the parent and under 06d.

The parent is the dispatch SHA's module (prov.PARENT_SHA, 06c's code ad9a5f4), read with `git show`
and imported under another name. "new" is the working tree's AdaptiveLevin/levin_quadrature.py.

Sections:
  A. test 6: every problem test_levin_quadrature.py drives -- the parent's own module, the
     in-process parent of AdaptiveLevin/tests/test_levin_cc_floor.py (_parent_cc(), checked equal
     to it bit for bit), and 06d; which parent calls have a fallback leaf; honesty of each.
  B. test 2: per cell, one fallback solve -- true error, phi_CC's terms, the nested pair and the
     parent's eq. (151) floor; and the call, under 06d.
  C. test 1 and test 3: the floor's terms, and test 3's parent floors level by level.
  D. test 4: the stationary-phase oracle, parent and 06d.
  E. 06b's test 3, (a) and (b), parent and 06d, with and without the bypass.
  F. the f' part of the node term, against the theta' part and the whole floor, over every
     fallback leaf of every 06d call above.
  G. the three-Bessel refinement test's five rungs, parent and 06d.

Writes ../out/parent_demo.txt. No Ray and no datastore; nothing is written outside ../out/
(prov.load_parent() writes the parent's source to a temporary directory).
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

REPO = prov.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
import AdaptiveLevin.tests.test_levin_cc_floor as T  # noqa: E402
import AdaptiveLevin.tests.test_levin_roundoff_exit as R  # noqa: E402
from AdaptiveLevin.tests.test_levin_global_budget import (  # noqa: E402
    levin_quadrature_references,
)

OUT = prov.OUT / "parent_demo.txt"
TEST_FILES = (
    REPO / "AdaptiveLevin" / "tests" / "test_levin_cc_floor.py",
    REPO / "AdaptiveLevin" / "tests" / "test_levin_roundoff_exit.py",
    REPO / "AdaptiveLevin" / "tests" / "test_levin_global_budget.py",
    REPO / "LiouvilleGreen" / "tests" / "test_three_bessel.py",
)

lines = []


def out(text=""):
    print(text, flush=True)
    lines.append(text)


def cc_terms(span, f, basis, N=lq.DEFAULT_LEVIN_CHEBSHEV_ORDER):
    """phi_CC's four parts for one fallback solve, recomputed from its samples as the code forms
    them: (sum term, argument term, node theta' part, node f' part)."""
    grid, Dmat = lq.chebyshev_matrices(span, N)
    _, theta_prime, _, _, _ = basis.build_Levin_data(grid, Dmat, need_AmatT=False)
    n = 2 * N - 1
    fine_grid, D_fine = lq.chebyshev_matrices(span, n)
    w = np.fabs(lq._cc_weights(span, n))
    f_fine = np.vstack([[func(x) for x in fine_grid] for func in f])
    pairs = [basis.eval_basis_and_arguments(x) for x in fine_grid]
    b = np.array([p[0] for p in pairs]).T
    a = np.array([p[1] for p in pairs], dtype=float).T
    fp = np.vstack([np.matmul(D_fine, row) for row in f_fine])
    u = lq._UNIT_ROUNDOFF
    S = float(np.dot(w, np.sum(np.fabs(f_fine * b), axis=0)))
    sum_term = (n + lq._CC_ROUNDOFF_PER_TERM_COUNT) * u * S
    arg_term = u * float(np.dot(w, np.sum(np.fabs(f_fine) * np.fabs(a), axis=0)))
    delta = lq._cc_node_error_bound(span)
    tp = float(np.max(np.fabs(theta_prime)))
    node_tp = delta * tp * float(np.dot(w, np.sum(np.fabs(f_fine), axis=0)))
    node_fp = delta * float(np.dot(w, np.sum(np.fabs(fp), axis=0)))
    return sum_term, arg_term, node_tp, node_fp


def signature(data):
    return R._call_signature(data)


def reference(name_or_expr):
    if callable(name_or_expr):
        return T._reference(name_or_expr)
    return T._reference(levin_quadrature_references()[name_or_expr])


FPRIME = []  # (label, node_fp, node_tp, phi) over 06d's fallback leaves


def collect_fprime(label, data, f, basis):
    for region in data["regions"]:
        if region.type != "direct":
            continue
        s, a, ntp, nfp = cc_terms((region.start, region.end), f, basis)
        FPRIME.append((label, nfp, ntp, s + a + ntp + nfp))


def honest_mark(value, abserr, ref):
    return "honest" if abs(value - ref) <= abserr else "DISHONEST"


def main():
    parent, parent_sha256 = prov.load_parent()
    for line in prov.header(
        __file__, "the numbers 06d's tests and log quote", TEST_FILES
    ):
        out(line)
    out(f"# parent {prov.PARENT_SHA} (06c's code ad9a5f4), module sha256 {parent_sha256}")
    out()

    # ------------------------------------------------------------------ A
    out("== A. test 6: the problems test_levin_quadrature.py drives")
    out(
        "   name | parent: fallback leaves, regions, depth, solves, value, abserr, true, honest |"
        " 06d: the same | in-process parent == parent module | 06d == parent"
    )
    references = levin_quadrature_references()
    with_fallback = []
    for name, span, f, theta, kwargs, basis, raises in R.levin_quadrature_problems():
        if raises is not None:
            out(f"   {name}: raises {raises.__name__} (not compared)")
            continue
        pb = parent._Basis_SinCos(theta) if basis is None else basis
        nb = lq._Basis_SinCos(theta) if basis is None else basis
        t0 = time.perf_counter()
        p = parent._adaptive_levin(span, f, pb, **kwargs)
        tp = time.perf_counter() - t0
        with T.parent_floor():
            q = lq._adaptive_levin(
                span, f, lq._Basis_SinCos(theta) if basis is None else basis, **kwargs
            )
        n = lq._adaptive_levin(span, f, nb, **kwargs)
        if basis is None:
            collect_fprime(f"test 6 {name}", n, f, nb)
        ref, _ = reference(name)
        same_inproc = signature(p) == signature(q)
        same_new = signature(p) == signature(n)
        if p["num_simple_regions"] > 0:
            with_fallback.append(name)
        out(
            f"   {name} | {p['num_simple_regions']}, {p['num_regions']}, {p['max_depth']},"
            f" {p['evaluations']}, {p['value']!r}, {p['abserr']:.4e},"
            f" {abs(p['value'] - ref):.3e}, {honest_mark(p['value'], p['abserr'], ref)},"
            f" conv {p['converged']} | {n['num_simple_regions']}, {n['num_regions']},"
            f" {n['max_depth']}, {n['evaluations']}, {n['value']!r}, {n['abserr']:.4e},"
            f" {abs(n['value'] - ref):.3e}, {honest_mark(n['value'], n['abserr'], ref)},"
            f" conv {n['converged']} | {same_inproc} | {same_new} ({tp:.1f} s at the parent)"
        )
    out(f"   parent calls with a fallback leaf: {with_fallback}")
    out()

    # ------------------------------------------------------------------ B
    out(
        "== B. test 2: one fallback solve per cell (true error against the floor alone), and"
        " the call under 06d (atol 1e-40, rtol 1e-17)"
    )
    out(
        "   family reduced scale | true | phi_CC = sum + arg + node(theta') + node(f') | pair |"
        " parent eq.(151) | true/phi | true/parent || call: regions, depth, phase-limited, abserr,"
        " true, true/abserr"
    )
    for family, reduced, scale in T.test2_cells():
        span, f, theta, expression = T._test2_problem(family, scale, reduced)
        basis = lq._Basis_SinCos(theta)
        data, parent_floor = T._cc_solve(span, f, theta)
        ref, _ = T._reference(expression)
        true = abs(data["value"] - ref)
        s, a, ntp, nfp = cc_terms(span, f, basis)
        FPRIME.append(
            (f"test 2 solve {family} {reduced} {scale:.0e}", nfp, ntp, s + a + ntp + nfp)
        )
        call = lq._adaptive_levin(span, f, basis, **T.TEST2_KWARGS)
        ctrue = abs(call["value"] - ref)
        collect_fprime(f"test 2 call {family} {reduced} {scale:.0e}", call, f, basis)
        out(
            f"   {family} {str(reduced):5s} {scale:.0e} | {true:.3e} | {data['phase_err']:.3e} ="
            f" {s:.2e} + {a:.2e} + {ntp:.2e} + {nfp:.2e} | {data['abserr_direct']:.3e} |"
            f" {parent_floor:.3e} | {true / data['phase_err']:.3g} |"
            f" {true / parent_floor if parent_floor > 0 else float('inf'):.3g} ||"
            f" {call['num_regions']}, {call['max_depth']}, {call['num_phase_limited_regions']},"
            f" {call['abserr']:.3e}, {ctrue:.3e}, {ctrue / call['abserr']:.3g}"
        )
    out()

    # ------------------------------------------------------------------ C
    out("== C. test 1 and test 3: the floor's terms")
    f, theta, _ = T._stationary_problem(0.5)
    span = (0.5, 0.5 + 2.0**-8)
    data, parent_floor = T._cc_solve(span, f, theta)
    s, a, ntp, nfp = cc_terms(span, f, lq._Basis_SinCos(theta))
    FPRIME.append(("test 1", nfp, ntp, s + a + ntp + nfp))
    ref, _ = T._stationary_reference(span, 0.5)
    out(
        f"   test 1 {span}: parent {parent_floor}; 06d {data['phase_err']:.4e} = {s:.3e} + {a:.3e}"
        f" + {ntp:.3e} + {nfp:.3e}; pair {data['abserr_direct']:.3e};"
        f" true {abs(data['value'] - ref):.3e}"
    )
    xs = 0.7236
    f, theta, _ = T._stationary_problem(xs)
    span = (0.72265625, 0.72265625 + 2.0**-8)
    for level in range(4):
        data, parent_floor = T._cc_solve(span, f, theta)
        c = 0.5 * (span[0] + span[1])
        left, _ = T._cc_solve((span[0], c), f, theta)
        right, _ = T._cc_solve((c, span[1]), f, theta)
        s, a, ntp, nfp = cc_terms(span, f, lq._Basis_SinCos(theta))
        FPRIME.append((f"test 3 level {level}", nfp, ntp, s + a + ntp + nfp))
        ref, _ = T._stationary_reference(span, xs)
        out(
            f"   test 3 level {level} [{span[0]!r}, {span[1]!r}]: parent eq.(151)"
            f" {parent_floor:.4e}; 06d {data['phase_err']:.4e} = {s:.3e} + {a:.3e} + {ntp:.3e} +"
            f" {nfp:.3e}; halves {left['phase_err'] + right['phase_err']:.4e}"
            f" ({(left['phase_err'] + right['phase_err']) / data['phase_err']:.3f} of the leaf's);"
            f" pair {data['abserr_direct']:.3e}; true {abs(data['value'] - ref):.3e}"
        )
        span = (span[0], c) if xs < c else (c, span[1])
    out()

    # ------------------------------------------------------------------ D
    out("== D. test 4: the stationary-phase oracle, atol = rtol = 1e-10")
    theta = {"theta": lambda x: 1.0e6 * (x - x * x)}
    f = [lambda x: math.exp(-x), lambda x: 0.0]
    ref, _ = reference("stationary_phase")
    for label, module in (("parent", parent), ("06d", lq)):
        t0 = time.perf_counter()
        d = module._adaptive_levin(
            (0.0, 1.0), f, module._Basis_SinCos(theta), atol=1e-10, rtol=1e-10
        )
        out(
            f"   {label}: leaves {d['num_regions']} (fallback {d['num_simple_regions']}), depth"
            f" {d['max_depth']}, solves {d['evaluations']}, phase-limited"
            f" {d['num_phase_limited_regions']}, converged {d['converged']}, value"
            f" {d['value']!r}, abserr {d['abserr']:.4e}, true {abs(d['value'] - ref):.4e},"
            f" {honest_mark(d['value'], d['abserr'], ref)}"
            f" ({time.perf_counter() - t0:.1f} s)"
        )
    out()

    # ------------------------------------------------------------------ E
    out("== E. 06b's test 3: (a) theta = 1e-6 e^x on [0, 6]; (b) the node-floor problem")
    for label, (f, theta), (ref, _), span, kwargs in (
        (
            "(a)",
            R._test3_problem(),
            R._test3_reference(),
            R.TEST3_SPAN,
            dict(atol=R.TEST3_ATOL, rtol=R.TEST3_RTOL),
        ),
        (
            "(b)",
            R._test3b_problem(),
            R._test3b_reference(),
            R.TEST3B_SPAN,
            dict(atol=R.TEST3B_ATOL, rtol=R.TEST3B_RTOL),
        ),
    ):
        for code, module in (("parent", parent), ("06d", lq)):
            for bypass in (True, False):
                d = module._adaptive_levin(
                    span,
                    f,
                    module._Basis_SinCos(theta),
                    _test_only_bypass_roundoff_premise=bypass,
                    **kwargs,
                )
                if code == "06d":
                    collect_fprime(
                        f"06b test 3{label} bypass={bypass}",
                        d,
                        f,
                        lq._Basis_SinCos(theta),
                    )
                out(
                    f"   {label} {code} bypass={bypass}: regions {d['num_regions']} (fallback"
                    f" {d['num_simple_regions']}), depth {d['max_depth']}, phase-limited"
                    f" {d['num_phase_limited_regions']}, converged {d['converged']}, abserr"
                    f" {d['abserr']:.4e}, true {abs(d['value'] - ref):.3e},"
                    f" {honest_mark(d['value'], d['abserr'], ref)}"
                )
    out()

    # ------------------------------------------------------------------ F
    out(
        "== F. the node term's f' part over every 06d fallback solve above: node(f') against"
        " node(theta') and against the whole floor"
    )
    ratios_node = [nfp / (nfp + ntp) if nfp + ntp > 0 else 0.0 for _, nfp, ntp, _ in FPRIME]
    ratios_phi = [nfp / phi if phi > 0 else 0.0 for _, nfp, _, phi in FPRIME]
    out(f"   solves {len(FPRIME)}")
    out(
        f"   node(f') / (node(theta') + node(f')): median {np.median(ratios_node):.3g}, max"
        f" {max(ratios_node):.3g}, min {min(ratios_node):.3g}"
    )
    out(
        f"   node(f') / phi_CC: median {np.median(ratios_phi):.3g}, max {max(ratios_phi):.3g};"
        f" solves where it exceeds 1/2 of phi_CC: {sum(r > 0.5 for r in ratios_phi)}"
    )
    worst = sorted(FPRIME, key=lambda r: -(r[1] / r[3] if r[3] > 0 else 0.0))[:8]
    for label, nfp, ntp, phi in worst:
        out(f"      {label}: node(f') {nfp:.3e}, node(theta') {ntp:.3e}, phi {phi:.3e}")
    out()

    # ------------------------------------------------------------------ G
    out("== G. the three-Bessel refinement test's rungs (quad_JJJ, k, q, s = 1.3, 1.7, 2.1)")
    from LiouvilleGreen.bessel_phase import bessel_phase
    from LiouvilleGreen.three_bessel_integrals import quad_JJJ

    k, q, s = 1.3, 1.7, 2.1
    analytic = (math.pi / 4.0) / (k * q * s)
    max_x = 1.0e12
    phase_data = [bessel_phase(0.5, 1.075 * c * max_x) for c in (k, q, s)]
    original = lq._adaptive_levin
    for code, driver in (("parent", parent._adaptive_levin), ("06d", original)):
        lq._adaptive_levin = driver
        try:
            for rtol in (1.0e-8, 1.0e-10, 1.0e-12, 1.0e-13, 1.0e-14):
                r = quad_JJJ(
                    *phase_data,
                    0.0,
                    0.0,
                    0.0,
                    k,
                    q,
                    s,
                    max_x,
                    atol=1.0e-16,
                    rtol=rtol,
                )
                out(
                    f"   {code} rtol {rtol:.0e}: value {r.value!r}, true"
                    f" {abs(r.value - analytic):.4e}, abserr {r.abserr:.4e}, phase_limited"
                    f" {r.phase_limited}"
                )
        finally:
            lq._adaptive_levin = original

    out()
    out(f"# uptime at end: {prov.uptime()}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
