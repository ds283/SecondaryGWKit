"""levin-phase-floor prompt 06g, section 2: the cells of tests 1-7, at the parent (06g's dispatch
commit's AdaptiveLevin/levin_quadrature.py, prov.load_parent()) and at the working tree's, with the
numbers their docstrings quote; test 3's descriptive table of checked leaves over tests 1 and 3's
cells (every leaf the trust check is taken on, those later bisected included); the budget clause as
the prompt writes it against as built (06g's log, deviation 1); and the calls on which the
node-limited stop, or the e' combination, lose a convergence the parent had (06g's log,
observations).

Run from the repository root: PYTHONPATH=. ./venv/bin/python <this script>. Writes ../out/cells.txt.
No Ray, no datastore.
"""

import math
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import mpmath
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
from AdaptiveLevin.tests import test_levin_leaf_values as T13  # noqa: E402
from AdaptiveLevin.tests import test_levin_node_sum as T46  # noqa: E402
from AdaptiveLevin.tests import test_levin_roundoff_exit as T7  # noqa: E402

OUT = prov.OUT / "cells.txt"
P, P_SHA = prov.load_parent()

BANDS = ("r<1/16", "1/16<=r<1/4", "1/4<=r<1", "r>=1")
SIDES = ("d1>phi", "d1<=phi")


def run(module, span, f, theta, **kwargs):
    return module._adaptive_levin(span, f, module._Basis_SinCos(theta), **kwargs)


@contextmanager
def recording(module):
    made = []
    original = module.used_interval

    class Recorded(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    with mock.patch.object(module, "used_interval", Recorded):
        yield made


def summary(data, ref=None):
    s = (
        f"value {data['value']!r} abserr {data['abserr']:.4e} converged {data['converged']}"
        f" regions {data['num_regions']} (fallback {data['num_simple_regions']}) depth"
        f" {data['max_depth']} solves {data['evaluations']} floor-limited"
        f" {data['num_phase_limited_regions']} untrusted {data['num_untrusted_regions']}"
    )
    if "node_limited" in data:
        s += (
            f" node-limited {data['node_limited']} sumE {data['node_error_signed_sum']:.4e}"
            f" sum|E| {data['node_error_abs_sum']:.4e}"
        )
    if ref is not None:
        s += f" | true error {abs(data['value'] - ref):.4e}"
    return s


def region_line(r, new):
    s = (
        f"      [{r.start!r}, {r.end!r}] d{r.depth} {r.type} res {r.abserr:.4e} floor"
        f" {r.phase_err:.4e}"
    )
    if r.convergence_ratio is not None:
        s += f" r {r.convergence_ratio:.4g} d1 {r.delta1:.4e} d2 {r.delta2:.4e}"
    s += f" trusted {r.trusted} limited {r.phase_limited}"
    if new:
        s += f" value {r.value_kind}"
        if r.node_error_signed is not None:
            s += f" E {r.node_error_signed:.4e} floor-less-E {r.phase_err_less_node:.4e}"
    return s


def classify(regions, table):
    for r in regions:
        if r.delta1 is None:
            continue
        band, side = T13.r_class(r)
        table[(band, side)] = table.get((band, side), 0) + 1


def table_lines(table, title):
    lines = [f"   {title}"]
    lines.append("      " + " ".join(f"{b:>12s}" for b in BANDS) + "   (counts, d1>phi / d1<=phi)")
    for side in SIDES:
        lines.append(
            f"   {side:>7s} " + " ".join(f"{table.get((b, side), 0):12d}" for b in BANDS)
        )
    lines.append(f"      total {sum(table.values())}")
    return lines


def main():
    lines = prov.header(
        __file__,
        "section 2 tests 1-7: the cells, at the parent and at 06g",
        inputs=(
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_leaf_values.py",
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_node_sum.py",
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_roundoff_exit.py",
        ),
    )
    lines.append(f"# parent {prov.PARENT_SHA} module sha256 {P_SHA}")
    lines.append("")

    checked_new = {}
    checked_parent = {}

    lines.append("== Test 1's cells: parent -> 06g")
    for name, span, f, theta, kwargs, expression, identical in T13.test1_cells():
        ref, _ = T13.reference(expression)
        with recording(P) as made_p:
            old = run(P, span, f, theta, **kwargs)
        with recording(lq) as made_n:
            new = run(lq, span, f, theta, **kwargs)
        classify(made_p, checked_parent)
        classify(made_n, checked_new)
        to, tn = abs(old["value"] - ref), abs(new["value"] - ref)
        lines.append(f"-- {name} (identical asserted: {identical}) kwargs {kwargs}")
        lines.append(f"   parent: {summary(old, ref)}")
        lines += [region_line(r, False) for r in old["regions"]]
        lines.append(f"   06g:    {summary(new, ref)}")
        lines += [region_line(r, True) for r in new["regions"]]
        lines.append(f"   true error parent/06g {to / tn if tn else float('inf'):.4g}")

    lines.append("")
    lines.append("== The noise-level cell (test_noise_level_cell)")
    span, f, theta = (-1.0, 1.0), [lambda x: 1.0 / (1.0 + 25.0 * x * x), lambda x: 0.0], {
        "theta": lambda x: x + 1.0,
        "theta_deriv": lambda x: 1.0 + 0.0 * x,
    }
    ref, _ = T13.reference(
        lambda: mpmath.quad(lambda x: mpmath.sin(x + 1) / (1 + 25 * x * x), [-1, 0, 1])
    )
    old = run(P, span, f, theta, atol=1e-20, rtol=1e-12)
    new = run(lq, span, f, theta, atol=1e-20, rtol=1e-12)
    lines.append(f"   parent: {summary(old, ref)}")
    lines.append(f"   06g:    {summary(new, ref)}")
    lines += [region_line(r, True) for r in new["regions"]]

    lines.append("")
    lines.append("== Test 3's cell: SincIntegral (06c's test 12b)")
    span, f, theta, kwargs, expression = T13.test3_sinc_problem()
    ref, _ = T13.reference(expression)
    with recording(P) as made_p:
        old = run(P, span, f, theta, **kwargs)
    with recording(lq) as made_n:
        new = run(lq, span, f, theta, **kwargs)
    classify(made_p, checked_parent)
    classify(made_n, checked_new)
    lines.append(f"   parent: {summary(old, ref)}")
    lines += [region_line(r, False) for r in old["regions"]]
    lines.append(f"   06g:    {summary(new, ref)}")
    lines += [region_line(r, True) for r in new["regions"]]

    lines.append("")
    lines.append(
        "== Test 3's descriptive table over tests 1 and 3's cells: every checked leaf (returned or"
        " later bisected), by r band and delta1 against the leaf's floor phi_P"
    )
    lines += table_lines(checked_parent, "parent")
    lines += table_lines(checked_new, "06g")

    lines.append("")
    lines.append("== Test 4: GRZIntegral(1000)")
    span, f, theta, kwargs = T46._grz1000()
    old = run(P, span, f, theta, **kwargs)
    new = run(lq, span, f, theta, **kwargs)
    signs = [r.node_error_signed for r in new["regions"] if r.node_error_signed is not None]
    combination = 0.0
    for r in new["regions"]:
        combination = combination + r.total_err
    lines.append(f"   parent: {summary(old)}")
    lines.append(f"   06g:    {summary(new)}")
    lines.append(
        f"   fallback leaves {len(signs)}, E > 0 {sum(e > 0 for e in signs)}, E < 0"
        f" {sum(e < 0 for e in signs)}; sum|E| / |sum E|"
        f" {new['node_error_abs_sum'] / abs(new['node_error_signed_sum']):.4g}; the parent's"
        f" combination on 06g's partition {combination:.4e}, / abserr"
        f" {combination / new['abserr']:.4g}"
    )

    for label, cell, kwargs in (
        ("Test 5", T46.TEST5, T46.TEST5_KWARGS),
        ("Test 6", T46.TEST6, T46.TEST6_KWARGS),
    ):
        span, f, theta, expression = T46._far_sine(*cell)
        ref, _ = T13.reference(expression)
        old = run(P, span, f, theta, **kwargs)
        new = run(lq, span, f, theta, **kwargs)
        lines.append("")
        lines.append(f"== {label}: sin({cell[2]} (x - {cell[0]:g})) on [x0, x0 + {cell[1]}], {kwargs}")
        lines.append(f"   parent: {summary(old, ref)}")
        lines += [region_line(r, False) for r in old["regions"]]
        lines.append(f"   06g:    {summary(new, ref)}")
        lines += [region_line(r, True) for r in new["regions"]]
        request = max(kwargs["atol"], kwargs["rtol"] * abs(new["value"]))
        lines.append(
            f"   request {request:.4e}; sum e' {new['abserr'] - new['abserr_node_signed']:.4e};"
            f" |V - sum E - ref| {abs(new['value'] - new['node_error_signed_sum'] - ref):.4e}"
        )

    lines.append("")
    lines.append("== Test 7")
    span, f, declared, plain, kwargs = T7.test7a_problem()
    b = lq._Basis_SinCos(declared)
    root = lq._adaptive_levin_subregion(span, f, b, None, rtol=1e-7)
    c = 0.5 * (span[0] + span[1])
    L = lq._adaptive_levin_subregion((span[0], c), f, b, None, rtol=1e-7)
    R = lq._adaptive_levin_subregion((c, span[1]), f, b, None, rtol=1e-7)
    lines.append(
        f"   7a: rtol {kwargs['rtol']!r}; root floor {root['phase_err']:.4e} (eq. 151"
        f" {root['phase_err_roundoff']:.4e}, declared {root['phase_err_declared']:.4e},"
        f" {root['phase_err_declared'] / root['phase_err']:.4f}); halves whole"
        f" {(L['phase_err'] + R['phase_err']) / root['phase_err']:.4f}, eq. 151"
        f" {(L['phase_err_roundoff'] + R['phase_err_roundoff']) / root['phase_err_roundoff']:.4f},"
        f" declared {(L['phase_err_declared'] + R['phase_err_declared']) / root['phase_err_declared']:.4f}"
        f"; request {kwargs['rtol'] * abs(root['value']):.4e}"
    )
    for tag, theta in (("with theta_abserr", declared), ("without", plain)):
        lines.append(f"   7a {tag}: parent: {summary(run(P, span, f, theta, **kwargs))}")
        lines.append(f"   7a {tag}: 06g:    {summary(run(lq, span, f, theta, **kwargs))}")
    f = [lambda x: 1.0, lambda x: 0.0]
    theta = T7._exp_phase(T7.TEST7A_A)
    root = T7._root_solve(T7.TEST7B_SPAN, f, theta)
    rtol = T7.TEST7B_REQUEST_FRACTION * root["phase_err"] / abs(root["value"])
    lines.append(f"   7b: root floor {root['phase_err']:.4e}, rtol {rtol!r}")
    lines.append(f"   7b: parent: {summary(run(P, T7.TEST7B_SPAN, f, theta, atol=1e-30, rtol=rtol))}")
    lines.append(f"   7b: 06g:    {summary(run(lq, T7.TEST7B_SPAN, f, theta, atol=1e-30, rtol=rtol))}")

    # the budget clause as written: with no "met now" condition (deviation 1)
    lines.append("")
    lines.append(
        "== The budget clause as the prompt writes it (no exemption when the call meets its"
        " request now) against as built, on test 7b's family (theta = 1e4 e^x on [0, L], f = 1)"
        " at requests kappa x the root's floor"
    )
    src = prov.LEVIN.read_text()
    old_text = (
        "        if running_eprime + running_node <= request:\n"
        "            return False\n\n"
        "        region = leaf.region"
    )
    assert src.count(old_text) == 1
    import importlib.util
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lpf06g-literal-"))
    path = tmp / "levin_quadrature_literal_clause.py"
    path.write_text(src.replace(old_text, "        region = leaf.region"))
    spec = importlib.util.spec_from_file_location("levin_quadrature_literal_clause", path)
    literal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(literal)
    for length in (1.0, 0.8, 0.6):
        span = (0.0, length)
        root = T7._root_solve(span, f, theta)
        for kappa in (0.75, 0.8, 0.9):
            kw = dict(atol=1e-30, rtol=kappa * root["phase_err"] / abs(root["value"]))
            n = run(lq, span, f, theta, **kw)
            m = run(literal, span, f, theta, **kw)
            lines.append(
                f"   L {length} kappa {kappa}: as built: converged {n['converged']} regions"
                f" {n['num_regions']} depth {n['max_depth']} solves {n['evaluations']} | as"
                f" written: converged {m['converged']} regions {m['num_regions']} depth"
                f" {m['max_depth']} solves {m['evaluations']} untrusted {m['num_untrusted_regions']}"
            )

    # convergence the parent had, lost under 06g
    lines.append("")
    lines.append(
        "== Convergence lost under 06g on fallback cells far from the origin: sin(c (x - x0))"
        " (or f = 1, theta = 4 (x - x0)) at requests where the parent converges"
    )
    cases = [
        ("f=1, theta=4(x-x0), x0=1e6, L=4", (1e6, 1e6 + 4.0), [lambda x: 1.0, lambda x: 0.0],
         {"theta": lambda x: 4.0 * (x - 1e6), "theta_deriv": lambda x: 4.0 + 0.0 * x}, 3.162e-10),
        ("sin(1.0 (x - 1e8)), L=2", (1e8, 1e8 + 2.0), [lambda x: 1.0, lambda x: 0.0],
         {"theta": lambda x: 1.0 * (x - 1e8), "theta_deriv": lambda x: 1.0 + 0.0 * x}, 1e-9),
        ("sin(1.0 (x - 3e7)), L=3", (3e7, 3e7 + 3.0), [lambda x: 1.0, lambda x: 0.0],
         {"theta": lambda x: 1.0 * (x - 3e7), "theta_deriv": lambda x: 1.0 + 0.0 * x}, 1e-9),
        ("sin(1.0 (x - 1e8)), L=3", (1e8, 1e8 + 3.0), [lambda x: 1.0, lambda x: 0.0],
         {"theta": lambda x: 1.0 * (x - 1e8), "theta_deriv": lambda x: 1.0 + 0.0 * x}, 1e-9),
    ]
    for label, span, f, theta, rtol in cases:
        old = run(P, span, f, theta, atol=1e-40, rtol=rtol)
        new = run(lq, span, f, theta, atol=1e-40, rtol=rtol)
        lines.append(f"   {label}, rtol {rtol:g}:")
        lines.append(f"      parent: {summary(old)}")
        lines.append(f"      06g:    {summary(new)}")
        lines += [region_line(r, True) for r in new["regions"]]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
