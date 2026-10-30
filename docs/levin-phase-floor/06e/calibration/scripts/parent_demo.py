"""levin-phase-floor prompt 06e, section 2: the numbers behind tests 1-6 and the changed existing
assertions, at the parent (06d's code: the dispatch SHA's own module, read with `git show`, see
prov.load_parent()) and at 06e's (the working tree).

Sections:
  A  test 1 and 2: the parent's end-node misses on the tests' seeded spans, and its untiled halves
  B  test 3: the offset chase, both codes, and the parent's residual over floor along its deepest
     line, with whether each level's ends tile
  C  test 4: every cell -- the true node error, 06d's third term, phi_node and its parts, both ratios;
     the measured relative error of a supplied theta'; P1's statistic (median over the window cells
     of 06d's third term over phi_node)
  D  test 5: the stationary-phase oracle, both codes
  E  test 6: every driven problem, both codes
  F  the existing assertions that move, both codes
  G  the cost of the node term per fallback solve, both codes

Run from the repository root: PYTHONPATH=. ./venv/bin/python <this script>. Writes
../out/parent_demo.txt. No Ray, no datastore.
"""

import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))

import mpmath  # noqa: E402
import numpy as np  # noqa: E402

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
import AdaptiveLevin.tests.test_levin_end_nodes as T  # noqa: E402
import AdaptiveLevin.tests.test_levin_roundoff_exit as RX  # noqa: E402
from AdaptiveLevin.tests.test_levin_global_budget import (  # noqa: E402
    levin_quadrature_references,
)
from AdaptiveLevin.tests.test_levin_roundoff_exit import (  # noqa: E402
    levin_quadrature_problems,
)

OUT = prov.OUT / "parent_demo.txt"
TEST_FILE = prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_end_nodes.py"

P, P_SHA = prov.load_parent()
# the test module's own parent loader is pointed at the same module, so that its helpers
# (node_term_06d()) use the module whose hash is recorded here
sys.modules[T._PARENT_NAME] = P

lines = []


def out(text=""):
    lines.append(text)
    print(text, flush=True)


def summary(data):
    regions = data["regions"]
    return (
        f"leaves {data['num_regions']} ({data['num_simple_regions']} fallback), depth"
        f" {data['max_depth']}, solves {data['evaluations']}, converged {data['converged']},"
        f" abserr {data['abserr']:.3e}"
    )


def section_A():
    out("== A. Tests 1 and 2: the parent's end nodes")
    spans = T.seeded_spans(T.TEST1_COUNT, T.TEST1_SEED)
    any_miss = 0
    per_order = {order: 0 for order in T.TEST1_ORDERS}
    for a, b in spans:
        missed = False
        for order in T.TEST1_ORDERS:
            x, _ = P.chebyshev_matrices((a, b), order)
            if x[0] != b or x[-1] != a:
                per_order[order] += 1
                missed = True
        any_miss += missed
    out(
        f"   test 1: {len(spans)} spans; the parent's end nodes miss on {any_miss} (at one"
        f" order or more); per order {per_order}"
    )
    spans = T.seeded_spans(T.TEST2_COUNT, T.TEST2_SEED)
    bisections = 0
    untiled = 0
    for a, b in spans:
        level = [(a, b)]
        for _ in range(T.TEST2_LEVELS):
            children = []
            for lo, hi in level:
                c = T._bisect(lo, hi)
                bisections += 1
                bad = False
                for order in (T.N, T.N_FINE):
                    xp, _ = P.chebyshev_matrices((lo, hi), order)
                    xl, _ = P.chebyshev_matrices((lo, c), order)
                    xr, _ = P.chebyshev_matrices((c, hi), order)
                    bad = bad or xp[0] != xr[0] or xp[-1] != xl[-1] or xl[0] != xr[-1]
                untiled += bad
                children += [(lo, c), (c, hi)]
            level = children
    out(
        f"   test 2: {bisections} bisections; at the parent a region and its halves fail to"
        f" tile (at order {T.N} or {T.N_FINE}) on {untiled}"
    )
    out()


def section_B():
    out("== B. Test 3: the offset chase")
    f, theta, expression = T._test3_problem()
    ref, ref_err = T._reference(expression)
    out(
        f"   theta = {T.TEST3_C:g} (x - {T.TEST3_X0:g}) on {T.TEST3_SPAN}, kwargs"
        f" {T.TEST3_KWARGS}; reference {ref!r} (its own error {ref_err:.1e})"
    )
    solves = {}
    original = P._adaptive_levin_subregion

    def spy(x_span, *args, **kwargs):
        data = original(x_span, *args, **kwargs)
        solves[(float(x_span[0]), float(x_span[1]))] = data
        return data

    P._adaptive_levin_subregion = spy
    try:
        start = time.perf_counter()
        old = P._adaptive_levin(T.TEST3_SPAN, f, P._Basis_SinCos(theta), **T.TEST3_KWARGS)
        old_wall = time.perf_counter() - start
    finally:
        P._adaptive_levin_subregion = original
    start = time.perf_counter()
    new = lq._adaptive_levin(T.TEST3_SPAN, f, lq._Basis_SinCos(theta), **T.TEST3_KWARGS)
    new_wall = time.perf_counter() - start
    for name, data, wall in (("06d", old, old_wall), ("06e", new, new_wall)):
        true = abs(data["value"] - ref)
        out(
            f"   {name}: {summary(data)}, true {true:.3e}, true/abserr {true / data['abserr']:.3g},"
            f" maximum Levin depth {max((r.depth for r in data['regions'] if r.type == 'Levin'), default=None)},"
            f" wall {wall:.2f} s"
        )
    deepest = max(old["regions"], key=lambda r: (r.depth, -r.start))
    target = 0.5 * (deepest.start + deepest.end)
    out(
        f"   the parent's deepest line: to its leaf [{deepest.start!r}, {deepest.end!r}] at depth"
        f" {deepest.depth}; per level the residual (Levin: |V - (V_L + V_R)|; fallback: the nested"
        f" pair), the floor (phase_err), their ratio, and whether the region's and its halves'"
        f" ends tile on the parent's grid"
    )
    a, b = T.TEST3_SPAN
    for depth in range(deepest.depth + 1):
        data = solves[(a, b)]
        c = (b + a) / 2.0
        left, right = solves.get((a, c)), solves.get((c, b))
        if data.get("is_direct"):
            residual = data["abserr_direct"]
            kind = "fallback"
        else:
            residual = abs(data["value"] - (left["value"] + right["value"]))
            kind = "Levin"
        order = T.N if not data.get("is_direct") else T.N_FINE
        xp, _ = P.chebyshev_matrices((a, b), order)
        xl, _ = P.chebyshev_matrices((a, c), order)
        xr, _ = P.chebyshev_matrices((c, b), order)
        tiles = xp[0] == xr[0] and xp[-1] == xl[-1] and xl[0] == xr[-1]
        ends = xp[0] == b and xp[-1] == a
        floor = data["phase_err"]
        out(
            f"     depth {depth:2d} {kind:8s} [{a!r}, {b!r}] residual {residual:.3e} floor"
            f" {floor:.3e} residual/floor {residual / floor if floor else float('inf'):.3g};"
            f" own ends exact {ends}; halves tile {tiles}"
        )
        if depth == deepest.depth or left is None:
            break
        a, b = (a, c) if target < c else (c, b)
    out()


def section_C():
    out("== C. Test 4: the node term, cell by cell")
    out(
        "   cell | true node error | 06d's third term | phi_node | |E| | R_g | R_2 | R_c |"
        " 06d/true | phi/true | 06d/phi | solve: true error / max(pair, floor) | max rel. error"
        " of a supplied theta' (u)"
    )
    rows = []
    worst_eta = 0.0
    for cell in T.test4_cells():
        data, (E, R_g, R_2, R_c), args = T.node_term_solve(cell)
        phi = abs(E) + R_g + R_2 + R_c
        node_error = T.true_node_error(cell)
        third = T.node_term_06d(cell)
        ref, _ = cell.reference()
        solve_ratio = abs(data["value"] - ref) / max(
            data["abserr_direct"], data["phase_err"]
        )
        eta = None
        if cell.theta_prime_mp is not None:
            nodes, _ = lq.chebyshev_matrices(cell.span, T.N_FINE)
            with mpmath.workdps(40):
                eta = max(
                    float(
                        abs(mpmath.mpf(v) - cell.theta_prime_mp(mpmath.mpf(x))) / abs(v)
                    )
                    for v, x in zip(args[9], nodes)
                    if v != 0.0
                )
            worst_eta = max(worst_eta, eta)
        rows.append((cell.label, node_error, third, phi, E, R_g, R_2, R_c))
        out(
            f"   {cell.label:36s} | {node_error:+.3e} | {third:.3e} | {phi:.3e} | {abs(E):.3e} |"
            f" {R_g:.3e} | {R_2:.2e} | {R_c:.2e} |"
            f" {third / abs(node_error) if node_error else float('inf'):.3g} |"
            f" {phi / abs(node_error) if node_error else float('inf'):.3g} | {third / phi:.3g} |"
            f" {solve_ratio:.3g} | {'' if eta is None else f'{eta / T.U:.2f}'}"
        )
    out()
    for prefix in ("T2", "T3", "W"):
        group = [r for r in rows if r[0].startswith(prefix)]
        ratios = [r[2] / r[3] for r in group]
        bound = [abs(r[1]) / r[3] for r in group]
        out(
            f"   {prefix}: {len(group)} cells; 06d/phi min {min(ratios):.3g}, median"
            f" {statistics.median(ratios):.4g}, max {max(ratios):.4g}; true/phi max {max(bound):.4g}"
        )
    out(f"   all cells: true/phi max {max(abs(r[1]) / r[3] for r in rows):.4g}")
    shares = [(r[5] / r[3], r[6] / r[3], r[7] / r[3]) for r in rows]
    out(
        f"   shares of phi_node, max over cells: R_g {max(s[0] for s in shares):.3g}, R_2"
        f" {max(s[1] for s in shares):.3g}, R_c {max(s[2] for s in shares):.3g}"
    )
    out(f"   worst relative error of a supplied theta' at a fine node: {worst_eta / T.U:.3f} u")
    window = [r[2] / r[3] for r in rows if r[0].startswith("W")]
    out(
        f"   P1: over the {len(window)} window-like cells, median of 06d's third term over"
        f" phi_node = {statistics.median(window):.4g} (rule: at least 10)"
    )
    out()


def section_D():
    out("== D. Test 5: the stationary-phase oracle")
    f = [lambda x: math.exp(-x), lambda x: 0.0]
    ref, _ = T._reference(levin_quadrature_references()["stationary_phase"])
    for name, M in (("06d", P), ("06e", lq)):
        start = time.perf_counter()
        data = M.adaptive_levin_sincos(
            (0.0, 1.0),
            f,
            theta={"theta": lambda x: 1.0e6 * (x - x * x)},
            atol=1e-10,
            rtol=1e-10,
        )
        wall = time.perf_counter() - start
        out(
            f"   {name}: {summary(data)}, phase-limited {data['num_phase_limited_regions']},"
            f" true {abs(data['value'] - ref):.3e}, wall {wall:.2f} s"
        )
    out()


def section_E():
    out("== E. Test 6: every driven problem, 06d -> 06e")
    references = levin_quadrature_references()
    for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
        if raises is not None:
            out(f"   {name}: raises {raises.__name__} under both")
            continue
        old = P._adaptive_levin(
            span, f, P._Basis_SinCos(theta) if basis is None else basis, **kwargs
        )
        new = lq._adaptive_levin(
            span, f, lq._Basis_SinCos(theta) if basis is None else basis, **kwargs
        )
        ref, _ = T._reference(references[name])
        same = old["value"] == new["value"] and old["abserr"] == new["abserr"]
        out(
            f"   {name}: 06d {old['num_regions']} leaves ({old['num_simple_regions']} fb), depth"
            f" {old['max_depth']}, solves {old['evaluations']}, conv {old['converged']}, abserr"
            f" {old['abserr']:.3e}, true {abs(old['value'] - ref):.3e} | 06e"
            f" {new['num_regions']} ({new['num_simple_regions']} fb), depth {new['max_depth']},"
            f" solves {new['evaluations']}, conv {new['converged']}, abserr {new['abserr']:.3e},"
            f" true {abs(new['value'] - ref):.3e} | |dv|/(abserr sum)"
            f" {abs(new['value'] - old['value']) / (old['abserr'] + new['abserr']):.3g}"
            f"{' | value and abserr bit-identical' if same else ''}"
        )
    out()


def section_F():
    out("== F. The existing assertions that move, 06d and 06e")
    a, b = 1.0, 1.0003
    f = [lambda x: math.exp(-x), lambda x: 0.0]
    theta = {"theta": lambda x: 1.0e6 * x}
    ref, _ = T._reference(levin_quadrature_references()["mode_filter_tight"])
    out(
        "   test_levin_quadrature test_mode_filter_truncation_visible_in_abserr (and"
        " test_levin_cc_floor test 6's mode_filter_tight)"
    )
    values = {}
    for name, M in (("06d", P), ("06e", lq)):
        for rtol in (1e-2, 1e-8):
            data = M.adaptive_levin_sincos(
                (a, b), f, theta, atol=1e-20, rtol=rtol, chebyshev_order=16, depth_max=20
            )
            values[(name, rtol)] = data
            out(
                f"     {name} rtol {rtol:g}: {summary(data)}, truncation"
                f" {data['abserr_truncation']:.3e}, true {abs(data['value'] - ref):.3e}"
            )
        jump = abs(values[(name, 1e-2)]["value"] - values[(name, 1e-8)]["value"])
        out(
            f"     {name}: jump {jump:.3e}; loose abserr >= jump:"
            f" {values[(name, 1e-2)]['abserr'] >= jump}"
        )
    for name, M in (("06d", P), ("06e", lq)):
        basis = M._Basis_SinCos(theta)
        solve = lambda s: M._adaptive_levin_subregion(  # noqa: E731
            s, f, basis, None, chebyshev_order=16, rtol=1e-2
        )["value"]
        c = (b + a) / 2.0
        cl, cr = (c + a) / 2.0, (b + c) / 2.0
        vp, vl, vr = solve((a, b)), solve((a, c)), solve((c, b))
        quarters = (solve((a, cl)) + solve((cl, c))) + (solve((c, cr)) + solve((cr, b)))
        d1 = abs(vp - (vl + vr))
        d2 = abs((vl + vr) - quarters)
        out(
            f"     {name} rtol 1e-2, the root's trust check: delta1 {d1:.3e}, delta2 {d2:.3e},"
            f" r {d2 / d1:.3g}"
        )
    out()
    out(
        "   test_levin_roundoff_exit test_exit_against_its_premise_fallback_branch, problem (b),"
        " with and without the bypass"
    )
    f3, theta3 = RX._test3b_problem()
    ref3, _ = RX._test3b_reference()
    kwargs = dict(atol=RX.TEST3B_ATOL, rtol=RX.TEST3B_RTOL)
    for name, M in (("06d", P), ("06e", lq)):
        for bypass in (True, False):
            data = M._adaptive_levin(
                RX.TEST3B_SPAN,
                f3,
                M._Basis_SinCos(theta3),
                _test_only_bypass_roundoff_premise=bypass,
                **kwargs,
            )
            out(
                f"     {name} {'bypass ' if bypass else 'premise'}: {summary(data)}, phase-limited"
                f" {data['num_phase_limited_regions']}, true {abs(data['value'] - ref3):.3e}"
            )
            for region in data["regions"]:
                out(
                    f"         [X0 + {region.start - RX.TEST3B_X0:.4f}, X0 +"
                    f" {region.end - RX.TEST3B_X0:.4f}] depth {region.depth} {region.type} pair"
                    f" {region.abserr:.3e} floor {region.phase_err:.3e} phase-limited"
                    f" {region.phase_limited}"
                )
    out()


def section_G():
    out(
        "== G. Cost of a fallback solve, 06d -> 06e: mean wall time per solve over the T3 and"
        " window cells (theta' supplied), and over the T2 cells (spectral theta'), 20 repeats"
    )
    for prefix in (("T3", "W"), ("T2",)):
        cells = [c for c in T.test4_cells() if c.label.startswith(prefix)]
        for name, M in (("06d", P), ("06e", lq)):
            prepared = []
            for cell in cells:
                basis = M._Basis_SinCos(cell.theta)
                grid, Dmat = M.chebyshev_matrices(cell.span, T.N)
                _, tp, _, _, ps = basis.build_Levin_data(grid, Dmat, need_AmatT=False)
                prepared.append((cell, basis, tp, ps))
            start = time.perf_counter()
            for _ in range(20):
                for cell, basis, tp, ps in prepared:
                    M._adaptive_levin_subregion_cc(
                        cell.span, cell.f, basis, T.N, ps, tp, None, None
                    )
            per = (time.perf_counter() - start) / (20 * len(prepared))
            out(f"   {'+'.join(prefix)} cells ({len(cells)}), {name}: {per * 1e6:.0f} us per solve")
    out()


def main():
    header = prov.header(__file__, "section 2, tests 1-6 at the parent and at 06e", [TEST_FILE])
    header.append(f"# parent module {prov.PARENT_SHA}, sha256 {P_SHA}")
    for line in header:
        out(line)
    out()
    for section in (section_A, section_B, section_C, section_D, section_E, section_F, section_G):
        section()
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
