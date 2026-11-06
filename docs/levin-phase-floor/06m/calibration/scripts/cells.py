"""levin-phase-floor prompt 06m, section 2 tests 2-5: the numbers the tests' docstrings quote. New in 06m.

For each of test 2's cells (AdaptiveLevin/tests/test_levin_trust_yardstick.test2_cells()), under the commit, 06l's
variant and the parent: regions, fallback regions, maximum depth, converged, value, abserr; whether the commit's
result equals the variant's and the parent's bit for bit (the test's own comparison); and the variant's firings,
each with its parent's kind, its halves' kinds, r, delta1, phi_P, nu_1 and nu_B. For tests 3-5, the leaf: its
solves afresh, E_P, E_L, E_R, the halves' phi and phi^-, nu_1 and nu_B, and its trust under both codes.

Run from the repository root: PYTHONPATH=. ./venv/bin/python -B <this script>. Writes ../out/cells.txt. No Ray, no
datastore.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
from AdaptiveLevin.tests import test_levin_trust_yardstick as T  # noqa: E402

OUT = HERE.parent / "out" / "cells.txt"


def main():
    test_file = Path(T.__file__)
    lines = prov.header(
        __file__,
        "section 2 tests 2-5, the cells and leaves",
        [test_file, T.VARIANT_PATH],
    )
    parent, variant = T.load_parent(), T.load_variant()
    lines.append(f"# parent: git show {T.DISPATCH_SHA}:AdaptiveLevin/levin_quadrature.py")
    lines.append(f"# variant: {T.VARIANT_PATH} (sha256 checked {T.VARIANT_SHA256})")
    lines.append("")
    lines.append("== Test 2: each cell under the commit (new), the variant (var) and the parent (old)")
    for cell in T.test2_cells():
        name = cell[0]
        new, var, old = T.sincos(lq, cell), T.sincos(variant, cell), T.sincos(parent, cell)
        eq_var = T.result_record(new) == T.result_record(var, exclude=("elapsed", "route_06l"))
        eq_old = T.result_record(new) == T.result_record(old)
        fired = [r for r in var["route_06l"]["records"] if r["kind"] == "fired"]
        n_cc = sum(1 for r in fired if r["is_direct"])
        lines.append(
            f"{name}: new regions {new['num_regions']} (fallback {new['num_simple_regions']}) depth"
            f" {new['max_depth']} converged {new['converged']} value {new['value']!r} abserr {new['abserr']!r};"
            f" old regions {old['num_regions']} (fallback {old['num_simple_regions']}) depth {old['max_depth']}"
            f" converged {old['converged']} value {old['value']!r} abserr {old['abserr']!r}"
        )
        lines.append(
            f"   new == var bit for bit: {eq_var}; new == old bit for bit: {eq_old}; variant firings"
            f" {len(fired)} (Clenshaw-Curtis parent {n_cc}, Levin parent {len(fired) - n_cc}); withdrawn"
            f" {var['route_06l']['n_withdrawn']}"
        )
        for r in fired:
            lines.append(
                f"      fired [{r['start']!r}, {r['end']!r}] depth {r['depth']} parent"
                f" {'CC' if r['is_direct'] else 'Levin'} halves"
                f" {'CC' if r['is_direct_L'] else 'Levin'}+{'CC' if r['is_direct_R'] else 'Levin'}"
                f" r {r['r']:.4f} delta1 {r['delta1']:.4e} phi_P {r['phi_P']:.4e} nu1 {r['nu1']:.4e}"
                f" nuB {r['nuB']:.4e} delta1/phi_P {r['delta1'] / r['phi_P']:.3g}"
            )

    lines.append("")
    lines.append("== Tests 3-5: the leaves")
    for test, cell, key in (
        ("test 3", T.far_cell(), (300003.640625, 300003.65625, 8)),
        ("test 4", T.noisy_512_cell(), (22.89453125, 23.125, 8)),
        ("test 5", T.far_cell(), (300002.0, 300002.5, 3)),
    ):
        new, made_new = T.run_recorded(lq, cell)
        old, made_old = T.run_recorded(parent, cell)
        ln, lo = T.checked_leaf(made_new, key), T.checked_leaf(made_old, key)
        P, L, R = T.leaf_solves(cell, key)
        nu_B = lq._trust_noise_yardstick(ln.phase_err, bool(P.get("is_direct", False)), L, R)
        nu_1 = T.nu1(ln.phase_err, L, R)
        lines.append(f"{test}: {cell[0]} leaf [{key[0]!r}, {key[1]!r}] depth {key[2]}")
        lines.append(
            f"   kinds: parent {'CC' if P.get('is_direct') else 'Levin'}, halves"
            f" {'CC' if L.get('is_direct') else 'Levin'}+{'CC' if R.get('is_direct') else 'Levin'}"
        )
        lines.append(
            f"   r {ln.convergence_ratio!r} delta1 {ln.delta1!r} delta2 {ln.delta2!r} phi_P {ln.phase_err!r}"
            f" (parent's: r {lo.convergence_ratio!r} delta1 {lo.delta1!r} phi_P {lo.phase_err!r})"
        )
        lines.append(
            f"   E_P {P.get('node_error_signed')!r} E_L {L.get('node_error_signed')!r} E_R"
            f" {R.get('node_error_signed')!r}; phi_L {L['phase_err']!r} phi_R {R['phase_err']!r}; phi^-_L"
            f" {L.get('phase_err_less_node')!r} phi^-_R {R.get('phase_err_less_node')!r}"
        )
        if L.get("is_direct") and R.get("is_direct"):
            EL, ER = L["node_error_signed"], R["node_error_signed"]
            lines.append(
                f"   |E_L + E_R| {abs(EL + ER):.4e}; |E_L| + |E_R| {abs(EL) + abs(ER):.4e}; ratio"
                f" {abs(EL + ER) / (abs(EL) + abs(ER)):.4g}"
            )
        lines.append(
            f"   nu_1 {nu_1!r} nu_B {nu_B!r}; delta1/phi_P {ln.delta1 / ln.phase_err:.4g} delta1/nu_B"
            f" {ln.delta1 / nu_B:.4g} delta1/nu_1 {ln.delta1 / nu_1:.4g}"
        )
        lines.append(
            f"   trusted: commit {ln.trusted} (returned {T.returned(new, key)}, value_kind {ln.value_kind});"
            f" parent {lo.trusted} (returned {T.returned(old, key)}); the call's regions: commit"
            f" {new['num_regions']} converged {new['converged']}, parent {old['num_regions']} converged"
            f" {old['converged']}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT} sha256 {prov.sha256(OUT)}")


if __name__ == "__main__":
    main()
