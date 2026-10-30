"""levin-phase-floor prompt 06e: does 06b's round-off premise still bind on the Clenshaw-Curtis
fallback under 06e's node term?

test_levin_roundoff_exit's test 3 (b) (06d section 5 point 3) asserts that, on a fallback problem
where the floor binds, the premise check at least halves the declared error against the bypass. Under
06e that problem's call is bit-identical across the bypass. This script searches the same family
for a problem on which it is not:

  theta = A e^(x - X0), f = (e^(-gamma (x - X0)), 0) on [X0, X0 + L], atol 1e-30, rtol 1e-16,

over X0, A, gamma and L, at the parent (06d's code) and at 06e's, each with and without the bypass.
Per problem it records the declared error with and without the bypass, and whether the two calls
are bit-identical apart from the evaluation count. It also records, over every fallback solve each 06e call made, the premise ratio
(phase_err_L + phase_err_R) / phase_err of every solved region whose halves were solved.

Run from the repository root: PYTHONPATH=. ./venv/bin/python <this script>. Writes
../out/exit_search.txt. No Ray, no datastore.
"""

import itertools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
from AdaptiveLevin.tests.test_levin_roundoff_exit import _call_signature  # noqa: E402

OUT = prov.OUT / "exit_search.txt"
P, P_SHA = prov.load_parent()

X0S = (1.0e3, 1.0e4, 1.0e6, 1.0e8)
AS = (0.025, 0.04, 0.3)
GAMMAS = (1.0, 2.0, 3.0)
LS = (4.0, 6.0)
KWARGS = dict(atol=1e-30, rtol=1e-16)


def problem(X0, A, gamma):
    f = [lambda x: math.exp(-gamma * (x - X0)), lambda x: 0.0]
    theta = {
        "theta": lambda x: A * math.exp(x - X0),
        "theta_deriv": lambda x: A * math.exp(x - X0),
    }
    return f, theta


def main():
    lines = prov.header(__file__, "06b's premise on the fallback branch under 06e")
    lines.append(f"# parent module {prov.PARENT_SHA}, sha256 {P_SHA}")
    lines.append(
        "# X0 A gamma L | 06d: abserr bypass, abserr premise, ratio, identical | 06e: the same"
    )
    ratios = []
    halving = {"06d": 0, "06e": 0}
    differ = {"06d": 0, "06e": 0}
    count = 0
    for X0, A, gamma, L in itertools.product(X0S, AS, GAMMAS, LS):
        f, theta = problem(X0, A, gamma)
        span = (X0, X0 + L)
        row = [f"{X0:.0e} {A:g} {gamma:g} {L:g}"]
        for name, M in (("06d", P), ("06e", lq)):
            solves = {}
            original = M._adaptive_levin_subregion

            def spy(x_span, *args, **kwargs):
                data = original(x_span, *args, **kwargs)
                solves[(float(x_span[0]), float(x_span[1]))] = data
                return data

            if name == "06e":
                M._adaptive_levin_subregion = spy
            try:
                bypass = M._adaptive_levin(
                    span,
                    f,
                    M._Basis_SinCos(theta),
                    _test_only_bypass_roundoff_premise=True,
                    **KWARGS,
                )
                premise = M._adaptive_levin(span, f, M._Basis_SinCos(theta), **KWARGS)
            finally:
                M._adaptive_levin_subregion = original
            # element 2 of the signature is the evaluation count, which the premise check
            # changes whenever it solves a fallback leaf's halves (test 4 of that module)
            sb, sp = _call_signature(bypass), _call_signature(premise)
            same = sb[:2] + sb[3:] == sp[:2] + sp[3:]
            ratio = premise["abserr"] / bypass["abserr"]
            halving[name] += ratio <= 0.5
            differ[name] += not same
            row.append(
                f"{name}: {bypass['abserr']:.3e} {premise['abserr']:.3e} {ratio:.3g}"
                f" {'identical' if same else 'DIFFER'}"
            )
            if name == "06e":
                for (a, b), data in solves.items():
                    if not data.get("is_direct"):
                        continue
                    c = (b + a) / 2.0
                    if (a, c) in solves and (c, b) in solves and data["phase_err"] > 0:
                        ratios.append(
                            (
                                solves[(a, c)]["phase_err"] + solves[(c, b)]["phase_err"]
                            )
                            / data["phase_err"]
                        )
        count += 1
        lines.append("   " + " | ".join(row))
    lines.append("")
    lines.append(
        f"problems {count}; calls that differ across the bypass: 06d {differ['06d']}, 06e"
        f" {differ['06e']}; premise abserr at most half the bypass's: 06d {halving['06d']}, 06e"
        f" {halving['06e']}"
    )
    ratios.sort()
    if ratios:
        below = sum(1 for r in ratios if r < lq._LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO)
        lines.append(
            f"06e fallback regions with both halves solved: {len(ratios)}; premise ratio"
            f" (halves' floors / own) min {ratios[0]:.4g}, median {ratios[len(ratios) // 2]:.4g},"
            f" max {ratios[-1]:.4g}; below 1/2: {below}"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
