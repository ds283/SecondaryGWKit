"""
Measurements behind AUDIT.md -- the ``_find_rho_equality`` root solve at
``CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:1008``.

Run from the repository root (the ``PYTHONPATH=.`` is required; without it the imports below
fail -- see RECONCILIATION.md R0):

    PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py

Needs neither Ray nor a datastore. Every figure quoted in AUDIT.md §2 and §3 comes from
this script; re-run it rather than trusting the numbers in the document.
"""

import sys
from math import pow, sqrt

from scipy.optimize import root_scalar, brentq

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.tests.test_wPerturbations import PureRadiationEOS, lambdaCDM_gstar
from Units import Mpc_units

# The two equality pairs production actually asks for, with the analytic initial guess
# LambdaCDM_GenericEOS.__init__ supplies at :496 and :499.
PAIRS = ("matter_radiation", "matter_lambda")


def initial_guess(model, pair: str) -> float:
    if pair == "matter_radiation":
        return model.omega_m / model.omega_r - 1.0
    return pow(model.omega_cc / model.omega_m, 1.0 / 3.0) - 1.0


def species(pair: str):
    return (
        ("matter", "radiation") if pair == "matter_radiation" else ("matter", "lambda")
    )


def counted_match_rho(model, pair: str):
    """``_find_rho_equality``'s own ``match_rho``, with an evaluation counter."""
    A, B = species(pair)
    calls = [0]

    def f(z: float) -> float:
        calls[0] += 1
        rho = model._rho_fluid(z)
        return rho[A] - rho[B]

    return f, calls


def bracketed_reference(f, z_root: float) -> float:
    """An independent Brent reference, at Brent's own floor of ~4*eps."""
    if z_root > 1.0:
        lo, hi = 0.95 * z_root, 1.05 * z_root
    else:
        lo, hi = max(0.5 * z_root, -0.9), 1.5 * z_root
    return brentq(f, lo, hi, xtol=1e-300, rtol=8.9e-16)


def section_2_achieved(model, label: str) -> None:
    """AUDIT.md §2.2 -- what the shipped tolerances achieve at the production call sites."""
    print(f"\n### achieved accuracy at the production call sites -- {label}")
    print(
        f"{'pair':>18} {'shipped root':>22} {'evals':>6} {'reference':>22} {'rel err':>10}"
    )
    for pair in PAIRS:
        f, calls = counted_match_rho(model, pair)
        z0 = initial_guess(model, pair)
        calls[0] = 0
        shipped = root_scalar(f, x0=z0, xtol=1e-6, rtol=1e-4)
        n = calls[0]
        ref = bracketed_reference(f, shipped.root)
        print(
            f"{pair:>18} {shipped.root:>22.14g} {n:>6d} {ref:>22.14g} "
            f"{(shipped.root - ref) / ref:>10.2e}"
        )


def section_2_why_exact(model, label: str) -> None:
    """
    AUDIT.md §2.3 -- the guess is the root because g_*(T) is flat at z_eq. Print g_* at the
    equality redshift and a decade either side; if these three agree, rho_r ~ (1+z)^4 exactly
    there and ``omega_m/omega_r - 1`` is the root to rounding.
    """
    print(f"\n### why the initial guess is already the root -- {label}")
    z_eq = initial_guess(model, "matter_radiation")
    for factor in (0.1, 1.0, 10.0):
        z = (1.0 + z_eq) * factor - 1.0
        T = model._rho_fluid(z)["T"]
        print(
            f"  z = {z:>16.6g}   T = {T / model._units.GeV:>12.6g} GeV   "
            f"G(T) = {model._eos.G(T):.12g}"
        )


def section_3_displaced(model, label: str) -> None:
    """AUDIT.md §3.1 -- what the tolerance is worth when the guess is not already the root."""
    print(f"\n### displaced-guess behaviour, matter = radiation -- {label}")
    f, calls = counted_match_rho(model, "matter_radiation")
    z0 = initial_guess(model, "matter_radiation")
    ref = bracketed_reference(f, z0)
    print(f"  reference root z = {ref:.14g}")
    print(
        f"{'offset':>10} {'shipped rel err':>17} {'evals':>6} | "
        f"{'tightened rel err':>19} {'evals':>6}"
    )
    for frac in (0.0, 0.01, 0.05, 0.2, 0.5, 2.0, -0.3):
        guess = z0 * (1.0 + frac)
        row = []
        for kwargs in ({"xtol": 1e-6, "rtol": 1e-4}, {"xtol": 1e-300, "rtol": 1e-14}):
            calls[0] = 0
            try:
                r = root_scalar(f, x0=guess, **kwargs)
                row.append((f"{(r.root - ref) / ref:.2e}", calls[0]))
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- the failure mode is the measurement
                row.append((type(exc).__name__, calls[0]))
        print(
            f"{frac:>+10.2f} {row[0][0]:>17} {row[0][1]:>6d} | "
            f"{row[1][0]:>19} {row[1][1]:>6d}"
        )


def section_3_failure_boundary(model, label: str) -> None:
    """
    AUDIT.md §3.2 -- the secant is unbracketed, so a displaced guess can leave the tabulated
    range. Locate the displacement at which it stops returning and report *how* it fails:
    ``ValueError`` escapes ``_find_rho_equality``'s ``if not root.converged`` guard entirely.
    """
    print(f"\n### failure boundary, matter = radiation -- {label}")
    f, _ = counted_match_rho(model, "matter_radiation")
    z0 = initial_guess(model, "matter_radiation")
    for frac in (-0.05, -0.1, -0.2, -0.3, -0.5, -0.8):
        guess = z0 * (1.0 + frac)
        try:
            r = root_scalar(f, x0=guess, xtol=1e-6, rtol=1e-4)
            outcome = f"converged={r.converged}, root={r.root:.10g}"
        except Exception as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}: {str(exc)[:60]}"
        print(f"  offset {frac:>+6.2f} (z0 = {guess:>14.6g}):  {outcome}")


def section_4_monotone(model, label: str) -> None:
    """
    AUDIT.md §4.1 -- the bracketing argument. Each ratio is strictly monotone in z across the
    range its own root lives in, so a bracket expanded geometrically from the analytic guess is
    guaranteed to straddle the root and ``brentq`` applies.

    ``rho_matter/rho_radiation ~ 1/((1+z) G(T))`` is **decreasing** (G rises with T, hence with
    z, so the two effects add); ``rho_matter/rho_lambda ~ (1+z)^3`` is **increasing**. The probe
    ranges differ because the two roots do: z ~ 3.4e3 and z ~ 0.3.
    """
    print(f"\n### monotonicity of the two density ratios -- {label}")
    probes = {
        "matter_radiation": (
            "decreasing",
            [33.0, 339.0, 1702.0, 3406.0, 6814.0, 34075.0, 340766.0],
        ),
        "matter_lambda": ("increasing", [0.0, 0.1, 0.303, 0.5, 1.0, 3.0, 10.0]),
    }
    for pair in PAIRS:
        A, B = species(pair)
        direction, z_values = probes[pair]
        print(
            f"  {A} / {B}   (guess z = {initial_guess(model, pair):.6g}, expect {direction})"
        )
        ratios = []
        for z in z_values:
            rho = model._rho_fluid(z)
            ratio = rho[A] / rho[B]
            ratios.append(ratio)
            print(f"    z = {z:>16.6g}   rho_{A}/rho_{B} = {ratio:>14.6e}")
        pairs_ok = zip(ratios, ratios[1:])
        ok = (
            all(b < a for a, b in pairs_ok)
            if direction == "decreasing"
            else all(b > a for a, b in zip(ratios, ratios[1:]))
        )
        print(f"    strictly {direction} across the probed range: {ok}")


def main() -> int:
    units = Mpc_units()
    params = Planck2018()

    qcd = QCD_Cosmology(store_id=10, units=units, params=params, max_z=1.0e12)
    radiation = LambdaCDM_GenericEOS(
        store_id=11,
        eos=PureRadiationEOS(units, lambdaCDM_gstar(params.Neff)),
        units=units,
        params=params,
        max_z=1.0e6,
    )

    for model, label in (
        (qcd, "QCD_Cosmology"),
        (radiation, "pure-radiation stand-in"),
    ):
        section_2_achieved(model, label)
        section_2_why_exact(model, label)

    section_3_displaced(qcd, "QCD_Cosmology")
    section_3_failure_boundary(qcd, "QCD_Cosmology")
    section_4_monotone(qcd, "QCD_Cosmology")
    return 0


if __name__ == "__main__":
    sys.exit(main())
