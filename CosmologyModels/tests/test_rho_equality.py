"""
Characterisation of ``LambdaCDM_GenericEOS._find_rho_equality`` -- the solve that locates the
matter/radiation and matter/Lambda equality redshifts.

WHAT THE SOLVE IS. ``__init__`` calls ``_find_rho_equality`` twice, once for each pair, handing
it an analytic initial guess: ``omega_m/omega_r - 1`` for matter = radiation and
``(omega_cc/omega_m)^(1/3) - 1`` for matter = Lambda. Both results are printed with ``:.4g`` in the
construction banner and then discarded.

WHY ITS ACCURACY IS NOT ITS TOLERANCES. As shipped the solve is an *unbracketed* secant at
``xtol=1e-6, rtol=1e-4`` -- tolerances two orders looser than the other two root solves in the same
file. It nevertheless returns the root to 1-3 ulp, and it does so for a reason that has nothing to
do with the tolerances: the caller hands it the answer. ``g_*(T)`` is flat at
``z_eq ~ 3.4e3`` (``T ~ 8e-10`` GeV, far below every QCD threshold), so ``rho_r ~ (1+z)^4``
*exactly* there and ``1 + z_eq = omega_m/omega_r`` is the closed solution; the Lambda pair has no
temperature dependence at all, so its cube-root guess is exact by construction. The secant
evaluates one to three times, finds the residual already at the rounding floor, and stops.
``prompts/background-solver-robustness/AUDIT.md`` §2.3 is the measurement behind that paragraph.

WHY THIS MODULE EXISTS. The accuracy above is inherited from the caller, not enforced by the code,
and the flat-``g_*`` argument is a property of *this* equation of state at *this* redshift. The
campaign that owns this file replaces the secant with a bracketed Brent solve at
``xtol=1e-300, rtol=1e-14``. This module exists so that the replacement cannot move either root:
it scores the solve against a reference that is *independent of the solve* -- a bracketed
``brentq`` at Brent's own floor, seeded from the closed form rather than from the shipped answer --
and against the closed forms themselves.

THE MODULE IS WRITTEN TO PASS BOTH BEFORE AND AFTER THAT CHANGE, AND THAT IS THE POINT OF IT.
Everything asserted here is an invariant that must outlive the bracketing change: the two roots,
the monotonicity a bracket-expansion policy rests on, and the two closed forms. Nothing here
asserts today's *behaviour* -- not the exception types raised when the guess is displaced, not the
evaluation counts, not the fact that ``_find_rho_equality``'s own ``converged`` guard never fires.
Those are precisely what the bracketing change is for; they are recorded in
``prompts/background-solver-robustness/logs/01-equality-solve-characterisation.md`` and the
assertions that distinguish the two trees are added by the prompt that makes the change. A test
that pinned them here would have to be rewritten by that prompt, which would destroy the evidence
that anything changed at all.

No Ray and no datastore is needed.
"""

import unittest
from math import pow

import numpy as np
from scipy.optimize import brentq

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.tests.test_wPerturbations import PureRadiationEOS, lambdaCDM_gstar
from Units import Mpc_units

# Brent's own convergence floor is 4*eps = 4 * 2.220446e-16 = 8.881784e-16; scipy's brentq
# rejects a smaller rtol outright, so 8.9e-16 is the tightest relative tolerance that can be
# asked for. Asking for less is asking scipy for something it cannot deliver. The absolute
# component is disabled with xtol=1e-300 for the same reason the T(z) solve at
# LambdaCDM_GenericEOS.py:583 disables it: the two roots here are four decades apart
# (z ~ 3.4e3 and z ~ 0.30), so no single absolute tolerance in z can serve both.
BRENT_RTOL_FLOOR = 8.9e-16
BRENT_XTOL_DISABLED = 1.0e-300

# max_z for the two anchor models, as prompts/background-solver-robustness/measure_rho_equality.py
# builds them. The QCD value is the production one; 1e6 is enough for the stand-in to cover every
# probe below.
QCD_MAX_Z = 1.0e12
RADIATION_MAX_Z = 1.0e6

# Agreement thresholds, in ulp of the reference and not as relative tolerances: both quantities
# sit at the double-precision floor, and a relative tolerance is an invitation to loosen it later
# without noticing that the floor is where it already is.
#
# The solve against the bracketed reference. Measured at 3e820eb: 3 ulp (QCD, matter = radiation;
# equivalently the -4.00e-16 relative AUDIT.md §2.2 reports, since one ulp at z = 3406.67 is
# 1.335e-16 relative), 2 ulp (QCD, matter = Lambda), 1 ulp and 2 ulp on the pure-radiation
# stand-in. The bound is 4 rather than 3 because the reference carries one ulp of arbitrariness of
# its own: re-taking it on brackets of +-1 %, +-5 %, +-10 %, [0.5x, 2x] and [0.8x, 1.5x] moves it
# over one ulp. It is not looser than that, and must not be made so to accommodate a change in the
# solve -- the campaign's claim is that the roots do not move at all.
SOLVE_VS_REFERENCE_ULP = 4

# The Lambda closed form against the bracketed reference: measured exactly 2 ulp on both models.
# There is no headroom in this one deliberately. The bracket is fixed, so the reference is
# deterministic, and this is the oracle that does not depend on the equation of state at all
# (below); if it moves, that is something a later reader should be told about rather than
# something to absorb.
LAMBDA_CLOSED_FORM_ULP = 2

# The matter/radiation closed form against the bracketed reference: measured 7 ulp on
# QCD_Cosmology and 1 ulp on the pure-radiation stand-in. Wider than the Lambda case because
# omega_m/omega_r is a ratio of two separately-rounded quantities, so the *evaluation* of an
# expression that is exact in exact arithmetic still costs a few ulp.
MATTER_RADIATION_CLOSED_FORM_ULP = 8

# AUDIT.md §4.1's probe sets, reproduced from measure_rho_equality.py §4.1 so the test and the
# measurement script agree. The two ranges differ because the two roots do.
MATTER_RADIATION_PROBES = (33.0, 339.0, 1702.0, 3406.0, 6814.0, 34075.0, 340766.0)
MATTER_LAMBDA_PROBES = (0.0, 0.1, 0.303, 0.5, 1.0, 3.0, 10.0)


def species_of(pair: str):
    """The two ``_rho_fluid`` keys ``_find_rho_equality`` is called with."""
    return (
        ("matter", "radiation") if pair == "matter_radiation" else ("matter", "lambda")
    )


def closed_form_guess(model, pair: str) -> float:
    """
    The analytic initial guess ``LambdaCDM_GenericEOS.__init__`` supplies at :501-508, written
    the same way it is written there.
    """
    if pair == "matter_radiation":
        return model.omega_m / model.omega_r - 1.0
    return pow(model.omega_cc / model.omega_m, 1.0 / 3.0) - 1.0


def match_rho(model, pair: str):
    """``_find_rho_equality``'s own residual, rebuilt here so the reference does not go through it."""
    A, B = species_of(pair)

    def f(z: float) -> float:
        rho = model._rho_fluid(z)
        return rho[A] - rho[B]

    return f


def bracketed_reference(model, pair: str) -> float:
    """
    An independent Brent reference for one equality redshift.

    This is ``measure_rho_equality.bracketed_reference``, reproduced rather than imported: a test
    must not depend on a campaign directory under ``prompts/``, which is deleted when the campaign
    closes.

    The bracket is expanded from the *closed form*, never from the production solve's answer, so
    that the reference is independent of the thing it is used to score. Both ratios are strictly
    monotone across these ranges -- which is what
    ``test_the_density_ratios_are_monotone_where_their_roots_live`` below exists to keep true --
    so the bracket straddles the root and Brent applies.
    """
    z_root = closed_form_guess(model, pair)
    if z_root > 1.0:
        lo, hi = 0.95 * z_root, 1.05 * z_root
    else:
        lo, hi = max(0.5 * z_root, -0.9), 1.5 * z_root
    return float(
        brentq(
            match_rho(model, pair),
            lo,
            hi,
            xtol=BRENT_XTOL_DISABLED,
            rtol=BRENT_RTOL_FLOOR,
        )
    )


def ulps_between(got: float, reference: float) -> float:
    """Signed separation of ``got`` from ``reference``, in ulp of the reference."""
    return (float(got) - float(reference)) / np.spacing(abs(float(reference)))


class TestRhoEquality(unittest.TestCase):
    """
    Constructing a LambdaCDM_GenericEOS is expensive -- the T(z) representation is a segmented
    entropy factor over DEFAULT_T_Z_SPLINE_SAMPLES nodes, each a bracketed root solve, plus the
    two equality solves under test -- so both models are built once for the whole class.
    """

    @classmethod
    def setUpClass(cls):
        cls.units = Mpc_units()
        cls.params = Planck2018()

        # QCD_Cosmology: the flat-g_* argument holds here, but by accident of where the equality
        # redshift falls relative to the QCD thresholds, not by construction.
        cls.qcd = QCD_Cosmology(
            store_id=10, units=cls.units, params=cls.params, max_z=QCD_MAX_Z
        )

        # The pure-radiation stand-in: g_* = g_{S,*} is constant, so T(z) = T_CMB (1+z) exactly
        # and *both* closed forms are exact. Any departure here is the representation's.
        cls.radiation = LambdaCDM_GenericEOS(
            store_id=11,
            eos=PureRadiationEOS(cls.units, lambdaCDM_gstar(cls.params.Neff)),
            units=cls.units,
            params=cls.params,
            max_z=RADIATION_MAX_Z,
        )

        cls.models = (
            ("QCD_Cosmology", cls.qcd),
            ("pure-radiation stand-in", cls.radiation),
        )

    def test_equality_redshifts_match_a_bracketed_reference(self):
        """
        The campaign's primary stop condition: both equality redshifts, on both models, reproduce
        an independent bracketed reference.

        The reference is a bracketed brentq at Brent's own floor, seeded from the closed form and
        never from the shipped answer. That independence is the whole design: the shipped solve is
        accurate because its caller hands it the root, so a test that merely pinned today's output
        would report success no matter what a later change did to the solve, the correct answer
        and the accidentally-correct answer being the same number.
        """
        print(
            "\n  equality redshifts against a bracketed brentq reference (17 digits):"
        )
        for label, model in self.models:
            for pair in ("matter_radiation", "matter_lambda"):
                with self.subTest(model=label, pair=pair):
                    A, B = species_of(pair)
                    got = float(
                        model._find_rho_equality(
                            A, B, init_z=closed_form_guess(model, pair)
                        )
                    )
                    reference = bracketed_reference(model, pair)
                    print(
                        f"    {label:>24} {pair:>17}: solve = {got!r:<22} "
                        f"reference = {reference!r:<22} "
                        f"({ulps_between(got, reference):+.1f} ulp)"
                    )
                    self.assertLessEqual(
                        abs(got - reference),
                        SOLVE_VS_REFERENCE_ULP * np.spacing(abs(reference)),
                        msg=f"{label}, {pair}: the equality redshift has moved away from the "
                        f"bracketed reference. solve = {got!r}, reference = {reference!r}, "
                        f"separation = {ulps_between(got, reference):+.3f} ulp, "
                        f"budget = {SOLVE_VS_REFERENCE_ULP} ulp",
                    )

    def test_the_density_ratios_are_monotone_where_their_roots_live(self):
        """
        Each density ratio is strictly monotone across the range its own root lives in.

        This test exists because it is the *precondition for bracketing*, not because the numbers
        are interesting. A bracket expanded geometrically from the analytic guess is guaranteed to
        straddle the root only if the residual changes sign exactly once over the expansion; an
        equation of state that broke monotonicity would make a bracket-expansion policy unsound
        rather than merely inaccurate, and would do it silently, because Brent reports success on
        any sign-changing bracket it is handed.

        rho_matter/rho_radiation ~ 1/((1+z) G(T)) is decreasing -- G rises with T and hence with
        z, so the two effects add rather than competing. rho_matter/rho_lambda ~ (1+z)^3 is
        increasing, with no temperature dependence at all. The two probe ranges differ because the
        two roots do: z ~ 3.4e3 against z ~ 0.30. AUDIT.md §4.1.
        """
        cases = (
            ("matter_radiation", MATTER_RADIATION_PROBES, "decreasing"),
            ("matter_lambda", MATTER_LAMBDA_PROBES, "increasing"),
        )
        for label, model in self.models:
            for pair, probes, direction in cases:
                with self.subTest(model=label, pair=pair):
                    A, B = species_of(pair)
                    ratios = [
                        model._rho_fluid(z)[A] / model._rho_fluid(z)[B] for z in probes
                    ]
                    for (z_lo, r_lo), (z_hi, r_hi) in zip(
                        zip(probes, ratios), zip(probes[1:], ratios[1:])
                    ):
                        if direction == "decreasing":
                            self.assertLess(
                                r_hi,
                                r_lo,
                                msg=f"{label}: rho_{A}/rho_{B} is not strictly decreasing "
                                f"between z={z_lo:g} ({r_lo:.12e}) and z={z_hi:g} "
                                f"({r_hi:.12e}); a bracket expanded from the analytic guess is "
                                "no longer guaranteed to straddle the root",
                            )
                        else:
                            self.assertGreater(
                                r_hi,
                                r_lo,
                                msg=f"{label}: rho_{A}/rho_{B} is not strictly increasing "
                                f"between z={z_lo:g} ({r_lo:.12e}) and z={z_hi:g} "
                                f"({r_hi:.12e}); a bracket expanded from the analytic guess is "
                                "no longer guaranteed to straddle the root",
                            )

    def test_the_lambda_equality_is_closed_form_on_any_equation_of_state(self):
        """
        (omega_cc/omega_m)^(1/3) - 1 is the matter/Lambda equality redshift by construction, on
        every equation of state.

        rho_matter/rho_lambda ~ (1+z)^3 with *no temperature dependence at all*: rho_m is
        rho_m0 (1+z)^3 and rho_lambda is a constant, so neither side sees T(z), g_*(T) or the
        representation that computes them. This is therefore an oracle rather than an
        approximation, and it is what tells a later reader that the Lambda root is safe on any
        equation of state while the matter/radiation root is not -- that one is exact only where
        g_* happens to be flat (see the test below).
        """
        for label, model in self.models:
            with self.subTest(model=label):
                closed_form = closed_form_guess(model, "matter_lambda")
                reference = bracketed_reference(model, "matter_lambda")
                self.assertLessEqual(
                    abs(closed_form - reference),
                    LAMBDA_CLOSED_FORM_ULP * np.spacing(abs(reference)),
                    msg=f"{label}: (omega_cc/omega_m)^(1/3) - 1 = {closed_form!r} is no longer "
                    f"the matter/Lambda equality redshift; bracketed reference = {reference!r}, "
                    f"separation = {ulps_between(closed_form, reference):+.3f} ulp, "
                    f"budget = {LAMBDA_CLOSED_FORM_ULP} ulp",
                )

    def test_the_matter_radiation_closed_form_is_exact_only_because_g_star_is_flat(
        self,
    ):
        """
        The finding this module records: on QCD_Cosmology the matter/radiation guess is the root
        *because* g_*(T) is flat at the equality redshift, and nothing in the production code
        checks that.

        z_eq ~ 3.4e3 puts the photon temperature at T ~ 8e-10 GeV, far below every QCD threshold,
        so G(T) is constant over a decade either side (AUDIT.md §2.3 measures 3.38 at all three
        probes). With g_* constant rho_r ~ (1+z)^4 exactly, so rho_m = rho_r has the closed
        solution 1 + z = omega_m/omega_r -- which is exactly what the caller passes in as the
        initial guess, and why the shipped secant converges in one to three evaluations.

        Unlike the Lambda case above this is a property of this equation of state at this
        redshift, not a theorem. An equation of state with a threshold near z_eq would leave the
        guess merely close, and the solve -- which is what is actually responsible for the answer
        -- would have to earn the accuracy the guess currently donates. This test is the standing
        statement of that assumption.
        """
        z_eq = closed_form_guess(self.qcd, "matter_radiation")
        g_star = [
            self.qcd._eos.G(self.qcd._rho_fluid((1.0 + z_eq) * factor - 1.0)["T"])
            for factor in (0.1, 1.0, 10.0)
        ]
        self.assertEqual(
            g_star[0],
            g_star[1],
            msg=f"g_*(T) is no longer flat a decade below z_eq: {g_star[0]!r} at z_eq/10 "
            f"against {g_star[1]!r} at z_eq = {z_eq:.6g}. The matter/radiation closed form is "
            "exact only where it is, so the initial guess is no longer the root and the solve "
            "now has to find it.",
        )
        self.assertEqual(
            g_star[2],
            g_star[1],
            msg=f"g_*(T) is no longer flat a decade above z_eq: {g_star[2]!r} at 10 z_eq "
            f"against {g_star[1]!r} at z_eq = {z_eq:.6g}. The matter/radiation closed form is "
            "exact only where it is, so the initial guess is no longer the root and the solve "
            "now has to find it.",
        )

        # ... and, because it is, the closed form is the root to rounding on both models.
        for label, model in self.models:
            with self.subTest(model=label):
                closed_form = closed_form_guess(model, "matter_radiation")
                reference = bracketed_reference(model, "matter_radiation")
                self.assertLessEqual(
                    abs(closed_form - reference),
                    MATTER_RADIATION_CLOSED_FORM_ULP * np.spacing(abs(reference)),
                    msg=f"{label}: omega_m/omega_r - 1 = {closed_form!r} is no longer the "
                    f"matter/radiation equality redshift to rounding; bracketed reference = "
                    f"{reference!r}, separation = "
                    f"{ulps_between(closed_form, reference):+.3f} ulp, "
                    f"budget = {MATTER_RADIATION_CLOSED_FORM_ULP} ulp",
                )


if __name__ == "__main__":
    unittest.main()
