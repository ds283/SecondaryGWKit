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

import builtins
import math
import unittest
from math import pow

import numpy as np
from scipy.optimize import brentq

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
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

# Fractions of the analytic guess at which AUDIT.md §3.2 measured the shipped secant failing or
# silently returning a wrong answer: -20 % returned converged=True with 2.7e-9 relative error,
# -30 % raised ValueError, and -50 % and -80 % walked to negative z and raised a
# TemperatureRepresentation bounds error from inside _rho_fluid. All four are recorded verbatim in
# logs/01-equality-solve-characterisation.md §4.1.
DISPLACED_GUESS_FRACTIONS = (0.8, 0.7, 0.5, 0.2)

# max_z for a model whose tabulated range cannot contain its own matter/radiation equality
# redshift, which is at z ~ 3.4e3 on these parameters. This is the only way to reach
# _find_rho_equality's bracketing failure: with the bracket clamped to the tabulated range and
# both density ratios monotone across it, no displacement of the *guess* can fail any more, which
# is the point of the change. 100 puts the representation's ceiling at 1.05 * 101 - 1 = 105.05.
TRUNCATED_MAX_Z = 100.0


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


# ----------------------------------------------------------------------------------------------
# The sites at which 1 + z_eq = omega_m/omega_r and 1 + z_Lambda = (omega_cc/omega_m)^(1/3) are
# still written out, each transcribed exactly as it is written in its own file -- including which
# ``pow`` is in scope there, which is the only way the two could differ at all:
#
#   LambdaCDM_GenericEOS.py, __init__    builtin ``pow``  (that module imports exp, sqrt, log,
#     -- the two *initial guesses* of                      log1p and expm1 from math -- not pow)
#        the bracketed solve, and nothing else
#   LambdaCDM.py, z_matter_*_equality    ``math.pow``     (``from math import sqrt, pow``)
#     -- that model's *answer*: it has no
#        equation of state, so the closed
#        form is exact rather than close
#
# **There is no longer a site in main.py.** ``cosmology_feature_redshifts`` used to compute the
# two redshifts itself, which made it a production sample location inside a BackgroundModel
# lookup key reached by a duplicated closed form; since prompt 09 of
# prompts/background-solver-robustness it reads BaseCosmology.z_matter_radiation_equality and
# BaseCosmology.z_matter_lambda_equality and computes nothing, so the model is authoritative and
# ``main.py`` holds no copy to drift. What reaches the grid on QCD_Cosmology is the bracketed
# solve, 7 ulp above the closed form; that substitution is what took the production source-grid
# digest from a2c32f67 to 4849552b on one sample of 1,996, and it is asserted in
# ComputeTargets/tests/test_source_grid.py, which can drive main.py's own function through
# load_main_py_functions (CLAUDE.md: main.py cannot be imported).
#
# ``test_the_remaining_closed_form_sites_agree`` below is what notices if the two that remain
# stop agreeing, and -- since LambdaCDM's is now an *answer* and not a diagnostic -- that its
# property still returns exactly what its expression says.
# ----------------------------------------------------------------------------------------------


def generic_eos_closed_form(model, pair: str) -> float:
    """``LambdaCDM_GenericEOS.__init__``'s initial guess, as written there."""
    if pair == "matter_radiation":
        return model.omega_m / model.omega_r - 1.0
    return builtins.pow(model.omega_cc / model.omega_m, 1.0 / 3.0) - 1.0


def lambdaCDM_closed_form(model, pair: str) -> float:
    """``LambdaCDM.z_matter_radiation_equality`` / ``z_matter_lambda_equality``, as written."""
    if pair == "matter_radiation":
        return model.omega_m / model.omega_r - 1.0
    return math.pow(model.omega_cc / model.omega_m, 1.0 / 3.0) - 1.0


CLOSED_FORM_SITES = (
    ("LambdaCDM_GenericEOS.py guess", generic_eos_closed_form),
    ("LambdaCDM.py property", lambdaCDM_closed_form),
)


def model_property(model, pair: str) -> float:
    """The model's own answer, through the ``BaseCosmology`` surface prompt 09 declares."""
    if pair == "matter_radiation":
        return float(model.z_matter_radiation_equality)
    return float(model.z_matter_lambda_equality)


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


def solve_recording_evaluations(model, pair: str, init_z: float):
    """
    Call ``_find_rho_equality`` and return ``(root, [every z at which _rho_fluid was evaluated])``.

    ``match_rho`` looks ``self._rho_fluid`` up at call time, so shadowing it on the instance
    records every evaluation the solve makes -- the guess, both endpoints of every expansion step,
    and every Brent iterate. That is what makes "the bracket does not leave the tabulated range" a
    direct assertion about the z values reached rather than a parse of the failure message, which
    is prose and will be reworded.
    """
    A, B = species_of(pair)
    evaluations = []
    underlying = model._rho_fluid

    def recording(z: float):
        evaluations.append(float(z))
        return underlying(z)

    model._rho_fluid = recording
    try:
        root = float(model._find_rho_equality(A, B, init_z=init_z))
    finally:
        # remove the instance attribute, exposing the class's own method again
        del model._rho_fluid

    return root, evaluations


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

        # LambdaCDM carries the third closed-form site (LambdaCDM.py:73-74) and is the model that
        # site actually runs on. It has no equation of state, no _rho_fluid and no
        # _find_rho_equality, so it can be scored for site agreement but not against a bracketed
        # reference; test_the_three_closed_form_sites_agree says so where it uses it.
        cls.lambdaCDM = LambdaCDM(store_id=13, units=cls.units, params=cls.params)

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

    def test_the_remaining_closed_form_sites_agree(self):
        """
        The sites that still write the closed forms out produce the *same float*, and each is the
        root to within the closed form's measured budget.

        WHY THIS TEST EXISTS, AND WHY ITS SUBJECT CHANGED. It shipped with prompt 03 as
        ``test_the_three_closed_form_sites_agree``, to make it safe to leave three copies of
        ``1 + z_eq = Omega_m/Omega_r`` standing, because the third -- ``main.py``'s -- was
        load-bearing: it became ``feature_z``, which ``CosmologyConcepts/wavenumber.py`` forces
        into the production source grid, whose content digest is a ``BackgroundModel`` lookup-key
        column. The user decided against that option
        (``[00-equality-redshift-closed-form-is-duplicated-three-times]``, README section 7 D2),
        and prompt 09 removed the ``main.py`` copy: the model now answers for itself and the
        consumer asks. Two sites remain, and **neither is load-bearing in the same way** -- one is
        a solver's initial guess, the other is ``LambdaCDM``'s own answer.

        IT IS NOT THE SAME GUARD, AND IT IS NOT A WEAKER ONE. ``LambdaCDM``'s expression was a
        banner diagnostic when this test was written; it is now the value that model returns from
        ``z_matter_radiation_equality``, so "the closed form equals the root" has stopped being a
        consistency claim about three transcriptions and become a *correctness* claim about one
        model's answer. That is what is asserted here, together with the check that the property
        really does return the expression below it.

        The only way the two remaining sites *can* disagree is the ``pow`` each file has in scope
        -- ``math.pow`` in ``LambdaCDM.py``, the builtin in ``LambdaCDM_GenericEOS.py`` -- and
        they agree bit for bit on these arguments today. (``closed_form_guess`` above reaches
        ``math.pow`` through this module's own ``from math import pow``, which is why it is a
        faithful stand-in for ``__init__``'s builtin one; this test is also the statement of
        that.)

        The budgets are the module's own measured constants, **not** "within 2 ulp": the
        matter-radiation closed form is 7 ulp from the reference on ``QCD_Cosmology``, which is
        the -9.34e-16 that ``RECONCILIATION.md`` §6 reports, so 2 ulp is arithmetically
        unreachable there (board standing note 7). Neither was widened by prompt 09.
        """
        print("\n  the remaining closed-form sites (17 digits):")
        for label, model in self.models + (("LambdaCDM(Planck2018)", self.lambdaCDM),):
            for pair, budget in (
                ("matter_radiation", MATTER_RADIATION_CLOSED_FORM_ULP),
                ("matter_lambda", LAMBDA_CLOSED_FORM_ULP),
            ):
                with self.subTest(model=label, pair=pair):
                    values = {
                        name: float(site(model, pair))
                        for name, site in CLOSED_FORM_SITES
                    }
                    for name, value in values.items():
                        print(f"    {label:>24} {pair:>17} {name:>34}: {value!r}")

                    distinct = set(values.values())
                    self.assertEqual(
                        len(distinct),
                        1,
                        msg=f"{label}, {pair}: the remaining closed-form sites no longer produce "
                        f"the same float -- "
                        + ", ".join(f"{n} = {v!r}" for n, v in values.items())
                        + ". One of them is LambdaCDM's own answer for this redshift and the "
                        "other is the guess LambdaCDM_GenericEOS seeds its solve with, so a site "
                        "that has drifted from the other is either wrong or about to be.",
                    )

                    if not hasattr(model, "_rho_fluid"):
                        # LambdaCDM has no equation of state and no residual to bracket, so there
                        # is no reference to score against. What can be said about it, and is,
                        # is that the property it answers with really is the expression above:
                        # prompt 09 made that expression its answer rather than a diagnostic
                        # printed beside one.
                        self.assertEqual(
                            model_property(model, pair).hex(),
                            values["LambdaCDM.py property"].hex(),
                            msg=f"{label}, {pair}: the model's own property returns "
                            f"{model_property(model, pair)!r}, which is not the closed form "
                            f"{values['LambdaCDM.py property']!r} this test transcribes. For a "
                            "model with no equation of state the closed form is exact, so the "
                            "property must be it and not an approximation to it.",
                        )
                        continue

                    reference = bracketed_reference(model, pair)
                    for name, value in values.items():
                        self.assertLessEqual(
                            abs(value - reference),
                            budget * np.spacing(abs(reference)),
                            msg=f"{label}, {pair}, {name}: the closed form {value!r} is no "
                            f"longer the equality redshift to within its measured budget; "
                            f"bracketed reference = {reference!r}, separation = "
                            f"{ulps_between(value, reference):+.3f} ulp, budget = {budget} ulp",
                        )

    def test_each_model_answers_for_its_own_equality_redshifts(self):
        """
        The ``BaseCosmology`` surface prompt 09 declares: each model supplies both equality
        redshifts, and *how* it supplies them is the model's business.

        ``LambdaCDM`` answers with the closed form, which for a model with no equation of state is
        the exact root rather than an approximation to it -- rho_r is rho_r0 (1+z)^4 identically,
        so 1 + z = Omega_m/Omega_r solves rho_m = rho_r with nothing left over.
        ``LambdaCDM_GenericEOS`` answers with the bracketed solve its constructor runs against its
        own rho_r = RadiationConstant G(T(z)) T(z)^4, and on ``QCD_Cosmology`` that is **not** the
        closed form: the two are 7 ulp apart, and which of them the model returns is what decides
        the production source grid's digest.

        The separation is asserted as an inequality of *bits*, not as a tolerance. 7 ulp is
        3.2e-12 in z; any tolerance loose enough to be worth writing would be satisfied by either
        quantity, which is exactly why the distinction went unnoticed until it was measured.
        """
        for label, model in self.models:
            for pair in ("matter_radiation", "matter_lambda"):
                with self.subTest(model=label, pair=pair):
                    A, B = species_of(pair)
                    solve = float(
                        model._find_rho_equality(
                            A, B, init_z=closed_form_guess(model, pair)
                        )
                    )
                    self.assertEqual(
                        model_property(model, pair).hex(),
                        solve.hex(),
                        msg=f"{label}, {pair}: the model answers {model_property(model, pair)!r}, "
                        f"which is not what _find_rho_equality returns ({solve!r}). A "
                        "LambdaCDM_GenericEOS must answer with its own solve, because the "
                        "radiation-domination closed form is exact for it only where g_* happens "
                        "to be flat at equality.",
                    )

        # LambdaCDM, where the closed form *is* the answer, on both pairs
        for pair in ("matter_radiation", "matter_lambda"):
            with self.subTest(model="LambdaCDM(Planck2018)", pair=pair):
                self.assertEqual(
                    model_property(self.lambdaCDM, pair).hex(),
                    float(lambdaCDM_closed_form(self.lambdaCDM, pair)).hex(),
                )

        # and the distinction that matters: on QCD_Cosmology the model's answer is not the closed
        # form, by the 7 ulp log 03 measured
        closed_form = float(closed_form_guess(self.qcd, "matter_radiation"))
        answer = model_property(self.qcd, "matter_radiation")
        self.assertNotEqual(
            answer.hex(),
            closed_form.hex(),
            msg=f"QCD_Cosmology answers {answer!r} for matter-radiation equality, which is the "
            f"closed form Omega_m/Omega_r - 1 rather than its own solve. The two agree here only "
            "because all of this equation of state's g_*(T) structure sits twelve orders above "
            "z_eq, which is a property of the equation of state and not of the code.",
        )
        self.assertAlmostEqual(ulps_between(answer, closed_form), 7.0, places=6)

    # ------------------------------------------------------------------------------------------
    # The three tests below are the ones that distinguish the trees. Everything above passes both
    # before and after the bracketing change, by design; these fail on the tree that ships the
    # unbracketed secant, and the failure output is quoted in
    # prompts/background-solver-robustness/logs/02-bracket-the-equality-solve.md.
    # (``test_each_model_answers_for_its_own_equality_redshifts`` above is prompt 09's equivalent:
    # it fails on the tree where BaseCosmology declares no such surface.)
    # ------------------------------------------------------------------------------------------

    def test_a_displaced_guess_now_finds_the_root_instead_of_failing(self):
        """
        The solve finds the root from a guess that is merely in the right region, which is what
        "bracketed" buys and what the shipped secant did not have.

        AUDIT.md §3.2 walked the matter/radiation guess down and measured the shipped solve
        breaking in three different ways within 30 % of the answer: at -20 % it reported
        converged=True with a root 2.7e-9 relative away -- a silently wrong answer well inside the
        rtol=1e-4 it asked for -- at -30 % it raised ValueError from a negative argument to a
        power, and at -50 % and beyond it had iterated to negative z, below the T(z) tabulation's
        floor, and raised a bounds error from two frames inside _rho_fluid.

        A bracket expanded about the guess and clamped to the tabulated range removes all four
        cases at once: the residual is monotone across the range its root lives in
        (test_the_density_ratios_are_monotone_where_their_roots_live above), so the expansion
        straddles and Brent converges on the same root whatever the guess was.
        """
        reference = bracketed_reference(self.qcd, "matter_radiation")
        z_guess = closed_form_guess(self.qcd, "matter_radiation")
        for fraction in DISPLACED_GUESS_FRACTIONS:
            with self.subTest(fraction=fraction):
                got, _ = solve_recording_evaluations(
                    self.qcd, "matter_radiation", fraction * z_guess
                )
                self.assertLessEqual(
                    abs(got - reference),
                    SOLVE_VS_REFERENCE_ULP * np.spacing(abs(reference)),
                    msg=f"a guess displaced to {fraction:g} x the analytic value "
                    f"({fraction * z_guess!r}) did not recover the matter/radiation equality "
                    f"redshift: solve = {got!r}, reference = {reference!r}, separation = "
                    f"{ulps_between(got, reference):+.3f} ulp, "
                    f"budget = {SOLVE_VS_REFERENCE_ULP} ulp",
                )

    def test_a_failure_to_bracket_raises_the_methods_own_error(self):
        """
        When there is genuinely no root to find, the error is _find_rho_equality's own and it
        names the species pair -- not a TemperatureRepresentation bounds error from two frames
        down.

        This is the guard at the end of the bracket expansion. The old `if not root.converged`
        guard was dead on every path that actually failed (log 01 §4.1: root.converged was True at
        every offset that returned at all, and every offset that did not returned raised from
        inside _rho_fluid), so a constructor that died this way reported a temperature-spline
        problem to a user whose actual mistake was a cosmology whose equality redshift is outside
        its own tabulated range. That is exactly the cosmology built here.

        The assertion is on the species names and the method name, never on the whole message,
        which is prose. What it must *not* contain is asserted separately, because "raises
        RuntimeError" alone is true on both trees.
        """
        with self.assertRaises(RuntimeError) as caught:
            LambdaCDM_GenericEOS(
                store_id=12,
                eos=PureRadiationEOS(self.units, lambdaCDM_gstar(self.params.Neff)),
                units=self.units,
                params=self.params,
                max_z=TRUNCATED_MAX_Z,
            )

        message = str(caught.exception)
        print(f"\n  failure to bracket:\n    {message}")

        for expected in ("_find_rho_equality", "matter", "radiation"):
            self.assertIn(
                expected,
                message,
                msg=f"a failure to bracket the equality redshift must name {expected!r}; the "
                f"message was: {message}",
            )
        self.assertNotIn(
            "TemperatureRepresentation",
            message,
            msg="a failure to bracket the equality redshift is being reported as a temperature "
            "representation bounds error, which means the solve reached _rho_fluid outside the "
            f"tabulated range instead of stopping at it. The message was: {message}",
        )

    def test_the_bracket_does_not_leave_the_tabulated_range(self):
        """
        Every z at which the solve evaluates _rho_fluid is inside the T(z) representation's own
        tabulated range, for guesses displaced far below and far above the root.

        This is the direct statement of the defect the bracket removes. The shipped secant, handed
        a guess 50 % low, iterated to z = -0.25719 and then to z = -0.38344 -- below the
        representation's floor of z = -0.24 -- and the only reason that was visible at all is that
        the representation rejects out-of-range arguments. Asserting it on the *arguments* rather
        than on the exception is what makes this test independent of what the representation
        chooses to do when it is asked out of range.
        """
        z_floor = self.qcd._T_z_spline._min_z
        z_ceil = self.qcd._T_z_spline._max_z
        z_guess = closed_form_guess(self.qcd, "matter_radiation")

        # 1 % of the analytic guess is far enough below the root that the expansion reaches the
        # floor before it straddles, so the clamp is exercised rather than merely present; 10x the
        # tabulated ceiling exercises the clamp applied to the guess itself.
        for init_z in (0.01 * z_guess, 10.0 * z_ceil):
            with self.subTest(init_z=init_z):
                _, evaluations = solve_recording_evaluations(
                    self.qcd, "matter_radiation", init_z
                )
                self.assertGreaterEqual(
                    min(evaluations),
                    z_floor,
                    msg=f"from init_z={init_z!r} the solve evaluated _rho_fluid at "
                    f"z={min(evaluations)!r}, below the tabulated floor z={z_floor!r}; the "
                    "bracket has left the range the representation is defined on",
                )
                self.assertLessEqual(
                    max(evaluations),
                    z_ceil,
                    msg=f"from init_z={init_z!r} the solve evaluated _rho_fluid at "
                    f"z={max(evaluations)!r}, above the tabulated ceiling z={z_ceil!r}; the "
                    "bracket has left the range the representation is defined on",
                )


if __name__ == "__main__":
    unittest.main()
