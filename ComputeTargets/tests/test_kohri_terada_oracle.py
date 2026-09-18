"""
Tests for ComputeTargets/tests/kohri_terada.py, the Kohri & Terada radiation-era oracle
(arXiv:1804.08577), after prompts/radiation-oracle prompt 01.

Offline: no Ray runtime, no datastore. Six tests, numbered as prompt 01 section 3 numbers them:

  1. eq. (22) against scipy.quad of eq. (15), over the audit's section 5 grid and more;
  2. the small-x limit, 2 x^2 / 9 (NOT the paper's x^2 / 2), and its next order;
  3. the large-x limit, eq. (25), approached like 1/x;
  4. the resonance u + v = sqrt(3), evaluated exactly on it;
  5. eq. (16) at w = 1/3 against eq. (20), away from eq. (20)'s cancellation;
  6. this repository's own `total` against eq. (22): N = -9/8.

READ THIS BEFORE TRUSTING 1-5. They compare the module's transcriptions of Kohri & Terada with
each other, and eq. (22) and the quadrature of eq. (15) share the Green's function, the measure
and the source, so 1-5 cannot detect a misread kernel. Test 6 is the only one with an object in it
that nobody in the radiation-oracle campaign wrote (docs/radiation-oracle/KOHRI-TERADA-ORACLE.md
section 0).

Every comparison is made against a bound built from the reference's own declared error -- quad's
error estimate, eq. (22)'s or eq. (20)'s rounding scale (eps times the sum of the absolute values
of the terms each adds), or the pipeline's `total_abserr` -- and every failure message prints it.
"""

import math
import unittest

from ComputeTargets.tests.kohri_terada import (
    KT_NORM,
    SQRT3,
    Phi,
    dPhi,
    Cin,
    f_RD,
    f_RD_eq20_with_rounding_scale,
    I_RD,
    I_RD_with_rounding_scale,
    I_RD_asymptotic,
    I_RD_quadrature,
    total_from_I_RD,
)
from ComputeTargets.tests.test_quadsource_integral import (
    Case,
    SHAPES,
    X_RESP_VALUES,
)

EPS = 2.0**-52

# the reference tolerance pair docs/radiation-oracle/kt_verification.py ran the pipeline at.
# prompt 01 section 3: fewer cases if test 6 is too slow, never a looser pair.
REF_ATOL, REF_RTOL = 1e-45, 1e-12

# KOHRI-TERADA-ORACLE.md section 5, the ten (v, u, x) of the audit's own eq. (22) check
AUDIT_GRID = (
    (0.7, 1.1, 1.0),
    (0.7, 1.1, 5.0),
    (0.7, 1.1, 20.0),
    (0.7, 1.1, 60.0),
    (1.0, 1.0, 10.0),
    (0.3, 1.6, 12.0),
    (1.2, 0.5, 25.0),
    (0.9, 0.87, 40.0),
    (0.86, 0.87, 15.0),
    (0.9, 0.9, 15.0),
)

# and the (v, u) shapes of test_quadsource_integral's SHAPES near their smallest x, where eq. (22)
# is least well conditioned: T-first (u = 10, v = 12) and q-smooth (u = 0.01, v = 1.1)
PIPELINE_SHAPED = (
    (1.2, 1.0, 14.43),
    (12.0, 10.0, 4.33),
    (1.1, 0.01, 47.24),
)

# u = v = sqrt(3)/2: u + v is sqrt(3) EXACTLY in binary floating point (doubling is exact), so
# 1 - (u + v)/sqrt(3) is exactly 0. A decimal literal such as 0.8660254 would miss the resonance.
RESONANT = SQRT3 / 2.0


def rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def assert_finite(test: unittest.TestCase, value: float, scale: float):
    """
    Every bound below is built from eq. (22)'s own rounding scale, so an inf or nan in the
    closed form would also make its bound inf or nan and could pass a <= comparison. Refuse it
    first.
    """
    test.assertTrue(
        math.isfinite(value) and math.isfinite(scale),
        f"eq. (22) = {value}, its rounding scale {scale}",
    )


class TestSeriesBranches(unittest.TestCase):
    """Phi, dPhi and Cin change formula at 1; the two sides must meet there to rounding."""

    def test_phi_and_dphi_branches_meet(self):
        # just below the cut the series is exact to rounding; just above it the closed form has
        # lost at most ~30x to cancellation (dPhi's two terms ~ 2/x against a result ~ x/15)
        lo, hi = math.nextafter(1.0, 0.0), 1.0
        self.assertLess(rel(Phi(lo), Phi(hi)), 16 * EPS)
        self.assertLess(rel(dPhi(lo), dPhi(hi)), 64 * EPS)
        # and the limits the source's x -> 0 behaviour rests on
        self.assertEqual(Phi(0.0), 1.0)
        self.assertEqual(dPhi(0.0), 0.0)
        self.assertLess(abs(f_RD(0.7, 1.1, 1e-8) - 4.0 / 3.0), 4 * EPS)

    def test_cin_branches_meet(self):
        lo, hi = math.nextafter(1.0, 0.0), 1.0
        self.assertLess(rel(Cin(lo), Cin(hi)), 16 * EPS)
        self.assertEqual(Cin(0.0), 0.0)
        # Cin(z) = z^2/4 - z^4/96 + ... : the leading term, where the next is 4e-20 of it
        self.assertLess(rel(Cin(1e-9), 0.25e-18), 4 * EPS)


class TestClosedFormAgainstQuadrature(unittest.TestCase):
    """Test 1: KT eq. (22) against scipy.quad of KT eq. (15)."""

    def test_eq22_matches_quadrature_of_eq15(self):
        """
        The bound is the two objects' own declared errors added: quad's error estimate for the
        quadrature, and eps times eq. (22)'s rounding scale for the closed form. The audit's worst
        on its grid is 3.5e-15 relative (KOHRI-TERADA-ORACLE.md section 5); quad declares
        2.3e-15 to 8.7e-15 absolute there, so on that grid the bound is quad's. The
        pipeline-shaped points add u = 0.01, where eq. (22)'s own rounding (~3e-9 relative) is
        the larger term. A quadrature declaring worse than 1e-12 relative is refused rather than
        allowed to loosen the bound (the audit's grid declares up to 1.9e-13).
        """
        for v, u, x in AUDIT_GRID + PIPELINE_SHAPED:
            with self.subTest(v=v, u=u, x=x):
                closed, scale = I_RD_with_rounding_scale(v, u, x)
                assert_finite(self, closed, scale)
                quad_value, quad_err = I_RD_quadrature(v, u, x)
                self.assertLess(
                    quad_err,
                    1e-12 * abs(quad_value),
                    f"quad's own declared error {quad_err:.2e} is too loose to score against",
                )
                bound = quad_err + EPS * scale
                self.assertLessEqual(
                    abs(closed - quad_value),
                    bound,
                    f"eq. (22) = {closed:+.15e}, quad = {quad_value:+.15e}: |diff| "
                    f"{abs(closed - quad_value):.2e} ({rel(closed, quad_value):.2e} relative) "
                    f"against quad's own error {quad_err:.2e} + eq. (22)'s rounding "
                    f"{EPS * scale:.2e}",
                )


class TestSmallX(unittest.TestCase):
    """Test 2: I_RD -> 2 x^2 / 9 as x -> 0."""

    def test_leading_term_is_two_ninths_x_squared(self):
        """
        ERRATUM 2. Kohri & Terada say below eq. (24) that I_RD ~ x^2/2 for small x. That remark
        is wrong by 4/9 and is deliberately not asserted anywhere: f_RD -> 4/3, so
        I -> (4/3) int_0^x (xbar/x)(x - xbar) dxbar = 2 x^2 / 9.

        The next order is derived the same way. Phi(y) = 1 - y^2/30 + O(y^4) gives
        f_RD = 4/3 - (2/27)(u^2 + v^2) x^2 + O(x^4), and
        sin(x - xbar) = (x - xbar) - (x - xbar)^3/6 + ..., so

            I_RD / (2 x^2/9) = 1 - alpha x^2 + O(x^4),   alpha = 1/20 + (u^2 + v^2)/60.

        The leading term is independent of u and v; alpha is not, so it tests the source's
        second-order term as well as its leading one. The O(x^4) coefficient measures 1.9e-3 to
        3.2e-3 on the quadrature over these pairs; the bound allows 1e-2.

        Two references. The quadrature of eq. (15) is exact to quad's declared error at every x.
        Eq. (22) cancels at small x -- its terms exceed its O(x^2) result by ~5e2 / x^4 -- so it is
        held to eps times its rounding scale, which at x = 0.01 is ~2e-5 of 2 x^2/9: the audit's
        0.9999861 of 2 x^2/9 at x = 0.01 is that rounding, and 1 - alpha x^2 = 0.9999922.
        """
        for v, u in (
            (0.7, 1.1),
            (1.2, 0.5),
            (RESONANT, RESONANT),
            (0.3, 1.6),
            (1.1, 0.01),
        ):
            alpha = 1.0 / 20.0 + (u * u + v * v) / 60.0
            for x in (0.3, 0.1, 0.03, 0.01):
                leading = 2.0 * x * x / 9.0
                expected = 1.0 - alpha * x * x
                truncation = 1e-2 * x**4

                with self.subTest(v=v, u=u, x=x, reference="quadrature"):
                    quad_value, quad_err = I_RD_quadrature(v, u, x)
                    ratio = quad_value / leading
                    self.assertLessEqual(
                        abs(ratio - expected),
                        truncation + quad_err / leading,
                        f"quadrature / (2x^2/9) = {ratio:.13f}, expected 1 - alpha x^2 = "
                        f"{expected:.13f}; quad's own error {quad_err / leading:.1e} of 2x^2/9",
                    )

                with self.subTest(v=v, u=u, x=x, reference="eq. (22)"):
                    closed, scale = I_RD_with_rounding_scale(v, u, x)
                    # an inf or nan must fail here, not slip through the skip below
                    assert_finite(self, closed, scale)
                    rounding = EPS * scale / leading
                    if rounding > 1e-3:
                        # u = 0.01 at x <= 0.03: eq. (22) has no digits left to test; the
                        # quadrature branch above still covers the pair
                        continue
                    ratio = closed / leading
                    self.assertLessEqual(
                        abs(ratio - expected),
                        truncation + rounding,
                        f"eq. (22) / (2x^2/9) = {ratio:.13f}, expected {expected:.13f}; "
                        f"eq. (22)'s own rounding {rounding:.1e} of 2x^2/9",
                    )


class TestLargeX(unittest.TestCase):
    """Test 3: I_RD -> eq. (25) as x -> infinity, like 1/x."""

    def test_approaches_eq25_like_one_over_x(self):
        """
        The trend, not a tolerance at one x. Both eq. (22) and eq. (25) oscillate, so a relative
        difference at a single x is dominated by where the zeros fall (the audit's 1.03e-01,
        3.30e-03, 7.06e-04 for (0.7, 1.1) at x = 200, 2000, 20000, and a non-monotone sequence
        for (0.3, 1.6)). Instead: D(X) = max |I_RD - eq. (25)| over four periods of sin x from
        X, divided by eq. (25)'s own envelope at X. Eq. (22)'s corrections to eq. (25) are
        Ci/Si tails, O(1/x) relative, so D must fall by a factor of 10 per decade of X. It
        measures 0.099 to 0.108 per decade; the bound is [0.08, 0.125], i.e. a log-slope of
        -1 +- 0.1.

        Eq. (22)'s rounding is no factor here: its rounding scale is < 1e-12 of the envelope at
        every X used.
        """
        samples = 256
        for v, u in ((0.7, 1.1), (1.2, 0.5), (0.3, 1.6), (1.0, 1.0), (0.9, 0.87)):
            d = u * u + v * v - 3.0
            log_arg = math.log(abs((3.0 - (u + v) ** 2) / (3.0 - (u - v) ** 2)))
            theta = 1.0 if u + v > SQRT3 else 0.0
            amplitude = (
                3.0
                * abs(d)
                / (4.0 * u**3 * v**3)
                * math.hypot(-4.0 * u * v + d * log_arg, math.pi * d * theta)
            )

            D = []
            for X in (200.0, 2000.0, 20000.0, 200000.0):
                worst = 0.0
                for i in range(samples):
                    x = X + 8.0 * math.pi * i / samples
                    closed, scale = I_RD_with_rounding_scale(v, u, x)
                    self.assertLess(EPS * scale, 1e-12 * amplitude / X)
                    worst = max(worst, abs(closed - I_RD_asymptotic(v, u, x)))
                D.append(worst / (amplitude / X))

            for (X0, D0), D1 in zip(zip((200, 2000, 20000), D[:-1]), D[1:]):
                with self.subTest(v=v, u=u, decade=f"{X0} -> {10 * X0}"):
                    self.assertTrue(
                        0.08 <= D1 / D0 <= 0.125,
                        f"max |eq. (22) - eq. (25)| / envelope fell {D0:.3e} -> {D1:.3e}, "
                        f"a factor {D1 / D0:.3f} per decade; 1/x approach is 0.1",
                    )


class TestResonance(unittest.TestCase):
    """Test 4: exactly at u + v = sqrt(3), where eq. (22) as printed is infinite."""

    def test_finite_and_correct_on_the_resonance(self):
        """
        u = v = sqrt(3)/2 puts u + v on sqrt(3) exactly (see RESONANT), so -Ci(|1 - (u+v)/sqrt3| x)
        is -Ci(0) = +inf and the log is ~ log(4e-16), and an unregularised eq. (22) returns inf.
        The Cin form must return the finite value, and match the quadrature of eq. (15) -- whose
        integrand has no singularity there at all -- to the two objects' declared errors.

        The audit's figure is 0.1391877548292 at x = 15, matching quadrature to 4.0e-16
        (KOHRI-TERADA-ORACLE.md section 6.1); it is printed to 13 digits, so it is held to half a
        unit in the last: 5e-14.
        """
        self.assertEqual(1.0 - (RESONANT + RESONANT) / SQRT3, 0.0)

        for x in (0.5, 15.0, 150.0):
            with self.subTest(x=x):
                closed, scale = I_RD_with_rounding_scale(RESONANT, RESONANT, x)
                self.assertTrue(
                    math.isfinite(closed) and math.isfinite(scale),
                    f"I_RD = {closed} on the resonance (rounding scale {scale})",
                )
                quad_value, quad_err = I_RD_quadrature(RESONANT, RESONANT, x)
                self.assertLessEqual(
                    abs(closed - quad_value),
                    quad_err + EPS * scale,
                    f"on the resonance at x = {x}: eq. (22) = {closed:+.15e}, quad = "
                    f"{quad_value:+.15e}, |diff| {abs(closed - quad_value):.2e} against quad's "
                    f"own error {quad_err:.2e} + eq. (22)'s rounding {EPS * scale:.2e}",
                )

        self.assertLessEqual(
            abs(I_RD(RESONANT, RESONANT, 15.0) - 0.1391877548292),
            5e-14,
            f"I_RD on the resonance at x = 15 is {I_RD(RESONANT, RESONANT, 15.0):.15f}; "
            f"the audit's 0.1391877548292 is good to 5e-14",
        )

    def test_continuous_approaching_the_resonance(self):
        """
        The audit's section 6.1 sequence: u + v - sqrt(3) from 1e-2 down to 0 at x = 15, each
        point against the quadrature at the two declared errors. A literal eq. (22) loses digits
        along it as Ci and the log grow; the Cin form does not.
        """
        for gap in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
            u = v = (SQRT3 + gap) / 2.0
            with self.subTest(gap=gap):
                closed, scale = I_RD_with_rounding_scale(v, u, 15.0)
                assert_finite(self, closed, scale)
                quad_value, quad_err = I_RD_quadrature(v, u, 15.0)
                self.assertLessEqual(
                    abs(closed - quad_value),
                    quad_err + EPS * scale,
                    f"u + v - sqrt3 = {gap:.0e}: eq. (22) = {closed:+.15e}, quad = "
                    f"{quad_value:+.15e}; quad's own error {quad_err:.2e}, eq. (22)'s rounding "
                    f"{EPS * scale:.2e}",
                )


class TestSource(unittest.TestCase):
    """Test 5: eq. (16) at w = 1/3 against eq. (20)."""

    def test_eq16_matches_eq20_away_from_the_cancellation(self):
        """
        Two formulas in the paper for the same function, sharing no code: this is what licenses
        integrating eq. (16) in place of eq. (20), and what confirms eq. (16)'s
        "xbar d_etabar Phi" is xbar d_xbar Phi. The audit's grid (KOHRI-TERADA-ORACLE.md
        section 4), all at x >= 0.5, where eq. (20) has lost at most ~1e-11 to cancellation.

        The bound is eq. (20)'s own rounding, eps times its scale, plus 16 ulp of f for
        eq. (16)'s arithmetic: four products of Phi and dPhi, each good to an ulp or two given
        its argument. Both formulas evaluate sin and cos of the same rounded u x / sqrt3 and
        v x / sqrt3, so that rounding is common to the two and cancels from the difference.
        """
        for v, u in ((0.7, 1.1), (1.0, 1.0), (0.3, 1.6), (1.2, 0.5), (0.9, 0.87)):
            for x in (0.5, 2.0, 7.3, 31.0, 100.0):
                with self.subTest(v=v, u=u, x=x):
                    eq16 = f_RD(v, u, x)
                    eq20, scale20 = f_RD_eq20_with_rounding_scale(v, u, x)
                    bound = EPS * scale20 + 16.0 * EPS * abs(eq16)
                    self.assertLessEqual(
                        abs(eq16 - eq20),
                        bound,
                        f"eq. (16) = {eq16:+.15e}, eq. (20) = {eq20:+.15e}, |diff| "
                        f"{abs(eq16 - eq20):.2e}; eq. (20)'s own rounding {EPS * scale20:.2e}",
                    )


class TestPipeline(unittest.TestCase):
    """Test 6: this repository's `total` is -9/8 / k_phys^2 times eq. (22)."""

    def test_total_is_minus_nine_eighths_of_I_RD(self):
        """
        THE TIE TO THE PIPELINE, AND THE ONLY TEST HERE THAT CAN CATCH A MISREAD KERNEL. On all
        nine b = 0 cases of test_quadsource_integral (three SHAPES x three X_RESP_VALUES, exact
        flavour, at the reference pair (1e-45, 1e-12)), evaluate_QuadSource_integral's `total`
        must equal

            total = KT_NORM / k_phys^2 * [ I_RD(r/k, q/k, k tau(z_resp)) - head ],  KT_NORM = -9/8,

        through total_from_I_RD, where the head 0 -> k tau(z_source_max) is quadratured because
        KT integrate from 0 and the code from z_source_max. It is scored against eq. (22) with
        the head subtracted -- NOT against a quadrature of KT's integrand between the pipeline's
        limits, which would pass with eq. (22) transcribed wrongly -- and it asserts N = -9/8,
        not merely that N is constant: a factor of two or a sign anywhere in the prefactor chain
        (1/c^2, h^us = h^them / 2, spec 04 section 0 (3)'s orientation) moves N off -9/8 and
        leaves it constant.

        THE BOUND is the sum of the two sides' own declared relative errors: the oracle's
        (eps times eq. (22)'s rounding scale, plus the head's quad error) and the pipeline's
        (its `total_abserr`, or the requested rtol applied to |numeric_quad| + |WKB_Levin| where
        that is larger, because the two halves cancel). Where eq. (22) is well conditioned that
        is 1.1e-12 to 9.9e-12, set by the pipeline. At q-smooth, u = q/k = 0.01, eq. (22)'s
        1/(u^3 v^3) prefactor multiplies terms that cancel, and the bound is 3.7e-9 to 1.6e-8,
        set by eq. (22)'s own declared rounding -- a conservative estimate (its actual error at
        these three points is 2.5e-11 to 9.7e-11 against 50-digit mpmath), which is the price
        of not trusting a number the suite cannot check. The audit's figure against eq. (22) was
        3.18e-10, on that shape, and is that rounding; its figure against the quadrature,
        3.09e-13, is not comparable and is not used. No case's bound may exceed 1e-7, an order
        above the loosest, so that a change to the rounding estimate that made this test
        vacuous would announce itself.

        THIS IS ALSO A LEVIN TEST, which is not obvious. total = numeric_quad + WKB_Levin, and on
        these nine cases the Levin half runs over 2-3 regions apiece and carries 0.099 to 3.846
        of |total| -- more than the total itself in four of them, so the halves cancel. An error
        eps_L in the Levin half enters `total` as eps_L |WKB_Levin| / |total|, so agreement here
        bounds it, and the cancellation tightens that bound rather than hiding behind it. Three
        limits on the claim:
          * it bounds the COMBINATION: a systematic Levin error cancelling against a compensating
            numeric_quad error would pass;
          * it exercises the `total` path only. It does NOT reach analytic_rad's
            _three_bessel_Levin, a separate code path, which is qsi-phase-groups'
            [06-analytic-rad-is-computed-at-the-callers-tolerance] and is untouched by this;
          * it is the EXACT flavour only, so the Levin algorithm is scored on exact ingredients
            with no representation floor present.
        """
        for shape in SHAPES:
            for x_resp in X_RESP_VALUES:
                with self.subTest(shape=shape.name, x_resp=x_resp):
                    case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=True)
                    out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
                    total = float(out["total"])
                    halves = abs(float(out["numeric_quad"])) + abs(
                        float(out["WKB_Levin"])
                    )
                    pipeline_err = max(float(out["total_abserr"]), REF_RTOL * halves)

                    tau = case.model.functions.tau
                    oracle = total_from_I_RD(
                        shape.k,
                        shape.q,
                        shape.r,
                        tau(case.z_resp),
                        tau(case.z_source_max),
                    )
                    predicted = oracle["total"]
                    N = KT_NORM * total / predicted

                    bound_rel = oracle["total_error"] / abs(predicted) + (
                        pipeline_err / abs(total)
                    )
                    self.assertLessEqual(bound_rel, 1e-7)
                    self.assertLessEqual(
                        abs(N - KT_NORM),
                        abs(KT_NORM) * bound_rel,
                        f"{case.label()}: N = {N:.15f}, |N + 9/8| = {abs(N - KT_NORM):.2e}; "
                        f"total = {total:+.15e}, eq. (22) - head predicts {predicted:+.15e}; "
                        f"oracle's own error {oracle['total_error'] / abs(predicted):.1e}, "
                        f"pipeline's own error {pipeline_err / abs(total):.1e} (relative)",
                    )


if __name__ == "__main__":
    unittest.main()
