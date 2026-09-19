"""
Tests for ComputeTargets/tests/domenech.py, the Domenech general-b kernel (arXiv:1912.05583v3 and
arXiv:2109.01398v2), after prompts/handover prompt 01.

Offline: no Ray runtime, no datastore. Eight groups, numbered as prompt 01 section 3 numbers them:

  1. the Legendre mapping -- each of the six mpmath calls of recon section 9 against an
     independent check: two Wronskians, the elementary closed forms, the parity relation, the
     papers' own special cases at b = 0 and b = +-1/2, and the type-3 phase conversion;
  2. the closed forms of A, B and C against those raw calls, at mp.dps = 30, at several b;
  3. I_target against I_quadrature: the 1/x approach of recon section 2.4's table, on BOTH
     Gervois-Navelet branches;
  4. I_asymptotic against I_target: the same 1/x, and their exact coincidence at b = 0;
  5. the b = 0 reduction -- (9/8) x kohri_terada, THE LOAD-BEARING EXTERNAL TIE;
  6. the small-x limit x^2/(2(2+b));
  7. the resonance c_s(u+v) = 1;
  8. N constant over the nine b = 0.2 fixture cases of test_quadsource_integral.

READ THIS BEFORE TRUSTING 1-4, 6 AND 7. They compare the module's own transcriptions with each
other. I_target and I_quadrature share the prefactor and the outer Bessels, so 3 checks the
Gervois-Navelet coefficients against the finite-x integrals -- a real cross-check between two
different pages -- but not the weight xbar^{1/2-b}, the source's rho, or the overall sign. Test 5
and test 8 are the only ones with an object in them that this campaign did not write: test 5's is
`ComputeTargets/tests/kohri_terada.py`, a transcription of a different paper by a different
campaign, and test 8's is this repository's own `total`. (KOHRI-TERADA-ORACLE.md section 0.)

Every comparison is made against a bound built from the references' own declared errors -- quad's
error estimate, eq. (22)'s rounding scale, an eq. (25) rounding scale built here to the same
recipe, or the pipeline's `total_abserr` -- or, where a single number would be dominated by where
the zeros fall, against a TREND with no tolerance in it at all (the 1/x of tests 3 and 4, the
|delta|^b of test 7, the x^2 of test 6).
"""

import math
import unittest

import mpmath as mp

from ComputeTargets.tests.domenech import (
    MPMATH_DPS,
    A_coefficient,
    B_at_resonance,
    B_coefficient,
    C_coefficient,
    I_asymptotic,
    I_quadrature,
    I_target,
    OutsideGervoisNavelet,
    cs_of_b,
    ferrers_P,
    ferrers_Q,
    kernel_constant,
    kinematics,
    kt_convention_norm,
    olver_Q,
    rho_of_b,
    small_x_limit,
    total_from_I_rev,
    two_c_squared,
)
from ComputeTargets.tests.kohri_terada import (
    SQRT3,
    I_RD,
    I_RD_asymptotic,
    I_RD_with_rounding_scale,
)
from ComputeTargets.tests.test_quadsource_integral import (
    Case,
    SHAPES,
    X_RESP_VALUES,
)

EPS = 2.0**-52

# the reference tolerance pair docs/radiation-oracle/kt_verification.py ran the pipeline at, and
# that test_kohri_terada_oracle's test 6 uses. Never a looser pair.
REF_ATOL, REF_RTOL = 1e-45, 1e-12

# b = 0.2 is the fixture value; the rest span the papers' -1/2 <= b < 1 without touching the two
# mpmath fragilities of recon section 3.4 (b = +-1/2, where legenp/legenq type=2 raise
# "hypsum() failed to converge" on a degenerate 2F1 connection case). Those two b are covered
# instead by the papers' own closed forms, in test_special_values_at_b_zero_and_half.
GENERIC_B = (0.2, 0.37, 0.8, 0.9, -0.3, -0.45)

# recon section 7's (u, v, x): the six points at which the reconnaissance measured a quadrature of
# the review's own G.f against (9/8) x eq. (22). (10, 12, 141) is the `T-first` shape, which at
# b = 0 has c_s|u - v| = 1.155 and y = 1.0042 -- see test_refuses_the_T_first_shape_at_b_zero.
KT_TIE_POINTS = (
    (0.7, 1.1, 20.0),
    (0.7, 1.1, 200.0),
    (1.2, 0.5, 100.0),
    (0.3, 1.6, 300.0),
    (0.01, 1.1, 50.0),
)
KT_TIE_T_FIRST = (10.0, 12.0, 141.0)


def rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def envelope(v: float, u: float, x: float, b: float) -> float:
    """
    The large-x envelope of the target object, recon section 2.4's unit of distance:
    K(b) x^{-b-1}/(c_s^2 u v) times the modulus of the (cos, 2/pi sin) coefficient pair. Every
    figure in recon sections 2.4, 4.2, 5.4 and 6.4 is a difference divided by this, so it is what
    a test has to reproduce them in.
    """
    kin = kinematics(v, u, b)
    pre = kernel_constant(b) * x ** (-b - 1.0) / (kin.cs * kin.cs * u * v)
    if kin.on_cut:
        return abs(pre) * math.hypot(
            A_coefficient(kin.one_plus_y, kin.one_minus_y, b),
            2.0 / math.pi * B_coefficient(kin.one_plus_y, kin.one_minus_y, b),
        )
    return (
        abs(pre)
        * 2.0
        / math.pi
        * abs(C_coefficient(kin.one_plus_y, kin.one_minus_y, b))
    )


def eq25_rounding_scale(v: float, u: float, x: float) -> float:
    """
    An absolute rounding estimate for kohri_terada.I_RD_asymptotic (KT eq. (25)), to the same
    recipe as that module's I_RD_with_rounding_scale: the magnitude of every term the formula
    adds, plus the error its inputs' rounding induces. Eq. (25) has no such helper of its own and
    this test needs one, because at u = q/k = 0.01 the two terms of its sin bracket,
    -4uv = -4.4e-02 and d log|.| = +4.4e-02, cancel to 1.4e-05 -- the same 1/(u^3 v^3)
    conditioning that KOHRI-TERADA-ORACLE.md section 7.2 traced eq. (22)'s 3.18e-10 to. It is
    computed here rather than in kohri_terada.py, which this prompt does not edit.

        eq. (25) = P ( sin x [-4uv + d L] - pi d Theta cos x ),  P = 3d/(4 u^3 v^3 x),
        d = u^2 + v^2 - 3,  L = log|(3-(u+v)^2)/(3-(u-v)^2)|.

    d is off by ~eps(u^2+v^2+3); L by ~eps times the sum of |(u+-v)^2 / (3 - (u+-v)^2)|, since
    log(a/b) inherits the relative error of each; and the outer d multiplies everything.
    """
    d = u * u + v * v - 3.0
    plus, minus = (u + v) ** 2, (u - v) ** 2
    L = math.log(abs((3.0 - plus) / (3.0 - minus)))
    theta = 1.0 if (v + u - SQRT3) > 0.0 else 0.0

    d_err = EPS * (u * u + v * v + 3.0)
    L_err = EPS * (plus / abs(3.0 - plus) + minus / abs(3.0 - minus))

    sin_coeff = -4.0 * u * v + d * L
    sin_err = EPS * 4.0 * u * v + abs(L) * d_err + abs(d) * L_err
    cos_coeff = math.pi * d * theta
    cos_err = math.pi * theta * d_err

    P = 3.0 * d / (4.0 * u**3 * v**3 * x)
    inner = abs(math.sin(x)) * abs(sin_coeff) + abs(math.cos(x)) * abs(cos_coeff)
    inner_err = abs(math.sin(x)) * sin_err + abs(math.cos(x)) * cos_err
    return abs(P) * inner_err + abs(P) * inner * (d_err / abs(d) + EPS)


# =============================================================================================
# Test 1 -- the Legendre mapping
# =============================================================================================


class TestLegendreMapping(unittest.TestCase):
    """
    Test 1. The foundation: if the six mpmath calls of recon section 3.2 are not the papers'
    functions, everything built on them is wrong in a way no other test here can see, because A,
    B and C are checked against those same calls in test 2. Each check below is INDEPENDENT of
    the calls' definitions -- a Wronskian, an elementary reduction, a parity relation, or the
    papers' own printed special cases.
    """

    def test_ferrers_wronskian(self):
        """
        DLMF 14.2.4: W{P^mu_nu, Q^mu_nu}(x) = Gamma(nu+mu+1) / (Gamma(nu-mu+1)(1-x^2)), for the
        Ferrers functions. With mu = -b and nu = b + j, j in {0, 2}, the right-hand side is
        Gamma(j+1) / (Gamma(2b+j+1)(1-y^2)), and this pins `legenp(nu, -b, y, type=2)` and
        `legenq(nu, -b, y, type=2)` TOGETHER: a wrong normalisation or a wrong convention in
        either would break it, and it is an identity, not a transcription.

        The right-hand side is written in j rather than in nu - b and nu + b because nu is a
        double: b + 2 - b is not b + 2 - b in floating point, and Gamma of an argument off by
        4e-16 moves by 5e-16, which is larger than the identity's own residue. Written this way
        the identity holds to 1.1e-16, which is nu = b + 2's own representation error inside the
        mpmath call; the bound is 1e-14, still five orders inside any normalisation slip.
        """
        for b in GENERIC_B:
            for j in (0, 2):
                nu = b + j
                for y in (-0.8, -0.3, 0.25, 0.75):
                    with self.subTest(b=b, nu=nu, y=y):
                        with mp.workdps(MPMATH_DPS):
                            P = lambda t: mp.legenp(nu, -b, t, type=2)
                            Q = lambda t: mp.legenq(nu, -b, t, type=2)
                            # the identity needs derivatives, so it is evaluated from mp
                            # calls rather than from the module's float-returning helpers --
                            # but it is anchored to them, so that it fails if `ferrers_P` or
                            # `ferrers_Q` ever stops being the call checked here
                            self.assertEqual(ferrers_P(nu, b, y), float(P(y)))
                            self.assertEqual(ferrers_Q(nu, b, y), float(Q(y)))
                            W = P(y) * mp.diff(Q, y) - mp.diff(P, y) * Q(y)
                            expected = (
                                mp.gamma(j + 1)
                                * mp.rgamma(2 * mp.mpf(b) + j + 1)
                                / (1 - mp.mpf(y) ** 2)
                            )
                            self.assertLess(
                                float(abs(W - expected) / abs(expected)),
                                1e-14,
                                f"DLMF 14.2.4 at b={b}, nu={nu}, y={y}: "
                                f"W = {mp.nstr(W, 20)}, expected {mp.nstr(expected, 20)}",
                            )

    def test_olver_wronskian(self):
        """
        DLMF 14.2.8: W{P^{-mu}_nu(x), **Q**^mu_nu(x)} = -1 / (Gamma(nu+mu+1)(x^2-1)) for the
        |x| > 1 functions, **Q** being Olver's. With mu = -b the pairing is
        `legenp(nu, +b, x, type=3)` against `olver_Q(nu, b, x)`, and the right-hand side is
        -1/(Gamma(nu-b+1)(x^2-1)). This is the check on the type-3 call AND on the
        e^{+i b pi}/Gamma(nu-b+1) conversion recon section 3.2 prescribes: drop the conversion and
        the Wronskian is out by the phase and the Gamma.

        Recon section 3.2 records misremembering the right-hand side as -1/(x^2-1) and getting
        -1/2 at nu = b+2; the equation is used here exactly as DLMF states it.
        """
        for b in GENERIC_B:
            for j in (0, 2):
                nu = b + j
                for x in (1.05, 1.8, 40.0, 150.0):
                    with self.subTest(b=b, nu=nu, x=x):
                        with mp.workdps(MPMATH_DPS):
                            P = lambda t: mp.legenp(nu, b, t, type=3)
                            Q = lambda t: mp.re(
                                mp.expjpi(b)
                                * mp.legenq(nu, -b, t, type=3)
                                * mp.rgamma(j + 1)
                            )
                            # anchored to the module's helper, as in the Ferrers Wronskian
                            self.assertEqual(olver_Q(nu, b, x), float(Q(x)))
                            W = P(x) * mp.diff(Q, x) - mp.diff(P, x) * Q(x)
                            expected = -mp.rgamma(j + 1) / (mp.mpf(x) ** 2 - 1)
                            self.assertLess(
                                float(abs(W - expected) / abs(expected)),
                                1e-14,
                                f"DLMF 14.2.8 at b={b}, nu={nu}, x={x}: "
                                f"W = {mp.nstr(W, 20)}, expected {mp.nstr(expected, 20)}",
                            )

    def test_ferrers_P_closed_forms(self):
        """
        Recon section 3.5: at mu + nu in {0, 2} both Ferrers P are elementary,

            P^{-b}_b(y)   = (1-y^2)^{b/2} / (2^b Gamma(1+b)),        [DLMF 14.5.18]
            P^{-b}_{b+2}(y) = that x [1 - (2b+3)(1-y^2)/(2(b+1))],

        from Euler's 2F1(a,b;a;z) and one Euler transformation. So the cos coefficient A needs no
        special function at all, which is why A_coefficient calls nothing. An independent check on
        `legenp(nu, -b, y, type=2)`, and the one A is actually built from.
        """
        for b in GENERIC_B:
            for y in (-0.95, -0.4, 0.1, 0.85):
                with self.subTest(b=b, y=y):
                    base = (1.0 - y * y) ** (b / 2.0) / (2.0**b * math.gamma(1.0 + b))
                    self.assertLess(rel(ferrers_P(b, b, y), base), 1e-14)
                    self.assertLess(
                        rel(
                            ferrers_P(b + 2.0, b, y),
                            base
                            * (
                                1.0
                                - (2.0 * b + 3.0) / (2.0 * (b + 1.0)) * (1.0 - y * y)
                            ),
                        ),
                        1e-13,
                    )

    def test_parity_at_mu_plus_nu_zero_or_two(self):
        """
        1912.tex:772-776: for mu + nu in {0, 2} the general reflection formulas reduce to
        P^mu_nu(-x) = P^mu_nu(x) and Q^mu_nu(-x) = -Q^mu_nu(x). Both of this module's nu satisfy
        mu + nu = -b + b = 0 and -b + b + 2 = 2, so this holds for every b -- an identity that a
        wrong order/degree assignment in the mpmath call would not satisfy. It is also what makes
        the b = 0 reduction of recon section 7 work on both branches at once.

        `ferrers_P` and `ferrers_Q` return doubles, so the identity cannot be checked below
        ~1e-16 through them; it holds to 1.1e-14 (at b = 0.9, nu = 2.9, y = 0.97, where Q is
        small) and the bound is 1e-13.
        """
        for b in GENERIC_B:
            for nu in (b, b + 2.0):
                for y in (0.2, 0.63, 0.97):
                    with self.subTest(b=b, nu=nu, y=y):
                        self.assertLess(
                            rel(ferrers_P(nu, b, -y), ferrers_P(nu, b, y)), 1e-13
                        )
                        self.assertLess(
                            rel(ferrers_Q(nu, b, -y), -ferrers_Q(nu, b, y)), 1e-13
                        )

    def test_special_values_at_b_zero_and_half(self):
        """
        1912.tex:804-822, the papers' OWN reduced forms, at the three b where recon section 3.4
        says the general route must not be used: b = 0 (integer order, where the Q definition is
        0/0) and b = +-1/2 (where mpmath 1.3.0's legenp/legenq type=2 raise
        "hypsum() failed to converge" on a degenerate 2F1). Written in mu = -b and
        nu in {b, b+2}:

            b = 0     P^0_0 = 1,  P^0_2 = (3y^2-1)/2,  Q^0_0 = (1/2) log((1+y)/(1-y)),
                      Q^0_2 = (1/4)(3y^2-1) log((1+y)/(1-y)) - 3y/2,
                      calQ^0_0 = (1/2) log((x+1)/(x-1)),
                      calQ^0_2 = (1/8)(3x^2-1) log((x+1)/(x-1)) - 3x/4;
            b = -1/2  P^{1/2}_{-1/2} = sqrt(2/pi)(1-y^2)^{-1/4},
                      P^{1/2}_{3/2} = sqrt(2/pi)(2y^2-1)(1-y^2)^{-1/4},
                      Q^{1/2}_{-1/2} = 0,  Q^{1/2}_{3/2} = -sqrt(2 pi) y (1-y^2)^{1/4};
            b = +1/2  P^{-1/2}_{1/2} = sqrt(2/pi)(1-y^2)^{1/4},
                      P^{-1/2}_{5/2} = (1/3) sqrt(2/pi)(4y^2-1)(1-y^2)^{1/4},
                      Q^{-1/2}_{1/2} = sqrt(pi/2) y (1-y^2)^{-1/4},
                      Q^{-1/2}_{5/2} = (1/3) sqrt(pi/2) y (4y^2-3)(1-y^2)^{-1/4},
                      calQ^{-1/2}_{1/2} = sqrt(pi/2)(x - sqrt(x^2-1))(x^2-1)^{-1/4},
                      calQ^{-1/2}_{5/2} = (1/6) sqrt(pi/2)(4x^3-3x+(1-4x^2)sqrt(x^2-1))
                                              (x^2-1)^{-1/4}.

        These are checked against A, B and C -- the objects the module actually ships -- not
        against the raw calls, precisely because the raw calls are the ones that fail there.
        A, B and C therefore have an independent reference at all three, and B's b = 0 branch is
        pinned by the first block.

        THE REFERENCES ARE EVALUATED IN mpmath, NOT IN DOUBLE, and that is not fussiness.
        Several of these reduced forms are the ill-conditioned way to write their function:
        calQ^0_0 + 4 calQ^0_2 = (3yt^2/2) log((yt+1)/(yt-1)) - 3yt cancels by 213 at yt = 6 and
        21597 at yt = 60, and calQ^{-1/2}_{5/2}'s numerator is 846.0 - 846.0 at yt = 6. In
        double they are 1.2e-13 to 2.7e-11 out and it is the REFERENCE that is wrong, not the
        module -- the same lesson as recon section 5.2's off-cut box (see domenech._S_nu).
        """

        # evaluated in mpmath at MPMATH_DPS, not in double. Several of the paper's reduced forms
        # are the ill-conditioned way to write their function: calQ^{-1/2}_{5/2} at yt = 6 is
        # (4x^3 - 3x) + (1 - 4x^2) sqrt(x^2-1) = 846.0 - 846.0, and calQ^0_0 + 4 calQ^0_2 cancels
        # by 213 at yt = 6 and 21597 at yt = 60. In double they lose those digits and it is the
        # REFERENCE that is wrong; at 30 digits they do not, and the comparison is against the
        # module's own double output as it should be.
        def _sq(t):
            return mp.sqrt(mp.mpf(t) ** 2 - 1)

        cases = {
            0.0: (
                lambda y: mp.mpf(1),
                lambda y: (3 * mp.mpf(y) ** 2 - 1) / 2,
                lambda y: mp.log((1 + mp.mpf(y)) / (1 - mp.mpf(y))) / 2,
                lambda y: (3 * mp.mpf(y) ** 2 - 1)
                * mp.log((1 + mp.mpf(y)) / (1 - mp.mpf(y)))
                / 4
                - 3 * mp.mpf(y) / 2,
                lambda t: mp.log((mp.mpf(t) + 1) / (mp.mpf(t) - 1)) / 2,
                lambda t: (3 * mp.mpf(t) ** 2 - 1)
                * mp.log((mp.mpf(t) + 1) / (mp.mpf(t) - 1))
                / 8
                - 3 * mp.mpf(t) / 4,
            ),
            -0.5: (
                lambda y: mp.sqrt(2 / mp.pi) * (1 - mp.mpf(y) ** 2) ** mp.mpf("-0.25"),
                lambda y: mp.sqrt(2 / mp.pi)
                * (2 * mp.mpf(y) ** 2 - 1)
                * (1 - mp.mpf(y) ** 2) ** mp.mpf("-0.25"),
                lambda y: mp.mpf(0),
                lambda y: -mp.sqrt(2 * mp.pi)
                * mp.mpf(y)
                * (1 - mp.mpf(y) ** 2) ** mp.mpf("0.25"),
                None,
                None,
            ),
            0.5: (
                lambda y: mp.sqrt(2 / mp.pi) * (1 - mp.mpf(y) ** 2) ** mp.mpf("0.25"),
                lambda y: mp.sqrt(2 / mp.pi)
                * (4 * mp.mpf(y) ** 2 - 1)
                * (1 - mp.mpf(y) ** 2) ** mp.mpf("0.25")
                / 3,
                lambda y: mp.sqrt(mp.pi / 2)
                * mp.mpf(y)
                * (1 - mp.mpf(y) ** 2) ** mp.mpf("-0.25"),
                lambda y: mp.sqrt(mp.pi / 2)
                * mp.mpf(y)
                * (4 * mp.mpf(y) ** 2 - 3)
                * (1 - mp.mpf(y) ** 2) ** mp.mpf("-0.25")
                / 3,
                lambda t: mp.sqrt(mp.pi / 2)
                * (mp.mpf(t) - _sq(t))
                * (mp.mpf(t) ** 2 - 1) ** mp.mpf("-0.25"),
                lambda t: mp.sqrt(mp.pi / 2)
                * (
                    4 * mp.mpf(t) ** 3
                    - 3 * mp.mpf(t)
                    + (1 - 4 * mp.mpf(t) ** 2) * _sq(t)
                )
                * (mp.mpf(t) ** 2 - 1) ** mp.mpf("-0.25")
                / 6,
            ),
        }

        for b, (P0, P2, Q0, Q2, cQ0, cQ2) in cases.items():
            rho = rho_of_b(b)
            with mp.workdps(MPMATH_DPS):
                for y in (-0.7, -0.25, 0.45, 0.9):
                    opy, omy = 1.0 + y, 1.0 - y
                    factor = abs(1.0 - y * y) ** (mp.mpf(b) / 2)
                    with self.subTest(b=b, y=y, coefficient="A"):
                        self.assertLess(
                            rel(
                                A_coefficient(opy, omy, b),
                                float(factor * (P0(y) + rho * P2(y))),
                            ),
                            1e-13,
                        )
                    with self.subTest(b=b, y=y, coefficient="B"):
                        self.assertLess(
                            rel(
                                B_coefficient(opy, omy, b),
                                float(factor * (Q0(y) + rho * Q2(y))),
                            ),
                            1e-13,
                        )
                if cQ0 is None:
                    # Q^{1/2}_{-1/2} = 0 and the paper prints no calQ at b = -1/2; the off-cut
                    # branch there is covered by test_C_against_raw_olver_Q at b = -0.45.
                    continue
                for t in (1.3, 2.5, 60.0):
                    with self.subTest(b=b, y_tilde=t, coefficient="C"):
                        self.assertLess(
                            rel(
                                C_coefficient(-(t - 1.0), t + 1.0, b),
                                float(
                                    (mp.mpf(t) ** 2 - 1) ** (mp.mpf(b) / 2)
                                    * (cQ0(t) + 2 * rho * cQ2(t))
                                ),
                            ),
                            1e-13,
                        )

    def test_olver_phase_conversion(self):
        """
        Recon section 3.2's two warnings about `legenq(nu, -b, yt, type=3)`, both of which produce
        a plausible wrong answer if ignored.

        1. The raw value is COMPLEX for non-integer b, with |Im| / |value| = |sin(b pi)| -- 0.59
           at b = 0.2, 1.00 at b = +-1/2. After the phase e^{-mu pi i} = e^{+i b pi} the residual
           imaginary part is at the working precision. `abs()` would give the modulus and lose
           the sign, which is what the off-cut branch carries.
        2. It must be called at yt = -y, never at y. At real argument < -1 the type-3 functions
           sit on the far side of the cut [-1, 1] and carry its phase: they are NOT the papers'
           calQ there, and the residual imaginary part after the conversion does not vanish.
        """
        for b in (0.2, 0.37, 0.8, -0.3):
            for nu in (b, b + 2.0):
                for t in (1.4, 3.0, 90.0):
                    with self.subTest(b=b, nu=nu, y_tilde=t):
                        with mp.workdps(MPMATH_DPS):
                            raw = mp.legenq(nu, -b, t, type=3)
                            self.assertLess(
                                abs(
                                    float(abs(mp.im(raw)) / abs(raw))
                                    - abs(math.sin(b * math.pi))
                                ),
                                1e-12,
                                "the raw type-3 value's imaginary part should be "
                                "|sin(b pi)| of its modulus",
                            )
                            turned = mp.expjpi(b) * raw * mp.rgamma(nu - b + 1)
                            self.assertLess(
                                float(abs(mp.im(turned)) / abs(turned)),
                                1e-25,
                                "the phase e^{+i b pi} should leave a real number",
                            )
                            self.assertEqual(olver_Q(nu, b, t), float(mp.re(turned)))
                            # across the cut the same conversion leaves a large imaginary part
                            wrong = (
                                mp.expjpi(b)
                                * mp.legenq(nu, -b, -t, type=3)
                                * mp.rgamma(nu - b + 1)
                            )
                            self.assertGreater(
                                float(abs(mp.im(wrong)) / abs(wrong)),
                                1e-3,
                                "calling type=3 at y < -1 instead of at yt = -y should not "
                                "give a real number",
                            )


# =============================================================================================
# Test 2 -- A, B and C against the raw calls
# =============================================================================================


class TestCoefficients(unittest.TestCase):
    """
    Test 2. A, B and C as recon section 9 gives them -- elementary for A, the regularised
    section 5.2 form for B, the 2020 paper's 1/yt^2 form for C -- against the six raw mpmath
    calls of recon section 3.2, away from y = +-1, at mp.dps = 30.
    """

    def test_A_against_raw_ferrers_P(self):
        for b in GENERIC_B:
            rho = rho_of_b(b)
            for y in (-0.999, -0.5, 0.4, 0.99):
                with self.subTest(b=b, y=y):
                    raw = (1.0 - y * y) ** (b / 2.0) * (
                        ferrers_P(b, b, y) + rho * ferrers_P(b + 2.0, b, y)
                    )
                    self.assertLess(
                        rel(A_coefficient(1.0 + y, 1.0 - y, b), raw),
                        1e-12,
                        f"A at b={b}, y={y}: closed form against legenp type=2",
                    )

    def test_B_against_raw_ferrers_Q(self):
        """
        B = R_b + rho R_{b+2} with R_nu recon section 5.2's regularised form, against
        |1-y^2|^{b/2} (Q^{-b}_b + rho Q^{-b}_{b+2}) from `legenq(nu, -b, y, type=2)`. The
        regularised form exists so that B is finite AT y = -1 for b > 0, where legenq returns
        nan; here the two are compared where legenq works.
        """
        for b in GENERIC_B:
            rho = rho_of_b(b)
            for y in (-0.999, -0.5, 0.4, 0.99):
                with self.subTest(b=b, y=y):
                    raw = (1.0 - y * y) ** (b / 2.0) * (
                        ferrers_Q(b, b, y) + rho * ferrers_Q(b + 2.0, b, y)
                    )
                    self.assertLess(
                        rel(B_coefficient(1.0 + y, 1.0 - y, b), raw),
                        1e-12,
                        f"B at b={b}, y={y}: section 5.2 form against legenq type=2",
                    )

    def test_C_against_raw_olver_Q(self):
        """
        C = S_b + 2 rho S_{b+2} against (yt^2-1)^{b/2} (calQ^{-b}_b + 2 rho calQ^{-b}_{b+2}) from
        the converted `legenq(nu, -b, yt, type=3)`. THE FACTOR 2 IS ASYMMETRIC AND IS TESTED HERE
        AS SUCH: the raw combination carries 2 rho where the on-cut one carries rho, because it
        is Gamma[nu-rho+1] = Gamma[3] in the Gervois-Navelet off-cut formula (recon section 4.2).

        yt = 150 is included deliberately. Recon section 5.2's off-cut box -- a difference of two
        2F1 at argument (1-yt)/2 -- is the same function but cancels 13 orders there and is 8.6e-8
        out in double-represented b, whatever mp.dps is set to; the module uses the 2020 paper's
        own 1/yt^2 form instead (see domenech._S_nu), which agrees with the raw call to rounding
        at every yt. `q-smooth` at b = 0.2 sits at yt = 149.5.
        """
        for b in GENERIC_B:
            rho = rho_of_b(b)
            for t in (1.001, 1.5, 40.0, 150.0):
                with self.subTest(b=b, y_tilde=t):
                    raw = (t * t - 1.0) ** (b / 2.0) * (
                        olver_Q(b, b, t) + 2.0 * rho * olver_Q(b + 2.0, b, t)
                    )
                    self.assertLess(
                        rel(C_coefficient(-(t - 1.0), t + 1.0, b), raw),
                        1e-12,
                        f"C at b={b}, yt={t}: 1/yt^2 form against legenq type=3",
                    )


# =============================================================================================
# Test 3 -- the target object against the exact kernel
# =============================================================================================


def worst_over_a_period(f, g, v, u, X, b, samples=5):
    """
    max |f - g| over one period of the outer oscillation from X, in envelope units. Both objects
    oscillate, so a difference at a single x is dominated by where the zeros fall -- recon
    section 2.4's own table has 1.5e-3 at x = 1600 and 9.9e-6 at 6400, a factor 150 for a factor
    4 in x. The maximum over a period is the envelope of the difference and falls like 1/x.
    """
    worst = 0.0
    for i in range(samples):
        x = X + 2.0 * math.pi * i / samples
        worst = max(worst, abs(f(v, u, x, b) - g(v, u, x, b)) / envelope(v, u, x, b))
    return worst


def _quadrature_value(v, u, x, b):
    return I_quadrature(v, u, x, b)[0]


class TestTargetAgainstExactKernel(unittest.TestCase):
    """
    Test 3. The x -> infinity coefficients (3.3)/(3.4) against the finite-x integrals (4.11),
    inside the same corrected (4.10). The paper states the error in terms -- "corrections from a
    finite upper integration limit ... will be suppressed by a further 1/x" -- so the assertion
    is that trend, which has no tolerance in it, and not a number at one x.
    """

    def test_difference_falls_like_one_over_x(self):
        """
        Recon section 2.4's table is the figure being reproduced (1.6e-2 / 1.5e-3 / 9.9e-6 at
        x = 100/1600/6400 for b = 0.2, (u,v) = (1.3,1.2)); this measures its envelope instead of
        its single-x values. D(X) = max |I_target - I_quadrature| over one period from X, in
        envelope units. Over a factor 4 in x it must fall by about 4, and over 16 by about 16.

        BOTH BRANCHES ARE HERE, and the off-cut rows are the ones that carry recon section 4.2's
        asymmetric factor of 2: with Gamma[1] in place of Gamma[3] on the second off-cut term the
        target does not approach the exact kernel at all -- it stalls at ~8e-2 of the envelope --
        so this trend is the discriminator for it.

        NOT INCLUDED: `q-smooth` (u = 0.01) and the resonance. The O(1/x) is not uniform; its
        control parameter is the smallest Bessel argument c_s min(u,v) x, and near the resonance
        |c_s(u+v)-1| x (recon sections 5.4 and 6.4). On `q-smooth` the target is still 5e-2 of
        the envelope off the exact kernel at x = 3200. That is why test 8 scores N against
        I_quadrature and not against I_target.
        """
        cases = (
            (0.2, 1.2, 1.3),  # on-cut,  y = -0.439
            (0.2, 3.0, 2.5),  # on-cut,  y = +0.717
            (0.5, 3.0, 2.5),  # on-cut,  y = +0.417
            (0.2, 1.091, 0.909),  # off-cut, y = -1.252  (the `together` shape)
            (0.2, 0.9, 0.6),  # off-cut, y = -3.083
        )
        for b, v, u in cases:
            D = [
                worst_over_a_period(I_target, _quadrature_value, v, u, X, b)
                for X in (100.0, 400.0, 1600.0)
            ]
            kin = kinematics(v, u, b)
            label = f"b={b}, (u,v)=({u},{v}), y={kin.y:+.4f}, on_cut={kin.on_cut}"
            for lo, hi, step in ((D[0], D[1], "100->400"), (D[1], D[2], "400->1600")):
                with self.subTest(case=label, decade=step):
                    self.assertTrue(
                        0.08 <= hi / lo <= 0.45,
                        f"{label}: max|target - exact|/envelope fell {lo:.3e} -> {hi:.3e} "
                        f"over {step}, a factor {hi / lo:.3f}; 1/x is 0.25",
                    )
            with self.subTest(case=label, decade="100->1600"):
                self.assertTrue(
                    0.01 <= D[2] / D[0] <= 0.11,
                    f"{label}: {D[0]:.3e} -> {D[2]:.3e} over 16x in x, a factor "
                    f"{D[2] / D[0]:.3f}; 1/x is 0.0625",
                )


# =============================================================================================
# Test 4 -- the doubly asymptotic form
# =============================================================================================


class TestAsymptoticForm(unittest.TestCase):
    """
    Test 4. I_asymptotic is I_target with the outer Bessels expanded, so they differ by the
    (4 nu^2 - 1)/(8x) Bessel correction and nothing else.
    """

    def test_asymptotic_approaches_target_like_one_over_x(self):
        """
        The same statistic as test 3, against I_target rather than the exact kernel, so it
        isolates the Bessel expansion: 2e-3 at x = 50 falling to 1e-6 at 3200 for b = 0.2
        (recon section 6.4). A sign error in the cos term -- which is exactly what the review's
        (4.12) as printed has -- makes this O(1) at every x instead.
        """
        for b, v, u in ((0.2, 1.2, 1.3), (0.2, 1.091, 0.909), (0.5, 3.0, 2.5)):
            D = [
                worst_over_a_period(I_asymptotic, I_target, v, u, X, b, samples=8)
                for X in (100.0, 400.0, 1600.0, 6400.0)
            ]
            for i, step in enumerate(("100->400", "400->1600", "1600->6400")):
                with self.subTest(b=b, u=u, v=v, decade=step):
                    self.assertTrue(
                        0.18 <= D[i + 1] / D[i] <= 0.34,
                        f"b={b}, (u,v)=({u},{v}): max|(4.12) - target|/envelope fell "
                        f"{D[i]:.3e} -> {D[i + 1]:.3e} over {step}, a factor "
                        f"{D[i + 1] / D[i]:.3f}; 1/x is 0.25",
                    )

    def test_asymptotic_equals_target_at_b_zero(self):
        """
        At b = 0 the outer Bessels are of order 1/2, where the large-argument asymptotics are
        EXACT: J_{1/2}(x) = sqrt(2/(pi x)) sin x and Y_{1/2}(x) = -sqrt(2/(pi x)) cos x. So the
        target object and the corrected (4.12) coincide there identically, and the distinction
        campaign README section 2 (i) draws between them is real only for b != 0.
        """
        for v, u, x in KT_TIE_POINTS:
            with self.subTest(v=v, u=u, x=x):
                self.assertLess(
                    rel(I_target(v, u, x, 0.0), I_asymptotic(v, u, x, 0.0)),
                    32.0 * EPS,
                )


# =============================================================================================
# Test 5 -- the b = 0 reduction: the load-bearing external tie
# =============================================================================================


class TestKohriTeradaTie(unittest.TestCase):
    """
    Test 5. `ComputeTargets/tests/kohri_terada.py` is the one object in this module's tests that
    this campaign did not write: a transcription of Kohri & Terada (arXiv:1804.08577) made by
    `prompts/radiation-oracle`, whose eq. (22) is a closed form in Si, Ci, sines and logarithms
    with no quadrature and no Legendre function in it anywhere.

    The bridging constant is DERIVED, not fitted (recon sections 7 and 8): the 2020 paper folds
    2 c^2 = 2((2+b)/(3+2b))^2 into its definition of I and the review does not, so
    I_rev = I_KT / (2c^2), which at b = 0 is exactly 9/8.
    """

    def test_quadrature_is_nine_eighths_of_eq22(self):
        """
        I_quadrature(b=0) = (9/8) eq. (22), at recon section 7's own six (u, v, x). This is the
        check that campaign README section 5.1 requires and the strongest one here: it scores the
        WHOLE chain -- the weight xbar^{1/2-b}, the source's rho, the corrected sign of (4.10),
        the prefactor N -- against a paper this module does not transcribe.

        The bound is the two references' own declared errors: quad's estimate on the two (4.11)
        integrals, and eps times eq. (22)'s rounding scale. Recon section 7 measured
        <= 4.5e-15 on five of the six points and 3.9e-10 at u = 0.01, where eq. (22)'s
        1/(u^3 v^3) prefactor multiplies terms that nearly cancel (KOHRI-TERADA-ORACLE.md
        section 7.2). Both figures are reproduced here: 4.0e-15 and 3.9e-10.

        THE `T-first` POINT IS INCLUDED. It has c_s|u - v| = 1.155 and y = 1.0042, where
        I_target is undefined -- but I_quadrature never forms y, so the exact kernel is perfectly
        well defined on a shape that is not a closable triangle.
        """
        for v, u, x in KT_TIE_POINTS + (KT_TIE_T_FIRST,):
            with self.subTest(v=v, u=u, x=x):
                quad_value, quad_err = I_quadrature(v, u, x, 0.0)
                closed, scale = I_RD_with_rounding_scale(v, u, x)
                predicted = 9.0 / 8.0 * closed
                bound = quad_err + 9.0 / 8.0 * EPS * scale
                self.assertLessEqual(
                    bound,
                    1e-8 * abs(predicted),
                    "a bound this loose would make the comparison vacuous",
                )
                self.assertLessEqual(
                    abs(quad_value - predicted),
                    bound,
                    f"(4.11) by quadrature = {quad_value:+.15e}, (9/8) eq. (22) = "
                    f"{predicted:+.15e}, |diff| {abs(quad_value - predicted):.2e} "
                    f"({rel(quad_value, predicted):.2e} relative) against quad's own error "
                    f"{quad_err:.2e} + eq. (22)'s rounding {9.0 / 8.0 * EPS * scale:.2e}",
                )

    def test_target_is_nine_eighths_of_eq25(self):
        """
        I_target(b=0) = (9/8) eq. (25), EXACTLY -- not to O(1/x). Both are the same object at
        b = 0: the order-1/2 Bessel asymptotics are exact, and recon section 7 writes out the
        reduction term by term, A -> 3y^2, B Theta_+ - C Theta_- -> (3y/2)(y log|(1+y)/(1-y)| - 2)
        on BOTH branches, with y = d/(2uv) in Kohri & Terada's d = u^2 + v^2 - 3. So a
        disagreement here is a transcription error, not a truncation.

        Two of the five points are on-cut and two off-cut, and one, (0.01, 1.1), has y = -81.4:
        the b = 0 limit of C is exercised far from the resonance as well as near it.

        The bound is eq. (25)'s own rounding, computed by eq25_rounding_scale to the same recipe
        kohri_terada uses for eq. (22), plus 8 ulp for I_target's own arithmetic. At u = 0.01
        eq. (25)'s sin bracket cancels 3 orders (-4uv = -4.4e-02 against d L = +4.4e-02) and the
        bound is 2e-11 relative; elsewhere it is ~1e-15.
        """
        for v, u, x in KT_TIE_POINTS:
            with self.subTest(v=v, u=u, x=x):
                target = I_target(v, u, x, 0.0)
                predicted = 9.0 / 8.0 * I_RD_asymptotic(v, u, x)
                bound = 9.0 / 8.0 * eq25_rounding_scale(v, u, x) + 8.0 * EPS * abs(
                    target
                )
                self.assertLessEqual(
                    bound,
                    1e-9 * abs(predicted),
                    "a bound this loose would make the comparison vacuous",
                )
                self.assertLessEqual(
                    abs(target - predicted),
                    bound,
                    f"I_target(b=0) = {target:+.15e}, (9/8) eq. (25) = {predicted:+.15e}, "
                    f"|diff| {abs(target - predicted):.2e} "
                    f"({rel(target, predicted):.2e} relative) against eq. (25)'s own "
                    f"rounding {9.0 / 8.0 * eq25_rounding_scale(v, u, x):.2e}",
                )

    def test_refuses_the_T_first_shape_at_b_zero(self):
        """
        `T-first` is (q, r) = (1e4, 1.2e4) against k = 1e3, so |q - r| = 2000 > k: not a closable
        triangle, and at b = 0 it has c_s|u - v| = 1.155 > 1, i.e. y = +1.0042. That is outside
        BOTH Gervois-Navelet cases either paper quotes, and the module refuses it rather than
        continuing into a region no source covers.

        Recon section 6.2 offers a continuation inferred numerically, and recon section 10 item 2
        records that it was not read from a source. This module declines it, for three reasons
        recorded on OutsideGervoisNavelet: campaign README section 5.1 forbids shipping a
        transcription that has not been scored against a source; there is nothing to score it
        against where it would be used, because Kohri & Terada's eq. (25) is equally inapplicable
        for |v - u| > sqrt3 (KOHRI-TERADA-ORACLE.md section 8 item 4); and eq. (22), which is
        exact and finite there, already covers that shape.

        At b = 0.2 the same shape has c_s|u - v| = 0.943 and y = +0.998, is on-cut, and is
        perfectly well defined -- which is why test 8 covers all nine fixture cases.
        """
        v, u, x = KT_TIE_T_FIRST
        self.assertGreaterEqual(kinematics(v, u, 0.0).y, 1.0)
        with self.assertRaises(OutsideGervoisNavelet):
            I_target(v, u, x, 0.0)
        with self.assertRaises(OutsideGervoisNavelet):
            I_asymptotic(v, u, x, 0.0)
        # ... and at b = 0.2 it is on-cut and finite
        kin = kinematics(v, u, 0.2)
        self.assertTrue(kin.on_cut and kin.one_minus_y > 0.0)
        self.assertTrue(math.isfinite(I_target(v, u, x, 0.2)))


# =============================================================================================
# Test 6 -- the small-x limit
# =============================================================================================


class TestSmallX(unittest.TestCase):
    """Test 6. I_rev(x << 1) -> +x^2/(2(2+b)), which at b = 0 is x^2/4 = (9/8)(2 x^2/9)."""

    def test_small_x_limit(self):
        """
        THE CHEAPEST TEST OF THE SIGN. G > 0 and f > 0 just after the source, so I_rev is POSITIVE
        at small x; the review's (4.10) as printed is negative there. Recon section 4.1 derives
        the constant from G_x -> xbar[1 - (xbar/x)^{1+2b}]/(1+2b), whose integral is
        x^2/(2(3+2b)), times f(0) = (3+2b)/(2+b).

        Asserted as a TREND, with no tolerance on the constant: (1 - I/(x^2/(2(2+b))))/x^2 must
        settle to a b-dependent constant alpha(b) as x -> 0, which pins the leading term AND the
        O(x^2) shape of the approach. Measured: 0.0783 at b = 0, 0.0638 at b = 0.2, 0.1166 at
        b = -0.3, each stable to 3e-5 relative between x = 0.1 and x = 0.01.

        AT b = 0 alpha IS DERIVED, from the same expansion Kohri & Terada's oracle asserts
        (test_kohri_terada_oracle's test 2): Phi(y) = 1 - y^2/30 + O(y^4) gives
        alpha = 1/20 + (u^2+v^2)/60. It is checked against that here, which ties the b = 0 end of
        the trend to a number rather than to itself.
        """
        for b in (0.0, 0.2, 0.5, -0.3, 0.8):
            for v, u in ((1.1, 0.7), (0.5, 1.2)):
                alphas = []
                for x in (0.3, 0.1, 0.03, 0.01):
                    # epsabs = 0: the module's 1e-16 default is an ABSOLUTE floor, and here
                    # I is 2.3e-05 at x = 0.01, so that floor would let quad stop with a
                    # declared error of 5.5e-09 relative -- on a value whose measured accuracy
                    # is 1e-13. Only the declared error moves; alpha below is identical either
                    # way to six figures.
                    value, err = I_quadrature(v, u, x, b, epsabs=0.0, epsrel=1e-13)
                    leading = small_x_limit(x, b)
                    with self.subTest(b=b, v=v, u=u, x=x):
                        self.assertGreater(
                            value,
                            0.0,
                            "I_rev is positive at small x; (4.10) as printed is not",
                        )
                        self.assertLess(err, 1e-11 * value)
                        self.assertLess(
                            abs(value / leading - 1.0),
                            0.2 * x * x,
                            f"I/(x^2/(2(2+b))) = {value / leading:.9f} at x = {x}",
                        )
                    alphas.append((1.0 - value / leading) / (x * x))
                with self.subTest(b=b, v=v, u=u, statistic="alpha settles"):
                    self.assertLess(
                        abs(alphas[-1] - alphas[-2]) / abs(alphas[-1]),
                        1e-3,
                        f"b={b}, (u,v)=({u},{v}): (1 - I/leading)/x^2 = "
                        + ", ".join(f"{a:.6f}" for a in alphas)
                        + " at x = 0.3, 0.1, 0.03, 0.01; it must settle",
                    )
                if b == 0.0:
                    with self.subTest(b=b, v=v, u=u, statistic="alpha at b=0"):
                        self.assertLess(
                            abs(alphas[-1] - (1.0 / 20.0 + (u * u + v * v) / 60.0)),
                            1e-4,
                            f"alpha = {alphas[-1]:.6f} against the derived "
                            f"1/20 + (u^2+v^2)/60 = "
                            f"{1.0 / 20.0 + (u * u + v * v) / 60.0:.6f}",
                        )


# =============================================================================================
# Test 7 -- the resonance
# =============================================================================================


class TestResonance(unittest.TestCase):
    """
    Test 7. At c_s(u+v) = 1 the kinematic variable y is -1 exactly and the x -> infinity
    COEFFICIENTS are singular: power-divergent as (1+y)^{-|b|} for b < 0, logarithmic at b = 0,
    finite for b > 0 (recon section 5.1). The finite-x kernel is regular there for every b.
    No fixture case is near the resonance (recon section 5.4); this is a limitation of the
    instrument, recorded as campaign README section 7 D6.
    """

    @staticmethod
    def _resonant_pair(b, delta=0.0):
        """u = v with c_s(u+v) - 1 = delta exactly."""
        return (1.0 + delta) / (2.0 * cs_of_b(b))

    def test_coefficients_at_the_resonance(self):
        """
        Recon section 5.2's closed form, evaluable AT y = -1 where mpmath's legenq returns nan
        and legenp returns -inf:

            B(-1) = -2^b Gamma(b)(3+2b)(1+b+b^2) / ((1+b) Gamma(2b+3)) = -C(+1),

        and A(-1) = 0 as (1+y)^b. B(-1) = -C(1) is what makes the sin coefficient
        B Theta_+ - C Theta_- continuous across the resonance while the cos coefficient vanishes,
        so the target object is continuous in (u, v) there for b > 0. -6.2147807897686 at
        b = 0.2, which recon section 5.2 quotes to 14 digits.

        Recon section 5.1's divergence structure is asserted too, in the only form it takes: at
        b = 0, A is finite (3) and B diverges logarithmically; for b < 0 both diverge.
        """
        for b in (0.2, 0.5, 0.8):
            with self.subTest(b=b):
                self.assertEqual(A_coefficient(0.0, 2.0, b), 0.0)
                self.assertLess(
                    rel(B_coefficient(0.0, 2.0, b), B_at_resonance(b)), 1e-14
                )
                # C at yt = 1 is reached as a limit: the point itself is taken on-cut
                self.assertLess(
                    rel(C_coefficient(-1e-300, 2.0, b), -B_at_resonance(b)), 1e-14
                )
        self.assertLess(
            abs(B_at_resonance(0.2) + 6.2147807897686),
            5e-14,
            "recon section 5.2's figure",
        )

        # b = 0: A -> 3 (finite and discontinuous across the resonance, which is the Theta in
        # Kohri & Terada's eq. (25)), B -> -infinity logarithmically
        self.assertLess(abs(A_coefficient(0.0, 2.0, 0.0) - 3.0), 4.0 * EPS)
        self.assertEqual(B_coefficient(0.0, 2.0, 0.0), -math.inf)
        # b < 0: both diverge, as (1+y)^{-|b|}
        self.assertEqual(A_coefficient(0.0, 2.0, -0.3), math.inf)
        self.assertFalse(math.isfinite(B_coefficient(0.0, 2.0, -0.3)))

    def test_target_one_sided_limits_agree_like_delta_to_the_b(self):
        """
        For b > 0 the target object is continuous at the resonance, and the rate is fixed by the
        coefficients: A ~ (1+y)^b and B - B(-1) ~ (1+y)^b, so the gap between I_target at
        c_s(u+v) = 1 +- delta and its value AT the resonance falls like |delta|^b, from BOTH
        sides. The assertion is that exponent -- a factor 100^{-b} per two decades of delta,
        0.398 at b = 0.2, 0.100 at b = 0.5, 0.025 at b = 0.8 -- which has no tolerance in it and
        which a wrong regularisation would not reproduce.
        """
        for b in (0.2, 0.5, 0.8):
            uv0 = self._resonant_pair(b)
            at_resonance = I_target(uv0, uv0, 1000.0, b)
            self.assertTrue(math.isfinite(at_resonance))
            for sign, side in ((+1.0, "on-cut"), (-1.0, "off-cut")):
                gaps = []
                for eps in (1e-6, 1e-8, 1e-10):
                    uv = self._resonant_pair(b, sign * eps)
                    gaps.append(abs(I_target(uv, uv, 1000.0, b) - at_resonance))
                for i, eps in enumerate((1e-8, 1e-10)):
                    with self.subTest(b=b, side=side, delta=eps):
                        ratio = gaps[i + 1] / gaps[i]
                        self.assertLess(
                            abs(ratio / 100.0**-b - 1.0),
                            0.12,
                            f"b={b}, {side}: |I_target(delta) - I_target(0)| fell "
                            f"{gaps[i]:.3e} -> {gaps[i + 1]:.3e} over two decades of delta, "
                            f"a factor {ratio:.4f}; |delta|^b is {100.0**-b:.4f}",
                        )

    def test_exact_kernel_is_regular_at_the_resonance(self):
        """
        The divergence of recon section 5.1 is a property of the x -> infinity COEFFICIENTS. The
        finite-x kernel is an integral of bounded functions over a finite range and is finite at
        the resonance for every b -- so I_quadrature is the only one of the three constructions
        that can score it, which is campaign README section 7 D6's whole subject.

        Two assertions. The exact kernel is finite ON the resonance with a declared error no
        worse than off it, and varies smoothly through it (no kink: a divergence would show as
        one). And at b = 0.8, where the coefficients are finite there, the target converges onto
        the exact kernel at the resonance in the ordinary 1/x way -- 4.3e-2 -> 1.4e-2 -> 4.5e-3
        over x = 400 -> 1600 -> 6400. At b = 0.2 the same sequence is 4.5e-1 -> 3.4e-1 -> 2.5e-1:
        finite, continuous, and a poor oracle, which is why no test scores a fixture case there.
        """
        b = 0.2
        values, errors = [], []
        for delta in (2e-6, 1e-6, 0.0, -1e-6, -2e-6):
            uv = self._resonant_pair(b, delta)
            value, err = I_quadrature(uv, uv, 400.0, b)
            self.assertTrue(math.isfinite(value))
            values.append(value)
            errors.append(err)
        self.assertLess(max(errors), 1e-12 * min(abs(f) for f in values))
        # no kink: on a spacing of 1e-6 in delta the four consecutive first differences agree
        # to 2e-4 of each other (7.290e-07 ... 7.292e-07). A divergence or a kink at delta = 0
        # would make the two straddling it differ by O(1) from the two outside.
        firsts = [values[i] - values[i + 1] for i in range(4)]
        self.assertLess(
            max(firsts) / min(firsts) - 1.0,
            1e-2,
            f"the exact kernel kinks at the resonance: first differences {firsts} of "
            f"{values}",
        )

        b = 0.8
        uv = self._resonant_pair(b)
        D = [
            worst_over_a_period(I_target, _quadrature_value, uv, uv, X, b)
            for X in (400.0, 1600.0, 6400.0)
        ]
        for i, step in enumerate(("400->1600", "1600->6400")):
            with self.subTest(b=b, decade=step):
                self.assertTrue(
                    0.15 <= D[i + 1] / D[i] <= 0.45,
                    f"on the resonance at b = {b}: max|target - exact|/envelope fell "
                    f"{D[i]:.3e} -> {D[i + 1]:.3e} over {step}, a factor "
                    f"{D[i + 1] / D[i]:.3f}; 1/x is 0.25",
                )


# =============================================================================================
# Test 8 -- the tie to the pipeline
# =============================================================================================


class TestNormalisation(unittest.TestCase):
    """Test 8. N over the nine b = 0.2 fixture cases of test_quadsource_integral."""

    def test_N_is_constant_over_the_nine_b_02_fixture_cases(self):
        """
        THE ONLY TEST HERE WITH AN OBJECT IN IT THAT NOBODY TRANSCRIBING A PAPER WROTE, and the
        one that closes `[01-general-w-normalisation-is-predicted-not-measured]`. On all nine
        b = 0.2 cases of test_quadsource_integral (three SHAPES x three X_RESP_VALUES, exact
        flavour, at the reference pair (1e-45, 1e-12)), evaluate_QuadSource_integral's `total`
        must satisfy

            total = -I_rev_trunc / k_phys^2,

        I_rev_trunc being (4.11) integrated between the code's OWN limits with the outer
        x = k tau(z_resp) in the kernel throughout. Equivalently, in the Kohri-Terada convention
        I_KT = 2 c^2 I_rev,

            N(b) = k_phys^2 total / I_KT = -(3+2b)^2/(2(2+b)^2) = -1.194214876... at b = 0.2.

        SCORED AGAINST I_quadrature, NOT I_target, AND THIS IS THE POINT. The target object's
        O(1/x) is not uniform: on `q-smooth` (u = 0.01) it is still 5e-2 of the envelope off the
        exact kernel at x = 3200, and the fixtures sit at x = 5.3 to 1.9e3. Scoring N against it
        would show a drift of a few per cent that looks like a pipeline defect and is not one
        (recon section 6.4; campaign README section 2 (i)).

        THE LOAD-BEARING STATISTIC IS CONSTANCY, and that is what the first assertion is. KT
        section 0's provenance argument: a wrong kernel, measure or Green's function would make N
        DRIFT with x -- here over x from 5.3 to 1.9e3, u from 0.01 to 10, and both
        Gervois-Navelet branches -- while a wrong normalisation is constant but wrong, which is
        the benign failure. The second assertion pins the value against recon section 8's
        derivation; if the first passes and the second fails, that is prompt 01 section 6's
        second stop condition (a statement about step 1 or step 3 of that derivation), not a
        tolerance to loosen.

        THE HEAD IS COMPUTED. Both papers integrate from xbar = 0 and the code from
        z_source_max; `head_over_I` is 1.2e-6 to 2.0e-5 on these cases and does not shrink with x
        (campaign README section 2 (k)). It is reported rather than subtracted, because
        I_rev_trunc is integrated between the code's own limits directly.

        THE BOUND is the two sides' own declared errors: quad's estimate on the two (4.11)
        integrals, and the pipeline's `total_abserr` or the requested rtol applied to
        |numeric_quad| + |WKB_Levin| where that is larger, because the two halves cancel. It is
        1.0e-12 to 1.5e-11, set by the pipeline except on `q-smooth` at x_resp = 980, where the
        quadrature's own 6.9e-12 dominates. No case's bound may exceed 1e-9.
        """
        b = 0.2
        measured = []
        for shape in SHAPES:
            for x_resp in X_RESP_VALUES:
                case = Case(b=b, shape=shape, x_resp=x_resp, exact=True)
                out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
                total = float(out["total"])
                halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
                pipeline_err = max(float(out["total_abserr"]), REF_RTOL * halves)

                tau = case.model.functions.tau
                oracle = total_from_I_rev(
                    shape.k,
                    shape.q,
                    shape.r,
                    tau(case.z_resp),
                    tau(case.z_source_max),
                    b,
                )
                predicted = oracle["total"]
                # N against I_rev is -1 exactly; against the Kohri-Terada convention
                # I_KT = 2c^2 I_rev it is N_rev/(2c^2) = -(3+2b)^2/(2(2+b)^2)
                N_rev = shape.k * shape.k * total / oracle["I_rev_trunc"]
                N = N_rev / two_c_squared(b)
                bound_rel = oracle["total_error"] / abs(predicted) + pipeline_err / abs(
                    total
                )
                with self.subTest(shape=shape.name, x_resp=x_resp):
                    self.assertLessEqual(bound_rel, 1e-9)
                    self.assertLessEqual(
                        abs(N - kt_convention_norm(b)),
                        abs(kt_convention_norm(b)) * bound_rel,
                        f"{case.label()}: N = {N:.15f}, |N - N(b)| = "
                        f"{abs(N - kt_convention_norm(b)):.2e}; total = {total:+.15e}, "
                        f"(4.11) between the code's own limits gives I_rev = "
                        f"{oracle['I_rev_trunc']:+.15e} at x = {oracle['x']:.4e}, "
                        f"head/I = {oracle['head_over_I']:.1e}; oracle's own error "
                        f"{oracle['total_error'] / abs(predicted):.1e}, pipeline's own error "
                        f"{pipeline_err / abs(total):.1e} (relative)",
                    )
                measured.append(N)

        spread = max(measured) - min(measured)
        mean = sum(measured) / len(measured)
        self.assertLess(
            abs(spread / mean),
            1e-11,
            f"N is not constant over the nine cases: min {min(measured):.15f}, max "
            f"{max(measured):.15f}, spread {spread:.3e} about the mean {mean:.15f}. A DRIFT "
            f"IS A FINDING, NOT A TOLERANCE (prompt 01 section 6)",
        )
        self.assertLess(
            abs(mean - kt_convention_norm(b)),
            1e-11,
            f"N is constant at {mean:.15f} but recon section 8 derives "
            f"{kt_convention_norm(b):.15f}; a constant-but-wrong N is a statement about that "
            f"derivation's step 1 or step 3 (prompt 01 section 6)",
        )


# =============================================================================================
# the oracle is fixed by construction
# =============================================================================================


class TestOracleIsFixed(unittest.TestCase):
    """
    Acceptance item 1: a pure function of (v, u, x, b) that moves with no tolerance. The module
    uses mpmath, whose precision is GLOBAL STATE, so this is not free -- it is bought by the
    `mp.workdps(MPMATH_DPS)` in each coefficient.
    """

    def test_answer_does_not_depend_on_the_callers_mp_dps(self):
        saved = mp.mp.dps
        try:
            results = {}
            for dps in (15, 30, 80):
                mp.mp.dps = dps
                results[dps] = (
                    A_coefficient(0.4, 1.6, 0.2),
                    B_coefficient(0.4, 1.6, 0.2),
                    C_coefficient(-0.5, 2.5, 0.2),
                    I_target(1.2, 1.3, 137.0, 0.2),
                    I_asymptotic(1.2, 1.3, 137.0, 0.2),
                )
            self.assertEqual(results[15], results[30])
            self.assertEqual(results[15], results[80])
        finally:
            mp.mp.dps = saved

    def test_two_c_squared_bridges_the_two_conventions(self):
        """
        I_KT = 2 c^2 I_rev with c = (2+b)/(3+2b), so N(b) = -1/(2c^2). At b = 0 that is
        kohri_terada.KT_NORM = -9/8 exactly, which is the arithmetic the b = 0 tie rests on.
        """
        self.assertLess(abs(two_c_squared(0.0) - 8.0 / 9.0), 4.0 * EPS)
        self.assertLess(abs(kt_convention_norm(0.0) + 9.0 / 8.0), 4.0 * EPS)
        self.assertLess(abs(kt_convention_norm(0.2) + 1.194214876033058), 1e-14)


if __name__ == "__main__":
    unittest.main()
