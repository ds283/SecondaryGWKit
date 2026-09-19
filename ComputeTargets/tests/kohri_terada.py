"""
Kohri & Terada's radiation-era closed form for the source time integral, as an exact oracle.

Kazunori Kohri and Takahiro Terada, "Semianalytic calculation of gravitational wave spectrum
nonlinearly induced from primordial curvature perturbations", arXiv:1804.08577,
Phys. Rev. D 97, 123532 (2018). Equation numbers below are theirs. They published no code;
everything here is a transcription of their published equations, productionised from
docs/radiation-oracle/kt_verification.py, whose measurement is
docs/radiation-oracle/KOHRI-TERADA-ORACLE.md.

    I_RD(v, u, x)           KT eq. (22), the closed form -- regular at u + v = sqrt(3)
    I_RD_asymptotic(v,u,x)  KT eq. (25), its x -> infinity limit
    I_RD_quadrature(...)    KT eq. (15), the defining integral, by scipy.quad
    Phi(x), dPhi(x)         KT eq. (19), the radiation-era transfer function, and d/dx
    f_RD(v, u, x)           KT eq. (16) at w = 1/3, the source -- NOT eq. (20), see f_RD
    f_RD_eq20(v, u, x)      KT eq. (20), kept only to test f_RD against; unusable below x ~ 0.1
    total_from_I_RD(...)    the mapping onto this repository's `total`, with N = -9/8

THE PAPER CARRIES THREE ERRATA, each recorded at the point where it bites:

  1. eq. (22): the Ci/Si arguments pair with the DIFFERENCE, not the sum -- see I_RD;
  2. below eq. (24): the small-x limit is 2 x^2 / 9, not the x^2 / 2 the paper states -- see I_RD;
  3. eq. (20) cannot be evaluated below x ~ 0.1 in double precision -- see f_RD and f_RD_eq20.

and eq. (22) as printed is singular at the resonance u + v = sqrt(3), which is removable and is
removed here -- see I_RD. Anyone comparing this file against the PDF will find it disagrees with
the PDF in the first two places. The PDF is wrong there; do not "fix" this file back to it.

WHERE THIS LIVES, AND WHAT IT IS NOT. It is an oracle for tests, not a pipeline component:
nothing in main.py or under ComputeTargets/ outside tests/ calls it, and it covers b = 0 (w = 1/3)
only, because Kohri & Terada give no closed form between pure radiation and pure matter.

WHAT ITS AGREEMENT WITH ITSELF DOES AND DOES NOT PROVE. I_RD and I_RD_quadrature share the Green's
function sin(x - xbar), the measure xbar/x and the source f_RD, so their agreement checks the
transcription of eq. (22) given the kernel and cannot detect a misread kernel. Only the comparison
with this repository's own `total` (total_from_I_RD, and test_kohri_terada_oracle's pipeline test)
has an object in it that this module did not write. KOHRI-TERADA-ORACLE.md section 0.
"""

import math
import sys

from scipy.integrate import quad
from scipy.special import sici

SQRT3 = math.sqrt(3.0)
EULER_GAMMA = 0.57721566490153286061

#: total = KT_NORM * I / k_phys^2 -- measured, not fitted, on the nine b = 0 cases of
#: test_quadsource_integral (KOHRI-TERADA-ORACLE.md sections 1 and 7). It factorises exactly as
#: -(1/2) * (9/4): the 9/4 is 1/c^2 with c = (2+b)/(3+2b) = 2/3 at b = 0
#: (docs/spec/03-source-term.md section 0.1), a constant Kohri & Terada fold into their f and the
#: code strips; the 1/2 is the author's h^us_ij = h^them_ij / 2 relative to Kohri & Terada
#: (docs/spec/05-one-loop.md, open question 11); the sign is the orientation convention of
#: docs/spec/04-source-integral.md section 0 (3). The sign is a convention, not an error.
KT_NORM = -9.0 / 8.0

# w = 1/3: the only equation of state this module covers
_W = 1.0 / 3.0

_EPS = sys.float_info.epsilon

# --------------------------------------------------------------------------------------------
# KT eq. (19), with a series branch
# --------------------------------------------------------------------------------------------

# Below this |x| Phi and dPhi are summed from their Taylor series. The closed forms cancel
# there -- sin(y)/y - cos(y) ~ y^2/3, and dPhi's two terms each ~ 2/x against a result ~ x/15 --
# and at x = 1 the series has converged to rounding in the terms kept.
_PHI_SERIES_CUTOFF = 1.0
_PHI_SERIES_TERMS = 12

# Phi(x) = 9 sum_{n>=1} (-1)^{n+1} 2n x^{2n-2} / (3^n (2n+1)!), which is 9/x^2 times the series
# of sin(y)/y - cos(y) at y = x/sqrt3. Coefficient of x^{2(n-1)}:
_PHI_COEFFS = tuple(
    9.0 * (-1.0) ** (n + 1) * 2.0 * n / (3.0**n * math.factorial(2 * n + 1))
    for n in range(1, _PHI_SERIES_TERMS + 1)
)


def Phi(x: float) -> float:
    """
    KT eq. (19): the radiation-era Newtonian potential transfer function,

        Phi(x) = (9/x^2) [ sin(x/sqrt3)/(x/sqrt3) - cos(x/sqrt3) ],   Phi(0) = 1.

    The closed form is 0/0 at x = 0 and cancels near it, so |x| < 1 is summed from the series
    1 - x^2/30 + x^4/2520 - ...
    """
    if abs(x) < _PHI_SERIES_CUTOFF:
        x2 = x * x
        total = 0.0
        for c in reversed(_PHI_COEFFS):
            total = total * x2 + c
        return total
    y = x / SQRT3
    return 9.0 / (x * x) * (math.sin(y) / y - math.cos(y))


def dPhi(x: float) -> float:
    """
    d Phi / d x, of KT eq. (19). Series branch below |x| = 1: -x/15 + x^3/630 - ...
    """
    if abs(x) < _PHI_SERIES_CUTOFF:
        x2 = x * x
        # d/dx of sum_n c_n x^{2(n-1)} = sum_{n>=2} 2(n-1) c_n x^{2n-3}
        total = 0.0
        for n in range(_PHI_SERIES_TERMS, 1, -1):
            total = total * x2 + 2.0 * (n - 1) * _PHI_COEFFS[n - 1]
        return total * x
    y = x / SQRT3
    s, c = math.sin(y), math.cos(y)
    g = s / y - c
    gp = (c / y - s / (y * y) + s) / SQRT3
    return 9.0 * gp / (x * x) - 18.0 * g / (x * x * x)


# --------------------------------------------------------------------------------------------
# KT eq. (16) at w = 1/3, and eq. (20)
# --------------------------------------------------------------------------------------------


def f_RD(v: float, u: float, x: float) -> float:
    """
    The source of KT eq. (15) in radiation domination: KT eq. (16), the general-w source, at
    w = 1/3 and with Phi from eq. (19),

        f = c1 Phi(vx) Phi(ux) + c2 [ x v Phi'(vx) Phi(ux) + x u Phi'(ux) Phi(vx) ]
            + c3 x v Phi'(vx) x u Phi'(ux),

        c1 = 6(w+1)/(3w+5),  c2 = 6(1+3w)(w+1)/(3w+5)^2,  c3 = 3(1+3w)^2(1+w)/(3w+5)^2,

    i.e. c1 = 4/3, c2 = c3 = 4/9 at w = 1/3, so f -> 4/3 as x -> 0. The eq. (16) term the paper
    writes "xbar d_etabar Phi" is read as xbar d_xbar Phi (etabar d_etabar = xbar d_xbar); the
    agreement with eq. (20) is what licenses that reading.

    ERRATUM 3 -- WHY THIS IS NOT EQ. (20). The paper's explicit radiation-era source, eq. (20), is
    the same function, but its bracket is a sum of O(x^2) terms that cancel to O(x^6) against an
    explicit x^6 denominator, so its relative error grows like eps / x^4 -- 1e-9 at x = 0.1,
    8e-6 at x = 0.01, O(1) near x = 1e-3 -- and diverges at x = 0, where eq. (15) starts. Used as
    the integrand of eq. (15) it puts I out by nine orders at x = 0.03 (4.5e5 against a true
    2.0e-4, KOHRI-TERADA-ORACLE.md section 2 item 3). Eq. (16) built from the series-guarded Phi
    above has no such cancellation, and it is what every quadrature in this module integrates.
    f_RD_eq20 exists only to test this function against.
    """
    c1 = 6.0 * (_W + 1.0) / (3.0 * _W + 5.0)
    c2 = 6.0 * (1.0 + 3.0 * _W) * (_W + 1.0) / (3.0 * _W + 5.0) ** 2
    c3 = 3.0 * (1.0 + 3.0 * _W) ** 2 * (1.0 + _W) / (3.0 * _W + 5.0) ** 2

    Pv, Pu = Phi(v * x), Phi(u * x)
    xdPv, xdPu = x * v * dPhi(v * x), x * u * dPhi(u * x)
    return c1 * Pv * Pu + c2 * (xdPv * Pu + xdPu * Pv) + c3 * xdPv * xdPu


def f_RD_eq20_with_rounding_scale(v: float, u: float, x: float) -> tuple:
    """
    KT eq. (20), returning (f, scale): `scale` is the sum of the absolute values of the terms its
    bracket adds, times the prefactor, so that its rounding error is about eps * scale. See
    f_RD_eq20.
    """
    cu, su = math.cos(u * x / SQRT3), math.sin(u * x / SQRT3)
    cv, sv = math.cos(v * x / SQRT3), math.sin(v * x / SQRT3)
    x2 = x * x
    terms = (
        18.0 * u * v * x2 * cu * cv,
        54.0 * su * sv,
        -6.0 * (u * u + v * v) * x2 * su * sv,
        u * u * v * v * x2 * x2 * su * sv,
        2.0 * SQRT3 * u * x * v * v * x2 * cu * sv,
        -18.0 * SQRT3 * u * x * cu * sv,
        2.0 * SQRT3 * v * x * u * u * x2 * su * cv,
        -18.0 * SQRT3 * v * x * su * cv,
    )
    prefactor = 12.0 / (u**3 * v**3 * x**6)
    return prefactor * math.fsum(terms), abs(prefactor) * sum(abs(t) for t in terms)


def f_RD_eq20(v: float, u: float, x: float) -> float:
    """
    KT eq. (20), the explicit radiation-era source,

        f = 12/(u^3 v^3 x^6) [ 18 u v x^2 cos(ux/sqrt3) cos(vx/sqrt3)
              + (54 - 6(u^2+v^2) x^2 + u^2 v^2 x^4) sin(ux/sqrt3) sin(vx/sqrt3)
              + 2 sqrt3 u x (v^2 x^2 - 9) cos(ux/sqrt3) sin(vx/sqrt3)
              + 2 sqrt3 v x (u^2 x^2 - 9) sin(ux/sqrt3) cos(vx/sqrt3) ].

    ERRATUM 3: UNUSABLE BELOW x ~ 0.1. The bracket's O(x^2) terms cancel to O(x^6) against the
    x^6 denominator, so the relative error grows like eps / x^4 (1e-9 at x = 0.1, 8e-6 at 0.01,
    O(1) near 1e-3) and diverges at x = 0. It is the same function as f_RD and is implemented
    only so that f_RD can be tested against it away from that region. Never integrate it: the
    integral of eq. (15) starts at xbar = 0.
    """
    return f_RD_eq20_with_rounding_scale(v, u, x)[0]


# --------------------------------------------------------------------------------------------
# sine and cosine integrals
# --------------------------------------------------------------------------------------------


def Si(z: float) -> float:
    """Si(z) = int_0^z sin(t)/t dt. Odd."""
    return sici(z)[0]


def Ci(z: float) -> float:
    """Ci(|z|) = gamma + ln|z| + int_0^|z| (cos t - 1)/t dt. Eq. (22) only ever takes Ci of |.|."""
    return sici(abs(z))[1]


# Cin's series: Cin(z) = sum_{n>=1} (-1)^{n+1} z^{2n} / (2n (2n)!). Below z = 1 the closed form
# gamma + ln z - Ci(z) cancels (to ~ z^2/4 against ln z), and at z = 1 the series has converged
# to rounding in the terms kept.
_CIN_SERIES_CUTOFF = 1.0
_CIN_SERIES_TERMS = 12
_CIN_COEFFS = tuple(
    (-1.0) ** (n + 1) / (2.0 * n * math.factorial(2 * n))
    for n in range(1, _CIN_SERIES_TERMS + 1)
)


def Cin(z: float) -> float:
    """
    Cin(z) = int_0^z (1 - cos t)/t dt = gamma + ln z - Ci(z): entire, even, and starting at
    z^2/4. It is what makes I_RD regular at the resonance. Summed from its series for |z| < 1,
    where the closed form cancels.
    """
    z = abs(z)
    if z < _CIN_SERIES_CUTOFF:
        z2 = z * z
        total = 0.0
        for c in reversed(_CIN_COEFFS):
            total = total * z2 + c
        return total * z2
    return EULER_GAMMA + math.log(z) - sici(z)[1]


# --------------------------------------------------------------------------------------------
# KT eq. (22)
# --------------------------------------------------------------------------------------------


def I_RD_with_rounding_scale(v: float, u: float, x: float) -> tuple:
    """
    KT eq. (22), returning (I_RD, scale). `scale` is the sum, over every term eq. (22) adds, of
    its magnitude plus the error its inputs' rounding induces (the trig arguments, d, and the
    Ci/Si arguments), times the prefactor. Rounding in I_RD is set by `scale`, not by I_RD,
    because eq. (22) cancels: at small x its terms exceed its O(x^2) result by a factor that
    grows like x^-4 (~5e2 / x^4 at (v, u) = (0.7, 1.1)), and when u or v is small its
    1/(u^3 v^3) prefactor multiplies terms that nearly cancel. eps * scale is the estimate the
    tests use for the closed form's own error; against a 50-digit mpmath evaluation of the same
    formula the actual error was at most 0.33 of it on 307 points (radiation-oracle log 01). See
    I_RD for the formula and the errata.
    """
    d = u * u + v * v - 3.0
    cu, su = math.cos(u * x / SQRT3), math.sin(u * x / SQRT3)
    cv, sv = math.cos(v * x / SQRT3), math.sin(v * x / SQRT3)
    sx, cx = math.sin(x), math.cos(x)

    rational_terms = (
        u * v * d * x**3 * sx,
        -6.0 * u * v * x * x * cu * cv,
        6.0 * SQRT3 * u * x * cu * sv,
        6.0 * SQRT3 * v * x * su * cv,
        -3.0 * (6.0 + d * x * x) * su * sv,
    )

    # ERRATUM 1: the (v - u) pair carries +Ci and -Si, the (v + u) pair carries -Ci and +Si
    a = 1.0 - (v - u) / SQRT3
    b = 1.0 + (v - u) / SQRT3
    c = 1.0 - (v + u) / SQRT3
    e = 1.0 + (v + u) / SQRT3

    # -Ci(|c| x) + log|(3 - (u+v)^2)/(3 - (u-v)^2)|, regularised: with
    # 3 - (u+v)^2 = 3 c e and 3 - (u-v)^2 = 3 a b, and -Ci(z) = Cin(z) - gamma - ln z,
    #   = Cin(|c| x) - gamma - ln x + log e - log|a b|,
    # the ln|c| having cancelled analytically.
    sin_terms = (
        Ci(a * x),
        Ci(b * x),
        -Ci(e * x),
        Cin(abs(c) * x),
        -EULER_GAMMA,
        -math.log(x),
        math.log(e),
        -math.log(abs(a * b)),
    )
    cos_terms = (-Si(a * x), -Si(b * x), Si(c * x), Si(e * x))

    prefactor = 3.0 / (4.0 * u**3 * v**3 * x)
    rational = math.fsum(rational_terms)

    # The rounding scale: for every term, its magnitude plus the absolute error that rounding
    # of its inputs induces, so that eps * scale estimates the error of `value`.
    #  - sin/cos of a rounded argument y = u x / sqrt3 move by ~eps |y cos y|, ~eps |y sin y|;
    #  - d = u^2 + v^2 - 3 is off by ~eps (u^2 + v^2 + 3), which matters as d -> 0; it enters
    #    d^2 multiplying the brackets' SUM, not their terms;
    #  - a, b, e are off by ~eps E, so Ci and Si of a x, b x, e x move by ~eps E / |a| etc.
    #    (Ci' = cos z / z, Si' = sin z / z), and Si(c x), Cin(|c| x) by ~eps E_ce min(x, 2/|c|),
    #    which is bounded AT the resonance: the regularisation costs nothing in rounding.
    yu, yv = abs(u * x / SQRT3), abs(v * x / SQRT3)
    es_u, ec_u = yu * abs(cu), yu * abs(su)
    es_v, ec_v = yv * abs(cv), yv * abs(sv)
    d_err = u * u + v * v + 3.0
    E_ab = 1.0 + abs(v - u) / SQRT3
    E_ce = 1.0 + (v + u) / SQRT3
    c_arg = E_ce * min(abs(x), 2.0 / abs(c)) if c != 0.0 else E_ce * abs(x)
    ci_si_arg = E_ab / abs(a) + E_ab / abs(b) + E_ce / e + c_arg

    rational_scale = (
        abs(u * v * x**3 * sx) * (abs(d) + d_err)
        + 6.0 * abs(u * v) * x * x * (abs(cu * cv) + abs(cv) * ec_u + abs(cu) * ec_v)
        + 6.0 * SQRT3 * abs(u * x) * (abs(cu * sv) + abs(sv) * ec_u + abs(cu) * es_v)
        + 6.0 * SQRT3 * abs(v * x) * (abs(su * cv) + abs(cv) * es_u + abs(su) * ec_v)
        + 3.0
        * (6.0 + (abs(d) + d_err) * x * x)
        * (abs(su * sv) + abs(sv) * es_u + abs(su) * es_v)
    )
    bracket = sx * math.fsum(sin_terms) + cx * math.fsum(cos_terms)
    special_scale = d * d * (
        abs(sx) * (sum(abs(t) for t in sin_terms) + ci_si_arg)
        + abs(cx) * (sum(abs(t) for t in cos_terms) + ci_si_arg)
    ) + 2.0 * abs(d) * d_err * abs(bracket)
    scale = abs(prefactor) * (4.0 / abs(x) ** 3 * rational_scale + special_scale)
    value = prefactor * (-4.0 / x**3 * rational + d * d * bracket)
    return value, scale


def I_RD(v: float, u: float, x: float) -> float:
    """
    KT eq. (22): the radiation-era closed form of eq. (15),

        I_RD(v, u, x) = int_0^x dxbar (xbar/x) sin(x - xbar) f_RD(v, u, xbar),

    in Si, Ci, sines and a logarithm. With d = u^2 + v^2 - 3,

        I_RD = 3/(4 u^3 v^3 x) { -(4/x^3) [ u v d x^3 sin x - 6 u v x^2 cos(ux/s3) cos(vx/s3)
                                             + 6 s3 u x cos(ux/s3) sin(vx/s3)
                                             + 6 s3 v x sin(ux/s3) cos(vx/s3)
                                             - 3 (6 + d x^2) sin(ux/s3) sin(vx/s3) ]
            + d^2 sin x [ Ci((1 - (v-u)/s3) x) + Ci((1 + (v-u)/s3) x)
                          - Ci(|1 - (v+u)/s3| x) - Ci((1 + (v+u)/s3) x)
                          + log| (3 - (u+v)^2) / (3 - (u-v)^2) | ]
            + d^2 cos x [ -Si((1 - (v-u)/s3) x) - Si((1 + (v-u)/s3) x)
                          + Si((1 - (v+u)/s3) x) + Si((1 + (v+u)/s3) x) ] },   s3 = sqrt(3).

    ERRATUM 1 -- THE Ci/Si ARGUMENTS PAIR WITH THE DIFFERENCE, NOT THE SUM. +Ci and -Si take
    (v - u); -Ci and +Si take (v + u). The rendered PDF makes the other pairing the natural
    reading -- the grouping of (v +- u)/sqrt3 inside (1 +- (v +- u)/sqrt3) is easy to invert --
    and the inverted function is smooth and plausible, misses the quadrature of eq. (15) by O(1)
    at every x, and tends to -12.08 instead of 0 as x -> 0. The authority is the ar5iv `alttext`
    of eq. (22), not the rendered equation; the form above is the one that reproduces eq. (15)
    by quadrature to 3.5e-15 (KOHRI-TERADA-ORACLE.md sections 2 item 1 and 5).

    ERRATUM 2 -- THE SMALL-x LIMIT IS 2 x^2 / 9. The paper says below eq. (24) that "for small x
    ... I_RD ~ x^2/2". That remark is wrong by 4/9: f_RD -> 4/3, so
    I -> (4/3) int_0^x (xbar/x)(x - xbar) dxbar = 2 x^2 / 9, which eq. (22) itself and the
    quadrature both reproduce. The formula is right; only the remark is wrong. Do not use x^2/2
    as an acceptance test -- it rejects a correct implementation.

    THE RESONANCE u + v = sqrt(3) IS REGULARISED, NOT AVOIDED. There -Ci(|1 - (u+v)/sqrt3| x) and
    the log each diverge and their sum does not, so eq. (22) as printed returns +-inf on the
    resonance and loses digits approaching it. Writing c = 1 - (u+v)/sqrt3, e = 1 + (u+v)/sqrt3,
    a = 1 - (v-u)/sqrt3, b = 1 + (v-u)/sqrt3 and Cin(z) = gamma + ln z - Ci(z),

        -Ci(|c| x) + log|(3-(u+v)^2)/(3-(u-v)^2)| = Cin(|c| x) - gamma - ln x + log e - log|a b|,

    in which ln|c| has cancelled analytically. This is exact, not an approximation, and it is
    finite AT the resonance, where it gives 0.1391877548292 at x = 15 (KOHRI-TERADA-ORACLE.md
    section 6.1). Nothing here raises, returns nan or perturbs u, v off the resonance: it is the
    physically interesting point and an oracle unusable there is unusable where it matters.

    The companion singularity |u - v| = sqrt(3), where a b = 0, is not regularised: it lies
    outside the triangle |u - v| <= 1 <= u + v that momentum conservation allows.

    ROUNDING. Eq. (22) cancels at small x and, through its 1/(u^3 v^3) prefactor, at small u or
    v, so its error is set by I_RD_with_rounding_scale's `scale`, not by I_RD: ~5e-6 relative
    at x = 0.01, and ~1e-10 at u = 0.01 for x of tens to thousands (whose conservative
    estimate eps * scale is 3e-9 to 1.4e-8).
    """
    return I_RD_with_rounding_scale(v, u, x)[0]


def I_RD_asymptotic(v: float, u: float, x: float) -> float:
    """
    KT eq. (25): the x -> infinity limit of eq. (22),

        I_RD -> 3 d / (4 u^3 v^3 x) { sin x [ -4 u v + d log|(3 - (u+v)^2)/(3 - (u-v)^2)| ]
                                        - pi d Theta(v + u - sqrt3) cos x },   d = u^2 + v^2 - 3,

    approached like 1/x relative to its own envelope. Singular at the resonance, as the paper
    has it: the large-x limit there is not uniform, and I_RD is what to use.
    """
    d = u * u + v * v - 3.0
    log_arg = abs((3.0 - (u + v) ** 2) / (3.0 - (u - v) ** 2))
    theta = 1.0 if (v + u - SQRT3) > 0.0 else 0.0
    return (
        3.0
        * d
        / (4.0 * u**3 * v**3 * x)
        * (
            math.sin(x) * (-4.0 * u * v + d * math.log(log_arg))
            - math.pi * d * theta * math.cos(x)
        )
    )


# --------------------------------------------------------------------------------------------
# KT eq. (15) by quadrature
# --------------------------------------------------------------------------------------------


def I_RD_quadrature(
    v: float,
    u: float,
    x: float,
    x_lo: float = 0.0,
    x_hi: float = None,
    epsabs: float = 1e-16,
    epsrel: float = 1e-13,
) -> tuple:
    """
    KT eq. (15) by scipy.quad, returning (value, declared_error):

        int_{x_lo}^{x_hi} dxbar (xbar/x) sin(x - xbar) f_RD(v, u, xbar),

    x_hi defaulting to x, so that the default range is eq. (15)'s own and the result is I_RD.
    The kernel carries the OUTER x throughout: a sub-range such as the head 0 -> x_min is its own
    integral, not I_RD(v, u, x_min).

    The integrand is f_RD, i.e. eq. (16), never eq. (20) (erratum 3, see f_RD). The range is
    broken at every period of every oscillation the integrand carries -- (u +- v)/sqrt3, u/sqrt3,
    v/sqrt3 from the source and 1 from the Green's function -- so quad never has to discover one.
    `declared_error` is the sum of quad's own error estimates over the pieces.

    This shares the Green's function, the measure and the source with I_RD, so agreement between
    the two checks eq. (22)'s transcription, not the kernel (KOHRI-TERADA-ORACLE.md section 0).
    """
    if x_hi is None:
        x_hi = x

    def integrand(xb):
        return (xb / x) * math.sin(x - xb) * f_RD(v, u, xb)

    pts = set()
    for fr in (1.0, (u + v) / SQRT3, abs(u - v) / SQRT3, u / SQRT3, v / SQRT3):
        if fr <= 0.0:
            continue
        period = 2.0 * math.pi / fr
        n = 1
        while n * period < x_hi and len(pts) < 4000:
            if n * period > x_lo:
                pts.add(n * period)
            n += 1

    edges = [x_lo] + sorted(pts) + [x_hi]
    value, err = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        piece, piece_err = quad(
            integrand, lo, hi, limit=200, epsabs=epsabs, epsrel=epsrel
        )
        value += piece
        err += piece_err
    return value, err


# --------------------------------------------------------------------------------------------
# the mapping onto this repository's `total`
# --------------------------------------------------------------------------------------------


def total_from_I_RD(
    k: float, q: float, r: float, tau_response: float, tau_source_max: float
) -> dict:
    """
    What evaluate_QuadSource_integral's `total` must be at b = 0, from KT eq. (22):

        total = KT_NORM / k_phys^2 * [ I_RD(r/k, q/k, k tau_resp) - head ],   KT_NORM = -9/8,

        head  = int_0^{k tau_source_max} dxbar (xbar/x) sin(x - xbar) f_RD(v, u, xbar).

    k, q, r are the code's wavenumbers k/a0 -- physical wavenumbers today -- and tau is the code's
    conformal time a0 eta, so x = k tau = (k/a0)(a0 eta) and u, v are ratios: every argument is
    a0-invariant. THE MAPPING IS WRITTEN IN 1/k_phys^2 BECAUSE a0 IS ABSORBED, NOT SET TO ONE
    (docs/spec/04-source-integral.md section 0 (1)): substituting the code's Green's function
    G_code(z,z') = -a0 H(z') Gr_k (spec 04 section 0 (2)), whose Gr_k is KT's G_k of their
    eq. (10), and d log(1+z') = -a0 H d eta'/(1+z') into the code's target leaves
    a0^2 int d eta' (a(etabar)/a(eta)) Gr_k f_code; against KT eq. (15),
    I = k^2 int d etabar (a/a) Gr_k f_KT, so total = (a0^2/k^2) (f_code/f_KT) I, and
    a0^2/k^2 = 1/k_phys^2 is invariant under a0 -> lambda a0 with k -> lambda k. The pure number
    f_code/f_KT is KT_NORM (see its comment for the factorisation and the sign convention).
    KOHRI-TERADA-ORACLE.md section 3.

    The head is subtracted because KT integrate from xbar = 0 and the code from
    tau(z_source_max); it is quadratured with the outer x in the kernel.

    Returns a dict: `total`; `total_error`, this prediction's own error estimate -- eps times
    eq. (22)'s rounding scale plus the head's declared quadrature error, mapped through the same
    prefactor; and the ingredients `u`, `v`, `x`, `x_min`, `I_RD`, `I_RD_scale`, `head`,
    `head_error`. At u = q/k = 0.01 the first term dominates the head's by nine orders and is
    3e-9 to 1.4e-8 of `total` (the actual error, against 50-digit mpmath, is ~1e-10): eq. (22)'s
    1/(u^3 v^3) cancellation, not the head, is what limits this prediction.
    """
    u, v = q / k, r / k
    x = k * tau_response
    x_min = k * tau_source_max

    I_closed, scale = I_RD_with_rounding_scale(v, u, x)
    head, head_err = I_RD_quadrature(
        v, u, x, x_lo=0.0, x_hi=x_min, epsabs=1e-18, epsrel=1e-13
    )
    mapping = KT_NORM / (k * k)
    return {
        "total": mapping * (I_closed - head),
        "total_error": abs(mapping) * (_EPS * scale + head_err),
        "u": u,
        "v": v,
        "x": x,
        "x_min": x_min,
        "I_RD": I_closed,
        "I_RD_scale": scale,
        "head": head,
        "head_error": head_err,
    }
