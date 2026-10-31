import math
from typing import NamedTuple, Sequence

import numpy as np
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from Quadrature.simple_quadrature import simple_quadrature


class BesselIntegralResult(NamedTuple):
    """
    Return type of quad_JJJ/quad_YJJ (and their internal building blocks). Replaces the bare float
    this module used to return (prompts/levin-refactor's prompt 09, audit sec 3.4 / recommendation
    15): the four sum-and-difference phase groups combine as (-G1+G2+G3-G4)/4, a cancellative
    combination whose *relative* error is amplified by the cancellation factor, and only the summed
    absolute error can reveal that -- four group values of order one cancelling to 1e-6, each
    accurate to 1e-12, give a relative error of 1e-6 with nothing in a bare float to say so.

    Component errors are combined **linearly**, not in quadrature, for the same reason the
    quadrature itself sums regions linearly (levin_quadrature.py): the four groups share a phase
    construction, so a systematically inaccurate phase produces a common drift across them rather
    than independent noise.

    IMPORTANT: "abserr" is the estimated error of the *quadrature*, and it now knows about the
    declared error of the phase but not about anything else.

    **What it includes, as of prompts/transfer-remedial's prompt 07.** Each of the four Levin calls
    is handed a "theta_abserr" for its group (BesselPhaseGroup.theta_abserr), so the phase construction's
    own declared error reaches the driver: it enters each region's achievable-accuracy floor
    (levin_quadrature._declared_endpoint_phase_err(), _roundoff_floor()) and therefore decides when
    a region is accepted as "phase_limited" rather than subdivided further. A tightening tolerance
    that would once have subdivided against a phase it could not resolve now stops and says so --
    measured, quad_YJJ(Y022) at k,q,s = 1.3,1.7,2.1, max_x = 1e12, rtol = 1e-14 takes 0.44 s
    instead of 2.15 s for a value agreeing to 3.3e-14 relative. This is why the reported "abserr"
    can *rise* when the declaration is supplied: the declaration is a statement about what is
    achievable, not a reduction of the error.

    Since prompts/levin-phase-floor/06h the declaration covers the group's own arithmetic too, not
    only its constituents' representations: the rounding of the abscissa x = fl(exp(u)) at which
    the group is evaluated, as eps_exp |dtheta/du|, and the evaluation of the group phase itself
    (its small angle's sum, angle addition, atan2 and the arguments m_i x). The rounding of the
    product K x and of K, which before 06h reached 148 times the declaration and was not declared,
    is no longer made: BesselPhaseGroup.sin_cos carries both exactly.

    **What it still does not include.** (i) The fit error of the *modulus*: adaptive_levin_sincos()
    has no analogue of theta_abserr for the f-vector, and bessel_phase's declared
    "amplitude_relerr" is not passed anywhere. (ii) The truncation of the integral at max_x -- the
    closed forms are integrals to infinity, and the omitted tail is oscillatory with a slowly
    decaying envelope, which is the largest part of the residual disagreement for the lowest-order
    oracles. (iii) Uncertainty in k, q, s themselves: near resonance x delta K is a property of the
    inputs, and no arithmetic in BesselPhaseGroup makes it smaller.

    **The floor, re-measured.** The uniform ~2e-8 relative accuracy this docstring used to record
    across all seven of test_3bessel_analytic.py's closed forms (see DEFAULT_3BESSEL_CHEBYSHEV_ORDER's
    comment above and prompts/levin-refactor/logs/09-caller-propagation.md) is gone: it was the
    phase and modulus fit error of the construction bessel_phase used before
    prompts/transfer-remedial's prompt 05. At k,q,s = 1.3,1.7,2.1, max_x = 1e12, atol = 1e-14,
    rtol = 1e-10 the residual relative disagreement with those same seven closed forms is now
    1.4e-10 (J000), 5.9e-12 (J110), 2.5e-14 (J220), 3.5e-14 (J222), 1.5e-13 (J231), 4.8e-11 (Y000)
    and 2.0e-13 (Y022), and the reported "abserr" bounds the true error on 7 of 7 -- against 2 of 7
    before that prompt (test_3bessel_analytic.test_abserr_bounds_truth, which is consequently now
    an *unexpected success* and belongs to prompt 08).

    Treat a small reported "abserr" here as "the Levin rule resolved the phase it was given
    accurately, and that phase declares itself accurate to theta_abserr", not as "this value is
    accurate to that many digits".
    """

    value: float
    abserr: float
    converged: bool
    phase_limited: bool


# The Levin rule integrates against a (sin, cos) basis carrying a single phase, so a product of three
# Bessel functions has to be decomposed into sum-and-difference phases first. Writing each Bessel factor
# in Liouville-Green form as (modulus) x (sine of phase), the triple product expands into four terms
# whose phases are theta_mu + e_nu*theta_nu + e_sigma*theta_sigma. These are the four sign pairs.
_PHASE_GROUP_SIGNS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

# Chebyshev spectral order used for the three-Bessel Levin evaluations. The cost of each Levin solve is
# dominated by evaluating the phase and modulus splines at the collocation points, so total cost scales
# roughly as (order) x (number of solves): a high order buys fewer subintervals but more than pays for
# it per solve.
#
# This was previously 64. Measured against the analytic oracles (J000, J231, Y022 at k,q,s = 1.3,1.7,2.1
# and max_x = 1e5), the relative error is *identical* at orders 12 through 64 -- it is set by the
# accuracy of the phase and modulus splines, not by the spectral order -- while the runtime rises by a
# factor of 3 to 6.6 over that range. So the order was costing several times the necessary work for no
# accuracy at all.
#
# Note the SVD-failure recovery in adaptive_levin_sincos() steps the order down in twos to a floor of 8,
# so 12 leaves it two steps of headroom.
DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12

# ----------------------------------------------------------------------------------------------
# The group phase's own rounding (prompts/levin-phase-floor/06h-group-phase-rounding.md).
#
# BesselPhaseGroup.sin_cos carries the roundings of fl(K x) and of K itself exactly in its small
# angle, and BesselPhaseGroup.theta_abserr declares what remains: the rounding of the abscissa
# x = fl(exp(u)) and the group's own evaluation (phi_eval). The constants below are those terms'
# bounds, each with its derivation. u = 2^-53 throughout.
# ----------------------------------------------------------------------------------------------

#: The unit roundoff of a double, 2^-53.
_UNIT_ROUNDOFF = 2.0**-53

#: Dekker's splitting constant 2^27 + 1, for the array form of TwoProduct (_two_product_error).
_TWO_PRODUCT_SPLITTER = 2.0**27 + 1.0

#: The multiple of the worst measured relative error of ``np.exp`` that :data:`EXP_RELERR` takes.
EXP_RELERR_MULTIPLE = 2.0

#: The worst relative error ``|fl(exp(u)) - e^u| / e^u`` of ``np.exp`` measured on 06h's sample:
#: 10^6 seeded points in each of u in [-30, 5] (the analytic block's log(eta); 06f's calls span
#: [-17.38, -1.79]) and u in [-2, log 1e12] (the LiouvilleGreen range), on a 0-d argument and on an
#: array, against mpmath at 30 digits. 1.1146e-16 = 1.004 u, attained at u = 0.696347627688485 on
#: both paths, which are bit-identical to each other and to math.exp at every point
#: (docs/levin-phase-floor/06h/calibration/scripts/measure_exp_relerr.py, out/exp_relerr.txt).
#: Re-measured on the same sample by
#: LiouvilleGreen/tests/test_group_phase_rounding.py's test of eps_exp's basis.
EXP_RELERR_MEASURED_WORST = 1.1146192306576078e-16

#: eps_exp: the bound on the relative error of the ``exp`` that turns the integration variable u
#: into the abscissa x = fl(exp(u)), so x = e^u (1 + delta_e) with |delta_e| <= eps_exp.
#:
#: **Measured, not derived.** The platform library documents no accuracy for exp (macOS's exp(3)
#: states none, and numpy 2.2.4 on aarch64 calls the platform's exp for float64). So it is taken
#: at EXP_RELERR_MULTIPLE = 2 times the worst measured (:data:`EXP_RELERR_MEASURED_WORST`), rounded
#: up: 2.23e-16 = 2.009 u. That is at least the bound 2u of any faithfully rounded exp (an error
#: below one ulp of the result, relative 2u at the bottom of a binade), which the measured worst
#: in ulps, 0.512, is comfortably inside.
EXP_RELERR = 2.23e-16

#: The allowance applied to ``sum_i |dr_i/d log y|`` wherever it enters a bound, for the error of
#: that sum's own evaluation: in the near region dr/d log y is the derivative of an interpolant,
#: whose relative error is not declared anywhere, so it is taken at 100 %, i.e. the magnitude is
#: doubled. In the tail it is a closed-form series and the factor is pure margin.
_LOG_DERIV_ALLOWANCE = 2.0

#: The number of roundings in the small angle of BesselPhaseGroup.sin_cos, as evaluated: C_red,
#: e_1 r_1, e_2 r_2, e_3 r_3, p_lo and K_lo x added left to right (each e_i r_i is exact, e_i being
#: +-1, and fl(K_lo x) is declared separately). Six summands and five additions, so by the
#: standard bound for recursive summation the computed sum is within gamma_5 = 5u/(1 - 5u) times
#: the sum of the six magnitudes (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd
#: ed., sec. 4.2).
_SMALL_ANGLE_ADDITIONS = 5

#: gamma_n for the small angle's sum.
_SMALL_ANGLE_GAMMA = (
    _SMALL_ANGLE_ADDITIONS
    * _UNIT_ROUNDOFF
    / (1.0 - _SMALL_ANGLE_ADDITIONS * _UNIT_ROUNDOFF)
)

#: The angle addition of BesselPhaseGroup.sin_cos as an equivalent phase error, in radians.
#:
#: s = sin(B) cos(S) + cos(B) sin(S) and c = cos(B) cos(S) - sin(B) sin(S) are formed from four
#: library values, each in error by at most eps_t, with two products and one sum per component.
#: To first order
#:     |delta s| <= eps_t (|cos S| + |sin S| + |sin B| + |cos B|) + u (|sin B cos S| + |cos B sin S|)
#:                  + u |s|  <=  2 sqrt(2) eps_t + 2u,
#: since |cos S| + |sin S| <= sqrt(2), likewise for B, and the products' magnitudes sum to at most 1
#: (Cauchy-Schwarz); the same for |delta c|. The phase the pair represents, atan2(s, c), moves by
#: the tangential component, at most |(delta s, delta c)| / |(s, c)| <= sqrt(2) (2 sqrt(2) eps_t +
#: 2u) = 4 eps_t + 2 sqrt(2) u. The library's sin and cos are taken at **two ulps**, eps_t = 2u
#: (two ulps of a value below 1 in magnitude): they are not documented, and 06h's measurement
#: (out/exp_relerr.txt) finds them faithful but not correctly rounded, worst 0.85 ulp on 10^5
#: seeded large and small arguments, so one ulp would leave a margin of 1.2 only. That gives
#: 8u + 2.83u = 10.83u, taken at 12u for the second-order terms and the normalisation |(s, c)| =
#: 1 + O(u).
#:
#: **Counted against the constituents' EVALUATION_FLOOR, and not credited to it.** That floor
#: (LiouvilleGreen.bessel_phase.EVALUATION_FLOOR) is each constituent's allowance for its *own*
#: angle addition in BesselPhaseFunction.sin_cos_theta, and for the rounding of its c_nu_reduced.
#: The group never calls sin_cos_theta: it inherits the three c_nu_reduced roundings, which stay
#: declared by the constituents and are not declared again here, and performs one angle addition
#: of its own on different arguments, which no constituent declares and which is therefore declared
#: here rather than omitted.
_ANGLE_ADDITION_PHASE_ERR = 12.0 * _UNIT_ROUNDOFF

#: The rounding of atan2 in BesselPhaseGroup.theta_mod_2pi, in radians: its result lies in
#: (-pi, pi], where an ulp is at most 2^-51 = 4u, and the library's atan2, like its sin and cos,
#: is taken at two ulps (measured worst 0.72 ulp, out/exp_relerr.txt): 8u. The driver integrates
#: this value, not the pair, so it is part of the phase it is given.
_ATAN2_PHASE_ERR = 8.0 * _UNIT_ROUNDOFF


def _ordered_sum(terms):
    """
    ``terms`` added left to right, one rounding per addition, for scalars and arrays alike.

    Not the builtin sum(): since Python 3.12 it compensates (Neumaier) a sum whose items are
    exact Python floats, but not one of arrays or of np.float64, so a scalar call and an array
    call of the same expression can differ in the last bit -- which
    AdaptiveLevin.levin_quadrature._detect_vectorized() does not allow of a callable it samples
    on an array, and which BesselPhaseGroup.theta_abserr must not do (prompts/levin-phase-floor/
    06h). The rounding of this sum is a few u of its terms, inside every bound it enters.
    """
    total = 0.0
    for term in terms:
        total = total + term
    return total


def _two_product_split(value):
    """Dekker's split of ``value`` into a high part of 26 bits and an exact remainder."""
    scaled = _TWO_PRODUCT_SPLITTER * value
    high = scaled - (scaled - value)
    return high, value - high


def _two_product_error(a: float, b, product):
    """
    The rounding error of a product, ``a b - fl(a b)``, **exactly**, given ``product = fl(a b)``
    (TwoProduct). ``a`` is a scalar; ``b`` and ``product`` are a scalar or an array of one shape.

    On a scalar it is ``math.fma(a, b, -product)``: the fused multiply-add rounds once, and
    ``a b - product`` is exactly representable, so the result is exact. On an array it is Dekker's
    algorithm with Veltkamp's split (T. J. Dekker, Numer. Math. 18 (1971) 224; "TwoProduct" in
    Ogita, Rump and Oishi, SIAM J. Sci. Comput. 26 (2005) 1955), which is exact in the same
    conditions. Both are exact barring overflow and underflow -- here |a| is a wavenumber sum and
    |b| = x <= 1e16, far from either -- so the two give the same double: the array result is
    bit-identical to the scalar one at every point, as
    AdaptiveLevin.levin_quadrature._detect_vectorized() requires of a phase callable.
    """
    if np.ndim(product) == 0:
        return math.fma(a, float(b), -float(product))
    b = np.asarray(b, dtype=float)
    a_high, a_low = _two_product_split(a)
    b_high, b_low = _two_product_split(b)
    return (
        (a_high * b_high - product) + a_high * b_low + a_low * b_high
    ) + a_low * b_low


def quad_JJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    min_x_mu = mu_phase["min_x"]
    min_x_nu = nu_phase["min_x"]
    min_x_sigma = sigma_phase["min_x"]

    # the integral from 0 up to max_x needs to be cut into two: numerical quadrature for low x,
    # where the Bessel phase representation is not reliable, and Levin integration
    # for the rest.
    # The first value of x where phase information is available for all three Bessel functions
    # min_cut = max(lowest x value) / min(k, q, s)
    x_cut = max(min_x_mu, min_x_nu, min_x_sigma)
    min_kqs = min(k, q, s)
    min_cut = x_cut / min_kqs

    numeric = _direct_JJJ(mu, nu, sigma, k, q, s, 0.0, min_cut, atol, rtol)
    Levin = _Levin_JJJ(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_cut,
        max_x,
        atol,
        rtol,
        chebyshev_order=chebyshev_order,
    )

    # print(f">> numeric on (0, {min_cut}) = {numeric}")
    # print(f">> Levin on ({min_cut}, {max_x}) = {Levin}")

    return BesselIntegralResult(
        value=numeric.value + Levin.value,
        abserr=numeric.abserr + Levin.abserr,
        converged=numeric.converged and Levin.converged,
        # plain quadrature (the "numeric" part) has no phase-limited concept; whether the combined
        # result is phase-limited is entirely a property of the Levin part
        phase_limited=Levin.phase_limited,
    )


def _direct_JJJ(
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
):
    def integrand(x):
        return (
            np.sqrt(x)
            * jv(mu + 0.5, k * x)
            * jv(nu + 0.5, q * x)
            * jv(sigma + 0.5, s * x)
        )

    data = simple_quadrature(
        integrand,
        a=min_x,
        b=max_x,
        atol=atol,
        rtol=rtol,
        label="numeric part",
        method="quad",
    )

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s)
    value = norm_factor * data["value"]
    abserr = norm_factor * data["abserr"]

    return BesselIntegralResult(
        value=value,
        abserr=abserr,
        converged=abserr <= max(atol, rtol * np.fabs(value)),
        # plain quadrature has no phase-limited concept
        phase_limited=False,
    )


def _coefficient_sum(terms: Sequence[float]) -> float:
    """
    The correctly rounded sum of the three signed coefficients or zero-points of a phase group.

    math.fsum() is exact-then-round -- one rounding for the whole sum instead of one per addition
    -- and this is the one place in the group assembly where that is worth having: K = k + e_nu q
    + e_sigma s is a *cancellative* sum, so a doubly rounded K carries two roundings of the
    largest coefficient where a correctly rounded one carries half an ulp of the (small) result,
    and every evaluation then multiplies that error by x. It is called three times per group, at
    construction.

    The per-point sums deliberately do **not** use it: `theta`, `theta_deriv`, `theta_abserr` and
    `sin_cos`'s small angle add their terms in the ordinary way. Two reasons, in order of weight.
    (i) An ordinary sum of a few smooth terms errs by ~eps*max|term|, which is below the
    interpolation error of the residuals the terms carry, so compensated summation would buy
    nothing measurable -- DRAFT-PLAN.md Sec 8.2 asks for "appropriately accurate" summation, and
    for these terms this is it. This note used to name the rounding of the product K*x as the
    dominant error as well; since prompts/levin-phase-floor/06h, `sin_cos` carries that rounding
    exactly (TwoProduct) rather than summing it away, and `theta_abserr` declares the small sum's
    own rounding (gamma_5 times its terms' magnitudes). `theta` and `theta_deriv` still round
    K*x, and neither is a value the driver integrates as a phase. (ii) fsum() does not accept
    arrays, and adaptive_levin_sincos() samples theta' over the whole Chebyshev grid in one array
    call only if the callable's array result is *bit-identical* to its scalar results
    (levin_quadrature._detect_vectorized()); a scalar path that used fsum and an array path that
    could not would fail that probe and give up the array sampling this module's callables do in
    fact get -- measured 13.5 ms against 71.8 ms for 200 array samplings of a 13-point grid, and
    _sample_vectorized()'s docstring puts the loop at 20-37 % of a subregion evaluation. (Note
    that docstring's parenthetical "they do not, as of this commit" is stale: prompt 05's
    BesselPhaseFunction accessors are array-safe, so both this assembly and the one it replaced
    are detected as vectorising.)
    """
    return math.fsum(terms)


class BesselPhaseGroup:
    """
    One sum-and-difference phase group, held as ``theta_group(x) = K x + C + R(x)`` rather than as
    a sum of three reconstructed phases.

    Public since prompts/qsi-phase-groups' prompt 01, which routes ComputeTargets'
    QuadSourceIntegral._three_bessel_Levin through it as well: that call site had the same
    three-raw-phase summation this class replaced, and reimplementing it there would have been a
    second copy of the same arithmetic. The name is spelled out because
    ComputeTargets/phase_groups.py owns a *different* ``PhaseGroup`` -- a composition of
    cosmological phase_spline objects over log(1+z), with no analytic leading term to split off,
    for which summing raw phases is the correct construction. The Bessel leading term ``x`` is
    what makes the K x + C + R(x) split possible here, and the two must not be unified.
    ``_PhaseGroup`` remains bound to this class below, for the callers that used the private name.

    With ``theta_nu(y) = y + c_nu + r_nu(y)`` (LiouvilleGreen.bessel_phase, whose zero-point is
    ``c_nu = pi/4 - pi nu/2``),

        theta_mu(k x) + e_nu theta_nu(q x) + e_sigma theta_sigma(s x) = K x + C + R(x),
        K = k + e_nu q + e_sigma s,
        C = c_mu + e_nu c_nu + e_sigma c_sigma,
        R(x) = r_mu(k x) + e_nu r_nu(q x) + e_sigma r_sigma(s x),

    exactly. K and C are formed once, at construction, from the supplied coefficients and orders;
    only R is evaluated per point. DRAFT-PLAN.md Sec 8.2 and prompt 07.

    **Why, quantitatively.** The route this replaced summed three raw_theta values. Each is a
    double of size ~m x, so it carries an absolute error ~eps*m*x -- from the rounding of the
    product m*x, whose sensitivity in the phase is theta' ~ 1, and again from the rounding of the
    sum x + c + r. Those errors do not cancel with the group's signs, so a group whose own phase
    is small (a near-resonant K) was returned with the absolute error of its *largest constituent*:
    measured against 60-digit mpmath at orders (1/2, 3/2, 5/2), signs (+, -, -) and 34.5 <= x <=
    1e12, |delta theta_group| was 2.9e-5 rad for K = 0 exactly and 7.6e-5 for K/max(k,q,s) ~ 1e-10,
    in both cases attained at the top of the range. Assembled as above the same two cases measure
    1.4e-13 -- the residual interpolation error of the three constituents, attained at the *bottom*
    of the range and no longer growing with x -- for ratios of 2.0e8 and 5.4e8. Every number here
    is printed by test_three_bessel.TestPhaseGroups, which re-measures both routes rather than
    trusting this paragraph.

    **The product floor, and what replaced it.** Until prompts/levin-phase-floor's prompt 06h what
    was left was one product and one sum of doubles, so the group phase was accurate to
    ~eps*(|K| x + |C + R|): K itself is exact whenever k + e_nu q + e_sigma s is representable
    and is a single rounding otherwise, but the *product* K*x was rounded whatever K was. This
    docstring called that floor "a genuine limit" -- 1.5e-5 rad at K = 0.1, x = 1e12, one ulp of
    K x, and what both routes measured there (prompts/transfer-remedial/IMPLEMENTATION_STATE.md
    [07-generic-K-product-rounding]) -- and theta_abserr did not declare it, so the driver was
    never told: 06f measured it at up to 148 times the declaration on production's tail Levin
    leaves ([06f-theta-abserr-omits-the-product-rounding]). It is not a limit of the
    representation, only of rounding into one double. :meth:`sin_cos` now carries the product's
    rounding (TwoProduct, exact) and K's (``K_lo``, from fsum) in its small angle, so the phase it
    represents is ``K x + C + R(x)`` at the evaluated x to a few u of the small angle, whatever
    ``|K| x`` is. What remains, and is declared by :meth:`theta_abserr`, is the rounding of the
    abscissa ``x = fl(exp(u))`` -- ``eps_exp |dtheta/du|``, which is of the size the product floor
    was, but is a displacement of the point rather than an error in the phase there -- and the
    group's own evaluation. :meth:`theta` (the raw value) still rounds K x into one double, and is
    not what the driver evaluates.

    Input coefficient uncertainty is a separate matter and is not addressed here at all: if k, q, s
    are themselves uncertain, x delta K is a physical uncertainty of the group phase and no
    arithmetic makes it smaller (DRAFT-PLAN.md Sec 8.2).
    """

    def __init__(self, phases: Sequence, coefficients: Sequence, signs: Sequence):
        """
        :param phases: the three ``BesselPhaseFunction`` objects, in the order (mu, nu, sigma).
        :param coefficients: the three wavenumbers (k, q, s), so that factor ``i`` is evaluated at
            ``coefficients[i] * x``.
        :param signs: the three group signs (1.0, e_nu, e_sigma); the first is always +1.
        """
        self.phases = tuple(phases)
        self.coefficients = tuple(float(value) for value in coefficients)
        self.signs = tuple(float(value) for value in signs)

        # K and C are formed here, once, from the supplied coefficients -- before any
        # multiplication by x. This is the whole point of the class; see the class docstring.
        signed_coefficients = [
            sign * value for sign, value in zip(self.signs, self.coefficients)
        ]
        self.K = _coefficient_sum(signed_coefficients)
        # K's own rounding, K_lo = K - fl(K), carried into sin_cos's small angle as K_lo x
        # (prompts/levin-phase-floor/06h). Each term is a float times +-1, so exact, and fsum of
        # them and -fl(K) is K - fl(K) correctly rounded: in error by at most u |K_lo|, which
        # theta_abserr declares. Zero whenever k + e_nu q + e_sigma s is representable.
        self.K_lo = math.fsum(signed_coefficients + [-self.K])
        self.C = _coefficient_sum(
            [sign * phase.c_nu for sign, phase in zip(self.signs, self.phases)]
        )
        # c_nu reduced mod 2 pi, for the trigonometric path only: each term is then a single
        # rounding of a multiple of pi below 2 in magnitude (< 7e-16), whereas summing the
        # unreduced c_nu injects the ulp of the largest order. sin/cos do not care which
        # representative they are given.
        self.C_reduced = _coefficient_sum(
            [sign * phase.c_nu_reduced for sign, phase in zip(self.signs, self.phases)]
        )

    def arguments(self, x):
        """The three Bessel arguments ``(k x, q x, s x)`` at which the factors are evaluated."""
        return tuple(value * x for value in self.coefficients)

    def residual(self, x):
        """
        ``R(x)``, the signed sum of the three constituent residuals at their own arguments.

        Each ``r_nu`` is O(1) to O(nu) -- at most ~571 rad at the largest supported order
        (RECONCILIATION.md C2) -- and smooth, so this sum is well conditioned: its error is the
        residual interpolation error of the three phase objects, not eps*theta.
        """
        return sum(
            sign * phase.residual(argument)
            for sign, phase, argument in zip(self.signs, self.phases, self.arguments(x))
        )

    def theta(self, log_x):
        """
        ``theta_group`` as a single number, for the ``"theta"`` key.

        Required by adaptive_levin_sincos() (levin_quadrature.py:948-952 raises without it) but
        never evaluated by it when "theta_mod_2pi" and "theta_deriv" are both supplied (:1038).
        Assembled from the split rather than from three raw_theta values, so it is as accurate as
        a double of its size can be, but a caller wanting trigonometric values should use
        :meth:`sin_cos` or :meth:`theta_mod_2pi` instead.
        """
        x = np.exp(log_x)
        return self.K * x + self.C + self.residual(x)

    def sin_cos(self, log_x):
        """
        ``(sin theta_group, cos theta_group)`` by angle addition on a big angle ``fl(K x)`` and a
        small one, ``C_red + R(x) + p_lo + K_lo x``.

        **The big angle** ``fl(fl(K) x)`` is handed to sin/cos **unreduced**: the library reduces
        it against a multi-hundred-bit pi (Payne-Hanek), while any reduction performed here would
        use a 53-bit 2 pi and be strictly worse. This is the argument
        LiouvilleGreen/range_reduce_mod_2pi.py's module docstring makes, and prompt 05's
        BesselPhaseFunction.sin_cos_theta makes for the single-factor case. (Its result is
        faithful, not correctly rounded: 06h measured up to 0.85 ulp; theta_abserr declares it.)

        **The small angle carries what the big one loses** (prompts/levin-phase-floor/06h). The
        product's rounding ``p_lo = fl(K) x - fl(fl(K) x)`` is computed exactly by TwoProduct
        (_two_product_error: math.fma on a scalar, Dekker's split on an array, bit-identical), and
        K's own rounding ``K_lo = K - fl(K)``, held from construction, enters as ``K_lo x``. So
        big + small is ``K x + C + R(x)`` at the evaluated ``x``, modulo 2 pi, to the rounding of
        the small sum alone -- a few u of ``|C_red| + sum |r_i| + |p_lo| + |K_lo x|`` -- and no
        longer to ``u |K x|``. Before 06h the product's rounding was left in the big angle; it
        reached 3.9e-13 rad on 06f's call g-1e-7-0-12689 #19, 148 times that call's declared
        theta_abserr ([06f-theta-abserr-omits-the-product-rounding]).

        What remains is declared by :meth:`theta_abserr`: the small sum, the angle addition below,
        the rounding of each argument m_i x, and the rounding of x = exp(u) itself.

        The alternative -- summing the three constituents' bounded angles -- is not the bounded
        angle of the sum: each constituent's angle carries the eps*(m x) rounding of its own
        argument, and those enter the sum additively rather than cancelling with the group's
        signs. Measured on the same triples as the class docstring, before 06h, the worst |sin| /
        |cos| pair error against 60-digit mpmath was 3.05e-5 for the summed-angle route at K = 0
        and 1.39e-13 here; at a generic K = 0.1 the two were 1.42e-5 and 1.24e-5, i.e. both at the
        product floor 06h removes from this route.
        """
        x = np.exp(log_x)
        big = self.K * x
        # six summands, five additions, left to right (the gamma_5 that theta_abserr declares).
        # Not self.residual(x): that adds the three residuals with sum(), which since Python 3.12
        # compensates a sum of exact floats but not of arrays (or of np.float64), so its scalar
        # result can differ from its array result in the last bit (_ordered_sum's comment).
        small = self.C_reduced
        for sign, phase, argument in zip(self.signs, self.phases, self.arguments(x)):
            small = small + sign * phase.residual(argument)
        small = small + _two_product_error(self.K, x, big)
        small = small + self.K_lo * x

        if np.isscalar(big) or np.ndim(big) == 0:
            sin_big, cos_big = math.sin(big), math.cos(big)
            sin_small, cos_small = math.sin(small), math.cos(small)
        else:
            sin_big, cos_big = np.sin(big), np.cos(big)
            sin_small, cos_small = np.sin(small), np.cos(small)

        return (
            sin_big * cos_small + cos_big * sin_small,
            cos_big * cos_small - sin_big * sin_small,
        )

    def theta_mod_2pi(self, log_x):
        """A bounded representative of ``theta_group`` in (-pi, pi], from atan2 of :meth:`sin_cos`."""
        sin_theta, cos_theta = self.sin_cos(log_x)
        if np.isscalar(sin_theta) or np.ndim(sin_theta) == 0:
            return math.atan2(sin_theta, cos_theta)
        return np.arctan2(sin_theta, cos_theta)

    def theta_deriv(self, log_x):
        """
        ``d theta_group / d log x = K x + sum_i e_i (dr_i/d log y)|_(y = m_i x)``, from the same
        expression as the value.

        Each factor contributes ``d/d log x theta_i(m_i x) = (m_i x) theta_i'(m_i x)``, and
        ``log(m_i x) = log m_i + log x``, so the constituent log-derivative evaluated at ``m_i x``
        is exactly that. The route this replaced summed those three log-derivatives directly. They
        are individually well conditioned -- prompt 05 supplies theta' as exp(-2 ell), a value read
        off the amplitude interpolant rather than a differentiated phase -- but each is a double of
        size ~m x, so the *sum* again inherits eps*max(m x) and loses the group's cancellation.
        Splitting off K x, which is exact in K, keeps it: measured against 60-digit mpmath at
        orders (1/2, 3/2, 5/2), signs (+, -, -) and 34.5 <= x <= 1e12, |delta d theta/d log x|
        falls from 3.05e-5 to 1.00e-12 at K = 0 exactly (a factor 3.0e7) and from 1.72e-5 to
        1.49e-8 for a generic K = 0.1, where the eps*|K| x product floor of the class docstring
        binds. Scaled to the constituent frequencies as README Sec 6 asks -- divided by
        max(k, q, s) x -- the two routes are instead *equal* at 1.45e-14, because that metric is
        maximised at the bottom of the near region where x is small, there is no leading term to
        cancel, and both routes sit on the interpolation floor. The sharpest statement is the one
        at exact resonance: at x = 1e12 the group log-derivative is 6.666667e-12, which this route
        returns to the last bit and the route it replaced gets wrong by 100 % of itself.

        The dr/d log x route is used here and *only* here. It differentiates an interpolant, so it
        is the weaker of prompt 05's two derivative primitives away from the tail
        (IMPLEMENTATION_STATE.md standing note 17), but it enters only as a correction to K x: at
        the bottom of the near region, where its relative error is worst, x is smallest, and the
        1.5e-11 measured above is that product. Levin uses theta' for basis conditioning *and* for
        deciding subdivision (phase_span at levin_quadrature.py:1090), so this is the group
        quantity it is most sensitive to.
        """
        x = np.exp(log_x)
        return self.K * x + sum(
            sign * phase.residual_log_deriv(argument)
            for sign, phase, argument in zip(self.signs, self.phases, self.arguments(x))
        )

    def theta_abserr(self, log_x):
        """
        The declared absolute error, in radians, of the phase :meth:`theta_mod_2pi` returns at
        ``u = log_x``, **as a function of u**: against ``theta_group(e^u)`` at the exact ``e^u``,
        which is the phase the Levin driver integrates in u. Three parts, combined linearly
        (prompts/levin-phase-floor/06h):

            theta_abserr(u) = sum_i theta_abserr_at_i(m_i x)  +  eps_exp |dtheta/du|  +  phi_eval(u),

        with ``x = fl(exp(u))``.

        * **The constituents' declarations** at their own arguments, as before 06h: each
          factor's representation error at the floating-point argument it is given, with its
          EVALUATION_FLOOR and the rounding of its c_nu_reduced.
        * **The abscissa.** The group is evaluated at ``x = e^u (1 + delta_e)``, not at ``e^u``,
          which moves the phase by ``(dtheta/du) delta_e`` to first order (the second-order term,
          ``(d^2 theta/du^2) delta_e^2 / 2 ~ |K x| u^2``, is below a ulp of anything here).
          ``bessel_phase`` defines a single factor's evaluation to be at the computed exp(u),
          which is right for its representation; but the function of u the driver integrates is
          this one, and an error in its value belongs here (levin_quadrature.py, "What it does not
          carry"). ``|delta_e| <= eps_exp`` (:data:`EXP_RELERR`, measured), and ``|dtheta/du|`` is
          bounded by ``(|fl(K)| + |K_lo|) x + 2 sum_i |dr_i/d log y|`` at ``m_i x`` -- data the class
          holds, **not** :meth:`theta_deriv`, whose signed sum can cancel; the factor 2 is
          :data:`_LOG_DERIV_ALLOWANCE`, for the evaluation of dr/d log y itself.
        * **phi_eval, the group's own evaluation**, term by term:
          - the small angle's sum, gamma_5 times the six summands' magnitudes, ``|C_red| +
            sum_i |r_i| + |p_lo| + |K_lo x|``, with ``|p_lo| <= u |fl(K)| x``
            (:data:`_SMALL_ANGLE_GAMMA`), plus ``u |C_red|``, the rounding of C_red's own fsum;
          - K_lo's: ``2 u |K_lo| x``, one u for the fsum that forms K_lo and one for the product;
          - the arguments: ``fl(m_i x)`` is in error by a relative u, which moves ``r_i`` by at
            most ``u |dr_i/d log y|``, with the same allowance of 2;
          - the angle addition of :meth:`sin_cos`, :data:`_ANGLE_ADDITION_PHASE_ERR` (12u), which
            no constituent declares (the constituents' EVALUATION_FLOOR covers their *own* angle
            additions, which the group never performs);
          - the atan2 of :meth:`theta_mod_2pi`, :data:`_ATAN2_PHASE_ERR` (8u).
          The roundings of fl(K x) and of K are *not* declared: :meth:`sin_cos` carries both
          exactly. The rounding of each c_nu_reduced is not declared twice: it is the
          constituents'.

        Combined **linearly**, not in quadrature, for the reason BesselIntegralResult's docstring
        gives for the group values: the three phases share a construction, so a systematically
        inaccurate phase produces a common drift rather than independent noise. The terms are also
        of very different sizes -- a factor lying in its closed-form tail declares ~1e-15 while one
        in its sampled near region declares ~5e-12, and the abscissa term is ~eps_exp |K| x -- so a
        quadrature sum would be dominated by the largest anyway.

        A scalar ``log_x`` gives a float, an array an array, bit-identical point by point.

        This is what adaptive_levin_sincos()'s "theta_abserr" key wants (levin_quadrature.py:962,
        :2360). Supplying it makes the reported abserr *larger*, not smaller: it is the statement
        that the phase construction, not the Levin rule, is the limit. See BesselIntegralResult.
        """
        constituents, abscissa, phi_eval = self._declared_error_parts(log_x)
        total = constituents + abscissa + phi_eval
        if np.ndim(total) == 0:
            return float(total)
        return total

    def _declared_error_parts(self, log_x):
        """
        The three parts of :meth:`theta_abserr` at ``u = log_x``, separately: ``(constituents,
        abscissa, phi_eval)``, the sum of the constituents' declarations, ``eps_exp`` times the
        bound on ``|dtheta/du|``, and the group's own evaluation. See :meth:`theta_abserr` for
        each, and the module's constants for their derivations.
        """
        x = np.exp(log_x)
        arguments = self.arguments(x)
        u = _UNIT_ROUNDOFF

        constituents = _ordered_sum(
            phase.theta_abserr_at(argument)
            for phase, argument in zip(self.phases, arguments)
        )
        residual_magnitude = _ordered_sum(
            np.abs(phase.residual(argument))
            for phase, argument in zip(self.phases, arguments)
        )
        log_deriv_magnitude = _LOG_DERIV_ALLOWANCE * _ordered_sum(
            np.abs(phase.residual_log_deriv(argument))
            for phase, argument in zip(self.phases, arguments)
        )
        K_hat_x = abs(self.K) * x
        K_lo_x = abs(self.K_lo) * x

        # the abscissa: eps_exp times a bound on |dtheta/du|
        abscissa = EXP_RELERR * ((K_hat_x + K_lo_x) + log_deriv_magnitude)

        # phi_eval, term by term (see the docstring and the constants' comments)
        small_angle = _SMALL_ANGLE_GAMMA * (
            abs(self.C_reduced) + residual_magnitude + u * K_hat_x + K_lo_x
        ) + u * abs(self.C_reduced)
        K_lo_rounding = 2.0 * u * K_lo_x
        argument_rounding = u * log_deriv_magnitude
        phi_eval = (
            small_angle
            + K_lo_rounding
            + argument_rounding
            + (_ANGLE_ADDITION_PHASE_ERR + _ATAN2_PHASE_ERR)
        )
        return constituents, abscissa, phi_eval

    def levin_theta(self) -> dict:
        """
        The phase dictionary adaptive_levin_sincos() expects. The integration variable is log(x).

        theta' is supplied explicitly rather than left to the Levin driver's spectral
        differentiation of theta: at large argument the raw phase is a large float whose absolute
        resolution is ~eps*theta, so differentiating sampled values of it loses precision in
        proportion to theta/(phase change across the subinterval). "theta_abserr" is new in prompt
        07 and is the group's own declared error.
        """
        return {
            "theta": self.theta,
            "theta_mod_2pi": self.theta_mod_2pi,
            "theta_deriv": self.theta_deriv,
            "theta_abserr": self.theta_abserr,
        }


#: The name this class carried when it was introduced (prompts/transfer-remedial prompt 07), kept
#: bound so that every existing caller and test continues to work unchanged. New code should use
#: :class:`BesselPhaseGroup`.
_PhaseGroup = BesselPhaseGroup


def _phase_group(
    phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma
) -> BesselPhaseGroup:
    """Assemble the :class:`BesselPhaseGroup` for one sum-and-difference sign pair."""
    return BesselPhaseGroup(
        phases=(phase_mu, phase_nu, phase_sigma),
        coefficients=(k, q, s),
        signs=(1.0, e_nu, e_sigma),
    )


def _Levin_3bessel(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    weights,
    combination,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    """
    Shared driver for the JJJ and YJJ three-Bessel integrals. These differ only in which slot of the
    (sin, cos) basis the slowly varying amplitude occupies, and in the signs with which the four
    sum-and-difference groups are recombined.

    :param weights: callable mapping the amplitude to the Levin f-vector
    :param combination: signs with which the four group values are combined
    """
    phase_mu = mu_phase["phase"]
    phase_nu = nu_phase["phase"]
    phase_sigma = sigma_phase["phase"]

    m_mu = mu_phase["mod"]
    m_nu = nu_phase["mod"]
    m_sigma = sigma_phase["mod"]

    x_span = (np.log(min_x), np.log(max_x))

    def Levin_f(log_x: float):
        x = np.exp(log_x)

        return np.pow(x, 3.0 / 2.0) * m_mu(k * x) * m_nu(q * x) * m_sigma(s * x)

    total = 0.0
    abserr_total = 0.0
    converged = True
    phase_limited = False
    for index, (e_nu, e_sigma) in enumerate(_PHASE_GROUP_SIGNS):
        data = adaptive_levin_sincos(
            x_span,
            f=weights(Levin_f),
            theta=_phase_group(
                phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma
            ).levin_theta(),
            atol=atol,
            rtol=rtol,
            chebyshev_order=chebyshev_order,
            notify_label=f"phase{index + 1}",
        )
        total = total + combination[index] * data["value"]
        # combined linearly, not in quadrature: the four groups share a phase construction, so a
        # systematically inaccurate phase produces a common drift rather than independent noise --
        # see BesselIntegralResult's docstring
        abserr_total = abserr_total + np.fabs(combination[index]) * data["abserr"]
        converged = converged and data["converged"]
        phase_limited = phase_limited or data["phase_limited"]

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) / 4.0

    return BesselIntegralResult(
        value=norm_factor * total,
        abserr=norm_factor * abserr_total,
        converged=converged,
        phase_limited=phase_limited,
    )


def _Levin_JJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    # J_nu = m sin(theta), so the amplitude sits in the sine slot
    return _Levin_3bessel(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_x,
        max_x,
        atol,
        rtol,
        weights=lambda amplitude: [amplitude, lambda x: 0.0],
        combination=(-1.0, 1.0, 1.0, -1.0),
        chebyshev_order=chebyshev_order,
    )


def quad_YJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    min_x_mu = mu_phase["min_x"]
    min_x_nu = nu_phase["min_x"]
    min_x_sigma = sigma_phase["min_x"]

    # cut integral in two, as in quad_JJJ: see documentation there
    x_cut = max(min_x_mu, min_x_nu, min_x_sigma)
    min_kqs = min(k, q, s)
    min_cut = x_cut / min_kqs

    numeric = _direct_YJJ(mu, nu, sigma, k, q, s, 0.0, min_cut, atol, rtol)
    Levin = _Levin_YJJ(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_cut,
        max_x,
        atol,
        rtol,
        chebyshev_order=chebyshev_order,
    )

    return BesselIntegralResult(
        value=numeric.value + Levin.value,
        abserr=numeric.abserr + Levin.abserr,
        converged=numeric.converged and Levin.converged,
        phase_limited=Levin.phase_limited,
    )


def _direct_YJJ(
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
):
    def integrand(x):
        return (
            np.sqrt(x)
            * yv(mu + 0.5, k * x)
            * jv(nu + 0.5, q * x)
            * jv(sigma + 0.5, s * x)
        )

    data = simple_quadrature(
        integrand,
        a=min_x,
        b=max_x,
        atol=atol,
        rtol=rtol,
        label="numeric part",
        method="quad",
    )

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s)
    value = norm_factor * data["value"]
    abserr = norm_factor * data["abserr"]

    return BesselIntegralResult(
        value=value,
        abserr=abserr,
        converged=abserr <= max(atol, rtol * np.fabs(value)),
        phase_limited=False,
    )


def _Levin_YJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    # Y_nu = -m cos(theta), so relative to JJJ the amplitude moves to the cosine slot and the four
    # groups recombine with the opposite signs
    return _Levin_3bessel(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_x,
        max_x,
        atol,
        rtol,
        weights=lambda amplitude: [lambda x: 0.0, amplitude],
        combination=(1.0, -1.0, -1.0, 1.0),
        chebyshev_order=chebyshev_order,
    )
