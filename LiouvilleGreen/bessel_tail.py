"""
Closed-form large-argument tail for the Bessel amplitude and phase construction.

This module supplies the **tail region** of the two-region construction of
``prompts/transfer-remedial/DRAFT-PLAN.md`` §1 and §7.2: above a crossover ``x_star(nu)`` the
normalized amplitude ``a_nu`` and the phase residual ``r_nu`` are evaluated directly from a
fixed-order asymptotic expansion. There are no samples, no interpolation, no branch tracking and
no cycle-count representation here, and there is deliberately no call to any SciPy Bessel routine:
``scipy.special`` is not imported.

Convention
----------

The repository's Bessel convention (which is **not** DLMF's; the two differ by exactly ``+pi/2``,
absorbed into ``c_nu``) is

    J_nu(x) = A_nu(x) sin theta_nu(x),        Y_nu(x) = -A_nu(x) cos theta_nu(x),

with ``theta_nu`` increasing in ``x`` and

    theta_nu(x) = x + c_nu + r_nu(x),   A_nu(x) = sqrt(2/(pi x)) a_nu(x),   c_nu = pi/4 - pi nu/2.

Do not "correct" this towards DLMF's convention.

Why the tail is mandatory, not an optimization
----------------------------------------------

Two independent reasons, both measured (``prompts/transfer-remedial/README.md`` §2 (e) and
``RECONCILIATION.md`` C1):

1. ``scipy.special.hankel1e`` fails *silently* above ``x ~ 7.13e8`` for ``nu >~ 100`` and above
   ``x ~ 2.247e15`` for every order -- it returns exactly ``-0j``, which ``isfinite`` passes and
   ``log(abs(.))`` turns into ``-inf``. With the crossover at or below ``100 nu`` the near-region
   sampler is never asked for a value within three decades of that boundary.
2. ``jv``/``yv`` become O(1)-relatively noisy above ``x ~ 2.5e15``, which is what stalls the
   existing phase ODE. Nothing here evaluates a Bessel routine at all, so the supported ``x_max``
   is limited by ``sin``/``cos`` (correctly rounded to ``1e16``) rather than by Amos.

The series
----------

With ``mu = 4 nu^2``, the phase residual in this repository's convention is DLMF 10.18.18
(equivalently Abramowitz & Stegun 9.2.29, whose ``(8x)^k`` grouping is the form the coefficients
below were checked against):

    r_nu(x) ~   (mu - 1)/(8 x)
              + 4 (mu - 1)(mu - 25) / (3 (8x)^3)
              + 32 (mu - 1)(mu^2 - 114 mu + 1073) / (5 (8x)^5)
              + 64 (mu - 1)(5 mu^3 - 1535 mu^2 + 54703 mu - 375733) / (7 (8x)^7)
              + ...

which reduces to the flat-denominator form used by :func:`tail_series_coefficients`,

    r_nu(x) ~   (mu - 1)/(8 x)
              + (mu - 1)(mu - 25) / (384 x^3)
              + (mu - 1)(mu^2 - 114 mu + 1073) / (5120 x^5)
              + (mu - 1)(5 mu^3 - 1535 mu^2 + 54703 mu - 375733) / (229376 x^7)
              + ...

**Note the third denominator is 5120, not 15360.** ``DRAFT-PLAN.md`` §7.2 and campaign prompt 03
§2 both print ``15360``, which is wrong by a factor of three; the campaign log
``prompts/transfer-remedial/logs/03-closed-form-tail.md`` records the numerical determination.
Briefly: fitting ``(r_true - two-term series) x^5`` against 120-digit ``mpmath`` gives 0.2000000
(nu=3/2), -5.4000000 (5/2) and 864675.13 (20.5), against numerators 1024, -27648 and 4427136000 --
ratios 5120.0 in all three cases, and ``32/(5 * 8^5) = 1/5120`` exactly. The fourth coefficient was
confirmed the same way: 229376, from ``64/(7 * 8^7)``.

**The amplitude needs no second series.** The Wronskian ``A^2 theta' = 2/(pi x)`` gives
``theta' = a^-2`` exactly, and ``theta = x + c_nu + r`` gives ``theta' = 1 + r'``, hence

    a_nu(x) = (1 + r_nu'(x))^(-1/2),

with ``r_nu'`` obtained by differentiating the expansion above term by term. DLMF 10.18.17 is not
required and is deliberately not used: one series governing both quantities is what makes the
crossover test on the phase cover the amplitude automatically.

``nu = 1/2`` is exactly degenerate: ``mu - 1 = 0`` in floating point (``4 * 0.5 * 0.5 == 1.0``
exactly), so ``r == 0``, ``r' == 0`` and ``a == 1`` **identically at every x**, not merely
asymptotically. :func:`tail_crossover` special-cases it so that the whole domain is closed form and
the near-region sampler never runs.

Evaluation form
---------------

Terms fall like ``x^-1, x^-3, x^-5``, so the series is evaluated by **Horner in ``w = 1/x^2``**,

    r = (1/x) (c0 + w (c1 + w c2)),

rather than by forming ``x**3`` and ``x**5`` independently. Three reasons, in order of weight:
``x**7`` overflows for ``x > 1.1e44`` while ``w**3`` merely underflows; Horner uses one rounding
per term instead of a power plus a division; and it accumulates the small corrections onto the
large leading term last, so the additions are of the "big plus tiny" kind that round exactly rather
than of the cancelling kind. Summing smallest-first would share the second and third properties but
not the first, and would need the same powers of ``w`` anyway. The derivative and the
log-derivative use the same nesting with the term index folded into the coefficients.

What lives elsewhere
--------------------

The *other* half of the remainder-tested crossover -- checking that the near-region interpolant and
this series agree at ``x_star`` -- belongs to campaign prompt 05, which stitches the two regions.
This module owns only the series side: how small the first omitted term is, and where.
"""

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np

#: Number of series terms this module can *evaluate*.
#:
#: A fourth coefficient is carried (see :func:`tail_series_coefficients`) but is used only by
#: :func:`tail_first_omitted_term`, so that the remainder test of :func:`tail_crossover` still has
#: an estimator at the maximum implemented order.
TAIL_SERIES_MAX_TERMS = 3

#: Number of coefficients held, i.e. :data:`TAIL_SERIES_MAX_TERMS` plus the omitted-term estimator.
TAIL_SERIES_COEFFICIENT_COUNT = TAIL_SERIES_MAX_TERMS + 1

#: Default number of terms used by every public entry point.
DEFAULT_TAIL_TERMS = TAIL_SERIES_MAX_TERMS

#: Fraction of the requested budget the first omitted term must fall below, in
#: :func:`tail_crossover`.
#:
#: The first omitted term is an *estimator* of an asymptotic remainder, not a bound on it. Measured
#: against 60-digit ``mpmath`` at the crossover the true ``|delta r|`` is 0.98--1.02 times the first
#: omitted term (campaign log 03), so requiring equality with the budget lands the measured error
#: marginally *above* it. A factor of four costs ``4^(1/7) = 1.22`` in ``x_star``, i.e. 0.2 of an
#: e-fold of extra near region, and buys a measured margin of ~4 rather than ~0.98.
DEFAULT_CROSSOVER_SAFETY = 0.25

#: Upper end of ``DRAFT-PLAN.md`` §5.3's safe matching window, as a multiple of ``max(nu, 1)``.
#:
#: Beyond ``4800 nu`` the cancellation floor of the residual right-hand side ``(2/pi)/m - x``
#: exceeds ``1e-8`` (``DRAFT-PLAN.md`` §5.2, §5.3), so a crossover placed further out could not be
#: cross-checked against a near-region construction even in principle; and every e-fold below
#: ``x_star`` has to be sampled and branch-tracked, so a crossover that far out is not a design
#: anyone wants. :func:`tail_crossover` refuses rather than silently extending past it.
SAFE_WINDOW_MAX_FACTOR = 4800.0

#: Lower bound of the domain for ``nu <= 1/2``, where ``sqrt(nu^2 - 1/4)`` degenerates to zero.
#: Mirrors ``LiouvilleGreen.bessel_phase`` and ``LiouvilleGreen.tests.bessel_reference``.
DEGENERATE_MIN_X = 1e-5


def construction_min_x(nu: float) -> float:
    """
    The lower edge of the domain on which a phase function is provided: ``sqrt(nu^2 - 1/4)`` for
    ``nu > 1/2`` (the turning point of ``omega_eff``, below which the solution is exponential
    rather than oscillatory) and :data:`DEGENERATE_MIN_X` for ``nu <= 1/2``.

    Duplicated from ``LiouvilleGreen.tests.bessel_reference.construction_min_x`` rather than
    imported, because production code must not depend on a test module. The two must agree; the
    campaign's prompt 05 is where they are used together.
    """
    if nu > 0.5:
        return math.sqrt(nu * nu - 0.25)
    return DEGENERATE_MIN_X


def tail_crossover_max_x(nu: float) -> float:
    """The default ceiling on ``x_star``: :data:`SAFE_WINDOW_MAX_FACTOR` times ``max(nu, 1)``."""
    return SAFE_WINDOW_MAX_FACTOR * max(nu, 1.0)


def tail_series_coefficients(nu: float) -> Tuple[float, float, float, float]:
    """
    The four DLMF 10.18.18 coefficients ``(c0, c1, c2, c3)`` for this order, where

        r_nu(x) ~ c0/x + c1/x^3 + c2/x^5 + c3/x^7 + ...

    ``c3`` is the omitted-term estimator: it is never summed into a returned residual, only used by
    :func:`tail_first_omitted_term`. See the module docstring for the provenance of the 5120 and
    229376 denominators, which differ from the ones printed in ``DRAFT-PLAN.md`` §7.2.

    ``mu - 1`` is a common factor of all four, so every coefficient vanishes identically at
    ``nu = 1/2``.
    """
    mu = 4.0 * nu * nu
    mu1 = mu - 1.0

    return (
        mu1 / 8.0,
        mu1 * (mu - 25.0) / 384.0,
        mu1 * (mu * mu - 114.0 * mu + 1073.0) / 5120.0,
        mu1
        * (5.0 * mu * mu * mu - 1535.0 * mu * mu + 54703.0 * mu - 375733.0)
        / 229376.0,
    )


def _check_terms(n_terms: int) -> int:
    if (
        not isinstance(n_terms, (int, np.integer))
        or not 1 <= n_terms <= TAIL_SERIES_MAX_TERMS
    ):
        raise ValueError(
            f"bessel_tail: n_terms must be an integer in [1, {TAIL_SERIES_MAX_TERMS}], "
            f"got {n_terms!r}. A fourth coefficient exists but is reserved for "
            f"tail_first_omitted_term, so that the remainder test still has an estimator at the "
            f"maximum implemented order."
        )
    return int(n_terms)


def _as_array(x) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)

    if np.any(arr <= 0.0):
        raise ValueError(
            "bessel_tail: the large-argument expansion is defined for x > 0 only; got "
            f"min(x)={float(np.min(arr)):.6g}."
        )

    return arr, scalar


def _restore(values: np.ndarray, scalar: bool):
    if scalar:
        return float(values[0])
    return values


def _horner(coefficients: Tuple[float, ...], w: np.ndarray) -> np.ndarray:
    """Evaluate ``coefficients[0] + w (coefficients[1] + w (...))`` by Horner in ``w = 1/x^2``."""
    acc = np.full_like(w, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        acc = coefficient + w * acc
    return acc


def tail_residual(nu: float, x, n_terms: int = DEFAULT_TAIL_TERMS):
    """
    The phase residual ``r_nu(x) = theta_nu(x) - x - c_nu`` from the large-argument expansion.

    Accepts a scalar or an array of ``x`` and returns the same shape. Exactly zero at ``nu = 1/2``.

    **This value is not reduced mod 2 pi and must never be.** At ``nu = 1000.5`` the residual is
    570.82 rad at the bottom of the domain and 5.0 rad at ``100 nu``
    (``RECONCILIATION.md`` C2), so folding it into ``(-pi, pi]`` would produce a value wrong by an
    exact multiple of ``2 pi`` -- the easiest way to poison every downstream comparison.
    """
    n = _check_terms(n_terms)
    arr, scalar = _as_array(x)
    coefficients = tail_series_coefficients(nu)[:n]

    w = 1.0 / (arr * arr)
    return _restore(_horner(coefficients, w) / arr, scalar)


def tail_residual_deriv(nu: float, x, n_terms: int = DEFAULT_TAIL_TERMS):
    """
    ``dr_nu/dx``, from term-by-term differentiation of :func:`tail_residual`:

        r' = -(1/x^2) (c0 + 3 w c1 + 5 w^2 c2),      w = 1/x^2.

    This is the quantity the tail amplitude is built from, via ``a = (1 + r')^(-1/2)``; it is not
    an independent approximation.
    """
    n = _check_terms(n_terms)
    arr, scalar = _as_array(x)
    coefficients = tail_series_coefficients(nu)[:n]
    weighted = tuple((2 * k + 1) * c for k, c in enumerate(coefficients))

    w = 1.0 / (arr * arr)
    return _restore(-_horner(weighted, w) * w, scalar)


def tail_residual_log_deriv(nu: float, x, n_terms: int = DEFAULT_TAIL_TERMS):
    """
    ``dr_nu/d(log x) = x r_nu'(x)``.

    Provided as a primitive because the near region interpolates in ``u = log x``, so prompt 05's
    crossover check compares this quantity rather than ``r'`` itself.
    """
    n = _check_terms(n_terms)
    arr, scalar = _as_array(x)
    coefficients = tail_series_coefficients(nu)[:n]
    weighted = tuple((2 * k + 1) * c for k, c in enumerate(coefficients))

    w = 1.0 / (arr * arr)
    return _restore(-_horner(weighted, w) / arr, scalar)


def tail_amplitude(nu: float, x, n_terms: int = DEFAULT_TAIL_TERMS):
    """
    The normalized amplitude ``a_nu(x) = (1 + r_nu'(x))^(-1/2)``, so that
    ``A_nu = sqrt(2/(pi x)) a_nu``.

    Exact consequence of the Wronskian ``A^2 theta' = 2/(pi x)`` together with
    ``theta = x + c_nu + r``; no second series is involved, and DLMF 10.18.17 is deliberately not
    used. Exactly 1 at ``nu = 1/2``.

    :raises ValueError: if ``1 + r' <= 0`` anywhere in ``x``. That happens only well below the
        region where the asymptotic series means anything -- at ``nu = 1000.5`` the three-term
        ``r'`` first exceeds 1 in magnitude near ``x = 2 nu`` -- and the alternative is a silent
        ``nan`` from a negative base, which is exactly the class of failure this campaign exists
        to remove. ``theta' = 1 + r'`` is also positive throughout the oscillatory region as a
        matter of fact, so a non-positive value is a statement about the series, not the function.
    """
    deriv = tail_residual_deriv(nu, x, n_terms=n_terms)
    one_plus = 1.0 + deriv

    if np.any(np.asarray(one_plus) <= 0.0):
        bad = np.min(np.asarray(one_plus))
        raise ValueError(
            f"tail_amplitude: 1 + r' = {float(bad):.6g} <= 0 for nu={nu} with {n_terms} series "
            f"term(s). The large-argument expansion has been evaluated far below its region of "
            f"validity; use tail_crossover to place the crossover and the near-region "
            f"construction below it."
        )

    if isinstance(deriv, float):
        return float(np.power(one_plus, -0.5))
    return np.power(one_plus, -0.5)


def tail_first_omitted_term(nu: float, x, n_terms: int = DEFAULT_TAIL_TERMS):
    """
    The magnitude of the first term the expansion does **not** include, ``|c_n| / x^(2n+1)``.

    This is the estimator the remainder test of :func:`tail_crossover` is built on. It is an
    estimator and not a bound: measured against 60-digit ``mpmath`` at the crossover the true
    ``|delta r|`` is 0.98--1.02 times this value, which is why :data:`DEFAULT_CROSSOVER_SAFETY`
    exists.

    It is zero at ``nu = 1/2``, where every coefficient vanishes, and **it can also be zero at an
    isolated order where that one coefficient happens to vanish while the series does not**: with
    ``n_terms = 1`` and ``nu = 5/2`` the polynomial ``mu - 25`` is exactly zero, so this function
    returns 0 although the true one-term error is 1.8e-10 at ``x = 50 nu``. At the shipped default
    of three terms the estimator is the ``c3`` coefficient, which does not vanish at any order the
    campaign uses; :func:`tail_crossover` refuses rather than believing a zero it did not get from
    ``mu = 1``.
    """
    n = _check_terms(n_terms)
    arr, scalar = _as_array(x)
    coefficient = abs(tail_series_coefficients(nu)[n])

    # Written as a power of 1/x rather than as a division by x**(2n+1): the tail runs to
    # x ~ 1e16 and beyond, where x**7 is still finite but x**(2n+1) for a longer series would
    # not be, and an underflow to zero is the right answer for a term this far out whereas an
    # overflow to inf is not.
    return _restore(coefficient * np.power(1.0 / arr, 2 * n + 1), scalar)


class TailCrossover(NamedTuple):
    """
    The outcome of the remainder test of :func:`tail_crossover`.

    :ivar nu: the order the test was run for.
    :ivar x_star: the crossover. The tail representation is used for ``x >= x_star``; everything
        below it must be sampled and branch-tracked by the near region.
    :ivar first_omitted_at_x_star: ``tail_first_omitted_term(nu, x_star, terms_used)``, the
        estimated series remainder there. Compare against ``phase_atol``, not against
        ``safety * phase_atol``: the safety factor is a construction margin, not a claim.
    :ivar terms_used: number of series terms the crossover was sized for; the same ``n_terms`` must
        be passed to :func:`tail_residual` and friends for the result to mean anything.
    :ivar phase_atol: the absolute phase budget requested, in radians.
    :ivar amplitude_rtol: the relative amplitude budget requested.
    :ivar safety: the margin factor applied to both budgets (:data:`DEFAULT_CROSSOVER_SAFETY`).
    :ivar x_phase: the smallest ``x`` satisfying the phase condition alone.
    :ivar x_amplitude: the smallest ``x`` satisfying the amplitude condition alone.
    :ivar amplitude_error_at_x_star: the estimated ``|delta a / a|`` at ``x_star``.
    :ivar binding: which condition set ``x_star`` -- ``"phase"``, ``"amplitude"`` or ``"domain"``
        (the last meaning the series already met both budgets at the bottom of the domain, as it
        does identically for ``nu = 1/2``).
    """

    nu: float
    x_star: float
    first_omitted_at_x_star: float
    terms_used: int
    phase_atol: float
    amplitude_rtol: float
    safety: float
    x_phase: float
    x_amplitude: float
    amplitude_error_at_x_star: float
    binding: str


def tail_crossover(
    nu: float,
    phase_atol: float,
    amplitude_rtol: float,
    n_terms: int = DEFAULT_TAIL_TERMS,
    x_max: Optional[float] = None,
    safety: float = DEFAULT_CROSSOVER_SAFETY,
) -> TailCrossover:
    """
    Choose the crossover ``x_star(nu)`` by a remainder test at the requested accuracy.

    ``DRAFT-PLAN.md`` §7.2 is explicit that ``x_star`` must be neither a fixed constant nor a fixed
    multiple of ``nu``: the series is asymptotic, so its remainder must be *tested*. The test here
    is the size of the first omitted term, and ``x_star`` is the **smallest** ``x`` that passes it,
    not a comfortable large value -- every e-fold below ``x_star`` has to be sampled and
    branch-tracked by the near region, so making the tail as wide as the budget allows is the whole
    economy of the two-region design.

    Two conditions are applied, both with the :data:`DEFAULT_CROSSOVER_SAFETY` margin:

    * **phase**: ``|c_n| / x^(2n+1) <= safety * phase_atol``;
    * **amplitude**: ``a = (1 + r')^(-1/2)`` gives ``|delta a / a| ~ |delta r'| / 2``, and the
      omitted term contributes ``|delta r'| = (2n+1) |c_n| / x^(2n+2)``, so the condition is
      ``(2n+1) |c_n| / (2 x^(2n+2)) <= safety * amplitude_rtol``.

    Each term is monotone decreasing in ``x``, so both conditions invert in closed form and the
    smallest admissible ``x`` is exact rather than searched. The amplitude condition is weaker than
    the phase one by a factor ``~2x`` and never binds at the campaign's budgets; it is applied
    anyway, and ``binding`` records which one won.

    ``nu = 1/2`` is special-cased on ``mu - 1 == 0`` rather than left to the numeric test: every
    coefficient is then exactly zero, the series is exact at every ``x``, and the correct crossover
    is the bottom of the domain so that the near-region sampler never runs. Leaving it to the
    numeric test would give the same answer here, but only because ``0 <= safety * atol`` happens
    to hold for a positive budget -- an explicit branch says what is meant.

    :param x_max: ceiling on ``x_star``; defaults to :func:`tail_crossover_max_x`. If neither
        condition can be met below it the function **raises** rather than silently extending, so
        that prompt 05 can fail construction loudly rather than accept a tail it cannot certify.
    :raises ValueError: if the budgets are not positive, or cannot be met below ``x_max``.
    """
    n = _check_terms(n_terms)

    if not (phase_atol > 0.0) or not (amplitude_rtol > 0.0):
        raise ValueError(
            f"tail_crossover: budgets must be positive, got phase_atol={phase_atol!r}, "
            f"amplitude_rtol={amplitude_rtol!r}."
        )
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"tail_crossover: safety must lie in (0, 1], got {safety!r}.")

    x_min = construction_min_x(nu)
    if x_max is None:
        x_max = tail_crossover_max_x(nu)
    if x_max < x_min:
        raise ValueError(
            f"tail_crossover: x_max={x_max:.6g} is below the domain lower bound "
            f"x_min={x_min:.6g} for nu={nu}."
        )

    coefficient = abs(tail_series_coefficients(nu)[n])
    power = 2 * n + 1

    if 4.0 * nu * nu - 1.0 == 0.0:
        # nu = 1/2 exactly: mu - 1 = 0, so every coefficient vanishes and r == 0, a == 1
        # identically at every x. The series is not merely accurate here, it is exact, so the
        # whole domain is closed form and the near-region sampler must never run.
        #
        # Special-cased on mu - 1 rather than left to the numeric test below. The numeric test
        # would return the same x_star, but only because "0 <= safety * atol" happens to hold for
        # a positive budget; and it would return it just as happily at an order where this one
        # coefficient vanishes by accident and the series does not (see the guard immediately
        # below), which would be wrong.
        return TailCrossover(
            nu=nu,
            x_star=x_min,
            first_omitted_at_x_star=0.0,
            terms_used=n,
            phase_atol=phase_atol,
            amplitude_rtol=amplitude_rtol,
            safety=safety,
            x_phase=x_min,
            x_amplitude=x_min,
            amplitude_error_at_x_star=0.0,
            binding="domain",
        )

    if coefficient == 0.0:
        raise ValueError(
            f"tail_crossover: the first omitted term vanishes identically at nu={nu} with "
            f"{n} series term(s) -- the polynomial factor of coefficient c{n} has a root there -- "
            f"so there is no remainder estimator to test and a crossover cannot be sized. This "
            f"is not the nu = 1/2 degeneracy, where every coefficient vanishes and the series is "
            f"exact. Use a different number of terms."
        )

    x_phase = (coefficient / (safety * phase_atol)) ** (1.0 / power)
    x_amplitude = (0.5 * power * coefficient / (safety * amplitude_rtol)) ** (
        1.0 / (power + 1)
    )

    x_star = max(x_phase, x_amplitude, x_min)
    if x_star == x_min:
        binding = "domain"
    elif x_phase >= x_amplitude:
        binding = "phase"
    else:
        binding = "amplitude"

    if x_star > x_max:
        best = tail_first_omitted_term(nu, x_max, n_terms=n)
        raise ValueError(
            f"tail_crossover: no crossover below x_max={x_max:.6g} for nu={nu} at "
            f"phase_atol={phase_atol:.6g}, amplitude_rtol={amplitude_rtol:.6g} with {n} series "
            f"term(s) and safety={safety:.6g}. The phase condition alone needs "
            f"x >= {x_phase:.6g} and the amplitude condition x >= {x_amplitude:.6g}; the best "
            f"first omitted term available below the ceiling is {best:.6g} at x={x_max:.6g}. "
            f"Widen x_max, loosen the budget, or extend the series."
        )

    return TailCrossover(
        nu=nu,
        x_star=x_star,
        first_omitted_at_x_star=float(tail_first_omitted_term(nu, x_star, n_terms=n)),
        terms_used=n,
        phase_atol=phase_atol,
        amplitude_rtol=amplitude_rtol,
        safety=safety,
        x_phase=x_phase,
        x_amplitude=x_amplitude,
        amplitude_error_at_x_star=float(
            0.5
            * power
            * coefficient
            / x_star ** (power + 1)
            / (1.0 + tail_residual_deriv(nu, x_star, n_terms=n))
        ),
        binding=binding,
    )
