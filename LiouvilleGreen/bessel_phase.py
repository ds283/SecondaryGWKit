"""
Amplitude and phase representation of the Bessel functions, built in two regions.

This module assembles the two regions that :mod:`LiouvilleGreen.bessel_near_region` and
:mod:`LiouvilleGreen.bessel_tail` supply into the object the rest of the repository consumes. It
replaces a construction that integrated ``Q = theta/x`` as an ODE, solved a scalar matching
equation for a phase offset ``phi``, and interpolated the *full* phase through the repository's
chunked full-phase spline module.

Convention
----------

    J_nu(x) = A_nu(x) sin theta_nu(x),        Y_nu(x) = -A_nu(x) cos theta_nu(x),

with ``theta_nu`` increasing in ``x``. DLMF's phase convention differs from this one by exactly
``+pi/2``; that difference is absorbed into ``c_nu`` and is **not** an error to be corrected.

Leading term plus residual
--------------------------

    theta_nu(x) = x + c_nu + r_nu(x),   A_nu(x) = sqrt(2/(pi x)) a_nu(x),   c_nu = pi/4 - pi nu/2.

Below the crossover ``x_star(nu)`` chosen by :func:`LiouvilleGreen.bessel_tail.tail_crossover`,
``r_nu`` and ``a_nu`` are sampled from the exponentially scaled Hankel function, branch-tracked and
interpolated in ``u = log x`` by :func:`LiouvilleGreen.bessel_near_region.build_near_region`. At and
above ``x_star`` they come from one asymptotic series (DLMF 10.18.18) and the exact Wronskian
relation ``a_nu = (1 + r_nu')^(-1/2)``; nothing above ``x_star`` evaluates a SciPy Bessel routine at
all.

The split is preserved all the way to ``sin`` and ``cos``: with ``d = c_nu + r_nu(x)``,

    sin theta = sin x cos d + cos x sin d,    cos theta = cos x cos d - sin x sin d,

so the supplied ``x`` reaches the platform trigonometric routines unreduced and the correction is
never rounded away against it. See :meth:`BesselPhaseFunction.sin_cos_theta`.

What was removed, and why
-------------------------

*The ``Q = theta/x`` ODE.* Reconstruction gives ``delta theta = x delta Q``, so a relative bound on
a state that approaches unity is no bound at all on absolute phase error, which is the error that
matters for ``sin theta`` (``DRAFT-PLAN.md`` §4.1). The phase is in any case a quadrature and not an
ODE: the right-hand side ``(2/pi)/(x m)`` does not contain ``theta`` (§5.1).

*The ``phi`` root solve.* For every ``nu > 1/2`` the domain lower bound satisfies
``log min_x > 0``, so the match point **is** the initial node, where the phase had already been
fixed exactly from the Bessel values; the matching function vanishes identically at ``phi = 0``.
The offset the solve returned was an artefact of its own ``xtol=1e-6, rtol=1e-4``, and it *was* the
whole tight-tolerance error: measured ``phi = -4.836537e-8`` against ``E_theta = 4.873e-8`` at
``nu = 5/2`` (§4.3). ``phi`` is now identically zero, and is reported rather than dropped.

*The full-phase spline.* For a leading phase ``theta ~ x = e^u`` the fourth ``u``
derivative is also ``~ x``, so cubic interpolation errs by ``h^4 x/384`` -- 6.7e-10 at ``x = 1e3``
and 6.7e-6 at ``1e7``, both confirmed. Subtracting whole cycles changes the constant term, not the
fourth derivative (§4.2), and chunking was measured to have no effect on it (§4.6). A second,
independent reason has since been measured on the cosmological side: chunk selection is a hard
switch between two fits, with a **1.08e-4 rad phase jump and a 3.51e-8 relative derivative jump** at
the switch point (``docs/gk-wkb-review-astra-pathfinder-2026-09-08.md`` §3, defect 3) -- a discontinuity a
Levin consumer sees directly. Interpolating ``r_nu`` instead removes the growth: ``r_nu`` decays
like ``1/x``, and above ``x_star`` it is not interpolated at all.

*The ``Q = theta/x`` diagnostic.* ``Q`` was the *pre-offset* ODE state, and there is no ODE state to
report. Keeping the key mapped to ``theta/x`` -- numerically the same thing once ``phi`` is zero --
would leave a member whose documented meaning ("the quantity the solver advances") no longer
describes anything, so the key is **removed** and reading it raises a ``KeyError`` naming its
replacement rather than returning a value that means something else
(``DRAFT-PLAN.md`` §8.1: "do not silently replace ``Q`` with a different diagnostic quantity under
the same undocumented meaning"). The smooth quantity the construction now actually represents is the
residual, and the diagnostic accessors for it are :meth:`BesselPhaseFunction.residual` and
:meth:`BesselPhaseFunction.log_amplitude`.

*The cycle-count-plus-remainder ``(div_2pi, mod_2pi)`` representation.* The reason is **not** that the
residual is small. On the continuously tracked branch used here ``r_nu`` reaches 570.82 rad at the
bottom of the domain for ``nu = 1000.5``, some 90 cycles (``RECONCILIATION.md`` C2, correcting
``DRAFT-PLAN.md`` §4.6). The reason is that the double-precision resolution of a quantity that
size, ``eps * 571 ~ 1.3e-13``, is an order of magnitude below the campaign's 1e-11 low-order target
and seven orders below its 1e-6 high-order target, so a split representation would buy nothing that
the budgets can see. That resolution is included in the accuracy this module reports.

Accuracy of the supplied argument
---------------------------------

The representation evaluates the Bessel functions accurately **at the supplied floating-point
``x``**. It cannot recover uncertainty already present in an ``x = k eta`` formed upstream, and it
cannot undo a lossy ``exp(log(x))`` round trip: when only ``u = log x`` is supplied, the evaluation
is *defined* to be at the computed ``exp(u)``, and that is the argument any reference must use. At
very large ``x`` an input-coordinate error can exceed the residual interpolation error by many
orders of magnitude (``DRAFT-PLAN.md`` §7.5). Where raw ``x`` is supplied it is preserved for the
leading oscillation, and ``log x`` is used only to query the correction interpolants.

Supported domain
----------------

``nu >= 1/2`` and ``construction_min_x(nu) <= x <= max_x <= MAX_SUPPORTED_X``. Requests outside it
raise :class:`BesselPhaseError` rather than being served by an implicit extension. Orders 1/2, 3/2,
7/4, 5/2, 20.5, 100.5 and 1000.5 are the ones the campaign validates.
"""

import math
import warnings
from typing import Optional

import numpy as np

from .bessel_near_region import (
    AmplitudeBandError,
    DEFAULT_INITIAL_PANEL_WIDTH,
    DEFAULT_PANEL_DEGREE,
    NearRegionData,
    NearRegionError,
    build_near_region,
)
from .bessel_tail import (
    DEFAULT_TAIL_TERMS,
    TailCrossover,
    construction_min_x,
    tail_amplitude,
    tail_crossover,
    tail_first_omitted_term,
    tail_residual,
    tail_residual_deriv,
    tail_residual_log_deriv,
    tail_series_coefficients,
)

#: Default absolute phase accuracy, in radians.
#:
#: The campaign's low-order acceptance target is ``E_theta <= 1e-11`` through ``x = 1e7``
#: (``DRAFT-PLAN.md`` §10), and this is that target used directly as the construction budget. It is
#: also comfortably inside the 1e-6 high-order target, where "accuracy is not the objective;
#: correctness is" -- so a single default serves both rows of the table and there is no order
#: dependence to get wrong. The cost of using the tight budget at high order is 817 nodes and 0.04 s
#: at ``nu = 1000.5`` (log 04), which is not worth a special case.
DEFAULT_PHASE_ATOL = 1.0e-11

#: Default relative amplitude accuracy. Same reasoning as :data:`DEFAULT_PHASE_ATOL`.
DEFAULT_AMPLITUDE_RTOL = 1.0e-11

#: Smallest order the construction accepts.
#:
#: Below ``nu = 1/2`` there is no turning point: ``construction_min_x`` returns the arbitrary floor
#: ``1e-5`` rather than ``sqrt(nu^2 - 1/4)``, and ``a_nu = sqrt(pi x/2)|H^(1)_nu|`` behaves like
#: ``x^(1/2 - nu)`` as ``x -> 0``, so it leaves the plausibility band and every sample would be
#: rejected. Failing at the entry point says that plainly; the old construction would instead have
#: returned an object built on an initial condition it could not justify. Nothing in the repository
#: builds an order below 1/2.
MIN_SUPPORTED_NU = 0.5

#: Largest ``max_x`` the construction accepts.
#:
#: The old construction had no declared ceiling and simply stalled above ``x ~ 2.5e15``, where
#: SciPy/Amos ``jv``/``yv`` become O(1)-relatively noisy and DOP853 at ``rtol=5e-14`` can no longer
#: pass its error test (``RECONCILIATION.md`` C1). Nothing here evaluates a Bessel routine above
#: ``x_star ~ 10-60 nu``, so that cliff is gone and the binding limit becomes the platform's
#: ``sin``/``cos``, which were verified correctly rounded to ``1e16`` (``RECONCILIATION.md`` §1).
#: This is a *declared* ceiling where there was none, not a widening of a validated domain: prompt
#: 09 measures how far the representation actually goes.
MAX_SUPPORTED_X = 1.0e16

#: Fractional cushion outside ``[min_x, max_x]`` in which an evaluation is clamped rather than
#: rejected. Unchanged from the previous implementation's ``XSplineWrapper``.
DOMAIN_CUSHION = 0.01

#: Absolute phase floor of a scaled-Hankel *sample*, in radians.
#:
#: ``NearRegionData``'s ``achieved_*`` estimates are measured by resampling ``hankel1e`` interior to
#: each panel and comparing against the interpolants there, so both sides of the comparison come
#: from the same function and a systematic bias in ``hankel1e``'s own phase cancels out of them:
#: they estimate *interpolation* error only (status-board issue
#: ``[04-achieved-estimates-exclude-the-sampling-floor]``). Since this module publishes those
#: numbers as ``theta_abserr``, whose entire purpose is that "the caller sees an honest number
#: instead of an artificially small one" (``AdaptiveLevin/levin_quadrature.py:2360``), the sampling
#: floor is added back explicitly rather than assumed away.
#:
#: The value is ``DRAFT-PLAN.md`` §4.4's measured 2.96e-13 ``hankel1e`` phase floor at
#: ``nu = 1000.5``, rounded up. It is used at every order, which is conservative at low order by
#: one to two orders of magnitude; and it is conservative at high order too, because it was measured
#: *before* prompt 04 removed a 6.6e-14--2.3e-13 argument-reduction contributor to it
#: (:func:`LiouvilleGreen.bessel_near_region.scaled_hankel_phase_constant`). A re-measurement
#: against ``mpmath`` would let it be reduced; nothing in the campaign's budgets needs that.
SAMPLED_PHASE_FLOOR = 3.0e-13

#: Relative amplitude floor of a scaled-Hankel sample. Same provenance as
#: :data:`SAMPLED_PHASE_FLOOR`.
SAMPLED_AMPLITUDE_FLOOR = 3.0e-13

#: Floor contributed by the evaluation arithmetic itself, in radians (and, as a relative error, for
#: the amplitude).
#:
#: ``DRAFT-PLAN.md`` §10 requires the phase error to be allocated among near-region sampling, branch
#: tracking, interpolation, the series remainder, the derivative representation **and the evaluation
#: arithmetic**. The last of those does not vanish even where the representation is exact: at
#: ``nu = 1/2`` the residual is identically zero and the series is exact at every ``x``, yet the
#: angle addition of :meth:`BesselPhaseFunction.sin_cos_theta` still performs four multiplications
#: and an addition, and the measured phase-pair error against the committed 40-digit corners is
#: 1.11e-16 -- half an ulp of unity. Four ulps covers that at every order and is negligible against
#: every other term wherever a near region exists.
EVALUATION_FLOOR = 4.0 * float(np.finfo(float).eps)

#: Margin applied to the tail series remainder **when it is declared as an error**, as opposed to
#: when it sizes the crossover.
#:
#: :func:`LiouvilleGreen.bessel_tail.tail_first_omitted_term` is an *estimator* of an asymptotic
#: remainder, not a bound on it: its own docstring records the true ``|delta r|`` as 0.98--1.02
#: times it, measured against 60-digit ``mpmath``. That is fine for choosing ``x_star``, where
#: ``bessel_tail.DEFAULT_CROSSOVER_SAFETY`` already buys a factor of four, but it is *not* fine for
#: :attr:`BesselPhaseFunction.theta_abserr` and :meth:`BesselPhaseFunction.theta_abserr_at`, whose
#: entire purpose is that the number cannot be optimistic
#: (``AdaptiveLevin/levin_quadrature.py:2360``). Declaring the estimator 1:1 was measured over 1500
#: points immediately above ``x_star`` at ``nu`` = 3/2, 7/4, 5/2, 20.5 and 100.5 to leave a margin
#: of 0.006 % -- the worst measured/estimator ratio was 0.99994, i.e. always below 1 but with
#: essentially nothing to spare, and the sign of that margin is not something a term-count or an
#: order change is obliged to preserve (log 06).
#:
#: Two costs it against: the declared low-order phase error becomes 5.0e-12 rather than 2.5e-12,
#: still a factor of two inside the 1e-11 the campaign accepts and inside the budget the caller
#: requested; and nothing else in the module changes, because the crossover is placed by
#: ``bessel_tail`` and this factor is applied nowhere near it.
DECLARED_SERIES_SAFETY = 2.0

#: Smallest initial panel width, in e-folds, that a ``sample_points`` request may ask for.
MIN_INITIAL_PANEL_WIDTH = 1.0e-3


class BesselPhaseError(RuntimeError):
    """
    A request outside the declared domain, or a construction that could not meet its budget.

    Derived from ``RuntimeError`` because that is what the previous implementation raised for a
    domain failure, and callers that catch it keep working.
    """


class BesselPhaseAccuracyError(BesselPhaseError):
    """The near region hit its refinement cap without meeting the requested accuracy."""


def c_nu(nu: float) -> float:
    """
    The zero-point ``c_nu = pi/4 - pi nu/2`` of ``theta_nu = x + c_nu + r_nu``.

    This is where DLMF's ``-(nu/2 + 1/4) pi`` and the repository's ``J = A sin theta`` convention
    are reconciled; the two differ by exactly ``+pi/2``. Do not "correct" it.
    """
    return 0.25 * math.pi - 0.5 * math.pi * nu


def c_nu_reduced(nu: float) -> float:
    """
    ``c_nu`` reduced modulo ``2 pi``, for use inside ``sin`` and ``cos`` only.

    Written as ``pi * fmod((1 - 2 nu)/4, 2)``. For a half-integer order ``(1 - 2 nu)/4`` is exact in
    binary and ``fmod`` is exact, so the result carries a single rounding of ``pi q`` with
    ``|q| <= 2``, below ``7e-16``. Forming ``pi/4 - pi nu/2`` directly gives ``-1570.87`` at
    ``nu = 1000.5``, whose ulp is ``2.3e-13`` -- an error injected into every evaluated angle at
    high order for no reason, since only ``sin`` and ``cos`` of it are ever needed and it differs
    from :func:`c_nu` by an exact multiple of ``2 pi``.

    :func:`c_nu`, not this, is what :meth:`BesselPhaseFunction.raw_theta` adds, because ``raw_theta``
    is documented to return ``x + c_nu + r_nu`` itself.
    """
    return math.pi * math.fmod((1.0 - 2.0 * nu) / 4.0, 2.0)


def _is_scalar(x) -> bool:
    return np.ndim(x) == 0


class XSplineWrapper:
    """
    Adapter presenting a function of ``log x`` as a function of ``x``, with domain clamping.

    Retained with its previous behaviour because it is public surface:
    ``LiouvilleGreen/tests/test_three_bessel.py`` imports it by name and annotates ``mod`` with it.
    An evaluation more than :data:`DOMAIN_CUSHION` outside ``[min_x, max_x]`` raises; one inside the
    cushion is clamped to the boundary.
    """

    def __init__(self, spline, min_x, max_x):
        self._spline = spline

        self._min_x = min_x
        self._max_x = max_x

        self._log_min_x = np.log(min_x)
        self._log_max_x = np.log(max_x)

    def _resolve(self, x, is_log: bool):
        """
        Map the supplied argument to a clamped ``(raw_x, log_x)`` pair.

        The raw argument is preserved where it was supplied raw, and recomputed as ``exp(u)`` where
        only ``u`` was supplied -- which is the point at which ``DRAFT-PLAN.md`` §7.5's "the
        evaluation is defined to be at the computed ``exp(u)``" takes effect. Scalars in, scalars
        out; arrays in, arrays out.
        """
        if _is_scalar(x):
            if is_log:
                log_x = float(x)
                raw_x = math.exp(log_x)
            else:
                raw_x = float(x)
                log_x = math.log(raw_x) if raw_x > 0.0 else -math.inf

            if raw_x < (1.0 - DOMAIN_CUSHION) * self._min_x:
                raise ValueError(
                    f"x={raw_x:.5g} too small (minimum value={self._min_x:.5g})"
                )
            if raw_x > (1.0 + DOMAIN_CUSHION) * self._max_x:
                raise ValueError(
                    f"x={raw_x:.5g} too large (maximum value={self._max_x:.5g})"
                )

            if raw_x < self._min_x:
                return self._min_x, float(self._log_min_x)
            if raw_x > self._max_x:
                return self._max_x, float(self._log_max_x)
            return raw_x, log_x

        if is_log:
            log_x = np.asarray(x, dtype=float)
            raw_x = np.exp(log_x)
        else:
            raw_x = np.asarray(x, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_x = np.log(raw_x)

        if np.any(raw_x < (1.0 - DOMAIN_CUSHION) * self._min_x):
            worst = float(np.min(raw_x))
            raise ValueError(
                f"x={worst:.5g} too small (minimum value={self._min_x:.5g})"
            )
        if np.any(raw_x > (1.0 + DOMAIN_CUSHION) * self._max_x):
            worst = float(np.max(raw_x))
            raise ValueError(
                f"x={worst:.5g} too large (maximum value={self._max_x:.5g})"
            )

        low = raw_x < self._min_x
        high = raw_x > self._max_x
        raw_x = np.where(low, self._min_x, np.where(high, self._max_x, raw_x))
        log_x = np.where(low, self._log_min_x, np.where(high, self._log_max_x, log_x))
        return raw_x, log_x

    def __call__(self, x, is_log=False):
        _, log_x = self._resolve(x, is_log)
        return self._spline(log_x)


class _TwoRegionCorrections:
    """
    The shared core: ``r_nu``, ``a_nu`` and their derivatives, from whichever region owns ``x``.

    Below ``x_star`` the near-region interpolants answer; at and above it the closed-form tail does.
    There is deliberately **no blending, taper or overlap**. A discontinuity that had to be hidden by
    blending would be a construction that failed its own crossover test, and blending would also
    destroy the ``C^1`` behaviour the Levin quadrature's subdivision logic reads through
    ``theta_deriv``. The seam is instead *tested* at construction time by
    :func:`bessel_phase`, which raises if the two sides disagree beyond the budget.
    """

    def __init__(
        self,
        nu: float,
        x_lo: float,
        x_star: float,
        near: Optional[NearRegionData],
        tail_terms: int,
    ):
        self.nu = float(nu)
        self.x_lo = float(x_lo)
        self.x_star = float(x_star)
        self.near = near
        self.tail_terms = int(tail_terms)

        self._r_interp = None if near is None else near.r_interp
        self._r_deriv_interp = None if near is None else near.r_interp.derivative()
        self._ell_interp = None if near is None else near.log_a_interp
        self._ell_deriv_interp = (
            None if near is None else near.log_a_interp.derivative()
        )

        self._coefficients = tail_series_coefficients(self.nu)

    # -- the tail ------------------------------------------------------------------------------

    def _tail_ell_log_deriv(self, raw_x):
        """
        ``d ell/d log x`` in the tail, where ``ell = log a = -log(1 + r')/2``.

        ``d ell/du = -x r''/(2 (1 + r'))`` with ``x r'' = sum (2j+1)(2j+2) c_j x^-(2j+2)``, taken
        term by term from the same series that supplies ``r`` -- the amplitude and the phase are
        governed by one series here (campaign ``README.md`` §2 (b)), so there is nothing else to
        consult.
        """
        x = np.asarray(raw_x, dtype=float)
        total = np.zeros_like(x)
        for j in range(self.tail_terms):
            coefficient = self._coefficients[j]
            if coefficient == 0.0:
                continue
            total = total + (2 * j + 1) * (2 * j + 2) * coefficient * x ** (
                -(2 * j + 2)
            )
        rp = np.asarray(
            tail_residual_deriv(self.nu, x, n_terms=self.tail_terms), dtype=float
        )
        return -0.5 * total / (1.0 + rp)

    # -- dispatch ------------------------------------------------------------------------------

    def _dispatch(self, raw_x, log_x, tail_fn, near_fn):
        if _is_scalar(raw_x):
            if raw_x >= self.x_star or self._r_interp is None:
                return float(tail_fn(raw_x))
            return float(near_fn(log_x))

        raw = np.asarray(raw_x, dtype=float)
        lg = np.asarray(log_x, dtype=float)
        out = np.empty(raw.shape, dtype=float)

        if self._r_interp is None:
            out[...] = tail_fn(raw)
            return out

        high = raw >= self.x_star
        if np.any(high):
            out[high] = tail_fn(raw[high])
        low = ~high
        if np.any(low):
            out[low] = near_fn(lg[low])
        return out

    # -- the quantities ------------------------------------------------------------------------

    def residual(self, raw_x, log_x):
        """``r_nu(x)``, on the branch with ``r -> 0`` as ``x -> infinity``. Never reduced mod 2 pi."""
        return self._dispatch(
            raw_x,
            log_x,
            lambda v: tail_residual(self.nu, v, n_terms=self.tail_terms),
            self._r_interp,
        )

    def residual_log_deriv(self, raw_x, log_x):
        """``dr_nu/d log x``."""
        return self._dispatch(
            raw_x,
            log_x,
            lambda v: tail_residual_log_deriv(self.nu, v, n_terms=self.tail_terms),
            self._r_deriv_interp,
        )

    def log_amplitude(self, raw_x, log_x):
        """``ell = log a_nu``."""
        return self._dispatch(
            raw_x,
            log_x,
            lambda v: np.log(tail_amplitude(self.nu, v, n_terms=self.tail_terms)),
            self._ell_interp,
        )

    def log_amplitude_log_deriv(self, raw_x, log_x):
        """``d ell/d log x``."""
        return self._dispatch(
            raw_x, log_x, self._tail_ell_log_deriv, self._ell_deriv_interp
        )

    def amplitude(self, raw_x, log_x):
        """``a_nu(x)``, the normalized amplitude, ``A_nu = sqrt(2/(pi x)) a_nu``."""
        return self._dispatch(
            raw_x,
            log_x,
            lambda v: tail_amplitude(self.nu, v, n_terms=self.tail_terms),
            lambda u: np.exp(self._ell_interp(u)),
        )

    def theta_prime(self, raw_x, log_x):
        """
        ``theta'(x)``.

        In the near region this is ``a^-2 = exp(-2 ell)``, read off the amplitude interpolant as a
        **value** rather than obtained by differentiating the residual interpolant: ``DRAFT-PLAN.md``
        §4.7 measures that route better in 7 of 8 configurations, by 4x to 15x. Its consequence is
        that the Wronskian identity ``a^2 theta' = 1`` becomes a tautology here, so the independent
        checks are ``exp(-2 ell)`` against ``1 + r_u/x`` and both against
        ``(2/pi)/(x (J^2 + Y^2))`` from reference functions (campaign ``README.md`` §2 (f)).

        In the tail it is ``1 + r'(x)`` taken directly from the series, which is a *primitive*
        there rather than a differentiated interpolant, and is therefore exact to the series
        remainder.
        """
        return self._dispatch(
            raw_x,
            log_x,
            lambda v: 1.0
            + np.asarray(
                tail_residual_deriv(self.nu, v, n_terms=self.tail_terms), dtype=float
            ),
            lambda u: np.exp(-2.0 * self._ell_interp(u)),
        )


class BesselAmplitude(XSplineWrapper):
    """
    The envelope ``A_nu(x) = sqrt(2/(pi x)) a_nu(x)``, which is ``sqrt(J_nu^2 + Y_nu^2)``.

    This is the object the returned dict calls ``mod``, and it is called exactly as before:
    ``mod(x)`` or ``mod(x, is_log=True)``. It subclasses :class:`XSplineWrapper` so that the
    annotation in ``LiouvilleGreen/tests/test_three_bessel.py`` stays truthful, but it overrides
    ``__call__``: the algebraic prefactor is evaluated at the *supplied* raw ``x``, not at
    ``exp(log x)``.
    """

    def __init__(self, corrections: _TwoRegionCorrections, min_x: float, max_x: float):
        super().__init__(None, min_x, max_x)
        self._corrections = corrections

    def __call__(self, x, is_log=False):
        raw_x, log_x = self._resolve(x, is_log)
        a = self._corrections.amplitude(raw_x, log_x)
        if _is_scalar(raw_x):
            return math.sqrt(2.0 / (math.pi * raw_x)) * float(a)
        return np.sqrt(2.0 / (math.pi * np.asarray(raw_x, dtype=float))) * a

    def a(self, x, is_log=False):
        """The normalized amplitude ``a_nu``, which tends to 1 from above as ``x`` grows."""
        raw_x, log_x = self._resolve(x, is_log)
        return self._corrections.amplitude(raw_x, log_x)

    def log_deriv(self, x, is_log=False):
        """
        ``d log A/dx = -1/(2x) + ell_u(log x)/x`` (``DRAFT-PLAN.md`` §7.3).

        The algebraic part is exact; only ``ell_u`` comes from the representation.
        """
        raw_x, log_x = self._resolve(x, is_log)
        ell_u = self._corrections.log_amplitude_log_deriv(raw_x, log_x)
        if _is_scalar(raw_x):
            return -0.5 / raw_x + float(ell_u) / raw_x
        raw = np.asarray(raw_x, dtype=float)
        return -0.5 / raw + ell_u / raw


class BesselPhaseFunction:
    """
    The phase object returned under the ``"phase"`` key.

    It implements the accessor protocol the consumers use --
    ``raw_theta(x, x_is_log=False)``, ``theta_mod_2pi(x, x_is_log=False)`` and
    ``theta_deriv(x, x_is_log=False, log_derivative=False)`` -- and adds the split-preserving
    accessors that make the two-region representation worth having.
    """

    def __init__(
        self,
        nu: float,
        corrections: _TwoRegionCorrections,
        min_x: float,
        max_x: float,
        amplitude: BesselAmplitude,
        theta_abserr: float,
    ):
        self.nu = float(nu)
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.x_star = corrections.x_star

        #: ``c_nu`` as defined, for a consumer that needs the leading coefficient of the phase.
        self.c_nu = c_nu(self.nu)
        #: ``c_nu`` reduced mod ``2 pi``; what the split evaluation actually uses.
        self.c_nu_reduced = c_nu_reduced(self.nu)

        #: Declared absolute phase error in radians, valid over the whole domain. This is the
        #: scalar form of ``AdaptiveLevin``'s ``theta_abserr``; :meth:`theta_abserr_at` is the
        #: ``x``-dependent form, and is the one to hand to ``adaptive_levin_sincos``.
        self.theta_abserr = float(theta_abserr)

        self._corrections = corrections
        self._amplitude = amplitude
        self._domain = XSplineWrapper(None, min_x=min_x, max_x=max_x)

    # -- domain -------------------------------------------------------------------------------

    def _resolve(self, x, x_is_log: bool):
        return self._domain._resolve(x, x_is_log)

    # -- the split ----------------------------------------------------------------------------

    def residual(self, x, x_is_log: bool = False):
        """
        ``r_nu(x) = theta_nu(x) - x - c_nu``, on the branch with ``r -> 0`` as ``x -> infinity``.

        Never reduced modulo ``2 pi``: at ``nu = 1000.5`` it reaches 570.82 rad at the bottom of the
        domain, and folding it would produce a value wrong by an exact multiple of ``2 pi``.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        return self._corrections.residual(raw_x, log_x)

    def residual_log_deriv(self, x, x_is_log: bool = False):
        """``dr_nu/d log x``. Used by the independent derivative check ``theta' = 1 + r_u/x``."""
        raw_x, log_x = self._resolve(x, x_is_log)
        return self._corrections.residual_log_deriv(raw_x, log_x)

    def log_amplitude(self, x, x_is_log: bool = False):
        """
        ``ell(x) = log a_nu(x)``, the quantity the near region actually interpolates.

        Exposed for diagnostics, as the amplitude counterpart of :meth:`residual`: together the two
        are the smooth pair the construction represents, and they are what replaced the old ``Q``
        diagnostic (module docstring, "What was removed, and why"). ``theta' = exp(-2 ell)`` exactly,
        so this is also the quantity a reader wanting to see where the derivative comes from should
        plot. The envelope itself is ``mod``: ``A_nu = sqrt(2/(pi x)) exp(ell)``.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        return self._corrections.log_amplitude(raw_x, log_x)

    def sin_cos_theta(self, x, x_is_log: bool = False):
        """
        ``(sin theta, cos theta)`` by angle addition on ``theta = x + d``, ``d = c_nu + r_nu(x)``.

        Forming ``x + d`` first and taking its sine rounds away part or all of the correction: at
        ``nu = 3/2``, measured against 70-digit ``mpmath``, the naive route errs by 3.5e-14 at
        ``x = 1e3``, 1.2e-10 at ``1e7``, 2.7e-6 at ``1e12`` and **4.7e-2** at ``1e15``, while angle
        addition stays at or below 1.1e-16 at every one of those points (``RECONCILIATION.md`` §1).

        The supplied ``x`` is handed to ``sin``/``cos`` **unreduced**. A quality libm performs
        Payne-Hanek reduction against a multi-hundred-bit ``pi`` and is correctly rounded out to
        ``1e16`` (verified); any reduction performed here would be done in double precision against
        a 53-bit ``2 pi`` and would be strictly worse. This is the argument
        ``LiouvilleGreen/range_reduce_mod_2pi.py``'s module docstring already makes.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        r = self._corrections.residual(raw_x, log_x)

        if _is_scalar(raw_x):
            d = self.c_nu_reduced + float(r)
            sin_x = math.sin(raw_x)
            cos_x = math.cos(raw_x)
            sin_d = math.sin(d)
            cos_d = math.cos(d)
            return sin_x * cos_d + cos_x * sin_d, cos_x * cos_d - sin_x * sin_d

        raw = np.asarray(raw_x, dtype=float)
        d = self.c_nu_reduced + r
        sin_x = np.sin(raw)
        cos_x = np.cos(raw)
        sin_d = np.sin(d)
        cos_d = np.cos(d)
        return sin_x * cos_d + cos_x * sin_d, cos_x * cos_d - sin_x * sin_d

    # -- the consumer protocol -----------------------------------------------------------------

    def raw_theta(self, x, x_is_log: bool = False):
        """
        ``theta_nu(x) = x + c_nu + r_nu(x)`` as a single float.

        **This value is limited to ``eps * theta`` in absolute terms**, i.e. about 2.2e-1 rad at
        ``x = 1e15``: the sum is formed in double precision and the correction is rounded against
        the leading term, which is precisely what :meth:`sin_cos_theta` exists to avoid. It is kept
        for compatibility and diagnostics, and because ``theta`` is a *required* key of the
        ``AdaptiveLevin`` phase dictionary (``levin_quadrature.py:948-952`` raises without it) --
        "compatibility-only" means Levin never *evaluates* it when ``theta_mod_2pi`` and
        ``theta_deriv`` are both supplied (``:1038``), not that the key may be omitted.

        Accurate oscillatory evaluation must use :meth:`sin_cos_theta` or :meth:`theta_mod_2pi`.
        This is **not** a cycle-count algorithm and must not be mistaken for one; if a consumer
        genuinely needs an accurate integer cycle count, that is separate design and validation
        work (``DRAFT-PLAN.md`` §7.4).
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        r = self._corrections.residual(raw_x, log_x)
        if _is_scalar(raw_x):
            return raw_x + self.c_nu + float(r)
        return np.asarray(raw_x, dtype=float) + self.c_nu + r

    def theta_mod_2pi(self, x, x_is_log: bool = False):
        """
        A bounded representative of ``theta_nu(x)``, in ``(-pi, pi]``, from
        ``atan2(sin theta, cos theta)`` on the split-evaluated pair.

        Reducing the raw phase against a double-precision ``2 pi`` instead would give an error
        proportional to the cycle count: ``fmod`` is exact for the arguments it is handed, but it
        faithfully computes the remainder with respect to the *wrong* modulus. Going through
        ``atan2`` of the split pair keeps the accuracy of :meth:`sin_cos_theta`, which is what
        consumers actually need, since they only ever take ``sin`` and ``cos`` of the result.

        The interval convention is applied only to this bounded value; nothing else in the module
        is reduced.
        """
        sin_theta, cos_theta = self.sin_cos_theta(x, x_is_log=x_is_log)
        if _is_scalar(sin_theta):
            return math.atan2(sin_theta, cos_theta)
        return np.arctan2(sin_theta, cos_theta)

    def theta_deriv(self, x, x_is_log: bool = False, log_derivative: bool = False):
        """
        ``theta'(x)``, or ``d theta/d log x = x theta'(x)`` when ``log_derivative`` is set.

        See :meth:`_TwoRegionCorrections.theta_prime` for why the near region reads this off the
        amplitude interpolant as a value rather than differentiating the residual.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        deriv = self._corrections.theta_prime(raw_x, log_x)
        if not log_derivative:
            return deriv
        if _is_scalar(raw_x):
            return raw_x * float(deriv)
        return np.asarray(raw_x, dtype=float) * deriv

    def theta_deriv_from_residual(self, x, x_is_log: bool = False):
        """
        The **independent** route ``theta' = 1 + r_u/x``, for the consistency check of
        ``README.md`` §2 (f).

        It is never used to evaluate anything; it exists so that the two interpolants can be
        cross-checked against each other, since a Wronskian check on a representation whose
        ``theta'`` is *derived from* ``a`` would be a tautology.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        r_u = self._corrections.residual_log_deriv(raw_x, log_x)
        if _is_scalar(raw_x):
            return 1.0 + float(r_u) / raw_x
        return 1.0 + r_u / np.asarray(raw_x, dtype=float)

    # -- the declared error --------------------------------------------------------------------

    def theta_abserr_at(self, x, x_is_log: bool = False):
        """
        The declared absolute phase error **at** ``x``, in radians: the callable form of
        ``theta_abserr`` that ``AdaptiveLevin`` accepts (``levin_quadrature.py:967-982`` takes a
        scalar or a callable of ``x``).

        Pass *this* to ``adaptive_levin_sincos`` as ``theta["theta_abserr"]`` in preference to the
        scalar :attr:`theta_abserr`. The two regions have genuinely different errors and the tail's
        is far smaller -- at ``nu = 5/2`` the near region declares 2.5e-12 while the tail at
        ``x = 1e7`` declares 8.9e-16, a factor of 2800 -- and Levin applies the declared value as an
        endpoint term on *every* region's round-off floor, so a single domain-wide scalar would
        inflate the reported error of every far-tail region by that factor. The scalar remains the
        honest domain-wide bound and is what the returned dict's ``theta_abserr`` key carries.

        The model, term by term:

        * **below** ``x_star``: the constant :attr:`theta_abserr`. Near-region interpolation error
          is not resolved per panel (``NearRegionData`` reports one number for the whole region),
          and it dominates every other term there, so there is nothing to gain from pretending to
          an ``x`` dependence the measurement does not have.
        * **at or above** ``x_star``: ``2 |c_n|/x^(2n+1) + 4 eps + eps |r_nu(x)|`` -- the first
          omitted series term with the :data:`DECLARED_SERIES_SAFETY` margin, which is the tail's
          only representation error; the angle-addition floor of :data:`EVALUATION_FLOOR`; and the
          double-precision resolution of the residual that is added to ``c_nu`` inside
          :meth:`sin_cos_theta`. These are **summed**, not maxed: they are independent
          contributions, and maxing them was measured to under-report by 3 % at
          ``nu = 20.5, x = 2050``, where the series remainder and the arithmetic floor are the same
          size (log 06). The result is capped at :attr:`theta_abserr`, which is the maximum of this
          model over the domain, so the cap is inert rather than binding.

        Verified not to under-report the phase-pair error at every cached 40-digit corner of every
        order the campaign covers, and on dense low-order sweeps of both regions; the measured
        margin is 1.2x to 2.3e4x (log 06). An estimator that under-reports would be worse than
        declaring nothing at all, which is why the inequality is asserted in
        ``test_bessel_compatibility.py`` rather than assumed.
        """
        raw_x, log_x = self._resolve(x, x_is_log)
        corrections = self._corrections
        nu = corrections.nu
        terms = corrections.tail_terms
        eps = float(np.finfo(float).eps)

        if _is_scalar(raw_x):
            if raw_x < corrections.x_star and corrections.near is not None:
                return self.theta_abserr
            r = float(corrections.residual(raw_x, log_x))
            series = float(tail_first_omitted_term(nu, raw_x, n_terms=terms))
            return min(
                self.theta_abserr,
                DECLARED_SERIES_SAFETY * series + EVALUATION_FLOOR + eps * abs(r),
            )

        raw = np.asarray(raw_x, dtype=float)
        r = np.asarray(corrections.residual(raw_x, log_x), dtype=float)
        series = np.asarray(
            tail_first_omitted_term(nu, raw, n_terms=terms), dtype=float
        )
        tail = np.minimum(
            self.theta_abserr,
            DECLARED_SERIES_SAFETY * series + EVALUATION_FLOOR + eps * np.abs(r),
        )
        if corrections.near is None:
            return tail
        return np.where(raw < corrections.x_star, self.theta_abserr, tail)

    # -- Bessel values -------------------------------------------------------------------------

    def bessel_j(self, x, is_log=False):
        """``J_nu(x) = A_nu(x) sin theta_nu(x)``, through the split evaluation."""
        sin_theta, _ = self.sin_cos_theta(x, x_is_log=is_log)
        return self._amplitude(x, is_log=is_log) * sin_theta

    def bessel_y(self, x, is_log=False):
        """``Y_nu(x) = -A_nu(x) cos theta_nu(x)``, through the split evaluation."""
        _, cos_theta = self.sin_cos_theta(x, x_is_log=is_log)
        return -self._amplitude(x, is_log=is_log) * cos_theta


#: Keys the returned mapping used to carry and deliberately no longer does, with the message a
#: reader of one gets. Each entry has to say what to use instead: the failure mode this exists to
#: prevent is a consumer silently receiving a *different* quantity under an old name, and the
#: second-worst outcome is a bare ``KeyError: 'Q'`` that sends a reader to ``git log``.
_REMOVED_KEYS = {
    "Q": (
        "bessel_phase: the 'Q' member has been removed. It was the *pre-offset* state of the "
        "phase ODE dQ/dlog x = (2/pi)/(x m) - Q, and there is no ODE and no offset any more: the "
        "phase is built as theta = x + c_nu + r_nu from a sampled near region and a closed-form "
        "tail. theta/x is still computable as phase.raw_theta(x)/x, but it is no longer the "
        "quantity 'Q' named, so it is not served under that key. For the smooth quantity the "
        "construction actually represents, plot phase.residual(x) (the phase residual r_nu) and "
        "phase.log_amplitude(x) (log a_nu); for the phase itself use phase.raw_theta(x), "
        "phase.theta_mod_2pi(x) or phase.sin_cos_theta(x)."
    ),
}


class _BesselPhaseData(dict):
    """
    The mapping :func:`bessel_phase` returns: a ``dict``, plus a message for a removed key.

    Subclassing ``dict`` rather than returning one keeps every consumer working unchanged --
    ``BesselPhaseProxy`` puts the whole thing through ``ray.put``, and this pickles exactly as a
    ``dict`` does because the class is module level and holds no state of its own. The only added
    behaviour is :meth:`__missing__`, so that a consumer of a member this construction dropped is
    told what to use instead at the point of the lookup, rather than getting ``KeyError: 'Q'``.
    """

    def __missing__(self, key):
        if key in _REMOVED_KEYS:
            raise KeyError(_REMOVED_KEYS[key])
        raise KeyError(key)


def _initial_panel_width(
    sample_points: Optional[int], log_min_x: float, log_max_x: float, degree: int
) -> float:
    """
    Translate the legacy ``sample_points`` into a **floor on the initial node density**.

    Under adaptive sampling ``sample_points`` can no longer mean "use exactly this many": the panel
    count is decided by the acceptance criteria, and the whole point of the two-sided adaptivity is
    that the tail is sampled far more coarsely than the turning point. It is therefore reinterpreted
    -- ``sample_points`` over the requested ``[log min_x, log max_x]`` is read as a requested density
    in nodes per e-fold, and the initial panels are made narrow enough to supply at least that
    density. It can only make the initial grid finer, never coarser, so refinement still decides the
    outcome.

    ``docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82`` is the caller that keeps this
    argument alive (``RECONCILIATION.md`` §3.3).
    """
    if sample_points is None or sample_points <= 0:
        return DEFAULT_INITIAL_PANEL_WIDTH

    span = max(log_max_x - log_min_x, 1.0e-12)
    density = float(sample_points) / span
    if density <= 0.0:
        return DEFAULT_INITIAL_PANEL_WIDTH

    width = float(degree) / density
    return min(DEFAULT_INITIAL_PANEL_WIDTH, max(MIN_INITIAL_PANEL_WIDTH, width))


def bessel_phase(
    nu: float,
    max_x: float,
    sample_points: Optional[int] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    phase_atol: Optional[float] = None,
    amplitude_rtol: Optional[float] = None,
    tail_terms: int = DEFAULT_TAIL_TERMS,
    interp_degree: int = DEFAULT_PANEL_DEGREE,
    strict_accuracy: bool = True,
) -> dict:
    """
    Build the two-region amplitude and phase representation of ``J_nu`` and ``Y_nu``.

    :param nu: the order. Must be at least :data:`MIN_SUPPORTED_NU`.
    :param max_x: the top of the domain served. Must lie in
        ``[construction_min_x(nu), MAX_SUPPORTED_X]``.
    :param sample_points: legacy. Reinterpreted as a floor on the initial node density; see
        :func:`_initial_panel_width`.
    :param atol: **deprecated**, and ignored. It described an ODE absolute tolerance, and there is
        no ODE. Use ``phase_atol``.
    :param rtol: **deprecated**, and ignored. It described an ODE relative tolerance. Use
        ``amplitude_rtol``.
    :param phase_atol: absolute phase accuracy requested, in radians.
    :param amplitude_rtol: relative amplitude accuracy requested.
    :param tail_terms: number of terms of DLMF 10.18.18 to sum. The same value sizes the crossover
        and evaluates the tail, and they must agree for either to mean anything.
    :param interp_degree: Chebyshev degree of a near-region panel.
    :param strict_accuracy: when ``True`` (the default) a near region that hits its refinement cap
        without meeting the budget raises :class:`BesselPhaseAccuracyError`. When ``False`` the
        object is returned with ``accuracy_met = False`` after a warning, for diagnostics.
    :raises BesselPhaseError: for a request outside the declared domain, a crossover that cannot be
        placed at the requested accuracy, or a seam whose two sides disagree beyond the budget.
    :raises BesselPhaseAccuracyError: for an unmet accuracy request under ``strict_accuracy``.
    :raises LiouvilleGreen.bessel_near_region.AmplitudeBandError: if a sampled ``a_nu`` leaves the
        two-sided plausibility band -- the guard that catches ``hankel1e`` returning exactly ``-0j``,
        which is *finite* and which an ``isfinite`` check would therefore accept.

    **Deprecated tolerance arguments.** ``atol`` and ``rtol`` map to nothing: they named the
    tolerances of an ODE solve that no longer exists, so translating them into the new budgets would
    be inventing a correspondence. Supplying either emits a ``DeprecationWarning`` naming the
    replacements, and the new defaults (or the new arguments, if supplied) are used. If old and new
    are supplied together the new ones win and the warning says so.

    **The returned mapping.** A ``dict`` (a :class:`_BesselPhaseData`, which differs only in the
    message it raises for a removed key) carrying ``phase``, ``mod``, ``phi``, ``bessel_j``,
    ``bessel_y``, ``min_x``, ``max_x`` under their previous meanings, and ``nu``, ``x_star``,
    ``theta_abserr``, ``amplitude_relerr``, ``theta_deriv_relerr``, ``accuracy``, ``accuracy_met``,
    ``crossover`` and ``near_region`` as of the two-region construction. ``Q`` is **gone**; see
    :data:`_REMOVED_KEYS`. A consumer assembling an ``AdaptiveLevin`` phase dictionary wants

        {"theta": phase.raw_theta, "theta_mod_2pi": phase.theta_mod_2pi,
         "theta_deriv": phase.theta_deriv, "theta_abserr": phase.theta_abserr_at}

    -- all four, in that shape. ``theta`` is required even though it is never *evaluated* when the
    other two accessors are present (``levin_quadrature.py:948-952`` and ``:1038``), and
    ``theta_deriv`` carries double duty there: it conditions the basis *and* decides subdivision,
    since ``phase_span`` is computed from it (``:1090``).
    """
    # ---- deprecated tolerance arguments ------------------------------------------------------
    deprecated_supplied = [
        name for name, value in (("atol", atol), ("rtol", rtol)) if value is not None
    ]
    if deprecated_supplied:
        replacement_supplied = phase_atol is not None or amplitude_rtol is not None
        detail = (
            "phase_atol/amplitude_rtol were also supplied and take precedence"
            if replacement_supplied
            else "the new defaults are being used instead"
        )
        warnings.warn(
            f"bessel_phase: {', '.join(deprecated_supplied)} is deprecated and has no effect. "
            f"It described a tolerance of the phase ODE solve, which no longer exists, so it maps "
            f"to nothing rather than to a translated value; {detail}. Use phase_atol (absolute "
            f"radians) and amplitude_rtol (relative) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if phase_atol is None:
        phase_atol = DEFAULT_PHASE_ATOL
    if amplitude_rtol is None:
        amplitude_rtol = DEFAULT_AMPLITUDE_RTOL

    # ---- the declared domain -----------------------------------------------------------------
    nu = float(nu)
    max_x = float(max_x)

    if not math.isfinite(nu) or nu < MIN_SUPPORTED_NU:
        raise BesselPhaseError(
            f"bessel_phase: nu={nu!r} is outside the supported order range nu >= "
            f"{MIN_SUPPORTED_NU}. Below nu = 1/2 there is no turning point, the domain lower bound "
            f"degenerates to an arbitrary floor, and the normalized amplitude a_nu ~ x^(1/2 - nu) "
            f"leaves the plausibility band as x -> 0."
        )

    if not math.isfinite(max_x) or max_x > MAX_SUPPORTED_X:
        raise BesselPhaseError(
            f"bessel_phase: max_x={max_x!r} is above the supported ceiling "
            f"MAX_SUPPORTED_X={MAX_SUPPORTED_X:.3g}. Above it the platform sin/cos are no longer "
            f"known to be correctly rounded, so the split evaluation has no certified accuracy."
        )

    min_x = construction_min_x(nu)
    if max_x < min_x:
        raise BesselPhaseError(
            f"bessel_phase: max_x={max_x:.6g} is below the domain lower bound "
            f"min_x={min_x:.6g} for nu={nu}, so the requested interval contains no oscillatory "
            f"region. The construction is not extended below sqrt(nu^2 - 1/4); please increase "
            f"max_x."
        )

    # ---- region 2: the closed-form tail ------------------------------------------------------
    try:
        crossover: TailCrossover = tail_crossover(
            nu, phase_atol, amplitude_rtol, n_terms=tail_terms
        )
    except ValueError as exc:
        raise BesselPhaseError(
            f"bessel_phase: could not place the tail crossover for nu={nu}, "
            f"phase_atol={phase_atol:.3g}, amplitude_rtol={amplitude_rtol:.3g}: {exc}"
        ) from exc

    x_star = crossover.x_star

    # ---- region 1: the sampled near region ---------------------------------------------------
    #
    # The near region is always built over the whole of [min_x, x_star], even when max_x falls
    # below x_star and part of it will never be evaluated. Two things depend on that. The branch
    # anchor takes its integer cycle from the series at the top of the interval, and the series is
    # certified to phase_atol only at x_star itself; and the crossover agreement of §2.2 has
    # nothing to compare unless both representations reach the same point. The cost of the unused
    # part is bounded by the whole near region, which is 817 nodes and 0.04 s at the largest order
    # the campaign supports.
    near: Optional[NearRegionData] = None
    if x_star > min_x:
        panel_width = _initial_panel_width(
            sample_points, math.log(min_x), math.log(max_x), interp_degree
        )
        try:
            near = build_near_region(
                nu,
                min_x,
                x_star,
                phase_atol=phase_atol,
                amplitude_rtol=amplitude_rtol,
                degree=interp_degree,
                initial_panel_width=panel_width,
                tail_terms=tail_terms,
            )
        except NearRegionError as exc:
            if isinstance(exc, AmplitudeBandError):
                raise
            raise BesselPhaseError(
                f"bessel_phase: near-region construction failed for nu={nu}, "
                f"[{min_x:.6g}, {x_star:.6g}]: {exc}"
            ) from exc

        if not near.converged:
            message = (
                f"bessel_phase: the near region for nu={nu} over [{min_x:.6g}, {x_star:.6g}] hit "
                f"its refinement cap ({near.refinement_passes} passes, {near.n_panels} panels) "
                f"without meeting the requested accuracy. Achieved: phase "
                f"{near.achieved_phase_abserr:.3g} rad against phase_atol={phase_atol:.3g}, "
                f"amplitude {near.achieved_amplitude_relerr:.3g} against "
                f"amplitude_rtol={amplitude_rtol:.3g}, theta' "
                f"{near.achieved_deriv_relerr:.3g} against {near.deriv_rtol:.3g}."
            )
            if strict_accuracy:
                raise BesselPhaseAccuracyError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    corrections = _TwoRegionCorrections(nu, min_x, x_star, near, tail_terms)

    # ---- the seam ----------------------------------------------------------------------------
    #
    # The second half of the remainder test DRAFT-PLAN.md §7.2 requires: prompt 03 could only test
    # the series against itself, and this is the test of the series against the sampled region.
    # Nothing is blended, tapered or averaged here; if the two sides disagree, the construction has
    # failed its own test and says so.
    crossover_phase_agreement = 0.0
    crossover_amplitude_agreement = 0.0
    if near is not None:
        u_star = math.log(x_star)
        near_r = float(near.r_interp(u_star))
        near_a = math.exp(float(near.log_a_interp(u_star)))
        tail_r = float(tail_residual(nu, x_star, n_terms=tail_terms))
        tail_a = float(tail_amplitude(nu, x_star, n_terms=tail_terms))

        crossover_phase_agreement = abs(near_r - tail_r)
        crossover_amplitude_agreement = abs(near_a / tail_a - 1.0)

        if (
            crossover_phase_agreement > phase_atol
            or crossover_amplitude_agreement > amplitude_rtol
        ):
            raise BesselPhaseError(
                f"bessel_phase: the two regions disagree at the crossover for nu={nu}. "
                f"x_star={x_star:.10g}: phase agreement {crossover_phase_agreement:.6g} rad "
                f"against phase_atol={phase_atol:.6g} (near {near_r:.17g}, tail {tail_r:.17g}); "
                f"amplitude agreement {crossover_amplitude_agreement:.6g} against "
                f"amplitude_rtol={amplitude_rtol:.6g} (near {near_a:.17g}, tail {tail_a:.17g}). "
                f"Neither side is blended into the other: a seam that has to be hidden is a "
                f"construction that failed its own remainder test."
            )

    # ---- achieved accuracy -------------------------------------------------------------------
    #
    # Every contribution to the phase budget, kept separate so that a later reader can see which
    # one binds: near-region interpolation, the scaled-Hankel sampling floor it cannot see, the
    # series remainder in the tail, the seam, and the double-precision resolution of |r| itself.
    if near is not None:
        max_abs_r = float(np.max(np.abs(near.r_nodes)))
    else:
        max_abs_r = abs(
            float(tail_residual(nu, max(min_x, x_star), n_terms=tail_terms))
        )
    residual_resolution = float(np.finfo(float).eps) * max_abs_r

    near_phase = (
        0.0 if near is None else near.achieved_phase_abserr + SAMPLED_PHASE_FLOOR
    )
    near_amplitude = (
        0.0
        if near is None
        else near.achieved_amplitude_relerr + SAMPLED_AMPLITUDE_FLOOR
    )
    near_deriv = (
        0.0
        if near is None
        else near.achieved_deriv_relerr + 2.0 * SAMPLED_AMPLITUDE_FLOOR
    )

    # The series remainder carries DECLARED_SERIES_SAFETY here and nowhere else: the estimator is
    # 0.98-1.02 times the truth, which is a margin of 0.006% in the direction that matters, and a
    # declared error is exactly the place where that is not good enough. The raw estimator is still
    # reported, under its own name, in the accuracy dict below. Both terms are largest at x_star
    # (each decays monotonically in x), so these are the maxima over the tail and hence the
    # domain-wide bounds theta_abserr_at() is capped by.
    tail_phase = DECLARED_SERIES_SAFETY * crossover.first_omitted_at_x_star
    tail_amplitude_err = DECLARED_SERIES_SAFETY * crossover.amplitude_error_at_x_star
    tail_deriv = 2.0 * tail_amplitude_err

    theta_abserr = max(
        near_phase,
        tail_phase,
        crossover_phase_agreement,
        residual_resolution,
        EVALUATION_FLOOR,
    )
    amplitude_relerr = max(
        near_amplitude,
        tail_amplitude_err,
        crossover_amplitude_agreement,
        EVALUATION_FLOOR,
    )
    theta_deriv_relerr = max(near_deriv, tail_deriv, 2.0 * EVALUATION_FLOOR)

    accuracy = {
        "phase_atol": phase_atol,
        "amplitude_rtol": amplitude_rtol,
        "theta_abserr": theta_abserr,
        "amplitude_relerr": amplitude_relerr,
        "theta_deriv_relerr": theta_deriv_relerr,
        "near_region_phase_abserr": (
            None if near is None else near.achieved_phase_abserr
        ),
        "near_region_amplitude_relerr": (
            None if near is None else near.achieved_amplitude_relerr
        ),
        "near_region_deriv_relerr": (
            None if near is None else near.achieved_deriv_relerr
        ),
        "near_region_deriv_alt_relerr": (
            None if near is None else near.achieved_deriv_alt_relerr
        ),
        "sampled_phase_floor": 0.0 if near is None else SAMPLED_PHASE_FLOOR,
        "sampled_amplitude_floor": 0.0 if near is None else SAMPLED_AMPLITUDE_FLOOR,
        # the *raw* estimator, without DECLARED_SERIES_SAFETY, so that a reader can see both the
        # measurement and the margin applied to it
        "tail_first_omitted_at_x_star": crossover.first_omitted_at_x_star,
        "tail_amplitude_at_x_star": crossover.amplitude_error_at_x_star,
        "declared_series_safety": DECLARED_SERIES_SAFETY,
        "crossover_phase_agreement": crossover_phase_agreement,
        "crossover_amplitude_agreement": crossover_amplitude_agreement,
        "residual_resolution": residual_resolution,
        "evaluation_floor": EVALUATION_FLOOR,
        "max_abs_residual": max_abs_r,
        "accuracy_met": True if near is None else bool(near.converged),
    }

    # ---- assembly ----------------------------------------------------------------------------
    mod = BesselAmplitude(corrections, min_x=min_x, max_x=max_x)
    phase = BesselPhaseFunction(
        nu,
        corrections,
        min_x=min_x,
        max_x=max_x,
        amplitude=mod,
        theta_abserr=theta_abserr,
    )

    return _BesselPhaseData(
        {
            "phase": phase,
            "mod": mod,
            # Identically zero, and reported rather than dropped: DRAFT-PLAN.md §4.3 shows the old
            # root solve's offset was a pure artefact of its own loose tolerances, at a match point
            # where the phase was already exact. There is no root solve here.
            "phi": 0.0,
            "bessel_j": phase.bessel_j,
            "bessel_y": phase.bessel_y,
            "min_x": min_x,
            "max_x": max_x,
            "nu": nu,
            "x_star": x_star,
            "theta_abserr": theta_abserr,
            "amplitude_relerr": amplitude_relerr,
            "theta_deriv_relerr": theta_deriv_relerr,
            "accuracy": accuracy,
            "accuracy_met": accuracy["accuracy_met"],
            "crossover": crossover,
            "near_region": near,
        }
    )
