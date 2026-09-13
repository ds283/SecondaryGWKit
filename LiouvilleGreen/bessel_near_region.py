"""
Near-region sampler for the Bessel amplitude and phase construction.

This module supplies the **near region** of the two-region construction of
``prompts/transfer-remedial/DRAFT-PLAN.md`` §1, §7.1 and §7.3: below the crossover ``x_star(nu)``
that :func:`LiouvilleGreen.bessel_tail.tail_crossover` places, the normalized amplitude ``a_nu``
and the phase residual ``r_nu`` are *sampled* from the exponentially scaled Hankel function,
branch-tracked into a continuous ``r_nu``, and interpolated in ``u = log x``. Above ``x_star``
nothing here is used at all; that region is closed form and belongs to
:mod:`LiouvilleGreen.bessel_tail`.

Convention
----------

The repository's Bessel convention (which is **not** DLMF's; the two differ by exactly ``+pi/2``,
absorbed into ``c_nu``) is

    J_nu(x) = A_nu(x) sin theta_nu(x),        Y_nu(x) = -A_nu(x) cos theta_nu(x),

with ``theta_nu`` increasing in ``x`` and

    theta_nu(x) = x + c_nu + r_nu(x),   A_nu(x) = sqrt(2/(pi x)) a_nu(x),   c_nu = pi/4 - pi nu/2.

Do not "correct" this towards DLMF's convention.

The sampled quantity, and why it is exact
-----------------------------------------

SciPy defines ``hankel1e(nu, x) = exp(-i x) H^(1)_nu(x)`` with ``H^(1)_nu = J_nu + i Y_nu``. Under
the convention above ``H^(1)_nu = A_nu exp(i(theta_nu - pi/2))``, so the two ``pi/4`` terms and the
``-pi/2`` cancel in

    S_nu(x) = sqrt(pi x / 2) exp(i(pi nu/2 + pi/4)) hankel1e(nu, x) = a_nu(x) exp(i r_nu(x))

and the identity is **exact, not asymptotic** (campaign ``README.md`` §2 (a); verified to
7.6e-15 in reconstructed ``J``, ``Y`` in ``RECONCILIATION.md`` §1). Therefore ``a_nu = |S_nu|``
and ``r_nu`` is the continuously tracked argument of ``S_nu``.

**The residual is never obtained by subtracting ``x`` from a large computed phase.** The scaled
routine supplies the oscillation-removed quantity directly, and that is the whole reason this
route works: at ``x = 1e8`` the difference ``(2/pi)/m - x`` has already lost every significant
digit (``DRAFT-PLAN.md`` §5.2).

Two small refinements to the sampling, both deliberate:

* the constant rotation angle ``pi nu/2 + pi/4 = pi (2 nu + 1)/4`` is reduced mod ``2 pi``
  *before* it is multiplied by ``pi`` (:func:`scaled_hankel_phase_constant`). Forming
  ``pi * 1000.5 / 2 = 1571.57...`` first and exponentiating it injects an argument-reduction error
  of order ``eps * 1571 ~ 3.5e-13`` into every sampled residual -- which is precisely the
  ``2.96e-13`` sampling floor tabulated for ``nu = 1000.5`` in ``DRAFT-PLAN.md`` §4.4. For a
  half-integer order ``(2 nu + 1)/4`` is exact in binary and ``fmod(., 2)`` is exact, so the
  reduced constant costs one rounding of ``pi * q``, i.e. ``<= 7e-16``;
* ``a_nu`` is formed as ``sqrt(pi x/2) |hankel1e|`` rather than by taking the modulus of the
  rotated value, since a unit rotation cannot change a modulus but can perturb it in floating
  point.

Why there is no ``(div_2pi, mod_2pi)`` representation here
----------------------------------------------------------

``r_nu`` is interpolated directly as a float, and neither ``phase_spline`` nor ``simple_mod_2pi``
is used. **The reason is not that the residual is small.** Measured on a continuously tracked
branch, ``r`` runs from 570.82 rad at the bottom of the domain to 5.00 rad at ``x = 100 nu`` for
``nu = 1000.5`` -- 90.05 cycles traversed (``RECONCILIATION.md`` C2, correcting
``DRAFT-PLAN.md`` §4.6). The reason is that the double-precision resolution of a quantity of that
size, ``eps * 571 ~ 1.3e-13``, sits an order of magnitude below the campaign's ``1e-11`` low-order
accuracy target and seven orders below its ``1e-6`` high-order target, so a split representation
would buy nothing that the budgets can see.

Why the near region is the hard part
------------------------------------

The naive expectation that the residual is always better conditioned than the full phase is false
near the turning point. From the exact identity ``dr/d log x = x (a^-2 - 1)``, the gain over the
full phase is 2.0 at ``nu = 1.5`` but 0.21 at ``nu = 100.5`` and **0.086** at ``nu = 1000.5``, all
at ``x = nu`` (``DRAFT-PLAN.md`` §4.5): at the turning point the residual is eleven times *worse*
conditioned. The split pays only for ``x >~ 1.5 nu`` and pays enormously only in the tail. Two
consequences shape everything below:

1. refinement near the turning point is mandatory and order dependent, and adaptivity must be
   **two-sided** -- in the tail ``r ~ C exp(-u)``, every ``u``-derivative is as small as ``r``
   itself, and the required node density collapses far below the 250 per e-fold of the prototype;
2. at a uniform 250 per e-fold the residual advance per interval reaches **3.685 rad at
   ``nu = 1000.5``**, above ``pi``. An ordinary ``unwrap`` cannot recover from that, raising the
   interpolation degree cannot repair an incorrectly unwrapped sample, and no density fixes it
   after the fact. Branch tracking, not sample density, is the binding requirement at high order.

The construction
----------------

*Panels.* The interval ``[log x_lo, log x_star]`` is divided into panels, each carrying a
Chebyshev-Lobatto grid of degree :data:`DEFAULT_PANEL_DEGREE`. Panels are **bisected** where any
acceptance criterion fails and left alone where none does, so refinement is automatically
concentrated at the turning point and the tail keeps the coarse initial spacing -- the two-sided
adaptivity of ``DRAFT-PLAN.md`` §4.5. Piecewise Chebyshev was chosen over a global quintic spline
for three reasons: the coefficient tail is a per-panel error estimate that costs nothing;
differentiation is spectral rather than a divided difference; and Lobatto grids put a node on each
panel edge, so adjacent panels agree there to the last bit and interpolation-boundary continuity
is exact by construction rather than by tolerance.

*Branch tracking.* The variation estimator is the exact ``dr/d log x = x (a^-2 - 1)``, evaluated
from the sampled ``a`` at the nodes we already have. Its panel Chebyshev interpolant is integrated
spectrally to predict the residual increment between consecutive nodes; each node's principal
angle is then snapped to the branch nearest that prediction, and the snap discrepancy is checked
against :data:`BRANCH_CONSISTENCY_FRACTION` times ``pi``. Independently, every node gap must
satisfy ``max|dr/d log x| * h <= BRANCH_SAFETY_FRACTION * pi``, so that the resolution would be
unique on the crude bound alone. Endpoint principal-angle differences are **never** used on their
own: an interval may contain an undetected full turn, which is exactly the failure the estimator
exists to exclude. There is no ``np.unwrap`` and no root solve anywhere in this module.

*Anchor.* The residual is defined only up to a constant integer number of cycles by ``J`` and
``Y`` alone, since ``theta -> theta + 2 pi n`` leaves both unchanged. This module fixes that
freedom to the branch on which ``r -> 0`` as ``x -> infinity`` -- the branch the tail series and
``RECONCILIATION.md`` C2 use -- by taking the integer cycle at the topmost node from
:func:`LiouvilleGreen.bessel_tail.tail_residual` at ``x_star``. Only the *integer* comes from the
series; the value is the sampled ``arg S_nu``, which uses ``J`` and ``Y`` together. So the series'
own error (at most ``phase_atol``, by the construction of ``x_star``) is not transferred into the
residual, and the anchor needs no root solve -- ``DRAFT-PLAN.md`` §4.3 measures the existing
``phi`` root solve to be a pure artefact worth ``-4.8e-8``.

*The guard.* Every sample is validated against a two-sided band on ``a_nu``, never against
``isfinite``: ``hankel1e(100.5, 1e9)`` is exactly ``-0j``, which *is* finite, so a finite-value
check passes it and ``log(abs(.))`` becomes ``-inf``. See :func:`check_amplitude_band`.

Accuracy reporting
------------------

:class:`NearRegionData` carries three ``achieved_*`` numbers, measured by resampling the scaled
Hankel function at points interior to each panel and comparing against the interpolants there.
They are **practical estimators, not mathematical supremum bounds** (campaign ``README.md`` §6),
and an honest over-estimate is much better than an optimistic under-estimate: prompt 05 propagates
them into the Levin quadrature's ``theta_abserr``, whose entire purpose is that "the caller sees an
honest number instead of an artificially small one" (``AdaptiveLevin/levin_quadrature.py:2360``).

When the refinement cap binds, :func:`build_near_region` returns with ``converged = False`` and the
achieved errors as measured, rather than silently accepting them: the decision to raise belongs to
the caller that owns the accuracy contract (``DRAFT-PLAN.md`` §9 Stage 2).

Module boundary
---------------

The public surface is a request for sampled data and interpolants over ``[x_lo, x_star]``. This
module knows nothing about ``bessel_phase``'s dict, its accessors, or its consumers, and it does
not import it. That is deliberate: ``DRAFT-PLAN.md`` §5.3's hybrid -- quadrature near the turning
point, series in the tail -- is the designated fallback if a SciPy change ever degrades
``hankel1e``'s phase, and it must be substitutable for this region alone without touching the tail,
the interpolation scheme, the evaluation path or the consumers.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy.special import hankel1e

from LiouvilleGreen.bessel_tail import DEFAULT_TAIL_TERMS, tail_residual

TWO_PI = 2.0 * math.pi

#: Lower edge of the two-sided plausibility band on ``a_nu``.
#:
#: ``a_nu`` is monotone decreasing in ``x`` and tends to 1 from above, and the measured minimum
#: over the whole near region is 1.0000000000 (``nu = 1/2``, where it is exactly 1) rising only to
#: 1.0000250016 at ``nu = 1000.5`` (``RECONCILIATION.md`` §3.1). A lower edge of 0.99 therefore
#: leaves a 1% margin below a quantity that is never below 1, and rejects the ``-0j`` failure of
#: ``README.md`` §2 (e), which lands at ``a = 0``.
AMPLITUDE_BAND_LO = 0.99

#: Upper edge of the two-sided plausibility band on ``a_nu``.
#:
#: ``a_nu`` attains its maximum at the turning point and that maximum grows only like ``nu^(1/6)``:
#: measured 1.000000 (``nu = 1/2``), 1.32288 (5/2), 1.85684 (20.5), 2.41795 (100.5), 3.54597
#: (1000.5) (``RECONCILIATION.md`` §3.1). An upper edge of 8.0 clears the largest order the
#: campaign supports by a factor 2.26, which on the ``nu^(1/6)`` scaling is headroom to
#: ``nu ~ 1.3e5``, and still rejects a spuriously *large* value -- which a one-sided ``a >~ 1``
#: test would let through.
AMPLITUDE_BAND_HI = 8.0

#: Chebyshev degree per panel. 16 gives 17 Lobatto nodes per panel.
#:
#: ``DRAFT-PLAN.md`` §4.7 and §6.2 make quintic the starting candidate for a spline-based design
#: (cubic misses the 1e-9 derivative target at every order). A degree-16 Chebyshev panel is a
#: strictly stronger local approximation and, unlike a fixed-degree spline, its coefficient tail
#: reports the error it is achieving, which is what drives the refinement here. Degree alone is
#: never the acceptance criterion (``DRAFT-PLAN.md`` §7.3): see :func:`build_near_region`.
#:
#: 16 is where the node count bottoms out. Swept at ``phase_atol = amplitude_rtol = 1e-11``, total
#: nodes for degree 8, 10, 12, 16, 20, 24 are 265/111/97/113/141/169 at ``nu = 5/2``,
#: 793/321/217/177/201/241 at 100.5 and 1393/861/829/817/841/841 at 1000.5 -- flat to within 15 %
#: from 12 to 24 at every order, and rising sharply below 10.
DEFAULT_PANEL_DEGREE = 16

#: Smallest panel degree :func:`build_near_region` accepts.
#:
#: Not a taste constraint. :func:`_coefficient_tail` estimates the truncation error as the sum of
#: the last three Chebyshev coefficients, and below degree 8 those three are a large fraction of
#: the function itself rather than of its truncation error, so the criterion never clears however
#: small the panel becomes. Measured at ``nu = 100.5``: degree 4 runs to the 8192-panel cap with
#: ``converged = False`` while its *interior residual* is already 1.3e-13, and degree 6 needs 1059
#: panels where degree 8 needs 99.
MIN_PANEL_DEGREE = 8

#: Width in e-folds of the initial, unrefined panels.
#:
#: With :data:`DEFAULT_PANEL_DEGREE` this is ~32 nodes per e-fold, i.e. about eight times *coarser*
#: than the 250 per e-fold of the prototype. That coarseness is the point: in the tail
#: ``r ~ C exp(-u)`` and the required density collapses (``DRAFT-PLAN.md`` §4.5), and panels there
#: are accepted at this spacing while panels at the turning point are bisected repeatedly.
DEFAULT_INITIAL_PANEL_WIDTH = 0.5

#: Fraction of ``pi`` that ``max|dr/d log x| * h`` may reach across one node gap.
#:
#: One half. The bound is the crude, estimator-only guarantee that a gap cannot conceal a wrap:
#: if the total variation of ``r`` across a gap is below ``pi`` then the principal-angle
#: difference determines the branch uniquely, and one half leaves a factor two for the fact that
#: ``max`` is taken over the two endpoints rather than over the interior. It is enforced *in
#: addition to* the spectral prediction of §"Branch tracking" above, not instead of it, so that
#: branch resolution does not rest on the quality of the predictor alone.
BRANCH_SAFETY_FRACTION = 0.5

#: Fraction of ``pi`` that the snap discrepancy ``|r_resolved - r_predicted|`` may reach.
#:
#: One quarter. The predicted increment is a spectral integral of the exact ``dr/d log x`` over the
#: gap, so on a resolved panel the discrepancy is many orders below this; a value approaching it
#: means the estimator no longer describes the interval, and the panel is bisected.
BRANCH_CONSISTENCY_FRACTION = 0.25

#: Maximum number of refinement passes. Each pass bisects every panel that failed a criterion.
DEFAULT_MAX_REFINEMENT_PASSES = 24

#: Maximum number of panels. Refinement stops rather than exceeding it.
DEFAULT_MAX_PANELS = 8192

#: ``x_star / x_lo - 1`` below which the near region is treated as empty.
DEGENERATE_SPAN_TOL = 1e-12

#: Slack in ``u``, outside the constructed interval, that the interpolants will still evaluate.
INTERPOLANT_RANGE_SLACK = 1e-9


class NearRegionError(ValueError):
    """Base class for near-region construction failures."""


class AmplitudeBandError(NearRegionError):
    """A sampled ``a_nu`` fell outside the two-sided plausibility band."""


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------


def scaled_hankel_phase_constant(nu: float) -> float:
    """
    The rotation angle ``pi nu/2 + pi/4`` of ``S_nu``, reduced mod ``2 pi`` before the
    multiplication by ``pi``.

    Written as ``pi * fmod((2 nu + 1)/4, 2)``. For a half-integer order ``(2 nu + 1)/4`` is exact
    in binary and ``fmod`` is exact, so the result carries a single rounding of ``pi * q`` with
    ``|q| <= 2``, i.e. an error below ``7e-16``. Forming ``pi * nu / 2 + pi / 4`` directly and
    handing it to ``exp(i .)`` instead costs ``eps * pi nu / 2``, which is ``3.5e-13`` at
    ``nu = 1000.5`` -- the sampling floor ``DRAFT-PLAN.md`` §4.4 measures at that order.
    """
    return math.pi * math.fmod((2.0 * nu + 1.0) / 4.0, 2.0)


def check_amplitude_band(
    nu: float,
    x,
    a,
    hankel=None,
    lo: float = AMPLITUDE_BAND_LO,
    hi: float = AMPLITUDE_BAND_HI,
) -> None:
    """
    Validate sampled normalized amplitudes against the two-sided plausibility band
    ``[lo, hi]`` and raise :class:`AmplitudeBandError` on the first offender.

    **This is deliberately not an ``isfinite`` check.** ``hankel1e(100.5, 1e9)`` returns exactly
    ``-0j``: ``np.isfinite`` passes it, ``np.abs`` gives 0 and ``np.log(np.abs(.))`` gives
    ``-inf``, so a finite-value guard admits the failure straight into a log-amplitude
    interpolant (``README.md`` §2 (e), ``DRAFT-PLAN.md`` §4.4). The silent boundary is
    ``x ~ 7.13e8`` for ``nu >~ 86`` and ``x ~ 2.25e15`` for every order; with the crossover at or
    below ``100 nu`` the sampler stays at least three decades below it even at ``nu = 1000.5``,
    so this guard is defence in depth rather than a routine occurrence.

    The band is two-sided because a one-sided ``a >~ 1`` test would pass a spuriously *large*
    value; see :data:`AMPLITUDE_BAND_LO` and :data:`AMPLITUDE_BAND_HI` for the measurements the
    constants are set against.

    :param hankel: the raw ``hankel1e`` values, reported in the message when supplied. They are
        what a reader needs in order to recognise the ``-0j`` failure for what it is.
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    av = np.atleast_1d(np.asarray(a, dtype=float))

    bad = ~((av >= lo) & (av <= hi))
    if not np.any(bad):
        return

    index = int(np.argmax(bad))
    detail = ""
    if hankel is not None:
        hv = np.atleast_1d(np.asarray(hankel))
        detail = f", hankel1e(nu, x) = {hv[index]!r}"

    raise AmplitudeBandError(
        f"bessel_near_region: the scaled-Hankel sample at nu={nu!r}, x={float(xv[index]):.17g} "
        f"gave a normalized amplitude a={float(av[index]):.17g}, outside the plausibility band "
        f"[{lo:.6g}, {hi:.6g}]{detail}. {int(np.count_nonzero(bad))} of {av.size} samples in this "
        f"batch failed. a_nu is monotone decreasing in x, tends to 1 from above and is at most "
        f"3.546 for every order up to 1000.5 (RECONCILIATION.md 3.1), so a value outside the band "
        f"is a failure of the sampled function, not of the construction. Note that a = 0 is the "
        f"signature of scipy.special.hankel1e returning exactly -0j, which is finite and which an "
        f"isfinite check would therefore have accepted."
    )


def sample_scaled_hankel(nu: float, x, band: Tuple[float, float] = None):
    """
    Sample ``S_nu(x) = sqrt(pi x/2) exp(i(pi nu/2 + pi/4)) hankel1e(nu, x)`` and return
    ``(a, raw_angle, hankel)``.

    ``a = |S_nu|`` is the normalized amplitude and ``raw_angle`` is ``arg S_nu`` *without* any
    branch resolution -- it is correct modulo ``2 pi`` only, and :func:`build_near_region` is what
    makes it continuous. The identity is exact, not asymptotic; see the module docstring.

    Every returned batch is validated against the plausibility band unless ``band`` is ``None``,
    in which case the caller is taking responsibility for the guard.
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    h = hankel1e(nu, xv)

    # |S| is computed from |hankel1e| rather than from the rotated value: a unit rotation cannot
    # change a modulus mathematically, but it can perturb it in floating point.
    a = np.sqrt(0.5 * math.pi * xv) * np.abs(h)
    raw_angle = np.angle(h) + scaled_hankel_phase_constant(nu)

    if band is not None:
        check_amplitude_band(nu, xv, a, hankel=h, lo=band[0], hi=band[1])

    return a, raw_angle, h


def residual_log_derivative(x, a):
    """
    The exact variation estimator ``dr_nu/d log x = x (a_nu^-2 - 1)``.

    Exact rather than approximate: the Wronskian gives ``theta' = a^-2``, and
    ``theta = x + c_nu + r`` gives ``theta' = 1 + r'(x)``, so ``r'(x) = a^-2 - 1`` identically and
    ``dr/d log x = x r'``. It is computed from the *sampled* ``a``, so it costs nothing beyond the
    samples already taken.

    Its known limitation is that ``a^-2 - 1`` cancels in the tail (``DRAFT-PLAN.md`` §5.2), which
    is harmless here: the tail is closed form, and the estimator is needed only where ``a^-2 - 1``
    is ``O(1)``. In the outer part of the near region it retains ~11 significant digits, which is
    ample for an estimator whose job is to bound a variation against ``pi``.
    """
    xv = np.asarray(x, dtype=float)
    av = np.asarray(a, dtype=float)
    return xv * (1.0 / (av * av) - 1.0)


# ----------------------------------------------------------------------------------------------
# Piecewise Chebyshev interpolation on Lobatto panels
# ----------------------------------------------------------------------------------------------


_LOBATTO_INVERSE_CACHE: Dict[int, np.ndarray] = {}


def _lobatto_nodes(degree: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes of the given degree on ``[-1, 1]``, ascending, endpoints included."""
    j = np.arange(degree + 1)
    return -np.cos(math.pi * j / degree)


def _lobatto_inverse(degree: int) -> np.ndarray:
    """Cached inverse of the Chebyshev-Vandermonde matrix on the Lobatto nodes."""
    cached = _LOBATTO_INVERSE_CACHE.get(degree)
    if cached is None:
        vander = _cheb.chebvander(_lobatto_nodes(degree), degree)
        cached = np.linalg.inv(vander)
        _LOBATTO_INVERSE_CACHE[degree] = cached
    return cached


def _coefficients_from_values(values: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev coefficients of the degree-``degree`` interpolant through Lobatto values."""
    return _lobatto_inverse(degree) @ np.asarray(values, dtype=float)


def _coefficient_tail(coefficients: np.ndarray, count: int = 3) -> float:
    """
    A per-panel error estimate: the sum of the magnitudes of the last ``count`` Chebyshev
    coefficients.

    For an analytic function the coefficients decay geometrically and the truncation error is
    dominated by the first omitted term, which the last retained ones bracket. Summing three
    rather than taking one guards against an accidental near-zero coefficient from a parity
    coincidence. Like every other number this module reports, it is a practical estimator and not
    a bound.
    """
    c = np.abs(np.asarray(coefficients, dtype=float))
    return float(np.sum(c[-count:]))


class PiecewiseChebyshev:
    """
    A Chebyshev interpolant on contiguous panels of ``u``, with spectral differentiation.

    Panels share their edges, and the underlying grids are Lobatto grids that include those edges,
    so the interpolant is exactly continuous at every panel boundary: the two panels interpolate
    the same sampled value there, bit for bit. The derivative is continuous only to the accuracy
    of the fits, which is what :func:`build_near_region` measures and reports.
    """

    def __init__(
        self, breakpoints: Sequence[float], coefficients: Sequence[np.ndarray]
    ):
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        if self.breakpoints.ndim != 1 or self.breakpoints.size != len(coefficients) + 1:
            raise ValueError(
                "PiecewiseChebyshev: need exactly one more breakpoint than coefficient block, "
                f"got {self.breakpoints.size} breakpoints and {len(coefficients)} blocks."
            )
        self.coefficients = [np.asarray(c, dtype=float) for c in coefficients]
        self._series = [
            _cheb.Chebyshev(c, domain=[self.breakpoints[i], self.breakpoints[i + 1]])
            for i, c in enumerate(self.coefficients)
        ]

    @property
    def n_panels(self) -> int:
        return len(self._series)

    def __call__(self, u):
        uv = np.asarray(u, dtype=float)
        scalar = uv.ndim == 0
        uv = np.atleast_1d(uv)

        lo = self.breakpoints[0] - INTERPOLANT_RANGE_SLACK
        hi = self.breakpoints[-1] + INTERPOLANT_RANGE_SLACK
        if np.any(uv < lo) or np.any(uv > hi):
            raise ValueError(
                f"PiecewiseChebyshev: u outside the constructed interval "
                f"[{self.breakpoints[0]:.17g}, {self.breakpoints[-1]:.17g}]; got "
                f"min(u)={float(np.min(uv)):.17g}, max(u)={float(np.max(uv)):.17g}. The near "
                f"region is defined only up to the crossover x_star; use the closed-form tail "
                f"above it."
            )

        index = np.searchsorted(self.breakpoints, uv, side="right") - 1
        index = np.clip(index, 0, len(self._series) - 1)

        out = np.empty_like(uv)
        for panel in np.unique(index):
            mask = index == panel
            out[mask] = self._series[int(panel)](uv[mask])

        return float(out[0]) if scalar else out

    def derivative(self) -> "PiecewiseChebyshev":
        """The spectral derivative ``d/du``, as another :class:`PiecewiseChebyshev`."""
        return PiecewiseChebyshev(
            self.breakpoints, [s.deriv().coef for s in self._series]
        )

    def on_panel(self, index: int, u):
        """
        Evaluate one panel's polynomial, ignoring which panel ``u`` actually falls in.

        Exists so that continuity at a shared edge can be measured as the difference of the two
        one-sided limits *at the edge itself*, rather than as a difference across a finite step,
        which would be dominated by the genuine slope: ``dr/du`` reaches 920 at ``nu = 1000.5``,
        so even a step of ``1e-11`` contributes ``1e-8`` and swamps the quantity of interest.
        """
        uv = np.asarray(u, dtype=float)
        scalar = uv.ndim == 0
        out = self._series[int(index)](np.atleast_1d(uv))
        return float(out[0]) if scalar else out


class _ConstantInterpolant(PiecewiseChebyshev):
    """
    A constant interpolant over a nominal unit-width panel, used only for a degenerate (empty)
    near region.

    A zero-width panel cannot be mapped to ``[-1, 1]``, so the nominal panel is widened to unit
    width about ``u_lo``. That is safe precisely because the value is constant: for ``nu = 1/2``,
    where the degenerate case actually arises, ``mu - 1 = 0`` makes ``r == 0`` and ``a == 1``
    identically at *every* ``x``, so the constant is not an extrapolation but the exact answer.
    """

    def __init__(self, u_lo: float, value: float):
        super().__init__([u_lo - 0.5, u_lo + 0.5], [np.array([float(value)])])


# ----------------------------------------------------------------------------------------------
# The near-region data structure
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NearRegionData:
    """
    Sampled data and interpolants for the near region ``[x_lo, x_star]``.

    :ivar nu: the order.
    :ivar x_lo: the bottom of the sampled region, as supplied.
    :ivar x_star: the crossover, as supplied. Nothing here was sampled above it.
    :ivar log_x_nodes: ``u = log x`` at every sample node, ascending, panel edges shared once.
    :ivar a_nodes: the sampled ``a_nu`` at those nodes.
    :ivar r_nodes: the **continuously tracked** ``r_nu`` at those nodes, on the branch with
        ``r -> 0`` as ``x -> infinity``. Not reduced mod ``2 pi``, and it must never be: at
        ``nu = 1000.5`` it reaches 570.82 rad at the bottom of the domain.
    :ivar r_interp: ``u -> r``, with ``.derivative()`` giving ``u -> dr/du``.
    :ivar log_a_interp: ``u -> ell = log a``, with ``.derivative()`` giving ``u -> d ell/du``.
        Interpolating ``ell`` rather than ``a`` preserves positivity and makes
        ``theta' = a^-2 = exp(-2 ell)`` a value read off an interpolant rather than a
        differentiated one (``README.md`` §2 (f), ``DRAFT-PLAN.md`` §7.3).
    :ivar achieved_phase_abserr: estimated ``max |r_interp - r|`` in radians, measured at points
        interior to the panels. A practical estimator, not a supremum bound.
    :ivar achieved_amplitude_relerr: the same for ``ell``, which is ``|delta a / a|`` to first
        order.
    :ivar achieved_deriv_relerr: the same for the shipped derivative route
        ``theta' = exp(-2 ell)``, relative, against ``a^-2`` from freshly sampled scaled-Hankel
        values at the interior points. To first order it is exactly twice
        ``achieved_amplitude_relerr``, since ``delta theta'/theta' = -2 delta ell``.
    :ivar achieved_deriv_alt_relerr: the same for the *alternative* route ``1 + r_u/x``, which
        differentiates the residual interpolant instead of reading a value off the amplitude
        interpolant. Reported, never used to drive refinement. Differentiating an interpolant
        amplifies the sampling floor of ``r`` by roughly ``N^2/h``, so this estimate *grows* as
        panels shrink: driving refinement with it produces a loop that diverges rather than
        converges (measured at ``nu = 1000.5``: it stalls at 1-10x any fixed threshold while the
        failing-panel count doubles every pass). It is the independent consistency check of
        ``README.md`` §2 (f), for which reporting is the right treatment, and it is also why
        ``DRAFT-PLAN.md`` §4.7 prefers ``exp(-2 ell)`` in the first place.
    :ivar phase_atol: the phase budget requested.
    :ivar amplitude_rtol: the amplitude budget requested.
    :ivar deriv_rtol: the derivative budget requested.
    :ivar interp_degree: Chebyshev degree per panel.
    :ivar refinement_passes: number of bisection passes executed.
    :ivar wraps_tracked: number of ``2 pi`` branch boundaries the tracker resolved across the
        region. Not decoration: it is the campaign's evidence that branch tracking was *verified*
        rather than assumed (``README.md`` §6, the high-order structural row).
    :ivar n_panels: final panel count.
    :ivar panel_edges: the final panel edges in ``u``.
    :ivar converged: ``True`` if every acceptance criterion was met before the refinement cap.
        When ``False`` the ``achieved_*`` values are the measured, unmet errors; the decision to
        raise belongs to the caller.
    :ivar degenerate: ``True`` if ``x_star <= x_lo``, i.e. the near region is empty and the whole
        domain is closed form. Holds identically at ``nu = 1/2``.
    :ivar amplitude_band: the plausibility band actually enforced.
    :ivar max_branch_advance: the largest ``max|dr/d log x| * h`` over any node gap, in radians.
        Compare against ``BRANCH_SAFETY_FRACTION * pi``.
    :ivar max_branch_snap: the largest ``|r_resolved - r_predicted|`` over any node, in radians.
        Compare against ``BRANCH_CONSISTENCY_FRACTION * pi``.
    """

    nu: float
    x_lo: float
    x_star: float
    log_x_nodes: np.ndarray
    a_nodes: np.ndarray
    r_nodes: np.ndarray
    r_interp: PiecewiseChebyshev
    log_a_interp: PiecewiseChebyshev
    achieved_phase_abserr: float
    achieved_amplitude_relerr: float
    achieved_deriv_relerr: float
    achieved_deriv_alt_relerr: float
    phase_atol: float
    amplitude_rtol: float
    deriv_rtol: float
    interp_degree: int
    refinement_passes: int
    wraps_tracked: int
    n_panels: int
    panel_edges: np.ndarray
    converged: bool
    degenerate: bool
    amplitude_band: Tuple[float, float]
    max_branch_advance: float
    max_branch_snap: float

    @property
    def n_nodes(self) -> int:
        return int(self.log_x_nodes.size)


# ----------------------------------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------------------------------


class _Panel:
    """Sampled scaled-Hankel data on one Lobatto panel, plus the interior test points."""

    __slots__ = (
        "u0",
        "u1",
        "u_nodes",
        "x_nodes",
        "a_nodes",
        "raw_angle",
        "f_nodes",
        "f_antiderivative",
        "u_test",
        "x_test",
        "a_test",
        "raw_angle_test",
    )

    def __init__(self, nu: float, u0: float, u1: float, degree: int, band):
        self.u0 = u0
        self.u1 = u1

        t = _lobatto_nodes(degree)
        mid = 0.5 * (u0 + u1)
        half = 0.5 * (u1 - u0)
        self.u_nodes = mid + half * t
        # The Lobatto nodes are symmetric, so the reconstructed edges are exact to a rounding;
        # pin them so that adjacent panels share their edge bit for bit.
        self.u_nodes[0] = u0
        self.u_nodes[-1] = u1

        self.x_nodes = np.exp(self.u_nodes)
        self.a_nodes, self.raw_angle, _ = sample_scaled_hankel(
            nu, self.x_nodes, band=band
        )
        self.f_nodes = residual_log_derivative(self.x_nodes, self.a_nodes)

        f_series = _cheb.Chebyshev(
            _coefficients_from_values(self.f_nodes, degree), domain=[u0, u1]
        )
        self.f_antiderivative = f_series.integ()

        # Interior test points: the midpoints in u of consecutive Lobatto nodes. They are interior
        # to the panel by construction and are never nodes of it, so comparing the interpolant
        # against fresh samples there is a genuine residual test rather than a tautology.
        self.u_test = 0.5 * (self.u_nodes[:-1] + self.u_nodes[1:])
        self.x_test = np.exp(self.u_test)
        self.a_test, self.raw_angle_test, _ = sample_scaled_hankel(
            nu, self.x_test, band=band
        )

    def increments(self) -> np.ndarray:
        """Predicted ``r`` increments between consecutive nodes, by spectral integration of ``f``."""
        primitive = self.f_antiderivative(self.u_nodes)
        return np.diff(primitive)


def _degenerate_result(
    nu: float,
    x_lo: float,
    x_star: float,
    phase_atol: float,
    amplitude_rtol: float,
    deriv_rtol: float,
    degree: int,
    band: Tuple[float, float],
    anchor_residual: Optional[float],
    tail_terms: int,
) -> NearRegionData:
    """The near region is empty: one sample at ``x_lo``, constant interpolants, no tracking."""
    u_lo = math.log(x_lo)
    a, raw_angle, _ = sample_scaled_hankel(nu, np.array([x_lo]), band=band)

    anchor = (
        float(anchor_residual)
        if anchor_residual is not None
        else float(tail_residual(nu, x_lo, n_terms=tail_terms))
    )
    r0 = float(raw_angle[0] + TWO_PI * round((anchor - raw_angle[0]) / TWO_PI))

    return NearRegionData(
        nu=nu,
        x_lo=x_lo,
        x_star=x_star,
        log_x_nodes=np.array([u_lo]),
        a_nodes=np.array([float(a[0])]),
        r_nodes=np.array([r0]),
        r_interp=_ConstantInterpolant(u_lo, r0),
        log_a_interp=_ConstantInterpolant(u_lo, math.log(float(a[0]))),
        achieved_phase_abserr=0.0,
        achieved_amplitude_relerr=0.0,
        achieved_deriv_relerr=0.0,
        achieved_deriv_alt_relerr=0.0,
        phase_atol=phase_atol,
        amplitude_rtol=amplitude_rtol,
        deriv_rtol=deriv_rtol,
        interp_degree=degree,
        refinement_passes=0,
        wraps_tracked=0,
        n_panels=0,
        panel_edges=np.array([u_lo]),
        converged=True,
        degenerate=True,
        amplitude_band=band,
        max_branch_advance=0.0,
        max_branch_snap=0.0,
    )


def build_near_region(
    nu: float,
    x_lo: float,
    x_star: float,
    phase_atol: float,
    amplitude_rtol: float,
    deriv_rtol: Optional[float] = None,
    degree: int = DEFAULT_PANEL_DEGREE,
    initial_panel_width: float = DEFAULT_INITIAL_PANEL_WIDTH,
    max_refinement_passes: int = DEFAULT_MAX_REFINEMENT_PASSES,
    max_panels: int = DEFAULT_MAX_PANELS,
    amplitude_band: Tuple[float, float] = (AMPLITUDE_BAND_LO, AMPLITUDE_BAND_HI),
    anchor_residual: Optional[float] = None,
    tail_terms: int = DEFAULT_TAIL_TERMS,
) -> NearRegionData:
    """
    Sample, branch-track and interpolate ``a_nu`` and ``r_nu`` over ``[x_lo, x_star]``.

    ``x_star`` must come from :func:`LiouvilleGreen.bessel_tail.tail_crossover`; this module never
    computes a crossover, a series or a reference of its own, and it never samples above
    ``x_star`` for any reason -- checking the crossover is the stitching layer's job, and it does
    it by comparing this interpolant against the series there.

    Acceptance is per panel, and a panel that fails any criterion is bisected. Degree alone is
    never the criterion (``DRAFT-PLAN.md`` §7.3); the four criteria are

    1. **branch safety** -- every node gap satisfies ``max|dr/d log x| * h <=
       BRANCH_SAFETY_FRACTION * pi``;
    2. **branch consistency** -- every resolved node lies within ``BRANCH_CONSISTENCY_FRACTION *
       pi`` of the value predicted by spectrally integrating ``dr/d log x`` from its neighbour;
    3. **coefficient tails** -- the Chebyshev tails of ``r`` and of ``ell = log a`` are within
       ``phase_atol`` and ``amplitude_rtol``;
    4. **interior residuals** -- the interpolants, evaluated at points strictly interior to the
       panel and never at its nodes, match freshly sampled values there within ``phase_atol``,
       ``amplitude_rtol`` and ``deriv_rtol``, the last for ``theta' = exp(-2 ell)``.

    The last of these is also what the ``achieved_*`` fields report. The alternative derivative
    route ``1 + r_u/x`` is measured at the same points and reported as
    ``achieved_deriv_alt_relerr``, but it is deliberately not a refinement criterion; see that
    field's documentation for the measured reason. Because refinement is by bisection of failing
    panels only, the tail keeps the coarse initial spacing while the turning point is refined
    repeatedly -- the two-sided adaptivity of ``DRAFT-PLAN.md`` §4.5, which a refine-only design
    on a uniform grid does not provide.

    :param deriv_rtol: budget for the relative error of ``theta'``. Defaults to twice
        ``amplitude_rtol``, which is the exact propagation of an ``ell`` error through
        ``theta' = exp(-2 ell)``.
    :param anchor_residual: overrides the value used to fix the integer cycle at the topmost node.
        Only the integer is taken from it, so its own accuracy does not enter the result; it
        exists so that a test can pin the anchor without the tail series.
    :raises NearRegionError: if the inputs are inconsistent.
    :raises AmplitudeBandError: if any sample falls outside the plausibility band. Construction
        fails loudly: no sample is dropped, interpolated across, or warned about and kept.
    """
    if not (x_lo > 0.0):
        raise NearRegionError(
            f"build_near_region: x_lo must be positive, got {x_lo!r}."
        )
    if not (x_star >= x_lo):
        raise NearRegionError(
            f"build_near_region: x_star={x_star!r} is below x_lo={x_lo!r}."
        )
    if not (phase_atol > 0.0) or not (amplitude_rtol > 0.0):
        raise NearRegionError(
            f"build_near_region: budgets must be positive, got phase_atol={phase_atol!r}, "
            f"amplitude_rtol={amplitude_rtol!r}."
        )
    if deriv_rtol is None:
        deriv_rtol = 2.0 * amplitude_rtol
    if not (deriv_rtol > 0.0):
        raise NearRegionError(
            f"build_near_region: deriv_rtol must be positive, got {deriv_rtol!r}."
        )
    if (
        not isinstance(degree, (int, np.integer))
        or degree < MIN_PANEL_DEGREE
        or degree % 2
    ):
        raise NearRegionError(
            f"build_near_region: degree must be an even integer >= {MIN_PANEL_DEGREE}, got "
            f"{degree!r}. Even, because a Lobatto grid of even degree carries a node at the panel "
            f"midpoint, which keeps the interior test points away from the nodes; and at least "
            f"{MIN_PANEL_DEGREE}, because the coefficient-tail estimate of "
            f"_coefficient_tail sums the last three coefficients, which below that degree is a "
            f"large fraction of the function itself rather than its truncation error."
        )
    degree = int(degree)
    band = (float(amplitude_band[0]), float(amplitude_band[1]))

    u_lo = math.log(x_lo)
    u_star = math.log(x_star)

    if u_star - u_lo <= DEGENERATE_SPAN_TOL:
        return _degenerate_result(
            nu,
            x_lo,
            x_star,
            phase_atol,
            amplitude_rtol,
            deriv_rtol,
            degree,
            band,
            anchor_residual,
            tail_terms,
        )

    anchor = (
        float(anchor_residual)
        if anchor_residual is not None
        else float(tail_residual(nu, x_star, n_terms=tail_terms))
    )

    span = u_star - u_lo
    n_initial = max(1, int(math.ceil(span / initial_panel_width)))
    edges = list(np.linspace(u_lo, u_star, n_initial + 1))

    cache: Dict[Tuple[float, float], _Panel] = {}
    passes = 0
    converged = False
    state = None

    while True:
        panels = []
        for i in range(len(edges) - 1):
            key = (edges[i], edges[i + 1])
            panel = cache.get(key)
            if panel is None:
                panel = _Panel(nu, edges[i], edges[i + 1], degree, band)
                cache[key] = panel
            panels.append(panel)

        state = _assemble(
            panels, degree, anchor, phase_atol, amplitude_rtol, deriv_rtol
        )

        if not np.any(state["failing"]):
            converged = True
            break
        if passes >= max_refinement_passes:
            break
        n_new = len(panels) + int(np.count_nonzero(state["failing"]))
        if n_new > max_panels:
            break

        new_edges = [edges[0]]
        for i in range(len(panels)):
            if state["failing"][i]:
                new_edges.append(0.5 * (edges[i] + edges[i + 1]))
            new_edges.append(edges[i + 1])
        edges = new_edges
        passes += 1

    return NearRegionData(
        nu=nu,
        x_lo=x_lo,
        x_star=x_star,
        log_x_nodes=state["u"],
        a_nodes=state["a"],
        r_nodes=state["r"],
        r_interp=state["r_interp"],
        log_a_interp=state["log_a_interp"],
        achieved_phase_abserr=state["achieved_phase_abserr"],
        achieved_amplitude_relerr=state["achieved_amplitude_relerr"],
        achieved_deriv_relerr=state["achieved_deriv_relerr"],
        achieved_deriv_alt_relerr=state["achieved_deriv_alt_relerr"],
        phase_atol=phase_atol,
        amplitude_rtol=amplitude_rtol,
        deriv_rtol=deriv_rtol,
        interp_degree=degree,
        refinement_passes=passes,
        wraps_tracked=state["wraps_tracked"],
        n_panels=len(edges) - 1,
        panel_edges=np.asarray(edges, dtype=float),
        converged=converged,
        degenerate=False,
        amplitude_band=band,
        max_branch_advance=state["max_branch_advance"],
        max_branch_snap=state["max_branch_snap"],
    )


def _assemble(
    panels: List[_Panel],
    degree: int,
    anchor: float,
    phase_atol: float,
    amplitude_rtol: float,
    deriv_rtol: float,
) -> dict:
    """
    Resolve the branch across all panels, fit the interpolants, measure the interior residuals,
    and mark the panels that fail a criterion.
    """
    # --- the global node list; panel edges appear once -----------------------------------------
    u = [panels[0].u_nodes]
    a = [panels[0].a_nodes]
    raw = [panels[0].raw_angle]
    f = [panels[0].f_nodes]
    increments = [panels[0].increments()]
    for panel in panels[1:]:
        u.append(panel.u_nodes[1:])
        a.append(panel.a_nodes[1:])
        raw.append(panel.raw_angle[1:])
        f.append(panel.f_nodes[1:])
        increments.append(panel.increments())

    u = np.concatenate(u)
    a = np.concatenate(a)
    raw = np.concatenate(raw)
    f = np.concatenate(f)
    increments = np.concatenate(increments)
    n = u.size

    # --- branch resolution, anchored at the top and walked downwards ---------------------------
    #
    # The anchor fixes the integer cycle only; the value is the sampled arg S_nu, which uses J and
    # Y together. Walking downwards from the top rather than upwards from the turning point keeps
    # every prediction re-anchored on an already-resolved sample, so the spectral integration
    # error does not accumulate over the ~90 cycles the region can span.
    r = np.empty(n)
    snap = np.zeros(n)
    r[n - 1] = raw[n - 1] + TWO_PI * round((anchor - raw[n - 1]) / TWO_PI)
    for i in range(n - 2, -1, -1):
        predicted = r[i + 1] - increments[i]
        r[i] = raw[i] + TWO_PI * round((predicted - raw[i]) / TWO_PI)
        snap[i] = abs(r[i] - predicted)

    gaps = np.diff(u)
    advance = np.maximum(np.abs(f[:-1]), np.abs(f[1:])) * gaps

    branch_ok = advance <= BRANCH_SAFETY_FRACTION * math.pi
    snap_ok = snap <= BRANCH_CONSISTENCY_FRACTION * math.pi

    # --- per-panel fits, tails and interior residuals ------------------------------------------
    failing = np.zeros(len(panels), dtype=bool)
    breakpoints = [panels[0].u0] + [p.u1 for p in panels]
    r_blocks: List[np.ndarray] = []
    l_blocks: List[np.ndarray] = []

    phase_err = 0.0
    amplitude_err = 0.0
    deriv_err = 0.0
    deriv_alt_err = 0.0

    offset = 0
    for k, panel in enumerate(panels):
        lo = offset
        hi = offset + degree + 1
        offset += degree  # the panel's last node is the next panel's first

        r_panel = r[lo:hi]
        l_panel = np.log(panel.a_nodes)

        r_coefficients = _coefficients_from_values(r_panel, degree)
        l_coefficients = _coefficients_from_values(l_panel, degree)
        r_blocks.append(r_coefficients)
        l_blocks.append(l_coefficients)

        if not np.all(branch_ok[lo : hi - 1]) or not np.all(snap_ok[lo:hi]):
            failing[k] = True

        if _coefficient_tail(r_coefficients) > phase_atol:
            failing[k] = True
        if _coefficient_tail(l_coefficients) > amplitude_rtol:
            failing[k] = True

        # --- interior residuals: the interpolants against fresh samples inside the panel --------
        r_series = _cheb.Chebyshev(r_coefficients, domain=[panel.u0, panel.u1])
        l_series = _cheb.Chebyshev(l_coefficients, domain=[panel.u0, panel.u1])

        r_predicted = r_series(panel.u_test)
        r_sampled = panel.raw_angle_test + TWO_PI * np.round(
            (r_predicted - panel.raw_angle_test) / TWO_PI
        )
        panel_phase_err = float(np.max(np.abs(r_predicted - r_sampled)))

        l_predicted = l_series(panel.u_test)
        l_sampled = np.log(panel.a_test)
        panel_amplitude_err = float(np.max(np.abs(l_predicted - l_sampled)))

        theta_prime_predicted = np.exp(-2.0 * l_predicted)
        theta_prime_sampled = 1.0 / (panel.a_test * panel.a_test)
        panel_deriv_err = float(
            np.max(np.abs(theta_prime_predicted / theta_prime_sampled - 1.0))
        )

        # The alternative derivative route, 1 + r_u/x, measured but deliberately *not* used to
        # drive refinement: see the note on `achieved_deriv_alt_relerr` in NearRegionData.
        r_u_predicted = r_series.deriv()(panel.u_test)
        panel_deriv_alt_err = float(
            np.max(
                np.abs((1.0 + r_u_predicted / panel.x_test) / theta_prime_sampled - 1.0)
            )
        )

        if (
            panel_phase_err > phase_atol
            or panel_amplitude_err > amplitude_rtol
            or panel_deriv_err > deriv_rtol
        ):
            failing[k] = True

        phase_err = max(phase_err, panel_phase_err)
        amplitude_err = max(amplitude_err, panel_amplitude_err)
        deriv_err = max(deriv_err, panel_deriv_err)
        deriv_alt_err = max(deriv_alt_err, panel_deriv_alt_err)

    cycles = np.floor(r / TWO_PI)
    wraps = int(np.sum(np.abs(np.diff(cycles))))

    return {
        "u": u,
        "a": a,
        "r": r,
        "r_interp": PiecewiseChebyshev(breakpoints, r_blocks),
        "log_a_interp": PiecewiseChebyshev(breakpoints, l_blocks),
        "failing": failing,
        "achieved_phase_abserr": phase_err,
        "achieved_amplitude_relerr": amplitude_err,
        "achieved_deriv_relerr": deriv_err,
        "achieved_deriv_alt_relerr": deriv_alt_err,
        "wraps_tracked": wraps,
        "max_branch_advance": float(np.max(advance)) if advance.size else 0.0,
        "max_branch_snap": float(np.max(snap)),
    }
