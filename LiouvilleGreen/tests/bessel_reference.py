"""
Independent references and error metrics for the Bessel amplitude/phase construction.

This module is deliberately *not* a test module: it defines no ``TestCase``. It is the single
place the transfer-function remedial campaign's metrics and references live, and later prompts
import it rather than redefining any of it.

Why it exists
-------------

The campaign replaces ``LiouvilleGreen.bessel_phase`` with a construction that is six to eight
orders more accurate. Measuring that requires references which do not share the error being
measured, and there are two distinct traps:

* **A reference built from ``bessel_phase`` itself conceals common error.** Several existing
  fixtures do exactly that (``ComputeTargets/tests/test_tk_source_functions.py:228-266`` defines
  its "exact" phase as ``pi - vartheta(x)`` read back out of ``bessel_phase``). Nothing in this
  module may import ``bessel_phase``, and nothing built here may be derived from it.

* **A reference built only from ``jv``/``yv`` shares the Amos library with ``hankel1e``.** Above
  ``x ~ 2.5e15`` SciPy/Amos ``jv``/``yv`` lose argument-reduction accuracy and become
  O(1)-relatively noisy, so at the top of the supported domain they are not a reference at all
  (``prompts/transfer-remedial/RECONCILIATION.md`` C1); and above ``nu ~ 86`` that boundary
  collapses to ``7.13e8``, where they return finite, non-zero values wrong by up to a factor 100.
  ``mpmath`` is the only usable reference beyond either boundary, which is why
  :func:`scipy_reference_max_x` is enforced by raising rather than warning.

Conventions
-----------

The repository's Bessel convention (which is *not* DLMF's; the two differ by exactly ``+pi/2``,
absorbed into ``c_nu``) is

    J_nu(x) = A_nu(x) sin theta_nu(x),        Y_nu(x) = -A_nu(x) cos theta_nu(x),

with ``theta_nu`` increasing in ``x``, ``A_nu = sqrt(2/(pi x)) a_nu`` and

    theta_nu(x) = x + c_nu + r_nu(x),         c_nu = pi/4 - pi nu/2.

Consequently ``H^(1)_nu = J_nu + i Y_nu = A_nu exp(i(theta_nu - pi/2))``, and the Wronskian
``A^2 theta' = 2/(pi x)`` gives the exact identities

    theta'(x) = (2/pi) / (x (J^2 + Y^2)),     a_nu = (1 + r_nu')^(-1/2),
    a_nu(x)   = sqrt(pi x / 2) hypot(J, Y).

``theta'`` is an *identity*, not an approximation, which is why it is the derivative oracle used
throughout; see :func:`reference_theta_deriv`.

Layout
------

* error metrics: :func:`phase_pair_error`, :func:`amplitude_error`, :func:`derivative_error`;
* three reference tiers with an explicit selector: :func:`exact_half_integer`,
  :func:`scipy_reference`, :func:`mpmath_reference`, selected through :func:`reference_JY` /
  :func:`reference_bundle`;
* a permanently cached table of ``mpmath`` corner references,
  ``LiouvilleGreen/tests/bessel_reference_data.json``, read by :func:`cached_corners` and written
  by :func:`regenerate_reference_table`.

``mpmath`` is imported **lazily**, inside the functions that need it. Reading the cached table
must never pull it in: ``test_bessel_reference.py`` asserts that importing this module leaves
``mpmath`` out of ``sys.modules``.
"""

import json
import math
import os
import platform
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import jv, yv

# ----------------------------------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------------------------------

#: Largest x at which SciPy/Amos ``jv``/``yv`` may be used as a reference.
#:
#: ``prompts/transfer-remedial/RECONCILIATION.md`` C1 measures ``(2/pi)/(x (J^2 + Y^2))``, which
#: must equal ``1 + O(nu^2/x^2)``, at five adjacent doubles: it is 1.000000 throughout
#: ``1e12 <= x <= 2e15`` and scatters over ``[0.73, 1.68]`` at ``3e15`` and above. The transition
#: is somewhere near ``2.5e15``; this constant sits below it with margin. ``jv``/``yv`` above it
#: are not wrong-and-flagged, they are silently noisy, so :func:`scipy_reference` raises rather
#: than warning. ``LiouvilleGreen/tests/test_scipy_bessel_domain.py`` (campaign prompt 02) pins
#: the boundary itself as a test.
SCIPY_REFERENCE_MAX_X = 2.0e15

#: Order above which the SciPy/Amos reference boundary collapses from 2e15 to 7.13e8, and the
#: collapsed boundary itself.
#:
#: ``RECONCILIATION.md`` 1 records the 7.13e8 boundary for ``hankel1e`` at ``nu`` in
#: ``{100.5, 1000.5}`` and 2.247e15 for ``nu <= 20.5``. **The same order-dependent boundary applies
#: to ``jv``/``yv``**, which that measurement did not cover and which matters here because ``jv``
#: and ``yv`` are what tier 2 is built from. Measured for this prompt (see
#: ``docs/transfer-remedial/baseline-2026-09.md``), with ``a = sqrt(pi x/2) hypot(J, Y)``, which is
#: ``1 + O(nu^2/x^2)`` and so must be 1.000000000000 to twelve places over this whole range:
#:
#: * ``max |a - 1|`` over ``1e8 <= x <= 2e15`` is 4.4e-16 (nu=2.5), 1.0e-14 (20.5), 6.4e-14 (50.5),
#:   1.6e-13 (80.5), 1.8e-13 (**85.5**) -- and then **1.0** at 88.5, 0.998 at 89.5, 1.0 at 90.5,
#:   100.5 and 1000.5;
#: * at ``nu = 100.5`` the failure in ``x`` is abrupt: ``a`` = 0.9999999872 at ``x = 7.108e8`` and
#:   0.0848 at ``7.188e8``, then wanders over ``[0.01, 0.99]`` -- finite, non-zero, and wrong by up
#:   to a factor 100.
#:
#: So the order threshold lies between 85.5 and 88.5 and is set conservatively at 85.5; the ``x``
#: boundary reproduces ``RECONCILIATION.md`` 1's 7.13e8 to three figures. Pinning both as tests is
#: prompt 02's; this module only has to refuse to hand back a value it knows to be wrong.
SCIPY_REFERENCE_HIGH_ORDER_NU = 85.5
SCIPY_REFERENCE_HIGH_ORDER_MAX_X = 7.13e8

TIER_EXACT = "exact"
TIER_SCIPY = "scipy"
TIER_MPMATH = "mpmath"

#: The reference tiers, in increasing cost and decreasing shared-library exposure.
REFERENCE_TIERS = (TIER_EXACT, TIER_SCIPY, TIER_MPMATH)

#: Orders for which :func:`exact_half_integer` has a closed form.
EXACT_HALF_INTEGER_ORDERS = (-0.5, 0.5, 1.5, 2.5)

#: Default working precision for the ``mpmath`` tier, in decimal digits.
DEFAULT_MPMATH_DPS = 70

#: Term and precision caps handed to ``mpmath.besselj``/``bessely``.
#:
#: Raised above mpmath 1.3.0's defaults because the ``0F1`` series it sums suffers severe
#: cancellation when the order is large and the argument is within a couple of decades of it. At
#: ``nu = 1000.5, x = 8436.9`` the default caps produce ``hypsum() failed to converge to the
#: requested 288 bits of accuracy using a working precision of 7323 bits``; the internal precision
#: escalation, not the term count, is what binds, and ``maxprec = 1e6`` bits clears it. These caps
#: cost nothing where the defaults would already have sufficed -- every cached corner evaluates in
#: under 60 ms -- but note that the *worst* interior points cost ~1.5 s each, which is why the
#: residual branch walk of :func:`regenerate_reference_table` does not use this tier.
MPMATH_MAXTERMS = 10**7
MPMATH_MAXPREC = 10**6

#: Number of significant decimal digits written to the cached table. Chosen well above double
#: precision so that the file is not itself limited to it.
CACHE_DIGITS = 40

#: Schema version of ``bessel_reference_data.json``. Bump on any incompatible layout change.
REFERENCE_DATA_SCHEMA_VERSION = 1

REFERENCE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bessel_reference_data.json"
)

#: Orders covered by the cached corner table.
CACHED_ORDERS = (0.5, 1.5, 1.75, 2.5, 20.5, 100.5, 1000.5)

_TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------------------------
# Small shape helpers
# ----------------------------------------------------------------------------------------------


def _as_1d(value) -> Tuple[np.ndarray, bool]:
    """Return (1-d float array, was_scalar)."""
    arr = np.asarray(value, dtype=float)
    scalar = arr.ndim == 0
    return np.atleast_1d(arr).ravel(), scalar


def _restore(arr: np.ndarray, scalar: bool):
    return float(arr[0]) if scalar else arr


# ----------------------------------------------------------------------------------------------
# Geometry of the construction
# ----------------------------------------------------------------------------------------------


def c_nu(nu: float) -> float:
    """
    The exact zero-point of the phase, ``c_nu = pi/4 - pi nu/2``.

    This is the repository's convention and not DLMF's: DLMF 10.18.18 writes the phase with a
    ``-(nu/2 + 1/4) pi`` offset, and the ``+pi/2`` difference between the two conventions is what
    turns that into ``c_nu``. Do not "correct" it.
    """
    return 0.25 * math.pi - 0.5 * math.pi * nu


def construction_min_x(nu: float) -> float:
    """
    The lower edge of the domain on which a phase function is provided, mirroring
    ``LiouvilleGreen.bessel_phase``: ``sqrt(nu^2 - 1/4)`` for ``nu > 1/2`` (the turning point of
    ``omega_eff``, below which the solution is exponential rather than oscillatory) and ``1e-5``
    for ``nu <= 1/2``, where ``sqrt(nu^2 - 1/4)`` degenerates to zero.

    Reproduced here rather than imported, because this module must not import ``bessel_phase``.
    """
    if nu > 0.5:
        return math.sqrt(nu * nu - 0.25)
    return 1e-5


def tail_residual_series(nu: float, x):
    """
    The two-term large-argument series for the phase residual ``r_nu``, DLMF 10.18.18 carried into
    this repository's convention:

        r_nu(x) ~ (mu - 1)/(8x) + (mu - 1)(mu - 25)/(384 x^3),      mu = 4 nu^2.

    Used here only to *anchor the branch* of the residual at the largest ``x`` of each order (see
    :func:`regenerate_reference_table`). The campaign's closed-form tail module, its remainder
    test and its crossover ``x_star`` are prompt 03's; nothing here selects a crossover.

    ``nu = 1/2`` gives ``mu - 1 = 0``, so ``r == 0`` identically at every ``x`` and not merely
    asymptotically.
    """
    arr, scalar = _as_1d(x)
    mu = 4.0 * nu * nu
    value = (mu - 1.0) / (8.0 * arr) + (mu - 1.0) * (mu - 25.0) / (384.0 * arr**3)
    return _restore(value, scalar)


# ----------------------------------------------------------------------------------------------
# Error metrics -- DRAFT-PLAN.md 6.1, README 6
# ----------------------------------------------------------------------------------------------


def phase_pair_error(sin_theta, minus_cos_theta, J, Y, amplitude) -> Tuple[float, int]:
    """
    The campaign's phase error ``E_theta``,

        E_theta = max_x max(|sin theta_ours - J/A|, |-cos theta_ours - Y/A|),

    with ``A = hypot(J, Y)`` taken from the *reference* functions.

    This is a phase-pair error and not an unwrapped phase difference, for two reasons. It measures
    the local effect of a phase error on the reconstructed Bessel values, normalized by their
    envelope, so a fixed ``E_theta`` means the same thing at every ``x``; and it never divides by a
    function that passes through zero, which a pointwise relative error in ``J`` or ``Y`` does at
    every Bessel zero. For small errors ``E_theta`` is the phase error in radians, to leading
    order.

    Both branches are compared and the larger is kept, because a phase error shows up in the sine
    or the cosine depending on where in the cycle ``x`` falls; taking only one would miss the
    maximum wherever that branch happens to be flat.

    :param sin_theta: our ``sin theta(x)``, formed by angle addition and *not* by ``sin(x + d)``.
    :param minus_cos_theta: our ``-cos theta(x)``.
    :param J: reference ``J_nu(x)``.
    :param Y: reference ``Y_nu(x)``.
    :param amplitude: reference ``hypot(J, Y)``.
    :return: ``(max_error, argmax_index)``. The location of the maximum is a **required** output,
        not a convenience: ``DRAFT-PLAN.md`` 4.7 finds every derivative maximum in the interval
        adjacent to the turning point, and prompts 04 and 05 have to be able to confirm that on
        their own output.
    """
    s, scalar = _as_1d(sin_theta)
    mc, _ = _as_1d(minus_cos_theta)
    Jv, _ = _as_1d(J)
    Yv, _ = _as_1d(Y)
    A, _ = _as_1d(amplitude)

    err = np.maximum(np.abs(s - Jv / A), np.abs(mc - Yv / A))
    index = int(np.argmax(err))
    return float(err[index]), index


def amplitude_error(A_ours, amplitude) -> Tuple[float, int]:
    """
    The campaign's amplitude error ``E_A = max_x |A_ours/A - 1|``, with ``A = hypot(J, Y)`` from
    the reference functions.

    Relative rather than absolute because the envelope falls like ``x^(-1/2)`` over up to twenty
    e-folds, so an absolute amplitude error would be meaningless at one end of the domain or the
    other. ``A`` never vanishes, so unlike ``J`` and ``Y`` it is safe to divide by.

    :return: ``(max_error, argmax_index)``.
    """
    ours, scalar = _as_1d(A_ours)
    ref, _ = _as_1d(amplitude)

    err = np.abs(ours / ref - 1.0)
    index = int(np.argmax(err))
    return float(err[index]), index


def derivative_error(theta_prime_ours, theta_prime_ref) -> Tuple[float, int]:
    """
    Relative error of the phase derivative against the exact oracle
    ``theta' = (2/pi)/(x (J^2 + Y^2))``.

    Relative, because ``theta'`` runs from ``O(nu)`` at the turning point down to ``1`` in the
    tail and is bounded away from zero throughout the supported domain, so the relative measure is
    both well defined and the one the existing contract at ``test_bessel_phase.py:139`` uses.

    :return: ``(max_error, argmax_index)``.
    """
    ours, scalar = _as_1d(theta_prime_ours)
    ref, _ = _as_1d(theta_prime_ref)

    err = np.abs(ours / ref - 1.0)
    index = int(np.argmax(err))
    return float(err[index]), index


# ----------------------------------------------------------------------------------------------
# Derived reference quantities, tier-independent
# ----------------------------------------------------------------------------------------------


def reference_amplitude(J, Y):
    """``A = hypot(J, Y)``, the Liouville-Green envelope of the reference functions."""
    Jv, scalar = _as_1d(J)
    Yv, _ = _as_1d(Y)
    return _restore(np.hypot(Jv, Yv), scalar)


def reference_theta_deriv(x, J, Y):
    """
    The exact derivative oracle ``theta'(x) = (2/pi)/(x (J^2 + Y^2))``.

    This is the Wronskian identity ``A^2 theta' = 2/(pi x)`` rearranged, so it is exact for any
    ``x`` in the oscillatory region and is *not* an asymptotic approximation. Its accuracy is
    exactly the accuracy of the ``J`` and ``Y`` it is handed, which is why the tier matters: fed
    ``jv``/``yv`` above ``SCIPY_REFERENCE_MAX_X`` it would inherit their noise, and
    :func:`scipy_reference` refuses to go there.
    """
    xv, scalar = _as_1d(x)
    Jv, _ = _as_1d(J)
    Yv, _ = _as_1d(Y)
    return _restore((2.0 / math.pi) / (xv * (Jv * Jv + Yv * Yv)), scalar)


def reference_theta_principal(J, Y):
    """
    The reference phase reduced to its principal branch, ``atan2(J, -Y)``, in ``(-pi, pi]``.

    Branch convention, stated explicitly because getting it wrong poisons every comparison
    downstream: with ``J = A sin theta`` and ``Y = -A cos theta``,
    ``atan2(J, -Y) = atan2(A sin theta, A cos theta)``, which is the continuous ``theta`` folded
    into ``(-pi, pi]``. It is therefore ``theta - 2 pi n`` for the integer ``n`` that lands it in
    that interval, **not** an unwrapped phase: the unwrapped phase at production arguments is of
    order ``1e15`` rad and cannot be recovered from ``J`` and ``Y`` alone at a single point.

    This is the anchor form ``DRAFT-PLAN.md`` 4.3 recommends in place of ``bessel_phase``'s
    ``root_scalar`` offset match: at any ``x`` it fixes the phase directly, with no bracketing, no
    convergence tolerance and no spurious offset.

    The cached corner table resolves the branch as well, storing both this principal value and the
    integer ``theta_div_2pi`` with ``theta_continuous = theta_principal + 2 pi theta_div_2pi``; see
    :func:`cached_corners`.
    """
    Jv, scalar = _as_1d(J)
    Yv, _ = _as_1d(Y)
    return _restore(np.arctan2(Jv, -Yv), scalar)


def reference_a(x, J, Y):
    """The normalized amplitude ``a_nu = sqrt(pi x / 2) hypot(J, Y)``, which tends to 1 at large x."""
    xv, scalar = _as_1d(x)
    Jv, _ = _as_1d(J)
    Yv, _ = _as_1d(Y)
    return _restore(np.sqrt(0.5 * math.pi * xv) * np.hypot(Jv, Yv), scalar)


# ----------------------------------------------------------------------------------------------
# Tier 1 -- exact half-integer closed forms
# ----------------------------------------------------------------------------------------------


def exact_half_integer(nu: float, x):
    """
    Closed-form ``(J_nu, Y_nu)`` for ``nu`` in :data:`EXACT_HALF_INTEGER_ORDERS`.

    Tier 1 of the reference hierarchy, and the only tier that shares **no** library at all with
    the object under test: it is built from ``numpy``'s ``sin``/``cos`` on the supplied ``x``,
    which ``RECONCILIATION.md`` 1 confirms correctly rounded out to ``x = 1e16`` on this platform.
    Prefer it wherever the order allows.

        J_{1/2}  =  sqrt(2/(pi x)) sin x        Y_{1/2}  = -sqrt(2/(pi x)) cos x
        J_{-1/2} =  sqrt(2/(pi x)) cos x        Y_{-1/2} =  sqrt(2/(pi x)) sin x

    and the orders above follow from the recurrence ``C_{nu+1} = (2 nu/x) C_nu - C_{nu-1}``, which
    holds for ``J`` and ``Y`` alike.

    Caveat, stated because it bounds what this tier can certify: upward recurrence in order is the
    unstable direction for ``J`` when ``x < nu``. On this module's domain ``x >= sqrt(nu^2 - 1/4)``
    and ``nu <= 5/2``, so at worst ``2 nu/x ~ 1.2`` for a single step and less than one decimal
    digit is lost; the tier-agreement test of ``test_bessel_reference.py`` measures the actual
    figure against the cached ``mpmath`` corners.
    """
    if not any(abs(nu - order) < 1e-13 for order in EXACT_HALF_INTEGER_ORDERS):
        raise ValueError(
            f"exact_half_integer: no closed form for nu={nu}; supported orders are "
            f"{EXACT_HALF_INTEGER_ORDERS}. Use scipy_reference() or mpmath_reference()."
        )

    xv, scalar = _as_1d(x)
    if np.any(xv <= 0.0):
        raise ValueError("exact_half_integer: x must be positive")

    envelope = np.sqrt(2.0 / (math.pi * xv))
    sin_x = np.sin(xv)
    cos_x = np.cos(xv)

    # nu = -1/2 and nu = +1/2
    J_lo, Y_lo = envelope * cos_x, envelope * sin_x
    J_hi, Y_hi = envelope * sin_x, -envelope * cos_x

    if nu < 0.0:
        return _restore(J_lo, scalar), _restore(Y_lo, scalar)

    order = 0.5
    while order < nu - 1e-13:
        factor = 2.0 * order / xv
        J_lo, J_hi = J_hi, factor * J_hi - J_lo
        Y_lo, Y_hi = Y_hi, factor * Y_hi - Y_lo
        order += 1.0

    return _restore(J_hi, scalar), _restore(Y_hi, scalar)


# ----------------------------------------------------------------------------------------------
# Tier 2 -- SciPy/Amos
# ----------------------------------------------------------------------------------------------


def scipy_reference_max_x(nu: float) -> float:
    """
    The largest ``x`` at which ``jv``/``yv`` may be used as a reference **at this order**.

    Two boundaries, both properties of the bundled Amos library rather than guarantees:

    * :data:`SCIPY_REFERENCE_MAX_X` (2e15), the universal argument-reduction boundary of
      ``RECONCILIATION.md`` C1;
    * :data:`SCIPY_REFERENCE_HIGH_ORDER_MAX_X` (7.13e8) for ``nu`` above
      :data:`SCIPY_REFERENCE_HIGH_ORDER_NU`, where the boundary collapses by six decades.

    The second is measured in this campaign's ``docs/transfer-remedial/baseline-2026-09.md``; see
    :data:`SCIPY_REFERENCE_HIGH_ORDER_MAX_X` for the numbers and for why it is not merely a
    restatement of ``RECONCILIATION.md`` 1.
    """
    if nu > SCIPY_REFERENCE_HIGH_ORDER_NU:
        return SCIPY_REFERENCE_HIGH_ORDER_MAX_X
    return SCIPY_REFERENCE_MAX_X


def scipy_reference(nu: float, x):
    """
    ``(J_nu, Y_nu)`` from SciPy's ``jv``/``yv``. Tier 2: available at every order, but sharing the
    Amos library with the ``hankel1e`` the campaign's near region samples, so a common-mode error
    is possible in principle and certain above the boundaries below.

    **Raises** ``ValueError`` for any ``x`` above :func:`scipy_reference_max_x` for that order.
    For ``nu <= 85.5`` that is ``SCIPY_REFERENCE_MAX_X`` = 2e15: above roughly ``2.5e15`` Amos
    loses argument reduction and ``jv``/``yv`` become O(1)-relatively noisy, and
    ``RECONCILIATION.md`` C1 measures ``(2/pi)/(x (J^2 + Y^2))`` -- identically
    ``1 + O(nu^2/x^2)`` -- taking the values ``0.989, 0.948, 1.029, 0.979, 1.033`` at five
    *adjacent* doubles near ``x = 3e15``. For higher orders the boundary is
    ``SCIPY_REFERENCE_HIGH_ORDER_MAX_X`` = 7.13e8 instead.

    A reference that answered silently in either regime would reproduce exactly the silent-failure
    mode this campaign exists to remove, so this refuses. Use the cached ``mpmath`` corners
    (:func:`cached_corners`) or :func:`mpmath_reference` above the boundary; the boundaries
    themselves are pinned by ``LiouvilleGreen/tests/test_scipy_bessel_domain.py`` (prompt 02).
    """
    xv, scalar = _as_1d(x)
    limit = scipy_reference_max_x(nu)
    if np.any(xv > limit):
        biggest = float(np.max(xv))
        raise ValueError(
            f"scipy_reference: x={biggest:.6g} exceeds the SciPy/Amos reference boundary "
            f"{limit:.6g} for nu={nu} (SCIPY_REFERENCE_MAX_X={SCIPY_REFERENCE_MAX_X:.6g}, "
            f"SCIPY_REFERENCE_HIGH_ORDER_MAX_X={SCIPY_REFERENCE_HIGH_ORDER_MAX_X:.6g} for "
            f"nu>{SCIPY_REFERENCE_HIGH_ORDER_NU}). SciPy/Amos jv/yv lose accuracy above these "
            f"boundaries -- silently, without a non-finite value -- so they are not a reference "
            f"there; see prompts/transfer-remedial/RECONCILIATION.md C1 and 1, and "
            f"LiouvilleGreen/tests/test_scipy_bessel_domain.py. Use the cached mpmath corners "
            f"(bessel_reference.cached_corners) or mpmath_reference() instead."
        )

    return _restore(jv(nu, xv), scalar), _restore(yv(nu, xv), scalar)


# ----------------------------------------------------------------------------------------------
# Tier 3 -- mpmath
# ----------------------------------------------------------------------------------------------


def mpmath_reference(nu: float, x, dps: int = DEFAULT_MPMATH_DPS):
    """
    ``(J_nu, Y_nu)`` from ``mpmath.besselj``/``bessely`` at ``dps`` decimal digits, returned as
    Python floats. Tier 3: no shared library with the object under test at all, and the only tier
    that is usable above :data:`SCIPY_REFERENCE_MAX_X`.

    The argument is converted with ``mpmath.mpf(float(x))``. This matters twice over:

    * ``mpf(repr(x))`` is wrong for a NumPy scalar, which stringifies as ``np.float64(...)`` and
      which ``mpf`` rejects;
    * more importantly, the reference must be evaluated **at exactly the supplied double**.
      ``DRAFT-PLAN.md`` 7.5 defines accuracy at the supplied ``x``, so a reference taken at a
      re-derived argument (``exp(log(x))``, say) would measure our error plus an input-coordinate
      perturbation of order ``eps x`` in the phase -- which at ``x = 1e15`` is 0.2 rad and would
      swamp everything.

    ``mpmath`` is imported here rather than at module scope so that reading the cached table never
    pulls it in.
    """
    import mpmath

    with mpmath.workdps(dps):
        arr, scalar = _as_1d(x)
        J = np.empty_like(arr)
        Y = np.empty_like(arr)
        for index, value in enumerate(arr):
            argument = mpmath.mpf(float(value))
            J[index] = float(
                mpmath.besselj(
                    nu, argument, maxterms=MPMATH_MAXTERMS, maxprec=MPMATH_MAXPREC
                )
            )
            Y[index] = float(
                mpmath.bessely(
                    nu, argument, maxterms=MPMATH_MAXTERMS, maxprec=MPMATH_MAXPREC
                )
            )

        return _restore(J, scalar), _restore(Y, scalar)


def mpmath_reference_extended(
    nu: float, x: float, dps: int = DEFAULT_MPMATH_DPS
) -> dict:
    """
    The full ``mpmath`` bundle at a single ``x``, kept in extended precision rather than rounded
    to double.

    Returns a dict of ``mpmath.mpf`` values with keys ``J``, ``Y``, ``amplitude``,
    ``theta_principal``, ``a`` and ``theta_deriv``. Used by :func:`regenerate_reference_table`,
    which needs more than double precision to resolve ``theta`` modulo ``2 pi`` at ``x ~ 1e16``
    (17 decimal digits of ``x`` must be subtracted away before 30 significant digits of the
    residual remain).

    The residual ``r_nu`` is deliberately **not** in this bundle: it needs a branch, the branch
    needs more than one point, and resolving it is :func:`regenerate_reference_table`'s job.
    """
    import mpmath

    with mpmath.workdps(dps):
        argument = mpmath.mpf(float(x))
        J = mpmath.besselj(
            nu, argument, maxterms=MPMATH_MAXTERMS, maxprec=MPMATH_MAXPREC
        )
        Y = mpmath.bessely(
            nu, argument, maxterms=MPMATH_MAXTERMS, maxprec=MPMATH_MAXPREC
        )
        m = J * J + Y * Y
        amplitude = mpmath.sqrt(m)

        return {
            "J": J,
            "Y": Y,
            "amplitude": amplitude,
            "theta_principal": mpmath.atan2(J, -Y),
            "a": mpmath.sqrt(mpmath.pi * argument / 2) * amplitude,
            "theta_deriv": (2 / mpmath.pi) / (argument * m),
        }


# ----------------------------------------------------------------------------------------------
# The explicit tier selector
# ----------------------------------------------------------------------------------------------


def reference_JY(nu: float, x, tier: str, dps: int = DEFAULT_MPMATH_DPS):
    """
    ``(J_nu, Y_nu)`` from a **named** tier.

    ``tier`` is required and positional-friendly on purpose. A caller that does not say which
    reference it is using will get one by accident, and the accident that matters is silently
    falling back to ``jv``/``yv`` in the regime where they are noise (``RECONCILIATION.md`` C1) or
    to a ``bessel_phase``-derived quantity that shares the error under test. There is no default
    and no automatic promotion between tiers.

    :param tier: one of :data:`REFERENCE_TIERS` -- ``"exact"``, ``"scipy"`` or ``"mpmath"``.
    """
    if tier == TIER_EXACT:
        return exact_half_integer(nu, x)
    if tier == TIER_SCIPY:
        return scipy_reference(nu, x)
    if tier == TIER_MPMATH:
        return mpmath_reference(nu, x, dps=dps)
    raise ValueError(
        f"reference_JY: unknown reference tier {tier!r}; expected one of {REFERENCE_TIERS}"
    )


class BesselReference(NamedTuple):
    """Every reference quantity the campaign's metrics need, at one tier, over one set of x."""

    nu: float
    tier: str
    x: Union[float, np.ndarray]
    J: Union[float, np.ndarray]
    Y: Union[float, np.ndarray]
    amplitude: Union[float, np.ndarray]
    a: Union[float, np.ndarray]
    theta_principal: Union[float, np.ndarray]
    theta_deriv: Union[float, np.ndarray]


def reference_bundle(
    nu: float, x, tier: str, dps: int = DEFAULT_MPMATH_DPS
) -> BesselReference:
    """
    :func:`reference_JY` plus every derived quantity: the envelope ``A``, the normalized amplitude
    ``a``, the principal phase ``atan2(J, -Y)`` and the exact derivative oracle. Available at all
    three tiers, so a caller can hold the tier fixed while scoring phase, amplitude and derivative
    together.
    """
    J, Y = reference_JY(nu, x, tier, dps=dps)
    return BesselReference(
        nu=nu,
        tier=tier,
        x=x,
        J=J,
        Y=Y,
        amplitude=reference_amplitude(J, Y),
        a=reference_a(x, J, Y),
        theta_principal=reference_theta_principal(J, Y),
        theta_deriv=reference_theta_deriv(x, J, Y),
    )


def best_available_tier(nu: float, x_max: float) -> str:
    """
    The cheapest tier that is *valid* for ``(nu, x_max)``: ``"exact"`` at the half-integer orders,
    otherwise ``"scipy"`` below :data:`SCIPY_REFERENCE_MAX_X`, otherwise ``"mpmath"``.

    A convenience for diagnostics that sweep many orders. It is not a default: callers still
    record which tier they were handed, and no metric function chooses a tier for itself.
    """
    if any(abs(nu - order) < 1e-13 for order in EXACT_HALF_INTEGER_ORDERS):
        return TIER_EXACT
    if x_max <= scipy_reference_max_x(nu):
        return TIER_SCIPY
    return TIER_MPMATH


# ----------------------------------------------------------------------------------------------
# The cached corner table
# ----------------------------------------------------------------------------------------------


class CachedCorner(NamedTuple):
    """
    One row of ``bessel_reference_data.json``, with the stored decimal strings decoded to double.

    ``theta_principal`` is in ``(-pi, pi]``; the continuous phase is
    ``theta_principal + 2 pi theta_div_2pi``, which also equals ``x + c_nu + r`` by construction.
    ``strings`` retains the full 40-digit decimal forms, for a consumer that wants more than
    double precision.
    """

    nu: float
    x: float
    J: float
    Y: float
    amplitude: float
    a: float
    r: float
    theta_principal: float
    theta_div_2pi: int
    theta_deriv: float
    strings: dict

    @property
    def theta_continuous(self) -> float:
        """``theta_principal + 2 pi theta_div_2pi``. Beware: at large x this is eps*x-limited."""
        return self.theta_principal + _TWO_PI * self.theta_div_2pi


_TABLE_CACHE = {}


def load_reference_table(path: str = REFERENCE_DATA_PATH) -> dict:
    """
    Read ``bessel_reference_data.json`` and return the decoded JSON document, memoized by path.

    Does not import ``mpmath``: the table exists precisely so that the per-commit test run does
    not pay for 70-digit Bessel evaluations. Regenerating it is :func:`regenerate_reference_table`,
    which must be invoked explicitly.
    """
    if path not in _TABLE_CACHE:
        with open(path, "r") as handle:
            document = json.load(handle)

        version = document.get("schema_version")
        if version != REFERENCE_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"load_reference_table: {path} has schema_version={version}, expected "
                f"{REFERENCE_DATA_SCHEMA_VERSION}. Regenerate with "
                f"bessel_reference.regenerate_reference_table()."
            )
        _TABLE_CACHE[path] = document

    return _TABLE_CACHE[path]


def cached_corners(
    nu: Optional[float] = None, path: str = REFERENCE_DATA_PATH
) -> Sequence[CachedCorner]:
    """
    The cached corner references as :class:`CachedCorner` rows, ascending in ``(nu, x)``.

    :param nu: if given, only rows for that order (matched to 1e-13).
    """
    document = load_reference_table(path)

    rows = []
    for entry in document["entries"]:
        entry_nu = float(entry["nu"])
        if nu is not None and abs(entry_nu - nu) > 1e-13:
            continue

        x_from_decimal = float(entry["x"])
        x_from_hex = float.fromhex(entry["x_hex"])
        if x_from_decimal != x_from_hex:
            raise ValueError(
                f"cached_corners: x does not round-trip for nu={entry_nu}: "
                f"{entry['x']} != {entry['x_hex']}"
            )

        rows.append(
            CachedCorner(
                nu=entry_nu,
                x=x_from_hex,
                J=float(entry["J"]),
                Y=float(entry["Y"]),
                amplitude=float(entry["amplitude"]),
                a=float(entry["a"]),
                r=float(entry["r"]),
                theta_principal=float(entry["theta_principal"]),
                theta_div_2pi=int(entry["theta_div_2pi"]),
                theta_deriv=float(entry["theta_deriv"]),
                strings=dict(entry),
            )
        )

    return rows


def cached_orders(path: str = REFERENCE_DATA_PATH) -> Sequence[float]:
    """The orders present in the cached table, ascending."""
    document = load_reference_table(path)
    return sorted({float(entry["nu"]) for entry in document["entries"]})


def cached_corner(nu: float, x: float, path: str = REFERENCE_DATA_PATH) -> CachedCorner:
    """The single cached row at ``(nu, x)``, matched on the exact stored double."""
    for row in cached_corners(nu, path=path):
        if row.x == x:
            return row
    raise KeyError(
        f"cached_corner: no cached reference at nu={nu}, x={x!r}. Available x for that order: "
        f"{[row.x for row in cached_corners(nu, path=path)]}"
    )


# ----------------------------------------------------------------------------------------------
# Generation of the cached corner table
# ----------------------------------------------------------------------------------------------

#: Extra ``x`` value beyond ``DRAFT-PLAN.md`` 9 Stage 1's list, so that the three tiers can be
#: compared at a common point without ``mpmath`` at test time (10 is not ``10 nu`` or ``100 nu``
#: for any covered order).
TIER_AGREEMENT_X = 10.0

#: The large arguments at which only ``mpmath`` is a reference (``RECONCILIATION.md`` C1.3).
LARGE_CORNER_X = (2.5e15, 1.0e16)

#: Step-size control for the residual branch walk of :func:`regenerate_reference_table`.
_WALK_TARGET_DR = 0.3  # rad of residual advance aimed at per step
_WALK_ACCEPT_DR = (
    1.2  # rad; predictor-vs-resolved disagreement above this halves the step
)
_WALK_MAX_DLOGX = 0.25  # e-folds; cap so the walk still samples where r is flat
_WALK_MIN_DLOGX = 1e-9  # e-folds; below this the walk gives up loudly


def corner_x_values(nu: float) -> Sequence[float]:
    """
    The ``x`` corners stored for one order: ``DRAFT-PLAN.md`` 9 Stage 1's set

        {x_0, 1.5 x_0, 10 nu, 100 nu, 1e3, 1e7, 1e12, 1e15}

    restricted to ``x >= x_0 = construction_min_x(nu)``, plus ``2.5e15`` and ``1e16`` where only
    ``mpmath`` is a reference, plus :data:`TIER_AGREEMENT_X`.
    """
    x0 = construction_min_x(nu)
    candidates = [
        x0,
        1.5 * x0,
        10.0 * nu,
        100.0 * nu,
        TIER_AGREEMENT_X,
        1.0e3,
        1.0e7,
        1.0e12,
        1.0e15,
    ] + list(LARGE_CORNER_X)

    kept = sorted({float(value) for value in candidates if value >= x0})
    return kept


def _resolve_residual(nu, x, theta_principal, r_predicted, mpmath):
    """
    Lift ``theta_principal`` to the branch nearest the prediction and return the residual there.

    ``theta = x + c_nu + r`` with ``theta = theta_principal + 2 pi n``, so
    ``n = round((x + c_nu + r_predicted - theta_principal) / 2 pi)`` and
    ``r = theta_principal + 2 pi n - x - c_nu``. Exact in extended precision; the only judgement is
    ``n``, and the caller checks that ``|r - r_predicted|`` is far below ``pi`` before believing it.
    """
    offset = mpmath.mpf(0.25) * mpmath.pi - mpmath.mpf(0.5) * mpmath.pi * mpmath.mpf(nu)
    n = mpmath.nint((x + offset + r_predicted - theta_principal) / (2 * mpmath.pi))
    return theta_principal + 2 * mpmath.pi * n - x - offset, int(n)


def _walk_probe(nu, x, mpmath, dps):
    """
    ``(theta_principal, a)`` as ``mpf``, from the cheapest tier that is *valid* at ``(nu, x)``, for
    use by the residual branch walk only.

    The walk is a cycle counter, not a reference: it needs ``theta_principal`` to a small fraction
    of a radian, so double precision is ample and SciPy is three orders of magnitude cheaper than
    ``mpmath`` in the regime that dominates the step count (near the turning point at high order,
    where ``mpmath.besselj``'s ``0F1`` series costs up to 1.5 s per point). Only the *arithmetic*
    is done in ``mpmath``, and that part is not optional: at ``x = 1e16`` the cycle count ``n`` is
    ~1.6e15, so ``2 pi n`` carries an absolute double-precision rounding error of order 2 rad and a
    walk done in double precision could not represent its own residual.

    Above :func:`scipy_reference_max_x` the walk falls back to ``mpmath``, which is cheap there:
    that regime is far from the turning point *and* is where ``r`` is flattest, so it costs only
    the :data:`_WALK_MAX_DLOGX` cap's worth of steps.
    """
    x_float = float(x)
    if x_float <= scipy_reference_max_x(nu):
        J, Y = scipy_reference(nu, x_float)
        return (
            mpmath.mpf(float(reference_theta_principal(J, Y))),
            mpmath.mpf(float(reference_a(x_float, J, Y))),
        )

    bundle = mpmath_reference_extended(nu, x_float, dps=dps)
    return bundle["theta_principal"], bundle["a"]


def regenerate_reference_table(
    path: str = REFERENCE_DATA_PATH,
    orders: Sequence[float] = CACHED_ORDERS,
    dps: int = DEFAULT_MPMATH_DPS,
    verbose: bool = True,
) -> dict:
    """
    Rebuild ``bessel_reference_data.json`` from ``mpmath`` at ``dps`` digits, and write it.

    **Explicitly invoked, never part of a test run.** Cost on the planning machine is of order a
    minute, dominated by ``nu = 1000.5``, because resolving the residual's branch requires walking
    the whole domain rather than evaluating the corners alone.

        PYTHONPATH=. ./venv/bin/python -c \\
            "from LiouvilleGreen.tests.bessel_reference import regenerate_reference_table as g; g()"

    How the residual branch is fixed
    --------------------------------

    ``r_nu`` is **not** ``theta - x - c_nu`` folded into ``(-pi, pi]``. That fold is wrong at high
    order: ``RECONCILIATION.md`` C2 measures ``r(x_0) = 570.82`` rad at ``nu = 1000.5``, about 90
    cycles from zero, and ``r`` traverses 565.82 rad across the near region. A naive fold produces
    a reference that is wrong by an exact multiple of ``2 pi``, which is the single easiest way to
    poison every later prompt.

    So the branch is anchored at the **largest** ``x`` of each order, where
    :func:`tail_residual_series` is accurate, and then tracked **down**:

    1. at the anchor, lift ``atan2(J, -Y)`` to the branch nearest the two-term series and *assert*
       that the lifted residual matches the series to 1e-8 -- an independent check that ``mpmath``'s
       phase, this module's ``c_nu`` and DLMF 10.18.18 all agree;
    2. step down in ``log x``, predicting ``r`` at the next point by first-order Taylor using the
       exact identity ``dr/dlog x = x (a^-2 - 1)`` evaluated at the current point, lifting to the
       branch nearest that prediction, and rejecting-and-halving the step if the lifted value
       disagrees with the prediction by more than :data:`_WALK_ACCEPT_DR` rad. The step is aimed at
       :data:`_WALK_TARGET_DR` rad of residual advance, so the acceptance test has an order of
       margin below the ``pi`` at which the lift would become ambiguous;
    3. corners are hit exactly -- the walk lands on ``mpf(float(x_corner))`` and never on
       ``exp(log(x_corner))`` -- so every stored value is the reference at the supplied double. At
       each corner the phase is then recomputed by :func:`mpmath_reference_extended` and lifted
       again with the walked residual as the predictor, and the two are required to agree; so the
       *stored* residual never depends on the cheap probe of :func:`_walk_probe`, only its branch
       does.

    Independent confirmation of the result: this produces ``r(x_0)`` = 0.615480 rad (nu=3/2),
    1.183200 (5/2), 11.443651 (20.5), 57.104238 (100.5) and 570.820039 (1000.5), against
    ``RECONCILIATION.md`` C2's independently measured 0.6155, 1.1832, 11.4437, 57.1042 and 570.8200
    -- which were obtained from continuously tracked ``hankel1e`` rather than from ``mpmath``, and
    with no series anchor.

    :return: the document that was written.
    """
    import mpmath

    document = {
        "schema_version": REFERENCE_DATA_SCHEMA_VERSION,
        "generated_by": "LiouvilleGreen.tests.bessel_reference.regenerate_reference_table",
        "campaign": "prompts/transfer-remedial (prompt 01)",
        "environment": _environment_block(dps),
        "conventions": {
            "bessel": "J_nu = A sin(theta), Y_nu = -A cos(theta), theta increasing in x",
            "c_nu": "pi/4 - pi*nu/2 (repository convention; DLMF differs by +pi/2)",
            "theta": "theta = x + c_nu + r; theta_continuous = theta_principal + 2*pi*theta_div_2pi",
            "theta_principal": "atan2(J, -Y), in (-pi, pi]",
            "a": "sqrt(pi*x/2)*hypot(J, Y); tends to 1 at large x",
            "theta_deriv": "(2/pi)/(x*(J^2 + Y^2)); an identity, not an approximation",
            "digits": f"decimal strings carry {CACHE_DIGITS} significant digits",
            "x": "stored as a round-tripping decimal ('x') and as a hex double ('x_hex')",
        },
        "entries": [],
    }

    with mpmath.workdps(dps):
        for nu in orders:
            corners = corner_x_values(nu)
            if verbose:
                print(
                    f"-- nu={nu}: {len(corners)} corners, x in [{corners[0]:.6g}, {corners[-1]:.6g}]"
                )

            # anchor at the largest x, where the two-term series is accurate
            anchor_x = mpmath.mpf(float(corners[-1]))
            anchor_bundle = mpmath_reference_extended(nu, corners[-1], dps=dps)
            series_r = mpmath.mpf(float(tail_residual_series(nu, corners[-1])))
            anchor_r, anchor_n = _resolve_residual(
                nu, anchor_x, anchor_bundle["theta_principal"], series_r, mpmath
            )

            series_mismatch = abs(anchor_r - series_r)
            if series_mismatch > mpmath.mpf("1e-8"):
                raise AssertionError(
                    f"regenerate_reference_table: residual branch anchor failed for nu={nu} at "
                    f"x={corners[-1]:.6g}: resolved r={mpmath.nstr(anchor_r, 20)} but the two-term "
                    f"series gives {mpmath.nstr(series_r, 20)}, a mismatch of "
                    f"{mpmath.nstr(series_mismatch, 6)} > 1e-8. Either the branch lift or the phase "
                    f"convention is wrong; do not write a table on this basis."
                )
            if verbose:
                print(
                    f"   anchor r={mpmath.nstr(anchor_r, 12)} matches series to "
                    f"{mpmath.nstr(series_mismatch, 4)}"
                )

            rows = [_entry(nu, corners[-1], anchor_bundle, anchor_r, anchor_n, dps)]

            # walk down, corner by corner
            current_x = anchor_x
            current_r = anchor_r
            current_a = anchor_bundle["a"]
            steps = 0

            for target in reversed(corners[:-1]):
                target_x = mpmath.mpf(float(target))
                target_log = mpmath.log(target_x)

                while current_x > target_x:
                    drdu = current_x * (1 / (current_a * current_a) - 1)
                    scale = max(abs(drdu), mpmath.mpf("1e-300"))
                    log_current = mpmath.log(current_x)
                    remaining = log_current - target_log
                    h = min(
                        mpmath.mpf(_WALK_MAX_DLOGX),
                        mpmath.mpf(_WALK_TARGET_DR) / scale,
                        remaining,
                    )

                    while True:
                        landing = h >= remaining
                        next_x = target_x if landing else mpmath.exp(log_current - h)

                        theta_probe, a_probe = _walk_probe(nu, next_x, mpmath, dps)
                        predicted = current_r - drdu * h
                        resolved, n = _resolve_residual(
                            nu, next_x, theta_probe, predicted, mpmath
                        )

                        if abs(resolved - predicted) <= mpmath.mpf(_WALK_ACCEPT_DR):
                            break

                        h = h / 2
                        if h < mpmath.mpf(_WALK_MIN_DLOGX):
                            raise AssertionError(
                                f"regenerate_reference_table: residual branch walk stalled for "
                                f"nu={nu} near x={float(current_x):.6g}: even at dlog(x)="
                                f"{float(h):.3g} the resolved residual differs from the Taylor "
                                f"prediction by {float(abs(resolved - predicted)):.3g} rad."
                            )

                    current_x = next_x
                    current_r = resolved
                    current_a = a_probe
                    steps += 1

                # the walk has landed exactly on the corner; re-evaluate it at full precision for
                # storage, and lift with the walked residual as the predictor
                bundle = mpmath_reference_extended(nu, target, dps=dps)
                resolved, n = _resolve_residual(
                    nu, target_x, bundle["theta_principal"], current_r, mpmath
                )
                drift = abs(resolved - current_r)
                if drift > mpmath.mpf(_WALK_ACCEPT_DR):
                    raise AssertionError(
                        f"regenerate_reference_table: at nu={nu}, x={target:.6g} the mpmath phase "
                        f"lifts to a residual {mpmath.nstr(resolved, 12)} that disagrees with the "
                        f"walked value {mpmath.nstr(current_r, 12)} by {float(drift):.3g} rad. The "
                        f"branch walk and the high-precision reference disagree; do not write a "
                        f"table on this basis."
                    )
                current_r = resolved
                current_a = bundle["a"]

                rows.append(_entry(nu, target, bundle, current_r, n, dps))

            if verbose:
                print(
                    f"   walked {steps} steps down to x={corners[0]:.6g}; "
                    f"r(x_0)={float(current_r):.6f} rad "
                    f"({float(current_r) / _TWO_PI:.2f} cycles from zero)"
                )

            document["entries"].extend(reversed(rows))

    with open(path, "w") as handle:
        json.dump(document, handle, indent=1, sort_keys=False)
        handle.write("\n")

    if verbose:
        print(f"wrote {len(document['entries'])} entries to {path}")

    _TABLE_CACHE.pop(path, None)
    return document


def _entry(nu, x, bundle, r, theta_div_2pi, dps) -> dict:
    import mpmath

    with mpmath.workdps(dps):
        x_float = float(x)
        return {
            "nu": repr(float(nu)),
            "x": repr(x_float),
            "x_hex": x_float.hex(),
            "J": mpmath.nstr(bundle["J"], CACHE_DIGITS),
            "Y": mpmath.nstr(bundle["Y"], CACHE_DIGITS),
            "amplitude": mpmath.nstr(bundle["amplitude"], CACHE_DIGITS),
            "a": mpmath.nstr(bundle["a"], CACHE_DIGITS),
            "r": mpmath.nstr(r, CACHE_DIGITS),
            "theta_principal": mpmath.nstr(bundle["theta_principal"], CACHE_DIGITS),
            "theta_div_2pi": theta_div_2pi,
            "theta_deriv": mpmath.nstr(bundle["theta_deriv"], CACHE_DIGITS),
        }


def _environment_block(dps: int) -> dict:
    """
    The environment the table was generated in.

    ``DRAFT-PLAN.md`` 9 Stage 1 requires this recorded: the ``hankel1e``/``jv``/``yv`` failure
    boundaries of 4.4 and ``RECONCILIATION.md`` C1 are properties of the bundled Amos library and
    of this platform, not guarantees, so a table that outlives a SciPy upgrade must say what it
    was measured against.
    """
    import datetime

    import mpmath
    import scipy

    return {
        "date": datetime.date.today().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "platform": platform.platform(),
        "mpmath_dps": dps,
    }
