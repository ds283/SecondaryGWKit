"""
Reference harness for the QCD background campaign (``prompts/qcd-background-audit/``, prompt 01).

This module holds the *measurement infrastructure* every later prompt in that campaign is scored
against: an accurate ``T(z)``, the entropy factor it is best expressed through, the locations of
the points at which ``T(z)`` genuinely jumps, the audit's probe geometry, and the three error
statistics (max / p90 / median) every prompt reports in that order.

**Independence.** The shipped representation is one of the things being measured, so it may not
appear anywhere in the reference path. Nothing in this module calls ``_solve_T_z``,
``_build_T_z_spline``, ``T_photon``, ``_T_z_spline``, ``_T_z_spline_knots_log1pz`` or
``integration_break_points`` when computing a reference value
(``prompts/qcd-background-audit/README.md`` §2 (a)). The reference is the defining equation

    T Gs(T)^(1/3) = T_CMB Gs(T_CMB)^(1/3) (1+z),

root-solved to ``rtol = 1e-14``. ``ComputeTargets/tests/wkb_reference.py`` states the same rule
for the phase campaign, one level downstream.

Two things are read from the cosmology that are *not* the representation, and are used only to
fix the reference's domain and its constants:

* ``_T_CMB``, ``_G_S_CMB_pow13`` and ``_eos`` -- the parameters of the defining equation itself;
* the tabulated redshift range, reconstructed in :func:`tabulated_u_range` from the model's
  declared ``max_z`` and ``DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT`` with ``_build_T_z_spline``'s 5%
  buffer in ``1+z``, so that a candidate representation is scored over the same interval the
  shipped one covers. That is arithmetic on the model's declared bounds, not an evaluation of the
  representation.

:func:`temperature_override` *does* assign to ``_T_z_spline``, but that is the measurement hook --
it is how a candidate ``T(z)`` is pushed through ``Hubble``/``rho`` -- and never a source of a
reference value.

**Where the probe geometry comes from.** The audit scored on production source-grid nodes *and*
the midpoints between them, because a spline is worst between its knots. The grid geometry is
reproduced here from the documented constants rather than imported from
``ComputeTargets.tests.wkb_reference``: that module imports ``ComputeTargets.BackgroundModel``,
hence Ray, into a package that does not otherwise need it, and its ``z_init`` comes from
``wkb_reference_data.json``, which later prompts in this campaign regenerate. Pinning
:data:`PRODUCTION_Z_INIT` keeps the probe set *fixed* for the whole campaign, which is what makes
the acceptance table in README §6.1 comparable from one prompt to the next.

**Redshift arithmetic.** Everything is carried in ``u = log(1+z)``. The ``expm1(u)`` calls below
are the lossy direction (``CLAUDE.md``), and are tolerable for exactly the reason
``temperature_crossing_log1pz`` gives for its own: the recovered ``z`` is only a probe location
or a quadrature limit, and the very next thing done with it is to take ``log1p`` again. No
equality-like comparison is ever made on a recovered ``z``.

No Ray and no datastore is needed.
"""

import warnings
from contextlib import contextmanager
from math import exp, log, log1p, log10, expm1, pow
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar

from CosmologyModels.GenericEOS.GenericEOS import HIGH_T_GSTAR
from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import (
    DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT,
)

# ---------------------------------------------------------------------------------------------
# production geometry
# ---------------------------------------------------------------------------------------------

# main.py:69-71, via ComputeTargets/tests/wkb_reference.py
PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z = 100
PRODUCTION_Z_END = 0.1

# The head of the universal source grid: five e-folds outside the horizon for the
# earliest-exiting production wavenumber, k = 3e8 /Mpc (main.py:411). The value is the one
# recorded in ComputeTargets/tests/wkb_reference_data.json under
# models/LambdaCDMModel/grid/z_init, and it is pinned here rather than recomputed because
# horizon exit is solved against H(z), which this campaign is about to move: recomputing it would
# shift the probe set out from under the acceptance table.
PRODUCTION_Z_INIT = 2.0636395964161516e16

# the audit's probe restriction: z in (1, 1e16), decimated by 5, giving 640 points
PROBE_Z_MIN = 1.0
PROBE_Z_MAX = 1.0e16
PROBE_DECIMATION = 5

# k*tau spans at the three production wavenumbers, from docs/gktk-remedial-verification.md §3.5.
# Used only to express a relative error in tau as radians of phase.
PRODUCTION_SPANS_RAD = {1.0e5: 1.3728e09, 1.0e7: 1.3728e11, 3.0e8: 4.1184e12}

# one ulp of each of those spans, the floor the consumer path has been polished to
PRODUCTION_FLOOR_RAD = {1.0e5: 3.05e-07, 1.0e7: 3.05e-05, 3.0e8: 9.15e-04}


# ---------------------------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------------------------


def accurate_T(cosmology, z: float, rtol: float = 1.0e-14) -> float:
    """
    ``T(z)`` from the defining equation ``T Gs(T)^(1/3) = T_CMB Gs(T_CMB)^(1/3) (1+z)``,
    root-solved to ``rtol``. **The reference for everything in this campaign.**

    The bracket is the shipped one: ``Gs(T) >= Gs(T_CMB)`` makes naive redshifting an
    overestimate, and ``Gs`` is bounded above by its asymptotic value, so
    ``[0.95 T_CMB (1+z) / g*^(1/3), 1.05 T_CMB (1+z)]`` contains the root. ``xtol`` is set
    absurdly small so that ``rtol`` alone decides convergence -- ``T`` is dimensionful and tiny in
    Mpc units, and an absolute tolerance would terminate the solve immediately.

    Cost is ~8 us per call, which is why the shipped code pays it once per spline node. A caller
    scoring a probe set should cache the result.

    :param cosmology: a ``LambdaCDM_GenericEOS`` (or subclass)
    :param z: redshift
    :param rtol: relative tolerance of the root solve
    :return: the temperature at ``z``, as a dimensionful quantity in the cosmology's units
    """
    T_CMB = cosmology._T_CMB
    eos = cosmology._eos

    bracket_hi = 1.05 * T_CMB * (1.0 + z)
    bracket_lo = 0.95 * T_CMB * (1.0 + z) / pow(HIGH_T_GSTAR, 1.0 / 3.0)
    target = T_CMB * cosmology._G_S_CMB_pow13 * (1.0 + z)

    def equation(T: float) -> float:
        return T * pow(eos.Gs(T), 1.0 / 3.0) - target

    root = root_scalar(
        equation, bracket=(bracket_lo, bracket_hi), xtol=1.0e-300, rtol=rtol
    )
    if not root.converged:
        raise RuntimeError(
            f"T_z_reference.accurate_T: root_scalar() did not converge at z={z:.6g}: "
            f'"{root.flag}"'
        )
    return root.root


def entropy_factor(cosmology, u: float, rtol: float = 1.0e-14) -> float:
    """
    ``F(u) = log( T / (T_CMB (1+z)) ) = -(1/3) log( Gs(T)/Gs(T_CMB) )`` at ``u = log(1+z)``.

    This is the whole of the difference between ``T(z)`` and the ``(1+z)`` ramp known in closed
    form. It is bounded, ``O(1)``, and *exactly constant* wherever ``Gs`` is -- which is why
    splining it rather than ``T`` is finding T3 of the audit.

    :param cosmology: a ``LambdaCDM_GenericEOS`` (or subclass)
    :param u: ``log(1+z)``
    :param rtol: relative tolerance of the underlying root solve
    :return: ``F(u)``, dimensionless
    """
    return log(accurate_T(cosmology, expm1(u), rtol) / (cosmology._T_CMB * exp(u)))


def tabulated_u_range(cosmology) -> tuple:
    """
    The interval in ``u = log(1+z)`` that ``_build_T_z_spline`` tabulates: the model's declared
    ``[DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, max_z]`` widened by 5% in ``1+z`` at each end.

    Reconstructed from the declared bounds rather than read off the representation, so that it
    stays meaningful when the representation changes under it. A candidate scored against this
    range is scored over the same interval the shipped spline covers.

    :param cosmology: a ``LambdaCDM_GenericEOS`` (or subclass)
    :return: ``(u_lo, u_hi)``
    """
    min_z = 0.95 * (1.0 + DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT) - 1.0
    max_z = 1.05 * (1.0 + cosmology._max_z) - 1.0
    return log1p(min_z), log1p(max_z)


def jump_locations(cosmology, rtol: float = 1.0e-14) -> list:
    """
    The values of ``u = log(1+z)``, strictly inside the tabulated range, at which ``T(z)`` crosses
    one of the equation of state's ``break_temperatures_GeV``. Ascending.

    **Found by bisecting the monotone ``T(z)``, never by root-finding on ``T(z) - T_break``**
    (README §2 (b), audit §2). ``T(z)`` *jumps* at exactly these points, so the difference
    ``T(z) - T_break`` has no root there at all: at the lowest crossing it steps from
    ``-7.53e-04`` to ``+8.84e-06`` relative across a single ulp of ``u``. A bracketing solver
    applied to it nevertheless reports ``converged`` and returns a non-root, at a place that
    depends on its tolerances -- measured between 4 and 317 ulp above the step, which straddles
    the padding a segmented build uses to keep each branch's nodes inside its own segment. A
    segment edge misplaced that way leaves the full 7.6e-04 error in place, and the audit records
    a first attempt that did exactly this and measured 5.7e-04.
    ``test_a_segment_edge_bisected_and_one_root_found_disagree`` in ``test_T_z_representation.py``
    is the standing demonstration.

    The bisection is geometric in ``1+z``, run to a relative width of 1e-15, which is the
    representable floor: the returned ``u`` is good to the last bit or two.

    :param cosmology: a ``LambdaCDM_GenericEOS`` (or subclass)
    :param rtol: relative tolerance of the underlying root solve
    :return: an ascending ``list`` of ``u`` values
    """
    u_lo_range, u_hi_range = tabulated_u_range(cosmology)

    out = []
    for T_break_GeV in sorted(set(cosmology._eos.break_temperatures_GeV)):
        T_break = T_break_GeV * cosmology.units.GeV

        lo, hi = 1.0e-6, 1.0e19
        if (
            not accurate_T(cosmology, lo, rtol)
            < T_break
            < accurate_T(cosmology, hi, rtol)
        ):
            # the crossing is outside the bracketable range; T_HI = 1e16 GeV is reached only at
            # z ~ 1e28, far above any tabulated range
            continue

        for _ in range(200):
            mid = (lo * hi) ** 0.5
            if accurate_T(cosmology, mid, rtol) < T_break:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1.0 < 1.0e-15:
                break

        u = log1p(0.5 * (lo + hi))
        if u_lo_range < u < u_hi_range:
            out.append(u)

    return sorted(out)


def temperature_crossing_log1pz(
    cosmology, T: float, u_lo: float, u_hi: float
) -> Optional[float]:
    """
    An *approximate* location for the point u = log(1+z), strictly inside (u_lo, u_hi), at
    which T_photon(z) reaches the dimensionful temperature T; None if it does not cross inside
    the range.

    **This is test machinery, not a production solver: nothing in production calls it, and
    nothing may put it back on the break-point path.** It is kept, not deleted, because it is
    two things at once -- the neighbourhood probe
    ComputeTargets/tests/test_numeric_break_points.py::
    test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink uses to find a neighbourhood
    of a crossing, and the documented illustration of the trap README §2 (b) of
    prompts/qcd-background-audit/ is about. Until prompt 07 of prompts/qcd-background-audit/ it
    was how integration_break_points located the equation-of-state crossings; that method now
    returns the bisected _break_point_crossings_log1pz, which is where the crossings actually
    are. The reason is README §2 (b): since prompt 06 the representation is segmented at exactly
    these temperatures, so T_photon genuinely *jumps* there, and log T_photon(z) - log T need not
    have a root at all. A bracketing solver applied to it reports converged and returns a
    non-root whose offset depends on its tolerances -- measured at +1.126e-12 in u at
    root_scalar's defaults, which is further from the jump than the 1e-12 by which each
    segment's nodes are held inside their own branch. CosmologyModels/tests/
    test_T_z_representation.py::test_a_segment_edge_bisected_and_one_root_found_disagree is
    the standing demonstration.

    It survives as a measurement probe: ComputeTargets/tests/test_numeric_break_points.py::
    test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink uses it to find a
    neighbourhood of a crossing, which is a use its ~1e-12 offset does not disturb, and it is
    the documented illustration of why a bracket is the wrong tool here.

    T_photon(z) is monotone in z, so the crossing is unique where it exists. It is solved for
    in u, which is the campaign's integration variable, to xtol = rtol = 1e-15. The expm1(u)
    inside q() is the lossy log(1+z) -> z direction (CLAUDE.md), but T_photon takes log(1+z)
    again internally, so it costs ~1 ulp of u.

    It lived as a private method on LambdaCDM_GenericEOS until
    prompts/background-solver-robustness/ prompt 04 moved it here, character-identical in its
    root_scalar call.

    :param cosmology: a ``LambdaCDM_GenericEOS`` (or subclass)
    :param T: the dimensionful temperature to cross
    :param u_lo: lower end of the search bracket, in ``u = log(1+z)``
    :param u_hi: upper end of the search bracket, in ``u = log(1+z)``
    :return: the crossing in ``u``, or ``None``
    """
    log_T = log(T)

    def q(u: float) -> float:
        return log(cosmology.T_photon(expm1(u))) - log_T

    q_lo = q(u_lo)
    q_hi = q(u_hi)
    if q_lo == 0.0 or q_hi == 0.0 or (q_lo > 0.0) == (q_hi > 0.0):
        return None

    root = root_scalar(q, bracket=(u_lo, u_hi), xtol=1e-15, rtol=1e-15)
    if not root.converged:
        raise RuntimeError(
            f"T_z_reference.temperature_crossing_log1pz: root_scalar() did not converge "
            f"for T = {T / cosmology._units.GeV:.5g} GeV between u = {u_lo:.6g} and {u_hi:.6g}: "
            f'"{root.flag}"'
        )
    u = float(root.root)
    if not u_lo < u < u_hi:
        return None
    return u


# ---------------------------------------------------------------------------------------------
# the probe geometry
# ---------------------------------------------------------------------------------------------


def production_source_z_values(
    z_init: float = PRODUCTION_Z_INIT,
    z_end: float = PRODUCTION_Z_END,
    samples_per_log10z: int = PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
) -> np.ndarray:
    """
    Reproduce ``wavenumber_exit_time.populate_z_sample`` (``CosmologyConcepts/wavenumber.py:250``):
    a descending grid, log-spaced **in z** (not in ``1+z``), with ``samples_per_log10z`` points per
    decade between ``z_init`` and ``z_end``. On the production defaults this is 1,732 samples.

    :return: a descending ``numpy`` array of redshifts
    """
    num = int(round(samples_per_log10z * (log10(z_init) - log10(z_end)) + 0.5, 0))
    return np.logspace(log10(z_init), log10(z_end), num=num)


def probe_set(
    z_values: Optional[Sequence[float]] = None,
    z_min: float = PROBE_Z_MIN,
    z_max: float = PROBE_Z_MAX,
    decimation: int = PROBE_DECIMATION,
) -> np.ndarray:
    """
    The audit's probe geometry (§3): the production source-grid nodes **and the midpoints between
    them** in ``u``, restricted to ``z in (z_min, z_max)`` and decimated by ``decimation``.

    The midpoints are the point of it. A spline is exact at its knots and worst between them, so a
    probe set drawn only from grid nodes would flatter any interpolating representation. On the
    production defaults this returns 640 ascending redshifts spanning ``z`` from ~1 to ~9.5e15,
    which is the set every figure in README §6.1 is measured on.

    :param z_values: the grid to probe on and between; the production source grid by default
    :param z_min: exclusive lower redshift bound
    :param z_max: exclusive upper redshift bound
    :param decimation: keep every ``decimation``-th probe
    :return: an ascending ``numpy`` array of redshifts
    """
    if z_values is None:
        z_values = production_source_z_values()

    zs = np.sort(np.asarray(z_values, dtype=float))
    u_nodes = np.log1p(zs)
    probe_u = np.unique(np.concatenate([u_nodes, 0.5 * (u_nodes[:-1] + u_nodes[1:])]))

    probe_z = np.expm1(probe_u)
    return probe_z[(probe_z > z_min) & (probe_z < z_max)][::decimation]


def reference_temperatures(
    cosmology, probe_z: Sequence[float], rtol: float = 1.0e-14
) -> np.ndarray:
    """
    :func:`accurate_T` over a probe set. ~8 us per point, so ~5 ms for the production 640: build
    it once per test class, not once per test.
    """
    return np.array(
        [accurate_T(cosmology, float(z), rtol) for z in probe_z], dtype=float
    )


# ---------------------------------------------------------------------------------------------
# error statistics
# ---------------------------------------------------------------------------------------------


def relative(candidate, reference) -> np.ndarray:
    """Elementwise ``|candidate - reference| / |reference|``."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return np.abs(candidate - reference) / np.abs(reference)


class Stats(NamedTuple):
    """
    The campaign's three error statistics, always in this order: max, p90, median.

    The order is not cosmetic. The audit's §4 table separates the three representation defects by
    exactly these three columns -- accurate nodes fix the p90, the entropy factor fixes the
    median, segmentation fixes the max -- so a prompt that reports them in this order can be read
    straight against README §6.1.
    """

    max: float
    p90: float
    median: float

    @staticmethod
    def of(rel) -> "Stats":
        """Summarise an array of relative errors."""
        rel = np.asarray(rel, dtype=float)
        return Stats(
            max=float(np.max(rel)),
            p90=float(np.percentile(rel, 90)),
            median=float(np.median(rel)),
        )

    def format(self, label: str = "") -> str:
        """One fixed-width line, for printing from a test."""
        return (
            f"{label:<46s} max {self.max:.3e}   p90 {self.p90:.3e}   "
            f"median {self.median:.3e}"
        )


# ---------------------------------------------------------------------------------------------
# the downstream hook: H(z) and conformal time under a substituted temperature
# ---------------------------------------------------------------------------------------------


@contextmanager
def temperature_override(cosmology, T_of_z: Callable[[float], float]):
    """
    Temporarily replace the cosmology's temperature representation by ``T_of_z``, so that
    ``Hubble``, ``rho`` and everything built on them can be evaluated against a *different*
    background.

    This is the measurement hook that makes a background-against-background comparison possible at
    all, and it is the only place in this module that touches ``_T_z_spline``. It is never a
    source of a reference value: the reference is whatever ``T_of_z`` computes.

    ``_rho_fluid`` (``LambdaCDM_GenericEOS.py:334``) calls ``self._T_z_spline(z)`` directly rather
    than going through ``T_photon``, so that attribute is the hook, not the method.
    """
    original = cosmology._T_z_spline
    cosmology._T_z_spline = T_of_z
    try:
        yield cosmology
    finally:
        cosmology._T_z_spline = original


def Hubble_with(cosmology, T_of_z, z_values) -> np.ndarray:
    """``H(z)`` over ``z_values``, with the cosmology's temperature replaced by ``T_of_z``."""
    with temperature_override(cosmology, T_of_z):
        return np.array([cosmology.Hubble(float(z)) for z in z_values], dtype=float)


def inverse_Hubble_integral(
    cosmology,
    T_of_z,
    u_a: float,
    u_b: float,
    points: Optional[Sequence[float]] = None,
    epsrel: float = 1.0e-11,
    limit: int = 400,
) -> float:
    """
    ``int dz/H`` between ``u_a`` and ``u_b`` in ``u = log(1+z)``, with the cosmology's temperature
    replaced by ``T_of_z``. Up to an additive constant this is the conformal time ``tau``, so a
    relative error here is a relative error in ``tau``, which is a phase error proportional to
    ``k tau`` (audit §5).

    The integrand is ``(1+z)/H`` because ``dz = (1+z) du``.

    ``points`` should be the interior jump locations: the integrand is discontinuous there, and an
    adaptive rule that is not told will either miss the step or spend its whole subdivision budget
    on it.

    ``IntegrationWarning`` is suppressed with the audit's justification. The exact integrand
    root-solves to machine precision on every call, so ``quad`` reports roundoff before it reaches
    ``epsrel``; the figure is nevertheless converged against the setting -- 3.4744e-08,
    3.4602e-08, 3.4605e-08 and 3.4605e-08 at ``epsrel`` = 1e-9, 1e-10, 1e-11, 1e-12, four digits
    of stability across four decades, well inside the effect being measured.
    """

    def integrand(u: float) -> float:
        z = expm1(u)
        return (1.0 + z) / cosmology.Hubble(z)

    with temperature_override(cosmology, T_of_z):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            value, _ = quad(
                integrand,
                u_a,
                u_b,
                limit=limit,
                epsabs=0,
                epsrel=epsrel,
                points=list(points) if points else None,
            )
    return value
