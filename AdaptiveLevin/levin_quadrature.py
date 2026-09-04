"""
Adaptive Levin quadrature for rapidly oscillatory integrals.

BASIS OF THE ALGORITHM
----------------------
The adaptive scheme implemented here is the algorithm of

    J. Bremer, Z. Chen, H. Yang,
    "Rapid evaluation of oscillatory integrals and their application to
     the numerical solution of the Helmholtz equation",
    arXiv:2211.13400,

specifically the adaptive Levin method of its section 5. Equation and step
numbers appearing in comments throughout this file ("Chen et al. (168)",
"Chen et al. step (4), below (173)", ...) refer to that preprint. Page
references are to **v3**; the numbering differs in v1/v2.

The underlying non-adaptive Levin rule is due to

    D. Levin, "Fast integration of rapidly oscillatory functions",
    J. Comput. Appl. Math. 67 (1996) 95-101,

whose equation numbers are cited as "Levin 96".

DEVIATIONS FROM THE REFERENCE ALGORITHM
---------------------------------------
This implementation has evolved beyond the bare algorithm of section 5. The
deliberate departures are:

  * Region acceptance (step (4)) additionally admits a *relative* tolerance;
    Chen et al.'s test is absolute only. See _adaptive_levin() for why this
    matters and how it is now gated.

  * The returned error estimate accounts for a round-off floor -- Chen et
    al.'s own eq. (151), see _roundoff_floor() -- as well as the step-(4)
    resolution residual (or, for a Clenshaw-Curtis fallback region, its
    nested-pair estimate). See the extended discussion in _adaptive_levin();
    the short version is that the step-(4) residual is a *resolution*
    criterion and is blind, by construction, to round-off, which is
    dominated by eps * ||f||_inf and the phase's local conditioning rather
    than by how well the Chebyshev interpolant resolves f and theta'. Chen
    et al. describe the existence of this loss of accuracy in prose (v3 p.
    30, immediately after the algorithm) rather than folding a bound for it
    into their estimate, which is legitimate for a paper but misleading in a
    library API. (An earlier version of this module modelled the floor as
    rounding of the phase specifically at the two region endpoints, with an
    inferred scale; that model was optimistic by up to 8.1e9 on the
    production code path and has been replaced -- see prompts/levin-refactor
    /logs/04-roundoff-floor.md.)

  * Weakly oscillatory regions (total variation of the phase, mean|theta'| * width,
    below 6*pi) are handed to a nested Clenshaw-Curtis rule rather than to the Levin
    rule. The extremal Chebyshev grid the Levin solve already uses *is* the
    Clenshaw-Curtis grid, and its N-point grid is exactly every other node of the
    (2N-1)-point grid (nesting holds for every N >= 2), so the pair (CC_N, CC_{2N-1})
    costs 2N-1 integrand and phase evaluations, always, and comes with its own error
    estimate |CC_{2N-1} - CC_N|. That estimate re-enters the same accept/bisect logic
    as a Levin estimate, so a fallback region that misses its tolerance is bisected
    rather than accepted outright. The gate is the phase's *total variation*, not its
    net change across the region: the two differ whenever theta' changes sign inside
    the region (a stationary point of the phase), and gating on the net change alone
    hands such a region to a direct rule that has to resolve the full oscillation in
    one panel, or -- if gated on net change *without even measuring the oscillation at
    all* -- can be handed a case whose net phase change happens to be small purely by
    cancellation. Levin's method itself is not applicable at a stationary point (its
    antiderivative p ~ f/(i*theta') is singular there); refining until the stationary
    neighbourhood is weakly oscillatory and then applying a direct rule to it, which is
    what this gate-and-bisect structure does, is the correct structural response.

  * A direct LU solve is used in place of the least-squares/SVD solve when the
    phase span shows the Levin super-operator to be well conditioned. Note
    that Remark 2 of Chen et al. recommends a *rank-revealing QR* in place of
    the truncated SVD throughout (they report ~5x faster with no apparent loss
    of accuracy). That suggestion has NOT been taken up here: the fast path is
    an LU solve gated on conditioning, with lstsq retained as the fallback for
    the ill-conditioned case where the minimum-norm solution is needed. An
    RRQR fallback in place of lstsq remains an unexplored optimisation.

LOGGING
-------
This module logs through the standard library logger "AdaptiveLevin.levin_quadrature"
(logging.getLogger(__name__)) rather than printing to stdout. As a library it attaches only a
logging.NullHandler(), so by default -- with no handler configured anywhere in the logging
hierarchy -- every message from this module is silently discarded, including warnings. This is a
deliberate, and as of prompts/levin-refactor/logs/07-diagnostics-hygiene.md a *new*, default:
earlier versions of this module printed warnings and progress notices unconditionally. A host
application that wants to see them should configure logging in the usual way, e.g.

    import logging
    logging.basicConfig(level=logging.INFO)   # or WARNING to see only warnings

or attach a handler directly to logging.getLogger("AdaptiveLevin.levin_quadrature"). Solve
failures/degradations and the aggregate/health-check warnings are logged at WARNING; progress
notifications and the depth-18 diagnostic history dump are logged at INFO.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from functools import lru_cache
from math import floor, ceil
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import toeplitz

from utilities import format_time

# seaborn and matplotlib.pyplot are imported inside _write_progress_data(), the only function
# that uses them, rather than at module scope. Measured with -X importtime (see prompts/levin-
# refactor/logs/07-diagnostics-hygiene.md): they accounted for 95-97% of this module's 1.3-2.1 s
# import cost, paid unconditionally -- including by every Ray worker that imports this module but
# never runs with emit_diagnostics=True.

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# default interval at which to log progress of the integration
DEFAULT_LEVIN_NOTIFY_INTERVAL = 5 * 60

# Default location for diagnostic output (the lstsq/pinv failure dump, and the emit_diagnostics
# progress plots/JSON payload), used when adaptive_levin_sincos()'s diagnostics_path argument is
# not supplied. Previously both writers used cwd-relative paths computed implicitly at call time
# (the failure dump wrote directly into whatever the process's current working directory happened
# to be; _write_progress_data wrote to a cwd-relative "SlowLevinData/" tree) -- hazardous under a
# Ray driver, where cwd is not guaranteed to be stable, writable, or distinct per worker. This is
# a named, explicit default rather than a Path.cwd() computed on demand, and it is a behavioural
# change for anyone relying on the old implicit locations -- see
# prompts/levin-refactor/logs/07-diagnostics-hygiene.md.
DEFAULT_LEVIN_DIAGNOSTICS_PATH = Path("levin_diagnostics")

# default Chebyshev spectral order (recommendation 13). Raised from 12 to 16 in prompt 08
# (prompts/levin-refactor/logs/08-order-and-sampling.md) after re-measuring order 8/12/16/24/32 on
# the four AdaptiveLevin/tests/ problems and three three-Bessel oracles the audit's own sec 4.5
# names: order 16 was faster than 12 on all three three-Bessel oracles (1.4-3.9x, best-of-2) and on
# every AdaptiveLevin test problem that subdivides at all (fewer regions -> fewer solves), with
# delivered accuracy against each closed form unaffected (three-Bessel: bit-identical to the
# reported digits) or improved. The useful band is 12-32: below it (order 8) is worse on every
# problem measured, sometimes catastrophically (500x on a QuadSourceIntegral-shaped problem -- see
# the log); above roughly 32 the per-solve cost keeps rising once a problem has already stopped
# subdividing, for no further accuracy. The optimum depends on how much the amplitude subdivides at
# a given order, so this is a default, not a universally-best value -- see the log's order-sweep
# table for the full evidence, including the one place raising the default has a real cost: eq.
# (151)'s max(G1, k^2)/G0 term rises by (16/12)^2 = 1.78x wherever k^2 > G1, measured in isolation
# at 1.68x on a region just above the SIX_PI Levin/fallback boundary.
DEFAULT_LEVIN_CHEBSHEV_ORDER = 16
_LEVIN_MINIMUM_ALLOWED_ORDER = 8

# default maximum bisection depth. 1/2^20 is roughly 1E-6
DEFAULT_LEVIN_MAX_DEPTH = 20

# default abs tolerance
DEFAULT_LEVIN_ABSTOL = 1e-15

# default rel tolerance
DEFAULT_LEVIN_RELTOL = 1e-7

# approximate machine epsilon for 64-bit floats
MACHINE_EPSILON = 1e-16

SIX_PI = 6.0 * np.pi

# Safety factor 'C' in Chen et al. eq. (151), the round-off floor applied to every region's
# estimate (Levin or Clenshaw-Curtis fallback alike):
#
#   round_off_err = C * eps * ||f||_inf * (h/2) * (1 + G1/G0 + max(G1, k^2)/G0)
#
# See _roundoff_floor() and prompts/levin-refactor/logs/04-roundoff-floor.md. Measured across 26
# resolved cells (three problem families x five decades of omega x two Chebyshev orders,
# audit's e29_paperbound.py) the worst true/bound ratio at C = 1.0 was 0.22 -- i.e. 4.5x margin --
# against 1.24 (an *exceeded*, i.e. wrong, bound) for the endpoint model this floor replaces.
# Reported as measured: the margin is already ample and there is no evidence here that would
# justify inflating every reported error bar for a wider one.
_LEVIN_ROUNDOFF_SAFETY = 1.0

# Safety factor applied only to the *declared* theta_abserr endpoint term (recommendation 5.2);
# see _phase_error(). This is deliberately a separate constant from _LEVIN_ROUNDOFF_SAFETY above:
# eq. (151) replaces the endpoint model as the *default* floor, so one shared factor could not
# serve both models at once. The audit's recommendation 10 measured the *old, inferred* endpoint
# model (theta_scale inferred as max|theta| or a hardwired 2*pi -- see prompt 04's log for why
# that inference was removed) exceeded by 1.24x in one of 30 cells and suggested raising a shared
# factor to 4-8. That does not transfer unchanged to this narrower use: a *declared* theta_abserr
# is a number the caller supplies about their own phase function, not a scale this module infers,
# and no production caller supplies one yet (see adaptive_levin_sincos()'s docstring and README
# Sec 6 of the levin-refactor campaign). Kept at 1.0 pending a real caller and a validation set of
# its own.
#
# CAVEAT carried over from the endpoint model this narrows to: the term assumes theta_abserr is a
# genuine *estimate* of the phase's own error, carrying full relative weight, not already a loose
# upper bound -- if a future caller's theta_abserr is itself conservative, this factor should not
# also inflate it. Erring loose rather than optimistic remains the safe default in the meantime.
_LEVIN_PHASE_ERROR_SAFETY = 1.0

# Relative noise floor for G0 = min|theta'| in _roundoff_floor()'s eq. (151) bound (prompt 04,
# note (b)): G0 <= this fraction of G1 = max|theta'| is treated as G0 being numerically
# indistinguishable from zero, not merely small, and the round-off floor is reported as +inf
# rather than as whatever large-but-finite number the ratio G1/G0 happens to produce.
#
# This is not cosmetic. A literal `G0 > 0.0` test is not enough: when theta' is obtained by
# spectral differentiation of a sampled phase (the common case -- no theta_deriv supplied), a
# region whose true minimum |theta'| is exactly zero at an interior stationary point still comes
# back from the Chebyshev differentiation matrix as a small nonzero float, because Chebyshev
# differentiation amplifies floating-point noise in the sampled theta values by a factor that
# grows with the collocation order (empirically ~k^2). Measured on the C2 stationary-phase
# regression (theta = 1e6*(x - x^2), true stationary point at x = 0.5, order 12): a region with
# one endpoint exactly at the stationary point produced G0 ~ 1.9e-9, G1 ~ 2.5e5 -- a ratio of
# ~7.5e-15, comfortably nonzero by a bare G0 > 0.0 test, but pure differentiation noise. Treating
# it at face value inflated the reported round-off floor to ~4e-3 (comparable to the integral's
# own value) and let it satisfy _adaptive_levin()'s phase_limited test, freezing a region whose
# value was in fact 490% wrong -- see prompts/levin-refactor/logs/04-roundoff-floor.md.
#
# 1e-10 sits several orders of magnitude above the observed noise level (~1e-15 to ~1e-12 for the
# collocation orders 8-32 this module clamps to) while remaining far below any ratio a genuinely
# resolvable (non-degenerate-to-double-precision) physical configuration could produce -- a real
# G0/G1 ratio below ~1e-10 is not meaningfully distinguishable from zero in float64 arithmetic
# regardless of its origin. Erring towards +inf here is safe in the sense _roundoff_floor()'s
# docstring explains: it can only ever delay acceptance of a region (forcing further bisection or
# a fall-through to depth_max), never force a wrong early acceptance.
_LEVIN_ROUNDOFF_G0_NOISE_FLOOR = 1.0e-10

# minimum phase change across a subinterval before we are prepared to invert the Levin super-operator
# with a direct LU solve rather than a least-squares (SVD) solve. Below this the operator is poorly
# conditioned and we need the minimum-norm solution; see _adaptive_levin_subregion_impl().
_LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20.0 * np.pi

INTERVAL_TYPE_LEVIN = 0
INTERVAL_TYPE_DIRECT = 1
types = {0: "Levin", 1: "direct"}


class _LazyUUID:
    """
    Defers uuid.uuid4() (measured ~2.3 microseconds, audit sec 4.3) until this run's id is first
    stringified -- which happens only inside a warning or progress message. A run that never emits
    one (the common case: well-conditioned, low-depth, no diagnostics) never pays for a uuid at
    all. Formats exactly like the uuid.UUID it wraps, so every existing f"...{id_label}..." call
    site works unchanged; only the *construction* site (_adaptive_levin()) changes.
    """

    __slots__ = ("_uuid",)

    def __init__(self):
        self._uuid = None

    def _get(self) -> uuid.UUID:
        if self._uuid is None:
            self._uuid = uuid.uuid4()
        return self._uuid

    def __str__(self) -> str:
        return str(self._get())

    def __repr__(self) -> str:
        return str(self._get())


def _format_label(notify_label: Optional[str], id_label) -> str:
    """
    Build the "<notify_label> id=<id_label>" tag used throughout this module's log messages, from
    the same two values every function already carries as parameters. Kept as a function (rather
    than a value precomputed once per call and threaded through as a "label" parameter, which is
    what earlier versions of this module did) so that it is only ever evaluated at an actual
    logging call site -- an unconditional precomputed "label" would stringify id_label (see
    _LazyUUID above) on every subregion solve regardless of whether a message is ever emitted,
    which is exactly the fixed per-call cost sec 4.3 measures.
    """
    if id_label is not None:
        return f"{notify_label} id={id_label}"
    return f"{id_label}"


def _phase_error(theta_abserr: float, p_endpoint_l1: float) -> float:
    """
    Estimate the absolute error contributed to a region's estimate by a declared absolute error
    theta_abserr in the phase at one region endpoint.

    As of prompt 04 (prompts/levin-refactor/logs/04-roundoff-floor.md) this function is used only
    for the *declared* theta_abserr endpoint term (recommendation 5.2): it is no longer the
    default accuracy floor, which is now _roundoff_floor()'s implementation of Chen et al. eq.
    (151). Previously theta_scale here was a *magnitude* the module inferred (max|theta|, or a
    hardwired 2*pi) and the true error was assumed to be eps * theta_scale; that inference is
    gone (see prompt 04's log for why it was wrong -- C4), so this function's first argument is
    now a caller-declared absolute error already in radians, and is used directly with no
    additional eps factor.

    The Levin estimate for a region is

        value = sum_i p_i(b) w_i(b) - sum_i p_i(a) w_i(a),        w = (sin theta, cos theta)

    i.e. it depends on theta *only* through its values at the two endpoints. If those values
    carry an absolute error d(theta) then, since |d w / d theta| <= 1 componentwise,

        |d value| <= d(theta) * ( sum_i |p_i(b)| + sum_i |p_i(a)| ).

    With d(theta) = theta_abserr this gives the bound returned here.

    Why this is needed at all: the step-(4) residual |val0 - valL - valR| cannot see this
    error. Bisecting [a, b] at c gives regions [a, c] and [c, b] whose contributions at c
    cancel in the sum, so parent and children are built from the *identical* values of
    theta(a) and theta(b). Endpoint phase rounding is therefore common-mode between the two
    sides of the step-(4) comparison and subtracts out exactly. It is invisible by
    construction, and no amount of subdivision will reveal it.

    Chen et al. (v3, p. 30) identify the same effect in prose immediately after stating the
    algorithm: they note that the condition number of the oscillatory integral grows with the
    magnitude of the phase, that the principal loss of accuracy occurs when the large-magnitude
    exponentials of their (171) -- the endpoint expression above -- are evaluated, and that
    since the integral itself typically shrinks with frequency the *absolute* error often stays
    constant or decays. So the step-(4) test is a valid resolution criterion; it is simply not
    a total-error estimate, and was never claimed to be one.

    :param theta_abserr: declared absolute error of the phase at the endpoint, in radians
    :param p_endpoint_l1: sum over components of |p| at the endpoint(s) this call covers
    """
    return _LEVIN_PHASE_ERROR_SAFETY * theta_abserr * p_endpoint_l1


def _levin_G0_G1(theta_prime_sample: np.ndarray, width: float) -> Tuple[float, float]:
    """
    G0 = min|theta'| and G1 = max|theta'|, on the *rescaled* interval [-1, 1] that Chen et al.
    (151) is stated on -- not on the raw x-interval. theta_prime_sample is theta' sampled on the
    raw interval (e.g. theta_prime_Cheb), so it is scaled here by (h/2), h = width, to convert.

    This distinction matters (prompt 04's note (a)): the ratio G1/G0 is scale-invariant, so it
    looks the same whichever interval you compute it on and a bug here would go unnoticed on it;
    max(G1, k^2)/G0 is *not* scale-invariant, and silently omitting the (h/2) factor gives a bound
    that is wrong by (h/2)^2 on every problem where the k^2 term dominates.

    :param theta_prime_sample: theta' sampled at the region's Chebyshev collocation points (any
        order; only its min/max magnitude is used).
    :param width: |b - a|, the raw interval width.
    """
    half_width = 0.5 * width
    abs_theta_prime = np.fabs(theta_prime_sample)
    G0 = float(np.min(abs_theta_prime)) * half_width
    G1 = float(np.max(abs_theta_prime)) * half_width
    return G0, G1


def _roundoff_floor(f_scale: float, width: float, G0: float, G1: float, k: float) -> float:
    """
    Round-off floor for a region's estimate, from Chen et al. (arXiv:2211.13400v3) eq. (151):

        |I_1 - I| <~ eps * ||f||_inf * (h/2) * (1 + G1/G0 + max(G1, k^2)/G0)

    where G0 = min|theta'| and G1 = max|theta'| on the rescaled interval [-1, 1] (see
    _levin_G0_G1()) and h = width is the raw interval width. This supersedes the module's former
    endpoint-phase model (_phase_error() with an inferred theta_scale) as the *default* accuracy
    floor -- see prompt 04's log (prompts/levin-refactor/logs/04-roundoff-floor.md) for the
    measurement showing the old model was optimistic by up to 8.1e9 on exactly the production
    code path, and for why this one is not: it is independent of whether the phase is presented
    raw or range-reduced, needs no evaluations the region's own solve does not already make, and
    (unlike the endpoint model) aggregates to a total that is roughly invariant under how finely
    an interval is subdivided rather than growing with the region count.

    G0 = min|theta'| can be exactly, or numerically, zero at an interior stationary point that
    happens to coincide with (or land extremely close to) a sampled collocation node -- reachable
    (see prompt 04, note (b)) for a Levin region with large total variation but a turning point
    strictly inside it (a region with *small* total variation and a turning point is routed to the
    Clenshaw-Curtis fallback by the total-variation gate before this function is ever called on it
    -- see _adaptive_levin_subregion_impl()).

    This is treated as +inf, not merely as a large finite number, whenever
    G0 <= _LEVIN_ROUNDOFF_G0_NOISE_FLOOR * G1 -- i.e. whenever G0 sits at or below a generous
    relative floating-point noise floor for theta', given its scale G1 elsewhere in the interval.
    See _LEVIN_ROUNDOFF_G0_NOISE_FLOOR's own comment for the constant's value and derivation; the
    short version is that this threshold is not cosmetic. An interior turning point that lands
    close to, but not exactly on, a sampled collocation node still gives a *finite* G0 -- and when
    theta' comes from spectral differentiation of a sampled phase (no theta_deriv supplied), even a
    node placed exactly at a true stationary point produces a small nonzero G0 from differentiation
    noise, not true zero. The naive reading -- "just check G0 > 0" -- was tried and measured to
    fail: on the C2 stationary-phase regression (theta = 1e6*(x - x^2), true stationary point at
    x = 0.5; prompts/levin-refactor/logs/04-roundoff-floor.md) a region with one endpoint exactly
    at the stationary point produced G0 ~ 1.9e-9 and G1 ~ 2.5e5 -- a ratio ~7.5e-15, comfortably
    nonzero by a bare G0 > 0.0 test but pure differentiation noise -- which inflated
    max(G1, k^2)/G0 to a bound comparable to the integral's own value and let it satisfy
    _adaptive_levin()'s phase_limited test, freezing a region whose value was in fact 490% wrong.
    Only the relative test below catches this; G0 > 0 alone does not. Eq. (151) is undefined in
    the true G0 = 0 limit (division by zero) and is not trustworthy in this noise-floor
    neighbourhood of it either, so this returns +inf in both cases rather than raising or
    fabricating a number the derivation does not support. This is deliberately NOT clamped to some
    other large-but-finite value: _adaptive_levin()'s phase_limited test additionally requires a
    *finite* floor before it will accept a region early on the strength of it (see the comment
    there), so an infinite floor here can only ever be reported once the region is accepted on some
    other basis (the step-(4) residual, or depth_max) -- it cannot itself
    freeze a region that genuine further bisection could still improve. See prompt 04's log for the
    full numerical consequence of this choice, including the regression this specific threshold
    fixes.

    :param f_scale: max_grid |f| for this region -- max sqrt(f_1(x)^2 + f_2(x)^2) over the grid
        already sampled to build the region's estimate (see prompt 04, note (c), for why the
        Euclidean combination of the two components was chosen over max_i max_grid|f_i| or their
        sum).
    :param width: |b - a|, the raw interval width. The (h/2) prefactor multiplying the whole
        bound is formed from this directly, rather than being threaded through from the caller's
        own (h/2), so it can never be confused with the (h/2) already folded into G0/G1.
    :param G0: min|theta'| * (h/2) on the rescaled interval; see _levin_G0_G1().
    :param G1: max|theta'| * (h/2) on the rescaled interval; see _levin_G0_G1().
    :param k: Chebyshev collocation order used for this region (note (d): not the region index,
        not a wave number).
    """
    if not (G0 > 0.0) or G0 <= _LEVIN_ROUNDOFF_G0_NOISE_FLOOR * G1:
        return float("inf")

    half_width = 0.5 * width
    k2 = float(k) * float(k)
    return (
        _LEVIN_ROUNDOFF_SAFETY
        * MACHINE_EPSILON
        * f_scale
        * half_width
        * (1.0 + G1 / G0 + max(G1, k2) / G0)
    )


def _declared_endpoint_phase_err(
    BasisData, a: float, b: float, p_endpoint_l1_a: float, p_endpoint_l1_b: float
) -> float:
    """
    Optional endpoint-rounding contribution from a caller-declared theta_abserr (recommendation
    5.2). Returns 0.0 when the phase dict supplies no "theta_abserr" key, so a region's floor is
    exactly the eq. (151) round-off floor and nothing more -- see _Basis_SinCos.theta_abserr_at()
    and prompt 04's log.

    :param a: lower endpoint of the region (x_span[0]).
    :param b: upper endpoint of the region (x_span[1]).
    :param p_endpoint_l1_a: sum over components of |p| (or an analogous proxy for the
        Clenshaw-Curtis fallback, which has no Levin antiderivative) attributed to endpoint a.
    :param p_endpoint_l1_b: as above, attributed to endpoint b.
    """
    err = 0.0

    theta_abserr_a = BasisData.theta_abserr_at(a)
    if theta_abserr_a is not None:
        err += _phase_error(theta_abserr_a, p_endpoint_l1_a)

    theta_abserr_b = BasisData.theta_abserr_at(b)
    if theta_abserr_b is not None:
        err += _phase_error(theta_abserr_b, p_endpoint_l1_b)

    return err


class used_interval:

    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        type: int,
        abserr: Optional[float] = None,
        relerr: Optional[float] = None,
        p_ratios: Optional[List[float]] = None,
        phase_err: Optional[float] = None,
        phase_limited: bool = False,
        abserr_truncation: Optional[float] = None,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._type = type

        # For a Levin region (type == INTERVAL_TYPE_LEVIN), abserr/relerr are the step-(4)
        # *resolution* residual: |val0 - valL - valR|. For a Clenshaw-Curtis fallback region
        # (type == INTERVAL_TYPE_DIRECT) there is no such comparison; abserr/relerr are instead
        # the nested-pair estimate |CC_{2N-1} - CC_N| (see _adaptive_levin_subregion_cc()).
        # phase_err is the round-off floor from Chen et al. eq. (151) (_roundoff_floor()), plus
        # any declared-theta_abserr endpoint contribution (_declared_endpoint_phase_err()) -- an
        # error source neither the resolution residual nor the nested-pair estimate can see (see
        # _adaptive_levin()). abserr_truncation (prompt 06, rec 9) is a third, independent source:
        # the endpoint contribution of p-modes the rtol gate in _adaptive_levin_subregion_impl()
        # dropped from this region's own value -- a deliberate truncation, not a resolution
        # residual or a round-off floor, and None for a Clenshaw-Curtis fallback region (which has
        # no Levin antiderivative to gate). The total error of the region is the resolution/
        # round-off max, plus the truncation term; see abserr_resolution/abserr_fallback/
        # abserr_roundoff/abserr_truncation below for the type-disambiguated accessors, and
        # total_err for the combination.
        self._abserr = abserr
        self._relerr = relerr

        self._phase_err = phase_err
        self._phase_limited = phase_limited

        self._abserr_truncation = abserr_truncation

        self._p_ratios = p_ratios if p_ratios is not None else []

    def __str__(self):
        if self._abserr is not None:
            abserr_label = f"abserr={self._abserr:.8g}"
        else:
            abserr_label = "abserr=(not recorded)"

        if self._relerr is not None:
            relerr_label = f"relerr={self._relerr:.8g}"
        else:
            relerr_label = "relerr=(not recorded)"

        if self._p_ratios is not None:
            p_ratios_label = f"[{", ".join(f"{p:.4g}" for p in self._p_ratios)}]"
        else:
            p_ratios_label = "(not recorded)"

        if self._phase_err is not None:
            phase_label = f"phase_err={self._phase_err:.8g}"
            if self._phase_limited:
                phase_label = phase_label + " (PHASE LIMITED)"
        else:
            phase_label = "phase_err=(not recorded)"

        return f"({self._start:.8g}, {self._end:.8g}), depth={self._depth} | {types[self._type]}, {abserr_label}, {relerr_label}, {phase_label} | p-ratios={p_ratios_label}"

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def type(self) -> str:
        return types[self._type]

    @property
    def abserr(self) -> float:
        return self._abserr

    @property
    def relerr(self) -> float:
        return self._relerr

    @property
    def phase_err(self) -> Optional[float]:
        """
        Round-off floor for this region (Chen et al. eq. (151), _roundoff_floor()), plus any
        declared-theta_abserr endpoint contribution. This is *not* included in abserr; the total
        error of the region is max(abserr, phase_err) -- see total_err. Can be +inf; see
        _roundoff_floor()'s docstring for when and why.
        """
        return self._phase_err

    @property
    def phase_limited(self) -> bool:
        """
        True if this region was accepted because the round-off floor, not lack of resolution, set
        the achievable accuracy. Subdividing such a region cannot improve it. Never True when
        phase_err is not finite -- see _adaptive_levin()'s phase_limited computation.
        """
        return self._phase_limited

    @property
    def total_err(self) -> Optional[float]:
        """
        Combined per-region error estimate: max(resolution/fallback residual, round-off floor),
        plus the truncation term (prompt 06) if this region has one. The truncation term is added
        rather than maxed because it is a distinct, always-present bias -- the endpoint
        contribution of modes the rtol gate discarded from this region's own value -- not another
        estimate of the same underlying quantity the max() is choosing between.
        """
        if self._abserr is None:
            base = self._phase_err
        elif self._phase_err is None:
            base = self._abserr
        else:
            base = max(self._abserr, self._phase_err)

        if base is None:
            return self._abserr_truncation
        if self._abserr_truncation is None:
            return base
        return base + self._abserr_truncation

    @property
    def abserr_resolution(self) -> Optional[float]:
        """
        Step-(4) resolution residual |val0 - valL - valR|. Defined only for a Levin region (type
        == "Levin"); None for a Clenshaw-Curtis fallback region, whose own nested-pair estimate
        is abserr_fallback instead. See prompts/levin-refactor/04-roundoff-floor.md item 5.
        """
        if self._type == INTERVAL_TYPE_LEVIN:
            return self._abserr
        return None

    @property
    def abserr_fallback(self) -> Optional[float]:
        """
        Clenshaw-Curtis nested-pair estimate |CC_{2N-1} - CC_N|. Defined only for a fallback
        region (type == "direct"); None for a Levin region, whose own resolution residual is
        abserr_resolution instead.
        """
        if self._type == INTERVAL_TYPE_DIRECT:
            return self._abserr
        return None

    @property
    def abserr_roundoff(self) -> Optional[float]:
        """
        Round-off floor (Chen et al. eq. (151), plus any declared-theta_abserr endpoint term).
        Defined for every region type -- an alias for phase_err, named to match
        abserr_resolution/abserr_fallback for the aggregate breakdown in _adaptive_levin()'s
        returned dict.
        """
        return self._phase_err

    @property
    def abserr_truncation(self) -> Optional[float]:
        """
        Endpoint contribution of p-modes dropped by the rtol gate in
        _adaptive_levin_subregion_impl() (prompt 06, rec 9): sum over discarded components i of
        (|p_i(a)| + |p_i(b)|), an upper bound on what those modes would have contributed to
        lower_limit/upper_limit (the sin/cos weights multiplying them have magnitude <= 1). None
        for a Clenshaw-Curtis fallback region (type == "direct"), which has no Levin antiderivative
        to gate.
        """
        return self._abserr_truncation

    @property
    def p_ratios(self) -> List[float]:
        return self._p_ratios


ErrorHistoryType = dict[int, float]
PScaleHistoryType = dict[int, List[float]]


class _levin_interval:
    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        abserr_history: Optional[ErrorHistoryType] = None,
        relerr_history: Optional[ErrorHistoryType] = None,
        p_ratios_history: Optional[PScaleHistoryType] = None,
        estimate: Optional[dict] = None,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._abserr_history = abserr_history if abserr_history is not None else {}
        self._relerr_history = relerr_history if relerr_history is not None else {}

        self._p_ratios_history = (
            p_ratios_history if p_ratios_history is not None else {}
        )

        # Levin estimate for this interval, if it has already been computed. When a region is bisected
        # its two halves have already been evaluated in order to produce the refined estimate, so we
        # carry those results forward rather than recomputing them when the children are popped.
        self._estimate = estimate

    @property
    def estimate(self) -> Optional[dict]:
        return self._estimate

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def break_point(self) -> float:
        return (self._end + self._start) / 2.0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def abserr_history(self) -> ErrorHistoryType:
        return self._abserr_history

    @property
    def relerr_history(self) -> ErrorHistoryType:
        return self._relerr_history

    @property
    def p_ratios_history(self) -> PScaleHistoryType:
        return self._p_ratios_history


@lru_cache(maxsize=32)
def _chebyshev_base(N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix, on the reference interval [-1, 1].

    Only the affine rescaling to an arbitrary interval depends on the endpoints, so this part of the
    construction is cached: the adaptive driver evaluates many subregions at the same Chebyshev order,
    and rebuilding these matrices each time is a significant fraction of the cost of the linear solve
    they feed.

    The returned arrays are marked read-only and shared between callers. chebyshev_matrices() below
    only ever forms new arrays from them, so this is safe; do not mutate them in place.

    Based on the implementation by Wiedeman & Reddy (http://dx.doi.org/10.1145/365723.365727)
    and pyddx (https://github.com/ronojoy/pyddx)
    :param N:
    :return:
    """
    n1 = floor(N / 2)
    n2 = ceil(N / 2)

    k = np.arange(N)
    theta = k * np.pi / (N - 1)

    # Compute the Chebyshev collocation points
    # x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way
    x = np.sin(
        np.pi * ((N - 1.0) - 2.0 * np.linspace(N - 1.0, 0, N)) / (2.0 * (N - 1.0))
    )  # W&R way
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(theta / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)  # matrix with entries c(k)/c(j)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    # set up differentiation matrix on [-1, 1]
    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)  # off-diagonals
    D[range(N), range(N)] = -np.sum(D, axis=1)  # negative sum trick

    x.setflags(write=False)
    D.setflags(write=False)

    return x, D


def chebyshev_matrices(x_span: Tuple[float, float], N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix, rescaled to the interval x_span.

    Note the returned grid is in *descending* order: x[0] is the upper endpoint of x_span and x[-1]
    is the lower endpoint. The Levin endpoint extraction depends on this convention.
    :param x_span:
    :param N:
    :return:
    """
    x_base, D_base = _chebyshev_base(N)

    # rescale for the arbitrary interval. Both operations below form new arrays, so the cached
    # read-only base matrices are left untouched.
    a, b = x_span
    x = (a + b) / 2.0 + (b - a) / 2.0 * x_base
    D = 2.0 * D_base / (b - a)

    return x, D


@lru_cache(maxsize=32)
def _cc_weights_base(N: int):
    """
    Clenshaw-Curtis quadrature weights for the extremal Chebyshev grid returned by
    _chebyshev_base(N) -- i.e. for the nodes x_k = cos(k*pi/(N-1)), k = 0, ..., N-1 on
    [-1, 1], in that order (x_0 = +1, x_{N-1} = -1). This is the standard closed-form
    construction (Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
    quadrature rules", BIT 46 (2006) 195-202; equivalent to the "clencurt" algorithm of
    Trefethen, Spectral Methods in MATLAB, ch. 12), specialised to real weights via the
    cosine-sum form rather than the FFT, since N here is at most a few dozen.

    Verified in AdaptiveLevin/tests/test_levin_quadrature.py by exact integration of
    polynomials of degree < N and against a known transcendental integral -- see
    prompt 03's log (prompts/levin-refactor/logs/03-total-variation-gate.md) for the
    numbers.

    Cached like _chebyshev_base(), for the same reason: the driver evaluates many
    fallback regions at the same order. The returned array is read-only and shared
    between callers; form new arrays from it rather than mutating in place.

    :param N: number of points (N >= 2).
    """
    n = N - 1  # number of intervals

    theta = np.pi * np.arange(N) / n

    w = np.zeros(N)
    if n % 2 == 0:
        w[0] = 1.0 / (n * n - 1)
    else:
        w[0] = 1.0 / (n * n)
    w[-1] = w[0]

    if n > 1:
        ii = np.arange(1, n)
        v = np.ones(n - 1)
        if n % 2 == 0:
            for k in range(1, n // 2):
                v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
            v -= np.cos(n * theta[ii]) / (n * n - 1)
        else:
            for k in range(1, (n - 1) // 2 + 1):
                v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        w[ii] = 2.0 * v / n

    w.setflags(write=False)
    return w


def _cc_weights(x_span: Tuple[float, float], N: int):
    """
    Clenshaw-Curtis weights for chebyshev_matrices(x_span, N)'s grid, i.e. _cc_weights_base(N)
    rescaled from [-1, 1] to x_span. Cheap (one array multiply of a cached base array), so not
    itself cached.
    """
    a, b = x_span
    return _cc_weights_base(N) * (np.fabs(b - a) / 2.0)


def _detect_vectorized(func, grid: np.ndarray) -> bool:
    """
    Probe once whether `func` accepts a whole grid of points as a single array argument and
    returns an array of the same shape whose entries agree exactly with calling func pointwise
    (recommendation 14; audit sec 4.1/4.3 -- see _sample_vectorized()).

    Two probe points, not one: a callable that silently broadcasts a scalar result across an
    array argument instead of evaluating pointwise (e.g. a constant amplitude `lambda x: 1.0`,
    which several of this module's own tests use) would pass a one-point check trivially. Using
    two distinct points and requiring exact agreement with the elementwise loop catches that
    failure mode -- the one the prior review's Sec 7.6 warns "produces silent garbage" -- rather
    than trusting the shape alone.

    Exceptions from either the scalar or the array call (e.g. math.atan, or any other function
    that assumes a scalar argument and raises when handed an array) are treated as "does not
    vectorize", not propagated: this is a capability probe, not a correctness check on `func`
    itself.

    :param grid: the actual sampling grid this call would use; only its first two points (or
        fewer, if the grid is shorter) are evaluated.
    """
    probe = grid[:2] if grid.size >= 2 else grid
    try:
        scalar_values = np.array([func(x) for x in probe], dtype=float)
    except Exception:
        return False

    try:
        vector_values = np.asarray(func(probe))
    except Exception:
        return False

    if vector_values.shape != probe.shape:
        return False
    if not np.issubdtype(vector_values.dtype, np.number):
        return False
    if not np.isfinite(vector_values).all() or not np.isfinite(scalar_values).all():
        return False

    return bool(np.array_equal(vector_values, scalar_values))


def _sample_vectorized(func, grid: np.ndarray, cache: dict, key) -> np.ndarray:
    """
    Sample `func` at every point of `grid`, using one array call in place of the Python loop
    `np.array([func(x) for x in grid])` whenever `func` supports it (recommendation 14).

    Audit sec 4.1/4.3: this loop -- for the amplitude f and, separately, for the phase
    derivative theta' -- is 20-37% of a subregion evaluation at the default order, more than the
    linear solve, and drops from 11.7 us to 2.5 us at N=12 (18x at N=64) when the callable
    accepts an array. Whether it helps in production depends entirely on whether the real
    callables vectorize -- see prompt 08's log (prompts/levin-refactor/logs/08-order-and-
    sampling.md) for the measurement on the actual three_bessel_integrals.py /
    QuadSourceIntegral.py callables (they do not, as of this commit: their phase and modulus
    splines branch on a scalar argument).

    Detection is by one-time probing (_detect_vectorized()), chosen over an opt-in flag (prior
    review Sec 7.6) so that a caller whose callables happen to vectorize gets the speedup for
    free. `cache` is created once per top-level adaptive_levin_sincos() call (in
    _adaptive_levin()) and threaded down through every subregion solve, so each distinct
    callable is probed at most once per run, not once per subregion -- the cost of the probe
    (one array call plus two scalar calls) is paid at most len(f) + 2 times per run, not
    len(f) + 2 times per region.

    :param cache: a dict, shared across every subregion of one run, mapping `key` to the
        previously-detected bool. Mutated in place.
    :param key: a hashable identifying `func` for the lifetime of `cache` -- typically
        (tag, id(func)). Using id() is safe here only because `cache` (and everything it keys)
        is discarded at the end of the run that created it, so an id cannot be reused by an
        unrelated object while the cache is still live.
    """
    vectorizes = cache.get(key)
    if vectorizes is None:
        vectorizes = _detect_vectorized(func, grid)
        cache[key] = vectorizes

    if vectorizes:
        result = np.asarray(func(grid), dtype=float)
        if result.shape == grid.shape:
            return result
        # The grid size changed since detection (e.g. a stepped-down chebyshev_order retry) and
        # this callable's apparent vectorisation turns out to be shape-dependent after all --
        # a genuinely vectorizing callable (np.sin and friends) matches any input shape, so this
        # is itself evidence of a broadcasting bug rather than a fluke. Stop trusting it for the
        # rest of this run and fall back to the loop, for this call and every later one.
        cache[key] = False

    return np.array([func(x) for x in grid], dtype=float)


class _Basis_SinCos:
    def __init__(self, theta):
        """
        :param theta: dict with a required "theta" key (the phase function) and optional
            "theta_mod_2pi" / "theta_deriv" / "theta_abserr" keys; see
            adaptive_levin_sincos()'s docstring for the full contract.
        """
        if callable(theta):
            raise TypeError(
                "levin_quadrature: theta must be a dict with a 'theta' key mapping to the phase "
                "function, not a bare callable -- did you mean theta={'theta': theta}?"
            )
        if "theta" not in theta:
            raise RuntimeError(
                "levin_quadrature: Levin phase function theta must be provided as theta['theta']"
            )

        self._theta = theta["theta"]

        if "theta_mod_2pi" in theta:
            self._theta_mod_2pi = theta["theta_mod_2pi"]

        if "theta_deriv" in theta:
            self._theta_deriv = theta["theta_deriv"]

        if "theta_abserr" in theta:
            self._theta_abserr = theta["theta_abserr"]

    def raw_theta(self, x):
        return self._theta(x)

    def theta_abserr_at(self, x) -> Optional[float]:
        """
        Declared absolute error of the phase at x, in radians (recommendation 5.2), or None when
        the phase dict supplied no "theta_abserr" key. theta_abserr may be a scalar (the same
        declared error everywhere) or a callable of x; follows the same hasattr convention as
        theta_mod_2pi/theta_deriv above rather than an explicit-None default (standing note 11 of
        prompts/levin-refactor/IMPLEMENTATION_STATE.md).

        Nothing in LiouvilleGreen/ supplies this today -- see prompt 04's log and README Sec 6 of
        the levin-refactor campaign. It exists so a future phase_spline accuracy API has somewhere
        to report its own construction error, which this module cannot otherwise see or express.
        """
        if not hasattr(self, "_theta_abserr"):
            return None
        theta_abserr = self._theta_abserr
        return float(theta_abserr(x)) if callable(theta_abserr) else float(theta_abserr)

    @property
    def supports_complexified_solve(self) -> bool:
        """
        True if this basis's Levin collocation system can be solved in the complexified N x N
        form (D + i diag(theta')) q = f1 + i f2 (Chen et al. (168)) instead of the realified
        2N x 2N one. This holds for the two-component (sin, cos) basis implemented here; see the
        module docstring and prompt 02's log for the derivation. A basis with a different number
        of components, or a genuinely different structure, would answer False here rather than
        the driver sniffing isinstance(BasisData, _Basis_SinCos).
        """
        return True

    def build_Levin_data(
        self,
        grid,
        Dmat,
        notify_label: Optional[str] = None,
        id_label=None,
        need_AmatT: bool = True,
        vectorize_cache: Optional[dict] = None,
    ):
        """
        Sample the phase function once and build everything derived from that sample: theta',
        the A^T super-operator block (unless need_AmatT is False), the basis vector w evaluated
        at each endpoint, and an estimate of the total phase change across the interval.

        :param notify_label: caller-supplied label for this run, used only (with id_label) to
            annotate a non-finite-phase-derivative warning -- see _format_label().
        :param id_label: caller-supplied run id (typically a _LazyUUID), likewise used only for
            that warning.
        :param need_AmatT: whether to build and return the A^T block. The complexified solve
            path (see supports_complexified_solve) needs theta_prime_Cheb directly and never
            needs A^T, so the caller passes False there to skip an O(N^2) allocation that would
            just be discarded.
        :param vectorize_cache: dict shared across every subregion of the current
            adaptive_levin_sincos() call, passed straight to _sample_vectorized() to decide
            whether theta/theta' can be sampled with one array call instead of a Python loop
            (recommendation 14). None (the default, used by any direct caller that does not
            thread one through) is equivalent to a fresh, un-shared cache -- correct, just
            without the cross-region reuse the driver otherwise gets.
        :return: (AmatT, theta_prime_Cheb, w0, wk, phase_span). AmatT is None when need_AmatT is
            False. Prior to prompt 04 this also returned a sixth element, theta_scale, used only
            by the endpoint phase-rounding model that eq. (151) (_roundoff_floor()) replaces as
            the default accuracy floor -- see prompt 04's log for why that model, and the
            inference it required, were removed.
        """
        if vectorize_cache is None:
            vectorize_cache = {}

        # we need theta sampled on the Chebyshev grid if either (a) we have to obtain theta' by spectral
        # differentiation, or (b) we have no range-reduced phase function and therefore have to evaluate
        # the basis functions from the raw phase at the endpoints.
        # These are independent conditions, so both have to be tested; testing only the first leaves
        # theta_Cheb undefined in the endpoint block below.
        need_theta_Cheb = not hasattr(self, "_theta_deriv") or not hasattr(
            self, "_theta_mod_2pi"
        )

        theta_Cheb = None
        if need_theta_Cheb:
            # sample the phase function theta on the Chebyshev grid
            theta_Cheb = _sample_vectorized(
                self._theta, grid, vectorize_cache, ("theta", id(self))
            )

        if hasattr(self, "_theta_deriv"):
            # prefer an explicitly supplied theta'. At large argument the raw phase theta is a large float
            # whose absolute resolution is ~ eps*theta, so spectral differentiation of the sampled values
            # below inherits an error ~ eps*theta/(phase span across the interval). A phase function that
            # can supply theta' directly (e.g. from a range-reduced spline) does not suffer from this.
            theta_prime_Cheb = _sample_vectorized(
                self._theta_deriv, grid, vectorize_cache, ("theta_deriv", id(self))
            )
        else:
            # multiply theta by the spectral differentiation matrix Dmat in order to produce an estimate of theta'(x)
            # evaluated at the collocation points
            theta_prime_Cheb = np.matmul(Dmat, theta_Cheb)

        if not np.isfinite(theta_prime_Cheb).all():
            _logger.warning(
                "!! WARNING (adaptive_levin_subregion, %s): sampled phase derivative theta' "
                "contains non-numeric values (np.nan, np.inf, or np.-inf)",
                _format_label(notify_label, id_label),
            )
            raise ValueError(
                "sampled phase derivative theta' contains non-numeric values (np.nan, np.inf, or np.-inf)"
            )

        AmatT = None
        if need_AmatT:
            # if w = (sin, cos) (regarded as a column vector), then w' = A w where A is the matrix
            #   Amat = ( 0,       theta' )
            #          ( -theta', 0      )
            # and therefore its transpose is
            #   Amat^T = ( 0,      -theta' )
            #            ( theta', 0       )
            theta_prime_I = np.diag(theta_prime_Cheb)

            zero_block = np.zeros_like(theta_prime_I)
            AmatT = np.block(
                [[zero_block, -theta_prime_I], [theta_prime_I, zero_block]]
            )

        # estimate the total phase change across the interval from quantities we have already computed.
        # This costs no further evaluations of the phase function, and is used to decide whether the
        # Levin super-operator is well enough conditioned to be inverted by a direct solve.
        phase_span = float(
            np.mean(np.fabs(theta_prime_Cheb)) * np.fabs(grid[0] - grid[-1])
        )

        # w0/wk are w = (sin theta, cos theta) at the two region endpoints, preferring a
        # range-reduced phase when one is supplied (an O(2*pi) argument to sin/cos rather than an
        # O(theta) one). This is still worth doing for its own sake -- it keeps the *values*
        # handed to sin/cos well-conditioned -- but, as of prompt 04, no longer changes the
        # reported accuracy floor: eq. (151) (_roundoff_floor()) does not depend on how the phase
        # is presented, only on f, theta' and the interval geometry. See prompt 04's log (C4) for
        # the measurement showing the previous model's dependence on this branch was the defect,
        # not a feature -- the reduced-phase floor was optimistic by up to 8.1e9 on the production
        # path while the raw-phase floor and the reduced-phase *value* were both fine.
        if hasattr(self, "_theta_mod_2pi"):
            theta0_mod_2pi = self._theta_mod_2pi(grid[-1])
            thetak_mod_2pi = self._theta_mod_2pi(grid[0])

            w0 = [np.sin(theta0_mod_2pi), np.cos(theta0_mod_2pi)]
            wk = [np.sin(thetak_mod_2pi), np.cos(thetak_mod_2pi)]
        else:
            # note grid is in reverse order, with largest value in position 1 and smallest value in last position -1
            theta0 = theta_Cheb[-1]
            thetak = theta_Cheb[0]

            w0 = [np.sin(theta0), np.cos(theta0)]
            wk = [np.sin(thetak), np.cos(thetak)]

        return AmatT, theta_prime_Cheb, w0, wk, phase_span

    def eval_basis(self, x):
        if hasattr(self, "_theta_mod_2pi"):
            return [np.sin(self._theta_mod_2pi(x)), np.cos(self._theta_mod_2pi(x))]

        return [np.sin(self._theta(x)), np.cos(self._theta(x))]


def _adaptive_levin_subregion(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    notify_label: Optional[str] = None,
    build_p_sample: bool = False,
    diagnostics_path: Path = DEFAULT_LEVIN_DIAGNOSTICS_PATH,
    vectorize_cache: Optional[dict] = None,
):
    working_order = max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER)
    num_order_changes = 0

    # to handle possible SVD failures, allow the working Chebyshev order to be stepped down.
    # this changes the matrices that we need to invert, so gives another change for the required SVD to converge
    #
    # NOTE the order actually reported below (working_order, after the loop) is always the order
    # that *succeeded* -- the loop's failure branch below only decrements working_order and
    # `continue`s to try again; if that retry (or every subsequent one down to the floor) also
    # fails, "if data['value'] is None: raise" below fires before the metadata update that reports
    # chebyshev_order, so a total failure never returns a chebyshev_order that was never attempted.
    # Confirmed with a forced-total-failure regression test (07's log).
    finished = False
    while not finished and working_order >= _LEVIN_MINIMUM_ALLOWED_ORDER:
        data = _adaptive_levin_subregion_impl(
            x_span,
            f,
            BasisData,
            id_label,
            chebyshev_order=working_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=build_p_sample,
            diagnostics_path=diagnostics_path,
            vectorize_cache=vectorize_cache,
        )
        if data["metadata"].get("SVD_failure", False):
            working_order = working_order - 2
            num_order_changes = num_order_changes + 1

            _logger.warning(
                "!! WARNING (adaptive_levin_subregion, %s): SVD failure - stepping down "
                "Chebyshev order to %d",
                _format_label(notify_label, id_label),
                working_order,
            )
            continue

        finished = True

    if data["value"] is None:
        raise LinAlgError("SVD failure")

    data["metadata"]["SVD_failure"] = data["metadata"].get("SVD_failure", False)
    data["metadata"].update(
        {
            "num_order_changes": num_order_changes,
            "chebyshev_order": working_order,
        }
    )

    return data


def _adaptive_levin_subregion_impl(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    notify_label: Optional[str] = None,
    build_p_sample: bool = False,
    diagnostics_path: Path = DEFAULT_LEVIN_DIAGNOSTICS_PATH,
    vectorize_cache: Optional[dict] = None,
):
    """
    f should be an m-vector of non-rapidly oscillating functions (Levin 96 eq. 2.1)
    theta should be an (m x m)-matrix representing the phase matrix of the system (Levin 96's A or A^t matrix)
    :param rtol:
    :param x_span:
    :param f: iterable of callables representing the integrand f-functions
    :param BasisData: callable
    :param chebyshev_order:
    :param build_p_sample: retain the sampled Levin antiderivatives p(x)? Only needed for diagnostics
    :param diagnostics_path: directory the lstsq-failure dump (LevinL_*.txt / f_Cheb_*.txt) is
        written under, on the rare path where lstsq itself raises. Not used otherwise.
    :param vectorize_cache: dict shared across every subregion of the current
        adaptive_levin_sincos() call (recommendation 14) -- see _sample_vectorized(). None
        (the default) is safe for a direct caller that does not thread one through; it just
        means each of this call's f-components is probed for vectorisation on its own rather
        than sharing a probe with sibling subregions.
    :return:
    """
    if vectorize_cache is None:
        vectorize_cache = {}

    # This function is on the hottest path in the module -- called once per popped region, plus
    # twice more for its comparison children -- so, unlike the wrapper above, it does not
    # precompute a "label" string here: that would stringify id_label (see _LazyUUID) on every
    # call regardless of whether a message is ever emitted. notify_label/id_label are passed
    # through unchanged to whichever branch below actually needs to log something, and
    # _format_label() builds the string only there.
    metadata = {}

    m = len(f)
    grid, Dmat = chebyshev_matrices(x_span, chebyshev_order)

    # Sample the phase once, before deciding anything else. build_Levin_data returns
    # phase_span = mean|theta'| * width, which is the total variation of the phase across this
    # region and costs no evaluations beyond theta' itself -- so it is available *before* we
    # decide whether to solve the Levin system at all, and before sampling f.
    #
    # A basis reports whether its Levin system can be solved in the complexified N x N form
    # (D + i diag(theta')) q = f1 + i f2 -- Chen et al. (168) -- instead of the realified 2N x 2N
    # one. This is asked of the basis object rather than sniffed via isinstance, so a future basis
    # with a different structure or component count simply answers False. It is also gated on
    # m == 2: the complex form only exists for a two-component (sin, cos) basis. This decision is
    # independent of phase_span, so it is safe to make before the gate below.
    use_complex_solve = m == 2 and getattr(
        BasisData, "supports_complexified_solve", False
    )

    AmatT, theta_prime_Cheb, w0, wk, phase_span = BasisData.build_Levin_data(
        grid,
        Dmat,
        notify_label=notify_label,
        id_label=id_label,
        need_AmatT=not use_complex_solve,
        vectorize_cache=vectorize_cache,
    )

    # C2 fix: gate on the phase's *total variation* across the region, not its net change
    # (theta(b) - theta(a)) -- the two differ whenever theta' changes sign inside the region, and
    # the net change can be accidentally small at a stationary point while the phase still
    # oscillates heavily either side of it. See the module docstring and prompt 03's log.
    #
    # This also means the fallback is chosen *before* f is sampled at all: a region routed to the
    # nested Clenshaw-Curtis rule below samples f on its own (2*chebyshev_order - 1)-point grid,
    # never on this one, so there is no wasted evaluation of f for a region that turns out not to
    # need the Levin solve.
    if phase_span < SIX_PI:
        return _adaptive_levin_subregion_cc(
            x_span,
            f,
            BasisData,
            chebyshev_order,
            phase_span,
            theta_prime_Cheb,
            notify_label,
            id_label,
            vectorize_cache=vectorize_cache,
        )

    # sample each component of f on the Chebyshev grid,
    # then assemble the result into a flattened vector in an m x k representation
    # Chen et al. around (166), (167)
    f_Cheb = np.hstack(
        [
            _sample_vectorized(func, grid, vectorize_cache, ("f", id(func)))
            for func in f
        ]
    )

    if not np.isfinite(f_Cheb).all():
        _logger.warning(
            "!! WARNING (adaptive_levin_subregion, %s): sampled amplitude f contains "
            "non-numeric values (np.nan, np.inf, or np.-inf)",
            _format_label(notify_label, id_label),
        )
        raise ValueError(
            "sampled amplitude f contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    if use_complex_solve:
        # Chen et al. (168) in complexified form: q = p1 + i*p2 solves (D + i diag(theta')) q =
        # f1 + i*f2, where f1/f2 are the sampled amplitudes for the sin/cos slots respectively
        # (f_Cheb is [f1(grid); f2(grid)] by construction above). D.astype(complex) always
        # allocates a fresh array (astype copies by default), so mutating its diagonal in place
        # cannot corrupt the shared, read-only base matrices from _chebyshev_base(), nor Dmat
        # itself -- see chebyshev_matrices()'s own note that it forms a new D on every call.
        LevinL = Dmat.astype(complex)
        LevinL[np.diag_indices(chebyshev_order)] += 1j * theta_prime_Cheb
        rhs = f_Cheb[:chebyshev_order] + 1j * f_Cheb[chebyshev_order:]
    else:
        # build the Levin superoperator corresponding to this system
        # Chen et al. (168), realified
        zero_block = np.zeros((chebyshev_order, chebyshev_order))

        row_list = []
        for i in range(m):
            row = [zero_block for _ in range(m)]
            row[i] = Dmat
            row_list.append(row)

        LevinL = np.block(row_list) + AmatT
        rhs = f_Cheb

    if not np.isfinite(LevinL).all():
        _logger.warning(
            "!! WARNING (adaptive_levin_subregion, %s): Levin super-operator contains "
            "non-numeric values (np.nan, np.inf, or np.-inf)",
            _format_label(notify_label, id_label),
        )

        raise ValueError(
            "Levin super-operator contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # now try to invert the Levin superoperator, to find the Levin antiderivatives p(x) (or, on
    # the complex path, q = p1 + i*p2)
    # Chen et al.
    success = False

    # Fast path. As theta' -> 0 the Levin super-operator degenerates to a block-diagonal matrix of
    # spectral differentiation matrices, and Dmat is singular (it annihilates constants). So the system
    # is badly conditioned on weakly oscillatory intervals, and there we need the minimum-norm solution
    # that lstsq provides. Once the interval carries enough phase the operator becomes well conditioned
    # (empirically cond ~ 4e2 at a phase span of 20*pi, and ~13 at 100*pi), and an ordinary LU solve
    # agrees with lstsq to machine precision while being an order of magnitude cheaper. The complex
    # N x N system has identical conditioning to the realified 2N x 2N one (verified: agrees to 4
    # significant figures at every order tested), so the same gate applies unchanged.
    #
    # NOTE the gate has to be the phase span, *not* the residual of the solve. In the near-singular
    # regime the residual is small (~1e-14) even when the computed p is completely wrong, so a residual
    # test alone would silently accept garbage.
    #
    # A prior version of this branch also checked ||Lq - rhs|| / ||rhs|| against a fixed tolerance
    # before accepting the direct solve, as a secondary guard against outright failure -- but the
    # phase-span gate above is what actually protects against a wrong-but-finite answer, and the
    # residual check cost 58-72% of the solve it guarded. Measured across the full AdaptiveLevin
    # test suite plus the J000 three-Bessel oracle (221 direct-solve calls), the relative residual
    # never exceeded ~5e-15 against a 1e-10 threshold, and every call the residual check would have
    # accepted, the finiteness check below also accepted, and vice versa -- so it was removed and
    # the finiteness check on the solved vector (below) is relied on instead. See prompt 02's log
    # for the measurement.
    if phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN:
        try:
            sol_direct = np.linalg.solve(LevinL, rhs)
        except LinAlgError:
            pass
        else:
            if np.isfinite(sol_direct).all():
                sol = sol_direct
                success = True
                metadata["direct_solve"] = 1

    # otherwise fall back to the least-squares (SVD) solution, which handles the ill-conditioned case
    #
    # NOTE on the choice of solver. Chen et al. (arXiv:2211.13400) solve the collocation system
    # (168) with a truncated SVD, but Remark 2 of that paper records that their own implementation
    # replaces it with a *rank-revealing QR* factorization, which they report to be roughly 5x
    # faster with no observed loss of accuracy. That substitution has deliberately NOT been made
    # here: scipy exposes RRQR only via pivoted QR (scipy.linalg.qr(pivoting=True)) with manual
    # rank determination and a triangular solve, so it is a non-trivial amount of new numerical
    # code to own, and the direct-solve fast path above already removes the SVD from the
    # well-conditioned majority of regions. Swapping lstsq for RRQR in this fallback branch
    # remains an unexplored optimisation; it would need A/B benchmarking against the closed-form
    # Bessel oracles before being trusted, because this branch is exactly the ill-conditioned
    # regime where the minimum-norm property of lstsq is doing real work.
    #
    # rcond=None truncates at eps * max(M, N) * sigma_1 (the paper's own step-5 truncation). On the
    # complex path M == N == chebyshev_order rather than 2*chebyshev_order, halving that threshold;
    # this is a real behavioural difference on the ill-conditioned branch but was confirmed
    # immaterial in practice -- see prompt 02's log for the measurement.
    if not success:
        try:
            sol, residuals, rank, s = np.linalg.lstsq(LevinL, rhs, rcond=None)
        except LinAlgError as e:
            label = _format_label(notify_label, id_label)
            _logger.warning(
                "!! WARNING (adaptive_levin_subregion, %s): could not solve Levin collocation "
                "system using numpy.linalg.lstsq (chebyshev_order=%d; will now attempt to use "
                "pseudo-inverse)",
                label,
                chebyshev_order,
            )
            # Parallel-safety: two workers failing in the same second must not collide, so the
            # run's id_label (unique per top-level adaptive_levin_sincos() call) is part of the
            # filename, not just the directory -- see prompts/levin-refactor/logs/
            # 07-diagnostics-hygiene.md. diagnostics_path is resolved once by _adaptive_levin()
            # (never left as None below this point).
            now = datetime.now().replace(microsecond=0)
            failure_dir = diagnostics_path / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            LevinL_filename = failure_dir / f"LevinL_{id_label}_{now.isoformat()}.txt"
            f_Cheb_filename = failure_dir / f"f_Cheb_{id_label}_{now.isoformat()}.txt"
            _logger.warning(
                '   -- Levin L super-operator written to file "%s", f_Cheb written to file "%s"',
                LevinL_filename,
                f_Cheb_filename,
            )
            np.savetxt(LevinL_filename, LevinL)
            np.savetxt(f_Cheb_filename, rhs)
            metadata["SVD_errors"] = 1
        else:
            success = True
            metadata["lstsq_solve"] = 1

    if not success:
        try:
            LevinL_inv = np.linalg.pinv(LevinL)
            sol = np.matmul(LevinL_inv, rhs)
        except LinAlgError as e:
            _logger.warning(
                "!! WARNING (adaptive_levin_subregion, %s): could not solve Levin collocation "
                "system using numpy.linalg.pinv (chebyshev_order=%d; final failure at this "
                "order)",
                _format_label(notify_label, id_label),
                chebyshev_order,
            )
            metadata["SVD_failure"] = True
            return {
                "value": None,
                "p_sample": None,
                "p_ratios": None,
                "metadata": metadata,
            }
        else:
            metadata["pinv_solve"] = 1

    if not np.isfinite(sol).all():
        _logger.warning(
            "!! WARNING (adaptive_levin_subregion, %s): solved Levin antiderivative p contains "
            "non-numeric values (np.nan, np.inf, or np.-inf)",
            _format_label(notify_label, id_label),
        )
        raise ValueError(
            "solved Levin antiderivative p contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # P[j, i] is Levin antiderivative component j at collocation point i. On the real path p is
    # stored flattened with component j at collocation point i in position j*chebyshev_order + i,
    # so this reshape recovers P[j, i]. On the complex path q = p1 + i*p2 by construction, so its
    # real/imaginary parts stacked in slot order are exactly the same P[j, i]. Taking the mean over
    # the collocation points in numpy avoids building a Python list of m-element lists on every
    # solve; that list is only needed for diagnostics.
    if use_complex_solve:
        P = np.vstack([sol.real, sol.imag])
    else:
        P = sol.reshape(m, chebyshev_order)

    # p_endpoint[i] = |p_i(a)| + |p_i(b)| -- note grid is in reverse order, with P[i, 0] at the
    # upper endpoint b and P[i, -1] at the lower endpoint a. This is the quantity the endpoint sum
    # below (lower_limit/upper_limit) actually consumes. Prompt 06 (problem (a)) gates on this
    # instead of the collocation-point mean |P[j, :]| the original heuristic (commit af85ef2)
    # used: a mode with a small mean and a large endpoint value could be discarded wrongly under
    # the old gate. The audit's own 400-problem randomised sweep found no case where this changed
    # a result, but verifying this prompt did find one, in this campaign's own standard problem
    # set (grz_1000, a mode whose mean-based and endpoint-based ratios sit on opposite sides of
    # rtol) -- see prompt 06's log for the measurement. Either way this is a fix to what the gate
    # tests, not a report of a field failure this module was producing.
    p_endpoint = np.fabs(P[:, -1]) + np.fabs(P[:, 0])
    p_endpoint_max = p_endpoint.max()
    if p_endpoint_max > 0.0:
        # ratio-to-maximum, not fraction-of-total: this preserves the meaning of the existing
        # rtol threshold and of the p_ratios values recorded in used_interval and printed by its
        # __str__ / the depth-18 diagnostic (prompt 06's log, "normalisation choice").
        p_ratios = [float(pe / p_endpoint_max) for pe in p_endpoint]
    else:
        # every component's endpoint value is identically zero -- reachable when the sampled
        # amplitude underflows to zero on this region. A zero solution genuinely contributes
        # zero, so report an all-ones ratio vector (rather than the nan that pe / 0.0 would
        # produce) so every component is kept by p_use below instead of being discarded by a
        # divide-by-zero artefact.
        p_ratios = [1.0 for _ in p_endpoint]

    p_sample = None
    if build_p_sample:
        p_sample = [(x, [P[j, i] for j in range(m)]) for i, x in enumerate(grid)]

    # don't keep p-modes that have relative amplitude smaller than the requested rtol.
    # Presumably we cannot compute these accurately anyway (especially if they are associated with small singular values that are
    # also not being handled correctly in the singular value decomposition of the Levin operator)
    # so they just behave as a source of numerical noise that pollutes our abserr estimates,
    # and can prevent convergence of the bisection step.
    p_use = [np.isfinite(r) and r > rtol for r in p_ratios]

    # note grid is in reverse order, with P[i, 0] at the upper endpoint b and P[i, -1] at the
    # lower endpoint a. Indexing P directly (rather than a flattened p vector) means this is the
    # same expression regardless of which solve path produced P.
    lower_limit = sum(P[i, -1] * w0[i] if p_use[i] else 0.0 for i in range(m))
    upper_limit = sum(P[i, 0] * wk[i] if p_use[i] else 0.0 for i in range(m))

    # Prompt 06 (problem (b)): the endpoint contribution of a discarded mode used to vanish
    # silently from lower_limit/upper_limit above with no trace in the reported error -- common-
    # mode between a parent and its dataL/dataR comparison children (both apply the same gate),
    # so the step-(4) resolution residual cannot see it either. |w0[i]|, |wk[i]| <= 1 (sin/cos),
    # so summing p_endpoint over the discarded components bounds what they would have added.
    # Measured (prompt 06's log): the value can jump by an amount bounded by rtol as rtol crosses
    # a mode's ratio, with abserr previously unmoved; this term makes that jump visible.
    abserr_truncation = float(
        sum(pe for pe, used in zip(p_endpoint, p_use) if not used)
    )

    # Round-off floor, Chen et al. eq. (151) -- see _roundoff_floor(). G0/G1 use theta' already
    # sampled by build_Levin_data(); f_scale = max_grid sqrt(f_1^2 + f_2^2) reuses f_Cheb, which
    # was already sampled and finiteness-checked above (F[j, i] recovers the same per-component
    # layout as P[j, i], by the same construction -- see the comment above P's reshape). Neither
    # costs an extra evaluation of anything.
    width = np.fabs(x_span[1] - x_span[0])
    G0, G1 = _levin_G0_G1(theta_prime_Cheb, width)
    F = f_Cheb.reshape(m, chebyshev_order)
    f_scale = float(np.max(np.sqrt(np.sum(F * F, axis=0))))
    round_off_err = _roundoff_floor(f_scale, width, G0, G1, chebyshev_order)

    # Optional declared-theta_abserr endpoint term (recommendation 5.2), 0.0 unless the phase dict
    # supplied "theta_abserr". Uses the same p-values and the same p_use gating as the estimate
    # itself, split by endpoint since a callable theta_abserr may differ at a and b.
    p_endpoint_l1_a = sum(np.fabs(P[i, -1]) if p_use[i] else 0.0 for i in range(m))
    p_endpoint_l1_b = sum(np.fabs(P[i, 0]) if p_use[i] else 0.0 for i in range(m))
    declared_err = _declared_endpoint_phase_err(
        BasisData, x_span[0], x_span[1], p_endpoint_l1_a, p_endpoint_l1_b
    )

    return {
        "value": upper_limit - lower_limit,
        "p_sample": p_sample,
        "p_ratios": p_ratios,
        "phase_span": phase_span,
        "phase_err": round_off_err + declared_err,
        "abserr_truncation": abserr_truncation,
        "metadata": metadata,
        "is_direct": False,
    }


def _adaptive_levin_subregion_cc(
    x_span: Tuple[float, float],
    f,
    BasisData,
    chebyshev_order: int,
    phase_span: float,
    theta_prime_Cheb: np.ndarray,
    notify_label: Optional[str],
    id_label,
    vectorize_cache: Optional[dict] = None,
):
    """
    Bounded-cost fallback for a region whose total phase variation (phase_span) falls below
    SIX_PI, replacing the module's former use of scipy.integrate.quad (C2, C7; see the module
    docstring and prompt 03's log).

    The extremal Chebyshev grid the Levin solve uses at order N *is* the Clenshaw-Curtis grid,
    and its N-point grid is exactly every other node of the (2*N - 1)-point grid (nesting holds
    for every N >= 2 -- see _cc_weights_base()). So the integrand is sampled once, on the
    (2*N - 1)-point grid, and both the order-N and order-(2*N - 1) Clenshaw-Curtis rules are
    formed from that one sample: CC_{2N-1} is returned as the region's value, and
    |CC_{2N-1} - CC_N| is returned as its error estimate -- a genuine nested-pair estimate, not a
    resolution proxy. Cost is exactly 2*N - 1 evaluations of each f_i and of the phase, always,
    against scipy.quad's unbounded panel count under a tight global tolerance.

    :param phase_span: total phase variation across x_span, already computed by the caller's
        call to BasisData.build_Levin_data() -- passed in rather than recomputed.
    :param theta_prime_Cheb: theta' sampled at the order-chebyshev_order collocation points,
        likewise already computed by the caller's call to build_Levin_data() -- used for the
        round-off floor's G0/G1 (see _levin_G0_G1()), not recomputed on the fine grid because the
        coarse sample is what the gate above already paid for.
    :return: a dict with the same "value"/"phase_span"/"phase_err"/"metadata" keys as the Levin
        path, "abserr_direct" in place of a p-based estimate, "is_direct": True, and
        "p_sample"/"p_ratios" both None (the Levin antiderivative concept does not apply here).
    """
    if vectorize_cache is None:
        vectorize_cache = {}

    m = len(f)
    fine_order = 2 * chebyshev_order - 1

    # Node reuse: fine_grid[::2] is exactly chebyshev_matrices(x_span, chebyshev_order)[0] (see
    # the nesting test in AdaptiveLevin/tests/test_levin_quadrature.py), so the coarse rule below
    # is formed from a subset of these same samples rather than by evaluating f or the basis a
    # second time.
    fine_grid, _ = chebyshev_matrices(x_span, fine_order)

    f_fine = np.vstack(
        [
            _sample_vectorized(func, fine_grid, vectorize_cache, ("f", id(func)))
            for func in f
        ]
    )

    if not np.isfinite(f_fine).all():
        _logger.warning(
            "!! WARNING (adaptive_levin_subregion, %s): sampled amplitude f contains "
            "non-numeric values (np.nan, np.inf, or np.-inf)",
            _format_label(notify_label, id_label),
        )
        raise ValueError(
            "sampled amplitude f contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # w_i(x) = (sin theta(x), cos theta(x)) at each of the 2*N - 1 nodes. Unlike the Levin path,
    # which needs only theta' on the grid and theta at the two endpoints, the CC rule integrates
    # the full integrand sum_i f_i(x) w_i(x) and therefore needs w_i sampled at every node.
    # eval_basis() already prefers theta_mod_2pi when the caller supplied one.
    basis_fine = np.array([BasisData.eval_basis(x) for x in fine_grid]).T

    integrand_fine = np.sum(f_fine * basis_fine, axis=0)

    w_fine = _cc_weights(x_span, fine_order)
    w_coarse = _cc_weights(x_span, chebyshev_order)

    value_fine = float(np.dot(w_fine, integrand_fine))
    value_coarse = float(np.dot(w_coarse, integrand_fine[::2]))

    abserr_direct = float(np.fabs(value_fine - value_coarse))

    # Round-off floor, Chen et al. eq. (151) -- the same model as the Levin branch
    # (_roundoff_floor()), replacing this fallback's own former endpoint-phase estimate (pending
    # prompt 04 note in earlier versions of this comment). f_scale = max_grid sqrt(sum_i f_i^2)
    # over the fine-grid samples already in hand, matching the Levin branch's choice (note (c) of
    # prompt 04) rather than the previous max_i sum_grid|f_i| L1-style estimate -- no extra
    # evaluations either way. G0/G1 reuse the *coarse* theta_prime_Cheb sample the phase_span gate
    # above already paid for, rather than resampling theta' on the fine grid.
    width = np.fabs(x_span[1] - x_span[0])
    G0, G1 = _levin_G0_G1(theta_prime_Cheb, width)
    f_scale = float(np.max(np.sqrt(np.sum(f_fine * f_fine, axis=0))))
    round_off_err = _roundoff_floor(f_scale, width, G0, G1, chebyshev_order)

    # Optional declared-theta_abserr endpoint term (recommendation 5.2), 0.0 unless the phase dict
    # supplied "theta_abserr". This fallback has no Levin antiderivative p to weight by, so it
    # reuses the same f_scale * (width / 2) proxy per endpoint that the pre-prompt-04 code used
    # for the whole interval (a crude sup-norm bound, not a per-endpoint quantity) -- see prompt
    # 04's log for why this, rather than the Levin branch's true per-endpoint p_endpoint_l1, was
    # judged an acceptable approximation here.
    endpoint_proxy = 0.5 * f_scale * width
    declared_err = _declared_endpoint_phase_err(
        BasisData, x_span[0], x_span[1], endpoint_proxy, endpoint_proxy
    )

    return {
        "value": value_fine,
        "abserr_direct": abserr_direct,
        "p_sample": None,
        "p_ratios": None,
        "phase_span": phase_span,
        "phase_err": round_off_err + declared_err,
        # No Levin antiderivative exists on this branch, so there is no p-mode to gate and
        # nothing for the rtol filter to discard -- see used_interval.abserr_truncation.
        "abserr_truncation": None,
        "metadata": {},
        "is_direct": True,
    }


def _local_atol(atol: float, a: float, b: float, x_span_width: float) -> float:
    """
    Scale the caller's atol by this region's share of the original integration interval's length
    (rec 8, C3b). Summing abserr over regions whose acceptance test uses this scaled value bounds
    the total by atol, by construction, because the length fractions of a partition sum to one.

    x_span_width is |b0 - a0| for the original x_span passed to _adaptive_levin, computed once
    before the driver loop starts; a and b are the current region's endpoints, in either order
    (Chen et al. step (4) does not require a < b).

    A zero-width x_span_width falls back to the unscaled atol rather than dividing by zero. That
    case is not otherwise guarded against here -- a zero-width call currently fails earlier, in
    build_Levin_data(), with a non-finite-theta-prime ValueError, before this function is ever
    reached -- but the fallback keeps this function itself total, and it would be the mathematically
    correct answer (fraction 1.0, since a zero-width original span can contain only a zero-width
    region) if that earlier failure were ever relaxed.
    """
    if x_span_width == 0.0:
        return atol
    return atol * (np.fabs(b - a) / x_span_width)


def _adaptive_levin(
    x_span: Tuple[float, float],
    f,
    BasisData,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics: bool = False,
    diagnostics_path: Optional[Union[str, Path]] = None,
):
    # Input validation (rec 4, C8, C9). This module cannot deliver a purely relative-error
    # contract: the phase-rounding floor computed by _phase_error() is absolute by nature, so a
    # caller asking for atol=0 leaves the relative-error denominator floor at :relerr_denom inert
    # and the phase_limited branch unable to fire, and was measured to subdivide an
    # identically-zero integrand to the full depth limit (256 regions / 1023 solves at
    # depth_max=8, against 1 region / 3 solves at atol=1e-15). Reject rather than silently
    # accepting a request this module cannot honour.
    if not (atol > 0):
        raise ValueError(
            f"levin_quadrature: atol must be strictly positive (received atol={atol!r}); "
            "this module cannot deliver a purely relative-error contract because its "
            "phase-rounding error floor is absolute by construction -- choose a small positive atol"
        )
    if rtol < 0:
        raise ValueError(
            f"levin_quadrature: rtol must be non-negative (received rtol={rtol!r})"
        )
    if depth_max < 0:
        raise ValueError(
            f"levin_quadrature: depth_max must be non-negative (received depth_max={depth_max!r})"
        )
    if len(x_span) != 2:
        raise ValueError(
            f"levin_quadrature: x_span must have exactly two entries (start, end); "
            f"received {len(x_span)} entries: {x_span!r}"
        )
    if not (np.isfinite(x_span[0]) and np.isfinite(x_span[1])):
        raise ValueError(
            f"levin_quadrature: x_span endpoints must be finite; received x_span={x_span!r}"
        )

    driver_start: float = time.perf_counter()
    start_time: float = time.time()
    last_notify: float = start_time
    updates_issued: int = 0

    # Unique id for this calculation, used only to tag log messages and (if emit_diagnostics or a
    # solve failure) diagnostic file paths. Deferred (sec 4.3, "fixed per-call overhead"): a
    # _LazyUUID does not call uuid.uuid4() (~2.3 microseconds) until it is first stringified, which
    # happens only inside an actual log message -- so a run that never logs anything (the common
    # case) never pays for one. See _LazyUUID and _format_label().
    id_label = _LazyUUID()

    # Vectorised-sampling detection cache (recommendation 14), created once per top-level call and
    # threaded to every subregion solve below -- see _sample_vectorized(). Each distinct f
    # component and phase callable is probed for array support at most once for the life of this
    # dict, not once per subregion.
    vectorize_cache: dict = {}

    # Resolved once, here, rather than per-subregion-call: every function below that can write a
    # diagnostic file receives this same concrete Path, never None (rec 12, C11 -- see
    # DEFAULT_LEVIN_DIAGNOSTICS_PATH's docstring for why the default is an explicit named
    # directory rather than an implicit Path.cwd()).
    resolved_diagnostics_path = (
        Path(diagnostics_path)
        if diagnostics_path is not None
        else DEFAULT_LEVIN_DIAGNOSTICS_PATH
    )

    if chebyshev_order < _LEVIN_MINIMUM_ALLOWED_ORDER:
        _logger.warning(
            "!! WARNING (adaptive_levin, %s): chebyshev_order=%d is below the minimum allowed "
            "order %d; every subregion solve will be clamped up to %d",
            _format_label(notify_label, id_label),
            chebyshev_order,
            _LEVIN_MINIMUM_ALLOWED_ORDER,
            _LEVIN_MINIMUM_ALLOWED_ORDER,
        )

    regions = [_levin_interval(start=x_span[0], end=x_span[1], depth=0)]

    # Original interval width, used by _local_atol() to distribute atol across subregions in
    # proportion to their share of it (rec 8, C3b) -- see that function's docstring.
    x_span_width = np.fabs(x_span[1] - x_span[0])

    val = 0.0
    used_regions = []
    p_points = []
    num_used_regions = 0
    num_simple_regions = 0
    num_evaluations = 0

    num_SVD_errors = 0
    num_order_changes = 0
    num_direct_solves = 0
    chebyshev_min_order = None
    max_depth = 0

    # True solve counts (rec 11, C10), accumulated from every Levin subregion solve actually
    # attempted -- the region's own solve (data) *and* the two comparison children (dataL, dataR)
    # computed for every Levin region regardless of whether the parent is accepted or bisected.
    # This is the honest total that num_direct_solves (below, kept for backward compatibility --
    # see its own note) is not: that counter accumulates only from each popped region's own
    # solve, so a solve performed purely to compute a comparison residual for an *accepted* parent
    # is never counted anywhere by it. See prompts/levin-refactor/logs/07-diagnostics-hygiene.md.
    num_solves_direct = 0
    num_solves_lstsq = 0
    num_solves_pinv = 0

    num_history_messages = 0

    while len(regions) > 0:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            _notify_progress(
                now,
                last_notify,
                start_time,
                val,
                num_used_regions,
                len(regions),
                num_simple_regions,
                max_depth,
                num_evaluations,
                num_SVD_errors,
                num_order_changes,
                chebyshev_min_order,
                updates_issued,
                id_label,
                notify_label,
            )

            # dump data every 3 notifications
            if emit_diagnostics and updates_issued % 3 == 1:
                _write_progress_data(
                    f,
                    BasisData,
                    regions,
                    chebyshev_order,
                    val,
                    id_label,
                    atol,
                    rtol,
                    notify_label,
                    resolved_diagnostics_path,
                    vectorize_cache=vectorize_cache,
                )

            last_notify = time.time()

        current_region = regions.pop()
        a = current_region.start
        b = current_region.end

        # atol scaled by this region's share of the original interval (rec 8, C3b); used by the
        # acceptance test, the relative-error denominator floor and the phase_limited guard below,
        # in both the fallback and Levin branches. rtol is deliberately not scaled here -- see the
        # module docstring and adaptive_levin_sincos()'s :param atol: note for why.
        local_atol = _local_atol(atol, a, b, x_span_width)

        # updated here, unconditionally, rather than only on the branch that accepts a Levin
        # region: the direct-quadrature branch below `continue`s before reaching that branch, so
        # updating it there missed every run that terminated in direct quadrature (C6).
        if current_region.depth > max_depth:
            max_depth = current_region.depth

        if current_region.depth >= 18 and num_history_messages < 20:
            num_history_messages += 1

            _logger.info(
                "@@ adaptive_levin (%s): encountered subinterval of depth %d (notification "
                "%d/20 for this quadrature)",
                _format_label(notify_label, id_label),
                current_region.depth,
                num_history_messages,
            )

            abs_history = current_region.abserr_history
            rel_history = current_region.relerr_history
            p_scale_history = current_region.p_ratios_history

            prev_abs = None
            prev_rel = None

            for i in range(0, current_region.depth):
                this_abs = abs_history[i]
                this_rel = rel_history[i]
                this_p_ratios = p_scale_history[i]

                if i == 0:
                    _logger.info(
                        "   -- %d. abserr=%.5g, relerr=%.5g, p-ratios=[%s]",
                        i + 1,
                        this_abs,
                        this_rel,
                        ", ".join(f"{p:.4g}" for p in this_p_ratios),
                    )

                else:
                    abs_improvement = prev_abs / this_abs
                    rel_improvement = prev_rel / this_rel
                    _logger.info(
                        "   -- %d. abserr=%.5g (improvement=%.3g), relerr=%.5g "
                        "(improvement=%.3g), p-ratios=[%s]",
                        i + 1,
                        this_abs,
                        abs_improvement,
                        this_rel,
                        rel_improvement,
                        ", ".join(f"{p:.3g}" for p in this_p_ratios),
                    )

                prev_abs = this_abs
                prev_rel = this_rel

        # Chen et al. (172).
        # If this region was produced by bisecting a parent, its estimate was already computed as one
        # half of the parent's refined estimate, and we can reuse it. Note that the metadata bookkeeping
        # below counts each region exactly once, when it is processed here as a parent -- the metadata
        # of the comparison regions dataL/dataR has never been accumulated, so reusing them preserves
        # the existing accounting exactly.
        #
        # This single call also makes the Levin-vs-fallback decision (C2): _adaptive_levin_subregion
        # samples theta' first, gates on the region's total phase variation, and only then either
        # solves the Levin system or evaluates the nested Clenshaw-Curtis pair -- see
        # _adaptive_levin_subregion_impl(). The two outcomes are told apart below by data["is_direct"].
        data = current_region.estimate
        if data is None:
            try:
                data = _adaptive_levin_subregion(
                    (a, b),
                    f,
                    BasisData,
                    id_label=id_label,
                    chebyshev_order=chebyshev_order,
                    rtol=rtol,
                    notify_label=notify_label,
                    build_p_sample=build_p_sample,
                    diagnostics_path=resolved_diagnostics_path,
                    vectorize_cache=vectorize_cache,
                )
                num_evaluations += 1
                # True-solve accounting (rec 11, C10): only inside this branch, i.e. only when a
                # solve is actually performed here -- when data is instead carried forward from a
                # parent's dataL/dataR (current_region.estimate was not None), that same solve was
                # already counted at the point dataL/dataR were computed, below. Counting it again
                # here (as the existing num_direct_solves/num_SVD_errors/num_order_changes
                # per-region accounting deliberately does, for a different reason -- see the
                # comment above this block) would double-count it for these new true totals.
                num_solves_direct += data["metadata"].get("direct_solve", 0)
                num_solves_lstsq += data["metadata"].get("lstsq_solve", 0)
                num_solves_pinv += data["metadata"].get("pinv_solve", 0)
            except LinAlgError as e:
                _logger.warning(
                    "!! adaptive_levin (%s): linear algebra error when estimating Levin "
                    "subregion (%s, %s), width=%.8g",
                    _format_label(notify_label, id_label),
                    a,
                    b,
                    current_region.width,
                )
                raise e

        order = data["metadata"].get("chebyshev_order", None)
        if order is not None:
            if chebyshev_min_order is None or order < chebyshev_min_order:
                chebyshev_min_order = order

        num_SVD_errors = num_SVD_errors + data["metadata"].get("SVD_errors", 0)
        num_order_changes = num_order_changes + data["metadata"].get(
            "num_order_changes", 0
        )
        num_direct_solves = num_direct_solves + data["metadata"].get("direct_solve", 0)

        c = current_region.break_point

        if data.get("is_direct", False):
            # Nested Clenshaw-Curtis fallback (C2, C7). Unlike the Levin branch below, this
            # region's error estimate -- |CC_{2N-1} - CC_N| -- comes from a nested pair evaluated
            # entirely within this one region, not from a comparison against children. So there is
            # no dataL/dataR to compute here, and none of the "reuse the comparison estimate as the
            # child's own estimate" bookkeeping applies: a bisected fallback region's children are
            # freshly evaluated (and independently re-gated on their own, roughly-halved phase span)
            # when they are popped.
            estimate = data["value"]
            abserr = data["abserr_direct"]

            relerr_denom = max(np.fabs(estimate), local_atol)
            relerr = abserr / relerr_denom

            phase_err = data.get("phase_err", 0.0) or 0.0

            resolved = abserr < local_atol or relerr < rtol

            # same precision-limited logic as the Levin branch below, adapted: a fallback region
            # has no step-(4) residual, so its own nested-pair abserr stands in for it. The
            # np.isfinite() guard matters here as of prompt 04: _roundoff_floor() returns +inf at
            # an interior stationary point whose G0 = min|theta'| lands on (or numerically at) a
            # sampled node, and an infinite floor must not be able to force early acceptance of a
            # region that further bisection could still improve -- see _roundoff_floor()'s
            # docstring. Without this guard "abserr <= phase_err" is trivially true against +inf
            # and phase_limited would fire on every such region regardless of resolution.
            phase_limited = (
                np.isfinite(phase_err)
                and phase_err > local_atol
                and phase_err > rtol * relerr_denom
                and abserr <= phase_err
            )
            phase_limited = phase_limited and not resolved

            if resolved or phase_limited or current_region.depth >= depth_max:
                val = val + estimate

                used_regions.append(
                    used_interval(
                        start=a,
                        end=b,
                        depth=current_region.depth,
                        type=INTERVAL_TYPE_DIRECT,
                        abserr=abserr,
                        relerr=relerr,
                        phase_err=phase_err,
                        phase_limited=phase_limited,
                        # no Levin antiderivative on this branch -- see
                        # used_interval.abserr_truncation.
                        abserr_truncation=data.get("abserr_truncation"),
                    )
                )
                num_used_regions = num_used_regions + 1
                num_simple_regions = num_simple_regions + 1

                # no Levin antiderivative sample exists for a fallback region
            else:
                new_abs_history = current_region.abserr_history | {
                    current_region.depth: abserr
                }
                new_rel_history = current_region.relerr_history | {
                    current_region.depth: relerr
                }
                new_p_ratios_history = current_region.p_ratios_history | {
                    current_region.depth: []
                }
                new_depth = current_region.depth + 1

                regions.extend(
                    [
                        _levin_interval(
                            start=a,
                            end=c,
                            depth=new_depth,
                            abserr_history=new_abs_history,
                            relerr_history=new_rel_history,
                            p_ratios_history=new_p_ratios_history,
                        ),
                        _levin_interval(
                            start=c,
                            end=b,
                            depth=new_depth,
                            abserr_history=new_abs_history,
                            relerr_history=new_rel_history,
                            p_ratios_history=new_p_ratios_history,
                        ),
                    ]
                )

            continue

        # Chen et al. (173)
        try:
            dataL = _adaptive_levin_subregion(
                (a, c),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                rtol=rtol,
                notify_label=notify_label,
                build_p_sample=build_p_sample,
                diagnostics_path=resolved_diagnostics_path,
                vectorize_cache=vectorize_cache,
            )
        except LinAlgError as e:
            _logger.warning(
                "!! adaptive_levin (%s): linear algebra error when estimating Levin "
                "left-comparison region (%s, %s), parent region = (%s, %s)",
                _format_label(notify_label, id_label),
                a,
                c,
                a,
                b,
            )
            raise e

        try:
            dataR = _adaptive_levin_subregion(
                (c, b),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                rtol=rtol,
                notify_label=notify_label,
                build_p_sample=build_p_sample,
                diagnostics_path=resolved_diagnostics_path,
                vectorize_cache=vectorize_cache,
            )
        except LinAlgError as e:
            _logger.warning(
                "!! adaptive_levin (%s): linear algebra error when estimating Levin "
                "right-comparison region (%s, %s), parent region = (%s, %s)",
                _format_label(notify_label, id_label),
                c,
                b,
                a,
                b,
            )
            raise e

        num_evaluations += 2
        # True-solve accounting (rec 11, C10): dataL/dataR are always two fresh solves, computed
        # here unconditionally regardless of whether this region ends up accepted (in which case
        # they are used only for the residual below and then discarded -- the case the existing
        # per-region num_direct_solves never captures at all) or bisected (in which case they also
        # become the two children's own "data" when popped, but are not re-counted there -- see
        # the note where num_solves_direct/lstsq/pinv are accumulated above).
        for _solve_data in (dataL, dataR):
            num_solves_direct += _solve_data["metadata"].get("direct_solve", 0)
            num_solves_lstsq += _solve_data["metadata"].get("lstsq_solve", 0)
            num_solves_pinv += _solve_data["metadata"].get("pinv_solve", 0)
        estimate = data["value"]
        refined_estimate = dataL["value"] + dataR["value"]

        abserr = np.fabs(estimate - refined_estimate)

        # guard the relative-error denominator. The integrand can pass through an accidental zero,
        # and there min(|estimate|, |refined_estimate|) is arbitrarily small, so relerr blows up
        # even though the region is perfectly well resolved -- driving subdivision to depth_max for
        # no gain. Below the requested absolute tolerance a relative test is meaningless anyway, so
        # floor the denominator at local_atol -- the same length-scaled tolerance the acceptance
        # test below uses (rec 8, C3b), so the two stay self-consistent. (min() rather than max()
        # of the two estimates is retained: it is the more conservative choice.)
        relerr_denom = max(min(np.fabs(estimate), np.fabs(refined_estimate)), local_atol)
        relerr = abserr / relerr_denom

        # Round-off floor for this region, from the parent estimate: it is the parent's endpoints
        # and grid that survive into the accumulated result. See _roundoff_floor().
        phase_err = data.get("phase_err", 0.0) or 0.0

        # Is this region precision-limited rather than under-resolved? All of the following must
        # hold: the round-off floor is finite, it exceeds what the caller asked for, AND the
        # step-(4) resolution residual has already come down to that floor. In that case
        # bisecting cannot help -- the children inherit essentially the same round-off floor (see
        # _roundoff_floor()), so subdivision buys additional cost and no accuracy. Accept and flag
        # it.
        #
        # The finiteness guard matters as of prompt 04: _roundoff_floor() returns +inf when G0 =
        # min|theta'| lands on (or numerically at) a sampled node -- an interior stationary point,
        # reachable here because the total-variation gate only routes a *weakly* oscillatory
        # neighbourhood of such a point to the Clenshaw-Curtis fallback (see the module docstring
        # and prompt 03's log); a strongly oscillatory region can still contain one. An infinite
        # floor must not be able to force early acceptance of a region that further bisection
        # could still improve -- without this guard "abserr <= phase_err" is trivially true
        # against +inf and phase_limited would fire regardless of resolution.
        #
        # The resolution-residual condition matters for a different reason: without it a region
        # with a large but still-reducible resolution residual would be accepted early merely
        # because its floor sits above atol, which would lose real accuracy.
        phase_limited = (
            np.isfinite(phase_err)
            and phase_err > local_atol
            and phase_err > rtol * relerr_denom
            and abserr <= phase_err
        )

        # Chen et al. step (4), below (173), adapted in two ways: a relative tolerance check is
        # also admitted, and regions whose accuracy is set by phase rounding rather than by lack
        # of resolution are accepted rather than subdivided.
        # Terminate in any case if we exceed the specified number of bisections.
        resolved = abserr < local_atol or relerr < rtol

        # Only report the region as phase-limited if the floor actually capped it, i.e. the
        # tolerance was *not* otherwise met. Eq. (151) (_roundoff_floor()) is a worst-case bound
        # with 4.5x measured margin at C = 1 (see _LEVIN_ROUNDOFF_SAFETY), so it can be loose on a
        # region that in fact already met its tolerance by resolution alone. Flagging those would
        # produce a warning about a result that in fact met its tolerance. The conservative bound
        # is still carried into the aggregate abserr either way; this only governs the diagnostic.
        phase_limited = phase_limited and not resolved

        if resolved or phase_limited or current_region.depth >= depth_max:
            val = val + estimate

            used_regions.append(
                used_interval(
                    start=a,
                    end=b,
                    depth=current_region.depth,
                    type=INTERVAL_TYPE_LEVIN,
                    abserr=abserr,
                    relerr=relerr,
                    p_ratios=data["p_ratios"],
                    phase_err=phase_err,
                    phase_limited=phase_limited,
                    # data is this region's own solve (not dataL/dataR): "value" above is
                    # data["value"], so data["abserr_truncation"] is the truncation actually
                    # incurred by the value this region contributes (prompt 06, problem (b)).
                    abserr_truncation=data.get("abserr_truncation"),
                )
            )
            num_used_regions = num_used_regions + 1

            if build_p_sample:
                p_points.extend(data["p_sample"])

        else:
            new_abs_history = current_region.abserr_history | {
                current_region.depth: abserr
            }
            new_rel_history = current_region.relerr_history | {
                current_region.depth: relerr
            }
            new_p_ratios_history = current_region.p_ratios_history | {
                current_region.depth: data["p_ratios"]
            }
            new_depth = current_region.depth + 1

            # carry the comparison estimates forward as the children's own estimates; they have just
            # been computed on exactly these intervals, so recomputing them when the children are
            # popped would repeat a third of all the linear solves performed by this driver
            regions.extend(
                [
                    _levin_interval(
                        start=a,
                        end=c,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                        p_ratios_history=new_p_ratios_history,
                        estimate=dataL,
                    ),
                    _levin_interval(
                        start=c,
                        end=b,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                        p_ratios_history=new_p_ratios_history,
                        estimate=dataR,
                    ),
                ]
            )

    used_regions.sort(key=lambda x: x.start)
    if build_p_sample:
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    # Aggregate error estimate. This function previously returned no error estimate at all, which
    # left callers with no option but to trust the result or re-run at a tighter tolerance and
    # compare. Each region contributes max(resolution/fallback residual, round-off floor): the two
    # are estimates of independent error sources, and the larger dominates.
    #
    # The regions are summed in absolute value rather than in quadrature. The round-off
    # contributions are not independent random errors -- neighbouring regions share endpoints, and
    # an inaccurate phase function produces a systematic drift common to all of them -- so a linear
    # sum is the defensible choice even though it is pessimistic when the residuals really do
    # behave randomly.
    #
    # abserr_resolution/abserr_roundoff/abserr_fallback/abserr_truncation (rec 9, rec 11, §3.4)
    # break the same total down by source, at no extra cost, so a caller can see *why* they
    # cannot get more digits: a large abserr_resolution means the driver could still make
    # progress by bisecting further (until depth_max); a large abserr_roundoff means it cannot,
    # no matter the tolerance requested; a large abserr_truncation means the rtol gate on small
    # p-modes (prompt 06) is the limiting factor, and a smaller rtol would help.
    abserr_total = 0.0
    abserr_resolution_total = 0.0
    abserr_roundoff_total = 0.0
    abserr_fallback_total = 0.0
    abserr_truncation_total = 0.0
    num_phase_limited = 0
    for region in used_regions:
        contribution = region.total_err
        if contribution is not None:
            abserr_total = abserr_total + contribution
        if region.abserr_resolution is not None:
            abserr_resolution_total = abserr_resolution_total + region.abserr_resolution
        if region.abserr_roundoff is not None:
            abserr_roundoff_total = abserr_roundoff_total + region.abserr_roundoff
        if region.abserr_fallback is not None:
            abserr_fallback_total = abserr_fallback_total + region.abserr_fallback
        if region.abserr_truncation is not None:
            abserr_truncation_total = abserr_truncation_total + region.abserr_truncation
        if region.phase_limited:
            num_phase_limited = num_phase_limited + 1

    # guarded exactly as the per-region relative test is guarded, and for the same reason
    relerr_total = abserr_total / max(np.fabs(val), atol)

    # Honest-aggregate check (rec 2, C3). As of prompt 05, region acceptance at step (4) tests each
    # region's residual against atol scaled by its own share of the original interval length
    # (_local_atol()), so the *summed* abserr_total is bounded by atol by construction (the length
    # fractions of a partition sum to one) -- converged should now almost always be True for a run
    # that terminates normally. It can still be False: a region accepted unconverged at depth_max
    # (the health check below), or a region whose round-off floor (Chen et al. eq. 151) exceeds its
    # local_atol share, can each still push the sum over the top. Report whether the aggregate in
    # fact meets the request; this is a report, not an additional acceptance criterion.
    requested_total = max(atol, rtol * np.fabs(val))
    converged = abserr_total <= requested_total
    if not converged:
        _logger.warning(
            "!! WARNING (adaptive_levin, %s): the aggregate error estimate exceeds what was "
            "requested | abserr_total=%.3g > requested=%.3g (atol=%.3g, rtol=%.3g)",
            _format_label(notify_label, id_label),
            abserr_total,
            requested_total,
            atol,
            rtol,
        )
        _logger.warning(
            "   -- atol is distributed across subintervals by length share, so this usually means "
            "a region hit depth_max unresolved, or the round-off floor (Chen et al. eq. 151) "
            "exceeds its share of atol; consider a tighter atol/rtol, a larger depth_max, or a "
            "smaller chebyshev_order"
        )

    # Health check. If neither atol nor rtol is attainable -- typically because the caller has asked for
    # an accuracy the phase function cannot deliver -- the bisection runs to the depth limit and we
    # accept whatever estimate we have. The answer is usually still good, but the cost can be two orders
    # of magnitude higher than necessary, so it is worth saying so rather than letting it pass silently.
    #
    # Note we deliberately do NOT warn on a high proportion of direct-quadrature subintervals. When the
    # integration variable is log(x) (as it is for the Bessel integrals this module was written for),
    # subintervals at small x carry very little phase and are *correctly* handled by direct quadrature;
    # difference-type phase groups, whose net frequency can be small, do the same over much of the
    # range. In that geometry a direct-quadrature fraction above 90% is normal rather than pathological,
    # and no threshold separates it from genuine non-convergence. num_simple_regions is reported in the
    # returned dictionary for callers who want to look.
    if max_depth >= depth_max:
        _logger.warning(
            "!! WARNING (adaptive_levin, %s): bisection reached the maximum depth %d without "
            "meeting either tolerance, so some subintervals were accepted unconverged | %d "
            "subintervals (%d direct), %d Levin solves | atol=%.3g, rtol=%.3g",
            _format_label(notify_label, id_label),
            depth_max,
            num_used_regions,
            num_simple_regions,
            num_evaluations,
            atol,
            rtol,
        )
        _logger.warning(
            "   -- consider relaxing atol/rtol, or check that the phase function is accurate "
            "enough to support the requested tolerance"
        )

    # Second health check, on the round-off floor rather than on resolution. This is the case the
    # step-(4) residual cannot detect on its own, so it has to be reported explicitly: the caller
    # asked for an accuracy below the eq. (151) round-off floor (_roundoff_floor()) for one or more
    # regions. Tightening atol/rtol will not help, and -- unlike before prompt 04 -- supplying a
    # range-reduced phase (theta_mod_2pi) will not help either: the floor is independent of how the
    # phase is presented (C4; see prompt 04's log). What can help is a smaller chebyshev_order
    # (lowering the max(G1, k^2) term) or, where the phase's own construction is the real limit
    # (e.g. a fitted spline), a declared theta_abserr so the caller sees an honest number instead
    # of an artificially small one.
    if num_phase_limited > 0:
        _logger.warning(
            "!! WARNING (adaptive_levin, %s): %d of %d subintervals were limited by the "
            "round-off floor (Chen et al. eq. 151), not by resolution | estimated abserr=%.3g, "
            "relerr=%.3g | atol=%.3g, rtol=%.3g",
            _format_label(notify_label, id_label),
            num_phase_limited,
            num_used_regions,
            abserr_total,
            relerr_total,
            atol,
            rtol,
        )
        _logger.warning(
            "   -- the requested tolerance is not attainable at this Chebyshev order and interval "
            "geometry; this floor does NOT depend on whether the phase is range-reduced"
        )

    return {
        "value": float(val),
        # estimated absolute and relative error of "value". abserr already includes the round-off
        # floor (Chen et al. eq. 151) as well as the step-(4)/nested-pair residual, so it does not
        # under-report at high frequency the way the resolution residual alone does.
        "abserr": float(abserr_total),
        "relerr": float(relerr_total),
        # abserr broken down by source (rec 11, rec 9, §3.4) -- see used_interval.abserr_resolution
        # / .abserr_roundoff / .abserr_fallback / .abserr_truncation. abserr_resolution/roundoff/
        # fallback are new as of prompt 04; abserr_truncation as of prompt 06. Existing consumers
        # keyed on "abserr"/"relerr" are unaffected.
        "abserr_resolution": float(abserr_resolution_total),
        "abserr_roundoff": float(abserr_roundoff_total),
        "abserr_fallback": float(abserr_fallback_total),
        "abserr_truncation": float(abserr_truncation_total),
        # True if the aggregate abserr actually meets max(atol, rtol*|value|). As of prompt 05,
        # atol is distributed across subintervals by length share (_local_atol()), so this is
        # usually True by construction for a run that terminates normally; see the warning printed
        # above for the remaining ways it can be False (depth_max reached unresolved, or the
        # round-off floor exceeding a region's share of atol).
        "converged": bool(converged),
        # True if at least one accepted region was limited by the round-off floor rather than by
        # resolution. When this is set, the requested tolerance was not attainable at this
        # Chebyshev order and interval geometry, and tightening atol/rtol will not help.
        "phase_limited": num_phase_limited > 0,
        "num_phase_limited_regions": num_phase_limited,
        "p_points": p_points,
        "num_regions": num_used_regions,
        "regions": used_regions,
        "num_simple_regions": num_simple_regions,
        # "evaluations" counts subregion *solves* (a fresh region solve, plus the two comparison
        # children for every region that takes the Levin branch), not integrand evaluations --
        # kept under this name because docs/adaptive-levin-benchmark/levin_bench/ reads it by key
        # (rec 11, C10). num_subregion_solves is the same value under a name that says what it
        # counts; prefer it in new code.
        "evaluations": int(num_evaluations),
        "num_subregion_solves": int(num_evaluations),
        "elapsed": float(elapsed),
        "num_SVD_errors": num_SVD_errors,
        "num_order_changes": num_order_changes,
        # num_direct_solves is kept exactly as before -- a *per-region* count, accumulated only
        # from each popped region's own solve, never from a comparison child (dataL/dataR) whose
        # parent is accepted rather than bisected (rec 11, C10; see the comment where it is
        # accumulated above). docs/adaptive-levin-benchmark/levin_bench/runners.py reads it by
        # this name. num_solves_direct/num_solves_lstsq/num_solves_pinv below are the true totals,
        # counting every solve actually attempted (region + both comparison children, whether or
        # not the parent is ultimately accepted); num_solves_total is their sum. A caller wanting
        # to check the LU fast path's coverage (audit sec 4.4) should use
        # num_solves_direct / num_solves_total, not num_direct_solves / evaluations.
        "num_direct_solves": num_direct_solves,
        "num_solves_direct": num_solves_direct,
        "num_solves_lstsq": num_solves_lstsq,
        "num_solves_pinv": num_solves_pinv,
        "num_solves_total": num_solves_direct + num_solves_lstsq + num_solves_pinv,
        "chebyshev_min_order": chebyshev_min_order,
        "max_depth": max_depth,
    }


def _notify_progress(
    now: float,
    last_notify: float,
    start_time: float,
    current_val: float,
    used_regions: int,
    remain_regions: int,
    simple_regions: int,
    max_depth: int,
    num_evaluations: int,
    num_SVD_errors: int,
    num_order_changes: int,
    chebyshev_min_order: int,
    update_number: int,
    id_label,
    notify_label: str = None,
):
    since_last_notify = now - last_notify
    since_start = now - start_time

    if notify_label is not None:
        _logger.info(
            '** STATUS UPDATE #%d: Levin quadrature "%s" (%s) has been running for %s (%s '
            "since last notification)",
            update_number,
            notify_label,
            id_label,
            format_time(since_start),
            format_time(since_last_notify),
        )
    else:
        _logger.info(
            "** STATUS UPDATE #%d: Levin quadrature %s has been running for %s (%s since last "
            "notification)",
            update_number,
            id_label,
            format_time(since_start),
            format_time(since_last_notify),
        )

    # "subregion solves", not "integrand evaluations" -- num_evaluations counts calls to
    # _adaptive_levin_subregion (rec 11, C10; see the "evaluations" / "num_subregion_solves" note
    # on the returned dictionary above), and this message previously named it the way the audit
    # itself misread it.
    _logger.info(
        "|  -- current value = %.7g, %d used subintervals (of which %d direct quadrature), %d "
        "subintervals remain | %d subregion solves | max depth %d",
        current_val,
        used_regions,
        simple_regions,
        remain_regions,
        num_evaluations,
        max_depth,
    )
    if num_SVD_errors > 0 or num_order_changes > 0:
        _logger.info(
            "|  -- %d SVD errors, %d Chebyshev order changes | minimum used Chebyshev order = %s",
            num_SVD_errors,
            num_order_changes,
            chebyshev_min_order,
        )


def _safe_fabs(x):
    if x is None:
        return None

    return np.fabs(x)


def _write_progress_data(
    f,
    BasisData,
    regions: List[_levin_interval],
    chebyshev_order: int,
    current_val: float,
    id_label: uuid,
    atol: float,
    rtol: float,
    notify_label: Optional[str] = None,
    diagnostics_path: Path = DEFAULT_LEVIN_DIAGNOSTICS_PATH,
    vectorize_cache: Optional[dict] = None,
):
    # Imported here, not at module scope: this is the only function in the module that uses
    # either, and both are expensive to import (95-97% of this module's own import cost -- see
    # the module docstring's LOGGING section neighbour and prompts/levin-refactor/logs/
    # 07-diagnostics-hygiene.md). Only reached under emit_diagnostics=True, so the cost is paid at
    # most once every three notification intervals, not on every import of this module.
    import seaborn as sns
    from matplotlib import pyplot as plt

    path = (
        diagnostics_path
        / "SlowLevinData"
        / str(id_label)
        / datetime.now().replace(microsecond=0).isoformat()
    ).resolve()
    path.mkdir(parents=True, exist_ok=True)

    payload = {
        "current_val": current_val,
        "id_label": str(id_label),
        "user_label": notify_label,
        "atol": atol,
        "rtol": rtol,
    }

    sns.set_theme()

    m = len(f)

    region_list = []
    for reg_num, region in enumerate(regions):
        start = region.start
        end = region.end
        region_data = {
            "id": reg_num,
            "start": start,
            "end": end,
            "depth": region.depth,
            "abserr_history": region.abserr_history,
            "relerr_history": region.relerr_history,
            "p_ratios_history": region.p_ratios_history,
        }

        x_grid = np.linspace(start, end, 500)
        y_grid = []
        f_grid = [[] for _ in range(m)]
        for x in x_grid:
            basis = BasisData.eval_basis(x)
            y = sum(f[i](x) * basis[i] for i in range(m))
            y_grid.append(_safe_fabs(y))
            for i in range(m):
                f_grid[i].append(_safe_fabs(f[i](x)))

        data = _adaptive_levin_subregion(
            (start, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
            diagnostics_path=diagnostics_path,
            vectorize_cache=vectorize_cache,
        )
        estimate = data["value"]
        region_data["estimate"] = estimate

        mid = region.break_point

        dataL = _adaptive_levin_subregion(
            (start, mid),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
            diagnostics_path=diagnostics_path,
            vectorize_cache=vectorize_cache,
        )
        estimateL = dataL["value"]

        dataR = _adaptive_levin_subregion(
            (mid, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            rtol=rtol,
            notify_label=notify_label,
            build_p_sample=True,
            diagnostics_path=diagnostics_path,
            vectorize_cache=vectorize_cache,
        )
        estimateR = dataR["value"]

        refined_estimate = estimateL + estimateR
        region_data["estimate_L"] = estimateL
        region_data["estimate_R"] = estimateR
        region_data["refined_estimate"] = refined_estimate

        abserr = np.fabs(estimate - refined_estimate)

        # Guarded exactly as the main path's relative-error denominator is guarded (:relerr_denom,
        # in the driver loop above) -- min(|estimate|, |refined_estimate|) is arbitrarily small
        # whenever the integrand passes through an accidental zero, which previously made this
        # diagnostic-only relerr blow up (or divide by exactly zero) even for a well-resolved
        # region. Floored at atol, the same absolute tolerance this function already receives.
        relerr_denom = max(min(np.fabs(estimate), np.fabs(refined_estimate)), atol)
        relerr = np.fabs(estimate - refined_estimate) / relerr_denom

        region_data["relerr"] = relerr
        region_data["abserr"] = abserr
        region_data["abserr_status"] = "fail" if abserr >= atol else "pass"
        region_data["relerr_status"] = "fail" if relerr >= rtol else "pass"

        # only report regions that will not be accepted in the next step
        if abserr >= atol and relerr >= rtol:
            region_list.append(region_data)

            # p_sample is None for a Clenshaw-Curtis fallback region (no Levin antiderivative
            # exists there -- see _adaptive_levin_subregion_cc()'s return dict): "or []" plots an
            # empty p-curve for such a region rather than crashing. Latent since prompt 03
            # introduced the fallback branch; this diagnostics-only function was never exercised
            # against it until prompt 07 verified emit_diagnostics end to end (see
            # prompts/levin-refactor/logs/07-diagnostics-hygiene.md).
            p_x_grid = []
            p_y_grid = [[] for _ in range(m)]
            for x, p_data in data["p_sample"] or []:
                p_x_grid.append(x)
                for i in range(m):
                    p_y_grid[i].append(_safe_fabs(p_data[i]))

            pL_x_grid = []
            pL_y_grid = [[] for _ in range(m)]
            for x, p_data in dataL["p_sample"] or []:
                pL_x_grid.append(x)
                for i in range(m):
                    pL_y_grid[i].append(_safe_fabs(p_data[i]))

            pR_x_grid = []
            pR_y_grid = [[] for _ in range(m)]
            for x, p_data in dataR["p_sample"] or []:
                pR_x_grid.append(x)
                for i in range(m):
                    pR_y_grid[i].append(_safe_fabs(p_data[i]))

            fig = plt.figure()
            ax = plt.gca()

            ax.plot(x_grid, y_grid, label="integrand", color="r")

            for i in range(m):
                ax.plot(x_grid, f_grid[i], linestyle="dashdot", label=f"Levin f{i+1}")
                ax.plot(
                    p_x_grid, p_y_grid[i], linestyle="dashed", label=f"Levin p{i+1}"
                )
                ax.plot(
                    pL_x_grid, pL_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} L"
                )
                ax.plot(
                    pR_x_grid, pR_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} R"
                )

            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True)

            fig_path = path / f"region{reg_num}_start{start:.5g}_end{end:.5g}.pdf"
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

            plt.close()

    payload["regions"] = region_list
    payload_path = path / "payload.json"
    with open(payload_path, "w", newline="") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def adaptive_levin_sincos(
    x_span: Tuple[float, float],
    f,
    theta: dict,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics=False,
    diagnostics_path: Optional[Union[str, Path]] = None,
):
    """
    Adaptive Levin quadrature of an integral of the form

        integral_{x_span} [ f[0](x) sin(theta(x)) + f[1](x) cos(theta(x)) ] dx

    using the adaptive Levin method of Bremer, Chen & Yang (arXiv:2211.13400, section 5), falling
    back to a nested Clenshaw-Curtis rule (a pair of Chebyshev quadratures at orders N and
    2*N - 1, sharing the same 2*N - 1 integrand samples) on subintervals whose total phase
    variation is too small for the Levin rule to offer an advantage. The fallback's own error
    estimate, |CC_{2N-1} - CC_N|, is tested against atol/rtol on the same terms as a Levin
    region's, so a fallback region that misses its tolerance is bisected rather than accepted
    outright.

    :param x_span: a 2-tuple (a, b) giving the integration limits. Must have exactly two finite
        entries.
    :param f: a 2-element sequence of callables [f_sin, f_cos]. f_sin multiplies sin(theta(x)),
        f_cos multiplies cos(theta(x)). This basis is fixed at two components (sin, cos); the
        underlying driver supports an arbitrary number of components for other bases, but this
        entry point does not expose that. Each of f_sin/f_cos, and the phase callables in theta
        below, is sampled with a single array call instead of a per-point Python loop
        (recommendation 14) if a one-time probe (on first use, not on every subregion) finds that
        the callable accepts an array argument and returns pointwise-correct results -- no flag or
        opt-in needed. A callable that does not vectorize (e.g. one built on scalar-only branching,
        as every phase/modulus spline in LiouvilleGreen/ is as of prompt 08 -- see
        prompts/levin-refactor/logs/08-order-and-sampling.md) is unaffected: detection falls back
        to the loop and costs one array call plus two scalar calls, once per run.
    :param theta: a dict describing the phase function, with keys:
          * "theta" (required) -- callable, the raw phase theta(x). Always used to decide
            whether a subinterval is oscillatory enough for the Levin rule (via the total phase
            change across it), and used to evaluate sin/cos directly if "theta_mod_2pi" is not
            supplied.
          * "theta_mod_2pi" (optional) -- callable, theta(x) reduced to the range [0, 2*pi) (or
            any range of length 2*pi). When supplied, sin/cos are evaluated from this
            range-reduced value instead of the raw phase, which is handed an O(2*pi) argument
            rather than an O(theta) one and so has O(eps) absolute rounding error instead of
            O(eps*theta). This keeps the *values* handed to sin/cos well-conditioned, but as of
            prompt 04 does NOT lower the reported error floor: the floor is now Chen et al. eq.
            (151) (see "abserr_roundoff" below), which is independent of how the phase is
            presented (C4 -- an earlier version of this module inferred the floor from whether
            this key was supplied, and that inference was optimistic by up to 8.1e9 at high
            frequency; see prompts/levin-refactor/logs/04-roundoff-floor.md).
          * "theta_deriv" (optional) -- callable, theta'(x). When supplied, it is used directly
            instead of differentiating a sampled theta(x) with the spectral differentiation
            matrix; this avoids inheriting rounding error from the magnitude of the raw phase
            into the derivative estimate.
          * "theta_abserr" (optional) -- scalar, or callable of x, giving a declared absolute
            error of the phase itself, in radians (recommendation 5.2). When supplied, each
            region's round-off floor gains an endpoint term proportional to it. This is the only
            way to represent error that the phase function's own construction introduces -- e.g.
            the fit error of a phase spline -- which is otherwise invisible to this module.
            Nothing in LiouvilleGreen/ supplies this today; it ships unused by production callers
            pending phase_spline.py growing an accuracy API of its own (README Sec 6 of the
            levin-refactor campaign).
    :param atol: requested absolute tolerance on the *summed* result. Must be strictly positive:
        this module cannot deliver a purely relative-error contract because its round-off error
        floor (Chen et al. eq. 151) is absolute by construction. As of prompt 05, each subregion's
        acceptance test is against atol scaled by that region's share of the original x_span's
        length (rec 8, C3b) rather than against atol directly, so the sum of accepted-region
        residuals is bounded by atol by construction (the length fractions of a partition sum to
        one) -- the returned "converged" flag is now usually True for a run that terminates
        normally. It can still be False if a region is accepted unresolved at depth_max, or if a
        region's round-off floor exceeds its length-scaled share of atol; see "converged" below.
    :param rtol: requested relative tolerance. Must be non-negative. Unlike atol, rtol is applied
        per region, unscaled: a relative tolerance is not additive over a partition the way an
        absolute one is, so there is no length-proportional analogue that would mean anything.
    :param chebyshev_order: spectral order used for each Levin subregion collocation grid.
        Values below 8 are clamped up, with a warning. Default 16 (recommendation 13; raised from
        12 in prompt 08 -- see DEFAULT_LEVIN_CHEBSHEV_ORDER's comment and
        prompts/levin-refactor/logs/08-order-and-sampling.md for the measurement). 12-32 is the
        useful band: below it, order 8 was slower on every problem measured (sometimes by two
        orders of magnitude); above it, a problem that has already stopped subdividing pays a
        rising per-solve cost for no further accuracy. The optimum is problem-dependent (it tracks
        how much the amplitude subdivides at a given order), so a caller with an unusually
        expensive integrand may still want to sweep this rather than trust the default.
    :param depth_max: maximum bisection depth. Must be non-negative.
    :param build_p_sample: if True, retain and return the sampled Levin antiderivatives p(x) on
        every accepted region (diagnostics only; costs additional memory).
    :param notify_interval: seconds between progress notifications for long-running calls.
    :param notify_label: optional label included in progress and warning messages.
    :param emit_diagnostics: if True, periodically write plots and a JSON payload describing
        slow-to-converge regions to disk (see _write_progress_data()).
    :param diagnostics_path: directory under which diagnostic output is written: the
        emit_diagnostics plots/JSON payload (in a "SlowLevinData/<run id>/<timestamp>"
        subdirectory, as before), and -- regardless of emit_diagnostics -- the rare lstsq-failure
        dump (LevinL_*.txt / f_Cheb_*.txt, under a "failures" subdirectory). Defaults to
        DEFAULT_LEVIN_DIAGNOSTICS_PATH ("levin_diagnostics", relative to the process cwd) rather
        than the process cwd directly, so that the location is a stable, named default instead of
        an implicit dependency on whatever cwd happens to be -- notably under a Ray driver, where
        cwd is not guaranteed to be stable, writable, or distinct per worker. This is a
        behavioural change from before prompts/levin-refactor/logs/07-diagnostics-hygiene.md,
        which wrote both outputs directly into cwd ("SlowLevinData/..." and, for the failure
        dump, cwd itself); pass diagnostics_path=Path(".") to recover the old failure-dump
        location, or diagnostics_path=Path(".") together with reading "SlowLevinData/" under it
        for the old progress-data location.

    :return: a dict with (at least) the following keys:
          * "value" -- the estimated value of the integral.
          * "abserr" -- estimated absolute error of "value". This is a sum, over accepted
            regions, of max(resolution/fallback residual, round-off floor) plus any truncation
            term (see abserr_truncation below); it is an estimate, not a proven bound.
          * "abserr_resolution", "abserr_roundoff", "abserr_fallback", "abserr_truncation" --
            "abserr" broken down by source: the summed step-(4) residual over Levin regions (new
            as of prompt 04), the summed round-off floor (Chen et al. eq. 151, plus any declared
            theta_abserr endpoint term) over every region (new as of prompt 04), the summed
            nested-pair estimate over Clenshaw-Curtis fallback regions (new as of prompt 04), and
            the summed endpoint contribution of p-modes dropped by the rtol gate on Levin regions
            (new as of prompt 06, rec 9), respectively. A caller that cannot get more digits can
            tell from these which source is responsible: a large abserr_resolution means further
            bisection (a tighter atol/rtol) could still help; a large abserr_roundoff means it
            cannot; a large abserr_truncation means a smaller rtol would recover more modes.
          * "relerr" -- "abserr" divided by max(|value|, atol).
          * "converged" -- True if "abserr" <= max(atol, rtol*|value|), i.e. whether the request
            was actually met in aggregate. Usually True as of prompt 05 (see the atol note above).
          * "phase_limited" -- True if at least one accepted region's achievable accuracy was set
            by the round-off floor rather than by lack of resolution; tightening atol/rtol will
            not help such regions -- see abserr_roundoff above.
          * "num_phase_limited_regions" -- count of such regions.
          * "num_regions" -- number of accepted subintervals (Levin + direct quadrature).
          * "num_simple_regions" -- of those, how many were handled by direct quadrature rather
            than the Levin rule.
          * "regions" -- list of used_interval objects describing each accepted subinterval.
          * "p_points" -- sampled Levin antiderivatives, if build_p_sample was True.
          * "evaluations" / "num_subregion_solves" -- identical values: the number of subregion
            solves performed (a region's own solve, plus two comparison-child solves for every
            region that takes the Levin branch). "evaluations" is the pre-existing name -- kept
            because docs/adaptive-levin-benchmark/levin_bench/ reads it by key -- despite naming
            them "evaluations" of the integrand, which they are not; "num_subregion_solves" is the
            same number under an accurate name (rec 11, C10).
          * "elapsed" -- wall-clock time in seconds.
          * "num_SVD_errors", "num_order_changes", "chebyshev_min_order", "max_depth" --
            diagnostic counters; see the source for exact semantics.
          * "num_direct_solves" -- kept for backward compatibility with
            docs/adaptive-levin-benchmark/levin_bench/runners.py, which reads it by this name.
            Despite the name, this is a *per-region* count (accumulated only from each popped
            region's own solve), not a total solve count: a comparison-child solve (dataL/dataR)
            whose parent region is accepted rather than bisected is never counted by it (rec 11,
            C10). "num_solves_direct", "num_solves_lstsq" and "num_solves_pinv" below are the
            true totals -- every solve actually attempted, by the method that succeeded --
            and "num_solves_total" is their sum; prefer these in new code, e.g. to check what
            fraction of solves took the fast LU path (audit sec 4.4) via
            num_solves_direct / num_solves_total.
    """
    if len(f) != 2:
        raise ValueError(
            f"levin_quadrature: adaptive_levin_sincos requires exactly two amplitude functions "
            f"f=[f_sin, f_cos] (the sin/cos basis _Basis_SinCos is fixed at two components); "
            f"received {len(f)}"
        )

    A = _Basis_SinCos(theta)

    return _adaptive_levin(
        x_span,
        f,
        A,
        atol=atol,
        rtol=rtol,
        chebyshev_order=chebyshev_order,
        depth_max=depth_max,
        build_p_sample=build_p_sample,
        notify_interval=notify_interval,
        notify_label=notify_label,
        emit_diagnostics=emit_diagnostics,
        diagnostics_path=diagnostics_path,
    )
