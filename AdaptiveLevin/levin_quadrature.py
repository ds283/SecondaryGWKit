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

  * Error control is global, not per region (prompts/levin-phase-floor/
    06c-global-error-control.md). Chen et al.'s step (4) accepts each region
    on its own absolute residual. This driver instead keeps every leaf of the
    current partition with its value v_i and error e_i, and stops when
    sum_i e_i <= max(atol, rtol * |sum_i v_i|): one budget for the call,
    with a relative tolerance taken against the current total, not against
    each region's own value. Until it is met, the refinable leaf with the
    largest error is bisected (QUADPACK's QAG). A leaf is not refinable at
    depth_max, or when it is floor-limited (below). See _adaptive_levin() for
    why the per-region tests this replaces failed where the regions' values
    cancel. Before the loop may stop, each leaf's estimate is checked one
    level down, against its grandchildren: its declared error is widened by
    the observed convergence ratio r, and a leaf with r >= 1 is refined
    whatever the budget says (the trust check; see _adaptive_levin()).

  * The returned error estimate accounts for a round-off floor -- on a Levin
    region Chen et al.'s own eq. (151), see _roundoff_floor(); on a
    Clenshaw-Curtis fallback region the rule's own, see _cc_roundoff_floor()
    -- as well as the step-(4) resolution residual (or, for a fallback
    region, its nested-pair estimate). Eq. (151) bounds the conditioning of
    the Levin collocation solve, with a 1/min|theta'| that grows at a
    stationary point of the phase; a fallback region solves no such system,
    so it carries a floor derived from the weighted sum it computes instead
    (prompts/levin-phase-floor/06d-cc-roundoff-floor.md). See the extended discussion in _adaptive_levin();
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

    A leaf chosen for refinement whose floor is finite, and whose residual
    is already down at it, is not bisected but flagged round-off-limited
    (phase_limited, "floor-limited") -- but only if bisecting it would not
    lower the floor: its two halves' floors must sum to at least
    _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its own. That premise used
    to be assumed, and it fails where max|theta'| / min|theta'| is large,
    because there the floor comes from fitting one region over many decades
    of frequency rather than from the integrand. See that constant's comment
    and prompts/levin-phase-floor/06b-roundoff-exit-premise.md. Under the
    global budget a leaf whose floor dominates the sum is the one chosen, so
    a floor its halves would cut is cut.

  * Weakly oscillatory regions (total variation of the phase, mean|theta'| * width,
    below 6*pi) are handed to a nested Clenshaw-Curtis rule rather than to the Levin
    rule. The extremal Chebyshev grid the Levin solve already uses *is* the
    Clenshaw-Curtis grid, and its N-point grid is exactly every other node of the
    (2N-1)-point grid (nesting holds for every N >= 2), so the pair (CC_N, CC_{2N-1})
    costs 2N-1 integrand and phase evaluations, always, and comes with its own error
    estimate |CC_{2N-1} - CC_N|. That estimate enters the call's error budget on the
    same terms as a Levin leaf's, so a fallback leaf holding the total back is bisected
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

import heapq
import json
import logging
import math
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

# Safety factor 'C' in Chen et al. eq. (151), the round-off floor applied to every Levin region's
# estimate (a Clenshaw-Curtis fallback region carries its own floor, _cc_roundoff_floor(), since
# prompts/levin-phase-floor/06d-cc-roundoff-floor.md):
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

# The round-off-limited exit's premise (prompts/levin-phase-floor/06b-roundoff-exit-premise.md).
# _adaptive_levin() does not bisect a leaf it has chosen while the call's error budget is unmet
# when the leaf's eq. (151) floor is finite and its residual is already down at that floor
# (floor-limited, phase_limited; under the per-region scheme before prompt 06c, a region was
# accepted on the same grounds when its floor was above its tolerance). That rests on one premise:
# that bisecting cannot lower the floor. It is checked, not assumed (since 06b): the leaf is
# floor-limited only if the floors of its two halves sum to at least this fraction of its own,
#
#     phase_err_L + phase_err_R >= _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO * phase_err_parent.
#
# Why the premise can fail. The floor is roughly eps * ||f|| * (h/2) * (1 + G1/G0 + max(G1, k^2)/G0)
# with G0 = min|theta'| and G1 = max|theta'| on the rescaled region. Where G1/G0 is large, the floor
# is set by fitting one region over many decades of frequency, not by the integrand: halving the
# region cuts that ratio, roughly to its square root when theta' grows exponentially, so the
# halves' floors sum to far less than the parent's. Found on a seeded LiouvilleGreen three-Bessel
# draw ([07-roundoff-exit-assumes-bisection-cannot-help]): a Levin region with G1/G0 ~ 1e9 was
# accepted at depth 0 on a floor of 9e-7 that bisection would have cut by ~1e4, outside the test's
# tolerance.
#
# Why 1/2.
#   * Where the floor is genuinely round-off it does not fall under bisection: the summed floor is
#     roughly invariant under subdivision (see _roundoff_floor()), and where the k^2 term dominates
#     it grows.
#   * Where it comes from a wide range of theta', the halves' sum falls like the ratio's square
#     root -- (1 + 2 sqrt(R)) / (1 + 2 R) of the parent's for R = G1/G0 with theta' exponential and
#     G1 above k^2: a factor ~1e4 at R = 1e9.
#   * 1/2 separates the two with a margin either way: R = 3 gives a halves' sum of about 0.64 of the
#     parent's, and the exit fires; R = 10 gives about 0.35, and the region is bisected.
#   * Not 1: that would never let the exit fire while G1/G0 > 1, however slowly the floor fell,
#     and would drive every such region on towards depth_max.
_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5

# The unit round-off of IEEE double precision, u = 2^-53: the bound on the relative error of one
# correctly rounded floating-point operation. Used by _cc_roundoff_floor(), the round-off floor of a
# Clenshaw-Curtis fallback solve (prompts/levin-phase-floor/06d-cc-roundoff-floor.md), which counts
# roundings term by term and so needs the exact unit, not MACHINE_EPSILON's approximate 1e-16.
_UNIT_ROUNDOFF = 2.0**-53

# The per-term rounding count added to the n-term dot product's gamma_n in _cc_roundoff_floor():
# each term w_k * sum_i f_ik b_ik carries about one u from each of
#   1. the product of the weight with the node's integrand value;
#   2. the weight's own computation (_cc_weights_base()'s cosine sum and _cc_weights()'s rescale);
#   3. the basis evaluation, sin or cos of the argument a_k;
#   4. the amplitude f_ik, correct to its last bit;
# so the computed value's rounding is bounded by (n + 4) u S, S = sum_k |w_k| sum_i |f_ik b_ik|.
# Derived, not fitted.
_CC_ROUNDOFF_PER_TERM_COUNT = 4

# The node-rounding count c_x in _cc_node_error_bound(): the computed fine-grid node x^_k against
# the exact one x_k, with M = max(|a|, |b|) and h = |b - a|. chebyshev_matrices() forms
#     x^_k = fl( fl(fl(a + b) / 2) + fl( fl(fl(b - a) / 2) * t^_k ) ),
# and _chebyshev_base() forms t^_k = fl(sin(fl(fl(pi * j) / (2 (N - 1))))), j an exact integer:
#   * t^_k: pi's representation, the product and the quotient each err by at most u relatively,
#     so the sine's argument (|z| <= pi/2) errs by at most 3u * pi/2 ~ 4.7u, which moves sin by at
#     most as much; the sine itself is correct to 1 ulp, 2u: |t^_k - t_k| <= 6.7u, counted as 7u;
#   * the midpoint: fl(a + b) errs by u|a + b|, halving is exact: at most u M;
#   * the half-width: fl(b - a) errs by u h, halving is exact: at most u h / 2;
#   * the product (h/2) t^_k: the half-width's error times |t| <= 1, (h/2) times t^_k's 7u, and its
#     own rounding u h/2: at most (1/2 + 7/2 + 1/2) u h = 4.5 u h;
#   * the final sum: its own rounding, u |x_k| <= u M.
# Together |x^_k - x_k| <= 2 u M + 4.5 u h <= c_x u (M + h) with c_x = 5, the larger of the two
# counts (prompts/levin-phase-floor/06d-cc-roundoff-floor.md, section 5 point 1). Derived, not
# fitted.
_CC_NODE_ROUNDING_COUNT = 5

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


def _roundoff_floor(
    f_scale: float, width: float, G0: float, G1: float, k: float
) -> float:
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

    Levin regions only, since prompts/levin-phase-floor/06d-cc-roundoff-floor.md. The bound models
    the conditioning of the Levin collocation solve p' + i theta' p = f -- its 1/G0 is that solve's
    sensitivity where theta' is small. From levin-refactor prompt 04 until 06d the Clenshaw-Curtis
    fallback used it too; but the fallback solves no such system, and there G0 is theta' at the
    node nearest a stationary point, so the "floor" grew as a region shrank round the point and was
    +inf when one sat at a node ([07-cc-solves-use-the-levin-roundoff-floor]). The fallback now
    carries its own floor, _cc_roundoff_floor(), with no theta' in it.

    G0 = min|theta'| can be exactly, or numerically, zero at an interior stationary point that
    happens to coincide with (or land extremely close to) a sampled collocation node -- reachable
    (see prompt 04, note (b)) for a Levin region with large total variation but a turning point
    strictly inside it (a region with *small* total variation and a turning point is routed to the
    Clenshaw-Curtis fallback by the total-variation gate, and so never reaches this function --
    see _adaptive_levin_subregion_impl()).

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
    other large-but-finite value: _adaptive_levin()'s floor-limited test additionally requires a
    *finite* floor before it will stop refining a leaf on the strength of it (see the comment
    there), so an infinite floor here cannot itself freeze a region that genuine further bisection
    could still improve. Under the global budget of prompts/levin-phase-floor/
    06c-global-error-control.md it does the opposite: a leaf with an infinite floor has an infinite
    error, so it is chosen before every finite one and bisected until its floor is finite or it
    reaches depth_max. See prompt 04's log for the full numerical consequence of this choice,
    including the regression this specific threshold fixes.

    A *finite* floor could, until prompts/levin-phase-floor/06b-roundoff-exit-premise.md. The exit
    assumed that bisecting a region leaves its floor where it is. That holds where the floor is
    genuinely round-off -- the summed floor is roughly invariant under subdivision, as above, and
    grows where the k^2 term dominates -- but not where G1/G0 is large: there the floor comes from
    fitting one region over many decades of theta', halving the region cuts G1/G0 roughly to its
    square root when theta' grows exponentially, and the halves' floors sum to far less than the
    region's own. A finite floor now stops refinement of a leaf only if its halves' floors sum to at least
    _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its own (_roundoff_exit_premise_holds(); see
    that constant's comment for why 1/2).

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


def _cc_node_error_bound(x_span: Tuple[float, float]) -> float:
    """
    delta, a bound on |x^_k - x_k| for every node of chebyshev_matrices(x_span, n): the computed
    node against the exact one, c_x u (max(|a|, |b|) + |b - a|). See _CC_NODE_ROUNDING_COUNT for
    the count, term by term.
    """
    a, b = x_span
    return _CC_NODE_ROUNDING_COUNT * _UNIT_ROUNDOFF * (max(abs(a), abs(b)) + abs(b - a))


def _cc_roundoff_floor(
    weights: np.ndarray,
    f_sample: np.ndarray,
    basis_sample: np.ndarray,
    argument_sample: np.ndarray,
    node_error: float,
    theta_prime_bound: float,
    f_prime_sample: np.ndarray,
) -> float:
    """
    Round-off floor for a Clenshaw-Curtis fallback solve
    (prompts/levin-phase-floor/06d-cc-roundoff-floor.md), derived from what the rule computes:

        phi_CC = (n + 4) u S  +  u sum_k |w_k| sum_i |f_ik| |a_ik|
                 +  sum_k |w_k| delta_k sum_i ( |theta'| |f_ik| + |f'_ik| ),
        S = sum_k |w_k| sum_i |f_ik b_ik|,

    with w_k (k = 1 ... n) the fine rule's weights, f_ik = f_i(x_k) the amplitudes, b_ik the basis
    values and a_ik the argument handed to the basis function that gives b_ik (for the (sin, cos)
    basis, one argument per node for both components), and u = _UNIT_ROUNDOFF.

    * The first term bounds the rounding of the computed value. The n-term dot product contributes
      n u S (the standard gamma_n bound), and each term about one u from each of four further
      roundings -- see _CC_ROUNDOFF_PER_TERM_COUNT for the count.
    * The second term is the rounding of the argument: an error u |a_ik| in it moves sin or cos by
      at most that much. With a range-reduced phase (theta_mod_2pi) |a| < 2 pi and the term is
      negligible; with a raw phase it is u |theta(x_k)|, which is real.
    * The third term is the rounding of the nodes (added by 06d's section 5, after the first two
      were measured short by up to 3.4e9 on a range-reduced phase): the computed node x^_k is
      displaced from the exact one by up to delta_k (_cc_node_error_bound()), and a rule whose
      weights are fixed while its nodes move commits sum_k w_k g'(x_k) (x^_k - x_k), with
      g' = (f' + i theta' f) e^{i theta} for g = f e^{i theta}. Both parts are kept. |theta'| is
      bounded by its maximum over the region's coarse sample (theta_prime_bound), the conservative
      side; f' is the fine grid's Chebyshev derivative of the samples, per node. With a raw phase
      theta ~ theta' x this is about the second term again; with a reduced phase it is what the
      second term no longer carries. Summed over leaves it is about u |x| ||f|| TV(theta), so it
      does not grow under subdivision.

    Why not eq. (151) (_roundoff_floor()). Eq. (151) bounds the conditioning of the Levin
    collocation solve p' + i theta' p = f, whose sensitivity grows like 1/theta' where theta' is
    small. A fallback solve solves no such system: it is a weighted sum of 2N - 1 samples of
    sum_i f_i w_i(theta), whose rounding has no 1/theta' in it. Carried over from the Levin branch
    (prompts/levin-refactor/logs/04-roundoff-floor.md), eq. (151)'s G0 = min|theta'| made a fallback
    region's floor grow as the region shrank round a stationary point, and become +inf when one sat
    at a node ([07-cc-solves-use-the-levin-roundoff-floor]). This floor has no 1/theta' in it:
    theta' enters only the node term, as a factor.

    What it does not carry: an error in the phase function's own value, e.g. a theta_mod_2pi
    computed by rounding a large raw phase. That belongs in the caller's theta_abserr, as the
    module's contract says, and enters through _declared_endpoint_phase_err().

    :param weights: the fine rule's weights, shape (n,).
    :param f_sample: amplitudes, shape (m, n).
    :param basis_sample: basis values, shape (m, n).
    :param argument_sample: the arguments the basis values were computed from, shape (m, n).
    :param node_error: delta, the bound on every node's rounding (_cc_node_error_bound()).
    :param theta_prime_bound: a bound on |theta'| over the region: max|theta'| on its sample.
    :param f_prime_sample: the amplitudes' derivatives at the nodes, shape (m, n).
    """
    abs_weights = np.fabs(weights)
    n = abs_weights.shape[0]
    abs_f = np.fabs(f_sample)
    S = float(np.dot(abs_weights, np.sum(np.fabs(f_sample * basis_sample), axis=0)))
    argument_term = float(
        np.dot(abs_weights, np.sum(abs_f * np.fabs(argument_sample), axis=0))
    )
    node_term = node_error * float(
        np.dot(
            abs_weights,
            np.sum(theta_prime_bound * abs_f + np.fabs(f_prime_sample), axis=0),
        )
    )
    return (
        (n + _CC_ROUNDOFF_PER_TERM_COUNT) * _UNIT_ROUNDOFF * S
        + _UNIT_ROUNDOFF * argument_term
        + node_term
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


def _roundoff_exit_premise_holds(
    phase_err: float, phase_err_L: float, phase_err_R: float
) -> bool:
    """
    The round-off-limited exit's premise: would bisecting this region leave its floor materially
    where it is? True if its two halves' floors ("phase_err", as each half's own subregion solve
    reports it) sum to at least _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO times its own -- see that
    constant's comment for the argument and the value.

    _adaptive_levin() asks this only once every other condition of the exit holds, so phase_err is
    always finite here. A half's floor can be +inf (an interior stationary point at or near one of
    its nodes; see _roundoff_floor()); the sum is then +inf, the premise holds and the exit may
    fire, exactly as it did before the premise was checked.
    """
    return (
        phase_err_L + phase_err_R
    ) >= _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO * phase_err


def _error_for_ordering(err: float) -> float:
    """
    A leaf's error as _adaptive_levin()'s global budget orders and sums it. A NaN error is read as
    +inf: such a leaf is chosen before every finite one, and while one is present the budget
    cannot be met. No path in this module is known to produce one -- every value is checked finite
    and every error is an absolute value, a max or a sum of finite or +inf terms -- but a NaN
    compares False with everything, so left as it is it would sink to the bottom of the order and
    drop out of the sum unnoticed. +inf is the reading that can only cost refinement, never accept
    a result.
    """
    if err != err:
        return float("inf")
    return err


class _ExactSum:
    """
    A running sum of floats held exactly, as a list of non-overlapping partials (Shewchuk's
    algorithm, as in the Python recipe "msum"): add() is exact, subtraction included, and total()
    is the correctly rounded sum of everything added. _adaptive_levin()'s global budget adds and
    removes one leaf's value and error per bisection, often thousands of times, while the total it
    tests can fall by many orders of magnitude below the terms that were added and removed on the
    way; a plain float accumulator would carry their rounding into the test. Only finite values
    are added here; the caller counts infinite errors separately.
    """

    __slots__ = ("_partials",)

    def __init__(self):
        self._partials = []

    def add(self, x: float) -> None:
        x = float(x)
        partials = self._partials
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]

    def total(self) -> float:
        return math.fsum(self._partials)


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
        # phase_err is the round-off floor -- Chen et al. eq. (151) (_roundoff_floor()) on a Levin
        # region, the rule's own (_cc_roundoff_floor()) on a fallback region -- plus any
        # declared-theta_abserr endpoint contribution (_declared_endpoint_phase_err()) -- an
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

        # The trust check one level down (prompts/levin-phase-floor/06c-global-error-control.md,
        # section 8, the third amendment; see _adaptive_levin()). delta1 = |V_P - (V_L + V_R)|,
        # delta2 = |(V_L + V_R) - (V_LL + V_LR + V_RL + V_RR)|, convergence_ratio = delta2 /
        # delta1, all None until the check is taken. trusted is None until then (and for a
        # floor-limited region, which is exempt), True if convergence_ratio < 1, False otherwise.
        # declared_resolution, when set, replaces abserr in total_err: max(abserr, delta1) /
        # (1 - r) for a trusted region, max(abserr, delta1 + delta2) for an untrusted one at
        # depth_max. abserr itself stays the raw estimate (the step-(4) residual, or the nested
        # pair), as do abserr_resolution/abserr_fallback.
        self._delta1 = None
        self._delta2 = None
        self._convergence_ratio = None
        self._trusted = None
        self._declared_resolution = None

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

        if self._trusted is not None:
            phase_label = phase_label + (
                f", r={self._convergence_ratio:.4g}"
                + ("" if self._trusted else " (UNTRUSTED)")
            )

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
        Round-off floor for this region -- Chen et al. eq. (151) (_roundoff_floor()) on a Levin
        region, the Clenshaw-Curtis rule's own (_cc_roundoff_floor()) on a fallback region -- plus
        any declared-theta_abserr endpoint contribution. This is *not* included in abserr; the
        total error of the region is max(abserr, phase_err) -- see total_err. Can be +inf on a
        Levin region; see _roundoff_floor()'s docstring for when and why. A fallback region's is
        finite whenever its samples are.
        """
        return self._phase_err

    @property
    def phase_limited(self) -> bool:
        """
        True if _adaptive_levin() chose this region (leaf) for refinement while the call's error
        budget was unmet, and found it floor-limited: its round-off floor is finite, its residual
        is at most that floor, and its two halves' floors sum to at least
        _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its own, so bisecting it would not lower its
        error materially. Never True when phase_err is not finite. A region the loop never chose
        is never flagged, whatever its floor -- see _adaptive_levin().
        """
        return self._phase_limited

    @property
    def delta1(self) -> Optional[float]:
        """|V_P - (V_L + V_R)| from the trust check one level down; None if it was not taken."""
        return self._delta1

    @property
    def delta2(self) -> Optional[float]:
        """|(V_L + V_R) - (V_LL + V_LR + V_RL + V_RR)| from the trust check; None if not taken."""
        return self._delta2

    @property
    def convergence_ratio(self) -> Optional[float]:
        """
        r = delta2 / delta1, the observed convergence ratio one level down (0 if both are 0;
        +inf if only delta1 is 0, or either is not finite); None if the check was not taken.
        """
        return self._convergence_ratio

    @property
    def trusted(self) -> Optional[bool]:
        """
        Whether this region's error estimate passed the trust check one level down (r < 1). None
        if the check was not taken on it: a floor-limited region is exempt, and a region at
        depth_max returned by a call whose budget was never met need not have been checked. A
        region returned with trusted False is at depth_max, and declares
        max(abserr, delta1 + delta2) as its resolution error; see _adaptive_levin().
        """
        return self._trusted

    @property
    def declared_resolution(self) -> Optional[float]:
        """
        The resolution error this region declares in total_err when the trust check changed it:
        max(abserr, delta1) / (1 - r) if trusted, max(abserr, delta1 + delta2) if untrusted at
        depth_max; None if total_err uses abserr as it stands.
        """
        return self._declared_resolution

    @property
    def total_err(self) -> Optional[float]:
        """
        Combined per-region error estimate: max(resolution/fallback residual, round-off floor),
        plus the truncation term (prompt 06) if this region has one. This is the leaf error e_i of
        _adaptive_levin()'s global budget: the loop stops when the sum of it over the leaves meets
        max(atol, rtol*|value|), and the call reports the same sum as "abserr". The truncation term is added
        rather than maxed because it is a distinct, always-present bias -- the endpoint
        contribution of modes the rtol gate discarded from this region's own value -- not another
        estimate of the same underlying quantity the max() is choosing between.

        The resolution error in the max is declared_resolution when the trust check one level
        down has set it (widened by the observed convergence ratio), else abserr.
        """
        resolution = (
            self._declared_resolution
            if self._declared_resolution is not None
            else self._abserr
        )
        if resolution is None:
            base = self._phase_err
        elif self._phase_err is None:
            base = resolution
        else:
            base = max(resolution, self._phase_err)

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
        Round-off floor (Chen et al. eq. (151) on a Levin region, _cc_roundoff_floor() on a
        fallback region, plus any declared-theta_abserr endpoint term).
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


class _levin_leaf:
    """
    One leaf of _adaptive_levin()'s current partition (prompts/levin-phase-floor/
    06c-global-error-control.md, section 1 item 1).

    A leaf is solved and its error known from the moment it is made: "region" is the
    used_interval the call will return for it, whose total_err is the leaf's error e_i, and
    "value" is what it contributes to the call's value -- the region's own solve on the Levin
    branch, CC_{2N-1} on the Clenshaw-Curtis fallback. "dataL"/"dataR" are its two halves'
    solves when they are in hand: on the Levin branch always (the step-(4) residual needs them),
    on the fallback branch only once the round-off premise has been checked on it. If the leaf is
    bisected they become its children's own solves, so no half is ever solved twice.

    "grandchildren" holds the solves of its halves' halves (LL, LR, RL, RR) once the trust check
    one level down has been taken on it; if the leaf is then bisected they become its children's
    halves, so they are not solved twice either. "checked" records that the check was taken, and
    "version" invalidates a stale entry for the leaf on the candidates' heap when its error or its
    trust changes.

    The three histories feed the depth-18 diagnostic log, as before.
    """

    __slots__ = (
        "start",
        "end",
        "depth",
        "seq",
        "value",
        "region",
        "is_direct",
        "data",
        "dataL",
        "dataR",
        "abserr_history",
        "relerr_history",
        "p_ratios_history",
        "grandchildren",
        "checked",
        "version",
    )

    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        seq: int,
        value: float,
        region: "used_interval",
        is_direct: bool,
        data: Optional[dict],
        dataL: Optional[dict],
        dataR: Optional[dict],
        abserr_history: "ErrorHistoryType",
        relerr_history: "ErrorHistoryType",
        p_ratios_history: "PScaleHistoryType",
    ):
        self.start = start
        self.end = end
        self.depth = depth
        self.seq = seq
        self.value = value
        self.region = region
        self.is_direct = is_direct
        self.data = data
        self.dataL = dataL
        self.dataR = dataR
        self.abserr_history = abserr_history
        self.relerr_history = relerr_history
        self.p_ratios_history = p_ratios_history
        self.grandchildren = None
        self.checked = False
        self.version = 0

    @property
    def width(self) -> float:
        return np.fabs(self.end - self.start)

    @property
    def break_point(self) -> float:
        return (self.end + self.start) / 2.0

    @property
    def error(self) -> float:
        """This leaf's e_i: used_interval.total_err of its region."""
        return self.region.total_err


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

    def eval_basis_and_arguments(self, x):
        """
        The basis at x, as eval_basis(), together with the argument handed to each component's
        sin or cos: (values, arguments), each a list with one entry per component. The phase is
        evaluated once, not once per component. Used by the Clenshaw-Curtis fallback, whose
        round-off floor counts the rounding of the argument (_cc_roundoff_floor()); the values are
        bit-identical to eval_basis(x)'s.
        """
        if hasattr(self, "_theta_mod_2pi"):
            argument = self._theta_mod_2pi(x)
        else:
            argument = self._theta(x)
        return [np.sin(argument), np.cos(argument)], [argument, argument]


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
        [_sample_vectorized(func, grid, vectorize_cache, ("f", id(func))) for func in f]
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
    The round-off floor, "phase_err", is this rule's own (_cc_roundoff_floor(), since
    prompts/levin-phase-floor/06d-cc-roundoff-floor.md): the rounding of the weighted sum, of
    the argument handed to sin and cos, and of the nodes, from the fine-grid samples -- plus the
    declared-phase term. It is not Chen et al. eq. (151), which the Levin branch uses: that bound is the
    conditioning of the Levin collocation solve, with a 1/min|theta'| this rule does not have.

    :param theta_prime_Cheb: theta' sampled at the order-chebyshev_order collocation points,
        likewise already computed by the caller's call to build_Levin_data(). Its maximum bounds
        |theta'| in the round-off floor's node term (until 06d it gave eq. (151)'s G0/G1).
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
    fine_grid, D_fine = chebyshev_matrices(x_span, fine_order)

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
    # eval_basis_and_arguments() prefers theta_mod_2pi when the caller supplied one, as
    # eval_basis() does, and also returns the argument it handed to sin and cos, which the
    # round-off floor below needs -- from the same single evaluation of the phase per node.
    basis_and_arguments = [BasisData.eval_basis_and_arguments(x) for x in fine_grid]
    basis_fine = np.array([values for values, _ in basis_and_arguments]).T
    arguments_fine = np.array(
        [arguments for _, arguments in basis_and_arguments], dtype=float
    ).T

    integrand_fine = np.sum(f_fine * basis_fine, axis=0)

    w_fine = _cc_weights(x_span, fine_order)
    w_coarse = _cc_weights(x_span, chebyshev_order)

    value_fine = float(np.dot(w_fine, integrand_fine))
    value_coarse = float(np.dot(w_coarse, integrand_fine[::2]))

    abserr_direct = float(np.fabs(value_fine - value_coarse))

    # Round-off floor of this rule, derived from what it computes (_cc_roundoff_floor();
    # prompts/levin-phase-floor/06d-cc-roundoff-floor.md), from the fine-grid samples already in
    # hand: no extra evaluation of anything. Not Chen et al. eq. (151) (_roundoff_floor()), which
    # this branch used from levin-refactor prompt 04 until 06d: that bound models the conditioning
    # of the Levin collocation solve, with its 1/min|theta'|, and this branch solves no such
    # system; carried over, it made a fallback region's floor grow, or become +inf, at a
    # stationary point of the phase.
    #
    # Its node term takes |theta'| as its maximum over the coarse theta_prime_Cheb the phase_span
    # gate already paid for, and f' from the fine grid's own Chebyshev differentiation matrix
    # applied to the samples in hand.
    f_prime_fine = np.vstack([np.matmul(D_fine, f_row) for f_row in f_fine])
    round_off_err = _cc_roundoff_floor(
        w_fine,
        f_fine,
        basis_fine,
        arguments_fine,
        _cc_node_error_bound(x_span),
        float(np.max(np.fabs(theta_prime_Cheb))),
        f_prime_fine,
    )

    # Optional declared-theta_abserr endpoint term (recommendation 5.2), 0.0 unless the phase dict
    # supplied "theta_abserr". This fallback has no Levin antiderivative p to weight by, so it
    # reuses the same f_scale * (width / 2) proxy per endpoint that the pre-prompt-04 code used
    # for the whole interval (a crude sup-norm bound, not a per-endpoint quantity) -- see prompt
    # 04's log for why this, rather than the Levin branch's true per-endpoint p_endpoint_l1, was
    # judged an acceptable approximation here. With f_scale = max_grid sqrt(sum_i f_i^2) over the
    # fine-grid samples, it bounds sum_k |w_k| |f_k| delta(theta) from above per endpoint, since
    # the weights are positive and sum to the width.
    width = np.fabs(x_span[1] - x_span[0])
    f_scale = float(np.max(np.sqrt(np.sum(f_fine * f_fine, axis=0))))
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


def _history_improvement(previous: float, current: float) -> Optional[float]:
    """
    previous / current for the depth-18 history log, or None where the ratio is not defined: the
    denominator zero or not finite. An ancestor's abserr can be exactly 0 -- under the global
    budget a leaf with a zero residual can still be split, by the trust check or a dominating floor
    -- and the plain division raised ZeroDivisionError from logging code
    ([07-depth18-diagnostic-divides-by-zero]; prompts/levin-phase-floor/06d-cc-roundoff-floor.md).
    """
    if current is None or previous is None:
        return None
    if current == 0.0 or not np.isfinite(current):
        return None
    return previous / current


def _log_depth_history(
    depth: int,
    abserr_history: "ErrorHistoryType",
    relerr_history: "ErrorHistoryType",
    p_ratios_history: "PScaleHistoryType",
) -> None:
    """
    The depth-18 diagnostic: log a deep leaf's ancestry, level by level, with each level's abserr
    and relerr and their improvement on the level above. An improvement whose denominator is zero
    or not finite is logged as not defined (_history_improvement()). Logging only: no value,
    error or decision depends on it.
    """

    def _format_improvement(improvement: Optional[float]) -> str:
        return "not defined" if improvement is None else f"{improvement:.3g}"

    prev_abs = None
    prev_rel = None

    for i in range(0, depth):
        this_abs = abserr_history[i]
        this_rel = relerr_history[i]
        this_p_ratios = p_ratios_history[i]

        if i == 0:
            _logger.info(
                "   -- %d. abserr=%.5g, relerr=%.5g, p-ratios=[%s]",
                i + 1,
                this_abs,
                this_rel,
                ", ".join(f"{p:.4g}" for p in this_p_ratios),
            )

        else:
            abs_improvement = _history_improvement(prev_abs, this_abs)
            rel_improvement = _history_improvement(prev_rel, this_rel)
            _logger.info(
                "   -- %d. abserr=%.5g (improvement=%s), relerr=%.5g "
                "(improvement=%s), p-ratios=[%s]",
                i + 1,
                this_abs,
                _format_improvement(abs_improvement),
                this_rel,
                _format_improvement(rel_improvement),
                ", ".join(f"{p:.3g}" for p in this_p_ratios),
            )

        prev_abs = this_abs
        prev_rel = this_rel


def _sum_leaves_ascending(leaves_ascending) -> Tuple[float, float]:
    """
    The call's value and abserr, each summed over the leaves in ascending x (the order the
    returned "regions" list is in), with plain floating-point addition. _adaptive_levin() computes
    its returned "value", "abserr" and "converged" from these, and confirms its running stop test
    against them before it stops, so a call that stops on the budget reports "converged" True by
    the same arithmetic it returns.
    """
    val = 0.0
    abserr_total = 0.0
    for leaf in leaves_ascending:
        val = val + leaf.value
        contribution = leaf.region.total_err
        if contribution is not None:
            abserr_total = abserr_total + contribution
    return val, abserr_total


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
    _test_only_bypass_roundoff_premise: bool = False,
):
    """
    The adaptive driver, under one error budget for the whole call
    (prompts/levin-phase-floor/06c-global-error-control.md). Global adaptivity, as QUADPACK's QAG:
    the driver keeps every *leaf* of the current partition of x_span, each with its value v_i and
    its error e_i (used_interval.total_err: max(residual, round-off floor) + truncation), and
    decides only against the whole call:

      1. Let E = sum_i e_i and V = sum_i v_i over the leaves. If E <= max(atol, rtol*|V|), stop:
         converged.
      2. Otherwise choose the *refinable* leaf with the largest e_i (ties: the smaller start), and
         replace it by its two halves.
      3. If no leaf is refinable, stop: unconverged.

    A leaf is refinable unless one of these holds:
      * it is at depth_max;
      * it is *floor-limited*: its round-off floor is finite, its residual is at most that floor,
        and 06b's premise holds -- its halves' floors sum to at least rho = 1/2 of its own
        (_roundoff_exit_premise_holds()) -- so bisecting it would not lower its error
        materially. Checked only on the leaf chosen; flagged used_interval.phase_limited.
    A leaf found not refinable is never chosen again. No rule stops a leaf because bisecting it has
    stopped lowering its error: a leaf whose error does not fall with h is chased to depth_max if it
    holds the total back, at a cost in work but not in honesty. (A stall rule was tried and
    withdrawn: it judged splitting by an error estimate that cannot yet be trusted on
    pre-asymptotic leaves and on Levin leaves across a stationary point, and stopped refinement
    where it works -- prompts/levin-phase-floor/06c-global-error-control.md, section 8.)

    The trust check one level down (section 8, the third amendment). The budget trusts each leaf's
    error estimate, and an estimate taken before the leaf is resolved can understate its error:
    the step-(4) residual |V_P - (V_L + V_R)| is V_P's error only if V_L + V_R is much better than
    V_P. So, for each leaf, from its halves and its grandchildren (each valued by its own branch's
    rule), delta1 = |V_P - (V_L + V_R)|, delta2 = |(V_L + V_R) - (V_LL + V_LR + V_RL + V_RR)| and
    r = delta2 / delta1. If the error falls from this level on at least as fast as observed, the
    leaf's error is at most delta1 / (1 - r), so:
      * a trusted leaf (r < 1) declares max(residual, delta1) / (1 - r) as its resolution error;
      * an untrusted leaf (r >= 1) is refinable whatever the budget says, and is chosen before
        every trusted leaf; the loop does not stop while a refinable leaf is untrusted. One at
        depth_max declares max(residual, delta1 + delta2), is flagged (used_interval.trusted
        False), and the call is not converged while one is returned;
      * a floor-limited leaf is exempt: its differences are round-off noise, which its floor
        bounds.
    The check is taken lazily: when the budget passes, every leaf not yet checked is checked, and
    a widened error or an untrusted leaf reopens the loop. A call that stops on the budget returns
    every leaf checked or floor-limited. The grandchildren are carried forward as the children's
    halves if a leaf is later bisected, and a leaf's value is unchanged. It does not reach an error
    common to a leaf and all its descendants ([06b-moved-calls-declare-less-than-their-error]).

    So the loop stops on exactly the quantity the call reports as "abserr", and "converged" keeps
    its formula, abserr <= max(atol, rtol*|value|), with no untrusted leaf returned. This
    replaces the earlier scheme, which
    accepted each region against its own value (relerr < rtol) or against atol scaled by its share
    of the span's length, and compared the summed error with the call's request only afterwards.
    Where the regions' values cancel, sum|v_i| >> |sum v_i|, per-region relative accuracy does not
    add up to the same relative accuracy in the total ([07-per-region-acceptance-misses-a-
    cancelling-total]); and a region accepted on its own residual could keep an eq. (151) floor
    that its halves would cut by orders ([06b-resolved-region-keeps-a-floor-bisection-would-cut]).
    Under one budget, the target is rtol times the current total; an inflated floor is the
    largest e_i, and is bisected where the premise fails; an infinite floor (a stationary point at
    a node) is always chosen first, and is bisected until it is finite or reaches depth_max.

    Termination: every bisection deepens a leaf, and a leaf at depth_max is never chosen.
    """
    # _test_only_bypass_roundoff_premise: TEST-ONLY. True skips 06b's premise
    # (_roundoff_exit_premise_holds()) in the floor-limited test, so that a leaf chosen while the
    # budget is unmet is found floor-limited on its floor and residual alone, without checking
    # that bisection would not lower its floor -- the round-off-limited exit's own test as it was
    # before prompts/levin-phase-floor/06b-roundoff-exit-premise.md, now inside the global budget.
    # It exists so the tests and calibration scripts can run that behaviour beside the default; it
    # is read only at the floor-limited test below, and adaptive_levin_sincos() does not expose
    # it. Never set it in production code.
    #
    # Input validation (rec 4, C8, C9). This module cannot deliver a purely relative-error
    # contract: the round-off floor computed by _roundoff_floor() is absolute by nature, so a
    # caller asking for atol=0 leaves the relative-error denominator floor at :relerr_denom inert,
    # and the budget max(atol, rtol*|V|) at zero for an integrand whose value is zero, and was
    # measured to subdivide an identically-zero integrand to the full depth limit (256 regions /
    # 1023 solves at depth_max=8, against 1 region / 3 solves at atol=1e-15). Reject rather than
    # silently accepting a request this module cannot honour.
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

    num_evaluations = 0

    num_SVD_errors = 0
    num_order_changes = 0
    num_direct_solves = 0
    chebyshev_min_order = None
    max_depth = 0

    # True solve counts (rec 11, C10), accumulated from every subregion solve actually attempted:
    # each leaf's own solve where it is not carried forward from its parent, the two comparison
    # halves (dataL, dataR) every Levin leaf solves for its step-(4) residual, and the two halves
    # a fallback leaf solves when it is checked for the round-off premise or bisected. Each solve
    # is counted once, where it is made; a half carried forward as a child's own solve is not
    # counted again. num_direct_solves (below, kept for backward compatibility -- see its own
    # note) is not this total: it accumulates only from each leaf's own solve. See
    # prompts/levin-refactor/logs/07-diagnostics-hygiene.md.
    num_solves_direct = 0
    num_solves_lstsq = 0
    num_solves_pinv = 0

    num_history_messages = 0

    # Counters of the global budget's own decisions, reported at DEBUG level when the call ends
    # (prompts/levin-phase-floor/06c-global-error-control.md, section 1 item 3, and section 8's
    # third amendment): how many bisections were made; how often the running stop test passed but
    # the returned sums did not; how many trust checks were taken, in how many passes, and how
    # many leaves they found untrusted.
    num_splits = 0
    num_resum_rejections = 0
    num_checks = 0
    num_check_passes = 0
    num_untrusted_found = 0

    def _solve(span, description):
        nonlocal num_evaluations, num_solves_direct, num_solves_lstsq, num_solves_pinv
        try:
            solve_data = _adaptive_levin_subregion(
                span,
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
                "!! adaptive_levin (%s): linear algebra error when estimating %s (%s, %s), "
                "width=%.8g",
                _format_label(notify_label, id_label),
                description,
                span[0],
                span[1],
                np.fabs(span[1] - span[0]),
            )
            raise e

        # True-solve accounting (rec 11, C10): counted here, where the solve is made, and not
        # again if this solve is later carried forward as a child leaf's own.
        num_evaluations += 1
        num_solves_direct += solve_data["metadata"].get("direct_solve", 0)
        num_solves_lstsq += solve_data["metadata"].get("lstsq_solve", 0)
        num_solves_pinv += solve_data["metadata"].get("pinv_solve", 0)
        return solve_data

    seq_counter = 0

    def _make_leaf(
        a,
        b,
        depth,
        data,
        abserr_history,
        relerr_history,
        p_ratios_history,
        halves=None,
    ) -> _levin_leaf:
        """
        Make a leaf of [a, b] at this depth, and compute its error. data is its own solve when it
        is carried forward from its parent (one of the parent's halves), else None (the root),
        when it is solved here. A Levin leaf solves its two halves for its step-(4) residual (Chen
        et al. (173)); a fallback leaf's error is its own nested-pair estimate. halves, when
        given, are the leaf's own halves' solves, carried forward from its parent's trust check
        (the parent's grandchildren), and are used rather than solved again.
        """
        nonlocal seq_counter, max_depth, num_history_messages
        nonlocal chebyshev_min_order, num_SVD_errors, num_order_changes, num_direct_solves

        if depth > max_depth:
            max_depth = depth

        if depth >= 18 and num_history_messages < 20:
            num_history_messages += 1

            _logger.info(
                "@@ adaptive_levin (%s): encountered subinterval of depth %d (notification "
                "%d/20 for this quadrature)",
                _format_label(notify_label, id_label),
                depth,
                num_history_messages,
            )

            _log_depth_history(depth, abserr_history, relerr_history, p_ratios_history)

        # Chen et al. (172). A leaf made by bisecting a parent carries its own solve forward from
        # the parent's halves, and only the root is solved here. This single call also makes the
        # Levin-vs-fallback decision (C2): _adaptive_levin_subregion samples theta' first, gates
        # on the region's total phase variation, and only then either solves the Levin system or
        # evaluates the nested Clenshaw-Curtis pair -- see _adaptive_levin_subregion_impl(). The
        # two outcomes are told apart below by data["is_direct"].
        if data is None:
            data = _solve((a, b), "Levin subregion")

        # Per-region bookkeeping, from each leaf's own solve, once per leaf -- including leaves
        # that are later bisected. The metadata of a half (dataL/dataR) is accumulated here when
        # it becomes a leaf, and never before.
        order = data["metadata"].get("chebyshev_order", None)
        if order is not None:
            if chebyshev_min_order is None or order < chebyshev_min_order:
                chebyshev_min_order = order

        num_SVD_errors = num_SVD_errors + data["metadata"].get("SVD_errors", 0)
        num_order_changes = num_order_changes + data["metadata"].get(
            "num_order_changes", 0
        )
        num_direct_solves = num_direct_solves + data["metadata"].get("direct_solve", 0)

        estimate = data["value"]

        # Round-off floor for this leaf, from its own solve: it is its own endpoints and grid that
        # enter the call's value. See _roundoff_floor().
        phase_err = data.get("phase_err", 0.0) or 0.0

        is_direct = bool(data.get("is_direct", False))
        if is_direct:
            # Nested Clenshaw-Curtis fallback (C2, C7). This leaf's error estimate --
            # |CC_{2N-1} - CC_N| -- comes from a nested pair evaluated entirely within it, not from
            # a comparison against halves, so none are solved here. They are solved when the leaf
            # is chosen: for the round-off premise, and as its children if it is bisected --
            # unless its parent's trust check has already solved them, when they are carried.
            dataL, dataR = halves if halves is not None else (None, None)
            abserr = data["abserr_direct"]

            # relerr is a diagnostic only -- the leaf's own error over its own value, with the
            # denominator floored at atol -- and no longer decides anything.
            relerr_denom = max(np.fabs(estimate), atol)
            relerr = abserr / relerr_denom

            region = used_interval(
                start=a,
                end=b,
                depth=depth,
                type=INTERVAL_TYPE_DIRECT,
                abserr=abserr,
                relerr=relerr,
                phase_err=phase_err,
                # no Levin antiderivative on this branch -- see used_interval.abserr_truncation.
                abserr_truncation=data.get("abserr_truncation"),
            )
        else:
            # Chen et al. (173): the step-(4) residual against the leaf's two halves. They are
            # kept, and become its children's own solves if it is bisected, so that they are not
            # solved a second time. If its parent's trust check solved them already, they are
            # carried.
            if halves is not None:
                dataL, dataR = halves
            else:
                c = (b + a) / 2.0
                dataL = _solve((a, c), "Levin left-comparison region")
                dataR = _solve((c, b), "Levin right-comparison region")

            refined_estimate = dataL["value"] + dataR["value"]
            abserr = np.fabs(estimate - refined_estimate)

            # relerr is a diagnostic only, and no longer decides anything. Its denominator is
            # guarded: the integrand can pass through an accidental zero, and there
            # min(|estimate|, |refined_estimate|) is arbitrarily small, so floor it at atol. (min()
            # rather than max() of the two estimates is retained: it is the more conservative
            # choice.)
            relerr_denom = max(min(np.fabs(estimate), np.fabs(refined_estimate)), atol)
            relerr = abserr / relerr_denom

            region = used_interval(
                start=a,
                end=b,
                depth=depth,
                type=INTERVAL_TYPE_LEVIN,
                abserr=abserr,
                relerr=relerr,
                p_ratios=data["p_ratios"],
                phase_err=phase_err,
                # data is this leaf's own solve (not dataL/dataR): its "value" is what the leaf
                # contributes, so data["abserr_truncation"] is the truncation actually incurred
                # by that value (prompt 06, problem (b)).
                abserr_truncation=data.get("abserr_truncation"),
            )

        seq_counter += 1
        return _levin_leaf(
            start=a,
            end=b,
            depth=depth,
            seq=seq_counter,
            value=estimate,
            region=region,
            is_direct=is_direct,
            # the leaf's own solve is kept only for its p_sample; its value is already in hand
            data=data if build_p_sample else None,
            dataL=dataL,
            dataR=dataR,
            abserr_history=abserr_history,
            relerr_history=relerr_history,
            p_ratios_history=p_ratios_history,
        )

    # The current partition: every leaf, by its sequence number. The candidates for bisection are
    # the refinable leaves, on a heap ordered first by trust -- a leaf the trust check found
    # untrusted before every other -- then by error, largest first, ties broken by the smaller
    # start and then by the sequence number, so the order is total and deterministic. A leaf at
    # depth_max, or found floor-limited, is never a candidate. When a leaf's error or trust
    # changes it is pushed again under a new version, and an entry whose version is stale, or whose
    # leaf has been bisected, is skipped when popped.
    leaves = {}
    candidates = []
    untrusted_refinable = set()

    # Running sums over the leaves, held exactly (_ExactSum): the finite errors, the count of
    # errors that are not finite (+inf, or NaN read as +inf -- see _error_for_ordering()), and the
    # values.
    error_sum = _ExactSum()
    value_sum = _ExactSum()
    num_nonfinite_errors = 0
    num_direct_leaves = 0

    def _add_leaf(leaf: _levin_leaf) -> None:
        nonlocal num_nonfinite_errors, num_direct_leaves
        leaves[leaf.seq] = leaf
        err = _error_for_ordering(leaf.error)
        if np.isfinite(err):
            error_sum.add(err)
        else:
            num_nonfinite_errors += 1
        value_sum.add(leaf.value)
        if leaf.is_direct:
            num_direct_leaves += 1
        _push_candidate(leaf)

    def _refinable(leaf: _levin_leaf) -> bool:
        return leaf.depth < depth_max and not leaf.region._phase_limited

    def _push_candidate(leaf: _levin_leaf) -> None:
        leaf.version += 1
        if _refinable(leaf):
            untrusted = leaf.region._trusted is False
            if untrusted:
                untrusted_refinable.add(leaf.seq)
            else:
                untrusted_refinable.discard(leaf.seq)
            err = _error_for_ordering(leaf.error)
            heapq.heappush(
                candidates,
                (0 if untrusted else 1, -err, leaf.start, leaf.seq, leaf.version),
            )
        else:
            untrusted_refinable.discard(leaf.seq)

    def _pop_candidate() -> Optional[_levin_leaf]:
        while candidates:
            _, _, _, seq, version = heapq.heappop(candidates)
            leaf = leaves.get(seq)
            if leaf is not None and leaf.version == version:
                untrusted_refinable.discard(seq)
                return leaf
        return None

    def _remove_error(leaf: _levin_leaf) -> None:
        nonlocal num_nonfinite_errors
        err = _error_for_ordering(leaf.error)
        if np.isfinite(err):
            error_sum.add(-err)
        else:
            num_nonfinite_errors -= 1

    def _restore_error(leaf: _levin_leaf) -> None:
        nonlocal num_nonfinite_errors
        err = _error_for_ordering(leaf.error)
        if np.isfinite(err):
            error_sum.add(err)
        else:
            num_nonfinite_errors += 1

    def _remove_leaf(leaf: _levin_leaf) -> None:
        nonlocal num_nonfinite_errors, num_direct_leaves
        del leaves[leaf.seq]
        untrusted_refinable.discard(leaf.seq)
        _remove_error(leaf)
        value_sum.add(-leaf.value)
        if leaf.is_direct:
            num_direct_leaves -= 1

    def _running_sums() -> Tuple[float, float]:
        running_error = float("inf") if num_nonfinite_errors > 0 else error_sum.total()
        return value_sum.total(), running_error

    def _leaves_ascending() -> List[_levin_leaf]:
        return sorted(leaves.values(), key=lambda leaf: leaf.start)

    def _solve_halves(leaf: _levin_leaf) -> None:
        # A fallback leaf has no halves of its own until it is chosen: they are solved here, on
        # demand, each a full subregion solve on its own samples -- the same call that would
        # produce it as a leaf, so its "phase_err" is the floor it would carry -- and they are
        # kept, to become the children's own solves if the leaf is bisected.
        c = leaf.break_point
        leaf.dataL = _solve((leaf.start, c), "left half of fallback region")
        leaf.dataR = _solve((c, leaf.end), "right half of fallback region")

    def _check_leaf(leaf: _levin_leaf) -> None:
        """
        The trust check one level down (prompts/levin-phase-floor/06c-global-error-control.md,
        section 8, the third amendment). With V each solve's own value (the region's own solve on
        the Levin branch, CC_{2N-1} on the fallback):

            delta1 = |V_P - (V_L + V_R)|,
            delta2 = |(V_L + V_R) - (V_LL + V_LR + V_RL + V_RR)|,
            r = delta2 / delta1   (0 if both are 0; +inf if only delta1 is 0, or either is not
                                   finite).

        If the error falls from this level on at least as fast as observed, the leaf's error is at
        most delta1 (1 + r + r^2 + ...) = delta1 / (1 - r), so a trusted leaf (r < 1) declares
        max(abserr, delta1) / (1 - r) as its resolution error. For r >= 1 no such bound exists: the
        leaf is untrusted, refinable whatever the budget says and chosen before every trusted
        leaf; at depth_max, where it cannot be refined, it declares max(abserr, delta1 + delta2)
        and the call is not converged. The grandchildren are kept, to become its children's halves
        if it is bisected. Its value is unchanged.
        """
        nonlocal num_checks, num_untrusted_found
        if leaf.dataL is None:
            _solve_halves(leaf)
        dataL, dataR = leaf.dataL, leaf.dataR
        a = leaf.start
        b = leaf.end
        c = leaf.break_point
        cL = (c + a) / 2.0
        cR = (b + c) / 2.0
        LL = _solve((a, cL), "trust check: left half of left half")
        LR = _solve((cL, c), "trust check: right half of left half")
        RL = _solve((c, cR), "trust check: left half of right half")
        RR = _solve((cR, b), "trust check: right half of right half")
        leaf.grandchildren = ((LL, LR), (RL, RR))

        halves_sum = dataL["value"] + dataR["value"]
        quarters_sum = (LL["value"] + LR["value"]) + (RL["value"] + RR["value"])
        delta1 = float(np.fabs(leaf.value - halves_sum))
        delta2 = float(np.fabs(halves_sum - quarters_sum))
        if not (np.isfinite(delta1) and np.isfinite(delta2)):
            ratio = float("inf")
        elif delta1 == 0.0:
            ratio = 0.0 if delta2 == 0.0 else float("inf")
        else:
            ratio = delta2 / delta1

        region = leaf.region
        residual = region.abserr
        _remove_error(leaf)
        region._delta1 = delta1
        region._delta2 = delta2
        region._convergence_ratio = ratio
        if ratio < 1.0:
            region._trusted = True
            region._declared_resolution = max(residual, delta1) / (1.0 - ratio)
        else:
            region._trusted = False
            num_untrusted_found += 1
            region._declared_resolution = (
                max(residual, delta1 + delta2) if leaf.depth >= depth_max else None
            )
        _restore_error(leaf)
        leaf.checked = True
        num_checks += 1
        _push_candidate(leaf)

    def _exempt_floor_limited(leaf: _levin_leaf) -> None:
        # A floor-limited leaf is exempt from the trust check: its differences are round-off
        # noise, and the floor bounds them. If the check was taken on it before it was found
        # floor-limited, its declared resolution error reverts to its own estimate; the
        # differences stay recorded.
        region = leaf.region
        if region._trusted is not None or region._declared_resolution is not None:
            _remove_error(leaf)
            region._trusted = None
            region._declared_resolution = None
            _restore_error(leaf)
        leaf.grandchildren = None

    _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))

    running_stop = False

    while True:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            current_val, _ = _running_sums()
            _notify_progress(
                now,
                last_notify,
                start_time,
                current_val,
                len(leaves),
                sum(1 for candidate in leaves.values() if _refinable(candidate)),
                num_direct_leaves,
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
                    [
                        leaves[entry[3]]
                        for entry in sorted(candidates)
                        if entry[3] in leaves and leaves[entry[3]].version == entry[4]
                    ],
                    chebyshev_order,
                    current_val,
                    id_label,
                    atol,
                    rtol,
                    notify_label,
                    resolved_diagnostics_path,
                    vectorize_cache=vectorize_cache,
                )

            last_notify = time.time()

        # Step 1: the budget. The running sums are exact, so this test is the correctly rounded
        # one; before stopping on it, it is confirmed on the sums the call returns (summed in
        # ascending x, _sum_leaves_ascending()), so that a call which stops here reports
        # converged by the same arithmetic. If they disagree -- a total at the boundary, within
        # rounding -- refinement continues.
        #
        # Then the trust check one level down, taken lazily: when the budget passes, every leaf not
        # yet checked (and not floor-limited, which is exempt) is checked, in ascending x. A
        # widened error, or an untrusted leaf, reopens the loop, which continues and checks again
        # at the next pass. The loop stops on the budget only with every leaf checked or
        # floor-limited, and no refinable leaf untrusted.
        running_value, running_error = _running_sums()
        running_stop = running_error <= max(atol, rtol * np.fabs(running_value))
        if running_stop:
            check_value, check_error = _sum_leaves_ascending(_leaves_ascending())
            if check_error <= max(atol, rtol * np.fabs(check_value)):
                unchecked = [
                    candidate
                    for candidate in _leaves_ascending()
                    if not candidate.checked and not candidate.region._phase_limited
                ]
                if unchecked:
                    num_check_passes += 1
                    for candidate in unchecked:
                        _check_leaf(candidate)
                    continue
                if not untrusted_refinable:
                    break
            else:
                num_resum_rejections += 1

        # Step 2: the refinable leaf chosen next -- an untrusted one first, else the one with the
        # largest error. Step 3: none is refinable.
        leaf = _pop_candidate()
        if leaf is None:
            break
        region = leaf.region

        # Floor-limited? Its round-off floor is finite, its residual (the step-(4) residual on the
        # Levin branch, the nested-pair estimate on the fallback branch) is already down at that
        # floor, and bisecting it would not lower the floor: its halves' floors sum to at least
        # _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its own (06b's premise; see that
        # constant's comment for why 1/2). Then bisection buys cost and no accuracy, so the leaf
        # is not refined, and is flagged phase_limited.
        #
        # The premise is checked from the halves: on the Levin branch they are dataL/dataR, in
        # hand already, so the check is free; on the fallback branch they are solved here, on
        # demand, and kept for the children if the leaf is bisected. Only the leaf chosen is ever
        # checked, so a leaf never chosen costs nothing extra.
        #
        # The finiteness guard matters (prompt 04): _roundoff_floor() returns +inf when G0 =
        # min|theta'| lands on (or numerically at) a sampled node -- an interior stationary point.
        # Such a leaf's error is +inf, so it is chosen before every finite one, and it must not
        # be found floor-limited: "residual <= floor" is trivially true against +inf. It is
        # bisected until its floor is finite, or it reaches depth_max.
        phase_err = region.phase_err
        if np.isfinite(phase_err) and region.abserr <= phase_err:
            if _test_only_bypass_roundoff_premise:
                floor_limited = True
            else:
                if leaf.dataL is None:
                    _solve_halves(leaf)
                floor_limited = _roundoff_exit_premise_holds(
                    phase_err,
                    leaf.dataL.get("phase_err", 0.0) or 0.0,
                    leaf.dataR.get("phase_err", 0.0) or 0.0,
                )
            if floor_limited:
                region._phase_limited = True
                _exempt_floor_limited(leaf)
                leaf.dataL = None
                leaf.dataR = None
                continue

        # Bisect. Its halves become its children's own solves, carried forward rather than solved
        # again: a Levin leaf's are dataL/dataR, already in hand; a fallback leaf's are solved now
        # unless the premise check above already solved them.
        if leaf.dataL is None:
            _solve_halves(leaf)

        a = leaf.start
        b = leaf.end
        c = leaf.break_point
        new_depth = leaf.depth + 1
        new_abs_history = leaf.abserr_history | {leaf.depth: region.abserr}
        new_rel_history = leaf.relerr_history | {leaf.depth: region.relerr}
        new_p_ratios_history = leaf.p_ratios_history | {
            leaf.depth: region.p_ratios if not leaf.is_direct else []
        }

        grandchildren = leaf.grandchildren
        left = _make_leaf(
            a,
            c,
            new_depth,
            leaf.dataL,
            new_abs_history,
            new_rel_history,
            new_p_ratios_history,
            halves=grandchildren[0] if grandchildren is not None else None,
        )
        right = _make_leaf(
            c,
            b,
            new_depth,
            leaf.dataR,
            new_abs_history,
            new_rel_history,
            new_p_ratios_history,
            halves=grandchildren[1] if grandchildren is not None else None,
        )

        num_splits += 1

        _remove_leaf(leaf)
        leaf.dataL = None
        leaf.dataR = None
        leaf.grandchildren = None
        _add_leaf(left)
        _add_leaf(right)

    leaves_ascending = _leaves_ascending()
    used_regions = [leaf.region for leaf in leaves_ascending]
    num_used_regions = len(used_regions)
    num_simple_regions = sum(1 for leaf in leaves_ascending if leaf.is_direct)

    p_points = []
    if build_p_sample:
        for leaf in leaves_ascending:
            if not leaf.is_direct:
                p_points.extend(leaf.data["p_sample"])
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    # Aggregate value and error estimate, each summed over the leaves in ascending x. Each leaf
    # contributes max(resolution/fallback residual, round-off floor), plus its truncation term:
    # used_interval.total_err, the e_i the loop's budget sums. The two are estimates of independent
    # error sources, and the larger dominates.
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
    val, abserr_total = _sum_leaves_ascending(leaves_ascending)

    abserr_resolution_total = 0.0
    abserr_roundoff_total = 0.0
    abserr_fallback_total = 0.0
    abserr_truncation_total = 0.0
    num_phase_limited = 0
    num_at_depth_max = 0
    num_untrusted = 0
    num_unchecked = 0
    for region in used_regions:
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
        if region.depth >= depth_max:
            num_at_depth_max = num_at_depth_max + 1
        if region.trusted is False:
            num_untrusted = num_untrusted + 1
        if region.trusted is None and not region.phase_limited:
            num_unchecked = num_unchecked + 1

    # guarded exactly as the per-region relative error is guarded, and for the same reason
    relerr_total = abserr_total / max(np.fabs(val), atol)

    # The call's budget, by the formula the loop stops on (step 1 above), from the returned sums.
    # A call that stopped on the budget meets it by construction; one that stopped because no
    # leaf was refinable does not, and every leaf still holding error back is then at depth_max,
    # or floor-limited (round-off, phase_limited). And the call is not converged while a leaf it
    # returns is untrusted (the trust check one level down; such a leaf is at depth_max).
    requested_total = max(atol, rtol * np.fabs(val))
    budget_met = abserr_total <= requested_total
    converged = budget_met and num_untrusted == 0

    _logger.debug(
        "adaptive_levin (%s): global budget | %d leaves, %d bisections, %d floor-limited, "
        "%d at depth_max | trust check: %d checks in %d passes, %d found untrusted, %d returned "
        "untrusted, %d returned unchecked | running test %s, returned budget met %s%s, running "
        "test passed but returned sums did not %d time(s)",
        _format_label(notify_label, id_label),
        num_used_regions,
        num_splits,
        num_phase_limited,
        num_at_depth_max,
        num_checks,
        num_check_passes,
        num_untrusted_found,
        num_untrusted,
        num_unchecked,
        running_stop,
        budget_met,
        " (DISAGREE)" if bool(running_stop) != bool(budget_met) else "",
        num_resum_rejections,
    )

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
            "   -- no leaf of the partition could be refined further: of %d leaves, %d are at "
            "depth_max (%d of them with an error estimate that failed the trust check one level "
            "down) and %d are limited by the round-off floor (Chen et al. eq. 151); consider a "
            "looser atol/rtol, a larger depth_max, or a smaller chebyshev_order",
            num_used_regions,
            num_at_depth_max,
            num_untrusted,
            num_phase_limited,
        )

    # Health check. If the requested accuracy is not attainable -- typically because the caller has
    # asked for an accuracy the phase function cannot deliver -- the bisection runs to the depth
    # limit along the lines holding the total back. The answer is usually still good, but the cost
    # can be two orders of magnitude higher than necessary, so it is worth saying so rather than
    # letting it pass silently.
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
            "!! WARNING (adaptive_levin, %s): bisection reached the maximum depth %d while the "
            "call's error budget was unmet | %d subintervals (%d direct), %d Levin solves | "
            "atol=%.3g, rtol=%.3g",
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
    # asked for an accuracy below the eq. (151) round-off floor (_roundoff_floor()) of one or more
    # leaves the budget needed to refine. Tightening atol/rtol will not help, and -- unlike before
    # prompt 04 -- supplying a range-reduced phase (theta_mod_2pi) will not help either: the floor
    # is independent of how the phase is presented (C4; see prompt 04's log). What can help is a
    # smaller chebyshev_order (lowering the max(G1, k^2) term) or, where the phase's own
    # construction is the real limit (e.g. a fitted spline), a declared theta_abserr so the caller
    # sees an honest number instead of an artificially small one.
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
        # True if the aggregate abserr meets max(atol, rtol*|value|) -- the call's budget, which
        # the loop refines until it is met or no leaf is refinable (06c) -- and no leaf returned is
        # untrusted. False means every leaf still holding error back is at depth_max or
        # floor-limited, or a leaf at depth_max failed the trust check one level down.
        "converged": bool(converged),
        # True if at least one leaf the loop chose while the budget was unmet was found
        # floor-limited: limited by the round-off floor rather than by resolution, on a floor its
        # halves would not lower (_roundoff_exit_premise_holds()). When this is set, the requested
        # tolerance was not attainable at this Chebyshev order and interval geometry, and
        # tightening atol/rtol will not help.
        "phase_limited": num_phase_limited > 0,
        "num_phase_limited_regions": num_phase_limited,
        # leaves returned untrusted by the trust check one level down: each is at depth_max, and
        # declares max(residual, delta1 + delta2); the call is not converged while one is returned
        "num_untrusted_regions": num_untrusted,
        "p_points": p_points,
        "num_regions": num_used_regions,
        "regions": used_regions,
        "num_simple_regions": num_simple_regions,
        # "evaluations" counts subregion *solves* (a leaf's own solve where it is not carried
        # forward, the two comparison halves of every Levin leaf, and the two halves of every
        # fallback leaf that is checked for the round-off premise or bisected), not integrand
        # evaluations -- kept under this name because docs/adaptive-levin-benchmark/levin_bench/
        # reads it by key (rec 11, C10). num_subregion_solves is the same value under a name that
        # says what it counts; prefer it in new code.
        "evaluations": int(num_evaluations),
        "num_subregion_solves": int(num_evaluations),
        "elapsed": float(elapsed),
        "num_SVD_errors": num_SVD_errors,
        "num_order_changes": num_order_changes,
        # num_direct_solves is kept exactly as before -- a *per-region* count, accumulated only
        # from each leaf's own solve, once per leaf (leaves later bisected included), never from a
        # comparison half (dataL/dataR) that does not become a leaf (rec 11, C10; see where it
        # is accumulated above). docs/adaptive-levin-benchmark/levin_bench/runners.py reads it by
        # this name. num_solves_direct/num_solves_lstsq/num_solves_pinv below are the true totals,
        # counting every solve actually attempted; num_solves_total is their sum. A caller wanting
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
    num_leaves: int,
    refinable_leaves: int,
    simple_leaves: int,
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
    #
    # Under the global budget (prompts/levin-phase-floor/06c-global-error-control.md) the value is
    # the current total V over the leaves of the partition, and "still refinable" counts the leaves
    # that remain candidates for bisection.
    _logger.info(
        "|  -- current value = %.7g, %d leaves (of which %d direct quadrature), %d still "
        "refinable | %d subregion solves | max depth %d",
        current_val,
        num_leaves,
        simple_leaves,
        refinable_leaves,
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
    regions: List[_levin_leaf],
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

        # only report regions whose own residual misses both atol and rtol. This is a per-region
        # filter for the plots, kept from the earlier per-region scheme; it is not the driver's
        # test, which is the call's global budget (see _adaptive_levin()). The regions handed in
        # are the leaves still refinable, largest error first.
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
    estimate, |CC_{2N-1} - CC_N|, enters the call's error budget on the same terms as a Levin
    region's.

    Error control is global (prompts/levin-phase-floor/06c-global-error-control.md): the driver
    keeps every subinterval (leaf) of the current partition with its value and its error, and
    bisects the refinable leaf with the largest error until the summed error meets
    max(atol, rtol*|summed value|), or no leaf is refinable. A leaf is not refinable at depth_max,
    when it is floor-limited (its round-off floor sets its error and bisection would not lower it;
    see "phase_limited"). So the budget goes
    where the error is, and the relative tolerance is taken against the call's own total, which
    matters wherever the subintervals' values cancel. Before it stops, each subinterval's error
    estimate is checked one level down, against its grandchildren: the estimate is widened by the
    observed convergence ratio, and a subinterval whose estimate is not converging is refined
    whatever the budget says (see "num_untrusted_regions").

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
            prompt 04 does NOT lower a Levin region's reported error floor: that floor is Chen et
            al. eq. (151) (see "abserr_roundoff" below), which is independent of how the phase is
            presented (C4 -- an earlier version of this module inferred the floor from whether
            this key was supplied, and that inference was optimistic by up to 8.1e9 at high
            frequency; see prompts/levin-refactor/logs/04-roundoff-floor.md). On a
            Clenshaw-Curtis fallback region, whose floor is the rule's own (as of
            prompts/levin-phase-floor/06d-cc-roundoff-floor.md), it does: there sin and cos are
            evaluated at every node, and the rounding of their argument, u*|theta(x)| with a raw
            phase, is a real error that the floor counts. An error in this function's own value
            belongs in "theta_abserr".
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
        floor (Chen et al. eq. 151) is absolute by construction. The call's budget is
        max(atol, rtol*|value|), met by the summed error over all subintervals; atol is not
        divided among them (as of prompts/levin-phase-floor 06c; from prompt 05 until then each
        subregion was tested against atol scaled by its share of x_span's length).
    :param rtol: requested relative tolerance on the summed result. Must be non-negative. It is
        taken against the call's own total |value|, not against each subinterval's value (as of
        06c), so a result whose subintervals cancel is held to rtol of what it sums to.
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
        every Levin subinterval of the final partition (diagnostics only; costs additional memory).
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
          * "abserr" -- estimated absolute error of "value". This is a sum, over the subintervals
            of the final partition, of max(resolution/fallback residual, round-off floor) plus any
            truncation term (see abserr_truncation below) -- the quantity the driver's budget is
            tested on; it is an estimate, not a proven bound.
          * "abserr_resolution", "abserr_roundoff", "abserr_fallback", "abserr_truncation" --
            "abserr" broken down by source: the summed step-(4) residual over Levin regions (new
            as of prompt 04), the summed round-off floor (Chen et al. eq. 151 on Levin regions,
            the Clenshaw-Curtis rule's own on fallback regions as of
            prompts/levin-phase-floor/06d-cc-roundoff-floor.md, plus any declared theta_abserr
            endpoint term) over every region (new as of prompt 04), the summed
            nested-pair estimate over Clenshaw-Curtis fallback regions (new as of prompt 04), and
            the summed endpoint contribution of p-modes dropped by the rtol gate on Levin regions
            (new as of prompt 06, rec 9), respectively. A caller that cannot get more digits can
            tell from these which source is responsible: a large abserr_resolution means further
            bisection (a tighter atol/rtol) could still help; a large abserr_roundoff means it
            cannot; a large abserr_truncation means a smaller rtol would recover more modes.
          * "relerr" -- "abserr" divided by max(|value|, atol).
          * "converged" -- True if "abserr" <= max(atol, rtol*|value|), i.e. whether the request
            was met in aggregate, and no subinterval returned failed the trust check. The driver
            refines until both hold, so False means no subinterval could be refined further:
            every one still holding error back is at depth_max or floor-limited, or one at
            depth_max is untrusted.
          * "phase_limited" -- True if at least one subinterval the driver chose to refine while
            the budget was unmet was found floor-limited: its achievable accuracy was set by the
            round-off floor rather than by lack of resolution, on a floor that bisecting it would
            not lower (its halves' floors sum to at least half its own); tightening atol/rtol will
            not help such regions -- see abserr_roundoff above.
          * "num_phase_limited_regions" -- count of such regions.
          * "num_untrusted_regions" -- count of subintervals returned that failed the trust
            check one level down: their halves' and grandchildren's values do not show the error
            falling (r = delta2 / delta1 >= 1), so their estimate cannot be relied on. Each is at
            depth_max, declares max(residual, delta1 + delta2), and makes "converged" False (new
            as of 06c). Each returned used_interval carries delta1, delta2, convergence_ratio,
            trusted and declared_resolution.
          * "num_regions" -- number of subintervals of the final partition (Levin + direct
            quadrature).
          * "num_simple_regions" -- of those, how many were handled by direct quadrature rather
            than the Levin rule.
          * "regions" -- list of used_interval objects describing each subinterval of the final
            partition, sorted by start.
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
            Despite the name, this is a *per-region* count (accumulated only from each
            subinterval's own solve), not a total solve count: a comparison-child solve
            (dataL/dataR) that never becomes a subinterval itself is never counted by it (rec 11,
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
