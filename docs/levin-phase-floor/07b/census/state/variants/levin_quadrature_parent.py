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
    (prompts/levin-phase-floor/06d-cc-roundoff-floor.md), whose node term is,
    since prompts/levin-phase-floor/06e-exact-end-nodes.md, the first-order
    error of the displacement each node actually has, signed, with bounded
    remainders, in place of 06d's worst case. The grid's end nodes are the
    region's endpoints exactly (chebyshev_matrices()), so every solve
    integrates its own region and a region and its halves tile. See the extended discussion in _adaptive_levin();
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

  * Weakly oscillatory regions (total variation of the phase below 6*pi) are handed to a
    nested Clenshaw-Curtis rule rather than to the Levin rule. The variation V is computed
    from the raw phase itself (prompts/levin-phase-floor/07a-window-stop-and-phase-
    variation.md, section 1.1): V = |theta(b) - theta(a)| on a region whose rate theta' has no
    two node samples of opposite strict sign, where theta is monotone and that is exact; else
    the variation on the node partition, sum_j |theta(x_{j+1}) - theta(x_j)|, exact where
    theta is monotone between adjacent nodes and a lower bound otherwise (see
    _Basis_SinCos.build_Levin_data()). Until 07a it was estimated as the plain mean of
    |theta'| on the region's Chebyshev nodes times its width, which leans to |theta'| at the
    region's ends, where the nodes crowd: measured 1.27-1.32 times high for a near-centred
    root, up to 8.43 times high with |theta'| weighted to one end, 0.47 of the truth on a
    narrow bump in the middle; and it was not monotone under bisection, so a half of a region
    the gate sent to the fallback could be sent to the Levin solve. V is additive on a
    monotone region, so a half never reads more than its parent. The threshold 6*pi itself
    is unchanged, and has never been measured ([07-c2-gate-threshold-was-never-measured]).
    The extremal Chebyshev grid the Levin solve already uses *is* the
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
    antiderivative p ~ f/(i*theta') is singular there). This item used to call the
    gate-and-bisect structure -- refining until the stationary neighbourhood is weakly
    oscillatory, then applying a direct rule to it -- "the correct structural response". It is
    not, by itself (prompts/levin-phase-floor/07-stationary-point-guard.md, after
    docs/levin-phase-floor/06/p1/report.md): the gate acts only on a region bisection has
    already produced, and nothing makes bisection reach a stationary point. Near one, a Levin
    region's estimate and its halves' all lack the stationary-phase term, so the step-(4)
    residual, and the trust check one level down, are blind to it; a Levin region of large
    total variation was measured accepting an interior stationary point with an error estimate
    about 1e5 times too small. The gate's regression test put its point at x = 1/2, the first
    bisection point, where bisection exposes it by construction. See the next item.

  * Stationary points are located before integrating, and their neighbourhoods integrated
    directly (prompts/levin-phase-floor/07-stationary-point-guard.md; _adaptive_levin()). A basis
    that supplies the closed-form rates of its independent phases (for adaptive_levin_sincos(),
    the caller's theta["theta_deriv"]) has every sign change of a rate, and every interior minimum
    of |theta'| deep enough to act as a nearly coalescing pair, located on a uniform scan. Round
    each, a window of phase half-width _LEVIN_WINDOW_HALF_PHASE is integrated by the nested
    Clenshaw-Curtis rule, in pieces of at most 2 pi, whatever their span; the rest is left to the
    loop as before, in the same budget. The location does not depend on any estimate of the
    phase's variation (prompt 07a section 1.1, in place of prompt 07 section 10's Rule A, which
    skipped a phase whose variation over the span, estimated on the scan, was below SIX_PI): a
    phase is left unlocated only when there is nothing to locate, its rate exactly zero at every
    finite scan sample. As a safety net, a Levin region whose phase rate changes sign on its own
    Chebyshev nodes is never returned as a Levin region. Without a closed-form rate nothing is
    located, and that guard is the only protection.

  * A direct LU solve is used in place of the least-squares/SVD solve when the
    phase span shows the Levin super-operator to be well conditioned. Note
    that Remark 2 of Chen et al. recommends a *rank-revealing QR* in place of
    the truncated SVD throughout (they report ~5x faster with no apparent loss
    of accuracy). That suggestion has NOT been taken up here: the fast path is
    an LU solve gated on conditioning, with lstsq retained as the fallback for
    the ill-conditioned case where the minimum-norm solution is needed. An
    RRQR fallback in place of lstsq remains an unexplored optimisation.

THE BASIS INTERFACE
-------------------
_adaptive_levin() is written for a basis of any number of components. It calls, on the basis
object: build_Levin_data() (theta' on a region's nodes, the super-operator, the basis at the two
endpoints and the phase's variation V across the region, computed from the raw phase as above),
theta_abserr_at(), and supports_complexified_solve if present; and,
on a region the nested Clenshaw-Curtis rule takes, eval_basis_and_arguments(),
theta_prime_on_fine_grid() and basis_derivative(). Since prompts/levin-phase-floor/
07-stationary-point-guard.md, two optional methods supply what the stationary-point guard needs:
independent_phases(), one callable per independent phase of the basis, and
independent_phase_rates(), their closed-form rates d theta_i / dx in the same order, or None when
the basis cannot supply them. A basis without them is treated as unable to supply rates: nothing is
located, and only the nodal guard protects it. A phase whose rate is exactly zero at every finite
scan sample is not located either (prompt 07a section 1.1); for a basis of several phases the nodal
guard tests such a phase for a sign change on a region's nodes rather than for one strict sign.
_Basis_SinCos
has one phase, theta["theta"], whose rate is theta["theta_deriv"] when the caller supplies it.

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

# The C2 gate's threshold: a region whose phase variation V is below SIX_PI goes to the nested
# Clenshaw-Curtis rule, any other to the Levin solve (_adaptive_levin_subregion_impl()). Since
# prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md, section 1.1, V is computed from
# the raw phase at the region's nodes (_Basis_SinCos.build_Levin_data()), exactly where the phase is
# monotone; until then it was estimated as mean|theta'| on the nodes times the width, which was
# biased by an amount that depends on the region's shape (1.27-1.32x high for a near-centred root,
# up to 8.43x high with |theta'| weighted to one end, 0.47x on a narrow central bump) and not
# monotone under bisection, so 6 pi did not mean one thing. The number 6 pi is unchanged by 07a,
# and has not been measured ([07-c2-gate-threshold-was-never-measured]).
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
#
# What is compared (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2
# (d)). The floors compared are each solve's "phase_err_roundoff": its floor less the declared
# endpoint term (_declared_endpoint_phase_err()), on both sides. That term is the caller's own phase
# error carried to the region's endpoints; bisection does not lower it, and the argument for rho
# above is about the round-off part alone -- eq. (151)'s floor on the Levin branch, the
# Clenshaw-Curtis rule's floor with its node term on the fallback branch, as 06b compared them.
# Left in, it dilutes the ratio on both sides and can make the premise hold where bisection would
# cut the round-off floor that sets the error: found on 06h's census
# ([07-roundoff-exit-assumes-bisection-cannot-help], g-1e-8-3-1072 call 13), where the declared term
# pushed the halves' sum past half the parent's and the call stopped unconverged at depth 0, which
# under 06e split once and converged.
#
# And the premise is not enough by itself: the halves are compared with the parent, not with what
# the call needs. A split that lowers a floor by 30 % fails the halving test, though it may be all
# the call needs. So the exit also needs the split to be unable to meet the budget: the call's
# error recomputed with the leaf replaced by its halves' floors (lower bounds on their errors) is
# still above the request. A leaf whose split might be all the call needs is bisected. It cannot
# chase: where floors do not fall, the recomputed error is the call's error now, which is above
# the request, so the exit fires as before; and every split it allows deepens a leaf. See
# _roundoff_exit_premise_holds() and _adaptive_levin().
_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5

# The trust check's cut-off on the observed convergence ratio r = delta2 / delta1
# (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (b)). A checked leaf is
# trusted -- its error bounded by max(residual, delta1) / (1 - r), the geometric tail from here on --
# only if r < this, or if r < 1 and delta1 <= nu_B, the noise yardstick of _trust_noise_yardstick()
# (then delta1 and delta2 are differences of rounding noise, r means nothing, and the rounding of
# the solves they are formed from bounds them). Otherwise it is untrusted, and refined whatever the
# budget says.
#
# The noise clause (since prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md; from 06g
# to 06m it compared delta1 with the leaf's own floor phi_P alone).
#   * Why phi_P alone was too small (06j, [06g-slow-leaves-at-half-cost-convergence-outside-
#     stationary-points]). delta1 = |V_P - (V_L + V_R)| is a difference of three solves, and phi_P
#     declares the rounding of one. On 75 of 06j's 142 sampled slow leaves (r ~ 1/2, delta1 / phi_P
#     median 1.49) delta1 was the difference of the three solves' signed Clenshaw-Curtis node
#     errors, |E_P - (E_L + E_R)|, to 0.6 %, which each scale with their solve's width and so fix r
#     at 1/2: rounding noise the clause did not recognise, refined to depth_max and returned
#     untrusted, leaving calls that met their request flagged unconverged.
#   * Why the halves' node errors are combined signed (06k, 06l). Adding the halves' full floors
#     (06k's route 1, nu_1 = phi_P + phi_L + phi_R) passed leaves far above noise: the halves' E
#     can be hundreds to 1e6 times the parent's and nearly equal and opposite, cancelling in
#     V_L + V_R, so |E_L| + |E_R| covered delta1 on a term that is not in it. nu_B carries them as
#     |E_L + E_R|, as the call's own error carries E (06g, section 0.2 (c)), and the floors less
#     their node terms beside it; then |E_P - (E_L + E_R)| <= |E_P| + |E_L + E_R| <= nu_B.
#   * Why a Levin parent's test leaves its Clenshaw-Curtis halves out (06k, 06l; the user's
#     modification of route 1). A Clenshaw-Curtis half's floor can be hundreds of times a Levin
#     parent's; carried into its test it passed leaves up to delta1 / phi_P = 1.1e4. Without it a
#     Levin parent's firings reach delta1 / phi_P = 2.85 at most (06l's census).
# 06l measured this yardstick as a scratch variant on A3 v2's census (route B; docs/levin-phase-
# floor/VERIFICATION.md, section 6l): it trusts none of 06h's wrongly trusted carrying leaves, moves
# no value past both errors, and trusts no leaf route 1 does not. It does not reach every slow leaf:
# on 12 of the 75, delta1 exceeds the first-order node-error difference by 0.2-0.6 %, and no margin
# is added for that remainder (prompt 06m, section 0.4).
#
# The cut-off 1/4 below is unchanged by 06m.
#
# Why 1/4.
#   * Every rule here converges, in its asymptotic regime, by far more than 4x per halving: a Levin
#     leaf of Chebyshev order n >= 8 by about 2^-n, and the nested Clenshaw-Curtis pair spectrally.
#     A leaf at r >= 1/4 is not in that regime, and there one ratio does not predict the next: on
#     06e's census (Z3's sample 3) a leaf at r = 0.507 was trusted, and its value was 1.245x its
#     declaration ([06c-budget-can-stop-on-a-pre-asymptotic-residual]).
#   * At r < 1/4 the geometric tail adds at most a third to delta1, so the bound leans little on
#     the extrapolation.
#   * A cut-off that ends refinement starts at the conservative end (the campaign's practice): 1/4
#     trusts fewer leaves than 1/2 would. 06g's tests and census report every checked leaf's r, so
#     the value can be revisited on that evidence.
# Until 06g the cut-off was 1, with no noise clause.
_LEVIN_TRUST_MAX_RATIO = 0.25

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

# The node term of _cc_roundoff_floor() (prompts/levin-phase-floor/06e-exact-end-nodes.md, which
# replaces 06d's worst case c_x u (max(|a|, |b|) + h) max|theta'| ... with the displacement each
# node actually has): phi_node = |E| + R_g + R_2 + R_c, E = sum_k w_k g'(x^_k) Delta_k, from the
# signed displacements Delta_k = x^_k - x_k (_chebyshev_node_displacements()). The constants below
# are its relative-error bounds and counts; see _cc_node_term_parts() for the terms.
#
# eta_theta, when theta' at the fine nodes is the caller's own theta_deriv, evaluated at each node
# (_Basis_SinCos.theta_prime_on_fine_grid()). Not derivable here: the module does nothing to that
# value but evaluate the caller's function, whose rounding it cannot see. So it is measured (prompt
# 06e, section 2 test 4): on every cell of that test whose theta' is supplied, the relative error of
# the supplied theta' against theta' evaluated in mpmath at the same float node is at most a few u
# (the log records the worst). It is taken at 2^-10, a margin of more than 10^12 over the worst
# measured, so that it also covers a caller whose theta' is a fit good to three digits rather than
# a closed form. A theta' worse than that is an error in the phase the Levin branch also builds its
# collocation operator from, and belongs in the caller's theta_abserr, as an error in theta does.
_CC_THETA_PRIME_REL_ERROR_SUPPLIED = 2.0**-10

# eta_theta, when theta' at the fine nodes is the fine grid's spectral derivative of a sampled raw
# theta (no theta_deriv supplied). Its error is set by how well 2N - 1 samples resolve theta and by
# the rounding of a raw phase of size |theta|, amplified by the differentiation matrix: neither is
# bounded relative to theta' by anything the solve holds, so no basis below 1 can be stated, and it
# is 1 -- a triangle-inequality bound on the exact displacements (section 0.2 item 2).
_CC_THETA_PRIME_REL_ERROR_SPECTRAL = 1.0

# eta_f, for f' at the fine nodes, the fine grid's Chebyshev derivative of the amplitude samples
# (as 06d took it). Its error is set by how well the samples resolve f, which nothing the solve
# holds bounds relative to f' (an amplitude near a singularity is resolved worst, and that is where
# f' matters). No basis below 1 can be stated, so it is 1, as above.
_CC_F_PRIME_REL_ERROR = 1.0

# The second-order remainder R_2 = (1/2) sum_k |w_k| Delta_k^2 |g''(xi_k)|, xi_k between x_k and
# x^_k, is bounded with |g''| <= max over the nodes of
#     sum_i ( |f''_i| + 2 |f'_i| |theta'| + |f_i| |theta''| + |f_i| theta'^2 )
# from the same data (f'' and theta'' the fine grid's Chebyshev derivatives of f' and theta'), times
# this factor: the derivatives taken good to within 100 %, as eta = 1 takes f' above, and the
# maximum over the nodes for that over the region. R_2 is of relative size u |x| theta' against
# E, so the factor never decides the floor; it keeps R_2 a bound.
_CC_NODE_SECOND_ORDER_SAFETY = 2.0

# The rounding count in R_c, the rounding of E itself, beyond the n of its weighted sum: E is formed
# as sum_k w_k (Delta_k g'_k), g'_k = sum_i ( f'_ik b_ik + (f_ik theta'_k) b'_ik ) over m components,
# so each of its n terms carries
#   * 2m + 1 roundings in g'_k: at most two products in each of its 2m summands, and 2m - 1 sums;
#   * 1 in the product with Delta_k, and n in the dot product with w_k (its product and n - 1 sums);
#   * 2 from Delta_k's own error (_chebyshev_node_displacements(): 2u |Delta_k| beyond its absolute
#     part, which R_c adds separately);
#   * 1 from the amplitude f_ik, correct to its last bit, as in 06d's count;
# so gamma_K with K = (2m + 1) + 1 + n + 2 + 1 = n + 2m + 5, this constant being the 5 beyond
# n + 2m. The weights carry none: the node error
# is that of the rule as computed, with its own float weights. The basis values' error, u (|a| + 2)
# per value, is added separately. Derived, not fitted.
_CC_NODE_TERM_ROUNDING_COUNT = 5

# minimum phase change across a subinterval before we are prepared to invert the Levin super-operator
# with a direct LU solve rather than a least-squares (SVD) solve. Below this the operator is poorly
# conditioned and we need the minimum-norm solution; see _adaptive_levin_subregion_impl(). The
# phase change compared is the region's variation V, computed from the raw phase as the C2 gate's is
# (see SIX_PI; since prompt 07a section 1.1, in place of the estimate mean|theta'| * width). The
# number 20 pi is unchanged.
_LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20.0 * np.pi

# The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md; see
# _locate_stationary_points() and _adaptive_levin()). Before the loop runs, each phase rate a
# basis supplies in closed form is scanned on this many points uniform in x over x_span, endpoints
# included: prompt 06a's grid, on which A3 v2's stationary points were first counted.
_LEVIN_STATIONARY_SCAN_POINTS = 4097

# Phi_w, the phase half-width of the window round each located point: the window is the largest
# interval holding the point on which |theta_i(x) - theta_i(x_*)| <= Phi_w, and it is integrated
# directly, by the nested Clenshaw-Curtis pair, in pieces. For a simple root, theta' ~ theta''
# (x - x_s), and at the window's edge theta'^2 / |theta''| = 2 Phi_w whatever theta'' is: one
# dimensionless number sets how well conditioned the Levin leaves beside the window are (prompt 07
# section 0.2 (beta)).
# Measured (prompt 07 section 2.2, the rule as the prompt fixes it, run at rtol 1e-9 after the section 2 stop,
# section 9 point 1): Phi_w is the smallest rung of {2, 8, 32, 128, 512} pi on which every case of the
# family (R) -- theta = omega x + (c/2)(x - x_s)^2 on [0, 1], c in {1e4, 1e6, 1e8}, two non-dyadic x_s, omega
# 0 and 1 -- is honest and converged. At 2 pi all ten are (err/abserr at most 0.03, abserr/request at most
# 0.89, no window stopped by the iroff1 rule); at 32 pi and above the cancelling case c = 1e8, x_s = 0.7236,
# omega = 0 is not converged. docs/levin-phase-floor/07/calibration/out/run2/calibrate_tables_rtol1e-09.md.
_LEVIN_WINDOW_HALF_PHASE = 2.0 * np.pi

# Lambda_c: an interior local minimum x_m of |theta_i'| with no sign change beside it is located,
# and windowed like a root, when its Airy parameter lambda = |theta_i'(x_m)|^(3/2) /
# sqrt(|theta_i'''(x_m)| / 2) is below this: the phase accumulated across its slow zone is then
# small, and the pair of stationary points nearly coalesces (prompt 07 section 0.2 (gamma)).
# Measured (prompt 07 section 2.2, at Phi_w = 2 pi and rtol 1e-9): Lambda_c is the smallest rung L of
# {4, 16, 64, 256, 1024} such that every case of the family (C) -- theta = a(x - x_0) + (b/3)(x - x_0)^3 on
# [0, 1], b in {1e4, 1e8}, lambda in {0.25, 1, 4, ..., 1024} -- with lambda >= L is honest with no minimum
# located; L = 4 (every case is honest, so the rule picks the lowest rung), and the cases below it,
# lambda = 0.25 and 1 at both b, are honest and converged with their minimum located.
# docs/levin-phase-floor/07/calibration/out/run2/calibrate_tables_rtol1e-09.md.
_LEVIN_COALESCENCE_LAMBDA = 4.0

# Delta phi_w, the largest phase span of one window piece, in every phase of the basis (prompt 07
# section 1.2 item 3).
_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi

# The stopping rule for window leaves: QUADPACK's iroff1, counted per window (prompts/levin-phase-floor/
# 07-stationary-point-guard.md section 1.3; its count made dqage's, with a value test no looser than
# the request, by prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.2, the
# user's choice of Part A's variant (d)). A bisection of a window leaf P into L and R counts against
# its window if and only if (_window_bisection_counts())
#   1. neither half is unresolved -- dqage's skip, its test defab1.eq.error1.or.defab2.eq.error2. A
#      Clenshaw-Curtis leaf is unresolved when _DQK21_SATURATION_FACTOR |CC_{2N-1} - CC_N| >= resasc,
#      both positive, where dqk21 would declare resasc itself (see that constant). A leaf with no resasc
#      is never unresolved;
#   2. |(V_L + V_R) - V_P| <= min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL, rtol) |V_L + V_R|, each V a leaf's own
#      solve's value: the halves barely change the value, measured against the request;
#   3. e_L + e_R >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO e_P, each e a leaf's own solve's error
#      (used_interval.own_err: the larger of its resolution and floor terms, plus truncation), taken
#      before the trust check widens a declaration -- dqage counts on its raw estimates, not on the
#      declared e' the budget sums.
# At _LEVIN_WINDOW_IROFF1_LIMIT counted bisections the window is round-off-limited, and every leaf of
# it is returned floor-limited (see _adaptive_levin()).
#
# Why each part.
#   * The count is dqage's, with its skip and its raw estimates. 07's port counted every bisection,
#     on e', which can be a trusted parent's widened declaration; dqage skips a bisection in which
#     either half's estimate is not yet in its asymptotic regime, and compares the estimates it
#     computed. The user kept dqage's rule defensively, "even though it barely makes any difference
#     here" (07a section 9.1): on Part A's calls neither the skip nor the raw estimates ever acted.
#   * The value test is no looser than the request: dqage's fixed 1e-5 would count, at rtol 1e-7 or
#     1e-8, a bisection that moves the value by 100-1000 times the request.
#   * dqage's outcome, ier = 2 ("roundoff error is detected, which prevents the requested tolerance
#     from being achieved"), is not adopted (07a section 9.3): dqage's stop ends the whole integration,
#     so "stopped unmet" there means "returned unmet"; here only the window stops, its leaves keep
#     declaring their error inside the call's sum, and the call is converged only if that sum meets
#     the request at the end -- the end-of-call test is the analogue of ier = 2. Flagging the call at
#     the stop would, on Part A's evidence, flag calls the stopped windows' declarations already cover
#     (they were measured at or above their true error), honest calls among them.
# The three numbers 1e-5, 0.99 and 6 are dqage's, unchanged. Why a rule is needed at all: window
# leaves are never floor-limited (07 section 0.2 (ii)), and 07's section 2 measured them chased to
# depth_max two ways, on a Clenshaw-Curtis floor flat in h above the request, and untrusted at the
# integrand's own noise far inside it; iroff1 counts on both (docs/levin-phase-floor/07/calibration/
# out/, the first run's calibrate_tables.md and replay_tables.md, section 2.5). Part A's measurement
# of the variants of this rule is docs/levin-phase-floor/07a/measurement/ (out/stop_2_report.md,
# out/tables_final_rg.md).
_LEVIN_WINDOW_IROFF1_VALUE_RTOL = 1.0e-5
_LEVIN_WINDOW_IROFF1_ERROR_RATIO = 0.99
_LEVIN_WINDOW_IROFF1_LIMIT = 6

# dqk21's saturation factor. QUADPACK's 21-point Gauss-Kronrod rule declares its error as
#     if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00)
#    *  abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)
# (dqk21.f), with abserr = |K - G| and resasc = int |f - mean f|, so the error it returns equals
# resasc exactly when 200 |K - G| >= resasc: the estimate is not yet in its asymptotic regime, and
# dqage's test defab.eq.error then skips the bisection's iroff1 count. A Clenshaw-Curtis leaf's
# analogue (07a section 1.2 item 1, Part A's definition): |CC_{2N-1} - CC_N| for |K - G|, and resasc
# formed on the fine rule (_adaptive_levin_subregion_cc()).
_DQK21_SATURATION_FACTOR = 200.0

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

    The +inf at a node near a stationary point is not a stationary-point guard. It fires only when
    a node falls close enough to the point; otherwise G0 is small but finite, the floor is finite,
    and a Levin region across the point could be left with an error estimate that does not see the
    stationary-phase term at all -- by its residual, or through the round-off-limited exit on this
    floor (docs/levin-phase-floor/06/p1/report.md, section 6: at rtol 1e-9 and 1e-11). Since
    prompts/levin-phase-floor/07-stationary-point-guard.md the driver locates stationary points
    before integrating, where the basis supplies closed-form phase rates, and integrates a window
    round each by the nested Clenshaw-Curtis pair, which carries _cc_roundoff_floor() and not this
    floor; and it never returns a Levin region whose theta' changes sign on its own nodes, nor
    lets such a region exit on this floor (see _adaptive_levin()).

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


def _cc_node_term_parts(
    weights: np.ndarray,
    displacements: np.ndarray,
    displacement_err_abs: float,
    f_sample: np.ndarray,
    f_prime_sample: np.ndarray,
    f_second_sample: np.ndarray,
    basis_sample: np.ndarray,
    basis_prime_sample: np.ndarray,
    argument_sample: np.ndarray,
    theta_prime_sample: np.ndarray,
    theta_second_sample: np.ndarray,
    eta_theta: float,
) -> Tuple[float, float, float, float]:
    """
    The node term of a Clenshaw-Curtis fallback solve's round-off floor, in its four parts
    (prompts/levin-phase-floor/06e-exact-end-nodes.md, section 0.2 item 2):

        phi_node = |E| + R_g + R_2 + R_c,      E = sum_k w_k g'(x^_k) Delta_k,

    with g = sum_i f_i b_i(theta), g' = sum_i ( f'_i b_i(theta) + f_i theta' b'_i(theta) ), and
    Delta_k = x^_k - x_k the signed displacement of each computed node from the exact one
    (_chebyshev_node_displacements()). Returns (E, R_g, R_2, R_c); E is signed.

    What it bounds. The rule as computed, sum_k w_k g(x^_k), differs from the same rule at the exact
    nodes by sum_k w_k (g(x^_k) - g(x_k)) = E - (1/2) sum_k w_k g''(xi_k) Delta_k^2. E is formed
    from the data the solve holds, so it errs; each part below bounds one source:
      * R_g = sum_k |w_k| |Delta_k| sum_i ( eta_theta |theta'_k| |f_ik| + eta_f |f'_ik| ): the
        error from using approximate theta'_k and f'_ik, whose relative errors are at most
        eta_theta (passed in; it depends on how theta' reached the nodes) and eta_f =
        _CC_F_PRIME_REL_ERROR. |b_i| and |b'_i| are at most 1 for the (sin, cos) basis.
      * R_2 = (1/2) sum_k |w_k| Delta_k^2 max|g''|, the second-order remainder, |g''| bounded from
        the same data; see _CC_NODE_SECOND_ORDER_SAFETY.
      * R_c, the rounding in computing E: gamma_K times sum_k |w_k| |Delta_k| sum_i |f'_ik b_ik| +
        |f_ik theta'_k b'_ik|, K = n + 2m + _CC_NODE_TERM_ROUNDING_COUNT (the count is that
        constant's comment); plus the basis values' error, u (|a_ik| + 2) per value (the argument
        to its last bit, and the sine or cosine to 1 ulp), at most 2, against the same sums with
        |b| replaced by 1; plus Delta's absolute error, displacement_err_abs, against
        sum_k |w_k| sum_i |f'_ik b_ik| + |f_ik theta'_k b'_ik|.

    The terms are summed in floating point, with relative errors of order n u of themselves, which
    no part of the bound depends on. No 1/theta' appears, as in 06d: theta' enters only as a factor.

    :param weights: the fine rule's weights, shape (n,).
    :param displacements: Delta_k for every node, shape (n,).
    :param displacement_err_abs: the absolute part of Delta's error bound.
    :param f_sample: amplitudes, shape (m, n).
    :param f_prime_sample: their derivatives at the nodes, shape (m, n).
    :param f_second_sample: their second derivatives at the nodes, shape (m, n).
    :param basis_sample: basis values b_i(theta(x^_k)), shape (m, n).
    :param basis_prime_sample: the basis values' derivatives with respect to theta, shape (m, n).
    :param argument_sample: the arguments the basis values were computed from, shape (m, n).
    :param theta_prime_sample: theta' at the nodes, shape (n,).
    :param theta_second_sample: theta'' at the nodes, shape (n,).
    :param eta_theta: the bound on theta_prime_sample's relative error.
    """
    m, n = f_sample.shape
    abs_w = np.fabs(weights)
    abs_wd = abs_w * np.fabs(displacements)
    abs_f = np.fabs(f_sample)
    abs_fp = np.fabs(f_prime_sample)
    abs_tp = np.fabs(theta_prime_sample)

    g_prime = np.sum(
        f_prime_sample * basis_sample
        + (f_sample * theta_prime_sample) * basis_prime_sample,
        axis=0,
    )
    E = float(np.dot(weights, displacements * g_prime))

    R_g = float(
        np.dot(
            abs_wd,
            np.sum(eta_theta * abs_tp * abs_f + _CC_F_PRIME_REL_ERROR * abs_fp, axis=0),
        )
    )

    g_second_bound = np.sum(
        np.fabs(f_second_sample)
        + 2.0 * abs_fp * abs_tp
        + abs_f * np.fabs(theta_second_sample)
        + abs_f * abs_tp * abs_tp,
        axis=0,
    )
    R_2 = (
        0.5
        * _CC_NODE_SECOND_ORDER_SAFETY
        * float(np.max(g_second_bound))
        * float(np.dot(abs_w, displacements * displacements))
    )

    g_prime_abs = np.sum(
        abs_fp * np.fabs(basis_sample) + abs_f * abs_tp * np.fabs(basis_prime_sample),
        axis=0,
    )
    K = n + 2 * m + _CC_NODE_TERM_ROUNDING_COUNT
    gamma_K = K * _UNIT_ROUNDOFF / (1.0 - K * _UNIT_ROUNDOFF)
    basis_err = np.minimum(2.0, _UNIT_ROUNDOFF * (np.fabs(argument_sample) + 2.0))
    R_c = (
        gamma_K * float(np.dot(abs_wd, g_prime_abs))
        + float(np.dot(abs_wd, np.sum(basis_err * (abs_fp + abs_f * abs_tp), axis=0)))
        + displacement_err_abs * float(np.dot(abs_w, g_prime_abs))
    )

    return E, R_g, R_2, R_c


def _cc_roundoff_floor(
    weights: np.ndarray,
    f_sample: np.ndarray,
    basis_sample: np.ndarray,
    argument_sample: np.ndarray,
    node_term: float,
) -> float:
    """
    Round-off floor for a Clenshaw-Curtis fallback solve
    (prompts/levin-phase-floor/06d-cc-roundoff-floor.md; its node term since
    prompts/levin-phase-floor/06e-exact-end-nodes.md), derived from what the rule computes:

        phi_CC = (n + 4) u S  +  u sum_k |w_k| sum_i |f_ik| |a_ik|  +  phi_node,
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
    * The third, phi_node (node_term, from _cc_node_term_parts()), is the rounding of the nodes:
      each computed node x^_k is displaced from the exact one by Delta_k, and a rule whose weights
      are fixed while its nodes move commits sum_k w_k g'(x^_k) Delta_k to first order, with
      g' = (f' + i theta' f) e^{i theta} for g = f e^{i theta}. Since 06e it is that first-order
      error itself, from the displacement each node actually has, signed, with theta' at each node,
      plus bounded remainders (its parts are _cc_node_term_parts()'s). 06d added the term (after
      the first two were measured short by up to 3.4e9 on a range-reduced phase) as a worst case:
      one displacement 5u (max(|a|, |b|) + h) for every node, the region's largest |theta'| at
      every node, and absolute values throughout. That bounded the node error by 20-1000 x, was
      98.8-99.7 % of the declared error on 07's leaves, and was flat under splitting, so a leaf it
      dominated was floor-limited far above its true error ([06d-cc-node-term-is-pessimistic]).
      With the end nodes exact (chebyshev_matrices()), Delta_0 = Delta_{n-1} = 0. Summed over
      leaves the term is still of order u |x| ||f|| TV(theta), so it does not grow under
      subdivision.

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
    :param node_term: phi_node, from _cc_node_term_parts().
    """
    abs_weights = np.fabs(weights)
    n = abs_weights.shape[0]
    abs_f = np.fabs(f_sample)
    S = float(np.dot(abs_weights, np.sum(np.fabs(f_sample * basis_sample), axis=0)))
    argument_term = float(
        np.dot(abs_weights, np.sum(abs_f * np.fabs(argument_sample), axis=0))
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
    phase_err: float,
    phase_err_L: float,
    phase_err_R: float,
    split_could_meet_budget: bool = False,
) -> bool:
    """
    The round-off-limited exit's premise: would bisecting this region leave its floor materially
    where it is, and leave the call short of its request? True if both hold:

      1. the halving test: its two halves' floors sum to at least
         _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO times its own;
      2. the budget clause: split_could_meet_budget is False.

    The floors passed are the part bisection can lower: each solve's "phase_err_roundoff", its floor
    less the declared endpoint term (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md;
    until then the whole "phase_err"). _adaptive_levin() computes split_could_meet_budget: True
    when the call's error is above its request now and would meet it with this leaf replaced by its
    halves' floors. See the comment above _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO for both.

    _adaptive_levin() asks this only once every other condition of the exit holds, so phase_err is
    always finite here. A half's floor can be +inf (an interior stationary point at or near one of
    its nodes; see _roundoff_floor()); the sum is then +inf, the halving test holds, the recomputed
    error is +inf so the budget clause holds too, and the exit may fire, exactly as it did before
    the premise was checked.
    """
    if split_could_meet_budget:
        return False
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


def _cc_leaf_unresolved(abserr: Optional[float], resasc: Optional[float]) -> bool:
    """
    Whether a Clenshaw-Curtis leaf's error estimate is unresolved in dqk21's sense (prompts/
    levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.2 item 1): True when
    _DQK21_SATURATION_FACTOR * abserr >= resasc, both finite and positive, abserr being the leaf's
    own nested-pair estimate |CC_{2N-1} - CC_N| and resasc its fine rule's sum_j |w_j| |g_j - gbar|.
    dqk21 would declare resasc itself there. A leaf with no resasc (not a Clenshaw-Curtis leaf), or
    with either number zero or not finite, is never unresolved, as dqk21 leaves its estimate alone
    when either is zero.
    """
    if resasc is None or abserr is None:
        return False
    if not (np.isfinite(resasc) and np.isfinite(abserr)):
        return False
    return bool(
        resasc > 0.0 and abserr > 0.0 and _DQK21_SATURATION_FACTOR * abserr >= resasc
    )


def _own_error(region: "used_interval") -> float:
    """A region's own solve's error e (used_interval.own_err) for the window's count; +inf if None."""
    own = region.own_err
    return float("inf") if own is None else float(own)


def _window_bisection_counts(
    value_parent: float,
    value_halves: float,
    error_parent: float,
    error_left: float,
    error_right: float,
    unresolved_left: bool,
    unresolved_right: bool,
    rtol: float,
) -> bool:
    """
    Whether a bisection of a window leaf P into L and R counts against its window's iroff1 count
    (prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.2 item 1; see the
    comment above _LEVIN_WINDOW_IROFF1_VALUE_RTOL). It counts if and only if neither half is
    unresolved (dqage's skip), |value_halves - value_parent| <= min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL,
    rtol) |value_halves|, and error_left + error_right >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO *
    error_parent; each inequality inclusive. value_parent is V_P and value_halves V_L + V_R, each a
    leaf's own solve's value; the errors are the leaves' own solves' errors e (used_interval.own_err),
    not the declared e'.
    """
    if unresolved_left or unresolved_right:
        return False
    value_tolerance = min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL, rtol)
    return bool(
        np.fabs(value_halves - value_parent) <= value_tolerance * np.fabs(value_halves)
        and error_left + error_right >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO * error_parent
    )


def _solve_floor_declared(data: dict) -> float:
    """A subregion solve's declared endpoint term, "phase_err_declared" (0.0 if it reports none)."""
    return data.get("phase_err_declared", 0.0) or 0.0


def _solve_floor_roundoff(data: dict) -> float:
    """
    A subregion solve's floor less its declared endpoint term, "phase_err_roundoff": the part of
    "phase_err" bisection can lower, which the round-off exit's halving test compares
    (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (d)). Every solve in
    this module reports it; a solve that does not (a test's stand-in) is read as having no declared
    term, its whole "phase_err".
    """
    if "phase_err_roundoff" in data:
        return data["phase_err_roundoff"]
    return data.get("phase_err", 0.0) or 0.0


def _solve_floor_in_budget(data: dict) -> float:
    """
    A subregion solve's floor as it enters the call's error, the lower bound the round-off exit's
    budget clause takes for its error: "phase_err_less_node" on a fallback solve, whose node error E
    the call carries signed, else "phase_err".
    """
    if data.get("phase_err_less_node") is not None:
        return data["phase_err_less_node"]
    return data.get("phase_err", 0.0) or 0.0


def _trust_noise_yardstick(
    phi_P: float, parent_is_direct: bool, data_L: dict, data_R: dict
) -> float:
    """
    The trust check's noise yardstick nu_B (prompts/levin-phase-floor/
    06m-signed-yardstick-in-the-driver.md, section 0.1; route B as
    prompts/levin-phase-floor/06l-signed-trust-yardstick.md measured it). A checked leaf with
    r < 1 is trusted on its noise clause if delta1 <= nu_B: its delta1 is then within the rounding
    of the three solves it is the difference of, V_P, V_L and V_R. Pure: it reads the parent's
    floor, whether the parent is a Clenshaw-Curtis solve, and the halves' solve dicts.

        nu_B = phi_P + sum over Levin halves of phi_h
                     + [Clenshaw-Curtis parent only: sum over Clenshaw-Curtis halves of phi^-_h
                                                     + |sum over Clenshaw-Curtis halves of E_h|]

    phi_P is the parent region's "phase_err"; phi_h a Levin half's "phase_err" (None read as 0.0);
    for a Clenshaw-Curtis half (its dict's "is_direct" true), phi^-_h is its "phase_err_less_node",
    its floor with |E| taken out of the node term, and E_h its "node_error_signed". The four cases:

        parent           halves                    nu_B
        Levin            Levin + Levin             phi_P + phi_L + phi_R
        Levin            one or two CC             phi_P + the Levin half's phi, if any
        CC               CC + CC                   phi_P + phi^-_L + phi^-_R + |E_L + E_R|
        CC               one CC, one Levin         phi_P + phi_Levin + phi^-_CC + |E_CC|

    On a Levin parent a Clenshaw-Curtis half adds nothing (the user's modification of route 1,
    prompts/levin-phase-floor/IMPLEMENTATION_STATE.md, Decisions 06k results and 06l).

    The basis (docs/levin-phase-floor/VERIFICATION.md, section 6l). On a Levin parent nu_B holds
    only Levin floors: a Clenshaw-Curtis half's floor can be hundreds of times a Levin parent's, and
    carried into its test (route 1, 06k) it passed leaves up to delta1 / phi_P = 1.1e4, which no
    noise level covers; under nu_B a Levin parent's firings reach 2.85 at most. On a
    Clenshaw-Curtis parent it holds the declared floors less the node term, plus the halves' node
    errors combined signed, as the call's own error combines them
    (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (c)): the halves' E can
    be large and nearly equal and opposite, cancelling in V_L + V_R, and their absolute values
    would pass a leaf on a term that is not in delta1. Where delta1 is the node-error difference
    |E_P - (E_L + E_R)|, the triangle inequality bounds it by |E_P| + |E_L + E_R|, both in nu_B
    (|E_P| through phi_P). nu_B <= phi_P + phi_L + phi_R to the rounding of the sum, so this
    yardstick trusts no leaf route 1's would not.

    The order is fixed: nu = phi_P; then each Levin half's phi in the order L, R; then, on a
    Clenshaw-Curtis parent with a Clenshaw-Curtis half, each such half's phi^- in the order L, R;
    then |S|, S the one half's E or E_L + E_R. It is the order 06l's variant summed in, whose
    census certifies this code by bit identity (prompt 06m, Gates M0 and M); floating-point
    addition is not associative, so another order can move the sum's last bit and with it a leaf's
    trust.

    A Clenshaw-Curtis half of a Clenshaw-Curtis parent without "phase_err_less_node" or
    "node_error_signed" raises RuntimeError. Every Clenshaw-Curtis solve returns both
    (_adaptive_levin_subregion_cc()), so this guards an internal inconsistency; it is not a path.
    """
    halves = (data_L, data_R)
    half_is_direct = [bool(half.get("is_direct", False)) for half in halves]
    nu = phi_P
    for half, is_direct in zip(halves, half_is_direct):
        if not is_direct:
            nu = nu + (half.get("phase_err", 0.0) or 0.0)
    if parent_is_direct and any(half_is_direct):
        cc_halves = [
            half for half, is_direct in zip(halves, half_is_direct) if is_direct
        ]
        for half in cc_halves:
            if (
                half.get("phase_err_less_node") is None
                or half.get("node_error_signed") is None
            ):
                raise RuntimeError(
                    "levin_quadrature: a Clenshaw-Curtis half of a Clenshaw-Curtis parent has no "
                    "phase_err_less_node or node_error_signed (_trust_noise_yardstick())"
                )
            nu = nu + half["phase_err_less_node"]
        if len(cc_halves) == 1:
            node_sum = cc_halves[0]["node_error_signed"]
        else:
            node_sum = (
                cc_halves[0]["node_error_signed"] + cc_halves[1]["node_error_signed"]
            )
        nu = nu + abs(node_sum)
    return float(nu)


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

    def total_with(self, *terms: float) -> float:
        """The correctly rounded sum of everything added and of terms, leaving the sum unchanged."""
        return math.fsum(self._partials + [float(t) for t in terms])


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
        value: Optional[float] = None,
        phase_err_declared: Optional[float] = None,
        phase_err_roundoff: Optional[float] = None,
        node_error_signed: Optional[float] = None,
        phase_err_less_node: Optional[float] = None,
        window: bool = False,
        guarded: bool = False,
        guard_at_depth_max: bool = False,
        resasc: Optional[float] = None,
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
        # floor-limited region, which is exempt), True if the trust test passes (convergence_ratio
        # < _LEVIN_TRUST_MAX_RATIO, or convergence_ratio < 1 with delta1 at most the noise
        # yardstick of _trust_noise_yardstick(); since 06g, the yardstick since 06m, phase_err
        # alone before it), False otherwise. declared_resolution, when set, replaces abserr in total_err:
        # max(abserr, delta1) / (1 - r) for a trusted region and for an untrusted one at depth_max
        # with r < 1, max(abserr, delta1 + delta2) for an untrusted one at depth_max with r >= 1.
        # abserr itself stays the raw estimate (the step-(4) residual, or the nested pair), as do
        # abserr_resolution/abserr_fallback.
        self._delta1 = None
        self._delta2 = None
        self._convergence_ratio = None
        self._trusted = None
        self._declared_resolution = None

        # The value this region contributes to the call's value, and which one it is
        # (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (a)): "own", the
        # region's own solve V_P (the Levin solve, or CC_{2N-1} on the fallback), or "quarters",
        # the quarters sum Q = (V_LL + V_LR) + (V_RL + V_RR) a trusted region returns. value_own
        # is V_P whichever is returned.
        self._value = value
        self._value_own = value
        self._value_kind = "own"

        # The floor's parts (06g, section 0.2 (c) and (d)): phase_err_declared is the declared
        # endpoint term (_declared_endpoint_phase_err()), phase_err_roundoff the rest of phase_err
        # (phase_err = phase_err_roundoff + phase_err_declared, as the solve formed it), the part
        # bisection can lower, which the round-off exit's halving test compares. On a fallback
        # region node_error_signed is E, its node term's signed first-order error, and
        # phase_err_less_node is phase_err with |E| taken out of the node term (the rule's floor
        # formed with R_g + R_2 + R_c alone, plus the declared term); both None on a Levin region.
        self._phase_err_declared = phase_err_declared
        self._phase_err_roundoff = phase_err_roundoff
        self._node_error_signed = node_error_signed
        self._phase_err_less_node = phase_err_less_node

        # The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md):
        # window marks a piece of the window round a located stationary point (or one of its
        # descendants), solved by the nested Clenshaw-Curtis pair whatever its phase span; guarded
        # marks a region whose Levin solve found a phase rate not of one strict sign on its own
        # Chebyshev nodes; guard_at_depth_max marks such a region at depth_max, re-solved by the
        # nested pair. See _adaptive_levin().
        self._window = window
        self._guarded = guarded
        self._guard_at_depth_max = guard_at_depth_max

        # dqk21's resasc on a Clenshaw-Curtis region's fine rule (prompts/levin-phase-floor/
        # 07a-window-stop-and-phase-variation.md section 1.2), None on a Levin region. Read only by
        # the window's iroff1 count (_cc_leaf_unresolved()); it enters no value or error.
        self._resasc = resasc

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
        if self._value_kind != "own":
            phase_label = phase_label + f", value={self._value_kind}"
        if self._window:
            phase_label = phase_label + " (WINDOW)"
        if self._guard_at_depth_max:
            phase_label = phase_label + " (GUARD AT DEPTH_MAX)"
        elif self._guarded:
            phase_label = phase_label + " (GUARDED)"

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
        True if _adaptive_levin() chose this region (leaf) for refinement and found it
        floor-limited: its round-off floor is finite, its residual is at most that floor, its two
        halves' floors less their declared endpoint terms sum to at least
        _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its own less its own, and the split could not
        meet the call's request (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md),
        so bisecting it would not lower its error materially. Never True when phase_err is not
        finite. A region the loop never chose is never flagged, whatever its floor -- see
        _adaptive_levin().

        Also True for every leaf of a stationary-point window that the iroff1 rule stopped
        (prompts/levin-phase-floor/07-stationary-point-guard.md section 1.3, its count since
        prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.2 dqage's;
        _LEVIN_WINDOW_IROFF1_LIMIT): six of the window's bisections, each with neither half
        unresolved in dqk21's sense, changed the value by at most min(1e-5, rtol) relatively
        without lowering the halves' own error estimates (before the trust check widens them), so
        the window is round-off-limited and no leaf of it is refined further, whether or not the
        loop chose that leaf.
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
        Whether this region's error estimate passed the trust check one level down: r <
        _LEVIN_TRUST_MAX_RATIO, or r < 1 with delta1 at most the noise yardstick of
        _trust_noise_yardstick() (since prompts/levin-phase-floor/
        06m-signed-yardstick-in-the-driver.md; from prompts/levin-phase-floor/
        06g-leaf-values-and-node-sum.md to then, at most phase_err; before 06g, r < 1). None if the
        check was not taken on it: a floor-limited region is exempt, and a region at depth_max
        returned by a call whose budget was never met need not have been checked. A region
        returned with trusted False is at depth_max, and declares max(abserr, delta1) / (1 - r) as
        its resolution error if r < 1, max(abserr, delta1 + delta2) if not; see _adaptive_levin().
        """
        return self._trusted

    @property
    def declared_resolution(self) -> Optional[float]:
        """
        The resolution error this region declares in total_err when the trust check changed it:
        max(abserr, delta1) / (1 - r) if trusted, or if untrusted at depth_max with r < 1;
        max(abserr, delta1 + delta2) if untrusted at depth_max with r >= 1; None if total_err uses
        abserr as it stands.
        """
        return self._declared_resolution

    @property
    def value(self) -> Optional[float]:
        """
        The value this region contributes to the call's value: its own solve's (value_own, V_P),
        or, if the trust check trusted it, its quarters sum Q (value_kind "quarters"; since
        prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md). The call's value is the sum of
        these over its regions, in ascending x.
        """
        return self._value

    @property
    def value_kind(self) -> str:
        """Which value the region returns: "own" (V_P) or "quarters" (Q); see value."""
        return self._value_kind

    @property
    def value_own(self) -> Optional[float]:
        """V_P, the region's own solve's value, whichever value it returns; see value."""
        return self._value_own

    @property
    def phase_err_declared(self) -> Optional[float]:
        """
        The declared endpoint term in phase_err (_declared_endpoint_phase_err(); 0.0 when the phase
        declares no theta_abserr). The round-off exit's halving test leaves it out; see
        _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO.
        """
        return self._phase_err_declared

    @property
    def phase_err_roundoff(self) -> Optional[float]:
        """phase_err less its declared endpoint term: eq. (151)'s floor on a Levin region, the
        Clenshaw-Curtis rule's floor, with its node term, on a fallback region."""
        return self._phase_err_roundoff

    @property
    def node_error_signed(self) -> Optional[float]:
        """
        E, the signed first-order error of a fallback region's nodes (_cc_node_term_parts()), whose
        absolute value is in phase_err's node term; None on a Levin region. _adaptive_levin()
        carries the sum of E over the call's fallback regions signed, |sum E| in place of
        sum |E| (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (c)).
        """
        return self._node_error_signed

    @property
    def phase_err_less_node(self) -> Optional[float]:
        """phase_err with |E| taken out of the node term, on a fallback region; None on a Levin
        region. See budget_err."""
        return self._phase_err_less_node

    @property
    def total_err(self) -> Optional[float]:
        """
        Combined per-region error estimate: max(resolution/fallback residual, round-off floor),
        plus the truncation term (prompt 06) if this region has one. The truncation term is added
        rather than maxed because it is a distinct, always-present bias -- the endpoint
        contribution of modes the rtol gate discarded from this region's own value -- not another
        estimate of the same underlying quantity the max() is choosing between.

        The resolution error in the max is declared_resolution when the trust check one level
        down has set it (widened by the observed convergence ratio), else abserr.

        This is the region's own error, its node term's |E| included. It is the leaf error e_i of
        _adaptive_levin()'s global budget on a Levin region; on a fallback region the budget takes
        budget_err, which leaves |E| out, and carries |sum E| over the call's fallback regions
        instead (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md; until then
        every region entered the budget by this).
        """
        return self._combine(self._phase_err)

    @property
    def budget_err(self) -> Optional[float]:
        """
        The region's contribution e'_i to _adaptive_levin()'s budget and its returned "abserr"
        (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (c)): total_err with
        |E| taken out of the floor, max(resolution, phase_err_less_node) plus truncation, on a
        fallback region; total_err itself on a Levin region, which has no node term. The call's
        error is sum_i e'_i + |sum_i E_i|: the node errors are carried signed, since their signs
        vary from leaf to leaf, and the sum of |E| does not fall under bisection.
        """
        if self._phase_err_less_node is None:
            return self._combine(self._phase_err)
        return self._combine(self._phase_err_less_node)

    @property
    def own_err(self) -> Optional[float]:
        """
        The error the region's own solve declares (prompts/levin-phase-floor/
        07a-window-stop-and-phase-variation.md section 1.2 item 1, the raw estimate e): budget_err
        with the resolution term its own abserr, before the trust check widens a declaration --
        max(abserr, floor) plus truncation, the floor less |E| on a fallback region. It is what
        QUADPACK's dqage compares in its iroff1 count; the window's count reads it.
        """
        floor = (
            self._phase_err_less_node
            if self._phase_err_less_node is not None
            else self._phase_err
        )
        return self._combine_terms(self._abserr, floor)

    @property
    def resasc(self) -> Optional[float]:
        """dqk21's resasc on a Clenshaw-Curtis region's fine rule; None on a Levin region."""
        return self._resasc

    def _combine(self, floor: Optional[float]) -> Optional[float]:
        resolution = (
            self._declared_resolution
            if self._declared_resolution is not None
            else self._abserr
        )
        return self._combine_terms(resolution, floor)

    def _combine_terms(
        self, resolution: Optional[float], floor: Optional[float]
    ) -> Optional[float]:
        if resolution is None:
            base = floor
        elif floor is None:
            base = resolution
        else:
            base = max(resolution, floor)

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

    @property
    def window(self) -> bool:
        """
        True for a piece of the window round a located stationary point, or a descendant of one:
        solved by the nested Clenshaw-Curtis pair whatever its phase span. The floor-limited exit
        never fires on it; it is floor-limited only when its window is stopped by the iroff1 rule
        (prompts/levin-phase-floor/07-stationary-point-guard.md sections 1.2-1.3; see
        _adaptive_levin()).
        """
        return self._window

    @property
    def guarded(self) -> bool:
        """
        True if this region's Levin solve found a phase rate not of one strict sign on its own
        Chebyshev nodes (the stationary-point guard; see _adaptive_levin()). Such a region is
        never returned as a Levin region: below depth_max it is bisected, and at depth_max it is
        re-solved by the nested Clenshaw-Curtis pair (guard_at_depth_max).
        """
        return self._guarded

    @property
    def guard_at_depth_max(self) -> bool:
        """
        True for a guarded region at depth_max, re-solved by the nested Clenshaw-Curtis pair
        whatever its phase span; its errors are that solve's, and the call is not converged while
        one is returned (see _adaptive_levin()).
        """
        return self._guard_at_depth_max


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

    The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md) adds two
    flags. "forced_direct": the leaf, its halves and its grandchildren are solved by the nested
    Clenshaw-Curtis pair whatever their phase span -- a window leaf, or a guarded leaf re-solved at
    depth_max. "guarded": a Levin leaf below depth_max whose phase rate is not of one strict sign
    on its own nodes; its error is read as +inf (see error), so it is chosen before every finite
    one and the budget cannot be met while it is in the partition, and it is never floor-limited.
    And, for section 1.3's rule, "window_index": the merged window a window leaf belongs to (None
    for every other leaf); "fell_at_last_bisection": for a window leaf made by a bisection, whether
    that bisection lowered the summed declared error (e'_L + e'_R below
    _LEVIN_WINDOW_IROFF1_ERROR_RATIO of the parent's e'), None for a leaf made by seeding. It reports
    on e', as at prompt 07, and is not the count's error test: since prompts/levin-phase-floor/
    07a-window-stop-and-phase-variation.md section 1.2 the count compares the leaves' own solves'
    errors e (used_interval.own_err), as dqage compares its raw estimates, so a bisection can count
    while its halves' e' fell (a parent's e' widened by the trust check) and fail to count while it
    did not. It feeds "leaves_falling_at_stop".
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
        "forced_direct",
        "guarded",
        "window_index",
        "fell_at_last_bisection",
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
        forced_direct: bool = False,
        guarded: bool = False,
        window_index: Optional[int] = None,
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
        self.forced_direct = forced_direct
        self.guarded = guarded
        self.window_index = window_index
        self.fell_at_last_bisection = None

    @property
    def width(self) -> float:
        return np.fabs(self.end - self.start)

    @property
    def break_point(self) -> float:
        return (self.end + self.start) / 2.0

    @property
    def error(self) -> float:
        """
        This leaf's contribution e'_i to the call's error: used_interval.budget_err of its region
        (total_err on a Levin leaf; on a fallback leaf, total_err with its node error |E| taken out
        of the floor, which the call carries signed; since
        prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md).

        A guarded leaf's is +inf (prompts/levin-phase-floor/07-stationary-point-guard.md section
        1.4): its Levin estimate holds a sign change of the phase rate, where neither the step-(4)
        residual nor the round-off floor sees the error, so it is never returned as it stands.
        """
        if self.guarded:
            return float("inf")
        return self.region.budget_err


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

    The end nodes are the endpoints exactly: x[0] == b and x[-1] == a, bit for bit (since
    prompts/levin-phase-floor/06e-exact-end-nodes.md). The affine map
    fl((a + b)/2 + fl((b - a)/2 * x_base)) does not give them, although x_base[0] = 1 and
    x_base[-1] = -1 exactly: the rounding of the midpoint, of the half-width and of their sum leaves
    an end node one ulp off its endpoint in about a third of spans
    ([06c-chebyshev-end-nodes-miss-the-span]). A Levin solve then integrates a shifted interval, an
    error of about |f| ulp(x) that is no round-off floor's, and a parent's ends and its halves' fail
    to tile, so the step-(4) residual sees that offset at every depth and _adaptive_levin() chased
    it to depth_max. So the two end nodes are set after the map. The interior nodes and D are the
    map's, unchanged. Since _adaptive_levin() hands both halves of a bisected region the same float
    midpoint, a region and its halves tile exactly.
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

    # the end nodes are the endpoints exactly (see above): x is a new array, so this cannot touch
    # the cached base.
    x[0] = b
    x[-1] = a

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


@lru_cache(maxsize=32)
def _chebyshev_base_corrections(N: int):
    """
    The base nodes of _chebyshev_base(N) in double-double: corr[k] = fl(t_k - x_base[k]), with
    t_k = cos(pi k / (N - 1)) the exact node, so that t_k = x_base[k] + corr[k] to within
    u |corr[k]| (prompts/levin-phase-floor/06e-exact-end-nodes.md, section 0.2 item 2). Returns
    (corr, max_k |corr[k]|).

    Computed once per order in a private mpmath context at 160 bits -- t_k to about 1e-48, far
    below the u^2 ~ 1e-32 the double-double holds -- and rounded to the nearest double. Cached, and
    read-only, as _chebyshev_base() is, for the same reason. corr[0] = corr[-1] = 0, since
    x_base[0] = 1 and x_base[-1] = -1 exactly. mpmath is imported here, not at module scope: it is
    needed only on the first fallback solve at each order.
    """
    import mpmath

    x_base, _ = _chebyshev_base(N)
    ctx = mpmath.MPContext()
    ctx.prec = 160
    corr = np.array(
        [
            float(ctx.cos(ctx.pi * k / (N - 1)) - ctx.mpf(float(x_base[k])))
            for k in range(N)
        ]
    )
    corr.setflags(write=False)
    return corr, float(np.max(np.fabs(corr)))


# Veltkamp's splitter, 2^27 + 1, for _two_product().
_DEKKER_SPLITTER = 134217729.0


def _two_sum(a, b):
    """
    Knuth's TwoSum, elementwise: (s, e) with s = fl(a + b) and a + b = s + e exactly, for any
    floats a, b (barring overflow).
    """
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_product(a, b):
    """
    Dekker's TwoProduct with Veltkamp's split, elementwise: (p, e) with p = fl(a b) and a b = p + e
    exactly, provided nothing overflows (|a|, |b| below about 1e300) or underflows (e above the
    smallest normal float, about 2e-308). Spans whose coordinates and widths are within 1e-280 ...
    1e300 of unity satisfy this by a wide margin.
    """
    p = a * b
    c = _DEKKER_SPLITTER * a
    a_hi = c - (c - a)
    a_lo = a - a_hi
    c = _DEKKER_SPLITTER * b
    b_hi = c - (c - b)
    b_lo = b - b_hi
    return p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def _chebyshev_node_displacements(x_span: Tuple[float, float], N: int):
    """
    Delta_k = x^_k - x_k for every node of chebyshev_matrices(x_span, N): the computed node against
    the exact one, x_k = (a + b)/2 + (b - a)/2 cos(pi k / (N - 1)) in exact arithmetic, signed
    (prompts/levin-phase-floor/06e-exact-end-nodes.md, section 0.2 item 2). Returns
    (Delta, err_abs): every computed Delta^_k satisfies

        |Delta^_k - Delta_k| <= 2 u |Delta^_k| + err_abs,
        err_abs = u (u (M + h) + h max_k |corr_k|),   M = max(|a|, |b|), h = |b - a|,

    so Delta is known to a few u of itself wherever |Delta_k| is above about u (M + h) times u.

    The construction, by error-free transformations (_two_sum(), _two_product()), with the base
    nodes in double-double, t_k = x_base[k] + corr[k] (_chebyshev_base_corrections()):
      * s + s_e = a + b and d + d_e = b - a exactly (TwoSum); chebyshev_matrices() forms the
        midpoint m^ = s/2 and the half-width r^ = d/2, both halvings exact;
      * p + p_e = r^ x_base[k] exactly (TwoProduct), p being the product chebyshev_matrices() forms;
      * x^_k + x_e = m^ + p exactly (TwoSum), x^_k being the node it returns;
      * so x_k = m^ + s_e/2 + (r^ + d_e/2)(x_base[k] + corr[k]) expands into x^_k and eight small
        terms, x_e, s_e/2, p_e, q + q_e = r^ corr[k], r + r_e = (d_e/2) x_base[k] (each pair exact by
        TwoProduct) and fl((d_e/2) corr[k]), and Delta_k is minus their sum.
    The eight terms are summed by Ogita, Rump and Oishi's Sum2 (cascaded TwoSum), which errs by at
    most u |sum| + gamma_7^2 sum |terms|; the terms are each at most about u (M + h), so the second
    part is below u^2 (M + h) u. The other errors: fl((d_e/2) corr[k]) by u of a term of size
    u^2 h; and corr[k] itself by at most u |corr[k]|, times r^ <= h/2. err_abs covers them all with
    a wide margin.

    The end nodes are the endpoints exactly (chebyshev_matrices()), so Delta_0 = Delta_{N-1} = 0.
    x_span's entries are taken as the floats chebyshev_matrices() is handed.
    """
    a, b = x_span
    a = float(a)
    b = float(b)
    x_base, _ = _chebyshev_base(N)
    corr, corr_max = _chebyshev_base_corrections(N)

    s, s_err = _two_sum(a, b)
    d, d_err = _two_sum(b, -a)
    midpoint = s / 2.0
    half_width = d / 2.0
    half_d_err = d_err / 2.0

    p, p_err = _two_product(half_width, x_base)
    _, x_err = _two_sum(midpoint, p)
    q, q_err = _two_product(half_width, corr)
    r, r_err = _two_product(half_d_err, x_base)
    tail = half_d_err * corr

    terms = (
        x_err,
        np.full_like(x_base, s_err / 2.0),
        p_err,
        q,
        q_err,
        r,
        r_err,
        tail,
    )
    total = terms[0]
    compensation = np.zeros_like(x_base)
    for term in terms[1:]:
        total, e = _two_sum(total, term)
        compensation = compensation + e
    delta = -(total + compensation)
    delta[0] = 0.0
    delta[-1] = 0.0

    M = max(abs(a), abs(b))
    h = abs(b - a)
    err_abs = _UNIT_ROUNDOFF * (_UNIT_ROUNDOFF * (M + h) + h * corr_max)
    return delta, err_abs


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


def _node_partition_variation(theta_nodes: np.ndarray) -> float:
    """
    The variation of a phase on a region's node partition, sum_j |theta(x_{j+1}) - theta(x_j)|
    over its nodes in order, summed correctly rounded (math.fsum, so the order of the nodes does
    not change the bits): exact where the phase is monotone between adjacent nodes, a lower bound
    on its total variation otherwise. The variation V a basis reports from build_Levin_data() on a
    region whose rate changes sign on its nodes (prompts/levin-phase-floor/
    07a-window-stop-and-phase-variation.md section 1.1 item 3).
    """
    return float(math.fsum(np.fabs(np.diff(np.asarray(theta_nodes, dtype=float)))))


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

    def independent_phases(self) -> list:
        """
        The basis interface's phases (prompts/levin-phase-floor/07-stationary-point-guard.md
        section 1.0): one callable per independent phase of the basis. This basis has one, the raw
        phase theta["theta"]. _adaptive_levin() finds the edges of a stationary point's window on
        it.
        """
        return [self._theta]

    def independent_phase_rates(self) -> Optional[list]:
        """
        The basis interface's phase rates (section 1.0): one callable per phase, its d theta / dx in
        closed form, in the order of independent_phases(); or None when the basis cannot supply
        them. Here [theta["theta_deriv"]] when the caller supplied it, else None -- a theta' got by
        differentiating sampled phase values is not a closed form, and the location would find its
        noise. With None, _adaptive_levin() locates nothing, and the nodal guard is the only
        protection at a stationary point.
        """
        if hasattr(self, "_theta_deriv"):
            return [self._theta_deriv]
        return None

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
        at each endpoint, and V, the phase's total variation across the interval, which the C2
        gate and the LU choice route on (_adaptive_levin_subregion_impl()).

        V (prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.1 item 3)
        is computed from the raw phase theta["theta"], not estimated from theta':
          * if theta' on the region's nodes (theta_prime_Cheb, from theta_deriv or by spectral
            differentiation) has no two samples of opposite strict sign (_changes_sign() False),
            V = |theta(b) - theta(a)|, with theta at the region's two end nodes, which are its
            endpoints exactly: exact where theta is monotone, from two evaluations of the phase.
            Its rounding is about ulp(theta), 1.5e-8 rad for a raw phase near 1e8, negligible
            against 6 pi;
          * otherwise theta is not monotone on the region and |theta(b) - theta(a)| can read near
            zero, so V is the variation on the node partition, sum_j |theta(x_{j+1}) - theta(x_j)|
            over the nodes in order, summed correctly rounded (math.fsum): exact where theta is
            monotone between adjacent nodes, a lower bound otherwise, which sends a region towards
            the Clenshaw-Curtis rule, right whatever its stationary points do.
        The raw phase is sampled on the nodes only in the second case (or where it is sampled
        already, below), and at the two ends otherwise, through _sample_vectorized(). Until 07a
        the method returned the estimate mean|theta'| * width, biased by the nodes' crowding to
        the ends and not monotone under bisection (see SIX_PI's comment). This basis has one
        phase, so V is that phase's.

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
        :return: (AmatT, theta_prime_Cheb, w0, wk, phase_span), phase_span being V above. AmatT
            is None when need_AmatT is False. Prior to prompt 04 this also returned a sixth element, theta_scale, used only
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

        # V, the phase's total variation across the region, from the raw phase (prompt 07a section
        # 1.1 item 3; see the docstring): |theta(b) - theta(a)| where theta' has no strict sign
        # change on the nodes, the node-partition sum where it has. It decides the C2 gate and
        # whether the Levin super-operator is well enough conditioned to be inverted by a direct
        # solve. grid is descending: grid[0] is the upper endpoint b, grid[-1] the lower a.
        if _changes_sign(theta_prime_Cheb):
            theta_nodes = theta_Cheb
            if theta_nodes is None:
                theta_nodes = _sample_vectorized(
                    self._theta, grid, vectorize_cache, ("theta", id(self))
                )
            phase_span = _node_partition_variation(theta_nodes)
        else:
            if theta_Cheb is not None:
                theta_b, theta_a = theta_Cheb[0], theta_Cheb[-1]
            else:
                theta_b, theta_a = _sample_vectorized(
                    self._theta,
                    np.array([grid[0], grid[-1]]),
                    vectorize_cache,
                    ("theta", id(self)),
                )
            phase_span = float(np.fabs(theta_b - theta_a))

        if not np.isfinite(phase_span):
            _logger.warning(
                "!! WARNING (adaptive_levin_subregion, %s): sampled phase theta contains "
                "non-numeric values (np.nan, np.inf, or np.-inf)",
                _format_label(notify_label, id_label),
            )
            raise ValueError(
                "sampled phase theta contains non-numeric values (np.nan, np.inf, or np.-inf)"
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

    @staticmethod
    def basis_derivative(basis_values: np.ndarray) -> np.ndarray:
        """
        The basis functions' derivatives with respect to theta, at the points whose basis values
        are given (shape (2, n), as the Clenshaw-Curtis fallback holds them): d sin/d theta = cos and
        d cos/d theta = -sin, so (b_1, -b_0). Exact: a permutation and a sign, no rounding. Used by
        the fallback's node term (_cc_node_term_parts()).
        """
        return np.vstack([basis_values[1], -basis_values[0]])

    def theta_prime_on_fine_grid(
        self,
        fine_grid: np.ndarray,
        D_fine: np.ndarray,
        theta_prime_coarse: np.ndarray,
        arguments_fine: np.ndarray,
        vectorize_cache: dict,
    ) -> Tuple[np.ndarray, float]:
        """
        theta' at every node of the Clenshaw-Curtis fallback's fine grid, for its node term
        (_cc_node_term_parts(); prompts/levin-phase-floor/06e-exact-end-nodes.md), with a bound on
        its relative error: (theta_prime_fine, eta_theta).

        * With "theta_deriv" supplied, the caller's theta' at each node: the even nodes are the
          coarse grid's (fine_grid[::2] is chebyshev_matrices(x_span, N)'s grid, bit for bit), where
          theta_prime_coarse already holds it, and the odd ones are evaluated here, one evaluation
          per node, N - 1 in all. eta_theta = _CC_THETA_PRIME_REL_ERROR_SUPPLIED.
        * Without it, the fine grid's spectral derivative of the raw theta sampled at the fine
          nodes, as build_Levin_data() takes theta' on the coarse grid. Without "theta_mod_2pi"
          the raw theta at the fine nodes is the argument the basis was evaluated from, already in
          hand (no evaluation); with it, theta is evaluated at each fine node, 2N - 1 in all.
          eta_theta = _CC_THETA_PRIME_REL_ERROR_SPECTRAL.

        :param theta_prime_coarse: theta' on the coarse grid fine_grid[::2], as build_Levin_data()
            returned it.
        :param arguments_fine: the arguments the fine-grid basis values were computed from, shape
            (2, n), as eval_basis_and_arguments() returned them.
        """
        if hasattr(self, "_theta_deriv"):
            theta_prime_fine = np.empty_like(fine_grid)
            theta_prime_fine[0::2] = theta_prime_coarse
            theta_prime_fine[1::2] = _sample_vectorized(
                self._theta_deriv,
                fine_grid[1::2],
                vectorize_cache,
                ("theta_deriv", id(self)),
            )
            eta_theta = _CC_THETA_PRIME_REL_ERROR_SUPPLIED
        else:
            if hasattr(self, "_theta_mod_2pi"):
                theta_fine = _sample_vectorized(
                    self._theta, fine_grid, vectorize_cache, ("theta", id(self))
                )
            else:
                theta_fine = np.asarray(arguments_fine[0], dtype=float)
            theta_prime_fine = np.matmul(D_fine, theta_fine)
            eta_theta = _CC_THETA_PRIME_REL_ERROR_SPECTRAL

        return theta_prime_fine, eta_theta


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
    force_direct: bool = False,
    guard_rates: Optional[list] = None,
    guard_sign_change_phases: Optional[frozenset] = None,
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
            force_direct=force_direct,
            guard_rates=guard_rates,
            guard_sign_change_phases=guard_sign_change_phases,
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
    force_direct: bool = False,
    guard_rates: Optional[list] = None,
    guard_sign_change_phases: Optional[frozenset] = None,
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
    :param force_direct: solve the region by the nested Clenshaw-Curtis pair whatever its phase
        span, bypassing the C2 gate below (prompts/levin-phase-floor/07-stationary-point-guard.md
        section 1.2 item 4): for a piece of a stationary point's window, and for a guarded region
        at depth_max. The value then keeps x_span's orientation (_oriented_cc()).
    :param guard_rates: for a basis with several independent phases, their closed-form rates
        (independent_phase_rates()), each sampled on this region's nodes for the nodal guard
        below; None for a basis with one phase, whose theta_prime_Cheb is its rate already.
    :param guard_sign_change_phases: the indices into guard_rates of the phases the location left
        unlocated, each with a rate exactly zero at every finite scan sample (prompts/
        levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.1; until 07a, prompt
        07 section 10's Rule A): the guard tests each of them for a sign change on the nodes
        (samples of both strict signs) rather than for one strict sign, so that a rate
        identically zero, or zero on a node, does not guard the region (_nodal_guard()). None, or
        empty, when no phase was skipped.

    The C2 gate, and what it does not see. A region whose total phase variation V is below 6 pi
    goes to the nested Clenshaw-Curtis rule. V is computed from the raw phase by the basis's
    build_Levin_data() (prompt 07a section 1.1): exactly, as |theta(b) - theta(a)|, where theta'
    has no strict sign change on the region's nodes, and as the variation on the node partition,
    a lower bound, where it has. Until 07a it was estimated as mean|theta'| on the nodes times the
    width, an estimate whose bias depended on the region's shape and which a half of a region
    could read above its parent's; V is additive where theta is monotone, so a half of a region
    the gate sends to the fallback goes to the fallback too. The threshold 6 pi is unchanged, and
    unmeasured ([07-c2-gate-threshold-was-never-measured]). That gate was the module's whole
    account of a stationary
    point, and the account was incomplete (prompt 07, after docs/levin-phase-floor/06/p1/report.md):
    it acts only on a region bisection has already produced, and nothing makes bisection reach a
    stationary point inside a region of large variation, whose step-(4) residual cannot see it
    (see _adaptive_levin()). So two things now sit beside the gate. The driver locates stationary
    points before integrating and hands their windows here with force_direct; and every Levin solve
    reports whether a phase rate fails to be of one strict sign on its own nodes ("guarded" in the
    result: theta_prime_Cheb, and each of guard_rates), which the driver never returns as a Levin
    region. A zero on a node counts as failing. The guard cannot see an even number of sign changes
    between two adjacent nodes; that is what the location is for. Since prompt 07a the location
    skips only a phase whose rate is exactly zero at every finite scan sample; for a basis of
    several phases the guard tests such a phase for a sign change on the nodes
    (guard_sign_change_phases).
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

    # Sample the phase once, before deciding anything else. build_Levin_data returns phase_span =
    # V, the total variation of the phase across this region computed from the raw phase (prompt
    # 07a section 1.1: exact where theta' has no strict sign change on the nodes, the node-partition
    # sum, a lower bound, where it has), at a cost of the raw phase at the two end nodes, or at
    # every node in the second case -- so it is available *before* we decide whether to solve the
    # Levin system at all, and before sampling f.
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
    # oscillates heavily either side of it. See the module docstring and prompt 03's log. Since
    # prompt 07a the variation is V, computed from the raw phase (build_Levin_data()): the net
    # change |theta(b) - theta(a)| only where theta' has no strict sign change on the nodes, where
    # it equals the total variation; the node-partition sum where it has. It replaces the estimate
    # mean|theta'| * width, which read 1.27-1.32 times high for a near-centred root and up to 8.43
    # times high with |theta'| weighted to one end, and could send a half of a Clenshaw-Curtis
    # region to the Levin solve (prompt 07a section 0.2). SIX_PI is unchanged, and unmeasured.
    #
    # This also means the fallback is chosen *before* f is sampled at all: a region routed to the
    # nested Clenshaw-Curtis rule below samples f on its own (2*chebyshev_order - 1)-point grid,
    # never on this one, so there is no wasted evaluation of f for a region that turns out not to
    # need the Levin solve.
    #
    # force_direct (prompt 07) takes the same rule whatever the span: a window piece, or a guarded
    # region at depth_max. Through _oriented_cc(), so that its value keeps x_span's orientation.
    if force_direct:
        return _oriented_cc(
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
    # The phase span compared is V, as at the C2 gate above (since prompt 07a; until then the
    # estimate mean|theta'| * width).
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

    # The nodal guard (prompt 07 section 1.4; see this function's docstring): from the samples of
    # theta' this solve already holds, and for a basis with several phases each of their rates on
    # the same nodes.
    guarded = _nodal_guard(
        theta_prime_Cheb, grid, guard_rates, vectorize_cache, guard_sign_change_phases
    )

    return {
        "value": upper_limit - lower_limit,
        "p_sample": p_sample,
        "p_ratios": p_ratios,
        "phase_span": phase_span,
        "phase_err": round_off_err + declared_err,
        # phase_err's two parts, each as it was added (prompts/levin-phase-floor/
        # 06g-leaf-values-and-node-sum.md, section 0.2 (d)): the round-off exit's halving test
        # compares the eq. (151) floor alone, the part bisection can lower
        "phase_err_roundoff": round_off_err,
        "phase_err_declared": declared_err,
        "abserr_truncation": abserr_truncation,
        "metadata": metadata,
        "is_direct": False,
        # prompt 07: a phase rate not of one strict sign on this region's nodes
        "guarded": guarded,
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
    against scipy.quad's unbounded panel count under a tight global tolerance. Its round-off floor
    adds N - 1 evaluations of the caller's theta_deriv, at the odd fine nodes, where one is
    supplied (with none, 2N - 1 of the raw phase if theta_mod_2pi is supplied, else none):
    see _Basis_SinCos.theta_prime_on_fine_grid().

    :param phase_span: total phase variation V across x_span, already computed by the caller's
        call to BasisData.build_Levin_data() -- passed in rather than recomputed, and only
        reported in the returned dict (since prompt 07a, computed from the raw phase).
    The round-off floor, "phase_err", is this rule's own (_cc_roundoff_floor(), since
    prompts/levin-phase-floor/06d-cc-roundoff-floor.md): the rounding of the weighted sum, of
    the argument handed to sin and cos, and of the nodes, from the fine-grid samples -- plus the
    declared-phase term. It is not Chen et al. eq. (151), which the Levin branch uses: that bound is the
    conditioning of the Levin collocation solve, with a 1/min|theta'| this rule does not have.

    :param theta_prime_Cheb: theta' sampled at the order-chebyshev_order collocation points,
        likewise already computed by the caller's call to build_Levin_data(). It gives theta' at
        the even fine nodes, which are those points, in the round-off floor's node term (until 06d
        it gave eq. (151)'s G0/G1; under 06d its maximum bounded |theta'| at every node).
    :return: a dict with the same "value"/"phase_span"/"phase_err"/"phase_err_roundoff"/
        "phase_err_declared"/"metadata" keys as the Levin path, "abserr_direct" in place of a
        p-based estimate, "is_direct": True, and "p_sample"/"p_ratios" both None (the Levin
        antiderivative concept does not apply here); and, since
        prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, the node term's parts,
        "node_error_signed" (E) and "node_error_bound" (R_g + R_2 + R_c), and "phase_err_less_node",
        the floor with |E| taken out of the node term; and, since prompts/levin-phase-floor/
        07a-window-stop-and-phase-variation.md, "resasc", dqk21's sum_j |w_j| |g_j - gbar| on the
        fine rule, read only by the window's iroff1 count.
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
    # Its node term (since prompts/levin-phase-floor/06e-exact-end-nodes.md) is the first-order
    # error of the displacement each fine node actually has, signed (_chebyshev_node_displacements()),
    # with bounded remainders (_cc_node_term_parts()). It takes theta' at each fine node -- the
    # caller's theta_deriv at the odd nodes, the coarse sample at the even ones, or a spectral
    # derivative where none is supplied (_Basis_SinCos.theta_prime_on_fine_grid()) -- and f', f''
    # and theta'' from the fine grid's own Chebyshev differentiation matrix applied to the samples
    # in hand.
    f_prime_fine = np.vstack([np.matmul(D_fine, f_row) for f_row in f_fine])
    f_second_fine = np.vstack([np.matmul(D_fine, f_row) for f_row in f_prime_fine])
    theta_prime_fine, eta_theta = BasisData.theta_prime_on_fine_grid(
        fine_grid, D_fine, theta_prime_Cheb, arguments_fine, vectorize_cache
    )
    if not np.isfinite(theta_prime_fine).all():
        _logger.warning(
            "!! WARNING (adaptive_levin_subregion, %s): sampled phase derivative theta' "
            "contains non-numeric values (np.nan, np.inf, or np.-inf)",
            _format_label(notify_label, id_label),
        )
        raise ValueError(
            "sampled phase derivative theta' contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )
    theta_second_fine = np.matmul(D_fine, theta_prime_fine)
    displacements, displacement_err_abs = _chebyshev_node_displacements(
        x_span, fine_order
    )
    E, R_g, R_2, R_c = _cc_node_term_parts(
        w_fine,
        displacements,
        displacement_err_abs,
        f_fine,
        f_prime_fine,
        f_second_fine,
        basis_fine,
        BasisData.basis_derivative(basis_fine),
        arguments_fine,
        theta_prime_fine,
        theta_second_fine,
        eta_theta,
    )
    round_off_err = _cc_roundoff_floor(
        w_fine,
        f_fine,
        basis_fine,
        arguments_fine,
        abs(E) + R_g + R_2 + R_c,
    )
    # The same floor with |E| taken out of its node term (prompts/levin-phase-floor/
    # 06g-leaf-values-and-node-sum.md, section 0.2 (c)): E's sign varies from leaf to leaf, so
    # _adaptive_levin() carries sum E signed over the call's fallback leaves, and each leaf enters
    # the call's error with this floor in place of round_off_err. Formed as its own sum, not by
    # subtracting |E| from round_off_err, so that it carries no cancellation.
    node_bound = R_g + R_2 + R_c
    round_off_err_less_node = _cc_roundoff_floor(
        w_fine,
        f_fine,
        basis_fine,
        arguments_fine,
        node_bound,
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

    # dqk21's resasc on the fine rule (prompts/levin-phase-floor/07a-window-stop-and-phase-
    # variation.md section 1.2 item 1): sum_j |w_j| |g_j - gbar|, gbar = CC_{2N-1} / sum_j w_j, the
    # rule's integral of |g - mean g|. It decides whether the leaf's error estimate is unresolved
    # (_cc_leaf_unresolved()), which the window's iroff1 count reads; it enters no value or error.
    weight_sum = float(np.sum(w_fine))
    integrand_mean = value_fine / weight_sum if weight_sum != 0.0 else 0.0
    resasc = float(np.sum(np.fabs(w_fine) * np.fabs(integrand_fine - integrand_mean)))

    return {
        "value": value_fine,
        "abserr_direct": abserr_direct,
        "resasc": resasc,
        "p_sample": None,
        "p_ratios": None,
        "phase_span": phase_span,
        "phase_err": round_off_err + declared_err,
        # phase_err's parts (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section
        # 0.2 (c) and (d)): the rule's floor, with its node term, and the declared endpoint term,
        # each as it was added; the node term's signed first-order error E and the rest of the
        # node term, R_g + R_2 + R_c; and phase_err with |E| taken out of the node term
        "phase_err_roundoff": round_off_err,
        "phase_err_declared": declared_err,
        "node_error_signed": E,
        "node_error_bound": node_bound,
        "phase_err_less_node": round_off_err_less_node + declared_err,
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


def _strictly_one_sign(values: np.ndarray) -> bool:
    """True if every entry of values is > 0, or every entry is < 0 (a zero or a NaN is neither)."""
    return bool(np.all(values > 0.0) or np.all(values < 0.0))


def _changes_sign(values: np.ndarray) -> bool:
    """True if values holds an entry > 0 and an entry < 0 (a zero or a NaN is neither)."""
    return bool(np.any(values > 0.0) and np.any(values < 0.0))


def _nodal_guard(
    theta_prime_Cheb: np.ndarray,
    grid: np.ndarray,
    guard_rates: Optional[list],
    vectorize_cache: dict,
    guard_sign_change_phases: Optional[frozenset] = None,
) -> bool:
    """
    The stationary-point guard on one Levin solve (prompts/levin-phase-floor/
    07-stationary-point-guard.md section 1.4): True if a phase rate is not of one strict sign on
    the region's Chebyshev nodes -- theta_prime_Cheb, and, for a basis with several phases that
    supplies their closed-form rates, each of guard_rates sampled on the same nodes. A zero on a
    node counts, as does a non-finite sample. It cannot see an even number of sign changes between
    two adjacent nodes; the location before the loop is for that (_locate_stationary_points()).

    A phase of guard_rates whose index is in guard_sign_change_phases -- one the location left
    unlocated, its rate exactly zero at every finite scan sample (prompts/levin-phase-floor/
    07a-window-stop-and-phase-variation.md section 1.1; until 07a, prompt 07 section 10's Rule A,
    whose phases had a variation over the call's span below SIX_PI) -- is tested instead for a
    sign change: True only if its samples hold both strict signs. A zero or a non-finite sample of
    it does not count, so that a rate identically zero does not guard every Levin region of the
    call and drive it to depth_max, while a root such a rate holds between nodes of both signs is
    still caught. theta_prime_Cheb, and every phase not skipped, keep the test above unchanged.
    """
    if not _strictly_one_sign(theta_prime_Cheb):
        return True
    if guard_rates is not None:
        for index, rate in enumerate(guard_rates):
            sample = _sample_vectorized(
                rate, grid, vectorize_cache, ("guard_rate", index, id(rate))
            )
            if guard_sign_change_phases and index in guard_sign_change_phases:
                if _changes_sign(sample):
                    return True
            elif not _strictly_one_sign(sample):
                return True
    return False


def _oriented_cc(
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
    The nested Clenshaw-Curtis pair, _adaptive_levin_subregion_cc(), for the stationary-point
    guard's forced solves (prompt 07 section 1.2 item 4: a window piece, or a guarded region at
    depth_max), keeping x_span's orientation. _adaptive_levin_subregion_cc() weights by |b - a|
    (_cc_weights()), so on a reversed span (a > b) its value has the wrong sign; prompt 07 records
    that defect of the existing fallback and does not change it, and keeps it out of its own paths:
    on a reversed span the pair is solved on the ascending span, with theta' sampled afresh on that
    span's nodes, and its value and its signed node error E are negated. On an ascending span this
    is _adaptive_levin_subregion_cc() itself.
    """
    if vectorize_cache is None:
        vectorize_cache = {}
    a, b = x_span
    if a <= b:
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

    ascending = (b, a)
    grid, Dmat = chebyshev_matrices(ascending, chebyshev_order)
    _, theta_prime_ascending, _, _, phase_span_ascending = BasisData.build_Levin_data(
        grid,
        Dmat,
        notify_label=notify_label,
        id_label=id_label,
        need_AmatT=False,
        vectorize_cache=vectorize_cache,
    )
    data = _adaptive_levin_subregion_cc(
        ascending,
        f,
        BasisData,
        chebyshev_order,
        phase_span_ascending,
        theta_prime_ascending,
        notify_label,
        id_label,
        vectorize_cache=vectorize_cache,
    )
    data["value"] = -data["value"]
    if data.get("node_error_signed") is not None:
        data["node_error_signed"] = -data["node_error_signed"]
    return data


def _scalar_callable(func):
    """func, returning a Python float: for scipy's scalar root finders and minimisers."""

    def scalar(x):
        return float(func(x))

    return scalar


def _second_difference(func, x: float, step: float, lo: float, hi: float) -> float:
    """
    The central second difference (func(c + step) - 2 func(c) + func(c - step)) / step^2, its
    centre c = x moved inward if need be so that the stencil stays inside [lo, hi] (a rate is not
    promised to be defined outside x_span). Applied to a phase rate it estimates theta'''
    (prompt 07 section 1.1 step 3): truncation error step^2 |theta^(5)| / 12, rounding about
    4 eps max|theta'| / step^2.
    """
    c = min(max(x, lo + step), hi - step)
    return (func(c + step) - 2.0 * func(c) + func(c - step)) / (step * step)


def _first_difference(func, x: float, step: float, lo: float, hi: float) -> float:
    """The central first difference, its centre kept inside [lo, hi] as in _second_difference()."""
    c = min(max(x, lo + step), hi - step)
    return (func(c + step) - func(c - step)) / (2.0 * step)


def _locate_stationary_points(
    x_span: Tuple[float, float],
    phase_fns: list,
    rate_fns: list,
    vectorize_cache: dict,
    window_half_phase: float,
    coalescence_lambda: float,
    piece_phase: float,
) -> dict:
    """
    The stationary-point guard's location and windows (prompts/levin-phase-floor/
    07-stationary-point-guard.md sections 1.1 and 1.2), for every independent phase of a basis,
    from its closed-form rates. Run by _adaptive_levin() before its loop.

    Location (section 1.1), for each phase i, from its rate theta_i':
      1. Scan. theta_i' is sampled through _sample_vectorized() on _LEVIN_STATIONARY_SCAN_POINTS
         points, numpy.linspace over x_span with both endpoints (prompt 06a's grid). A non-finite
         sample is skipped, as 06a's scan skipped them.
         Nothing to locate (prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md
         section 1.1 item 1). If theta_i' is exactly 0.0 at every finite scan sample, steps 2-4 do
         not run for phase i: it gives no point, no window and no unlocated minimum, and is
         reported in "skipped_phases" with the reason. Without this a rate identically zero made
         every scan sample a root, and its call one window cut at every sample (R28's q - r calls
         with q = r, [07-location-on-an-identically-zero-rate]). Every other phase is located,
         whatever its variation: its roots, its exact zeros and its minima of |theta_i'|, steps
         2-4. This replaces prompt 07 section 10's Rule A, which skipped a phase whose variation
         over x_span, estimated on the scan by the trapezoidal rule, was below the C2 gate's
         SIX_PI, on the reasoning that the gate would send every region of it to the
         Clenshaw-Curtis rule. That held only where the gate agreed with the scan's estimate, and
         the gate's own estimate, from a region's nodes, could read above SIX_PI where the scan's
         did not, sending a region holding an unlocated root to the Levin solve, where only the
         nodal guard stood between it and a wrong value (07a section 0.2). A phase of one strict
         sign is still located: it can hold a nearly coalescing pair, a minimum of |theta_i'|
         whose Airy parameter is below coalescence_lambda.
      2. Roots. Each exact zero sample is a root as it stands. Each adjacent pair of finite samples
         of opposite sign is refined by scipy's brentq on the rate, with xtol = eps max(|x_a|,
         |x_b|) and rtol = 4 eps (scipy's least), eps = 2^-52: finer than the prompt's
         4 eps max(|x_a|, |x_b|). Kind "root".
      3. Minima. Each strict interior local minimum of |theta_i'| on the scan, at which and at
         whose two neighbours the rate has one strict sign (no sign change within a scan spacing
         h), is refined by bounded Brent minimisation of |theta_i'| between its neighbours
         (xatol = 4 eps max|x|, which scipy floors at about sqrt(eps)|x|; a refined value above the
         grid's is replaced by the grid point). theta_i''' there is the central second difference
         of the rate with step h/2 (_second_difference(); truncation error h^2 |theta^(5)| / 48,
         rounding about 16 eps max|theta'| / h^2). The minimum is located, kind "minimum", if its
         Airy parameter lambda = |theta_i'(x_m)|^(3/2) / sqrt(|theta_i'''| / 2) is below
         coalescence_lambda; otherwise it is recorded in "unlocated_minima".
      4. A point within h/2 of an end of x_span is moved to that end, kind "end" ("found_as" keeps
         what it was found as). Points of one phase within 4 eps |x| of one another are merged.
         Each point records its phase.

    Windows (section 1.2):
      1. Round each point x_* of phase i, the largest interval holding it on which
         |theta_i(x) - theta_i(x_*)| <= window_half_phase, clipped to x_span. Each edge is the
         first scan point outward from x_* at which that bound is reached, refined by brentq on
         |theta_i(x) - theta_i(x_*)| - window_half_phase between it and the point before it (x_*
         itself for the first). Edge precision: brentq returns x within about xtol + rtol |x| =
         5 eps |x| of a zero of the computed function, and the phase is evaluated to about
         ulp(|theta|), so at a returned edge | |theta_i - theta_i(x_*)| - window_half_phase | is at
         most about 5 eps |x| |theta_i'| + ulp(|theta_i(x)|) + ulp(|theta_i(x_*)|): of order 1e-8
         rad for a raw phase of 1e8, negligible against window_half_phase >= 2 pi.
      2. Windows that overlap or touch are merged.
      3. Each window is cut into pieces of at most piece_phase in every phase of the basis. A
         phase's variation is measured on its monotone segments -- the window split at that
         phase's located roots inside it, which are cuts -- exactly, as |Delta theta|, not
         estimated on the scan: a window can be narrower than a few scan spacings. Each segment
         is cut into ceil(|Delta theta| / piece_phase) pieces of equal phase, each cut found by
         brentq on theta - level; a level whose bracket does not change sign (a phase not monotone
         there, i.e. a point the location missed) is skipped. The cuts of every phase are united,
         and cuts within 4 eps |x| of one another merged.

    :param x_span: the call's x_span, in either order.
    :param phase_fns: the basis's independent_phases().
    :param rate_fns: the basis's independent_phase_rates(), the same length.
    :return: dict with "points" (one dict per located point, ascending in x: "x", "kind", "found_as",
        "phase", "theta", "theta_deriv", "lambda" and "theta_third_deriv" for a minimum,
        "own_window", "own_window_edges" with theta' and theta'^2 / |theta''| at each edge,
        "window_index", "window", "window_pieces"), "windows" (one dict per merged window,
        ascending: "lo", "hi", "cuts" -- the piece edges from lo to hi -- and "points", indices into
        "points"), "unlocated_minima", "skipped_phases" (one dict per phase with nothing to locate,
        in phase order: "phase", its index, and "reason", "zero_rate": the rate exactly 0.0 at
        every finite scan sample) and "scan_spacing".
    """
    from scipy.optimize import brentq, minimize_scalar

    eps = float(np.finfo(float).eps)
    a0, b0 = float(x_span[0]), float(x_span[1])
    lo, hi = min(a0, b0), max(a0, b0)
    n_scan = _LEVIN_STATIONARY_SCAN_POINTS
    scan = np.linspace(a0, b0, n_scan)
    h = (hi - lo) / (n_scan - 1)
    ascending = a0 <= b0
    xs = scan if ascending else scan[::-1]

    def root_in(func, xa, xb):
        return float(
            brentq(
                func,
                xa,
                xb,
                xtol=eps * max(abs(xa), abs(xb)),
                rtol=4.0 * eps,
                maxiter=200,
            )
        )

    points = []
    unlocated_minima = []
    skipped_phases = []
    for phase_index, rate in enumerate(rate_fns):
        sample = _sample_vectorized(
            rate, scan, vectorize_cache, ("location_rate", phase_index, id(rate))
        )
        d = sample if ascending else sample[::-1]
        r = _scalar_callable(rate)
        finite = np.isfinite(d)
        sgn = np.where(finite, np.sign(d), np.nan)

        # step 1 (prompt 07a section 1.1 item 1): a phase whose rate is exactly 0.0 at every finite
        # scan sample has nothing to locate; every other phase is located, whatever its variation
        if not np.any(finite & (d != 0.0)):
            skipped_phases.append({"phase": phase_index, "reason": "zero_rate"})
            continue

        # step 2: exact zeros, then adjacent finite pairs of opposite sign
        for j in np.nonzero(finite & (d == 0.0))[0]:
            points.append({"x": float(xs[j]), "kind": "root", "phase": phase_index})
        opposite = finite[:-1] & finite[1:] & (sgn[:-1] * sgn[1:] < 0.0)
        for j in np.nonzero(opposite)[0]:
            xa, xb = float(xs[j]), float(xs[j + 1])
            fa, fb = r(xa), r(xb)
            if fa == 0.0:
                x_root = xa
            elif fb == 0.0:
                x_root = xb
            elif np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                x_root = root_in(r, xa, xb)
            else:
                # the array sample and the scalar call disagree in sign at the bracket's ends (a
                # vectorised rate rounding differently from its scalar form): the zero is taken at
                # the end where the rate is smaller
                x_root = xa if abs(fa) <= abs(fb) else xb
            points.append({"x": x_root, "kind": "root", "phase": phase_index})

        # step 3: strict interior local minima of |theta'| with no sign change beside them
        absd = np.abs(d)
        is_min = (
            finite[:-2]
            & finite[1:-1]
            & finite[2:]
            & (sgn[1:-1] != 0.0)
            & (sgn[:-2] == sgn[1:-1])
            & (sgn[1:-1] == sgn[2:])
            & (absd[1:-1] < absd[:-2])
            & (absd[1:-1] < absd[2:])
        )
        for j in np.nonzero(is_min)[0] + 1:
            xl, xr = float(xs[j - 1]), float(xs[j + 1])
            result = minimize_scalar(
                lambda x: abs(r(x)),
                bounds=(xl, xr),
                method="bounded",
                options={"xatol": 4.0 * eps * max(abs(xl), abs(xr)), "maxiter": 500},
            )
            x_m = float(result.x)
            a_m = abs(r(x_m))
            if not (np.isfinite(a_m) and a_m <= absd[j]):
                x_m, a_m = float(xs[j]), float(absd[j])
            third = _second_difference(r, x_m, 0.5 * h, lo, hi)
            if np.isfinite(third) and third != 0.0:
                lam = a_m**1.5 / np.sqrt(0.5 * abs(third))
            else:
                lam = float("inf")
            record = {
                "x": x_m,
                "kind": "minimum",
                "phase": phase_index,
                "lambda": float(lam),
                "theta_third_deriv": float(third),
            }
            if lam < coalescence_lambda:
                points.append(record)
            else:
                unlocated_minima.append(record)

    # step 4: ends, then merging within a phase
    for p in points:
        p["found_as"] = p["kind"]
        if p["x"] - lo <= 0.5 * h:
            p["x"], p["kind"] = lo, "end"
        elif hi - p["x"] <= 0.5 * h:
            p["x"], p["kind"] = hi, "end"
    points.sort(key=lambda p: (p["phase"], p["x"]))
    merged = []
    for p in points:
        if merged:
            q = merged[-1]
            if q["phase"] == p["phase"] and abs(p["x"] - q["x"]) <= 4.0 * eps * max(
                abs(p["x"]), abs(q["x"])
            ):
                continue
        merged.append(p)
    points = sorted(merged, key=lambda p: (p["x"], p["phase"]))

    # section 1.2 item 1: each point's own window, from the phase sampled on the scan
    phase_on_scan = {}

    def window_edge(phase_index, x_p, t_p, direction):
        theta = phase_fns[phase_index]

        def g(x):
            value = float(theta(x))
            if not np.isfinite(value):
                raise ValueError(
                    "levin_quadrature: a phase returned a non-numeric value while locating "
                    "the edge of a stationary-point window"
                )
            return abs(value - t_p) - window_half_phase

        if phase_index not in phase_on_scan:
            sampled = _sample_vectorized(
                theta, scan, vectorize_cache, ("location_phase", phase_index, id(theta))
            )
            phase_on_scan[phase_index] = sampled if ascending else sampled[::-1]
        theta_scan = phase_on_scan[phase_index]

        if direction > 0:
            indices = np.nonzero(xs > x_p)[0]
        else:
            indices = np.nonzero(xs < x_p)[0][::-1]
        previous = x_p
        for j in indices:
            x_c = float(xs[j])
            if not (np.abs(theta_scan[j] - t_p) >= window_half_phase):
                previous = x_c
                continue
            g_c = g(x_c)
            if g_c < 0.0:
                previous = x_c
                continue
            if g_c == 0.0:
                return x_c
            if previous != x_p and not (g(previous) < 0.0):
                # the scan's sample and the scalar call disagree at the previous point (a
                # vectorised phase rounding differently from its scalar form)
                return previous
            u, v = (previous, x_c) if previous < x_c else (x_c, previous)
            return root_in(g, u, v)
        return hi if direction > 0 else lo

    for p in points:
        rate = _scalar_callable(rate_fns[p["phase"]])
        t_p = float(phase_fns[p["phase"]](p["x"]))
        p["theta"] = t_p
        p["theta_deriv"] = rate(p["x"])
        left = window_edge(p["phase"], p["x"], t_p, -1)
        right = window_edge(p["phase"], p["x"], t_p, +1)
        p["own_window"] = [left, right]
        edges_info = []
        for x_e in (left, right):
            first = rate(x_e)
            second = _first_difference(rate, x_e, 0.5 * h, lo, hi)
            edges_info.append(
                {
                    "x": x_e,
                    "theta_minus_theta_point": float(phase_fns[p["phase"]](x_e)) - t_p,
                    "theta_deriv": first,
                    "theta_second_deriv": float(second),
                    "theta_deriv_sq_over_second": (
                        first * first / abs(second) if second != 0.0 else float("inf")
                    ),
                }
            )
        p["own_window_edges"] = edges_info

    # section 1.2 item 2: merge windows that overlap or touch
    windows = []
    for index in sorted(range(len(points)), key=lambda k: points[k]["own_window"][0]):
        left, right = points[index]["own_window"]
        if windows and left <= windows[-1]["hi"]:
            windows[-1]["hi"] = max(windows[-1]["hi"], right)
            windows[-1]["points"].append(index)
        else:
            windows.append({"lo": left, "hi": right, "points": [index]})

    # section 1.2 item 3: pieces of at most piece_phase in every phase, each phase's located roots
    # inside the window among the cuts
    for window in windows:
        w_lo, w_hi = window["lo"], window["hi"]
        cuts = {w_lo, w_hi}
        for phase_index, theta in enumerate(phase_fns):
            roots = sorted(
                {
                    points[k]["x"]
                    for k in window["points"]
                    if points[k]["phase"] == phase_index
                    and points[k]["found_as"] == "root"
                    and w_lo < points[k]["x"] < w_hi
                }
            )
            cuts.update(roots)
            bounds = [w_lo] + roots + [w_hi]
            for j in range(len(bounds) - 1):
                s0, s1 = bounds[j], bounds[j + 1]
                t0, t1 = float(theta(s0)), float(theta(s1))
                span = abs(t1 - t0)
                if not (span > piece_phase):
                    continue
                direction = 1.0 if t1 >= t0 else -1.0
                count = int(ceil(span / piece_phase))
                start = s0
                for k in range(1, count):
                    level = t0 + direction * span * k / count

                    def g(x, level=level):
                        return float(theta(x)) - level

                    g_start, g_end = g(start), g(s1)
                    if g_start == 0.0:
                        x_cut = start
                    elif not (np.sign(g_start) != np.sign(g_end)):
                        continue
                    else:
                        x_cut = root_in(g, start, s1)
                    cuts.add(x_cut)
                    start = x_cut
        ordered = sorted(cuts)
        kept = [ordered[0]]
        for x in ordered[1:]:
            if x - kept[-1] > 4.0 * eps * max(abs(x), abs(kept[-1])):
                kept.append(x)
        if kept[-1] != w_hi:
            if len(kept) > 1:
                kept[-1] = w_hi
            else:
                kept.append(w_hi)
        window["cuts"] = kept

    for window_index, window in enumerate(windows):
        for k in window["points"]:
            points[k]["window_index"] = window_index
            points[k]["window"] = [window["lo"], window["hi"]]
            points[k]["window_pieces"] = len(window["cuts"]) - 1

    return {
        "points": points,
        "windows": windows,
        "unlocated_minima": unlocated_minima,
        "skipped_phases": skipped_phases,
        "scan_spacing": h,
    }


def _sum_leaves_ascending(leaves_ascending) -> Tuple[float, float, float]:
    """
    The call's value, the sum of its leaves' errors e'_i (used_interval.budget_err), and the
    signed sum of its fallback leaves' node errors E, over the leaves in ascending x (the order
    the returned "regions" list is in): the first two with plain floating-point addition, the
    third correctly rounded (math.fsum, the value the loop's exact running sum holds). The call's
    error is the second plus the absolute value of the third (prompts/levin-phase-floor/
    06g-leaf-values-and-node-sum.md, section 0.2 (c)). _adaptive_levin() computes its returned
    "value", "abserr" and "converged" from these, and confirms its running stop test against them
    before it stops, so a call that stops on the budget reports "converged" True by the same
    arithmetic it returns.
    """
    val = 0.0
    abserr_total = 0.0
    node_errors = []
    for leaf in leaves_ascending:
        val = val + leaf.value
        contribution = leaf.region.budget_err
        if contribution is not None:
            abserr_total = abserr_total + contribution
        if leaf.region.node_error_signed is not None:
            node_errors.append(leaf.region.node_error_signed)
    return val, abserr_total, math.fsum(node_errors)


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
    _test_only_bypass_location: bool = False,
):
    """
    The adaptive driver, under one error budget for the whole call
    (prompts/levin-phase-floor/06c-global-error-control.md). Global adaptivity, as QUADPACK's QAG:
    the driver keeps every *leaf* of the current partition of x_span, each with its value v_i and
    its error e'_i (used_interval.budget_err: max(residual, round-off floor) + truncation, with a
    Clenshaw-Curtis leaf's node error |E_i| taken out of its floor), and decides only against the
    whole call. The call's error is

        E_call = sum_i e'_i + |sum_i E_i|,

    the node errors carried signed over the fallback leaves (prompts/levin-phase-floor/
    06g-leaf-values-and-node-sum.md, section 0.2 (c): E's sign varies from leaf to leaf, so
    sum |E_i| overstates the call's node error, by 54x on 06e's Z2 call, and since E is set by
    ulp(x), not by a leaf's width, sum |E_i| does not fall systematically when a leaf is split,
    06f). With V = sum_i v_i and the request max(atol, rtol*|V|):

      1. If E_call meets the request, stop: converged.
      2. If sum_i e'_i meets it and E_call does not, stop unconverged, flagged node_limited:
         |sum E| alone holds the call above its request, and bisection does not lower it
         systematically, so refining further would chase it.
      3. Otherwise choose the *refinable* leaf with the largest e'_i (ties: the smaller start),
         and replace it by its two halves.
      4. If no leaf is refinable, stop: unconverged.

    Stops 1 and 2 also need every leaf checked or floor-limited and no refinable leaf untrusted
    (the trust check, below).

    A leaf is refinable unless one of these holds:
      * it is at depth_max;
      * it is *floor-limited*: its round-off floor is finite, its residual is at most that floor
        (the whole floor, its own |E| included), and 06b's premise holds, as 06g states it
        (_roundoff_exit_premise_holds()): its halves' floors less their declared endpoint terms
        sum to at least rho = 1/2 of its own less its own, and the call's error, recomputed with
        the leaf replaced by its halves' floors, would still be above the request -- so bisecting
        it would not lower its error materially, and could not meet the budget. Checked only on
        the leaf chosen; flagged used_interval.phase_limited.
    A leaf found not refinable is never chosen again. No rule stops a leaf because bisecting it has
    stopped lowering its error: a leaf whose error does not fall with h is chased to depth_max if it
    holds the total back, at a cost in work but not in honesty. (A stall rule was tried and
    withdrawn: it judged splitting by an error estimate that cannot yet be trusted on
    pre-asymptotic leaves and on Levin leaves across a stationary point, and stopped refinement
    where it works -- prompts/levin-phase-floor/06c-global-error-control.md, section 8.)

    The trust check one level down (06c section 8, the third amendment; the trust test and the value
    returned since 06g). The budget trusts each leaf's error estimate, and an estimate taken before
    the leaf is resolved can understate its error: the step-(4) residual |V_P - (V_L + V_R)| is
    V_P's error only if V_L + V_R is much better than V_P. So, for each leaf, from its halves and
    its grandchildren (each valued by its own branch's rule), delta1 = |V_P - (V_L + V_R)|,
    delta2 = |(V_L + V_R) - Q|, Q = (V_LL + V_LR) + (V_RL + V_RR), and r = delta2 / delta1. If the
    error falls from this level on at least as fast as observed, the leaf's error is at most
    delta1 / (1 - r), so:
      * a leaf is trusted if r < _LEVIN_TRUST_MAX_RATIO = 1/4, or if r < 1 and delta1 is at most
        the noise yardstick nu_B (_trust_noise_yardstick(); since prompts/levin-phase-floor/
        06m-signed-yardstick-in-the-driver.md, in place of the leaf's own floor alone): its own
        round-off floor, plus its Levin halves' floors, plus, on a Clenshaw-Curtis leaf, its
        Clenshaw-Curtis halves' floors less their node terms and the absolute value of those
        halves' node errors summed signed (then delta1 and delta2 are rounding noise, which nu_B
        bounds). A trusted leaf declares max(residual, delta1) / (1 - r) as its resolution error
        and returns Q, the finest value computed, in place of V_P; the declaration bounds Q's
        error under the same assumption that makes it bound V_P's, and is not tightened;
      * any other leaf is untrusted: refinable whatever the budget says, and chosen before every
        trusted leaf; the loop does not stop while a refinable leaf is untrusted. One at
        depth_max declares max(residual, delta1) / (1 - r) if r < 1 and max(residual,
        delta1 + delta2) if not, keeps V_P, is flagged (used_interval.trusted False), and the call
        is not converged while one is returned;
      * a floor-limited leaf is exempt: its differences are round-off noise, which its floor
        bounds. If it was checked first, its declared error and its value revert to its own.
    The check is taken lazily: when the budget passes, every leaf not yet checked is checked, and
    a widened error or an untrusted leaf reopens the loop. A call that stops on the budget returns
    every leaf checked or floor-limited. The grandchildren are carried forward as the children's
    halves if a leaf is later bisected. The running value sum follows each leaf's returned value
    exactly. It does not reach an error common to a leaf and all its descendants
    ([06b-moved-calls-declare-less-than-their-error]).

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
    largest e'_i, and is bisected where the premise fails; an infinite floor (a stationary point at
    a node) is always chosen first, and is bisected until it is finite or reaches depth_max.

    Termination: every bisection deepens a leaf, and a leaf at depth_max is never chosen.

    The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md). Neither
    the step-(4) residual nor the trust check can see an interior stationary point of the phase:
    near one the Levin antiderivative p ~ f / (i theta') is singular, and a leaf's estimate, its
    halves' and its grandchildren's all lack the stationary-phase term, so they agree with one
    another and not with the integral (docs/levin-phase-floor/06/p1/report.md: a residual 1e5
    times smaller than the true error). The eq. (151) floor is infinite only when a node falls
    close enough to the point, and the C2 gate acts only once bisection has reached a region of
    small variation, which nothing forced. So, generically, for a basis of any number of
    components:
      1. Location. If the basis supplies the closed-form rates of its independent phases
         (independent_phase_rates() not None; for adaptive_levin_sincos(), the caller's
         theta["theta_deriv"]), _locate_stationary_points() finds, before the loop, every sign
         change of every rate on _LEVIN_STATIONARY_SCAN_POINTS points, and every interior minimum
         of |rate| whose Airy parameter is below _LEVIN_COALESCENCE_LAMBDA (a nearly coalescing
         pair). The location does not depend on any estimate of the phase's variation (prompts/
         levin-phase-floor/07a-window-stop-and-phase-variation.md section 1.1): a phase is not
         located only when its rate is exactly zero at every finite scan sample, where there is
         nothing to locate -- without that, a rate identically zero made every scan sample a
         root, and its call one window cut at every sample (R28's q - r calls with q = r). Every
         other phase is located, so every region the Levin solve receives has its stationary
         points located and cut out, whatever its variation, save a pair the scan missed between
         two of its samples, which the guard (item 4) is for. This replaces prompt 07 section
         10's Rule A, which skipped a phase whose variation over x_span, estimated on the scan,
         was below SIX_PI, relying on the C2 gate to send its regions to the fallback; the gate,
         deciding on its own estimate from a region's nodes, could disagree, and a region holding
         an unlocated root then went to the Levin solve with only the guard to catch it.
      2. Windows. Round each point, the interval on which its phase stays within
         _LEVIN_WINDOW_HALF_PHASE of its value at the point, cut into pieces of at most
         _LEVIN_WINDOW_PIECE_PHASE in every phase. Each piece is a leaf at depth 0, solved -- with
         its halves and grandchildren -- by the nested Clenshaw-Curtis pair whatever its span
         (force_direct; used_interval.window). It is a Clenshaw-Curtis leaf in every other
         respect: its errors, its trust check and its node error are the pair's. The floor-limited
         exit never fires on it (prompt 07 section 0.2 (ii)), so it stays refinable until
         depth_max or until the window is stopped by the next rule.
      2a. The stopping rule for window leaves (prompt 07 section 1.3; its count since prompt 07a
         section 1.2 QUADPACK dqage's): iroff1, counted per merged window. A bisection of one of
         its leaves counts (_window_bisection_counts()) if neither half's error estimate is
         unresolved in dqk21's sense (_cc_leaf_unresolved(); dqage's skip), the halves change the
         value by at most min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL, rtol) relatively -- a test no
         looser than the request, where dqage's fixed 1e-5 would count, at rtol 1e-8, a bisection
         that moves the value by 1000 times the request -- and their own solves' summed error
         (used_interval.own_err, before the trust check widens a declaration, as dqage compares
         its raw estimates) is at least _LEVIN_WINDOW_IROFF1_ERROR_RATIO of the parent's. Until 07a
         every bisection was counted, at a fixed 1e-5, on the declared e'. At
         _LEVIN_WINDOW_IROFF1_LIMIT counted
         bisections the window is round-off-limited: every leaf of it, made then or later, is not
         refinable and is returned floor-limited, exempt from the trust check, declaring what it
         carries (an untrusted leaf declares as an untrusted leaf at depth_max does), and the
         call's "converged" follows its budget: dqage's outcome at the stop, ier = 2, is not
         adopted (07a section 9.3; see the comment above _LEVIN_WINDOW_IROFF1_VALUE_RTOL). It
         ends the two ways 07's section 2 measured window leaves chased to depth_max: on a
         Clenshaw-Curtis floor flat in h above the request, and untrusted at the integrand's own
         noise far inside it. Guarded leaves are not counted.
      3. The rest. The initial partition is the window pieces and the ranges of x_span between
         windows, each a leaf at depth 0, all in the one budget, heap, trust check and node sum.
         With nothing located it is the single leaf x_span, and the call takes exactly the path it
         takes without location.
      4. The guard. A Levin leaf whose phase rate is not of one strict sign on its own Chebyshev
         nodes (the location missed a point, or the basis has no rates; see
         _adaptive_levin_subregion_impl()) is guarded (used_interval.guarded): never returned as a
         Levin leaf. Below depth_max its error is read as +inf (_levin_leaf.error), as for a leaf
         whose G0 is at noise, so it is chosen first, the budget cannot be met while it is in the
         partition, it is never trusted and never floor-limited, and it is bisected; below 6 pi
         the C2 gate sends its halves to the fallback. At depth_max it is re-solved by the nested
         pair whatever its span, its errors are that solve's, it is flagged guard_at_depth_max and
         exempt from the trust check, and the call is not converged while one is returned. For a
         basis of several phases, a phase the location skipped (its rate exactly zero at every
         finite scan sample; until 07a, a phase Rule A skipped) is tested for a sign change
         (samples of both strict signs on the nodes) rather than for one strict sign, so that a
         rate identically zero does not guard every Levin leaf; theta_prime_Cheb and every
         located phase keep the strict test. Since 07a the guard is a backstop: every phase with
         something to locate is located, and it fires only where the scan missed a pair of
         roots between two samples, or the basis supplies no rates.
    An unguarded leaf outside every window behaves exactly as without the guard. Without rates
    nothing is located, and the guard alone protects the call: it misses a pair of sign changes
    between two adjacent nodes of a leaf, and a minimum of |theta'| that does not reach zero.

    The result adds "stationary_points" (one dict per located point: its x, phase and kind --
    "root", "minimum" or "end" -- theta' there, lambda for a minimum, its window's edges and number
    of pieces, and theta'^2 / |theta''| at its own window's edges), "stationary_windows" (each
    merged window's edges and piece cuts, its iroff1 count, and whether the rule stopped it, with
    what was true of its leaves then), "unlocated_minima", "num_window_pieces" (window leaves
    returned), "num_guarded_regions" (leaves found guarded when made, returned or bisected) and
    "num_guard_at_depth_max"; and "location_skipped_phases" (one dict per phase the location did
    not locate, its rate exactly zero at every finite scan sample: its index "phase" and "reason",
    "zero_rate"; until 07a, Rule A's phases with their estimated "variation"). Each window's
    "iroff1_count" is the count of item 2a.

    :param _test_only_bypass_location: TEST-ONLY, off by default: skips step 1, and so 2 and 3, so
        that a test can show a call in which nothing is located is bit-identical to one in which the
        location never ran (prompt 07 section 3 test 8). The nodal guard still runs. Read only at
        the location below; adaptive_levin_sincos() does not expose it. Never set it in production
        code.
    """
    # _test_only_bypass_roundoff_premise: TEST-ONLY. True skips 06b's premise
    # (_roundoff_exit_premise_holds()) in the floor-limited test, so that a leaf chosen while the
    # budget is unmet is found floor-limited on its floor and residual alone, without checking
    # that bisection would not lower its floor -- the round-off-limited exit's own test as it was
    # before prompts/levin-phase-floor/06b-roundoff-exit-premise.md, now inside the global budget.
    # It skips the premise whole: its halving test and, since 06g, its budget clause.
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

    # The stationary-point guard (prompt 07; see the docstring): the basis's independent phases and
    # their closed-form rates. A basis without the two methods, or whose rates are None, cannot
    # supply rates: nothing is located, and only the nodal guard protects the call.
    phases_method = getattr(BasisData, "independent_phases", None)
    rates_method = getattr(BasisData, "independent_phase_rates", None)
    phase_fns = phases_method() if callable(phases_method) else None
    rate_fns = rates_method() if callable(rates_method) else None
    if phase_fns is None or rate_fns is None or len(rate_fns) != len(phase_fns):
        rate_fns = None

    # For a basis with several phases the nodal guard samples each phase's rate on a Levin leaf's
    # nodes; for one phase theta_prime_Cheb is its rate already (_nodal_guard()).
    guard_rates = rate_fns if rate_fns is not None and len(rate_fns) > 1 else None

    location = None
    if (
        rate_fns is not None
        and not _test_only_bypass_location
        and x_span[1] != x_span[0]
    ):
        location = _locate_stationary_points(
            x_span,
            phase_fns,
            rate_fns,
            vectorize_cache,
            window_half_phase=_LEVIN_WINDOW_HALF_PHASE,
            coalescence_lambda=_LEVIN_COALESCENCE_LAMBDA,
            piece_phase=_LEVIN_WINDOW_PIECE_PHASE,
        )

    # For a basis of several phases, the nodal guard tests each phase the location skipped -- since
    # prompt 07a section 1.1 only a rate exactly zero at every finite scan sample; until then prompt
    # 07 section 10's Rule A -- for a sign change on a leaf's nodes, not for one strict sign
    guard_sign_change_phases = None
    if guard_rates is not None and location is not None:
        guard_sign_change_phases = frozenset(
            skipped["phase"] for skipped in location["skipped_phases"]
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

    # The stationary-point guard's counts (prompt 07 section 1.5): leaves found guarded when made,
    # returned or bisected; and guarded leaves re-solved by the nested pair at depth_max.
    num_guarded_regions = 0
    num_guard_at_depth_max = 0

    def _solve(span, description, force_direct=False):
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
                force_direct=force_direct,
                guard_rates=guard_rates,
                guard_sign_change_phases=guard_sign_change_phases,
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
        window=False,
        window_index=None,
    ) -> _levin_leaf:
        """
        Make a leaf of [a, b] at this depth, and compute its error. data is its own solve when it
        is carried forward from its parent (one of the parent's halves), else None (the root),
        when it is solved here. A Levin leaf solves its two halves for its step-(4) residual (Chen
        et al. (173)); a fallback leaf's error is its own nested-pair estimate. halves, when
        given, are the leaf's own halves' solves, carried forward from its parent's trust check
        (the parent's grandchildren), and are used rather than solved again.

        window (prompt 07): the leaf is a window piece, or a descendant of one, solved by the
        nested pair whatever its span. A Levin leaf whose solve is guarded is made guarded below
        depth_max, and at depth_max is re-solved by the nested pair and flagged
        guard_at_depth_max (see _adaptive_levin()'s docstring).
        """
        nonlocal seq_counter, max_depth, num_history_messages
        nonlocal chebyshev_min_order, num_SVD_errors, num_order_changes, num_direct_solves
        nonlocal num_guarded_regions, num_guard_at_depth_max

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
            data = _solve(
                (a, b),
                "stationary-point window piece" if window else "Levin subregion",
                force_direct=window,
            )

        # The stationary-point guard (prompt 07 section 1.4). A guarded Levin solve is never
        # returned as a Levin leaf: below depth_max the leaf is made guarded, with an infinite
        # error (_levin_leaf.error), and is bisected; at depth_max it is re-solved by the nested
        # pair whatever its span, and that solve is the leaf's.
        guarded = False
        guard_at_depth_max = False
        if not data.get("is_direct", False) and data.get("guarded", False):
            num_guarded_regions += 1
            guarded = True
            if depth >= depth_max:
                num_guard_at_depth_max += 1
                guard_at_depth_max = True
                data = _solve((a, b), "guarded leaf at depth_max", force_direct=True)
                halves = None

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
                value=estimate,
                phase_err_declared=_solve_floor_declared(data),
                phase_err_roundoff=_solve_floor_roundoff(data),
                node_error_signed=data.get("node_error_signed"),
                phase_err_less_node=data.get("phase_err_less_node"),
                window=window,
                guarded=guarded,
                guard_at_depth_max=guard_at_depth_max,
                resasc=data.get("resasc"),
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
                value=estimate,
                phase_err_declared=_solve_floor_declared(data),
                phase_err_roundoff=_solve_floor_roundoff(data),
                guarded=guarded,
            )

        seq_counter += 1
        leaf = _levin_leaf(
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
            forced_direct=window or guard_at_depth_max,
            guarded=guarded and not guard_at_depth_max,
            window_index=window_index if window else None,
        )
        if guard_at_depth_max:
            # exempt from the trust check: never trusted, its errors the nested pair's own, and
            # the call not converged while it is returned
            leaf.checked = True
        return leaf

    # The current partition: every leaf, by its sequence number. The candidates for bisection are
    # the refinable leaves, on a heap ordered first by trust -- a leaf the trust check found
    # untrusted before every other -- then by error e'_i (_levin_leaf.error), largest first, ties
    # broken by the smaller start and then by the sequence number, so the order is total and
    # deterministic. A leaf at depth_max, or found floor-limited, is never a candidate. When a
    # leaf's error or trust changes it is pushed again under a new version, and an entry whose
    # version is stale, or whose leaf has been bisected, is skipped when popped.
    leaves = {}
    candidates = []
    untrusted_refinable = set()

    # Running sums over the leaves, held exactly (_ExactSum): the finite errors e'_i, the count of
    # errors that are not finite (+inf, or NaN read as +inf -- see _error_for_ordering()), the
    # values, and (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md) the fallback
    # leaves' signed node errors E, a non-finite E counted with the non-finite errors. Each is
    # updated when a leaf is added, removed, or changes its error or its value.
    error_sum = _ExactSum()
    value_sum = _ExactSum()
    node_sum = _ExactSum()
    num_nonfinite_errors = 0
    num_direct_leaves = 0

    def _add_node_error(leaf: _levin_leaf, sign: float) -> None:
        nonlocal num_nonfinite_errors
        node = leaf.region.node_error_signed
        if node is None:
            return
        if np.isfinite(node):
            node_sum.add(sign * node)
        else:
            num_nonfinite_errors += 1 if sign > 0 else -1

    def _add_leaf(leaf: _levin_leaf) -> None:
        nonlocal num_nonfinite_errors, num_direct_leaves
        leaves[leaf.seq] = leaf
        err = _error_for_ordering(leaf.error)
        if np.isfinite(err):
            error_sum.add(err)
        else:
            num_nonfinite_errors += 1
        _add_node_error(leaf, 1.0)
        value_sum.add(leaf.value)
        if leaf.is_direct:
            num_direct_leaves += 1
        if leaf.window_index is not None and leaf.window_index in windows_stopped:
            _floor_limit_stopped_window_leaf(leaf)
            return
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
        _add_node_error(leaf, -1.0)
        value_sum.add(-leaf.value)
        if leaf.is_direct:
            num_direct_leaves -= 1

    def _set_value(leaf: _levin_leaf, value: float, kind: str) -> None:
        # A leaf's returned value changes (06g, section 0.2 (a)): to its quarters sum when the trust
        # check trusts it, back to its own when the floor-limited test exempts it. The running
        # value sum follows exactly.
        value_sum.add(-leaf.value)
        value_sum.add(value)
        leaf.value = value
        leaf.region._value = value
        leaf.region._value_kind = kind

    def _running_sums() -> Tuple[float, float, float]:
        """
        The running value, the running sum of e'_i, and |sum E|; the call's error is the sum of
        the last two (+inf while any error or E is not finite).
        """
        if num_nonfinite_errors > 0:
            return value_sum.total(), float("inf"), float("inf")
        return value_sum.total(), error_sum.total(), abs(node_sum.total())

    def _leaves_ascending() -> List[_levin_leaf]:
        return sorted(leaves.values(), key=lambda leaf: leaf.start)

    def _solve_halves(leaf: _levin_leaf) -> None:
        # A fallback leaf has no halves of its own until it is chosen: they are solved here, on
        # demand, each a full subregion solve on its own samples -- the same call that would
        # produce it as a leaf, so its "phase_err" is the floor it would carry -- and they are
        # kept, to become the children's own solves if the leaf is bisected.
        # A window leaf's halves are solved by the nested pair whatever their span (prompt 07).
        c = leaf.break_point
        leaf.dataL = _solve(
            (leaf.start, c),
            "left half of fallback region",
            force_direct=leaf.forced_direct,
        )
        leaf.dataR = _solve(
            (c, leaf.end),
            "right half of fallback region",
            force_direct=leaf.forced_direct,
        )

    def _check_leaf(leaf: _levin_leaf) -> None:
        """
        The trust check one level down (prompts/levin-phase-floor/06c-global-error-control.md,
        section 8, the third amendment; its trust test and the value returned since
        prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md). With V each solve's own value
        (the region's own solve on the Levin branch, CC_{2N-1} on the fallback):

            delta1 = |V_P - (V_L + V_R)|,
            delta2 = |(V_L + V_R) - Q|,     Q = (V_LL + V_LR) + (V_RL + V_RR),
            r = delta2 / delta1   (0 if both are 0; +inf if only delta1 is 0, or either is not
                                   finite).

        If the error falls from this level on at least as fast as observed, the leaf's error is at
        most delta1 (1 + r + r^2 + ...) = delta1 / (1 - r). The leaf is trusted if r <
        _LEVIN_TRUST_MAX_RATIO (1/4; that constant's comment gives the basis), or if r < 1 and
        delta1 is at most the noise yardstick nu_B of _trust_noise_yardstick(), formed from the
        leaf's own floor phase_err, whether it is a Clenshaw-Curtis solve, and its halves' solves
        (since prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md; from 06g to 06m,
        phase_err alone): there delta1 and delta2 are differences of rounding noise, r means
        nothing, and the rounding of the solves they are formed from bounds them. A trusted leaf
        declares max(abserr, delta1) / (1 - r) as its resolution error and returns Q, the finest
        value computed, in place of V_P: under the assumption that makes delta1 / (1 - r) a bound
        on V_P's error it bounds Q's too, and the declaration is not tightened (06e's leaf
        breakdown measured Q's error at about 3 delta2 on Z3's sample 8, so a declaration built
        from delta2 would understate it). The running value sum is moved by Q - V_P exactly.

        Otherwise the leaf is untrusted, keeps V_P, and is refinable whatever the budget says,
        chosen before every trusted leaf; at depth_max, where it cannot be refined, it declares
        max(abserr, delta1) / (1 - r) if r < 1 and max(abserr, delta1 + delta2) if not, and the
        call is not converged. The grandchildren are kept, to become its children's halves if it
        is bisected.
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
        forced = leaf.forced_direct
        LL = _solve((a, cL), "trust check: left half of left half", force_direct=forced)
        LR = _solve(
            (cL, c), "trust check: right half of left half", force_direct=forced
        )
        RL = _solve(
            (c, cR), "trust check: left half of right half", force_direct=forced
        )
        RR = _solve(
            (cR, b), "trust check: right half of right half", force_direct=forced
        )
        leaf.grandchildren = ((LL, LR), (RL, RR))

        region = leaf.region
        own_value = region._value_own
        halves_sum = dataL["value"] + dataR["value"]
        quarters_sum = (LL["value"] + LR["value"]) + (RL["value"] + RR["value"])
        delta1 = float(np.fabs(own_value - halves_sum))
        delta2 = float(np.fabs(halves_sum - quarters_sum))
        if not (np.isfinite(delta1) and np.isfinite(delta2)):
            ratio = float("inf")
        elif delta1 == 0.0:
            ratio = 0.0 if delta2 == 0.0 else float("inf")
        else:
            ratio = delta2 / delta1

        residual = region.abserr
        noise_yardstick = _trust_noise_yardstick(
            region.phase_err, leaf.is_direct, dataL, dataR
        )
        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (
            ratio < 1.0 and delta1 <= noise_yardstick
        )
        _remove_error(leaf)
        region._delta1 = delta1
        region._delta2 = delta2
        region._convergence_ratio = ratio
        if trusted:
            region._trusted = True
            region._declared_resolution = max(residual, delta1) / (1.0 - ratio)
        else:
            region._trusted = False
            num_untrusted_found += 1
            if leaf.depth >= depth_max:
                region._declared_resolution = (
                    max(residual, delta1) / (1.0 - ratio)
                    if ratio < 1.0
                    else max(residual, delta1 + delta2)
                )
            else:
                region._declared_resolution = None
        _restore_error(leaf)
        if trusted:
            _set_value(leaf, quarters_sum, "quarters")
        leaf.checked = True
        num_checks += 1
        _push_candidate(leaf)

    def _exempt_floor_limited(leaf: _levin_leaf) -> None:
        # A floor-limited leaf is exempt from the trust check: its differences are round-off
        # noise, and the floor bounds them. If the check was taken on it before it was found
        # floor-limited, its declared resolution error reverts to its own estimate, and its value
        # to its own, V_P, if the check had trusted it (06g); the differences stay recorded.
        region = leaf.region
        if region._trusted is not None or region._declared_resolution is not None:
            _remove_error(leaf)
            region._trusted = None
            region._declared_resolution = None
            _restore_error(leaf)
        if region._value_kind != "own":
            _set_value(leaf, region._value_own, "own")
        leaf.grandchildren = None

    # Section 1.3's rule: each merged window's count of iroff1 bisections, and the windows stopped,
    # with what was true of their leaves when they stopped (reported in "stationary_windows").
    window_counts = {}
    windows_stopped = {}

    def _floor_limit_stopped_window_leaf(leaf: _levin_leaf) -> None:
        """
        A leaf of a window section 1.3's rule has stopped: not refinable, returned floor-limited,
        exempt from the trust check as a floor-limited leaf is. It declares what it carries: its
        e', except that a leaf the trust check found untrusted declares as an untrusted leaf at
        depth_max does, max(residual, delta1) / (1 - r) if r < 1 and max(residual, delta1 +
        delta2) if not. A trusted leaf keeps its widened declaration and returns its own value, V_P,
        as a floor-limited leaf does.
        """
        region = leaf.region
        _remove_error(leaf)
        if region._trusted is False:
            ratio = region._convergence_ratio
            residual = region.abserr
            region._declared_resolution = (
                max(residual, region._delta1) / (1.0 - ratio)
                if ratio < 1.0
                else max(residual, region._delta1 + region._delta2)
            )
        region._trusted = None
        region._phase_limited = True
        _restore_error(leaf)
        if region._value_kind != "own":
            _set_value(leaf, region._value_own, "own")
        leaf.dataL = None
        leaf.dataR = None
        leaf.grandchildren = None
        _push_candidate(leaf)

    def _stop_window(window_index: int) -> None:
        members = [
            leaf for leaf in leaves.values() if leaf.window_index == window_index
        ]
        windows_stopped[window_index] = {
            "stopped_after_split": num_splits,
            "leaves_at_stop": len(members),
            "leaves_falling_at_stop": sum(
                1 for leaf in members if leaf.fell_at_last_bisection
            ),
            "max_depth_at_stop": max(leaf.depth for leaf in members),
        }
        for member in members:
            _floor_limit_stopped_window_leaf(member)

    def _split_could_meet_budget(leaf: _levin_leaf) -> bool:
        """
        The round-off exit's budget clause (06g, section 0.2 (d) item 2): True if the call's error
        is above its request now, and would meet it with this leaf's e' replaced by its halves'
        floors as each would enter the call's error (_solve_floor_in_budget(): less |E| on a
        fallback half) and sum E updated by the halves' E in place of the leaf's. The floors are
        lower bounds on the halves' errors, so a False here means the split cannot meet the
        request. The recomputed sums are exact (_ExactSum.total_with()); a non-finite error
        elsewhere in the call, or a non-finite half floor, makes the recomputed error +inf.

        When the call meets its request already -- the leaf was chosen because the trust check
        found it untrusted -- the split is not what the call needs for its budget, and this is
        False: the premise is then decided by the halving test alone, as before 06g. (As written,
        the clause would bisect every such leaf, a noise-level one included, on towards depth_max;
        see 06g's log.)
        """
        running_value, running_eprime, running_node = _running_sums()
        request = max(atol, rtol * np.fabs(running_value))
        if running_eprime + running_node <= request:
            return False

        region = leaf.region
        e_leaf = _error_for_ordering(leaf.error)
        E_leaf = region.node_error_signed
        own_nonfinite = (0 if np.isfinite(e_leaf) else 1) + (
            1 if E_leaf is not None and not np.isfinite(E_leaf) else 0
        )
        if num_nonfinite_errors - own_nonfinite > 0:
            return False

        error_terms = []
        node_terms = []
        if np.isfinite(e_leaf):
            error_terms.append(-e_leaf)
        if E_leaf is not None and np.isfinite(E_leaf):
            node_terms.append(-E_leaf)
        for half in (leaf.dataL, leaf.dataR):
            floor = _solve_floor_in_budget(half)
            node = half.get("node_error_signed")
            if not np.isfinite(floor) or (node is not None and not np.isfinite(node)):
                return False
            error_terms.append(floor)
            if node is not None:
                node_terms.append(node)

        recomputed = error_sum.total_with(*error_terms) + abs(
            node_sum.total_with(*node_terms)
        )
        return recomputed <= request

    # The initial partition. With nothing located, the root leaf x_span, as before prompt 07. With
    # windows (prompt 07 section 1.2 item 5): one leaf for each range of x_span between windows,
    # an ordinary leaf, and one for each window piece, a window leaf, all at depth 0 and oriented as
    # x_span.
    if location is None or not location["windows"]:
        _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))
    else:
        lo_span, hi_span = min(x_span[0], x_span[1]), max(x_span[0], x_span[1])
        seed_spans = []
        cursor = lo_span
        for window_index, stationary_window in enumerate(location["windows"]):
            if stationary_window["lo"] > cursor:
                seed_spans.append((cursor, stationary_window["lo"], None))
            cuts = stationary_window["cuts"]
            for piece_lo, piece_hi in zip(cuts[:-1], cuts[1:]):
                if piece_hi > piece_lo:
                    seed_spans.append((piece_lo, piece_hi, window_index))
            cursor = max(cursor, stationary_window["hi"])
        if hi_span > cursor:
            seed_spans.append((cursor, hi_span, None))
        reversed_span = x_span[0] > x_span[1]
        for seed_lo, seed_hi, seed_window_index in seed_spans:
            seed_a, seed_b = (seed_hi, seed_lo) if reversed_span else (seed_lo, seed_hi)
            _add_leaf(
                _make_leaf(
                    seed_a,
                    seed_b,
                    0,
                    None,
                    {},
                    {},
                    {},
                    window=seed_window_index is not None,
                    window_index=seed_window_index,
                )
            )

    running_stop = False
    node_limited = False

    while True:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            current_val, _, _ = _running_sums()
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

        # Step 1: the budget. The call's error is sum e'_i + |sum E| (06g, section 0.2 (c)). The
        # running sums are exact, so this test is the correctly rounded one; before stopping on
        # it, it is confirmed on the sums the call returns (summed in ascending x,
        # _sum_leaves_ascending()), so that a call which stops here reports converged by the same
        # arithmetic. If they disagree -- a total at the boundary, within rounding -- refinement
        # continues.
        #
        # Then the trust check one level down, taken lazily: when the budget passes, every leaf not
        # yet checked (and not floor-limited, which is exempt) is checked, in ascending x. A
        # widened error, or an untrusted leaf, reopens the loop, which continues and checks again
        # at the next pass. The loop stops on the budget only with every leaf checked or
        # floor-limited, and no refinable leaf untrusted.
        #
        # The node-limited stop (06g). The test above is taken on sum e'_i alone, which the stop
        # needs either way. When sum e'_i meets the request and sum e'_i + |sum E| does not, with
        # every other condition of the stop holding, the call stops unconverged, flagged
        # node_limited: |sum E| alone holds it above its request, and bisection does not lower
        # sum E systematically (06f), so refining further would chase it.
        running_value, running_eprime, running_node = _running_sums()
        running_request = max(atol, rtol * np.fabs(running_value))
        running_stop = running_eprime + running_node <= running_request
        if running_eprime <= running_request:
            check_value, check_eprime, check_node = _sum_leaves_ascending(
                _leaves_ascending()
            )
            check_request = max(atol, rtol * np.fabs(check_value))
            if check_eprime <= check_request:
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
                    if not (check_eprime + np.fabs(check_node) <= check_request):
                        node_limited = True
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
        # floor, and bisecting it would not help: 06b's premise holds, as 06g states it
        # (_roundoff_exit_premise_holds(); the comment above _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO
        # gives the argument and the values) --
        #   1. the halving test: its halves' floors, less their declared endpoint terms, sum to at
        #      least 1/2 of its own, less its own;
        #   2. the budget clause: the split could not meet the call's request
        #      (_split_could_meet_budget()).
        # Then bisection buys cost and no accuracy, so the leaf is not refined, and is flagged
        # phase_limited. The residual is compared with the leaf's whole floor, its node error |E|
        # included: the residual of one leaf sees its own node error, not the call's sum (06g).
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
        #
        # The exit is barred on every window leaf and every guarded leaf (prompt 07 section 0.2
        # (ii)): they are never floor-limited, and stay refinable until depth_max.
        phase_err = region.phase_err
        if (
            not region.window
            and not leaf.guarded
            and np.isfinite(phase_err)
            and region.abserr <= phase_err
        ):
            if _test_only_bypass_roundoff_premise:
                floor_limited = True
            else:
                if leaf.dataL is None:
                    _solve_halves(leaf)
                floor_limited = _roundoff_exit_premise_holds(
                    region.phase_err_roundoff,
                    _solve_floor_roundoff(leaf.dataL),
                    _solve_floor_roundoff(leaf.dataR),
                    split_could_meet_budget=_split_could_meet_budget(leaf),
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
            window=region.window,
            window_index=leaf.window_index,
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
            window=region.window,
            window_index=leaf.window_index,
        )

        num_splits += 1

        # The window's stopping rule (prompt 07 section 1.3; its count dqage's since prompt 07a
        # section 1.2): count this bisection against its window if neither half's estimate is
        # unresolved (dqage's skip), the halves change the value by at most min(1e-5, rtol)
        # relatively, and the halves' own solves' errors do not fall (QUADPACK's iroff1, on the
        # raw estimates e, before the trust check widens a declaration); see the constants'
        # comment and _window_bisection_counts(). fell_at_last_bisection still reports whether the
        # declared e' fell, as at 07: it feeds "leaves_falling_at_stop", and is not the count's
        # test.
        stop_window = None
        if leaf.window_index is not None:
            value_halves = left.region.value_own + right.region.value_own
            error_halves = _error_for_ordering(left.error) + _error_for_ordering(
                right.error
            )
            error_parent = _error_for_ordering(leaf.error)
            error_flat = error_halves >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO * error_parent
            left.fell_at_last_bisection = not error_flat
            right.fell_at_last_bisection = not error_flat
            if _window_bisection_counts(
                region.value_own,
                value_halves,
                _own_error(region),
                _own_error(left.region),
                _own_error(right.region),
                _cc_leaf_unresolved(left.region.abserr, left.region.resasc),
                _cc_leaf_unresolved(right.region.abserr, right.region.resasc),
                rtol,
            ):
                window_counts[leaf.window_index] = (
                    window_counts.get(leaf.window_index, 0) + 1
                )
                if (
                    window_counts[leaf.window_index] >= _LEVIN_WINDOW_IROFF1_LIMIT
                    and leaf.window_index not in windows_stopped
                ):
                    stop_window = leaf.window_index

        _remove_leaf(leaf)
        leaf.dataL = None
        leaf.dataR = None
        leaf.grandchildren = None
        _add_leaf(left)
        _add_leaf(right)
        if stop_window is not None:
            _stop_window(stop_window)

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
    # contributes max(resolution/fallback residual, round-off floor), plus its truncation term --
    # with its node error |E| taken out of a fallback leaf's floor (used_interval.budget_err, the
    # e'_i the loop's budget sums) -- and the fallback leaves' node errors enter signed, as
    # |sum E| (since prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (c):
    # E's sign varies from leaf to leaf, and sum |E| does not fall under bisection). The
    # resolution and round-off estimates are of independent error sources, and the larger
    # dominates.
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
    val, abserr_less_node, node_error_signed_sum = _sum_leaves_ascending(
        leaves_ascending
    )
    abserr_node_signed = float(np.fabs(node_error_signed_sum))
    abserr_total = abserr_less_node + abserr_node_signed
    node_error_abs_sum = math.fsum(
        abs(region.node_error_signed)
        for region in used_regions
        if region.node_error_signed is not None
    )

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

    # The stationary-point guard's returned counts (prompt 07 section 1.5). Every guarded leaf
    # below depth_max was bisected (its error is +inf, so the budget cannot stop the loop while
    # one is in the partition, and it is never floor-limited), so the guarded leaves returned are
    # those re-solved at depth_max.
    num_window_pieces = sum(1 for region in used_regions if region.window)
    num_guard_at_depth_max_returned = sum(
        1 for region in used_regions if region.guard_at_depth_max
    )

    # guarded exactly as the per-region relative error is guarded, and for the same reason
    relerr_total = abserr_total / max(np.fabs(val), atol)

    # The call's budget, by the formula the loop stops on (step 1 above), from the returned sums.
    # A call that stopped on the budget meets it by construction; one that stopped because no
    # leaf was refinable does not, and every leaf still holding error back is then at depth_max,
    # or floor-limited (round-off, phase_limited). And the call is not converged while a leaf it
    # returns is untrusted (the trust check one level down; such a leaf is at depth_max). A call
    # that stopped node-limited does not meet it either: there sum e'_i meets the request and
    # |sum E| holds the call above it. Nor is it converged while it returns a guarded leaf
    # re-solved at depth_max (prompt 07 section 1.4).
    requested_total = max(atol, rtol * np.fabs(val))
    budget_met = abserr_total <= requested_total
    converged = (
        budget_met and num_untrusted == 0 and num_guard_at_depth_max_returned == 0
    )

    _logger.debug(
        "adaptive_levin (%s): global budget | %d leaves, %d bisections, %d floor-limited, "
        "%d at depth_max | trust check: %d checks in %d passes, %d found untrusted, %d returned "
        "untrusted, %d returned unchecked | running test %s, returned budget met %s%s, running "
        "test passed but returned sums did not %d time(s) | node errors: sum E %.3g, sum |E| "
        "%.3g, node-limited %s",
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
        node_error_signed_sum,
        node_error_abs_sum,
        node_limited,
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
        if node_limited:
            _logger.warning(
                "   -- node-limited: the leaves' errors meet the request, and the signed sum of "
                "the Clenshaw-Curtis leaves' node errors, |sum E| = %.3g (sum |E| = %.3g), holds "
                "the call above it; bisection does not lower it systematically, so the call was "
                "not refined further",
                abserr_node_signed,
                node_error_abs_sum,
            )
        else:
            _logger.warning(
                "   -- no leaf of the partition could be refined further: of %d leaves, %d are at "
                "depth_max (%d of them with an error estimate that failed the trust check one "
                "level down) and %d are limited by the round-off floor (Chen et al. eq. 151); "
                "consider a looser atol/rtol, a larger depth_max, or a smaller chebyshev_order",
                num_used_regions,
                num_at_depth_max,
                num_untrusted,
                num_phase_limited,
            )
        if num_guard_at_depth_max_returned > 0:
            _logger.warning(
                "   -- %d leaves held a sign change of a phase rate on their nodes at depth_max "
                "(the stationary-point guard) and were re-solved by the nested Clenshaw-Curtis "
                "pair; the call is not converged while one is returned",
                num_guard_at_depth_max_returned,
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
        # The Clenshaw-Curtis leaves' node errors (prompts/levin-phase-floor/
        # 06g-leaf-values-and-node-sum.md, section 0.2 (c)): their signed sum, correctly rounded
        # (math.fsum over the regions in ascending x); the sum of their absolute values, which
        # "abserr_roundoff" carries as before; and |sum E|, the node errors' part of "abserr",
        # which carries them signed. The correction by -sum E is measured, not applied.
        "node_error_signed_sum": float(node_error_signed_sum),
        "node_error_abs_sum": float(node_error_abs_sum),
        "abserr_node_signed": abserr_node_signed,
        # True if the call stopped because |sum E| alone held it above its request: the leaves'
        # errors met it, every leaf was checked or floor-limited, and no refinable leaf was
        # untrusted. "converged" is then False.
        "node_limited": bool(node_limited),
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
        # The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md
        # section 1.5; see this function's docstring): each located point, with its window; each
        # merged window, with its piece cuts; the minima found and not located (lambda at or above
        # _LEVIN_COALESCENCE_LAMBDA); the window leaves returned; the leaves found guarded when
        # made, returned or bisected; and the guarded leaves re-solved by the nested pair at
        # depth_max, none of which may be returned by a converged call. All empty or zero when
        # the basis supplies no rates and no leaf is guarded.
        "stationary_points": (
            [dict(point) for point in location["points"]]
            if location is not None
            else []
        ),
        "stationary_windows": (
            [
                {
                    "lo": stationary_window["lo"],
                    "hi": stationary_window["hi"],
                    "cuts": list(stationary_window["cuts"]),
                    "points": list(stationary_window["points"]),
                    # section 1.3's rule: the window's count of iroff1 bisections (dqage's count,
                    # since prompt 07a section 1.2), and, if the rule stopped it, after which
                    # bisection of the call, how many of its leaves there were then, how many of
                    # them a bisection had just lowered the declared error e' of
                    # (fell_at_last_bisection), and their deepest depth
                    "iroff1_count": window_counts.get(window_index, 0),
                    "stopped": window_index in windows_stopped,
                    **windows_stopped.get(window_index, {}),
                }
                for window_index, stationary_window in enumerate(location["windows"])
            ]
            if location is not None
            else []
        ),
        "unlocated_minima": (
            [dict(point) for point in location["unlocated_minima"]]
            if location is not None
            else []
        ),
        "num_window_pieces": num_window_pieces,
        "num_guarded_regions": num_guarded_regions,
        "num_guard_at_depth_max": num_guard_at_depth_max,
        # each phase the location skipped, its rate exactly zero at every finite scan sample (prompt
        # 07a section 1.1, in place of prompt 07 section 10's Rule A), with its index and the reason;
        # empty when nothing was skipped or the location did not run
        "location_skipped_phases": (
            [dict(skipped) for skipped in location["skipped_phases"]]
            if location is not None
            else []
        ),
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
            into the derivative estimate. Since prompts/levin-phase-floor/
            07-stationary-point-guard.md it also drives the stationary-point location: the
            driver scans it over x_span before integrating, locates every sign change of theta'
            and every interior minimum of |theta'| deep enough to act as a nearly coalescing
            pair, and integrates a window round each directly (see "stationary_points" below and
            _adaptive_levin()), unless theta' is exactly zero at every finite sample of that
            scan, when there is nothing to locate (see "location_skipped_phases" below; since
            prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md, in place of Rule
            A, which skipped a phase whose variation over x_span was below SIX_PI). The raw
            phase "theta" is also what the C2 gate's variation is computed from, at each
            region's end nodes, or at all its nodes where theta' changes sign on them (see
            _Basis_SinCos.build_Levin_data()). It must be the closed form of theta's
            derivative. Without it nothing is located, and the only protection at a stationary
            point is the guard that never returns a Levin subinterval whose theta' changes sign on
            its own nodes, which misses a pair of sign changes between two adjacent nodes.
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
            truncation term (see abserr_truncation below) -- with the Clenshaw-Curtis
            subintervals' node errors taken out of their floors and carried signed, as
            |node_error_signed_sum| (as of prompts/levin-phase-floor/
            06g-leaf-values-and-node-sum.md) -- the quantity the driver's budget is tested on; it
            is an estimate, not a proven bound.
          * "node_error_signed_sum", "node_error_abs_sum", "abserr_node_signed" -- the
            Clenshaw-Curtis subintervals' node errors E (the signed first-order error of their
            computed nodes): their signed sum, the sum of their absolute values (which
            abserr_roundoff carries), and the absolute value of the signed sum (which abserr
            carries). New as of 06g; all zero when no subinterval took the fallback.
          * "node_limited" -- True if the driver stopped because |node_error_signed_sum| alone
            held the call above its request, with every subinterval's own error within it;
            "converged" is then False (new as of 06g).
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
            depth_max is untrusted; or the call stopped node-limited (see "node_limited").
          * "phase_limited" -- True if at least one subinterval the driver chose to refine while
            the budget was unmet was found floor-limited: its achievable accuracy was set by the
            round-off floor rather than by lack of resolution, on a floor that bisecting it would
            not lower (its halves' floors sum to at least half its own); tightening atol/rtol will
            not help such regions -- see abserr_roundoff above.
          * "num_phase_limited_regions" -- count of such regions.
          * "num_untrusted_regions" -- count of subintervals returned that failed the trust
            check one level down: their halves' and grandchildren's values do not show the error
            falling fast enough to extrapolate (r = delta2 / delta1 at least 1/4, unless r < 1 and
            delta1 is within the subinterval's round-off floor; until 06g, r >= 1), so their
            estimate cannot be relied on. Each is at depth_max, declares max(residual, delta1) /
            (1 - r) if r < 1 and max(residual, delta1 + delta2) if not, and makes "converged" False
            (new as of 06c). Each returned used_interval carries delta1, delta2,
            convergence_ratio, trusted and declared_resolution; and value, value_kind and value_own:
            a trusted subinterval returns its quarters sum, the finest value computed, in place of
            its own solve's (as of 06g).
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
          * "stationary_points", "stationary_windows", "unlocated_minima", "num_window_pieces",
            "num_guarded_regions", "num_guard_at_depth_max" -- the stationary-point guard (as of
            prompts/levin-phase-floor/07-stationary-point-guard.md): each located point (its x,
            kind "root", "minimum" or "end", theta' there, lambda for a minimum, and its window),
            each merged window with its piece cuts, the minima found but not located, the window
            subintervals returned, the subintervals found guarded (theta' changing sign on their
            nodes) when made, and those re-solved by the Clenshaw-Curtis rule at depth_max, which
            make "converged" False. Each used_interval carries window, guarded and
            guard_at_depth_max. See _adaptive_levin().
          * "location_skipped_phases" -- a phase whose theta_deriv is exactly zero at every
            finite sample of the location's scan has nothing to locate, and is not located; each
            such phase is listed with its index and the reason ("phase", and "reason":
            "zero_rate"). Every other phase is located whatever its variation (as of prompts/
            levin-phase-floor/07a-window-stop-and-phase-variation.md; from prompt 07 section 10
            until then, Rule A skipped a phase whose variation over x_span, estimated on the
            scan, was below SIX_PI, and listed it with that estimate, "variation").
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
