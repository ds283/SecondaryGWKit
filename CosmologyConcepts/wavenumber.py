from functools import total_ordering
from math import log10, log, fabs, exp
from typing import Iterable, Optional, Mapping, List, NamedTuple, Sequence, Tuple

import numpy as np
import ray
from numpy import logspace
from scipy.optimize import root_scalar

from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance
from Units import check_units
from config.defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_REDSHIFT_RELATIVE_PRECISION,
)
from utilities import WallclockTimer

# ---------------------------------------------------------------------------------------------
# the cosmology-aware source grid (prompt 11 of prompts/qcd-background-audit, audit section 7)
# ---------------------------------------------------------------------------------------------

# The identity of the *algorithm* that builds the source grid: everything in this section, plus
# build_z_sample below and the way wavenumber_exit_time.populate_source_grid calls it.
#
# **What it identifies, and what it does not.** It names the construction, not the grid. Two runs
# at the same version can still produce different grids -- a different z_end, a different
# samples_per_log10z, a different cosmology -- and the content digest
# (CosmologyConcepts.redshift.redshift_grid_digest, prompt 11) is what separates those. The pair
# answers two different questions, and neither substitutes for the other: the digest says
# *which exact grid*, and cannot be inverted or range-queried; this integer says *which
# generation of the algorithm*, and is the only thing that can be compared across grids whose
# values legitimately differ.
#
# **A bump is the only signal a datastore gets.** Nothing about how a grid was constructed is
# recoverable from the stored samples, so an unbumped change to the code in this section serves
# objects computed under the old construction as if they were the new one -- which is the failure
# Datastore/SQL/ObjectFactories/QCD_Cosmology.py's module docstring describes one level in, for
# the T(z) representation. **Any prompt that changes how the grid is constructed bumps this**,
# and records what it changed in the table below.
#
#   version 1 (prompt 11 of prompts/qcd-background-audit, recorded by prompt 14) -- a uniform
#     logspace base grid at the caller's samples_per_log10z, plus, for a cosmology that declares
#     integration_break_points: a pair straddling each declared crossing at
#     SOURCE_GRID_BREAK_STANDOFF = 0.25 of a base interval, the
#     SOURCE_GRID_BREAK_HALF_WIDTH = 5 intervals either side of it refined by
#     SOURCE_GRID_BREAK_REFINEMENT = 2, and the two equality redshifts. A cosmology that declares
#     nothing gets the bare logspace, element for element.
#   version 2 (prompt 15 of prompts/qcd-background-audit) -- version 1, plus a *base density* set
#     by the measured curvature criterion of docs/qcd-background-verification.md section 10:
#     every base interval that the fourth-derivative equidistribution condition
#     h^4 |phi''''| / 384 <= eps finds too wide is subdivided, in the base grid's own coordinate,
#     by the smallest integer that satisfies it. The criterion reaches build_z_sample as a
#     `spacing` profile computed by the caller (main.source_grid_spacing_profile), so no cosmology
#     object reaches this module; SOURCE_GRID_MAX_SPACING_FACTOR = 1.0 is the cap, and at that
#     value the construction may only ever *refine*. A cosmology that declares nothing no longer
#     gets the bare logspace from a production run -- it gets the bare logspace refined -- though
#     build_z_sample called without a `spacing` still reproduces it element for element.
SOURCE_GRID_CONSTRUCTION_VERSION = 2

# The standoff at which the pair of samples straddling a declared break point is placed, **as a
# fraction of the grid's own spacing**: the pair goes at
#
#     z_break -+ SOURCE_GRID_BREAK_STANDOFF * (base grid's relative step in z) * (1 + z_break),
#
# which is a relative standoff in (1+z), sized by the grid rather than by an absolute constant.
#
# **This is not Quadrature/integrators/numeric_with_phase_cut.BREAK_POINT_STANDOFF = 1e-12, and
# must not be set from it.** That constant places an ODE *restart* boundary: nothing is
# interpolated there, nothing is stored there, and its only requirement is to clear the
# cosmology's ~1e-16 evaluation noise, which is why it can be that small and why its own
# docstring says the choice "is not delicate". The number here is a *sample* location -- a
# datastore row, an interpolation site for the consumer's cubic, and an ordinate carrying the
# consumer's storage granularity -- and its choice is delicate. Four constraints bracket it, the
# last two of which are why an absolute 1e-12, or even 1e-5, is the wrong answer:
#
#  * **Above 1e-7 relative, by the datastore.** Datastore/SQL/ObjectFactories/redshift.py matches
#    an existing row with |z_stored - z| / z < DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7, so a
#    pair closer than that is *one row*: the two samples would silently collapse into a single
#    redshift object and the straddle would not exist. build_z_sample refuses such a standoff
#    rather than let it happen quietly.
#  * **Far above 3.6e-15 relative, by the redshift arithmetic.** A cosmology declares its break
#    points in u = log(1+z) (LambdaCDM_GenericEOS.integration_break_points), so the value
#    reaching this module has been through the lossy u -> z direction (CLAUDE.md). The recovered
#    (1+z) is good to ~ulp(u) = 3.6e-15 relative on the production range, so any standoff well
#    above that provably brackets the true crossing and no equality-like comparison is needed.
#  * **Above the consumer's own noise floor.** This is the binding constraint and it is not
#    obvious. PrimitivePhase is handed phi values whose granularity is that of the stored phase
#    -- about 1 ulp of the phase span, and `[02-consumer-phi-below-the-storage-granularity]` is
#    the standing record that this is real. Two samples a distance d apart therefore carry a
#    *slope* uncertainty ~2 ulp / d, and a cubic through them propagates it to the neighbouring
#    intervals. Measured at QCD k=1e5 (log 11 section 3): a pair at 1e-5 relative leaves the
#    consumer at 2.52 / 123.45 ulp where the neighbourhood refinement alone reaches 1.64 / 74.60,
#    a factor of 1.5 to 1.7 *worse*; the damage disappears once the pair is a fixed fraction of a
#    grid interval apart. Tying the standoff to the grid spacing is what keeps it above this
#    floor at any grid density, which an absolute constant cannot do.
#  * **Below the grid spacing**, or the pair is not straddling anything in particular.
#
# A quarter of a grid interval satisfies all four with room, and it was **scored** from 1/2 down
# to 1/10000 of an interval on both QCD k=1e5 consumer rows (log 11 section 3). It is the only
# value in that scan that beats the neighbourhood refinement alone on *both* rows -- 1.61 ulp
# against 1.64 (G_k) and 65.78 against 74.60 (T_k) -- and below ~1/32 the pair stops helping and
# then hurts, reaching a plateau 1.5x to 1.7x worse than no pair at all as the slope it implies
# drowns in the consumer's storage granularity. Geometrically it is also the natural value: the
# neighbourhood is refined by SOURCE_GRID_BREAK_REFINEMENT = 2, so the local spacing is half an
# interval and the pair sits half a *refined* interval either side of the break.
#
# On the production grid it is 5.82e-03 relative in (1+z): 5.8e4 times the datastore's
# resolution, 1.6e12 times the redshift-recovery granularity, and 4 times tighter than the base
# samples either side.
SOURCE_GRID_BREAK_STANDOFF = 0.25

# Number of base grid intervals either side of a declared break point whose spacing is refined,
# and the factor by which they are refined.
#
# **These are prompt 10's measurement, not a guess** (logs/10-primitive-phase-break-points.md
# section 4, and docs/qcd-background-verification.md section 8). The consumer's error at
# QCD_EOS's T_LO crossing is *not* confined to the break's own interval: four extra samples
# inside that interval alone buy 1.96x and then stall, while refining +-5 intervals by 2x --
# 10 extra samples in 1,016, 1.0 % -- takes QCD k=1e5 from 34.11 to 1.64 ulp (G_k) and from
# 1876.61 to 74.60 ulp (T_k). The feature is three to five grid intervals wide, so a grid design
# that protects only the declared point does not work, and the straddling pair above is necessary
# but on its own nowhere near sufficient.
SOURCE_GRID_BREAK_HALF_WIDTH = 5
SOURCE_GRID_BREAK_REFINEMENT = 2

# The closest two samples of the grid are ever allowed to be, relative in (1+z). This is a
# *degeneracy guard*, not a spacing policy: its only job is to keep any two samples further apart
# than DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7, the tolerance with which
# Datastore/SQL/ObjectFactories/redshift.py identifies an existing redshift row, so that no two
# grid points can silently become the same redshift object. Ten times that leaves a margin and is
# still 4.3e-05 of a production grid interval.
SOURCE_GRID_MIN_SEPARATION = 10.0 * DEFAULT_REDSHIFT_RELATIVE_PRECISION

# ... and the second half of the guard, which *is* about spacing: no candidate may be placed
# closer than this fraction of the straddling standoff to a sample already accepted. Without it
# the grid is only as well conditioned as the accident of where a break falls inside its interval
# -- a break a quarter of an interval from a base point would put a pair member exactly on that
# point. With it, the tightest gap anywhere near a break is a quarter of the standoff and the
# widest is the refined spacing, so the local mesh ratio is bounded by 8 whatever the accident.
# On the production QCD grid it fires on nothing: the closest a base sample comes to a pair
# member is 0.108 of an interval, against the 0.0625 this permits (log 11 section 3).
SOURCE_GRID_MESH_GUARD = 0.25


# ---------------------------------------------------------------------------------------------
# the curvature criterion that sets the base density (prompt 15; verification document section 10)
# ---------------------------------------------------------------------------------------------

# **The cap: the coarsest the grid is allowed to be, as a multiple of the spacing today's uniform
# `samples_per_log10z` lattice puts at the same place.** At 1.0 the criterion may only ever
# *refine*: no interval of the grid this module builds is wider than the base interval that
# contains it, on any cosmology, anywhere.
#
# **This is a decision, and the saving it declines is measured.**
# docs/qcd-background-verification.md section 10.5 costs a ladder of caps: at a cap of 2 the
# criterion needs 1.75x fewer samples on QCD_Cosmology and 2.06x fewer on LambdaCDM at the *same*
# accuracy, and the ladder goes on to 3.4x and 7.9x. The whole of that saving is coarsening at low
# z, where the phase residual's curvature is smaller than the grid's density by up to 1e19 in
# error -- so it is real. It is declined, on the user's instruction (prompt 15 section 1):
#
#     "This is a science code. We want to use compute resource sensibly, efficiently, and without
#      extravagance, but not take any short cuts or risks: there is no reward for doing so. The
#      only thing we get is an unreliable published result. I would much rather have the grid
#      slightly more dense than needed, than have it underdense."
#
# Three things the criterion cannot see stand behind that instruction, and each of them lives on
# this grid: the numeric ODE's own sample points, the four CumulativeTables' Gauss panels, and the
# abscissae QuadSource and QuadSourceIntegral integrate the source over. A criterion derived from
# the consumer's phase alone is a lower bound on the density and never an upper one (section
# 10.0). The fourth is measured: [03-derivative-pad-clamp-on-coarse-grids] clamps
# BackgroundModel._build_derivative_fit_grid's padding the moment the lowest grid interval exceeds
# -log(0.9)/12 = 8.7800e-03 in u, which is exactly where the coarsening would begin.
#
# **Moving it is not a change of a number.** Raising this above 1.0 does not by itself coarsen
# anything: the construction below only ever subdivides a base interval, so a cap above 1 would
# additionally need a code path that *drops* base samples, and writing that path is the
# deliberate act this constant exists to force.
SOURCE_GRID_MAX_SPACING_FACTOR = 1.0

# The largest subdivision of a single base interval the criterion is allowed to ask for before the
# construction refuses. On the production envelope -- both models, both sectors, all fifty
# production wavenumbers -- the largest it asks for is 4. A criterion that asks for far more has
# either met a cosmology nothing here has been measured on or gone wrong, and a grid of unknown
# size is worse than a refusal.
SOURCE_GRID_MAX_REFINEMENT = 32

# The finite stand-in for "this part of the profile constrains nothing", in u. Larger than the
# whole production range (u runs to ~37.6), so it can never bind, and finite so that the profile
# can be interpolated.
_SPACING_UNCONSTRAINED = 1.0e6

# The cubic interpolation error constant, max|e| ~ CONST * h^4 |f''''| on an interval of width h.
# 1/384 is the textbook bound for a local cubic through four equally spaced points, and section
# 10.3 measures it rather than assuming it: in the transfer-function sector the indicator predicts
# the realised error to +-2 % over 500-odd production intervals, on both models, at all three
# reference wavenumbers, with this constant and no fitting.
SOURCE_GRID_CUBIC_ERROR_CONST = 1.0 / 384.0

# Step in u = log(1+z) for the five-point stencil that takes the third derivative of the
# closed-form phi'. Prompt 12's value, unchanged: roundoff in the stencil is ~ eps|g|/(2 delta^3)
# ~ 5e-08 |g| and truncation ~ delta^2 |g^(5)|, both far below anything a density criterion
# resolves.
SOURCE_GRID_CURVATURE_STEP_U = 1.0e-3

# Step in u for the finite differences that supply epsilon, d_epsilon_dz and d_wPerturbations_dz
# from a cosmology that does not compute them in closed form. Section 10.1 licenses this: epsilon
# from a central difference of the cosmology's own pointwise Hubble agrees with epsilon as a
# BackgroundModel supplies it to max 2.5e-09 (LambdaCDM) and 3.9e-09 (QCD_Cosmology), median
# 8.1e-10, away from a declared crossing.
SOURCE_GRID_CURVATURE_FD_STEP_U = 1.0e-4

# How far either side of a declared crossing the curvature is not evaluated, in u. H has a genuine
# *step* there, so no derivative of it means anything within a few stencil widths of one, and the
# criterion is log-interpolated across the gap. What happens inside the gap is prompt 11's
# mechanism -- a straddling pair and a refined neighbourhood placed on a rule -- and not a density
# set by a curvature. Six stencil steps, as prompt 12 masked it.
SOURCE_GRID_CROSSING_MASK_U = 6.0 * SOURCE_GRID_CURVATURE_STEP_U

# prompt 10 section 5's consumer target, the ceiling on the per-case error target
SOURCE_GRID_CONSUMER_TARGET_RAD = 1.0e-6

# **The end condition, and why the criterion alone is not enough.** SOURCE_GRID_CUBIC_ERROR_CONST
# is the constant of an interpolating cubic in its *interior*. scipy's make_interp_spline closes
# the system with a not-a-knot condition at each end, whose error constant in the outermost
# intervals is much larger, and the two rows section 10.2 records as a miss are exactly there --
# in the topmost interval of the Liouville-Green band, within a few grid intervals of horizon
# entry. Measured, realised/predicted per interval on the production grid:
#
#     interval from the band edge      1st      2nd      3rd      interior
#     LambdaCDM Tk k = 1e5            9.897    4.292    1.230    p50 0.923, p90 0.934
#     QCD       Tk k = 1e5            9.792    4.189    ~1       p50 0.928, p90 0.941
#     QCD       Gk k = 1e5            9.327    4.546    1.379    p50 1.223
#
# So the error target is divided by SOURCE_GRID_SPLINE_EDGE_FACTOR over the outermost
# SOURCE_GRID_SPLINE_EDGE_INTERVALS intervals at each end of each band, which asks for a spacing
# smaller there by 10^(1/4) = 1.78. It is applied at the *edges only* rather than as a global
# safety factor because the same measurement shows the interior constant is right: a global factor
# of ten would refine 560 production intervals instead of 226, all of them where the realised
# error is 1e-12 against a 1e-07 floor.
SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3
SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0

# **The ceiling on what the density criterion's guard may absorb**, as a fraction of one
# (k, sector) band's nodes. Prompt 02a of prompts/tolerance-convergence.
#
# The band is established node-wise by residual_node_range and the stencil is evaluated off-node,
# at u +- delta and u +- 2 delta, so where H steps omega^2 can be negative between two nodes that
# both pass the margin test. Such a node has no fourth derivative to equidistribute and is marked
# unusable and log-interpolated across, exactly as a declared crossing's neighbourhood is. Before
# that guard existed the construction raised outright and a QCD production run could not build its
# source grid at all ([01-v2-density-raises-at-the-qcd-production-anchor]).
#
# A guard with no ceiling is the worse defect, because the raise at least stops: it would let an
# arbitrarily misplaced band be absorbed silently, and the profile would then be a log-interpolation
# through whatever few nodes survived. So the guard refuses above this fraction and names the count.
#
# Measured, at production geometry -- fifty wavenumbers, both sectors, 100 samples per decade of z:
#
#     cosmology / anchor                       guarded    band    worst single band
#     QCD at LambdaCDM's anchor (1996 samples)       0   136492            0
#     LambdaCDM at its own anchor (1778)             0   150932            0
#     QCD at its own anchor (2034)                  53   136453     7.716e-04
#
# The only production case that guards anything reaches **7.716e-04** of a band -- one node of
# 1,304, on 53 of the 100 (k, sector) cases -- and that figure is stable at every relative
# perturbation of z_init from 1e-16 to 1e-8, so it is not one float's accident. 0.05 sits **64.8x**
# above it. The margin is deliberately large in that direction and small in the other: at 5% of a
# band, 95% of its nodes remain to fit the log-interpolation from, so anything that trips this
# ceiling is a misplaced band rather than a boundary effect, which is the distinction the constant
# exists to draw. Whether the criterion should run over a horizon-based band of its own instead is
# [01-density-criterion-imposed-outside-the-wkb-region], and the guarded counts above are its
# evidence, not this constant's business.
SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05


class SourceGrid(NamedTuple):
    """
    The source sample grid, together with the record of which of its points are there because the
    cosmology asked for them.

    ``z_values`` is what ``populate_z_sample`` has always returned: a descending array of
    redshifts. ``protected_z`` is the subset that must survive any later decimation -- the pair
    straddling each break point, and each feature redshift -- as a descending array;
    ``breaks`` and ``features`` are the declared points themselves, restricted to the grid's range.
    A cosmology that declares nothing gives ``z_values`` bit-identical to today's ``logspace`` and
    three empty arrays.
    """

    z_values: np.ndarray
    protected_z: np.ndarray
    breaks: np.ndarray
    features: np.ndarray


def _relative_separation(z: float, others: np.ndarray) -> float:
    """
    The smallest separation between ``z`` and ``others``, relative in (1+z). Infinite if
    ``others`` is empty.

    Relative in (1+z) rather than in z because that is the coordinate every tolerance in this
    campaign is expressed in, and because it stays meaningful as z -> 0. Note that a separation
    of eps relative in (1+z) is *at least* eps relative in z, which is the direction that matters
    for staying clear of the datastore's DEFAULT_REDSHIFT_RELATIVE_PRECISION.
    """
    if len(others) == 0:
        return float("inf")
    return float(np.min(np.abs(others - z) / (1.0 + np.minimum(others, z))))


def build_z_sample(
    z_init: float,
    z_end: float,
    samples_per_log10z: int,
    *,
    break_z: Sequence[float] = (),
    feature_z: Sequence[float] = (),
    standoff: float = SOURCE_GRID_BREAK_STANDOFF,
    half_width: int = SOURCE_GRID_BREAK_HALF_WIDTH,
    refinement: int = SOURCE_GRID_BREAK_REFINEMENT,
    spacing: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> SourceGrid:
    """
    Build the source sample grid: the log-spaced lattice this function has always returned, plus
    whatever the cosmology has asked to be resolved.

    ``break_z`` are redshifts at which the background loses smoothness -- in practice the
    crossings of an equation of state's branch temperatures, which is what
    ``ComputeTargets.BackgroundModel._cosmology_break_points`` returns. Each one in range gets

      * a **pair of samples straddling it** at ``standoff`` relative in (1+z), so that the
        consumer's spline has a data point on each side of the step and never has to bridge it,
        and so that no equality-like comparison on a recovered z is needed to say which side a
        sample is on (CLAUDE.md's redshift rule); and
      * a **refinement of the ``half_width`` intervals either side** by ``refinement``, because
        prompt 10 measured that the feature is three to five grid intervals wide and that
        protecting the crossing alone stalls at a factor of two.

    ``feature_z`` are redshifts that merely have to be *present* -- matter-radiation and
    matter-Lambda equality -- where the background is perfectly smooth and one sample is enough.

    ``spacing`` is the **base density**, as ``(u_profile, h_profile)``: ascending samples of
    ``u = log(1+z)`` and, at each of them, the largest spacing in ``u`` that the consumer's phase
    spline can tolerate there. Every base interval wider than the profile allows is subdivided, in
    the base grid's own coordinate, by the smallest integer that satisfies it, and
    ``SOURCE_GRID_MAX_SPACING_FACTOR`` caps how coarse the result may be against the base lattice
    -- at its shipped value of 1.0 the subdivision is by at least 1, i.e. the construction may
    only ever refine. The profile is computed by ``main.source_grid_spacing_profile`` from the
    cosmology's ``Hubble`` and ``wPerturbations`` alone; the criterion it implements, and the
    measurement that chose it, are docs/qcd-background-verification.md section 10.

    **No cosmology object reaches this function and no equation-of-state module is imported
    here.** It takes values -- break redshifts, feature redshifts and a spacing profile --
    following ``_cosmology_break_points``'s own duck-typed precedent. With none of them supplied
    it takes the early return below and reproduces ``numpy.logspace`` element for element, which
    is what ``ComputeTargets/tests/test_source_grid.py`` asserts; a *production* grid now always
    carries a spacing profile, on every cosmology, because the density question is not a question
    about the equation of state.

    :param z_init: the highest redshift in the grid
    :param z_end: the lowest redshift in the grid
    :param samples_per_log10z: base density, in samples per decade of z
    :param break_z: redshifts at which the background is not smooth (descending or ascending)
    :param feature_z: redshifts that must appear in the grid
    :param standoff: standoff for the straddling pair, as a fraction of a base grid interval
    :param half_width: number of base intervals either side of a break that are refined
    :param refinement: factor by which those intervals are refined
    :param spacing: ``(u_profile, h_profile)``, the largest spacing in ``u`` the criterion permits
    """
    num = int(round(samples_per_log10z * (log10(z_init) - log10(z_end)) + 0.5, 0))

    # the base grid, computed exactly as it has always been computed
    base = logspace(log10(z_init), log10(z_end), num=num)

    breaks = np.array(
        sorted((float(z) for z in break_z if z_end < z < z_init), reverse=True),
        dtype=float,
    )
    features = np.array(
        sorted((float(z) for z in feature_z if z_end < z < z_init), reverse=True),
        dtype=float,
    )

    empty = np.empty(0, dtype=float)
    if len(breaks) == 0 and len(features) == 0 and spacing is None:
        # nothing was declared and no density was asked for: this is today's grid, and it is
        # today's array object
        return SourceGrid(
            z_values=base, protected_z=empty, breaks=empty, features=empty
        )

    if refinement < 1 or half_width < 0:
        raise ValueError(
            f"build_z_sample: refinement must be >= 1 and half_width >= 0 "
            f"(got {refinement}, {half_width})"
        )

    # the exponents logspace itself used, so that a refinement point is placed in the base grid's
    # own coordinate rather than in a recovered one
    t = np.linspace(log10(z_init), log10(z_end), num)

    # the grid's own relative step in z, and the standoff it sets
    delta_log10 = fabs(float(t[0] - t[1]))
    standoff_rel = standoff * (pow(10.0, delta_log10) - 1.0)
    if standoff_rel <= SOURCE_GRID_MIN_SEPARATION:
        raise ValueError(
            f"build_z_sample: a break-point standoff of {standoff:.3g} of a grid interval is "
            f"{standoff_rel:.3g} relative on this grid, at or below the minimum separation "
            f"{SOURCE_GRID_MIN_SEPARATION:.3g} the datastore's redshift resolution imposes; the "
            f"two samples straddling a break would be the same redshift row"
        )

    # two tolerances, because two different things are being guarded against. Everything must
    # clear SOURCE_GRID_MIN_SEPARATION, or two samples become one datastore row; anything near a
    # straddling pair must additionally clear mesh_tol, or the pair's own conditioning is only as
    # good as the accident of where the break fell inside its interval.
    mesh_tol = max(SOURCE_GRID_MIN_SEPARATION, SOURCE_GRID_MESH_GUARD * standoff_rel)

    def _admits(z: float, pairs: np.ndarray, others: np.ndarray) -> bool:
        return (
            _relative_separation(z, pairs) > mesh_tol
            and _relative_separation(z, others) > SOURCE_GRID_MIN_SEPARATION
        )

    # ---- highest priority: the pair straddling each break ----
    pairs: List[float] = []
    pairs_arr = np.empty(0, dtype=float)
    for z_break in breaks:
        for sign in (+1.0, -1.0):
            z_new = float(z_break) + sign * standoff_rel * (1.0 + float(z_break))
            if _relative_separation(z_new, pairs_arr) > SOURCE_GRID_MIN_SEPARATION:
                pairs.append(z_new)
                pairs_arr = np.array(pairs, dtype=float)

    # ---- then the feature redshifts, which need to be present but need no pair ----
    others: List[float] = []
    others_arr = np.empty(0, dtype=float)
    for z in features:
        if _admits(float(z), pairs_arr, others_arr):
            others.append(float(z))
            others_arr = np.array(others, dtype=float)

    protected = sorted(pairs + others, reverse=True)

    # ---- then the base grid ----
    accepted = list(protected)
    for z in base:
        if _admits(float(z), pairs_arr, others_arr):
            accepted.append(float(z))
    accepted_arr = np.array(sorted(accepted), dtype=float)

    # ---- then the refinement of the neighbourhood of each break ----
    z_asc = base[::-1]
    t_asc = t[::-1]
    n = len(z_asc)
    for z_break in breaks:
        j = int(np.searchsorted(z_asc, float(z_break)))
        first = max(j - half_width, 1)
        last = min(j + half_width, n - 1)
        for i in range(first, last + 1):
            for m in range(1, refinement):
                frac = float(m) / float(refinement)
                z_new = float(
                    np.power(10.0, t_asc[i - 1] + frac * (t_asc[i] - t_asc[i - 1]))
                )
                if _admits(z_new, pairs_arr, accepted_arr):
                    accepted.append(z_new)
                    accepted_arr = np.array(sorted(accepted), dtype=float)

    # ---- and last, the base density the curvature criterion asks for ----
    #
    # This runs after the break-point work, not before it, so that every sample prompt 11 places
    # is placed exactly where prompt 11 placed it: the mesh guards see the same accepted set they
    # saw before, and a grid built with a spacing profile is a strict *superset* of the one built
    # without. That is what makes "never coarser than today, anywhere" a property of the
    # construction rather than a number to be checked afterwards.
    if spacing is not None:
        u_prof = np.asarray(spacing[0], dtype=float)
        h_prof = np.asarray(spacing[1], dtype=float)
        if u_prof.ndim != 1 or h_prof.shape != u_prof.shape or u_prof.size < 2:
            raise ValueError(
                "build_z_sample: spacing must be a pair of one-dimensional arrays of equal "
                f"length >= 2 (got shapes {u_prof.shape} and {h_prof.shape})"
            )
        if not np.all(np.diff(u_prof) > 0.0):
            raise ValueError(
                "build_z_sample: the spacing profile must be strictly ascending in u = log(1+z)"
            )
        # an unconstrained stretch of the profile carries +inf; the cap bounds it anyway, and a
        # finite value is what numpy.interp needs
        h_prof = np.where(np.isfinite(h_prof), h_prof, _SPACING_UNCONSTRAINED)

        t_asc = t[::-1]
        u_asc = np.log1p(base[::-1])
        h_at = np.interp(u_asc, u_prof, h_prof)
        for i in range(1, len(u_asc)):
            h_base = float(u_asc[i] - u_asc[i - 1])
            if h_base <= 0.0:
                continue
            h_allow = min(
                float(h_at[i - 1]),
                float(h_at[i]),
                SOURCE_GRID_MAX_SPACING_FACTOR * h_base,
            )
            m = max(1, int(np.ceil(h_base / h_allow)))
            if m > SOURCE_GRID_MAX_REFINEMENT:
                raise ValueError(
                    f"build_z_sample: the spacing profile asks for a subdivision of {m} in the "
                    f"base interval z = {base[::-1][i - 1]:.6g} .. {base[::-1][i]:.6g}, above "
                    f"the SOURCE_GRID_MAX_REFINEMENT = {SOURCE_GRID_MAX_REFINEMENT} this "
                    "construction will build; the largest asked for on the production envelope "
                    "is 4"
                )
            for j in range(1, m):
                frac = float(j) / float(m)
                z_new = float(
                    np.power(10.0, t_asc[i - 1] + frac * (t_asc[i] - t_asc[i - 1]))
                )
                if _admits(z_new, pairs_arr, accepted_arr):
                    accepted.append(z_new)
                    accepted_arr = np.array(sorted(accepted), dtype=float)

    z_values = np.array(sorted(accepted, reverse=True), dtype=float)

    return SourceGrid(
        z_values=z_values,
        protected_z=np.array(protected, dtype=float),
        breaks=breaks,
        features=features,
    )


@total_ordering
class wavenumber(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        k_inv_Mpc: float,
        units,
        is_source: bool = False,
        is_response: bool = False,
    ):
        """
        Represents a wavenumber, e.g.,
        used to sample a transfer function or power spectrum
        :param store_id: unique Datastore id. Should not be None
        :param k_inv_Mpc: wavenumber, measured in 1/Mpc
        :param units: units block (e.g. Mpc-based units)
        """
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        # units are available for inspection
        self.units = units

        self.k_inv_Mpc = k_inv_Mpc
        self.k = k_inv_Mpc / units.Mpc

        self.is_source = is_source
        self.is_response = is_response

    def __float__(self):
        """
        Cast to float. Returns dimensionful wavenumber.
        :return:
        """
        return float(self.k)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.k < other.k

    def __hash__(self):
        return ("wavenumber", self.store_id).__hash__()


class wavenumber_array:
    def __init__(self, k_array: Iterable[wavenumber]):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        """
        # store array in ascending order of k; the conversion to set ensure that we remove any duplicates
        self._k_array = sorted(set(k_array), key=lambda x: x.k)

    def __iter__(self):
        for k in self._k_array:
            yield k

    def __getitem__(self, key):
        return self._k_array[key]

    def __len__(self):
        return len(self._k_array)

    def __add__(self, other):
        full_set = set(self._k_array)
        full_set.update(set(other._k_array))
        return wavenumber_array(full_set)

    def as_list(self) -> list[float]:
        return [float(k) for k in self._k_array]

    def extend(self, k_array: Iterable[wavenumber]):
        full_set = set(self._k_array)
        full_set.update(set(k_array))
        self._k_array = sorted(full_set, key=lambda x: x.k)


WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS = [1, 2, 3, 4, 5]
WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS = [1, 2, 3, 4, 5, 6]


@total_ordering
class wavenumber_exit_time(DatastoreObject):
    def __init__(
        self,
        payload,
        k: wavenumber,
        cosmology: BaseCosmology,
        atol: tolerance,
        rtol: tolerance,
    ):
        """
        Represents the horizon exit time for a mode of wavenumber k
        :param store_id: unique Datastore id. May be None if the object has not yet been fully serialized
        :param k: wavenumber object
        :param cosmology: cosmology object satisfying the CosmologyBase concept
        """
        check_units(k, cosmology)

        # store the provided z_exit value and compute_time value
        # these may be None if store_id is also None. This represents the case that the computation has not yet been done.
        # In this case, the client code needs to call compute() in order to populate the z_exit value
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._z_exit = None

            for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                setattr(self, f"_z_exit_suph_e{z_offset}", None)
            for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                setattr(self, f"_z_exit_subh_e{z_offset}", None)

            self._compute_time = None
            self._stepping = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._z_exit = payload["z_exit"]

            for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                setattr(
                    self,
                    f"_z_exit_suph_e{z_offset}",
                    payload[f"z_exit_suph_e{z_offset}"],
                )
            for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                setattr(
                    self,
                    f"_z_exit_subh_e{z_offset}",
                    payload[f"z_exit_subh_e{z_offset}"],
                )

            self._compute_time = payload["compute_time"]
            self._stepping = payload["stepping"]

        # store parameters
        self.k = k
        self.cosmology = cosmology

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.k < other.k

    def __hash__(self):
        return ("wavenumber_exit_time", self.store_id).__hash__()

    def compute(self, label: Optional[str] = None):
        if self._z_exit is not None:
            raise RuntimeError("z_exit has already been computed")
        self._compute_ref = find_horizon_exit_time.remote(
            self.cosmology,
            self.k,
            suph_efolds=WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
            subh_efolds=WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "wavenumber_exit_time: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._z_exit = data["z_exit"]

        for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
            setattr(self, f"_z_exit_suph_e{z_offset}", data[f"z_exit_suph_e{z_offset}"])
        for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
            setattr(self, f"_z_exit_subh_e{z_offset}", data[f"z_exit_subh_e{z_offset}"])

        self._compute_time = data["compute_time"]
        self._stepping = 0

        return True

    @property
    def z_exit(self) -> float:
        if self._z_exit is None:
            raise RuntimeError("z_exit has not yet been populated")
        return self._z_exit

    @property
    def compute_time(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_time has not yet been populated")
        return self._compute_time

    @property
    def stepping(self) -> int:
        if self._stepping is None:
            raise RuntimeError("stepping has not yet been populated")
        return self._stepping

    @property
    def atol(self) -> float:
        return self._atol.tol

    @property
    def rtol(self) -> float:
        return self._rtol.tol

    def populate_z_sample(
        self,
        samples_per_log10z: int = 50,
        z_end: float = 0.1,
        outside_horizon_efolds: int = 3,
        **kwargs,
    ):
        """
        Build a set of z sample points, with specified density per log_10(z), and ending at the specified z_end.
        The initial time is taken to be the horizon re-entry time for this k-mode, or possibly offset by a specified
        number of e-folds in/outside the horizon, specified in 'outside_horizon_efolds'

        Keyword arguments are forwarded to :func:`build_z_sample`; with none of them supplied
        this returns exactly the ``logspace`` it has always returned. Use
        :meth:`populate_source_grid` instead when the protected subset is needed as well.
        :param samples_per_log10z:
        :param z_end:
        :param outside_horizon_efolds:
        :return:
        """
        return self.populate_source_grid(
            samples_per_log10z=samples_per_log10z,
            z_end=z_end,
            outside_horizon_efolds=outside_horizon_efolds,
            **kwargs,
        ).z_values

    def populate_source_grid(
        self,
        samples_per_log10z: int = 50,
        z_end: float = 0.1,
        outside_horizon_efolds: int = 3,
        **kwargs,
    ) -> SourceGrid:
        """
        As :meth:`populate_z_sample`, but returning the whole :class:`SourceGrid` -- the sample
        values together with the subset that the cosmology asked for and that must therefore
        survive ``redshift_array.winnow``.

        :param samples_per_log10z:
        :param z_end:
        :param outside_horizon_efolds:
        :return:
        """
        if outside_horizon_efolds == 0:
            z_init = self._z_exit
        elif outside_horizon_efolds > 0:
            if outside_horizon_efolds not in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                raise RuntimeError(
                    f"wavenumber_exit_time: z_exit + superhorizon({outside_horizon_efolds}) is not computed"
                )

            z_init = getattr(self, f"z_exit_suph_e{outside_horizon_efolds}")
        elif outside_horizon_efolds < 0:
            if outside_horizon_efolds not in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                raise RuntimeError(
                    f"wavenumber_exit_time: z_exit + subhorizon({outside_horizon_efolds}) is not computed"
                )

            z_init = getattr(self, f"z_exit_subh_e{outside_horizon_efolds}")
        else:
            raise RuntimeError(
                f"Unknown outside_horizon_efolds value {outside_horizon_efolds}"
            )

        # now we want to build a set of sample points for redshifts between z_init and
        # the final point z = z_final, using the specified number of redshift sample points,
        # plus whatever the cosmology has asked to have resolved (audit section 7)
        return build_z_sample(
            z_init=z_init,
            z_end=z_end,
            samples_per_log10z=samples_per_log10z,
            **kwargs,
        )


# create accessors
def _create_accessor(attr_label):
    def accessor_template(self):
        if not hasattr(self, attr_label):
            raise RuntimeError(
                f'wavenumber_exit_time: object does not have attribute "{attr_label}"'
            )
        value = getattr(self, attr_label)
        if value is None:
            raise RuntimeError(
                f'wavenumber_exit_time: attribute "{attr_label}" has not yet been populated'
            )
        return value

    return accessor_template


for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
    setattr(
        wavenumber_exit_time,
        f"z_exit_suph_e{z_offset}",
        property(_create_accessor(f"_z_exit_suph_e{z_offset}")),
    )

for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
    setattr(
        wavenumber_exit_time,
        f"z_exit_subh_e{z_offset}",
        property(_create_accessor(f"_z_exit_subh_e{z_offset}")),
    )


class wavenumber_exit_time_array:
    def __init__(self, k_exit_array: Iterable[wavenumber_exit_time]):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        """
        # store array in ascending order of k; the conversion to set ensure that we remove any duplicates
        self._k_exit_array = sorted(set(k_exit_array), key=lambda x: x.k)

    def __iter__(self):
        for k in self._k_exit_array:
            yield k

    def __getitem__(self, key):
        return self._k_exit_array[key]

    def __len__(self):
        return len(self._k_exit_array)

    def __add__(self, other):
        full_set = set(self._k_exit_array)
        full_set.update(set(other._k_exit_array))
        return wavenumber_exit_time_array(full_set)

    def as_list(self) -> list[wavenumber_exit_time]:
        return [k for k in self._k_exit_array]

    def extend(self, k_array: Iterable[wavenumber]):
        full_set = set(self._k_exit_array)
        full_set.update(set(k_array))
        self._k_exit_array = sorted(full_set, key=lambda x: x.k)

    @property
    def max(self) -> wavenumber_exit_time:
        return self._k_exit_array[-1]

    @property
    def min(self) -> wavenumber_exit_time:
        return self._k_exit_array[0]


DEFAULT_HEXIT_TOLERANCE = 1e-2


def _solve_horizon_exit(
    cosmology: BaseCosmology,
    k: wavenumber,
    offset_subh: int,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
):
    """
    Solve the implicit equation log(k/aH) - offset_subh = 0 to find the horizon exit time (plus offset) associated with wavenumber k, i.e.
    (k/aH) = exp(offset).
    offset_subh should be a number representing the number of e-folds *inside* the horizon that we wish to locate.
    If it is a positive number, it represents the number of e-folds *inside* the horizon.
    If it is a negative number, it represents the number of e-folds *outside* the horizon.
    :param cosmology:
    :param k:
    :param offset:
    :param atol:
    :param rtol:
    :return:
    """

    # GUESS A SENSIBLE INITIAL REDSHIFT FOR THE CALCULATION

    # in radiation domination H(z) = H0 (1+z)^2 because H^2 ~ rho ~ T^4 and T ~ (1+z).
    # Therefore a(z)H(z) ~ H0(1+z).
    # The normalization a0 of a(z) is absorbed into k/a0 = k_phys.
    # Hence we can guess a possible redshift of horizon exit for this mode as
    #   1 + z_exit(k) = k/H0

    def q(log_z: float) -> float:
        z: float = exp(log_z) - 1.0
        return log(float(k) * (1.0 + z) / cosmology.Hubble(z)) - offset_subh

    log_z_guess = log(float(k) / cosmology.H0) - offset_subh
    q_guess = q(log_z_guess)

    # If q_guess is a positive number, then z_guess is a LOWER BOUND for the desired crossing time.
    # On the other hand, if q_quess is a negative number, then z_guess is an UPPER BOUND for the desired crossing time.

    COUNT_MAX = 25
    SEARCH_MULTIPLIER = 1.5
    LOG_SEARCH_OFFSET = log(SEARCH_MULTIPLIER)

    if q_guess > DEFAULT_HEXIT_TOLERANCE:
        log_z_lo = log_z_guess
        q_lo = q_guess

        log_z_hi = log_z_lo + LOG_SEARCH_OFFSET
        q_hi = q(log_z_hi)
        count = 0
        while q_hi > -DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX:
            log_z_hi = log_z_hi + LOG_SEARCH_OFFSET
            q_hi = q(log_z_hi)

            count += 1

        if count >= COUNT_MAX:
            raise RuntimeError(
                f"_solve_horizon_exit: failed to find upper bound z_hi for k={k.k_inv_Mpc:.5}/Mpc (z_lo={log_z_lo:.5g}, q_lo={q_lo:.5g}, last z_hi={log_z_hi:.5g}, last q_hi={q_hi:.5g})"
            )

    elif q_guess < -DEFAULT_HEXIT_TOLERANCE:
        log_z_hi = log_z_guess
        q_hi = q_guess

        log_z_lo = log_z_hi - LOG_SEARCH_OFFSET
        q_lo = q(log_z_lo)
        count = 0
        while q_lo < DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX:
            log_z_lo = log_z_lo - LOG_SEARCH_OFFSET
            q_lo = q(log_z_lo)

            count += 1

        if count >= COUNT_MAX:
            raise RuntimeError(
                f"_solve_horizon_exit: failed to find lower bound z_lo for k={k.k_inv_Mpc:.5}/Mpc (z_hi={log_z_hi:.5g}, q_hi={q_hi:.5g}, last z_lo={log_z_lo:.5g}, last q_lo={q_lo:.5g})"
            )

    else:
        # q_guess is very close to zero
        log_z_lo = log_z_guess - LOG_SEARCH_OFFSET
        log_z_hi = log_z_guess + LOG_SEARCH_OFFSET

        q_lo = q(log_z_lo)
        q_hi = q(log_z_hi)

    if q_hi * q_lo > 0.0:
        raise RuntimeError(
            f"_solve_horizon_exit: failed to bracket horizon crossing time for k={k.k_inv_Mpc:.5}/Mpc (z_lo={log_z_lo:.5g}, q_lo={q_lo:.5g}, z_hi={log_z_hi:.5g}, q_hi={q_hi:.5g})"
        )

    root = root_scalar(
        q,
        bracket=(log_z_lo, log_z_hi),
        xtol=atol,
        rtol=rtol,
    )

    if not root.converged:
        raise RuntimeError(
            f'_solve_horizon_exit: root_scalar() did not converge to a solution for k={k.k_inv_Mpc:.5}/Mpc: x_bracket=({log_z_lo:.5g}, {log_z_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    log_z_root = root.root
    q_root = q(log_z_root)
    if fabs(q_root) > DEFAULT_HEXIT_TOLERANCE:
        raise RuntimeError(
            f"_solve_horizon_exit: root_scalar() converged, but root is out of tolerance for k={k.k_inv_Mpc:.5}/Mpc: log_z_root={log_z_root:.5g}, |q_root|={fabs(q_root):.5g}, x_bracket=({log_z_lo:.5g}, {log_z_hi:.5g}), q_bracket-({q_lo:.5g}, {q_hi:.5g}), iterations={root.iterations}, method={root.method}"
        )

    return exp(log_z_root) - 1.0


@ray.remote
def find_horizon_exit_time(
    cosmology: BaseCosmology,
    k: wavenumber,
    suph_efolds: List[int],
    subh_efolds: List[int],
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> Mapping[str, float]:
    """
    Compute the redshift of horizon exit for a mode of wavenumber k in the specified cosmology
    :param cosmology:
    :param k:
    :return:
    """
    check_units(k, cosmology)

    payload = {}
    with WallclockTimer() as timer:
        z_exit = _solve_horizon_exit(
            cosmology, k, offset_subh=0.0, atol=atol, rtol=rtol
        )
        payload["z_exit"] = z_exit

        for z_subh in subh_efolds:
            z_exit = _solve_horizon_exit(
                cosmology, k, offset_subh=z_subh, atol=atol, rtol=rtol
            )
            payload[f"z_exit_subh_e{z_subh}"] = z_exit

        for z_suph in suph_efolds:
            z_exit = _solve_horizon_exit(
                cosmology, k, offset_subh=-z_suph, atol=atol, rtol=rtol
            )
            payload[f"z_exit_suph_e{z_suph}"] = z_exit

    payload["compute_time"] = timer.elapsed

    return payload
