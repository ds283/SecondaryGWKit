from bisect import bisect_right
from collections import namedtuple
from math import sqrt, log
from typing import Optional, List, Sequence, Union

import numpy as np
import ray
from ray import ObjectRef
from scipy.interpolate import make_interp_spline

from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyConcepts import redshift_array, redshift, wavenumber
from CosmologyModels import BaseCosmology

# the vocabulary of the cosmology's break-point declaration, re-exported so that a consumer of
# _cosmology_break_points() need not import an equation-of-state module to name the kind of
# non-smoothness it cares about (Quadrature/integrators/numeric_with_phase_cut.py does exactly
# this). CosmologyModels/GenericEOS/GenericEOS.py is the authority on what they mean.
from CosmologyModels.GenericEOS.GenericEOS import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
)
from Datastore import DatastoreObject
from MetadataConcepts import store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.supervisors.base import RHS_timer, IntegrationSupervisor
from Units.base import UnitsLike

# Gauss-Legendre order per production interval for the conformal-time table. Fixed by measurement
# in prompts/GkTk-remedial/logs/02-qcd-residual-convergence.md (N_tau = 4): order 4 is at the
# double-precision floor on LambdaCDM (review §7) and, once every interval is split at the
# cosmology's break points, on QCD_Cosmology as well. Raising it buys nothing.
TAU_GAUSS_ORDER = 4

# Gauss-Legendre orders for the two transfer-function primitives, fixed by the same measurement
# (N_cs_tau = N_F = 4; maximum increment error at order 4 under the break-point scheme is
# 4.55e-15 for c_s/H and 7.53e-16 for the friction integrand on QCD_Cosmology). They are named
# separately from TAU_GAUSS_ORDER so that a future re-measurement can move one without the
# others, but while all three agree the tables share main.py's single IntegrationSolver
# registration (see TAU_SOLVER_LABEL below, and [03-integrationsolver-stepping-minimum-lookup]).
CS_TAU_GAUSS_ORDER = 4
FRICTION_F_GAUSS_ORDER = 4

# The IntegrationSolver label under which the table is registered in main.py; "stepping" carries
# the Gauss order, following the existing "<label>-stepping<n>" convention.
TAU_SOLVER_LABEL_BASE = "cumulative-GL"
TAU_SOLVER_LABEL = f"{TAU_SOLVER_LABEL_BASE}-stepping{TAU_GAUSS_ORDER}"

# Settings for the private grid on which _build_derivative fits its splines when a cosmology model
# supplies no analytic derivative. See the comment in compute_background().
# - number of extra points added beyond each end of the production grid
DERIVATIVE_FIT_PAD_POINTS = 12
# - number of sub-intervals each production interval is divided into
DERIVATIVE_FIT_REFINE = 3
# - the low-end padding is never allowed to take 1+z below this multiple of 1+z_min. With z_min >= 0
#   this keeps the padded grid at z >= -0.1, inside the range over which the GenericEOS models build
#   their T(z) spline (DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2)
DERIVATIVE_FIT_PAD_FLOOR = 0.9
# - the padding at either end never extends the fitted range in log(1+z) by more than this fraction
DERIVATIVE_FIT_PAD_FRACTION = 0.05
# - degree of the interpolating spline that is differentiated
DERIVATIVE_SPLINE_ORDER = 5
# - degree of the interpolating spline BackgroundModel._create_functions fits through the *stored*
#   samples of any quantity the cosmology does not supply as a method. A cubic, which is what
#   make_interp_spline defaults to and what this site has always used; named here only so that the
#   degeneracy check below can quote the number of nodes a branch has to hold.
STORED_SAMPLE_SPLINE_ORDER = 3


def _build_derivative_fit_grid(z_sample: redshift_array):
    """
    Build the private, padded and refined grid in x = log(1+z) on which compute_background fits the
    splines it differentiates.
    Returns (fit_x, fit_z, fit_select, sample_order) where fit_x[fit_select][sample_order] are the
    production redshifts in the order they appear in z_sample.
    """
    z_prod = np.array([z.z for z in z_sample], dtype=float)

    # z_sample may run in either direction; fit on an ascending grid and record how to get back
    ascending = np.argsort(z_prod)
    sample_order = np.empty_like(ascending)
    sample_order[ascending] = np.arange(len(z_prod))

    x_prod = np.log1p(z_prod[ascending])

    # refine: insert DERIVATIVE_FIT_REFINE-1 equally spaced points inside each production interval
    refine = max(int(DERIVATIVE_FIT_REFINE), 1)
    if refine > 1:
        subdivided = x_prod[:-1, None] + (x_prod[1:, None] - x_prod[:-1, None]) * (
            np.arange(refine)[None, :] / refine
        )
        x_core = np.concatenate([subdivided.reshape(-1), x_prod[-1:]])
    else:
        x_core = x_prod

    # pad: extend beyond each end at the local grid spacing, clamping the low end so that
    # 1+z cannot approach (or cross) zero on a coarse grid
    pad = max(int(DERIVATIVE_FIT_PAD_POINTS), 0)
    if pad > 0 and len(x_core) >= 2:
        # on any sensibly dense grid the padding is pad grid spacings; the two caps only bite on a
        # very coarse or very short grid, and mirror the 5% buffer LambdaCDM_GenericEOS uses
        max_extension = DERIVATIVE_FIT_PAD_FRACTION * (x_core[-1] - x_core[0])
        h_lo = min(
            x_core[1] - x_core[0],
            -log(DERIVATIVE_FIT_PAD_FLOOR) / pad,
            max_extension / pad,
        )
        h_hi = min(x_core[-1] - x_core[-2], max_extension / pad)
        lo = x_core[0] - h_lo * np.arange(pad, 0, -1)
        hi = x_core[-1] + h_hi * np.arange(1, pad + 1)
        fit_x = np.concatenate([lo, x_core, hi])
    else:
        pad = 0
        fit_x = x_core

    fit_select = pad + refine * np.arange(len(x_prod))

    fit_z = np.expm1(fit_x)

    # restore the production redshifts exactly at the points we will select, so that a cosmology
    # supplying analytic derivatives is evaluated at exactly the requested z (no expm1(log1p(z))
    # round trip) and its stored values are unchanged
    fit_x[fit_select] = x_prod
    fit_z[fit_select] = z_prod[ascending]

    return fit_x, fit_z, fit_select, sample_order


class SegmentedSpline:
    """
    One interpolating spline per branch of a cosmology that declares itself non-smooth, dispatching
    on ``u = log(1+z)`` alone.

    This is ``CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.SegmentedEntropyFactor`` applied one
    level further out. Prompt 06 of ``prompts/qcd-background-audit/`` segmented the entropy factor
    ``F(u)`` at the redshifts where ``T(z)`` jumps, and prompt 07 split every cumulative table's
    Gauss panels at the same points; the lattice this module fits its *derivative* splines on was
    the last place in the tree where a smooth interpolant still ran straight across a point the
    cosmology declares it is not smooth at. ``ln H`` genuinely **steps** at two of the three
    crossings ``QCD_Cosmology`` declares, so a quintic through samples either side of one rings:
    2.04e-02 relative in ``epsilon`` at ``T_LO`` and 1.03e-03 at ``T_120_MEV`` against 3.9e-09 away
    from a crossing (``docs/qcd-background-verification.md`` §10.4).

    Dispatch is ``bisect_right`` on ``u``, never on a recovered ``z`` (``CLAUDE.md``; README §2
    (i)): ``log(1+z) -> z`` is irreducibly lossy at large ``z`` and must never appear in an
    equality-like comparison, which a segment-edge test is. ``bisect_right`` places a ``u`` exactly
    equal to an edge in the segment **above** it, which is the branch the bisected edge itself
    belongs to -- the same convention, and for the same reason, as ``SegmentedEntropyFactor``.

    With no edges the callers below build a plain ``BSpline`` instead, so every cosmology that
    declares nothing (``LambdaCDM``, ``RadiationModel``, every test stand-in, any ``GenericEOSBase``
    with a constant ``g_s``) takes a code path that is not merely equivalent to the unsegmented one
    but is *literally* it.
    """

    __slots__ = ("_edges", "_splines")

    def __init__(self, edges: Sequence[float], splines: Sequence):
        if len(splines) != len(edges) + 1:
            raise RuntimeError(
                f"SegmentedSpline: {len(splines)} splines for {len(edges)} interior edges "
                f"(expected {len(edges) + 1})"
            )

        self._edges = [float(u) for u in edges]
        self._splines = list(splines)

    @property
    def segment_edges(self) -> tuple:
        """The interior segment edges, ascending, in ``u = log(1+z)``."""
        return tuple(self._edges)

    @property
    def splines(self) -> tuple:
        """The per-branch splines, in the order :attr:`segment_edges` separates them."""
        return tuple(self._splines)

    def __call__(self, u):
        return self._splines[bisect_right(self._edges, float(u))](u)


def _segment_slices(x: np.ndarray, edges: Sequence[float]) -> List[slice]:
    """
    Partition the ascending array ``x`` into one contiguous slice per branch, the branches
    separated by the interior ``edges`` (also in ``u = log(1+z)``, also ascending).

    ``side="left"`` is what makes this partition agree with :class:`SegmentedSpline`'s
    ``bisect_right`` dispatch: an ``x`` exactly equal to an edge begins the segment above it, so
    the sample that sits *on* a declared crossing is fitted with the branch whose value it carries
    (the bisected edge is the first ``u`` at or above the crossing; see
    ``LambdaCDM_GenericEOS._bisect_temperature_crossing_log1pz``).
    """
    if len(edges) == 0:
        return [slice(0, len(x))]

    cuts = [int(c) for c in np.searchsorted(x, np.asarray(edges, dtype=float), "left")]
    bounds = [0] + cuts + [len(x)]
    return [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _refuse_degenerate_segment(
    site: str, index: int, count: int, lo: float, hi: float, nodes: int, order: int
):
    """
    Refuse a branch that cannot hold an order-``order`` interpolating spline.

    The two alternatives are both silent and both wrong: dropping to a lower order changes the
    accuracy of a *stored* background quantity without saying so, and merging the branch with its
    neighbour puts the discontinuity back inside a single spline, which is the whole of what
    segmentation exists to prevent. ``build_segmented_entropy_spline`` refuses the analogous
    geometry for the same reason.
    """
    raise RuntimeError(
        f"{site}: segment {index + 1} of {count} spans u in [{lo!r}, {hi!r}] and holds "
        f"{nodes} node(s), fewer than the {order + 1} an order-{order} interpolating spline "
        f"needs. The segment edges are the redshifts at which the cosmology declares it is not "
        f"smooth (BackgroundModel._cosmology_break_points); a grid that does not resolve one of "
        f"them cannot carry a background at all, and dropping the order or fitting across the "
        f"break instead would hide that rather than fix it"
    )


def build_stored_sample_spline(
    attr: str,
    x_data,
    y_data,
    min_z: float,
    max_z: float,
    break_points: Sequence[float] = (),
):
    """
    The interpolant ``BackgroundModel._create_functions`` fits through the **stored** samples of a
    background quantity: an order-``STORED_SAMPLE_SPLINE_ORDER`` interpolating spline in
    ``x = log(1+z)``, one per branch of the cosmology, wrapped in the usual ``ZSplineWrapper`` for
    its range check and soft clamp.

    ``x_data`` must be ascending and ``break_points`` the cosmology's declared non-smooth points in
    the same variable (``_cosmology_break_points``), ascending and strictly inside the range. With
    ``break_points`` empty this is the single unsegmented spline this site has always built, built
    by the same call with the same defaulted order.

    It is a module-level function, and not a closure inside ``_create_functions`` where it used to
    live, because ``ComputeTargets/tests/wkb_reference.py`` builds a ``ModelFunctions`` from a
    ``compute_background`` payload and its docstring undertakes to reproduce this site exactly.
    While the two were separate implementations of the same three lines they could -- and did --
    disagree the moment one of them changed: prompt 13's whole measurement is taken through that
    harness, and an unsegmented copy of this function there would have scored the production change
    as a partial failure. One implementation, two callers.
    """
    segments = _segment_slices(np.asarray(x_data, dtype=float), break_points)
    if len(segments) == 1:
        spline = make_interp_spline(x_data, y_data)
    else:
        splines = []
        for i, sl in enumerate(segments):
            if sl.stop - sl.start < STORED_SAMPLE_SPLINE_ORDER + 1:
                _refuse_degenerate_segment(
                    f'build_stored_sample_spline("{attr}")',
                    i,
                    len(segments),
                    x_data[sl.start] if sl.stop > sl.start else None,
                    x_data[sl.stop - 1] if sl.stop > sl.start else None,
                    sl.stop - sl.start,
                    STORED_SAMPLE_SPLINE_ORDER,
                )
            splines.append(
                make_interp_spline(x_data[sl], y_data[sl], k=STORED_SAMPLE_SPLINE_ORDER)
            )
        spline = SegmentedSpline(break_points, splines)

    return ZSplineWrapper(
        spline,
        label=attr,
        min_z=min_z,
        max_z=max_z,
        log_z=True,
    )


ModelFunctions = namedtuple(
    "ModelFunctions",
    [
        "Hubble",
        "epsilon",
        "d_epsilon_dz",
        "d2_epsilon_dz2",
        "wBackground",
        "wPerturbations",
        "tau",
        "T_photon",
        "d_lnH_dz",
        "d2_lnH_dz2",
        "d3_lnH_dz3",
        "d_wPerturbations_dz",
        "d2_wPerturbations_dz2",
        "cs_tau",
        "friction_F",
    ],
    # cs_tau and friction_F are appended with None defaults so that every stand-in that builds a
    # ModelFunctions with the thirteen original fields -- test_tk_source_functions.FakeModel,
    # test_phase_groups, docs/gk-wkb-review-fable-2026-09-09/realbg.py, the audit scripts --
    # keeps constructing unchanged (RECONCILIATION.md §2 item 1, README §5 rule 7).
    defaults=(None, None),
)


def _sound_speed_sq(cosmology, z: float) -> float:
    """
    ``c_s^2 = wPerturbations(z)`` (the author's convention in the transfer-function sector),
    refused if negative: a negative squared sound speed has no sound horizon, and silently
    propagating a NaN through the table would be worse than stopping here.
    """
    cs_sq = cosmology.wPerturbations(z)
    if cs_sq < 0.0:
        raise ValueError(
            f"sound horizon: wPerturbations(z) = {cs_sq:.8g} is negative at z = {z:.8g} for "
            f"cosmology {type(cosmology).__name__} "
            f"(store_id={getattr(cosmology, 'store_id', None)}); c_s^2 < 0 has no sound horizon, "
            "so the cs_tau and friction_F tables cannot be built"
        )
    return cs_sq


def _cs_over_Hubble(cosmology):
    """
    The sound-horizon table's integrand in ``z``: ``f = c_s/H``, so that the table holds
    ``tau_s(z) = int_z^{z_top} c_s dz/H`` (up to the absolute anchor), the leading primitive of
    the transfer-function phase (review §12.2).
    """

    def f(z: float) -> float:
        return sqrt(_sound_speed_sq(cosmology, z)) / cosmology.Hubble(z)

    return f


def _friction_integrand(cosmology):
    """
    The friction table's integrand in ``z``: ``f = -(3/2)(1 + c_s^2)/(1+z)``.

    ``CumulativeTable`` accumulates ``T(z) = int_z^{z_top} f dz``, so ``T' = -f``: the integrand
    handed to it is always *minus* the derivative of the primitive it is to hold (for tau,
    ``dtau/dz = -1/H`` and the integrand is ``+1/H``). The Liouville-Green friction integral is
    the primitive of ``TkWKBIntegration.friction_RHS``, ``dF/dz = +(3/2)(1 + c_s^2)/(1+z)``, so
    ``F`` *decreases* towards lower redshift where ``tau`` and ``cs_tau`` increase; the minus
    sign here is what makes the table hold ``F`` itself rather than ``-F``. With it,
    ``friction_F.delta(z_a, z_b) = F(z_b) - F(z_a)`` in the campaign's sign convention
    (README §2 (c)) and ``friction_F.delta(z_init, z)`` is exactly what the friction ODE
    accumulates from ``F(z_init) = 0``.
    """

    def f(z: float) -> float:
        return -1.5 * (1.0 + _sound_speed_sq(cosmology, z)) / (1.0 + z)

    return f


def _cosmology_break_points(
    cosmology, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
) -> np.ndarray:
    """
    The points in u = log(1+z), strictly inside (log(1+z_lo), log(1+z_hi)), at which the
    cosmology's background quantities lose smoothness of the requested ``kind``, as an ascending
    array; empty if the cosmology declares none. Duck-typed like the analytic-derivative shortcuts
    below: a cosmology that does not implement ``integration_break_points`` (LambdaCDM, the test
    stand-ins) is treated as smooth. LambdaCDM_GenericEOS implements it, and what it declares is
    the redshifts at which T(z) crosses one of its equation of state's branch temperatures -- 3 of
    them on QCD_Cosmology's production range, 2 of which are genuine jumps. See
    prompts/GkTk-remedial/logs/02 for why every Gauss panel has to be split there. Until prompt 07
    of prompts/qcd-background-audit/ it also declared every interior knot of its own T(z)
    tabulation (404 points, and 2,414 once the tabulation was refined); those are an artefact of
    the approximation rather than a feature of the cosmology, and the measurement that a panel
    edge at one of them now buys nothing is in that method's docstring.

    ``kind`` is ``BREAK_POINT_ALL`` (every non-smooth point -- what a fixed-order quadrature panel
    needs, and the default, so that every existing caller is unchanged) or
    ``BREAK_POINT_DISCONTINUITY`` (the subset at which a quantity jumps -- what an adaptive ODE
    solver needs; see Quadrature/integrators/numeric_with_phase_cut.py). The two names are
    re-exported here so that a consumer outside CosmologyModels need not import an
    equation-of-state module to name the kind it wants.
    """
    method = getattr(cosmology, "integration_break_points", None)
    if method is None:
        return np.empty(0, dtype=float)
    return np.asarray(method(z_lo, z_hi, kind=kind), dtype=float)


@ray.remote
def compute_background(
    cosmology: BaseCosmology,
    z_sample: redshift_array,
) -> dict:
    """
    Tabulate the background quantities on ``z_sample``.

    The conformal time tau = a_0 eta (with dtau/dz = -1/H) is *not* integrated as an ODE: it is
    accumulated as a Gauss-Legendre cumulative table on the sample grid itself
    (``ComputeTargets.cumulative_table.CumulativeTable``, order ``TAU_GAUSS_ORDER`` per interval,
    split at the cosmology's break points), held as double-double (hi, lo) pairs so that
    downstream phase differences ``k [tau(z_a) - tau(z_b)]`` over short baselines do not inherit
    the rounding of the absolute tau (review §7, §13.2, §13.3). The absolute normalisation is the
    author's radiation-era closed form ``tau_init = sqrt(3) M_P / sqrt(rho(z_init)) (1 + z_init)``
    at the top of the grid, added to the table in double-double.

    There are no tolerances here, and since prompt 05 of ``prompts/tolerance-convergence`` there
    are none in the lookup key either: the accuracy of all three tables is set by the Gauss orders
    ``TAU_GAUSS_ORDER``, ``CS_TAU_GAUSS_ORDER`` and ``FRICTION_F_GAUSS_ORDER``, and those are what
    the ``BackgroundModel`` row is keyed on. The three module constants read below are the *only*
    declaration of each order, and the payload echoes the three values this call actually used as
    ``tau_order``, ``cs_tau_order`` and ``friction_F_order``. Since prompt 05b those echoes are
    what ``BackgroundModel.store()`` records and what the factory's ``store()`` writes into the
    key, so a row cannot record an order its tables were not built at; ``build()`` filters on the
    module constants, which is the lookup semantics rather than a statement about any object.
    """
    z_nodes = np.array(z_sample.as_float_list(), dtype=float)
    z_init = float(z_nodes[0])
    z_stop = float(z_nodes[-1])

    break_points = _cosmology_break_points(cosmology, z_stop, z_init)

    with IntegrationSupervisor() as supervisor:

        def inverse_Hubble(z: float) -> float:
            with RHS_timer(supervisor):
                return 1.0 / cosmology.Hubble(z)

        table = CumulativeTable(
            z_nodes,
            inverse_Hubble,
            TAU_GAUSS_ORDER,
            break_points=break_points,
            label="tau",
        )

        # the author's radiation-era asymptote for tau at the top of the grid; a convention, kept
        rho_init = cosmology.rho(z_init)
        tau_init = (
            sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) * (1.0 + z_init)
        )
        table = table.shifted(tau_init)

        # the two transfer-function siblings, by the same machinery and on the same grid and
        # break points (review §12.7): the sound horizon tau_s = int c_s dz/H, and the
        # Liouville-Green friction integral F with dF/dz = (3/2)(1 + c_s^2)/(1+z), which replaces
        # TkWKBIntegration's friction ODE (whose relative error was 2.3-4.1e-7, review §12.3).
        # Their integrand evaluations are inside the supervisor's timing window -- compute_time
        # covers all three tables -- but are reported separately from RHS_evaluations; see the
        # comment on the returned IntegrationData.
        cs_tau_table = CumulativeTable(
            z_nodes,
            _cs_over_Hubble(cosmology),
            CS_TAU_GAUSS_ORDER,
            break_points=break_points,
            label="cs_tau",
        )
        # the sibling of tau_init, and a convention in the same way: c_s is constant in the
        # radiation era, so the asymptote that fixes tau at the top of the grid fixes
        # tau_s(z_init) = c_s(z_init) tau_init. Only cs_tau.delta is used downstream.
        cs_tau_init = sqrt(_sound_speed_sq(cosmology, z_init)) * tau_init
        cs_tau_table = cs_tau_table.shifted(cs_tau_init)

        # F is anchored at zero at the top of the grid: it has no closed-form absolute value and
        # enters the transfer function only as exp(F(z) - F(z_i)). A single double suffices --
        # |F| <= 60 over the production range, so 1e-14 absolute in F is 1e-14 relative in the
        # amplitude -- and no low limb is persisted (README §7 D1).
        friction_F_table = CumulativeTable(
            z_nodes,
            _friction_integrand(cosmology),
            FRICTION_F_GAUSS_ORDER,
            break_points=break_points,
            label="friction_F",
        )

    tau_hi_sample = [float(v) for v in table.hi]
    tau_lo_sample = [float(v) for v in table.lo]
    cs_tau_hi_sample = [float(v) for v in cs_tau_table.hi]
    cs_tau_lo_sample = [float(v) for v in cs_tau_table.lo]
    friction_F_sample = [float(v) for v in friction_F_table.hi]

    # each BaseCosmology instance provides methods to evaluate H(z), rho(z), and the value of the equation of state
    # for the background and perturbations
    H_sample = [cosmology.Hubble(z.z) for z in z_sample]
    rho_sample = [cosmology.rho(z.z) for z in z_sample]
    T_photon_sample = [cosmology.T_photon(z.z) for z in z_sample]
    wBackground_sample = [cosmology.wBackground(z.z) for z in z_sample]
    wPerturbations_sample = [cosmology.wPerturbations(z.z) for z in z_sample]

    # further, each BaseCosmology instance may provide methods to evaluate the derivatives of H(z) and w(z), but if it doesn't,
    # we estimate these derivatives using a spline.
    #
    # A spline fitted only on the production grid is badly biased at the two ends: the not-a-knot end
    # condition has no data to constrain it, and each differentiation of a stacked derivative amplifies
    # that error. To avoid this we fit on a *private* grid that is padded beyond both ends of the
    # production grid and refined between its points, evaluate the derivative there, and select the
    # production points only at the end. The padding has to be carried through the whole stack
    # (d2_lnH_dz2 is built from d_lnH_dz, d3_lnH_dz3 from d2_lnH_dz2, ...), so every level of the stack
    # is computed on the padded grid; only the returned samples are truncated.
    # LambdaCDM_GenericEOS._build_T_z_spline already buffers its own grid for exactly this reason.

    fit_x, fit_z, fit_select, sample_order = _build_derivative_fit_grid(z_sample)
    fit_opz = fit_z + 1.0

    # a not-a-knot quintic is used rather than the cubic of the original implementation: with the
    # stacked derivatives the cubic's discontinuous third derivative is the dominant residual error
    # once the end bias has been removed by padding
    fit_k = DERIVATIVE_SPLINE_ORDER if len(fit_x) >= DERIVATIVE_SPLINE_ORDER + 1 else 3

    # ...and it is fitted one spline per branch of the cosmology, not one across the whole grid.
    # ln H genuinely *steps* at two of the three crossings QCD_Cosmology declares, so a single
    # quintic through samples either side of one rings: 2.04e-02 relative in epsilon at T_LO and
    # 1.03e-03 at T_120_MEV against 3.9e-09 away from a crossing, with EOS_T_LO -- where g_s is
    # continuous and only w kinks -- the control at 1.6e-09
    # (docs/qcd-background-verification.md §10.4; prompt 13 of prompts/qcd-background-audit/).
    #
    # The break points are taken over the *padded* range rather than over z_sample's own, so that
    # a crossing lying in the padding is split too; on any cosmology that declares nothing the
    # list is empty, _segment_slices returns the single whole-grid slice, and the spline built
    # below is bit-for-bit the one this code built before segmentation existed.
    #
    # PADDING (prompt 13 §2 item 4). The pad above extends the *outer* ends of the fit grid,
    # because a not-a-knot end has no data beyond it to constrain it. A segment edge is not an
    # outer end and cannot be padded the same way: what lies beyond it is the other branch, whose
    # values are exactly what must not enter this fit, and the cosmology exposes no analytic
    # continuation of one branch past the crossing (its own T(z) representation dispatches on u
    # and would hand back the other branch's value). Each branch therefore keeps the two interior
    # ends it inherits from the cut, unpadded, and the fit grid is not modified. That is a
    # deliberate decision and it was measured, not assumed: with it, the ringing at both genuine
    # steps falls to the away-from-a-crossing regime (log 13 §"Verification performed"), which is
    # the whole of what the padding at the outer ends exists to deliver.
    fit_break_points = _cosmology_break_points(
        cosmology, float(fit_z[0]), float(fit_z[-1])
    )
    fit_segments = _segment_slices(fit_x, fit_break_points)
    for i, sl in enumerate(fit_segments):
        if sl.stop - sl.start < fit_k + 1:
            _refuse_degenerate_segment(
                "compute_background._build_derivative",
                i,
                len(fit_segments),
                float(fit_x[sl.start]) if sl.stop > sl.start else None,
                float(fit_x[sl.stop - 1]) if sl.stop > sl.start else None,
                sl.stop - sl.start,
                fit_k,
            )

    def _build_derivative(attr: str, f_to_diff=None, fit_sample_to_diff=None):
        """
        Evaluate a derivative on the *padded fit grid*, either from the cosmology's own analytic
        method, or by differentiating a spline through the supplied samples/function.
        """
        if f_to_diff is None and fit_sample_to_diff is None:
            raise RuntimeError(
                "compute_background._build_derivative: f_to_diff and fit_sample_to_diff cannot both be None"
            )

        if hasattr(cosmology, attr):
            method = getattr(cosmology, attr)
            return np.array([method(z) for z in fit_z])

        if f_to_diff is not None:
            y_data = np.array([f_to_diff(z) for z in fit_z])
        else:
            y_data = np.asarray(fit_sample_to_diff)

        # one spline per branch; with no declared break points fit_segments is the single
        # whole-grid slice and this is exactly the unsegmented fit it replaces
        d_du = np.empty(len(fit_x), dtype=float)
        for sl in fit_segments:
            deriv = make_interp_spline(fit_x[sl], y_data[sl], k=fit_k).derivative()
            d_du[sl] = np.asarray(deriv(fit_x[sl]))

        # the spline computes d/d(log(1+z)), so divide by 1+z to obtain the raw z-derivative
        return d_du / fit_opz

    def _truncate(fit_values) -> List[float]:
        """Select the production grid points, restoring the ordering of z_sample."""
        return [float(v) for v in np.asarray(fit_values)[fit_select][sample_order]]

    d_lnH_dz_fit = _build_derivative(
        "d_lnH_dz", f_to_diff=lambda z: log(cosmology.Hubble(z))
    )
    d2_lnH_dz2_fit = _build_derivative("d2_lnH_dz2", fit_sample_to_diff=d_lnH_dz_fit)
    d3_lnH_dz3_fit = _build_derivative("d3_lnH_dz3", fit_sample_to_diff=d2_lnH_dz2_fit)
    d_wPerturbations_dz_fit = _build_derivative(
        "d_wPerturbations_dz", f_to_diff=cosmology.wPerturbations
    )
    d2_wPerturbations_dz2_fit = _build_derivative(
        "d2_wPerturbations_dz2", fit_sample_to_diff=d_wPerturbations_dz_fit
    )

    d_lnH_dz_sample = _truncate(d_lnH_dz_fit)
    d2_lnH_dz2_sample = _truncate(d2_lnH_dz2_fit)
    d3_lnH_dz3_sample = _truncate(d3_lnH_dz3_fit)
    d_wPerturbations_dz_sample = _truncate(d_wPerturbations_dz_fit)
    d2_wPerturbations_dz2_sample = _truncate(d2_wPerturbations_dz2_fit)

    return {
        # compute_steps is the node count and RHS_evaluations the number of Hubble evaluations
        # spent building the tau table (one per Gauss abscissa); compute_time covers all three
        # tables. IntegrationData is a fixed namedtuple with one evaluation counter, and prompt
        # 03's test_background_tau pins RHS_evaluations to exactly TAU_GAUSS_ORDER * (nodes - 1)
        # on a break-free cosmology, so the cs_tau and friction_F integrand counts are reported
        # in their own payload keys rather than folded into it
        # ([04-background-rhs-evaluations-count] on the campaign board).
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=len(table),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "tau_hi_sample": tau_hi_sample,
        "tau_lo_sample": tau_lo_sample,
        "tau_order": TAU_GAUSS_ORDER,
        "cs_tau_hi_sample": cs_tau_hi_sample,
        "cs_tau_lo_sample": cs_tau_lo_sample,
        "cs_tau_order": CS_TAU_GAUSS_ORDER,
        "cs_tau_evaluations": cs_tau_table.evaluations,
        "friction_F_sample": friction_F_sample,
        "friction_F_order": FRICTION_F_GAUSS_ORDER,
        "friction_F_evaluations": friction_F_table.evaluations,
        "H_sample": H_sample,
        "rho_sample": rho_sample,
        "T_photon_sample": T_photon_sample,
        "wBackground_sample": wBackground_sample,
        "wPerturbations_sample": wPerturbations_sample,
        "d_lnH_dz_sample": d_lnH_dz_sample,
        "d2_lnH_dz2_sample": d2_lnH_dz2_sample,
        "d3_lnH_dz3_sample": d3_lnH_dz3_sample,
        "d_wPerturbations_dz_sample": d_wPerturbations_dz_sample,
        "d2_wPerturbations_dz2_sample": d2_wPerturbations_dz2_sample,
        "solver_label": TAU_SOLVER_LABEL,
    }


class TablePrimitive:
    """
    A background primitive held as a ``CumulativeTable``: ``functions.tau``, ``functions.cs_tau``
    and ``functions.friction_F``.

    Two accessors, both returning plain floats:

    * ``primitive(z)`` -- the absolute value pointwise, for the existing consumers
      (``compute_analytic_G/T``, the eta limits in ``QuadSourceIntegral``, main.py's Bessel
      ``x_max``). Carries half an ulp of the primitive itself.
    * ``primitive.delta(z_a, z_b) = primitive(z_b) - primitive(z_a)``, the interval accessor the
      WKB phase uses. Formed from the double-double node table and local Gauss partials, never
      as a difference of two pointwise values (README §2 (c); ``CumulativeTable.delta``).
      For tau this is ``int_{z_b}^{z_a} dz/H``, positive when ``z_b < z_a``.
    """

    def __init__(self, table: CumulativeTable, label: str):
        self._table = table
        self._label = label

    def __call__(self, z: float) -> float:
        return self._table.value(float(z))

    def delta(self, z_a: float, z_b: float) -> float:
        return self._table.delta(float(z_a), float(z_b))

    @property
    def table(self) -> CumulativeTable:
        return self._table

    @property
    def label(self) -> str:
        return self._label


class BackgroundModel(DatastoreObject):
    """
    Encapsulates the time history of a cosmological model.
    This bakes-in all the quantities we need such as the conformal time \tau (for the WKB phases,
    and for analytic approximations to the transfer functions and Green's functions). \tau is
    tabulated on the sample grid as a double-double Gauss-Legendre cumulative table rather than
    integrated as an ODE, and ``functions.tau`` exposes it both pointwise, ``tau(z)``, and as an
    interval, ``tau.delta(z_a, z_b)`` (see ``TablePrimitive``). Two siblings are tabulated the
    same way for the transfer-function sector: ``functions.cs_tau``, the sound horizon
    ``int c_s dz/H``, and ``functions.friction_F``, the Liouville-Green friction integral
    ``F`` with ``dF/dz = (3/2)(1 + c_s^2)/(1+z)`` (review §12.2, §12.7).
    It also means we have an explicit record in the database of the values of H(z), w(z), etc.,
    that yielded a particular set of results
    """

    # the solver label compute_background reports, and the registration main.py must make
    TAU_GAUSS_ORDER = TAU_GAUSS_ORDER
    TAU_SOLVER_LABEL_BASE = TAU_SOLVER_LABEL_BASE
    TAU_SOLVER_LABEL = TAU_SOLVER_LABEL

    def __init__(
        self,
        payload,
        solver_labels: dict,
        cosmology: BaseCosmology,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._solver_labels = solver_labels
        self._z_sample = z_sample

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None
            self._solver = None
            self._values = None

            # nothing has been tabulated and no row is behind us, so there are no orders to
            # report yet; store() fills them from the compute_background payload
            self._tau_gauss_order = None
            self._cs_tau_gauss_order = None
            self._friction_F_gauss_order = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._data: Optional[IntegrationData] = payload["data"]
            self._solver: Optional[IntegrationSolver] = payload["solver"]
            self._values: Optional[List[BackgroundModelValue]] = payload["values"]

            # the three orders the row records, which are the orders this model's three tables
            # were tabulated at. _build_*_primitive reassembles the tables at these and not at
            # the module constants, so an off-grid partial evaluated against a rehydrated table
            # uses the rule its nodes were integrated with
            # (prompts/tolerance-convergence, prompt 05b)
            self._tau_gauss_order = payload["tau_gauss_order"]
            self._cs_tau_gauss_order = payload["cs_tau_gauss_order"]
            self._friction_F_gauss_order = payload["friction_F_gauss_order"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._cosmology = cosmology
        self._units = cosmology.units

        self._functions = None

        self._compute_ref = None

    # The three Gauss orders that set this model's accuracy, and -- since prompt 05 of
    # prompts/tolerance-convergence -- its datastore lookup key.
    #
    # Prompt 05 made each accessor re-read its module constant at call time, which is right about
    # the key and wrong about the object: it reports what the module currently says, not what
    # this model's tables were built with. Since prompt 05b each reports the order recorded on
    # the path this object came into existence by -- the compute_background payload on the
    # compute path (store()), the row's own column on the rehydration path (__init__ above) --
    # and _build_*_primitive reassembles the tables at those same orders.
    # sqla_BackgroundModelFactory.build() still *filters* on the module constants, which is the
    # lookup semantics: a model tabulated at another order is a different row (README §7 D10).
    def _order(self, value: Optional[int], name: str) -> int:
        if value is None:
            raise RuntimeError(
                f"BackgroundModel: {name} was read before this object had one. It is the Gauss "
                "order one of the three cumulative tables was built at, so it exists only once "
                "the model has been computed (store()) or rehydrated from a row."
            )

        return value

    @property
    def tau_gauss_order(self) -> int:
        return self._order(self._tau_gauss_order, "tau_gauss_order")

    @property
    def cs_tau_gauss_order(self) -> int:
        return self._order(self._cs_tau_gauss_order, "cs_tau_gauss_order")

    @property
    def friction_F_gauss_order(self) -> int:
        return self._order(self._friction_F_gauss_order, "friction_F_gauss_order")

    @property
    def cosmology(self):
        return self._cosmology

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_sample(self):
        return self._z_sample

    def efolds_subh(self, k: wavenumber, z: Union[redshift, float]) -> float:
        if isinstance(z, redshift):
            z_float = z.z
        else:
            z_float = float(z)

        H = self.functions.Hubble(z_float)
        return log((1.0 + z_float) * k.k / H)

    @property
    def data(self) -> IntegrationData:
        if self.values is None:
            raise RuntimeError("values have not yet been populated")

        return self._data

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    @property
    def functions(self) -> ModelFunctions:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        # the second of the two sites at which this module fits a smooth interpolant across the
        # sample grid, and it carries the same defect as compute_background's derivative fit: the
        # *stored* samples of any quantity the cosmology does not supply as a method are splined
        # here, and at a declared crossing they step. Fixing compute_background alone would make
        # the stored d_lnH_dz right and then ring a cubic through it, so both are segmented
        # (prompt 13 §2 item 2 of prompts/qcd-background-audit/; the separate contribution of each
        # site is in log 13). Empty on any cosmology that declares nothing, in which case the
        # spline built is literally the one this code built before.
        stored_break_points = _cosmology_break_points(
            self._cosmology, self.z_sample.min.z, self.z_sample.max.z
        )

        def _build_func(attr: str):
            if hasattr(self._cosmology, attr):
                return getattr(self._cosmology, attr)

            data = [(log(1.0 + v.z.z), getattr(v, attr)) for v in self.values]
            data.sort(key=lambda pair: pair[0])

            x_data, y_data = zip(*data)
            return build_stored_sample_spline(
                attr,
                x_data,
                y_data,
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                break_points=stored_break_points,
            )

        # tau is reconstructed from the persisted (hi, lo) limbs with no quadrature; the integrand
        # is needed only for off-grid partials (the per-object anchor z_init of a numeric hand-over,
        # RECONCILIATION.md §2 item 5). No cosmology supplies an analytic tau, and a pointwise
        # analytic tau could not supply delta at the required accuracy, so there is no
        # hasattr(cosmology, "tau") shortcut here (RECONCILIATION.md §2 item 2). A cubic spline of
        # the nodes -- the previous accessor -- is 1.4e-9 relative off-grid, ~2 rad of phase at
        # k = 1e5/Mpc (review §7, §13.2).
        tau_func = self._build_tau_primitive()

        # the two transfer-function primitives, the same object of the same kind: the sound
        # horizon tau_s (double-double, like tau: k Delta tau_s reaches 1.85e11 rad) and the
        # Liouville-Green friction integral F (a single limb; see _build_friction_F_primitive)
        cs_tau_func = self._build_cs_tau_primitive()
        friction_F_func = self._build_friction_F_primitive()

        T_photon_func = _build_func("T_photon")
        d_lnH_dz_func = _build_func("d_lnH_dz")
        d2_lnH_dz2_func = _build_func("d2_lnH_dz2")
        d3_lnH_dz3_func = _build_func("d3_lnH_dz3")
        d_wPerturbations_dz_func = _build_func("d_wPerturbations_dz")
        d2_wPerturbations_dz2_func = _build_func("d2_wPerturbations_dz2")

        def epsilon(z: float) -> float:
            """
            Evaluate the conventional epsilon parameter eps = -dot(H)/H^2
            :param z: redshift of evaluation
            :return:
            """
            one_plus_z = 1.0 + z
            return one_plus_z * d_lnH_dz_func(z)

        def d_epsilon_dz(z: float) -> float:
            """
            Evaluate the z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return d_lnH_dz_func(z) + one_plus_z * d2_lnH_dz2_func(z)

        def d2_epsilon_dz2(z: float) -> float:
            """
            Evaluate the 2nd z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return 2.0 * d2_lnH_dz2_func(z) + one_plus_z * d3_lnH_dz3_func(z)

        # the underlying cosmology object is guaranteed to provide methods for Hubble, rho, wBackground, and wPerturbations.
        # it may or may not provide methods for other quantities.
        # The functions built above will pass through to the underlying cosmology object when it can perform the computation
        # (so we can make use of e.g. analytic expressions where they are available), but otherwise we use splines
        # constructed by differentiation (usually of another spline). Differentiating a spline gives us much less noisy
        # results than finite difference formulae.
        self._functions = ModelFunctions(
            Hubble=self._cosmology.Hubble,
            epsilon=epsilon,
            d_epsilon_dz=d_epsilon_dz,
            d2_epsilon_dz2=d2_epsilon_dz2,
            wBackground=self._cosmology.wBackground,
            wPerturbations=self._cosmology.wPerturbations,
            tau=tau_func,
            T_photon=T_photon_func,
            d_lnH_dz=d_lnH_dz_func,
            d2_lnH_dz2=d2_lnH_dz2_func,
            d3_lnH_dz3=d3_lnH_dz3_func,
            d_wPerturbations_dz=d_wPerturbations_dz_func,
            d2_wPerturbations_dz2=d2_wPerturbations_dz2_func,
            cs_tau=cs_tau_func,
            friction_F=friction_F_func,
        )

    def _build_tau_primitive(self) -> TablePrimitive:
        values = sorted(self.values, key=lambda v: v.z.z, reverse=True)
        z_nodes = [v.z.z for v in values]
        cosmology = self._cosmology

        def inverse_Hubble(z: float) -> float:
            return 1.0 / cosmology.Hubble(z)

        table = CumulativeTable(
            z_nodes,
            inverse_Hubble,
            self.tau_gauss_order,
            hi=[v.tau for v in values],
            lo=[v.tau_lo for v in values],
            break_points=_cosmology_break_points(cosmology, z_nodes[-1], z_nodes[0]),
            label="tau",
        )
        return TablePrimitive(table, label="tau")

    def _persisted_limbs(self, values, attr: str, label: str) -> List[float]:
        """The persisted limb ``attr`` of every value, in node order, refusing a missing one."""
        limbs = [getattr(v, attr) for v in values]
        if any(limb is None for limb in limbs):
            raise RuntimeError(
                f'BackgroundModel: the "{label}" table cannot be rebuilt because at least one '
                f"BackgroundModelValue carries no {attr}. Values written before "
                "prompts/GkTk-remedial prompt 04 have none, and there is no migration: the "
                "datastore must be regenerated."
            )
        return [float(limb) for limb in limbs]

    def _build_cs_tau_primitive(self) -> TablePrimitive:
        """
        The sound-horizon primitive ``tau_s = int c_s dz/H`` (review §12.2), reconstructed from
        the persisted (hi, lo) limbs with no quadrature; the integrand is needed only for
        off-grid partials. Double-double for the same reason as ``tau``: ``k Delta tau_s``
        reaches 1.85e11 rad at k = 3e8/Mpc (review §12.2), so a difference of two correctly
        rounded pointwise values would carry ~4e-5 rad however short the baseline.
        """
        values = sorted(self.values, key=lambda v: v.z.z, reverse=True)
        z_nodes = [v.z.z for v in values]
        cosmology = self._cosmology

        table = CumulativeTable(
            z_nodes,
            _cs_over_Hubble(cosmology),
            self.cs_tau_gauss_order,
            hi=self._persisted_limbs(values, "cs_tau", "cs_tau"),
            lo=self._persisted_limbs(values, "cs_tau_lo", "cs_tau"),
            break_points=_cosmology_break_points(cosmology, z_nodes[-1], z_nodes[0]),
            label="cs_tau",
        )
        return TablePrimitive(table, label="cs_tau")

    def _build_friction_F_primitive(self) -> TablePrimitive:
        """
        The Liouville-Green friction integral ``F``, ``dF/dz = (3/2)(1 + c_s^2)/(1+z)``, anchored
        at zero at the top of the grid and reconstructed from a **single** persisted limb (the
        low limbs are zeros). One double suffices: ``|F| <= 60`` over the production range and
        ``F`` enters the transfer function only through ``exp(F(z) - F(z_i))``, so 1e-14
        absolute in ``F`` is 1e-14 relative in the amplitude -- four orders below the 2.3-4.1e-7
        the friction ODE it replaces carried (review §12.3).

        ``friction_F.delta(z_a, z_b) = F(z_b) - F(z_a)``, so ``friction_F.delta(z_init, z)`` is
        exactly the quantity ``TkWKBIntegration`` accumulates today from ``F(z_init) = 0``.
        """
        values = sorted(self.values, key=lambda v: v.z.z, reverse=True)
        z_nodes = [v.z.z for v in values]
        cosmology = self._cosmology

        hi = self._persisted_limbs(values, "friction_F", "friction_F")
        table = CumulativeTable(
            z_nodes,
            _friction_integrand(cosmology),
            self.friction_F_gauss_order,
            hi=hi,
            lo=[0.0] * len(hi),
            break_points=_cosmology_break_points(cosmology, z_nodes[-1], z_nodes[0]),
            label="friction_F",
        )
        return TablePrimitive(table, label="friction_F")

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values has not yet been populated")

        if self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        self._compute_ref = compute_background.remote(
            self.cosmology,
            self._z_sample,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkWKBIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._data = data["data"]
        self._values = self.values_from_payload(self._z_sample, data)
        self._solver = self._solver_labels[data["solver_label"]]

        # the orders the three tables were actually built at, as compute_background reports them.
        # This is what makes the record faithful: the factory's store() writes these into the key
        # columns, so a row cannot claim an order its tables were not built at
        self._tau_gauss_order = int(data["tau_order"])
        self._cs_tau_gauss_order = int(data["cs_tau_order"])
        self._friction_F_gauss_order = int(data["friction_F_order"])

        return True

    @staticmethod
    def values_from_payload(
        z_sample: redshift_array, data: dict
    ) -> List["BackgroundModelValue"]:
        """
        Build the per-redshift ``BackgroundModelValue`` list from a ``compute_background``
        payload, in ``z_sample`` order. Shared by ``store()`` and the offline tests.
        """
        H_sample = data["H_sample"]
        wB_sample = data["wBackground_sample"]
        wP_sample = data["wPerturbations_sample"]
        rho_sample = data["rho_sample"]
        T_photon_sample = data["T_photon_sample"]
        tau_hi_sample = data["tau_hi_sample"]
        tau_lo_sample = data["tau_lo_sample"]
        cs_tau_hi_sample = data["cs_tau_hi_sample"]
        cs_tau_lo_sample = data["cs_tau_lo_sample"]
        friction_F_sample = data["friction_F_sample"]

        d_lnH_ds_sample = data["d_lnH_dz_sample"]
        d2_lnH_dz2_sample = data["d2_lnH_dz2_sample"]
        d3_lnH_dz3_sample = data["d3_lnH_dz3_sample"]

        d_wPerturbations_dz_sample = data["d_wPerturbations_dz_sample"]
        d2_wPerturbations_dz2_sample = data["d2_wPerturbations_dz2_sample"]

        values = []
        for i in range(len(H_sample)):
            values.append(
                BackgroundModelValue(
                    None,
                    z_sample[i],
                    Hubble=H_sample[i],
                    wBackground=wB_sample[i],
                    wPerturbations=wP_sample[i],
                    rho=rho_sample[i],
                    tau=tau_hi_sample[i],
                    T_photon=T_photon_sample[i],
                    d_lnH_dz=d_lnH_ds_sample[i],
                    d2_lnH_dz2=d2_lnH_dz2_sample[i],
                    d3_lnH_dz3=d3_lnH_dz3_sample[i],
                    d_wPerturbations_dz=d_wPerturbations_dz_sample[i],
                    d2_wPerturbations_dz2=d2_wPerturbations_dz2_sample[i],
                    tau_lo=tau_lo_sample[i],
                    cs_tau=cs_tau_hi_sample[i],
                    cs_tau_lo=cs_tau_lo_sample[i],
                    friction_F=friction_F_sample[i],
                )
            )
        return values


class BackgroundModelValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        Hubble: float,
        wBackground: float,
        wPerturbations: float,
        rho: float,
        tau: float,
        T_photon: float,
        d_lnH_dz: float,
        d2_lnH_dz2: Optional[float] = None,
        d3_lnH_dz3: Optional[float] = None,
        d_wPerturbations_dz: Optional[float] = None,
        d2_wPerturbations_dz2: Optional[float] = None,
        tau_lo: float = 0.0,
        cs_tau: Optional[float] = None,
        cs_tau_lo: Optional[float] = None,
        friction_F: Optional[float] = None,
    ):
        """
        ``tau`` is the high limb and ``tau_lo`` the low limb of the double-double conformal time
        at this redshift (``tau + tau_lo`` is the value to ~1e-32 relative). ``tau_lo`` is a
        keyword with a zero default so that stand-ins and the factory's ``build()`` path keep
        constructing.

        ``cs_tau`` and ``cs_tau_lo`` are the two limbs of the sound horizon
        ``tau_s = int c_s dz/H``, and ``friction_F`` the single double holding the Liouville-Green
        friction integral ``F`` (``|F| <= 60``, so one limb is enough; see
        ``BackgroundModel._build_friction_F_primitive``). All three are keywords with ``None``
        defaults for the same reason as ``tau_lo``; a value carrying ``None`` cannot take part in
        a rebuilt table and the rebuild says so.
        """
        DatastoreObject.__init__(self, store_id)

        self._z = z

        self._Hubble: float = Hubble
        self._wBackground: float = wBackground
        self._wPerturbations: float = wPerturbations

        self._rho: float = rho
        self._tau: float = tau
        self._tau_lo: float = tau_lo
        self._cs_tau: Optional[float] = cs_tau
        self._cs_tau_lo: Optional[float] = cs_tau_lo
        self._friction_F: Optional[float] = friction_F
        self._T_photon: float = T_photon

        self._d_lnH_dz: float = d_lnH_dz
        self._d2_lnH_dz2: float = d2_lnH_dz2
        self._d3_lnH_dz3: float = d3_lnH_dz3

        self._d_wPerturbations_dz: float = d_wPerturbations_dz
        self._d2_wPerturbations_dz2: float = d2_wPerturbations_dz2

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def Hubble(self) -> float:
        return self._Hubble

    @property
    def wBackground(self) -> float:
        return self._wBackground

    @property
    def wPerturbations(self) -> float:
        return self._wPerturbations

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def tau(self) -> float:
        """The high limb of the double-double conformal time at this redshift."""
        return self._tau

    @property
    def tau_lo(self) -> float:
        """The low limb of the double-double conformal time at this redshift."""
        return self._tau_lo

    @property
    def cs_tau(self) -> Optional[float]:
        """The high limb of the double-double sound horizon at this redshift."""
        return self._cs_tau

    @property
    def cs_tau_lo(self) -> Optional[float]:
        """The low limb of the double-double sound horizon at this redshift."""
        return self._cs_tau_lo

    @property
    def friction_F(self) -> Optional[float]:
        """
        The Liouville-Green friction integral at this redshift, anchored at zero at the top of
        the grid (dimensionless, so it is persisted without a unit conversion).
        """
        return self._friction_F

    @property
    def T_photon(self) -> float:
        return self._T_photon

    @property
    def d_lnH_dz(self) -> float:
        return self._d_lnH_dz

    @property
    def d2_lnH_dz2(self) -> Optional[float]:
        return self._d2_lnH_dz2

    @property
    def d3_lnH_dz3(self) -> Optional[float]:
        return self._d3_lnH_dz3

    @property
    def d_wPerturbations_dz(self) -> Optional[float]:
        return self._d_wPerturbations_dz

    @property
    def d2_wPerturbations_dz2(self) -> Optional[float]:
        return self._d2_wPerturbations_dz2


class ModelProxy:
    def __init__(self, model: BackgroundModel):
        self._ref: ObjectRef = ray.put(model)

        self._store_id: int = model.store_id if model.available else None

        self._units: UnitsLike = model.cosmology.units
        self._cosmology: BaseCosmology = model.cosmology

    @property
    def store_id(self) -> int:
        return self._store_id

    @property
    def available(self) -> bool:
        return self._store_id is not None

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def cosmology(self) -> BaseCosmology:
        return self._cosmology

    def get(self) -> BackgroundModel:
        """
        The return value should only be held locally and not persisted, otherwise the entire
        BackgroundModel instance may be serialized when it is passed around by Ray.
        That would defeat the purpose of the proxy.
        :return:
        """
        return ray.get(self._ref)
