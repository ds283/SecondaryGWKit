from bisect import bisect_right
from math import exp, sqrt, log, log1p, expm1
from typing import Callable, List, Mapping, Optional, Sequence

import numpy as np
from numpy import linspace
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

from ComputeTargets.spline_wrappers import _outward
from CosmologyModels import BaseCosmology
from CosmologyModels.GenericEOS.GenericEOS import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
    BREAK_POINT_KINDS,
    GenericEOSBase,
    HIGH_T_GSTAR,
)
from Units.base import UnitsLike
from constants import RadiationConstant

DEFAULT_MAX_TEMPERATURE_Z_REDSHIFT = 1e20
# The T(z) tabulation extends a little into the future so that numerical derivatives at z = 0 are
# accurate; see the comment where the spline is built. Four steps of 0.05 is where this value came
# from, but nothing reads a step size any more.
DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2

# Total number of tabulation nodes, and the order of the spline through them, used by
# _build_T_z_spline. The nodes are shared out between the segments in proportion to each
# segment's width in u, so this is a total and not a per-segment count. Measured on the audit's
# 640-point probe set (CosmologyModels/tests/T_z_reference.py, probe_set()) against the defining
# equation root-solved to rtol = 1e-14, with the entropy factor as the splined quantity:
#
#   nodes  order  segmented     max        p90      median    build   knots  in range
#   -----  -----  ---------  ---------  ---------  ---------  ------  -----  --------
#     500    3       no      7.236e-04  8.912e-08  2.599e-10  12.4ms    498      404
#     500    5       no      7.226e-04  2.022e-08  8.099e-13  11.9ms    496
#    2000    3       no      5.066e-05  2.508e-10  1.476e-13  50.5ms   1998
#    2000    5       no      5.580e-05  9.002e-14  2.320e-16  48.7ms   1996
#    3000    5       no      3.730e-04  7.062e-15  1.943e-16  71.9ms   2996
#     500    3      yes      1.013e-05  6.544e-08  9.018e-12  18.8ms    492      398
#    2000    5      yes      1.292e-09  4.481e-14  2.374e-16  61.4ms   1984     1604
#    3000    5      yes      6.807e-11  3.237e-15  1.765e-16  88.2ms   2983     2411
#
# (the first five rows are prompt 05's, the last three prompt 06's, measured the same way on the
# same probe set. "knots" is the tabulation's unique knots, "in range" those of them that fall
# inside the production source grid. Until prompt 07 the "in range" column was also the number of
# break points the cosmology declared; it is not any more, and that is the point of the paragraph
# below.)
#
# 3000 / k = 5 is the audit's recommendation (docs/qcd-background-audit-2026-09.md §4,
# prompts/qcd-background-audit/ README §7 D3) and is what is shipped. Unsegmented, the maximum
# error is pinned near the height of the jump in T(z) at every node count, because one spline is
# being run straight across a genuine discontinuity; segmented, it falls by seven orders and the
# node count buys the p90 and the median in the ordinary way. The cheaper 2000 / k = 5 was
# measured too, and it is not enough: it reaches the same median and the same (bit-identical)
# conformal time, but sits an order above on the p90 and four orders above on the max, missing
# two of the four accuracy rows README §6.1 sets for this representation. Order 5 rather than 3
# is what buys the p90: a cubic would need ~25,000 nodes to reach 1e-14 there, and every node is
# a declared break point.
#
# The nodes used to be paid for in declared break points: every interior knot was returned by
# integration_break_points below, so every quadrature and every ODE in the tree split a panel at
# each of them, and raising this constant from 500 to 3,000 took the declared set from 404 points
# on the production source grid to 2,414. **Prompt 07 of prompts/qcd-background-audit/ removed
# the knots from that set** (finding G1 of the audit), on the measurement recorded at
# integration_break_points: at k = 5 the first discontinuous derivative of F is the fifth, three
# levels below the deepest derivative anything in the tree builds, and the observable residual
# across a knot is 6.3e-12 in H against the 2.1e-04 the old 500-node k = 3 lattice carried. The
# node count is therefore now free of the break-point set, and this constant can be raised on
# accuracy alone.
DEFAULT_T_Z_SPLINE_SAMPLES = 3000
DEFAULT_T_Z_SPLINE_ORDER = 5

# How far inside its own branch each segment's node set is held, in u = log(1+z). The segment
# edges are the redshifts at which T(z) *jumps*, so a node placed on the far side of one is a
# node from the wrong branch and the discontinuity is interpolated across after all -- which is
# the whole of what segmentation exists to prevent. 1e-12 is the audit's value
# (docs/qcd-background-audit/measure_T_z_representation.py, build_segmented); at the production
# edges, u ~ 17.6 to 27.5, it is a few hundred ulp, and the spline it excludes is extrapolated
# over that distance instead, which costs ~1e-17 relative on an F whose slope is O(1e-5) per unit
# u. CosmologyModels/tests/test_T_z_representation.py's SEGMENT_PAD is the same number.
SEGMENT_EDGE_PAD_LOG1PZ = 1.0e-12


class SegmentedEntropyFactor:
    """
    The entropy factor ``F(u)``, interpolated **one spline per branch** of the equation of state,
    with the segment edges placed exactly at the redshifts where ``T(z)`` jumps.

    ``T(z)`` is genuinely discontinuous at those redshifts, not merely kinked
    (``docs/qcd-background-audit-2026-09.md`` §2): ``g_s`` steps across an equation-of-state branch
    join, so ``F = -(1/3) log(g_s(T)/g_s(T_CMB))`` steps with it -- by 7.6229229003969e-04 at the
    lowest crossing, between two values each flat to twelve decimals. No single smooth approximant
    represents a step at any node count, which is why the maximum error of the unsegmented
    representation was pinned near the jump height at 500, 2,000 and 3,000 nodes alike. One spline
    per branch reproduces it exactly.

    Callable on ``u = log(1+z)``, dispatching on ``u`` alone -- never on a recovered ``z``, which
    is irreducibly lossy at large ``z`` (``CLAUDE.md``, README §2 (i)). ``bisect_right`` places an
    ``u`` exactly equal to an edge in the segment **above** it, which is the branch the bisected
    edge itself belongs to: the last representable ``u`` below the crossing is the cold branch's,
    the edge is the hot branch's, and the two are one ulp apart.

    ``t`` is the concatenation of the segments' knot vectors, so that this object stands in for a
    single ``BSpline`` wherever one is read for its knots -- ``_build_T_z_spline``, which records
    them as ``_T_z_spline_knots_log1pz``, and ``docs/gktk-remedial/residual_convergence.py``,
    which reaches into ``cosmology._T_z_spline._spline.t``. Callers take ``np.unique`` of it.
    Since prompt 07 those knots are *not* what ``integration_break_points`` declares; they are
    read only to measure that they need not be.
    """

    __slots__ = ("_edges", "_splines", "t")

    def __init__(self, edges: Sequence[float], splines: Sequence):
        if len(splines) != len(edges) + 1:
            raise RuntimeError(
                f"SegmentedEntropyFactor: {len(splines)} splines for {len(edges)} interior "
                f"edges (expected {len(edges) + 1})"
            )

        self._edges = list(edges)
        self._splines = list(splines)
        self.t = np.concatenate([np.asarray(s.t, dtype=float) for s in splines])

    @property
    def segment_edges(self) -> tuple:
        """The interior segment edges, ascending, in ``u = log(1+z)``."""
        return tuple(self._edges)

    @property
    def splines(self) -> tuple:
        """
        The per-branch splines, in the same order as :attr:`segment_edges` separates them: the
        spline for ``u`` is ``splines[bisect_right(segment_edges, u)]``.

        Public because ``TemperatureRepresentation.__call__`` does that indexing in line rather
        than calling this object, to save a Python-level call on the hot path.
        """
        return tuple(self._splines)

    def __call__(self, u: float):
        return self._splines[bisect_right(self._edges, u)](u)


def build_segmented_entropy_spline(
    F_of_u: Callable[[float], float],
    edges: Sequence[float],
    u_lo: float,
    u_hi: float,
    samples: int,
    order: int,
    pad: float = SEGMENT_EDGE_PAD_LOG1PZ,
):
    """
    Tabulate ``F_of_u`` over ``[u_lo, u_hi]`` as one interpolating spline per segment, the segments
    separated by ``edges``.

    ``samples`` is a **total**: each segment receives a share in proportion to its width in ``u``,
    and never fewer than ``order + 1`` nodes, which is the fewest an order-``order`` interpolating
    spline can be built through. Each segment's nodes are held ``pad`` inside its own edges so that
    no segment interpolates across a discontinuity.

    With ``edges`` empty this returns the plain ``BSpline`` over the whole range -- the
    unsegmented representation, unchanged, which is what every cosmology that declares no break
    temperatures gets (``LambdaCDM``, ``RadiationModel``, the test stand-ins, and any
    ``GenericEOSBase`` with a constant ``g_s``).

    Degenerate geometry raises rather than quietly producing a lower-order or cross-branch fit:
    that failure is silent and expensive, and it is the one this whole design exists to prevent.

    :param F_of_u: the quantity to tabulate, evaluated at ``u = log(1+z)``
    :param edges: interior segment edges in ``u``, strictly ascending and strictly inside the range
    :param u_lo: lower end of the tabulated range in ``u``
    :param u_hi: upper end of the tabulated range in ``u``
    :param samples: total number of nodes across all segments
    :param order: spline order ``k``
    :param pad: how far inside its own edges each segment's node set is held, in ``u``
    :return: a ``BSpline`` if ``edges`` is empty, otherwise a :class:`SegmentedEntropyFactor`
    """
    if not u_lo < u_hi:
        raise RuntimeError(
            f"build_segmented_entropy_spline: empty tabulation range "
            f"u in [{u_lo!r}, {u_hi!r}]"
        )

    edges = [float(u) for u in edges]
    for previous, current in zip([u_lo] + edges, edges + [u_hi]):
        if not previous < current:
            raise RuntimeError(
                f"build_segmented_entropy_spline: segment edges {edges} are not strictly "
                f"ascending and strictly inside the tabulated range "
                f"u in [{u_lo!r}, {u_hi!r}]"
            )

    bounds = [u_lo] + edges + [u_hi]
    widths = np.diff(np.asarray(bounds, dtype=float))
    fractions = widths / widths.sum()

    splines = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]

        # The outermost ends of the tabulation are the range itself, and are used as they are.
        # Every interior end is a jump in the quantity being tabulated, and the node set is held
        # `pad` inside it: a node taken from the far side of a jump belongs to the other branch,
        # and one such node puts the discontinuity back inside a single spline.
        lo_in = lo + pad if i > 0 else lo
        hi_in = hi - pad if i + 2 < len(bounds) else hi

        nodes = max(order + 1, int(round(samples * fractions[i])))
        if not lo_in < hi_in:
            raise RuntimeError(
                f"build_segmented_entropy_spline: segment {i + 1} of {len(bounds) - 1} spans "
                f"u in [{lo!r}, {hi!r}], which is narrower than twice the {pad:.1e} padding "
                f"that keeps its nodes inside their own branch"
            )

        us = linspace(lo_in, hi_in, nodes)
        if not np.all(np.diff(us) > 0.0):
            raise RuntimeError(
                f"build_segmented_entropy_spline: segment {i + 1} of {len(bounds) - 1} spans "
                f"u in [{lo!r}, {hi!r}] and cannot hold {nodes} distinct nodes -- its node "
                f"spacing is below the representable resolution of u there"
            )

        splines.append(make_interp_spline(us, [F_of_u(float(u)) for u in us], k=order))

    if len(edges) == 0:
        return splines[0]

    return SegmentedEntropyFactor(edges, splines)


class TemperatureRepresentation:
    """
    ``T(z)`` for a ``LambdaCDM_GenericEOS``: the closed-form ``(1+z)`` ramp times the exponential
    of an interpolated *entropy factor*.

    Entropy conservation gives ``T g_s(T)^(1/3) = T_CMB g_s(T_CMB)^(1/3) (1+z)``, so

        T(z) = T_CMB (1+z) exp F(u),    F(u) = log( T / [T_CMB (1+z)] )
                                             = -(1/3) log( g_s(T) / g_s(T_CMB) ),

    with ``u = log(1+z)``. Only ``F`` is interpolated. It is bounded, ``O(1)``, and *exactly
    constant* wherever ``g_s`` is -- which is most of the range -- whereas ``T`` itself is
    dominated over twenty decades by the ramp, which is known in closed form. Splining ``T``
    against ``u`` spends the interpolant's degrees of freedom re-deriving something exact; at 500
    nodes that costs a factor of 400 in the median error (1.071e-07 against 2.599e-10). This is
    finding T3 of ``docs/qcd-background-audit-2026-09.md`` §3.

    ``F`` is interpolated **per branch** of the equation of state, by
    :class:`SegmentedEntropyFactor`, because ``T(z)`` genuinely jumps where ``g_s`` does (audit §2,
    finding T4). The segment edges are located by *bisecting the monotone* ``T(z)``, never by
    root-finding on ``T(z) - T_break``: at a discontinuity that difference has no root, and a
    bracketing solver reports convergence onto a point beside the jump whose offset depends on its
    tolerances. A cosmology that declares no break temperatures gets a single ``BSpline`` over the
    whole range and is numerically unchanged from prompt 05's representation.

    The range logic -- the hard rejection outside ``SPLINE_BOUND_SLACK`` of the tabulated range
    and the soft clamp inside it -- is ``ZSplineWrapper``'s, reproduced here rather than delegated
    to because the ramp has to be clamped along with the interpolant: a wrapper that clamped only
    ``F`` would let the ``(1+z)`` factor run away above ``max_z``. ``_outward`` is imported rather
    than re-derived, so there is one definition of "outward" in the repository, and the
    ``RuntimeError`` text is kept character-for-character identical to ``ZSplineWrapper``'s so that
    nothing that reads it changes. The prefix was literally ``GkSource.function:`` at all six sites
    across the three classes until 2026-09-16, which meant every out-of-bounds ``T(z)`` reported
    itself as a ``GkSource`` failure; it is now ``type(self).__name__``, so the three classes still
    share one expression while each names itself. Nothing parses the message -- there is no
    ``assertRaisesRegex`` against it anywhere in the tree.
    """

    def __init__(
        self,
        spline,
        T_CMB: float,
        label: str,
        min_z: float,
        max_z: float,
    ):
        # F(u): a SciPy BSpline where the cosmology declares no break temperatures, and a
        # SegmentedEntropyFactor -- one BSpline per branch, with the same call signature and the
        # same `t` -- where it does. Deliberately named `_spline`, as in ZSplineWrapper:
        # docs/gktk-remedial/residual_convergence.py reads `cosmology._T_z_spline._spline.t` to
        # recover the tabulation's knots, and that script is the generator of the reference
        # fixture's `convergence` block.
        self._spline = spline

        # The segment dispatch is done in line in __call__ rather than by calling the
        # SegmentedEntropyFactor, because a Python-level call costs about as much as the bisect
        # and the dispatch itself put together on this hot path. These two are that object's
        # `segment_edges` and `splines`, hoisted; for an unsegmented representation `_edges` is
        # empty and __call__ takes the branch that evaluates `_spline` exactly as it did before
        # this class learned to segment.
        self._edges = list(getattr(spline, "segment_edges", ()))
        self._splines = list(getattr(spline, "splines", ()))

        self._T_CMB = T_CMB
        self._label = label

        self._min_z = min_z
        self._max_z = max_z

        self._min_log_z = log(1.0 + min_z)
        self._max_log_z = log(1.0 + max_z)

    @property
    def segment_edges(self) -> tuple:
        """
        The interior segment edges in ``u = log(1+z)``, ascending; empty for a cosmology whose
        equation of state declares no break temperatures.
        """
        return tuple(self._edges)

    def __call__(self, z: float, z_is_log: bool = False) -> float:
        if z_is_log:
            log_z = z
            raw_z = exp(z) - 1.0
        else:
            log_z = log(1.0 + z)
            raw_z = z

        # if some way out of bounds, reject
        if log_z > _outward(self._max_log_z, +1):
            raise RuntimeError(
                f"{type(self).__name__}: evaluated {self._label} out of bounds @ z={raw_z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {_outward(self._max_z, -1):.5g})"
            )

        # otherwise, softly cushion the representation at the top end. The ramp is clamped with
        # the interpolant, so the value returned is T(max_z) exactly as it was when T itself was
        # the splined quantity.
        if log_z > self._max_log_z:
            log_z = self._max_log_z
            raw_z = self._max_z

        # same at lower limit
        if log_z < _outward(self._min_log_z, -1):
            raise RuntimeError(
                f"{type(self).__name__}: evaluated {self._label} out of bounds @ z={raw_z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {_outward(self._min_z, +1):.5g})"
            )

        if log_z < self._min_log_z:
            log_z = self._min_log_z
            raw_z = self._min_z

        # dispatch on u against the ascending edge list -- never on a recovered z, which is
        # irreducibly lossy at large z (CLAUDE.md). bisect_right puts a u exactly equal to an edge
        # in the segment *above* it, which is the branch the bisected edge itself belongs to.
        if self._edges:
            spline = self._splines[bisect_right(self._edges, log_z)]
        else:
            spline = self._spline

        return np.float64(self._T_CMB * (1.0 + raw_z) * exp(float(spline(log_z))))


class LambdaCDM_GenericEOS(BaseCosmology):
    """
    Construct a datastore
    """

    # The identity of the T(z) representation this class builds its background from. It is part of
    # the QCD cosmology's datastore lookup key
    # (Datastore/SQL/ObjectFactories/QCD_Cosmology.py), and it exists because nothing else in that
    # key can see the representation: build() matches on the seven parameter values and
    # log10_max_z, every one of which is an *input* to the model rather than a property of how
    # T(z) is approximated from those inputs.
    #
    # What it identifies -- everything about the background that the parameter key does not
    # otherwise capture:
    #
    #   * the tolerance to which _solve_T_z root-solves each node of the tabulation;
    #   * the quantity that is tabulated and splined -- T itself, or the entropy factor
    #     F(u) = log(T / [T_CMB (1+z)]) whose ramp is known in closed form;
    #   * the number of nodes and the order of the spline through them;
    #   * whether the representation is segmented at the redshifts where T(z) genuinely jumps, and
    #     where those segment edges are placed;
    #   * the set of points integration_break_points declares, because every quadrature and every
    #     ODE in the tree splits its panels there, so a BackgroundModel built against a different
    #     set is a different background.
    #
    # Every prompt in prompts/qcd-background-audit/ that changes any of those bumps this constant,
    # and a bump is the only signal a datastore ever gets:
    #
    #   version | prompt | what changed
    #   --------+--------+-----------------------------------------------------------------------
    #      1    |   03   | nothing numerically; the key exists
    #      2    |   04   | _solve_T_z tightened from xtol=1e-6, rtol=1e-4 to xtol=1e-300, rtol=1e-14
    #      3    |   05   | the entropy factor F(u) is splined, not T itself; T = T_CMB (1+z) e^F
    #      4    |   06   | F is splined per branch, edges bisected onto the jumps; 3000 nodes, k=5
    #      5    |   07   | integration_break_points declares the EOS crossings alone, not the knots
    #      6    |   13   | BackgroundModel splines its derivative fields one per branch, not across
    #           |        | the declared crossings; every stored omega_eff, and so every stored
    #           |        | phase, moves
    #
    # (prompt 08 appends a row here if it moves a per-sector break-point policy; it did not.)
    #
    # Version 6 is not a change to this class at all -- it is a change to
    # ComputeTargets/BackgroundModel.py, which is what makes the constant's *scope* worth stating.
    # It keys the background a cosmology yields, not the cosmology's own parameters: the list above
    # is what the parameter key does not capture, and "the lattice the background's derivative
    # fields are splined on, and whether it is split where this cosmology says it is not smooth"
    # belongs on it for exactly the reason the break-point set does. A datastore written before
    # prompt 13 holds BackgroundModel rows whose epsilon, d_epsilon_dz and d2_epsilon_dz2 ring at
    # two of the three crossings declared here -- 2.04e-02 relative at T_LO on the production grid
    # (docs/qcd-background-verification.md section 10.4) -- and nothing else in the key would say so.
    #
    # Why this is not optional. Without it the same cosmology row is returned under the same
    # serial when the representation changes; every BackgroundModel keyed on that serial is found
    # and deserialised; its tau, cs_tau and friction_F limbs are the *old* background's; and every
    # Gk/Tk numeric and WKB row built on it is served against a background that no longer exists
    # in the code. There is no exception, no warning and no column that differs -- a stale row is
    # otherwise undetectable, which is the whole reason this constant exists. The discrepancy the
    # campaign removes is 3.461e-08 relative in conformal time, which is of order 1.4e5 radians of
    # oscillation phase at k = 3e8/Mpc, so a stale row is not a small inaccuracy.
    #
    # This follows TkNumericIntegration.BREAK_POINT_KIND: a single declaration, readable from the
    # class without an instance, so that the factory can filter on it before any model is built.
    T_Z_REPRESENTATION_VERSION: int = 6

    def __init__(
        self,
        store_id: int,
        eos: GenericEOSBase,
        units: UnitsLike,
        params,
        max_z: float = DEFAULT_MAX_TEMPERATURE_Z_REDSHIFT,
    ):
        BaseCosmology.__init__(self, store_id)

        self._params = params
        self._units = units
        self._eos = eos

        self._max_z = max_z

        # unpack details of the parameter block so we can access them without extensive nesting
        self._name = f"{eos.name} | {params.name}"

        # Omega factors are all measured today
        self.omega_cc = params.omega_cc
        self.omega_m = params.omega_m
        self.f_baryon = params.f_baryon
        self.h = params.h
        self.T_CMB_Kelvin = params.T_CMB_Kelvin

        # Neff not used here because it is baked into the G(T) and G_S(T) parametersa computed by the EOS object
        # self.Neff = params.Neff

        # derived dimensionful quantities, expressed in whatever system of units we require
        self._H0 = 100.0 * params.h * units.Kilometre / (units.Second * units.Mpc)
        self._T_CMB = params.T_CMB_Kelvin * units.Kelvin

        self.H0sq = self._H0 * self._H0
        self.Mpsq = units.PlanckMass * units.PlanckMass

        # POPULATE KEY DATA NOT PROVIDED AS PART OF THE PARAMS BLOCK

        T_CMB_2 = self._T_CMB * self._T_CMB
        T_CMB_4 = T_CMB_2 * T_CMB_2

        Omega_factor = 3.0 * self.H0sq * self.Mpsq

        self.rho_m0 = Omega_factor * self.omega_m
        # note the effective G* reported by the EOS object should have reheating
        # of the thermal bath relative to the neutrinos already included.
        # Therefore, we don't need the extra famous factor (4/11)^(4/3)
        self.rho_r0 = RadiationConstant * self._eos.G(self._T_CMB) * T_CMB_4
        self.rho_cc = Omega_factor * self.omega_cc

        self.omega_r = self.rho_r0 / Omega_factor

        # cache values of G(T_CMB), G_S(T_CMB), [G(T_CMB)]^(4/3) and [G_S(T_CMB)]^(4/3) which we need to use later
        self._G_CMB = eos.G(self._T_CMB)
        self._G_S_CMB = eos.Gs(self._T_CMB)
        self._G_CMB_pow13 = pow(self._G_CMB, 1.0 / 3.0)
        self._G_S_CMB_pow13 = pow(self._G_S_CMB, 1.0 / 3.0)

        # Locate, once, the redshift at which T(z) reaches each of the equation of state's break
        # temperatures. These are the *only* points integration_break_points declares, and they
        # are also where _build_T_z_spline segments the entropy factor, so they are bisected here
        # -- before there is a representation to evaluate -- and both consumers read this one
        # cache. See _build_break_point_crossings_log1pz.
        self._break_point_crossings_log1pz = self._build_break_point_crossings_log1pz()

        # Build the spline used to map z to a temperature, given the specified equation of state.
        # We need to go all the way to z=0 so that we can compute the radiation temperature today
        # (needed to match to the CMB), but we need to compute a bit into the future (negative z)
        # so that we can accurately compute numerical derivatives at z=0
        self._T_z_spline = self._build_T_z_spline(
            min_z=DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, max_z=max_z
        )

        # COMPUTE BASIC DATA ABOUT THIS COSMOLOGICAL MODEL

        rho_today = self.rho(0)
        gram_per_m3 = units.Gram / (units.Metre * units.Metre * units.Metre)
        rho_today_gram_m3 = rho_today / gram_per_m3

        # Solve for epochs of matter/radiation and matter/Lambda equality. These used to be
        # printed in the banner below and discarded; they are now what z_matter_radiation_equality
        # and z_matter_lambda_equality return, and on a cosmology that declares break points they
        # are production source-grid sample locations (main.py's cosmology_feature_redshifts).
        # Nothing about the solve changes: same two calls, same closed-form initial guesses, same
        # bracket and tolerances.
        #
        # The guesses stay closed forms deliberately. 1 + z_eq = Omega_m/Omega_r is only a *guess*
        # here -- it is exact solely where g_* is flat at equality, which is the accident this
        # method's own comment refuses to build on -- but it is a sound one, since z_eq cannot
        # move by a large factor without wrecking the CMB, and the bracketing below makes a wrong
        # guess harmless rather than fatal.
        self._z_matter_radiation_equality = self._find_rho_equality(
            "matter", "radiation", init_z=self.omega_m / self.omega_r - 1.0
        )
        self._z_matter_lambda_equality = self._find_rho_equality(
            "matter",
            "lambda",
            init_z=pow(self.omega_cc / self.omega_m, 1.0 / 3.0) - 1.0,
        )

        print(f'@@ Parametrized equation-of-state LambdaCDM-like model "{self._name}"')
        print(f'|  equation of state = "{self._eos.name}", max_z = {self._max_z:.5g}')
        print(f"|  Omega_m = {self.omega_m:.4g}")
        print(f"|  Omega_cc = {self.omega_cc:.4g}")
        print(f"|  Omega_r = {self.omega_r:.4g}")
        print(f"|  present-day energy density = {rho_today_gram_m3:.4g} g/m^3")
        print(
            f"|  matter-radiation equality at z = {self._z_matter_radiation_equality:.4g}"
        )
        print(f"|  matter-Lambda equality at z = {self._z_matter_lambda_equality:.4g}")

    @property
    def type_id(self) -> int:
        # inherit our unique ID from the underlying choice of equation of state
        return self._eos.type_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def H0(self) -> float:
        return self._H0

    @property
    def z_matter_radiation_equality(self) -> float:
        """
        The equality redshift ``__init__`` located with :meth:`_find_rho_equality`, on this
        model's *own* rho_r = RadiationConstant G(T(z)) T(z)^4.

        This is the solve and not ``Omega_m/Omega_r - 1``. The two agree to 7 ulp on
        ``QCD_Cosmology`` at production parameters, but only because all of that equation of
        state's g_*(T) structure sits at z ~ 1e12, twelve orders above equality; that is a
        property of where the QCD transition happens to fall and not of this class, which is
        parametrized by an arbitrary equation of state. See
        ``CosmologyModels.base.BaseCosmology.z_matter_radiation_equality``.

        :return: the matter-radiation equality redshift
        """
        return self._z_matter_radiation_equality

    @property
    def z_matter_lambda_equality(self) -> float:
        """
        The matter-Lambda equality redshift, likewise from ``__init__``'s solve.

        Here the closed form is exact on any equation of state -- rho_m/rho_Lambda has no
        temperature dependence -- and the solve is measured to return the same double on both
        production models. It is still the solve that answers, so that this class has one route
        to both numbers rather than two.

        :return: the matter-Lambda equality redshift
        """
        return self._z_matter_lambda_equality

    def T_photon(self, z: float) -> float:
        return self._T_z_spline(z)

    def _solve_T_z(self, z: float) -> float:
        """
        Solve for T(z), the temperature as a function of redshift z
        :param z: redshift
        :return: temperature at this redshift, as a dimensionful quantity
        """

        # in the absence of entropy effects, T(z) a(z) = T_CMB a0, where a0 is the value of
        # the scale factor today, and T_CMB is the radiation temperature today. Then
        #   T(z) = T_CMB (a0/a) = T_CMB (1 + z)
        # With entropy effects included, this scaling is no longer exact. Instead, T(z)
        # should solve the implicit equation
        #   T(z) [G_S(T(z))]^(1/3) = T_CMB [G_S(T_CMB)]^(1/3) (1 + z)

        # a good initial guess for T(z) is given by simple redshfting, without including factors
        # of G_S(T). We always work with dimensionful values of T.
        # This initial guess will be an overestimate, because G_S(T) >= G_S(T_CMB)
        bracket_hi = 1.05 * self._T_CMB * (1.0 + z)

        # meanwhile, the largest G_S(T) can be is given by its asymptotic value
        bracket_lo = 0.95 * self._T_CMB * (1.0 + z) / pow(HIGH_T_GSTAR, 1.0 / 3.0)

        target = self._T_CMB * self._G_S_CMB_pow13 * (1.0 + z)

        def T_equation(T: float) -> float:
            G_S = self._eos.Gs(T)
            G_S_pow13 = pow(G_S, 1.0 / 3.0)
            return T * G_S_pow13 - target

        if T_equation(bracket_lo) * T_equation(bracket_hi) >= 0.0:
            raise RuntimeError(
                f"Could not bracket target temperature T(z) at z={z:.4g}, bracket_lo={bracket_lo:.5g}, bracket_hi={bracket_hi:.5g}"
            )

        # This solve fixes one node of the T(z) spline built in _build_T_z_spline, so its cost is
        # paid once per node at build time (~500 nodes, 8.3 us each -- a few ms total), never at
        # evaluation time. Each node converges independently, so a loose tolerance here does not
        # give a uniformly-scaled error: it gives an *uncorrelated scatter* between neighbouring
        # nodes, and a cubic spline through scattered nodes has a scattered derivative too
        # (prompts/qcd-background-audit/, prompt 04; audit §3, T2). rtol=1e-14 sits just above
        # Brent's own floor of ~4*eps=8.9e-16, so it is the tightest tolerance root_scalar can
        # actually resolve; xtol=1e-300 disables the absolute component entirely -- T spans twenty
        # decades over the tabulated range, so any finite absolute tolerance binds at the cold end
        # long before the hot end is resolved, and the previous xtol=1e-6 did exactly that.
        root = root_scalar(
            T_equation, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=1e-14
        )

        if not root.converged:
            raise RuntimeError(
                f'root_scalar() did not converge to a solution: x_bracket=({bracket_lo:.5g}, {bracket_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
            )

        return root.root

    def _build_T_z_spline(
        self,
        min_z: float,
        max_z: float,
        samples: int = DEFAULT_T_Z_SPLINE_SAMPLES,
        order: int = DEFAULT_T_Z_SPLINE_ORDER,
    ) -> TemperatureRepresentation:
        # Add a 5% buffer to the min/max z range, so that a caller asking for exactly the
        # requested bounds is inside the tabulated range rather than on its edge.
        #
        # The buffer has to be applied to 1+z, not to z. Scaling z itself works only while z is
        # positive: with min_z = DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2, `0.95 * min_z` is
        # -0.19, which moves the lower bound *inward* and shrinks the range by 5% instead of
        # widening it -- so the model could not be evaluated at its own declared floor. 1+z is
        # positive throughout (z > -1), so scaling that is correct at both ends.
        min_z = 0.95 * (1.0 + min_z) - 1.0
        max_z = 1.05 * (1.0 + max_z) - 1.0

        u_lo, u_hi = log(1.0 + min_z), log(1.0 + max_z)

        # What is tabulated is the *entropy factor*
        #
        #     F(u) = log( T(z) / [T_CMB (1+z)] ) = -(1/3) log( g_s(T) / g_s(T_CMB) ),
        #
        # not T itself, and TemperatureRepresentation multiplies the ramp back in on evaluation.
        # The class docstring says why: F is bounded, O(1) and exactly constant wherever g_s is,
        # while T over this range is almost entirely the (1+z) ramp, which is exact in closed
        # form and needs no interpolant at all.
        #
        # It is tabulated one spline per branch, with the segment edges bisected onto the
        # redshifts at which T(z) jumps (_entropy_segment_edges_log1pz below). A single spline
        # across a genuine discontinuity carries the height of the jump as its maximum error at
        # every node count -- 7.2e-04 at 500, 2,000 and 3,000 nodes alike -- and that error is
        # 3.5e-08 of the conformal time, which is 1.4e5 radians of oscillation phase at
        # k = 3e8/Mpc. Segmenting removes it: on each branch F is a smooth function of u with
        # nothing left to resolve but g_s itself.
        spline = build_segmented_entropy_spline(
            self._entropy_factor_log1pz,
            self._entropy_segment_edges_log1pz(u_lo, u_hi),
            u_lo,
            u_hi,
            samples=samples,
            order=order,
        )

        # The knot vector, in log(1+z). It is NOT a break-point declaration and has not been one
        # since prompt 07 of prompts/qcd-background-audit/: integration_break_points below returns
        # the equation of state's temperature crossings and nothing else. It is kept because the
        # *measurement* that the knots are not worth declaring has to be repeatable -- the
        # statement "none of the declared break points is a knot of an interpolant" is only
        # checkable if the knots are still knowable from outside -- and because
        # docs/qcd-background-audit/measure_T_z_representation.py §5 and
        # ComputeTargets/tests/test_background_tau.py::test_qcd_break_points both read it to make
        # exactly that check. Recorded here, where the spline is built, so that nothing outside
        # this class has to reach into the representation for it; a segmented representation
        # contributes the knots of every segment.
        #
        # Why the knots stopped being declared (prompt 07 §2 item 1, measured): an order-k
        # interpolating spline through simple interior knots is C(k-1) there, so at k = 5 the
        # first genuinely discontinuous derivative of F is the *fifth* -- measured, d1 to d4 jump
        # by at most 1.3e-13, 7.1e-12, 5.0e-10 and 6.2e-08 absolute (floating-point noise of a
        # one-sided evaluation; the continuity is exact by construction) while d5 jumps by 26 %
        # relative. The deepest derivative anything in the tree actually builds is d3_lnH_dz3
        # (ComputeTargets/BackgroundModel.py), three levels below that. The observable agrees:
        # across a knot the step in d ln H/du is 8.3e-11 at worst and |H/H_exact - 1| in a knot's
        # neighbourhood is 6.3e-12. The 500-node k = 3 lattice this replaced was C2, so its third
        # derivative -- the one d3_lnH_dz3 reads -- jumped by a median 74 % relative, the step in
        # d ln H/du reached 6.3e-07 and the residual against the exact background reached
        # 2.1e-04. That is the 1e-4-level defect the old lattice carried, and why splitting a
        # Gauss panel at every one of its knots was buying something; there is nothing left at a
        # knot of this representation to buy.
        self._T_z_spline_knots_log1pz = np.unique(np.asarray(spline.t, dtype=float))

        return TemperatureRepresentation(
            spline,
            T_CMB=self._T_CMB,
            label="T(z)",
            min_z=min_z,
            max_z=max_z,
        )

    def _entropy_factor_log1pz(self, u: float) -> float:
        """
        ``F(u) = log( T(z) / [T_CMB (1+z)] )`` at ``u = log(1+z)``, from the node solve.

        There is no second root solve: this is the same ``_solve_T_z`` call that used to supply
        the node value of ``T``. The division is written with ``(1.0 + z)`` for exactly the
        ``z = exp(u) - 1`` that ``TemperatureRepresentation.__call__`` recovers, and multiplies
        back, so that the representation reproduces its own node values to the last bit rather
        than to within a rounding of ``exp(u)`` against ``1+z``.

        :param u: ``log(1+z)``
        :return: the entropy factor at ``u``, dimensionless
        """
        z = exp(u) - 1.0
        return log(self._solve_T_z(z) / (self._T_CMB * (1.0 + z)))

    def _entropy_segment_edges_log1pz(self, u_lo: float, u_hi: float) -> List[float]:
        """
        The points ``u = log(1+z)`` strictly inside ``(u_lo, u_hi)`` at which ``T(z)`` crosses one
        of the equation of state's ``break_temperatures_GeV``, ascending -- the edges at which
        ``_build_T_z_spline`` segments the entropy factor.

        These are the same crossings ``integration_break_points`` declares, and since prompt 07 of
        ``prompts/qcd-background-audit/`` they are literally the same numbers: both read
        :attr:`_break_point_crossings_log1pz`, which is bisected once in ``__init__``. They were
        once found two different ways -- here by bisection, there by a root solve on
        ``T_photon(z) - T_break`` -- and that is exactly the arrangement README §2 (b) warns
        against, since the two methods disagree by ~1e-12 in ``u`` and the segment padding is
        1e-12. See :meth:`_bisect_temperature_crossing_log1pz`.

        All four break temperatures are used, not only the subset at which ``g_s`` jumps
        (``discontinuity_temperatures_GeV``). Segmenting where the equation of state is in fact
        continuous costs nothing -- the two branches agree there to 1.8e-11 and the pair of
        splines reproduces one smooth function -- while failing to segment at a genuine jump costs
        the whole of the error this design removes. ``QCD_EOS``'s join at ``EOS_T_LO = 0.002`` GeV
        is the continuous one (audit §1); it is segmented at anyway
        (``prompts/qcd-background-audit/README.md`` §7 D4).

        :param u_lo: lower end of the tabulated range in ``u``
        :param u_hi: upper end of the tabulated range in ``u``
        :return: an ascending list of ``u`` values, possibly empty
        """
        return sorted(
            u for u in self._break_point_crossings_log1pz.values() if u_lo < u < u_hi
        )

    def _build_break_point_crossings_log1pz(self) -> Mapping[float, float]:
        """
        The ``u = log(1+z)`` at which ``T(z)`` reaches each of the equation of state's
        ``break_temperatures_GeV``, keyed by that temperature in GeV; temperatures not crossed
        inside :meth:`_bisect_temperature_crossing_log1pz`'s search bracket are absent.

        **This is the single definition of where the cosmology's break points are.** Both
        consumers read it: :meth:`_entropy_segment_edges_log1pz`, which segments the entropy
        factor at these redshifts, and :meth:`integration_break_points`, which declares them to
        every quadrature and every ODE in the tree. One definition is not a tidiness point. Until
        prompt 07 of ``prompts/qcd-background-audit/`` the two were computed by different methods
        -- bisection here, a ``root_scalar`` bracket on ``T_photon(z) - T_break`` in what is now
        ``CosmologyModels.tests.T_z_reference.temperature_crossing_log1pz`` (moved out of this
        class by ``prompts/background-solver-robustness/`` prompt 04) -- and they disagreed by
        ~1e-12 in ``u``, which is the same size as the padding that holds each segment's nodes
        inside its own branch. Declaring a panel edge on the wrong side of a jump by that margin
        is precisely the silent failure README §2 (b) exists to prevent.

        Called from ``__init__`` *before* the temperature representation is built, because the
        representation needs the edges. It therefore cannot use ``T_photon``; it bisects
        ``_solve_T_z`` directly, which is the defining equation at ``rtol = 1e-14`` and is the
        only temperature available at that point in construction. That is also why the result is
        independent of the representation: the crossings are a property of the equation of state.

        Cost: one bisection per break temperature, ~1.5 ms each, ~6 ms for the production four.
        Paid once per cosmology instead of on every ``integration_break_points`` call, which used
        to root-solve the crossings afresh each time it was asked.

        :return: a mapping from break temperature in GeV to ``u = log(1+z)``
        """
        GeV = self._units.GeV

        crossings = {}
        for T_break_GeV in sorted(set(self._eos.break_temperatures_GeV)):
            u = self._bisect_temperature_crossing_log1pz(T_break_GeV * GeV)
            if u is not None:
                crossings[T_break_GeV] = u

        return crossings

    def _bisect_temperature_crossing_log1pz(
        self,
        T_break: float,
        z_lo: float = 1.0e-6,
        z_hi: float = 1.0e19,
        rtol: float = 1.0e-15,
    ) -> Optional[float]:
        """
        The ``u = log(1+z)`` at which the monotone ``T(z)`` reaches ``T_break``, by **geometric
        bisection**; ``None`` if the crossing is not inside ``[z_lo, z_hi]``.

        **This may not be replaced by a root solve on** ``T(z) - T_break``
        (``prompts/qcd-background-audit/README.md`` §2 (b), audit §2). ``T(z)`` *jumps* at exactly
        these points, so that difference need not have a root at all: at the lowest QCD crossing
        it steps from ``-7.53e-04`` to ``+8.84e-06`` relative across a single ulp of ``u``,
        never passing through zero. A bracketing solver applied to it nevertheless reports
        ``converged`` and returns a non-root, at a place that depends on its tolerances --
        measured at ``+1.126e-12`` in ``u`` at ``root_scalar``'s defaults over the tabulated
        range, which is *further* from the jump than the 1e-12 by which the segments' nodes are
        held inside their branches. The segment below the jump would then be fitted through a node
        taken from the branch above it, the discontinuity would be interpolated across after all,
        and every other number in the representation would still improve: the audit records a
        first attempt that did exactly this and measured 5.7e-04, the full error, still in place.
        ``CosmologyModels/tests/T_z_reference.py``'s ``jump_locations`` is the independent
        implementation this one is scored against, and
        ``test_a_segment_edge_bisected_and_one_root_found_disagree`` is the standing demonstration
        of the trap.

        Bisection has no such freedom. It brackets the crossing between one ``z`` whose
        temperature is below ``T_break`` and one whose temperature is not, and replaces one end by
        the geometric mean of the two until they are a relative ``rtol`` apart -- which at
        ``rtol = 1e-15`` is the representable floor, so the returned ``u`` is good to the last bit
        or two. The test at each step is an inequality against a monotone function, which a
        discontinuity does not disturb.

        The bisection is geometric in ``z`` rather than in ``1+z``, which is the one place in this
        campaign where ``z`` rather than ``u`` is carried: it mirrors ``jump_locations`` step for
        step, so the two agree bit for bit and the test that scores this against it is an equality
        rather than a tolerance. The crossings are at ``z ~ 4e7`` and above, where the two
        geometries differ by 1e-8 of a bracket width in the early iterations and by nothing at
        all in the last ones; ``log1p`` is taken of the final bracket, and no recovered ``z`` ever
        reaches an equality-like comparison (``CLAUDE.md``).

        ``_solve_T_z`` is the temperature here, not the spline being built: the edges have to be
        known before there is a representation to evaluate. Each call costs a root solve at
        ``rtol = 1e-14``, and about 57 halvings are needed to cross twenty-five decades, so an
        edge costs ~1.5 ms and the whole set ~6 ms of the build.

        :param T_break: the dimensionful temperature to cross
        :param z_lo: lower end of the search bracket, in ``z``
        :param z_hi: upper end of the search bracket, in ``z``
        :param rtol: relative width in ``1+z`` at which the bisection stops
        :return: the crossing in ``u = log(1+z)``, or ``None``
        """
        if not self._solve_T_z(z_lo) < T_break < self._solve_T_z(z_hi):
            return None

        lo, hi = z_lo, z_hi
        for _ in range(200):
            mid = sqrt(lo * hi)
            if self._solve_T_z(mid) < T_break:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1.0 < rtol:
                break

        return log1p(0.5 * (lo + hi))

    def integration_break_points(
        self, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
    ) -> np.ndarray:
        """
        Every point in u = log(1+z), strictly inside (log(1+z_lo), log(1+z_hi)), at which
        Hubble(z), rho(z), T_photon(z), wBackground(z) or wPerturbations(z) loses smoothness of
        the requested ``kind``.

        Both kinds return crossings of an equation-of-state temperature and nothing else:

        * ``kind = BREAK_POINT_ALL`` (the default) gives the redshifts at which T(z) reaches one
          of break_temperatures_GeV, where G, Gs or w change analytic form -- a jump in H(z) at a
          G/Gs boundary, a kink in c_s^2 at a w clamp. On the production range of QCD_Cosmology
          that is **3** points.
        * ``kind = BREAK_POINT_DISCONTINUITY`` gives the subset at which a quantity actually
          *jumps*, the crossings of discontinuity_temperatures_GeV: **2** on the same range.

        A fixed-order Gauss-Legendre panel that straddles a break converges only as N^-2
        (docs/gktk-remedial/RESIDUAL-CONVERGENCE.md, §3), so the cumulative tables of
        ComputeTargets/BackgroundModel.py split every production interval at the points returned
        here and integrate the pieces separately; an adaptive ODE stepper absorbs a merely C2
        point and asks for the jumps alone (Quadrature/integrators/numeric_with_phase_cut.py,
        whose module docstring says why the distinction matters there and not in a quadrature).

        **The interior knots of the T(z) tabulation are not returned, and have not been since
        prompt 07 of prompts/qcd-background-audit/ (finding G1 of the audit).** They were: the set
        was 404 points on the production source grid when the tabulation carried 500 nodes, and
        2,414 when prompt 06 raised it to 3,000 -- a Gauss panel split roughly every 0.67 grid
        intervals for the node lattice of an auxiliary interpolant, and the sole cause of
        prompts/phase-representation prompt 02's Schoenberg-Whitney failure. A knot lattice is a
        property of the approximation, not of the cosmology, and this method's contract is the
        cosmology's non-smoothness.

        What made that defensible once, and does not now, is smoothness at a knot, and prompt 07
        measured it rather than asserting it. An order-k interpolating spline through simple
        interior knots is C(k-1) there, so at the shipped k = 5 the first genuinely discontinuous
        derivative of F is the fifth (d1 to d4 agree across a knot to 1.3e-13, 7.1e-12, 5.0e-10
        and 6.2e-08 absolute, which is the noise of a one-sided evaluation; d5 jumps by 26 %
        relative). The deepest derivative anything in the tree builds is d3_lnH_dz3. The
        observable says the same: across a knot the step in d ln H/du is at most 8.3e-11 and
        |H/H_exact - 1| near a knot at most 6.3e-12, against 6.3e-07 and 2.1e-04 for the 500-node
        k = 3 lattice that used to be declared. There is nothing left at a knot for a panel edge
        to protect against.

        :param z_lo: lower redshift of the range (inclusive; a break exactly here is not returned)
        :param z_hi: upper redshift of the range
        :param kind: BREAK_POINT_ALL for every non-smooth point, BREAK_POINT_DISCONTINUITY for
            the subset at which a quantity jumps
        :return: an ascending numpy array of u values, empty if none fall inside the range
        """
        if kind not in BREAK_POINT_KINDS:
            raise ValueError(
                f"LambdaCDM_GenericEOS.integration_break_points: unknown break-point kind "
                f'"{kind}" (expected one of {", ".join(BREAK_POINT_KINDS)})'
            )

        u_lo = log1p(z_lo)
        u_hi = log1p(z_hi)
        if not u_lo < u_hi:
            raise ValueError(
                f"LambdaCDM_GenericEOS.integration_break_points: need z_lo < z_hi "
                f"(got z_lo={z_lo:.6g}, z_hi={z_hi:.6g})"
            )

        breaks = tuple(self._eos.break_temperatures_GeV)
        jumps = tuple(self._eos.discontinuity_temperatures_GeV)
        if not set(jumps).issubset(set(breaks)):
            raise RuntimeError(
                f"LambdaCDM_GenericEOS.integration_break_points: the equation of state "
                f'"{self._eos.name}" declares discontinuity temperatures that are not among its '
                f"break temperatures ({sorted(set(jumps) - set(breaks))} GeV). Every "
                f"discontinuity is a break, so the quadrature path would not be split where the "
                f"ODE path is."
            )

        temperatures = breaks if kind == BREAK_POINT_ALL else jumps

        # The crossings were bisected once, in __init__, and are read here rather than solved for
        # again: _build_break_point_crossings_log1pz says why there is exactly one definition of
        # where they are, and why a root solve on T_photon(z) - T_break is not it.
        points = []
        for T_in_GeV in temperatures:
            u = self._break_point_crossings_log1pz.get(T_in_GeV)
            if u is not None and u_lo < u < u_hi:
                points.append(u)

        if len(points) == 0:
            return np.empty(0, dtype=float)
        return np.unique(np.asarray(points, dtype=float))

    def _rho_fluid(self, z: float) -> Mapping[str, float]:
        """
        Determine the densities of the matter, radiation (etc.) fluids at redshift z
        :param z:
        :return:
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z

        T: float = self._T_z_spline(z)
        T_2 = T * T
        T_4 = T_2 * T_2

        rho_m = self.rho_m0 * one_plus_z_3

        # reheating of the thermal bath due to annihilations, and splitting of the photon and
        # neutrino temperatures at low redshift, should be included already in the EOS object
        rho_r = RadiationConstant * self._eos.G(T) * T_4

        return {
            "T": T,
            "matter": rho_m,
            "radiation": rho_r,
            "lambda": self.rho_cc,
        }

    def _find_rho_equality(
        self, species_A: str, species_B: str, init_z: float
    ) -> float:
        """
        Determine the redshift at which the energy density in species A equals the energy density
        in species B.

        ``init_z`` is a starting *guess*, not an answer: the bracket is expanded about it until
        the residual changes sign, and Brent is run on that bracket. It used to be handed straight
        to an unbracketed secant, which is why it mattered that the callers' guesses happen to be
        the closed-form roots; see the comment at the solve.

        :param species_A: a key of :meth:`_rho_fluid`
        :param species_B: a key of :meth:`_rho_fluid`
        :param init_z: a starting guess for the equality redshift
        :return: the redshift at which the two densities are equal
        """

        def match_rho(z: float) -> float:
            rho = self._rho_fluid(z)
            return rho[species_A] - rho[species_B]

        # The bracket is expanded multiplicatively in 1+z, never additively in z. 1+z is positive
        # throughout the tabulated range (z > -1), so a multiplicative step is scale-free: one
        # policy serves both roots this method is called for -- matter/radiation at z ~ 3.4e3 and
        # matter/Lambda at z ~ 0.30, four decades apart -- and it can never propose 1+z <= 0,
        # which is how the secant this replaced reached negative z. It is also the same argument
        # _build_T_z_spline makes above for applying its 5% buffer to 1+z rather than to z.
        #
        # sqrt(2) per step, capped at 140 steps: sqrt(2)**140 = 2**70 = 1.2e21, which carries 1+z
        # from the representation's floor (0.76, at the default min_z = -0.2) past its ceiling at
        # the default max_z = 1e20 in a single direction, so the cap can never bind before the
        # clamp below does. In practice the loop runs once: the callers' analytic guesses are the
        # roots to a few ulp on every production model, so the first expansion already straddles.
        BRACKET_EXPANSION_FACTOR = sqrt(2.0)
        BRACKET_EXPANSION_MAX_STEPS = 140

        # The expansion is clamped to the tabulated range of T(z), which is the widest interval on
        # which _rho_fluid can be evaluated at all. Clamping there and then failing to bracket is
        # the intended behaviour, and it is what this change buys: the old solve walked past the
        # floor to negative z and died inside _rho_fluid two frames below this method's own guard,
        # reporting a temperature-spline bounds error to a caller whose actual mistake was a
        # cosmology whose equality redshift is not in its own tabulated range.
        z_floor = self._T_z_spline._min_z
        z_ceil = self._T_z_spline._max_z

        # The guess is clamped into that range before it is evaluated, not only the endpoints
        # expanded from it. A caller can hand this method a guess that is itself outside the
        # tabulated range -- __init__ does, on any cosmology whose max_z is below its own equality
        # redshift -- and evaluating it would raise the same TemperatureRepresentation bounds
        # error from inside _rho_fluid that this method exists to stop reporting. Clamped, the
        # expansion below runs, fails to bracket, and says which pair it was looking for.
        z_guess = min(max(init_z, z_floor), z_ceil)
        one_plus_z_guess = 1.0 + z_guess

        # The guess is evaluated first, and its residual is used only in the failure message.
        # There is deliberately NO short-circuit returning it when the residual is small. That the
        # guess is already the root to rounding is a property of this equation of state at this
        # redshift (prompts/background-solver-robustness/AUDIT.md §2.3), not of the code, and
        # writing it in would make permanent the very accident this change exists to remove.
        f_guess = match_rho(z_guess)

        scale = 1.0
        bracket_lo = bracket_hi = z_guess
        f_lo = f_hi = f_guess
        bracketed = False

        for _ in range(BRACKET_EXPANSION_MAX_STEPS):
            scale *= BRACKET_EXPANSION_FACTOR

            bracket_lo = max(one_plus_z_guess / scale - 1.0, z_floor)
            bracket_hi = min(one_plus_z_guess * scale - 1.0, z_ceil)

            f_lo = match_rho(bracket_lo)
            f_hi = match_rho(bracket_hi)

            # Compare signs rather than testing f_lo * f_hi <= 0, as _solve_T_z does at :568. The
            # residual there is a difference of temperatures and is O(1); here it is a difference
            # of energy densities, which reach ~1e183 at the top of the default tabulated range,
            # and their product overflows to +inf -- a sign change that the product test would
            # then silently fail to see.
            if f_lo == 0.0 or f_hi == 0.0 or (f_lo < 0.0) != (f_hi < 0.0):
                bracketed = True
                break

            # Both ends are against the tabulated range and the residual has not changed sign, so
            # no further expansion is possible and there is no root to find.
            if bracket_lo <= z_floor and bracket_hi >= z_ceil:
                break

        if not bracketed:
            raise RuntimeError(
                f"LambdaCDM_GenericEOS._find_rho_equality: could not bracket the redshift at "
                f"which rho[{species_A}] = rho[{species_B}]. Expanding multiplicatively in 1+z "
                f"about the initial guess z={init_z:.5g}, clamped to z={z_guess:.5g} "
                f"(residual {f_guess:.5g}), reached "
                f"z_lo={bracket_lo:.5g} (residual {f_lo:.5g}) and z_hi={bracket_hi:.5g} "
                f"(residual {f_hi:.5g}), which have the same sign, so no root is enclosed. The "
                f"search is clamped to the tabulated range of T(z), z in [{z_floor:.5g}, "
                f"{z_ceil:.5g}]."
            )

        # rtol=8.9e-16 is Brent's own convergence floor of 4*eps = 8.881784e-16 -- scipy rejects
        # anything smaller outright -- so it is the tightest relative tolerance root_scalar can
        # resolve. xtol=1e-300 disables the absolute component deliberately, for the reason
        # _solve_T_z gives at :574-582: this method serves two roots four decades apart, so any
        # finite absolute tolerance in z is meaningless at z ~ 3.4e3 and is the only thing acting
        # at z ~ 0.30. The xtol=1e-6, rtol=1e-4 this replaces did both at once, and neither bound
        # the error nor predicted it.
        #
        # The floor, and not the rtol=1e-14 that _solve_T_z uses at :584. That was this campaign's
        # recommendation (prompts/background-solver-robustness/ README §7 D1) until it was
        # measured: the residual here is a cancellation between two densities of order 1e112,
        # quantised at ~7e100 near the root, so the sign change is not localised to a single
        # float. On QCD_Cosmology at matter/radiation equality the residual is exactly zero one
        # ulp above the closed form, non-zero either side of it, and changes sign six to seven ulp
        # higher -- the root is a band a few ulp wide, and where a solver stops inside it is set
        # by rtol. rtol=1e-14 is 75 ulp of slack at z ~ 3.4e3 and Brent stopped 7 ulp away from an
        # independent reference; at the floor it lands on it. The user took that decision on
        # 2026-09-16 (README §7 D1; prompt 02 §2 of that campaign holds the measurement).
        #
        # Cost, measured rather than predicted: the four production calls go from 3, 1, 1, 1
        # evaluations of match_rho to 23, 25, 21, 25 -- between +20 and +24 each, paid twice per
        # model construction, each one spline evaluation. (AUDIT.md §3.1 predicted +6 to +9; that
        # figure is for tightening the secant, and it does not survive bracketing, which buys the
        # two bracket endpoints and Brent's bisection steps as well.)
        #
        # Why this is worth doing at all, which is the finding of AUDIT.md §2.3: until this change
        # the solve was accurate because its caller handed it the closed-form root, not because
        # its tolerances were adequate. g_* is flat at z_eq, so rho_r ~ (1+z)^4 exactly there and
        # 1+z = Omega_m/Omega_r is the closed solution -- which is exactly what __init__ passes in
        # as init_z, so the secant confirmed it in one to three evaluations and stopped. That is a
        # property of this equation of state at this redshift, and nothing checked it. The bracket
        # is what makes the solve correct by construction on an equation of state where g_* is
        # *not* flat at equality, and what makes a failure to find the root say so in its own
        # words instead of dying inside _rho_fluid.
        root = root_scalar(
            match_rho, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=8.9e-16
        )

        if not root.converged:
            raise RuntimeError(
                f"LambdaCDM_GenericEOS._find_rho_equality: root_scalar() did not converge to the "
                f"redshift at which rho[{species_A}] = rho[{species_B}]: "
                f"z_bracket=({bracket_lo:.5g}, {bracket_hi:.5g}), "
                f'iterations={root.iterations}, method={root.method}: "{root.flag}"'
            )

        return root.root

    def rho(self, z: float) -> float:
        """
        Determine the total matter density at redshift z
        :param z:
        :return:
        """
        rho = self._rho_fluid(z)
        return rho["matter"] + rho["radiation"] + rho["lambda"]

    def Hubble(self, z: float) -> float:
        """
        Evaluate the Hubble rate H(z) at the specified redshift z
        :param z: required redshift
        :return: value of H(z)
        """
        rho_total = self.rho(z)
        H0sq = rho_total / (3.0 * self.Mpsq)
        return sqrt(H0sq)

    def wBackground(self, z: float) -> float:
        rho = self._rho_fluid(z)

        T = rho["T"]

        # background w(z) includes contributions from radiation, cosmological constant, and matter
        # (but matter has w=0 and drops out)
        numerator = self._eos.w(T) * rho["radiation"] + (-1.0) * rho["lambda"]
        denominator = self.rho(z)

        return numerator / denominator

    def wPerturbations(self, z: float) -> float:
        rho = self._rho_fluid(z)
        T = rho["T"]

        # perturbations w(z) includes contributions from radiation and matter, but not the cosmological constant,
        # which we take not to have perturbations. (Possibly we shouldn't do that, but instead allow the cosmological
        # constant to cluster with c_s=1?)
        # As for the background, matter has w=0 and drops out.
        numerator = self._eos.w(T) * rho["radiation"]
        denominator = rho["matter"] + rho["radiation"]

        return numerator / denominator
