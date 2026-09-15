"""
What does the source grid's density buy the consumer's ``phi`` spline, and what would a criterion
on ``phi``'s curvature buy instead?

Written for prompt 12 of ``prompts/qcd-background-audit/``, which implements the research half of
audit §8 recommendation 5 -- *"its density should be set by a measured criterion on phi's
curvature rather than by a uniform ``samples_per_log10z``"*. The prompt **measures and
recommends**; it changes no production file, and this script changes none either.

**The oracle, and why it is not the stored phase.** Prompts 10 and 11 scored the consumer against
``phi`` recovered from a *stored* ``theta``, which carries the ``eps k tau`` rounding of
``[02-consumer-phi-below-the-storage-granularity]``: 3.05e-07 rad at k = 1e5/Mpc and 9.15e-04 rad
at k = 3e8/Mpc. That floor is exactly what a density measurement must see *past*, because a grid
question is about interpolation error and not about rounding. So the reference here is the phase
residual itself:

    phi(z) = rho(z) + const,   rho(z) = int_z^{z_top} C/(omega + omega_0) dz'

evaluated through ``ComputeTargets.phase_residual.build_phase_residual``'s ``CumulativeTable``,
whose ``delta`` does genuine Gauss quadrature between arbitrary endpoints rather than
interpolating (``ComputeTargets/cumulative_table.py:360``). ``PrimitivePhase``'s own module
docstring is the warrant: ``phi`` is ``rho`` plus the producer's constant offsets, and a constant
is invisible to an interpolation error. The rounding floor then enters this script only where it
belongs -- as the line below which no grid decision means anything (section B).

**The criterion's inputs are pre-grid, and that is the point.** ``populate_z_sample`` runs before
any ``BackgroundModel`` exists, so a criterion may use only what the *cosmology* supplies
pointwise. It does supply enough:

    d phi / du = -(1+z) C(z) / (omega(z,k) + omega_0(z,k)),   u = log(1+z)

with ``omega_0^2 = w (k/H)^2`` (``w = 1`` in the ``Gk`` sector, ``w = c_s^2`` in ``Tk``) and ``C``
built from ``H``, ``c_s^2`` and their z-derivatives. ``H`` and ``c_s^2`` are closed form on every
production cosmology; the derivatives are closed form on ``LambdaCDM`` and are taken here by
finite differences of the cosmology's own pointwise values for ``QCD_Cosmology``, which is what a
pre-grid criterion would have to do. Section A checks the closed form against the table.

Sections:

* **A.** The closed-form ``phi'`` against the oracle -- the licence for everything below.
* **B.** What the shipped grid delivers: interpolation error per decade, per model, per sector, at
  all three reference wavenumbers, against the storage floor and the 1e-06 rad consumer target.
* **C.** Does ``h^4 |phi''''| / 384`` predict the measurement? The criterion has to earn its place.
* **D.** Two candidate criteria, costed: samples at fixed accuracy and accuracy at fixed samples,
  against the uniform grid.
* **E.** The response grid (it must stay a decimation of the source grid) and the ``k tau``
  oscillation an Omega_GW post-processing step would have to resolve.

No Ray, no datastore. ~90 s for the default set.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py
    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py \
        --k 1e5 --models QCDModel --json /tmp/density.json
"""

import argparse
import json
import sys
import time
from math import expm1, fabs, log, log10, log1p, sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).parent))

from scipy.interpolate import make_interp_spline  # noqa: E402

from ComputeTargets.BackgroundModel import (  # noqa: E402
    DERIVATIVE_FIT_PAD_FLOOR,
    DERIVATIVE_FIT_PAD_FRACTION,
    DERIVATIVE_FIT_PAD_POINTS,
    DERIVATIVE_FIT_REFINE,
    _cosmology_break_points,
)
from ComputeTargets.phase_residual import (  # noqa: E402
    build_phase_residual,
    phase_residual_integrand,
    residual_node_range,
)
from ComputeTargets.tests.test_gk_wkb_phase import (  # noqa: E402
    lambdacdm_model_with_tables,
    qcd_model_with_tables,
)
from ComputeTargets.tests.test_main_plumbing import (  # noqa: E402
    load_main_py_functions,
)
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    PRODUCTION_RESPONSE_SPARSENESS,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_Z_END,
    load_references,
    to_redshift_array,
)
from CosmologyConcepts import (  # noqa: E402
    SOURCE_GRID_BREAK_HALF_WIDTH,
    SOURCE_GRID_BREAK_REFINEMENT,
    SOURCE_GRID_BREAK_STANDOFF,
    build_z_sample,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology  # noqa: E402
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018  # noqa: E402
from Units import Mpc_units  # noqa: E402

K_VALUES = (1.0e5, 1.0e7, 3.0e8)
MODEL_KEYS = ("LambdaCDMModel", "QCDModel")
SECTORS = ("Gk", "Tk")
SPLINE_ORDER = 3

# points inside each production interval at which a candidate spline is scored against the oracle
SCORE_POINTS_PER_INTERVAL = 4

# the cubic-interpolation error constant: max |e| ~ CUBIC_ERROR_CONST * h^4 |f''''| on an interval
# of width h. 1/384 is the textbook bound for a local cubic through four equally spaced points;
# section C measures the constant that is actually realised and does not assume this one.
CUBIC_ERROR_CONST = 1.0 / 384.0

# step in u for the numerical third derivative of the closed-form phi'. Roundoff in the 5-point
# stencil is ~ eps |g| / (2 delta^3) ~ 5e-8 |g| here and truncation ~ delta^2 |g^(5)|; both are
# far below anything a density criterion resolves.
DERIV_STEP_U = 1.0e-3

# the declared crossings of QCD_EOS, in u = log(1+z); prompt 07's bisected values, transcribed
# from logs/07-rederive-break-points.md. A criterion built on derivatives of H is meaningless
# inside a few steps of a genuine step in H, and those neighbourhoods are prompt 11's business.
U_CROSSINGS = (17.565806941870026, 23.197460552819653, 27.485391822044257)
CROSSING_MASK_U = 6.0 * DERIV_STEP_U

# How far either side of a declared crossing this prompt declines to answer. 0.15 in u is about
# 6.5 production grid intervals, comfortably outside prompt 11's +-5-interval neighbourhood: what
# happens inside is that prompt's local mechanism (a straddling pair and a 2x refinement placed on
# a rule), not a density set by a curvature.
CROSSING_HALO_U = 0.15

# 1 ulp of the stored phase at each reference wavenumber: the floor of
# [02-consumer-phi-below-the-storage-granularity], from PrimitivePhase's module docstring and
# docs/gk-wkb-review-fable-2026-09-09.md §1 / §13.4. Recomputed per case from the span below;
# these are the quoted figures and are printed beside the computed ones.
QUOTED_ULP_FLOOR = {1.0e5: 3.05e-07, 1.0e7: 3.05e-05, 3.0e8: 9.15e-04}

# prompt 10 §5's consumer target
CONSUMER_TARGET_RAD = 1.0e-06

# No candidate criterion may be more than this many times coarser than today's uniform spacing.
# The cap is not cosmetic: a criterion that divides by |phi'| takes an unbounded step wherever phi
# is stationary, and the source grid carries far more than phi -- the numeric ODE's samples, the
# cumulative tables' panels and the source integral's abscissae all live on it, and none of those
# requirements is measured here.
H_MAX_FACTOR = 10.0

# the coarsening the user would actually be choosing between, as multiples of today's spacing
CAP_LADDER = (1.0, 2.0, 4.0, 8.0, 32.0)

# below this the measured error is the oracle's own rounding (rho is O(0.1-1) rad, so a double
# carries it to ~1e-17) rather than an interpolation error, and section C's indicator is scored
# only where both sides are above it
ORACLE_NOISE_RAD = 1.0e-15

cosmology_feature_redshifts = load_main_py_functions(
    ["cosmology_feature_redshifts"],
    extra_globals={"np": np, "_cosmology_break_points": _cosmology_break_points},
)["cosmology_feature_redshifts"]


# ---------------------------------------------------------------------------------------------
# the grids
# ---------------------------------------------------------------------------------------------


def production_grids(cosmology):
    """``(base, shipped)`` as descending float arrays: the uniform grid as it was before prompt
    11, and the cosmology-aware grid as prompt 11 ships it."""
    references = load_references()
    z_init = float(references["models"]["LambdaCDMModel"]["grid"]["z_init"])
    base = build_z_sample(
        z_init, PRODUCTION_Z_END, PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z
    ).z_values
    break_z, feature_z = cosmology_feature_redshifts(
        cosmology, PRODUCTION_Z_END, z_init
    )
    grid = build_z_sample(
        z_init,
        PRODUCTION_Z_END,
        PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
        standoff=SOURCE_GRID_BREAK_STANDOFF,
        half_width=SOURCE_GRID_BREAK_HALF_WIDTH,
        refinement=SOURCE_GRID_BREAK_REFINEMENT,
    )
    return base, grid, np.asarray(break_z, dtype=float)


# ---------------------------------------------------------------------------------------------
# one (model, sector, k): the oracle, the closed form, and the scoring set
# ---------------------------------------------------------------------------------------------


class Case:
    """The oracle and the pre-grid closed form for one (model, sector, k)."""

    def __init__(self, model, model_key, sector, k, grid_z, breaks_u=()):
        self.model, self.model_key, self.sector, self.k = model, model_key, sector, k
        self.cosmology = model.cosmology
        self.breaks_u = np.asarray(breaks_u, dtype=float)

        nodes = np.asarray(residual_node_range(model, k, grid_z, sector), dtype=float)
        self.table = build_phase_residual(model, k, nodes, sector)
        self.integrand = phase_residual_integrand(model, k, sector)
        self.z_top = float(nodes[0])
        self.nodes = nodes
        self.u_node = np.log1p(nodes[::-1])  # ascending in u

        # the scoring set: SCORE_POINTS_PER_INTERVAL strictly inside every production interval,
        # and the oracle's value at each. Shared by every candidate grid.
        u_score = []
        for i in range(len(self.u_node) - 1):
            a, b = self.u_node[i], self.u_node[i + 1]
            for j in range(1, SCORE_POINTS_PER_INTERVAL + 1):
                u_score.append(a + (b - a) * j / (SCORE_POINTS_PER_INTERVAL + 1.0))
        self.u_score = np.array(u_score)
        self.phi_score = np.array([self.phi(u) for u in self.u_score])
        self.phi_node = np.array([self.phi(u) for u in self.u_node])

        # the span of the stored phase, and therefore the storage floor this k sits on
        self.span = fabs(k * self._leading_span())
        self.ulp = float(np.spacing(self.span))

        # A candidate grid puts samples anywhere, and an exact oracle call costs ~0.1 ms, so a
        # ladder of thirty candidates of 10^4 samples would cost minutes of table quadrature.
        # `phi_fast` is an order-5 interpolant of the exact values already computed above, whose
        # sites are SCORE_POINTS_PER_INTERVAL+1 times denser than the production grid; against a
        # cubic on the production spacing its own error is of order (1/5)^6 h^2, i.e. eight orders
        # down. `phi_fast_error` below is that claim measured rather than asserted.
        u_all = np.concatenate([self.u_node, self.u_score])
        phi_all = np.concatenate([self.phi_node, self.phi_score])
        order = np.argsort(u_all)
        self._phi_fast = make_interp_spline(u_all[order], phi_all[order], k=5)

        # the control: phi_fast against the exact table at points it was not built on
        probe = 0.5 * (self.u_score[:-1] + self.u_score[1:])[::37]
        diff = np.abs(self._phi_fast(probe) - np.array([self.phi(u) for u in probe]))
        away = self._away(probe)
        self.phi_fast_error = float(diff.max())
        self.phi_fast_error_away = float(diff[away].max()) if away.any() else 0.0

    def _away(self, u, width: float = CROSSING_HALO_U):
        """True where ``u`` is more than ``width`` in ``u`` from every declared crossing."""
        u = np.asarray(u, dtype=float)
        ok = np.ones_like(u, dtype=bool)
        for b in self.breaks_u:
            ok &= np.abs(u - b) > width
        return ok

    # -- the oracle -----------------------------------------------------------------------
    def phi(self, u: float) -> float:
        """``rho`` at ``u = log(1+z)``, from the residual table's exact interval accessor."""
        return self.table.delta(self.z_top, expm1(float(u)))

    def _leading_span(self) -> float:
        lead = (
            self.model.functions.tau
            if self.sector == "Gk"
            else self.model.functions.cs_tau
        )
        return lead.delta(float(self.nodes[0]), float(self.nodes[-1]))

    # -- the pre-grid closed form ---------------------------------------------------------
    def dphi_du(self, u: float) -> float:
        """``d phi / du`` from ``H``, ``c_s^2`` and ``k`` alone -- no grid, no stored phase."""
        z = expm1(float(u))
        return -(1.0 + z) * self.integrand(z)

    def d4phi_du4(self, u: float, delta: float = DERIV_STEP_U) -> float:
        """The fourth derivative of ``phi``, i.e. the third of the closed-form ``phi'``.

        The stencil is slid inwards at the two ends of the band rather than made one-sided: the
        model's derivative accessors refuse an evaluation outside the grid, and a criterion that
        cannot be evaluated in the last two steps of its own range is no use.
        """
        u = min(
            max(float(u), float(self.u_node[0]) + 2.0 * delta),
            float(self.u_node[-1]) - 2.0 * delta,
        )
        g = self.dphi_du
        return (
            -g(u - 2.0 * delta)
            + 2.0 * g(u - delta)
            - 2.0 * g(u + delta)
            + g(u + 2.0 * delta)
        ) / (2.0 * delta**3)

    # -- scoring --------------------------------------------------------------------------
    def score(self, u_fit, exact: bool = False, away_only: bool = False):
        """Max and median |spline(phi) - phi| over the scoring set, for a candidate grid.

        ``exact=True`` takes the candidate's own ordinates from the table rather than from
        ``phi_fast`` -- used for the shipped grid, whose samples are the production nodes and
        whose error is quoted on its own. ``away_only=True`` restricts the scoring set to points
        away from a declared crossing: the crossing neighbourhoods are prompt 11's local
        mechanism and are not what a density criterion is answering.
        """
        u_fit = np.asarray(sorted(float(v) for v in u_fit), dtype=float)
        phi_fit = (
            np.array([self.phi(u) for u in u_fit]) if exact else self._phi_fast(u_fit)
        )
        spline = make_interp_spline(u_fit, phi_fit, k=SPLINE_ORDER)
        inside = (self.u_score > u_fit[0]) & (self.u_score < u_fit[-1])
        if away_only:
            inside &= self._away(self.u_score)
        err = np.abs(spline(self.u_score[inside]) - self.phi_score[inside])
        return {
            "n": int(len(u_fit)),
            "max": float(err.max()),
            "median": float(np.median(err)),
            "max_u": float(self.u_score[inside][int(err.argmax())]),
            "err": err,
            "u": self.u_score[inside],
        }


def masked(u, breaks_u):
    """True where ``u`` is far enough from a declared crossing for a derivative of ``H`` to
    mean anything."""
    ok = np.ones_like(np.asarray(u, dtype=float), dtype=bool)
    for b in breaks_u:
        ok &= np.abs(np.asarray(u, dtype=float) - b) > CROSSING_MASK_U
    return ok


# ---------------------------------------------------------------------------------------------
# the candidate criteria
# ---------------------------------------------------------------------------------------------
#
# Both are of the form h(u) = F(scale) * g(u): the *shape* g is a property of the cosmology, the
# wavenumber and the sector alone, and the scale is the one number a user would turn. So the shape
# is computed once per case and the scale is bisected against it, which is what makes a ladder of
# thirty candidate grids affordable.


def criterion_shapes(case, u, breaks_u):
    """``(d1, d4)`` on ``u``: ``|phi'|`` and ``|phi''''|`` from the closed form, with the
    neighbourhood of every declared crossing filled in by log-interpolation from either side.

    The mask is not a convenience. ``C(z)`` is built from derivatives of ``H``, and ``H`` has a
    genuine **step** at each declared crossing (audit section 2), so no derivative of it means
    anything within a few steps of one. Those neighbourhoods are prompt 11's business -- it puts a
    straddling pair and a refined neighbourhood there on a rule, not on a curvature -- and this
    criterion is about the density everywhere else.
    """
    u = np.asarray(u, dtype=float)
    d1 = np.array([fabs(case.dphi_du(float(v))) for v in u])
    d4 = np.array([fabs(case.d4phi_du4(float(v))) for v in u])
    ok = masked(u, breaks_u) & (d1 > 0.0) & (d4 > 0.0)
    if not ok.all() and ok.sum() >= 2:
        d1 = np.exp(np.interp(u, u[ok], np.log(d1[ok])))
        d4 = np.exp(np.interp(u, u[ok], np.log(d4[ok])))
    return d1, d4


def spacing_profile(kind, d1, d4, scale):
    """``h(u)`` for one criterion at one scale.

    * ``CURV`` -- equidistribute the cubic interpolation error:
      ``h = (eps / (CUBIC_ERROR_CONST |phi''''|))^(1/4)``. Four derivatives of ``H``.
    * ``SLOPE`` -- equidistribute ``phi`` itself: ``h |phi'| = delta``, an equal increment of the
      residual phase per interval. One derivative, and no numerical differentiation at all.
    """
    if kind == "CURV":
        return (scale / (CUBIC_ERROR_CONST * np.maximum(d4, 1e-300))) ** 0.25
    if kind == "SLOPE":
        return scale / np.maximum(d1, 1e-300)
    raise ValueError(kind)


class TooFine(RuntimeError):
    """A spacing profile that would need more samples than any grid could carry."""


MAX_MARCH_POINTS = 200000


def march(u_lo, u_hi, u_prof, h_prof, h_min=1.0e-6, h_max=2.0):
    """Build a grid from ``u_lo`` to ``u_hi`` by stepping ``h(u)``, clamped to ``[h_min, h_max]``.

    A forward Euler on ``du/di = h(u)``, which is what any practical ``populate_z_sample`` would
    do; it is deliberately not iterated to a fixed point, so the counts quoted below are the counts
    of the simple implementation and not of an optimal one.
    """
    log_h = np.log(np.maximum(h_prof, 1e-300))
    out = [float(u_lo)]
    u = float(u_lo)
    while u < u_hi:
        h = min(max(float(np.exp(np.interp(u, u_prof, log_h))), h_min), h_max)
        u = u + h
        out.append(u)
        if len(out) > MAX_MARCH_POINTS:
            raise TooFine(f"march: more than {MAX_MARCH_POINTS} samples")
    out[-1] = float(u_hi)
    return np.array(out)


def tune_to_target(case, kind, d1, d4, u_prof, u_lo, u_hi, target, h_max=2.0, iters=34):
    """The largest scale whose grid still meets ``target``, by bisection in ``log scale``.

    Both criteria are monotone -- a larger scale is a coarser grid and a larger error -- so a
    bisection is exact up to the resolution of the ladder.
    """
    lo, hi = 1.0e-36, 1.0e6
    best = None
    for _ in range(iters):
        mid = sqrt(lo * hi)
        try:
            cand = march(
                u_lo, u_hi, u_prof, spacing_profile(kind, d1, d4, mid), h_max=h_max
            )
        except TooFine:
            lo = mid  # too fine to build: coarsen
            continue
        if len(cand) < 8:
            hi = mid
            continue
        r = case.score(cand, away_only=True)
        if r["max"] <= target:
            best = (mid, r)
            lo = mid
        else:
            hi = mid
    return best


def tune_to_count(
    case, kind, d1, d4, u_prof, u_lo, u_hi, n_target, h_max=2.0, iters=38
):
    """The scale whose grid has about ``n_target`` samples, and what it achieves."""
    lo, hi = 1.0e-36, 1.0e6
    best = None
    for _ in range(iters):
        mid = sqrt(lo * hi)
        try:
            cand = march(
                u_lo, u_hi, u_prof, spacing_profile(kind, d1, d4, mid), h_max=h_max
            )
        except TooFine:
            lo = mid
            continue
        if len(cand) >= 8:
            best = (mid, cand)
        if len(cand) > n_target:
            lo = mid
        else:
            hi = mid
    if best is None:
        return None, None
    scale, cand = best
    return scale, case.score(cand, away_only=True)


def uniform_to_target(case, u_lo, u_hi, target, in_log10z: bool):
    """The fewest uniform samples that meet ``target``: uniform in ``u = log(1+z)``, or uniform in
    ``log10 z``, which is what ``populate_z_sample`` actually builds."""

    def grid(n):
        if in_log10z:
            z_lo, z_hi = expm1(u_lo), expm1(u_hi)
            return np.log1p(np.logspace(log10(z_lo), log10(z_hi), num=n))
        return np.linspace(u_lo, u_hi, n)

    lo, hi = 8, 200000
    if case.score(grid(hi), away_only=True)["max"] > target:
        return hi, case.score(grid(hi), away_only=True)
    while lo < hi:
        mid = (lo + hi) // 2
        if case.score(grid(mid), away_only=True)["max"] <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo, case.score(grid(lo), away_only=True)


def derivative_pad(u_grid):
    """``h_lo``, and which of its three terms binds, for ``_build_derivative_fit_grid``.

    A transcription of ``ComputeTargets/BackgroundModel.py:99-104`` -- ``h_lo = min(first refined
    interval, -log(FLOOR)/pad, FRACTION*(span)/pad)`` in ``x = log(1+z)``. This is
    ``[03-derivative-pad-clamp-on-coarse-grids]``, whose whole content is that the clamp is
    harmless at the shipped 100 samples per decade and binds on a coarser grid; any recommendation
    to coarsen has to say where it lands.
    """
    x = np.sort(np.asarray(u_grid, dtype=float))
    refine = max(int(DERIVATIVE_FIT_REFINE), 1)
    first = (x[1] - x[0]) / refine
    floor_cap = -log(DERIVATIVE_FIT_PAD_FLOOR) / DERIVATIVE_FIT_PAD_POINTS
    frac_cap = DERIVATIVE_FIT_PAD_FRACTION * (x[-1] - x[0]) / DERIVATIVE_FIT_PAD_POINTS
    h_lo = min(first, floor_cap, frac_cap)
    binds = h_lo < first - 0.0
    return {
        "first": float(first),
        "floor_cap": float(floor_cap),
        "fraction_cap": float(frac_cap),
        "h_lo": float(h_lo),
        "clamped": bool(binds),
    }


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def decade_rows(case, res, floor, breaks_u):
    """Per-decade breakdown of a scored grid, with the declared crossings separated out: the
    crossing neighbourhoods are prompt 11's mechanism, not the density's."""
    z = np.expm1(res["u"])
    dec = np.floor(np.log10(np.maximum(z, 1e-30))).astype(int)
    away = case._away(res["u"])
    rows = []
    for d in sorted(set(dec.tolist())):
        m = dec == d
        u_d = res["u"][m]
        i_lo = int(np.searchsorted(case.u_node, u_d.min()))
        i_hi = int(np.searchsorted(case.u_node, u_d.max()))
        h = float(np.median(np.diff(case.u_node[max(i_lo - 1, 0) : i_hi + 1])))
        m_away = m & away
        rows.append(
            {
                "decade": int(d),
                "h_u": h,
                "n_int": int(i_hi - i_lo + 1),
                "max": float(res["err"][m].max()),
                "max_away": float(res["err"][m_away].max()) if m_away.any() else 0.0,
                "median": float(np.median(res["err"][m])),
                "has_crossing": bool(m.sum() != m_away.sum()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    parser.add_argument("--k", nargs="+", type=float, default=list(K_VALUES))
    parser.add_argument("--sectors", nargs="+", default=list(SECTORS))
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="which sections to run",
    )
    args = parser.parse_args()
    want = set(s.upper() for s in args.sections)

    t_start = time.perf_counter()
    print("=" * 102)
    print("SOURCE-GRID DENSITY CRITERION -- prompt 12, prompts/qcd-background-audit")
    print("=" * 102)

    references = load_references()
    z_init = float(references["models"]["LambdaCDMModel"]["grid"]["z_init"])

    out = {}
    built, built_base, grids, cases = {}, {}, {}, []

    # The cosmology exists before the grid -- which is the premise of the whole criterion -- so
    # each model's grid is built from its own cosmology first, and its tables are then built on
    # the grid it actually ships.
    for model_key in args.models:
        cosmology = (
            LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
            if model_key == "LambdaCDMModel"
            else QCD_Cosmology(
                store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
            )
        )
        base, shipped, break_z = production_grids(cosmology)
        if model_key == "LambdaCDMModel":
            model = lambdacdm_model_with_tables(shipped.z_values)
            model_base = model
        else:
            model = qcd_model_with_tables(to_redshift_array(shipped.z_values))
            model_base = qcd_model_with_tables(to_redshift_array(base))
        built[model_key] = model
        built_base[model_key] = model_base
        grids[model_key] = (base, shipped, break_z)
        print(
            f"   [{model_key}] base grid {len(base)}, shipped grid "
            f"{len(shipped.z_values)}, declared crossings in range {len(break_z)}"
        )

    for model_key in args.models:
        model = built[model_key]
        _, shipped, _ = grids[model_key]
        for sector in args.sectors:
            for k in args.k:
                t0 = time.perf_counter()
                bu = np.log1p(grids[model_key][2]) if len(grids[model_key][2]) else ()
                case = Case(model, model_key, sector, k, shipped.z_values, breaks_u=bu)
                cases.append(case)
                print(
                    f"   [{model_key} {sector} k={k:.1e}] band z "
                    f"{case.nodes[0]:.4e} .. {case.nodes[-1]:.4e}, "
                    f"{len(case.nodes)} grid samples, k*tau span {case.span:.4e} rad, "
                    f"1 ulp = {case.ulp:.4e} rad; phi_fast control {case.phi_fast_error:.2e} rad "
                    f"({case.phi_fast_error_away:.2e} away from a crossing)"
                    f"  ({time.perf_counter() - t0:.2f} s)"
                )

    def breaks_of(case):
        return (
            np.log1p(grids[case.model_key][2])
            if len(grids[case.model_key][2])
            else np.empty(0)
        )

    # =========================================================================================
    if "A" in want:
        print("\n" + "=" * 102)
        print(
            "A. THE CRITERION'S INPUTS ARE PRE-GRID (the licence for everything below)"
        )
        print("=" * 102)
        print(
            "\n   A1. the closed-form phi' against the residual table's own derivative"
        )
        print(
            f"   {'model':<14}{'sector':<7}{'k':<11}{'u':<10}"
            f"{'numeric phi(u)':>18}{'closed form':>18}{'ratio-1':>12}"
        )
        a1 = []
        for case in cases:
            u_mid = float(case.u_node[len(case.u_node) // 2])
            d = 1.0e-4
            num = (case.phi(u_mid + d) - case.phi(u_mid - d)) / (2.0 * d)
            closed = case.dphi_du(u_mid)
            rel = num / closed - 1.0 if closed != 0.0 else float("nan")
            a1.append(
                dict(
                    model=case.model_key,
                    sector=case.sector,
                    k=case.k,
                    u=u_mid,
                    numeric=num,
                    closed=closed,
                    rel=rel,
                )
            )
            print(
                f"   {case.model_key:<14}{case.sector:<7}{case.k:<11.2e}{u_mid:<10.4f}"
                f"{num:>18.8e}{closed:>18.8e}{rel:>12.2e}"
            )

        print(
            "\n   A2. epsilon as the *model* supplies it (a spline over the grid's own samples on "
            "QCD) against\n       epsilon from a central difference of the cosmology's pointwise "
            "Hubble -- what a pre-grid\n       criterion would have to use"
        )
        a2 = {}
        for model_key in args.models:
            model = built[model_key]
            cosmology = model.cosmology
            _, _, break_z = grids[model_key]
            breaks_u = np.log1p(break_z) if len(break_z) else np.empty(0)

            def eps_fd(z, d=1.0e-6):
                u = log1p(z)
                return (
                    log(cosmology.Hubble(expm1(u + d)))
                    - log(cosmology.Hubble(expm1(u - d)))
                ) / (2.0 * d)

            u_probe = np.linspace(
                float(np.log1p(1.0e-1)) + 0.5, float(np.log1p(1.0e16)) - 0.5, 400
            )
            ok = (
                masked(u_probe, breaks_u)
                if len(breaks_u)
                else np.ones_like(u_probe, dtype=bool)
            )
            if len(breaks_u):
                far = np.ones_like(u_probe, dtype=bool)
                near_any = np.zeros_like(u_probe, dtype=bool)
                for b in breaks_u:
                    far &= np.abs(u_probe - b) > 0.5
                    near_any |= np.abs(u_probe - b) < 0.5
            else:
                far = np.ones_like(u_probe, dtype=bool)
                near_any = np.zeros_like(u_probe, dtype=bool)
            rel = []
            for u in u_probe:
                z = expm1(float(u))
                e_fd = eps_fd(z)
                rel.append(
                    fabs(model.functions.epsilon(z) / e_fd - 1.0) if e_fd != 0 else 0.0
                )
            rel = np.array(rel)
            a2[model_key] = {
                "max_far": float(rel[far].max()),
                "median_far": float(np.median(rel[far])),
                "max_near": float(rel[near_any].max()) if near_any.any() else None,
            }
            print(
                f"   {model_key:<14}away from a declared crossing (|du| > 0.5): "
                f"max {rel[far].max():.3e}, median {np.median(rel[far]):.3e}"
                + (
                    f"   |   within 0.5 of one: max {rel[near_any].max():.3e}"
                    if near_any.any()
                    else "   |   (declares none)"
                )
            )

        print(
            "\n   A3. the same comparison at a ladder of offsets from each declared crossing, on\n"
            "       the background built on the base grid and on the one built on the shipped grid.\n"
            "       epsilon is a spline of ln H over a refinement of the source grid and ln H has a\n"
            "       genuine STEP at each crossing, so this is a spline ringing at a step and not an\n"
            "       error of the cosmology, which is smooth on each branch."
        )
        a3 = {}
        for model_key in args.models:
            _, _, break_z = grids[model_key]
            if len(break_z) == 0:
                print(f"   {model_key}: declares no crossings; nothing to probe")
                continue
            cosmology = built[model_key].cosmology

            def eps_fd(z, d=1.0e-6):
                u = log1p(z)
                return (
                    log(cosmology.Hubble(expm1(u + d)))
                    - log(cosmology.Hubble(expm1(u - d)))
                ) / (2.0 * d)

            for b in np.log1p(np.asarray(break_z, dtype=float)):
                print(
                    f"\n   {model_key}, crossing at u = {b:.9f} (z = {expm1(float(b)):.6e}); "
                    f"one grid interval is {2.3032e-02:.4e} in u"
                )
                print(
                    f"      {'du':>10}{'eps, base grid':>18}{'eps, shipped':>18}"
                    f"{'eps, cosmology':>18}{'base rel':>12}{'shipped rel':>13}"
                )
                rows = []
                for du in (-0.20, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.20):
                    z = expm1(float(b) + du)
                    e_fd = eps_fd(z)
                    e_b = built_base[model_key].functions.epsilon(z)
                    e_s = built[model_key].functions.epsilon(z)
                    rows.append(
                        {
                            "du": du,
                            "base": e_b,
                            "shipped": e_s,
                            "cosmology": e_fd,
                            "base_rel": e_b / e_fd - 1.0,
                            "shipped_rel": e_s / e_fd - 1.0,
                        }
                    )
                    print(
                        f"      {du:>10.3f}{e_b:>18.8e}{e_s:>18.8e}{e_fd:>18.8e}"
                        f"{e_b / e_fd - 1.0:>12.2e}{e_s / e_fd - 1.0:>13.2e}"
                    )
                a3[f"{model_key} u={b:.9f}"] = rows
        out["A"] = {"closed_form": a1, "epsilon": a2, "epsilon_at_crossings": a3}

    # =========================================================================================
    shipped_scores = {}
    if "B" in want or "C" in want or "D" in want:
        for case in cases:
            shipped_scores[id(case)] = case.score(case.u_node, exact=True)

    if "B" in want:
        print("\n" + "=" * 102)
        print("B. WHAT THE SHIPPED GRID DELIVERS -- phi interpolation error per decade")
        print("=" * 102)
        print(
            "   'floor' is 1 ulp of the k*tau span -- the granularity of\n"
            "   [02-consumer-phi-below-the-storage-granularity], below which no grid decision is "
            "visible.\n"
            "   'away' excludes the 0.15 in u either side of a declared crossing: those "
            "neighbourhoods are\n   prompt 11's mechanism, not the density's. 'headroom' is "
            "floor / max(away)."
        )
        b_rows = {}
        for case in cases:
            res = shipped_scores[id(case)]
            floor, bu = case.ulp, breaks_of(case)
            rows = decade_rows(case, res, floor, bu)
            away_max = max(r["max_away"] for r in rows)
            print(
                f"\n   {case.model_key} {case.sector} k = {case.k:.3e}: "
                f"{res['n']} samples; max {res['max']:.4e} rad ({res['max'] / floor:.2f} ulp) "
                f"at z = {expm1(res['max_u']):.4e}; away from a crossing {away_max:.4e} rad "
                f"({away_max / floor:.2f} ulp); floor {floor:.4e} rad"
            )
            print(
                f"      {'decade':>7}{'h_u':>11}{'max err/rad':>14}{'away':>13}"
                f"{'median':>13}{'away/floor':>12}{'headroom':>12}"
            )
            for r in rows:
                mark = " *" if r["has_crossing"] else "  "
                print(
                    f"      {r['decade']:>7}{r['h_u']:>11.4e}{r['max']:>14.4e}"
                    f"{r['max_away']:>13.4e}{r['median']:>13.4e}"
                    f"{r['max_away'] / floor:>12.2e}"
                    f"{floor / max(r['max_away'], 1e-300):>12.2e}{mark}"
                )
            b_rows[f"{case.model_key} {case.sector} {case.k:.3e}"] = {
                "n": res["n"],
                "max": res["max"],
                "max_away": away_max,
                "max_z": expm1(res["max_u"]),
                "floor": floor,
                "span": case.span,
                "quoted_floor": QUOTED_ULP_FLOOR.get(case.k),
                "decades": rows,
            }
        print("\n   (* the decade holds a declared equation-of-state crossing)")
        out["B"] = b_rows

        # ---- B2: does the background move when the grid does? -----------------------------
        print("\n" + "-" * 102)
        print(
            "B2. THE GRID IS NOT ONLY A SAMPLE SET -- on QCD it is also the lattice the "
            "background's own\n    derivative fields are splined on "
            "(BackgroundModel._build_derivative_fit_grid). Four combinations."
        )
        print("-" * 102)
        b2 = {}
        for model_key in args.models:
            base, shipped, break_z = grids[model_key]
            if np.array_equal(np.asarray(base), np.asarray(shipped.z_values)):
                print(
                    f"   {model_key}: base and shipped grids identical; nothing to separate"
                )
                continue
            for sector in args.sectors:
                for k in args.k:
                    row = {}
                    for bg_tag, model in (
                        ("base", built_base[model_key]),
                        ("shipped", built[model_key]),
                    ):
                        c = Case(
                            model,
                            model_key,
                            sector,
                            k,
                            shipped.z_values,
                            breaks_u=np.log1p(break_z) if len(break_z) else (),
                        )
                        for s_tag, s_z in (
                            ("base", base),
                            ("shipped", shipped.z_values),
                        ):
                            u_fit = np.sort(
                                np.log1p(
                                    np.array(
                                        [
                                            z
                                            for z in s_z
                                            if c.nodes[-1] <= z <= c.nodes[0]
                                        ],
                                        dtype=float,
                                    )
                                )
                            )
                            r = c.score(u_fit, exact=True)
                            bu = np.log1p(break_z)
                            near = np.zeros_like(r["u"], dtype=bool)
                            for b in bu:
                                near |= np.abs(r["u"] - b) < 0.15
                            row[f"bg={bg_tag},samples={s_tag}"] = {
                                "n": r["n"],
                                "max": r["max"],
                                "max_near": (
                                    float(r["err"][near].max()) if near.any() else 0.0
                                ),
                                "max_away": float(r["err"][~near].max()),
                                "max_z": expm1(r["max_u"]),
                            }
                    print(
                        f"\n   {model_key} {sector} k = {k:.3e}   (1 ulp = {c.ulp:.4e} rad)"
                    )
                    print(
                        f"      {'background on':<16}{'samples from':<16}{'n':>7}"
                        f"{'max/rad':>13}{'near a crossing':>18}{'away':>13}"
                    )
                    for key, v in row.items():
                        bg, sm = key.split(",")
                        print(
                            f"      {bg.split('=')[1]:<16}{sm.split('=')[1]:<16}{v['n']:>7}"
                            f"{v['max']:>13.4e}{v['max_near']:>18.4e}{v['max_away']:>13.4e}"
                        )
                    b2[f"{model_key} {sector} {k:.3e}"] = row
        out["B2"] = b2

    # =========================================================================================
    if "C" in want:
        print("\n" + "=" * 102)
        print("C. DOES h^4 |phi''''| / 384 PREDICT THE MEASUREMENT?")
        print("=" * 102)
        print(
            "   Per production interval, away from a declared crossing and above the 1e-17 rad "
            "at which the\n   oracle itself runs out: the realised error against the predicted "
            "one. A usable indicator has a\n   narrow spread, not a ratio of one."
        )
        c_rows = {}
        for case in cases:
            bu = breaks_of(case)
            res = shipped_scores[id(case)]
            u_mid = 0.5 * (case.u_node[:-1] + case.u_node[1:])
            h = np.diff(case.u_node)
            ok = case._away(u_mid)
            d4 = np.array([fabs(case.d4phi_du4(float(v))) for v in u_mid])
            pred = CUBIC_ERROR_CONST * h**4 * d4
            real = np.zeros_like(pred)
            which = np.searchsorted(case.u_node, res["u"]) - 1
            for i in range(len(pred)):
                m = which == i
                if m.any():
                    real[i] = res["err"][m].max()
            good = ok & (real > ORACLE_NOISE_RAD) & (pred > ORACLE_NOISE_RAD)
            if int(good.sum()) < 10:
                print(
                    f"   {case.model_key:<14}{case.sector:<4}k={case.k:<10.2e}too few usable intervals"
                )
                continue
            ratio = real[good] / pred[good]
            p10, p50, p90 = (float(np.percentile(ratio, q)) for q in (10, 50, 90))
            print(
                f"   {case.model_key:<14}{case.sector:<4}k={case.k:<10.2e}"
                f"intervals {int(good.sum()):>5}   realised/predicted  "
                f"p10 {p10:8.3f}  median {p50:8.3f}  p90 {p90:8.3f}   "
                f"spread {p90 / max(p10, 1e-300):6.1f}x"
            )
            c_rows[f"{case.model_key} {case.sector} {case.k:.3e}"] = {
                "n": int(good.sum()),
                "p10": p10,
                "median": p50,
                "p90": p90,
            }
        out["C"] = c_rows

    # =========================================================================================
    if "D" in want:
        print("\n" + "=" * 102)
        print("D. TWO CANDIDATE CRITERIA, COSTED, PER CASE")
        print("=" * 102)
        print(
            "   target = min(1 ulp of the k*tau span, 1e-06 rad): below the first nothing is "
            "visible, and the\n   second is prompt 10 section 5's consumer target. 'samples' "
            "counts the band only, not the whole grid."
        )
        d_out, shapes = {}, {}
        for case in cases:
            bu = breaks_of(case)
            u_prof = case.u_node.copy()
            u_lo, u_hi = float(u_prof[0]), float(u_prof[-1])
            t0 = time.perf_counter()
            d1, d4 = criterion_shapes(case, u_prof, bu)
            t_shape = time.perf_counter() - t0
            shapes[id(case)] = (d1, d4, u_prof)
            floor = case.ulp
            target = min(floor, CONSUMER_TARGET_RAD)
            n_ship = len(u_prof)
            ship_max = shipped_scores[id(case)]["max"]
            print(
                f"\n   {case.model_key} {case.sector} k = {case.k:.3e}: target {target:.4e} rad; "
                f"shipped {n_ship} band samples at {ship_max:.4e} rad "
                f"(shape cost {t_shape:.2f} s for {5 * len(u_prof)} closed-form evaluations)"
            )
            print(
                f"      {'criterion':<22}{'scale':>12}{'samples':>9}{'max err/rad':>14}"
                f"{'/target':>11}{'vs shipped':>12}"
            )
            # No criterion is allowed to be more than H_MAX_FACTOR times coarser than today's
            # uniform spacing anywhere. Without a cap a criterion that divides by |phi'| takes one
            # unbounded step wherever phi is stationary, and the comparison is then about that
            # accident rather than about the criterion.
            h_cap = H_MAX_FACTOR * float(np.median(np.diff(case.u_node)))
            entry = {
                "target": target,
                "floor": floor,
                "shipped_n": n_ship,
                "shipped_max": ship_max,
                "shape_seconds": t_shape,
                "h_cap": h_cap,
            }
            for kind in ("CURV", "SLOPE"):
                best = tune_to_target(
                    case, kind, d1, d4, u_prof, u_lo, u_hi, target, h_max=h_cap
                )
                if best is None:
                    print(
                        f"      {kind + ' @target':<22}{'--':>12}{'--':>9}{'never meets it':>14}"
                    )
                    continue
                scale, r = best
                print(
                    f"      {kind + ' @target':<22}{scale:>12.3e}{r['n']:>9}{r['max']:>14.4e}"
                    f"{r['max'] / target:>11.2f}{n_ship / r['n']:>11.2f}x"
                )
                scale_n, r_n = tune_to_count(
                    case, kind, d1, d4, u_prof, u_lo, u_hi, n_ship, h_max=h_cap
                )
                if r_n is None:
                    entry[kind] = {"scale": scale, "n": r["n"], "max": r["max"]}
                    continue
                print(
                    f"      {kind + ' @shipped n':<22}{scale_n:>12.3e}{r_n['n']:>9}"
                    f"{r_n['max']:>14.4e}{r_n['max'] / target:>11.2e}"
                    f"{ship_max / max(r_n['max'], 1e-300):>11.2f}x"
                )
                entry[kind] = {"scale": scale, "n": r["n"], "max": r["max"]}
                entry[kind + "_at_n"] = {
                    "scale": scale_n,
                    "n": r_n["n"],
                    "max": r_n["max"],
                }
            for tag, in_log10z in (
                ("UNIFORM in u", False),
                ("UNIFORM in log10 z", True),
            ):
                n_u, r_u = uniform_to_target(case, u_lo, u_hi, target, in_log10z)
                print(
                    f"      {tag + ' @target':<22}{'--':>12}{n_u:>9}{r_u['max']:>14.4e}"
                    f"{r_u['max'] / target:>11.2f}{n_ship / n_u:>11.2f}x"
                )
                entry[tag] = {"n": n_u, "max": r_u["max"]}
            d_out[f"{case.model_key} {case.sector} {case.k:.3e}"] = entry
        out["D"] = d_out

        # ---- D2: one grid for the whole production set ------------------------------------
        print("\n" + "-" * 102)
        print(
            "D2. ONE GRID FOR THE WHOLE PRODUCTION SET -- main.py builds a single universal source\n"
            "    grid, so the criterion must be the envelope over every (model, sector, k) it "
            "serves."
        )
        print("-" * 102)
        d2 = {}
        for model_key in args.models:
            base, shipped, break_z = grids[model_key]
            mine = [c for c in cases if c.model_key == model_key]
            if not mine:
                continue
            u_all = np.log1p(np.asarray(shipped.z_values, dtype=float)[::-1])
            u_lo, u_hi = float(u_all[0]), float(u_all[-1])
            h_today = float(np.max(np.diff(u_all)))
            for kind in ("CURV", "SLOPE"):
                # The envelope: each case constrains only its own band and each is tuned to its
                # own target, so the k and the sector that bind are measured rather than assumed.
                # The per-case scale does not depend on the cap, so it is found once.
                h_env_raw = np.full_like(u_all, np.inf)
                per_case_scale = {}
                for c in mine:
                    d1, d4, u_prof = shapes[id(c)]
                    target = min(c.ulp, CONSUMER_TARGET_RAD)
                    best = tune_to_target(
                        c,
                        kind,
                        d1,
                        d4,
                        u_prof,
                        float(u_prof[0]),
                        float(u_prof[-1]),
                        target,
                        h_max=H_MAX_FACTOR * float(np.median(np.diff(c.u_node))),
                    )
                    if best is None:
                        continue
                    scale = best[0]
                    per_case_scale[f"{c.sector} {c.k:.1e}"] = scale
                    h_c = spacing_profile(kind, d1, d4, scale)
                    h_on_all = np.exp(np.interp(u_all, u_prof, np.log(h_c)))
                    inside = (u_all >= u_prof[0]) & (u_all <= u_prof[-1])
                    h_env_raw = np.where(
                        inside, np.minimum(h_env_raw, h_on_all), h_env_raw
                    )
                if not np.isfinite(h_env_raw).any():
                    print(f"\n   {model_key} {kind}: no case admits a scale; skipped")
                    continue
                print(
                    f"\n   {model_key} {kind}: binding scale per case "
                    + ", ".join(f"{a} {b:.3e}" for a, b in per_case_scale.items())
                )
                for cap_factor in CAP_LADDER:
                    h_cap = cap_factor * h_today
                    h_env = np.where(np.isfinite(h_env_raw), h_env_raw, h_cap)
                    h_env = np.minimum(h_env, h_cap)
                    grid_u = march(u_lo, u_hi, u_all, h_env, h_max=h_cap)
                    print(
                        f"\n   {model_key} {kind}, spacing capped at {cap_factor:g}x today's "
                        f"{h_today:.4e} in u: one grid of {len(grid_u)} samples against the shipped "
                        f"{len(shipped.z_values)} ({len(shipped.z_values) / len(grid_u):.2f}x fewer)"
                    )
                    scored = {}
                    for c in mine:
                        u_fit = grid_u[
                            (grid_u >= c.u_node[0]) & (grid_u <= c.u_node[-1])
                        ]
                        u_fit = np.unique(
                            np.concatenate([[c.u_node[0]], u_fit, [c.u_node[-1]]])
                        )
                        r = c.score(u_fit, away_only=True)
                        tgt = min(c.ulp, CONSUMER_TARGET_RAD)
                        a_max = float(r["max"])
                        scored[f"{c.sector} {c.k:.1e}"] = {
                            "n": r["n"],
                            "max_away": a_max,
                            "target": tgt,
                            "ratio": a_max / tgt,
                            "shipped_n": int(len(c.u_node)),
                        }
                        print(
                            f"      {c.sector:<4}k={c.k:<10.2e}{r['n']:>6} samples in band "
                            f"(shipped {len(c.u_node)})   max away from a crossing {a_max:.4e}   "
                            f"target {tgt:.4e}   ratio {a_max / tgt:6.2f}"
                        )
                    src = to_redshift_array([float(expm1(u)) for u in grid_u[::-1]])
                    resp = src.winnow(sparseness=PRODUCTION_RESPONSE_SPARSENESS)
                    subset = set(np.array(resp.as_float_list()).tolist()) <= set(
                        np.array(src.as_float_list()).tolist()
                    )
                    pad = derivative_pad(grid_u)
                    print(
                        f"      response grid by the production stride {PRODUCTION_RESPONSE_SPARSENESS}: "
                        f"{len(resp)} samples, a subset of the source grid: {subset}; "
                        f"derivative pad h_lo {pad['h_lo']:.4e} "
                        f"(first {pad['first']:.4e}, floor cap {pad['floor_cap']:.4e}) -- "
                        f"[03-derivative-pad-clamp-on-coarse-grids] "
                        f"{'BINDS' if pad['clamped'] else 'does not bind'}"
                    )
                    d2[f"{model_key} {kind} cap{cap_factor:g}"] = {
                        "cap_factor": cap_factor,
                        "pad": pad,
                        "n": int(len(grid_u)),
                        "shipped_n": int(len(shipped.z_values)),
                        "scales": per_case_scale,
                        "scored": scored,
                        "response_n": int(len(resp)),
                        "response_is_subset": bool(subset),
                    }
        out["D2"] = d2

    # =========================================================================================
    if "E" in want:
        print("\n" + "=" * 102)
        print(
            "E. THE RESPONSE GRID, AND THE k tau OSCILLATION (audit section 9, second bullet)"
        )
        print("=" * 102)
        e_out = {}
        for model_key in args.models:
            model = built[model_key]
            _, shipped, _ = grids[model_key]
            # a *blind* stride, as `wkb_reference.production_response_grid` takes it: prompt 11's
            # production call adds `protect=`, which puts QCD's response grid at 156 rather than
            # the 148 here. Nothing below turns on the difference -- the shortfall is measured in
            # powers of ten -- and the blind stride is the like-for-like comparison with LambdaCDM.
            src = to_redshift_array([float(z) for z in shipped.z_values])
            resp = src.winnow(sparseness=PRODUCTION_RESPONSE_SPARSENESS)
            resp_z = np.array(resp.as_float_list(), dtype=float)
            subset = set(resp_z.tolist()) <= set(np.asarray(shipped.z_values).tolist())
            tau = model.functions.tau
            total = tau.delta(float(shipped.z_values[0]), float(shipped.z_values[-1]))
            per_k = {}
            for k in args.k:
                d = np.array(
                    [
                        fabs(k * tau.delta(float(resp_z[i]), float(resp_z[i + 1])))
                        for i in range(len(resp_z) - 1)
                    ]
                )
                need = 4.0 * fabs(k * total) / (2.0 * np.pi)
                per_k[f"{k:.3e}"] = {
                    "max_dphase": float(d.max()),
                    "median_dphase": float(np.median(d)),
                    "total_ktau": float(fabs(k * total)),
                    "samples_for_4_per_cycle": float(need),
                }
                print(
                    f"   {model_key:<14}k={k:<11.2e}k*tau over the range {fabs(k * total):.4e} rad; "
                    f"per response interval max {d.max():.4e} rad, median {np.median(d):.4e}; "
                    f"4 samples/cycle would need {need:.3e} response samples "
                    f"({need / len(resp_z):.2e}x the {len(resp_z)} it has)"
                )
            e_out[model_key] = {
                "source_n": int(len(shipped.z_values)),
                "response_n": int(len(resp_z)),
                "response_is_subset": bool(subset),
                "k": per_k,
            }
            print(
                f"   {model_key:<14}response grid is a subset of the source grid: {subset}"
            )
        out["E"] = e_out

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"raw results written to {args.json}")


if __name__ == "__main__":
    main()
