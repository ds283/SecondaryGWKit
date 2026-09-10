import time
from math import log, exp, pow, sqrt, pi, gamma, inf
from typing import Optional, List, Union, Callable, Sequence, Tuple

import ray
from ray import ObjectRef
from scipy.special import yv, jv

from AdaptiveLevin import adaptive_levin_sincos
from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy, ModelFunctions
from ComputeTargets.GkSource import GkSource
from ComputeTargets.GkSourcePolicyData import GkSourcePolicyData, GkSourceFunctions
from ComputeTargets.QuadSource import QuadSource
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.phase_groups import build_phase_groups
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance, GkSourcePolicy
from Quadrature.integration_metadata import IntegrationData, LevinData
from Quadrature.simple_quadrature import simple_quadrature
from config.defaults import (
    DEFAULT_QUADRATURE_RTOL,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_QUADRATURE_ATOL,
)
from utilities import WallclockTimer

# NO QUADRATURE GATE LIVES IN THIS MODULE. Until prompt 08 of prompts/source-remediation (audit
# QI-6) the Levin quadrature of the source time integral was gated on the *Green's-function*
# phase alone: GkSourcePolicyData.Levin_z (where |d theta_G/d log(1+z)| first exceeds
# Levin_threshold) and the net phase change theta_G(z_response) - theta_G(Levin_z) >
# LEVIN_MIN_PHASE_DIFF = 10 * 2 pi. Neither gate consulted T_q or T_r, so a region in which
# theta_q - theta_r turned over thousands of cycles could be classified "not worth Levin" and
# handed to scipy.quad. Every sub-interval with at least one oscillatory factor now goes to
# adaptive_levin_sincos, whose total-variation gate (AdaptiveLevin/levin_quadrature.py,
# docs/adaptive-levin-verification.md section 4.2) routes weakly oscillatory sub-regions to
# Clenshaw-Curtis itself; GkPolicy.Levin_z is still read (it is persisted) and recorded in
# metadata["partition"], but drives nothing.
#
# Prompt 10 removed the two retired constants (LEVIN_MIN_2PI_CYCLES, LEVIN_MIN_PHASE_DIFF) rather
# than reviving them through MetadataConcepts/QuadSourcePolicy.Levin_threshold. Prompt 08 section 6
# measured the weakly oscillatory case the old gate used to divert (theta_G turning over 3.5-5.4
# cycles): the single-group Levin call is 2.65-3.56x slower in wall-clock than the old direct
# quadrature while doing only 1.00-1.44x the integrand evaluations, i.e. the excess is the Levin
# driver's per-region overhead once its own gate has already sent every region to Clenshaw-Curtis.
# The author's decision was to accept that cost here and make the *driver* choose its fallback
# better -- routing a weakly oscillatory integrand to Clenshaw-Curtis wholesale at the outset
# instead of discovering it after many bisections -- which is work in AdaptiveLevin/, not a second
# threshold duplicating the decision in this module. See prompts/source-remediation
# logs/08-qsi-phase-group-integration.md (deviation 10) and logs/10-qsi-main-plumbing.md.

# Retuned from 64 to 24 in prompts/levin-refactor's prompt 08 (see
# prompts/levin-refactor/logs/08-order-and-sampling.md for the measurement). This module has no
# analytic oracle, so the order was chosen by self-consistency against a synthetic problem built
# directly from bessel_phase() with the same domain, amplitude form and phase contract (theta +
# theta_mod_2pi, no theta_deriv) as this file's own nine call sites, swept across three
# configurations spanning the phase magnitude the then-current LEVIN_MIN_PHASE_DIFF gate allowed (a
# near-threshold ~14-cycle case up to a ~4.5e6-cycle deep-sub-horizon case). Orders 12 through 64
# agreed with each other to within their own round-off floor in every configuration -- the same
# "accuracy is set by the phase/modulus splines, not the spectral order" finding that took
# three_bessel_integrals.py's DEFAULT_3BESSEL_CHEBYSHEV_ORDER from 64 to 12 in commit cc64ae4, here
# transferred by self-consistency rather than re-derived against an analytic reference. 24 was
# chosen over three_bessel_integrals.py's more aggressive 12 specifically because that finding
# rests on self-consistency here, not an analytic oracle (the audit sec 7 caveat on same-core
# references applies): at 24 every large-phase configuration measured had already collapsed to its
# minimal one-region solution (matching order 64's region count while costing 2-2.6x less per
# solve), whereas order 16 had not always done so. See the log for the full sweep table; this is
# an "evidence supports choosing an order, not evidence of absolute accuracy" result -- the
# self-consistency check cannot rule out an order-independent bias shared by every order tested.
CHEBYSHEV_ORDER = 24

# The module constants LEVIN_RELERR = 1e-8 / LEVIN_ABSERR = 1e-23 were deleted by
# prompts/source-remediation prompt 09 (audit B5/QI-8). Their only remaining use was the "Y3"
# call of _three_bessel_Levin, which passed them where its seven siblings pass the caller's
# atol/rtol; that asymmetry made the four cancelling phase groups' error bars non-uniform.
# Every Levin call in this module now takes the tolerances it was given.

# Relative tolerance of the Bessel-order guard _check_bessel_order below, as a fraction of the
# local Liouville-Green envelope m = sqrt(J^2 + Y^2). bessel_phase() reconstructs J_nu to ~2e-8
# of that envelope (audit QI-1), while an order wrong by delta shifts the phase by delta*pi/2
# -- 0.31 rad for the smallest interesting mismatch, delta = 0.2 -- and so moves J by O(0.1) of
# the envelope. 1e-3 sits five orders above the fit floor and two below the smallest defect it
# has to catch.
BESSEL_ORDER_CHECK_TOL = 1e-3

# Whether the composed phase derivative d Psi/d log(1+z') (closed-form omega (1+z') for each
# transfer function, the phase-spline log-derivative for the Green's function; see
# ComputeTargets/phase_groups.py) is handed to adaptive_levin_sincos as "theta_deriv". Without it
# the driver differentiates the sampled raw phase spectrally, which loses precision in proportion
# to |theta| / (phase change across the sub-region). Validated end-to-end in prompt 08 on the
# exact constant-w fixtures (prompts/source-remediation/logs/08-qsi-phase-group-integration.md,
# "theta_deriv decision"): see that log for the numbers behind this setting.
LEVIN_USE_THETA_DERIV = True

# A transfer function's hand-over redshift (TkSourceFunctions.crossover_z) need not be a grid
# point: main.py truncates the WKB grid to the largest source-grid point at or below z_init, so
# TkSourceFunctions.WKB_region[0] can sit up to one grid step below crossover_z, and no accessor
# of that factor is evaluable in between (IMPLEMENTATION_STATE.md section 5 note 6). The
# partition is made at crossover_z, and inside that gap the factor's accessors are evaluated at
# the nearest end of their own range ("clamped"). The gap is allowed to be at most this many
# mean source-grid steps (in log(1+z)) wide; anything larger is a data problem and raises.
HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5

# A sub-interval narrower than this in log(1+z') is treated as empty (two breakpoints that
# coincide, e.g. q = r, or a breakpoint landing on an end of the range). This is audit B11's
# fix (QI-4): before prompt 08 the region-nonempty guards compared get_z(max_z)/get_z(min_z)
# against 1 + DEFAULT_QUADRATURE_RTOL, a ratio in z rather than in 1+z, so the smallest accepted
# interval varied by orders of magnitude across the range and the test divided by zero at
# z_response = 0. A width in the integration variable log(1+z') does neither.
#
# Value: DEFAULT_FLOAT_PRECISION = 1e-7 is the tolerance at which this codebase treats two
# redshifts as the same, which is what a coincident breakpoint (q = r, or a hand-over landing on
# an end of the range) is. It is also small enough to be numerically irrelevant: the production
# range is log(1+1e5) - log(1+0.1) = 11.4 wide, so a discarded sub-interval carries at most
# 1e-7/11.4 ~ 1e-8 of the integral -- at the quadrature rtol, below every representation floor
# recorded on the campaign's status board. Discarded breakpoints are recorded in
# metadata["partition"]["skipped"] rather than dropped silently.
MIN_SUBINTERVAL_LOG_WIDTH = DEFAULT_FLOAT_PRECISION


class BesselPhaseProxy:
    def __init__(self, obj: dict):
        self._ref: ObjectRef = ray.put(obj)

    def get(self):
        """
        The return value should only be held locally and not persisted
        """
        return ray.get(self._ref)


def get_z(z):
    # duck-typed: a redshift, or any stand-in exposing .z (as the audit's QI_* scripts use)
    if isinstance(z, redshift) or hasattr(z, "z"):
        return z.z

    elif isinstance(z, float):
        return z

    return float(z)


def _log1p_z(z) -> float:
    return log(1.0 + get_z(z))


# ------------------------------------------------------------------------------------------
# clamping adapters
#
# phase_groups' factor adapters call the TkSourceFunctions accessors with log(1+z') and do not
# clamp (logs/07 "State handed to the next prompt" item 1). Each sub-interval therefore wraps
# its transfer functions in one of these, which pins the abscissa into the evaluable range of
# the representation the regime selects. The permitted overhang is checked when the partition
# is built (_check_gap), not here.


class _ClampedPhase:
    """phase_spline protocol (raw_theta, theta_mod_2pi, theta_deriv) with the abscissa clamped."""

    def __init__(self, phase, clamp: Callable[[float], float]):
        self._phase = phase
        self._clamp = clamp

    def raw_theta(self, x: float, x_is_log: bool = False) -> float:
        return self._phase.raw_theta(self._clamp(_as_log(x, x_is_log)), x_is_log=True)

    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float:
        return self._phase.theta_mod_2pi(
            self._clamp(_as_log(x, x_is_log)), x_is_log=True
        )

    def theta_deriv(
        self, x: float, x_is_log: bool = False, log_derivative: bool = False
    ) -> float:
        return self._phase.theta_deriv(
            self._clamp(_as_log(x, x_is_log)),
            x_is_log=True,
            log_derivative=log_derivative,
        )


def _as_log(x: float, is_log: bool) -> float:
    return x if is_log else log(1.0 + x)


class _ClampedTk:
    """
    The TkSourceFunctions protocol phase_groups reads (T, dT_dz for a smooth factor; M, dlnM_dz,
    omega, phase for an oscillatory one), with log(1+z') clamped into the evaluable range of
    the representation in use: numeric_region for a smooth factor (open above, because T = 1
    exactly above the numeric samples), WKB_region for an oscillatory one.
    """

    def __init__(self, functions, oscillatory: bool):
        self._f = functions
        self.oscillatory = oscillatory
        if oscillatory:
            z_max, z_min = functions.WKB_region
            self._log_hi = log(1.0 + z_max)
            self._log_lo = log(1.0 + z_min)
            self.phase = _ClampedPhase(functions.phase, self._clamp)
        else:
            z_max, z_min = functions.numeric_region
            self._log_hi = inf
            self._log_lo = log(1.0 + z_min)
            self.phase = None

    def _clamp(self, log_z: float) -> float:
        if log_z < self._log_lo:
            return self._log_lo
        if log_z > self._log_hi:
            return self._log_hi
        return log_z

    # smooth protocol
    def T(self, x: float, z_is_log: bool = False) -> float:
        return self._f.T(self._clamp(_as_log(x, z_is_log)), z_is_log=True)

    def dT_dz(self, x: float, z_is_log: bool = False) -> float:
        return self._f.dT_dz(self._clamp(_as_log(x, z_is_log)), z_is_log=True)

    # oscillatory protocol
    def M(self, x: float, z_is_log: bool = False) -> float:
        return self._f.M(self._clamp(_as_log(x, z_is_log)), z_is_log=True)

    def dlnM_dz(self, x: float, z_is_log: bool = False) -> float:
        return self._f.dlnM_dz(self._clamp(_as_log(x, z_is_log)), z_is_log=True)

    def omega(self, x: float, z_is_log: bool = False) -> float:
        return self._f.omega(self._clamp(_as_log(x, z_is_log)), z_is_log=True)


class _ClampedSource:
    """The QuadSource dense-output spline f(z') with log(1+z') clamped into its numeric_region."""

    def __init__(self, source_f: Callable, numeric_region: Tuple[float, float]):
        self._f = source_f
        z_max, z_min = numeric_region
        self._log_hi = log(1.0 + z_max)
        self._log_lo = log(1.0 + z_min)

    def __call__(self, x: float, z_is_log: bool = False) -> float:
        log_z = _as_log(x, z_is_log)
        if log_z < self._log_lo:
            log_z = self._log_lo
        elif log_z > self._log_hi:
            log_z = self._log_hi
        return self._f(log_z, z_is_log=True)


# ------------------------------------------------------------------------------------------
# the partition


def _factor_breakpoint(
    label: str, z_break: Optional[float], log_top: float, log_bottom: float
):
    """
    Classify one factor's hand-over redshift against the integration range
    [z_response, z_source_max]. Returns (mode, record) where mode is "smooth" (smooth over the
    whole range), "oscillatory" (oscillatory over the whole range) or "split" (breakpoint
    strictly inside the range), and record is the metadata["partition"]["breakpoints"] entry.
    """
    if z_break is None:
        return "smooth", None

    log_break = log(1.0 + z_break)
    record = {"factor": label, "z": float(z_break), "inside_range": False}

    if log_break >= log_top - MIN_SUBINTERVAL_LOG_WIDTH:
        # hands over at or above the top of the range: oscillatory throughout
        return "oscillatory", record
    if log_break <= log_bottom + MIN_SUBINTERVAL_LOG_WIDTH:
        # hands over at or below the response redshift: never oscillatory here
        return "smooth", record

    record["inside_range"] = True
    return "split", record


def _check_gap(
    label: str,
    gap: float,
    max_gap: float,
    z_max: float,
    z_min: float,
    detail: str,
) -> float:
    """
    A representation is allowed to fall short of a sub-interval end by at most max_gap in
    log(1+z) (see HANDOVER_CLAMP_MAX_GRID_STEPS); that shortfall is bridged by clamping.
    Returns the (non-negative) gap, or raises if it is too large.
    """
    if gap <= 0.0:
        return 0.0
    if gap > max_gap:
        raise RuntimeError(
            f"compute_QuadSource_integral: {label} is not evaluable over the whole sub-interval "
            f"z in ({z_max:.5g}, {z_min:.5g}): {detail}; the shortfall is {gap:.3e} in log(1+z), "
            f"but at most {max_gap:.3e} ({HANDOVER_CLAMP_MAX_GRID_STEPS:g} mean source-grid steps) "
            f"can be bridged by clamping"
        )
    return gap


def _mean_source_grid_step(source: QuadSource) -> float:
    """Mean spacing of the source redshift grid in log(1+z)."""
    z_sample = source.z_sample
    n = len(z_sample)
    if n < 2:
        raise RuntimeError(
            f"compute_QuadSource_integral: QuadSource (store_id={source.store_id}) has fewer than two sampled redshifts"
        )
    return (_log1p_z(z_sample.max) - _log1p_z(z_sample.min)) / (n - 1)


def build_partition(
    GkPolicy: GkSourcePolicyData,
    Tq_f,
    Tr_f,
    source: QuadSource,
    z_response: float,
    z_source_max: float,
) -> dict:
    """
    Partition [z_response, z_source_max] at the hand-over redshifts of all three factors of the
    integrand G_k(z, z') f(z' | q, r) / H(z')^2 (README section 6 of prompts/source-remediation/):

        factor   smooth representation above   oscillatory (Liouville-Green) below   breakpoint
        G        Gk_f.numeric_Gk               Gk_f.sin_amplitude, Gk_f.phase       GkPolicy.crossover_z (type "mixed";
                                                                                     "numeric" = smooth throughout,
                                                                                     "WKB" = oscillatory throughout)
        T_q      Tq_f.T, Tq_f.dT_dz (T = 1 above its grid)   Tq_f.M, dlnM_dz, omega, phase   Tq_f.crossover_z
        T_r      likewise                                     likewise                        Tr_f.crossover_z

    Each sub-interval carries a regime (G_osc, q_osc, r_osc) and, for every factor, the
    representation it uses is checked to be evaluable over the sub-interval up to a clamp gap
    of at most HANDOVER_CLAMP_MAX_GRID_STEPS source-grid steps (see _ClampedTk).

    :return: dict with "subintervals" (list of dicts, descending in z, each with z_max, z_min,
        regime, method, clamp gaps and the clamped factor objects) and "metadata" (the JSON-able
        record stored as metadata["partition"]).
    """
    Gk_f: GkSourceFunctions = GkPolicy.functions

    log_top = log(1.0 + z_source_max)
    log_bottom = log(1.0 + z_response)
    if log_top - log_bottom <= MIN_SUBINTERVAL_LOG_WIDTH:
        raise RuntimeError(
            f"compute_QuadSource_integral: empty integration range z_source_max={z_source_max:.5g}, z_response={z_response:.5g}"
        )

    # --- G ---------------------------------------------------------------------------------
    G_type = GkPolicy.type
    if G_type == "numeric":
        G_mode, G_record = "smooth", None
    elif G_type == "WKB":
        G_mode, G_record = "oscillatory", None
    elif G_type == "mixed":
        if GkPolicy.crossover_z is None:
            raise RuntimeError(
                f"compute_QuadSource_integral: GkSourcePolicyData (store_id={GkPolicy.store_id}) has type 'mixed' but no crossover_z"
            )
        G_mode, G_record = _factor_breakpoint(
            "G", float(GkPolicy.crossover_z), log_top, log_bottom
        )
    else:
        raise NotImplementedError(f"Gk {G_type} not implemented")

    # --- T_q, T_r --------------------------------------------------------------------------
    q_mode, q_record = _factor_breakpoint("Tq", Tq_f.crossover_z, log_top, log_bottom)
    r_mode, r_record = _factor_breakpoint("Tr", Tr_f.crossover_z, log_top, log_bottom)

    def log_break(mode, record):
        if mode == "oscillatory":
            return inf
        if mode == "smooth":
            return -inf
        return log(1.0 + record["z"])

    G_log_break = log_break(G_mode, G_record)
    q_log_break = log_break(q_mode, q_record)
    r_log_break = log_break(r_mode, r_record)

    # --- edges: the range ends plus every breakpoint strictly inside, descending, deduplicated
    # A breakpoint within MIN_SUBINTERVAL_LOG_WIDTH of the edge above it would open a sub-interval
    # too narrow to carry any of the integral (see that constant); it is merged into that edge and
    # recorded as skipped (audit B11), so that a reader of the stored metadata can tell a merged
    # hand-over from one that never happened.
    edges = [log_top]
    skipped = []
    for value, label in sorted(
        (
            (value, label)
            for value, label in (
                (G_log_break, "G"),
                (q_log_break, "Tq"),
                (r_log_break, "Tr"),
            )
            if log_bottom < value < log_top
        ),
        reverse=True,
    ):
        if edges[-1] - value > MIN_SUBINTERVAL_LOG_WIDTH:
            edges.append(value)
        else:
            skipped.append(
                {
                    "factor": label,
                    "z": exp(value) - 1.0,
                    "log_width": edges[-1] - value,
                    "reason": "hand-over within MIN_SUBINTERVAL_LOG_WIDTH of the sub-interval above it",
                }
            )
    if edges[-1] - log_bottom > MIN_SUBINTERVAL_LOG_WIDTH:
        edges.append(log_bottom)
    else:
        skipped.append(
            {
                "factor": None,
                "z": exp(edges[-1]) - 1.0,
                "log_width": edges[-1] - log_bottom,
                "reason": "lowest hand-over within MIN_SUBINTERVAL_LOG_WIDTH of z_response; snapped to it",
            }
        )
        edges[-1] = log_bottom

    max_gap = HANDOVER_CLAMP_MAX_GRID_STEPS * _mean_source_grid_step(source)

    source_region = source.numeric_region
    if source_region is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: QuadSource (store_id={source.store_id}) has no numeric_region"
        )

    subintervals = []
    records = []
    for log_hi, log_lo in zip(edges[:-1], edges[1:]):
        z_hi = exp(log_hi) - 1.0
        z_lo = exp(log_lo) - 1.0

        # a factor is oscillatory on this sub-interval if the sub-interval lies below its breakpoint
        regime = (
            G_log_break >= log_hi - MIN_SUBINTERVAL_LOG_WIDTH,
            q_log_break >= log_hi - MIN_SUBINTERVAL_LOG_WIDTH,
            r_log_break >= log_hi - MIN_SUBINTERVAL_LOG_WIDTH,
        )
        G_osc, q_osc, r_osc = regime
        gaps = {}

        # Green's function: GkSourcePolicyData guarantees its regions cover the crossover with
        # clearance, so no clamping is offered -- the checks are strict, as they were before.
        if G_osc:
            if (
                Gk_f.sin_amplitude is None
                or Gk_f.phase is None
                or Gk_f.WKB_region is None
            ):
                raise RuntimeError(
                    f"compute_QuadSource_integral: Green's function is oscillatory on z in ({z_hi:.5g}, {z_lo:.5g}) but has no WKB representation (type={G_type}, quality={GkPolicy.quality}, {_Gk_diagnostics(GkPolicy)})"
                )
            _check_region_covers("Gk WKB", Gk_f.WKB_region, log_hi, log_lo, z_hi, z_lo)
        else:
            if Gk_f.numeric_Gk is None or Gk_f.numeric_region is None:
                raise RuntimeError(
                    f"compute_QuadSource_integral: Green's function is smooth on z in ({z_hi:.5g}, {z_lo:.5g}) but has no numeric representation (type={G_type}, quality={GkPolicy.quality}, {_Gk_diagnostics(GkPolicy)})"
                )
            _check_region_covers(
                "Gk numeric", Gk_f.numeric_region, log_hi, log_lo, z_hi, z_lo
            )

        # transfer functions: partition on crossover_z, clamp to the evaluable range
        factors = {}
        for label, functions, osc in (("Tq", Tq_f, q_osc), ("Tr", Tr_f, r_osc)):
            if osc:
                region_max, region_min = functions.WKB_region
                if log(1.0 + region_min) > log_lo + MIN_SUBINTERVAL_LOG_WIDTH:
                    raise RuntimeError(
                        f"compute_QuadSource_integral: z_response={z_response:.5g} (sub-interval bottom z={z_lo:.5g}) lies below the lowest Liouville-Green sample z={region_min:.5g} of {label} (WKB region = ({region_max:.5g}, {region_min:.5g}))"
                    )
                gaps[label] = _check_gap(
                    f"the Liouville-Green representation of {label}",
                    log_hi - log(1.0 + region_max),
                    max_gap,
                    z_hi,
                    z_lo,
                    f"WKB region = ({region_max:.5g}, {region_min:.5g}), hand-over z={functions.crossover_z:.5g}",
                )
            else:
                region_max, region_min = functions.numeric_region
                gaps[label] = _check_gap(
                    f"the numeric representation of {label}",
                    log(1.0 + region_min) - log_lo,
                    max_gap,
                    z_hi,
                    z_lo,
                    f"numeric region = ({region_max:.5g}, {region_min:.5g}), hand-over z={functions.crossover_z:.5g}",
                )
            factors[label] = _ClampedTk(functions, osc)

        if not any(regime):
            # ordinary quadrature of the QuadSource spline of f, valid on source.numeric_region
            source_max, source_min = source_region
            if log(1.0 + source_max) < log_hi - DEFAULT_FLOAT_PRECISION:
                raise RuntimeError(
                    f"compute_QuadSource_integral: all-smooth sub-interval z in ({z_hi:.5g}, {z_lo:.5g}) starts above the QuadSource spline range ({source_max:.5g}, {source_min:.5g})"
                )
            gaps["source"] = _check_gap(
                "the QuadSource spline of f",
                log(1.0 + source_min) - log_lo,
                max_gap,
                z_hi,
                z_lo,
                f"QuadSource numeric region = ({source_max:.5g}, {source_min:.5g})",
            )
            method = "quad"
        else:
            method = "Levin"

        subintervals.append(
            {
                "z_max": z_hi,
                "z_min": z_lo,
                "log_width": log_hi - log_lo,
                "regime": regime,
                "method": method,
                "gaps": gaps,
                "Tq": factors["Tq"],
                "Tr": factors["Tr"],
            }
        )
        records.append(
            {
                "z_max": z_hi,
                "z_min": z_lo,
                "regime": list(regime),
                "method": method,
                "clamp_gaps_log1pz": gaps,
            }
        )

    metadata = {
        "z_source_max": float(z_source_max),
        "z_response": float(z_response),
        "G_type": G_type,
        "breakpoints": [r for r in (G_record, q_record, r_record) if r is not None],
        "crossover_z_q": Tq_f.crossover_z,
        "crossover_z_r": Tr_f.crossover_z,
        # persisted by GkSourcePolicyData, read here for the record only; not used (see the
        # "no quadrature gate lives in this module" comment at the top of this module)
        "Levin_z_unused": (
            get_z(GkPolicy.Levin_z) if GkPolicy.Levin_z is not None else None
        ),
        "max_clamp_gap_log1pz": max_gap,
        "min_subinterval_log_width": MIN_SUBINTERVAL_LOG_WIDTH,
        "skipped": skipped,
        "subintervals": records,
    }

    return {"subintervals": subintervals, "metadata": metadata}


def _check_region_covers(
    label: str,
    region,
    log_z_max: float,
    log_z_min: float,
    z_max: float,
    z_min: float,
) -> None:
    """
    Check that a factor's region of validity covers a sub-interval, comparing in the integration
    variable log(1+z') with MIN_SUBINTERVAL_LOG_WIDTH as the tolerance.

    This comparison must not be made in z. build_partition carries its edges in log(1+z) and
    recovers z for the quadrature by z = exp(log_z) - 1; at z ~ 5e14, log(1+z) ~ 32.3 exhausts the
    ~16 significant digits of a double, so the recoverable 1+z has a granularity of ~0.02 -- five
    orders of magnitude above DEFAULT_FLOAT_PRECISION. The information is gone in the log
    representation and no amount of expm1 care recovers it. Because the Green's-function region
    always ends exactly at z_response, the bottom of the lowest sub-interval always coincides with
    a region boundary, and an absolute tolerance in z therefore let the sign of the round-trip
    rounding decide whether the guard fired: 1372 of 3185 production work items raised
    (docs/source-remediation-verification.md section 5.1). Converting the region bounds
    z -> log(1+z) costs ~1 ulp of the log and does not amplify, so only that direction appears in
    the decision path below; this is the shape the Tq/Tr branch of build_partition already used.

    The error messages stay in z: they are for humans, and 5 significant figures is the right
    resolution there.

    Note this is a guard on an *equality-like* comparison. The same z_max/z_min are also passed on
    as quadrature limits, where the round trip is harmless -- 0.02 absolute at z = 5e14 is 4e-17
    relative -- so nothing about how those are computed should be "fixed" to match.
    """
    region_max_z, region_min_z = region

    # log(1+z) needs z > -1. The pipeline never goes below DEFAULT_ZEND = 0.1 and z_response >= 0,
    # so this is unreachable today, but the bounds arrive from the factor objects as plain floats.
    if region_min_z <= -1.0 or region_max_z <= -1.0:
        raise RuntimeError(
            f"compute_QuadSource_integral: the {label} region ({region_max_z:.5g}, {region_min_z:.5g}) contains a redshift z <= -1, which has no log(1+z) [domain={z_max:.5g}, {z_min:.5g}]"
        )

    if log_z_max > log(1.0 + region_max_z) + MIN_SUBINTERVAL_LOG_WIDTH:
        raise RuntimeError(
            f"compute_QuadSource_integral: sub-interval top z={z_max:.5g} is out-of-bounds for the {label} region ({region_max_z:.5g}, {region_min_z:.5g}) [domain={z_max:.5g}, {z_min:.5g}]"
        )
    if log_z_min < log(1.0 + region_min_z) - MIN_SUBINTERVAL_LOG_WIDTH:
        raise RuntimeError(
            f"compute_QuadSource_integral: sub-interval bottom z={z_min:.5g} is out-of-bounds for the {label} region ({region_max_z:.5g}, {region_min_z:.5g}) [domain={z_max:.5g}, {z_min:.5g}]"
        )


def _Gk_diagnostics(GkPolicy) -> str:
    """Best-effort description of the underlying GkSource for error messages."""
    proxy = getattr(GkPolicy, "_source_proxy", None)
    if proxy is None:
        return "GkSource unavailable"
    try:
        Gk: GkSource = proxy.get()
        return f"lowest numeric z={_extract_z(Gk._numeric_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(GkPolicy.crossover_z)}, z_Levin={_extract_z(GkPolicy.Levin_z)}"
    except Exception as e:  # diagnostics only; never mask the original error
        return f"GkSource diagnostics unavailable ({type(e).__name__})"


# ------------------------------------------------------------------------------------------
# aggregation of per-sub-interval integration records


def _aggregate_IntegrationData(
    items: Sequence[IntegrationData],
) -> Optional[IntegrationData]:
    if len(items) == 0:
        return None
    evaluations = sum(item.RHS_evaluations for item in items)
    if evaluations > 0:
        mean_RHS_time = (
            sum(item.mean_RHS_time * item.RHS_evaluations for item in items)
            / evaluations
        )
    else:
        mean_RHS_time = 0.0
    return IntegrationData(
        compute_time=sum(item.compute_time for item in items),
        compute_steps=sum(item.compute_steps for item in items),
        mean_RHS_time=mean_RHS_time,
        max_RHS_time=max(item.max_RHS_time for item in items),
        min_RHS_time=min(item.min_RHS_time for item in items),
        RHS_evaluations=evaluations,
    )


def _aggregate_LevinData(items: Sequence[dict]) -> Optional[LevinData]:
    """
    Aggregate the adaptive_levin_sincos result dicts of every Levin call made for one integral:
    counters are summed, chebyshev_min_order is the minimum, max_depth the maximum.
    """
    if len(items) == 0:
        return None
    return LevinData(
        num_regions=sum(item["num_regions"] for item in items),
        evaluations=sum(item["evaluations"] for item in items),
        num_simple_regions=sum(item["num_simple_regions"] for item in items),
        num_SVD_errors=sum(item["num_SVD_errors"] for item in items),
        num_order_changes=sum(item["num_order_changes"] for item in items),
        chebyshev_min_order=min(item["chebyshev_min_order"] for item in items),
        max_depth=max(item["max_depth"] for item in items),
        elapsed=sum(item["elapsed"] for item in items),
    )


# ------------------------------------------------------------------------------------------
# the source time integral


def _check_bessel_order(label: str, phase_data: dict, nu: float) -> None:
    """
    The Levin branch of the analytic oracle uses the caller-supplied bessel_phase splines while
    _three_bessel_quad recomputes jv(nu + b, .) for itself; the two agree only if the splines
    were built at the same b that is passed as `b` (audit QI-1: "correct at HEAD but an
    unguarded invariant"). bessel_phase() does not record its own order -- its return dict
    (LiouvilleGreen/bessel_phase.py) carries phase, mod, Q, phi, bessel_j, bessel_y, min_x,
    max_x and no "nu" -- so the order is checked numerically instead: the spline's own
    reconstruction J_nu = m sin(theta) is compared against scipy's J_nu, normalised by the local
    envelope m (which is never zero, unlike J itself). An order wrong by delta puts the phase out
    by delta*pi/2 -- 0.31 rad for delta = 0.2 -- so J moves by O(0.1) of the envelope, while the
    reconstruction's own error is ~2e-8 of it (audit QI-1). LiouvilleGreen/ is out of scope for
    this campaign, so nothing here asks bessel_phase to start recording its order.
    """
    min_x = phase_data["min_x"]
    max_x = phase_data["max_x"]

    # Two abscissae just inside the oscillatory region, x ~ 2 nu to 5 nu: far apart in phase, so
    # that a wrong order cannot escape both by sitting at a node of the difference, but small
    # enough that the phase spline is at its most accurate (its error grows with x, and
    # production builds these splines out to x ~ 1e9). Deep inside the turning point, x << nu,
    # the check would lose its sensitivity instead: there J is exponentially small compared with
    # the envelope, so even a wrong order moves it by far less than the envelope.
    x_lo = min(max(1.01 * min_x, 2.0 * max(nu, 0.5)), max_x)
    for x_test in sorted({x_lo, min(2.5 * x_lo, max_x)}):
        envelope = phase_data["mod"](x_test)
        residual = abs(phase_data["bessel_j"](x_test) - jv(nu, x_test))
        if residual > BESSEL_ORDER_CHECK_TOL * envelope:
            raise RuntimeError(
                f"compute_QuadSource_integral: the supplied {label} phase spline does not appear to be "
                f"a Liouville-Green representation of the Bessel function of order nu={nu:.8g} required by "
                f"b={nu - (0.5 if label == 'Bessel_0pt5' else 2.5):.8g}: at x={x_test:.5g} the spline gives "
                f"J={phase_data['bessel_j'](x_test):.8g} against scipy's J_nu={jv(nu, x_test):.8g}, a "
                f"discrepancy of {residual / envelope:.3e} of the local envelope m={envelope:.5g} (tolerance "
                f"{BESSEL_ORDER_CHECK_TOL:.1e}). The Bessel phase splines must be built with the same b that "
                f"is passed to the source time integral (audit QI-1)"
            )


def evaluate_QuadSource_integral(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    q: wavenumber_exit_time,
    r: wavenumber_exit_time,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    z_source_max: redshift,
    b: float,
    Bessel_0pt5: dict,
    Bessel_2pt5: dict,
    Tq_numeric,
    Tq_WKB,
    Tr_numeric,
    Tr_WKB,
    atol: float = DEFAULT_QUADRATURE_ATOL,
    rtol: float = DEFAULT_QUADRATURE_RTOL,
    Tk_functions_builder=TkSourceFunctions,
) -> dict:
    """
    The source time integral (spec 03 R28 with Q_s/a_0^2 stripped, audit section 3.1),

        total = (1 + z_response) int d log(1+z') G_k(z_response, z') f(z' | q, r) / H(z')^2

    over z' in [z_response, z_source_max], together with the analytic constant-w oracle
    `analytic_rad`. The range is partitioned at the hand-over redshifts of G, T_q and T_r
    (build_partition); an all-smooth sub-interval is integrated by scipy.quad of the QuadSource
    spline (numeric_quad_integral), every other sub-interval by one adaptive_levin_sincos call
    per phase group of ComputeTargets.phase_groups (phase_group_Levin_integral).

    This is the body of the Ray task compute_QuadSource_integral, callable directly (no Ray
    runtime needed) with the Bessel phase dicts rather than their proxies. Tk_functions_builder
    exists so that tests can substitute an exact two-region representation of T_k for
    TkSourceFunctions; production never passes it.
    """
    model_f: ModelFunctions = model.functions
    Gk_f: GkSourceFunctions = GkPolicy.functions

    with WallclockTimer() as timer:
        # the Bessel phase splines must have been built at this b (audit QI-1)
        _check_bessel_order("Bessel_0pt5", Bessel_0pt5, 0.5 + b)
        _check_bessel_order("Bessel_2pt5", Bessel_2pt5, 2.5 + b)

        # two-region (numeric + Liouville-Green) representations of the transfer functions
        Tq_f = Tk_functions_builder(model, q.k, Tq_numeric, Tq_WKB)
        Tr_f = Tk_functions_builder(model, r.k, Tr_numeric, Tr_WKB)

        for label, functions, recorded in (
            ("q", Tq_f, source.crossover_z_q),
            ("r", Tr_f, source.crossover_z_r),
        ):
            if recorded is not None and abs(
                functions.crossover_z - recorded
            ) > DEFAULT_FLOAT_PRECISION * max(1.0, abs(recorded)):
                raise RuntimeError(
                    f"compute_QuadSource_integral: hand-over redshift of T_{label} from the supplied Tk objects is z={functions.crossover_z:.8g}, but the QuadSource (store_id={source.store_id}) records z={recorded:.8g}"
                )

        z_resp = get_z(z_response)
        z_top = get_z(z_source_max)

        partition = build_partition(GkPolicy, Tq_f, Tr_f, source, z_resp, z_top)
        subintervals = partition["subintervals"]
        total_log_width = sum(item["log_width"] for item in subintervals)

        source_f = _ClampedSource(source.functions.source, source.numeric_region)

        numeric_quad: float = 0.0
        numeric_quad_abserr: float = 0.0
        numeric_quad_items: List[IntegrationData] = []
        numeric_quad_records = []

        WKB_Levin: float = 0.0
        WKB_Levin_abserr: float = 0.0
        WKB_Levin_converged: bool = True
        WKB_Levin_phase_limited: bool = False
        WKB_Levin_items: List[dict] = []
        WKB_Levin_records = []

        for item in subintervals:
            z_hi = item["z_max"]
            z_lo = item["z_min"]
            # the caller's atol bounds the whole integral; each sub-interval gets its share by
            # length in log(1+z'), matching how adaptive_levin_sincos itself distributes atol
            # over its sub-regions (prompts/levin-refactor prompt 05, audit C3)
            atol_sub = atol * item["log_width"] / total_log_width

            if item["method"] == "quad":
                payload = numeric_quad_integral(
                    model,
                    k.k,
                    q.k,
                    r.k,
                    source,
                    GkPolicy,
                    z_response,
                    max_z=z_hi,
                    min_z=z_lo,
                    atol=atol_sub,
                    rtol=rtol,
                    source_f=source_f,
                )
                numeric_quad += payload["value"]
                numeric_quad_abserr += payload["abserr"]
                numeric_quad_items.append(payload["data"])
                numeric_quad_records.append(
                    {
                        "z_max": z_hi,
                        "z_min": z_lo,
                        "value": payload["value"],
                        "abserr": payload["abserr"],
                    }
                )
            else:
                payload = phase_group_Levin_integral(
                    model,
                    k.k,
                    q.k,
                    r.k,
                    item["regime"],
                    Gk_f,
                    item["Tq"],
                    item["Tr"],
                    z_response,
                    max_z=z_hi,
                    min_z=z_lo,
                    atol=atol_sub,
                    rtol=rtol,
                )
                WKB_Levin += payload["value"]
                WKB_Levin_abserr += payload["abserr"]
                WKB_Levin_converged = WKB_Levin_converged and payload["converged"]
                WKB_Levin_phase_limited = (
                    WKB_Levin_phase_limited or payload["phase_limited"]
                )
                WKB_Levin_items.extend(payload["Levin_results"])
                WKB_Levin_records.append(
                    {
                        "z_max": z_hi,
                        "z_min": z_lo,
                        "regime": list(item["regime"]),
                        "value": payload["value"],
                        "abserr": payload["abserr"],
                        "converged": payload["converged"],
                        "phase_limited": payload["phase_limited"],
                        "groups": payload["groups"],
                    }
                )

        # calculate analytic approximation for specified value of b using pre-supplied Bessel function splines
        analytic_data = analytic_integral(
            model,
            k.k,
            q.k,
            r.k,
            z_response,
            max_z=z_source_max,
            min_z=z_response,
            b=b,
            Bessel_0pt5=Bessel_0pt5,
            Bessel_2pt5=Bessel_2pt5,
            rtol=rtol,
            atol=atol,
        )

    # A bound on the error of "total" (audit B8/QI-11). Both contributions are absolute errors
    # already scaled by (1 + z_response) -- numeric_quad_integral and phase_group_Levin_integral
    # each scale their own "abserr" exactly as they scale their "value" -- so they are summed
    # directly, and linearly rather than in quadrature: neighbouring sub-intervals share the same
    # representation of every factor, so a representation error is a common drift and not
    # independent noise (the same reasoning as the three-Bessel machinery below).
    #
    # This is an absolute bound and must stay one: numeric_quad and WKB_Levin cancel by up to 5x
    # on the prompt-08 fixtures (logs/08-qsi-phase-group-integration.md observation 5), so a
    # relative error scaled by |total| would understate the error of a cancelling case.
    total_abserr = numeric_quad_abserr + WKB_Levin_abserr

    # The persisted columns keep their names, with these meanings since prompt 08:
    #   numeric_quad  -- sum over the all-smooth sub-intervals (scipy.quad of the QuadSource spline)
    #   WKB_Levin     -- sum over every sub-interval with at least one oscillatory factor
    #   total         -- numeric_quad + WKB_Levin
    #   total_abserr, total_converged, total_phase_limited -- the bound above and the Levin
    #                    driver's flags, aggregated over every group of every sub-interval
    #   b             -- the b these were computed at (audit B7): it fixes c_s, the Bessel orders
    #                    and the eta' weight of analytic_rad
    return {
        "total": numeric_quad + WKB_Levin,
        "total_abserr": total_abserr,
        "total_converged": WKB_Levin_converged,
        "total_phase_limited": WKB_Levin_phase_limited,
        "b": b,
        "numeric_quad": numeric_quad,
        "WKB_Levin": WKB_Levin,
        "GkPolicy_serial": GkPolicy.store_id,
        "source_serial": source.store_id,
        "numeric_quad_data": _aggregate_IntegrationData(numeric_quad_items),
        "WKB_Levin_data": _aggregate_LevinData(WKB_Levin_items),
        "WKB_phase_spline_chunks": (
            getattr(Gk_f.phase, "num_chunks", None) if Gk_f.phase is not None else None
        ),
        "eta_source_max": model_f.tau(z_top),
        "eta_response": model_f.tau(z_resp),
        "analytic_rad": analytic_data["value"],
        "compute_time": timer.elapsed,
        "analytic_compute_time": analytic_data["elapsed"],
        # Error bounds: the Levin sub-intervals' abserr (summed linearly across groups and
        # sub-intervals) and the quad sub-intervals' scipy error estimates. Their sum is the
        # top-level "total_abserr" above; the per-part and per-sub-interval detail stays here.
        "metadata": {
            "analytic": {
                **analytic_data["metadata"],
                "abserr": analytic_data["abserr"],
                "converged": analytic_data["converged"],
            },
            "numeric_quad": {
                "abserr": numeric_quad_abserr,
                "subintervals": numeric_quad_records,
            },
            "WKB_Levin": {
                "abserr": WKB_Levin_abserr,
                "converged": WKB_Levin_converged,
                "phase_limited": WKB_Levin_phase_limited,
                "theta_deriv_supplied": LEVIN_USE_THETA_DERIV,
                "subintervals": WKB_Levin_records,
            },
            "partition": partition["metadata"],
        },
    }


@ray.remote
def compute_QuadSource_integral(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    q: wavenumber_exit_time,
    r: wavenumber_exit_time,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    z_source_max: redshift,
    b: float,
    Bessel_0pt5: BesselPhaseProxy,
    Bessel_2pt5: BesselPhaseProxy,
    Tq_numeric,
    Tq_WKB,
    Tr_numeric,
    Tr_WKB,
    atol: float = DEFAULT_QUADRATURE_ATOL,
    rtol: float = DEFAULT_QUADRATURE_RTOL,
) -> dict:
    # we are entitled to assume that the GkSource embedded in GkPolicy and source are evaluated at the same z_response
    # also that source and Gk have z_samples at least as far back as z_source_max
    model: BackgroundModel = model_proxy.get()

    return evaluate_QuadSource_integral(
        model,
        k,
        q,
        r,
        source,
        GkPolicy,
        z_response,
        z_source_max,
        b,
        Bessel_0pt5.get(),
        Bessel_2pt5.get(),
        Tq_numeric,
        Tq_WKB,
        Tr_numeric,
        Tr_WKB,
        atol=atol,
        rtol=rtol,
    )


def phase_group_Levin_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    regime: Tuple[bool, bool, bool],
    Gk_f: GkSourceFunctions,
    Tq_f,
    Tr_f,
    z_response: redshift,
    max_z: float,
    min_z: float,
    atol: float,
    rtol: float,
) -> dict:
    """
    (1 + z_response) int_{log(1+min_z)}^{log(1+max_z)} G f / H^2 d log(1+z') over one sub-interval
    on which at least one factor is oscillatory, as one adaptive_levin_sincos call per phase
    group of build_phase_groups(regime, ...). Group values and error estimates are summed
    linearly (the groups share a phase construction, so an inaccurate phase produces a common
    drift, not independent noise -- the same reasoning as _three_bessel_Levin below).

    atol is the absolute tolerance for this sub-interval; each group receives atol / n_groups so
    that the linear sum of the groups' error bounds is bounded by atol. rtol is passed to every
    group unchanged (a relative tolerance has no additive share).

    :param Tq_f, Tr_f: TkSourceFunctions-protocol objects (normally _ClampedTk wrappers) whose
        numeric accessors are read if the factor is smooth in `regime`, WKB accessors if not
    :return: dict with "value", "abserr", "converged", "phase_limited", "groups" (per-group
        metadata, values scaled by (1 + z_response) like "value") and "Levin_results" (the raw
        driver result dicts, for aggregation into LevinData)
    """
    model_f: ModelFunctions = model.functions
    z_resp = get_z(z_response)

    groups = build_phase_groups(
        regime,
        Gk=Gk_f if regime[0] else Gk_f.numeric_Gk,
        Tq=Tq_f,
        Tr=Tr_f,
        model_functions=model_f,
        w_background=model_f.wBackground,
    )

    x_span = (log(1.0 + min_z), log(1.0 + max_z))
    atol_group = atol / len(groups)
    regime_label = "".join(name for name, flag in zip(("G", "q", "r"), regime) if flag)

    results = []
    for group in groups:
        data = adaptive_levin_sincos(
            x_span,
            [group.f_sin, group.f_cos],
            theta=group.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV),
            atol=atol_group,
            rtol=rtol,
            chebyshev_order=CHEBYSHEV_ORDER,
            notify_label=f"k={k.k_inv_Mpc:.3g}/Mpc, q={q.k_inv_Mpc:.3g}/Mpc, r={r.k_inv_Mpc:.3g}/Mpc @ z_response={z_resp:.5g}, z in ({max_z:.5g}, {min_z:.5g}), regime {regime_label}, group {group.label}",
        )
        results.append((group, data))

    scale = 1.0 + z_resp
    value = scale * sum(data["value"] for _, data in results)
    # abserr is an absolute error, so it scales linearly under the same rescaling as "value"
    abserr = scale * sum(data["abserr"] for _, data in results)

    return {
        "value": value,
        "abserr": abserr,
        "converged": all(data["converged"] for _, data in results),
        "phase_limited": any(data["phase_limited"] for _, data in results),
        "groups": [
            {
                "label": group.label,
                "value": scale * data["value"],
                "abserr": scale * data["abserr"],
                "converged": data["converged"],
                "phase_limited": data["phase_limited"],
                "regions": data["num_regions"],
                "simple_regions": data["num_simple_regions"],
                "evaluations": data["evaluations"],
                "elapsed": data["elapsed"],
            }
            for group, data in results
        ],
        "Levin_results": [data for _, data in results],
    }


def _three_bessel_integrals(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    phase_data: dict,
    nu_type: str,
    atol: float,
    rtol: float,
):
    if nu_type not in ["0pt5", "2pt5"]:
        raise RuntimeError('phase_data should be "0pt5" or "2pt5"')

    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)

    phase_data_Gk = phase_data["0pt5"]
    phase_data_Tk = phase_data[nu_type]

    min_x_Gk = phase_data_Gk["min_x"]
    min_x_Tk = phase_data_Tk["min_x"]

    # the nu = 0.5 Bessel function will have min_x nearly zero, and in principle we can use it for almost any values
    # of eta. But this doesn't actually produce good results, probably because of cancellations between the different
    # phases in the 4 sin/cos integrals that are needed. In this region, we are better using ordinary numerical quadrature.
    x_cut = max(min_x_Gk, min_x_Tk, 0.5)

    min_k_mode = min(k.k, q.k, r.k)

    eta_cut = x_cut / min_k_mode / cs

    if eta_cut / min_eta <= 1.0 + DEFAULT_FLOAT_PRECISION:
        Levin = _three_bessel_Levin(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            phase_data_Gk=phase_data_Gk,
            phase_data_Tk=phase_data_Tk,
            x_cut=x_cut,
            atol=atol,
            rtol=rtol,
        )
        value = Levin["value"]
        abserr = Levin["abserr"]
        converged = Levin["converged"]
        return {
            "J": value[0],
            "Y": value[1],
            "J_abserr": abserr[0],
            "Y_abserr": abserr[1],
            "J_converged": converged[0],
            "Y_converged": converged[1],
            "metadata": Levin["metadata"] | {"compute_time": Levin["compute_time"]},
        }

    if eta_cut / max_eta >= 1.0 - DEFAULT_FLOAT_PRECISION:
        quad = _three_bessel_quad(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            nu_type=nu_type,
            atol=atol,
            rtol=rtol,
        )
        value = quad["value"]
        abserr = quad["abserr"]
        converged = quad["converged"]
        return {
            "J": value[0],
            "Y": value[1],
            "J_abserr": abserr[0],
            "Y_abserr": abserr[1],
            "J_converged": converged[0],
            "Y_converged": converged[1],
            "metadata": quad["metadata"] | {"compute_time": quad["compute_time"]},
        }

    # otherwise, eta_cut calls between min_eta and max_eta

    quad = _three_bessel_quad(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=eta_cut,
        b=b,
        nu_type=nu_type,
        atol=atol,
        rtol=rtol,
    )
    Levin = _three_bessel_Levin(
        k,
        q,
        r,
        min_eta=eta_cut,
        max_eta=max_eta,
        b=b,
        phase_data_Gk=phase_data_Gk,
        phase_data_Tk=phase_data_Tk,
        x_cut=x_cut,
        atol=atol,
        rtol=rtol,
    )
    quad_value = quad["value"]
    Levin_value = Levin["value"]
    quad_abserr = quad["abserr"]
    Levin_abserr = Levin["abserr"]
    return {
        "J": quad_value[0] + Levin_value[0],
        "Y": quad_value[1] + Levin_value[1],
        "J_abserr": quad_abserr[0] + Levin_abserr[0],
        "Y_abserr": quad_abserr[1] + Levin_abserr[1],
        "J_converged": quad["converged"][0] and Levin["converged"][0],
        "Y_converged": quad["converged"][1] and Levin["converged"][1],
        "metadata": {
            "quad": quad["metadata"],
            "Levin": Levin["metadata"],
            "compute_time": quad["compute_time"] + Levin["compute_time"],
        },
    }


def _three_bessel_Levin(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    phase_data_Gk,
    phase_data_Tk,
    x_cut: float,
    atol: float,
    rtol: float,
):
    assert min_eta <= max_eta

    start = time.perf_counter()

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    x_span = (log(min_eta), log(max_eta))

    min_k_mode = min(k.k, q.k, r.k)
    if min_k_mode * min_eta * cs < (1.0 - DEFAULT_FLOAT_PRECISION) * x_cut:
        raise RuntimeError(
            f"!! three_bessel_Levin: ERROR: smallest required x is smaller than x_cut (smallest x={min_k_mode * min_eta * cs:.5g}, x_cut={x_cut:.5g}, min_eta={min_eta:.5g}, min_k_mode={min_k_mode:.5g}, cs={cs:.5g}, coeff={min_k_mode*cs:.5g}, x_cut/coeff={x_cut/(min_k_mode*cs):.5g})"
        )

    phase_Gk = phase_data_Gk["phase"]
    mod_Gk = phase_data_Gk["mod"]

    phase_Tk = phase_data_Tk["phase"]
    mod_Tk = phase_data_Tk["mod"]

    def Levin_f(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)
        B = mod_Gk(x1) * mod_Tk(x2) * mod_Tk(x3)

        return A * B

    def phase1(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk.raw_theta(x1) + phase_Tk.raw_theta(x2) + phase_Tk.raw_theta(x3)

    def phase1_mod_2pi(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return (
            phase_Gk.theta_mod_2pi(x1)
            + phase_Tk.theta_mod_2pi(x2)
            + phase_Tk.theta_mod_2pi(x3)
        )

    J1_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase1, "theta_mod_2pi": phase1_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic J1",
    )
    Y1_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta={"theta": phase1, "theta_mod_2pi": phase1_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic Y1",
    )

    def phase2(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk.raw_theta(x1) + phase_Tk.raw_theta(x2) - phase_Tk.raw_theta(x3)

    def phase2_mod_2pi(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return (
            phase_Gk.theta_mod_2pi(x1)
            + phase_Tk.theta_mod_2pi(x2)
            - phase_Tk.theta_mod_2pi(x3)
        )

    J2_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase2, "theta_mod_2pi": phase2_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic J2",
    )
    Y2_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta={"theta": phase2, "theta_mod_2pi": phase2_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic Y2",
    )

    def phase3(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk.raw_theta(x1) - phase_Tk.raw_theta(x2) + phase_Tk.raw_theta(x3)

    def phase3_mod_2pi(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return (
            phase_Gk.theta_mod_2pi(x1)
            - phase_Tk.theta_mod_2pi(x2)
            + phase_Tk.theta_mod_2pi(x3)
        )

    J3_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase3, "theta_mod_2pi": phase3_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic J3",
    )
    Y3_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta={"theta": phase3, "theta_mod_2pi": phase3_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic Y3",
    )

    def phase4(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk.raw_theta(x1) - phase_Tk.raw_theta(x2) - phase_Tk.raw_theta(x3)

    def phase4_mod_2pi(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return (
            phase_Gk.theta_mod_2pi(x1)
            - phase_Tk.theta_mod_2pi(x2)
            - phase_Tk.theta_mod_2pi(x3)
        )

    J4_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase4, "theta_mod_2pi": phase4_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic J4",
    )
    Y4_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta={"theta": phase4, "theta_mod_2pi": phase4_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=CHEBYSHEV_ORDER,
        notify_label="analytic Y4",
    )

    J1_value = J1_data["value"]
    J2_value = J2_data["value"]
    J3_value = J3_data["value"]
    J4_value = J4_data["value"]

    Y1_value = Y1_data["value"]
    Y2_value = Y2_data["value"]
    Y3_value = Y3_data["value"]
    Y4_value = Y4_data["value"]

    norm_factor = 1.0 / 4.0
    J = norm_factor * (-J1_value + J2_value + J3_value - J4_value)
    Y = norm_factor * (-Y1_value + Y2_value + Y3_value - Y4_value)

    J_blocks = [J1_data, J2_data, J3_data, J4_data]
    Y_blocks = [Y1_data, Y2_data, Y3_data, Y4_data]

    # combined linearly, not in quadrature, for the same reason as three_bessel_integrals.py's
    # BesselIntegralResult (prompts/levin-refactor's prompt 09): the four groups share a phase
    # construction, so an inaccurate phase produces a common drift, not independent noise. Every
    # combination coefficient here has magnitude 1, so |coefficient| is elided.
    J_abserr = norm_factor * sum(item["abserr"] for item in J_blocks)
    Y_abserr = norm_factor * sum(item["abserr"] for item in Y_blocks)
    J_converged = all(item["converged"] for item in J_blocks)
    Y_converged = all(item["converged"] for item in Y_blocks)
    J_phase_limited = any(item["phase_limited"] for item in J_blocks)
    Y_phase_limited = any(item["phase_limited"] for item in Y_blocks)

    data_blocks = {
        "J1_data": J1_data,
        "J2_data": J2_data,
        "J3_data": J3_data,
        "J4_data": J4_data,
        "Y1_data": Y1_data,
        "Y2_data": Y2_data,
        "Y3_data": Y3_data,
        "Y4_data": Y4_data,
    }
    metadata = {
        key: {
            "elapsed": item["elapsed"],
            "regions": len(item["regions"]),
            "abserr": item["abserr"],
            "converged": item["converged"],
            "phase_limited": item["phase_limited"],
        }
        for key, item in data_blocks.items()
    }

    stop = time.perf_counter()

    return {
        "value": [J, Y],
        "abserr": [J_abserr, Y_abserr],
        "converged": [J_converged, Y_converged],
        "phase_limited": [J_phase_limited, Y_phase_limited],
        "compute_time": stop - start,
        "metadata": metadata,
    }


def _three_bessel_quad(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    nu_type: str,
    atol: float,
    rtol: float,
):
    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    log_min_eta = log(min_eta)
    log_max_eta = log(max_eta)

    nu_types = {"0pt5": 0.5, "2pt5": 2.5}
    nu = nu_types[nu_type]

    def J_integrand(log_eta):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)
        B = jv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)

        return A * B

    def Y_integrand(log_eta):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)
        B = yv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)

        return A * B

    data = simple_quadrature(
        [J_integrand, Y_integrand],
        a=log_min_eta,
        b=log_max_eta,
        atol=atol,
        rtol=rtol,
        label=f"three_bessel_quad for  k={k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}), q={q.k_inv_Mpc:.5g}/Mpc (store_id={q.store_id}), r={r.k_inv_Mpc:.5g}/Mpc (store_id={r.store_id})",
        method="quad",
    )

    J_value, Y_value = data["value"]
    J_abserr, Y_abserr = data["abserr"]

    return {
        "value": data["value"],
        "abserr": data["abserr"],
        "converged": [
            J_abserr <= max(atol, rtol * abs(J_value)),
            Y_abserr <= max(atol, rtol * abs(Y_value)),
        ],
        "compute_time": data["data"].compute_time,
        "metadata": {},
    }


def analytic_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    z_response: redshift,
    max_z: redshift,
    min_z: redshift,
    b: float,
    Bessel_0pt5,
    Bessel_2pt5,
    rtol: float,
    atol: float,
):
    start = time.perf_counter()

    functions: ModelFunctions = model.functions
    min_eta: float = functions.tau(max_z.z)
    max_eta: float = functions.tau(min_z.z)

    eta_response: float = functions.tau(z_response.z)

    cs_sq = (1.0 - b) / (1.0 + b) / 3.0

    # atol and rtol are the caller's -- i.e. the QuadSourceIntegral row's own atol_serial and
    # rtol_serial. They used to be dead arguments here, with 1e-21 / 1e-8 hardwired on both
    # calls, so the stored tolerance columns did not describe analytic_rad (audit B6/QI-9). The
    # pipeline supplies DEFAULT_QUADRATURE_ATOL = 1e-25 and DEFAULT_QUADRATURE_RTOL = 1e-8, so
    # the absolute tolerance is now 1e4 tighter than the retired literal; measured effect on
    # analytic_rad and on runtime: prompts/source-remediation/logs/09-qsi-errors-schema-tolerances.md.
    data0pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": Bessel_0pt5},
        nu_type="0pt5",
        atol=atol,
        rtol=rtol,
    )
    data2pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": Bessel_0pt5, "2pt5": Bessel_2pt5},
        nu_type="2pt5",
        atol=atol,
        rtol=rtol,
    )

    metadata = {
        "0pt5": data0pt5["metadata"],
        "2pt5": data2pt5["metadata"],
    }

    A = (2.0 + b) / (1.0 + b)

    Y_factor = data0pt5["J"] + A * data2pt5["J"]
    J_factor = data0pt5["Y"] + A * data2pt5["Y"]

    # combined linearly, not in quadrature, for the same reason as everywhere else in this file's
    # three-Bessel machinery (prompts/levin-refactor's prompt 09): a common phase-spline drift, not
    # independent noise
    Y_factor_abserr = data0pt5["J_abserr"] + abs(A) * data2pt5["J_abserr"]
    J_factor_abserr = data0pt5["Y_abserr"] + abs(A) * data2pt5["Y_abserr"]
    converged = (
        data0pt5["J_converged"]
        and data0pt5["Y_converged"]
        and data2pt5["J_converged"]
        and data2pt5["Y_converged"]
    )

    B = pi / 2.0
    C = pow(2.0, 3.0 + 2.0 * b) / (3.0 + 2.0 * b) / (2.0 + b)
    D = gamma(2.5 + b) * gamma(2.5 + b)
    E = pow(q.k * r.k * cs_sq * eta_response, -0.5 - b)

    F = -B * C * D * E
    x = k.k * eta_response

    Y_bessel = yv(0.5 + b, x)
    J_bessel = jv(0.5 + b, x)

    value = F * (Y_bessel * Y_factor - J_bessel * J_factor)
    abserr = abs(F) * (
        abs(Y_bessel) * Y_factor_abserr + abs(J_bessel) * J_factor_abserr
    )

    stop = time.perf_counter()

    return {
        "value": value,
        "abserr": abserr,
        "converged": converged,
        "elapsed": stop - start,
        "metadata": metadata,
    }


def _extract_z(z: Union[type(None), redshift, float]) -> str:
    if z is None:
        return "(not set)"

    if isinstance(z, redshift):
        return f"{z.z:.5g}"

    if isinstance(z, float):
        return f"{z:.5g}"

    return f"{float(z):..5g}"


def numeric_quad_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    max_z: float,
    min_z: float,
    atol: float,
    rtol: float,
    source_f: Optional[Callable] = None,
) -> dict:
    """
    (1 + z_response) int_{log(1+min_z)}^{log(1+max_z)} G f / H^2 d log(1+z') by scipy.quad, on a
    sub-interval where every factor is smooth: G from Gk_f.numeric_Gk and f from the QuadSource
    dense-output spline (or `source_f`, a callable f(log(1+z'), z_is_log=True) standing in for it,
    e.g. the clamped wrapper evaluate_QuadSource_integral builds). The measure is spec 03 R28 /
    spec 04 R1 (audit QI-2): d log(1+z') supplies the 1/(1+z'), the post-multiplication the (1+z).

    :return: dict with "value", "abserr" (scipy's estimate, scaled like "value") and "data"
        (an IntegrationData record)
    """
    Gk_f: GkSourceFunctions = GkPolicy.functions
    model_f: ModelFunctions = model.functions

    if source_f is None:
        source_f = source.functions.source

    if GkPolicy.type not in ["numeric", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk object is not of "numeric" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.numeric_Gk is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numeric_Gk is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, {_Gk_diagnostics(GkPolicy)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.numeric_region is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numeric_region is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, {_Gk_diagnostics(GkPolicy)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    # This duplicates the "Gk numeric" check build_partition already made on the same region over
    # the same sub-interval (_check_region_covers), and it must be made in the same variable, for
    # the same reason: min_z and max_z reach here as exp(log_z) - 1 (build_partition:439-440), and
    # at z ~ 5e14 that round trip moves z by ~1e7 times DEFAULT_FLOAT_PRECISION. Comparing in z
    # with an absolute tolerance let the sign of the rounding decide whether the guard fired, and
    # because the Green's-function numeric region ends exactly at z_response the bottom of the
    # lowest sub-interval always coincides with the region boundary. build_partition raised first
    # and masked this copy: with that guard corrected, 771 of 3185 production work items raised
    # here instead (prompts/source-remediation/logs/13-region-guard-tolerance.md). Recovering
    # log(1+z) from the round-tripped z costs ~1 ulp of the log and is safe; it is only the other
    # direction, log(1+z) -> z, that is lossy at large z.
    region_max_z, region_min_z = Gk_f.numeric_region
    if region_min_z <= -1.0 or region_max_z <= -1.0:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but the region ({region_max_z:.5g}, {region_min_z:.5g}) where a numerical solution is available contains a redshift z <= -1, which has no log(1+z) [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if log_max_z > log(1.0 + region_max_z) + MIN_SUBINTERVAL_LOG_WIDTH:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z:.5g}) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if log_min_z < log(1.0 + region_min_z) - MIN_SUBINTERVAL_LOG_WIDTH:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z:.5g}) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    def integrand(log_z_source) -> float:
        Green = Gk_f.numeric_Gk(log_z_source, z_is_log=True)
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f(log_z_source, z_is_log=True)

        return Green * f / H_sq

    data = simple_quadrature(
        integrand,
        a=log_min_z,
        b=log_max_z,
        atol=atol,
        rtol=rtol,
        label=f"numeric_quad_integral for k={k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}), q={q.k_inv_Mpc:.5g}/Mpc (store_id={q.store_id}), r={r.k_inv_Mpc:.5g}/Mpc (store_id={r.store_id})",
        method="quad",
    )

    scale = 1.0 + get_z(z_response)
    data["value"] = scale * data["value"]
    # abserr is an absolute error, so it scales linearly under the same rescaling as "value"
    data["abserr"] = scale * data["abserr"]

    return data


class QuadSourceIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model: ModelProxy,
        policy: GkSourcePolicy,
        z_response: redshift,
        z_source_max: redshift,
        k: wavenumber_exit_time,
        q: wavenumber_exit_time,
        r: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._model_proxy = model
        self._policy = policy

        self._k_exit = k
        self._q_exit = q
        self._r_exit = r

        self._z_response = z_response
        self._z_source_max = z_source_max

        if payload is None:
            DatastoreObject.__init__(self, None)

            self._total = None
            self._total_abserr = None
            self._total_converged = None
            self._total_phase_limited = None
            self._b = None
            self._numeric_quad = None
            self._WKB_Levin = None

            self._numeric_quad_data = None
            self._WKB_Levin_data = None
            self._WKB_phase_spline_chunks = None

            self._eta_source_max = None
            self._eta_response = None
            self._analytic_rad = None

            self._compute_time = None
            self._analytic_compute_time = None

            self._data_serial = None
            self._source_serial = None

            self._metadata = {}

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._total = payload["total"]
            self._total_abserr = payload["total_abserr"]
            self._total_converged = payload["total_converged"]
            self._total_phase_limited = payload["total_phase_limited"]
            self._b = payload["b"]
            self._numeric_quad = payload["numeric_quad"]
            self._WKB_Levin = payload["WKB_Levin"]
            self._WKB_phase_spline_chunks = payload["WKB_phase_spline_chunks"]

            self._eta_source_max = payload["eta_source_max"]
            self._eta_response = payload["eta_response"]
            self._analytic_rad = payload["analytic_rad"]

            self._source_serial = payload["source_serial"]
            self._data_serial = payload["data_serial"]

            self._numeric_quad_data = payload["numeric_quad_data"]
            self._WKB_Levin_data = payload["WKB_Levin_data"]

            self._compute_time = payload["compute_time"]
            self._analytic_compute_time = payload["analytic_compute_time"]

            self._metadata = payload["metadata"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._atol = atol
        self._rtol = rtol

        self._compute_ref = None

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def q(self) -> wavenumber:
        return self._q_exit.k

    @property
    def r(self) -> wavenumber:
        return self._r_exit.k

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def z_source_max(self) -> redshift:
        return self._z_source_max

    @property
    def total(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total

    @property
    def total_abserr(self) -> Optional[float]:
        """
        Bound on the *quadrature* error of `total`: the linear sum of every sub-interval's
        absolute error estimate, scipy's on the all-smooth sub-intervals and the Levin driver's
        on the rest (audit B8). Absolute, deliberately: numeric_quad and WKB_Levin cancel, so do
        not turn this into a relative error by dividing by |total|.

        It does *not* include the error of the ingredients -- the QuadSource spline of f, the
        Liouville-Green closed forms, the re-splined phases, the hand-over clamp -- which is
        four to five orders larger at production settings and is recorded on the campaign's
        status board (prompts/source-remediation/IMPLEMENTATION_STATE.md section 3, issue
        [09-abserr-is-a-quadrature-bound]).
        """
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total_abserr

    @property
    def total_converged(self) -> Optional[bool]:
        """Whether every phase group of every Levin sub-interval reported convergence."""
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total_converged

    @property
    def total_phase_limited(self) -> Optional[bool]:
        """Whether any phase group of any Levin sub-interval was phase-limited."""
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total_phase_limited

    @property
    def b(self) -> Optional[float]:
        """
        The b at which this row was computed (audit B7): it fixes c_s^2 = (1-b)/(3(1+b)), the
        Bessel orders 1/2 + b and 5/2 + b, and the eta' weight of `analytic_rad`.
        """
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._b

    @property
    def numeric_quad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad

    @property
    def WKB_Levin(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin

    @property
    def analytic_rad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._analytic_rad

    @property
    def eta_source_max(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._eta_source_max

    @property
    def eta_response(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._eta_response

    @property
    def data_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._data_serial

    @property
    def source_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._source_serial

    @property
    def numeric_quad_data(self) -> Optional[IntegrationData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad_data

    @property
    def WKB_Levin_data(self) -> Optional[LevinData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin_data

    @property
    def compute_time(self) -> Optional[float]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._compute_time

    @property
    def analytic_compute_time(self) -> Optional[float]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._analytic_compute_time

    @property
    def metadata(self) -> dict:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._metadata

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    # payload keys compute() requires. The four Tk keys were added by prompts/source-remediation
    # prompt 08 and are supplied by main.py from prompt 10 onwards; until then compute() fails
    # loudly here rather than with a KeyError.
    REQUIRED_PAYLOAD_KEYS = (
        "source",
        "GkPolicy",
        "b",
        "Bessel_0pt5",
        "Bessel_2pt5",
        "Tq_numeric",
        "Tq_WKB",
        "Tr_numeric",
        "Tr_WKB",
    )

    def compute(self, payload, label: Optional[str] = None):
        if self._total is not None:
            raise RuntimeError(
                "QuadSourceIntegral: compute() called, but value has already been computed"
            )

        missing = [key for key in self.REQUIRED_PAYLOAD_KEYS if key not in payload]
        if len(missing) > 0:
            raise RuntimeError(
                f"QuadSourceIntegral: compute() payload is missing the required key(s) {', '.join(repr(key) for key in missing)} "
                f"(the transfer-function keys 'Tq_numeric', 'Tq_WKB', 'Tr_numeric', 'Tr_WKB' are the TkNumericIntegration and "
                f"TkWKBIntegration objects for q and r, needed since prompts/source-remediation prompt 08)"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        source: QuadSource = payload["source"]
        GkPolicy: GkSourcePolicyData = payload["GkPolicy"]

        if self._policy.store_id != GkPolicy.policy.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied GkSourcePolicyData object does not match specified policy (required policy store_id={self._policy.store_id}, supplied GkSourcePolicyData object has policy store_id={GkPolicy.policy.store_id})"
            )

        Gk = GkPolicy._source_proxy.get()

        # TODO: improve compatibility check between source and Gk
        if self._z_response.store_id != Gk.z_response.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied GkSource object does not match specified z_response (z_response={self._z_response.z:.5g}, GkSource object evaluated at z_response={Gk.z_response.z:.5g})"
            )

        if source.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied quadratic source term has maximum z_source={source.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk has maximum z_source={Gk.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.k.store_id != self._k_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={self._k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._k_exit.k.store_id}])"
            )

        if source.q.store_id != self._q_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={self._q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._q_exit.k.store_id}])"
            )

        if source.r.store_id != self._r_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={self._r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._r_exit.k.store_id}])"
            )

        # the transfer-function objects must describe the same q and r modes as the source
        for key, required in (
            ("Tq_numeric", self._q_exit),
            ("Tq_WKB", self._q_exit),
            ("Tr_numeric", self._r_exit),
            ("Tr_WKB", self._r_exit),
        ):
            Tk = payload[key]
            if Tk.k.store_id != required.k.store_id:
                raise RuntimeError(
                    f"QuadSourceIntegral: supplied {key} is evaluated for a mode that does not match the required value (supplied {key} is for k={Tk.k.k_inv_Mpc:.3g}/Mpc [store_id={Tk.k.store_id}], required value is k={required.k.k_inv_Mpc:.3g}/Mpc [store_id={required.k.store_id}])"
                )

        self._compute_ref = compute_QuadSource_integral.remote(
            self._model_proxy,
            self._k_exit,
            self._q_exit,
            self._r_exit,
            source=source,
            GkPolicy=GkPolicy,
            z_response=self._z_response,
            z_source_max=self._z_source_max,
            b=payload["b"],
            Bessel_0pt5=payload["Bessel_0pt5"],
            Bessel_2pt5=payload["Bessel_2pt5"],
            Tq_numeric=payload["Tq_numeric"],
            Tq_WKB=payload["Tq_WKB"],
            Tr_numeric=payload["Tr_numeric"],
            Tr_WKB=payload["Tr_WKB"],
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )

        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "QuadSourceIntegral: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None

        payload = ray.get(self._compute_ref)
        self._compute_ref = None

        self._total = payload["total"]
        self._total_abserr = payload["total_abserr"]
        self._total_converged = payload["total_converged"]
        self._total_phase_limited = payload["total_phase_limited"]
        self._b = payload["b"]
        self._numeric_quad = payload["numeric_quad"]
        self._WKB_Levin = payload["WKB_Levin"]

        self._eta_source_max = payload["eta_source_max"]
        self._eta_response = payload["eta_response"]
        self._analytic_rad = payload["analytic_rad"]

        self._data_serial = payload["GkPolicy_serial"]
        self._source_serial = payload["source_serial"]

        self._numeric_quad_data = payload["numeric_quad_data"]
        self._WKB_Levin_data = payload["WKB_Levin_data"]
        self._WKB_phase_spline_chunks = payload["WKB_phase_spline_chunks"]

        self._compute_time = payload["compute_time"]
        self._analytic_compute_time = payload["analytic_compute_time"]

        self._metadata = payload["metadata"]
