"""
What does a break-point-aware knot vector buy ``PrimitivePhase`` on the *corrected* background?

Written for prompt 10 of ``prompts/qcd-background-audit/``, which re-takes the measurement
``prompts/phase-representation`` prompt 02 stopped on. That prompt found a repeated
(multiplicity-``spline_order``) knot vector **singular on all six production grids** at
``BREAK_POINT_ALL``, and, where it did construct, **2x worse** at
``BREAK_POINT_DISCONTINUITY``. Both findings were taken against a background whose ``T(z)``
spline smeared the equation of state's genuine step over ~4 production grid intervals; prompt 09
measured that only **25.55 %** of the step in ``dlnH/du`` then fell inside the crossing's own grid
interval, against **99.84 %** of a step **2.78x taller** now
(``docs/qcd-background-verification.md`` §3.3). So the premise has changed and the measurement has
to be re-taken, not inherited.

**What this scores.** Exactly the geometry of ``docs/gktk-remedial/verify_production_path.py``'s
``consumers`` section -- the source of ``docs/gktk-remedial-verification.md`` §3.5 and §3.6 -- for
both models, both sectors and all three production wavenumbers, and for each of a family of knot
schemes. The expensive half (the producer runs that supply the stored samples and the dense
reference) is done **once** per (model, sector, k) and every scheme is scored against it, which is
what makes a scan of seven schemes affordable where seven runs of the verification script would
not be.

**Why it overrides the spline rather than passing ``break_points``.** The production parameter
carries one construction (a multiplicity-``spline_order`` knot at each declared break: the C0 knot
prompt 02 named). The controls this scan needs -- multiplicity 1 and 2, and the "no phi derivative
at all" column that separates the break points' share of ``theta_deriv`` from
``[02-consumer-phi-below-the-storage-granularity]``'s -- are not constructions anything should be
able to ask for in production. So the scan builds the production ``PrimitivePhase`` (through
``TkSourceFunctions`` in the ``Tk`` sector, exactly as the production call site does) and then
substitutes the spline object. ``base`` reproduces §3.5 and §3.6 to the printed digits, which is
this script's own check that it is measuring the same thing.

**The trap this exists to avoid** (prompt 02 log, "State handed to the next prompt", item 5):
*do not score a knot scheme at k = 1e5 alone*. The two schemes that helped there in 2026-09-13's
measurement were the two that regressed 1e7 and 3e8. Every table below is all three wavenumbers,
both sectors, both models.

No Ray, no datastore. ~60 s for the default scheme list.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_knot_scheme_scan.py
    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_knot_scheme_scan.py \
        --schemes base DISCx3 --json /tmp/scan.json
"""

import argparse
import json
import sys
import time
from math import fabs, pi, sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.interpolate import make_interp_spline  # noqa: E402

from ComputeTargets.BackgroundModel import (  # noqa: E402
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
    _cosmology_break_points,
)
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq  # noqa: E402
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq  # noqa: E402
from ComputeTargets.primitive_phase import (  # noqa: E402
    PrimitivePhase,
    build_phi_samples,
)
from ComputeTargets.TkSourceFunctions import TkSourceFunctions  # noqa: E402
from ComputeTargets.tests.test_gk_wkb_phase import (  # noqa: E402
    _run_Gk,
    _unwrapped,
    lambdacdm_model_with_tables,
    qcd_model_with_tables,
)
from ComputeTargets.tests.test_tk_wkb_phase import build_and_store  # noqa: E402
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    PRODUCTION_Z_END,
    horizon_exit_z,
    load_references,
    phase_error,
    production_source_grid,
    to_redshift_array,
)
from LiouvilleGreen.constants import TWO_PI  # noqa: E402

K_VALUES = (1.0e5, 1.0e7, 3.0e8)
MODEL_KEYS = ("LambdaCDMModel", "QCDModel")
CONSUMER_POINTS_PER_INTERVAL = 10
SPLINE_ORDER = 3

# (label, kind or None, multiplicity). multiplicity 0 is the control that removes the phi
# derivative altogether -- not a scheme, the denominator of the theta_deriv attribution.
SCHEMES = {
    "base": (None, 0),
    "DISCx1": (BREAK_POINT_DISCONTINUITY, 1),
    "DISCx2": (BREAK_POINT_DISCONTINUITY, 2),
    "DISCx3": (BREAK_POINT_DISCONTINUITY, 3),
    "ALLx1": (BREAK_POINT_ALL, 1),
    "ALLx2": (BREAK_POINT_ALL, 2),
    "ALLx3": (BREAK_POINT_ALL, 3),
    "segDISC": (BREAK_POINT_DISCONTINUITY, "segment"),
    "segALL": (BREAK_POINT_ALL, "segment"),
}
DEFAULT_SCHEMES = (
    "base",
    "segDISC",
    "segALL",
    "DISCx3",
    "ALLx3",
    "DISCx1",
    "ALLx1",
    "DISCx2",
    "ALLx2",
)


def repeated_knot_vector(sites, breaks, order: int, multiplicity: int) -> np.ndarray:
    """
    ``make_interp_spline``'s interpolating knot vector for ``sites``, with a knot of the given
    multiplicity at each break point, paid for **locally**.

    This is ``ComputeTargets/tests/test_numeric_break_points.py::_repeated_knot_vector`` widened
    by one parameter; prompt 02 of ``prompts/phase-representation`` measured that local payment is
    the only placement that can satisfy Schoenberg-Whitney (a fixed-length knot vector tolerates
    at most one net removal below any site and one above, so knots taken from the smooth interior
    fail at dozens of sites).
    """
    half = order // 2 + 1
    interior = list(float(t) for t in sites[half:-half])
    for b in sorted(float(x) for x in breaks):
        nearest = sorted(range(len(interior)), key=lambda i: abs(interior[i] - b))
        drop = set(nearest[:multiplicity])
        interior = [t for i, t in enumerate(interior) if i not in drop]
        interior.extend([b] * multiplicity)
        interior.sort()
    ends = [float(sites[0])] * (order + 1), [float(sites[-1])] * (order + 1)
    return np.asarray(ends[0] + interior + ends[1], dtype=float)


class SegmentedSpline:
    """
    One interpolating spline per interval between the declared break points, with a dispatch on
    evaluation. Strictly freer than a multiplicity-``order`` knot -- it drops even C0 -- and,
    unlike a repeated-knot vector, it **steals no knots**: every segment keeps its own data sites
    as knots, so the spline is not coarsened in the intervals around the break, which is what
    prompt 02 of ``prompts/phase-representation`` measured a repeated knot to cost.

    Each break point lies strictly inside a grid interval (measured: fractional position
    0.32-0.64 on the production grids), so the samples nearest it belong to different segments and
    the interval containing it is covered by a short **extrapolation** from each side, out to the
    break and no further.
    """

    def __init__(self, sites, values, breaks, order):
        sites = np.asarray(sites, dtype=float)
        values = np.asarray(values, dtype=float)
        self._breaks = np.asarray(sorted(float(b) for b in breaks), dtype=float)
        edges = [-np.inf, *self._breaks, np.inf]
        self._splines = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sites > lo) & (sites < hi)
            if int(mask.sum()) < order + 1:
                raise RuntimeError(
                    f"SegmentedSpline: segment ({lo:.6g}, {hi:.6g}) holds {int(mask.sum())} "
                    f"samples, fewer than the {order + 1} an order-{order} spline needs"
                )
            self._splines.append(make_interp_spline(sites[mask], values[mask], k=order))

    def __call__(self, u):
        return self._splines[int(np.searchsorted(self._breaks, float(u)))](u)

    def derivative(self):
        out = object.__new__(SegmentedSpline)
        out._breaks = self._breaks
        out._splines = [s.derivative() for s in self._splines]
        return out


class _ZeroSpline:
    """The control: phi contributes nothing, so theta_deriv is the closed-form leading term
    alone. Used only for the `phi'=0` column of the theta_deriv attribution."""

    def __call__(self, u):
        return 0.0 * np.asarray(u, dtype=float)

    def derivative(self):
        return self


def _apply_scheme(phase: PrimitivePhase, cosmology, scheme: str):
    """Substitute the phi spline of ``phase`` for the one ``scheme`` names. Returns
    ``(n_breaks, constructed)``; a scheme that raises ``LinAlgError`` leaves the phase untouched
    and reports ``constructed=False``."""
    kind, multiplicity = SCHEMES[scheme]
    sites = phase._u_points
    breaks = (
        np.empty(0)
        if kind is None
        else _cosmology_break_points(
            cosmology, float(phase.min_x), float(phase.max_x), kind=kind
        )
    )
    try:
        if multiplicity == "segment" and len(breaks) > 0:
            spline = SegmentedSpline(sites, phase._phi_points, breaks, SPLINE_ORDER)
        else:
            t = (
                None
                if (kind is None or len(breaks) == 0 or multiplicity == "segment")
                else repeated_knot_vector(sites, breaks, SPLINE_ORDER, multiplicity)
            )
            spline = make_interp_spline(sites, phase._phi_points, k=SPLINE_ORDER, t=t)
    except Exception as exc:  # LinAlgError and friends
        return len(breaks), False, repr(exc)
    phase._spline = spline
    phase._spline_deriv = spline.derivative()
    return len(breaks), True, None


def _zero_phi(phase: PrimitivePhase):
    phase._spline = _ZeroSpline()
    phase._spline_deriv = _ZeroSpline()


def _midpoint_grid(z_nodes, points_per_interval: int) -> np.ndarray:
    u = np.log1p(np.asarray(z_nodes, dtype=float))
    out = []
    for i in range(len(u) - 1):
        a, b = u[i], u[i + 1]
        for j in range(1, points_per_interval + 1):
            out.append(a + (b - a) * j / (points_per_interval + 1.0))
    return np.expm1(np.array(out, dtype=float))


def _score_phase(phase, ref_theta, delta=0.0):
    worst, worst_z, span = -1.0, None, 0.0
    for z, t_ref in ref_theta.items():
        err = phase_error(phase.raw_theta(z), t_ref + delta)
        span = max(span, fabs(t_ref))
        if err > worst:
            worst, worst_z = err, z
    ulp = float(np.spacing(span))
    return {
        "worst": float(worst),
        "worst_z": float(worst_z),
        "span": float(span),
        "ulp": worst / ulp if ulp > 0 else float("nan"),
    }


def _score_deriv(model, k, phase, z_nodes, omega_sq):
    z_sorted = np.sort(np.asarray(z_nodes, dtype=float))
    errs = []
    for z in z_sorted:
        omega = sqrt(fabs(omega_sq(model, float(k), float(z))))
        errs.append(fabs(fabs(phase.theta_deriv(float(z))) - omega) / omega)
    errs = np.array(errs)
    n = len(errs)
    return {
        "max_all": float(errs.max()),
        "max_at_z": float(z_sorted[int(errs.argmax())]),
        "max_3_3": float(errs[3 : n - 3].max()) if n > 6 else float("nan"),
        "interior": float(errs[10 : n - 10].max()) if n > 20 else float("nan"),
    }


class _Wavenumber:
    def __init__(self, k: float):
        self._k = float(k)

    def __float__(self):
        return self._k


class _FakeNumericSample:
    def __init__(self, z: float, T: float, Tprime: float):
        self.z = type("Z", (), {"z": float(z)})()
        self.T = float(T)
        self.Tprime = float(Tprime)


class _FakeTkNumeric:
    """``verify_production_path.py``'s stand-in, copied so the two harnesses build the same
    ``TkSourceFunctions``: the samples at or above the hand-over that ``_build_numeric``
    requires, and nothing else."""

    def __init__(self, z_init: float, model):
        zs = [z_init * (1.0 + 0.02 * j) for j in range(8)][::-1]
        self.values = [_FakeNumericSample(z, 1.0e-3, 0.0) for z in zs]
        self.stop_deltaz_subh = None
        self.z_exit = None


def build_cases(models, k_values, verbose=True):
    """The expensive half: the stored samples and the dense reference, once per
    (model, sector, k)."""
    references = load_references()
    grid = production_source_grid(
        references["models"]["LambdaCDMModel"]["grid"]["z_init"]
    )
    z_nodes = np.array(grid.as_float_list(), dtype=float)

    built = {}
    cases = []
    for model_key in models:
        if model_key not in built:
            t0 = time.perf_counter()
            built[model_key] = (
                lambdacdm_model_with_tables(z_nodes)
                if model_key == "LambdaCDMModel"
                else qcd_model_with_tables(grid)
            )
            if verbose:
                print(
                    f"   [built {model_key} with its tables in "
                    f"{time.perf_counter() - t0:.2f} s]"
                )
        model = built[model_key]
        units = model.cosmology.units
        cosmology = model.cosmology

        for k in k_values:
            k_key = f"{k:.6e}"

            # ------------------------------ Gk ------------------------------
            z_e3 = horizon_exit_z(cosmology, k, 3.0)
            z_e4 = horizon_exit_z(cosmology, k, 4.0)
            z_source_limit = sqrt(z_e3 * z_e4)
            z_r = PRODUCTION_Z_END
            band = z_nodes[(z_nodes < z_source_limit) & (z_nodes > z_r)]
            if len(band) >= 8:
                theta_stored = []
                for z_s in band:
                    p = _run_Gk(model, units, k, float(z_s), to_redshift_array([z_r]))
                    theta_stored.append(
                        p["theta_div_2pi_sample"][0] * TWO_PI
                        + p["theta_mod_2pi_sample"][0]
                    )
                z_points = [float(z) for z in band]
                phi_points = build_phi_samples(
                    k, model.functions.tau, z_r, z_points, theta_stored, sign=-1
                )
                mid = _midpoint_grid(np.array(z_points), CONSUMER_POINTS_PER_INTERVAL)
                mid = mid[(mid > min(z_points)) & (mid < max(z_points))]
                ref_payload = _run_Gk(
                    model, units, k, z_r, to_redshift_array(sorted(mid, reverse=True))
                )
                ref_theta = {
                    z: -t
                    for z, t in zip(sorted(mid, reverse=True), _unwrapped(ref_payload))
                }
                cases.append(
                    {
                        "model_key": model_key,
                        "model": model,
                        "cosmology": cosmology,
                        "k": k,
                        "k_key": k_key,
                        "sector": "Gk",
                        "z_points": z_points,
                        "phi_points": phi_points,
                        "ref_theta": ref_theta,
                        "delta": 0.0,
                        "omega_sq": Gk_omegaEff_sq,
                        "make_phase": (
                            lambda model=model, k=k, z_r=z_r, z_points=z_points, phi_points=phi_points: PrimitivePhase(
                                k,
                                model.functions.tau,
                                z_r,
                                z_points,
                                phi_points,
                                sign=-1,
                                model_functions=model.functions,
                                label="G_k WKB phase (scan)",
                            )
                        ),
                        "deriv_nodes": z_points,
                    }
                )
                if verbose:
                    print(
                        f"   [{model_key} Gk k={k:.3e}] {len(z_points)} sources, "
                        f"{len(ref_theta)} reference points"
                    )

            # ------------------------------ Tk ------------------------------
            z_init = float(references["models"][model_key]["rho_anchor_z"][k_key])
            wkb_nodes = z_nodes[z_nodes <= z_init]
            if len(wkb_nodes) < 8:
                continue
            wkb_samples = to_redshift_array([float(z) for z in wkb_nodes])
            tk_obj = build_and_store(
                model,
                k,
                z_init,
                wkb_samples,
                T_init=1.0e-3,
                Tprime_init=0.0,
                units=units,
            )
            functions = TkSourceFunctions(
                model, _Wavenumber(k), _FakeTkNumeric(z_init, model), tk_obj
            )
            mid_t = _midpoint_grid(wkb_nodes, CONSUMER_POINTS_PER_INTERVAL)
            mid_t = mid_t[
                (mid_t > float(wkb_nodes[-1])) & (mid_t < float(wkb_nodes[0]))
            ]
            from ComputeTargets.tests.test_gk_wkb_phase import _run_Tk

            ref_t_payload = _run_Tk(
                model, units, k, z_init, to_redshift_array(sorted(mid_t, reverse=True))
            )
            ref_t = dict(zip(sorted(mid_t, reverse=True), _unwrapped(ref_t_payload)))
            from ComputeTargets.phase_residual import cached_phase_residual

            def tk_delta(model=model, k=k, z_init=z_init, tk_obj=tk_obj):
                stored_first = max(tk_obj.values, key=lambda v: v.z.z)
                rho, _ = cached_phase_residual(
                    model,
                    float(k),
                    model.functions.cs_tau.table.z_nodes,
                    "Tk",
                    store_id=None,
                )
                from_tables = -(
                    float(k) * model.functions.cs_tau.delta(z_init, stored_first.z.z)
                    + rho.delta(z_init, stored_first.z.z)
                )
                return (
                    stored_first.theta_div_2pi * TWO_PI + stored_first.theta_mod_2pi
                ) - from_tables

            cases.append(
                {
                    "model_key": model_key,
                    "model": model,
                    "cosmology": cosmology,
                    "k": k,
                    "k_key": k_key,
                    "sector": "Tk",
                    "z_points": [float(z) for z in wkb_nodes],
                    "phi_points": np.asarray(functions.phase.phi_samples),
                    "ref_theta": ref_t,
                    "delta": tk_delta(),
                    "omega_sq": Tk_omegaEff_sq,
                    "make_phase": (
                        lambda model=model, k=k, z_init=z_init, tk_obj=tk_obj: TkSourceFunctions(
                            model, _Wavenumber(k), _FakeTkNumeric(z_init, model), tk_obj
                        ).phase
                    ),
                    "deriv_nodes": [float(z) for z in wkb_nodes],
                }
            )
            if verbose:
                print(
                    f"   [{model_key} Tk k={k:.3e}] {len(wkb_nodes)} samples, "
                    f"{len(ref_t)} reference points"
                )
    return cases


# ---------------------------------------------------------------------------------------------
# the resolution ladder: is the feature at the crossing the knot vector's problem or the grid's?
# ---------------------------------------------------------------------------------------------


def resolution_ladder(case, u_break: float):
    """
    The decisive control. Reconstruct ``phi`` from the **dense reference** (ten points per
    production interval), then ask what a plain cubic spline of it -- default knots, no break-point
    treatment whatever -- achieves when it is given *more samples* instead of better knots.

    If a denser sample set reaches the floor where no knot vector does, the residue at the
    crossing is the production source grid's and not ``PrimitivePhase``'s, and it belongs to the
    grid campaign (``prompts/qcd-background-audit`` G2, prompts 11-12) rather than to
    ``[13-consumer-spline-crosses-eos-break-points]``. That is
    ``prompts/phase-representation/IMPLEMENTATION_STATE.md`` §5 note 6 -- *a knot vector cannot
    resolve a break the sample grid does not resolve* -- put to the test on the corrected
    background rather than inherited.

    Every ordinate below is the reference's own ``phi``; the stored node values are kept exactly
    where they exist, so the ``1x`` row *is* the shipped consumer.
    """
    phase = case["make_phase"]()
    sign, lead, z_anchor, k = phase.sign, phase._leading, phase.z_anchor, case["k"]

    z_ref = np.array(sorted(case["ref_theta"].keys()))
    u_ref = np.log1p(z_ref)
    theta_ref = np.array([case["ref_theta"][z] + case["delta"] for z in z_ref])
    phi_ref = theta_ref - sign * k * np.array(
        [lead.delta(float(z), z_anchor) for z in z_ref]
    )

    u_node = np.log1p(np.asarray(phase.z_samples))
    phi_node = np.asarray(phase.phi_samples)
    h = float(np.median(np.diff(u_node)))
    span = max(fabs(v) for v in case["ref_theta"].values())
    ulp = float(np.spacing(span))
    j = int(np.searchsorted(u_node, u_break))

    def score(u_fit, phi_fit, tag):
        spline = make_interp_spline(u_fit, phi_fit, k=SPLINE_ORDER)
        fitted = set(np.round(u_fit, 15))
        idx = [i for i in range(len(u_ref)) if round(u_ref[i], 15) not in fitted]
        err = np.array([abs(float(spline(u_ref[i])) - phi_ref[i]) for i in idx])
        near = np.array(
            [
                abs(float(spline(u_ref[i])) - phi_ref[i])
                for i in idx
                if abs(u_ref[i] - u_break) < 3.0 * h
            ]
        )
        return {
            "tag": tag,
            "n_samples": len(u_fit),
            "worst": float(err.max()),
            "worst_ulp": float(err.max() / ulp),
            "near_ulp": float(near.max() / ulp) if near.size else float("nan"),
        }

    def with_extra(u_extra):
        u_fit = np.sort(np.concatenate([u_node, np.asarray(u_extra, dtype=float)]))
        phi_fit = np.interp(u_fit, u_ref, phi_ref)
        phi_fit[np.searchsorted(u_fit, u_node)] = (
            phi_node  # keep the stored values exact
        )
        return u_fit, phi_fit

    rows = [
        score(u_node, phi_node, "production grid, default knots (the shipped consumer)")
    ]

    # (a) refine the break's own interval alone
    for n_extra in (1, 2, 4):
        lo, hi = u_node[j - 1], u_node[j]
        extra = [lo + t * (hi - lo) for t in np.linspace(0.0, 1.0, n_extra + 2)[1:-1]]
        rows.append(
            score(
                *with_extra(extra),
                f"+{n_extra} sample(s) inside the break interval alone",
            )
        )

    # (b) refine a neighbourhood of the break
    for width in (1, 2, 3, 5, 10):
        for factor in (2, 5):
            extra = []
            for i in range(max(j - width, 0), min(j + width, len(u_node) - 1)):
                for t in np.linspace(0.0, 1.0, factor + 1)[1:-1]:
                    extra.append(u_node[i] + t * (u_node[i + 1] - u_node[i]))
            rows.append(
                score(*with_extra(extra), f"refine +-{width} interval(s) by {factor}x")
            )

    # (c) uniform refinement, for scale
    for stride, factor in ((5, 2), (2, 5)):
        rows.append(
            score(
                u_ref[::stride],
                phi_ref[::stride],
                f"uniform {factor}x over the whole range",
            )
        )

    return {"ulp": ulp, "h": h, "rows": rows}


def kink_fit(case, u_break: float):
    """
    ``prompts/phase-representation`` prompt 02's item 3, re-taken on the corrected background --
    the `GkTk-remedial` board's explicit instruction to prompt 10, because that fit was made on a
    step the old ``T(z)`` spline had smeared over about four grid intervals.

    One-sided cubics are fitted to ``phi`` reconstructed from the dense reference, over windows of
    1, 2 and 3 production grid intervals either side of the crossing. **A genuine slope
    discontinuity gives a window-independent jump**; smooth-but-unresolved data does not, and that
    distinction is what decides whether a C0 knot at the break can help at all.
    """
    phase = case["make_phase"]()
    sign, lead, z_anchor, k = phase.sign, phase._leading, phase.z_anchor, case["k"]
    z_ref = np.array(sorted(case["ref_theta"].keys()))
    u_ref = np.log1p(z_ref)
    phi_ref = np.array(
        [case["ref_theta"][z] + case["delta"] for z in z_ref]
    ) - sign * k * np.array([lead.delta(float(z), z_anchor) for z in z_ref])
    h = float(np.median(np.diff(np.log1p(np.asarray(phase.z_samples)))))

    rows = []
    for w in (1.0, 2.0, 3.0):
        lo = (u_ref < u_break) & (u_ref > u_break - w * h)
        hi = (u_ref > u_break) & (u_ref < u_break + w * h)
        if int(lo.sum()) < 5 or int(hi.sum()) < 5:
            continue
        cl = np.polyfit(u_ref[lo] - u_break, phi_ref[lo], 3)
        ch = np.polyfit(u_ref[hi] - u_break, phi_ref[hi], 3)
        d_phi = float(np.polyval(ch, 0.0) - np.polyval(cl, 0.0))
        d_dphi = float(
            np.polyval(np.polyder(ch), 0.0) - np.polyval(np.polyder(cl), 0.0)
        )
        rows.append(
            {
                "window_h": w,
                "n_lo": int(lo.sum()),
                "n_hi": int(hi.sum()),
                "jump_phi": d_phi,
                "jump_dphi": d_dphi,
                "kink_term": abs(d_dphi) * h / 8.0,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", nargs="+", default=list(DEFAULT_SCHEMES))
    parser.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    parser.add_argument("--k", nargs="+", type=float, default=list(K_VALUES))
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--no-ladder",
        action="store_true",
        help="skip section E (the sample-density control)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 100)
    print("CONSUMER KNOT-SCHEME SCAN -- prompt 10, prompts/qcd-background-audit")
    print("=" * 100)
    cases = build_cases(args.models, args.k)

    results = {}
    for case in cases:
        key = (case["model_key"], case["sector"], case["k_key"])
        row = {}
        for scheme in args.schemes:
            phase = case["make_phase"]()
            n_breaks, ok, err = _apply_scheme(phase, case["cosmology"], scheme)
            if not ok:
                row[scheme] = {"constructed": False, "error": err, "n_breaks": n_breaks}
                continue
            entry = _score_phase(phase, case["ref_theta"], case["delta"])
            entry.update(
                {
                    "constructed": True,
                    "n_breaks": n_breaks,
                    "deriv": _score_deriv(
                        case["model"],
                        case["k"],
                        phase,
                        case["deriv_nodes"],
                        case["omega_sq"],
                    ),
                }
            )
            row[scheme] = entry
        # the attribution control: phi' removed altogether
        phase = case["make_phase"]()
        _zero_phi(phase)
        row["phi_zero"] = {
            "constructed": True,
            "n_breaks": 0,
            "deriv": _score_deriv(
                case["model"],
                case["k"],
                phase,
                case["deriv_nodes"],
                case["omega_sq"],
            ),
        }
        phi = np.asarray(case["phi_points"], dtype=float)
        span = max(abs(v) for v in case["ref_theta"].values())
        row["phi_range_ulp"] = float((phi.max() - phi.min()) / np.spacing(span))
        results[key] = row

    _report(results, args.schemes)

    if not args.no_ladder:
        print("\n" + "=" * 100)
        print(
            "E. THE RESOLUTION LADDER -- a plain cubic of the same phi, given more SAMPLES"
        )
        print("=" * 100)
        for case in cases:
            breaks = (
                np.empty(0)
                if not hasattr(case["cosmology"], "integration_break_points")
                else _cosmology_break_points(
                    case["cosmology"],
                    min(case["z_points"]),
                    max(case["z_points"]),
                    kind=BREAK_POINT_DISCONTINUITY,
                )
            )
            if len(breaks) == 0:
                continue
            ladder = resolution_ladder(case, float(breaks[0]))
            print(
                f"\n   {case['model_key']} {case['sector']} k = {case['k']:.3e}: "
                f"1 ulp = {ladder['ulp']:.4e} rad, grid spacing h = {ladder['h']:.4e} in u"
            )
            for r in ladder["rows"]:
                print(
                    f"      {r['tag']:<56} {r['n_samples']:>6} samples  "
                    f"whole range {r['worst']:.4e} rad = {r['worst_ulp']:8.2f} ulp | "
                    f"near the break {r['near_ulp']:8.2f} ulp"
                )
            fits = kink_fit(case, float(breaks[0]))
            if fits:
                print(
                    "      one-sided cubic fits either side of the crossing "
                    "(prompt 02 item 3, re-taken):"
                )
                for r in fits:
                    print(
                        f"         window {r['window_h']:>3.0f} h "
                        f"({r['n_lo']}/{r['n_hi']} pts):  [phi] = {r['jump_phi']:+.4e}   "
                        f"[phi'] = {r['jump_dphi']:+.4e}   "
                        f"|[phi']| h/8 = {r['kink_term']:.4e} rad"
                    )
            entry = results[(case["model_key"], case["sector"], case["k_key"])]
            entry["ladder"] = ladder
            entry["kink_fit"] = fits

    print(f"\nTotal wall time {time.perf_counter() - t0:.1f} s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=1, default=str)
        print(f"Raw results written to {args.json}")


def _report(results, schemes):
    print("\n" + "=" * 100)
    print("A. CONSUMER PHASE ERROR  max |theta - ref|, rad (ulp of the span)")
    print("=" * 100)
    header = f"{'model':<14}{'sector':<7}{'k':<11}" + "".join(
        f"{s:>22}" for s in schemes
    )
    print(header)
    for (model_key, sector, k_key), row in results.items():
        cells = ""
        for s in schemes:
            e = row.get(s)
            if e is None or not e["constructed"]:
                cells += f"{'SINGULAR':>22}"
            else:
                cells += f"{e['worst']:>12.4e}({e['ulp']:5.1f})"
        print(f"{model_key:<14}{sector:<7}{float(k_key):<11.2e}{cells}")

    print("\n" + "=" * 100)
    print(
        "B. theta_deriv vs omega, deep interior (relative); 'phi_zero' removes phi' entirely"
    )
    print("=" * 100)
    cols = list(schemes) + ["phi_zero"]
    print(
        f"{'model':<14}{'sector':<7}{'k':<11}"
        + "".join(f"{s:>14}" for s in cols)
        + f"{'phi range/ulp':>16}"
    )
    for (model_key, sector, k_key), row in results.items():
        cells = ""
        for s in cols:
            e = row.get(s)
            if e is None or not e["constructed"]:
                cells += f"{'SINGULAR':>14}"
            else:
                cells += f"{e['deriv']['interior']:>14.4e}"
        print(
            f"{model_key:<14}{sector:<7}{float(k_key):<11.2e}{cells}"
            f"{row['phi_range_ulp']:>16.1f}"
        )

    print("\n" + "=" * 100)
    print("C. theta_deriv vs omega, max over all samples (relative)")
    print("=" * 100)
    print(f"{'model':<14}{'sector':<7}{'k':<11}" + "".join(f"{s:>14}" for s in cols))
    for (model_key, sector, k_key), row in results.items():
        cells = ""
        for s in cols:
            e = row.get(s)
            if e is None or not e["constructed"]:
                cells += f"{'SINGULAR':>14}"
            else:
                cells += f"{e['deriv']['max_all']:>14.4e}"
        print(f"{model_key:<14}{sector:<7}{float(k_key):<11.2e}{cells}")

    print("\n" + "=" * 100)
    print("D. where the consumer-phase maximum sits, and how many breaks were declared")
    print("=" * 100)
    for (model_key, sector, k_key), row in results.items():
        base = row.get("base")
        parts = []
        for s in schemes:
            e = row.get(s)
            if e is not None and e["constructed"]:
                parts.append(f"{s}: z={e['worst_z']:.6g} (n={e['n_breaks']})")
        print(f"{model_key:<14}{sector:<5}{float(k_key):<10.2e} " + "; ".join(parts))


if __name__ == "__main__":
    main()
