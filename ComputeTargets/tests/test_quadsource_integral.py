"""
Tests for ComputeTargets/QuadSourceIntegral.py after prompts/source-remediation prompt 08: the
partition of the source time integral at the hand-over redshifts of G, T_q and T_r, ordinary
quadrature on the all-smooth sub-interval and one adaptive_levin_sincos call per phase group of
ComputeTargets.phase_groups everywhere else.

Offline: no Ray runtime, no datastore. Everything is built on the exact constant-w background of
ComputeTargets/tests/test_tk_source_functions.py, in two flavours:

  * "exact" -- the transfer functions are ExactTk stand-ins (test_phase_groups) injected through
    evaluate_QuadSource_integral's Tk_functions_builder hook, the Green's function is ExactGk and
    the source spline is replaced by the exact f. Every ingredient is then exact to rounding and
    the comparison against the oracles tests the partition and the integrator alone.
  * "realistic" -- the transfer functions are TkSourceFunctions built from prompt 05's exact
    fixture (bessel_phase amplitude and phase re-splined on the production 100-per-decade grid,
    Liouville-Green closed forms for omega and d ln M/dz), the Green's function's phase is a real
    phase_spline through bessel_phase samples (BesselPhaseGk) and f is a spline through exact
    samples on the production grid, as QuadSource stores it. This carries the representation
    floors logs 05, 06 and 07 measured, and is the accuracy production can expect.

The code's own oracle is analytic_integral (spec 04 R14, audit QI-1); a second, independent
oracle is scipy.quad of the exact integrand on a short range.
"""

import importlib
import importlib.util
import subprocess
import tempfile
import unittest
from math import log, exp, sqrt, inf
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sqlalchemy as sqla
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline

from ComputeTargets.GkSourcePolicyData import GkSourceFunctions
from ComputeTargets.QuadSource import QuadSourceFunctions, source_function
from ComputeTargets.QuadSourceIntegral import (
    QuadSourceIntegral,
    analytic_integral,
    evaluate_QuadSource_integral,
    phase_group_Levin_integral,
    build_partition,
    _ClampedTk,
    _check_region_covers,
    numeric_quad_integral,
    LEVIN_USE_THETA_DERIV,
    HANDOVER_CLAMP_MAX_GRID_STEPS,
    MIN_SUBINTERVAL_LOG_WIDTH,
    BESSEL_ORDER_CHECK_TOL,
)
from ComputeTargets.analytic_Gk import compute_analytic_G
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.phase_groups import (
    build_phase_groups,
    evaluate_sum,
    evaluate_envelope,
)
from ComputeTargets.spline_wrappers import ZSplineWrapper
from config.defaults import DEFAULT_FLOAT_PRECISION
import ComputeTargets.tests.test_tk_source_functions as tk_fixtures
from ComputeTargets.tests.test_tk_source_functions import (
    Fixture,
    FakeModel,
    log_grid,
    PRODUCTION_SAMPLES_PER_LOG10Z,
)
from ComputeTargets.tests.test_phase_groups import ExactTk, ExactGk, BesselPhaseGk
from LiouvilleGreen.bessel_phase import bessel_phase

# the two values of b the prompt asks for; w = (1 - b) / (3 (1 + b))
B_VALUES = (0.0, 0.2)

# the pre-commit QuadSourceIntegral.py, for the old-regime regression (prompt 08 section 7 item 5)
PRE_COMMIT_SHA = "39ed7fc"

REPO_ROOT = Path(__file__).resolve().parents[2]

# the module itself, for the tolerance spies. `ComputeTargets.QuadSourceIntegral` as an attribute
# of the package is the *class* (ComputeTargets/__init__.py re-exports it), so the module has to
# be fetched explicitly.
qsi_module = importlib.import_module("ComputeTargets.QuadSourceIntegral")


def w_of_b(b: float) -> float:
    return (1.0 - b) / (3.0 * (1.0 + b))


def z_at_tau(model: FakeModel, tau: float) -> float:
    """Invert FakeModel.tau: a0 eta = 1 / (H0 (p-1) (1+z)^(p-1))."""
    p = model.p
    return (1.0 / (model.H0 * (p - 1.0) * tau)) ** (1.0 / (p - 1.0)) - 1.0


# =============================================================================================
# stand-ins shaped like the objects evaluate_QuadSource_integral touches
# =============================================================================================


class FakeWavenumber:
    """wavenumber: .k is k/a0, float() likewise; k_inv_Mpc/store_id feed log labels."""

    def __init__(self, k: float, store_id: int):
        self.k = k
        self.k_inv_Mpc = k
        self.store_id = store_id

    def __float__(self):
        return float(self.k)


class FakeExitTime:
    """wavenumber_exit_time: .k is the wavenumber, .z_exit horizon crossing."""

    def __init__(self, k: float, store_id: int, z_exit: float):
        self.k = FakeWavenumber(k, store_id)
        self.store_id = store_id
        self.z_exit = z_exit


class FakeZ:
    def __init__(self, z: float, store_id: int = 0):
        self.z = z
        self.store_id = store_id


class FakeZSample:
    """redshift_array as the integral reads it: .max, .min, len()."""

    def __init__(self, z_values):
        self._z = sorted(z_values, reverse=True)
        self.max = FakeZ(self._z[0])
        self.min = FakeZ(self._z[-1])

    def __len__(self):
        return len(self._z)

    def __iter__(self):
        return (FakeZ(z) for z in self._z)


class FakeQuadSource:
    """
    QuadSource after prompt 06: `functions` is a spline of f valid on numeric_region =
    (z_max, z_min) -- here a spline through exact samples on the production grid, or the exact
    f itself (exact=True) -- and the two hand-over redshifts are exposed.
    """

    store_id = 11

    def __init__(
        self,
        q: FakeWavenumber,
        r: FakeWavenumber,
        z_grid,
        f_exact,
        crossover_z_q,
        crossover_z_r,
        exact: bool = False,
    ):
        self.q = q
        self.r = r
        self.z_sample = FakeZSample(z_grid)
        z_max, z_min = self.z_sample.max.z, self.z_sample.min.z
        self.numeric_region = (z_max, z_min)
        self.crossover_z_q = crossover_z_q
        self.crossover_z_r = crossover_z_r

        if exact:

            def source(x, z_is_log=False):
                z = exp(x) - 1.0 if z_is_log else x
                return f_exact(z)

        else:
            log_x = sorted(log(1.0 + z) for z in z_grid)
            y = [f_exact(exp(u) - 1.0) for u in log_x]
            source = ZSplineWrapper(
                make_interp_spline(log_x, y),
                "quadratic source",
                z_max,
                z_min,
                log_z=True,
            )

        self.functions = QuadSourceFunctions(
            source=source, numeric_region=self.numeric_region
        )


class FakeGkPolicy:
    """GkSourcePolicyData as the integral reads it: type, crossover_z, Levin_z, functions."""

    store_id = 5
    quality = "complete"

    def __init__(self, gk, type_: str, crossover_z, numeric_region, WKB_region):
        self.type = type_
        self.crossover_z = crossover_z
        self.Levin_z = None
        self.functions = GkSourceFunctions(
            numeric_region=numeric_region,
            WKB_region=WKB_region,
            numeric_Gk=gk.numeric_Gk if numeric_region is not None else None,
            WKB_Gk=None,
            phase=gk.phase if WKB_region is not None else None,
            sin_amplitude=gk.sin_amplitude if WKB_region is not None else None,
            type=type_,
            quality=self.quality,
            crossover_z=crossover_z,
        )


class OffsetBesselPhaseGk(BesselPhaseGk):
    """
    BesselPhaseGk with the sampled Green's-function phase shifted by a whole number of cycles.
    The unshifted theta_G = vartheta(k eta') - vartheta(k eta_resp) vanishes at z' = z_resp, and
    when the phase grid reaches z_resp a rounding-level positive value there gets
    wrap_theta's div2pi = +1 while every other sample has div2pi <= 0, which phase_spline's
    logarithmic chunking refuses (it requires one sign). Shifting theta by -2 cycles keeps
    sin/cos, theta_deriv and every phase *difference* identical and gives the spline a
    single-signed div2pi range. (Log 07's fixture avoided the issue by stopping its grid 5 %
    above z_resp, which an integral down to z_resp cannot do.)
    """

    OFFSET_CYCLES = -2

    def __init__(self, k, w, z_resp, model, z_grid):
        from LiouvilleGreen.WKBtools import wrap_theta
        from LiouvilleGreen.constants import TWO_PI
        from LiouvilleGreen.phase_spline import phase_spline

        self.k = k
        self.w = w
        self.model = model
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
        self.nu = 0.5 + self.b
        self.z_resp = z_resp
        self.tau_resp = model.tau(z_resp)

        self._bessel = bessel_phase(self.nu, 1.02 * k * self.tau_resp)
        self._m_resp = self._bessel["mod"](k * self.tau_resp)
        self._vartheta_resp = self._bessel["phase"].raw_theta(k * self.tau_resp)

        samples = []
        for z in z_grid:
            theta = (
                self._bessel["phase"].raw_theta(k * model.tau(z))
                - self._vartheta_resp
                + self.OFFSET_CYCLES * TWO_PI
            )
            div, mod = wrap_theta(theta)
            samples.append((log(1.0 + z), div, mod))
        samples.sort(key=lambda s: s[0])
        log_x, div_2pi, mod_2pi = zip(*samples)
        self.phase = phase_spline(
            list(log_x),
            list(div_2pi),
            list(mod_2pi),
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=125,
            increasing=False,
        )
        self.WKB_region = (max(z_grid), min(z_grid))


class ExactTkFunctions(ExactTk):
    """
    ExactTk (exact amplitude-phase decomposition of the analytic T, valid at every z) with the
    region bookkeeping TkSourceFunctions exposes, so that it can be injected through
    evaluate_QuadSource_integral's Tk_functions_builder. Its regions never bind.
    """

    def __init__(self, k, w, model, crossover_z):
        super().__init__(k, w, model)
        self.crossover_z = crossover_z
        self.numeric_region = (inf, crossover_z)
        self.WKB_region = (crossover_z, 0.0)


def exact_Tk_inputs(fixture: Fixture):
    """
    The (Tk_numeric, Tk_WKB) stand-ins prompt 05's Fixture.exact_functions() feeds to
    TkSourceFunctions, captured instead of consumed.
    """
    captured = {}

    def capture(model, k, Tk_numeric, Tk_WKB):
        captured["inputs"] = (Tk_numeric, Tk_WKB)
        return None

    with patch.object(tk_fixtures, "TkSourceFunctions", capture):
        fixture.exact_functions()
    return captured["inputs"]


# =============================================================================================
# the (k, q, r) shapes of prompt 08 section 7
# =============================================================================================


class Shape:
    """
    One (k, q, r) configuration. G_cross_x is the value of x_G = k a0 eta at which the
    Green's-function policy hands over from numeric to Liouville-Green (type "mixed").
    """

    def __init__(self, name, k, q, r, G_cross_x):
        self.name = name
        self.k = k
        self.q = q
        self.r = r
        self.G_cross_x = G_cross_x


SHAPES = [
    # all three hand-overs within ~0.3 e-folds of each other
    Shape("together", k=1.1e4, q=1.0e4, r=1.2e4, G_cross_x=33.0),
    # q ~ r >> k: both T oscillate while G is still numeric, then G hands over too
    Shape("T-first", k=1.0e3, q=1.0e4, r=1.2e4, G_cross_x=4.0),
    # q << k ~ r: T_q stays numeric down to the response redshift
    Shape("q-smooth", k=1.0e4, q=1.0e2, r=1.1e4, G_cross_x=4.0),
]

# x_r = r c_s a0 eta at the response redshift, from "every factor oscillatory over most of the
# range" (the LG fixtures end at x = 1e3) to "hand-overs inside the range" (T hands over at
# x ~ 19, so x_r = 30 puts T_r's hand-over just above z_response)
X_RESP_VALUES = (980.0, 100.0, 30.0)


class Case:
    """
    Everything one integral needs, for one b, one shape and one response redshift, in either
    flavour. `run()` calls evaluate_QuadSource_integral directly.
    """

    _fixtures = {}

    @classmethod
    def fixture(cls, w, k):
        key = (w, k)
        if key not in cls._fixtures:
            cls._fixtures[key] = Fixture(w, k=k)
        return cls._fixtures[key]

    def __init__(
        self,
        b: float,
        shape: Shape,
        x_resp: float,
        exact: bool,
        G_type: str = "mixed",
        z_source_max: float = None,
    ):
        self.b = b
        self.w = w = w_of_b(b)
        self.shape = shape
        self.exact = exact
        self.model = FakeModel(w)
        model = self.model

        self.k = FakeExitTime(shape.k, 1, None)
        self.q = FakeExitTime(shape.q, 2, None)
        self.r = FakeExitTime(shape.r, 3, None)

        self.Fq = self.fixture(w, shape.q)
        self.Fr = self.fixture(w, shape.r)
        self.q.z_exit = self.Fq.z_exit
        self.r.z_exit = self.Fr.z_exit

        # response redshift from x_r; the r fixture's LG region must reach it
        cs = sqrt(w)
        self.z_resp = z_at_tau(model, x_resp / (shape.r * cs))
        assert self.z_resp >= self.Fr.z_WKB[-1] * (1.0 - 1e-12), (
            self.z_resp,
            self.Fr.z_WKB[-1],
        )

        # top of the range: 5 e-folds super-horizon for the larger source wavenumber (the
        # numeric fixtures start there; above it T = 1 exactly)
        if z_source_max is None:
            z_source_max = max(self.Fq.z_numeric[0], self.Fr.z_numeric[0])
        self.z_source_max = z_source_max

        # Green's function hand-over
        self.G_crossover_z = z_at_tau(model, shape.G_cross_x / shape.k)
        self.G_type = G_type

        # transfer-function inputs and the builder that turns them into two-region functions
        self.Tq_inputs = exact_Tk_inputs(self.Fq)
        self.Tr_inputs = exact_Tk_inputs(self.Fr)
        if exact:
            exact_functions = {
                float(shape.q): ExactTkFunctions(
                    shape.q, w, model, self.Fq.crossover_z
                ),
                float(shape.r): ExactTkFunctions(
                    shape.r, w, model, self.Fr.crossover_z
                ),
            }
            self.Tk_builder = lambda model, k, Tn, Tw: exact_functions[float(k)]
        else:
            self.Tk_builder = tk_fixtures.TkSourceFunctions

        # source: the exact f, splined on the production grid over the both-numeric region
        z_floor = max(self.Fq.crossover_z, self.Fr.crossover_z)
        z_grid = log_grid(
            1.0 + z_source_max, 1.0 + z_floor, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        self.source = FakeQuadSource(
            self.q.k,
            self.r.k,
            z_grid,
            self.f_exact,
            self.Fq.crossover_z,
            self.Fr.crossover_z,
            exact=exact,
        )

        # Green's function
        if exact:
            gk = ExactGk(shape.k, w, self.z_resp, model)
        else:
            top = z_source_max if G_type == "WKB" else self.G_crossover_z
            gk = OffsetBesselPhaseGk(
                shape.k,
                w,
                self.z_resp,
                model,
                log_grid(1.0 + top, 1.0 + self.z_resp, PRODUCTION_SAMPLES_PER_LOG10Z),
            )
        self.gk = gk
        whole = (z_source_max, self.z_resp)
        if G_type == "numeric":
            self.GkPolicy = FakeGkPolicy(gk, "numeric", None, whole, None)
        elif G_type == "WKB":
            self.GkPolicy = FakeGkPolicy(gk, "WKB", None, None, whole)
        else:
            self.GkPolicy = FakeGkPolicy(gk, "mixed", self.G_crossover_z, whole, whole)

        self._bessel = None

    # --- exact ingredients -----------------------------------------------------------------

    def f_exact(self, z: float) -> float:
        tau = self.model.tau(z)
        H = self.model.Hubble(z)
        return source_function(
            compute_analytic_T(self.shape.q, self.w, tau),
            compute_analytic_T(self.shape.r, self.w, tau),
            compute_analytic_Tprime(self.shape.q, self.w, tau, H),
            compute_analytic_Tprime(self.shape.r, self.w, tau, H),
            z,
            self.w,
        )["source"]

    def integrand_exact(self, log_z: float) -> float:
        """G f / H^2 at log(1+z'), from scipy's Bessel functions only."""
        z = exp(log_z) - 1.0
        H = self.model.Hubble(z)
        G = compute_analytic_G(
            self.shape.k, self.w, self.model.tau(z), self.model.tau(self.z_resp), H
        )
        return G * self.f_exact(z) / (H * H)

    def bessel_phase_data(self):
        """bessel_phase splines for nu = 1/2 + b and 5/2 + b, as main.py builds them."""
        if self._bessel is None:
            eta_resp = self.model.tau(self.z_resp)
            cs = sqrt(self.w)
            x_max = 1.1 * max(
                self.shape.k * eta_resp,
                self.shape.q * cs * eta_resp,
                self.shape.r * cs * eta_resp,
            )
            self._bessel = (
                bessel_phase(0.5 + self.b, x_max, atol=1e-25, rtol=5e-14),
                bessel_phase(2.5 + self.b, x_max, atol=1e-25, rtol=5e-14),
            )
        return self._bessel

    # --- running the integral ----------------------------------------------------------------

    def partition(self):
        Tq_f = self.Tk_builder(self.model, self.q.k, *self.Tq_inputs)
        Tr_f = self.Tk_builder(self.model, self.r.k, *self.Tr_inputs)
        return build_partition(
            self.GkPolicy, Tq_f, Tr_f, self.source, self.z_resp, self.z_source_max
        )

    def run(self, atol=1e-25, rtol=1e-8):
        B05, B25 = self.bessel_phase_data()
        return evaluate_QuadSource_integral(
            self.model,
            self.k,
            self.q,
            self.r,
            self.source,
            self.GkPolicy,
            FakeZ(self.z_resp),
            FakeZ(self.z_source_max),
            self.b,
            B05,
            B25,
            self.Tq_inputs[0],
            self.Tq_inputs[1],
            self.Tr_inputs[0],
            self.Tr_inputs[1],
            atol=atol,
            rtol=rtol,
            Tk_functions_builder=self.Tk_builder,
        )

    def label(self):
        return f"b={self.b:g} {self.shape.name} x_resp={self.shape.r * sqrt(self.w) * self.model.tau(self.z_resp):.0f} {'exact' if self.exact else 'realistic'}"


def regimes_of(result: dict):
    return [
        tuple(item["regime"])
        for item in result["metadata"]["partition"]["subintervals"]
    ]


def regime_row(regime) -> str:
    """The README section 6 row a regime belongs to."""
    G, q, r = regime
    n = G + q + r
    if n == 0:
        return "none"
    if n == 3:
        return "all three"
    if n == 2:
        return "G and one T" if G else "Tq and Tr, G smooth"
    return "G only" if G else "one T only"


ALL_ROWS = {"G only", "one T only", "G and one T", "Tq and Tr, G smooth", "all three"}


# =============================================================================================
# acceptance oracle 1: total vs analytic_rad (spec 04 R14 through the code's own oracle)
# =============================================================================================


class TestAnalyticOracle(unittest.TestCase):
    """
    total vs analytic_rad for b in {0, 0.2}, the three (k, q, r) shapes and three response
    redshifts per shape.

    Exact flavour: every ingredient is exact to rounding, so the residual is the integrator's
    plus analytic_rad's own (audit QI-1: 1e-8 to 1e-6). Threshold 1e-5 as the prompt asks.

    Realistic flavour: the residual is the representation floors recorded on the board --
    4.5e-4 of the local envelope for the QuadSource spline of f at the hand-over
    ([06-source-spline-residual-vs-handover]), 7e-6 (b=0) to 1.4e-4 (b=0.25) for the LG closed
    forms just below it ([07-lg-derivative-truncation-at-handover]), 6e-6 for the re-splined
    phase -- and, because numeric_quad and WKB_Levin cancel (by up to 5x in these cases), the
    error of `total` is that floor times |numeric_quad / total|. So the realistic assertions
    are on the two parts separately, each against the exact-ingredient run of the same case and
    normalised by scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|): the smooth part to
    1e-3 (prompt 06's 4.5e-4 with 2x headroom), the Levin part to 5e-4 (log 07's number), and
    total - analytic_rad to 1.5e-3 of the same scale. This is NOT the prompt's 1e-5; see
    logs/08-qsi-phase-group-integration.md deviation 1 for the arithmetic.
    """

    EXACT_THRESHOLD = 1.0e-5
    REALISTIC_SMOOTH_THRESHOLD = 1.0e-3
    REALISTIC_LEVIN_THRESHOLD = 5.0e-4
    REALISTIC_TOTAL_THRESHOLD = 1.5e-3

    seen_regimes = set()

    def test_exact_ingredients(self):
        rows = []
        worst = 0.0
        for b in B_VALUES:
            for shape in SHAPES:
                for x_resp in X_RESP_VALUES:
                    case = Case(b, shape, x_resp, exact=True)
                    out = case.run()
                    rel = abs(out["total"] - out["analytic_rad"]) / abs(
                        out["analytic_rad"]
                    )
                    worst = max(worst, rel)
                    regimes = regimes_of(out)
                    type(self).seen_regimes.update(regime_row(r) for r in regimes)
                    rows.append(
                        f"  {case.label():<36s} total={out['total']:+.10e} analytic={out['analytic_rad']:+.10e} rel={rel:.3e} "
                        f"regimes={[''.join(n for n, f in zip('Gqr', r) if f) or '-' for r in regimes]} "
                        f"Levin abserr/|total|={out['metadata']['WKB_Levin']['abserr'] / abs(out['total']):.1e} "
                        f"converged={out['metadata']['WKB_Levin']['converged']} "
                        f"solves={out['WKB_Levin_data'].evaluations if out['WKB_Levin_data'] else 0}"
                    )
                    with self.subTest(case=case.label()):
                        self.assertLess(rel, self.EXACT_THRESHOLD)
        print(f"\n[oracle 1, exact] total vs analytic_rad, worst {worst:.3e}:")
        print("\n".join(rows))

    def test_realistic_fixtures(self):
        rows = []
        worst = {"total": 0.0, "smooth": 0.0, "Levin": 0.0, "total/analytic": 0.0}
        for b in B_VALUES:
            for shape in SHAPES:
                for x_resp in X_RESP_VALUES:
                    exact = Case(b, shape, x_resp, exact=True).run()
                    case = Case(b, shape, x_resp, exact=False)
                    out = case.run()
                    A = out["analytic_rad"]
                    scale = max(
                        abs(exact["numeric_quad"]), abs(exact["WKB_Levin"]), abs(A)
                    )
                    smooth = abs(out["numeric_quad"] - exact["numeric_quad"]) / scale
                    Levin = abs(out["WKB_Levin"] - exact["WKB_Levin"]) / scale
                    total = abs(out["total"] - A) / scale
                    rel = abs(out["total"] - A) / abs(A)
                    worst["smooth"] = max(worst["smooth"], smooth)
                    worst["Levin"] = max(worst["Levin"], Levin)
                    worst["total"] = max(worst["total"], total)
                    worst["total/analytic"] = max(worst["total/analytic"], rel)
                    rows.append(
                        f"  {case.label():<40s} rel(total, analytic)={rel:.3e} |numeric_quad/total|={abs(exact['numeric_quad'] / A):.1f} "
                        f"/scale: smooth-part dev={smooth:.3e} Levin-part dev={Levin:.3e} total dev={total:.3e} "
                        f"solves={out['WKB_Levin_data'].evaluations if out['WKB_Levin_data'] else 0} t={out['compute_time']:.2f}s"
                    )
                    with self.subTest(case=case.label()):
                        self.assertLess(smooth, self.REALISTIC_SMOOTH_THRESHOLD)
                        self.assertLess(Levin, self.REALISTIC_LEVIN_THRESHOLD)
                        self.assertLess(total, self.REALISTIC_TOTAL_THRESHOLD)
        print(
            f"\n[oracle 1, realistic] worst: total vs analytic_rad {worst['total/analytic']:.3e} relative; "
            f"normalised by scale: smooth part {worst['smooth']:.3e}, Levin part {worst['Levin']:.3e}, total {worst['total']:.3e}"
        )
        print("\n".join(rows))


# =============================================================================================
# acceptance oracle 2: total vs scipy.quad of the exact integrand on a short range
# =============================================================================================


class TestDirectQuadOracle(unittest.TestCase):
    """
    On a short range straddling every hand-over (x_r from ~4 to 60, so quad converges), total
    with exact ingredients must agree with (1 + z_resp) * quad(G_exact f_exact / H^2) in
    log(1+z') to 1e-8. This isolates the partition and the Levin integration from the analytic
    branch's own bessel_phase splines.
    """

    THRESHOLD = 1.0e-8

    def test_short_range(self):
        for b in B_VALUES:
            for shape in SHAPES:
                case = Case(b, shape, 60.0, exact=True)
                cs = sqrt(case.w)
                z_top = z_at_tau(case.model, 4.0 / (shape.r * cs))
                case = Case(b, shape, 60.0, exact=True, z_source_max=z_top)
                out = case.run(rtol=1e-10)
                ref, err = quad(
                    case.integrand_exact,
                    log(1.0 + case.z_resp),
                    log(1.0 + case.z_source_max),
                    limit=2000,
                    epsabs=0.0,
                    epsrel=1e-12,
                )
                ref *= 1.0 + case.z_resp
                rel = abs(out["total"] - ref) / abs(ref)
                regimes = regimes_of(out)
                print(
                    f"\n[oracle 2] {case.label()}: total={out['total']:+.12e} quad={ref:+.12e} rel={rel:.3e} "
                    f"(quad err {err * (1 + case.z_resp) / abs(ref):.1e}) regimes={[''.join(n for n, f in zip('Gqr', r) if f) or '-' for r in regimes]}"
                )
                with self.subTest(case=case.label()):
                    self.assertLess(rel, self.THRESHOLD)
                    # every shape straddles at least one hand-over on this range
                    self.assertGreaterEqual(len(regimes), 2)


# =============================================================================================
# regime coverage, partition structure, GkSourcePolicyData types
# =============================================================================================


class TestPartition(unittest.TestCase):
    def test_every_row_of_the_phase_group_table_is_reached(self):
        """
        Across the acceptance cases every README section 6 row with n >= 1 occurs, including
        "T_q and T_r oscillatory, G smooth". Checked from build_partition (no integration).
        """
        seen = set()
        for b in B_VALUES:
            for shape in SHAPES:
                for x_resp in X_RESP_VALUES:
                    partition = Case(b, shape, x_resp, exact=True).partition()
                    for item in partition["metadata"]["subintervals"]:
                        seen.add(regime_row(tuple(item["regime"])))
        print(f"\n[partition] rows reached: {sorted(seen)}")
        self.assertTrue(ALL_ROWS <= seen, ALL_ROWS - seen)

    def test_subintervals_are_contiguous_descending_and_monotone_in_regime(self):
        for shape in SHAPES:
            case = Case(0.0, shape, 100.0, exact=True)
            partition = case.partition()
            items = partition["metadata"]["subintervals"]
            self.assertAlmostEqual(items[0]["z_max"], case.z_source_max, delta=1e-9)
            self.assertAlmostEqual(items[-1]["z_min"], case.z_resp, delta=1e-9)
            for a, b_ in zip(items[:-1], items[1:]):
                self.assertAlmostEqual(a["z_min"], b_["z_max"], delta=1e-9)
                # once a factor is oscillatory it stays oscillatory as z' descends
                for fa, fb in zip(a["regime"], b_["regime"]):
                    self.assertTrue(fb or not fa)
            self.assertEqual(items[0]["regime"], [False, False, False])
            self.assertEqual(items[0]["method"], "quad")
            self.assertTrue(all(i["method"] == "Levin" for i in items[1:]))
            breakpoints = {bp["factor"] for bp in partition["metadata"]["breakpoints"]}
            self.assertEqual(breakpoints, {"G", "Tq", "Tr"})

    def test_GkSourcePolicyData_types(self):
        """
        type "numeric": G smooth everywhere, so the T-driven rows are the only ones; type "WKB":
        G oscillatory everywhere, so the first sub-interval is already a Levin one.
        """
        shape = SHAPES[1]
        numeric = Case(0.0, shape, 100.0, exact=True, G_type="numeric").partition()
        self.assertTrue(
            all(not item["regime"][0] for item in numeric["metadata"]["subintervals"])
        )
        self.assertEqual(
            [tuple(i["regime"]) for i in numeric["metadata"]["subintervals"]],
            [(False, False, False), (False, False, True), (False, True, True)],
        )
        WKB = Case(0.0, shape, 100.0, exact=True, G_type="WKB").partition()
        self.assertTrue(
            all(item["regime"][0] for item in WKB["metadata"]["subintervals"])
        )
        self.assertEqual(WKB["metadata"]["subintervals"][0]["method"], "Levin")
        # and both integrate against the oracle
        for G_type in ("numeric", "WKB"):
            case = Case(0.0, shape, 100.0, exact=True, G_type=G_type)
            out = case.run()
            rel = abs(out["total"] - out["analytic_rad"]) / abs(out["analytic_rad"])
            print(f"\n[G type {G_type}] {case.label()}: rel vs analytic_rad {rel:.3e}")
            self.assertLess(rel, TestAnalyticOracle.EXACT_THRESHOLD)
            if G_type == "numeric":
                self.assertEqual(out["metadata"]["partition"]["G_type"], "numeric")

    def test_equal_wavenumbers(self):
        """q = r: the two T breakpoints coincide and are merged; the q-r group has zero phase."""
        shape = Shape("q=r", k=1.1e4, q=1.0e4, r=1.0e4, G_cross_x=33.0)
        case = Case(0.0, shape, 100.0, exact=True)
        out = case.run()
        regimes = regimes_of(out)
        self.assertEqual(
            regimes, [(False, False, False), (True, False, False), (True, True, True)]
        )
        rel = abs(out["total"] - out["analytic_rad"]) / abs(out["analytic_rad"])
        print(f"\n[q = r] rel vs analytic_rad {rel:.3e}")
        self.assertLess(rel, TestAnalyticOracle.EXACT_THRESHOLD)

    def test_outputs_are_populated_as_specified(self):
        case = Case(0.0, SHAPES[0], 100.0, exact=True)
        out = case.run()
        self.assertEqual(out["WKB_quad"], 0.0)
        self.assertIsNone(out["WKB_quad_data"])
        self.assertAlmostEqual(
            out["total"], out["numeric_quad"] + out["WKB_Levin"], delta=0.0
        )
        self.assertIsNotNone(out["numeric_quad_data"])
        Levin = out["WKB_Levin_data"]
        subs = out["metadata"]["WKB_Levin"]["subintervals"]
        self.assertEqual(
            Levin.evaluations, sum(g["evaluations"] for s in subs for g in s["groups"])
        )
        self.assertEqual(
            Levin.num_regions, sum(g["regions"] for s in subs for g in s["groups"])
        )
        self.assertAlmostEqual(
            out["WKB_Levin"],
            sum(s["value"] for s in subs),
            delta=1e-12 * abs(out["WKB_Levin"]),
        )
        self.assertAlmostEqual(
            out["metadata"]["WKB_Levin"]["abserr"],
            sum(g["abserr"] for s in subs for g in s["groups"]),
            delta=1e-12 * out["metadata"]["WKB_Levin"]["abserr"],
        )
        self.assertEqual(
            out["metadata"]["WKB_Levin"]["theta_deriv_supplied"], LEVIN_USE_THETA_DERIV
        )
        self.assertIn("partition", out["metadata"])
        self.assertIn("abserr", out["metadata"]["numeric_quad"])
        self.assertIsNone(out["metadata"]["partition"]["Levin_z_unused"])


# =============================================================================================
# seams
# =============================================================================================


class TestSeams(unittest.TestCase):
    def test_artificial_split_of_a_subinterval(self):
        """
        Splitting a Levin sub-interval at an interior point: the two halves sum to the unsplit
        value to 1e-10 (exact ingredients, all-oscillatory regime).
        """
        for b in B_VALUES:
            case = Case(b, SHAPES[0], 980.0, exact=True)
            Tq_f = case.Tk_builder(case.model, case.q.k, *case.Tq_inputs)
            Tr_f = case.Tk_builder(case.model, case.r.k, *case.Tr_inputs)
            cs = sqrt(case.w)
            z_hi = z_at_tau(case.model, 40.0 / (case.shape.r * cs))
            z_lo = case.z_resp
            z_mid = exp(0.5 * (log(1.0 + z_hi) + log(1.0 + z_lo))) - 1.0

            def piece(a, c):
                return phase_group_Levin_integral(
                    case.model,
                    case.k.k,
                    case.q.k,
                    case.r.k,
                    (True, True, True),
                    case.GkPolicy.functions,
                    _ClampedTk(Tq_f, True),
                    _ClampedTk(Tr_f, True),
                    FakeZ(case.z_resp),
                    max_z=a,
                    min_z=c,
                    atol=1e-25,
                    rtol=1e-10,
                )["value"]

            whole = piece(z_hi, z_lo)
            split = piece(z_hi, z_mid) + piece(z_mid, z_lo)
            rel = abs(whole - split) / abs(whole)
            print(
                f"\n[split] b={b:g}: whole={whole:+.12e} split={split:+.12e} rel={rel:.3e}"
            )
            with self.subTest(b=b):
                self.assertLess(rel, 1.0e-10)

    def test_integrand_continuity_at_the_hand_over(self):
        """
        Just above T_r's hand-over the integrand is G f_spline / H^2 with f from the QuadSource
        spline; just below it is the phase-group sum with T_r in its LG form. On the realistic
        fixtures the two agree to the seam accuracy log 07 measured (~1e-3 of the envelope at
        the hand-over, threshold 2e-3); on exact ingredients to rounding.
        """
        for exact in (True, False):
            for b in B_VALUES:
                # shape "T-first": G hands over below both T, so the regime just below T_r's
                # hand-over is (G smooth, T_q smooth, T_r oscillatory) for both b
                case = Case(b, SHAPES[1], 100.0, exact=exact)
                Tq_f = case.Tk_builder(case.model, case.q.k, *case.Tq_inputs)
                Tr_f = case.Tk_builder(case.model, case.r.k, *case.Tr_inputs)
                z_cross = Tr_f.crossover_z
                # the region just below: G smooth (G hands over lower in this shape), T_q smooth
                partition = case.partition()["metadata"]["subintervals"]
                below = next(
                    item
                    for item in partition
                    if abs(item["z_max"] - z_cross) < 1e-9 * (1.0 + z_cross)
                )
                self.assertEqual(below["regime"], [False, False, True])
                groups = build_phase_groups(
                    (False, False, True),
                    Gk=case.GkPolicy.functions.numeric_Gk,
                    Tq=_ClampedTk(Tq_f, False),
                    Tr=_ClampedTk(Tr_f, True),
                    model_functions=case.model.functions,
                    w_background=case.model.functions.wBackground,
                )
                # both representations are evaluated AT the hand-over (the spline's last node and
                # the LG region's first sample), so the difference is the seam alone
                log_c = log(1.0 + z_cross)
                H_c = case.model.Hubble(z_cross)
                smooth = (
                    case.GkPolicy.functions.numeric_Gk(log_c, z_is_log=True)
                    * case.source.functions.source(log_c, z_is_log=True)
                    / (H_c * H_c)
                )
                osc = evaluate_sum(groups, log_c)
                scale = max(abs(smooth), evaluate_envelope(groups, log_c))
                rel = abs(smooth - osc) / scale
                print(
                    f"\n[hand-over seam, {'exact' if exact else 'realistic'}, b={b:g}] smooth side {smooth:+.6e}, LG side {osc:+.6e}, |diff|/envelope {rel:.3e}"
                )
                with self.subTest(exact=exact, b=b):
                    self.assertLess(rel, 1.0e-9 if exact else 2.0e-3)


# =============================================================================================
# hand-over gap (production grid shape) and its clamp
# =============================================================================================


class TestHandOverClamp(unittest.TestCase):
    """
    In production the first WKB sample sits up to one grid step below the hand-over
    (Fixture(drop_first_WKB_sample=True) reproduces this), so TkSourceFunctions.WKB_region[0] <
    crossover_z and no accessor is evaluable in between. The integral clamps the LG accessors
    to WKB_region[0] across that gap (board section 5 note 6) and records the gap; the effect on
    the total is measured here against the same case with the gap closed: 5.1e-3 for a
    one-step gap, the largest single error term in the chain (board issue
    [08-handover-clamp-error]).
    """

    def test_gap_is_recorded_and_bridged(self):
        b = 0.0
        w = w_of_b(b)
        shape = SHAPES[0]
        reference = Case(b, shape, 100.0, exact=False)
        gapped = Case(b, shape, 100.0, exact=False)
        gapped.Tr_inputs = exact_Tk_inputs(
            Fixture(w, k=shape.r, drop_first_WKB_sample=True)
        )
        out_ref = reference.run()
        out_gap = gapped.run()
        gaps = [
            item["clamp_gaps_log1pz"]
            for item in out_gap["metadata"]["partition"]["subintervals"]
        ]
        Tr_gaps = [g["Tr"] for g in gaps if g.get("Tr", 0.0) > 0.0]
        self.assertEqual(len(Tr_gaps), 1)
        step = (
            out_gap["metadata"]["partition"]["max_clamp_gap_log1pz"]
            / HANDOVER_CLAMP_MAX_GRID_STEPS
        )
        self.assertLess(Tr_gaps[0], 1.05 * step)
        rel = abs(out_gap["total"] - out_ref["total"]) / abs(out_ref["total"])
        rel_oracle = abs(out_gap["total"] - out_gap["analytic_rad"]) / abs(
            out_gap["analytic_rad"]
        )
        print(
            f"\n[hand-over clamp] gap {Tr_gaps[0]:.3e} in log(1+z) ({Tr_gaps[0] / step:.2f} grid steps); "
            f"total with gap vs without: rel {rel:.3e}; with gap vs analytic_rad: rel {rel_oracle:.3e}"
        )
        # measured 5.1e-3 for a one-step gap (board issue [08-handover-clamp-error]); the
        # assertion only pins the order of magnitude
        self.assertLess(rel, 1.0e-2)

    def test_gap_beyond_tolerance_raises(self):
        b = 0.0
        shape = SHAPES[0]
        case = Case(b, shape, 100.0, exact=False)
        Tk_numeric, Tk_WKB = case.Tr_inputs
        # drop five WKB samples: the gap is then ~5 grid steps, beyond the clamp tolerance
        Tk_WKB.values = Tk_WKB.values[5:]
        with self.assertRaises(RuntimeError) as ctx:
            case.run()
        self.assertIn("Liouville-Green representation of Tr", str(ctx.exception))

    def test_response_below_the_LG_samples_raises(self):
        shape = SHAPES[0]
        case = Case(0.0, shape, 100.0, exact=False)
        Tk_numeric, Tk_WKB = case.Tr_inputs
        # keep only the first 60 LG samples (x_r up to ~76, above the response at x_r = 100)
        Tk_WKB.values = Tk_WKB.values[:60]
        with self.assertRaises(RuntimeError) as ctx:
            case.run()
        self.assertIn(
            "lies below the lowest Liouville-Green sample", str(ctx.exception)
        )


# =============================================================================================
# old-regime regression
# =============================================================================================


def load_pre_commit_module():
    """
    The pre-commit ComputeTargets/QuadSourceIntegral.py (PRE_COMMIT_SHA), loaded as a module
    from `git show`. Returns None if git or the object is unavailable.
    """
    try:
        text = subprocess.run(
            ["git", "show", f"{PRE_COMMIT_SHA}:ComputeTargets/QuadSourceIntegral.py"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    with tempfile.NamedTemporaryFile("w", suffix="_old_qsi.py", delete=False) as fh:
        fh.write(text)
        path = fh.name
    spec = importlib.util.spec_from_file_location("old_QuadSourceIntegral", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOldRegimeRegression(unittest.TestCase):
    """
    With both T super-horizon throughout (T = 1, dT/dz = 0 exactly, f = alpha), the pre-commit
    numeric_quad_integral + WKB_Levin_integral (the "G only" Levin row, the only one the old code
    implemented) and the new total agree to 1e-10. The old functions are run from the pre-commit
    file at PRE_COMMIT_SHA.
    """

    def test_G_only_case_reproduces_the_pre_commit_total(self):
        old = load_pre_commit_module()
        if old is None:
            self.skipTest("pre-commit QuadSourceIntegral.py not available from git")

        b = 0.0
        w = w_of_b(b)
        model = FakeModel(w)
        # q = r = 1e4 five to six e-folds super-horizon; k = 1e8 so that G turns over ~7 cycles
        Fq = Case.fixture(w, 1.0e4)
        z_top = exp(log(1.0 + Fq.z_numeric[0]) + 1.0) - 1.0
        z_resp = Fq.z_numeric[0] * 1.0001
        k_G = 1.0e8
        shape = Shape("regression", k=k_G, q=1.0e4, r=1.0e4, G_cross_x=1.0)
        # TkSourceFunctions (exact=False) return T = 1, dT/dz = 0 exactly above their grids, so
        # that f = alpha exactly on both sides of the comparison; G is exact
        case = Case(b, shape, 980.0, exact=False, z_source_max=z_top)
        # override the response redshift and the Green's function accordingly
        case.z_resp = z_resp
        case.G_crossover_z = z_at_tau(model, 40.0 / k_G)
        gk = ExactGk(k_G, w, z_resp, model)
        whole = (z_top, z_resp)
        case.GkPolicy = FakeGkPolicy(gk, "mixed", case.G_crossover_z, whole, whole)
        alpha = source_function(1.0, 1.0, 0.0, 0.0, 0.0, w)["source"]

        class ConstantSource(FakeQuadSource):
            def __init__(self):
                self.q = case.q.k
                self.r = case.r.k
                self.z_sample = FakeZSample(log_grid(1.0 + z_top, 1.0 + z_resp, 100))
                self.numeric_region = (z_top, z_resp)
                self.crossover_z_q = None
                self.crossover_z_r = None
                self.functions = QuadSourceFunctions(
                    source=lambda x, z_is_log=False: alpha,
                    numeric_region=self.numeric_region,
                )

        case.source = ConstantSource()
        # the T fixtures' crossovers lie far below z_resp, so both T are smooth (T = 1) throughout
        new = case.run(rtol=1e-10)
        self.assertEqual(regimes_of(new), [(False, False, False), (True, False, False)])

        z_cross = case.G_crossover_z
        old_numeric = old.numeric_quad_integral(
            model,
            case.k.k,
            case.q.k,
            case.r.k,
            case.source,
            case.GkPolicy,
            FakeZ(z_resp),
            max_z=z_top,
            min_z=z_cross,
            atol=1e-25,
            rtol=1e-10,
        )
        old_Levin = old.WKB_Levin_integral(
            model,
            case.k.k,
            case.q.k,
            case.r.k,
            case.source,
            case.GkPolicy,
            FakeZ(z_resp),
            max_z=z_cross,
            min_z=z_resp,
            atol=1e-25,
            rtol=1e-10,
        )
        old_total = old_numeric["value"] + old_Levin["value"]
        rel = abs(new["total"] - old_total) / abs(old_total)
        theta_span = abs(
            gk.phase.raw_theta(log(1.0 + z_cross), x_is_log=True)
            - gk.phase.raw_theta(log(1.0 + z_resp), x_is_log=True)
        )
        print(
            f"\n[old-regime regression] theta_G span {theta_span / (2 * np.pi):.1f} cycles; "
            f"old total {old_total:+.12e} (numeric {old_numeric['value']:+.6e}, Levin {old_Levin['value']:+.6e}), "
            f"new total {new['total']:+.12e} (numeric {new['numeric_quad']:+.6e}, Levin {new['WKB_Levin']:+.6e}); rel {rel:.3e}"
        )
        self.assertLess(rel, 1.0e-10)
        self.assertLess(
            abs(new["numeric_quad"] - old_numeric["value"]) / abs(old_numeric["value"]),
            1.0e-12,
        )


# =============================================================================================
# compute() payload check
# =============================================================================================


class TestComputePayload(unittest.TestCase):
    def test_missing_transfer_function_keys_raise_RuntimeError_naming_them(self):
        class Tol:
            tol = 1e-8

        class Policy:
            store_id = 9

        qsi = QuadSourceIntegral(
            None,
            model=None,
            policy=Policy(),
            z_response=FakeZ(10.0),
            z_source_max=FakeZ(1e5),
            k=FakeExitTime(1.0, 1, None),
            q=FakeExitTime(1.0, 2, None),
            r=FakeExitTime(1.0, 3, None),
            atol=Tol(),
            rtol=Tol(),
        )
        payload = {
            "source": None,
            "GkPolicy": None,
            "b": 0.0,
            "Bessel_0pt5": None,
            "Bessel_2pt5": None,
        }
        with self.assertRaises(RuntimeError) as ctx:
            qsi.compute(payload)
        message = str(ctx.exception)
        for key in ("Tq_numeric", "Tq_WKB", "Tr_numeric", "Tr_WKB"):
            self.assertIn(key, message)
        self.assertNotIsInstance(ctx.exception, KeyError)


# =============================================================================================
# prompt 09: error bound, b provenance, tolerance plumbing, region guards, schema
# =============================================================================================


class TestErrorBound(unittest.TestCase):
    """
    B8/QI-11. `total_abserr` is the linear sum of every sub-interval's absolute error estimate
    (scipy's on the all-smooth sub-intervals, the Levin driver's on the rest), already scaled by
    (1 + z_response). It is a bound on the *quadrature* error and on nothing else: the error of
    the ingredients -- the QuadSource spline of f, the Liouville-Green closed forms, the
    re-splined phases, the hand-over clamp -- is not visible to either integrator, and neither
    is analytic_rad's own phase/modulus-spline floor. So `total_abserr` does not bound
    |total - analytic_rad|; the ratio is reported here and is ~1e0-1e4 (see the log). What is
    asserted instead is what the number claims: re-running the same integral at a much tighter
    rtol moves `total` by less than the two runs' bounds allow.
    """

    QUADRATURE_BOUND_FRACTION = 1.0e-5

    def test_abserr_is_positive_sums_the_parts_and_is_small(self):
        rows = []
        worst_ratio = {"exact": 0.0, "realistic": 0.0}
        worst_fraction = 0.0
        not_converged = []
        for exact in (True, False):
            for b in B_VALUES:
                for shape in SHAPES:
                    for x_resp in X_RESP_VALUES:
                        case = Case(b, shape, x_resp, exact=exact)
                        out = case.run()
                        err = out["total_abserr"]
                        parts = (
                            out["metadata"]["numeric_quad"]["abserr"]
                            + out["metadata"]["WKB_Levin"]["abserr"]
                        )
                        dev = abs(out["total"] - out["analytic_rad"])
                        ratio = dev / err
                        fraction = err / abs(out["total"])
                        key = "exact" if exact else "realistic"
                        worst_ratio[key] = max(worst_ratio[key], ratio)
                        worst_fraction = max(worst_fraction, fraction)
                        if not out["total_converged"]:
                            not_converged.append(case.label())
                        rows.append(
                            f"  {case.label():<40s} abserr={err:.3e} (quad {out['metadata']['numeric_quad']['abserr']:.2e} "
                            f"+ Levin {out['metadata']['WKB_Levin']['abserr']:.2e}) = {fraction:.2e} of |total|; "
                            f"|total-analytic|/abserr={ratio:.3e} converged={out['total_converged']} "
                            f"phase_limited={out['total_phase_limited']}"
                        )
                        with self.subTest(case=case.label()):
                            self.assertGreater(err, 0.0)
                            self.assertAlmostEqual(err, parts, delta=1e-15 * parts)
                            # the quadrature bound is a small fraction of the value: if it were
                            # not, the integrator would be telling us it had failed
                            self.assertLess(fraction, self.QUADRATURE_BOUND_FRACTION)
                            self.assertIsInstance(out["total_converged"], bool)
                            self.assertIsInstance(out["total_phase_limited"], bool)
                            self.assertEqual(out["b"], b)
        print(
            f"\n[error bound] total_abserr: worst |total-analytic|/abserr {worst_ratio['exact']:.3e} (exact), "
            f"{worst_ratio['realistic']:.3e} (realistic); worst abserr/|total| {worst_fraction:.3e}; "
            f"{len(not_converged)} of 36 cases report total_converged=False:"
        )
        print("\n".join(rows))

    def test_abserr_bounds_the_quadrature_error(self):
        """
        The meaning of the bound: |total(rtol=1e-8) - total(rtol=1e-11)| must be within the sum
        of the two runs' reported bounds. Exact ingredients, so that only the integrators move.
        """
        rows = []
        worst = 0.0
        for b in B_VALUES:
            for shape in SHAPES:
                for x_resp in X_RESP_VALUES:
                    case = Case(b, shape, x_resp, exact=True)
                    loose = case.run(rtol=1e-8)
                    tight = case.run(rtol=1e-11)
                    dev = abs(loose["total"] - tight["total"])
                    bound = loose["total_abserr"] + tight["total_abserr"]
                    worst = max(worst, dev / bound)
                    rows.append(
                        f"  {case.label():<36s} |Delta total|={dev:.3e} bound={bound:.3e} ratio={dev / bound:.3e}"
                    )
                    with self.subTest(case=case.label()):
                        self.assertLessEqual(dev, bound)
        print(f"\n[error bound] rtol 1e-8 vs 1e-11, worst ratio {worst:.3e}:")
        print("\n".join(rows))


class TestTolerancePlumbing(unittest.TestCase):
    """B5/QI-8 and B6/QI-9: every tolerance the analytic branch uses is the caller's."""

    ATOL = 3.25e-23
    RTOL = 7.5e-9

    def _call(self, case):
        B05, B25 = case.bessel_phase_data()
        return analytic_integral(
            case.model,
            case.k.k,
            case.q.k,
            case.r.k,
            FakeZ(case.z_resp),
            max_z=FakeZ(case.z_source_max),
            min_z=FakeZ(case.z_resp),
            b=case.b,
            Bessel_0pt5=B05,
            Bessel_2pt5=B25,
            rtol=self.RTOL,
            atol=self.ATOL,
        )

    def test_analytic_integral_forwards_its_tolerances(self):
        case = Case(0.0, SHAPES[0], 100.0, exact=True)
        seen = []
        real = qsi_module._three_bessel_integrals

        def spy(*args, **kwargs):
            seen.append((kwargs["nu_type"], kwargs["atol"], kwargs["rtol"]))
            return real(*args, **kwargs)

        with patch.object(qsi_module, "_three_bessel_integrals", spy):
            self._call(case)

        print(f"\n[tolerances] _three_bessel_integrals saw {seen}")
        self.assertEqual(len(seen), 2)
        self.assertEqual({nu for nu, _, _ in seen}, {"0pt5", "2pt5"})
        for nu_type, atol, rtol in seen:
            with self.subTest(nu_type=nu_type):
                self.assertEqual(atol, self.ATOL)
                self.assertEqual(rtol, self.RTOL)

    def test_every_analytic_Levin_call_uses_the_same_tolerances(self):
        """
        The Y3 call used to pass the module constants LEVIN_ABSERR / LEVIN_RELERR while its
        seven siblings passed the caller's, making the four cancelling phase groups' error bars
        non-uniform. All eight now agree, and the constants are gone.
        """
        self.assertFalse(hasattr(qsi_module, "LEVIN_ABSERR"))
        self.assertFalse(hasattr(qsi_module, "LEVIN_RELERR"))

        case = Case(0.0, SHAPES[0], 100.0, exact=True)
        seen = []
        real = qsi_module.adaptive_levin_sincos

        def spy(*args, **kwargs):
            seen.append((kwargs["notify_label"], kwargs["atol"], kwargs["rtol"]))
            return real(*args, **kwargs)

        with patch.object(qsi_module, "adaptive_levin_sincos", spy):
            self._call(case)

        labels = sorted(label for label, _, _ in seen)
        print(f"\n[tolerances] analytic Levin calls: {labels}")
        # both three-Bessel calls take the Levin branch on this case, so all eight groups of
        # each are exercised
        self.assertEqual(len(seen), 16)
        self.assertTrue(any(label.endswith("Y3") for label, _, _ in seen))
        self.assertEqual(
            {(atol, rtol) for _, atol, rtol in seen}, {(self.ATOL, self.RTOL)}
        )


class TestBesselOrderGuard(unittest.TestCase):
    """
    B7/QI-1: the Levin branch of the analytic oracle uses the caller's bessel_phase splines
    while _three_bessel_quad recomputes jv(nu + b, .); they agree only if both were built at the
    same b. bessel_phase() does not record its order, so the guard checks it numerically.
    """

    def test_splines_built_at_the_wrong_b_are_rejected(self):
        # b = 0 splines, claimed as b = 0.2 (the run itself is a b = 0.2 configuration)
        case = Case(0.2, SHAPES[0], 100.0, exact=True)
        wrong = Case(0.0, SHAPES[0], 100.0, exact=True).bessel_phase_data()
        case._bessel = wrong
        with self.assertRaises(RuntimeError) as ctx:
            case.run()
        message = str(ctx.exception)
        print(f"\n[b guard] {message[:220]}")
        self.assertIn("Bessel", message)
        self.assertIn("order", message)

    def test_correctly_built_splines_are_accepted(self):
        # the same configuration with its own splines runs (every other test relies on this)
        case = Case(0.2, SHAPES[0], 100.0, exact=True)
        out = case.run()
        self.assertEqual(out["b"], 0.2)
        self.assertGreater(BESSEL_ORDER_CHECK_TOL, 0.0)


# --- lightweight stand-ins for the region guards (no bessel_phase, no integration) -----------


class _MinimalGk:
    """Only numeric_Gk is read for a type "numeric" policy."""

    @staticmethod
    def numeric_Gk(x, z_is_log=False):
        return 1.0


class _MinimalTk:
    """
    The region bookkeeping build_partition reads; no accessor is called. `crossover_z` is a
    float in production (TkSourceFunctions.crossover_z = TkWKBIntegration.z_init), so a factor
    that is smooth over the whole range is given a hand-over at the bottom of it.
    """

    def __init__(self, crossover_z: float):
        self.crossover_z = crossover_z
        self.numeric_region = (inf, crossover_z)
        self.WKB_region = (crossover_z, 0.0)
        self.phase = None


class _MinimalSource:
    store_id = 77

    def __init__(self, z_max, z_min):
        self.z_sample = FakeZSample(log_grid(1.0 + z_max, 1.0 + z_min, 100))
        self.numeric_region = (z_max, z_min)
        self.crossover_z_q = None
        self.crossover_z_r = None


class TestRegionGuards(unittest.TestCase):
    """
    B11/QI-4. The region-nonempty test is a width in the integration variable log(1+z'),
    MIN_SUBINTERVAL_LOG_WIDTH, not a ratio in z: it means the same thing everywhere in the range
    and it does not divide by zero at z_response = 0. A hand-over that falls inside that width is
    merged into the sub-interval above it and recorded in metadata["partition"]["skipped"].
    """

    def _partition(self, z_top, z_bottom, crossover_q, crossover_r):
        policy = FakeGkPolicy(_MinimalGk(), "numeric", None, (z_top, z_bottom), None)
        return build_partition(
            policy,
            _MinimalTk(crossover_q),
            _MinimalTk(crossover_r),
            _MinimalSource(z_top, z_bottom),
            z_bottom,
            z_top,
        )

    def test_narrow_subinterval_is_skipped_and_recorded(self):
        z_top, z_bottom = 1.0e5, 10.0
        z_q = 1.0e3
        # a hand-over 0.4 * MIN_SUBINTERVAL_LOG_WIDTH below the one above it: too narrow to
        # integrate, so it is merged and recorded rather than dropped
        z_r = exp(log(1.0 + z_q) - 0.4 * MIN_SUBINTERVAL_LOG_WIDTH) - 1.0
        partition = self._partition(z_top, z_bottom, z_q, z_r)
        metadata = partition["metadata"]
        skipped = metadata["skipped"]
        print(
            f"\n[B11] hand-overs z={z_q:.10g} and z={z_r:.10g} "
            f"(separation {log(1.0 + z_q) - log(1.0 + z_r):.3e} in log(1+z), "
            f"MIN_SUBINTERVAL_LOG_WIDTH={MIN_SUBINTERVAL_LOG_WIDTH:.1e}): "
            f"{len(metadata['subintervals'])} sub-intervals, skipped={skipped}"
        )
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["factor"], "Tr")
        self.assertLess(skipped[0]["log_width"], MIN_SUBINTERVAL_LOG_WIDTH)
        self.assertEqual(
            metadata["min_subinterval_log_width"], MIN_SUBINTERVAL_LOG_WIDTH
        )
        # two sub-intervals, not three: both transfer functions turn oscillatory at the merged
        # breakpoint
        regimes = [tuple(item["regime"]) for item in metadata["subintervals"]]
        self.assertEqual(regimes, [(False, False, False), (False, True, True)])

    def test_well_separated_hand_overs_are_not_skipped(self):
        partition = self._partition(1.0e5, 10.0, 1.0e3, 5.0e2)
        metadata = partition["metadata"]
        self.assertEqual(metadata["skipped"], [])
        self.assertEqual(len(metadata["subintervals"]), 3)

    def test_hand_over_within_the_width_of_the_response_redshift_is_not_a_breakpoint(
        self,
    ):
        """
        A hand-over closer than MIN_SUBINTERVAL_LOG_WIDTH to z_response would open a degenerate
        region at the bottom of the range. It is classified as "never oscillatory here" and
        recorded with inside_range False, and the last sub-interval ends exactly at z_response.
        """
        z_bottom = 10.0
        z_r = exp(log(1.0 + z_bottom) + 0.4 * MIN_SUBINTERVAL_LOG_WIDTH) - 1.0
        partition = self._partition(1.0e5, z_bottom, 1.0e3, z_r)
        metadata = partition["metadata"]
        self.assertEqual(metadata["skipped"], [])
        record = {bp["factor"]: bp["inside_range"] for bp in metadata["breakpoints"]}
        self.assertFalse(record["Tr"])
        self.assertTrue(record["Tq"])
        self.assertEqual(len(metadata["subintervals"]), 2)
        self.assertAlmostEqual(
            metadata["subintervals"][-1]["z_min"], z_bottom, delta=1e-12
        )
        self.assertFalse(metadata["subintervals"][-1]["regime"][2])

    def test_z_response_zero_does_not_raise(self):
        """
        The retired guard divided get_z(max_z) by get_z(min_z); at z_response = 0 that is a
        division by zero (unreachable at DEFAULT_ZEND = 0.1, but only by accident).
        """
        partition = self._partition(1.0e5, 0.0, 0.0, 0.0)
        metadata = partition["metadata"]
        print(
            f"\n[B11] z_response = 0: {len(metadata['subintervals'])} sub-interval(s), "
            f"method {metadata['subintervals'][0]['method']}"
        )
        self.assertEqual(len(metadata["subintervals"]), 1)
        self.assertEqual(metadata["subintervals"][0]["method"], "quad")
        self.assertEqual(metadata["subintervals"][0]["z_min"], 0.0)

    def test_coincident_hand_overs_of_equal_wavenumbers_are_recorded(self):
        """q = r: the two transfer functions hand over at exactly the same redshift."""
        shape = Shape("q=r", k=1.1e4, q=1.0e4, r=1.0e4, G_cross_x=33.0)
        partition = Case(0.0, shape, 100.0, exact=True).partition()
        skipped = partition["metadata"]["skipped"]
        print(f"\n[B11] q = r: skipped={skipped}")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["log_width"], 0.0)


class TestRegionCoverageTolerance(unittest.TestCase):
    """
    Prompt 13. `_check_region_covers` compares in log(1+z), the variable build_partition actually
    carries its edges in, with MIN_SUBINTERVAL_LOG_WIDTH as the tolerance -- the shape the Tq/Tr
    branch of build_partition already used. It used to compare z = exp(log_z) - 1 against the
    region boundary with an absolute tolerance of DEFAULT_FLOAT_PRECISION = 1e-7; at z ~ 5e14 one
    ulp of 1+z is ~0.06, so the sign of the round-trip rounding alone decided whether the guard
    fired, and 1372 of 3185 production work items raised
    (docs/source-remediation-verification.md section 5.1).

    These two redshifts are the shape the live run hit: z_bottom round-trips *downward* by 1.19
    and z_top round-trips *upward* by 1.13, so both ends of the retired comparison are exercised.
    """

    # z_source_max and z_response of a production-shaped sub-interval at the top of the grid
    Z_TOP = 6.8e14
    Z_BOTTOM = 4.8682e14

    def _partition(self, region_min_z=None, region_max_z=None):
        """
        One all-smooth sub-interval spanning [Z_BOTTOM, Z_TOP], with the Green's function's
        numeric region defaulting to exactly the range -- which is what production hands it,
        because that region always ends exactly at z_response.
        """
        z_top, z_bottom = self.Z_TOP, self.Z_BOTTOM
        policy = FakeGkPolicy(
            _MinimalGk(),
            "numeric",
            None,
            (
                z_top if region_max_z is None else region_max_z,
                z_bottom if region_min_z is None else region_min_z,
            ),
            None,
        )
        return build_partition(
            policy,
            _MinimalTk(z_bottom),
            _MinimalTk(z_bottom),
            _MinimalSource(z_top, z_bottom),
            z_bottom,
            z_top,
        )

    def test_region_ending_exactly_at_z_response_does_not_raise(self):
        z_top, z_bottom = self.Z_TOP, self.Z_BOTTOM
        rt_top = exp(log(1.0 + z_top)) - 1.0
        rt_bottom = exp(log(1.0 + z_bottom)) - 1.0
        print(
            f"\n[13] log/exp round trip at the sub-interval ends: "
            f"z_top {z_top:.6g} -> {rt_top:.6g} ({rt_top - z_top:+.4g}), "
            f"z_bottom {z_bottom:.6g} -> {rt_bottom:.6g} ({rt_bottom - z_bottom:+.4g}); "
            f"the retired absolute tolerance was {DEFAULT_FLOAT_PRECISION:.1e} in z"
        )
        # both ends move by ~1e7 times the retired tolerance, in opposite directions
        self.assertGreater(rt_top - z_top, DEFAULT_FLOAT_PRECISION)
        self.assertLess(rt_bottom - z_bottom, -DEFAULT_FLOAT_PRECISION)

        partition = self._partition()
        subintervals = partition["metadata"]["subintervals"]
        self.assertEqual(len(subintervals), 1)
        self.assertEqual(subintervals[0]["method"], "quad")

    def test_a_region_short_of_the_subinterval_still_raises(self):
        """
        A fix that merely stops the guard complaining is not the fix. A region whose bottom sits
        several source-grid steps above z_response must still raise, naming both the region and
        the sub-interval.
        """
        z_top, z_bottom = self.Z_TOP, self.Z_BOTTOM
        source = _MinimalSource(z_top, z_bottom)
        step = (log(1.0 + z_top) - log(1.0 + z_bottom)) / (len(source.z_sample) - 1)
        short_by = 5.0 * step
        region_min_z = exp(log(1.0 + z_bottom) + short_by) - 1.0
        print(
            f"\n[13] region bottom raised {short_by:.4g} in log(1+z) "
            f"(5 source-grid steps of {step:.4g}) to z={region_min_z:.6g}, "
            f"against a sub-interval bottom of z={z_bottom:.6g}"
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._partition(region_min_z=region_min_z)
        message = str(ctx.exception)
        print(f"[13] raised: {message}")
        self.assertIn("Gk numeric", message)
        self.assertIn(f"{region_min_z:.5g}", message)
        self.assertIn(f"{z_bottom:.5g}", message)

        # ... and the same, several grid steps the other way at the top
        region_max_z = exp(log(1.0 + z_top) - short_by) - 1.0
        with self.assertRaises(RuntimeError) as ctx:
            self._partition(region_max_z=region_max_z)
        message = str(ctx.exception)
        self.assertIn("sub-interval top", message)
        self.assertIn(f"{region_max_z:.5g}", message)

    def test_round_trip_reproduction_selects_nothing_the_log_comparison_rejects(self):
        """
        The pure-arithmetic reproduction of verification document section 5.1
        (`exp(log(1+z)) - 1 < z - 1e-7`, and its upward counterpart) over a production-shaped
        grid: every redshift it selects must now be accepted by _check_region_covers when the
        region ends exactly there, which is what the Green's-function region always does.
        """
        # the grid the pipeline actually builds -- logspace in z, not in log(1+z), as
        # wavenumber_exit_time.populate_z_sample does -- over run A's own endpoints
        z_init, z_end = 6.8788e14, 1.0e7
        n = int(
            round(
                PRODUCTION_SAMPLES_PER_LOG10Z * (np.log10(z_init) - np.log10(z_end))
                + 0.5
            )
        )
        grid = list(np.logspace(np.log10(z_init), np.log10(z_end), num=n))
        down = [
            z for z in grid if exp(log(1.0 + z)) - 1.0 < z - DEFAULT_FLOAT_PRECISION
        ]
        up = [z for z in grid if exp(log(1.0 + z)) - 1.0 > z + DEFAULT_FLOAT_PRECISION]
        print(
            f"\n[13] {len(grid)} redshifts at {PRODUCTION_SAMPLES_PER_LOG10Z} per log10(z) "
            f"over z in [1e7, 6.8788e14]: the retired comparison selected "
            f"{len(down)} downward ({100.0 * len(down) / len(grid):.0f} %) and "
            f"{len(up)} upward ({100.0 * len(up) / len(grid):.0f} %)"
        )
        self.assertGreater(len(down), 0)
        self.assertGreater(len(up), 0)

        for z in grid:
            log_z = log(1.0 + z)
            round_trip = exp(log_z) - 1.0
            # a region ending exactly at this redshift, checked against a sub-interval whose end
            # is the same redshift in log(1+z) -- the production case
            _check_region_covers("test", (z, z), log_z, log_z, round_trip, round_trip)

    def test_the_quadrature_helper_makes_the_same_comparison(self):
        """
        `numeric_quad_integral` re-checks the same "Gk numeric" region over the same sub-interval,
        and it received min_z/max_z as exp(log_z) - 1 too. build_partition raised first and masked
        it: once the guard above was corrected, 771 of the 3185 production work items raised here
        instead. It must accept a region ending exactly at the sub-interval bottom, and must still
        reject one that falls short.
        """
        z_top, z_bottom = self.Z_TOP, self.Z_BOTTOM
        log_top, log_bottom = log(1.0 + z_top), log(1.0 + z_bottom)
        # exactly the round-tripped ends evaluate_QuadSource_integral hands on as limits
        z_hi, z_lo = exp(log_top) - 1.0, exp(log_bottom) - 1.0

        model = FakeModel(1.0 / 3.0)
        k = FakeWavenumber(1.0e6, 1)

        def run(region):
            return numeric_quad_integral(
                model,
                k,
                k,
                k,
                None,
                FakeGkPolicy(_MinimalGk(), "numeric", None, region, None),
                z_bottom,
                max_z=z_hi,
                min_z=z_lo,
                atol=1.0e-25,
                rtol=1.0e-10,
                source_f=lambda x, z_is_log=False: 1.0,
            )

        payload = run((z_top, z_bottom))
        print(
            f"\n[13] numeric_quad_integral over the round-tripped limits "
            f"({z_hi:.6g}, {z_lo:.6g}) against a region ({z_top:.6g}, {z_bottom:.6g}): "
            f"value={payload['value']:.6g}"
        )
        self.assertTrue(np.isfinite(payload["value"]))

        step = (log_top - log_bottom) / 15.0
        short = exp(log_bottom + 5.0 * step) - 1.0
        with self.assertRaises(RuntimeError) as ctx:
            run((z_top, short))
        message = str(ctx.exception)
        print(f"[13] raised: {message}")
        self.assertIn("min_z", message)
        self.assertIn(f"{short:.5g}", message)


class TestPersistedSchema(unittest.TestCase):
    """
    The one schema change of the campaign: `b` (non-null), `total_abserr`, `total_converged`,
    `total_phase_limited`. Checked against the factory's own table definition -- no datastore,
    no Ray, no ShardedPool.
    """

    def _table(self):
        from Datastore.SQL.ObjectFactories.QuadSourceIntegral import (
            sqla_QuadSourceIntegral_factory,
        )

        columns = sqla_QuadSourceIntegral_factory.register()["columns"]
        return sqla.Table("QuadSourceIntegral", sqla.MetaData(), *columns)

    def test_new_columns_exist_with_the_intended_types(self):
        table = self._table()
        expected = {
            "b": (sqla.Float, False),
            "total_abserr": (sqla.Float, True),
            "total_converged": (sqla.Boolean, True),
            "total_phase_limited": (sqla.Boolean, True),
        }
        print(
            "\n[schema] "
            + ", ".join(
                f"{name}: {table.c[name].type} nullable={table.c[name].nullable}"
                for name in expected
            )
        )
        for name, (type_, nullable) in expected.items():
            with self.subTest(column=name):
                self.assertIn(name, table.c)
                self.assertIsInstance(table.c[name].type, type_)
                self.assertEqual(table.c[name].nullable, nullable)

    def test_WKB_quad_columns_are_kept(self):
        """
        WKB_quad has been identically 0.0 since prompt 08, but
        extract_QuadSourceIntegral_data.py reads obj.WKB_quad and that script is out of scope
        for this campaign, so the column (and its six timing siblings) stay.
        """
        table = self._table()
        for name in (
            "WKB_quad",
            "WKB_quad_compute_time",
            "WKB_quad_compute_steps",
            "WKB_quad_RHS_evaluations",
            "WKB_quad_mean_RHS_time",
            "WKB_quad_max_RHS_time",
            "WKB_quad_min_RHS_time",
        ):
            with self.subTest(column=name):
                self.assertIn(name, table.c)

    def test_object_round_trips_the_new_fields(self):
        """
        What the factory's build()/read_batch() do: hand the row's columns to the constructor as
        a payload, and read them back off the properties.
        """

        class Tol:
            tol = 1e-8
            store_id = 1

        class Policy:
            store_id = 9

        payload = {
            "store_id": 4321,
            "b": 0.25,
            "total": 1.5e-10,
            "total_abserr": 2.5e-18,
            "total_converged": True,
            "total_phase_limited": False,
            "numeric_quad": 1.0e-10,
            "WKB_quad": 0.0,
            "WKB_Levin": 0.5e-10,
            "analytic_rad": 1.4999e-10,
            "eta_source_max": 1.0,
            "eta_response": 2.0,
            "numeric_quad_data": None,
            "WKB_quad_data": None,
            "WKB_Levin_data": None,
            "WKB_phase_spline_chunks": 3,
            "compute_time": 0.5,
            "analytic_compute_time": 0.1,
            "source_serial": 7,
            "data_serial": 8,
            "metadata": {},
        }
        common = dict(
            model=None,
            policy=Policy(),
            z_response=FakeZ(10.0),
            z_source_max=FakeZ(1e5),
            k=FakeExitTime(1.0, 1, None),
            q=FakeExitTime(1.0, 2, None),
            r=FakeExitTime(1.0, 3, None),
            atol=Tol(),
            rtol=Tol(),
        )
        obj = QuadSourceIntegral(payload, **common)
        self.assertEqual(obj.b, 0.25)
        self.assertEqual(obj.total_abserr, 2.5e-18)
        self.assertIs(obj.total_converged, True)
        self.assertIs(obj.total_phase_limited, False)

        # query-only object: the new fields default to None like every other value
        empty = QuadSourceIntegral(None, **common)
        for name in ("_b", "_total_abserr", "_total_converged", "_total_phase_limited"):
            self.assertIsNone(getattr(empty, name))
        with self.assertRaises(RuntimeError):
            empty.b


if __name__ == "__main__":
    unittest.main()
