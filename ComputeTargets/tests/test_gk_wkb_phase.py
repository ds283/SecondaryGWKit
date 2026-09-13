"""
Tests for the Green's-function WKB phase evaluated from the conformal-time table
(``prompts/GkTk-remedial``, prompt 06 §4): ``Quadrature/integrators/WKB_phase_function.py``,
``LiouvilleGreen.WKBtools.apply_phase_offset`` and the ``(B, delta)`` store algebra of
``GkWKBIntegration.store()``.

The producer being tested computes, for every sample ``z`` of a ``GkWKBIntegration`` object,

    theta(z; z_init) = -[ k tau.delta(z_init, z) + rho.delta(z_init, z) ]

from the background model's double-double ``tau`` table (prompt 03) and a per-``k`` residual
table (prompt 05), and splits it as ``(theta_div_2pi, theta_mod_2pi)`` in the negative-remainder
convention. It replaces a two-stage phase ODE whose stored phase was wrong by 13.9 rad
(``k = 1e5/Mpc``) and 7366 rad (``k = 3e8/Mpc``) at ``z = 0.1`` and cost 63.7 s per object
(review §2-§4).

The seven items of prompt 06 §4, in order:

1. exact-radiation control at production span (``k = 1e7``: 1e7 rad, error ``<= 1e-8``;
   ``k = 1e9``: 1e9 rad, ``<= 1e-6``), the table built from ``1/H`` on the production grid;
2. the real background against prompt 01's references at the JSON checkpoints (LambdaCDM
   ``k = 1e5``: ``<= 1e-5`` rad; ``k = 3e8``: ``<= 5e-3``; QCD ``k = 3e8``: ``<= 5e-3``);
3. an off-grid anchor 37 % of the way through a grid interval, as a numeric hand-over is;
4. the ``(B, delta)`` algebra reproduces the initial data and ``sin_coeff = B > 0`` always,
   including ``G_init = 0`` and ``G_init < 0`` -- the removed sign fix was a no-op;
5. cross-object cycle consistency in the numeric-initialised band (review §8.3's ``t6_sweep``
   without the rebase): the stored unwrapped phase is the exact phase plus an integer number of
   cycles, that integer changes only where the stop point moves to the next extremum, and it
   changes there by exactly one cycle; no object carries a rebase offset;
6. the payload contract, including the ``initial_data_only`` and ``friction=True`` cases;
7. the cost per object on LambdaCDM at ``k = 3e8`` over the full production response grid.

Plus the §5 greps as a source-inspection test, and the ``main.py`` solver registration.

**Stand-ins.** The reference harness's models (``wkb_reference.RadiationModel``,
``LambdaCDMModel``, ``QCDModel``) carry ``functions.tau = None``; here each is given the three
``TablePrimitive`` accessors exactly as ``BackgroundModel._build_*_primitive`` builds them --
from the integrands on LambdaCDM and radiation, from the persisted limbs of the
``compute_background`` payload on QCD (zero quadrature). ``WKB_phase_function`` is called
through its undecorated ``_function`` with a ``ModelProxy`` stand-in exposing ``.get()`` and
``.units`` and a ``wavenumber_exit_time`` stand-in exposing ``.k.k``, ``.k.k_inv_Mpc``,
``.k.store_id``, ``.k.units``. No Ray, no datastore.
"""

import re
import time
import unittest
from math import atan2, ceil, cos, exp, expm1, fabs, log1p, pi, sin, sqrt
from pathlib import Path

import numpy as np

from ComputeTargets.BackgroundModel import (
    CS_TAU_GAUSS_ORDER,
    FRICTION_F_GAUSS_ORDER,
    TAU_GAUSS_ORDER,
    TablePrimitive,
    _cosmology_break_points,
    _cs_over_Hubble,
    _friction_integrand,
)
from ComputeTargets.WKB_Gk import Gk_d_ln_omegaEff_dz, Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import (
    Tk_d_ln_omegaEff_dz,
    Tk_omegaEff_sq,
    Tk_omegaEff_sq_leading,
)
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.phase_residual import (
    RESIDUAL_WKB_REGION_MARGIN,
    RHO_GAUSS_ORDER,
    residual_node_range,
)
from ComputeTargets.tests.wkb_reference import (
    PRODUCTION_Z_END,
    LambdaCDMModel,
    QCDModel,
    RadiationModel,
    load_references,
    phase_error,
    production_response_grid,
    production_source_grid,
    production_source_z_values,
    to_redshift_array,
)
from LiouvilleGreen.WKBtools import (
    WKB_mod_2pi,
    apply_phase_offset,
    shift_theta_sample,
    wrap_theta,
)
from LiouvilleGreen.constants import TWO_PI
from Quadrature.integration_metadata import IntegrationData
from Quadrature.integrators.WKB_phase_function import (
    PHASE_SOLVER_LABEL,
    PHASE_SOLVER_LABEL_BASE,
    PHASE_SOLVER_STEPPING,
    WKB_phase_function,
)
from Units import Mpc_units

REPO_ROOT = Path(__file__).parents[2]

# the undecorated function (the pattern of wkb_reference.QCDModel's compute_background call)
_phase = WKB_phase_function._function

# prompt 06 §4 item 1 / README §6: the exact-radiation control
RADIATION_H0 = 1.0
RADIATION_CASES = ((1.0e7, 1.0e-8), (1.0e9, 1.0e-6))  # (k, max phase error in rad)

# item 2 / README §6: the real background against prompt 01's references
LAMBDACDM_TOL = {"1.000000e+05": 1.0e-5, "3.000000e+08": 5.0e-3}
QCD_TOL = {"3.000000e+08": 5.0e-3}

# item 3: the off-grid anchor sits this far through a grid interval, in u = log(1+z)
OFF_GRID_FRACTION = 0.37

# item 4: the store algebra reproduces the initial data to this relative accuracy
STORE_ALGEBRA_REL_TOL = 1.0e-12
STORE_ALGEBRA_SAMPLES = 200

# item 5: the review's sweep geometry (t6_sweep.py): 15 k values, three response points
SWEEP_K_VALUES = np.geomspace(1.0e6, 1.0e8, 15)
SWEEP_X_R_VALUES = (1.0e2, 1.0e3, 1.0e4)
SWEEP_SAMPLES_PER_DECADE = 100
SWEEP_RESPONSE_SPARSENESS = 12
SWEEP_CYCLE_TOL = 1.0e-9

# item 7 / README §6: wall time per GkWKBIntegration object at k = 3e8 on LambdaCDM
COST_WALL_TIME_LIMIT = 0.05
COST_REPEATS = 3

# the theta_T control in item 6 uses the exact rho_T primitive of the radiation model
RADIATION_TK_TOL = 1.0e-8

# --------------------------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------------------------


class _Wavenumber:
    def __init__(self, k: float, store_id: int, units):
        self.k = float(k)
        self.k_inv_Mpc = float(k)
        self.store_id = store_id
        self.units = units


class _KExit:
    """A ``wavenumber_exit_time`` stand-in: ``.k`` and, where a test needs them, the e-fold
    redshifts."""

    def __init__(self, k: float, units, store_id: int = 1, z_exit_subh_e3=None):
        self.k = _Wavenumber(k, store_id, units)
        self.z_exit_subh_e3 = z_exit_subh_e3


class _Proxy:
    """A ``ModelProxy`` stand-in: ``.get()`` and ``.units`` (for ``check_units``)."""

    def __init__(self, model, units):
        self._model = model
        self.units = units

    def get(self):
        return self._model


def _attach_tables_from_integrands(model, z_nodes, inverse_Hubble, cs_over_H, friction):
    """Give a stand-in the three ``TablePrimitive`` accessors, built from their integrands on
    ``z_nodes`` exactly as ``compute_background`` builds them (order 4, the cosmology's break
    points)."""
    cosmology = model.cosmology
    z_nodes = np.asarray(z_nodes, dtype=float)
    break_points = (
        _cosmology_break_points(cosmology, float(z_nodes[-1]), float(z_nodes[0]))
        if cosmology is not None
        else np.empty(0)
    )
    tau = CumulativeTable(
        z_nodes, inverse_Hubble, TAU_GAUSS_ORDER, break_points=break_points, label="tau"
    )
    cs_tau = CumulativeTable(
        z_nodes,
        cs_over_H,
        CS_TAU_GAUSS_ORDER,
        break_points=break_points,
        label="cs_tau",
    )
    friction_F = CumulativeTable(
        z_nodes,
        friction,
        FRICTION_F_GAUSS_ORDER,
        break_points=break_points,
        label="friction_F",
    )
    model.functions = model.functions._replace(
        tau=TablePrimitive(tau, "tau"),
        cs_tau=TablePrimitive(cs_tau, "cs_tau"),
        friction_F=TablePrimitive(friction_F, "friction_F"),
    )
    return model


def radiation_model_with_tables(z_nodes, H0: float = RADIATION_H0) -> RadiationModel:
    """``RadiationModel`` with its tables built from ``f = 1/H`` on ``z_nodes``."""
    model = RadiationModel(H0)
    return _attach_tables_from_integrands(
        model,
        z_nodes,
        lambda z: 1.0 / model.Hubble(z),
        lambda z: 1.0 / (sqrt(3.0) * model.Hubble(z)),
        # minus the derivative of F, as _friction_integrand hands CumulativeTable:
        # dF/dz = (3/2)(1 + 1/3)/(1+z) = 2/(1+z)
        lambda z: -2.0 / (1.0 + z),
    )


def lambdacdm_model_with_tables(z_nodes) -> LambdaCDMModel:
    model = LambdaCDMModel()
    c = model.cosmology
    return _attach_tables_from_integrands(
        model,
        z_nodes,
        lambda z: 1.0 / c.Hubble(z),
        _cs_over_Hubble(c),
        _friction_integrand(c),
    )


def qcd_model_with_tables(grid) -> QCDModel:
    """``QCDModel`` with its tables reconstructed from the ``compute_background`` payload's
    persisted limbs, as ``BackgroundModel._build_*_primitive`` does (no quadrature)."""
    model = QCDModel(grid)
    c = model.cosmology
    payload = model.background_payload
    z_nodes = np.array(grid.as_float_list(), dtype=float)
    break_points = _cosmology_break_points(c, float(z_nodes[-1]), float(z_nodes[0]))

    tau = CumulativeTable(
        z_nodes,
        lambda z: 1.0 / c.Hubble(z),
        TAU_GAUSS_ORDER,
        hi=payload["tau_hi_sample"],
        lo=payload["tau_lo_sample"],
        break_points=break_points,
        label="tau",
    )
    cs_tau = CumulativeTable(
        z_nodes,
        _cs_over_Hubble(c),
        CS_TAU_GAUSS_ORDER,
        hi=payload["cs_tau_hi_sample"],
        lo=payload["cs_tau_lo_sample"],
        break_points=break_points,
        label="cs_tau",
    )
    friction_F = CumulativeTable(
        z_nodes,
        _friction_integrand(c),
        FRICTION_F_GAUSS_ORDER,
        hi=payload["friction_F_sample"],
        lo=[0.0] * len(z_nodes),
        break_points=break_points,
        label="friction_F",
    )
    model.functions = model.functions._replace(
        tau=TablePrimitive(tau, "tau"),
        cs_tau=TablePrimitive(cs_tau, "cs_tau"),
        friction_F=TablePrimitive(friction_F, "friction_F"),
    )
    return model


def _radiation_z_e3(k: float, H0: float = RADIATION_H0) -> float:
    """``k(1+z)/H = e^3`` with ``H = H0 (1+z)^2``: ``1+z = k/(H0 e^3)``."""
    return k / (H0 * exp(3.0)) - 1.0


def _unwrapped(payload) -> np.ndarray:
    return np.array(
        [
            d * TWO_PI + m
            for d, m in zip(
                payload["theta_div_2pi_sample"], payload["theta_mod_2pi_sample"]
            )
        ],
        dtype=float,
    )


def _run_Gk(model, units, k: float, z_init: float, z_sample, friction=False) -> dict:
    return _phase(
        _Proxy(model, units),
        _KExit(k, units),
        z_init,
        z_sample,
        sector="Gk",
        omega_sq=Gk_omegaEff_sq,
        d_ln_omega_dz=Gk_d_ln_omegaEff_dz,
        friction=friction,
        task_label="test_gk_wkb_phase",
        object_label="test",
    )


def _run_Tk(model, units, k: float, z_init: float, z_sample) -> dict:
    return _phase(
        _Proxy(model, units),
        _KExit(k, units),
        z_init,
        z_sample,
        sector="Tk",
        omega_sq=Tk_omegaEff_sq,
        d_ln_omega_dz=Tk_d_ln_omegaEff_dz,
        friction=True,
        task_label="test_gk_wkb_phase",
        object_label="test",
    )


def store_algebra(
    omega_sq_init: float,
    d_ln_omega_init: float,
    eps_init: float,
    z_init: float,
    G_init: float,
    Gprime_init: float,
):
    """
    A verbatim replica of the ``(B, deltaTheta)`` algebra of ``GkWKBIntegration.store()``
    (README §2 (f)), which cannot be exercised directly without Ray. Returns
    ``(B, deltaTheta, raw_cos, raw_sin)``.
    """
    one_plus_z_init = 1.0 + z_init
    omega_init = sqrt(omega_sq_init)
    sqrt_omega_init = sqrt(omega_init)

    raw_cos_coeff = sqrt_omega_init * G_init
    raw_sin_coeff = (
        Gprime_init + (G_init / 2.0) * (d_ln_omega_init + eps_init / one_plus_z_init)
    ) / sqrt_omega_init

    deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
    B = sqrt(raw_cos_coeff * raw_cos_coeff + raw_sin_coeff * raw_sin_coeff)
    return B, deltaTheta, raw_cos_coeff, raw_sin_coeff


def removed_sign_fix(deltaTheta: float, G_init: float) -> int:
    """The factor ``sgn(sin deltaTheta) * sgn(G_init)`` that ``store()`` used to multiply ``B``
    by, with its ``>= 0`` conventions (review §8.1)."""
    sgn_sin = +1 if sin(deltaTheta) >= 0.0 else -1
    sgn_G = +1 if G_init >= 0.0 else -1
    return sgn_sin * sgn_G


# --------------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------------


class _Shared:
    """The references, the production grid and both production models with tables, built
    once for the module."""

    references = None
    grid = None
    z_nodes = None
    lambdacdm = None
    qcd = None

    @classmethod
    def build(cls):
        if cls.references is not None:
            return
        cls.references = load_references()
        cls.grid = production_source_grid(
            cls.references["models"]["LambdaCDMModel"]["grid"]["z_init"]
        )
        cls.z_nodes = np.array(cls.grid.as_float_list(), dtype=float)
        cls.lambdacdm = lambdacdm_model_with_tables(cls.z_nodes)

    @classmethod
    def build_qcd(cls):
        cls.build()
        if cls.qcd is None:
            cls.qcd = qcd_model_with_tables(cls.grid)

    @classmethod
    def reference_theta(cls, model_key: str, k_key: str):
        """
        ``[(z, theta_ref)]`` at the JSON checkpoints below the ``rho`` anchor of ``k``:
        ``theta_ref = -k [tau_minus_top(z) - tau_minus_top(anchor)] - rho_G``, in the README §2
        convention (the JSON's own ``"schema"`` block).
        """
        block = cls.references["models"][model_key]
        k = float(k_key)
        anchor_tau = block["primitives_at_rho_anchor"][k_key]["tau_minus_top"]
        checkpoint_position = {
            cp["index"]: i for i, cp in enumerate(block["checkpoints"])
        }
        out = []
        for entry in block["rho_G"][k_key]:
            tau_minus_top = block["tau_minus_top"][checkpoint_position[entry["index"]]]
            theta_ref = -k * (tau_minus_top - anchor_tau) - float(entry["value"])
            out.append((float(entry["z"]), theta_ref))
        return out


# --------------------------------------------------------------------------------------------
# 1. exact-radiation control at production span
# --------------------------------------------------------------------------------------------


class TestRadiationControl(unittest.TestCase):
    """Prompt 06 §4 item 1."""

    def _run(self, k: float, tol: float):
        z_e3 = _radiation_z_e3(k)
        source_z = production_source_z_values(z_e3, PRODUCTION_Z_END)
        model = radiation_model_with_tables(source_z)
        units = Mpc_units()
        response = production_response_grid(to_redshift_array(source_z))
        z_init = float(source_z[0])

        payload = _run_Gk(model, units, k, z_init, response)
        theta = _unwrapped(payload)

        worst, worst_z = 0.0, None
        for z, got in zip(response.as_float_list(), theta):
            ref = -k * model.tau_delta(z_init, z)
            err = phase_error(got, ref)
            if err > worst:
                worst, worst_z = err, z
        span = fabs(theta[-1])
        floor = np.finfo(float).eps * span
        print(
            f"\n[radiation] k = {k:.3e}: span {span:.4e} rad over {len(response)} response "
            f"samples; max phase error {worst:.4e} rad at z = {worst_z:.6g} "
            f"(eps*theta floor {floor:.2e}; threshold {tol:g}); {payload['metadata']}"
        )
        self.assertLessEqual(worst, tol)
        # the mod-2pi remainders are in the negative-remainder convention
        for m in payload["theta_mod_2pi_sample"]:
            self.assertGreater(m, -TWO_PI)
            self.assertLessEqual(m, 0.0)
        return span, worst

    def test_span_1e7_rad(self):
        """The ODE gave 9.7e-3 rad here (review §2); the floor is ~2e-9."""
        k, tol = RADIATION_CASES[0]
        span, worst = self._run(k, tol)
        self.assertGreater(span, 5.0e6)

    def test_span_1e9_rad(self):
        """The ODE gave 0.98 rad here (review §2)."""
        k, tol = RADIATION_CASES[1]
        span, worst = self._run(k, tol)
        self.assertGreater(span, 5.0e8)

    def test_residual_is_exactly_zero_in_radiation(self):
        """``C == 0`` in exact radiation, so the residual table contributes nothing and
        ``theta`` is ``k Delta tau`` and nothing else; the metadata records ``rho_end == 0``.
        """
        k, _ = RADIATION_CASES[0]
        z_e3 = _radiation_z_e3(k)
        source_z = production_source_z_values(z_e3, PRODUCTION_Z_END)
        model = radiation_model_with_tables(source_z)
        response = production_response_grid(to_redshift_array(source_z))
        payload = _run_Gk(model, Mpc_units(), k, float(source_z[0]), response)
        self.assertEqual(payload["metadata"]["rho_end"], 0.0)


# --------------------------------------------------------------------------------------------
# 2. the real background at production spans
# --------------------------------------------------------------------------------------------


class TestRealBackground(unittest.TestCase):
    """Prompt 06 §4 item 2: LambdaCDM at k = 1e5 and 3e8, QCD at 3e8, against prompt 01's
    references at the JSON checkpoints."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def _score(self, model, model_key: str, k_key: str, tol: float):
        block = self.s.references["models"][model_key]
        k = float(k_key)
        z_anchor = float(block["rho_anchor_z"][k_key])
        reference = self.s.reference_theta(model_key, k_key)
        samples = to_redshift_array([z for z, _ in reference])

        payload = _run_Gk(model, model.cosmology.units, k, z_anchor, samples)
        theta = dict(zip(samples.as_float_list(), _unwrapped(payload)))

        worst, worst_z, span = 0.0, None, 0.0
        for z, theta_ref in reference:
            err = phase_error(theta[z], theta_ref)
            span = max(span, fabs(theta_ref))
            if err > worst:
                worst, worst_z = err, z
        floor = np.finfo(float).eps * span
        print(
            f"\n[{model_key}] k = {k:.3e}: {len(reference)} checkpoints from the anchor "
            f"z = {z_anchor:.6g} to z = {min(theta):.3g}, span {span:.4e} rad; max phase "
            f"error {worst:.4e} rad at z = {worst_z:.6g} (eps*k*tau floor {floor:.2e}; "
            f"threshold {tol:g}); {payload['metadata']}"
        )
        self.assertTrue(payload["metadata"]["offgrid_init"])
        self.assertLessEqual(worst, tol)
        return worst, worst_z

    def test_lambdacdm_k_1e5(self):
        """Review §4: the ODE was 13.9 rad off at z = 0.1."""
        self._score(
            self.s.lambdacdm,
            "LambdaCDMModel",
            "1.000000e+05",
            LAMBDACDM_TOL["1.000000e+05"],
        )

    def test_lambdacdm_k_3e8(self):
        """Review §4: the ODE was 7366 rad off at z = 0.1. The reference and the reconstructed
        ``div*2pi + mod`` each carry the eps*k*tau floor of ~5e-4 rad here (README §2 (d)).
        """
        self._score(
            self.s.lambdacdm,
            "LambdaCDMModel",
            "3.000000e+08",
            LAMBDACDM_TOL["3.000000e+08"],
        )

    def test_qcd_k_3e8(self):
        """The LG truncation floor on QCD is ~1e-3 rad (README §2 (d)); the table's own error
        is far below it, but the reference is the same LG integral so this scores the
        quadrature, not the physics."""
        _Shared.build_qcd()
        self._score(self.s.qcd, "QCDModel", "3.000000e+08", QCD_TOL["3.000000e+08"])


# --------------------------------------------------------------------------------------------
# 3. off-grid anchor
# --------------------------------------------------------------------------------------------


class TestOffGridAnchor(unittest.TestCase):
    """Prompt 06 §4 item 3: a numeric hand-over's z_init is a root, not a node."""

    def test_anchor_37_percent_through_a_grid_interval(self):
        k, tol = RADIATION_CASES[0]
        z_e3 = _radiation_z_e3(k)
        source_z = production_source_z_values(z_e3, PRODUCTION_Z_END)
        model = radiation_model_with_tables(source_z)
        units = Mpc_units()

        z_hi, z_lo = float(source_z[0]), float(source_z[1])
        width = log1p((z_hi - z_lo) / (1.0 + z_lo))
        z_init = z_lo + (1.0 + z_lo) * expm1(OFF_GRID_FRACTION * width)
        self.assertIsNone(model.functions.tau.table.node_index(z_init))

        response = production_response_grid(to_redshift_array(source_z))
        samples = to_redshift_array(
            [z for z in response.as_float_list() if z <= z_init]
        )

        payload = _run_Gk(model, units, k, z_init, samples)
        theta = _unwrapped(payload)

        worst, worst_z = 0.0, None
        for z, got in zip(samples.as_float_list(), theta):
            err = phase_error(got, -k * model.tau_delta(z_init, z))
            if err > worst:
                worst, worst_z = err, z
        print(
            f"\n[off-grid anchor] k = {k:.3e}, z_init = {z_init:.10g} ({OFF_GRID_FRACTION:.0%} "
            f"through [{z_lo:.6g}, {z_hi:.6g}] in u): max phase error {worst:.4e} rad at "
            f"z = {worst_z:.6g}; {payload['metadata']}"
        )
        self.assertLessEqual(worst, tol)
        self.assertTrue(payload["metadata"]["offgrid_init"])
        # one leading-table partial per sample, none for the residual (z_init is its top node)
        self.assertEqual(payload["metadata"]["lead_partials"], len(samples))
        self.assertEqual(
            payload["metadata"]["lead_evals"], TAU_GAUSS_ORDER * len(samples)
        )

    def test_residual_node_range_is_the_grid_cut_at_the_wkb_region(self):
        """Prompt 14 §2 item 1 replaces prompt 06's per-object ``residual_nodes(grid, z_sample,
        z_init)`` by a range that depends only on ``(model, k, sector)``: the whole grid, cut at
        the top where the Liouville-Green frequency stops keeping the fraction
        ``RESIDUAL_WKB_REGION_MARGIN`` of its leading term. In exact radiation ``C = 0`` for the
        Green's function, so ``omega^2 = (k/H)^2`` keeps all of it at every node and nothing is
        cut; for the transfer function ``C_T = -2/s^2``, so
        ``omega_T^2/omega_{T,0}^2 = 1 - 6 H0^2 s^2/k^2`` falls through one half at
        ``s = k/(2 sqrt(3) H0)``, which the grid used here straddles."""
        k = 1.0e7
        grid = production_source_z_values(1.0e8, PRODUCTION_Z_END)
        model = RadiationModel(RADIATION_H0)

        nodes = residual_node_range(model, k, grid, "Gk")
        self.assertEqual(nodes.tolist(), grid.tolist())

        nodes = residual_node_range(model, k, grid, "Tk")
        self.assertLess(nodes[0], grid[0])
        self.assertEqual(nodes[-1], grid[-1])

        def kept(z: float) -> float:
            leading = Tk_omegaEff_sq_leading(model, k, z)
            return Tk_omegaEff_sq(model, k, z) / leading

        # the top is the highest node that keeps the margin, and every node below keeps it
        j = int(np.argmax(grid == nodes[0]))
        self.assertEqual(nodes.tolist(), grid[j:].tolist())
        self.assertGreaterEqual(kept(float(grid[j])), RESIDUAL_WKB_REGION_MARGIN)
        self.assertLess(kept(float(grid[j - 1])), RESIDUAL_WKB_REGION_MARGIN)
        self.assertTrue(
            all(kept(float(z)) >= RESIDUAL_WKB_REGION_MARGIN for z in nodes)
        )
        # and it sits at the analytic crossing s = k/(2 sqrt(3) H0)
        self.assertLess(
            fabs(nodes[0] + 1.0 - k / (2.0 * sqrt(3.0) * RADIATION_H0)) / nodes[0], 0.05
        )


# --------------------------------------------------------------------------------------------
# 4. the (B, delta) store algebra
# --------------------------------------------------------------------------------------------


class TestStoreAlgebra(unittest.TestCase):
    """Prompt 06 §4 item 4."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_initial_data_reproduced_and_sin_coeff_positive(self):
        model = self.s.lambdacdm
        block = self.s.references["models"]["LambdaCDMModel"]
        k = 1.0e5
        z_init = float(block["rho_anchor_z"]["1.000000e+05"])

        omega_sq_init = Gk_omegaEff_sq(model, k, z_init)
        d_ln_omega_init = Gk_d_ln_omegaEff_dz(model, k, z_init)
        eps_init = model.functions.epsilon(z_init)
        omega_init = sqrt(omega_sq_init)
        one_plus_z = 1.0 + z_init

        rng = np.random.default_rng(20260911)
        cases = []
        for _ in range(STORE_ALGEBRA_SAMPLES):
            scale = 10.0 ** rng.uniform(-6.0, 6.0)
            cases.append((scale * rng.normal(), scale * omega_init * rng.normal()))
        # the special cases the removed sign fix was written for
        cases += [
            (0.0, 1.0),
            (0.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 0.5),
            (-2.5, -0.5),
            (1.0, 0.0),
        ]

        worst_G, worst_Gprime = 0.0, 0.0
        for G_init, Gprime_init in cases:
            B, delta, raw_cos, raw_sin = store_algebra(
                omega_sq_init, d_ln_omega_init, eps_init, z_init, G_init, Gprime_init
            )
            # sin_coeff = B, and it is positive
            self.assertGreater(B, 0.0)
            # the removed factor sgn(sin delta) sgn(G_init) was +1 in every case
            self.assertEqual(removed_sign_fix(delta, G_init), +1)

            # the LG solution norm(z) B sin(theta + delta) with norm = sqrt(H_i/(H omega)),
            # theta(z_init) = 0 and dtheta/dz = omega, matched at z_init:
            #   G(z_i)  = B sin(delta)/sqrt(omega_i)
            #   G'(z_i) = -(G_i/2)(d ln omega + eps/(1+z)) + sqrt(omega_i) B cos(delta)
            amplitude = B / sqrt(omega_init)
            G_recon = B * sin(delta) / sqrt(omega_init)
            Gprime_recon = -(G_recon / 2.0) * (
                d_ln_omega_init + eps_init / one_plus_z
            ) + sqrt(omega_init) * B * cos(delta)
            worst_G = max(worst_G, fabs(G_recon - G_init) / amplitude)
            worst_Gprime = max(
                worst_Gprime,
                fabs(Gprime_recon - Gprime_init) / (amplitude * omega_init),
            )

        print(
            f"\n[store algebra] {len(cases)} (G, G') pairs at k = {k:.1e}, z_init = {z_init:.6g}: "
            f"G reproduced to {worst_G:.3e}, G' to {worst_Gprime:.3e} of the LG amplitude; "
            f"sin_coeff = B > 0 and the removed sign factor was +1 in every case"
        )
        self.assertLessEqual(worst_G, STORE_ALGEBRA_REL_TOL)
        self.assertLessEqual(worst_Gprime, STORE_ALGEBRA_REL_TOL)


# --------------------------------------------------------------------------------------------
# 5. cross-object cycle consistency (review §8.3's t6 sweep, without the rebase)
# --------------------------------------------------------------------------------------------


def _sweep(k: float, x_r: float, stop: str):
    """
    Review §8.3's simulation in exact radiation (``H0 = 1``, ``s = 1+z``), for one ``k`` and one
    response point: for every source ``s_s`` in the numeric-initialised band
    ``[sqrt(s_e3 s_e4), s_e3]`` at 100 per decade, the stop point is the first extremum of
    ``G(s; s_s) ~ sin(theta(s_s, s))`` past the 4-e-fold point -- a *maximum* for
    ``stop="maximum"`` (the production geometry: README §2 (h), ``G_stop/env = +1.000000`` in
    every run), alternating maxima and minima for ``stop="extremum"`` (the review's
    ``t6_sweep``). The object is built from the exact initial data at the stop, the store
    algebra and ``apply_phase_offset`` are applied, and the record for the response point is
    returned along with the exact phase ``theta(s_s, s_r)`` and the stop index.

    Phases are formed with the exact radiation primitive rather than ``WKB_phase_function`` (the
    table is exact in radiation and its accuracy is items 1 and 3); this isolates the
    cycle-bookkeeping question.
    """
    e3, e4 = exp(3.0), exp(4.0)
    s_e3, s_e4 = k / e3, k / e4
    s_lim = sqrt(s_e3 * s_e4)
    sources = np.geomspace(
        s_lim, s_e3, int(np.log10(s_e3 / s_lim) * SWEEP_SAMPLES_PER_DECADE) + 1
    )
    full = np.geomspace(1.0, s_e3, int(np.log10(s_e3) * SWEEP_SAMPLES_PER_DECADE) + 1)
    response = full[::-1][::SWEEP_RESPONSE_SPARSENESS][::-1]
    s_r = response[np.argmin(np.abs(response - k / x_r))]

    def theta(s_a: float, s_b: float) -> float:
        # k(1/s_a - 1/s_b), factored: negative for s_b < s_a
        return -k * (s_a - s_b) / (s_a * s_b)

    records = []
    for s_s in sources:
        x_s = k / s_s
        p4 = x_s - e4  # theta(s_s, s_e4): the phase accumulated to the 4-e-fold point
        if stop == "maximum":
            # theta_i = -(3pi/2 + 2 pi n): maxima of sin(theta) on the negative axis
            n = max(0, ceil((-p4 - 1.5 * pi) / TWO_PI))
            theta_i = -(1.5 * pi + TWO_PI * n)
        else:
            # theta_i = -(pi/2 + pi n): every extremum
            n = max(0, ceil((-p4 - 0.5 * pi) / pi))
            theta_i = -(0.5 * pi + pi * n)
        s_i = 1.0 / (1.0 / s_s - theta_i / k)

        # exact initial data at the stop: G ~ (s_s^2/k) sin(theta), dG/dz = cos(theta) s_s^2/s_i^2
        G_i = (s_s * s_s / k) * sin(theta_i)
        Gprime_i = cos(theta_i) * s_s * s_s / (s_i * s_i)
        omega_sq_i = (k / (s_i * s_i)) ** 2
        # in exact radiation d ln omega/dz = -2/s and eps/(1+z) = 2/s
        B, delta, _, _ = store_algebra(
            omega_sq_i, -2.0 / s_i, 2.0, s_i - 1.0, G_i, Gprime_i
        )
        assert B > 0.0

        rs = response[response <= min(s_e3, s_i)][
            ::-1
        ]  # descending, as a redshift_array is
        div, mod = map(list, zip(*[WKB_mod_2pi(theta(s_i, s)) for s in rs]))
        div2, mod2 = apply_phase_offset(div, mod, delta)

        j = int(np.argmin(np.abs(rs - s_r)))
        records.append(
            {
                "s_s": float(s_s),
                "z_s": float(s_s - 1.0),
                "n": int(n),
                "delta": float(delta),
                "stored": div2[j] * TWO_PI + mod2[j],
                "exact": theta(s_s, s_r),
                # the invariant of apply_phase_offset: no cross-sample rebase
                # (formed from the small per-sample shift, not from the ~1e8 rad totals,
                # whose ulp is 1e-8)
                "offset_defect": max(
                    fabs((d2 - d) * TWO_PI + m2 - (m + delta))
                    for d, m, d2, m2 in zip(div, mod, div2, mod2)
                ),
                # what shift_theta_sample would have subtracted from every div of this object
                "rebase_base": wrap_theta(mod[0] + delta)[0],
                "would_rebase": shift_theta_sample(div, mod, delta)[0][j] != div2[j],
            }
        )
    return records


class TestCrossObjectConsistency(unittest.TestCase):
    """Prompt 06 §4 item 5."""

    def _analyse(self, stop: str, assert_count: bool):
        totals = {"objects": 0, "jumps": 0, "transitions": 0, "would_rebase": 0}
        worst_integer_defect = 0.0
        worst_offset_defect = 0.0
        example = None
        for k in SWEEP_K_VALUES:
            for x_r in SWEEP_X_R_VALUES:
                recs = _sweep(float(k), x_r, stop)
                D = np.array([r["stored"] - r["exact"] for r in recs])
                cycles = D / TWO_PI
                worst_integer_defect = max(
                    worst_integer_defect,
                    float(np.max(np.abs(cycles - np.round(cycles)))),
                )
                worst_offset_defect = max(
                    worst_offset_defect, max(r["offset_defect"] for r in recs)
                )
                rounded = np.round(cycles).astype(int)
                jumps = np.diff(rounded)
                # every jump between neighbours is exactly one cycle
                self.assertTrue(np.all(np.abs(jumps) <= 1), msg=f"k={k}, x_r={x_r}")
                n = np.array([r["n"] for r in recs])
                transitions = np.diff(n) != 0
                totals["objects"] += len(recs)
                totals["jumps"] += int(np.sum(jumps != 0))
                totals["transitions"] += int(np.sum(transitions))
                totals["would_rebase"] += sum(1 for r in recs if r["would_rebase"])
                if assert_count:
                    # the stored cycle count changes exactly where the stop point moves to
                    # the next extremum (RECONCILIATION.md §2 item 6), and nowhere else
                    self.assertTrue(
                        np.array_equal(jumps != 0, transitions), msg=f"k={k}, x_r={x_r}"
                    )
                if (
                    example is None
                    and fabs(k - 1.0e7) / 1.0e7 < 1.0e-9
                    and x_r == 1.0e3
                ):
                    example = (
                        float(k),
                        x_r,
                        [
                            (recs[i]["z_s"], recs[i + 1]["z_s"], int(jumps[i]))
                            for i in range(len(jumps))
                            if jumps[i] != 0
                        ],
                        len(recs),
                    )
        print(
            f"\n[sweep, stop at {stop}] {totals['objects']} objects over {len(SWEEP_K_VALUES)} k "
            f"x {len(SWEEP_X_R_VALUES)} response points: stored - exact is an integer number "
            f"of cycles to {worst_integer_defect * TWO_PI:.2e} rad; apply_phase_offset "
            f"defect {worst_offset_defect:.2e} rad; {totals['jumps']} one-cycle jumps between "
            f"neighbours against {totals['transitions']} stop-point transitions; "
            f"shift_theta_sample would have rebased {totals['would_rebase']} of them"
        )
        if example is not None:
            k, x_r, wraps, count = example
            print(
                f"[sweep, stop at {stop}] k = {k:.3e}, x_r = {x_r:.0e}: {count} objects, "
                f"{len(wraps)} cycle jumps, between neighbouring z_s = "
                + ", ".join(f"({a:.6g} -> {b:.6g}: {j:+d})" for a, b, j in wraps)
            )
        self.assertLessEqual(worst_integer_defect * TWO_PI, SWEEP_CYCLE_TOL)
        self.assertLessEqual(worst_offset_defect, SWEEP_CYCLE_TOL)
        return totals

    def test_production_geometry_stop_at_maxima(self):
        """The stop is a maximum of G (README §2 (h)); delta = +pi/2 for every object, and
        the stored cycle count steps by one exactly where the stop moves to the next
        maximum -- the delta-wrap mechanism of RECONCILIATION.md §2 item 6 that the GkSource
        rectifier repairs and prompt 09 must build phi from the rectified div."""
        totals = self._analyse("maximum", assert_count=True)
        self.assertGreater(totals["jumps"], 0)
        self.assertEqual(totals["jumps"], totals["transitions"])

    def test_review_geometry_alternating_extrema(self):
        """The review's t6 geometry (alternating maxima and minima, delta = +-pi/2). The cycle
        count steps at every second transition -- where delta goes from -pi/2 to +pi/2 -- so
        the jumps are a strict subset of the transitions; reported, and the invariants
        asserted."""
        totals = self._analyse("extremum", assert_count=False)
        self.assertGreater(totals["jumps"], 0)
        self.assertLessEqual(totals["jumps"], totals["transitions"])


# --------------------------------------------------------------------------------------------
# 6. payload contract
# --------------------------------------------------------------------------------------------


class TestPayloadContract(unittest.TestCase):
    """Prompt 06 §4 item 6."""

    REQUIRED_KEYS = {
        "stage_1_data",
        "stage_2_data",
        "theta_div_2pi_sample",
        "theta_mod_2pi_sample",
        "phase_solver_label",
        "has_WKB_violation",
        "WKB_violation_z",
        "WKB_violation_efolds_subh",
        "metadata",
    }
    FRICTION_KEYS = {"friction_sample", "friction_data", "friction_solver_label"}

    @classmethod
    def setUpClass(cls):
        k = RADIATION_CASES[0][0]
        cls.k = k
        cls.source_z = production_source_z_values(_radiation_z_e3(k), PRODUCTION_Z_END)
        cls.model = radiation_model_with_tables(cls.source_z)
        cls.units = Mpc_units()
        cls.response = production_response_grid(to_redshift_array(cls.source_z))
        cls.z_init = float(cls.source_z[0])

    def _assert_empty(self, data):
        self.assertIsInstance(data, IntegrationData)
        for field in data._fields:
            self.assertIsNone(getattr(data, field), msg=field)

    def test_keys_and_stage_data(self):
        payload = _run_Gk(self.model, self.units, self.k, self.z_init, self.response)
        self.assertEqual(set(payload.keys()), self.REQUIRED_KEYS)
        self.assertEqual(payload["phase_solver_label"], PHASE_SOLVER_LABEL)
        self.assertEqual(
            PHASE_SOLVER_LABEL,
            f"{PHASE_SOLVER_LABEL_BASE}-stepping{PHASE_SOLVER_STEPPING}",
        )
        self.assertEqual(PHASE_SOLVER_STEPPING, RHO_GAUSS_ORDER)

        self._assert_empty(payload["stage_2_data"])
        stage_1 = payload["stage_1_data"]
        self.assertIsInstance(stage_1, IntegrationData)
        self.assertGreater(stage_1.compute_time, 0.0)
        self.assertEqual(stage_1.compute_steps, len(self.response))
        self.assertEqual(
            stage_1.RHS_evaluations,
            payload["metadata"]["rho_evals"] + payload["metadata"]["lead_evals"],
        )
        self.assertEqual(len(payload["theta_div_2pi_sample"]), len(self.response))
        self.assertTrue(
            all(isinstance(d, int) for d in payload["theta_div_2pi_sample"])
        )
        self.assertFalse(payload["has_WKB_violation"])
        self.assertIsNone(payload["WKB_violation_z"])
        for key in (
            "solver",
            "sector",
            "N_rho",
            "N_lead",
            "rho_nodes",
            "rho_evals",
            "rho_reused",
            "lead_partials",
            "lead_evals",
            "offgrid_init",
            "rho_end",
        ):
            self.assertIn(key, payload["metadata"])
        self.assertNotIn("initial_data_only", payload["metadata"])
        # on-grid anchor and on-grid samples: no leading-table partials at all
        self.assertFalse(payload["metadata"]["offgrid_init"])
        self.assertEqual(payload["metadata"]["lead_partials"], 0)
        self.assertEqual(payload["metadata"]["lead_evals"], 0)

    def test_initial_data_only(self):
        payload = _run_Gk(
            self.model,
            self.units,
            self.k,
            self.z_init,
            to_redshift_array([self.z_init]),
        )
        self.assertTrue(payload["metadata"]["initial_data_only"])
        self.assertEqual(payload["theta_div_2pi_sample"], [0])
        self.assertEqual(payload["theta_mod_2pi_sample"], [0.0])
        self._assert_empty(payload["stage_1_data"])
        self._assert_empty(payload["stage_2_data"])
        self.assertEqual(payload["phase_solver_label"], PHASE_SOLVER_LABEL)

        payload = _run_Tk(
            self.model,
            self.units,
            self.k,
            self.z_init,
            to_redshift_array([self.z_init]),
        )
        self.assertEqual(payload["friction_sample"], [0.0])
        self._assert_empty(payload["friction_data"])
        self.assertEqual(payload["friction_solver_label"], PHASE_SOLVER_LABEL)

    def test_friction_and_the_transfer_function_sector(self):
        """``friction=True`` returns ``friction_F.delta(z_init, z)`` per sample (bit-equal to
        the accessor, and equal to the closed form ``2 log((1+z)/(1+z_init))``), and the
        ``Tk`` sector's phase is ``-k cs_tau.delta - rho_T`` against the exact radiation
        primitive -- the interim transfer-function path of prompt 06 §3.4."""
        payload = _run_Tk(self.model, self.units, self.k, self.z_init, self.response)
        self.assertTrue(self.FRICTION_KEYS.issubset(payload.keys()))
        self.assertEqual(payload["friction_solver_label"], PHASE_SOLVER_LABEL)
        self._assert_empty(payload["friction_data"])
        self.assertEqual(payload["metadata"]["sector"], "Tk")

        zs = self.response.as_float_list()
        friction_F = self.model.functions.friction_F
        worst_F = 0.0
        for z, f in zip(zs, payload["friction_sample"]):
            self.assertEqual(f, friction_F.delta(self.z_init, z))
            worst_F = max(
                worst_F, fabs(f - self.model.friction_F_delta(z, self.z_init))
            )
            self.assertLess(f, 0.0)

        theta = _unwrapped(payload)
        worst_T, worst_z = 0.0, None
        for z, got in zip(zs, theta):
            ref = -self.k * self.model.cs_tau_delta(self.z_init, z) - self.model.rho_T(
                self.k, z, self.z_init
            )
            err = phase_error(got, ref)
            if err > worst_T:
                worst_T, worst_z = err, z
        print(
            f"\n[Tk sector] k = {self.k:.1e}: friction vs closed form {worst_F:.3e}; theta_T vs "
            f"the exact radiation primitive {worst_T:.3e} rad at z = {worst_z:.6g} over "
            f"{len(zs)} samples; rho_T over the range {payload['metadata']['rho_end']:.6e} rad"
        )
        self.assertLessEqual(worst_F, 1.0e-13)
        self.assertLessEqual(worst_T, RADIATION_TK_TOL)
        # rho_T is not negligible and is carried (review §12.2): 1/x - 1/x_i with x_i = e^3
        self.assertLess(payload["metadata"]["rho_end"], -0.04)

    def test_guards(self):
        with self.assertRaises(ValueError):
            _phase(
                _Proxy(self.model, self.units),
                _KExit(self.k, self.units),
                self.z_init,
                self.response,
                sector="Qk",
                omega_sq=Gk_omegaEff_sq,
                d_ln_omega_dz=Gk_d_ln_omegaEff_dz,
            )
        # a model without the table accessor is refused with a message naming the regeneration
        bare = RadiationModel(RADIATION_H0)
        with self.assertRaises(RuntimeError):
            _run_Gk(bare, self.units, self.k, self.z_init, self.response)


# --------------------------------------------------------------------------------------------
# 7. cost
# --------------------------------------------------------------------------------------------


class TestCost(unittest.TestCase):
    """Prompt 06 §4 item 7 / README §6: <= 0.05 s per object at k = 3e8 on LambdaCDM over the
    full production response grid (the ODE: 63.7 s, 2.5e6 RHS evaluations)."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_wall_time_per_object(self):
        block = self.s.references["models"]["LambdaCDMModel"]
        k_key = "3.000000e+08"
        k = float(k_key)
        z_anchor = float(block["rho_anchor_z"][k_key])
        response = production_response_grid(self.s.grid)
        samples = to_redshift_array(
            [z for z in response.as_float_list() if z <= z_anchor]
        )
        model = self.s.lambdacdm

        best, payload = None, None
        for _ in range(COST_REPEATS):
            start = time.perf_counter()
            payload = _run_Gk(model, model.cosmology.units, k, z_anchor, samples)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)

        meta = payload["metadata"]
        stage_1 = payload["stage_1_data"]
        print(
            f"\n[cost] LambdaCDM k = {k:.1e}, {len(samples)} response samples from z = "
            f"{z_anchor:.4g} to {samples.min.z:.3g}: best of {COST_REPEATS} {best:.4f} s "
            f"(in-call {stage_1.compute_time:.4f} s); {stage_1.RHS_evaluations} integrand "
            f"evaluations = {meta['rho_evals']} residual (order {meta['N_rho']}, "
            f"{meta['rho_nodes']} nodes) + {meta['lead_evals']} leading-table partials; "
            f"threshold {COST_WALL_TIME_LIMIT} s"
        )
        self.assertLessEqual(best, COST_WALL_TIME_LIMIT)
        self.assertGreater(len(samples), 100)


# --------------------------------------------------------------------------------------------
# apply_phase_offset, the §5 greps and the main.py registration
# --------------------------------------------------------------------------------------------


class TestApplyPhaseOffset(unittest.TestCase):
    def test_per_sample_wrap_without_rebase(self):
        div = [-3, -4, -5, 0]
        mod = [-0.5, -6.0, -TWO_PI + 1.0e-12, 0.0]
        for delta in (1.0, -1.0, 0.0, 7.0, -7.5, pi / 2):
            new_div, new_mod = apply_phase_offset(div, mod, delta)
            self.assertEqual(len(new_div), len(div))
            for d, m, d2, m2 in zip(div, mod, new_div, new_mod):
                self.assertIsInstance(d2, int)
                self.assertGreater(m2, -TWO_PI)
                self.assertLessEqual(m2, 0.0)
                self.assertAlmostEqual(
                    d2 * TWO_PI + m2, d * TWO_PI + m + delta, delta=1.0e-12
                )
        # and against the rebased helper it replaces: identical remainders, and divs that differ
        # by the constant shift_theta_sample subtracts (its first sample's wrap shift)
        delta = 1.0
        new_div, new_mod = apply_phase_offset(div, mod, delta)
        old_div, old_mod = shift_theta_sample(div, mod, delta)
        self.assertEqual(list(old_mod), new_mod)
        base = wrap_theta(mod[0] + delta)[0]
        self.assertEqual([d - base for d in new_div], list(old_div))
        self.assertNotEqual(base, 0)


class TestSourceHygiene(unittest.TestCase):
    """Prompt 06 §2 item 6 and §5: the ODE and the dead store logic are gone, and main.py
    registers the new solver label."""

    def _source(self, relative: str) -> str:
        return (REPO_ROOT / relative).read_text()

    def test_phase_function_has_no_ode(self):
        source = self._source("Quadrature/integrators/WKB_phase_function.py")
        for pattern in (
            "solve_ivp",
            "Q_INDEX",
            "DEFAULT_PHASE_RUN_LENGTH",
            "THETA_INDEX",
            "FRICTION_INDEX",
            "stage_1_evolution",
            "stage_2_evolution",
            "ThetaSupervisor",
            "QSupervisor",
            "WKB_product_mod_2pi",
        ):
            self.assertNotIn(pattern, source, msg=pattern)
        self.assertFalse((REPO_ROOT / "Quadrature/supervisors/WKB.py").exists())

    def test_gk_store_has_no_sign_fix_or_rebase(self):
        source = self._source("ComputeTargets/GkWKBIntegration.py")
        for pattern in ("sgn_sin_deltaTheta", "shift_theta_sample", "atol=self._atol"):
            self.assertNotIn(pattern, source, msg=pattern)
        self.assertIn("apply_phase_offset", source)

    def test_tk_compute_uses_the_primitive(self):
        source = self._source("ComputeTargets/TkWKBIntegration.py")
        self.assertIn('sector="Tk"', source)
        self.assertIn("friction=True", source)
        self.assertNotIn("friction=friction_RHS", source)
        self.assertNotIn("atol=self._atol.tol", source)

    def test_main_registers_the_solver_label(self):
        source = self._source("main.py")
        self.assertIn("GkWKBIntegration.PHASE_SOLVER_LABEL_BASE", source)
        self.assertIn("GkWKBIntegration.PHASE_SOLVER_STEPPING", source)
        self.assertRegex(
            source, re.compile(r"GkWKBIntegration\.PHASE_SOLVER_LABEL\s*:")
        )


if __name__ == "__main__":
    unittest.main()
