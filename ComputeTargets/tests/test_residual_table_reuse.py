"""
Tests for the per-wavenumber reuse of the Liouville-Green phase residual
(``prompts/GkTk-remedial`` prompt 14 §3.2): ``ComputeTargets.phase_residual.residual_node_range``
and ``cached_phase_residual``, and the way ``Quadrature/integrators/WKB_phase_function.py``
consumes them.

Prompt 06 built the residual table on every call, anchored on the object's own ``z_init``: 5,536
of the 6,000 integrand evaluations and ~92 % of the 0.031 s of an object at ``k = 3e8`` on
LambdaCDM, repeated identically for each of the ~1,700 source redshifts of that wavenumber
(``[06-residual-table-per-object]``). ``rho`` depends only on ``(model, k, sector)``, so the
table is now built on a node range that depends on nothing else -- the background grid, cut at
the top where the frequency stops keeping ``RESIDUAL_WKB_REGION_MARGIN`` of its leading term --
and memoised on that key; the object's anchor is reached through ``CumulativeTable.delta``'s
off-grid partial, split off once per object at the nearest node.

This is a **cost** change. The five items below are, in order: the table is shared; the key
discriminates (including two offline stand-ins, which have no datastore id); the answer did not
move; the exact-radiation control survives; and the amortised cost.

Stand-ins and fixtures are prompt 06's, imported from ``test_gk_wkb_phase``. No Ray, no
datastore.
"""

import json
import unittest
from math import expm1, fabs, log1p
from typing import Optional

import numpy as np

from ComputeTargets.WKB_Gk import Gk_d_ln_omegaEff_dz, Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq
from ComputeTargets.phase_residual import (
    RHO_GAUSS_ORDER,
    build_phase_residual,
    cached_phase_residual,
    clear_phase_residual_cache,
    phase_residual_cache_size,
    residual_node_range,
)
from ComputeTargets.tests.test_gk_wkb_phase import (
    OFF_GRID_FRACTION,
    RADIATION_H0,
    _KExit,
    _Proxy,
    _radiation_z_e3,
    lambdacdm_model_with_tables,
    qcd_model_with_tables,
    radiation_model_with_tables,
)
from ComputeTargets.tests.wkb_reference import (
    PRODUCTION_Z_END,
    load_references,
    production_response_grid,
    production_source_grid,
    production_source_z_values,
    to_redshift_array,
)
from LiouvilleGreen.constants import TWO_PI
from Quadrature.integrators.WKB_phase_function import (
    SECTOR_LEADING_PRIMITIVE,
    WKB_phase_function,
    nearest_table_node,
)
from Units import Mpc_units
from config.defaults import DEFAULT_STRING_LENGTH

_phase = WKB_phase_function._function

# item 3: the shared table must reproduce the per-object table's phase to this absolute accuracy.
# Two orders below README §6's rho acceptance (1e-6 rad) and far below the eps*k*tau floor
# (3e-7 rad at k = 1e5, 9e-4 rad at 3e8).
ANCHORING_TOL = 1.0e-9

# item 3: the wavenumbers scored on each model
REUSE_K_VALUES = (1.0e5, 1.0e7, 3.0e8)
REFERENCE_K_KEYS = ("1.000000e+05", "1.000000e+07", "3.000000e+08")

# item 5: how many objects of one wavenumber the amortised cost is measured over
COST_OBJECT_COUNT = 50

_SECTOR_DIAGNOSTICS = {
    "Gk": (Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz),
    "Tk": (Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz),
}


class _ProxyWithStoreId(_Proxy):
    """A ``ModelProxy`` stand-in that also carries a datastore id, as the production proxy does
    (``BackgroundModel.ModelProxy.store_id``; ``None`` when the model is unavailable).
    """

    def __init__(self, model, units, store_id: Optional[int]):
        super().__init__(model, units)
        self.store_id = store_id


def _run(model, units, k, z_init, z_sample, sector="Gk", store_id=None) -> dict:
    omega_sq, d_ln_omega_dz = _SECTOR_DIAGNOSTICS[sector]
    return _phase(
        _ProxyWithStoreId(model, units, store_id),
        _KExit(k, units),
        z_init,
        z_sample,
        sector=sector,
        omega_sq=omega_sq,
        d_ln_omega_dz=d_ln_omega_dz,
        friction=False,
        task_label="test_residual_table_reuse",
        object_label="test",
    )


def prompt_06_residual_nodes(grid, z_sample, z_init) -> np.ndarray:
    """Prompt 06's per-object node rule, kept here as the control item 3 scores against: the
    background grid restricted to ``[min(z_sample), z_init]``, plus every sample, plus ``z_init``
    itself as the top node (``WKB_phase_function.residual_nodes`` before prompt 14)."""
    grid = np.asarray(grid, dtype=float)
    z_init = float(z_init)
    z_min = min(float(z) for z in z_sample)
    inside = grid[(grid >= z_min) & (grid <= z_init)]
    nodes = set(inside.tolist())
    nodes.update(float(z) for z in z_sample)
    nodes.add(z_init)
    return np.array(sorted(nodes, reverse=True), dtype=float)


def off_grid_z(z_hi: float, z_lo: float, fraction: float = OFF_GRID_FRACTION) -> float:
    """A redshift ``fraction`` of the way through the interval ``[z_lo, z_hi]`` in
    ``u = log(1+z)``, as a numeric hand-over's ``z_init`` is."""
    width = log1p((z_hi - z_lo) / (1.0 + z_lo))
    return z_lo + (1.0 + z_lo) * expm1(fraction * width)


class _Shared:
    """The production grid and the three models, built once for the module."""

    references = None
    grid = None
    z_nodes = None
    lambdacdm = None
    qcd = None

    @classmethod
    def build(cls):
        if cls.references is None:
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
    def anchor(cls, model_key: str, k_key: str) -> float:
        return float(cls.references["models"][model_key]["rho_anchor_z"][k_key])


# ---------------------------------------------------------------------------------------------
# 1. the table is shared between objects of one (model, k, sector)
# ---------------------------------------------------------------------------------------------


class TestTableIsShared(unittest.TestCase):
    """Prompt 14 §3.2 item 1."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def test_objects_with_different_anchors_and_samples_share_one_table(self):
        k = 1.0e7
        model = self.s.lambdacdm
        units = model.cosmology.units
        z_nodes = self.s.z_nodes
        anchor = self.s.anchor("LambdaCDMModel", "1.000000e+07")

        # the table this (model, k, sector) resolves to, and its node range
        nodes = residual_node_range(model, k, z_nodes, "Gk")
        table, reused = cached_phase_residual(model, k, z_nodes, "Gk")
        self.assertFalse(reused)
        self.assertEqual(table.z_nodes.tolist(), nodes.tolist())
        build_evaluations = table.evaluations

        # four objects: an on-grid anchor, an anchor 37 % through the interval below it, a
        # much lower anchor, and one with a different (shorter) sample set
        on_grid = [z for z in z_nodes if z <= anchor]
        anchors = [
            float(on_grid[0]),
            off_grid_z(float(on_grid[0]), float(on_grid[1])),
            off_grid_z(float(on_grid[40]), float(on_grid[41])),
            float(on_grid[80]),
        ]
        sample_sets = [
            to_redshift_array([z for z in on_grid if z <= a][:: (i + 1) * 7 + 1])
            for i, a in enumerate(anchors)
        ]

        payloads = []
        for a, samples in zip(anchors, sample_sets):
            payloads.append(_run(model, units, k, a, samples))
            # every call resolves to the same table *object*
            again, _ = cached_phase_residual(model, k, z_nodes, "Gk")
            self.assertIs(again, table)

        self.assertEqual(phase_residual_cache_size(), 1)
        # the table was built once: no call added a build evaluation
        self.assertEqual(table.evaluations, build_evaluations)

        # ... and every one of the four objects reports the reuse and spends no build evaluation
        for payload, a in zip(payloads, anchors):
            meta = payload["metadata"]
            self.assertTrue(meta["rho_reused"], msg=f"z_init={a}")
            self.assertEqual(meta["rho_nodes"], len(nodes))
            on_grid_anchor = table.node_index(a) is not None
            if on_grid_anchor:
                # an on-grid anchor needs no partial at all
                self.assertEqual(meta["rho_evals"], 0, msg=f"z_init={a}")
            else:
                self.assertGreater(meta["rho_evals"], 0, msg=f"z_init={a}")
                self.assertLessEqual(
                    meta["rho_evals"], 4 * RHO_GAUSS_ORDER, msg=f"z_init={a}"
                )
            self.assertEqual(
                payload["stage_1_data"].RHS_evaluations,
                meta["rho_evals"] + meta["lead_evals"],
            )

        print(
            f"\n[shared table] LambdaCDM k = {k:.1e}, sector Gk: one table of {len(nodes)} "
            f"nodes and {build_evaluations} integrand evaluations serves {len(anchors)} objects "
            f"with anchors {', '.join(f'{a:.6g}' for a in anchors)} and "
            f"{', '.join(str(len(s)) for s in sample_sets)} samples; their rho_evals are "
            f"{[p['metadata']['rho_evals'] for p in payloads]} (the anchor partials alone)"
        )

    def test_metadata_still_fits_the_column(self):
        """``[06-metadata-column-headroom]``: the payload is persisted as ``json.dumps`` into a
        ``String(DEFAULT_STRING_LENGTH)``. The new ``rho_reused`` key must not overflow it.
        """
        k = 3.0e8
        model = self.s.lambdacdm
        units = model.cosmology.units
        anchor = self.s.anchor("LambdaCDMModel", "3.000000e+08")
        samples = to_redshift_array([z for z in self.s.z_nodes if z <= anchor])

        worst, worst_label = 0, None
        for sector in ("Gk", "Tk"):
            for call in (1, 2):
                payload = _run(model, units, k, anchor, samples, sector=sector)
                length = len(json.dumps(payload["metadata"]))
                if length > worst:
                    worst, worst_label = length, f"{sector}, call {call}"
        print(
            f"\n[metadata] longest JSON payload {worst} characters ({worst_label}) against the "
            f"String({DEFAULT_STRING_LENGTH}) column; prompt 06 measured 206 without "
            f"'rho_reused'"
        )
        self.assertLessEqual(worst, DEFAULT_STRING_LENGTH)


# ---------------------------------------------------------------------------------------------
# 2. the key discriminates
# ---------------------------------------------------------------------------------------------


class TestKeyDiscriminates(unittest.TestCase):
    """Prompt 14 §3.2 item 2."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def test_wavenumber_and_sector_each_force_a_rebuild(self):
        model = self.s.lambdacdm
        z_nodes = self.s.z_nodes

        first, reused = cached_phase_residual(model, 1.0e7, z_nodes, "Gk")
        self.assertFalse(reused)

        same, reused = cached_phase_residual(model, 1.0e7, z_nodes, "Gk")
        self.assertTrue(reused)
        self.assertIs(same, first)

        other_k, reused = cached_phase_residual(model, 2.0e7, z_nodes, "Gk")
        self.assertFalse(reused)
        self.assertIsNot(other_k, first)

        other_sector, reused = cached_phase_residual(model, 1.0e7, z_nodes, "Tk")
        self.assertFalse(reused)
        self.assertIsNot(other_sector, first)
        # the Tk table is the shorter one: its frequency turns over inside the horizon
        self.assertLess(len(other_sector), len(first))

        self.assertEqual(phase_residual_cache_size(), 3)

    def test_two_offline_models_with_no_store_id_do_not_share(self):
        """``ModelProxy.store_id`` is ``None`` for an unavailable model
        (``BackgroundModel.py:963``) and for every offline stand-in, so ``None`` cannot be the
        key: identity takes its place."""
        k = 1.0e7
        z_e3 = _radiation_z_e3(k)
        z_nodes = production_source_z_values(z_e3, PRODUCTION_Z_END)

        first = radiation_model_with_tables(z_nodes, RADIATION_H0)
        second = radiation_model_with_tables(z_nodes, 2.0 * RADIATION_H0)
        self.assertIsNot(first, second)

        table_1, reused = cached_phase_residual(first, k, z_nodes, "Tk", store_id=None)
        self.assertFalse(reused)
        table_2, reused = cached_phase_residual(second, k, z_nodes, "Tk", store_id=None)
        self.assertFalse(
            reused, msg="a second offline model must not inherit the first's table"
        )
        self.assertIsNot(table_2, table_1)
        self.assertEqual(phase_residual_cache_size(), 2)

        # each model gets its own back
        again_1, reused = cached_phase_residual(first, k, z_nodes, "Tk", store_id=None)
        self.assertTrue(reused)
        self.assertIs(again_1, table_1)
        again_2, reused = cached_phase_residual(second, k, z_nodes, "Tk", store_id=None)
        self.assertTrue(reused)
        self.assertIs(again_2, table_2)

        # the two are different tables, not merely different objects: H0 differs by 2, so the
        # sound horizon and the residual do too
        rho_1 = table_1.delta(float(z_nodes[0]), float(z_nodes[-1]))
        rho_2 = table_2.delta(float(z_nodes[0]), float(z_nodes[-1]))
        print(
            f"\n[key] two offline stand-ins (store_id=None) at k = {k:.1e}: separate tables, "
            f"rho_T over the range {rho_1:.6e} and {rho_2:.6e} rad"
        )
        self.assertNotEqual(rho_1, rho_2)

    def test_different_models_do_not_share_through_the_store_id(self):
        """Two models with different datastore ids are different keys; the same id is the same
        model, which is what the id means."""
        k = 1.0e7
        _Shared.build_qcd()
        z_nodes = self.s.z_nodes

        lam, reused = cached_phase_residual(
            self.s.lambdacdm, k, z_nodes, "Gk", store_id=11
        )
        self.assertFalse(reused)
        qcd, reused = cached_phase_residual(self.s.qcd, k, z_nodes, "Gk", store_id=12)
        self.assertFalse(reused)
        self.assertIsNot(qcd, lam)
        self.assertNotEqual(len(qcd), len(lam))

        again, reused = cached_phase_residual(
            self.s.lambdacdm, k, z_nodes, "Gk", store_id=11
        )
        self.assertTrue(reused)
        self.assertIs(again, lam)
        self.assertEqual(phase_residual_cache_size(), 2)


# ---------------------------------------------------------------------------------------------
# 3. the answer did not move
# ---------------------------------------------------------------------------------------------


class TestAnchoringIsFree(unittest.TestCase):
    """Prompt 14 §3.2 item 3: the phase from the shared table against the phase from a table
    built prompt 06's way, anchored at the object's own ``z_init``.

    Both are formed unreduced, as doubles, because the payload's ``div*2pi + mod`` carries the
    ``eps*k*tau`` representation floor -- 9.8e-4 rad at ``k = 3e8`` -- which would swamp the
    difference being measured. That the production path really computes the unreduced phase
    scored here is asserted separately against the payload, at its own floor."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        _Shared.build_qcd()
        cls.s = _Shared

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def _compare(self, model, units, grid, k, z_init, samples, sector="Gk"):
        leading = getattr(model.functions, SECTOR_LEADING_PRIMITIVE[sector])

        shared, _ = cached_phase_residual(model, k, grid, sector)
        node = nearest_table_node(shared, z_init)
        anchor_partial = shared.delta(z_init, node)

        old = build_phase_residual(
            model, k, prompt_06_residual_nodes(grid, samples, z_init), sector
        )

        payload = _run(
            model, units, k, z_init, to_redshift_array(samples), sector=sector
        )
        stored = [
            d * TWO_PI + m
            for d, m in zip(
                payload["theta_div_2pi_sample"], payload["theta_mod_2pi_sample"]
            )
        ]

        worst, worst_z, worst_stored = 0.0, samples[0], 0.0
        worst_rho, worst_rho_z = 0.0, samples[0]
        for z, reconstructed in zip(samples, stored):
            lead_term = k * leading.delta(z_init, z)
            rho_new = anchor_partial + shared.delta(node, z)
            rho_old = old.delta(z_init, z)
            theta_new = -(lead_term + rho_new)
            theta_old = -(lead_term + rho_old)
            err = fabs(theta_new - theta_old)
            if err > worst:
                worst, worst_z = err, z
            err_rho = fabs(rho_new - rho_old)
            if err_rho > worst_rho:
                worst_rho, worst_rho_z = err_rho, z
            worst_stored = max(worst_stored, fabs(reconstructed - theta_new))

        # the payload is the phase scored here, to the representation floor of div*2pi + mod
        floor = 8.0 * np.finfo(float).eps * max(fabs(t) for t in stored)
        self.assertLessEqual(worst_stored, max(floor, 1.0e-12))
        return worst, worst_z, worst_rho, worst_rho_z

    def test_phase_is_unchanged_on_all_three_models(self):
        cases = []

        # exact radiation: a fresh grid per k (the 3-e-fold point moves with k), off-grid anchor
        for k in REUSE_K_VALUES:
            grid = production_source_z_values(_radiation_z_e3(k), PRODUCTION_Z_END)
            model = radiation_model_with_tables(grid, RADIATION_H0)
            z_init = off_grid_z(float(grid[0]), float(grid[1]))
            samples = [
                z
                for z in production_response_grid(
                    to_redshift_array(grid)
                ).as_float_list()
                if z <= z_init
            ]
            cases.append(
                ("RadiationModel", model, Mpc_units(), grid, k, z_init, samples)
            )

        response = production_response_grid(self.s.grid).as_float_list()
        for model_key, model in (
            ("LambdaCDMModel", self.s.lambdacdm),
            ("QCDModel", self.s.qcd),
        ):
            for k, k_key in zip(REUSE_K_VALUES, REFERENCE_K_KEYS):
                z_init = self.s.anchor(model_key, k_key)
                samples = [z for z in response if z <= z_init]
                cases.append(
                    (
                        model_key,
                        model,
                        model.cosmology.units,
                        self.s.z_nodes,
                        k,
                        z_init,
                        samples,
                    )
                )

        worst, worst_case = 0.0, None
        worst_rho, worst_rho_case = 0.0, None
        for name, model, units, grid, k, z_init, samples in cases:
            err, err_z, err_rho, err_rho_z = self._compare(
                model, units, grid, k, z_init, samples
            )
            print(
                f"[anchoring] {name} k = {k:.3e}: {len(samples)} samples from z_init = "
                f"{z_init:.6g}; max |theta_shared - theta_per_object| = {err:.3e} rad at "
                f"z = {err_z:.6g} (in rho alone {err_rho:.3e} rad at z = {err_rho_z:.6g})"
            )
            if err > worst:
                worst, worst_case = err, (name, f"{k:.3e}", err_z)
            if err_rho > worst_rho:
                worst_rho, worst_rho_case = err_rho, (name, f"{k:.3e}", err_rho_z)
        print(
            f"[anchoring] worst over {len(cases)} (model, k) cases: theta {worst:.3e} rad at "
            f"{worst_case}, rho {worst_rho:.3e} rad at {worst_rho_case}; threshold "
            f"{ANCHORING_TOL:g}"
        )
        self.assertLessEqual(worst, ANCHORING_TOL)
        self.assertLessEqual(worst_rho, ANCHORING_TOL)

    def test_phase_is_unchanged_in_the_transfer_function_sector(self):
        """The ``Tk`` sector carries ``rho_T ~ -0.09`` rad, six orders larger than ``rho_G``, so
        it is the sector where an anchoring change would show."""
        response = production_response_grid(self.s.grid).as_float_list()
        worst, worst_case = 0.0, None
        worst_rho, worst_rho_case = 0.0, None
        for model_key, model in (
            ("LambdaCDMModel", self.s.lambdacdm),
            ("QCDModel", self.s.qcd),
        ):
            for k, k_key in zip(REUSE_K_VALUES, REFERENCE_K_KEYS):
                z_init = self.s.anchor(model_key, k_key)
                samples = [z for z in response if z <= z_init]
                err, err_z, err_rho, err_rho_z = self._compare(
                    model,
                    model.cosmology.units,
                    self.s.z_nodes,
                    k,
                    z_init,
                    samples,
                    sector="Tk",
                )
                if err > worst:
                    worst, worst_case = err, (model_key, f"{k:.3e}", err_z)
                if err_rho > worst_rho:
                    worst_rho, worst_rho_case = err_rho, (
                        model_key,
                        f"{k:.3e}",
                        err_rho_z,
                    )
        print(
            f"\n[anchoring, Tk] worst over six (model, k) cases: theta {worst:.3e} rad at "
            f"{worst_case}, rho {worst_rho:.3e} rad at {worst_rho_case}; threshold "
            f"{ANCHORING_TOL:g}"
        )
        self.assertLessEqual(worst, ANCHORING_TOL)
        self.assertLessEqual(worst_rho, ANCHORING_TOL)


# ---------------------------------------------------------------------------------------------
# 4. the exact-radiation control survives
# ---------------------------------------------------------------------------------------------


class TestRadiationControlSurvives(unittest.TestCase):
    """Prompt 14 §3.2 item 4."""

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def test_rho_G_is_bit_exactly_zero_through_the_shared_table(self):
        k = 1.0e7
        grid = production_source_z_values(_radiation_z_e3(k), PRODUCTION_Z_END)
        model = radiation_model_with_tables(grid, RADIATION_H0)
        units = Mpc_units()

        table, _ = cached_phase_residual(model, k, grid, "Gk")
        # C = (3 eps/2 - eps^2/4 - 2)/s^2 = 0 identically in exact radiation
        self.assertTrue(np.all(table.hi == 0.0))
        self.assertTrue(np.all(table.lo == 0.0))
        # the range is the whole grid: omega^2 = (k/H)^2 keeps all of its leading term
        self.assertEqual(table.z_nodes.tolist(), list(grid))

        response = production_response_grid(to_redshift_array(grid))
        for z_init in (
            float(grid[0]),
            off_grid_z(float(grid[0]), float(grid[1])),
        ):
            samples = to_redshift_array(
                [z for z in response.as_float_list() if z <= z_init]
            )
            payload = _run(model, units, k, z_init, samples)
            self.assertEqual(payload["metadata"]["rho_end"], 0.0)
            # the off-grid anchor's partial is exactly zero too, and costs nothing but the
            # integrand calls that return zero
            node = nearest_table_node(table, z_init)
            self.assertEqual(table.delta(z_init, node), 0.0)
            # the phase is k Delta tau and nothing else
            worst = 0.0
            for z, d, m in zip(
                samples.as_float_list(),
                payload["theta_div_2pi_sample"],
                payload["theta_mod_2pi_sample"],
            ):
                worst = max(
                    worst, fabs(d * TWO_PI + m - (-k * model.tau_delta(z_init, z)))
                )
            print(
                f"\n[radiation] k = {k:.1e}, z_init = {z_init:.10g}: rho_G is bit-exactly zero "
                f"through the shared table; max |theta + k tau.delta| = {worst:.3e} rad"
            )
            self.assertLessEqual(worst, 1.0e-8)


# ---------------------------------------------------------------------------------------------
# 5. cost
# ---------------------------------------------------------------------------------------------


class TestAmortisedCost(unittest.TestCase):
    """Prompt 14 §3.2 item 5: the integrand evaluations per object over ``COST_OBJECT_COUNT``
    objects of one wavenumber, on both production models."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        _Shared.build_qcd()
        cls.s = _Shared

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def _measure(self, name: str, model, k: float):
        units = model.cosmology.units
        z_nodes = self.s.z_nodes
        anchor = self.s.anchor(name, f"{k:.6e}")
        on_grid = [z for z in z_nodes if z <= anchor]
        response = [
            z
            for z in production_response_grid(self.s.grid).as_float_list()
            if z <= anchor
        ]

        # COST_OBJECT_COUNT objects of one k, each with its own off-grid anchor, as the numeric
        # hand-over gives them
        anchors = [
            off_grid_z(float(on_grid[i]), float(on_grid[i + 1]))
            for i in range(COST_OBJECT_COUNT)
        ]

        rho_evals = []
        build_evaluations = None
        for i, a in enumerate(anchors):
            payload = _run(
                model,
                units,
                k,
                a,
                to_redshift_array([z for z in response if z <= a]),
            )
            meta = payload["metadata"]
            rho_evals.append(meta["rho_evals"])
            table, _ = cached_phase_residual(model, k, z_nodes, "Gk")
            if i == 0:
                self.assertFalse(meta["rho_reused"])
                build_evaluations = table.evaluations
                # the first object pays the build plus its own anchor partial
                self.assertGreaterEqual(meta["rho_evals"], build_evaluations)
                self.assertLessEqual(
                    meta["rho_evals"], build_evaluations + 4 * RHO_GAUSS_ORDER
                )
            else:
                # the second and every subsequent object spends no table-build evaluation
                self.assertTrue(meta["rho_reused"])
                self.assertEqual(table.evaluations, build_evaluations)
                self.assertLessEqual(meta["rho_evals"], 4 * RHO_GAUSS_ORDER)

        amortised = sum(rho_evals) / float(COST_OBJECT_COUNT)
        per_object_partial = rho_evals[1]
        print(
            f"\n[cost] {name} k = {k:.1e}, {COST_OBJECT_COUNT} objects of one wavenumber: "
            f"table of {len(table)} nodes built once for {build_evaluations} integrand "
            f"evaluations; every later object spends {per_object_partial} (its anchor partial) "
            f"and no build evaluation. Amortised {amortised:.1f} residual-integrand evaluations "
            f"per object, against {build_evaluations} per object before prompt 14 "
            f"({build_evaluations / amortised:.0f}x)"
        )
        self.assertLess(amortised, build_evaluations / 10.0)
        return amortised, build_evaluations

    def test_lambdacdm(self):
        self._measure("LambdaCDMModel", self.s.lambdacdm, 3.0e8)

    def test_qcd(self):
        self._measure("QCDModel", self.s.qcd, 3.0e8)


if __name__ == "__main__":
    unittest.main()
