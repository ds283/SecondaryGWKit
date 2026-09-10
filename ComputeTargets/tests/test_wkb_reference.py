"""
Tests for ``ComputeTargets/tests/wkb_reference.py`` and the cached reference values in
``wkb_reference_data.json`` (Gk/Tk WKB phase remedial campaign, prompt 01).

These are *guard* tests for the measurement infrastructure the rest of the campaign is scored
against: they check that the JSON is complete and self-consistent, that the exact-radiation
closed forms reproduce the independently-quadratured references, that the three error
definitions behave as documented, and that the two real-background stand-ins wire up.

They must stay fast (<10 s) and must not need Ray, a datastore or ``mpmath``: the 40-digit work
lives in ``docs/gktk-remedial/generate_references.py`` and its output is read from the JSON.
"""

import unittest
from math import log1p, fabs
from pathlib import Path

from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    MODEL_KEYS,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    QCDModel,
    RadiationModel,
    REFERENCE_K_VALUES,
    difference_error,
    envelope_relative_error,
    horizon_exit_z,
    phase_error,
    production_response_grid,
    production_source_grid,
    load_references,
)

# the exact-radiation closed forms are compared against a 40-digit mpmath quadrature of the same
# integrand; the residual is the conditioning of the closed form, not the reference
CLOSED_FORM_RTOL = 1.0e-15

# rho_T's closed form is an antiderivative difference in which the two terms of g(s) cancel by a
# factor two, so it carries a few ulps more than the primitives do (measured worst case
# 8.3e-16 at k = 3e8, z = 9.9e6; the primitives reach 1.8e-16, 1.8e-16 and 2.3e-16)
RHO_T_CLOSED_FORM_RTOL = 1.0e-14

# LambdaCDM is radiation-dominated at z = 1e10 to a part in 1e7
LAMBDACDM_EPSILON_TOLERANCE = 1.0e-6

# QCD_Cosmology is NOT: e+e- annihilation makes g_*(T) genuinely vary there, so epsilon really
# does depart from 2 (measured: 1.99796 at z = 1e10, 1.9517 at z = 1e9 -- review §6 quotes a
# departure of up to 0.150 at the QCD transition, z ~ 1.2e12). This is a wiring smoke test, not
# an accuracy test.
QCD_EPSILON_TOLERANCE = 6.0e-2


def _k_key(k: float) -> str:
    return f"{k:.6e}"


class TestReferenceJSON(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_references()

    def test_json_is_complete(self):
        """Every model, checkpoint, quantity and k listed in prompt 01 §2.2 is present."""
        data = self.data
        self.assertEqual(set(data["models"].keys()), set(MODEL_KEYS))
        self.assertEqual(tuple(data["k_values"]), tuple(REFERENCE_K_VALUES))

        for model_key in MODEL_KEYS:
            block = data["models"][model_key]

            with self.subTest(model=model_key):
                self.assertIn("method", block)
                self.assertTrue(len(block["method"]) > 40)
                self.assertIn("reference_floor", block)
                self.assertIn("grid", block)
                self.assertIn("z_top", block)

                checkpoints = block["checkpoints"]
                self.assertGreaterEqual(len(checkpoints), 10)

                # the checkpoints must span the grid: top node, bottom node, and one near each
                # of z = 1e9, 1e6, 1e3, 10, 1
                z_values = [c["z"] for c in checkpoints]
                self.assertEqual(z_values[0], block["grid"]["z_init"])
                self.assertAlmostEqual(z_values[-1], block["grid"]["z_end"], places=12)
                for target in (1e9, 1e6, 1e3, 1e1, 1.0):
                    self.assertTrue(
                        any(0.5 * target < z < 2.0 * target for z in z_values),
                        f"{model_key}: no checkpoint near z = {target:g}",
                    )

                for key in (
                    "tau_minus_top",
                    "cs_tau_minus_top",
                    "friction_F_minus_top",
                ):
                    self.assertEqual(len(block[key]), len(checkpoints), key)

                # sign conventions: tau and cs_tau increase towards lower z, F decreases
                self.assertTrue(all(v >= 0.0 for v in block["tau_minus_top"]))
                self.assertTrue(all(v >= 0.0 for v in block["cs_tau_minus_top"]))
                self.assertTrue(all(v <= 0.0 for v in block["friction_F_minus_top"]))

                for k in REFERENCE_K_VALUES:
                    key = _k_key(k)
                    self.assertIn(key, block["rho_anchor_z"])
                    anchor = block["rho_anchor_z"][key]

                    # the primitives at the (off-grid) anchor, so that a consumer can form the
                    # increment from the production WKB start point without re-quadraturing
                    at_anchor = block["primitives_at_rho_anchor"][key]
                    for name in (
                        "tau_minus_top",
                        "cs_tau_minus_top",
                        "friction_F_minus_top",
                    ):
                        self.assertIn(name, at_anchor)
                    self.assertGreaterEqual(at_anchor["tau_minus_top"], 0.0)
                    self.assertLessEqual(at_anchor["friction_F_minus_top"], 0.0)

                    for sector in ("rho_G", "rho_T"):
                        self.assertIn(key, block[sector])
                        entries = block[sector][key]
                        self.assertGreater(
                            len(entries), 0, f"{model_key}/{sector}/{key} is empty"
                        )
                        for entry in entries:
                            self.assertLess(entry["z"], anchor)
                            self.assertEqual(
                                entry["z"],
                                checkpoints[
                                    [c["index"] for c in checkpoints].index(
                                        entry["index"]
                                    )
                                ]["z"],
                            )

                # three short baselines: one grid interval and its 37% fraction
                short = block["short_baseline"]
                self.assertEqual(len(short), 3)
                for record in short:
                    self.assertLess(record["z_node_lo"], record["z_node_hi"])
                    self.assertLess(record["z_fraction"], record["z_node_hi"])
                    self.assertGreater(record["z_fraction"], record["z_node_lo"])
                    self.assertGreater(record["delta_tau_full"], 0.0)
                    self.assertGreater(record["delta_tau_fraction"], 0.0)
                    self.assertLess(
                        record["delta_tau_fraction"], record["delta_tau_full"]
                    )

    def test_radiation_closed_forms(self):
        """
        The exact-radiation closed forms of §2.1 reproduce the independently-quadratured JSON.
        This is the generator's self-test.
        """
        block = self.data["models"]["RadiationModel"]
        model = RadiationModel(H0=block["H0"])
        z_top = block["z_top"]
        z_values = [c["z"] for c in block["checkpoints"]]

        for z, reference in zip(z_values, block["tau_minus_top"]):
            expected = model.tau_delta(z_top, z)
            if expected == 0.0:
                self.assertEqual(reference, 0.0)
                continue
            self.assertLessEqual(
                difference_error(expected, reference), CLOSED_FORM_RTOL, f"tau at z={z}"
            )

        for z, reference in zip(z_values, block["cs_tau_minus_top"]):
            expected = model.cs_tau_delta(z_top, z)
            if expected == 0.0:
                self.assertEqual(reference, 0.0)
                continue
            self.assertLessEqual(
                difference_error(expected, reference),
                CLOSED_FORM_RTOL,
                f"cs_tau at z={z}",
            )

        for z, reference in zip(z_values, block["friction_F_minus_top"]):
            expected = 2.0 * (log1p(z) - log1p(z_top))
            if expected == 0.0:
                self.assertEqual(fabs(reference), 0.0)
                continue
            self.assertLessEqual(
                difference_error(expected, reference), CLOSED_FORM_RTOL, f"F at z={z}"
            )

        for k in REFERENCE_K_VALUES:
            key = _k_key(k)
            anchor = block["rho_anchor_z"][key]

            # C vanishes identically in exact radiation, so rho_G is exactly zero
            for entry in block["rho_G"][key]:
                self.assertEqual(entry["value"], 0.0, f"rho_G at z={entry['z']}")

            for entry in block["rho_T"][key]:
                expected = model.rho_T(k, entry["z"], anchor)
                self.assertLessEqual(
                    difference_error(expected, entry["value"]),
                    RHO_T_CLOSED_FORM_RTOL,
                    f"rho_T at k={k:g}, z={entry['z']}",
                )

        for record in block["short_baseline"]:
            for key, z_lo in (
                ("delta_tau_full", record["z_node_lo"]),
                ("delta_tau_fraction", record["z_fraction"]),
            ):
                expected = model.tau_delta(record["z_node_hi"], z_lo)
                self.assertLessEqual(
                    difference_error(expected, record[key]),
                    CLOSED_FORM_RTOL,
                    f"{key} at z={record['z_node_hi']}",
                )

    def test_baselines_block(self):
        """
        The two cheap production baselines re-measured by docs/gktk-remedial/baseline_k1e5.py
        must be recorded, and must reproduce review §4 and §12.3 to 5 %.
        """
        baselines = self.data["baselines"]["results"]
        for sector in ("Gk", "Tk"):
            record = baselines[sector]
            self.assertEqual(record["k_inv_Mpc"], 1.0e5)
            self.assertLessEqual(record["relative_difference_from_review"], 0.05)

    def test_reference_module_does_not_import_mpmath(self):
        """
        The reference harness must be importable, and the tests runnable, without mpmath: the
        40-digit work is done once by the generator and cached in the JSON.
        """
        source = Path(Path(__file__).parent / "wkb_reference.py").read_text()
        offenders = [
            line
            for line in source.splitlines()
            if line.strip().startswith(("import mpmath", "from mpmath"))
        ]
        self.assertEqual(offenders, [])


class TestErrorDefinitions(unittest.TestCase):
    """README §6's three error definitions, on constructed inputs."""

    def test_phase_error_is_absolute_and_unwrapped(self):
        self.assertEqual(phase_error(10.0, 10.0), 0.0)
        self.assertAlmostEqual(phase_error(-1.0e9 + 13.9, -1.0e9), 13.9, places=6)
        # symmetric
        self.assertEqual(phase_error(3.0, 5.0), phase_error(5.0, 3.0))
        # NOT reduced mod 2pi: a whole-cycle error is reported in full
        from LiouvilleGreen.constants import TWO_PI

        self.assertAlmostEqual(phase_error(TWO_PI, 0.0), TWO_PI, places=12)

    def test_difference_error_is_relative_to_the_interval(self):
        # a short baseline is scored against its own size, not against the primitive
        self.assertAlmostEqual(difference_error(1.0e-6, 1.0e-6), 0.0)
        self.assertAlmostEqual(difference_error(1.01e-6, 1.0e-6), 0.01, places=12)
        # sign of the reference does not matter
        self.assertAlmostEqual(difference_error(-1.01e-6, -1.0e-6), 0.01, places=12)
        with self.assertRaises(ValueError):
            difference_error(1.0, 0.0)

    def test_envelope_relative_error(self):
        self.assertAlmostEqual(
            envelope_relative_error(1.5, 1.0, 100.0), 0.005, places=12
        )
        # a zero crossing of the value does not blow up
        self.assertAlmostEqual(
            envelope_relative_error(1.0e-9, 0.0, 1.0e-3), 1.0e-6, places=15
        )
        with self.assertRaises(ValueError):
            envelope_relative_error(1.0, 1.0, 0.0)


class TestStandInModels(unittest.TestCase):
    """Smoke test: the two real-background stand-ins construct and are wired up correctly."""

    @classmethod
    def setUpClass(cls):
        cls.lam = LambdaCDMModel()
        cls.z_init = horizon_exit_z(
            cls.lam.cosmology,
            PRODUCTION_LARGEST_K_INV_MPC,
            -PRODUCTION_SUPERHORIZON_EFOLDS,
        )
        cls.grid = production_source_grid(cls.z_init)
        cls.qcd = QCDModel(cls.grid)

    def test_production_grid_shape(self):
        grid = self.grid
        # descending, 100 per decade of z, from the 5-e-fold super-horizon point of k = 3e8
        self.assertGreater(len(grid), 1500)
        # np.logspace(log10(z_init), ...) does not round-trip z_init exactly; production has
        # the same property (CosmologyConcepts/wavenumber.py:292)
        self.assertLessEqual(difference_error(grid.max.z, self.z_init), 1.0e-15)
        self.assertAlmostEqual(grid.min.z, 0.1, places=12)
        self.assertGreater(grid[0].z, grid[1].z)

        response = production_response_grid(grid)
        self.assertAlmostEqual(len(response), len(grid) / 12.0, delta=1.0)
        # the response grid is a subset of the source grid (main.py:422)
        source_z = {z.z for z in grid}
        self.assertTrue(all(z.z in source_z for z in response))

    def test_epsilon_in_the_radiation_era(self):
        self.assertLessEqual(
            fabs(self.lam.functions.epsilon(1.0e10) - 2.0),
            LAMBDACDM_EPSILON_TOLERANCE,
        )
        self.assertLessEqual(
            fabs(self.qcd.functions.epsilon(1.0e10) - 2.0), QCD_EPSILON_TOLERANCE
        )

    def test_radiation_model_is_self_consistent(self):
        model = RadiationModel()
        z = 1234.5
        self.assertAlmostEqual(model.functions.Hubble(z), (1.0 + z) ** 2, places=6)
        self.assertEqual(model.functions.epsilon(z), 2.0)
        self.assertAlmostEqual(model.tau(z), 1.0 / (1.0 + z), places=15)
        # theta_G = k(1/s_i - 1/s) is negative for z < z_init
        self.assertLess(model.theta_G(1.0e5, 100.0, 10000.0), 0.0)
        self.assertEqual(model.rho_G(1.0e5, 100.0, 10000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
