"""
Tests for ``ComputeTargets/phase_residual.py`` (``prompts/GkTk-remedial``, prompt 05 §2).

The residual is the whole of the Liouville-Green phase that is *not* ``k`` times a shared
primitive:

    theta(z; z_i) = -[ k tau.delta(z_i, z) + rho.delta(z_i, z) ],
    rho.delta(z_i, z) = int_z^{z_i} C/(omega + omega_0) dz

with ``omega_0 = k/H`` and the ``tau`` table in the ``Gk`` sector, ``omega_0 = k c_s/H`` and the
``cs_tau`` table in the ``Tk`` sector (review §6, §12.2; README §2 (a), (c)).

Four things are checked, in the order prompt 05 §2 lists them.

1. **Radiation controls.** ``rho_G`` is bit-exactly zero, because ``C`` is
   ``(3 eps/2 - eps^2/4 - 2)/s^2 = 0`` with no rounding when ``eps = 2``. ``rho_T`` is not:
   it is scored against ``RadiationModel``'s exact closed-form primitive.
2. **Against prompt 01's references**, both production models, both sectors, three ``k``, at the
   JSON checkpoints, anchored at each ``k``'s 3-e-fold sub-horizon redshift (which is *off* the
   node grid, so ``delta`` must form the partial panel above the top node).
3. **Size sanity**, reproducing the review's §6 and §12.2 tables, so that a sign or factor slip
   in the integrand cannot pass.
4. **Cost**: integrand evaluations and wall time per table.

**A note on the sign of the radiation control.** Prompt 05 §2 quotes review §12.4's
``rho_T = 1/x_i - 1/x``. That is the review's convention (its ``rho`` is ``theta - (x_i - x)``);
in the campaign's convention, fixed by README §2 (a), (c) and used by ``wkb_reference`` and by
the ``rho_T`` block of ``wkb_reference_data.json``, the same quantity is ``1/x - 1/x_i``,
negative for ``x > x_i``. The magnitude is the review's (0.0416 rad from ``x_i = 24``), and
``test_rho_T_matches_the_review_asymptote`` records both.

The reference is the **exact** primitive, not the ``1/x - 1/x_i`` asymptote: the asymptote is
itself in error by ``2/(3 x^3)``, i.e. 2.9e-4 relative from ``x_i = 24``, which is eleven orders
above the 1e-13 absolute accuracy the prompt asks the table to reach. Prompt 02's convergence
block scores the same way (``convergence.radiation_controls.rho_T.*.max_rel_error_vs_closed_form``
alongside ``asymptote_relative_departure``).

No Ray and no datastore; ``QCDModel`` is built once for the module.
"""

import time
import unittest
from math import fabs, sqrt

import numpy as np

from ComputeTargets.phase_residual import (
    RHO_ADAPTIVE_FALLBACK_REQUIRED,
    RHO_GAUSS_ORDER,
    build_phase_residual,
    phase_residual_integrand,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    QCDModel,
    RadiationModel,
    REFERENCE_K_VALUES,
    load_references,
    production_source_grid,
)

# prompt 05 §2 test 2 / README §6: the residual target is 1e-6 rad; prompt 02 chose N_rho = 4 to
# leave a decade of margin, so this asserts 1e-7
RHO_REFERENCE_ABS_TOL = 1.0e-7

# prompt 05 §2 test 1: the exact-radiation control for rho_T, over x in [24, 1e4]
RADIATION_ABS_TOL = 1.0e-13
RADIATION_K_INV_MPC = 1.0e5
RADIATION_X_LO = 24.0
RADIATION_X_HI = 1.0e4
RADIATION_SAMPLES_PER_LOG10Z = 100

# prompt 05 §2 test 3: the size windows that protect against a sign or factor slip (review §6
# and §12.2). |rho_G| on LambdaCDM at k = 1e5 is 2.6e-7 rad; on QCD at k = 3e8 it is 1.2e-3;
# rho_T is -0.086 to -0.093 on both models at every k.
LAMBDACDM_RHO_G_MAX = 3.0e-7
QCD_RHO_G_MIN = 5.0e-4
QCD_RHO_G_MAX = 3.0e-3
RHO_T_MIN = -0.12
RHO_T_MAX = -0.06

# prompt 05 §2 test 4: with no break points the build costs exactly RHO_GAUSS_ORDER evaluations
# per interval; QCD_Cosmology's break-point subdivision (log 02) costs 24 % more.
#
# Loosened 1.30 -> 2.40 by prompts/qcd-background-audit/ prompt 06, measured 2.337. This is a
# real cost, not a stale figure: the T(z) tabulation went from 500 nodes to 3,000 (segmenting the
# representation at the jumps is what buys the accuracy, but the node count is what buys the p90
# and the median), every interior knot of it is declared by integration_break_points, and every
# Gauss panel is split at each one. BREAK_POINT_ALL on the production source grid is 2,414 where
# it was 407, so a panel is now split roughly every 0.7 grid intervals rather than every 4.
# **Prompt 07 removes the knots from that set entirely** -- they are an artefact of the
# representation and not of the cosmology, which is finding G1 of the audit -- and this factor
# should then come back below its original 1.30 rather than merely to it, because what will be
# left is three genuine crossings ([05-break-point-set-grew-with-the-node-count]).
COST_BREAK_POINT_FACTOR = 2.40


def _nodes_at_or_below(z_nodes: np.ndarray, z_anchor: float) -> np.ndarray:
    """
    The production nodes inside the WKB region for this ``k``.

    The anchor itself is a ``root_scalar`` root of ``k(1+z)/H(z) = e^3`` and is never a node
    (``RECONCILIATION.md`` §2 item 5), so it sits less than one grid interval above the top node
    returned here and is reached by ``CumulativeTable.delta``'s off-grid partial. Nodes above it
    are excluded because ``omega^2`` is not positive there for the smaller wavenumbers.
    """
    return z_nodes[z_nodes <= z_anchor]


class _Shared:
    """The references, the production grid, both production models, built once."""

    references = None
    z_nodes = None
    models = None
    radiation = None
    radiation_z = None
    build_cost = None

    @classmethod
    def build(cls):
        if cls.references is not None:
            return
        cls.references = load_references()

        grid = production_source_grid(
            cls.references["models"]["LambdaCDMModel"]["grid"]["z_init"]
        )
        cls.z_nodes = np.array(grid.as_float_list(), dtype=float)

        cls.models = {
            "LambdaCDMModel": LambdaCDMModel(),
            "QCDModel": QCDModel(grid),
        }

        cls.radiation = RadiationModel()
        # x = k c_s tau = k/(sqrt(3) H0 (1+z)) in exact radiation, so the x window fixes the grid
        H0 = cls.radiation.H0
        z_of_x = lambda x: RADIATION_K_INV_MPC / (sqrt(3.0) * H0 * x) - 1.0
        z_top = z_of_x(RADIATION_X_LO)
        z_end = z_of_x(RADIATION_X_HI)
        num = (
            int(
                round(
                    RADIATION_SAMPLES_PER_LOG10Z * (np.log10(z_top) - np.log10(z_end))
                )
            )
            + 1
        )
        cls.radiation_z = np.logspace(np.log10(z_top), np.log10(z_end), num=int(num))

        # one table per (model, k, sector), with its build cost
        cls.build_cost = {}
        for model_key, model in cls.models.items():
            block = cls.references["models"][model_key]
            for k_key, z_anchor in block["rho_anchor_z"].items():
                nodes = _nodes_at_or_below(cls.z_nodes, float(z_anchor))
                for sector in ("Gk", "Tk"):
                    start = time.perf_counter()
                    table = build_phase_residual(model, float(k_key), nodes, sector)
                    seconds = time.perf_counter() - start
                    cls.build_cost[(model_key, k_key, sector)] = (
                        table,
                        len(nodes),
                        seconds,
                    )

    @classmethod
    def table(cls, model_key, k_key, sector):
        return cls.build_cost[(model_key, k_key, sector)][0]


class TestRadiationControls(unittest.TestCase):
    """Prompt 05 §2 test 1."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_rho_G_is_bit_exactly_zero_in_radiation(self):
        """
        ``C`` vanishes identically in exact radiation, so every panel integrates ``0.0`` and
        every node of the table -- both limbs -- is exactly zero. This is the strongest form of
        review §6's statement that the Green's-function phase in radiation is ``k Delta tau``
        with nothing left over.
        """
        for k in REFERENCE_K_VALUES:
            table = build_phase_residual(self.s.radiation, k, self.s.radiation_z, "Gk")
            self.assertTrue(
                np.all(table.hi == 0.0), msg=f"rho_G hi limb is not zero at k={k:.6e}"
            )
            self.assertTrue(
                np.all(table.lo == 0.0), msg=f"rho_G lo limb is not zero at k={k:.6e}"
            )
            self.assertEqual(
                table.delta(
                    float(self.s.radiation_z[0]), float(self.s.radiation_z[-1])
                ),
                0.0,
            )

    def test_rho_T_against_the_exact_radiation_primitive(self):
        """
        ``rho_T.delta(z_i, z)`` against ``RadiationModel.rho_T``, the exact primitive of
        ``C_T/(omega_T + k c_s/H)`` for ``C_T = -2/s^2``, over ``x`` in [24, 1e4].
        """
        model = self.s.radiation
        k = RADIATION_K_INV_MPC
        z = self.s.radiation_z
        z_anchor = float(z[0])

        table = build_phase_residual(model, k, z, "Tk")

        worst_abs, worst_z, worst_rel = 0.0, None, 0.0
        for z_sample in z:
            z_sample = float(z_sample)
            got = table.delta(z_anchor, z_sample)
            exact = model.rho_T(k, z_sample, z_anchor)
            err = fabs(got - exact)
            if err > worst_abs:
                worst_abs, worst_z = err, z_sample
            if exact != 0.0:
                worst_rel = max(worst_rel, err / fabs(exact))

        print(
            f"\n[rho_T] radiation control, x in [{RADIATION_X_LO:g}, {RADIATION_X_HI:g}]: "
            f"max {worst_abs:.4e} rad absolute at z = {worst_z}, {worst_rel:.4e} relative "
            f"({len(z)} nodes, order {RHO_GAUSS_ORDER})"
        )
        self.assertLessEqual(worst_abs, RADIATION_ABS_TOL)

    def test_rho_T_matches_the_review_asymptote(self):
        """
        Over the whole window the table gives ``rho_T = -0.041579`` rad, whose magnitude is
        review §12.4's 0.0416 from ``x_i = 24``; the sign is opposite because the review's
        ``rho`` is ``theta - (x_i - x)`` (see the module docstring). The departure from the
        asymptote ``1/x - 1/x_i`` is the ``2/(3 x_i^3)`` term, 2.9e-4 relative, which is why the
        exact primitive and not the asymptote is the reference above.
        """
        model = self.s.radiation
        k = RADIATION_K_INV_MPC
        z = self.s.radiation_z
        z_anchor, z_end = float(z[0]), float(z[-1])

        table = build_phase_residual(model, k, z, "Tk")
        got = table.delta(z_anchor, z_end)

        x_i = model.x_T(k, z_anchor)
        x = model.x_T(k, z_end)
        asymptote = 1.0 / x - 1.0 / x_i
        departure = fabs(got - asymptote) / fabs(got)

        print(
            f"[rho_T] whole window: table {got:.12e} rad; asymptote 1/x - 1/x_i "
            f"{asymptote:.12e}; departure {departure:.4e} relative; review §12.4 quotes "
            f"1/x_i - 1/x = {-asymptote:.6f} at x_i = {x_i:g}"
        )
        self.assertAlmostEqual(x_i, RADIATION_X_LO, places=9)
        self.assertAlmostEqual(x, RADIATION_X_HI, places=6)
        # the asymptote is right to its own 2/(3 x^3), and no better
        self.assertLess(departure, 1.0e-3)
        self.assertGreater(departure, 1.0e-5)


class TestAgainstReferences(unittest.TestCase):
    """Prompt 05 §2 test 2: both production models, both sectors, three ``k``."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_rho_at_the_reference_checkpoints(self):
        worst, worst_where = 0.0, None
        per_case = []

        for model_key in ("LambdaCDMModel", "QCDModel"):
            block = self.s.references["models"][model_key]
            for k_key, z_anchor in block["rho_anchor_z"].items():
                z_anchor = float(z_anchor)
                for sector, reference_key in (("Gk", "rho_G"), ("Tk", "rho_T")):
                    table = self.s.table(model_key, k_key, sector)
                    case_worst, case_z = 0.0, None
                    for entry in block[reference_key][k_key]:
                        got = table.delta(z_anchor, float(entry["z"]))
                        err = fabs(got - float(entry["value"]))
                        if err > case_worst:
                            case_worst, case_z = err, float(entry["z"])
                    per_case.append(
                        (model_key, sector, float(k_key), case_worst, case_z)
                    )
                    if case_worst > worst:
                        worst = case_worst
                        worst_where = (model_key, sector, float(k_key), case_z)

        print("\n[rho] against prompt 01's references (rad, absolute):")
        for model_key, sector, k, err, z in per_case:
            print(f"    {model_key:<14s} {sector}  k = {k:.6e}   {err:.4e}  at z = {z}")
        print(
            f"    worst {worst:.4e} rad at "
            f"(model, sector, k, z) = {worst_where}; threshold "
            f"{RHO_REFERENCE_ABS_TOL:g}"
        )
        self.assertLessEqual(worst, RHO_REFERENCE_ABS_TOL)


class TestSizeSanity(unittest.TestCase):
    """
    Prompt 05 §2 test 3: the review's §6 and §12.2 magnitudes, over the whole WKB range (the
    3-e-fold anchor down to ``z = 0.1``).
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def _whole_range(self, model_key, k_key, sector):
        block = self.s.references["models"][model_key]
        z_anchor = float(block["rho_anchor_z"][k_key])
        table = self.s.table(model_key, k_key, sector)
        return table.delta(z_anchor, float(table.z_nodes[-1]))

    def test_rho_G_magnitudes(self):
        smallest_k = f"{REFERENCE_K_VALUES[0]:.6e}"
        largest_k = f"{REFERENCE_K_VALUES[-1]:.6e}"

        lambdacdm = self._whole_range("LambdaCDMModel", smallest_k, "Gk")
        qcd = self._whole_range("QCDModel", largest_k, "Gk")
        print(
            f"\n[rho_G] LambdaCDM k = {smallest_k}: {lambdacdm:.6e} rad; "
            f"QCD k = {largest_k}: {qcd:.6e} rad"
        )

        # review §6: -2.5e-7 rad on LambdaCDM at k = 1e5 -- negligible, but not zero
        self.assertLessEqual(fabs(lambdacdm), LAMBDACDM_RHO_G_MAX)
        # review §6: -1.26e-3 rad on QCD_Cosmology at k = 3e8 -- the QCD transition
        self.assertGreaterEqual(fabs(qcd), QCD_RHO_G_MIN)
        self.assertLessEqual(fabs(qcd), QCD_RHO_G_MAX)

    def test_rho_T_magnitudes(self):
        print("")
        for model_key in ("LambdaCDMModel", "QCDModel"):
            for k in REFERENCE_K_VALUES:
                value = self._whole_range(model_key, f"{k:.6e}", "Tk")
                print(f"[rho_T] {model_key:<14s} k = {k:.6e}: {value:.6e} rad")
                # review §12.2: -0.086 (LambdaCDM) to -0.093 (QCD) at every k, and *negative*
                # in the campaign's sign convention
                self.assertGreaterEqual(value, RHO_T_MIN)
                self.assertLessEqual(value, RHO_T_MAX)


class TestCost(unittest.TestCase):
    """Prompt 05 §2 test 4."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_integrand_evaluations_and_wall_time(self):
        print("")
        for (model_key, k_key, sector), (
            table,
            num_nodes,
            seconds,
        ) in self.s.build_cost.items():
            panels = num_nodes - 1
            baseline = RHO_GAUSS_ORDER * panels
            print(
                f"[cost] {model_key:<14s} {sector}  k = {float(k_key):.6e}  "
                f"{num_nodes} nodes  {table.evaluations} evaluations "
                f"({table.evaluations / baseline:.3f} x order*intervals)  {seconds:.3f} s"
            )
            if model_key == "LambdaCDMModel":
                # no break points: exactly one Gauss panel per interval
                self.assertEqual(table.evaluations, baseline)
            else:
                # QCD_Cosmology splits every panel at the T(z) spline knots and the
                # equation-of-state branch temperatures (log 02): 24 % more evaluations
                self.assertLessEqual(
                    table.evaluations, COST_BREAK_POINT_FACTOR * baseline
                )
                self.assertGreater(table.evaluations, baseline)


class TestGuards(unittest.TestCase):
    """The integrand refuses to run outside the WKB region, and the builder refuses a bad sector."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_non_positive_omega_sq_raises(self):
        """
        Deep outside the horizon ``omega^2 = (k/H)^2 + C`` turns negative (``C = -2/s^2`` in
        radiation), and there is no Liouville-Green phase to take a residual of.
        ``WKB_phase_function`` raises there today; so does this.
        """
        model = self.s.radiation
        # exact radiation has C == 0 for Gk, so use the Tk sector, where C_T = -2/s^2 and
        # omega_T^2 < 0 once x_T < sqrt(2)
        f = phase_residual_integrand(model, RADIATION_K_INV_MPC, "Tk")
        z_bad = RADIATION_K_INV_MPC / (sqrt(3.0) * model.H0 * 1.0) - 1.0  # x_T = 1
        with self.assertRaises(ValueError):
            f(z_bad)

    def test_unknown_sector_raises(self):
        with self.assertRaises(ValueError):
            phase_residual_integrand(self.s.radiation, 1.0e5, "Qk")
        with self.assertRaises(ValueError):
            build_phase_residual(self.s.radiation, 1.0e5, self.s.radiation_z, "Qk")

    def test_no_adaptive_fallback_was_required(self):
        """
        Review §11 holds an adaptive rule for ``rho`` alone in reserve if a fixed-order Gauss
        rule fails to converge across ``QCD_Cosmology``'s spline knots. Prompt 02 measured that
        it does converge, so none is implemented; this pins the record.
        """
        self.assertFalse(RHO_ADAPTIVE_FALLBACK_REQUIRED)
        self.assertEqual(
            RHO_GAUSS_ORDER,
            int(self.s.references["convergence"]["decision"]["N_rho"]),
        )
        self.assertFalse(
            self.s.references["convergence"]["decision"][
                "rho_adaptive_fallback_required"
            ]
        )


if __name__ == "__main__":
    unittest.main()
