"""
The convergence facility and the named source-grid generations
(``prompts/tolerance-convergence/`` prompt 01, board items T1 and T2).

Four things are asserted here, and the first two are the ones that must not be weakened:

1. **A drift figure cannot be obtained without its verdict.** ``reference_drift`` has no default
   for the smallest difference the caller intends to report, and what it returns carries the
   threshold, the verdict and the numbers as one object. ``GkTk-remedial`` prompt 17 reported
   candidate errors through a reference that had not converged on ``QCDModel`` at four
   wavenumbers; the API is shaped so that doing that again takes deliberate effort.

2. **"One step tighter" is the knob's property.** A decade for a tolerance pair, one order for a
   Gauss-Legendre rule. Prompt 04 audits four integer orders and has no tolerance to move.

3. **The grid generations are named, and the hoist did not move them.** Version 0, 1 and 2 are
   distinct constructions with distinct sample counts and distinct digests; the version-2 one is
   the grid ``main.py`` builds and is bit-identical to the private helper it was hoisted out of;
   and ``SOURCE_GRID_V2_REPRODUCES_VERSION`` is *cross-checked* against production's own
   ``SOURCE_GRID_CONSTRUCTION_VERSION`` rather than asserted, so that a bump says so here instead
   of quietly leaving a stale construction called "production".

4. **The constant-w anchors are the anchors README §3.1 claims.** In particular ``rho_G`` vanishes
   identically, which is what makes the ``N_rho`` measurement pure quadrature error, and
   ``z_exit`` is an elementary inversion, which is why board item T6's "never measured" was never
   a statement about difficulty.

Dry: no Ray, no datastore, and the only cosmologies built are ``QCD_Cosmology`` and ``LambdaCDM``.
"""

import unittest
from math import cos, sin, sqrt

import numpy as np

from ComputeTargets.tests.convergence_reference import (
    CRITERION_RATIO,
    SCIPY_RTOL_FLOOR,
    V0_PER_K_GRID,
    V2_PRODUCTION_GRID,
    AccuracyKnob,
    DriftVerdict,
    GaussOrder,
    SourceGridSpec,
    TolerancePair,
    analytic_T,
    analytic_Tprime,
    anchor_error,
    converged_reference,
    exact_z_exit,
    radiation_anchors,
    reference_drift,
    sector_reference,
    tk_geometry,
    tk_run,
    x_local,
)
from ComputeTargets.tests.wkb_reference import (
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_INIT,
    SOURCE_GRID_V0,
    SOURCE_GRID_V1,
    SOURCE_GRID_V2,
    SOURCE_GRID_V2_REPRODUCES_VERSION,
    RadiationModel,
    horizon_exit_z,
    production_source_grid,
    production_source_z_values,
    source_grid,
)
from CosmologyConcepts import redshift_grid_digest
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

# The three generations, by length and by digest over their exact bits, at production geometry.
# These are ComputeTargets/tests/test_source_grid.py's own recorded values: the hoist is a move,
# so they have to be the same numbers on the same cosmologies.
GRID_FINGERPRINTS = {
    (SOURCE_GRID_V0, None): (1732, "0960e169"),
    (SOURCE_GRID_V1, "QCD"): (1773, "81c6e682"),
    (SOURCE_GRID_V2, "QCD"): (1996, "4849552b"),
    (SOURCE_GRID_V1, "LambdaCDM"): (1732, "0960e169"),
    (SOURCE_GRID_V2, "LambdaCDM"): (1778, "60a3205a"),
}


class TestTheKnobKnowsItsOwnStep(unittest.TestCase):
    """README §3.1: a decade for a tolerance pair, one order for a Gauss order."""

    def test_a_tolerance_pair_steps_by_a_decade_on_both_axes(self):
        knob = TolerancePair(1e-18, 1e-12)
        self.assertEqual(knob.tighter(), TolerancePair(1e-19, 1e-13))
        self.assertEqual(knob.looser(), TolerancePair(1e-17, 1e-11))
        self.assertEqual(knob.tighter(2), TolerancePair(1e-20, 1e-14))

    def test_a_tolerance_pair_can_step_one_axis_alone(self):
        """
        The axis prompt 17 separated in the transfer-function sector and review §10.1 did not in
        the Green's-function sector (README §2 (d)). Prompt 03 needs both axes separately, so the
        knob has to express one.
        """
        self.assertEqual(
            TolerancePair(1e-10, 1e-8, axis="rtol").tighter(),
            TolerancePair(1e-10, 1e-9, axis="rtol"),
        )
        self.assertEqual(
            TolerancePair(1e-10, 1e-8, axis="atol").tighter(),
            TolerancePair(1e-11, 1e-8, axis="atol"),
        )

    def test_a_gauss_order_steps_by_one_order(self):
        self.assertEqual(GaussOrder(4, "N_rho").tighter(), GaussOrder(5, "N_rho"))
        self.assertEqual(GaussOrder(4).tighter(3).order, 7)
        self.assertEqual(GaussOrder(4, "N_tau").label, "N_tau = 4")

    def test_a_ladder_runs_loose_to_tight(self):
        """README §6.1 rule 3: the target is the *first* setting that clears the floor."""
        self.assertEqual(
            [k.order for k in GaussOrder(4).ladder(3)],
            [4, 5, 6, 7],
        )
        self.assertEqual(
            [k.rtol for k in TolerancePair(1e-10, 1e-8).ladder(2)],
            [1e-8, 1e-9, 1e-10],
        )

    def test_the_knob_refuses_what_it_cannot_step(self):
        with self.assertRaises(ValueError):
            TolerancePair(1e-10, 1e-8, axis="nonsense")
        with self.assertRaises(ValueError):
            GaussOrder(0)
        with self.assertRaises(NotImplementedError):
            AccuracyKnob().tighter()

    def test_the_scipy_rtol_clamp_is_known(self):
        """
        SciPy clamps ``rtol`` at ``100 * eps``
        (``scipy/integrate/_ivp/common.py:47-51``), so ``TIGHTENED_RTOL = 1e-13`` is the last
        tightening a Runge-Kutta stepper really applies -- the reason ``GkTk-remedial`` prompt
        17's convergence check is one decade and not two.
        """
        self.assertTrue(TolerancePair(1e-19, 1e-13).rtol_step_is_effective)
        self.assertFalse(TolerancePair(1e-20, 1e-14).rtol_step_is_effective)
        self.assertEqual(SCIPY_RTOL_FLOOR, 100.0 * float(np.spacing(1.0)))


class TestADriftCannotBeObtainedWithoutItsVerdict(unittest.TestCase):
    """
    The facility's central constraint (README §5 rule 5).

    The build function here is synthetic -- a number that converges geometrically in the knob --
    so that the criterion logic is tested on its own, without an ODE solve deciding the outcome.
    """

    @staticmethod
    def _measure(candidate, reference):
        """Two samples, so that the summary's second-largest column has something to report."""
        return [(abs(candidate - reference), 0.0, 0.0), (0.0, 1.0, 1.0)]

    @staticmethod
    def _build(scale):
        return lambda knob: 1.0 + scale * knob.rtol

    def test_the_verdict_and_the_numbers_arrive_together(self):
        verdict = reference_drift(
            self._build(1.0),
            TolerancePair(1e-18, 1e-12),
            error_measure=self._measure,
            smallest_reported_difference=1e-7,
        )
        self.assertIsInstance(verdict, DriftVerdict)
        self.assertEqual(verdict.threshold, 1e-7 / CRITERION_RATIO)
        self.assertAlmostEqual(verdict.max, 0.9e-12)
        self.assertTrue(verdict.passed)
        self.assertGreater(verdict.headroom, 1.0)
        self.assertIn("converged", str(verdict))

    def test_a_reference_that_has_not_converged_says_so(self):
        """
        The prompt 17 failure, in miniature: a drift of 9e-7 against a smallest reported
        difference of 3.45e-7 is a measurement of the reference, and the verdict has to be false
        rather than a footnote.
        """
        verdict = reference_drift(
            self._build(1.0e6),
            TolerancePair(1e-18, 1e-12),
            error_measure=self._measure,
            smallest_reported_difference=3.45e-7,
        )
        self.assertFalse(verdict.passed)
        self.assertLess(verdict.headroom, 1.0)
        self.assertIn("NOT CONVERGED", str(verdict))

    def test_there_is_no_default_for_the_smallest_reported_difference(self):
        with self.assertRaises(TypeError):
            reference_drift(
                self._build(1.0),
                TolerancePair(1e-18, 1e-12),
                error_measure=self._measure,
            )
        with self.assertRaises(ValueError):
            reference_drift(
                self._build(1.0),
                TolerancePair(1e-18, 1e-12),
                error_measure=self._measure,
                smallest_reported_difference=0.0,
            )

    def test_a_step_scipy_would_ignore_is_recorded(self):
        verdict = reference_drift(
            self._build(1.0),
            TolerancePair(1e-19, 1e-13),
            error_measure=self._measure,
            smallest_reported_difference=1e-7,
        )
        self.assertEqual(len(verdict.notes), 1)
        self.assertIn("silently ignored", verdict.notes[0])

    def test_a_converged_reference_carries_its_own_payload_and_drift(self):
        reference = converged_reference(
            self._build(1.0),
            TolerancePair(1e-18, 1e-12),
            error_measure=self._measure,
            smallest_reported_difference=1e-7,
        )
        self.assertTrue(bool(reference))
        self.assertTrue(reference.converged)
        self.assertEqual(reference.payload, 1.0 + 1e-12)
        scored = reference.score(1.0 + 2e-12)
        self.assertAlmostEqual(scored["max"], 1e-12)
        self.assertEqual(scored["reference_drift"], reference.drift.max)
        self.assertTrue(scored["reference_converged"])


class TestTheGridGenerationsAreNamedAndUnmoved(unittest.TestCase):
    """
    Board item T2, and the narrowing of ``[00-three-production-grid-reproductions]``.

    The version-2 construction was complete and correct before this prompt; what it was not was
    *reachable*. These assertions are the ones ``test_source_grid.py`` already makes, re-taken
    through the hoisted helper, which is what makes the move a move.
    """

    @classmethod
    def setUpClass(cls):
        cls.cosmologies = {
            "QCD": QCD_Cosmology(
                store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
            ),
            "LambdaCDM": LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018()),
        }

    def test_each_generation_is_the_grid_the_campaign_recorded(self):
        for (generation, model), (count, digest) in GRID_FINGERPRINTS.items():
            with self.subTest(generation=generation, model=model):
                grid = source_grid(
                    generation,
                    cosmology=None if model is None else self.cosmologies[model],
                )
                self.assertEqual(len(grid.z_values), count)
                self.assertEqual(redshift_grid_digest(grid.z_values), digest)

    def test_version_0_is_still_the_logspace_every_published_figure_used(self):
        self.assertEqual(
            source_grid(SOURCE_GRID_V0).z_values.tobytes(),
            production_source_z_values(PRODUCTION_Z_INIT).tobytes(),
            "the named version-0 helper is not bit-identical to the construction every figure in "
            "docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md was scored on",
        )

    def test_the_three_generations_are_three_different_grids(self):
        """The control on the control: a test that could pass on code that did nothing."""
        qcd = self.cosmologies["QCD"]
        lengths = {
            generation: len(
                source_grid(
                    generation, cosmology=None if generation == SOURCE_GRID_V0 else qcd
                ).z_values
            )
            for generation in (SOURCE_GRID_V0, SOURCE_GRID_V1, SOURCE_GRID_V2)
        }
        self.assertEqual(len(set(lengths.values())), 3, lengths)

    def test_the_version_2_tag_is_cross_checked_against_production(self):
        """
        Not ``assertEqual(SOURCE_GRID_CONSTRUCTION_VERSION, 2)`` for its own sake: the point is
        that the test tree's claim to reproduce production is tied to a number production owns.
        When production moves to version 3 this fails, and the message says what to do.
        """
        self.assertEqual(
            SOURCE_GRID_V2_REPRODUCES_VERSION,
            SOURCE_GRID_CONSTRUCTION_VERSION,
            msg="ComputeTargets/tests/wkb_reference.py's SOURCE_GRID_V2 reproduces construction "
            f"version {SOURCE_GRID_V2_REPRODUCES_VERSION}, but main.py now builds version "
            f"{SOURCE_GRID_CONSTRUCTION_VERSION}. The test tree's 'production grid' is a "
            "generation behind: add the new construction as a named generation rather than "
            "editing the old one, and re-score anything that quoted version "
            f"{SOURCE_GRID_V2_REPRODUCES_VERSION}.",
        )

    def test_a_caller_has_to_say_which_generation(self):
        with self.assertRaises(ValueError):
            source_grid("production")
        with self.assertRaises(ValueError):
            source_grid(SOURCE_GRID_V2)  # no cosmology
        with self.assertRaises(ValueError):
            source_grid(SOURCE_GRID_V0, cosmology=self.cosmologies["QCD"])

    def test_a_cosmology_aware_generation_exists_only_as_the_universal_grid(self):
        with self.assertRaises(ValueError):
            SourceGridSpec(generation=SOURCE_GRID_V2, universal=False)
        self.assertEqual(V0_PER_K_GRID.label, "v0 per-k")
        self.assertEqual(V2_PRODUCTION_GRID.label, "v2 universal")

    def test_a_built_grid_labels_itself_with_its_generation_and_digest(self):
        built = V2_PRODUCTION_GRID.build(self.cosmologies["LambdaCDM"])
        self.assertEqual(built.samples, 1778)
        self.assertEqual(built.digest, "60a3205a")
        self.assertIn("v2 universal", built.label)
        self.assertIn("60a3205a", built.label)


class TestTheFoldDidNotMoveTheGeometry(unittest.TestCase):
    """
    ``docs/gktk-remedial/tk_numeric_atol_sweep.py``'s geometry, built through the facility, is the
    geometry that script built inline. A measurement that changes when its code moves was not
    measuring what it claimed.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = RadiationModel()
        cls.k = 1.0e6

    def test_the_version_0_per_k_geometry_is_prompt_17s(self):
        geo = tk_geometry(self.model, self.k, V0_PER_K_GRID.build(self.model))

        z_source = horizon_exit_z(
            self.model, self.k, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
        )
        z_e6 = horizon_exit_z(self.model, self.k, 6.0)
        expected = production_source_grid(z_source).truncate(
            0.85 * z_e6, keep="higher-include"
        )

        self.assertEqual(geo["grid"].as_float_list(), expected.as_float_list())
        self.assertEqual(geo["z_exit"], horizon_exit_z(self.model, self.k, 0.0))
        self.assertEqual(geo["grid_spec"].generation, SOURCE_GRID_V0)

    def test_the_universal_geometry_truncates_rather_than_rebuilds(self):
        """
        ``main.py:1209-1211``: one grid per model, cut to this work item. Every sample of the
        cut grid is a sample of the universal one -- the thing a per-wavenumber lattice cannot
        say.
        """
        built = V0_PER_K_GRID.__class__(
            generation=SOURCE_GRID_V0, universal=True, z_init=PRODUCTION_Z_INIT
        ).build(self.model)
        geo = tk_geometry(self.model, self.k, built)
        universal = set(built.z_values.tolist())
        self.assertTrue(set(geo["grid"].as_float_list()) <= universal)
        self.assertLess(len(geo["grid"]), built.samples)


class TestTheConstantWAnchors(unittest.TestCase):
    """README §3.1's table: the closed forms an exact-radiation control provides."""

    @classmethod
    def setUpClass(cls):
        cls.model = RadiationModel()
        cls.anchors = radiation_anchors(cls.model)

    def test_the_table_is_complete(self):
        self.assertEqual(
            set(self.anchors),
            {
                "T",
                "Tprime",
                "G",
                "Gprime",
                "tau",
                "cs_tau",
                "friction_F",
                "theta_G",
                "rho_G",
                "rho_T",
                "z_exit",
            },
        )

    def test_rho_G_vanishes_identically(self):
        """
        ``C == 0`` in exact radiation, so the Green's-function residual is **zero** and any
        non-zero answer from a quadrature is pure quadrature error. This is what gives prompt
        04's ``N_rho`` measurement a two-sided oracle and no reference to build.
        """
        kind, rho_G = self.anchors["rho_G"]
        self.assertEqual(kind, "phase")
        for k in (1.0e5, 1.0e7, 3.0e8):
            for z in (1.0e3, 1.0e6, 1.0e9):
                self.assertEqual(rho_G(k, z, 10.0 * z), 0.0)

    def test_z_exit_is_the_elementary_inversion(self):
        """
        The closed form against the **harness's** bracketed solve
        (``wkb_reference.horizon_exit_z``, ``xtol = rtol = 1e-14``) over the production range.

        This is *not* board item T6: the production quantity is
        ``CosmologyConcepts.wavenumber``'s ``root_scalar`` at ``xtol = 1e-10, rtol = 1e-8``, and
        measuring that is prompt 03's. What is asserted here is only that the oracle exists and
        is exact, so that prompt 03 has something to score against -- which is the fact README
        §3.1 consequence 3 says nobody had noticed.
        """
        worst = 0.0
        for k in (1.0e3, 1.0e5, 1.0e7, 3.0e8):
            for efolds in (-3.0, 0.0, 4.0):
                exact = exact_z_exit(self.model.H0, k, efolds)
                solved = horizon_exit_z(self.model, k, efolds)
                worst = max(worst, abs(solved - exact) / abs(exact))
        self.assertLess(worst, 1.0e-15, f"worst relative disagreement {worst:.3g}")

    def test_the_transfer_oracle_reduces_to_the_radiation_closed_form(self):
        """``compute_analytic_T`` at ``w = 1/3`` is ``3(sin x - x cos x)/x^3`` (review §12.4)."""
        k = 1.0e6
        for z in (1.0e4, 1.0e6, 1.0e8):
            with self.subTest(z=z):
                x = x_local(self.model, k, z)
                expected = 3.0 * (sin(x) - x * cos(x)) / (x * x * x)
                # 1e-8, not machine precision: the two routes differ by SciPy's own ``jv``
                # evaluation against the elementary form, and by the cancellation in
                # ``sin x - x cos x`` at large x. It is the agreement of two representations,
                # not a convergence statement
                self.assertLess(
                    abs(analytic_T(self.model, k, z) / expected - 1.0), 1.0e-8
                )

    def test_the_interval_anchors_beat_a_difference_of_primitives(self):
        """
        README §3.1 consequence 2: the interval form is the one that matters, and it carries
        ~1e-16 against ~5e-15 for the naive difference over one production grid interval.
        """
        kind, tau_delta = self.anchors["tau"]
        self.assertEqual(kind, "interval")
        z_a, z_b = 1.0e12, 1.0e12 * 0.99
        exact = tau_delta(z_a, z_b)
        naive = self.model.tau(z_b) - self.model.tau(z_a)
        self.assertLess(anchor_error("interval", exact, exact), 1.0e-17)
        self.assertGreater(anchor_error("interval", naive, exact), 0.0)

    def test_rho_T_refuses_to_walk_outside_its_validity_bound(self):
        """
        ``omega_T^2 > 0``, i.e. ``1 + z < k/(sqrt6 H0)``. A caller that trips this has chosen its
        ``z_init`` wrongly; it is not a finding about the representation (prompt 01 §3.1).
        """
        _, rho_T = self.anchors["rho_T"]
        k = 1.0e6
        bound = k / (sqrt(6.0) * self.model.H0) - 1.0
        self.assertIsInstance(rho_T(k, 0.5 * bound, 0.9 * bound), float)
        with self.assertRaises(ValueError):
            rho_T(k, 0.5 * bound, 1.5 * bound)

    def test_a_model_without_closed_forms_is_refused(self):
        class _NotRadiation:
            name = "not radiation"

        with self.assertRaises(TypeError):
            radiation_anchors(_NotRadiation())

    def test_an_anchor_error_needs_the_measure_its_row_calls_for(self):
        with self.assertRaises(ValueError):
            anchor_error("value", 1.0, 1.0)
        with self.assertRaises(ValueError):
            anchor_error("nonsense", 1.0, 1.0)


class TestTheFacilityMeasuresTheTreeItRunsOn(unittest.TestCase):
    """
    One end-to-end use, on the control where truth is known: the self-convergence drift and the
    distance to the oracle, side by side, which is README §0.2's calibration and the thing §5
    rule 5 asks every prompt to do.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = RadiationModel()
        cls.k = 1.0e6
        cls.geo, cls.reference = sector_reference(
            "Tk",
            cls.model,
            cls.model,
            cls.k,
            V0_PER_K_GRID.build(cls.model),
            TolerancePair(1e-18, 1e-12),
            smallest_reported_difference=1.32e-7,
        )

    def test_the_reference_converges_and_says_so(self):
        self.assertTrue(self.reference.converged)
        self.assertLess(self.reference.drift.max, self.reference.drift.threshold)

    def _oracle_errors(self, payload):
        """Envelope-relative error of a run against the exact radiation ``T``, sample by sample."""
        out = []
        for z, value in zip(self.geo["grid"], payload["value_sample"]):
            x = x_local(self.model, self.k, z.z)
            envelope = 3.0 * sqrt(1.0 + x * x) / (x * x * x)
            out.append(
                anchor_error(
                    "value", value, analytic_T(self.model, self.k, z.z), envelope
                )
            )
        return out

    def test_the_drift_is_calibrated_against_the_oracle(self):
        """
        README §0.2: the self-convergence drift is a *surrogate*, and on a constant-w control it
        can be checked against truth. Run the reference again from the **exact** initial data, so
        that the initial-condition error is out of the comparison, and score it against the
        closed-form ``T``: the two must be the same order, which is what licenses the drift
        statistic on ``LambdaCDMModel`` and ``QCDModel``, where no oracle exists.

        ``GkTk-remedial`` prompt 17 §4 found 4.21e-11 drift against 2.3-4.3e-11 from the oracle
        over the whole grid; this is the same comparison at one wavenumber.
        """
        z_top = self.geo["grid"].max.z
        exact_ic = (
            analytic_T(self.model, self.k, z_top),
            analytic_Tprime(self.model, self.k, z_top),
        )
        exact_run = tk_run(self.model, self.k, self.geo, 1e-18, 1e-12, ic=exact_ic)
        oracle = max(self._oracle_errors(exact_run))

        self.assertLess(oracle / self.reference.drift.max, 10.0)
        self.assertGreater(oracle / self.reference.drift.max, 0.1)

    def test_the_initial_condition_floor_is_where_it_was_declared(self):
        """
        The same reference from the **production** ``T = 1, T' = 0`` misses the exact ``T`` by
        2.52e-6 of the envelope, k-independent (README §2 (f), ``GkTk-remedial`` prompt 17 §8).

        It is asserted here because the facility's users have to read every candidate error
        against it: a reported accuracy below a declared floor is a campaign-wide stop, not a
        result. It is *not* re-derived here and this prompt does not revisit it.
        """
        self.assertAlmostEqual(
            max(self._oracle_errors(self.reference.payload)) / 2.52e-6, 1.0, places=2
        )

    def test_a_candidate_is_scored_with_the_references_drift_attached(self):
        candidate = tk_run(self.model, self.k, self.geo, 1e-13, 1e-8)
        scored = self.reference.score(candidate)
        self.assertEqual(scored["reference_drift"], self.reference.drift.max)
        self.assertTrue(scored["reference_converged"])
        self.assertGreater(
            scored["max"],
            10.0 * scored["reference_drift"],
            "the candidate's error does not exceed its reference's drift, so nothing about the "
            "candidate is measurable here",
        )


if __name__ == "__main__":
    unittest.main()
