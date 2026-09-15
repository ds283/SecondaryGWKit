"""
A smooth interpolant may not run across a point at which the cosmology declares it is not smooth.

``ComputeTargets/BackgroundModel.py`` splines background quantities at two sites, and until prompt
13 of ``prompts/qcd-background-audit/`` both of them ran straight across the equation of state's
declared crossings:

* ``compute_background._build_derivative`` fits an order-5 spline of ``log H`` over
  ``_build_derivative_fit_grid(z_sample)`` -- a padded, 3x refined copy of the source grid -- and
  differentiates it, then stacks ``d2_lnH_dz2`` and ``d3_lnH_dz3`` on the result;
* ``BackgroundModel._create_functions`` (now ``build_stored_sample_spline``) fits a cubic through
  the **stored** samples of anything the cosmology does not supply as a method.

``H`` genuinely *steps* at two of the three crossings ``QCD_Cosmology`` declares (a ``g_s`` jump at
an equation-of-state branch join, ``docs/qcd-background-audit-2026-09.md`` §1, §2), so both splines
rang there: 2.04e-02 relative in ``epsilon`` at ``T_LO`` and 1.03e-03 at ``T_120_MEV`` on the
production grid, against 3.9e-09 max / 8.1e-10 median away from a crossing
(``docs/qcd-background-verification.md`` §10.4). ``EOS_T_LO`` is the control: there ``g_s`` is
continuous to 1.8e-11 and only ``w`` kinks, ``H`` does not step, and nothing rang -- which is the
evidence that the other two were a spline ringing at a step and not an error of the cosmology.

This module re-takes that measurement, both sites separately, and pins it. It needs no Ray and no
datastore: ``compute_background``'s pieces are called directly and the production source grid is
rebuilt from ``CosmologyConcepts.build_z_sample`` and ``main.py``'s own helper, exactly as
``docs/qcd-background-audit/grid_density_criterion.py`` does.
"""

import unittest
from math import expm1, log

import numpy as np
from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import (
    DERIVATIVE_SPLINE_ORDER,
    STORED_SAMPLE_SPLINE_ORDER,
    SegmentedSpline,
    _build_derivative_fit_grid,
    _cosmology_break_points,
    _segment_slices,
    build_stored_sample_spline,
)
from ComputeTargets.spline_wrappers import ZSplineWrapper
from ComputeTargets.tests.test_main_plumbing import load_main_py_functions
from ComputeTargets.tests.wkb_reference import (
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_Z_END,
    load_references,
    to_redshift_array,
)
from CosmologyConcepts import (
    SOURCE_GRID_BREAK_HALF_WIDTH,
    SOURCE_GRID_BREAK_REFINEMENT,
    SOURCE_GRID_BREAK_STANDOFF,
    build_z_sample,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

# The offsets in u = log(1+z) either side of a declared crossing at which epsilon is scored, and
# the step of the central difference that supplies the reference. Both are
# docs/qcd-background-verification.md §10.4's, unchanged, so that the numbers here and there are
# the same measurement.
PROBE_OFFSETS_U = (-0.20, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.20)
CENTRAL_DIFFERENCE_STEP_U = 1.0e-6

# The regime that holds away from any declared crossing, measured over the whole production range
# on the same comparison: 3.907e-09 max, 8.091e-10 median (§10.4 / prompt 12's section A2). The
# acceptance for the two genuine steps is that they come down into it.
AWAY_FROM_CROSSING_MAX = 3.907e-09
RING_TOLERANCE = 1.0e-8

# The control. EOS_T_LO reads 1.6e-09 on the shipped grid both before and after segmentation; the
# threshold is set just above it so that the test says "unchanged", not "small".
CONTROL_TOLERANCE = 2.0e-9

# What the defect was worth before either site was segmented, on the production grid. Asserted as
# a floor rather than a ceiling: a test that passes both before and after a change has measured
# nothing (prompts/qcd-background-audit/README §0.2), so the unsegmented control has to be seen to
# fail the tolerance above by orders.
UNSEGMENTED_RINGING_AT_T_LO = 1.0e-3

cosmology_feature_redshifts = load_main_py_functions(
    ["cosmology_feature_redshifts"],
    extra_globals={"np": np, "_cosmology_break_points": _cosmology_break_points},
)["cosmology_feature_redshifts"]


def production_source_grid(cosmology):
    """The grid ``main.py`` builds for this cosmology, prompt 11's cosmology-aware one."""
    z_init = float(load_references()["models"]["LambdaCDMModel"]["grid"]["z_init"])
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
    return to_redshift_array(grid.z_values), np.asarray(break_z, dtype=float)


def epsilon_accessor(cosmology, z_sample, segment_fit: bool, segment_stored: bool):
    """
    ``epsilon(z)`` as ``BackgroundModel`` delivers it, with each of the two spline sites segmented
    or not, so that what each one buys can be scored on its own.

    This is ``compute_background``'s derivative stack for ``d_lnH_dz`` followed by
    ``_create_functions``'s stored-sample spline, with nothing else in between -- no cumulative
    table is built, so the whole thing costs one pass of ``Hubble`` over the padded fit grid.
    """
    fit_x, fit_z, fit_select, _ = _build_derivative_fit_grid(z_sample)
    fit_edges = _cosmology_break_points(cosmology, float(fit_z[0]), float(fit_z[-1]))

    y = np.array([log(cosmology.Hubble(z)) for z in fit_z])
    d_du = np.empty(len(fit_x), dtype=float)
    for sl in _segment_slices(fit_x, fit_edges if segment_fit else ()):
        spline = make_interp_spline(fit_x[sl], y[sl], k=DERIVATIVE_SPLINE_ORDER)
        d_du[sl] = np.asarray(spline.derivative()(fit_x[sl]))
    d_lnH_dz_fit = d_du / (fit_z + 1.0)

    stored_edges = _cosmology_break_points(cosmology, z_sample.min.z, z_sample.max.z)
    accessor = build_stored_sample_spline(
        "d_lnH_dz",
        fit_x[fit_select],
        d_lnH_dz_fit[fit_select],
        min_z=z_sample.min.z,
        max_z=z_sample.max.z,
        break_points=stored_edges if segment_stored else (),
    )
    return lambda z: (1.0 + z) * accessor(z)


def epsilon_from_cosmology(cosmology, u: float) -> float:
    """
    ``epsilon = -dot H / H^2 = d log H / d log(1+z)`` from a central difference of the cosmology's
    own pointwise ``Hubble``, which is what the whole comparison is scored against. The stencil is
    1e-6 wide in ``u``, far narrower than the distance from any probe point to a crossing, so it
    never straddles one.
    """
    d = CENTRAL_DIFFERENCE_STEP_U
    hi = log(cosmology.Hubble(expm1(u + d)))
    lo = log(cosmology.Hubble(expm1(u - d)))
    return (hi - lo) / (2.0 * d)


class TestDeclaredCrossingsAreNotSplinedAcross(unittest.TestCase):
    """The §10.4 measurement, re-taken on the production grid and pinned."""

    @classmethod
    def setUpClass(cls):
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        cls.z_sample, cls.break_z = production_source_grid(cls.cosmology)
        cls.break_u = np.log1p(cls.break_z)
        cls.eps = {
            (fit, stored): epsilon_accessor(cls.cosmology, cls.z_sample, fit, stored)
            for fit, stored in ((False, False), (True, False), (True, True))
        }

    def _worst_at(self, crossing_u: float, key) -> float:
        eps = self.eps[key]
        worst = 0.0
        for du in PROBE_OFFSETS_U:
            u = float(crossing_u) + du
            reference = epsilon_from_cosmology(self.cosmology, u)
            got = eps(expm1(u))
            worst = max(worst, abs(got / reference - 1.0))
        return worst

    def test_the_cosmology_declares_three_crossings_on_the_production_grid(self):
        """Prompt 07's set, and the anchor for every index below."""
        self.assertEqual(len(self.break_u), 3)
        np.testing.assert_allclose(
            self.break_u,
            [17.565806941870026, 23.197460552819653, 27.485391822044257],
            rtol=0.0,
            atol=1e-12,
        )

    def test_both_sites_segmented_removes_the_ringing(self):
        """
        The acceptance row of prompt 13 §4: the two genuine steps come down into the regime that
        holds away from a crossing.
        """
        for i, u in enumerate(self.break_u):
            with self.subTest(crossing=i):
                worst = self._worst_at(u, (True, True))
                self.assertLess(
                    worst,
                    RING_TOLERANCE,
                    msg=f"epsilon rings by {worst:.3e} at the crossing u = {u!r}",
                )

    def test_the_control_crossing_is_not_made_worse(self):
        """
        ``EOS_T_LO`` is where ``g_s`` is continuous to 1.8e-11 and only ``w`` kinks, so ``H`` does
        not step and nothing rang there before. Segmenting must not disturb it.
        """
        before = self._worst_at(self.break_u[1], (False, False))
        after = self._worst_at(self.break_u[1], (True, True))
        self.assertLess(before, CONTROL_TOLERANCE)
        self.assertLess(after, CONTROL_TOLERANCE)

    def test_an_unsegmented_fit_still_rings(self):
        """
        The guard that this test can fail. With neither site segmented -- which is what the tree
        did before prompt 13 -- ``T_LO`` is orders outside the tolerance above.
        """
        worst = self._worst_at(self.break_u[0], (False, False))
        self.assertGreater(worst, UNSEGMENTED_RINGING_AT_T_LO)

    def test_the_stored_sample_spline_is_needed_as_well_as_the_fit(self):
        """
        Segmenting ``compute_background``'s fit alone leaves a cubic running across the step in the
        stored samples, and at ``T_120_MEV`` that alone is worth four orders of the residue. Both
        sites, or neither: prompt 13 §2 item 2.
        """
        fit_only = self._worst_at(self.break_u[2], (True, False))
        both = self._worst_at(self.break_u[2], (True, True))
        self.assertGreater(fit_only, RING_TOLERANCE)
        self.assertLess(both, RING_TOLERANCE)
        self.assertLess(both, fit_only)

    def test_away_from_a_crossing_nothing_of_consequence_moves(self):
        """
        An interpolating spline is global over the interval it is fitted on, so segmenting one
        does move every value on both branches, not only the values beside the cut -- there is no
        "locality" to appeal to and this test does not claim any. What it pins is the *size*: half
        a decade or more from any crossing the two constructions agree to a few tens of ulp, five
        orders below the 3.9e-09 the comparison against the cosmology itself reads there. The
        ringing tests above are what say the change is real where it is meant to be.
        """
        unsegmented = self.eps[(False, False)]
        segmented = self.eps[(True, True)]
        for u in (5.0, 10.0, 14.0, 20.0, 25.0, 31.0, 34.0):
            with self.subTest(u=u):
                self.assertTrue(
                    np.all(np.abs(np.asarray(self.break_u) - u) > 0.5),
                    msg="probe point is not away from a crossing",
                )
                z = expm1(u)
                before, after = unsegmented(z), segmented(z)
                self.assertLess(abs(after / before - 1.0), 1.0e-12)


class TestSegmentSlices(unittest.TestCase):
    """``_segment_slices`` against ``SegmentedSpline``'s own dispatch, and the refusals."""

    def test_no_edges_is_the_whole_grid(self):
        x = np.linspace(0.0, 1.0, 11)
        self.assertEqual(_segment_slices(x, ()), [slice(0, 11)])

    def test_the_partition_agrees_with_the_dispatch(self):
        """
        The partition and the dispatch have to place the same ``u`` in the same segment, including
        a ``u`` that lands exactly on an edge -- which belongs to the branch **above** it, because
        the edge is the first representable ``u`` at or above the crossing and carries that
        branch's value.
        """
        x = np.linspace(0.0, 10.0, 101)
        edges = [2.0, 5.0, 7.5]
        slices = _segment_slices(x, edges)
        self.assertEqual(len(slices), len(edges) + 1)

        marker = SegmentedSpline(
            edges, [(lambda i: (lambda u: i))(i) for i in range(4)]
        )
        for index, sl in enumerate(slices):
            for value in x[sl]:
                self.assertEqual(marker(float(value)), index)

        # x[20] == 2.0 exactly: it must open the second segment, not close the first
        self.assertEqual(x[20], 2.0)
        self.assertEqual(slices[0].stop, 20)
        self.assertEqual(slices[1].start, 20)

    def test_segments_cover_the_grid_exactly_once(self):
        x = np.sort(np.random.default_rng(20260915).uniform(0.0, 10.0, 200))
        slices = _segment_slices(x, [2.0, 5.0, 7.5])
        covered = np.concatenate([np.arange(sl.start, sl.stop) for sl in slices])
        np.testing.assert_array_equal(covered, np.arange(len(x)))

    def test_a_branch_too_narrow_for_the_order_is_refused(self):
        """
        Never a silent drop to a lower order, and never a fit across the step after all: the
        failure this design exists to prevent is exactly the silent one.
        """
        x = np.array([0.0, 1.0, 1.9, 2.1, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = np.arange(len(x), dtype=float)
        with self.assertRaises(RuntimeError) as caught:
            build_stored_sample_spline(
                "d_lnH_dz", x, y, min_z=0.0, max_z=10.0, break_points=[1.95]
            )
        message = str(caught.exception)
        self.assertIn("build_stored_sample_spline", message)
        self.assertIn(f"{STORED_SAMPLE_SPLINE_ORDER + 1}", message)

    def test_an_unsegmented_build_is_the_spline_it_always_was(self):
        """
        README §2 (g): a cosmology that declares nothing must take a path that is not merely
        equivalent to the old one but is literally it.
        """
        x = np.linspace(0.1, 4.0, 60)
        y = np.sin(3.0 * x) + 0.25 * x**2
        min_z, max_z = expm1(x[0]), expm1(x[-1])
        built = build_stored_sample_spline(
            "probe", x, y, min_z=min_z, max_z=max_z, break_points=()
        )
        # what _create_functions built before prompt 13, written out in full
        reference = ZSplineWrapper(
            make_interp_spline(x, y),
            label="probe",
            min_z=min_z,
            max_z=max_z,
            log_z=True,
        )
        for u in np.linspace(x[0], x[-1], 97):
            z = expm1(float(u))
            self.assertEqual(float(built(z)), float(reference(z)))


class TestCosmologyThatDeclaresNothing(unittest.TestCase):
    """LambdaCDM declares no break points, so every site above is the unsegmented one."""

    def test_lambdacdm_declares_no_break_points(self):
        cosmology = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        self.assertEqual(
            len(_cosmology_break_points(cosmology, 0.1, 1.0e16)),
            0,
        )


if __name__ == "__main__":
    unittest.main()
