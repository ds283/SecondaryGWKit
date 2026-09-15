"""
The cosmology-aware source grid: prompt 11 of ``prompts/qcd-background-audit/``, audit section 7.

Three things are asserted here, and the first is the most important:

1. **A cosmology that declares nothing gets the grid it has always got, element for element.**
   That is ``prompts/qcd-background-audit/README.md`` section 0.5 and section 2 (g) -- every
   LambdaCDM, RadiationModel and stand-in number bit-identical across the whole campaign -- and
   it is a stop condition, not an expectation. It is checked twice over: at the level of
   ``build_z_sample`` (nothing declared reproduces ``numpy.logspace`` bit for bit) and at the
   level of ``main.py``'s own policy function, which is what decides whether anything is declared
   at all.

2. **A cosmology that does declare something gets what prompt 10 measured it needs.** The pair
   straddling each break at the chosen standoff, the neighbourhood of each break refined, every
   protected point present, the grid still strictly descending and free of duplicates, and no
   two samples closer together than the datastore can tell apart.

3. **The grid tag identifies the grid.** Two grids of equal length that differ in one sample get
   different tags, where before they collided; and the labels ``main.py`` actually builds come
   from ``main.py``, read with ``ast`` and executed in isolation (``main.py`` cannot be imported,
   ``CLAUDE.md``), rather than from a transcription of them.

This is a *dry* test: no Ray, no datastore, and the only cosmology it builds is
``QCD_Cosmology``, for its ``integration_break_points`` and its Omega values. The consumer
measurement that chose the standoff and the neighbourhood is a 30 s script and lives in
``docs/qcd-background-audit/source_grid_consumer_check.py``.
"""

import unittest
from math import expm1, log10

import numpy as np
from numpy import logspace

from ComputeTargets.BackgroundModel import _cosmology_break_points
from ComputeTargets.tests.test_main_plumbing import load_main_py_functions
from CosmologyConcepts import (
    SOURCE_GRID_BREAK_HALF_WIDTH,
    SOURCE_GRID_BREAK_REFINEMENT,
    SOURCE_GRID_BREAK_STANDOFF,
    build_z_sample,
    redshift,
    redshift_array,
    redshift_grid_digest,
)
from CosmologyConcepts.wavenumber import SOURCE_GRID_MIN_SEPARATION
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from Units import Mpc_units

# the production grid of docs/gktk-remedial/verify_production_path.py and
# ComputeTargets/tests/wkb_reference.py: the z_exit + 5 e-folds of k = 3e8/Mpc, 100 samples per
# decade of z, down to z = 0.1
PRODUCTION_Z_INIT = 2.0636395964161516e16
PRODUCTION_Z_END = 0.1
PRODUCTION_SAMPLES_PER_LOG10Z = 100
PRODUCTION_NUM_NODES = 1732
PRODUCTION_RESPONSE_SPARSENESS = 12

# prompt 07's bisected crossings of QCD_EOS's break_temperatures_GeV, in u = log(1+z), all three
# in range on the production grid (logs/06, logs/07, logs/10 all quote them to these 17 digits)
QCD_CROSSINGS_LOG1PZ = (
    17.565806941870026,
    23.197460552819653,
    27.485391822044257,
)

_main = load_main_py_functions(
    ["cosmology_feature_redshifts", "build_grid_tag_labels"],
    extra_globals={
        "np": np,
        "_cosmology_break_points": _cosmology_break_points,
        "redshift_grid_digest": redshift_grid_digest,
    },
)
cosmology_feature_redshifts = _main["cosmology_feature_redshifts"]
build_grid_tag_labels = _main["build_grid_tag_labels"]


def _production_base_grid() -> np.ndarray:
    """``populate_z_sample``'s arithmetic, transcribed, as it stood before prompt 11."""
    num = int(
        round(
            PRODUCTION_SAMPLES_PER_LOG10Z
            * (log10(PRODUCTION_Z_INIT) - log10(PRODUCTION_Z_END))
            + 0.5,
            0,
        )
    )
    return logspace(log10(PRODUCTION_Z_INIT), log10(PRODUCTION_Z_END), num=num)


def _to_redshift_array(z_values) -> redshift_array:
    return redshift_array(
        [redshift(store_id=i, z=float(z)) for i, z in enumerate(z_values)]
    )


class _SmoothCosmology:
    """
    A cosmology that declares nothing -- the duck type of every LambdaCDM model, RadiationModel
    and test stand-in. It has Omega values, so it would have equality redshifts if anything asked
    for them; the point of the first test below is that nothing does.
    """

    omega_m = 0.3111
    omega_r = 9.139e-05
    omega_cc = 0.6889


class TestACosmologyThatDeclaresNothingIsUntouched(unittest.TestCase):
    """
    README section 0.5 and section 2 (g). This is the campaign's stop condition, and it is
    asserted here at both levels at which it could fail.
    """

    def test_build_z_sample_reproduces_logspace_bit_for_bit(self):
        base = _production_base_grid()
        grid = build_z_sample(
            PRODUCTION_Z_INIT, PRODUCTION_Z_END, PRODUCTION_SAMPLES_PER_LOG10Z
        )

        self.assertEqual(len(grid.z_values), PRODUCTION_NUM_NODES)
        self.assertEqual(
            grid.z_values.tobytes(),
            base.tobytes(),
            "an undeclared cosmology's source grid is not bit-identical to today's logspace",
        )
        self.assertEqual(len(grid.protected_z), 0)
        self.assertEqual(len(grid.breaks), 0)
        self.assertEqual(len(grid.features), 0)

    def test_main_declares_nothing_for_a_smooth_cosmology(self):
        """
        The gate is in ``main.py``, not in ``build_z_sample``: a cosmology with no break points
        gets **no** protected set at all, equality redshifts included. See
        ``cosmology_feature_redshifts``'s docstring for why the two equality samples are not
        worth breaking the campaign's bit-identity invariant for.
        """
        break_z, feature_z = cosmology_feature_redshifts(
            _SmoothCosmology(), PRODUCTION_Z_END, PRODUCTION_Z_INIT
        )
        self.assertEqual(break_z, [])
        self.assertEqual(feature_z, [])

        grid = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=break_z,
            feature_z=feature_z,
        )
        self.assertEqual(grid.z_values.tobytes(), _production_base_grid().tobytes())

    def test_a_declared_grid_is_not_bit_identical(self):
        """The control on the control: the test above would pass on code that did nothing."""
        grid = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=[expm1(u) for u in QCD_CROSSINGS_LOG1PZ],
        )
        self.assertGreater(len(grid.z_values), PRODUCTION_NUM_NODES)


class TestTheProtectedSetOnQCD(unittest.TestCase):
    """The production QCD grid, built the way ``main.py`` builds it."""

    @classmethod
    def setUpClass(cls):
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        cls.break_z, cls.feature_z = cosmology_feature_redshifts(
            cls.cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
        )
        cls.base = _production_base_grid()
        cls.grid = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=cls.break_z,
            feature_z=cls.feature_z,
        )

    def test_the_declared_points_are_prompt_07s_crossings(self):
        """
        What reaches the grid is ``integration_break_points``, converted out of u. The conversion
        is the lossy direction, so this asserts agreement in u, where it is exact, rather than
        in z.
        """
        self.assertEqual(len(self.break_z), 3)
        for z_break, u_expected in zip(self.break_z, QCD_CROSSINGS_LOG1PZ):
            self.assertAlmostEqual(np.log1p(z_break), u_expected, places=13)

        # and the two equality redshifts, which the model computes in its constructor and throws
        # away; the closed form agrees with its own root solve far inside a grid interval
        self.assertEqual(len(self.feature_z), 2)
        self.assertAlmostEqual(self.feature_z[0], 3406.668974249948, places=6)
        self.assertAlmostEqual(self.feature_z[1], 0.3034230329964074, places=12)

    def test_every_protected_point_is_in_the_grid(self):
        values = set(self.grid.z_values.tolist())
        self.assertEqual(len(self.grid.protected_z), 2 * 3 + 2)
        for z in self.grid.protected_z:
            self.assertIn(float(z), values)

    def test_each_break_is_straddled_at_the_chosen_standoff(self):
        """
        Two samples per break, one each side, at
        ``SOURCE_GRID_BREAK_STANDOFF`` of a grid interval. Nothing here compares a recovered
        redshift for equality: the assertions are on separations.
        """
        delta_log10 = (log10(PRODUCTION_Z_INIT) - log10(PRODUCTION_Z_END)) / (
            PRODUCTION_NUM_NODES - 1
        )
        expected = SOURCE_GRID_BREAK_STANDOFF * (pow(10.0, delta_log10) - 1.0)

        values = set(self.grid.z_values.tolist())
        protected = self.grid.protected_z
        for z_break in self.break_z:
            above = protected[protected > z_break].min()
            below = protected[protected < z_break].max()

            self.assertAlmostEqual(
                (above - z_break) / (1.0 + z_break), expected, places=12
            )
            self.assertAlmostEqual(
                (z_break - below) / (1.0 + z_break), expected, places=12
            )

            # the pair is in the grid, and it is the grid's closest approach to the break from
            # each side among the *protected* points; a refinement sample may legitimately fall
            # between a pair member and the break, which is why this asserts the pair rather
            # than the nearest neighbour
            self.assertIn(float(above), values)
            self.assertIn(float(below), values)

            # the pair brackets the break, and it is tighter than the base grid: the two
            # members are 2 * standoff apart against a base spacing of one whole interval
            self.assertLess(below, z_break)
            self.assertGreater(above, z_break)
            self.assertLess(
                (above - below) / (1.0 + z_break),
                pow(10.0, delta_log10) - 1.0,
                "the straddling pair is no tighter than the base grid",
            )

    def test_the_neighbourhood_of_each_break_is_refined(self):
        """
        Prompt 10's measurement: the feature is three to five grid intervals wide, so the grid
        refines ``+-SOURCE_GRID_BREAK_HALF_WIDTH`` intervals by
        ``SOURCE_GRID_BREAK_REFINEMENT``. Counted as the number of samples the grid has that the
        base grid does not.
        """
        extra = set(self.grid.z_values.tolist()) - set(self.base.tolist())
        per_break = 2 + (  # the straddling pair
            2 * SOURCE_GRID_BREAK_HALF_WIDTH + 1
        ) * (SOURCE_GRID_BREAK_REFINEMENT - 1)
        self.assertEqual(len(extra), 3 * per_break + len(self.feature_z))

        # and they really are around the breaks: every refinement sample is inside the declared
        # neighbourhood of some break
        u_base = np.log1p(self.base)
        h = float(np.median(np.abs(np.diff(u_base))))
        for z in extra:
            if float(z) in set(self.grid.features.tolist()):
                continue
            self.assertTrue(
                any(
                    abs(np.log1p(float(z)) - np.log1p(float(b)))
                    < (SOURCE_GRID_BREAK_HALF_WIDTH + 1) * h
                    for b in self.break_z
                ),
                f"sample z={z:.6e} is not in the neighbourhood of any declared break",
            )

    def test_no_base_sample_was_displaced(self):
        """
        The grid only ever *adds*. The mesh guard can in principle drop a base sample that lands
        on top of a straddling pair member; on the production grid it does not, and the closest
        approach is recorded in log 11.
        """
        self.assertTrue(
            set(self.base.tolist()) <= set(self.grid.z_values.tolist()),
            "a base grid sample was displaced by the cosmology-aware construction",
        )

    def test_the_grid_is_descending_distinct_and_resolvable_by_the_datastore(self):
        values = self.grid.z_values
        self.assertTrue(
            np.all(np.diff(values) < 0.0), "the grid is not strictly descending"
        )
        self.assertEqual(len(set(values.tolist())), len(values))

        separation = np.abs(np.diff(values)) / (1.0 + values[1:])
        self.assertGreater(
            float(separation.min()),
            SOURCE_GRID_MIN_SEPARATION,
            "two samples are closer than the datastore can tell apart",
        )


class TestTheResponseGridIsASubsetAndKeepsTheProtectedPoints(unittest.TestCase):
    """
    ``main.py``'s own comment: *the response sample must remain a subset of the source sample*.
    ``winnow`` was a blind stride, which drops a protected point with probability
    1 - 1/sparseness.
    """

    @classmethod
    def setUpClass(cls):
        cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        break_z, feature_z = cosmology_feature_redshifts(
            cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
        )
        cls.grid = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=break_z,
            feature_z=feature_z,
        )
        cls.source = _to_redshift_array(cls.grid.z_values)
        protected_ids = {
            z.store_id
            for z in cls.source
            if float(z) in set(cls.grid.protected_z.tolist())
        }
        cls.protected = [z for z in cls.source if z.store_id in protected_ids]

    def test_an_unprotected_winnow_drops_protected_points(self):
        """The defect, asserted before the fix, so the fix is not vacuous."""
        response = self.source.winnow(sparseness=PRODUCTION_RESPONSE_SPARSENESS)
        kept = {z.store_id for z in response}
        missing = [z for z in self.protected if z.store_id not in kept]
        self.assertGreater(
            len(missing),
            0,
            "a blind stride happened to retain every protected point; the test is vacuous",
        )

    def test_a_protected_winnow_keeps_them_all(self):
        response = self.source.winnow(
            sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=self.protected
        )
        kept = {z.store_id for z in response}
        for z in self.protected:
            self.assertIn(z.store_id, kept)

    def test_the_response_grid_is_a_subset_of_the_source_grid(self):
        response = self.source.winnow(
            sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=self.protected
        )
        source_ids = {z.store_id for z in self.source}
        for z in response:
            self.assertIn(z.store_id, source_ids)

        # descending, distinct, and it still reaches the bottom of the source grid
        values = [float(z) for z in response]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual(float(response.min), float(self.source.min))

    def test_protect_ignores_redshifts_that_are_not_in_the_array(self):
        """``protect`` can only ever retain; it can never add."""
        outsider = redshift(store_id=10**9, z=1.234)
        response = self.source.winnow(
            sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=[outsider]
        )
        self.assertNotIn(outsider.store_id, {z.store_id for z in response})


class TestTheGridTagIdentifiesTheGrid(unittest.TestCase):
    """
    ``SourceRedshiftGrid_{len}`` labels size alone, so two different grids of equal length
    collide in the datastore. This is that collision, and its closure.
    """

    def test_two_grids_of_equal_length_used_to_collide(self):
        a = _production_base_grid()
        b = a.copy()
        b[100] = float(b[100]) * (1.0 + 1.0e-3)

        self.assertEqual(len(a), len(b))
        self.assertEqual(
            f"SourceRedshiftGrid_{len(a)}",
            f"SourceRedshiftGrid_{len(b)}",
            "the old tag scheme did not collide; this test no longer measures anything",
        )
        self.assertNotEqual(redshift_grid_digest(a), redshift_grid_digest(b))
        self.assertNotEqual(
            build_grid_tag_labels(a, a)[0], build_grid_tag_labels(b, b)[0]
        )

    def test_two_grids_differing_only_in_their_protected_set_get_different_tags(self):
        breaks = [expm1(u) for u in QCD_CROSSINGS_LOG1PZ]
        one = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=breaks,
        )
        two = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=breaks,
            standoff=2.0 * SOURCE_GRID_BREAK_STANDOFF,
        )
        self.assertEqual(len(one.z_values), len(two.z_values))
        self.assertNotEqual(
            build_grid_tag_labels(one.z_values, one.z_values)[0],
            build_grid_tag_labels(two.z_values, two.z_values)[0],
        )

    def test_the_labels_carry_the_length_and_the_digest(self):
        source = _production_base_grid()
        response = source[::-PRODUCTION_RESPONSE_SPARSENESS][::-1]
        source_label, response_label = build_grid_tag_labels(source, response)

        self.assertEqual(
            source_label,
            f"SourceRedshiftGrid_{len(source)}_{redshift_grid_digest(source)}",
        )
        self.assertEqual(
            response_label,
            f"ResponseRedshiftGrid_{len(response)}_{redshift_grid_digest(response)}",
        )
        self.assertTrue(source_label.startswith(f"SourceRedshiftGrid_{len(source)}_"))
        self.assertTrue(
            response_label.startswith(f"ResponseRedshiftGrid_{len(response)}_")
        )

    def test_the_digest_is_stable_and_order_sensitive(self):
        source = _production_base_grid()
        self.assertEqual(redshift_grid_digest(source), redshift_grid_digest(source))
        self.assertEqual(
            redshift_grid_digest(source),
            redshift_grid_digest(
                [redshift(store_id=i, z=z) for i, z in enumerate(source)]
            ),
        )
        self.assertNotEqual(
            redshift_grid_digest(source), redshift_grid_digest(source[::-1])
        )

    def test_the_tag_is_built_from_main_py(self):
        """
        ``build_grid_tag_labels`` above was extracted from ``main.py`` with ``ast`` and executed
        in isolation, so every assertion in this class is about the code the pipeline runs. This
        asserts that the pipeline does not build the labels any other way.
        """
        import ast
        from pathlib import Path

        source = (Path(__file__).parents[2] / "main.py").read_text()
        self.assertNotIn('label=f"SourceRedshiftGrid_', source)
        self.assertNotIn('label=f"ResponseRedshiftGrid_', source)

        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_grid_tag_labels"
        ]
        self.assertEqual(len(calls), 1)


class TestTheStandoffIsRefused(unittest.TestCase):
    def test_a_standoff_the_datastore_cannot_resolve_is_refused(self):
        """
        ``Quadrature/integrators/numeric_with_phase_cut.BREAK_POINT_STANDOFF = 1e-12`` is the
        right number for an ODE restart and the wrong number for a sample: the two samples
        straddling a break would be the same datastore row. The construction refuses it rather
        than producing a grid with a silent duplicate in it.
        """
        with self.assertRaises(ValueError):
            build_z_sample(
                PRODUCTION_Z_INIT,
                PRODUCTION_Z_END,
                PRODUCTION_SAMPLES_PER_LOG10Z,
                break_z=[expm1(QCD_CROSSINGS_LOG1PZ[0])],
                standoff=1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
