"""
The cosmology-aware source grid: prompt 11 of ``prompts/qcd-background-audit/``, audit section 7,
and its density, prompt 15 of the same campaign and section 10 of the verification document.

Four things are asserted here, and the first two are the ones that must not be weakened:

1. **The criterion may only ever refine.** No interval of a production grid is wider than the
   interval the uniform ``samples_per_log10z`` lattice puts at the same place, on either model.
   That is ``SOURCE_GRID_MAX_SPACING_FACTOR = 1.0``, and it is the user's decision (prompt 15
   section 1) expressed as a test: the 1.75x/2.06x saving prompt 12 measured at a coarser cap is
   real, is declined, and the whole of it is coarsening at low z, where the derivative-fit padding
   clamps and where three consumers the criterion cannot see -- the numeric ODE, the cumulative
   tables' Gauss panels and the source integral's abscissae -- all live.

2. **A cosmology that declares nothing still takes the unchanged *break-point* path.** Its grid is
   no longer bit-identical to ``numpy.logspace``, because the density is not a question about the
   equation of state and LambdaCDM's grid is refined too; what still holds, and is checked here,
   is that ``build_z_sample`` called without a ``spacing`` reproduces ``logspace`` element for
   element, that a smooth cosmology gets **no** protected points and no break neighbourhoods, and
   that the construction imports no equation-of-state module and receives no cosmology object.

3. **A cosmology that does declare something gets what prompt 10 measured it needs.** The pair
   straddling each break at the chosen standoff, the neighbourhood of each break refined, every
   protected point present, the grid still strictly descending and free of duplicates, and no
   two samples closer together than the datastore can tell apart. Prompt 15 does not disturb any
   of it: its grid is a strict *superset* of prompt 11's.

4. **The grid tag identifies the grid.** Two grids of equal length that differ in one sample get
   different tags, where before they collided; and the labels ``main.py`` actually builds come
   from ``main.py``, read with ``ast`` and executed in isolation (``main.py`` cannot be imported,
   ``CLAUDE.md``), rather than from a transcription of them.

This is a *dry* test: no Ray, no datastore, and the only cosmologies it builds are
``QCD_Cosmology`` and ``LambdaCDM``. The consumer measurement that chose the standoff and the
neighbourhood is ``docs/qcd-background-audit/source_grid_consumer_check.py``; the measurement that
scores the density against the phase-residual oracle is
``docs/qcd-background-audit/equidistributed_grid_check.py``.
"""

import unittest
from math import expm1, log10

import numpy as np
from numpy import logspace

from ComputeTargets.BackgroundModel import _cosmology_break_points
from ComputeTargets.phase_residual import (
    phase_residual_integrand,
    residual_node_range,
)
from ComputeTargets.tests.test_main_plumbing import load_main_py_functions
from CosmologyConcepts import (
    SOURCE_GRID_BREAK_HALF_WIDTH,
    SOURCE_GRID_BREAK_REFINEMENT,
    SOURCE_GRID_BREAK_STANDOFF,
    SOURCE_GRID_CONSUMER_TARGET_RAD,
    SOURCE_GRID_CROSSING_MASK_U,
    SOURCE_GRID_CUBIC_ERROR_CONST,
    SOURCE_GRID_CURVATURE_FD_STEP_U,
    SOURCE_GRID_CURVATURE_STEP_U,
    SOURCE_GRID_MAX_SPACING_FACTOR,
    SOURCE_GRID_SPLINE_EDGE_FACTOR,
    SOURCE_GRID_SPLINE_EDGE_INTERVALS,
    build_z_sample,
    redshift,
    redshift_array,
    redshift_grid_digest,
)
from CosmologyConcepts.wavenumber import (
    SOURCE_GRID_CONSTRUCTION_VERSION,
    SOURCE_GRID_MAX_REFINEMENT,
    SOURCE_GRID_MIN_SEPARATION,
)
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
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

# main.py:3292 -- the production source and response wavenumber arrays are the same fifty
# logspaced values, and one universal source grid has to serve every one of them
PRODUCTION_K_INV_MPC = np.logspace(np.log10(1.0e5), np.log10(3.0e8), 50)

_main = load_main_py_functions(
    [
        "cosmology_feature_redshifts",
        "build_grid_tag_labels",
        "pre_grid_background_proxy",
        "source_grid_spacing_profile",
    ],
    extra_globals={
        "np": np,
        "_cosmology_break_points": _cosmology_break_points,
        "redshift_grid_digest": redshift_grid_digest,
        "phase_residual_integrand": phase_residual_integrand,
        "residual_node_range": residual_node_range,
        "SOURCE_GRID_CONSUMER_TARGET_RAD": SOURCE_GRID_CONSUMER_TARGET_RAD,
        "SOURCE_GRID_CROSSING_MASK_U": SOURCE_GRID_CROSSING_MASK_U,
        "SOURCE_GRID_CUBIC_ERROR_CONST": SOURCE_GRID_CUBIC_ERROR_CONST,
        "SOURCE_GRID_CURVATURE_FD_STEP_U": SOURCE_GRID_CURVATURE_FD_STEP_U,
        "SOURCE_GRID_CURVATURE_STEP_U": SOURCE_GRID_CURVATURE_STEP_U,
        "SOURCE_GRID_SPLINE_EDGE_FACTOR": SOURCE_GRID_SPLINE_EDGE_FACTOR,
        "SOURCE_GRID_SPLINE_EDGE_INTERVALS": SOURCE_GRID_SPLINE_EDGE_INTERVALS,
    },
)
cosmology_feature_redshifts = _main["cosmology_feature_redshifts"]
build_grid_tag_labels = _main["build_grid_tag_labels"]
pre_grid_background_proxy = _main["pre_grid_background_proxy"]
source_grid_spacing_profile = _main["source_grid_spacing_profile"]


def _production_grid(cosmology, with_spacing: bool = True):
    """The grid ``main.py`` builds for this cosmology, at production geometry."""
    base = _production_base_grid()
    break_z, feature_z = cosmology_feature_redshifts(
        cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
    )
    spacing = (
        source_grid_spacing_profile(
            cosmology,
            base,
            [k / Mpc_units().Mpc for k in PRODUCTION_K_INV_MPC],
        )
        if with_spacing
        else None
    )
    return build_z_sample(
        PRODUCTION_Z_INIT,
        PRODUCTION_Z_END,
        PRODUCTION_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
        spacing=spacing,
    )


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
    and test stand-in. It carries Omega values and **no** equality-redshift properties, which
    since prompt 09 of ``prompts/background-solver-robustness`` is the only surface
    ``cosmology_feature_redshifts`` will read; the point of the first test below is that it never
    gets that far, because the whole cosmology-aware path is gated on declaring non-smoothness.
    """

    omega_m = 0.3111
    omega_r = 9.139e-05
    omega_cc = 0.6889


class _BrokenCosmology:
    """
    A cosmology that declares non-smoothness but cannot say where its own equality redshifts are.

    It is deliberately not a ``BaseCosmology`` -- ``cosmology_feature_redshifts`` duck-types
    everything it touches, and an abstract property cannot be left unimplemented on a real
    subclass anyway -- so this is the shape a nonstandard cosmology would actually arrive in.
    """

    name = "a cosmology that cannot answer"

    omega_m = 0.3111
    omega_r = 9.139e-05
    omega_cc = 0.6889

    def integration_break_points(self, z_lo: float, z_hi: float, kind: str = ""):
        return [float(u) for u in QCD_CROSSINGS_LOG1PZ]


class TestACosmologyThatDeclaresNothingDeclaresNothing(unittest.TestCase):
    """
    README section 0.5 and section 2 (g), as prompt 15 leaves them.

    **What changed, and why the change is not a weakening.** Until prompt 15 the campaign's stop
    condition was that every LambdaCDM, RadiationModel and stand-in *value* is bit-identical,
    and the source grid was part of that: prompt 11 gated its whole mechanism on the cosmology
    declaring some non-smoothness, so a smooth cosmology got the bare ``logspace``. Prompt 15
    changes the grid's **density**, and density is not a question about the equation of state --
    section 10.2 measures LambdaCDM's transfer-function row at k = 1e5 missing its storage floor
    by 7.86x, the same miss and the same place as QCD's 7.84x. So LambdaCDM's grid moves too, and
    it must.

    What survives, and is asserted here, is the *structural* half of the invariant: the
    construction is handed values and never a cosmology, it imports no equation-of-state module,
    and a cosmology that declares no break points gets no protected points, no straddling pairs
    and no break neighbourhoods -- only a refinement of the lattice it already had.
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

    def test_a_smooth_cosmology_with_a_density_gets_no_protected_points(self):
        """
        Prompt 15: LambdaCDM's *production* grid is no longer the bare lattice, because the
        density criterion applies to it too. What it still gets is nothing from the break-point
        machinery.
        """
        grid = _production_grid(
            LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        )
        self.assertEqual(len(grid.protected_z), 0)
        self.assertEqual(len(grid.breaks), 0)
        self.assertEqual(len(grid.features), 0)
        self.assertGreater(len(grid.z_values), PRODUCTION_NUM_NODES)
        self.assertTrue(
            set(_production_base_grid().tolist()) <= set(grid.z_values.tolist()),
            "the density criterion displaced a base sample on a smooth cosmology",
        )

    def test_the_construction_imports_no_equation_of_state_module(self):
        """
        ``build_z_sample`` takes values -- break redshifts, feature redshifts and a spacing
        profile -- and never a cosmology. Nothing in the module that builds the grid may reach
        into ``CosmologyModels``; the criterion that needs ``Hubble`` and ``wPerturbations`` lives
        in ``main.py``, one layer up, and hands down numbers.
        """
        import inspect

        import CosmologyConcepts.wavenumber as module

        source = inspect.getsource(module)
        self.assertNotIn("CosmologyModels.", source)
        self.assertNotIn("import CosmologyModels", source)
        self.assertNotIn("cosmology", inspect.signature(build_z_sample).parameters)

    def test_a_declared_grid_is_not_bit_identical(self):
        """The control on the control: the test above would pass on code that did nothing."""
        grid = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=[expm1(u) for u in QCD_CROSSINGS_LOG1PZ],
        )
        self.assertGreater(len(grid.z_values), PRODUCTION_NUM_NODES)


class TestTheModelIsAuthoritativeForItsEqualityRedshifts(unittest.TestCase):
    """
    ``feature_z`` is what the *cosmology* says its equality redshifts are, and nothing else.

    Prompt 09 of ``prompts/background-solver-robustness``, implementing that campaign's README
    section 7 D2 as the user decided it. Until then ``cosmology_feature_redshifts`` recomputed
    both from ``omega_m`` / ``omega_r`` / ``omega_cc``. Omega_r is a *present-day* density
    parameter, so ``1 + z_eq = Omega_m/Omega_r`` is exact only while rho_r ~ (1+z)^4 holds from
    today back to equality -- true on ``QCD_Cosmology`` to 7 ulp for the single reason that all
    of that equation of state's g_*(T) structure sits twelve orders above z_eq, and false, by far
    more than ulps and silently, on a cosmology with late entropy injection. Only the model knows
    whether the closed form is valid for it, so ``BaseCosmology`` declares the obligation and each
    model answers.

    The two tests below are the ones that distinguish the trees: both fail on the tree that
    computed the closed form here, and the failure output is quoted in
    ``prompts/background-solver-robustness/logs/09-make-the-model-authoritative.md``.
    """

    @classmethod
    def setUpClass(cls):
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        cls.break_z, cls.feature_z = cosmology_feature_redshifts(
            cls.cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
        )

    def test_feature_z_is_the_models_answer_and_not_the_closed_form(self):
        """
        Each entry of ``feature_z`` is the model's property **as the same float**, and the
        matter-radiation entry is *not* the closed form.

        The second half is what gives this test teeth. The two agree to 7 ulp on this cosmology,
        which is 3.2e-12 in z and far inside a grid interval, so every assertion that compares
        them at any tolerance a human would write passes either way; only bit-identity can tell
        which quantity actually reached the grid. That distinction is not cosmetic -- it is one
        sample of 1,996 and therefore a different ``BackgroundModel`` source-grid digest.
        """
        self.assertEqual(len(self.feature_z), 2)

        for index, attribute in (
            (0, "z_matter_radiation_equality"),
            (1, "z_matter_lambda_equality"),
        ):
            with self.subTest(attribute=attribute):
                supplied = float(getattr(self.cosmology, attribute))
                self.assertEqual(
                    self.feature_z[index].hex(),
                    supplied.hex(),
                    msg=f"feature_z[{index}] = {self.feature_z[index]!r} is not the model's own "
                    f"{attribute} = {supplied!r}; something is computing this redshift instead "
                    "of asking for it",
                )

        closed_form = float(self.cosmology.omega_m / self.cosmology.omega_r - 1.0)
        self.assertNotEqual(
            self.feature_z[0].hex(),
            closed_form.hex(),
            msg=f"feature_z[0] = {self.feature_z[0]!r} is the radiation-domination closed form "
            f"Omega_m/Omega_r - 1 = {closed_form!r}, not the model's own solve. The two differ "
            "by 7 ulp on this cosmology and the closed form is the one that is only accidentally "
            "right.",
        )
        self.assertAlmostEqual(
            (self.feature_z[0] - closed_form) / np.spacing(closed_form), 7.0, places=6
        )

        # the matter-Lambda pair is the same double from either route -- rho_m/rho_Lambda has no
        # temperature dependence at all -- which is why exactly one sample of the grid moves
        self.assertEqual(
            self.feature_z[1].hex(),
            float(
                pow(self.cosmology.omega_cc / self.cosmology.omega_m, 1.0 / 3.0) - 1.0
            ).hex(),
        )

    def test_a_cosmology_that_cannot_answer_raises_instead_of_falling_back(self):
        """
        A cosmology that declares break points but supplies no equality redshift **raises**, and
        the error names it and says which of the two was missing.

        ``_BrokenCosmology`` carries ``omega_m``, ``omega_r`` and ``omega_cc``, so a fallback to
        the closed form would succeed here and hand the grid a number nothing has vouched for.
        That is the one mistake this change exists to prevent, and it is the one that looks most
        like care, so it is asserted rather than left to review.
        """
        with self.assertRaises(RuntimeError) as caught:
            cosmology_feature_redshifts(
                _BrokenCosmology(), PRODUCTION_Z_END, PRODUCTION_Z_INIT
            )

        message = str(caught.exception)
        print(f"\n  a cosmology that cannot answer:\n    {message}")
        for expected in (
            "a cosmology that cannot answer",
            "_BrokenCosmology",
            "z_matter_radiation_equality",
        ):
            self.assertIn(
                expected,
                message,
                msg=f"a cosmology that cannot supply its equality redshifts must be named in the "
                f"error, along with what it could not supply; expected {expected!r} in: {message}",
            )


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

        # and the two equality redshifts, which the *model* supplies -- z_matter_radiation_equality
        # and z_matter_lambda_equality, answered from the bracketed solve its constructor runs
        # (prompt 09 of prompts/background-solver-robustness). The matter-radiation literal is
        # that solve's value and not the Omega_m/Omega_r closed form, which sits 7 ulp below it;
        # the matter-Lambda pair is the same double either way, since neither side sees T(z).
        self.assertEqual(len(self.feature_z), 2)
        self.assertAlmostEqual(self.feature_z[0], 3406.6689742499511, places=6)
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


class TestTheCurvatureCriterionSetsTheBaseDensity(unittest.TestCase):
    """
    Prompt 15 of ``prompts/qcd-background-audit``, and section 10 of the verification document.

    The accuracy the criterion buys is scored against the phase-residual oracle in
    ``docs/qcd-background-audit/equidistributed_grid_check.py``, which needs a background model
    and four minutes. What is asserted here is what the *construction* must guarantee whatever the
    criterion says, and the first of them is the user's decision.
    """

    @classmethod
    def setUpClass(cls):
        cls.base = _production_base_grid()
        cls.grids, cls.prompt11 = {}, {}
        for key, cosmology in (
            (
                "LambdaCDM",
                LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018()),
            ),
            (
                "QCD",
                QCD_Cosmology(
                    store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
                ),
            ),
        ):
            cls.grids[key] = _production_grid(cosmology)
            cls.prompt11[key] = _production_grid(cosmology, with_spacing=False)

    def test_no_interval_is_coarser_than_the_uniform_lattice_puts_there(self):
        """
        **The cap, asserted directly, and the one test here that must not be weakened.**
        ``SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`` is "never coarser than today, anywhere", and it
        is the user's decision (prompt 15 section 1) rather than a numerical convenience: a
        criterion built on the consumer's phase alone is a lower bound on the density and never an
        upper one, because the numeric ODE's samples, the four cumulative tables' Gauss panels and
        the source integral's abscissae all live on this grid and none of them is measured
        anywhere.
        """
        self.assertEqual(SOURCE_GRID_MAX_SPACING_FACTOR, 1.0)

        u_base = np.log1p(np.sort(self.base))
        for key, grid in self.grids.items():
            u = np.log1p(np.sort(grid.z_values))
            for i in range(len(u) - 1):
                mid = 0.5 * (u[i] + u[i + 1])
                j = min(max(int(np.searchsorted(u_base, mid)) - 1, 0), len(u_base) - 2)
                self.assertLessEqual(
                    u[i + 1] - u[i],
                    (u_base[j + 1] - u_base[j]) * (1.0 + 1.0e-12),
                    f"{key}: the interval at z = {expm1(mid):.6e} is coarser than the uniform "
                    "lattice puts there",
                )

    def test_the_prompt_11_grid_survives_element_for_element(self):
        """
        The density is refined *around* prompt 11's mechanism, never through it: the criterion
        runs after the straddling pairs, the feature redshifts and the break neighbourhoods have
        been placed, so every sample prompt 11 placed is still there and still where it was.
        """
        for key, grid in self.grids.items():
            before = set(self.prompt11[key].z_values.tolist())
            self.assertTrue(
                before <= set(grid.z_values.tolist()),
                f"{key}: the density criterion displaced a sample prompt 11 placed",
            )
            self.assertGreater(len(grid.z_values), len(self.prompt11[key].z_values))
            self.assertEqual(
                grid.protected_z.tobytes(), self.prompt11[key].protected_z.tobytes()
            )

    def test_the_protected_points_are_still_present_and_still_straddled(self):
        grid = self.grids["QCD"]
        values = set(grid.z_values.tolist())
        self.assertEqual(len(grid.protected_z), 2 * 3 + 2)
        for z in grid.protected_z:
            self.assertIn(float(z), values)

        delta_log10 = (log10(PRODUCTION_Z_INIT) - log10(PRODUCTION_Z_END)) / (
            PRODUCTION_NUM_NODES - 1
        )
        expected = SOURCE_GRID_BREAK_STANDOFF * (pow(10.0, delta_log10) - 1.0)
        protected = grid.protected_z
        for z_break in grid.breaks:
            above = protected[protected > z_break].min()
            below = protected[protected < z_break].max()
            self.assertAlmostEqual(
                (above - z_break) / (1.0 + z_break), expected, places=12
            )
            self.assertAlmostEqual(
                (z_break - below) / (1.0 + z_break), expected, places=12
            )

    def test_the_grid_is_still_descending_distinct_and_resolvable(self):
        for key, grid in self.grids.items():
            values = grid.z_values
            self.assertTrue(np.all(np.diff(values) < 0.0), f"{key}: not descending")
            self.assertEqual(len(set(values.tolist())), len(values))
            separation = np.abs(np.diff(values)) / (1.0 + values[1:])
            self.assertGreater(float(separation.min()), SOURCE_GRID_MIN_SEPARATION)

    def test_the_derivative_pad_clamp_does_not_bind(self):
        """
        ``[03-derivative-pad-clamp-on-coarse-grids]``: ``_build_derivative_fit_grid`` clamps its
        padding once the lowest grid interval exceeds ``-log(0.9)/12 = 8.7800e-03`` in u. That is
        the measured consequence of choosing the 1x cap -- at every coarser cap on prompt 12's
        ladder it binds -- and it should fail loudly if a later prompt coarsens.
        """
        clamp = -np.log(0.9) / 12.0
        self.assertAlmostEqual(clamp, 8.7800e-03, places=7)
        for key, grid in self.grids.items():
            u = np.log1p(np.sort(grid.z_values))
            self.assertLess(
                float(u[1] - u[0]),
                clamp,
                f"{key}: the lowest grid interval would clamp the derivative-fit padding",
            )

    def test_the_response_grid_is_still_a_decimation_of_the_source_grid(self):
        for key, grid in self.grids.items():
            source = _to_redshift_array(grid.z_values)
            protected_values = set(grid.protected_z.tolist())
            protected = [z for z in source if float(z) in protected_values]
            response = source.winnow(
                sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=protected
            )
            source_ids = {z.store_id for z in source}
            for z in response:
                self.assertIn(z.store_id, source_ids)
            self.assertEqual(float(response.min), float(source.min))
            for z in protected:
                self.assertIn(z.store_id, {r.store_id for r in response})

    def test_the_response_grids_are_the_ones_prompt_15_recorded(self):
        """A response grid of 44 samples over twenty decades would not be usable (section 10.6);
        these are at the safe end of that ladder, and the numbers are the record."""
        sizes = {}
        for key, grid in self.grids.items():
            source = _to_redshift_array(grid.z_values)
            protected_values = set(grid.protected_z.tolist())
            protected = [z for z in source if float(z) in protected_values]
            sizes[key] = len(
                source.winnow(
                    sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=protected
                )
            )
        self.assertEqual(sizes["QCD"], 175)
        self.assertEqual(sizes["LambdaCDM"], 149)

    def test_an_absurd_spacing_profile_is_refused_rather_than_built(self):
        """
        A grid of unknown size is worse than a refusal. The largest subdivision the criterion asks
        for on the production envelope is 4.
        """
        u = np.log1p(np.sort(self.base))
        with self.assertRaises(ValueError):
            build_z_sample(
                PRODUCTION_Z_INIT,
                PRODUCTION_Z_END,
                PRODUCTION_SAMPLES_PER_LOG10Z,
                spacing=(u, np.full(len(u), 1.0e-6)),
            )
        self.assertEqual(SOURCE_GRID_MAX_REFINEMENT, 32)

    def test_a_malformed_spacing_profile_is_refused(self):
        u = np.log1p(np.sort(self.base))
        with self.assertRaises(ValueError):
            build_z_sample(
                PRODUCTION_Z_INIT,
                PRODUCTION_Z_END,
                PRODUCTION_SAMPLES_PER_LOG10Z,
                spacing=(u, np.full(len(u) - 1, 1.0)),
            )
        with self.assertRaises(ValueError):
            build_z_sample(
                PRODUCTION_Z_INIT,
                PRODUCTION_Z_END,
                PRODUCTION_SAMPLES_PER_LOG10Z,
                spacing=(u[::-1], np.full(len(u), 1.0)),
            )

    def test_an_unconstrained_profile_changes_nothing(self):
        """The criterion's own null: a profile that permits any spacing reproduces the grid built
        without one, element for element, on both models."""
        u = np.log1p(np.sort(self.base))
        for key, cosmology in (
            (
                "LambdaCDM",
                LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018()),
            ),
            (
                "QCD",
                QCD_Cosmology(
                    store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
                ),
            ),
        ):
            break_z, feature_z = cosmology_feature_redshifts(
                cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
            )
            grid = build_z_sample(
                PRODUCTION_Z_INIT,
                PRODUCTION_Z_END,
                PRODUCTION_SAMPLES_PER_LOG10Z,
                break_z=break_z,
                feature_z=feature_z,
                spacing=(u, np.full(len(u), np.inf)),
            )
            self.assertEqual(
                grid.z_values.tobytes(), self.prompt11[key].z_values.tobytes()
            )


class TestThePreGridCriterionNeedsNoBackgroundModel(unittest.TestCase):
    """
    The premise of the whole criterion: ``populate_z_sample`` runs before any ``BackgroundModel``
    exists, so the density may be computed from what the *cosmology* supplies pointwise and
    nothing else. Section 10.1 licenses the substitution by measurement; this asserts that the
    code actually makes it.
    """

    @classmethod
    def setUpClass(cls):
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        cls.proxy = pre_grid_background_proxy(cls.cosmology)

    def test_the_proxy_supplies_exactly_what_the_phase_residual_asks_for(self):
        for name in (
            "Hubble",
            "epsilon",
            "d_epsilon_dz",
            "wPerturbations",
            "d_wPerturbations_dz",
        ):
            self.assertTrue(hasattr(self.proxy.functions, name))
        self.assertIs(self.proxy.cosmology, self.cosmology)

        # and it is enough to build the integrand the criterion differentiates
        integrand = phase_residual_integrand(self.proxy, 1.0e5 / Mpc_units().Mpc, "Tk")
        self.assertIsInstance(integrand(1.0e6), float)

    def test_epsilon_agrees_with_a_central_difference_of_the_cosmology(self):
        """Section 10.1's second licence check, in miniature and away from a declared crossing:
        max 3.9e-09 on QCD_Cosmology. Here, over 40 probes spanning twelve decades."""
        crossings = np.asarray(
            [float(u) for u in _cosmology_break_points(self.cosmology, 0.1, 1.0e16)]
        )
        worst = 0.0
        for u in np.linspace(3.0, 33.0, 40):
            if any(abs(u - b) < 0.5 for b in crossings):
                continue
            z = float(np.expm1(u))
            d = 1.0e-6
            reference = (
                np.log(self.cosmology.Hubble(float(np.expm1(u + d))))
                - np.log(self.cosmology.Hubble(float(np.expm1(u - d))))
            ) / (2.0 * d)
            worst = max(worst, abs(self.proxy.functions.epsilon(z) / reference - 1.0))
        self.assertLess(worst, 1.0e-07)

    def test_the_spacing_profile_is_capped_by_nothing_and_ascending(self):
        base = _production_base_grid()
        u_profile, h_profile = source_grid_spacing_profile(
            self.cosmology, base, [1.0e5 / Mpc_units().Mpc]
        )
        self.assertEqual(len(u_profile), len(base))
        self.assertTrue(np.all(np.diff(u_profile) > 0.0))
        self.assertTrue(np.all(h_profile > 0.0))
        # above the Liouville-Green band of the only wavenumber supplied, nothing constrains
        # anything
        self.assertTrue(np.isinf(h_profile[-1]))


class TestTheConstructionVersionNamesThisAlgorithm(unittest.TestCase):
    """
    Prompt 14 of ``prompts/qcd-background-audit``. ``SOURCE_GRID_CONSTRUCTION_VERSION`` names the
    *algorithm* that builds the grid, and a bump is the only signal a datastore gets that the
    algorithm has moved: nothing about how a grid was constructed is recoverable from the stored
    samples. Version 1 is prompt 11's construction, and the three constants below are what it is.

    **A prompt that changes one of these and not the version breaks here.** That is the point: it
    is cheaper to update this test deliberately than to discover, later, that objects computed by
    two different constructions were served for one another. The digest cannot catch it, because
    two constructions can agree on a grid for one cosmology and differ for another.
    """

    def test_version_2_is_prompt_15s_construction(self):
        self.assertEqual(SOURCE_GRID_CONSTRUCTION_VERSION, 2)
        # version 1's constants, unchanged: prompt 15 refines around them, not through them
        self.assertEqual(SOURCE_GRID_BREAK_STANDOFF, 0.25)
        self.assertEqual(SOURCE_GRID_BREAK_HALF_WIDTH, 5)
        self.assertEqual(SOURCE_GRID_BREAK_REFINEMENT, 2)
        # and version 2's own
        self.assertEqual(SOURCE_GRID_MAX_SPACING_FACTOR, 1.0)
        self.assertEqual(SOURCE_GRID_CUBIC_ERROR_CONST, 1.0 / 384.0)
        self.assertEqual(SOURCE_GRID_CURVATURE_STEP_U, 1.0e-3)
        self.assertEqual(SOURCE_GRID_CURVATURE_FD_STEP_U, 1.0e-4)
        self.assertEqual(SOURCE_GRID_CONSUMER_TARGET_RAD, 1.0e-6)
        self.assertEqual(SOURCE_GRID_SPLINE_EDGE_INTERVALS, 3)
        self.assertEqual(SOURCE_GRID_SPLINE_EDGE_FACTOR, 10.0)

    def test_the_production_grids_are_the_ones_the_campaign_recorded(self):
        """
        The production grids, by length and by digest over their exact bits.

        Three of them now: the bare lattice (unmoved since the campaign began, and the control
        that ``build_z_sample`` without a ``spacing`` still reproduces ``logspace``), prompt
        11/14's break-point grid (unmoved -- prompt 15 does not disturb it), and the grid
        ``main.py`` builds, which is the first two plus the density the criterion asks for.
        """
        cosmology = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        break_z, feature_z = cosmology_feature_redshifts(
            cosmology, PRODUCTION_Z_END, PRODUCTION_Z_INIT
        )
        qcd = build_z_sample(
            PRODUCTION_Z_INIT,
            PRODUCTION_Z_END,
            PRODUCTION_SAMPLES_PER_LOG10Z,
            break_z=break_z,
            feature_z=feature_z,
        )
        self.assertEqual(len(qcd.z_values), 1773)
        # 303f9ce7 until prompt 09 of prompts/background-solver-robustness made the model
        # authoritative for its own equality redshifts: feature_z[0] is now the constructor's
        # bracketed solve rather than Omega_m/Omega_r - 1, 7 ulp above it, which moves exactly one
        # sample of this grid and therefore its digest. Nothing else about the grid changed --
        # same length, same break points, same lattice.
        self.assertEqual(redshift_grid_digest(qcd.z_values), "81c6e682")

        smooth = build_z_sample(
            PRODUCTION_Z_INIT, PRODUCTION_Z_END, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        self.assertEqual(len(smooth.z_values), PRODUCTION_NUM_NODES)
        self.assertEqual(redshift_grid_digest(smooth.z_values), "0960e169")

        production_qcd = _production_grid(cosmology)
        self.assertEqual(len(production_qcd.z_values), 1996)
        # a2c32f67 until prompt 09, for the same single moved sample as above; the count is
        # unchanged, and LambdaCDM's digest below is unchanged, because its closed form *is* its
        # answer and it never reaches the feature path at all
        self.assertEqual(redshift_grid_digest(production_qcd.z_values), "4849552b")

        production_lcdm = _production_grid(
            LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        )
        self.assertEqual(len(production_lcdm.z_values), 1778)
        self.assertEqual(redshift_grid_digest(production_lcdm.z_values), "60a3205a")


if __name__ == "__main__":
    unittest.main()
