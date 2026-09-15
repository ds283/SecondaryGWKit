"""
Retiring the samples-per-decade tag and the arguments it fed: prompt 16 of
``prompts/qcd-background-audit``.

Prompt 15 replaced the source grid's uniform density with a measured curvature criterion, capped
so that no interval is ever coarser than the uniform lattice would have put there
(``SOURCE_GRID_MAX_SPACING_FACTOR = 1.0``). After that commit ``SourceSamplesPerLog10ZTag`` was
false as a description -- it labels stored objects ``SourceSamplesPerLog10Z_100`` as though the
grid had 100 samples per decade of z, when QCD carries 1,996 samples and LambdaCDM 1,778 against
the 1,773 and 1,732 of the lattice that number now merely *bounds* -- and the two ``delta_logz``
arguments it fed at ``main.py:1203`` and ``:1820`` (line numbers before this commit) were already
established as vestigial by prompt 14: ``numeric_with_phase_cut`` still accepts ``delta_logz`` and
still hands it to ``NumericIntegrationSupervisor``, but the oscillation-resolution diagnostic that
constant used to feed (``NumericIntegrationSupervisor.report_wavelength``) has not been called by
either numeric integrator's ``RHS`` since prompt 11 of ``prompts/GkTk-remedial`` -- the diagnostic
now runs after the solve, against the *actual* spacing of the returned sample grid
(``scan_sample_grid_for_unresolved_osc``).

Four things are asserted here, in the order prompt 16 section 3 states them:

1. **The tag is gone from every writer.** By source-text scan over ``main.py`` (following prompt
   14's ``TestMainPyNamesItsRun`` shape): no tag list mentions ``SourceSamplesPerLog10ZTag``,
   nothing constructs the ``SourceSamplesPerLog10Z_<n>`` label, and the two ``delta_logz=``
   keyword arguments are gone. ``--source-samples-log10z`` itself is *not* retired -- its job
   changed rather than ended, from setting the grid's density to setting the base lattice the cap
   is measured against -- so its declaration and its four surviving uses must remain.

2. **The grid did not move.** This prompt touches no construction code, so
   ``ComputeTargets/tests/test_source_grid.py::TestTheConstructionVersionNamesThisAlgorithm``
   (unmodified by this prompt) is the proof: it pins the production grids to 1,996 samples /
   digest ``a2c32f67`` (QCD) and 1,778 / ``60a3205a`` (LambdaCDM), element for element, and
   ``SOURCE_GRID_CONSTRUCTION_VERSION`` to 2. Repeated here only as direct pins on the two version
   constants this prompt must not move.

3. **Removing a tag from a lookup broadens it, and a pre-prompt-16 store still resolves.** Every
   tagged ``object_get`` in the tree -- ``BackgroundModel``, ``TkNumericIntegration``,
   ``GkNumericIntegration`` and the rest -- filters by joining, once per requested tag, on a
   table that associates the candidate row with that tag's serial. There is no companion check
   that the row carries *only* the requested tags, so a tag no longer requested is simply never
   joined on: dropping ``SourceSamplesPerLog10ZTag`` from a caller's list can only admit rows that
   used to be excluded by it, never exclude rows that used to pass. This is demonstrated against
   ``BackgroundModel``'s factory, using the real-SQL harness prompt 14 built in
   ``test_run_identity.py``, because the join idiom is the same in every tagged factory and the
   claim is about that idiom, not about ``BackgroundModel`` specifically. **Consequence: this
   commit carries no datastore regeneration** -- every row any pre-prompt-16 run wrote still
   carries ``SourceSamplesPerLog10ZTag`` as archival history, and still satisfies the shorter
   post-prompt-16 query built from the same run label and construction version.

4. **Neither version constant moved.** ``SOURCE_GRID_CONSTRUCTION_VERSION`` is still 2 -- nothing
   about how the grid is built changes here, only what is said about it -- and
   ``LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION`` is still 6, since no ``CosmologyModels/``
   file is touched.

Nothing here needs Ray or a datastore: item 3 runs against a real in-memory SQLite database built
from the production factories' own ``register()``, exactly as ``test_run_identity.py`` does.
"""

import unittest
from pathlib import Path

from ComputeTargets.tests.test_run_identity import _Schema
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS

MAIN_PY = Path(__file__).parents[2] / "main.py"


class TestTheTagIsGoneFromEveryWriter(unittest.TestCase):
    """Prompt 16 section 3 item 1."""

    def setUp(self):
        self.source = MAIN_PY.read_text()

    def test_the_tag_name_and_label_are_not_referenced(self):
        # a single substring check covers every one of the 28 call sites prompt 16 names (the
        # declaration, the label construction and every tags=[...] use): none of them can mention
        # the retired tag without the literal name "SourceSamplesPerLog10Z" appearing somewhere in
        # the source
        self.assertNotIn("SourceSamplesPerLog10Z", self.source)

    def test_the_delta_logz_arguments_are_gone(self):
        # numeric_with_phase_cut still accepts delta_logz (Quadrature/integrators, unchanged by
        # this prompt), but main.py no longer has anything to pass it: the diagnostic it used to
        # feed has taken no input from it since prompts/GkTk-remedial prompt 11
        self.assertNotIn("delta_logz", self.source)

    def test_the_switch_itself_survives_with_corrected_help_text(self):
        # the user's decision (prompt 16 section 1 item 1): the switch stays, its job changed
        self.assertIn('"--source-samples-log10z"', self.source)

        # the old help text described the number as the density; it is now the base lattice a
        # cap is measured against, and the old sentence must not survive verbatim
        self.assertNotIn(
            "specify number of z-sample points per log10(z) for the source term",
            self.source,
        )

    def test_the_switch_still_feeds_the_base_lattice_at_all_four_sites(self):
        # the four surviving uses prompt 16 section 1 item 1 names: the base grid construction,
        # the production source grid construction, the status print, and the argparse default
        self.assertEqual(
            self.source.count("source_samples_per_log10z"),
            4,
            "expected exactly the four surviving uses prompt 16 section 1 names -- a count that "
            "moves means either a use was missed or a new one was introduced",
        )


class TestNeitherVersionConstantMoved(unittest.TestCase):
    """Prompt 16 section 3 item 4 (and section 4's acceptance table)."""

    def test_the_construction_version_is_unchanged(self):
        self.assertEqual(SOURCE_GRID_CONSTRUCTION_VERSION, 2)

    def test_the_representation_version_is_unchanged(self):
        self.assertEqual(LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION, 6)


class TestRemovingATagFromTheLookupBroadensIt(unittest.TestCase):
    """
    Prompt 16 section 2 item 2 and section 3 item 3 -- the one question in the prompt that is not
    mechanical, and the test that carries the answer rather than a docstring's assertion of it.
    """

    def setUp(self):
        self.db = _Schema()
        self.addCleanup(self.db.close)

        self.grid = self.db.z_sample([100.0, 10.0, 1.0])
        self.run_tag = self.db.tag("Run_default")
        # the tag a pre-prompt-16 run would have attached, spelled the way main.py built it
        self.retired_tag = self.db.tag("SourceSamplesPerLog10Z_100")

    def test_a_row_carrying_the_retired_tag_still_resolves_without_it(self):
        written = self.db.write_model(
            self.grid, tags=[self.run_tag, self.retired_tag], label="pre-prompt-16"
        )

        # post-prompt-16 main.py queries with a list that no longer names the retired tag; the
        # row still carries it as harmless archival history
        read = self.db.read_model(self.grid, tags=[self.run_tag])

        self.assertTrue(read.available)
        self.assertEqual(read.store_id, written.store_id)

    def test_a_tag_the_row_lacks_is_still_required(self):
        """
        The converse, so the first result is not mistaken for "tags are not checked at all": a
        row missing a tag the caller *does* still ask for continues to miss, exactly as before
        this prompt.
        """
        self.db.write_model(self.grid, tags=[self.run_tag], label="no-retired-tag")

        other_tag = self.db.tag("SomeUnrelatedTag")
        read = self.db.read_model(self.grid, tags=[self.run_tag, other_tag])

        self.assertFalse(read.available)


if __name__ == "__main__":
    unittest.main()
