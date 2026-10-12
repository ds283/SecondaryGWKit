"""
Tests for prompt 09b of ``prompts/datastore-integrity`` (board items K6, and K7's set-valued
parent): a ``GkSource`` is keyed on the set of Gk numeric and WKB rows it is assembled from
(docs/datastore-integrity-audit.md §2.4), and the store inventory resolves that set.

A ``GkSource`` for ``(k, z_response)`` is assembled, for each ``z_source`` of its grid, from the
value at ``z_response`` of one ``GkNumericIntegration``, one ``GkWKBIntegration``, or both. Its
lookup named none of them. Now ``GkSource_parents`` records the set, one row per ``z_source``, and
``parents_digest``, the digest of that set, is in ``build()``'s filter.

The prompt's §3 tests, by class:

  1. ``TestKeyedOnItsSet`` -- two parent sets differing in one ``z_source``'s row give two rows,
     each found by its own set.
  2. ``TestTheRecordAgreesWithTheKey`` -- ``parents_digest`` is the digest of the
     ``GkSource_parents`` rows written, which are the set assembled, not the set asked for; and
     ``store()`` refuses an empty set (the user's decision on the sketch, 2026-09-30).
  3. ``TestTheValueReadsHandBackTheirParent`` -- ``GkNumericValue.read_batch`` and
     ``GkWKBValue.read_batch`` attach their parent's serial, and their serial-only form
     (``parents_only``) returns the ``(z_source, parent)`` pairs.
  4. ``TestTheReaderMode`` and ``TestTheValueModelRead`` -- ``labels_only`` and
     ``GkSourceValue._build_impl_model`` refuse two matches, naming their digests and the
     ``z_source``s at which their parents differ; one match is served without being given the set.
  5. ``TestTheInventoryResolvesTheSet`` -- two ``GkSource``s differing only in one parent are two
     records with different identities; an unresolvable member, an orphan member row and an empty
     set are named problems.
  7. ``TestTheWKBValueReadsOnACurrentStore`` -- the WKB value reads, on the current schema's
     tables, find their rows.
  9. ``TestTheForeignKeys`` -- ``GkSource_parents``' four keys under ``PRAGMA
     foreign_key_check``.

Test 8 (``main.py``), the compute-time assertion and the pipeline's own logic end to end are in
``ComputeTargets/tests/test_gksource_parent_set_main.py``.

No Ray, nothing under ``var/``: the schema is built in memory with the real factories, and the
inventory tests write stores in temporary directories with ``build_full_store``.
"""

import tempfile
import unittest
from pathlib import Path

import ray
import sqlalchemy as sqla
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import GkSource, GkWKBIntegration
from CosmologyConcepts import redshift, redshift_array
from ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericValue_factory,
)
from ObjectFactories.GkSource import (
    GkSource_parents_digest,
    sqla_GkSource_factory,
    sqla_GkSourceValue_factory,
)
from ObjectFactories.GkWKBIntegration import sqla_GkWKBValue_factory
from datastorekit.store_inventory import read_inventory
from ObjectFactories.tests.real_store_fixtures import (
    build_full_store,
    full_rows,
    vary_row,
    with_rows,
)
from ObjectFactories.tests.test_wkb_numeric_parent_key import (
    MODEL,
    Store,
    _exit_time,
    _Ref,
)
from ComputeTargets import GkNumericIntegration
import ComputeTargets.phase_residual as phase_residual
from MetadataConcepts import tolerance
from config.datastore import factories as _factories

K_EXIT = _exit_time(serial=11, z_exit=1.0e8)
TAG = _Ref(1)
OTHER_TAG = _Ref(2)
ATOL = tolerance(store_id=1, log10_tol=-10.0)
OTHER_ATOL = tolerance(store_id=3, log10_tol=-12.0)
RTOL = tolerance(store_id=2, log10_tol=-9.0)

# the source grid: three z_sources above one z_response
Z_SOURCES = {
    21: redshift(store_id=21, z=1.0e6, is_source=True, is_response=False),
    22: redshift(store_id=22, z=1.0e5, is_source=True, is_response=False),
    23: redshift(store_id=23, z=1.0e4, is_source=True, is_response=False),
}
Z_RESPONSE = redshift(store_id=24, z=10.0, is_source=False, is_response=True)
OTHER_Z_RESPONSE = redshift(store_id=25, z=5.0, is_source=False, is_response=True)
Z_SAMPLE = redshift_array([Z_SOURCES[s] for s in (21, 22, 23)])

# Gk rows, each serial held by its own table only (so that a key declared to the wrong table
# cannot be satisfied by coincidence, test 9): numeric 41 and 43 at z_source 21 (43 at another
# atol, so a lookup at ATOL sees 41 only), 42 at z_source 22; WKB 51 at z_source 22 seeded from 42,
# 52 at z_source 23 pure WKB
GK_NUMERIC = {41: (21, ATOL), 42: (22, ATOL), 43: (21, OTHER_ATOL)}
GK_WKB = {51: (22, 42), 52: (23, None)}

# two parent sets differing in one z_source's row: z_source 21's numeric row
SET_A = ((21, 41, None), (22, 42, 51), (23, None, 52))
SET_B = ((21, 43, None), (22, 42, 51), (23, None, 52))


def _gk_store() -> Store:
    s = Store()
    s.insert("version", serial=1, label="test_gksource_parent_set")
    for z in list(Z_SOURCES.values()) + [Z_RESPONSE, OTHER_Z_RESPONSE]:
        s.insert(
            "redshift",
            serial=z.store_id,
            z=z.z,
            source=z.is_source,
            response=z.is_response,
        )
    s.insert("store_tag", serial=1, label="T")
    s.insert("store_tag", serial=2, label="U")
    s.insert("IntegrationSolver", serial=1, label="solver", stepping=0)
    for t in (ATOL, RTOL, OTHER_ATOL):
        s.insert("tolerance", serial=t.store_id, log10_tol=t.log10_tol)
    s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5)
    s.insert(
        "BackgroundModel",
        serial=MODEL.store_id,
        cosmology_type=1,
        cosmology_serial=1,
        tau_gauss_order=4,
        cs_tau_gauss_order=4,
        friction_F_gauss_order=4,
        solver_serial=1,
        z_init_serial=21,
        z_samples=0,
        source_grid_digest="fixture",
        source_grid_construction="fixture",
        validated=True,
    )
    s.insert(
        "wavenumber_exit_time",
        serial=K_EXIT.store_id,
        wavenumber_serial=1,
        cosmology_type=1,
        cosmology_serial=1,
        atol_serial=1,
        rtol_serial=2,
        z_exit=K_EXIT.z_exit,
    )
    for serial, (z_source, atol) in GK_NUMERIC.items():
        s.insert(
            "GkNumericIntegration",
            serial=serial,
            label=f"gknum-{serial}",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            atol_serial=atol.store_id,
            rtol_serial=RTOL.store_id,
            break_point_kind=GkNumericIntegration.BREAK_POINT_KIND,
            solver_serial=1,
            z_source_serial=z_source,
            z_min_serial=OTHER_Z_RESPONSE.store_id,
            z_samples=1,
            validated=True,
        )
        s.insert("GkNumeric_tags", integration_serial=serial, tag_serial=1)
        s.insert(
            "GkNumericValue",
            serial=100 + serial,
            integration_serial=serial,
            z_serial=Z_RESPONSE.store_id,
            G=0.5,
            Gprime=0.25,
            analytic_G_rad=0.5,
            analytic_Gprime_rad=0.25,
        )
    for serial, (z_source, numeric) in GK_WKB.items():
        s.insert(
            "GkWKBIntegration",
            serial=serial,
            label=f"gkwkb-{serial}",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            numeric_serial=numeric,
            rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
            solver_serial=1,
            z_source_serial=z_source,
            z_min_serial=OTHER_Z_RESPONSE.store_id,
            z_samples=1,
            z_init=1.0e4,
            sin_coeff=0.0,
            cos_coeff=1.0,
            G_init=0.0,
            Gprime_init=1.0,
            validated=True,
        )
        s.insert("GkWKB_tags", wkb_serial=serial, tag_serial=1)
        s.insert(
            "GkWKBValue",
            serial=100 + serial,
            wkb_serial=serial,
            z_serial=Z_RESPONSE.store_id,
            # sqrt(H_ratio / omega_WKB) (cos_coeff cos theta + sin_coeff sin theta) = G_WKB, so
            # that assemble_GkSource_values finds the phase consistent
            H_ratio=2.0,
            theta_mod_2pi=0.0,
            theta_div_2pi=0,
            omega_WKB_sq=4.0,
            G_WKB=1.0,
            analytic_G_rad=0.5,
            analytic_Gprime_rad=0.25,
        )
    s.conn.commit()
    return s


def _call(s: Store, factory, name: str, payload):
    return factory.build(
        payload, s.conn, s.tables[name], s.inserters[name], s.tables, s.inserters
    )


def source_lookup(s: Store, z_response=Z_RESPONSE, tags=(TAG,), **extra):
    """The lookup main.py makes: the labels, the tags, and whatever ``extra`` names."""
    payload = {
        "model": MODEL,
        "k": K_EXIT,
        "z_response": z_response,
        "z_sample": None,
        "tags": list(tags),
        "_do_not_populate": True,
    }
    payload.update(extra)
    return _call(s, sqla_GkSource_factory, "GkSource", payload)


def store_source(
    s: Store, parents, assembled=None, z_response=Z_RESPONSE, tags=(TAG,), values=()
):
    """
    A validated GkSource stored through the real factory, keyed on ``parents`` and recording
    ``assembled`` (by default the same set), as the target's ``store()`` leaves it once
    ``assemble_GkSource_values`` has returned. Returns the object.
    """
    obj = GkSource(
        payload=None,
        model=MODEL,
        k=K_EXIT,
        z_response=z_response,
        z_sample=Z_SAMPLE,
        tags=list(tags),
        parents=parents,
    )
    obj._values = list(values)
    obj._assembled_parents = parents if assembled is None else assembled
    sqla_GkSource_factory.store(
        obj,
        s.conn,
        s.tables["GkSource"],
        s.inserters["GkSource"],
        s.tables,
        s.inserters,
    )
    table = s.tables["GkSource"]
    s.conn.execute(
        sqla.update(table).where(table.c.serial == obj.store_id).values(validated=True)
    )
    s.conn.commit()
    return obj


def stored_members(s: Store, serial: int):
    table = s.tables["GkSource_parents"]
    rows = s.conn.execute(
        sqla.select(
            table.c.z_source_serial, table.c.numeric_serial, table.c.wkb_serial
        ).where(table.c.gk_source_serial == serial)
    ).all()
    return tuple(sorted((tuple(r) for r in rows), key=lambda m: m[0]))


def stored_digest(s: Store, serial: int) -> str:
    table = s.tables["GkSource"]
    return s.conn.execute(
        sqla.select(table.c.parents_digest).where(table.c.serial == serial)
    ).scalar_one()


class _WithAStore(unittest.TestCase):
    def setUp(self):
        self.s = _gk_store()
        self.addCleanup(self.s.close)


class TestTheDigest(unittest.TestCase):
    def test_it_is_the_sha256_of_the_sorted_members(self):
        import hashlib

        expected = hashlib.sha256(b"[[21,41,null],[22,42,51],[23,null,52]]").hexdigest()
        self.assertEqual(GkSource_parents_digest(SET_A), expected)
        # the order the members are given in does not matter; their content does
        self.assertEqual(GkSource_parents_digest(tuple(reversed(SET_A))), expected)
        self.assertNotEqual(GkSource_parents_digest(SET_B), expected)

    def test_a_malformed_member_is_refused(self):
        for members in (
            ((21, None, None),),
            ((21, 41, None), (21, 43, None)),
            ((21, True, None),),
            ((21.0, 41, None),),
            ((21, 41),),
        ):
            with self.subTest(members=members):
                with self.assertRaises(ValueError):
                    GkSource_parents_digest(members)


class TestKeyedOnItsSet(_WithAStore):
    """§3 test 1."""

    def test_found_by_its_own_set_and_missed_by_another(self):
        stored = store_source(self.s, SET_A)
        found = source_lookup(self.s, parents=SET_A)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        self.assertEqual(found.parents, SET_A)
        self.assertFalse(source_lookup(self.s, parents=SET_B).available)

    def test_two_sets_differing_in_one_row_give_two_rows_each_found_by_its_own(self):
        a = store_source(self.s, SET_A)
        b = store_source(self.s, SET_B)
        self.assertNotEqual(a.store_id, b.store_id)
        self.assertEqual(source_lookup(self.s, parents=SET_A).store_id, a.store_id)
        self.assertEqual(source_lookup(self.s, parents=SET_B).store_id, b.store_id)

    def test_a_miss_carries_the_set_it_was_asked_under(self):
        missed = source_lookup(self.s, parents=SET_B)
        self.assertFalse(missed.available)
        self.assertEqual(missed.parents, SET_B)
        self.assertFalse(missed.labels_only)

    def test_a_subset_is_not_the_set(self):
        store_source(self.s, SET_A)
        self.assertFalse(source_lookup(self.s, parents=SET_A[:2]).available)

    def test_the_labels_and_tags_are_still_part_of_the_key(self):
        store_source(self.s, SET_A)
        self.assertFalse(
            source_lookup(self.s, parents=SET_A, tags=(TAG, OTHER_TAG)).available
        )
        self.assertFalse(
            source_lookup(self.s, parents=SET_A, z_response=OTHER_Z_RESPONSE).available
        )

    def test_a_lookup_naming_neither_the_set_nor_the_mode_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            source_lookup(self.s)
        self.assertIn("labels_only=True", str(caught.exception))

    def test_the_set_and_the_mode_together_are_an_error(self):
        with self.assertRaises(ValueError):
            source_lookup(self.s, parents=SET_A, labels_only=True)


class TestTheRecordAgreesWithTheKey(_WithAStore):
    """§3 test 2."""

    def test_parents_digest_is_the_digest_of_the_rows_written(self):
        stored = store_source(self.s, SET_A)
        members = stored_members(self.s, stored.store_id)
        self.assertEqual(members, SET_A)
        self.assertEqual(
            stored_digest(self.s, stored.store_id), GkSource_parents_digest(members)
        )

    def test_the_rows_written_are_the_set_assembled_not_the_set_asked_for(self):
        """An object asked for under SET_A whose values were assembled from SET_B: the row records
        SET_B, and its key column is SET_B's digest, recomputed from the rows written. (compute()
        refuses such an object before this point; ComputeTargets/tests has that half.)
        """
        stored = store_source(self.s, SET_A, assembled=SET_B)
        members = stored_members(self.s, stored.store_id)
        self.assertEqual(members, SET_B)
        self.assertEqual(
            stored_digest(self.s, stored.store_id), GkSource_parents_digest(members)
        )
        self.assertNotEqual(
            stored_digest(self.s, stored.store_id), GkSource_parents_digest(SET_A)
        )
        # and so the row is found by what it is, not by what it was asked to be
        self.assertEqual(source_lookup(self.s, parents=SET_B).store_id, stored.store_id)
        self.assertFalse(source_lookup(self.s, parents=SET_A).available)

    def test_an_empty_set_is_refused_and_nothing_is_written(self):
        for assembled in ((), None):
            with self.subTest(assembled=assembled):
                obj = GkSource(
                    payload=None,
                    model=MODEL,
                    k=K_EXIT,
                    z_response=Z_RESPONSE,
                    z_sample=Z_SAMPLE,
                    tags=[TAG],
                    parents=(),
                )
                obj._values = []
                obj._assembled_parents = assembled
                with self.assertRaises(ValueError) as caught:
                    sqla_GkSource_factory.store(
                        obj,
                        self.s.conn,
                        self.s.tables["GkSource"],
                        self.s.inserters["GkSource"],
                        self.s.tables,
                        self.s.inserters,
                    )
                self.assertIn("empty parent set", str(caught.exception))
                for table in ("GkSource", "GkSource_parents", "GkSource_tags"):
                    self.assertEqual(
                        self.s.conn.execute(
                            sqla.text(f'SELECT count(*) FROM "{table}"')
                        ).scalar_one(),
                        0,
                    )


class TestTheValueReadsHandBackTheirParent(_WithAStore):
    """§3 test 3."""

    def numeric(self, **extra):
        payload = {
            "model": MODEL,
            "k": K_EXIT,
            "z": Z_RESPONSE,
            "atol": ATOL,
            "rtol": RTOL,
            "tags": [TAG],
        }
        payload.update(extra)
        return sqla_GkNumericValue_factory.read_batch(
            payload, self.s.conn, self.s.tables["GkNumericValue"], self.s.tables
        )

    def wkb(self, **extra):
        payload = {"model": MODEL, "k": K_EXIT, "z": Z_RESPONSE, "tags": [TAG]}
        payload.update(extra)
        return sqla_GkWKBValue_factory.read_batch(
            payload, self.s.conn, self.s.tables["GkWKBValue"], self.s.tables
        )

    def test_each_numeric_value_carries_its_integration(self):
        found = {v._z_source.store_id: v._parent_serial for v in self.numeric()}
        self.assertEqual(found, {21: 41, 22: 42})

    def test_each_wkb_value_carries_its_integration(self):
        found = {v._z_source.store_id: v._parent_serial for v in self.wkb()}
        self.assertEqual(found, {22: 51, 23: 52})

    def test_the_serial_only_forms_return_the_pairs_and_no_values(self):
        self.assertEqual(self.numeric(parents_only=True), [(21, 41), (22, 42)])
        self.assertEqual(self.wkb(parents_only=True), [(22, 51), (23, 52)])
        # the other atol's row is another key
        self.assertEqual(self.numeric(parents_only=True, atol=OTHER_ATOL), [(21, 43)])

    def test_the_two_forms_agree(self):
        full = sorted((v._z_source.store_id, v._parent_serial) for v in self.numeric())
        self.assertEqual(full, self.numeric(parents_only=True))
        full = sorted((v._z_source.store_id, v._parent_serial) for v in self.wkb())
        self.assertEqual(full, self.wkb(parents_only=True))

    def test_the_serial_only_form_still_refuses_a_second_parent(self):
        # a second WKB row at z_source 23, under the same labels
        self.s.insert(
            "GkWKBIntegration",
            serial=53,
            label="gkwkb-53",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            numeric_serial=None,
            rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
            solver_serial=1,
            z_source_serial=23,
            z_min_serial=OTHER_Z_RESPONSE.store_id,
            z_samples=1,
            z_init=1.0e4,
            sin_coeff=0.0,
            cos_coeff=1.0,
            G_init=0.0,
            Gprime_init=1.0,
            validated=True,
        )
        self.s.insert("GkWKB_tags", wkb_serial=53, tag_serial=1)
        self.s.insert(
            "GkWKBValue",
            serial=153,
            wkb_serial=53,
            z_serial=Z_RESPONSE.store_id,
            H_ratio=1.0,
            theta_mod_2pi=-0.5,
            theta_div_2pi=0,
            omega_WKB_sq=4.0,
            G_WKB=0.5,
        )
        self.s.conn.commit()
        for form in ({}, {"parents_only": True}):
            with self.subTest(form=form):
                with self.assertRaises(MultipleResultsFound) as caught:
                    self.wkb(**form)
                self.assertIn("[52, 53]", str(caught.exception))


class TestTheReaderMode(_WithAStore):
    """§3 test 4, for GkSource.build()'s labels_only (D1)."""

    def test_two_matches_are_refused_naming_their_digests_and_where_they_differ(self):
        a = store_source(self.s, SET_A)
        b = store_source(self.s, SET_B)
        with self.assertRaises(MultipleResultsFound) as caught:
            source_lookup(self.s, labels_only=True)
        message = str(caught.exception)
        self.assertIn(
            f"serial={a.store_id}, parents_digest={GkSource_parents_digest(SET_A)}",
            message,
        )
        self.assertIn(
            f"serial={b.store_id}, parents_digest={GkSource_parents_digest(SET_B)}",
            message,
        )
        # they differ at z_source 21 only
        self.assertIn("differ at 1 z_source serial(s): [21]", message)

    def test_one_match_is_returned_without_being_given_the_set(self):
        stored = store_source(self.s, SET_A)
        found = source_lookup(self.s, labels_only=True)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        # carrying the set its row records
        self.assertEqual(found.parents, SET_A)

    def test_no_match_is_a_labels_only_miss_that_cannot_be_computed(self):
        missed = source_lookup(self.s, labels_only=True)
        self.assertFalse(missed.available)
        self.assertTrue(missed.labels_only)
        self.assertIsNone(missed.parents)

    def test_the_tags_still_select_under_the_mode(self):
        store_source(self.s, SET_A)
        store_source(self.s, SET_B, tags=(TAG, OTHER_TAG))
        found = source_lookup(self.s, labels_only=True, tags=(TAG, OTHER_TAG))
        self.assertEqual(found.parents, SET_B)


class TestTheValueModelRead(_WithAStore):
    """§3 test 4, for GkSourceValue._build_impl_model (the user's decision on the sketch,
    2026-09-30): a labels-only read of one value, which joined every matching GkSource.
    """

    def read(self):
        return _call(
            self.s,
            sqla_GkSourceValue_factory,
            "GkSourceValue",
            {
                "model": MODEL,
                "k": K_EXIT,
                "z_response": Z_RESPONSE,
                "z_source": Z_SOURCES[22],
                "tags": [TAG],
            },
        )

    def value_row(self, parent: int, serial: int, G: float):
        self.s.insert(
            "GkSourceValue",
            serial=serial,
            parent_serial=parent,
            z_source_serial=22,
            G=G,
            Gprime=0.0,
        )
        self.s.conn.commit()

    def test_one_match_serves_its_value(self):
        a = store_source(self.s, SET_A)
        self.value_row(a.store_id, 801, 11.0)
        found = self.read()
        self.assertEqual(found.store_id, 801)
        self.assertEqual(found.numeric.G, 11.0)

    def test_two_matches_are_refused_naming_what_distinguishes_them(self):
        a = store_source(self.s, SET_A)
        b = store_source(self.s, SET_B)
        self.value_row(a.store_id, 801, 11.0)
        self.value_row(b.store_id, 802, 22.0)
        with self.assertRaises(MultipleResultsFound) as caught:
            self.read()
        message = str(caught.exception)
        self.assertIn(GkSource_parents_digest(SET_A), message)
        self.assertIn(GkSource_parents_digest(SET_B), message)
        self.assertIn("[21]", message)

    def test_no_match_is_a_miss(self):
        self.assertFalse(self.read().available)


class TestTheWKBValueReadsOnACurrentStore(unittest.TestCase):
    """
    The WKB value reads on a current store: ``GkWKBValue.read_batch`` and its keyed
    ``_build_impl_model`` read the rows the schema holds. (Moved, with its body unchanged, from
    ``TestTheWKBValueReadsRefuseAPre07Store``, whose other tests exercised the refusal of a store
    with no ``numeric_serial`` and went with it.)
    """

    def test_a_current_table_is_still_read(self):
        """The control: the same four reads on the current schema's tables find their rows."""
        s = _gk_store()
        self.addCleanup(s.close)
        payload = {"model": MODEL, "k": K_EXIT, "z": Z_RESPONSE, "tags": [TAG]}
        values = sqla_GkWKBValue_factory.read_batch(
            payload, s.conn, s.tables["GkWKBValue"], s.tables
        )
        self.assertEqual(sorted(v.store_id for v in values), [151, 152])
        value = _call(
            s,
            sqla_GkWKBValue_factory,
            "GkWKBValue",
            {
                "model": MODEL,
                "k": K_EXIT,
                "z": Z_RESPONSE,
                "z_source": Z_SOURCES[23],
                "numeric_serial": GkWKBIntegration.PURE_WKB,
            },
        )
        self.assertEqual(value.store_id, 152)


class TestTheForeignKeys(_WithAStore):
    """
    §3 test 9: ``GkSource_parents``' four foreign keys, checked under ``PRAGMA
    foreign_key_check`` on rows of the test's own.

    Each serial a member row names is held by its own target table **only** -- GkSource 71,
    redshift 21 to 23, GkNumericIntegration 41 and 42, GkWKBIntegration 51 and 52 -- which
    ``test_the_rows_are_not_satisfied_by_coincidence`` asserts. So a key declared to the wrong
    table could not be satisfied by coincidence, as prompt 06 found the fixture's own rows could.
    """

    TARGETS = {
        "gk_source_serial": "GkSource",
        "z_source_serial": "redshift",
        "numeric_serial": "GkNumericIntegration",
        "wkb_serial": "GkWKBIntegration",
    }

    def setUp(self):
        super().setUp()
        self.s.insert(
            "GkSource",
            serial=71,
            label="gksource-71",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            z_response_serial=Z_RESPONSE.store_id,
            parents_digest=GkSource_parents_digest(SET_A),
            z_max_serial=21,
            z_samples=0,
            validated=True,
        )
        for serial, (z_source, numeric, wkb) in zip((91, 92, 93), SET_A):
            self.s.insert(
                "GkSource_parents",
                serial=serial,
                gk_source_serial=71,
                z_source_serial=z_source,
                numeric_serial=numeric,
                wkb_serial=wkb,
            )
        self.s.conn.commit()

    def violations(self):
        return self.s.conn.execute(sqla.text("PRAGMA foreign_key_check")).all()

    def test_the_keys_are_declared(self):
        table = self.s.tables["GkSource_parents"]
        declared = {
            column.name: fk.column.table.name
            for column in table.columns
            for fk in column.foreign_keys
        }
        self.assertEqual(declared, self.TARGETS)

    def test_the_rows_are_not_satisfied_by_coincidence(self):
        named = {
            "gk_source_serial": {71},
            "z_source_serial": {21, 22, 23},
            "numeric_serial": {41, 42},
            "wkb_serial": {51, 52},
        }
        tables = set(self.TARGETS.values())
        for column, serials in named.items():
            for other in tables - {self.TARGETS[column]}:
                with self.subTest(column=column, other=other):
                    held = set(
                        self.s.conn.execute(
                            sqla.text(f'SELECT serial FROM "{other}"')
                        ).scalars()
                    )
                    self.assertEqual(held & serials, set())

    def test_the_store_passes_foreign_key_check(self):
        self.assertEqual(self.violations(), [])

    def test_a_store_written_by_the_factory_passes_too(self):
        store_source(self.s, SET_B)
        self.assertEqual(self.violations(), [])

    def test_a_dangling_reference_is_reported_for_each_key(self):
        # each column in turn names a serial its own table does not hold, but another table does
        wrong = {
            "gk_source_serial": 41,
            "z_source_serial": 51,
            "numeric_serial": 51,
            "wkb_serial": 41,
        }
        table = self.s.tables["GkSource_parents"]
        for column, serial in wrong.items():
            with self.subTest(column=column):
                self.s.conn.execute(
                    sqla.update(table)
                    .where(table.c.serial == 92)
                    .values(**{column: serial})
                )
                found = self.violations()
                self.assertEqual(
                    [(row[0], row[2]) for row in found],
                    [("GkSource_parents", self.TARGETS[column])],
                )
                self.s.conn.rollback()
                self.assertEqual(self.violations(), [])


class TestTheInventoryResolvesTheSet(unittest.TestCase):
    """§3 test 5, on the full real store (``build_full_store``) and variations of it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls._count = 0
        cls.baseline = cls.read()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def read(cls, **kwargs):
        cls._count += 1
        directory = cls.root / f"store-{cls._count}"
        directory.mkdir()
        return read_inventory(build_full_store(directory, **kwargs).primary, _factories)

    @staticmethod
    def kinds(problems):
        return sorted(p.split(":", 1)[0] for p in problems)

    def test_the_fixture_has_every_set_resolved(self):
        cls = self.baseline["GkSource"]
        self.assertEqual(cls.problems, ())
        self.assertEqual(cls.count, 3)
        self.assertIn("parents", cls.parent_sets)
        for record in cls.records:
            members = cls.members["parents"][record.key["parents"]]
            self.assertEqual(len(members), 2)
            self.assertEqual(
                list(cls.parent_sets["parents"]), ["z_source", "numeric", "wkb"]
            )
            for member in members:
                self.assertEqual(len(member), 3)

    def test_a_member_is_the_parents_identity_and_not_its_serial(self):
        """Each member names the reference digests of its rows, as a key parent does."""
        cls = self.baseline["GkSource"]
        numeric = {
            self.baseline.digest_of("GkNumericIntegration", r)
            for r in self.baseline["GkNumericIntegration"].records
        }
        wkb = {
            self.baseline.digest_of("GkWKBIntegration", r)
            for r in self.baseline["GkWKBIntegration"].records
        }
        for record in cls.records:
            for z_source, n, w in cls.members["parents"][record.key["parents"]]:
                self.assertTrue(n is None or n in numeric)
                self.assertTrue(w is None or w in wkb)
                self.assertIsInstance(z_source, str)

    def test_two_sources_differing_only_in_one_parent_are_two_records(self):
        """GkSource 7 is GkSource 1 in every column and tag, but for one member: at z_source 2 it
        was assembled from Gk numeric 1 as well as the pure-WKB row."""
        replicated, sharded, keys = full_rows()
        rows = sharded[0]
        source = dict(next(r for r in rows["GkSource"] if r["serial"] == 1))
        source.update(serial=7, label="fixture-gksource-7")
        sharded[0] = with_rows(
            rows,
            {
                "GkSource": [source],
                "GkSource_tags": [
                    {"parent_serial": 7, "tag_serial": 1},
                    {"parent_serial": 7, "tag_serial": 2},
                ],
                "GkSource_parents": [
                    {
                        "serial": 607,
                        "gk_source_serial": 7,
                        "z_source_serial": 1,
                        "numeric_serial": 1,
                        "wkb_serial": None,
                    },
                    {
                        "serial": 608,
                        "gk_source_serial": 7,
                        "z_source_serial": 2,
                        "numeric_serial": 1,
                        "wkb_serial": 1,
                    },
                ],
            },
        )
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        cls = inventory["GkSource"]
        self.assertEqual(cls.problems, ())
        self.assertEqual(cls.count, 4)
        identities = {r.identity() for r in cls.records}
        self.assertEqual(len(identities), 4)
        # the new record and GkSource 1's differ in their parent set and in nothing else
        base = set(self.baseline["GkSource"].records)
        (new,) = [r for r in cls.records if r not in base]
        (twin,) = [
            r
            for r in cls.records
            if r is not new
            and {f: v for f, v in r.key.items() if f != "parents"}
            == {f: v for f, v in new.key.items() if f != "parents"}
            and r.tags == new.tags
        ]
        self.assertNotEqual(twin.key["parents"], new.key["parents"])

    def test_an_unresolvable_member_is_a_named_problem(self):
        replicated, sharded, keys = full_rows()
        # GkSource 1's member at z_source 2 names a Gk WKB row that is not there
        vary_row(replicated, sharded, "GkSource_parents", 602, "wkb_serial", 99)
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        problems = inventory["GkSource"].problems
        self.assertEqual(self.kinds(problems), ["unresolved-parent"])
        self.assertIn("field(s) parents", problems[0])
        self.assertIn("serials 1", problems[0])
        self.assertEqual(inventory["GkSource"].count, 2)
        # and its GkSourcePolicyData, which is keyed on it, cannot be resolved either
        self.assertEqual(
            self.kinds(inventory["GkSourcePolicyData"].problems), ["unresolved-parent"]
        )

    def test_an_orphan_member_row_is_a_named_problem(self):
        replicated, sharded, keys = full_rows()
        sharded[1] = with_rows(
            sharded[1],
            {
                "GkSource_parents": [
                    {
                        "serial": 609,
                        "gk_source_serial": 99,
                        "z_source_serial": 1,
                        "numeric_serial": 2,
                        "wkb_serial": None,
                    }
                ]
            },
        )
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        problems = inventory["GkSource"].problems
        self.assertEqual(self.kinds(problems), ["orphan-member"])
        self.assertIn("shard #1", problems[0])
        self.assertIn("owner serials 99", problems[0])
        # nothing else moves
        self.assertEqual(
            inventory["GkSource"].records, self.baseline["GkSource"].records
        )

    def test_an_empty_set_is_a_named_problem(self):
        replicated, sharded, keys = full_rows()
        sharded[0]["GkSource_parents"] = [
            r for r in sharded[0]["GkSource_parents"] if r["gk_source_serial"] != 2
        ]
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        problems = inventory["GkSource"].problems
        self.assertEqual(self.kinds(problems), ["empty-parent-set"])
        self.assertIn("serials 2", problems[0])
        self.assertEqual(inventory["GkSource"].count, 2)
        self.assertEqual(
            self.kinds(inventory["GkSourcePolicyData"].problems), ["unresolved-parent"]
        )

    def test_parents_digest_is_not_identity(self):
        replicated, sharded, keys = full_rows()
        vary_row(replicated, sharded, "GkSource", 1, "parents_digest", "0" * 64)
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        self.assertEqual(
            inventory["GkSource"].records, self.baseline["GkSource"].records
        )
        self.assertEqual(inventory["GkSource"].problems, ())

    def test_resolve_names_every_member(self):
        cls = self.baseline["GkSource"]
        for record in cls.records:
            resolved = self.baseline.resolve("GkSource", record)
            members = resolved["parents"]["set"]
            self.assertEqual(len(members), 2)
            for member in members:
                self.assertIn("key", member["z_source"])
                self.assertTrue(member["numeric"] is None or "key" in member["numeric"])
                self.assertTrue(member["wkb"] is None or "key" in member["wkb"])


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
