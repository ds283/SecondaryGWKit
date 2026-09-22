"""
Round-trip guard for ``sqla_BackgroundModelValue_factory.build()``, which until 2026-09-22 had
never been executed by anything.

`prompts/datastore-readback` prompt 01's audit found two defects in it, both fatal on the first
call and both invisible because nothing calls it: ``BackgroundModel.build()`` reads its sample
rows with its own ``SELECT`` rather than through this factory, and nothing in the tree calls
``object_get("BackgroundModelValue", ...)``.

  * the insert branch wrote the key ``"wkb_serial"``, which is not a column of this table
    (``register()`` declares ``model_serial``; ``wkb_serial`` is a real column of the *GkWKB* and
    *TkWKB* value tables, which is where it was copied from). SQLAlchemy Core's ``insert()``
    drops an unmatched key silently rather than raising -- the note at
    ``OneLoopIntegral.py:253`` records the same behaviour for a different key -- so the row was
    attempted with ``model_serial`` unset, and the column's own ``nullable=False`` is what
    refused it: ``IntegrityError: NOT NULL constraint failed``. The write branch could never have
    succeeded, but not for the reason the key's spelling suggests, and had the column been
    nullable it would have written a silently orphaned row instead;
  * the read-back branch compared ``row_data.Hubble``, which is not a column either -- the column
    is ``Hubble_GeV`` -- so the read branch raised ``NoSuchColumnError`` on the attribute.

The second was not a misspelling that could be renamed away. The comparison sat *below* the block
that replaces the payload values with the stored ones, so ``Hubble`` had already become the stored
value: with the attribute name corrected in place, the check would have compared the stored value
against itself and could never have fired. It is now above that block, and compares in the stored
GeV representation with a relative bound, because the internal-unit value spans some thirty
decades and an absolute ``DEFAULT_FLOAT_PRECISION`` there is meaningless at both ends.

The tests below therefore exercise all three branches -- insert, agreeing read-back, disagreeing
read-back -- against an in-memory SQLite built from the factory's own ``register()`` output, so
the schema under test is the production schema rather than a copy of it. No Ray and no datastore
file, per `CLAUDE.md`. The helpers follow ``ComputeTargets/tests/test_run_identity.py``, which
builds the same table the same way.

This is prompt 01 §3's shape **(b)**, the in-memory round trip, applied to the one factory that
needed it: the static guard in ``test_factory_select_columns.py`` sees the read side of this
defect class, and by construction cannot see the write side, which is a dict of keys against a
column list.
"""

import unittest
from datetime import datetime
from math import ulp
from typing import Optional

import sqlalchemy as sqla

from CosmologyConcepts import redshift
from Datastore.SQL.ObjectFactories.BackgroundModel import (
    sqla_BackgroundModelValue_factory,
)
from Units import Mpc_units
from config.defaults import DEFAULT_FLOAT_PRECISION


def _append_table(name: str, factory, metadata: sqla.MetaData) -> sqla.Table:
    """The table Datastore._build_schema builds for this factory, from its own register()."""
    registration = factory.register()

    tab = sqla.Table(name, metadata)
    if registration.get("serial", True):
        tab.append_column(sqla.Column("serial", sqla.Integer, primary_key=True))
    if registration.get("version", False):
        tab.append_column(sqla.Column("version", sqla.Integer))
    if registration.get("timestamp", False):
        tab.append_column(sqla.Column("timestamp", sqla.DateTime()))
    if registration.get("stepping", False):
        tab.append_column(sqla.Column("stepping", sqla.Integer))

    for column in registration.get("columns", []):
        tab.append_column(column)

    return tab


def _inserter_for(table: sqla.Table):
    """What Datastore._insert does for this table."""
    has_timestamp = "timestamp" in table.c

    def inserter(conn, data):
        payload = dict(data)
        if has_timestamp:
            payload["timestamp"] = datetime.now()
        return conn.execute(sqla.insert(table), payload).lastrowid

    return inserter


# a realistic sample: the Hubble rate at the top of a production redshift grid, where the
# internal-unit value is ~1e+26 and an absolute tolerance on it is worthless
UNITS = Mpc_units()
MODEL_SERIAL = 17
Z_SERIAL = 4242
HUBBLE_GeV = 5.8773840062414546e-12
W_BACKGROUND = 1.0 / 3.0


class _Schema:
    """One in-memory SQLite database carrying the single table this factory touches."""

    def __init__(self):
        self.metadata = sqla.MetaData()

        # the two foreign-key targets the registered columns name. Stubs rather than the real
        # factories' tables: nothing here reads them -- SQLite does not enforce foreign keys
        # unless PRAGMA foreign_keys is on, and the production Datastore does not turn it on --
        # but they have to exist in this MetaData for create_all to resolve the references.
        for target in ("BackgroundModel", "redshift"):
            sqla.Table(
                target,
                self.metadata,
                sqla.Column("serial", sqla.Integer, primary_key=True),
            )

        self.table = _append_table(
            "BackgroundModelValue", sqla_BackgroundModelValue_factory, self.metadata
        )
        self.engine = sqla.create_engine("sqlite://", future=True)
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()
        self.inserter = _inserter_for(self.table)

    def close(self):
        self.conn.close()
        self.engine.dispose()

    def build(self, hubble_GeV: float = HUBBLE_GeV, wBackground: float = W_BACKGROUND):
        return sqla_BackgroundModelValue_factory.build(
            payload=_payload(hubble_GeV, wBackground),
            conn=self.conn,
            table=self.table,
            inserter=self.inserter,
            tables={},
            inserters={},
        )

    def rows(self):
        return list(self.conn.execute(sqla.select(self.table)))


def _payload(hubble_GeV: float, wBackground: float) -> dict:
    """
    Everything build() reads out of its payload. The values other than Hubble and wBackground are
    arbitrary but fixed: this factory is a straight column-for-column round trip and nothing here
    evaluates a background.
    """
    GeV = UNITS.GeV
    Mpc = UNITS.Mpc
    return {
        "model_serial": MODEL_SERIAL,
        "units": UNITS,
        "z": redshift(store_id=Z_SERIAL, z=1.0e12, is_source=True, is_response=False),
        "Hubble": hubble_GeV * GeV,
        "wBackground": wBackground,
        "wPerturbations": 0.25,
        "rho": 3.0 * pow(GeV, 4.0),
        "tau": 1.5e-3 * Mpc,
        "tau_lo": 1.0e-19 * Mpc,
        "cs_tau": 8.7e-4 * Mpc,
        "cs_tau_lo": 5.0e-20 * Mpc,
        "friction_F": -13.75,
        "T_photon": 1.0e-3 * GeV,
        "d_lnH_dz": 2.0e-12,
        "d2_lnH_dz2": -3.0e-24,
        "d3_lnH_dz3": 4.0e-36,
        "d_wPerturbations_dz": 1.0e-13,
        "d2_wPerturbations_dz2": -2.0e-25,
    }


class TestBackgroundModelValueRoundTrip(unittest.TestCase):
    def setUp(self):
        self.db = _Schema()
        self.addCleanup(self.db.close)

    def test_the_insert_branch_writes_a_row(self):
        """
        The write side. With the key spelled "wkb_serial" the key was dropped in silence and the
        insert died on model_serial's NOT NULL constraint, so the row below could never have been
        written. Verified by restoring the old spelling: all six tests in this module fail with
        `IntegrityError: NOT NULL constraint failed: BackgroundModelValue.model_serial`.
        """
        obj = self.db.build()

        rows = self.db.rows()
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(MODEL_SERIAL, row.model_serial)
        self.assertEqual(Z_SERIAL, row.z_serial)
        self.assertEqual(HUBBLE_GeV, row.Hubble_GeV)
        self.assertEqual(W_BACKGROUND, row.wBackground)
        self.assertEqual(row.serial, obj.store_id)

    def test_the_insert_payload_names_only_real_columns(self):
        """
        The defect stated as what it was rather than as its symptom: every key build() inserts has
        to be a column of the table its own register() declares. This is the assertion that would
        still hold if the column were made nullable -- the NOT NULL constraint is what caught this
        one, and it is a property of the schema rather than of the insert, so it is not the thing
        to rely on.
        """
        self.db.build()
        columns = set(self.db.table.c.keys())
        written = set(self.db.rows()[0]._mapping.keys())
        self.assertEqual(set(), written - columns)
        self.assertIn("model_serial", columns)
        self.assertNotIn("wkb_serial", columns)

    def test_an_existing_row_reads_back(self):
        """
        The read side, and the defect the campaign exists for: build() read row_data.Hubble where
        the column is Hubble_GeV, so this second call raised NoSuchColumnError.
        """
        first = self.db.build()
        second = self.db.build()

        self.assertEqual(
            1, len(self.db.rows()), "the second call must not insert again"
        )
        self.assertEqual(first.store_id, second.store_id)
        self.assertEqual(HUBBLE_GeV * UNITS.GeV, second.Hubble)
        self.assertEqual(W_BACKGROUND, second.wBackground)
        self.assertEqual(Z_SERIAL, second.z.store_id)

    def test_a_disagreeing_Hubble_is_refused(self):
        """
        The check is load-bearing, which it has never been. Placed where it used to be -- below
        the block that overwrites the payload Hubble with the stored one -- this would pass with
        any stored value whatsoever.
        """
        self.db.build()

        with self.assertRaises(ValueError) as caught:
            self.db.build(hubble_GeV=HUBBLE_GeV * 1.01)
        self.assertIn("Hubble", str(caught.exception))

    def test_a_disagreeing_wBackground_is_refused(self):
        self.db.build()

        with self.assertRaises(ValueError) as caught:
            self.db.build(wBackground=W_BACKGROUND + 1.0e-3)
        self.assertIn("w_Background", str(caught.exception))

    def test_no_absolute_bound_can_serve_this_column(self):
        """
        Why the Hubble comparison is relative, and in the stored representation. Over a production
        redshift grid the internal-unit Hubble spans thirty decades, and a single absolute bound
        is the wrong instrument at both ends of it at once -- much too loose at one, unsatisfiable
        at the other. This is a property of the numbers, not of any particular pair of values, so
        it is asserted directly rather than inferred from a round trip.
        """
        GeV = UNITS.GeV
        low = 1.5162712131717023e-42 * GeV  # bottom of a production grid: 2.37e-04
        high = 5.8773840062414546e-12 * GeV  # top of one: 9.19e+26

        # at the bottom, an absolute DEFAULT_FLOAT_PRECISION is a *relative* tolerance of 4.2e-04:
        # it would accept a stored Hubble wrong in its fourth significant figure
        self.assertGreater(DEFAULT_FLOAT_PRECISION / low, 1.0e-4)

        # at the top, one ulp is 1.37e+11, so the smallest disagreement a double can express is
        # some 1.4e+18 times the bound: the check would degenerate to bit equality
        self.assertGreater(ulp(high) / DEFAULT_FLOAT_PRECISION, 1.0e17)

        # the relative bound that replaced it accepts both ends, which is the point
        for hubble_GeV in (1.5162712131717023e-42, 5.8773840062414546e-12):
            db = _Schema()
            self.addCleanup(db.close)
            db.build(hubble_GeV=hubble_GeV)
            db.build(hubble_GeV=hubble_GeV)
            self.assertEqual(1, len(db.rows()))
