"""
Test for prompt 07 of ``prompts/background-solver-robustness``:
``sqla_QCDCosmology_factory.inventory()`` reports the ``T_z_representation`` column.

Since ``prompts/qcd-background-audit`` prompt 03, ``T_z_representation`` is part of the QCD
cosmology's datastore lookup key (see the module docstring on
``Datastore/SQL/ObjectFactories/QCD_Cosmology.py``): two rows can share every physical parameter
and ``log10_max_z`` and still be different backgrounds. Before this commit ``inventory()`` selected
only ``name``, ``omega_m``, ``omega_cc``, ``h`` and ``log10_max_z``, so two such rows rendered as
indistinguishable duplicates to ``tools/inventory_report.py`` -- the only tool in the tree that
inspects a datastore ([03-qcd-inventory-does-not-report-the-representation]).

This runs the production ``inventory()`` against a real in-memory SQLite table built from the
factory's own ``register()`` output, in the shape ``ComputeTargets/tests/test_cosmology_representation_key.py``
already uses -- no Ray, no project ``Datastore`` machinery.
"""

import unittest
from datetime import datetime

import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory

KEY_COLUMN = "T_z_representation"


def _build_table(metadata: sqla.MetaData) -> sqla.Table:
    """
    Reproduce the table Datastore._build_schema builds for this factory, from the factory's own
    register() output -- so the schema under test is the production schema.
    """
    registration = sqla_QCDCosmology_factory.register()

    table = sqla.Table("QCD_Cosmology", metadata)
    if registration.get("serial", True):
        table.append_column(sqla.Column("serial", sqla.Integer, primary_key=True))
    if registration.get("timestamp", False):
        table.append_column(sqla.Column("timestamp", sqla.DateTime()))
    for column in registration.get("columns", []):
        table.append_column(column)

    return table


def _row(representation: int, **overrides) -> dict:
    payload = {
        "name": "QCD_Cosmology",
        "omega_m": 0.3,
        "omega_cc": 0.7,
        "h": 0.7,
        "f_baryon": 0.05,
        "T_CMB_Kelvin": 2.725,
        "Neff": 3.046,
        "log10_max_z": 12.0,
        "T_z_representation": representation,
        "timestamp": datetime.now(),
    }
    payload.update(overrides)
    return payload


class TestInventoryReportsTheRepresentation(unittest.TestCase):
    """Prompt 07 §3: two rows differing only in T_z_representation must render distinguishably."""

    def setUp(self):
        self.metadata = sqla.MetaData()
        self.table = _build_table(self.metadata)
        self.engine = sqla.create_engine("sqlite://", future=True)
        self.table.create(self.engine)
        self.conn = self.engine.connect()
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.conn.close)

    def _insert(self, **overrides):
        self.conn.execute(sqla.insert(self.table), _row(**overrides))
        self.conn.commit()

    def test_two_rows_differing_only_in_representation_render_distinguishably(self):
        # same name, same seven parameters, same log10_max_z -- differ only in the column that
        # prompt 03 added to the lookup key
        self._insert(representation=4)
        self._insert(representation=6)

        report = sqla_QCDCosmology_factory.inventory(self.conn, self.table, None)
        values = report["values"]

        self.assertEqual(len(values), 2)
        # every other reported field is identical between the two rows ...
        for value in values:
            self.assertEqual(value["name"], "QCD_Cosmology")
            self.assertEqual(value["omega_m"], 0.3)
            self.assertEqual(value["omega_cc"], 0.7)
            self.assertEqual(value["h"], 0.7)
            self.assertEqual(value["log10_max_z"], 12.0)

        # ... so the representation is the only thing that can, and must, tell them apart
        representations = {value[KEY_COLUMN] for value in values}
        self.assertEqual(representations, {4, 6})
        self.assertNotEqual(values[0], values[1])

    def test_the_column_appears_before_log10_max_z_not_at_the_end(self):
        """
        Prompt 07 §2 item 3: the representation is placed next to the field it qualifies, not
        appended as an afterthought. It and log10_max_z both describe how the row's background was
        computed and over what range, so it precedes log10_max_z rather than trailing it.
        """
        self._insert(representation=6)
        report = sqla_QCDCosmology_factory.inventory(self.conn, self.table, None)
        keys = list(report["values"][0].keys())

        self.assertIn(KEY_COLUMN, keys)
        self.assertLess(keys.index(KEY_COLUMN), keys.index("log10_max_z"))

    def test_the_same_representation_still_reports_one_label_each(self):
        """a sanity check that the new column does not change the row count or the other fields"""
        self._insert(representation=6, name="A")
        self._insert(representation=6, name="B")

        report = sqla_QCDCosmology_factory.inventory(self.conn, self.table, None)
        values = report["values"]

        self.assertEqual(len(values), 2)
        self.assertEqual({v["name"] for v in values}, {"A", "B"})
        self.assertEqual({v[KEY_COLUMN] for v in values}, {6})


if __name__ == "__main__":
    unittest.main()
