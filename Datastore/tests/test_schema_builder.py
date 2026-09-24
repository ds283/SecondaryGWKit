"""
The one schema builder, ``Datastore/SQL/schema.py`` ``build_schema`` (store-fingerprint prompt 01).

- **The schema is unchanged.** Both ``build_schema`` and the actor's ``_build_schema`` reproduce
  ``Datastore/tests/data/schema_at_base.json`` exactly. That file was captured from the code
  before the builder moved (the prompt's log records how, and at which SHA); it is never
  regenerated to make these tests pass. The actor's method is called without Ray and without its
  ``__init__`` (``schema_description.actor_with_built_schema``), so the second test shows that the
  actor delegates to the function and has not kept its own copy.
- **The fix** of ``[00-build-schema-reads-registration-before-its-none-check]``: a factory whose
  ``register()`` returns ``None`` gets a record with no table, and no ``AttributeError``.

No Ray, no datastore, nothing under ``var/``.
"""

import functools
import json
import unittest
from pathlib import Path

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema
from Datastore.tests.schema_description import (
    actor_with_built_schema,
    describe_schema,
    dumps,
)

WITNESS = Path(__file__).resolve().parent / "data" / "schema_at_base.json"


def _witness_text() -> str:
    return WITNESS.read_text(encoding="utf-8")


class _NoTableFactory:
    """A factory that asks for no table."""

    @staticmethod
    def register():
        return None


class _OneColumnFactory:
    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("label", sqla.String(64))],
        }


class TestSchemaIsUnchanged(unittest.TestCase):
    def assert_matches_witness(self, description):
        witness = json.loads(_witness_text())
        # class by class first, so that a failure names the class that changed
        self.assertEqual(witness["class_order"], description["class_order"])
        for cls_name in witness["classes"]:
            with self.subTest(cls=cls_name):
                self.assertEqual(
                    witness["classes"][cls_name], description["classes"][cls_name]
                )
        self.assertEqual(witness, description)
        # and byte for byte, as the file was written
        self.assertEqual(_witness_text(), dumps(description))

    def test_build_schema_reproduces_the_witness(self):
        metadata = sqla.MetaData()
        built = build_schema(metadata, _factories)
        self.assert_matches_witness(
            describe_schema(built.records, built.tables, metadata)
        )

    def test_build_schema_records_hold_no_inserter(self):
        built = build_schema(sqla.MetaData(), _factories)
        for cls_name, record in built.records.items():
            with self.subTest(cls=cls_name):
                self.assertNotIn("insert", record)

    def test_actor_build_schema_reproduces_the_witness(self):
        actor = actor_with_built_schema()
        self.assert_matches_witness(
            describe_schema(actor._schema, actor._tables, actor._metadata)
        )

    def test_actor_adds_only_the_inserters(self):
        actor = actor_with_built_schema()
        self.assertEqual(list(actor._tables), list(actor._inserters))
        for cls_name, record in actor._schema.items():
            with self.subTest(cls=cls_name):
                tab = record["table"]
                self.assertIs(actor._tables.get(cls_name), tab)
                inserter = record["insert"]
                self.assertIs(actor._inserters[cls_name], inserter)
                self.assertIsInstance(inserter, functools.partial)
                self.assertEqual(inserter.func, actor._insert)
                self.assertIs(inserter.args[0], record)
                self.assertIs(inserter.args[1], tab)

    def test_actor_tables_are_in_its_metadata(self):
        actor = actor_with_built_schema()
        for cls_name, tab in actor._tables.items():
            with self.subTest(cls=cls_name):
                self.assertIs(actor._metadata.tables[cls_name], tab)


class TestNoneRegistration(unittest.TestCase):
    FACTORIES = {"NoTable": _NoTableFactory, "OneColumn": _OneColumnFactory}

    def test_build_schema_gives_a_record_with_no_table(self):
        metadata = sqla.MetaData()
        built = build_schema(metadata, self.FACTORIES)

        self.assertEqual(
            built.records["NoTable"],
            {"name": "NoTable", "validate_on_startup": False, "table": None},
        )
        self.assertNotIn("NoTable", built.tables)
        self.assertNotIn("NoTable", metadata.tables)
        # the class after it is built as usual
        self.assertEqual(
            [c.name for c in built.tables["OneColumn"].columns],
            ["serial", "timestamp", "label"],
        )

    def test_actor_gives_a_record_with_no_table_and_no_inserter(self):
        import importlib

        datastore_module = importlib.import_module("Datastore.SQL.Datastore")
        cls = datastore_module.Datastore.__ray_metadata__.modified_class
        actor = object.__new__(cls)
        actor._factories = {}
        actor.register_factories(self.FACTORIES)
        actor._metadata = sqla.MetaData()
        actor._tables = {}
        actor._inserters = {}
        actor._schema = {}
        actor._build_schema()

        self.assertIsNone(actor._schema["NoTable"]["table"])
        self.assertIsNone(actor._schema["NoTable"]["insert"])
        self.assertFalse(actor._schema["NoTable"]["validate_on_startup"])
        self.assertNotIn("NoTable", actor._tables)
        self.assertNotIn("NoTable", actor._inserters)
        self.assertIn("OneColumn", actor._inserters)


if __name__ == "__main__":
    unittest.main()
