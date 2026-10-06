"""
The inventory's declarations (prompts/datastore-generic, prompt 08; audit C1 rows 12 and 13).

Each factory declares how the inventory reads its class, as data (``inventory_spec()``); the layer
derives which classes it reads and in what order (``inventory_classes``), and resolves a
polymorphic parent through the type map its referencing factory declares. These tests pin:

1. the derived order is today's order, written here as a literal, and it is the roots-first rule,
   not a plain dependency sort;
2. on hand-built registries: a cycle and a parent with no spec are refused, by name; a polymorphic
   parent's classes are built before it; a malformed ``Parent`` is refused;
3. ``resolve()`` goes through the declared type map;
4. ``store_inventory`` names no project package or table and loads none.

No Ray; stores are built in temporary directories; nothing under ``var/``.
"""

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
import unittest
from pathlib import Path

from Datastore.contract import TAG_TABLE
from Datastore.store_inventory import (
    ClassInventory,
    InventorySpec,
    Parent,
    ParentSet,
    Record,
    StoreInventory,
    inventory_classes,
    inventory_specs,
    read_inventory,
    reference_digest,
)
from Datastore.tests.real_store_fixtures import build_full_store
from config.datastore import factories as _factories

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_INVENTORY = REPO_ROOT / "Datastore" / "store_inventory.py"

# the order the inventory was built in before prompt 08, when it was a hand-kept list
ORDER = (
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "tolerance",
    "LambdaCDM",
    "QCD_Cosmology",
    "IntegrationSolver",
    "GkSourcePolicy",
    "QuadSourcePolicy",
    "wavenumber_exit_time",
    "BackgroundModel",
    "TkNumericIntegration",
    "TkWKBIntegration",
    "GkNumericIntegration",
    "GkWKBIntegration",
    "GkSource",
    "GkSourcePolicyData",
    "QuadSource",
    "QuadSourceIntegral",
    "OneLoopIntegral",
)


def registry(**specs):
    """A hand-built registry, in keyword order: each name a stand-in factory whose
    ``inventory_spec()`` returns the spec given (``None`` for a class outside the inventory).
    """
    return {
        name: type(
            f"{name}_factory", (), {"inventory_spec": staticmethod(lambda s=spec: s)}
        )
        for name, spec in specs.items()
    }


def plain_sort(factories):
    """A dependency sort with no roots-first step: at each step the earliest class in registry
    order whose dependencies are placed."""
    specs = inventory_specs(factories)
    order = []
    while len(order) < len(specs):
        order.append(
            next(
                n
                for n, s in specs.items()
                if n not in order
                and all(d in order for d in s.dependencies() if d != n)
            )
        )
    return order


# ------------------------------------------------------------------------------------------------
# 1. the derived order
# ------------------------------------------------------------------------------------------------


class TestTheDerivedOrder(unittest.TestCase):
    def test_it_is_the_order_the_inventory_was_built_in(self):
        self.assertEqual(tuple(inventory_classes(_factories)), ORDER)

    def test_the_classes_are_those_that_declare_a_spec(self):
        declared = [n for n, f in _factories.items() if f.inventory_spec() is not None]
        self.assertEqual(sorted(declared), sorted(ORDER))
        # the association, value and member tables declare none
        for name in ("BackgroundModelValue", "GkSource_tags", "GkSource_parents"):
            self.assertIsNone(_factories[name].inventory_spec())

    def test_a_spec_takes_no_connection(self):
        import inspect

        for name in ORDER:
            with self.subTest(cls=name):
                hook = _factories[name].inventory_spec
                self.assertEqual(list(inspect.signature(hook).parameters), [])
                self.assertIsInstance(hook(), InventorySpec)

    def test_roots_come_first_unlike_a_plain_sort(self):
        derived = inventory_classes(_factories)
        plain = plain_sort(_factories)
        self.assertNotEqual(derived, plain)
        # every class with no reference precedes every class with one
        specs = inventory_specs(_factories)
        roots = [n for n in derived if len(specs[n].dependencies()) == 0]
        self.assertEqual(derived[: len(roots)], roots)

    def test_roots_first_on_a_hand_built_registry(self):
        factories = registry(
            R1=InventorySpec(),
            C=InventorySpec(parents={"r": Parent("r_serial", "R1")}),
            R2=InventorySpec(),
        )
        self.assertEqual(inventory_classes(factories), ["R1", "R2", "C"])
        self.assertEqual(plain_sort(factories), ["R1", "C", "R2"])

    def test_the_earliest_ready_class_in_registry_order(self):
        factories = registry(
            R=InventorySpec(),
            B=InventorySpec(parents={"a": Parent("a_serial", "A")}),
            A=InventorySpec(parents={"r": Parent("r_serial", "R")}),
            C=InventorySpec(parents={"r": Parent("r_serial", "R")}),
        )
        self.assertEqual(inventory_classes(factories), ["R", "A", "B", "C"])

    def test_every_class_follows_every_class_it_references(self):
        order = inventory_classes(_factories)
        specs = inventory_specs(_factories)
        for name in order:
            for dep in specs[name].dependencies():
                with self.subTest(cls=name, dep=dep):
                    self.assertLess(order.index(dep), order.index(name))

    def test_a_tagged_class_follows_the_tag_table(self):
        factories = registry(
            T=InventorySpec(tags=("T_tags", "t_serial")),
            store_tag=InventorySpec(leaves=("label",)),
        )
        self.assertEqual(inventory_classes(factories), ["store_tag", "T"])


# ------------------------------------------------------------------------------------------------
# 2. refusals and the polymorphic parent
# ------------------------------------------------------------------------------------------------


class TestRefusals(unittest.TestCase):
    def test_a_cycle_is_refused_by_name(self):
        factories = registry(
            R=InventorySpec(),
            A=InventorySpec(parents={"b": Parent("b_serial", "B")}),
            B=InventorySpec(parents={"a": Parent("a_serial", "A")}),
        )
        with self.assertRaisesRegex(ValueError, r"A, B form a cycle"):
            inventory_classes(factories)

    def test_a_self_reference_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"A references itself"):
            inventory_classes(
                registry(A=InventorySpec(parents={"a": Parent("a_serial", "A")}))
            )

    def test_a_parent_with_no_spec_is_refused_by_name(self):
        for spec in (None, "absent"):
            with self.subTest(parent=spec):
                classes = dict(B=InventorySpec(parents={"a": Parent("a_serial", "A")}))
                if spec is None:
                    classes["A"] = None
                with self.assertRaisesRegex(
                    ValueError, r"B references A, which declares no inventory_spec"
                ):
                    inventory_classes(registry(**classes))

    def test_a_tagged_class_whose_tag_table_declares_no_spec_is_refused(self):
        for tag_table in (None, "absent"):
            with self.subTest(tag_table=tag_table):
                classes = dict(T=InventorySpec(tags=("T_tags", "t_serial")))
                if tag_table is None:
                    classes[TAG_TABLE] = None
                with self.assertRaisesRegex(
                    ValueError,
                    rf"T references {TAG_TABLE}, which declares no inventory_spec",
                ):
                    inventory_classes(registry(**classes))

    def test_a_polymorphic_parents_classes_come_before_it(self):
        factories = registry(
            C=InventorySpec(
                leaves=("kind",),
                parents={
                    "p": Parent("p_serial", type_column="kind", types={0: "A", 1: "B"})
                },
            ),
            A=InventorySpec(),
            D=InventorySpec(parents={"c": Parent("c_serial", "C")}),
            B=InventorySpec(parents={"a": Parent("a_serial", "A")}),
        )
        self.assertEqual(inventory_classes(factories), ["A", "B", "C", "D"])

    def test_a_type_column_that_is_not_a_leaf_is_refused(self):
        factories = registry(
            A=InventorySpec(),
            C=InventorySpec(
                parents={"p": Parent("p_serial", type_column="kind", types={0: "A"})}
            ),
        )
        with self.assertRaisesRegex(ValueError, r"type column kind is not one of"):
            inventory_classes(factories)

    def test_a_malformed_parent_is_refused(self):
        cases = {
            "neither": dict(),
            "both": dict(of="A", type_column="t", types={0: "A"}),
            "types without type_column": dict(types={0: "A"}),
            "type_column without types": dict(of="A", type_column="t"),
            "empty types": dict(type_column="t", types={}),
        }
        for case, kwargs in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    Parent("c", **kwargs)

    def test_a_polymorphic_member_of_a_parent_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"member field\(s\) b are polymorphic"):
            ParentSet(
                table="T",
                owner="o",
                members={"b": Parent("b", type_column="t", types={0: "A"})},
            )

    def test_a_parent_stays_hashable(self):
        a = Parent("c", type_column="t", types={0: "A"})
        b = Parent("c", type_column="t", types={0: "A"})
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(a, b)
        self.assertNotEqual(a, Parent("c", type_column="t", types={0: "B"}))

    def test_the_reader_refuses_a_registry_before_the_order_is(self):
        """A registry that lacks a class its store records is refused by the reader's
        ``RuntimeError``, which names the primary's ``replicated_tables``, as it always was. The
        order is derived after the store is opened: derived before, the same registry would be
        refused by ``inventory_classes``' ``ValueError``, which this test shows it would be.
        """
        with tempfile.TemporaryDirectory() as tmp:
            primary = build_full_store(Path(tmp)).primary
            dropped = (
                "BackgroundModel",
                "BackgroundModel_tags",
                "BackgroundModelValue",
            )
            reduced = {n: f for n, f in _factories.items() if n not in dropped}
            with self.assertRaises(ValueError):
                inventory_classes(reduced)
            with self.assertRaises(RuntimeError) as caught:
                read_inventory(primary, reduced)
        self.assertIn("replicated_tables", str(caught.exception))
        self.assertIn("BackgroundModel", str(caught.exception))


# ------------------------------------------------------------------------------------------------
# 3. resolve() through the declared map
# ------------------------------------------------------------------------------------------------


def _class(name, records, parents=None, parent_types=None):
    return ClassInventory(
        name=name,
        replicated=True,
        tagged=False,
        parents=parents or {},
        records=tuple(records),
        count=len(records),
        earliest_timestamp=None,
        latest_timestamp=None,
        problems=(),
        parent_types=parent_types or {},
    )


class TestResolve(unittest.TestCase):
    def test_a_hand_built_inventory(self):
        a = Record(key={"x": 1}, tags=(), validated=None, value_count=None)
        b = Record(key={"x": 2}, tags=(), validated=None, value_count=None)
        da = reference_digest(a.key, (), False)
        db = reference_digest(b.key, (), False)
        children = [
            Record(key={"kind": 0, "p": da}, tags=(), validated=None, value_count=None),
            Record(key={"kind": 1, "p": db}, tags=(), validated=None, value_count=None),
            Record(key={"kind": 7, "p": da}, tags=(), validated=None, value_count=None),
        ]
        inventory = StoreInventory(
            primary=Path("x"),
            shards=(0,),
            classes={
                "A": _class("A", [a]),
                "B": _class("B", [b]),
                "C": _class(
                    "C",
                    children,
                    parents={"p": None},
                    parent_types={"p": ("kind", {0: "A", 1: "B"})},
                ),
            },
        )
        resolved = [inventory.resolve("C", r) for r in children]
        self.assertEqual(resolved[0]["p"], {"class": "A", "key": {"x": 1}, "tags": []})
        self.assertEqual(resolved[1]["p"], {"class": "B", "key": {"x": 2}, "tags": []})
        self.assertEqual(
            resolved[2]["p"], {"class": None, "digest": da, "unresolved": 0}
        )
        self.assertEqual(resolved[0]["kind"], 0)

    def test_the_full_store_resolves_both_cosmology_types(self):
        from CosmologyModels.model_ids import LAMBDACDM_IDENTIFIER, QCD_EOS_IDENTIFIER

        with tempfile.TemporaryDirectory() as tmp:
            inventory = read_inventory(build_full_store(Path(tmp)).primary, _factories)
        expected = {
            LAMBDACDM_IDENTIFIER: "LambdaCDM",
            QCD_EOS_IDENTIFIER: "QCD_Cosmology",
        }
        for name in ("wavenumber_exit_time", "BackgroundModel"):
            seen = set()
            for record in inventory[name].records:
                with self.subTest(cls=name, record=record.key):
                    resolved = inventory.resolve(name, record)["cosmology"]
                    kind = record.key["cosmology_type"]
                    self.assertEqual(resolved["class"], expected[kind])
                    self.assertIn("key", resolved)
                    seen.add(kind)
            self.assertEqual(seen, set(expected))
            self.assertIsNone(inventory[name].parents["cosmology"])
            self.assertEqual(
                inventory[name].parent_types["cosmology"],
                ("cosmology_type", expected),
            )

    def test_both_referencing_factories_declare_the_one_map(self):
        from Datastore.SQL.ObjectFactories.cosmology_types import COSMOLOGY_TYPES

        for name in ("wavenumber_exit_time", "BackgroundModel"):
            with self.subTest(cls=name):
                parent = _factories[name].inventory_spec().parents["cosmology"]
                self.assertIs(parent.types, COSMOLOGY_TYPES)
                self.assertEqual(parent.type_column, "cosmology_type")
                self.assertIsNone(parent.of)


# ------------------------------------------------------------------------------------------------
# 4. the layer names and loads no project package or table
# ------------------------------------------------------------------------------------------------

PROJECT_PACKAGES = (
    "CosmologyModels",
    "CosmologyConcepts",
    "ComputeTargets",
    "MetadataConcepts",
    "config",
)


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheLayerKnowsNoProject(unittest.TestCase):
    def test_it_imports_no_project_package(self):
        tree = ast.parse(STORE_INVENTORY.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(
            [m for m in imported if m.split(".")[0] in PROJECT_PACKAGES], []
        )

    def test_reading_records_loads_no_project_module(self):
        code = (
            "import json, sys\n"
            "import sqlalchemy as sqla\n"
            "from Datastore.store_inventory import ShardContext, read_records\n"
            "md = sqla.MetaData()\n"
            "t = sqla.Table('thing', md, sqla.Column('serial', sqla.Integer, primary_key=True),"
            " sqla.Column('x', sqla.Float))\n"
            "e = sqla.create_engine('sqlite://')\n"
            "md.create_all(e)\n"
            "with e.begin() as c:\n"
            "    c.execute(t.insert(), [{'serial': 1, 'x': 0.5}])\n"
            "    read_records(c, t, {'thing': t}, ShardContext(0, {}, {}), leaves=('x',))\n"
            f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {PROJECT_PACKAGES!r})))\n"
        )
        self.assertEqual(_run(code), [])

    def test_read_inventory_loads_only_what_the_registry_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = build_full_store(Path(tmp)).primary
            code = (
                "import json, sys\n"
                "from Datastore.store_inventory import read_inventory\n"
                "from config.datastore import factories\n"
                "before = set(sys.modules)\n"
                f"read_inventory({str(primary)!r}, factories)\n"
                "print(json.dumps(sorted(set(sys.modules) - before)))\n"
            )
            added = _run(code)
        self.assertEqual([m for m in added if m.split(".")[0] in PROJECT_PACKAGES], [])

    def test_no_project_name_in_its_code_or_text(self):
        names = (
            set(_factories)
            | set(PROJECT_PACKAGES)
            | {
                "cosmology",
                "cosmology_type",
                "cosmology_serial",
                "COSMOLOGY",
                "model_ids",
            }
        )
        # ``validated`` is not in the list: it is the name of the record's own field and of
        # ``read_records``' keyword, which the layer defines, and it is not a column it assumes
        text = STORE_INVENTORY.read_text()
        found = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NAME:
                if tok.string in names:
                    found.append((tok.start[0], tok.string))
            elif tok.type in (tokenize.STRING, tokenize.COMMENT):
                for name in names:
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                        tok.string,
                    ):
                        found.append((tok.start[0], name))
                # the naming conventions of the project's tables
                for fragment in ("_tags", "*Value", "Run_<"):
                    if fragment in tok.string:
                        found.append((tok.start[0], fragment))
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
