"""
The structured inventory, ``Datastore/store_inventory.py`` ``read_inventory`` (store-fingerprint
prompt 02), on the full real store of ``real_store_fixtures.build_full_store``.

1. the keys are physical: the same content under other serials, on other shards, gives an equal
   inventory;
2. every identity column matters: changing one changes the class's records;
3. what is not identity does not matter: labels, timestamps, payload, solver serials, cosmology
   names, version foreign keys;
4. floats: the stored bits, through ``canonical`` and nothing else;
5. tags: a record's own association rows, and a tagged parent's digest covers them;
6. value counts: per parent, from one ``GROUP BY``;
7. replicated divergence is a named problem naming the shard;
8. old stores and orphans are named problems, and change nothing else;
9. duplicates are a named problem, and both are kept;
10. read-only and no Ray.

No test here starts Ray or constructs a ``ShardedPool``, and nothing is opened under ``var/``.
"""

import ast
import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest import mock

import sqlalchemy as sqla

import Datastore.store_inventory as store_inventory
from Datastore.store_inventory import (
    INVENTORY_CLASSES,
    Record,
    canonical,
    canonical_json,
    read_inventory,
)
from Datastore.tests.real_store_fixtures import (
    build_full_store,
    file_state,
    find_row,
    full_rows,
    relabel_serials,
    vary_row,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORIES = REPO_ROOT / "Datastore" / "SQL" / "ObjectFactories"

VALUE_TABLES = (
    "BackgroundModelValue",
    "TkNumericValue",
    "TkWKBValue",
    "GkNumericValue",
    "GkWKBValue",
    "GkSourceValue",
    "QuadSourceValue",
)

# class -> key field -> (table, serial, column, a new value) that changes that one identity column
# of one row. Each list is read from the class's build() lookup, optional filters included
IDENTITY = {
    "version": {"label": ("version", 2, "label", "2023.1.1")},
    "store_tag": {"label": ("store_tag", 3, "label", "renamed-tag")},
    "redshift": {"z": ("redshift", 5, "z", 2.0)},
    "wavenumber": {"k_inv_Mpc": ("wavenumber", 3, "k_inv_Mpc", 0.25)},
    "tolerance": {"log10_tol": ("tolerance", 3, "log10_tol", -11.0)},
    "LambdaCDM": {
        c: ("LambdaCDM", 2, c, v)
        for c, v in (
            ("omega_m", 0.35),
            ("omega_cc", 0.65),
            ("h", 0.72),
            ("f_baryon", 0.17),
            ("T_CMB_Kelvin", 2.8),
            ("Neff", 3.2),
        )
    },
    "QCD_Cosmology": {
        c: ("QCD_Cosmology", 1, c, v)
        for c, v in (
            ("omega_m", 0.35),
            ("omega_cc", 0.65),
            ("h", 0.72),
            ("f_baryon", 0.17),
            ("T_CMB_Kelvin", 2.8),
            ("Neff", 3.2),
            ("log10_max_z", 11.0),
            ("T_z_representation", 4),
        )
    },
    "IntegrationSolver": {
        "label": ("IntegrationSolver", 2, "label", "renamed-solver"),
        "stepping": ("IntegrationSolver", 2, "stepping", 2),
    },
    "GkSourcePolicy": {
        "Levin_threshold": ("GkSourcePolicy", 2, "Levin_threshold", 0.6),
        "numeric_policy": ("GkSourcePolicy", 2, "numeric_policy", "maximize-other"),
    },
    "QuadSourcePolicy": {
        "Levin_threshold": ("QuadSourcePolicy", 1, "Levin_threshold", 0.6),
        "numeric_policy": ("QuadSourcePolicy", 1, "numeric_policy", "maximize-other"),
    },
    "wavenumber_exit_time": {
        "cosmology_type": ("wavenumber_exit_time", 4, "cosmology_type", 0),
        "stepping": ("wavenumber_exit_time", 3, "stepping", 1),
        "k": ("wavenumber_exit_time", 4, "wavenumber_serial", 3),
        "cosmology": ("wavenumber_exit_time", 3, "cosmology_serial", 2),
        "atol": ("wavenumber_exit_time", 3, "atol_serial", 3),
        "rtol": ("wavenumber_exit_time", 3, "rtol_serial", 3),
    },
    "BackgroundModel": {
        "cosmology_type": ("BackgroundModel", 2, "cosmology_type", 0),
        "cosmology": ("BackgroundModel", 1, "cosmology_serial", 2),
        "tau_gauss_order": ("BackgroundModel", 2, "tau_gauss_order", 7),
        "cs_tau_gauss_order": ("BackgroundModel", 2, "cs_tau_gauss_order", 7),
        "friction_F_gauss_order": ("BackgroundModel", 2, "friction_F_gauss_order", 7),
        "source_grid_construction": (
            "BackgroundModel",
            2,
            "source_grid_construction",
            2,
        ),
        "source_grid_digest": (
            "BackgroundModel",
            2,
            "source_grid_digest",
            "other-digest",
        ),
        "z_init": ("BackgroundModel", 2, "z_init_serial", 5),
    },
    "TkNumericIntegration": {
        "break_point_kind": (
            "TkNumericIntegration",
            3,
            "break_point_kind",
            "discontinuity",
        ),
        "model": ("TkNumericIntegration", 3, "model_serial", 1),
        "k": ("TkNumericIntegration", 3, "wavenumber_exit_serial", 3),
        "atol": ("TkNumericIntegration", 3, "atol_serial", 3),
        "rtol": ("TkNumericIntegration", 3, "rtol_serial", 3),
        "z_init": ("TkNumericIntegration", 3, "z_init_serial", 5),
    },
    "TkWKBIntegration": {
        "rho_gauss_order": ("TkWKBIntegration", 2, "rho_gauss_order", 12),
        "z_init": ("TkWKBIntegration", 2, "z_init", 300.0),
        "model": ("TkWKBIntegration", 2, "model_serial", 2),
        "k": ("TkWKBIntegration", 2, "wavenumber_exit_serial", 4),
    },
    "GkNumericIntegration": {
        "break_point_kind": (
            "GkNumericIntegration",
            1,
            "break_point_kind",
            "discontinuity",
        ),
        "model": ("GkNumericIntegration", 1, "model_serial", 2),
        "k": ("GkNumericIntegration", 1, "wavenumber_exit_serial", 4),
        "atol": ("GkNumericIntegration", 1, "atol_serial", 3),
        "rtol": ("GkNumericIntegration", 1, "rtol_serial", 3),
        "z_source": ("GkNumericIntegration", 1, "z_source_serial", 5),
    },
    "GkWKBIntegration": {
        "rho_gauss_order": ("GkWKBIntegration", 1, "rho_gauss_order", 12),
        "z_init": ("GkWKBIntegration", 1, "z_init", 60.0),
        "model": ("GkWKBIntegration", 1, "model_serial", 2),
        "k": ("GkWKBIntegration", 1, "wavenumber_exit_serial", 4),
        "z_source": ("GkWKBIntegration", 1, "z_source_serial", 5),
    },
    "GkSource": {
        "model": ("GkSource", 2, "model_serial", 2),
        "k": ("GkSource", 2, "wavenumber_exit_serial", 4),
        "z_response": ("GkSource", 2, "z_response_serial", 5),
    },
    "GkSourcePolicyData": {
        "source": ("GkSourcePolicyData", 2, "source_serial", 1),
        "policy": ("GkSourcePolicyData", 2, "policy_serial", 1),
        "k": ("GkSourcePolicyData", 2, "wavenumber_exit_serial", 4),
    },
    "QuadSource": {
        "model": ("QuadSource", 1, "model_serial", 2),
        "q": ("QuadSource", 1, "q_wavenumber_exit_serial", 4),
        "r": ("QuadSource", 1, "r_wavenumber_exit_serial", 3),
    },
    "QuadSourceIntegral": {
        "model": ("QuadSourceIntegral", 2, "model_serial", 2),
        "policy": ("QuadSourceIntegral", 2, "policy_serial", 2),
        "k": ("QuadSourceIntegral", 2, "k_wavenumber_exit_serial", 4),
        "q": ("QuadSourceIntegral", 2, "q_wavenumber_exit_serial", 4),
        "r": ("QuadSourceIntegral", 2, "r_wavenumber_exit_serial", 3),
        "z_response": ("QuadSourceIntegral", 2, "z_response_serial", 5),
        "z_source_max": ("QuadSourceIntegral", 2, "z_source_max_serial", 4),
        "atol": ("QuadSourceIntegral", 2, "atol_serial", 3),
        "rtol": ("QuadSourceIntegral", 2, "rtol_serial", 3),
    },
    "OneLoopIntegral": {
        "model": ("OneLoopIntegral", 1, "model_serial", 2),
        "k": ("OneLoopIntegral", 1, "wavenumber_exit_serial", 4),
        "z_response": ("OneLoopIntegral", 1, "z_response_serial", 5),
        "atol": ("OneLoopIntegral", 1, "atol_serial", 3),
        "rtol": ("OneLoopIntegral", 1, "rtol_serial", 3),
    },
}

# (table, serial, column, a new value): columns that are not identity
NON_IDENTITY = [
    # compute-target labels
    ("TkNumericIntegration", 1, "label", "relabelled"),
    ("TkWKBIntegration", 1, "label", "relabelled"),
    ("GkNumericIntegration", 1, "label", "relabelled"),
    ("GkWKBIntegration", 1, "label", "relabelled"),
    ("GkSource", 1, "label", "relabelled"),
    ("QuadSource", 1, "label", "relabelled"),
    ("QuadSourceIntegral", 1, "label", "relabelled"),
    ("OneLoopIntegral", 1, "label", "relabelled"),
    ("BackgroundModel", 1, "label", "relabelled"),
    # payload and provenance
    ("TkNumericIntegration", 1, "stop_T", 0.75),
    ("TkNumericIntegration", 1, "z_samples", 9),
    ("TkNumericIntegration", 1, "z_min_serial", 2),
    ("TkWKBIntegration", 1, "T_init", 2.0),
    ("QuadSource", 1, "Tq_serial", 9),
    ("QuadSourceIntegral", 1, "total", 9.0),
    ("QuadSourceIntegral", 1, "source_serial", 2),
    ("QuadSourceIntegral", 1, "data_serial", 3),
    ("QuadSourceIntegral", 1, "metadata", "{}"),
    ("GkSourcePolicyData", 1, "crossover_z", 30.0),
    ("GkSourcePolicyData", 1, "type", 2),
    ("OneLoopIntegral", 1, "value", 0.25),
    ("wavenumber_exit_time", 1, "z_exit", 1.0),
    ("TkNumericValue", 11, "T", 7.0),
    ("BackgroundModelValue", 1, "Hubble_GeV", 7.0),
    ("GkSourceValue", 401, "G", 1.0),
    # solver serials
    ("BackgroundModel", 1, "solver_serial", 2),
    ("TkNumericIntegration", 1, "solver_serial", 2),
    ("TkWKBIntegration", 1, "phase_solver_serial", 2),
    ("TkWKBIntegration", 1, "friction_solver_serial", 2),
    ("GkNumericIntegration", 1, "solver_serial", 2),
    ("GkWKBIntegration", 1, "solver_serial", 2),
    # descriptive names and labels
    ("LambdaCDM", 1, "name", "renamed"),
    ("QCD_Cosmology", 1, "name", "renamed"),
    ("GkSourcePolicy", 1, "label", "renamed"),
    ("QuadSourcePolicy", 1, "label", "renamed"),
    # version foreign keys
    ("TkNumericIntegration", 1, "version", 2),
    ("wavenumber_exit_time", 1, "version", 2),
    ("GkSourcePolicy", 1, "version", 2),
    ("BackgroundModel", 1, "version", 2),
    ("QuadSourceIntegral", 1, "version", 2),
    # the source/response flags, which accumulate by OR
    ("redshift", 1, "source", False),
    ("wavenumber", 1, "response", False),
]


def records_of(inventory):
    return {name: c.records for name, c in inventory.classes.items()}


def changed(before, after):
    """The classes whose records differ."""
    return {n for n in before if before[n] != after[n]}


def kinds(problems):
    return [p.split(":", 1)[0] for p in problems]


class _Stores(unittest.TestCase):
    """A temporary directory, and a builder of full stores in it, each in its own directory."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls._count = 0
        cls.baseline = cls.inventory()
        cls.base = records_of(cls.baseline)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def store(cls, **kwargs):
        cls._count += 1
        directory = cls.root / f"store-{cls._count}"
        directory.mkdir()
        return build_full_store(directory, **kwargs)

    @classmethod
    def inventory(cls, **kwargs):
        return read_inventory(cls.store(**kwargs).primary)

    def varied(self, table, serial, column, value, **kwargs):
        replicated, sharded, keys = full_rows()
        vary_row(replicated, sharded, table, serial, column, value)
        return self.inventory(
            replicated=replicated, sharded=sharded, shard_keys=keys, **kwargs
        )


class TestTheFullStore(_Stores):
    def test_every_class_has_a_record(self):
        self.assertEqual(tuple(self.baseline.classes), INVENTORY_CLASSES)
        for name in INVENTORY_CLASSES:
            with self.subTest(cls=name):
                self.assertGreater(self.baseline[name].count, 0)
                self.assertEqual(
                    self.baseline[name].count, len(self.baseline[name].records)
                )

    def test_the_full_store_has_no_problem(self):
        self.assertEqual(self.baseline.problems, ())

    def test_records_are_json_safe(self):
        for name, cls in self.baseline.classes.items():
            for record in cls.records:
                with self.subTest(cls=name):
                    text = json.dumps(record.as_json(), allow_nan=False)
                    self.assertEqual(
                        set(json.loads(text)),
                        {"key", "tags", "validated", "value_count"},
                    )
                    for leaf in record.key.values():
                        self.assertIsInstance(leaf, (str, int, bool, type(None)))

    def test_classes_with_tags_validated_and_values(self):
        tagged = {
            "BackgroundModel",
            "TkNumericIntegration",
            "TkWKBIntegration",
            "GkNumericIntegration",
            "GkWKBIntegration",
            "GkSource",
            "QuadSource",
            "QuadSourceIntegral",
            "OneLoopIntegral",
        }
        validated = tagged - {"QuadSourceIntegral", "OneLoopIntegral"}
        for name, cls in self.baseline.classes.items():
            with self.subTest(cls=name):
                self.assertEqual(cls.tagged, name in tagged)
                for record in cls.records:
                    if name not in tagged:
                        self.assertEqual(record.tags, ())
                    else:
                        self.assertIn("Run_fixture", record.tags)
                        self.assertEqual(record.tags, tuple(sorted(record.tags)))
                    if name in validated:
                        self.assertIsInstance(record.validated, bool)
                        self.assertIsNotNone(record.value_count)
                    else:
                        self.assertIsNone(record.validated)
                        self.assertIsNone(record.value_count)
        # unvalidated rows are recorded, with their flag
        for name in ("BackgroundModel", "TkNumericIntegration", "QuadSource"):
            self.assertIn(False, [r.validated for r in self.baseline[name].records])

    def test_replicated_and_sharded(self):
        from config.sharding import replicated_tables

        for name, cls in self.baseline.classes.items():
            self.assertEqual(cls.replicated, name in replicated_tables)

    def test_resolve_names_every_parent(self):
        qsi = self.baseline["QuadSourceIntegral"]
        for record in qsi.records:
            resolved = self.baseline.resolve("QuadSourceIntegral", record)
            for field in qsi.parents:
                self.assertIn("key", resolved[field], field)


class TestKeysArePhysical(_Stores):
    """Test 1: the same content, built with other serials and on other shards, is equal."""

    def assertSameInventory(self, other):
        self.assertEqual(tuple(other.classes), tuple(self.baseline.classes))
        for name in self.baseline.classes:
            with self.subTest(cls=name):
                self.assertEqual(other[name].records, self.baseline[name].records)
                self.assertEqual(other[name].count, self.baseline[name].count)
                self.assertEqual(other[name].problems, ())
                self.assertEqual(
                    other[name].earliest_timestamp,
                    self.baseline[name].earliest_timestamp,
                )

    def test_different_serial_assignments(self):
        replicated, sharded, keys = relabel_serials(*full_rows(), offset=100)
        base_replicated, base_sharded, _ = full_rows()
        # every replicated class, and every sharded one, really has other serials
        for table in ("BackgroundModel", "wavenumber_exit_time", "tolerance"):
            self.assertTrue(
                {r["serial"] for r in replicated[table]}.isdisjoint(
                    {r["serial"] for r in base_replicated[table]}
                )
            )
        self.assertTrue(
            {r["serial"] for r in sharded[0]["QuadSourceIntegral"]}.isdisjoint(
                {r["serial"] for r in base_sharded[0]["QuadSourceIntegral"]}
            )
        )
        self.assertSameInventory(
            self.inventory(replicated=replicated, sharded=sharded, shard_keys=keys)
        )

    def test_sharded_rows_on_different_shards(self):
        replicated, sharded, keys = full_rows()
        moved = {0: {}, 1: sharded[1], 2: sharded[0]}
        moved_keys = {k: {0: 2, 1: 1}[s] for k, s in keys.items()}
        self.assertSameInventory(
            self.inventory(
                replicated=replicated, sharded=moved, shard_keys=moved_keys, shards=3
            )
        )

    def test_replicated_serials_permuted_and_rows_moved(self):
        replicated, sharded, keys = relabel_serials(*full_rows(), offset=37)
        moved = {0: sharded[1], 1: sharded[0]}
        moved_keys = {k: 1 - s for k, s in keys.items()}
        self.assertSameInventory(
            self.inventory(replicated=replicated, sharded=moved, shard_keys=moved_keys)
        )


class TestIdentityColumnsMatter(_Stores):
    """Test 2: each identity column, one at a time, changes the class's records."""

    def test_the_key_fields_are_the_lookup_columns(self):
        for name in INVENTORY_CLASSES:
            with self.subTest(cls=name):
                fields = {f for r in self.baseline[name].records for f in r.key}
                self.assertEqual(fields, set(IDENTITY[name]))

    def test_each_identity_column_changes_the_records(self):
        for name, fields in IDENTITY.items():
            for field, (table, serial, column, value) in fields.items():
                with self.subTest(cls=name, field=field):
                    after = self.varied(table, serial, column, value)[name].records
                    self.assertNotEqual(after, self.base[name])
                    self.assertNotEqual(
                        Counter(r.key[field] for r in after),
                        Counter(r.key[field] for r in self.base[name]),
                    )


class TestNonIdentityDoesNotMatter(_Stores):
    """Test 3: labels, timestamps, payload, solver serials, names and version keys."""

    def test_non_identity_columns(self):
        for table, serial, column, value in NON_IDENTITY:
            with self.subTest(table=table, column=column):
                self.assertEqual(
                    records_of(self.varied(table, serial, column, value)), self.base
                )

    def test_timestamps(self):
        replicated, sharded, keys = full_rows()
        tables = store_inventory_tables()
        stamp = datetime(2030, 1, 2, 3, 4, 5)
        for rows in [replicated] + list(sharded.values()):
            for table, table_rows in rows.items():
                if "timestamp" in tables[table].c:
                    for row in table_rows:
                        row["timestamp"] = stamp
        after = self.inventory(replicated=replicated, sharded=sharded, shard_keys=keys)
        self.assertEqual(records_of(after), self.base)
        self.assertEqual(after["QuadSourceIntegral"].earliest_timestamp, stamp)


def store_inventory_tables():
    from Datastore.SQL.Datastore import _factories
    from Datastore.SQL.schema import build_schema

    return build_schema(sqla.MetaData(), _factories).tables


class TestFloats(_Stores):
    """Test 4: the stored bits, through ``canonical`` only (decision D1)."""

    def test_canonical(self):
        self.assertEqual(canonical(0.1), (0.1).hex())
        self.assertEqual(canonical(-5.0), "-0x1.4000000000000p+2")
        self.assertEqual(canonical(3), 3)
        self.assertIs(canonical(True), True)
        self.assertEqual(canonical("all"), "all")
        self.assertIsNone(canonical(None))
        self.assertNotEqual(canonical(0.1), canonical(math.nextafter(0.1, 1.0)))
        for bad in (datetime(2026, 1, 1), b"x", [1.0]):
            with self.assertRaises(TypeError):
                canonical(bad)

    def test_canonical_json(self):
        self.assertEqual(
            canonical_json({"b": 1.0, "a": [0.5, "x", None]}),
            '{"a":["0x1.0000000000000p-1","x",null],"b":"0x1.0000000000000p+0"}',
        )

    def test_the_last_bit_of_k_changes_the_records(self):
        after = self.varied("wavenumber", 3, "k_inv_Mpc", math.nextafter(0.5, 1.0))
        self.assertIn("wavenumber", changed(self.base, records_of(after)))
        self.assertIn("wavenumber_exit_time", changed(self.base, records_of(after)))

    def test_log10_tol_is_used_as_stored(self):
        self.assertEqual(
            sorted(r.key["log10_tol"] for r in self.base["tolerance"]),
            sorted(float.hex(v) for v in (-5.0, -7.0, -9.0)),
        )

    def test_every_float_leaf_goes_through_canonical(self):
        real = store_inventory.canonical

        def marked(value):
            out = real(value)
            return "F:" + out if isinstance(value, float) else out

        with mock.patch.object(store_inventory, "canonical", marked):
            inventory = self.inventory()
        self.assertEqual(
            sorted(r.key["log10_tol"] for r in inventory["tolerance"].records),
            sorted("F:" + float.hex(v) for v in (-5.0, -7.0, -9.0)),
        )
        self.assertTrue(
            all(
                r.key["k_inv_Mpc"].startswith("F:")
                for r in inventory["wavenumber"].records
            )
        )

    def test_only_canonical_formats_a_float(self):
        """No code of the inventory but ``canonical`` calls ``float.hex``, ``.hex()``, ``round``,
        ``format`` or ``repr``, and no factory's ``inventory_records`` formats anything.
        """
        tree = ast.parse((REPO_ROOT / "Datastore" / "store_inventory.py").read_text())
        offenders = []
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if func.name == "canonical":
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == "hex":
                    offenders.append(f"store_inventory.{func.name}")
                if isinstance(node, ast.Name) and node.id in ("round", "format"):
                    offenders.append(f"store_inventory.{func.name}")
        canonical_def = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "canonical"
        ]
        self.assertEqual(len(canonical_def), 1)
        self.assertTrue(
            any(
                isinstance(n, ast.Attribute) and n.attr == "hex"
                for n in ast.walk(canonical_def[0])
            )
        )

        builders = 0
        for path in sorted(FACTORIES.glob("*.py")):
            for func in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(func, ast.FunctionDef)
                    and func.name == "inventory_records"
                ):
                    builders += 1
                    for node in ast.walk(func):
                        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
                            offenders.append(f"{path.name}: an f-string")
                        if isinstance(node, ast.Attribute) and node.attr in (
                            "hex",
                            "format",
                        ):
                            offenders.append(f"{path.name}: .{node.attr}")
                        if isinstance(node, ast.Name) and node.id in (
                            "float",
                            "round",
                            "repr",
                            "str",
                            "format",
                        ):
                            offenders.append(f"{path.name}: {node.id}")
        self.assertEqual(builders, len(INVENTORY_CLASSES))
        self.assertEqual(offenders, [])


class TestTags(_Stores):
    """Test 5: tags come from each record's own association table, and a tagged parent's digest
    covers them."""

    def _with(self, shard, table, row):
        replicated, sharded, keys = full_rows()
        target = replicated if shard is None else sharded[shard]
        target.setdefault(table, []).append(row)
        return records_of(
            self.inventory(replicated=replicated, sharded=sharded, shard_keys=keys)
        )

    def test_adding_a_tag_changes_that_record_and_nothing_else(self):
        after = self._with(
            0, "QuadSourceIntegral_tags", {"parent_serial": 2, "tag_serial": 4}
        )
        self.assertEqual(changed(self.base, after), {"QuadSourceIntegral"})
        gone = set(self.base["QuadSourceIntegral"]) - set(after["QuadSourceIntegral"])
        new = set(after["QuadSourceIntegral"]) - set(self.base["QuadSourceIntegral"])
        self.assertEqual((len(gone), len(new)), (1, 1))
        (gone,), (new,) = gone, new
        self.assertEqual(new.key, gone.key)
        self.assertEqual(new.validated, gone.validated)
        self.assertEqual(new.value_count, gone.value_count)
        self.assertEqual(gone.tags, ("Run_fixture",))
        self.assertEqual(new.tags, ("Run_fixture", "grid-B"))

    def test_a_tagged_parents_digest_covers_its_tags(self):
        """Two GkSource rows can differ only in their tags, so a GkSourcePolicyData's reference
        to its GkSource changes when a tag is added to the GkSource."""
        after = self._with(0, "GkSource_tags", {"parent_serial": 2, "tag_serial": 4})
        self.assertEqual(changed(self.base, after), {"GkSource", "GkSourcePolicyData"})
        gone = set(self.base["GkSourcePolicyData"]) - set(after["GkSourcePolicyData"])
        new = set(after["GkSourcePolicyData"]) - set(self.base["GkSourcePolicyData"])
        self.assertEqual((len(gone), len(new)), (1, 1))
        (gone,), (new,) = gone, new
        self.assertEqual(
            {f for f in gone.key if gone.key[f] != new.key[f]},
            {"source"},
        )

    def test_a_tag_on_the_background_model_reaches_its_children(self):
        after = self._with(
            None, "BackgroundModel_tags", {"model_serial": 2, "tag_serial": 4}
        )
        # model 2 is the parent of TkNumericIntegration 3 only
        self.assertEqual(
            changed(self.base, after), {"BackgroundModel", "TkNumericIntegration"}
        )

    def test_oneloop_tags_come_from_its_own_table(self):
        (record,) = self.base["OneLoopIntegral"]
        self.assertEqual(record.tags, ("Run_fixture", "oneloop-tag"))
        # QuadSourceIntegral 1 has the same serial as OneLoopIntegral 1, and other tags
        after = self._with(
            0, "QuadSourceIntegral_tags", {"parent_serial": 1, "tag_serial": 4}
        )
        self.assertEqual(changed(self.base, after), {"QuadSourceIntegral"})
        after = self._with(
            0, "OneLoopIntegral_tags", {"parent_serial": 1, "tag_serial": 4}
        )
        self.assertEqual(changed(self.base, after), {"OneLoopIntegral"})
        (record,) = after["OneLoopIntegral"]
        self.assertEqual(record.tags, ("Run_fixture", "grid-B", "oneloop-tag"))

    def test_a_tag_no_row_carries_is_only_a_store_tag(self):
        labels = {r.key["label"] for r in self.base["store_tag"]}
        self.assertIn("unused-tag", labels)
        carried = {t for c in self.base.values() for r in c for t in r.tags}
        self.assertNotIn("unused-tag", carried)


class TestValueCounts(_Stores):
    """Test 6: each record carries its own value count."""

    # value table -> (the parent class, the value serial deleted)
    DELETIONS = {
        "BackgroundModelValue": ("BackgroundModel", 4),
        "TkNumericValue": ("TkNumericIntegration", 12),
        "TkWKBValue": ("TkWKBIntegration", 101),
        "GkNumericValue": ("GkNumericIntegration", 201),
        "GkWKBValue": ("GkWKBIntegration", 301),
        "GkSourceValue": ("GkSource", 401),
        "QuadSourceValue": ("QuadSource", 501),
    }

    def test_the_counts_are_per_parent(self):
        expected = {
            "BackgroundModel": [3, 2],
            "TkNumericIntegration": [3, 4, 2],
            "TkWKBIntegration": [3, 1, 2],
            "GkNumericIntegration": [3, 1],
            "GkWKBIntegration": [2, 1],
            "GkSource": [3, 2, 1],
            "QuadSource": [3, 2],
        }
        for name, counts in expected.items():
            with self.subTest(cls=name):
                self.assertEqual(
                    Counter(r.value_count for r in self.base[name]), Counter(counts)
                )

    def test_deleting_a_value_row_lowers_one_count_by_one(self):
        for table, (name, serial) in self.DELETIONS.items():
            with self.subTest(table=table):
                replicated, sharded, keys = full_rows()
                for rows in [replicated] + list(sharded.values()):
                    if table in rows:
                        rows[table] = [r for r in rows[table] if r["serial"] != serial]
                after = records_of(
                    self.inventory(
                        replicated=replicated, sharded=sharded, shard_keys=keys
                    )
                )
                self.assertEqual(changed(self.base, after), {name})
                gone = set(self.base[name]) - set(after[name])
                new = set(after[name]) - set(self.base[name])
                self.assertEqual((len(gone), len(new)), (1, 1))
                (gone,), (new,) = gone, new
                self.assertEqual((new.key, new.tags), (gone.key, gone.tags))
                self.assertEqual(new.value_count, gone.value_count - 1)

    def test_value_tables_are_only_counted(self):
        """No statement reads a *Value table except one GROUP BY count (F7)."""
        statements = []

        def capture(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        store = self.store()
        sqla.event.listen(sqla.engine.Engine, "before_cursor_execute", capture)
        try:
            read_inventory(store.primary)
        finally:
            sqla.event.remove(sqla.engine.Engine, "before_cursor_execute", capture)

        touching = [
            s
            for s in statements
            if any(re.search(rf'\b"?{t}"?\b', s) for t in VALUE_TABLES)
            and not s.startswith("PRAGMA")
        ]
        self.assertGreater(len(touching), 0)
        for statement in touching:
            self.assertIn("count(*)", statement)
            self.assertIn("GROUP BY", statement)


class TestReplicatedDivergence(_Stores):
    """Test 7: a replicated row changed on one shard only is named, with its shard."""

    def _diverged(self, statements, shards=2):
        inventory = self.inventory(extra_sql=statements, shards=shards)
        return inventory, {
            n: [p for p in c.problems if p.startswith("replicated-divergence")]
            for n, c in inventory.classes.items()
            if len(c.problems) > 0
        }

    def test_a_changed_row(self):
        inventory, problems = self._diverged(
            {1: ["UPDATE tolerance SET log10_tol = -10.0 WHERE serial = 3"]}
        )
        self.assertEqual(set(problems), {"tolerance"})
        (problem,) = problems["tolerance"]
        self.assertIn("shard #1 differs from shard #0", problem)
        # the records are the lowest-serial shard's
        self.assertEqual(inventory["tolerance"].records, self.base["tolerance"])

    def test_a_changed_tag_set(self):
        inventory, problems = self._diverged(
            {
                1: [
                    "DELETE FROM BackgroundModel_tags WHERE model_serial = 1 "
                    "AND tag_serial = 2"
                ]
            }
        )
        self.assertIn("BackgroundModel", problems)
        self.assertIn("shard #1", problems["BackgroundModel"][0])
        self.assertEqual(
            inventory["BackgroundModel"].records, self.base["BackgroundModel"]
        )

    def test_a_changed_value_count(self):
        _, problems = self._diverged(
            {1: ["DELETE FROM BackgroundModelValue WHERE serial = 5"]}
        )
        self.assertEqual(set(problems), {"BackgroundModel"})
        self.assertIn("shard #1", problems["BackgroundModel"][0])

    def test_the_right_shard_of_three(self):
        _, problems = self._diverged(
            {2: ["UPDATE tolerance SET log10_tol = -10.0 WHERE serial = 3"]}, shards=3
        )
        (problem,) = problems["tolerance"]
        self.assertIn("shard #2 differs from shard #0", problem)
        self.assertNotIn("shard #1", problem)


class TestOldStoresAndOrphans(_Stores):
    """Test 8: each case is a named problem, and nothing else changes. On a three-shard store,
    whose shard 2 holds only replicated rows."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.three = records_of(cls.inventory(shards=3))

    def check(self, name, kind, shard, expected=None, **kwargs):
        inventory = self.inventory(shards=3, **kwargs)
        problems = [p for c in inventory.classes.values() for p in c.problems]
        self.assertEqual(len(problems), 1, problems)
        self.assertEqual(kinds(inventory[name].problems), [kind])
        self.assertIn(f"shard #{shard}", inventory[name].problems[0])
        after = records_of(inventory)
        if expected is None:
            self.assertEqual(after, self.three)
        else:
            self.assertEqual(changed(self.three, after), {name})
            self.assertEqual(after[name], expected)
        return inventory[name].problems[0]

    def test_an_absent_table(self):
        problem = self.check(
            "OneLoopIntegral",
            "absent-table",
            2,
            missing_tables={2: ["OneLoopIntegral", "OneLoopIntegral_tags"]},
        )
        self.assertIn("OneLoopIntegral", problem)

    def test_an_absent_association_table(self):
        problem = self.check(
            "OneLoopIntegral",
            "absent-table",
            2,
            missing_tables={2: ["OneLoopIntegral_tags"]},
        )
        self.assertIn("OneLoopIntegral_tags", problem)

    def test_an_absent_value_table(self):
        problem = self.check(
            "TkWKBIntegration", "absent-table", 2, missing_tables={2: ["TkWKBValue"]}
        )
        self.assertIn("TkWKBValue", problem)

    def test_an_absent_replicated_table(self):
        self.check(
            "QuadSourcePolicy",
            "absent-table",
            2,
            missing_tables={2: ["QuadSourcePolicy"]},
        )

    def test_an_absent_replicated_association_table(self):
        self.check(
            "BackgroundModel",
            "absent-table",
            2,
            missing_tables={2: ["BackgroundModel_tags"]},
        )

    def test_a_missing_key_column_makes_the_class_incomplete(self):
        # TkWKBIntegration 3 is the only one on shard 1: the class is the store without it
        replicated, sharded, keys = full_rows()
        sharded[1]["TkWKBIntegration"] = []
        sharded[1]["TkWKB_tags"] = []
        sharded[1]["TkWKBValue"] = []
        without = self.inventory(
            shards=3, replicated=replicated, sharded=sharded, shard_keys=keys
        )["TkWKBIntegration"].records
        self.assertEqual(len(without), len(self.three["TkWKBIntegration"]) - 1)
        problem = self.check(
            "TkWKBIntegration",
            "incomplete",
            1,
            expected=without,
            missing_columns={1: {"TkWKBIntegration": ["z_init"]}},
        )
        self.assertIn("z_init", problem)

    def test_a_missing_column_the_key_does_not_need(self):
        inventory = self.inventory(
            shards=3, missing_columns={0: {"TkNumericIntegration": ["stop_Tprime"]}}
        )
        self.assertEqual(inventory.problems, ())
        self.assertEqual(records_of(inventory), self.three)

    def test_an_orphan_value_row(self):
        problem = self.check(
            "TkNumericIntegration",
            "orphan-value",
            0,
            extra_sql={
                0: [
                    "INSERT INTO TkNumericValue (serial, integration_serial, z_serial, T, "
                    "Tprime) VALUES (9001, 999, 1, 0.0, 0.0)"
                ]
            },
        )
        self.assertIn("999", problem)

    def test_a_tag_row_whose_parent_is_missing(self):
        problem = self.check(
            "TkNumericIntegration",
            "orphan-tag",
            0,
            extra_sql={
                0: [
                    "INSERT INTO TkNumeric_tags (integration_serial, tag_serial) "
                    "VALUES (999, 1)"
                ]
            },
        )
        self.assertIn("999", problem)

    def test_a_tag_row_whose_tag_is_missing(self):
        problem = self.check(
            "TkNumericIntegration",
            "orphan-tag",
            1,
            extra_sql={
                1: [
                    "INSERT INTO TkNumeric_tags (integration_serial, tag_serial) "
                    "VALUES (2, 999)"
                ]
            },
        )
        self.assertIn("999", problem)

    def test_a_reference_that_cannot_be_resolved(self):
        replicated, sharded, keys = full_rows()
        row = copy.deepcopy(find_row(replicated, sharded, "TkNumericIntegration", 1))
        row.update(serial=77, model_serial=999)
        sharded[0]["TkNumericIntegration"].append(row)
        problem = self.check(
            "TkNumericIntegration",
            "unresolved-parent",
            0,
            replicated=replicated,
            sharded=sharded,
            shard_keys=keys,
        )
        self.assertIn("77", problem)
        self.assertIn("model", problem)

    def test_an_unknown_cosmology_type(self):
        replicated, sharded, keys = full_rows()
        row = copy.deepcopy(find_row(replicated, sharded, "wavenumber_exit_time", 3))
        row.update(serial=9, cosmology_type=7)
        replicated["wavenumber_exit_time"].append(row)
        inventory = self.inventory(
            shards=3, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        problems = inventory["wavenumber_exit_time"].problems
        self.assertEqual(kinds(problems), ["unresolved-parent"] * 3)
        for shard, problem in enumerate(problems):
            self.assertIn(f"shard #{shard}", problem)
            self.assertIn("cosmology", problem)
        self.assertEqual(records_of(inventory), self.three)


class TestDuplicates(_Stores):
    """Test 9: two records with the same key and tags are a named problem, and both are kept."""

    def _duplicate(self, table, serial, new_serial, shard, tag_table=None, column=None):
        replicated, sharded, keys = full_rows()
        row = copy.deepcopy(find_row(replicated, sharded, table, serial))
        row["serial"] = new_serial
        target = replicated if shard is None else sharded[shard]
        target.setdefault(table, []).append(row)
        if tag_table is not None:
            for rows in [replicated] + list(sharded.values()):
                for tag in [t for t in rows.get(tag_table, []) if t[column] == serial]:
                    target.setdefault(tag_table, []).append(
                        dict(tag, **{column: new_serial})
                    )
        return self.inventory(replicated=replicated, sharded=sharded, shard_keys=keys)

    def test_on_one_shard(self):
        inventory = self._duplicate(
            "QuadSourceIntegral", 1, 9, 0, "QuadSourceIntegral_tags", "parent_serial"
        )
        self.assertEqual(kinds(inventory["QuadSourceIntegral"].problems), ["duplicate"])
        self.assertIn("#0/1", inventory["QuadSourceIntegral"].problems[0])
        self.assertIn("#0/9", inventory["QuadSourceIntegral"].problems[0])
        self.assertEqual(
            inventory["QuadSourceIntegral"].count,
            self.baseline["QuadSourceIntegral"].count + 1,
        )

    def test_across_shards(self):
        inventory = self._duplicate(
            "QuadSourceIntegral", 1, 9, 1, "QuadSourceIntegral_tags", "parent_serial"
        )
        self.assertEqual(kinds(inventory["QuadSourceIntegral"].problems), ["duplicate"])
        self.assertIn("#1/9", inventory["QuadSourceIntegral"].problems[0])

    def test_a_replicated_class(self):
        inventory = self._duplicate("tolerance", 1, 4, None)
        self.assertEqual(kinds(inventory["tolerance"].problems), ["duplicate"])
        self.assertEqual(inventory["tolerance"].count, 4)

    def test_other_tags_are_not_a_duplicate(self):
        replicated, sharded, keys = full_rows()
        row = copy.deepcopy(find_row(replicated, sharded, "QuadSourceIntegral", 1))
        row["serial"] = 9
        sharded[0]["QuadSourceIntegral"].append(row)
        inventory = self.inventory(
            replicated=replicated, sharded=sharded, shard_keys=keys
        )
        self.assertEqual(inventory.problems, ())


class TestReadOnlyAndNoRay(_Stores):
    """Test 10: prompt 01's never-writes check across a full read_inventory, and no Ray."""

    def test_read_inventory_writes_nothing(self):
        store = self.store()
        before = file_state(store.directory)
        inventory = read_inventory(store.primary)
        self.assertGreater(inventory["QuadSourceIntegral"].count, 0)
        after = file_state(store.directory)
        self.assertEqual(before, after)
        self.assertFalse(
            any(n.endswith(("-journal", "-wal", "-shm")) for n in after[0])
        )

    def test_no_ray(self):
        store = self.store()
        code = (
            "import json, sys\n"
            "from Datastore.store_inventory import read_inventory\n"
            "inventory = read_inventory(sys.argv[1])\n"
            "import ray\n"
            "print(json.dumps({'ray': ray.is_initialized(), "
            "'qsi': inventory['QuadSourceIntegral'].count}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(store.primary)],
            cwd=str(self.root),
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report, {"ray": False, "qsi": 3})


if __name__ == "__main__":
    unittest.main()
