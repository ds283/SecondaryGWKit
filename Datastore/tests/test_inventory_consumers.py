"""
Tests for store-fingerprint prompt 03, F8 items 4-5: the two consumers of the inventory read the
structured inventory.

4. **``main.py --inventory`` is read-only and needs no Ray.** It is run in a child interpreter,
   from the repository root, through a guard that replaces ``ray.init`` with a function that
   raises, so a ``ray.init`` reached by mistake fails at once, whatever cluster is running. It
   exits 0 and prints the report; the store's directory is unchanged, and its unvalidated rows are
   all still there ([00-inventory-run-prunes-unvalidated-rows-by-default]). With ``--drop`` it is
   refused, and on a path that does not exist it exits non-zero and no file appears.
5. **The branch comes first.** In ``main.py``, read with ``ast`` as ``load_main_py_functions``
   reads it, the top-level ``if args.inventory:`` comes before ``ray.init`` and before the ``with
   ShardedPool(...)`` statement, and nothing between ``parse_args`` and it constructs a pool, an
   actor or an engine.
6. **The run labels.** ``available_run_labels`` reads the ``store_tag`` class of the structured
   inventory through ``pool.primary``, refuses a problem in that class, and is indifferent to
   problems elsewhere. ``ShardedPool.primary`` returns ``_primary_file``.

Stores are built with ``build_full_store`` in a temporary directory. No Ray; nothing under
``var/``.
"""

import ast
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.tests.inventory_report_parsing import sections
from Datastore.tests.real_store_fixtures import (
    build_full_store,
    file_state,
    full_rows,
    with_rows,
)
from extract_common import available_run_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"
EXTRACT_COMMON = REPO_ROOT / "extract_common.py"

# run main.py as __main__ in this interpreter, with ray.init replaced by a function that raises
_GUARDED_MAIN = """
import runpy
import sys

import ray


def _refuse(*args, **kwargs):
    raise RuntimeError("GUARD: ray.init was reached")


ray.init = _refuse
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
"""


def run_main(*args):
    return subprocess.run(
        [sys.executable, "-c", _GUARDED_MAIN, *[str(a) for a in args]],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )


def unvalidated_rows(store):
    """(shard, table) -> the serials of its rows with validated = 0, read with sqlite3 mode=ro."""
    out = {}
    for serial, path in sorted(store.shard_files.items()):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            ]
            for table in tables:
                columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                if "validated" in columns:
                    out[(serial, table)] = sorted(
                        r[0]
                        for r in conn.execute(
                            f'SELECT serial FROM "{table}" WHERE validated = 0'
                        )
                    )
        finally:
            conn.close()
    return out


class _Stores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls._count = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def store(cls, **kwargs):
        cls._count += 1
        directory = cls.root / f"store-{cls._count}"
        directory.mkdir()
        return build_full_store(directory, **kwargs)


class TestMainInventoryIsReadOnly(_Stores):
    def test_it_prints_the_report_and_writes_nothing(self):
        store = self.store()
        before = file_state(store.directory)
        unvalidated = unvalidated_rows(store)
        self.assertGreater(sum(len(v) for v in unvalidated.values()), 0)

        result = run_main("--database", store.primary, "--inventory")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("GUARD", result.stderr)
        self.assertTrue(
            result.stdout.startswith(f"== Datastore inventory: {store.primary} ==\n"),
            result.stdout[:200],
        )
        self.assertEqual(sections(result.stdout)["QuadSourceIntegral"].count, 3)
        self.assertEqual(file_state(store.directory), before)
        # the issue this closes: the default --prune-unvalidated no longer deletes anything
        self.assertEqual(unvalidated_rows(store), unvalidated)

    def test_verbose(self):
        store = self.store()
        before = file_state(store.directory)
        result = run_main(
            "--database", store.primary, "--inventory", "--inventory-verbose"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(sections(result.stdout)["redshift"].records), 5)
        self.assertEqual(file_state(store.directory), before)

    def test_drop_is_refused(self):
        store = self.store()
        before = file_state(store.directory)
        result = run_main(
            "--database", store.primary, "--inventory", "--drop", "tk-numeric"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("GUARD", result.stderr)
        self.assertIn("refuses --drop", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(file_state(store.directory), before)

    def test_a_path_that_does_not_exist_is_refused_and_not_created(self):
        directory = self.root / "empty"
        directory.mkdir()
        missing = directory / "nothing-here.sqlite"
        result = run_main("--database", missing, "--inventory")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("GUARD", result.stderr)
        self.assertIn(str(missing), result.stderr)
        self.assertEqual(list(directory.iterdir()), [])


class TestTheBranchComesFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY)).body

    def index_of(self, predicate, what):
        found = [i for i, node in enumerate(self.body) if predicate(node)]
        self.assertEqual(len(found), 1, f"{what}: {found}")
        return found[0]

    def test_the_inventory_branch_precedes_ray_and_the_pool(self):
        def is_parse_args(node):
            return (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "parse_args"
            )

        def is_inventory_branch(node):
            return (
                isinstance(node, ast.If) and ast.unparse(node.test) == "args.inventory"
            )

        def is_ray_init(node):
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "ray.init"
            )

        def is_pool(node):
            return isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) == "ShardedPool"
                for item in node.items
            )

        parse = self.index_of(is_parse_args, "parse_args")
        branch = self.index_of(is_inventory_branch, "if args.inventory:")
        ray_init = self.index_of(is_ray_init, "ray.init(...)")
        pool = self.index_of(is_pool, "with ShardedPool(...)")

        self.assertLess(parse, branch)
        self.assertLess(branch, ray_init)
        self.assertLess(branch, pool)

        # nothing between parse_args and the branch constructs a pool, an actor or an engine
        forbidden = ("ShardedPool", "Datastore", "create_engine", "remote", "options")
        for node in self.body[parse + 1 : branch]:
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                name = ast.unparse(call.func)
                for word in forbidden:
                    self.assertNotIn(word, name, ast.unparse(node))
                self.assertNotEqual(name, "ray.init")

    def test_the_branch_reads_the_closed_store_and_exits(self):
        (branch,) = [
            node
            for node in self.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.inventory"
        ]
        calls = {
            ast.unparse(n.func) for n in ast.walk(branch) if isinstance(n, ast.Call)
        }
        self.assertIn("read_inventory", calls)
        self.assertIn("format_inventory_report", calls)
        self.assertIn("sys.exit", calls)
        self.assertFalse(any("ShardedPool" in c or "ray" in c for c in calls), calls)


class TestRunLabels(_Stores):
    def test_the_run_labels_from_the_structured_inventory(self):
        replicated, sharded, keys = full_rows()
        replicated = with_rows(
            replicated,
            {"store_tag": [{"serial": 20, "label": "Run_second"}]},
        )
        store = self.store(replicated=replicated, sharded=sharded, shard_keys=keys)
        before = file_state(store.directory)

        pool = SimpleNamespace(primary=store.primary)
        # "Run_second" is carried by no record, and is still a run: the answer is unchanged
        self.assertEqual(available_run_labels(pool), ["fixture", "second"])
        self.assertEqual(file_state(store.directory), before)

    def test_a_problem_in_store_tag_is_a_refusal(self):
        store = self.store(
            extra_sql={1: ["UPDATE store_tag SET label = 'Run_other' WHERE serial = 3"]}
        )
        pool = SimpleNamespace(primary=store.primary)
        with self.assertRaises(RuntimeError) as raised:
            available_run_labels(pool)
        message = str(raised.exception)
        self.assertIn("replicated-divergence: store_tag", message)
        self.assertIn("shard #1", message)

    def test_a_problem_elsewhere_does_not_matter(self):
        store = self.store(missing_tables={1: ["QuadSourceIntegral_tags"]})
        pool = SimpleNamespace(primary=store.primary)
        self.assertEqual(available_run_labels(pool), ["fixture"])

    def test_extract_common_does_not_call_the_pool_inventory(self):
        tree = ast.parse(EXTRACT_COMMON.read_text())
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn("inventory", attributes)

    def test_sharded_pool_primary(self):
        pool = ShardedPool.__new__(ShardedPool)
        pool._primary_file = self.root / "somewhere.sqlite"
        self.assertEqual(pool.primary, self.root / "somewhere.sqlite")
        with self.assertRaises(AttributeError):
            pool.primary = self.root / "elsewhere.sqlite"


if __name__ == "__main__":
    unittest.main()
