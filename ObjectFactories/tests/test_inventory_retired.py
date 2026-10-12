"""
Tests for store-fingerprint prompt 03, F9: the old inventory service is gone, so there is one.

Every factory lacks ``inventory``; the ``Datastore`` actor class lacks it; ``ShardedPool`` lacks
``inventory`` and its merge helper, and its constructor has no parameter for the merge policies;
``config.sharding`` has no merge policies. And prompt 03 §2 F9 item 7's ``git grep`` finds no
reference to a retired name in any Python file.

**Since dk-and-rpk-migration prompt 04 the layer is the installed package ``datastorekit``**, so
the two layer checks read the release SGK pins: the actor's source is the package's
``datastorekit/SQL/Datastore.py`` (found by ``importlib.util.find_spec``), and ``ShardedPool`` is
``datastorekit.SQL.ShardedPool``'s. SGK thereby still checks that the release it pins lacks the old
service. The grep runs over every tracked ``*.py`` file with no exclusion: the one file it used to
exclude, ``ObjectFactories/tests/test_store_inventory.py``, whose helper ``_Stores.inventory`` called the
new service, was deleted with the other tests the package now carries, and so was the test that
checked what matched in it.

**Every retired name in this module is built at run time**, so that the module does not match
the pattern it runs.
"""

import ast
import importlib.util
import inspect
import re
import subprocess
import unittest
from pathlib import Path

import config.sharding as sharding
from config.datastore import factories as _factories
from datastorekit.SQL.ShardedPool import ShardedPool

REPO_ROOT = Path(__file__).resolve().parents[2]

# the retired names, assembled so that this file does not contain them
_INVENTORY = "inven" + "tory"
_CONFIG = _INVENTORY + "_" + "config"
_MERGE = "_merge" + "_queue"
_CONFIG_TYPE = "Inven" + "tory" + "Config" + "Type"

# prompt 03 §2 F9 item 7's pattern
PATTERN = "|".join(
    [
        r"\." + _INVENTORY + r"\(",
        r"def " + _INVENTORY + r"\(",
        _CONFIG,
        _MERGE,
        _CONFIG_TYPE,
    ]
)


def git_grep(*pathspecs):
    return subprocess.run(
        ["git", "grep", "-nE", PATTERN, "--", *pathspecs],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


class TestTheOldServiceIsGone(unittest.TestCase):
    def test_no_factory_has_the_old_method(self):
        for name, factory in _factories.items():
            with self.subTest(cls=name):
                self.assertFalse(hasattr(factory, _INVENTORY))

    def test_the_datastore_actor_class_has_no_method(self):
        # Datastore is a Ray actor class, so its methods are read from its source, not its object:
        # the installed package's module, found without importing it
        source = importlib.util.find_spec("datastorekit.SQL.Datastore").origin
        tree = ast.parse(Path(source).read_text())
        (actor,) = [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Datastore"
        ]
        methods = {n.name for n in actor.body if isinstance(n, ast.FunctionDef)}
        self.assertNotIn(_INVENTORY, methods)
        self.assertIn("read_table", methods)
        assigned = {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        self.assertNotIn(_CONFIG_TYPE, assigned)

    def test_sharded_pool_has_neither_method_nor_parameter(self):
        self.assertFalse(hasattr(ShardedPool, _INVENTORY))
        self.assertFalse(hasattr(ShardedPool, _MERGE))
        parameters = inspect.signature(ShardedPool.__init__).parameters
        self.assertNotIn(_CONFIG, parameters)
        self.assertIn("read_table_config", parameters)
        self.assertIsInstance(inspect.getattr_static(ShardedPool, "primary"), property)

    def test_config_sharding_has_no_merge_policies(self):
        self.assertFalse(hasattr(sharding, _CONFIG))
        self.assertTrue(hasattr(sharding, "read_table_config"))


class TestNoReferenceRemains(unittest.TestCase):
    def test_the_grep_prints_nothing(self):
        result = git_grep("*.py")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, "")
        # git grep exits 1 when nothing matches, and 2 or more on an error
        self.assertEqual(result.returncode, 1)

    def test_this_module_does_not_match_itself(self):
        # read directly, so that the check holds before this module is committed, when git grep
        # (which searches tracked files) would not see it
        self.assertIsNone(re.search(PATTERN, Path(__file__).read_text()))


if __name__ == "__main__":
    unittest.main()
