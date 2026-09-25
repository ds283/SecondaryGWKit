"""
Tests for store-fingerprint prompt 03, F9: the old inventory service is gone, so there is one.

Every factory lacks ``inventory``; the ``Datastore`` actor class lacks it; ``ShardedPool`` lacks
``inventory`` and its merge helper, and its constructor has no parameter for the merge policies;
``config.sharding`` has no merge policies. And prompt 03 §2 F9 item 7's ``git grep`` finds no
reference to a retired name in any Python file.

**The grep is narrowed by one file** (the user's decision of 2026-09-25, relayed by the
orchestrator). ``Datastore/tests/test_store_inventory.py``, prompt 02's test module, has a helper
``_Stores.inventory`` that calls ``read_inventory`` -- the new service -- and whose definition and
calls the pattern matches. README §5 rule 7 keeps that module as it is. So the grep excludes it by
a pathspec, and a second test requires that every match of the pattern in that file is the helper's
definition or one of its calls through ``cls`` or ``self``, so that a real caller of the old
service cannot hide there.

**Every retired name in this module is built at run time**, so that the module does not match
the pattern it runs.
"""

import ast
import inspect
import re
import subprocess
import unittest
from pathlib import Path

import config.sharding as sharding
from Datastore.SQL.Datastore import _factories
from Datastore.SQL.ShardedPool import ShardedPool

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

# the one file the grep excludes, and what is allowed to match in it
HELPER_MODULE = "Datastore/tests/test_store_inventory.py"
HELPER_MATCH = re.compile(
    r"(?:\b(?:cls|self)\." + _INVENTORY + r"\(|\bdef " + _INVENTORY + r"\(cls\b)"
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
        # Datastore is a Ray actor class, so its methods are read from its source, not its object
        tree = ast.parse((REPO_ROOT / "Datastore" / "SQL" / "Datastore.py").read_text())
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
        result = git_grep("*.py", f":!{HELPER_MODULE}")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, "")
        # git grep exits 1 when nothing matches, and 2 or more on an error
        self.assertEqual(result.returncode, 1)

    def test_the_excluded_file_holds_only_the_new_service_helper(self):
        result = git_grep(HELPER_MODULE)
        self.assertEqual(result.stderr, "")
        lines = result.stdout.splitlines()
        self.assertGreater(len(lines), 0)
        pattern = re.compile(PATTERN)
        for line in lines:
            text = line.split(":", 2)[2]
            with self.subTest(line=line):
                matches = [m.start() for m in pattern.finditer(text)]
                self.assertGreater(len(matches), 0)
                # every match of the pattern on the line lies inside an allowed helper match
                allowed = [(m.start(), m.end()) for m in HELPER_MATCH.finditer(text)]
                for start in matches:
                    self.assertTrue(
                        any(a <= start < b for a, b in allowed),
                        f"a match at column {start} is not the helper",
                    )

        # and the helper is _Stores.inventory, which reads the new service
        tree = ast.parse((REPO_ROOT / HELPER_MODULE).read_text())
        defined = [
            (cls.name, fn)
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef)
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name == _INVENTORY
        ]
        self.assertEqual([name for name, _ in defined], ["_Stores"])
        calls = {
            ast.unparse(n.func)
            for n in ast.walk(defined[0][1])
            if isinstance(n, ast.Call)
        }
        self.assertIn("read_inventory", calls)
        module_functions = [
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == _INVENTORY
        ]
        self.assertEqual(module_functions, [])

    def test_this_module_does_not_match_itself(self):
        # read directly, so that the check holds before this module is committed, when git grep
        # (which searches tracked files) would not see it
        self.assertIsNone(re.search(PATTERN, Path(__file__).read_text()))


if __name__ == "__main__":
    unittest.main()
