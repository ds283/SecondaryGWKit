"""
Every Ray instance this repository starts gives Ray the temp dir of ``config/ray_local.py``
(prompts/rpk-v030-adoption, prompt 01).

The shared rehearsal lock, ``raypoolkit.rehearsal_lock`` (RayPoolKit's ``docs/rehearsal-lock.md``
§6), finds a Ray instance by its temp dir: a process is the instance's if its command line names
``/tmp/ray-sgk``, or if it descends from one that does. An instance started without that temp dir
is one the lock cannot see: ``acquire`` would not refuse while it runs, ``release`` would not
refuse while it is still up, and ``ray-processes`` would say none after a hard kill.

There are ten starts (the campaign README §0):

- three wrappers that replace ``ray.init`` in the drivers, each calling the real one as
  ``_real_ray_init(address="local", ..., _temp_dir=RAY_TEMP_DIR)``: ``run_child``'s and
  ``run_build``'s in ``RunDrivers/quadsource_atol_sweep.py``, and ``main``'s in
  ``RunDrivers/scoped_pipeline_run.py``;
- seven module-level ``ray.init`` calls, in ``main.py`` and the six readers, which give the temp
  dir only with ``--ray-address local``: with ``auto`` it would change which cluster Ray finds,
  and with a ``host:port`` it would split the driver's files from the cluster's.

The scripts and the drivers cannot be imported here, so every live ``.py`` file is read with
``ast``, as ``ComputeTargets/tests/test_ray_address_default.py`` reads the scripts. A start is keyed
by its file and the qualified name of the function it is called in (``<module>`` at module scope),
so the two wrappers called ``_local_ray_init`` in one file stay two. A start added anywhere fails
``test_the_ray_starts_are_the_ten`` until it is added to ``STARTS`` and checked by the others.

No test here starts Ray.
"""

import ast
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAY_LOCAL = "config/ray_local.py"
TEMP_DIR = "/tmp/ray-sgk"

# never walked: the venv, records, run state, every test package, and hidden directories
# (.git, .idea, .claude and its worktrees)
EXCLUDED_DIRS = {"venv", "docs", "prompts", "var", "tests"}

WRAPPERS = (
    ("RunDrivers/quadsource_atol_sweep.py", "run_build._local_ray_init"),
    ("RunDrivers/quadsource_atol_sweep.py", "run_child._local_ray_init"),
    ("RunDrivers/scoped_pipeline_run.py", "main._local_ray_init"),
)

SCRIPTS = tuple(
    (name, "<module>")
    for name in (
        "main.py",
        "extract_Gk_data.py",
        "extract_GkSource_data.py",
        "extract_GkWKB_data.py",
        "extract_QuadSourceIntegral_data.py",
        "extract_TkWKB_data.py",
        "extract_tensor_source_data.py",
    )
)

STARTS = WRAPPERS + SCRIPTS

SCRIPT_TEMP_DIR = "RAY_TEMP_DIR if args.ray_address == 'local' else None"


def _live_python_files():
    """Every ``.py`` file under the repository root outside ``EXCLUDED_DIRS`` and hidden
    directories, as paths relative to the root with ``/`` separators."""
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = sorted(
            d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if name.endswith(".py"):
                found.append(Path(dirpath, name).relative_to(REPO_ROOT).as_posix())
    return found


def _is_start(node) -> bool:
    """A call to ``ray.init`` or to a driver's saved ``_real_ray_init``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "_real_ray_init"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "init"
        and isinstance(func.value, ast.Name)
        and func.value.id == "ray"
    )


def _starts_in(tree):
    """``(qualified name of the enclosing function or <module>, call)`` for each start."""
    found = []

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = scope + (node.name,)
        if _is_start(node):
            found.append((".".join(scope) or "<module>", node))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return found


def _keywords(call: ast.Call) -> dict:
    return {kw.arg: kw.value for kw in call.keywords}


class TestRayTempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _live_python_files()
        cls.trees = {
            name: ast.parse((REPO_ROOT / name).read_text(), filename=name)
            for name in cls.files
        }
        cls.starts = {}
        for name, tree in cls.trees.items():
            for scope, call in _starts_in(tree):
                cls.starts.setdefault((name, scope), []).append(call)

    def test_the_temp_dir_is_the_constant(self):
        """The one spelling of the temp dir. The lock compares it as a string (contract §5), so a
        different spelling here, a trailing ``/`` included, gives every start a temp dir the
        lock's ``--ray-temp-dir /tmp/ray-sgk`` does not name: an SGK Ray instance the lock
        cannot see."""
        tree = self.trees[RAY_LOCAL]
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "RAY_TEMP_DIR" for t in node.targets
            )
        ]
        self.assertEqual(len(values), 1, "one module-level RAY_TEMP_DIR")
        self.assertIsInstance(values[0], ast.Constant)
        self.assertEqual(values[0].value, TEMP_DIR)

    def test_the_ray_starts_are_the_ten(self):
        """A Ray start anywhere in the live tree that is not one of the ten, or a second start
        where there was one, is a start no other test here checks: possibly an SGK Ray instance
        the lock cannot see."""
        found = sorted(key for key, calls in self.starts.items() for _ in calls)
        self.assertEqual(len(found), len(STARTS), found)
        self.assertEqual(found, sorted(STARTS))

    def test_each_wrapper_starts_a_local_instance_in_the_temp_dir(self):
        """A wrapper whose real ``ray.init`` has no ``_temp_dir`` starts an SGK Ray instance the
        lock cannot see; one with no ``address="local"`` may instead attach, with that temp dir,
        to a cluster ``RAY_ADDRESS`` names."""
        for key in WRAPPERS:
            with self.subTest(start=key):
                calls = self.starts.get(key, [])
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertIsInstance(call.func, ast.Name)
                kw = _keywords(call)
                self.assertIn("address", kw, "the real ray.init passes no address")
                self.assertIsInstance(kw["address"], ast.Constant)
                self.assertEqual(kw["address"].value, "local")
                self.assertIn("_temp_dir", kw, "the real ray.init passes no temp dir")
                self.assertIsInstance(kw["_temp_dir"], ast.Name)
                self.assertEqual(kw["_temp_dir"].id, "RAY_TEMP_DIR")

    def test_each_script_gives_the_temp_dir_with_local_only(self):
        """A script whose ``ray.init`` has no ``_temp_dir`` starts, with ``--ray-address local``,
        an SGK Ray instance the lock cannot see. Given unconditionally, the temp dir would
        change which cluster ``auto`` finds, or split a ``host:port`` driver's files from the
        cluster's."""
        for key in SCRIPTS:
            with self.subTest(start=key):
                calls = self.starts.get(key, [])
                self.assertEqual(len(calls), 1)
                kw = _keywords(calls[0])
                self.assertIn("_temp_dir", kw, "ray.init passes no temp dir")
                self.assertEqual(ast.unparse(kw["_temp_dir"]), SCRIPT_TEMP_DIR)

    def test_each_start_takes_the_temp_dir_from_config(self):
        """Each file with a start imports ``RAY_TEMP_DIR`` from ``config.ray_local`` and binds no
        ``RAY_TEMP_DIR`` of its own: a local value is a spelling the lock may not name, and so
        an SGK Ray instance the lock cannot see."""
        for name in sorted({name for name, _ in STARTS}):
            with self.subTest(file=name):
                tree = self.trees[name]
                imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and any(alias.name == "RAY_TEMP_DIR" for alias in node.names)
                ]
                self.assertGreaterEqual(len(imports), 1, "RAY_TEMP_DIR is not imported")
                for node in imports:
                    self.assertEqual(node.module, "config.ray_local")
                    self.assertEqual(node.level, 0)
                    alias = [a for a in node.names if a.name == "RAY_TEMP_DIR"][0]
                    self.assertIsNone(alias.asname)
                bound = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and node.id == "RAY_TEMP_DIR"
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                ]
                self.assertEqual(bound, [], "RAY_TEMP_DIR is assigned here")


if __name__ == "__main__":
    unittest.main()
