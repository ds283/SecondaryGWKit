"""The two drivers print their refusals as one clean line (rrk-v020-adoption prompt 05).

A refusal is a check that stops a driver before any work starts: an existing store, a path that is
not a directory, a store a registered run already names. It used to end the process with a Python
traceback, under which the one line that matters sat last. Each driver
(`docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py`) now defines
`Refusal(RuntimeError)`, raises it at exactly the refusal sites, and prints it from its `__main__`
block as one `!! ` line on stderr with exit status 1. Everything else keeps its traceback.

What is pinned, and how:

  1. **The class.** `Refusal` is a subclass of `RuntimeError` in each driver, so every caller and
     test that catches `RuntimeError` still does.
  2. **The sites**, with `ast`: the `raise Refusal(` sites lie in exactly the functions named here,
     with their counts, and the `raise RuntimeError(` sites that are failures and invariants stay.
     Functions are named by their dotted path, so a nested model filter is not confused with its
     enclosing function. At each `runregistrykit.begin` call the enclosing `try` has two handlers,
     in order: `runregistrykit.ConfigError` with a bare `raise`, then `RuntimeError` raising
     `Refusal` `from` the error. The order matters: `ConfigError` is a `RuntimeError` at
     `runregistrykit` v0.2.0 and is not a refusal.
  3. **The clean print**, end to end: one subprocess per driver, on a refusal reached before any
     registration. Exit 1, stderr's first non-warning line begins `!! `, no `Traceback`, and no Ray
     started.
  4. **`begin`'s refusal becomes a `Refusal`.** At `register`, against a real `begin` and a live
     run on the same store, in a temporary runs root. At `run_build` and `sweep`, which have no
     runs-root seam, with `runregistrykit.begin` patched.
  5. **A failure keeps its type.** An `OSError`, and a `ConfigError`, from `begin` are not made
     into a `Refusal`.

No Ray is started, nothing under `var/` is read or written, and the only runs root is a temporary
one.
"""

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import runregistrykit

from RunDrivers.tests.registry_fixtures import REPO_ROOT
from RunDrivers.tests.test_hard_killed_predecessor import (
    HardKilledPredecessorTestCase,
)
from RunDrivers.tests.test_quadsource_atol_sweep_prepare import _load_script

LAUNCHER = REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py"
SCOPED = REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py"

# the functions each driver's `raise Refusal(` and `raise RuntimeError(` sites lie in, by dotted
# path, with how many in each. A new refusal, or a failure turned into one, fails test 2 until it
# is changed with it.
EXPECTED = {
    LAUNCHER: {
        "Refusal": {
            "run_build": 3,  # the existing store, the not-a-directory, and `begin`'s
            "assert_store_is_self_consistent": 1,
            "prepare": 2,  # `--force`, and `copy_store`'s
            "sweep": 1,  # `begin`'s
        },
        "RuntimeError": {
            "make_case_selector.select": 1,
            "substitute_main_py": 1,
            "run_child._filtered_build": 1,
            "run_build._filtered_build": 1,
            "sweep": 1,  # a child's exit
        },
    },
    SCOPED: {
        "Refusal": {
            "main": 2,  # the existing datastore, the not-a-directory
            "register": 2,  # the missing package, and `begin`'s
        },
        "RuntimeError": {
            "main._filtered_build_model_list": 1,
            "main": 1,  # the main.py literal count
        },
    },
}

BEGIN_REFUSAL = (
    "Cannot begin run x-20261019T000000: a store a run that has not finished names. "
    "No run directory was created"
)


def _tree(path):
    return ast.parse(Path(path).read_text(), filename=str(path))


def _raise_sites(tree):
    """`Counter` of `(name raised, dotted path of the innermost enclosing def)` for every
    `raise Name(...)`."""
    counts = Counter()

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, stack + [child.name])
                continue
            if (
                isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and isinstance(child.exc.func, ast.Name)
            ):
                counts[(child.exc.func.id, ".".join(stack))] += 1
            walk(child, stack)

    walk(tree, [])
    return counts


def _is_begin_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "begin"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runregistrykit"
    )


def _begin_trys(tree):
    """For each `runregistrykit.begin(` call, the `ast.Try` whose body holds it (or None)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            if any(_is_begin_call(n) for stmt in node.body for n in ast.walk(stmt)):
                found.append(node)
    calls = sum(_is_begin_call(n) for n in ast.walk(tree))
    return calls, found


class TestTheClass(unittest.TestCase):
    def test_refusal_is_a_runtime_error_in_each_driver(self):
        for path in (LAUNCHER, SCOPED):
            with self.subTest(driver=path.name):
                tree = _tree(path)
                classes = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "Refusal"
                ]
                self.assertEqual(len(classes), 1)
                self.assertEqual(
                    [ast.unparse(base) for base in classes[0].bases], ["RuntimeError"]
                )

    def test_the_two_classes_are_distinct(self):
        launcher = _load_script()
        import importlib.util

        spec = importlib.util.spec_from_file_location("_scoped_under_test", str(SCOPED))
        scoped = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoped)
        self.assertTrue(issubclass(launcher.Refusal, RuntimeError))
        self.assertTrue(issubclass(scoped.Refusal, RuntimeError))
        self.assertIsNot(launcher.Refusal, scoped.Refusal)


class TestTheSites(unittest.TestCase):
    def test_the_raise_sites_are_the_expected_ones(self):
        for path, expected in EXPECTED.items():
            with self.subTest(driver=path.name):
                counts = _raise_sites(_tree(path))
                for name in ("Refusal", "RuntimeError"):
                    got = {
                        where: n for (what, where), n in counts.items() if what == name
                    }
                    self.assertEqual(got, expected[name], f"raise {name}( sites")

    def test_each_begin_call_has_the_two_handlers_in_order(self):
        wanted = {LAUNCHER: 2, SCOPED: 1}
        for path, calls_wanted in wanted.items():
            with self.subTest(driver=path.name):
                calls, trys = _begin_trys(_tree(path))
                self.assertEqual(calls, calls_wanted)
                self.assertEqual(len(trys), calls_wanted)
                for node in trys:
                    self.assertEqual(node.orelse, [])
                    self.assertEqual(node.finalbody, [])
                    self.assertEqual(
                        [
                            ast.unparse(h.type) if h.type is not None else None
                            for h in node.handlers
                        ],
                        ["runregistrykit.ConfigError", "RuntimeError"],
                    )
                    config, runtime = node.handlers
                    self.assertEqual(len(config.body), 1)
                    self.assertIsInstance(config.body[0], ast.Raise)
                    self.assertIsNone(config.body[0].exc, "a bare raise")
                    self.assertEqual(len(runtime.body), 1)
                    raised = runtime.body[0]
                    self.assertIsInstance(raised, ast.Raise)
                    self.assertEqual(
                        ast.unparse(raised.exc), f"Refusal(str({runtime.name}))"
                    )
                    self.assertIsNotNone(raised.cause, "raised from the error")
                    self.assertEqual(ast.unparse(raised.cause), runtime.name)


def _run_driver(script, arguments):
    """The driver as a subprocess, from the repository root, with `sys.executable`."""
    return subprocess.run(
        [sys.executable, str(script)] + arguments,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


class TestTheCleanPrint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.database = os.path.join(self._tmp.name, "existing.sqlite")
        Path(self.database).write_bytes(b"")

    def check(self, completed, text):
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertNotIn("bootstrapping", completed.stdout)
        lines = completed.stderr.splitlines()
        # the strict form: no import-time warning is seen on this machine, so the first line of
        # stderr is the refusal's, and it is the only line of stderr
        self.assertEqual(len(lines), 1, completed.stderr)
        self.assertTrue(lines[0].startswith("!! "), completed.stderr)
        self.assertIn(text, lines[0])

    def test_the_launcher_prints_its_refusal_cleanly(self):
        completed = _run_driver(
            LAUNCHER,
            ["--build", "--database", self.database]
            + ["--atol", "1e-32", "--rtol", "1e-7"],
        )
        self.check(completed, "Refusing to build into an existing store")

    def test_the_scoped_driver_prints_its_refusal_cleanly(self):
        completed = _run_driver(
            SCOPED,
            ["--k-min", "1e5", "--k-max", "1e6", "--k-count", "2"]
            + ["--", "--database", self.database],
        )
        self.check(completed, "already exists; refusing to reuse")


def _captured_args(argv, which):
    """A fresh copy of the launcher, and the `Namespace` its own `main()` hands to the mode
    function ``which`` (`run_build` or `sweep`). Returns (module, the real function, namespace).
    """
    module = _load_script()
    real = getattr(module, which)
    seen = []
    for name in ("run_build", "run_child", "sweep", "prepare", "report"):
        setattr(module, name, lambda *a, _n=name, **k: seen.append((_n, a)))
    with mock.patch.object(sys, "argv", ["quadsource_atol_sweep.py"] + argv):
        module.main()
    assert len(seen) == 1 and seen[0][0] == which, seen
    return module, real, seen[0][1][0]


class TestBeginsRefusalAtRegister(HardKilledPredecessorTestCase):
    def test_a_live_run_on_the_store_is_a_refusal_with_begins_message(self):
        child, path, _ = self.a_child_run()
        listing = sorted(os.listdir(self.root))

        outcome, taken = self.register()

        self.assertIsInstance(outcome, self.driver.Refusal, repr(outcome))
        message = str(outcome)
        self.assertIn(os.path.basename(path), message)
        self.assertTrue(message.endswith("No run directory was created"), message)
        self.assertIsInstance(outcome.__cause__, RuntimeError)
        self.assertNotIsInstance(outcome.__cause__, self.driver.Refusal)
        self.assertEqual(str(outcome.__cause__), message)
        self.assertEqual(sorted(os.listdir(self.root)), listing)
        self.assertEqual(taken, {"stdout": False, "stderr": False, "handlers": False})


class LauncherCase(unittest.TestCase):
    """The launcher's fixtures, and no test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.database = os.path.join(self._tmp.name, "absent.sqlite")  # never created

    def assertNothingStarted(self):
        import ray

        self.assertFalse(ray.is_initialized())

    def build_args(self):
        return _captured_args(
            [
                "--build",
                "--database",
                self.database,
                "--atol",
                "1e-32",
                "--rtol",
                "1e-7",
                "--register",
                "x",
                "--purpose",
                "a build, refused in a test",
            ],
            "run_build",
        )

    def sweep_args(self):
        return _captured_args(
            ["--register", "x", "--purpose", "a sweep, refused in a test"], "sweep"
        )


class TestBeginsRefusalAtTheLauncher(LauncherCase):
    def test_run_build_makes_begins_refusal_a_refusal(self):
        module, run_build, args = self.build_args()
        error = RuntimeError(BEGIN_REFUSAL)
        with mock.patch.object(
            runregistrykit, "begin", side_effect=error
        ), mock.patch.object(module.subprocess, "Popen") as popen:
            with self.assertRaises(module.Refusal) as caught:
                run_build(args)
        self.assertEqual(str(caught.exception), BEGIN_REFUSAL)
        self.assertIs(caught.exception.__cause__, error)
        popen.assert_not_called()
        self.assertNothingStarted()
        self.assertFalse(os.path.exists(self.database))

    def test_sweep_makes_begins_refusal_a_refusal(self):
        module, sweep, args = self.sweep_args()
        error = RuntimeError(BEGIN_REFUSAL)
        with mock.patch.object(
            runregistrykit, "begin", side_effect=error
        ), mock.patch.object(module.subprocess, "Popen") as popen:
            with self.assertRaises(module.Refusal) as caught:
                sweep(args)
        self.assertEqual(str(caught.exception), BEGIN_REFUSAL)
        self.assertIs(caught.exception.__cause__, error)
        popen.assert_not_called()
        self.assertNothingStarted()


class TestAFailureKeepsItsType(LauncherCase):
    def test_an_oserror_from_begin_is_not_a_refusal(self):
        module, run_build, args = self.build_args()
        with mock.patch.object(runregistrykit, "begin", side_effect=OSError("disk")):
            with self.assertRaises(OSError) as caught:
                run_build(args)
        self.assertNotIsInstance(caught.exception, module.Refusal)
        self.assertNothingStarted()

    def test_a_config_error_from_begin_is_not_a_refusal(self):
        module, run_build, args = self.build_args()
        error = runregistrykit.ConfigError("no runregistrykit.toml was found")
        self.assertIsInstance(error, RuntimeError)
        with mock.patch.object(runregistrykit, "begin", side_effect=error):
            with self.assertRaises(runregistrykit.ConfigError) as caught:
                run_build(args)
        self.assertIs(caught.exception, error)
        self.assertNotIsInstance(caught.exception, module.Refusal)
        self.assertNothingStarted()

    def test_an_oserror_and_a_config_error_at_sweep_are_not_refusals(self):
        for error in (OSError("disk"), runregistrykit.ConfigError("no config")):
            with self.subTest(error=type(error).__name__):
                module, sweep, args = self.sweep_args()
                with mock.patch.object(runregistrykit, "begin", side_effect=error):
                    with self.assertRaises(type(error)) as caught:
                        sweep(args)
                self.assertIs(caught.exception, error)
                self.assertNotIsInstance(caught.exception, module.Refusal)


if __name__ == "__main__":
    unittest.main()
