"""
Every Ray instance this repository starts runs with Ray's token authentication named explicitly
(prompts/py313-venv, after the review of prompt 01; closes
``[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]``).

Since Ray 2.59 a new local instance runs in token mode when ``RAY_AUTH_MODE`` is unset, and logs a
warning saying so. ``config/ray_local.py`` sets ``RAY_AUTH_MODE=token`` at import unless the
environment already holds a value, so the mode is this repository's choice and not Ray's default,
the warning goes, and an attach to an explicit ``host:port`` sends the token.

That holds only if every Ray start imports the module before it calls ``ray.init``. The ten starts
are ``test_ray_temp_dir``'s; each file is read with ``ast``. The module itself is imported in a
fresh interpreter, so the test's own environment is not changed. No test here starts Ray.
"""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from RunDrivers.tests.test_ray_temp_dir import SCRIPTS, WRAPPERS

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = "import os, config.ray_local; print(os.environ.get('RAY_AUTH_MODE'))"


def _mode_after_import(environment):
    """``RAY_AUTH_MODE`` as a fresh interpreter sees it after importing ``config.ray_local``."""
    env = {k: v for k, v in os.environ.items() if k != "RAY_AUTH_MODE"}
    env.update(environment)
    env["PYTHONPATH"] = str(REPO_ROOT)
    done = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


class TestTheModuleSetsTheMode(unittest.TestCase):
    def test_an_unset_mode_becomes_token(self):
        """With nothing set, Ray would choose token mode itself and warn in every log."""
        self.assertEqual(_mode_after_import({}), "token")

    def test_a_mode_already_set_is_left_as_it_is(self):
        """A person's explicit choice in the environment wins over this repository's."""
        self.assertEqual(_mode_after_import({"RAY_AUTH_MODE": "disabled"}), "disabled")


class TestEveryRayStartImportsIt(unittest.TestCase):
    def test_each_file_that_starts_ray_imports_config_ray_local(self):
        """A start whose file does not import the module would run in Ray's default mode."""
        for name in sorted({name for name, _ in SCRIPTS + WRAPPERS}):
            with self.subTest(file=name):
                tree = ast.parse((REPO_ROOT / name).read_text(), filename=name)
                modules = {
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                }
                self.assertIn("config.ray_local", modules)

    def test_the_wrappers_import_it_before_they_call_ray_init(self):
        """In the drivers the import is inside the function that wraps ``ray.init``."""
        for name, qualified in WRAPPERS:
            outer = qualified.split(".")[0]
            with self.subTest(start=f"{name}:{qualified}"):
                tree = ast.parse((REPO_ROOT / name).read_text(), filename=name)
                function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == outer
                )
                imports = [
                    node.lineno
                    for node in ast.walk(function)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "config.ray_local"
                ]
                calls = [
                    node.lineno
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_real_ray_init"
                ]
                self.assertTrue(imports and calls)
                self.assertLess(min(imports), min(calls))


if __name__ == "__main__":
    unittest.main()
