"""
``main.py`` and the six readers start their own Ray instance unless told otherwise
(prompts/handover, prompt 05; closes ``[00-main-and-the-readers-default-to-ray-address-auto]``).

Their ``--ray-address`` used to default to ``auto``, which attaches to whatever Ray instance is up
on the machine: ``RAY_ADDRESS``, then ``/tmp/ray/ray_current_cluster``, then any ``gcs_server`` it
finds among the running processes. The default is now ``local``, which starts a fresh instance for
this process and ignores ``RAY_ADDRESS``; attaching takes ``--ray-address auto`` or a ``host:port``.

The seven scripts call ``ray.init`` at module scope and cannot be imported, so they are read with
``ast``, as ``test_reader_read_only`` reads the readers and ``test_main_plumbing`` reads ``main.py``:

- the ``--ray-address`` argument's default is ``"local"`` -- for ``main.py`` through the
  module-level ``DEFAULT_RAY_ADDRESS`` it names -- and the argument is not required, because the
  A3 v2 launcher (``RunDrivers/quadsource_atol_sweep._main_py_argv``) passes none;
- each script has exactly one ``ray.init``, and its ``address`` is ``args.ray_address``, so the
  default is what reaches Ray.

Ray's half -- that ``local`` means a new instance even with ``RAY_ADDRESS`` set -- is read from
``ray._private.services`` and pinned here without starting Ray, so a Ray upgrade that changes it
fails in this module rather than in a build.
"""

import ast
import os
import unittest
from pathlib import Path
from unittest import mock

import ray
from ray._private import services

REPO_ROOT = Path(__file__).resolve().parents[2]

MAIN_PY = "main.py"

READERS = (
    "extract_Gk_data.py",
    "extract_GkWKB_data.py",
    "extract_GkSource_data.py",
    "extract_TkWKB_data.py",
    "extract_tensor_source_data.py",
    "extract_QuadSourceIntegral_data.py",
)

DEFAULT = "local"


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_ray_address_default")


def _tree(name):
    return ast.parse((REPO_ROOT / name).read_text(), filename=name)


def _ray_address_argument(tree) -> ast.Call:
    """The one ``parser.add_argument("--ray-address", ...)`` call."""
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "--ray-address"
            for arg in node.args
        )
    ]
    if len(found) != 1:
        raise AssertionError(f"{len(found)} --ray-address arguments")
    return found[0]


def _keywords(call: ast.Call) -> dict:
    return {kw.arg: kw.value for kw in call.keywords}


def _module_constant(tree, name):
    """The value of the module-level ``name = <constant>``; fails if it is not exactly one."""
    found = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    if len(found) != 1:
        raise AssertionError(f"{len(found)} module-level assignments to {name}")
    if not isinstance(found[0], ast.Constant):
        raise AssertionError(f"{name} is not assigned a constant")
    return found[0].value


def _ray_init_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ray"
    ]


class TestTheDefaultIsLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = {name: _tree(name) for name in (MAIN_PY,) + READERS}

    def test_main_py_constant_is_local(self):
        self.assertEqual(
            _module_constant(self.trees[MAIN_PY], "DEFAULT_RAY_ADDRESS"), DEFAULT
        )

    def test_main_py_argument_defaults_to_the_constant(self):
        default = _keywords(_ray_address_argument(self.trees[MAIN_PY]))["default"]
        self.assertIsInstance(default, ast.Name)
        self.assertEqual(default.id, "DEFAULT_RAY_ADDRESS")

    def test_each_reader_defaults_to_local(self):
        for name in READERS:
            with self.subTest(reader=name):
                default = _keywords(_ray_address_argument(self.trees[name]))["default"]
                self.assertIsInstance(default, ast.Constant)
                self.assertEqual(default.value, DEFAULT)

    def test_no_script_requires_the_flag(self):
        # the A3 v2 launcher's main.py argv carries no --ray-address
        for name, tree in self.trees.items():
            with self.subTest(script=name):
                required = _keywords(_ray_address_argument(tree)).get("required")
                if required is not None:
                    self.assertIsInstance(required, ast.Constant)
                    self.assertIs(required.value, False)

    def test_each_ray_init_takes_the_flag(self):
        for name, tree in self.trees.items():
            with self.subTest(script=name):
                calls = _ray_init_calls(tree)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].args, [])
                address = _keywords(calls[0])["address"]
                self.assertEqual(ast.unparse(address), "args.ray_address")


class TestRayResolvesLocalToANewInstance(unittest.TestCase):
    """What the default relies on, in the venv's Ray: ``local`` never attaches."""

    ELSEWHERE = "10.0.0.1:6379"

    def test_local_ignores_RAY_ADDRESS(self):
        with mock.patch.dict(os.environ, {"RAY_ADDRESS": self.ELSEWHERE}):
            self.assertIsNone(services.canonicalize_bootstrap_address(DEFAULT))

    def test_auto_follows_RAY_ADDRESS(self):
        # the contrast: the old default would have attached to the environment's cluster
        with mock.patch.dict(os.environ, {"RAY_ADDRESS": self.ELSEWHERE}):
            self.assertEqual(
                services.canonicalize_bootstrap_address("auto"), self.ELSEWHERE
            )


if __name__ == "__main__":
    unittest.main()
