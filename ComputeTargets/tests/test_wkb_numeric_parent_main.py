"""
Prompt 07 of ``prompts/datastore-integrity`` (§3 test 8): every place ``main.py`` asks "does this
WKB row exist?", or builds one, says which numeric row it means, and the numeric row is looked up
before the question is asked. Read with ``ast``, in the style of ``test_main_plumbing.py``;
``main.py`` is never imported (it parses ``sys.argv`` and opens Ray at module scope).

Also here:

  * ``Gk_WKB_numeric_parent``, the Gk stage's seeded-or-pure choice, lifted unchanged to module
    level so that the existence query can be keyed on it, exercised through
    ``test_main_plumbing.load_main_py_functions``;
  * the readers: the two extraction scripts ask for the labels-only mode explicitly, and the
    verification driver's Tk WKB lookup is keyed on the numeric row it has just looked up.

No Ray, no datastore.
"""

import ast
import math
import types
import unittest
from pathlib import Path
from typing import Optional

import ray

from ComputeTargets.tests.test_main_plumbing import (
    MAIN_PY,
    _object_get_calls,
    _own_nodes,
    load_main_py_functions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WKB_CLASSES = ("TkWKBIntegration", "GkWKBIntegration")
NUMERIC_OF = {
    "TkWKBIntegration": "TkNumericIntegration",
    "GkWKBIntegration": "GkNumericIntegration",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _functions(tree) -> dict:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dict_keys_under(node) -> set:
    keys = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for key in sub.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _batch_values(scope, batch_name) -> list:
    """The expressions assigned to `batch_name` in `scope`'s own nodes."""
    values = []
    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == batch_name:
                    values.append(node.value)
    return values


def wkb_sites(scope):
    """Each WKB object_get in `scope`: (class, kind, keys, line), where kind is "query" for a
    RayWorkPool dispatch over a batch (an existence query or a stage lookup) and "construction" for
    a direct call, and keys are the payload keys it carries."""
    own = _own_nodes(scope)
    sites = []
    dispatched = set()
    for node in own:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RayWorkPool"
        ):
            continue
        builders = [kw.value for kw in node.keywords if kw.arg == "task_builder"]
        for call in _object_get_calls(
            [sub for builder in builders for sub in ast.walk(builder)]
        ):
            dispatched.add(id(call))
            cls = call.args[0].value
            if cls not in WKB_CLASSES:
                continue
            batch = node.args[1].id
            keys = set()
            for value in _batch_values(scope, batch):
                keys |= _dict_keys_under(value)
            sites.append((cls, "query", keys, call.lineno, node.lineno))
    for call in _object_get_calls(own):
        if id(call) in dispatched:
            continue
        cls = call.args[0].value
        if cls not in WKB_CLASSES:
            continue
        keys = {kw.arg: kw.value for kw in call.keywords}
        sites.append((cls, "construction", keys, call.lineno, call.lineno))
    return sites


def numeric_lookup_lines(scope, cls) -> list:
    """The RayWorkPool lines of `scope` dispatching a lookup of `cls`."""
    lines = []
    for node in _own_nodes(scope):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RayWorkPool"
        ):
            continue
        builders = [kw.value for kw in node.keywords if kw.arg == "task_builder"]
        for call in _object_get_calls(
            [sub for builder in builders for sub in ast.walk(builder)]
        ):
            if call.args[0].value == cls:
                lines.append(node.lineno)
    return lines


class TestMainPyKeysEveryWKBSite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = _functions(_tree(MAIN_PY))

    def sites(self, function):
        return wkb_sites(self.functions[function])

    def test_the_sites_are_where_they_were(self):
        """Three per class, as test_main_plumbing.EXPECTED_SITES pins them: a finder that found
        nothing would pass everything below."""
        found = {}
        for function in (
            "build_Tk_WKB_work",
            "build_Gk_WKB_work",
            "build_QuadSourceIntegral_batch",
        ):
            for cls, kind, _, _, _ in self.sites(function):
                found[(function, cls, kind)] = found.get((function, cls, kind), 0) + 1
        self.assertEqual(
            found,
            {
                ("build_Tk_WKB_work", "TkWKBIntegration", "query"): 1,
                ("build_Tk_WKB_work", "TkWKBIntegration", "construction"): 1,
                ("build_Gk_WKB_work", "GkWKBIntegration", "query"): 1,
                ("build_Gk_WKB_work", "GkWKBIntegration", "construction"): 2,
                ("build_QuadSourceIntegral_batch", "TkWKBIntegration", "query"): 1,
            },
        )

    def test_every_query_passes_the_parent_and_no_z_init(self):
        for function in (
            "build_Tk_WKB_work",
            "build_Gk_WKB_work",
            "build_QuadSourceIntegral_batch",
        ):
            for cls, kind, keys, line, _ in self.sites(function):
                if kind != "query":
                    continue
                with self.subTest(function=function, line=line):
                    self.assertIn("numeric_serial", keys)
                    self.assertNotIn("z_init", keys)
                    self.assertNotIn("labels_only", keys)

    def test_every_construction_passes_the_parent(self):
        for function in ("build_Tk_WKB_work", "build_Gk_WKB_work"):
            for cls, kind, keys, line, _ in self.sites(function):
                if kind != "construction":
                    continue
                with self.subTest(function=function, line=line):
                    self.assertIn("numeric_serial", keys)
                    self.assertNotIn("labels_only", keys)

    def test_the_seeded_constructions_name_the_numeric_row_and_the_pure_one_the_marker(
        self,
    ):
        for function in ("build_Tk_WKB_work", "build_Gk_WKB_work"):
            parents = []
            for cls, kind, keys, line, _ in self.sites(function):
                if kind != "construction":
                    continue
                parents.append(ast.unparse(keys["numeric_serial"]))
            with self.subTest(function=function):
                if function == "build_Tk_WKB_work":
                    self.assertEqual(parents, ["Tk.store_id"])
                else:
                    self.assertEqual(
                        parents, ["Gk.store_id", "GkWKBIntegration.PURE_WKB"]
                    )

    def test_the_numeric_row_is_looked_up_before_the_wkb_query(self):
        for function in (
            "build_Tk_WKB_work",
            "build_Gk_WKB_work",
            "build_QuadSourceIntegral_batch",
        ):
            scope = self.functions[function]
            for cls, kind, _, line, pool_line in self.sites(function):
                if kind != "query":
                    continue
                with self.subTest(function=function, cls=cls):
                    numeric = numeric_lookup_lines(scope, NUMERIC_OF[cls])
                    self.assertEqual(len(numeric), 1)
                    self.assertLess(numeric[0], pool_line)

    def test_the_gk_stage_decides_with_the_module_level_choice(self):
        calls = [
            node
            for node in ast.walk(self.functions["build_Gk_WKB_work"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Gk_WKB_numeric_parent"
        ]
        self.assertEqual(len(calls), 1)
        # ... and no longer compares against the limit itself
        source = ast.unparse(self.functions["build_Gk_WKB_work"])
        self.assertNotIn("z_source_limit", source)

    def test_the_gk_numeric_lookup_is_serial_only(self):
        """Only the serial and the stop_* columns of the numeric row are read in the Gk stage, so
        its lookup does not populate the values (it runs over the whole batch since prompt 07).
        """
        scope = self.functions["build_Gk_WKB_work"]
        keys = set()
        for value in _batch_values(scope, "payload_batch"):
            keys |= _dict_keys_under(value)
        self.assertIn("_do_not_populate", keys)

    def test_main_py_never_asks_for_the_reader_mode(self):
        for node in ast.walk(_tree(MAIN_PY)):
            if isinstance(node, ast.keyword):
                self.assertNotEqual(node.arg, "labels_only")
            if isinstance(node, ast.Constant):
                self.assertNotEqual(node.value, "labels_only")


def _k_exit(e3, e4):
    return types.SimpleNamespace(z_exit_subh_e3=e3, z_exit_subh_e4=e4)


def _z(value):
    return types.SimpleNamespace(z=value)


def _numeric(available):
    return types.SimpleNamespace(available=available)


class TestTheSeededOrPureChoice(unittest.TestCase):
    """`Gk_WKB_numeric_parent`, main.py's own code: seeded when z_source is at or above the
    geometric mean of the 3- and 4-e-fold points *and* a numeric row exists; pure WKB otherwise,
    including when the numeric row exists but z_source is below the limit."""

    @classmethod
    def setUpClass(cls):
        # a staticmethod, so that the class attribute is not bound as a method
        cls.choose = staticmethod(
            load_main_py_functions(
                ["Gk_WKB_numeric_parent"],
                extra_globals={
                    "sqrt": math.sqrt,
                    "Optional": Optional,
                    "GkNumericIntegration": object,
                },
            )["Gk_WKB_numeric_parent"]
        )

    def test_above_the_limit_with_a_numeric_row_is_seeded(self):
        Gk = _numeric(True)
        self.assertIs(self.choose(_k_exit(1.0e4, 1.0e2), _z(2.0e3), Gk), Gk)

    def test_at_the_limit_is_seeded(self):
        Gk = _numeric(True)
        limit = math.sqrt(1.0e4 * 1.0e2)
        self.assertIs(self.choose(_k_exit(1.0e4, 1.0e2), _z(limit), Gk), Gk)

    def test_below_the_limit_is_pure_even_with_a_numeric_row(self):
        limit = math.sqrt(1.0e4 * 1.0e2)
        self.assertIsNone(
            self.choose(
                _k_exit(1.0e4, 1.0e2), _z(math.nextafter(limit, 0.0)), _numeric(True)
            )
        )

    def test_without_a_numeric_row_is_pure(self):
        self.assertIsNone(
            self.choose(_k_exit(1.0e4, 1.0e2), _z(5.0e3), _numeric(False))
        )


class TestTheReaders(unittest.TestCase):
    """§2 K1 readers: the extraction scripts select by physical labels in the explicit reader
    mode; the verification driver keys on the numeric row it holds."""

    def _wkb_calls(self, path):
        return [
            call
            for call in _object_get_calls(list(ast.walk(_tree(path))))
            if call.args[0].value in WKB_CLASSES
        ]

    def test_the_extraction_scripts_ask_for_labels_only(self):
        for script, cls in (
            ("extract_TkWKB_data.py", "TkWKBIntegration"),
            ("extract_GkWKB_data.py", "GkWKBIntegration"),
        ):
            with self.subTest(script=script):
                calls = self._wkb_calls(REPO_ROOT / script)
                self.assertEqual([c.args[0].value for c in calls], [cls])
                keywords = {kw.arg: kw.value for kw in calls[0].keywords}
                self.assertIn("labels_only", keywords)
                self.assertIs(keywords["labels_only"].value, True)

    def test_the_verification_driver_keys_on_its_numeric_row(self):
        calls = self._wkb_calls(
            REPO_ROOT
            / "docs"
            / "source-remediation-verification"
            / "run_quadsource_integrals.py"
        )
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertEqual(ast.unparse(keywords["numeric_serial"]), "Tk_numeric.store_id")
        self.assertNotIn("z_init", keywords)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
