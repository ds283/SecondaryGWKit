"""
The six readers open read-only, take ``--models``, and skip a model with no background
(prompts/a3-v2-readiness, prompt 03, O3–O4).

The scripts call ``ray.init`` at module scope and cannot be imported, so they are read with
``ast``, as ``test_main_plumbing`` reads ``main.py``:

- each constructs ``ShardedPool`` with ``read_only=True`` and with neither ``prune_unvalidated``
  nor ``drop_actions``;
- each adds ``--models`` through ``extract_common.add_models_argument``, filters
  ``build_model_list``'s result through ``extract_common.select_models(..., args.models)``, runs
  its models through ``extract_common.read_models(model_list, run_pipeline)``, and in
  ``run_pipeline`` calls ``extract_common.require_background(model, model_label)`` where it used
  to raise ``RuntimeError("Could not locate suitable background model…")``, which no script
  raises any more.

The helpers are pure and are run here: an unknown ``--models`` label is an error naming the known
ones; the filter keeps ``build_model_list``'s order; a model with no background is skipped with its
line; a script that reads no model exits non-zero, naming what it skipped and why. No Ray, no
datastore.
"""

import argparse
import ast
import contextlib
import io
import unittest
from pathlib import Path

import ray

import extract_common
from extract_common import (
    NO_BACKGROUND_REASON,
    NoBackgroundModel,
    add_models_argument,
    read_models,
    require_background,
    select_models,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

READERS = (
    "extract_Gk_data.py",
    "extract_GkWKB_data.py",
    "extract_GkSource_data.py",
    "extract_TkWKB_data.py",
    "extract_tensor_source_data.py",
    "extract_QuadSourceIntegral_data.py",
)

# the text of every message a test below provokes, by case (for the prompt's log)
MESSAGES = {}

MODEL_LIST = [
    {"label": "LambdaCDM", "cosmology": "lcdm"},
    {"label": "QCD_Cosmology", "cosmology": "qcd"},
]


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_reader_read_only")


def _name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _name(node.func) == name
    ]


def _function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level function {name}")


def _pool_with(tree) -> ast.With:
    """The module-level ``with ShardedPool(...) as pool:`` block."""
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _name(item.context_expr.func) == "ShardedPool"
            for item in node.items
        )
    ]
    if len(found) != 1:
        raise AssertionError(
            f"{len(found)} module-level `with ShardedPool(...)` blocks"
        )
    return found[0]


class TestTheSixReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = {
            name: ast.parse((REPO_ROOT / name).read_text(), filename=name)
            for name in READERS
        }
        cls.sources = {name: (REPO_ROOT / name).read_text() for name in READERS}

    def test_each_opens_its_pool_read_only(self):
        for name, tree in self.trees.items():
            with self.subTest(reader=name):
                calls = _calls(tree, "ShardedPool")
                self.assertEqual(len(calls), 1)
                keywords = {kw.arg: kw.value for kw in calls[0].keywords}
                self.assertIn("read_only", keywords)
                self.assertIsInstance(keywords["read_only"], ast.Constant)
                self.assertIs(keywords["read_only"].value, True)
                self.assertNotIn("prune_unvalidated", keywords)
                self.assertNotIn("drop_tables", keywords)
                # and it is the pool the with block opens
                self.assertIs(_pool_with(tree).items[0].context_expr, calls[0])

    def test_each_adds_models_through_extract_common(self):
        for name, tree in self.trees.items():
            with self.subTest(reader=name):
                calls = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and _name(node.value.func) == "add_models_argument"
                ]
                self.assertEqual(len(calls), 1)
                self.assertEqual([_name(a) for a in calls[0].value.args], ["parser"])
                # before the arguments are parsed
                parse = [
                    i
                    for i, node in enumerate(tree.body)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and _name(node.value.func) == "parse_args"
                ]
                self.assertEqual(len(parse), 1)
                self.assertLess(tree.body.index(calls[0]), parse[0])

    def test_each_filters_the_model_list_and_reads_it_through_read_models(self):
        for name, tree in self.trees.items():
            with self.subTest(reader=name):
                block = _pool_with(tree)
                assigns = [
                    node
                    for node in block.body
                    if isinstance(node, ast.Assign)
                    and [_name(t) for t in node.targets] == ["model_list"]
                ]
                self.assertEqual(len(assigns), 1)
                value = assigns[0].value
                self.assertEqual(_name(value.func), "select_models")
                self.assertEqual(_name(value.args[0].func), "build_model_list")
                self.assertEqual(ast.unparse(value.args[1]), "args.models")

                loops = [
                    node
                    for node in block.body
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and _name(node.value.func) == "read_models"
                ]
                self.assertEqual(len(loops), 1)
                self.assertEqual(
                    [_name(a) for a in loops[0].value.args],
                    ["model_list", "run_pipeline"],
                )
                # no other loop over the models is left
                self.assertFalse(
                    any(
                        isinstance(node, ast.For) and _name(node.iter) == "model_list"
                        for node in ast.walk(block)
                    )
                )

    def test_each_calls_the_skip_helper_after_its_background_lookup(self):
        for name, tree in self.trees.items():
            with self.subTest(reader=name):
                run_pipeline = _function(tree, "run_pipeline")
                calls = _calls(run_pipeline, "require_background")
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    [_name(a) for a in calls[0].args], ["model", "model_label"]
                )
                self.assertNotIn(
                    "Could not locate suitable background model", self.sources[name]
                )

    def test_each_imports_the_helpers_from_extract_common(self):
        wanted = {
            "add_models_argument",
            "select_models",
            "read_models",
            "require_background",
        }
        for name, tree in self.trees.items():
            with self.subTest(reader=name):
                imported = {
                    alias.name
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "extract_common"
                    for alias in node.names
                }
                self.assertTrue(wanted <= imported, wanted - imported)


class TestModelsArgument(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_models_argument(parser)
        return parser

    def test_absent_is_none(self):
        self.assertIsNone(self.parser().parse_args([]).models)

    def test_one_or_more_labels(self):
        self.assertEqual(
            self.parser().parse_args(["--models", "LambdaCDM"]).models, ["LambdaCDM"]
        )
        self.assertEqual(
            self.parser().parse_args(["--models", "LambdaCDM", "QCD_Cosmology"]).models,
            ["LambdaCDM", "QCD_Cosmology"],
        )


class TestSelectModels(unittest.TestCase):
    def test_none_keeps_every_model(self):
        self.assertEqual(select_models(MODEL_LIST, None), MODEL_LIST)

    def test_the_filter_keeps_the_model_lists_order(self):
        kept = select_models(MODEL_LIST, ["QCD_Cosmology", "LambdaCDM"])
        self.assertEqual([m["label"] for m in kept], ["LambdaCDM", "QCD_Cosmology"])
        kept = select_models(MODEL_LIST, ["QCD_Cosmology"])
        self.assertEqual(kept, [MODEL_LIST[1]])

    def test_an_unknown_label_is_an_error_naming_the_known_ones(self):
        for requested in (["LambdaCMD"], ["LambdaCDM", "LambdaCMD"]):
            with self.subTest(requested=requested):
                with self.assertRaises(RuntimeError) as ctx:
                    select_models(MODEL_LIST, requested)
                text = str(ctx.exception)
                self.assertIn("LambdaCMD", text)
                self.assertIn("Models it returns: LambdaCDM, QCD_Cosmology.", text)
                MESSAGES["unknown label"] = text


class _Background:
    def __init__(self, available: bool):
        self.available = available


class TestTheSkip(unittest.TestCase):
    def pipeline(self, held):
        """A run_pipeline that looks its background up as the scripts do: available only for
        the labels in ``held``."""
        ran = []

        def run_pipeline(model_data):
            model_label = model_data["label"]
            model = _Background(model_label in held)
            require_background(model, model_label)
            ran.append(model_label)

        return run_pipeline, ran

    def test_a_model_with_no_background_is_skipped_with_its_line(self):
        run_pipeline, ran = self.pipeline({"LambdaCDM"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            read = read_models(MODEL_LIST, run_pipeline)
        self.assertEqual(read, ["LambdaCDM"])
        self.assertEqual(ran, ["LambdaCDM"])
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        self.assertEqual(
            lines,
            [
                ">> SKIPPING MODEL QCD_Cosmology: the datastore holds no background "
                "model for it"
            ],
        )

    def test_the_skip_does_not_stop_the_next_model(self):
        run_pipeline, ran = self.pipeline({"QCD_Cosmology"})
        with contextlib.redirect_stdout(io.StringIO()):
            read = read_models(MODEL_LIST, run_pipeline)
        self.assertEqual(read, ["QCD_Cosmology"])

    def test_every_model_held_reads_every_model(self):
        run_pipeline, ran = self.pipeline({"LambdaCDM", "QCD_Cosmology"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            read = read_models(MODEL_LIST, run_pipeline)
        self.assertEqual(read, ["LambdaCDM", "QCD_Cosmology"])
        self.assertEqual(out.getvalue(), "")

    def test_all_skipped_exits_non_zero_with_the_reason(self):
        run_pipeline, ran = self.pipeline(set())
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                read_models(MODEL_LIST, run_pipeline)
        code = ctx.exception.code
        # SystemExit with a message exits with status 1, printing the message
        self.assertIsInstance(code, str)
        self.assertIn("no model was read", code)
        self.assertIn(f"LambdaCDM was skipped: {NO_BACKGROUND_REASON}", code)
        self.assertIn(f"QCD_Cosmology was skipped: {NO_BACKGROUND_REASON}", code)
        MESSAGES["all skipped"] = code

    def test_an_empty_model_list_exits_non_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            read_models([], lambda m: None)
        self.assertIn("the model list is empty", ctx.exception.code)

    def test_any_other_error_is_not_a_skip(self):
        def run_pipeline(model_data):
            raise RuntimeError("something else")

        with self.assertRaises(RuntimeError):
            read_models(MODEL_LIST, run_pipeline)

    def test_the_skip_is_not_a_runtime_error(self):
        self.assertFalse(issubclass(NoBackgroundModel, RuntimeError))
        self.assertIs(extract_common.NoBackgroundModel, NoBackgroundModel)


if __name__ == "__main__":
    unittest.main()
