"""`main.py`'s argv from `docs/handover/quadsource_atol_sweep.py` never disables pruning, for
`a3-v2-readiness` prompt 05 (**P2**).

Pruning unvalidated rows at startup is the designed recovery for a row stored but not validated
when a build was interrupted. Both of the sweep script's `main.py` invocations once passed
`--no-prune-unvalidated`, so an interrupted A3 v2 build resumed with `--build --resume` stored a
second row beside every such row (`[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`;
`docs/datastore-integrity-verification.md` §2.1.4). The two argument lists are now built by
`build_main_argv` and `child_main_argv`, which need no Ray; this module keeps the flag out.

What is tested, and how:

* **The lists, element by element** (`TestTheListsAreWhatTheyWere`). Each is compared with the
  list the script built before prompt 05, less the flag, so that moving the lists into functions
  moved nothing else. The `Namespace` each is called with comes from the script's own parser
  (`main()` with `run_build` / `run_child` replaced by a function that keeps its argument), so the
  defaults are the real ones.
* **Nothing in a list disables pruning** (`TestNothingDisablesPruning`). Not the token, and not
  an abbreviation: `main.py`'s parser is `argparse.ArgumentParser()` with `allow_abbrev` at its
  default, so `--no-prune` resolves to `--no-prune-unvalidated`. Each list is therefore also
  parsed by an `argparse` parser rebuilt from `main.py`'s own `add_argument` calls, read with
  `ast` (`main.py` cannot be imported, `CLAUDE.md`), and the result must have
  `prune_unvalidated` true.
* **`main.py`'s default is pruning** (`TestMainPyDefault`), read from its source, so that the
  lists' silence means pruning.
* **The drivers use the functions** (`TestTheDriversCallThem`), read with `ast`, so that a list
  written back into `run_build` or `run_child` is seen.

Nothing here needs Ray, a datastore or anything under `var/`.
"""

import argparse
import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py"
MAIN_PY = REPO_ROOT / "main.py"
USAGE_DOCSTRINGS = (
    REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py",
    REPO_ROOT / "docs" / "source-remediation-verification" / "scoped_pipeline_run.py",
)

FLAG = "--no-prune-unvalidated"


def _load_script():
    """A fresh import of the sweep script, isolated from every other test's monkeypatching."""
    spec = importlib.util.spec_from_file_location(
        "_quadsource_atol_sweep_prune_under_test", str(SCRIPT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed_arguments(command_line):
    """The `Namespace` the script's own `main()` hands to `run_build` or `run_child`.

    `run_build` and `run_child` are replaced, on a fresh copy of the module, by functions that
    keep their argument; `main()` then runs its real parser on `command_line`, its defaults
    included. Returns `(module, which, namespace)`.
    """
    module = _load_script()
    seen = []
    module.run_build = lambda args: seen.append(("build", args))
    module.run_child = lambda args: seen.append(("child", args))
    with mock.patch.object(sys, "argv", ["quadsource_atol_sweep.py"] + command_line):
        module.main()
    assert len(seen) == 1, seen
    which, namespace = seen[0]
    return module, which, namespace


def _main_py_add_argument_calls():
    """Every `parser.add_argument(...)` call of `main.py`, as `(option strings, keywords)`.

    `main.py` is read, never imported (it parses `sys.argv` and opens Ray at module scope).
    A keyword whose value is not a literal (`type=str`, `default=DEFAULT_ZEND`,
    `choices=allowed_drop_actions`) is left out of `keywords`: the parser rebuilt from these needs
    the option strings and their actions, not the values' types. `argparse.BooleanOptionalAction`
    is kept, by name, because it is what adds the `--no-` spelling.
    """
    tree = ast.parse(MAIN_PY.read_text())
    calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "parser"
        ):
            continue
        strings = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        keywords = {}
        for kw in node.keywords:
            if (
                kw.arg == "action"
                and isinstance(kw.value, ast.Attribute)
                and kw.value.attr == "BooleanOptionalAction"
            ):
                keywords["action"] = argparse.BooleanOptionalAction
                continue
            try:
                keywords[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                pass
        calls.append((strings, keywords))
    return calls


def main_py_parser():
    """An `ArgumentParser` with `main.py`'s options, built from its source.

    It is `argparse.ArgumentParser()` exactly as `main.py`'s is, so `allow_abbrev` is at its
    default and an abbreviation resolves as it does there.
    """
    parser = argparse.ArgumentParser()
    for strings, keywords in _main_py_add_argument_calls():
        keywords = {k: v for k, v in keywords.items() if k != "help"}
        parser.add_argument(*strings, **keywords)
    return parser


def prune_default_call():
    """The one `add_argument` call for `--prune-unvalidated`, as `(strings, keywords)`."""
    calls = [
        call
        for call in _main_py_add_argument_calls()
        if "--prune-unvalidated" in call[0]
    ]
    assert len(calls) == 1, calls
    return calls[0]


# the command lines of README-style launches: one per path, each through the script's own parser
BUILD = ["--build", "--database", "{db}", "--atol", "1e-32", "--rtol", "1e-7"]
CHILD = ["--child", "--atol", "1e-32", "--rtol", "1e-7"]


def build_namespace(resume, database):
    command = [token.format(db=database) for token in BUILD]
    if resume:
        command.append("--resume")
    module, which, namespace = parsed_arguments(command)
    assert which == "build" and namespace.resume is resume
    return module, namespace


def child_namespace():
    module, which, namespace = parsed_arguments(CHILD)
    assert which == "child"
    return module, namespace


class ArgvTestCase(unittest.TestCase):
    """The three argument lists, each from the script's own parsed `Namespace`."""

    # `Path.resolve()` turns macOS's `/tmp` into `/private/tmp`, and `run_build` resolves the path
    # it is given, so the database passed here is already a resolved path
    DATABASE = "/private/tmp/prune-on-resume/store.sqlite"

    def setUp(self):
        self.lists = {}
        module, namespace = build_namespace(False, self.DATABASE)
        self.module = module
        self.lists["build"] = module.build_main_argv(namespace, Path(self.DATABASE))
        module, namespace = build_namespace(True, self.DATABASE)
        self.lists["build --resume"] = module.build_main_argv(
            namespace, Path(self.DATABASE)
        )
        module, namespace = child_namespace()
        self.lists["child"] = module.child_main_argv(namespace, module.SWEEP_STORE)
        self.sweep_store = str(module.SWEEP_STORE)


class TestTheListsAreWhatTheyWere(ArgvTestCase):
    """E4: each list, less the flag, is the list the script built before prompt 05."""

    def test_the_build_list_is_what_run_build_built_less_the_flag(self):
        expected = [
            "main.py",
            "--database",
            self.DATABASE,
            "--job-name",
            "handover-A3-baseline-v2",
            "--shards",
            "4",
            "--zend",
            "0.1",
            "--source-samples-log10z",
            "100",
        ]
        self.assertEqual(self.lists["build"], expected)

    def test_the_resumed_build_list_is_the_same_list(self):
        # `--resume` decides run_build's pre-check and nothing in main.py's argument list
        self.assertEqual(self.lists["build --resume"], self.lists["build"])

    def test_the_child_list_is_what_run_child_built_less_the_flag(self):
        expected = [
            "main.py",
            "--database",
            self.sweep_store,
            "--job-name",
            "handover-atol-sweep-a1e-32-r1e-07",
            "--shards",
            "4",
            "--zend",
            "0.1",
            "--source-samples-log10z",
            "100",
        ]
        self.assertEqual(self.lists["child"], expected)

    def test_every_element_is_a_string(self):
        for name, argv in self.lists.items():
            with self.subTest(list=name):
                self.assertTrue(all(type(token) is str for token in argv), argv)

    def test_the_job_name_is_the_namespaces(self):
        module = self.module
        namespace = argparse.Namespace(
            job_name="some-other-label", resume=False, rehearsal=False
        )
        self.assertEqual(
            module.build_main_argv(namespace, Path(self.DATABASE))[4],
            "some-other-label",
        )
        namespace = argparse.Namespace(atol=3e-30, rtol=2e-6)
        self.assertEqual(
            module.child_main_argv(namespace, Path(self.DATABASE))[4],
            "handover-atol-sweep-a3e-30-r2e-06",
        )


class TestNothingDisablesPruning(ArgvTestCase):
    """E1: not the token, not an abbreviation, and not anything `main.py`'s parser reads as one."""

    def test_no_list_contains_the_flag(self):
        for name, argv in self.lists.items():
            with self.subTest(list=name):
                self.assertNotIn(FLAG, argv)

    def test_no_list_contains_an_abbreviation_of_it(self):
        # main.py's parser keeps `allow_abbrev`, so `--no-prune` and `--no-p` both resolve to the
        # flag; no other option of main.py begins `--no-p`
        for name, argv in self.lists.items():
            with self.subTest(list=name):
                for token in argv:
                    self.assertFalse(token.startswith("--no-p"), token)
                    self.assertFalse(
                        token.startswith("--no") and FLAG.startswith(token), token
                    )

    def test_main_pys_own_parser_reads_every_list_as_pruning(self):
        for name, argv in self.lists.items():
            with self.subTest(list=name):
                parsed = main_py_parser().parse_args(argv[1:])
                self.assertIs(parsed.prune_unvalidated, True)

    def test_the_parser_rebuilt_from_main_py_sees_what_main_pys_does(self):
        # without this the previous test could pass because the instrument is blind
        parser = main_py_parser()
        for spelling in (FLAG, "--no-prune", "--no-prune-unvalid", "--no-p"):
            with self.subTest(spelling=spelling):
                parsed = parser.parse_args(["--database", "x", spelling])
                self.assertIs(parsed.prune_unvalidated, False)
        parsed = parser.parse_args(["--database", "x", "--prune-unvalidated"])
        self.assertIs(parsed.prune_unvalidated, True)
        parsed = parser.parse_args(["--database", "x"])
        self.assertIs(parsed.prune_unvalidated, True)

    def test_the_rebuilt_parser_reads_a_list_with_the_flag_as_not_pruning(self):
        # the same instrument on a list with the flag put back, `--resume` or not
        for argv in (
            self.lists["build"] + [FLAG],
            self.lists["child"] + ["--no-prune"],
        ):
            with self.subTest(argv=argv[-1]):
                parsed = main_py_parser().parse_args(argv[1:])
                self.assertIs(parsed.prune_unvalidated, False)


class TestMainPyDefault(unittest.TestCase):
    """E3: the lists' silence means pruning, because `main.py`'s default is true."""

    def test_prune_unvalidated_defaults_to_true_in_main_pys_source(self):
        strings, keywords = prune_default_call()
        self.assertIs(keywords["default"], True)

    def test_it_is_a_boolean_optional_action(self):
        # which is what makes `--no-prune-unvalidated` (and its abbreviations) exist at all
        strings, keywords = prune_default_call()
        self.assertIs(keywords["action"], argparse.BooleanOptionalAction)
        self.assertEqual(strings, ["--prune-unvalidated"])

    def test_no_other_option_of_main_py_begins_no_p(self):
        spellings = []
        for strings, keywords in _main_py_add_argument_calls():
            for option in strings:
                spellings.append(option)
                if keywords.get("action") is argparse.BooleanOptionalAction:
                    spellings.append("--no-" + option[2:])
        self.assertEqual(
            [s for s in spellings if s.startswith("--no-p")], [FLAG], spellings
        )


class TestTheDriversCallThem(unittest.TestCase):
    """E2: `run_build` and `run_child` build their lists with the functions, and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT_PATH.read_text())
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def _sys_argv_assignments(self, name):
        found = []
        for node in ast.walk(self.functions[name]):
            if isinstance(node, ast.Assign) and any(
                ast.unparse(target) == "sys.argv" for target in node.targets
            ):
                found.append(node)
        return found

    def test_run_build_sets_sys_argv_from_build_main_argv(self):
        (assign,) = self._sys_argv_assignments("run_build")
        self.assertEqual(ast.unparse(assign.value), "build_main_argv(args, database)")

    def test_run_child_sets_sys_argv_from_child_main_argv(self):
        (assign,) = self._sys_argv_assignments("run_child")
        self.assertEqual(
            ast.unparse(assign.value), "child_main_argv(args, SWEEP_STORE)"
        )

    def test_neither_driver_writes_an_option_string_of_its_own(self):
        # a list written back into either function would carry its options as literals
        for name in ("run_build", "run_child"):
            with self.subTest(driver=name):
                options = [
                    node.value
                    for node in ast.walk(self.functions[name])
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("--")
                    and " " not in node.value
                ]
                self.assertEqual(options, [])

    def test_the_two_builders_need_no_ray(self):
        for name in ("_main_py_argv", "build_main_argv", "child_main_argv"):
            with self.subTest(function=name):
                imported = [
                    node
                    for node in ast.walk(self.functions[name])
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                self.assertEqual(imported, [])


class TestTheUsageLinesAreClean(unittest.TestCase):
    """The two module docstrings that teach a pipeline launch by copying do not teach the flag."""

    def test_neither_usage_line_carries_the_flag_or_an_abbreviation(self):
        for path in USAGE_DOCSTRINGS:
            with self.subTest(script=path.parent.name):
                docstring = ast.get_docstring(ast.parse(path.read_text()))
                self.assertIn("--source-samples-log10z 100", docstring)
                self.assertNotIn("--no-p", docstring)


if __name__ == "__main__":
    unittest.main()
