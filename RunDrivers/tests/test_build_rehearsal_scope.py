"""`quadsource_atol_sweep.py --build --rehearsal`, for `a3-v2-readiness` prompt 06a (**B2**).

`--build` runs A3 v2's production work list, which takes days. Prompt 07 rehearses A3 v2's launch
through `run_build`, the code prompts 04 and 05 changed, and needs a smaller scope reached through
that same code. So `--build` gained one flag, `--rehearsal`, which selects one fixed scope:
`datastore-integrity` prompt 10's, three log-spaced k over 1e5-1e6 /Mpc with `--zend 250`.
Without the flag every list, string and value is what it was. A scope that moved `--zend` and not
the grid, or the grid and not the manifest's description, would run without error and be recorded
wrongly, so one function, `build_scope`, decides the k sample, the `--zend` and the manifest's
`scope` text, and `run_build` and `build_main_argv` read them from it.

What is tested, and how:

* **Production is unchanged** (`TestProduction`): the k sample is `k_sample()`, the argv list is
  the production list and the scope text is today's string, each as a literal.
* **The rehearsal moves all three together** (`TestRehearsal`): the k sample, the argv list (with
  and without `--resume`) and the scope text, each as a literal or a literal's parts.
* **It is refused outside `--build`, before any mode runs** (`TestRefusals`): `main()` on a loaded
  copy of the module with `run_build`, `run_child`, `prepare`, `report` and `sweep` all stubbed,
  so that a refusal that fails to happen cannot copy a store or launch a child.
* **The drivers read the scope** (`TestTheDriversReadTheScope`), with `ast`: `run_build`'s
  namespace takes its grid from the scope function's result, and its manifest takes its `scope`
  from the scope's text, and it holds no call of `k_sample()` of its own; `run_child` is still
  production.

Nothing here needs Ray, a datastore or anything under `var/`. The parser is the script's own, as in
`test_prune_on_resume`, whose `parsed_arguments` this module reuses.
"""

import ast
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

from RunDrivers.tests.test_prune_on_resume import parsed_arguments

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "RunDrivers" / "quadsource_atol_sweep.py"

# `Path.resolve()` turns macOS's `/tmp` into `/private/tmp`, and `run_build` resolves the path it
# is given, so the database passed here is already a resolved path
DATABASE = "/private/tmp/build-rehearsal-scope/store.sqlite"
PAIR = ["--atol", "1e-32", "--rtol", "1e-7"]


def argv_with_zend(zend):
    return [
        "main.py",
        "--database",
        DATABASE,
        "--job-name",
        "handover-A3-baseline-v2",
        "--shards",
        "4",
        "--zend",
        zend,
        "--source-samples-log10z",
        "100",
    ]


# the manifest's `scope` as `run_build` wrote it before 06a, at atol 1e-32, rtol 1e-7, the
# default 6 cpus and a store named `store.sqlite`
PRODUCTION_TEXT = (
    "full build: main.py's own 9280-item QuadSourceIntegral work list at atol=1e-32 "
    "rtol=1e-07, 6 cpus, LambdaCDM, 8 log-spaced k over 100000-3e+08 /Mpc, into store.sqlite"
)


def scope_of(command):
    """`(module, namespace, k sample, build_main_argv's list, scope text)` for a `--build` line.

    Parsed by the script's own parser, defaults and all. The text is filled with the run's own
    pair, cpus and store name, as `run_build` fills it.
    """
    module, which, namespace = parsed_arguments(
        ["--build", "--database", DATABASE] + PAIR + command
    )
    assert which == "build"
    k_values, _zend, template = module.build_scope(namespace)
    text = template.format(
        atol=namespace.atol,
        rtol=namespace.rtol,
        cpus=namespace.cpus,
        database=Path(DATABASE).name,
    )
    return (
        module,
        namespace,
        k_values,
        module.build_main_argv(namespace, Path(DATABASE)),
        text,
    )


class TestProduction(unittest.TestCase):
    """E1: without the flag, the k sample, the argv list and the scope text are today's."""

    def test_the_flag_defaults_to_off(self):
        module, namespace, *_ = scope_of([])
        self.assertIs(namespace.rehearsal, False)

    def test_the_k_sample_is_k_sample(self):
        module, namespace, k_values, argv, text = scope_of([])
        self.assertEqual(k_values, module.k_sample())

    def test_the_k_sample_is_the_eight_point_production_grid(self):
        # k_sample() could itself drift; this pins what it is
        module, namespace, k_values, argv, text = scope_of([])
        self.assertEqual(len(k_values), 8)
        self.assertEqual(k_values[0], 1e5)
        self.assertAlmostEqual(k_values[-1] / 3e8, 1.0, places=12)

    def test_the_argv_list_is_the_production_list(self):
        module, namespace, k_values, argv, text = scope_of([])
        self.assertEqual(argv, argv_with_zend("0.1"))

    def test_the_resumed_argv_list_is_the_same_list(self):
        module, namespace, k_values, argv, text = scope_of(["--resume"])
        self.assertEqual(argv, argv_with_zend("0.1"))

    def test_the_scope_text_is_what_run_build_wrote_before(self):
        module, namespace, k_values, argv, text = scope_of([])
        self.assertEqual(text, PRODUCTION_TEXT)


class TestRehearsal(unittest.TestCase):
    """E2: with the flag the k sample, `--zend` and the text move together."""

    def test_the_flag_is_on(self):
        module, namespace, *_ = scope_of(["--rehearsal"])
        self.assertIs(namespace.rehearsal, True)

    def test_the_k_sample_is_three_points_over_1e5_to_1e6(self):
        module, namespace, k_values, argv, text = scope_of(["--rehearsal"])
        self.assertEqual(k_values, [1e5, 316227.7660168379, 1e6])

    def test_the_argv_list_has_zend_250_and_nothing_else_moved(self):
        module, namespace, k_values, argv, text = scope_of(["--rehearsal"])
        self.assertEqual(argv, argv_with_zend("250"))

    def test_the_resumed_argv_list_is_the_same_list(self):
        module, namespace, k_values, argv, text = scope_of(["--rehearsal", "--resume"])
        self.assertEqual(argv, argv_with_zend("250"))

    def test_the_resumed_k_sample_is_the_same(self):
        module, namespace, k_values, argv, text = scope_of(["--rehearsal", "--resume"])
        self.assertEqual(k_values, [1e5, 316227.7660168379, 1e6])

    def test_the_scope_text_says_what_ran(self):
        module, namespace, k_values, argv, text = scope_of(["--rehearsal"])
        self.assertIn("REHEARSAL build, not A3 v2", text)
        self.assertIn("3 log-spaced k over 100000-1e+06 /Mpc", text)
        self.assertIn("zend 250", text)
        self.assertIn("atol=1e-32 rtol=1e-07, 6 cpus", text)
        self.assertTrue(text.endswith("into store.sqlite"), text)

    def test_the_scope_text_does_not_say_the_production_size(self):
        # 9280 is the production work list's size, not this one's
        module, namespace, k_values, argv, text = scope_of(["--rehearsal"])
        self.assertNotIn("9280", text)
        self.assertNotIn("8 log-spaced", text)
        self.assertNotIn("3e+08", text)

    def test_the_two_scopes_differ_in_the_three_places_and_no_others(self):
        _, _, production_k, production_argv, production_text = scope_of([])
        _, _, rehearsal_k, rehearsal_argv, rehearsal_text = scope_of(["--rehearsal"])
        differing = [
            i for i, (a, b) in enumerate(zip(production_argv, rehearsal_argv)) if a != b
        ]
        self.assertEqual(len(production_argv), len(rehearsal_argv))
        self.assertEqual(differing, [8])  # the token after `--zend`
        self.assertNotEqual(production_k, rehearsal_k)
        self.assertNotEqual(production_text, rehearsal_text)


class StubbedMain:
    """`main()` on a fresh copy of the script, with every mode replaced by a recorder.

    A refusal that does not happen must not be able to copy a store (`prepare`), print a report,
    launch a child build per tolerance pair (`sweep`) or start Ray (`run_build`, `run_child`).
    """

    MODES = ("run_build", "run_child", "prepare", "report", "sweep")

    def __init__(self, command_line):
        spec = importlib.util.spec_from_file_location(
            "_quadsource_atol_sweep_rehearsal_refusals", str(SCRIPT_PATH)
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.reached = []
        for name in self.MODES:
            setattr(
                self.module,
                name,
                lambda *a, _name=name, **k: self.reached.append(_name),
            )
        self.stderr = io.StringIO()
        self.exit_code = None
        with mock.patch.object(
            sys, "argv", ["quadsource_atol_sweep.py"] + command_line
        ), contextlib.redirect_stderr(self.stderr):
            try:
                self.module.main()
            except SystemExit as exc:
                self.exit_code = exc.code


class TestRefusals(unittest.TestCase):
    """E3: `--rehearsal` is refused outside `--build`, and before any mode runs."""

    def assertRefused(self, command_line):
        main = StubbedMain(command_line)
        self.assertEqual(main.exit_code, 2, main.stderr.getvalue())
        self.assertEqual(main.reached, [])
        self.assertIn("--rehearsal", main.stderr.getvalue())
        self.assertIn("--build", main.stderr.getvalue())

    def test_with_child_is_refused(self):
        self.assertRefused(["--child", "--rehearsal"] + PAIR)

    def test_with_prepare_is_refused(self):
        self.assertRefused(["--prepare", "--rehearsal"])

    def test_with_report_is_refused(self):
        self.assertRefused(["--report", "--rehearsal"])

    def test_with_no_mode_is_refused(self):
        # the sweep, which launches a child build per tolerance pair
        self.assertRefused(["--rehearsal"])

    def test_with_no_mode_and_a_registration_is_refused(self):
        self.assertRefused(
            ["--rehearsal", "--register", "some-slug", "--purpose", "a purpose"]
        )

    def test_the_stubs_reach_a_build_with_the_flag(self):
        # the instrument is not blind: the same stubs see `run_build` reached when it is allowed
        main = StubbedMain(["--build", "--rehearsal", "--database", DATABASE] + PAIR)
        self.assertIsNone(main.exit_code)
        self.assertEqual(main.reached, ["run_build"])

    def test_the_stubs_reach_each_mode_without_the_flag(self):
        for command_line, mode in (
            (["--prepare"], "prepare"),
            (["--report"], "report"),
            (["--child"] + PAIR, "run_child"),
            ([], "sweep"),
        ):
            with self.subTest(mode=mode):
                main = StubbedMain(command_line)
                self.assertIsNone(main.exit_code, main.stderr.getvalue())
                self.assertEqual(main.reached, [mode])


class TestTheDriversReadTheScope(unittest.TestCase):
    """`run_build` takes its grid and its manifest's text from the scope; `run_child` does not."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT_PATH.read_text())
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def _namespace_value(self, name):
        """The value `name` gives `K_GRID_NAME` in its `exec` namespace dict, unparsed."""
        found = [
            ast.unparse(value)
            for node in ast.walk(self.functions[name])
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Name) and key.id == "K_GRID_NAME"
        ]
        self.assertEqual(len(found), 1, found)
        return found[0]

    def _scope_unpacking(self):
        """The one statement of `run_build` that unpacks `build_scope(args)`: its target names."""
        found = [
            [ast.unparse(element) for element in node.targets[0].elts]
            for node in ast.walk(self.functions["run_build"])
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Tuple)
            and ast.unparse(node.value) == "build_scope(args)"
        ]
        self.assertEqual(len(found), 1, found)
        return found[0]

    def test_run_builds_namespace_takes_the_grid_from_the_scope(self):
        k_name, _zend_name, _text_name = self._scope_unpacking()
        self.assertEqual(self._namespace_value("run_build"), k_name)

    def test_run_build_holds_no_call_of_k_sample(self):
        calls = [
            ast.unparse(node)
            for node in ast.walk(self.functions["run_build"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "k_sample"
        ]
        self.assertEqual(calls, [])

    def test_run_builds_manifest_takes_its_scope_from_the_scopes_text(self):
        _k_name, _zend_name, text_name = self._scope_unpacking()
        scopes = [
            ast.unparse(keyword.value)
            for node in ast.walk(self.functions["run_build"])
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "runregistrykit.begin"
            for keyword in node.keywords
            if keyword.arg == "scope"
        ]
        self.assertEqual(len(scopes), 1, scopes)
        self.assertTrue(scopes[0].startswith(f"{text_name}.format("), scopes[0])

    def test_run_build_still_sets_sys_argv_from_build_main_argv(self):
        # `test_prune_on_resume` pins its form; this is the one that 06a had to keep
        assigns = [
            ast.unparse(node.value)
            for node in ast.walk(self.functions["run_build"])
            if isinstance(node, ast.Assign)
            and any(ast.unparse(t) == "sys.argv" for t in node.targets)
        ]
        self.assertEqual(assigns, ["build_main_argv(args, database)"])

    def test_run_child_is_still_production(self):
        self.assertEqual(self._namespace_value("run_child"), "k_sample()")

    def test_run_child_does_not_read_the_scope(self):
        names = {
            node.id
            for node in ast.walk(self.functions["run_child"])
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("build_scope", names)
        self.assertNotIn("rehearsal", ast.unparse(self.functions["run_child"]))

    def test_build_scope_needs_no_ray(self):
        imported = [
            node
            for node in ast.walk(self.functions["build_scope"])
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(imported, [])


if __name__ == "__main__":
    unittest.main()
