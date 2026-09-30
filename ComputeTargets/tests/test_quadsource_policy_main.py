"""
Tests for prompt 08 of ``prompts/datastore-integrity`` (board item K3), the parts that are not the
factory (``Datastore/tests/test_quadsource_policy_key.py`` holds those): a ``QuadSourceIntegral``
is keyed on ``QuadSource_policy_1pt5`` (decision D6).

  3. ``TestTheConstructorRequiresIt`` -- the compute target takes the policy, required, as
     ``quad_policy`` beside the ``GkSourcePolicy`` it already takes as ``policy``.
  5. ``TestMainPyKeysOnQuadSourcePolicy1pt5`` -- ``main.py``'s two ``QuadSourceIntegral`` sites,
     the existence query and the compute call, pass ``QuadSource_policy_1pt5``, and no site in
     ``run_pipeline`` reaches for ``QuadSource_policy_5pt0`` (which stays created and unused,
     ``[00-the-5pt0-policy-pair-is-created-and-never-used]``). Read with ``ast``, in the style of
     ``test_main_plumbing.py``; ``main.py`` is never imported (it parses ``sys.argv`` and opens
     Ray at module scope).

and, from the orchestrator note (§0.1, §0.2):

  * ``TestTheParametersAreOneConstant`` -- ``config.defaults.DEFAULT_QUADSOURCE_POLICY_1PT5``
    holds the three values ``main.py`` used to spell out, and ``main.py`` creates
    ``QuadSource_policy_1pt5`` from it;
  * ``TestTheExtractionScript`` -- ``extract_QuadSourceIntegral_data.py`` finds that policy with
    ``read_table``, never ``object_get``, and refuses -- naming the parameters, or the serials --
    when the store holds none or more than one. **The refusal writes nothing**: run against a real
    ``ShardedPool`` on stand-in shards, it makes no call but ``read_table`` and leaves the primary
    with no ``replication_in_flight`` row. That is the reason for ``read_table``: an
    ``object_get`` of a replicated class commits that record before its call, and a refusal from
    inside it would leave it set, so that the check at open would refuse the store the script was
    reading.

No Ray, nothing under ``var/``: every store is in a temporary directory.
"""

import ast
import sqlite3
import tempfile
import unittest
from pathlib import Path

import ray

import Datastore.tests.standin_pool as sp
from ComputeTargets import QuadSourceIntegral
from ComputeTargets.tests.test_main_plumbing import (
    MAIN_PY,
    _object_get_calls,
    _own_nodes,
    _tolerance_values_under,
)
from config.defaults import DEFAULT_QUADSOURCE_POLICY_1PT5
from config.sharding import read_table_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_QSI = REPO_ROOT / "extract_QuadSourceIntegral_data.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _functions(tree) -> dict:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class _Stand:
    """Anything a constructor only stores."""

    def __init__(self, store_id=1):
        self.store_id = store_id


class TestTheConstructorRequiresIt(unittest.TestCase):
    """§3 test 3."""

    def arguments(self, **kwargs):
        arguments = dict(
            model=None,
            policy=_Stand(9),
            quad_policy=_Stand(4),
            z_response=_Stand(),
            z_source_max=_Stand(),
            k=_Stand(),
            q=_Stand(),
            r=_Stand(),
            atol=_Stand(),
            rtol=_Stand(),
            source_serial=1,
            data_serial=2,
            Tq_numeric_serial=3,
            Tq_WKB_serial=4,
            Tr_numeric_serial=5,
            Tr_WKB_serial=6,
        )
        arguments.update(kwargs)
        return arguments

    def test_it_is_carried(self):
        qsi = QuadSourceIntegral(None, **self.arguments())
        self.assertEqual(qsi.quad_policy.store_id, 4)
        # ... beside, and not instead of, the GkSourcePolicy
        self.assertEqual(qsi._policy.store_id, 9)
        self.assertEqual(qsi._quad_policy.store_id, 4)

    def test_omitting_it_is_an_error(self):
        arguments = self.arguments()
        del arguments["quad_policy"]
        with self.assertRaises(TypeError) as caught:
            QuadSourceIntegral(None, **arguments)
        self.assertIn("quad_policy", str(caught.exception))

    def test_none_is_an_error(self):
        with self.assertRaises(ValueError) as caught:
            QuadSourceIntegral(None, **self.arguments(quad_policy=None))
        self.assertIn("quad_policy", str(caught.exception))

    def test_a_populated_object_needs_it_too(self):
        arguments = self.arguments()
        del arguments["quad_policy"]
        with self.assertRaises(TypeError):
            QuadSourceIntegral({"store_id": 1}, **arguments)


class TestMainPyKeysOnQuadSourcePolicy1pt5(unittest.TestCase):
    """§3 test 5. ``main.py`` is read with ``ast``, never imported."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _tree(MAIN_PY)
        cls.functions = _functions(cls.tree)
        cls.scope = cls.functions["build_QuadSourceIntegral_batch"]

    def qsi_sites(self, key="quad_policy"):
        """Each ``QuadSourceIntegral`` object_get among the scope's own nodes: (kind, values of
        the ``key`` entry or keyword). "query" is a RayWorkPool dispatch over a batch of payload
        dicts, "construction" a direct call with keywords."""
        own = _own_nodes(self.scope)
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
                if call.args[0].value != "QuadSourceIntegral":
                    continue
                batch = node.args[1].id
                values = []
                for assign in own:
                    if isinstance(assign, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == batch
                        for t in assign.targets
                    ):
                        values += _tolerance_values_under(assign.value, key)
                sites.append(("query", values))
        for call in _object_get_calls(own):
            if id(call) in dispatched or call.args[0].value != "QuadSourceIntegral":
                continue
            values = [kw.value for kw in call.keywords if kw.arg == key]
            sites.append(("construction", values))
        return sites

    def test_the_two_sites_are_found(self):
        """A finder that found nothing would pass everything below. ``test_main_plumbing``'s
        ``EXPECTED_SITES`` pins the same count: 2."""
        self.assertEqual(
            sorted(kind for kind, _ in self.qsi_sites()), ["construction", "query"]
        )

    def test_each_site_passes_QuadSource_policy_1pt5(self):
        for kind, values in self.qsi_sites():
            with self.subTest(site=kind):
                self.assertEqual(
                    [ast.unparse(value) for value in values], ["QuadSource_policy_1pt5"]
                )

    def test_the_GkSourcePolicy_is_still_passed_beside_it(self):
        """The key gained a column; it did not swap one for the other."""
        sites = self.qsi_sites(key="policy")
        self.assertEqual(sorted(kind for kind, _ in sites), ["construction", "query"])
        for kind, values in sites:
            with self.subTest(site=kind):
                self.assertEqual(
                    [ast.unparse(value) for value in values], ["GkSource_policy_1pt5"]
                )

    def test_nothing_in_run_pipeline_reaches_for_the_5pt0_policy(self):
        """It is created, and passed into run_pipeline, and unused (D6)."""
        names = [
            node
            for node in ast.walk(self.functions["run_pipeline"])
            if isinstance(node, ast.Name) and node.id == "QuadSource_policy_5pt0"
        ]
        self.assertEqual(names, [])
        parameters = [arg.arg for arg in self.functions["run_pipeline"].args.args]
        self.assertIn("QuadSource_policy_5pt0", parameters)

    def test_the_5pt0_policy_is_still_created_from_literals(self):
        creations = [
            call
            for call in _object_get_calls(list(ast.walk(self.tree)))
            if call.args[0].value == "QuadSourcePolicy"
        ]
        self.assertEqual(len(creations), 2)
        literal = [c for c in creations if all(kw.arg for kw in c.keywords)]
        self.assertEqual(len(literal), 1)
        self.assertEqual(
            {kw.arg: ast.literal_eval(kw.value) for kw in literal[0].keywords},
            {
                "label": 'policy="maximize-WKB"-Levin-threshold="5.0"',
                "Levin_threshold": 5.0,
                "numeric_policy": "maximize-WKB",
            },
        )


class TestTheParametersAreOneConstant(unittest.TestCase):
    """§0.2: the three values are held once, in config.defaults."""

    def test_the_values(self):
        """What ``main.py`` spelled out before prompt 08. They name the row in every existing
        store, so they do not change."""
        self.assertEqual(
            DEFAULT_QUADSOURCE_POLICY_1PT5,
            {
                "label": 'policy="maximize-WKB"-Levin-threshold="1.5"',
                "Levin_threshold": 1.5,
                "numeric_policy": "maximize-WKB",
            },
        )

    def test_main_py_creates_the_1pt5_policy_from_it(self):
        creations = [
            call
            for call in _object_get_calls(list(ast.walk(_tree(MAIN_PY))))
            if call.args[0].value == "QuadSourcePolicy"
        ]
        from_constant = [
            call
            for call in creations
            if [(kw.arg, ast.unparse(kw.value)) for kw in call.keywords]
            == [(None, "DEFAULT_QUADSOURCE_POLICY_1PT5")]
        ]
        self.assertEqual(len(from_constant), 1)

    def test_main_py_imports_it(self):
        imported = [
            alias.name
            for node in ast.walk(_tree(MAIN_PY))
            if isinstance(node, ast.ImportFrom) and node.module == "config.defaults"
            for alias in node.names
        ]
        self.assertIn("DEFAULT_QUADSOURCE_POLICY_1PT5", imported)

    def test_the_extraction_script_imports_it_and_hard_codes_no_quad_policy(self):
        tree = _tree(EXTRACT_QSI)
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "config.defaults"
            for alias in node.names
        ]
        self.assertIn("DEFAULT_QUADSOURCE_POLICY_1PT5", imported)
        # no call names the class by its own literals
        for call in _object_get_calls(list(ast.walk(tree))):
            self.assertNotEqual(call.args[0].value, "QuadSourcePolicy")


def _load_find_QuadSource_policy():
    """``find_QuadSource_policy`` compiled out of the extraction script, which is a program (it
    parses ``sys.argv`` and opens a datastore at module scope) and cannot be imported. What it
    needs from the script's globals is ``ray`` and the constant, supplied as the real objects.
    """
    tree = _tree(EXTRACT_QSI)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "find_QuadSource_policy"
    ]
    if len(wanted) != 1:
        raise RuntimeError("find_QuadSource_policy is not a module-level function")
    namespace = {
        "ray": ray,
        "DEFAULT_QUADSOURCE_POLICY_1PT5": DEFAULT_QUADSOURCE_POLICY_1PT5,
    }
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(EXTRACT_QSI), "exec"),
        namespace,
    )
    return namespace["find_QuadSource_policy"]


class TestTheExtractionScript(unittest.TestCase):
    """§0.1: the script finds the policy without inserting one, and refuses -- writing nothing --
    when the store cannot say which it is."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cluster = sp.StandinCluster()
        self.cluster.controller = 0
        self.stack = self.cluster.active()
        self.stack.__enter__()
        self.addCleanup(self.stack.__exit__, None, None, None)
        self.pool = self.cluster.open_pool(
            Path(tmp.name) / "store.sqlite",
            shards=2,
            read_table_config=read_table_config,
        )
        self.addCleanup(self.cluster.close_pool)
        self.find = _load_find_QuadSource_policy()

    def create_policy(self):
        """Create the 1.5 policy the way main.py does, through object_get."""
        return ray.get(
            self.pool.object_get("QuadSourcePolicy", **DEFAULT_QUADSOURCE_POLICY_1PT5)
        )

    def add_a_duplicate(self, serial):
        """A second row build() would call the same policy, written raw to every shard file (the
        pool's own writes cannot make one)."""
        for path in self.pool._shard_db_files.values():
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    'INSERT INTO "QuadSourcePolicy" '
                    "(serial, label, Levin_threshold, numeric_policy) VALUES (?, ?, ?, ?)",
                    (serial, "duplicate", 1.5, "maximize-WKB"),
                )
                conn.commit()
            finally:
                conn.close()

    def state(self):
        return (
            sp.shard_rows(self.pool, "QuadSourcePolicy"),
            sp.in_flight_records(self.pool),
        )

    def find_and_check_nothing_was_written(self):
        """Call the script's lookup; whatever it does, it makes no call but read_table and
        changes no row, and leaves no in-flight record."""
        before, calls = self.state(), len(self.cluster.calls)
        try:
            return self.find(self.pool)
        finally:
            made = self.cluster.calls[calls:]
            self.assertGreater(len(made), 0)
            self.assertEqual({call["method"] for call in made}, {"read_table"})
            self.assertEqual({call["class"] for call in made}, {"QuadSourcePolicy"})
            self.assertEqual(self.state(), before)
            self.assertEqual(sp.in_flight_records(self.pool), [])

    def test_it_finds_the_policy_main_py_created(self):
        created = self.create_policy()
        found = self.find_and_check_nothing_was_written()
        self.assertEqual(found.store_id, created.store_id)
        self.assertEqual(found.Levin_threshold, 1.5)
        self.assertEqual(found.numeric_policy, "maximize-WKB")

    def test_it_is_not_confused_by_the_5pt0_policy(self):
        created = self.create_policy()
        ray.get(
            self.pool.object_get(
                "QuadSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize-WKB",
            )
        )
        self.assertEqual(
            self.find_and_check_nothing_was_written().store_id, created.store_id
        )

    def test_no_row_is_refused_naming_the_parameters_and_writes_nothing(self):
        with self.assertRaises(RuntimeError) as caught:
            self.find_and_check_nothing_was_written()
        message = str(caught.exception)
        self.assertIn("Levin_threshold=1.5", message)
        self.assertIn('numeric_policy="maximize-WKB"', message)
        self.assertIn("holds no QuadSourcePolicy", message)
        self.assertIn("does not create one", message)
        # and the store still has none: the refusal did not insert one
        self.assertEqual(self.state()[0], {0: [], 1: []})

    def test_a_different_policy_alone_is_no_row(self):
        ray.get(
            self.pool.object_get(
                "QuadSourcePolicy",
                label="other",
                Levin_threshold=5.0,
                numeric_policy="maximize-WKB",
            )
        )
        with self.assertRaises(RuntimeError) as caught:
            self.find_and_check_nothing_was_written()
        self.assertIn("holds no QuadSourcePolicy", str(caught.exception))

    def test_more_than_one_row_is_refused_naming_their_serials(self):
        created = self.create_policy()
        self.add_a_duplicate(serial=created.store_id + 1)
        with self.assertRaises(RuntimeError) as caught:
            self.find_and_check_nothing_was_written()
        message = str(caught.exception)
        self.assertIn("holds 2 QuadSourcePolicy rows", message)
        self.assertIn(f"serials {created.store_id}, {created.store_id + 1}", message)
        self.assertIn("Levin_threshold=1.5", message)

    def test_the_control_object_get_is_a_different_call(self):
        """The reason for read_table (§0.1): the pool's replicated write commits an in-flight
        record on the primary before the call. This shows that the calls the other tests check for
        are what tell the two apart: creating the policy through ``object_get`` makes an
        ``object_get`` call (and leases serials), which the script's lookup never does.
        """
        calls = len(self.cluster.calls)
        self.create_policy()
        created_by = {c["method"] for c in self.cluster.calls[calls:]}
        self.assertIn("object_get", created_by)
        self.assertNotIn("read_table", created_by)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
