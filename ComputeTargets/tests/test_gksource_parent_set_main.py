"""
Tests for prompt 09b of ``prompts/datastore-integrity`` (board item K6), the parts that are not the
factory, the value reads or the inventory (``ObjectFactories/tests/test_gksource_parent_set.py`` holds
those): a ``GkSource`` is keyed on the set of Gk numeric and WKB rows it is assembled from.

  * ``TestTheConstructor`` -- an object to be computed names the parent set it is keyed on;
    ``labels_only`` is the reader mode's parentless miss, which cannot be computed.
  * ``TestTheAssertion`` -- ``compute()`` raises, naming the ``z_source``s at which they differ,
    when the values it is handed were read from rows other than the set it is keyed on (charter
    §2: a derived row is keyed on the rows its ``compute()`` receives), and refuses an empty set.
  * ``TestThePipelinesOwnLogic`` -- the prompt's first stop condition, on a store: ``main.py``'s
    own ``resolve_GkSource_parents``, extracted with ``ast`` and run against the real factories,
    resolves the set; the values ``build_missing_GkSource`` would read are assembled by the real
    ``assemble_GkSource_values``; and the row the real factory writes records exactly the set
    assembled, which is the set the lookup was keyed on.
  8. ``TestMainPyKeysEveryGkSourceLookup`` -- ``main.py``, read with ``ast`` and never imported:
     the GkSource stage resolves the set before its existence query, and all six ``GkSource``
     lookups are keyed on a set that the one shared helper resolved.
  * ``TestTheReaders`` -- the four ``GkSource`` readers hold only physical labels and ask for the
    reader mode, which refuses more than one stored match.

No Ray, nothing under ``var/``: the store is in memory.
"""

import ast
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ray

from ComputeTargets import GkSource
from ComputeTargets.GkSource import (
    GkSource_parent_set_of_serials,
    GkSource_parent_set_of_values,
)
from ComputeTargets.tests.test_main_plumbing import MAIN_PY
from config.datastore import factories as _factories
from ObjectFactories.GkSource import (
    GkSource_parents_digest,
    sqla_GkSource_factory,
)
from ObjectFactories.tests.test_gksource_parent_set import (
    ATOL,
    K_EXIT,
    MODEL,
    RTOL,
    SET_A,
    TAG,
    Z_RESPONSE,
    Z_SAMPLE,
    _gk_store,
    source_lookup,
    stored_members,
)

# the module itself: `import ComputeTargets.GkSource as ...` binds the package's re-exported class
# of the same name, not the module that holds assemble_GkSource_values
gk_source_module = importlib.import_module("ComputeTargets.GkSource")

REPO_ROOT = Path(__file__).resolve().parents[2]
READERS = (
    REPO_ROOT / "extract_GkSource_data.py",
    REPO_ROOT / "extract_QuadSourceIntegral_data.py",
    REPO_ROOT
    / "docs"
    / "source-remediation-verification"
    / "analyse_greens_and_source.py",
    REPO_ROOT
    / "docs"
    / "source-remediation-verification"
    / "run_quadsource_integrals.py",
)

# the six functions of run_pipeline that look a GkSource up, each once
GKSOURCE_LOOKUPS = (
    "build_GkSource_batch",
    "build_missing_GkSource",
    "build_incomplete_GkSourcePolicy",
    "build_GkSourcePolicy_batch",
    "build_missing_GkSourcePolicy",
    "build_QuadSourceIntegral_batch",
)
HELPER = "resolve_GkSource_parents"


def _value(parent):
    """Stands in for a GkNumericValue or GkWKBValue, as far as the parent set goes."""
    return SimpleNamespace(_parent_serial=parent)


def _maps(members):
    """The numeric and WKB maps (z_source serial -> value) that ``members`` would be read as."""
    numeric = {z: _value(n) for z, n, _ in members if n is not None}
    WKB = {z: _value(w) for z, _, w in members if w is not None}
    return numeric, WKB


def _source(parents=SET_A, **kwargs):
    return GkSource(
        payload=None,
        model=MODEL,
        k=K_EXIT,
        z_response=Z_RESPONSE,
        z_sample=Z_SAMPLE,
        tags=[TAG],
        parents=parents,
        **kwargs,
    )


def _compute(source, numeric, WKB):
    """compute(), with the Ray task stood in for."""
    task = mock.MagicMock()
    task.remote.return_value = "a Ray reference"
    with mock.patch.object(gk_source_module, "assemble_GkSource_values", task):
        return source.compute(payload={"numeric": numeric, "WKB": WKB}), task


class TestTheConstructor(unittest.TestCase):
    def test_it_carries_the_set_it_is_keyed_on_in_canonical_form(self):
        source = _source(parents=tuple(reversed(SET_A)))
        self.assertEqual(source.parents, SET_A)
        self.assertIsNone(source.assembled_parents)
        self.assertFalse(source.labels_only)

    def test_an_object_to_be_computed_names_its_set(self):
        with self.assertRaises(ValueError) as caught:
            GkSource(
                payload=None,
                model=MODEL,
                k=K_EXIT,
                z_response=Z_RESPONSE,
                z_sample=Z_SAMPLE,
            )
        self.assertIn("parents=", str(caught.exception))

    def test_a_labels_only_miss_names_none_and_cannot_be_computed(self):
        miss = GkSource(
            payload=None,
            model=MODEL,
            k=K_EXIT,
            z_response=Z_RESPONSE,
            z_sample=Z_SAMPLE,
            labels_only=True,
        )
        self.assertIsNone(miss.parents)
        with self.assertRaises(RuntimeError) as caught:
            _compute(miss, *_maps(SET_A))
        self.assertIn("labels_only", str(caught.exception))

    def test_a_labels_only_object_given_a_set_is_an_error(self):
        with self.assertRaises(ValueError):
            _source(labels_only=True)

    def test_a_malformed_set_is_refused(self):
        with self.assertRaises(ValueError):
            _source(parents=((21, None, None),))


class TestTheAssertion(unittest.TestCase):
    def test_the_rows_it_is_keyed_on_are_accepted(self):
        ref, task = _compute(_source(), *_maps(SET_A))
        self.assertEqual(ref, "a Ray reference")
        task.remote.assert_called_once()

    def test_another_row_at_one_z_source_raises_naming_it(self):
        other = ((21, 43, None),) + SET_A[1:]
        with self.assertRaises(RuntimeError) as caught:
            _compute(_source(), *_maps(other))
        message = str(caught.exception)
        self.assertIn("differ at 1 z_source serial(s), e.g. [21]", message)
        self.assertIn(f"k store_id={K_EXIT.store_id}", message)

    def test_a_missing_or_an_extra_parent_raises(self):
        for members in (
            ((21, 41, None), (22, 42, None), (23, None, 52)),
            ((21, 41, None), (22, 42, 51), (23, 45, 52)),
            SET_A[:2],
        ):
            with self.subTest(members=members):
                with self.assertRaises(RuntimeError):
                    _compute(_source(), *_maps(members))

    def test_a_value_outside_the_sample_is_not_a_member(self):
        """assemble_GkSource_values reads only the z_sources of its sample, so a value at another
        z_source is not assembled, and is not a parent."""
        numeric, WKB = _maps(SET_A)
        numeric[99] = _value(77)
        ref, _ = _compute(_source(), numeric, WKB)
        self.assertEqual(ref, "a Ray reference")

    def test_nothing_is_computed_when_it_raises(self):
        task = mock.MagicMock()
        with mock.patch.object(gk_source_module, "assemble_GkSource_values", task):
            with self.assertRaises(RuntimeError):
                _source().compute(
                    payload=dict(zip(("numeric", "WKB"), _maps(SET_A[:1])))
                )
        task.remote.assert_not_called()

    def test_a_value_that_does_not_say_its_parent_raises(self):
        numeric, WKB = _maps(SET_A)
        numeric[21] = SimpleNamespace()
        with self.assertRaises(RuntimeError) as caught:
            _compute(_source(), numeric, WKB)
        self.assertIn("does not say which integration", str(caught.exception))

    def test_an_empty_set_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _compute(_source(parents=()), {}, {})
        self.assertIn("empty parent set", str(caught.exception))


class _WorkPool:
    """Stands in for RayWorkPool: runs the task builder over the items, in order, and keeps the
    results, which the stand-in pool returns directly rather than as references."""

    def __init__(self, pool, items, task_builder, **kwargs):
        self.items = list(items)
        self.task_builder = task_builder

    def run(self):
        self.results = [self.task_builder(item) for item in self.items]


class _Pool:
    """Stands in for ShardedPool.object_read_batch on one in-memory shard: the payload gains the
    shard key, as ShardedPool's does, and the real factory's read_batch runs."""

    def __init__(self, store):
        self.store = store
        self.calls = []

    def object_read_batch(self, cls_name, shard_key, **payload):
        payload.update(shard_key)
        self.calls.append((cls_name, dict(payload)))
        return _factories[cls_name].read_batch(
            payload, self.store.conn, self.store.tables[cls_name], self.store.tables
        )


def _run_pipeline_function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY))
    (run_pipeline,) = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline"
    ]
    found = [
        n
        for n in ast.walk(run_pipeline)
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(found) != 1:
        raise AssertionError(f"{len(found)} functions {name} in run_pipeline")
    return found[0]


def _load_helper(pool):
    """main.py's own resolve_GkSource_parents, compiled from its source alone, its free names
    bound to stand-ins: the pool, the work pool, the run's model, tolerances, tags and grid.
    """
    node = _run_pipeline_function(HELPER)
    code = compile(ast.Module(body=[node], type_ignores=[]), str(MAIN_PY), "exec")
    namespace = {
        "pool": pool,
        "RayWorkPool": _WorkPool,
        "model_proxy": MODEL,
        "Gk_numeric_atol": ATOL,
        "Gk_numeric_rtol": RTOL,
        "z_source_sample": Z_SAMPLE,
        "GkSource_parent_set_of_serials": GkSource_parent_set_of_serials,
    }
    for tag in (
        "GkProductionTag",
        "RunLabelTag",
        "SourceGridConstructionTag",
        "SourceZGridSizeTag",
        "ResponseZGridSizeTag",
        "LargestSourceZTag",
        "SmallestSourceZTag",
        "ResponseSparsenessZTag",
    ):
        # the store's rows carry one tag, TAG; main.py asks for eight, which here are all it
        namespace[tag] = TAG
    exec(code, namespace)
    return namespace[HELPER]


class TestThePipelinesOwnLogic(unittest.TestCase):
    """
    The prompt's first stop condition: in a store written by the pipeline's own logic, the set
    recorded must be the set assembled. The GkSource stage is run here piece by piece, on the real
    factories: main.py's helper resolves the set serial-only; the values are read as
    build_missing_GkSource reads them; the real assemble_GkSource_values runs (its Ray task called
    directly); the target's store() collects what it assembled; the real factory writes the row.
    """

    def setUp(self):
        self.s = _gk_store()
        self.addCleanup(self.s.close)
        self.pool = _Pool(self.s)
        self.resolve = _load_helper(self.pool)

    def full_read(self, cls_name):
        payload = {"model": MODEL, "z": Z_RESPONSE, "tags": [TAG]}
        if cls_name == "GkNumericValue":
            payload.update(atol=ATOL, rtol=RTOL)
        return self.pool.object_read_batch(cls_name, {"k": K_EXIT}, **payload)

    def test_the_helper_resolves_the_set_serial_only(self):
        parents = self.resolve([(K_EXIT, [Z_RESPONSE])])
        self.assertEqual(parents, {(K_EXIT.store_id, Z_RESPONSE.store_id): SET_A})
        self.assertEqual(
            [(c, p["parents_only"]) for c, p in self.pool.calls],
            [("GkNumericValue", True), ("GkWKBValue", True)],
        )

    def test_nothing_wanted_reads_nothing(self):
        self.assertEqual(self.resolve([]), {})
        self.assertEqual(self.resolve([(K_EXIT, [])]), {})
        self.assertEqual(self.pool.calls, [])

    def test_the_row_records_the_set_assembled_which_is_the_set_it_was_keyed_on(self):
        parents = self.resolve([(K_EXIT, [Z_RESPONSE])])[
            (K_EXIT.store_id, Z_RESPONSE.store_id)
        ]
        self.assertFalse(source_lookup(self.s, parents=parents).available)

        # build_missing_GkSource's read and its maps
        numeric = {v._z_source.store_id: v for v in self.full_read("GkNumericValue")}
        WKB = {v._z_source.store_id: v for v in self.full_read("GkWKBValue")}
        self.assertEqual(GkSource_parent_set_of_values(Z_SAMPLE, numeric, WKB), parents)

        obj = GkSource(
            payload=None,
            model=MODEL,
            k=K_EXIT,
            z_response=Z_RESPONSE,
            z_sample=Z_SAMPLE,
            tags=[TAG],
            parents=parents,
        )
        results = {}
        # the real remote's undecorated function, called in-process in place of the Ray task
        assemble = gk_source_module.assemble_GkSource_values._function

        class _Task:
            @staticmethod
            def remote(*args):
                results["payload"] = assemble(*args)
                return "a Ray reference"

        with mock.patch.object(gk_source_module, "assemble_GkSource_values", _Task):
            ref = obj.compute(payload={"numeric": numeric, "WKB": WKB})
        with mock.patch.object(
            gk_source_module.ray, "wait", return_value=([ref], [])
        ), mock.patch.object(
            gk_source_module.ray, "get", return_value=results["payload"]
        ):
            obj.store()

        self.assertEqual(obj.assembled_parents, parents)
        sqla_GkSource_factory.store(
            obj,
            self.s.conn,
            self.s.tables["GkSource"],
            self.s.inserters["GkSource"],
            self.s.tables,
            self.s.inserters,
        )
        self.s.conn.commit()

        self.assertEqual(stored_members(self.s, obj.store_id), parents)
        self.assertEqual(len(obj.values), 3)
        table = self.s.tables["GkSource"]
        digest = self.s.conn.execute(
            table.select()
            .with_only_columns(table.c.parents_digest)
            .where(table.c.serial == obj.store_id)
        ).scalar_one()
        self.assertEqual(digest, GkSource_parents_digest(parents))

        # and a resume, resolving the set again, finds it
        self.s.conn.execute(
            table.update().where(table.c.serial == obj.store_id).values(validated=True)
        )
        self.s.conn.commit()
        again = self.resolve([(K_EXIT, [Z_RESPONSE])])[
            (K_EXIT.store_id, Z_RESPONSE.store_id)
        ]
        self.assertEqual(source_lookup(self.s, parents=again).store_id, obj.store_id)


def _lookups(scope, cls_name="GkSource"):
    """The object_get / object_get_vectorized calls in ``scope`` whose first argument is
    ``cls_name``."""
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("object_get", "object_get_vectorized")
        and len(node.args) > 0
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == cls_name
    ]


def _keys(node: ast.Dict):
    return [k.value for k in node.keys if isinstance(k, ast.Constant)]


def _gk_source_payloads(scope):
    """The dict literals in ``scope`` that are a GkSource lookup's payload: they name the
    z_response and a z_sample, and no source (a GkSourcePolicyData payload names its source).
    """
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Dict)
        and {"z_response", "z_sample", "tags"} <= set(_keys(node))
        and "source" not in _keys(node)
        and "q" not in _keys(node)
    ]


def _is_the_resolved_set(node) -> bool:
    """``GkSource_parents[(k.store_id, z_response.store_id)]``: the helper's result, indexed."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "GkSource_parents"
    )


class TestMainPyKeysEveryGkSourceLookup(unittest.TestCase):
    """§3 test 8."""

    @classmethod
    def setUpClass(cls):
        cls.functions = {
            name: _run_pipeline_function(name) for name in GKSOURCE_LOOKUPS
        }
        cls.helper = _run_pipeline_function(HELPER)

    def test_each_function_looks_a_gk_source_up_once(self):
        total = 0
        for name, fn in self.functions.items():
            with self.subTest(function=name):
                self.assertEqual(len(_lookups(fn)), 1)
                total += 1
        # the six stage lookups prompt 09b keyed. test_main_plumbing.EXPECTED_SITES counts seven
        # GkSource sites since prompt 09c: the seventh is resolve_QuadSourceIntegral_parents'
        # serial-only lookup, which is not one of these functions
        self.assertEqual(total, 6)

    def test_every_lookup_is_keyed_on_a_resolved_set(self):
        for name, fn in self.functions.items():
            with self.subTest(function=name):
                (call,) = _lookups(fn)
                keywords = {k.arg: k.value for k in call.keywords}
                if call.func.attr == "object_get":
                    self.assertIn("parents", keywords)
                    self.assertTrue(_is_the_resolved_set(keywords["parents"]))
                    continue
                (payload,) = _gk_source_payloads(fn)
                self.assertIn("parents", _keys(payload))
                value = payload.values[_keys(payload).index("parents")]
                self.assertTrue(_is_the_resolved_set(value))

    def test_no_stage_asks_for_the_reader_mode(self):
        for name, fn in list(self.functions.items()) + [(HELPER, self.helper)]:
            with self.subTest(function=name):
                self.assertNotIn("labels_only", ast.dump(fn))

    def test_every_set_comes_from_the_one_helper(self):
        """Each function either calls the helper itself or is handed its result by the one that
        does, so no stage can form a set of its own."""
        callers = {}
        for name, fn in self.functions.items():
            calls = [
                node
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == HELPER
            ]
            parameters = [a.arg for a in fn.args.args]
            with self.subTest(function=name):
                self.assertEqual(
                    len(calls) + parameters.count("GkSource_parents"), 1, parameters
                )
            callers[name] = len(calls) == 1
        self.assertEqual(
            {name for name, calls in callers.items() if calls},
            {
                "build_GkSource_batch",
                "build_GkSourcePolicy_batch",
                "build_QuadSourceIntegral_batch",
            },
        )

    def test_the_gk_source_stage_resolves_the_set_before_its_existence_query(self):
        fn = self.functions["build_GkSource_batch"]
        (resolve,) = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == HELPER
        ]
        (query,) = _lookups(fn)
        (payload,) = _gk_source_payloads(fn)
        self.assertLess(resolve.lineno, payload.lineno)
        self.assertLess(resolve.lineno, query.lineno)

    def test_the_downstream_stages_resolve_before_their_lookups(self):
        for name in ("build_GkSourcePolicy_batch", "build_QuadSourceIntegral_batch"):
            fn = self.functions[name]
            with self.subTest(function=name):
                (resolve,) = [
                    node
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == HELPER
                ]
                (query,) = _lookups(fn)
                self.assertLess(resolve.lineno, query.lineno)

    def test_the_helper_reads_both_classes_serial_only(self):
        (payload,) = [
            node
            for node in ast.walk(self.helper)
            if isinstance(node, ast.Dict) and "parents_only" in _keys(node)
        ]
        value = payload.values[_keys(payload).index("parents_only")]
        self.assertIsInstance(value, ast.Constant)
        self.assertIs(value.value, True)
        self.assertIn(
            "['GkNumericValue', 'GkWKBValue']",
            ast.unparse(self.helper),
        )
        calls = [
            node.func.attr
            for node in ast.walk(self.helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("object_read_batch", calls)
        self.assertNotIn("object_get", calls)

    def test_the_helper_forms_the_set_over_the_grid_the_gk_source_stage_assembles_over(
        self,
    ):
        def truncations(fn):
            return {
                ast.unparse(node)
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "truncate"
            }

        self.assertEqual(
            truncations(self.helper),
            {"z_source_sample.truncate(z_response, keep='higher')"},
        )
        self.assertIn(
            "z_source_sample.truncate(z_response, keep='higher')",
            truncations(self.functions["build_GkSource_batch"]),
        )

    def test_the_construction_passes_the_set_its_existence_query_used(self):
        fn = self.functions["build_missing_GkSource"]
        self.assertIn("GkSource_parents", [a.arg for a in fn.args.args])
        (call,) = _lookups(fn)
        self.assertEqual(
            ast.unparse({k.arg: k.value for k in call.keywords}["parents"]),
            "GkSource_parents[k_exit.store_id, z_response.store_id]",
        )


class TestTheReaders(unittest.TestCase):
    def test_each_gk_source_lookup_asks_for_labels_only(self):
        for path in READERS:
            tree = ast.parse(path.read_text(), filename=str(path))
            calls = _lookups(tree)
            with self.subTest(reader=path.name):
                self.assertEqual(len(calls), 1)
                keywords = {k.arg: k.value for k in calls[0].keywords}
                self.assertIn("labels_only", keywords)
                self.assertIs(keywords["labels_only"].value, True)
                self.assertNotIn("parents", keywords)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
