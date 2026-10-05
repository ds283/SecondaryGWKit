"""
Tests for prompt 09c of ``prompts/datastore-integrity`` (board item K5), the parts that are not the
factory or the inventory (``Datastore/tests/test_quadsource_integral_parent_key.py`` holds those):
a ``QuadSourceIntegral`` is keyed on the six stored rows it is computed from -- its
``QuadSource``, its ``GkSourcePolicyData``, and the Tk numeric and WKB rows for q and r.

  * ``TestTheConstructor`` -- an object to be computed names all six from construction; a stored
    object takes them from its row.
  2. ``TestTheAssertion`` -- ``compute()`` raises, naming both sides, unless the six rows it is
     handed are the six it is keyed on, the Tq and Tr numeric rows are its source's ``Tq_serial``
     and ``Tr_serial``, and each WKB row's ``numeric_serial`` is its numeric row. Nothing is
     computed when it raises.
  4. ``TestTheExtractionScript`` -- ``extract_QuadSourceIntegral_data.py`` selects by physical
     labels (D1): with one row per z_response it uses them without being given any parent; with
     two at one z_response it refuses, naming the parents that distinguish them. The rows are
     real ``read_batch`` objects.
  7. ``TestMainPy`` -- ``main.py``, read with ``ast`` and never imported: the
     ``QuadSourceIntegral`` stage resolves the six parents, serial-only, before its existence
     query, which is keyed on them, as is the object to be computed; the work-list lines are
     unchanged.
  * ``TestThePipelinesOwnLogic`` -- the prompt's first stop condition, on a store:
    ``main.py``'s own ``resolve_QuadSourceIntegral_parents``, compiled from its source and run
    against the real factories, finds each item's six parents by their own parents, and the rows
    it finds satisfy the relations ``compute()`` asserts.

No Ray, nothing under ``var/``: the stores are in memory.
"""

import ast
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ray
import sqlalchemy as sqla

from ComputeTargets import QuadSourceIntegral
from ComputeTargets.tests.test_main_plumbing import MAIN_PY
from config.datastore import factories as _factories
from Datastore.SQL.ObjectFactories.GkSource import GkSource_parents_digest
from Datastore.tests.test_quadsource_integral_parent_key import (
    OTHER,
    PARENTS,
    _WithAStore,
)
from Datastore.tests.test_wkb_numeric_parent_key import (
    K_EXIT,
    MODEL,
    OTHER_K_EXIT,
    RTOL,
    TOL,
    _base_store,
    _Ref,
)
import ComputeTargets.phase_residual as phase_residual
from CosmologyConcepts import redshift

# the module itself: `import ComputeTargets.QuadSourceIntegral as ...` binds the package's
# re-exported class of the same name, not the module that holds compute_QuadSource_integral
qsi_module = importlib.import_module("ComputeTargets.QuadSourceIntegral")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_QSI = REPO_ROOT / "extract_QuadSourceIntegral_data.py"

STAGE = "build_QuadSourceIntegral_batch"
HELPER = "resolve_QuadSourceIntegral_parents"

# ---------------------------------------------------------------------------------------------
# the compute target
# ---------------------------------------------------------------------------------------------


def _mode(k_serial):
    return SimpleNamespace(
        k=SimpleNamespace(store_id=k_serial, k_inv_Mpc=float(k_serial))
    )


K, Q, R = _mode(1), _mode(2), _mode(3)
Z_RESPONSE = SimpleNamespace(store_id=5, z=10.0)
Z_SOURCE_MAX = SimpleNamespace(store_id=6, z=1.0e5)
TOLERANCE = SimpleNamespace(store_id=1, tol=1.0e-8)
GK_POLICY = SimpleNamespace(store_id=9)


def _integral(r=R, **parents):
    return QuadSourceIntegral(
        None,
        model=None,
        policy=GK_POLICY,
        quad_policy=SimpleNamespace(store_id=4),
        z_response=Z_RESPONSE,
        z_source_max=Z_SOURCE_MAX,
        k=K,
        q=Q,
        r=r,
        atol=TOLERANCE,
        rtol=TOLERANCE,
        **dict(PARENTS, **parents),
    )


def _rows(
    Tq_serial=None,
    Tr_serial=None,
    Tq_WKB_numeric=None,
    Tr_WKB_numeric=None,
    **store_ids,
):
    """The payload compute() is handed: rows whose serials are PARENTS', and related as the
    pipeline relates them, unless an argument says otherwise."""
    ids = dict(
        source=PARENTS["source_serial"],
        data=PARENTS["data_serial"],
        Tq_numeric=PARENTS["Tq_numeric_serial"],
        Tq_WKB=PARENTS["Tq_WKB_serial"],
        Tr_numeric=PARENTS["Tr_numeric_serial"],
        Tr_WKB=PARENTS["Tr_WKB_serial"],
    )
    ids.update(store_ids)
    sample = SimpleNamespace(max=SimpleNamespace(z=2.0e5))
    Gk = SimpleNamespace(z_response=Z_RESPONSE, z_sample=sample, k=K.k)
    return {
        "source": SimpleNamespace(
            store_id=ids["source"],
            q=Q.k,
            r=R.k,
            z_sample=sample,
            Tq_serial=ids["Tq_numeric"] if Tq_serial is None else Tq_serial,
            Tr_serial=ids["Tr_numeric"] if Tr_serial is None else Tr_serial,
        ),
        "GkPolicy": SimpleNamespace(
            store_id=ids["data"],
            policy=GK_POLICY,
            _source_proxy=SimpleNamespace(get=lambda: Gk),
        ),
        "b": 0.0,
        "Bessel_0pt5": None,
        "Bessel_2pt5": None,
        "Tq_numeric": SimpleNamespace(store_id=ids["Tq_numeric"], k=Q.k),
        "Tq_WKB": SimpleNamespace(
            store_id=ids["Tq_WKB"],
            k=Q.k,
            numeric_serial=(
                ids["Tq_numeric"] if Tq_WKB_numeric is None else Tq_WKB_numeric
            ),
        ),
        "Tr_numeric": SimpleNamespace(store_id=ids["Tr_numeric"], k=R.k),
        "Tr_WKB": SimpleNamespace(
            store_id=ids["Tr_WKB"],
            k=R.k,
            numeric_serial=(
                ids["Tr_numeric"] if Tr_WKB_numeric is None else Tr_WKB_numeric
            ),
        ),
    }


def _compute(integral, payload):
    """compute(), with the Ray task stood in for. Returns (the reference, the task)."""
    task = mock.MagicMock()
    task.remote.return_value = "a Ray reference"
    with mock.patch.object(qsi_module, "compute_QuadSource_integral", task):
        return integral.compute(payload=payload), task


class TestTheConstructor(unittest.TestCase):
    def test_an_object_to_be_computed_names_all_six(self):
        for name in QuadSourceIntegral.PARENTS:
            with self.subTest(parent=name):
                with self.assertRaises(ValueError) as caught:
                    _integral(**{name: None})
                self.assertIn(name, str(caught.exception))

    def test_it_carries_them_from_construction(self):
        integral = _integral()
        self.assertEqual(integral.parents, PARENTS)
        self.assertEqual(integral.Tq_numeric_serial, PARENTS["Tq_numeric_serial"])
        self.assertEqual(integral.Tr_WKB_serial, PARENTS["Tr_WKB_serial"])

    def test_a_stored_object_takes_them_from_its_row(self):
        (obj,) = _StoredRows.read()
        self.assertEqual(obj.parents, PARENTS)

    def test_store_does_not_replace_them(self):
        """The target's store() used to overwrite the source and data serials with what the
        computation reported; they are the key, fixed at construction (prompt 09c)."""
        integral = _integral()
        integral._compute_ref = "a Ray reference"
        result = {
            "total": 1.0,
            "total_abserr": None,
            "total_converged": None,
            "total_phase_limited": None,
            "b": 0.0,
            "numeric_quad": 0.25,
            "WKB_Levin": 0.75,
            "eta_source_max": None,
            "eta_response": None,
            "analytic_rad": None,
            "GkPolicy_serial": 999,
            "source_serial": 998,
            "numeric_quad_data": None,
            "WKB_Levin_data": None,
            "WKB_phase_spline_chunks": None,
            "compute_time": None,
            "analytic_compute_time": None,
            "metadata": {},
        }
        with mock.patch.object(
            qsi_module.ray, "wait", return_value=(["a Ray reference"], [])
        ), mock.patch.object(qsi_module.ray, "get", return_value=result):
            integral.store()
        self.assertEqual(integral.parents, PARENTS)


class TestTheAssertion(unittest.TestCase):
    """§3 test 2."""

    def test_the_rows_it_is_keyed_on_are_accepted(self):
        ref, task = _compute(_integral(), _rows())
        self.assertEqual(ref, "a Ray reference")
        task.remote.assert_called_once()

    def test_another_row_for_any_parent_raises_naming_both(self):
        for field, name in (
            ("source", "source_serial"),
            ("data", "data_serial"),
            ("Tq_numeric", "Tq_numeric_serial"),
            ("Tq_WKB", "Tq_WKB_serial"),
            ("Tr_numeric", "Tr_numeric_serial"),
            ("Tr_WKB", "Tr_WKB_serial"),
        ):
            with self.subTest(parent=name):
                with self.assertRaises(RuntimeError) as caught:
                    _compute(_integral(), _rows(**{field: OTHER[name]}))
                message = str(caught.exception)
                self.assertIn(f"{name}={PARENTS[name]}", message)
                self.assertIn(f"store_id={OTHER[name]}", message)

    def test_a_tq_numeric_row_that_is_not_its_sources_tq_serial_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_integral(), _rows(Tq_serial=77))
        message = str(caught.exception)
        self.assertIn("Tq numeric row", message)
        self.assertIn(f"store_id={PARENTS['Tq_numeric_serial']}", message)
        self.assertIn("Tq_serial=77", message)

    def test_a_tr_numeric_row_that_is_not_its_sources_tr_serial_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_integral(), _rows(Tr_serial=78))
        message = str(caught.exception)
        self.assertIn("Tr numeric row", message)
        self.assertIn("Tr_serial=78", message)

    def test_a_tq_wkb_row_that_does_not_start_from_the_tq_numeric_row_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_integral(), _rows(Tq_WKB_numeric=79))
        message = str(caught.exception)
        self.assertIn("Tq WKB row", message)
        self.assertIn("numeric_serial=79", message)
        self.assertIn(f"store_id={PARENTS['Tq_numeric_serial']}", message)

    def test_a_tr_wkb_row_that_does_not_start_from_the_tr_numeric_row_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_integral(), _rows(Tr_WKB_numeric=80))
        message = str(caught.exception)
        self.assertIn("Tr WKB row", message)
        self.assertIn("numeric_serial=80", message)

    def test_nothing_is_computed_when_it_raises(self):
        for payload in (
            _rows(Tq_numeric=OTHER["Tq_numeric_serial"]),
            _rows(Tq_serial=77),
            _rows(Tr_WKB_numeric=80),
        ):
            task = mock.MagicMock()
            with mock.patch.object(qsi_module, "compute_QuadSource_integral", task):
                with self.assertRaises(RuntimeError):
                    _integral().compute(payload=payload)
            task.remote.assert_not_called()

    def test_the_diagonal_hands_one_row_twice(self):
        """q == r: the Tk rows for q and r are one pair of rows, handed twice."""
        integral = _integral(r=Q, Tr_numeric_serial=31, Tr_WKB_serial=41)
        payload = _rows(Tr_numeric=31, Tr_WKB=41)
        payload["source"].r = Q.k
        payload["Tr_numeric"] = payload["Tq_numeric"]
        payload["Tr_WKB"] = payload["Tq_WKB"]
        ref, _ = _compute(integral, payload)
        self.assertEqual(ref, "a Ray reference")

    def test_the_checks_that_were_there_still_run_first(self):
        payload = _rows()
        payload["Tq_numeric"] = SimpleNamespace(store_id=99, k=R.k)
        with self.assertRaises(RuntimeError) as caught:
            _compute(_integral(), payload)
        self.assertIn("does not match the required value", str(caught.exception))


# ---------------------------------------------------------------------------------------------
# the extraction script
# ---------------------------------------------------------------------------------------------


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _module_function(path: Path, name: str, namespace: dict):
    (node,) = [
        n for n in _tree(path).body if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class _StoredRows(_WithAStore):
    """Rows stored through the real factory and read back by the real ``read_batch``, as the
    extraction script reads them: by physical labels, naming no parent. Not a test case itself.
    """

    @classmethod
    def read(cls, rows=((PARENTS, 1),)):
        case = cls()
        case.setUp()
        try:
            for parents, z_response_serial in rows:
                stored = case.store_row(parents=parents)
                table = case.table()
                case.s.conn.execute(
                    sqla.update(table)
                    .where(table.c.serial == stored.store_id)
                    .values(z_response_serial=z_response_serial)
                )
            case.s.conn.commit()
            return case.read_batch()
        finally:
            case.doCleanups()


class TestTheExtractionScript(unittest.TestCase):
    """§3 test 4."""

    @classmethod
    def setUpClass(cls):
        cls.refuse = staticmethod(
            _module_function(
                EXTRACT_QSI,
                "refuse_ambiguous_QuadSourceIntegrals",
                {"QuadSourceIntegral": QuadSourceIntegral},
            )
        )

    def test_one_row_per_z_response_is_used_as_it_is(self):
        rows = _StoredRows.read(((PARENTS, 1), (OTHER, 2)))
        self.assertEqual(len(rows), 2)
        self.assertIs(self.refuse(rows), rows)

    def test_two_rows_at_one_z_response_are_refused_naming_what_distinguishes_them(
        self,
    ):
        other = dict(PARENTS, Tq_WKB_serial=OTHER["Tq_WKB_serial"])
        rows = _StoredRows.read(((PARENTS, 1), (other, 1)))
        with self.assertRaises(RuntimeError) as caught:
            self.refuse(rows)
        message = str(caught.exception)
        self.assertIn("2 stored QuadSourceIntegral rows", message)
        self.assertIn("They differ in Tq_WKB_serial", message)
        for obj in rows:
            self.assertIn(
                f"serial={obj.store_id}: Tq_WKB_serial={obj.parents['Tq_WKB_serial']}",
                message,
            )
        # the parents they share are not named as distinguishing them
        self.assertNotIn("source_serial=", message)

    def test_every_differing_parent_is_named(self):
        rows = _StoredRows.read(((PARENTS, 1), (OTHER, 1)))
        with self.assertRaises(RuntimeError) as caught:
            self.refuse(rows)
        message = str(caught.exception)
        for name in QuadSourceIntegral.PARENTS:
            self.assertIn(name, message)

    def test_no_rows_is_nothing_to_refuse(self):
        self.assertEqual(self.refuse([]), [])

    def test_the_plot_refuses_before_it_uses_the_rows(self):
        (plot,) = [
            n
            for n in _tree(EXTRACT_QSI).body
            if isinstance(n, ast.FunctionDef) and n.name == "plot_QuadSourceIntegral"
        ]
        first = plot.body[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertEqual(
            ast.unparse(first),
            "integral_data = refuse_ambiguous_QuadSourceIntegrals(integral_data)",
        )

    def test_the_read_names_no_parent(self):
        calls = [
            node
            for node in ast.walk(_tree(EXTRACT_QSI))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "object_read_batch"
            and any(
                kw.arg == "ObjectClass"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "QuadSourceIntegral"
                for kw in node.keywords
            )
        ]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg for kw in calls[0].keywords}
        self.assertEqual(keywords & set(QuadSourceIntegral.PARENTS), set())


# ---------------------------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------------------------


def _run_pipeline_function(name: str) -> ast.FunctionDef:
    (run_pipeline,) = [
        n
        for n in _tree(MAIN_PY).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline"
    ]
    (found,) = [
        n
        for n in ast.walk(run_pipeline)
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    return found


def _lookups(scope):
    """``(lineno, method, class name, call)`` for each object_get / object_get_vectorized in
    ``scope`` naming its class as a string, in source order."""
    out = []
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("object_get", "object_get_vectorized")
            and len(node.args) > 0
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.append((node.lineno, node.func.attr, node.args[0].value, node))
    return sorted(out, key=lambda item: item[0])


def _calls_of(scope, name):
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _keys(node: ast.Dict):
    return [k.value if isinstance(k, ast.Constant) else None for k in node.keys]


# the work list as it stood before this prompt (9a18063), which the sweep script's BATCH_LINE
# substitution filters: it must be kept exactly
WORK_LIST = """        qsi_work_items = itertools.product(
            z_source_integral_response_sample,
            itertools.combinations_with_replacement(source_k_exit_times, 2),
            response_k_exit_times,
        )
        qsi_work_items = [(z, k, q, r) for z, (q, r), k in qsi_work_items]

        # a (k, q, r) triple has no theta in [0, pi] realizing r = |k - q| unless it closes a
        # triangle (spec 03 Sec 0.3); computing QuadSourceIntegral for such a triple is pure waste
        num_before_triangle_filter = len(qsi_work_items)
        qsi_work_items = [
            (z, k, q, r) for z, k, q, r in qsi_work_items if closes_triangle(k, q, r)
        ]
        num_after_triangle_filter = len(qsi_work_items)
"""
BATCH_LINE = '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))\n'


class TestMainPy(unittest.TestCase):
    """§3 test 7. ``main.py`` is read with ``ast``, never imported."""

    @classmethod
    def setUpClass(cls):
        cls.stage = _run_pipeline_function(STAGE)
        cls.helper = _run_pipeline_function(HELPER)

    def existence_query(self):
        (query,) = [
            call
            for _, method, name, call in _lookups(self.stage)
            if name == "QuadSourceIntegral" and method == "object_get_vectorized"
        ]
        return query

    def test_the_stage_resolves_the_parents_before_its_existence_query(self):
        (resolve_gk,) = _calls_of(self.stage, "resolve_GkSource_parents")
        (resolve,) = _calls_of(self.stage, HELPER)
        query = self.existence_query()
        self.assertLess(resolve_gk.lineno, resolve.lineno)
        self.assertLess(resolve.lineno, query.lineno)
        # the helper is handed the whole batch, not only its missing items, and the set the
        # GkSource stage's helper resolved
        self.assertEqual(
            [ast.unparse(a) for a in resolve.args], ["batch", "GkSource_parents"]
        )
        # and every other lookup of the stage, of the missing items' full parents, follows it
        others = [
            call
            for _, _, name, call in _lookups(self.stage)
            if name != "QuadSourceIntegral"
        ]
        self.assertGreater(len(others), 0)
        for call in others:
            self.assertGreater(call.lineno, query.lineno)

    def test_the_existence_query_is_keyed_on_them(self):
        (payload,) = [
            node
            for node in ast.walk(self.stage)
            if isinstance(node, ast.Dict)
            and {"quad_policy", "z_response", "z_source_max"} <= set(_keys(node))
        ]
        spread = [v for k, v in zip(payload.keys, payload.values) if k is None]
        self.assertEqual(len(spread), 1)
        self.assertEqual(
            ast.unparse(spread[0]),
            "QuadSourceIntegral_parents[z_response.store_id, k.store_id, q.store_id, r.store_id]",
        )

    def test_the_object_to_be_computed_is_keyed_on_the_same_parents(self):
        (construction,) = [
            call
            for _, method, name, call in _lookups(self.stage)
            if name == "QuadSourceIntegral" and method == "object_get"
        ]
        spread = [kw.value for kw in construction.keywords if kw.arg is None]
        self.assertEqual([ast.unparse(v) for v in spread], ["parents"])
        (assigned,) = [
            node
            for node in ast.walk(self.stage)
            if isinstance(node, ast.Assign)
            and [ast.unparse(t) for t in node.targets] == ["parents"]
        ]
        self.assertEqual(
            ast.unparse(assigned.value),
            "QuadSourceIntegral_parents[z_response.store_id, k.store_id, q.store_id, r.store_id]",
        )

    def test_the_helper_resolves_each_parent_serial_only_by_its_own_parents(self):
        order = [(method, name) for _, method, name, _ in _lookups(self.helper)]
        self.assertEqual(
            order,
            [
                ("object_get", "TkNumericIntegration"),
                ("object_get", "TkWKBIntegration"),
                ("object_get_vectorized", "QuadSource"),
                ("object_get_vectorized", "GkSource"),
                ("object_get_vectorized", "GkSourcePolicyData"),
            ],
        )
        flagged = [
            node
            for node in ast.walk(self.helper)
            if isinstance(node, ast.Dict) and "_do_not_populate" in _keys(node)
        ]
        # the four lookups that could populate: GkSourcePolicyData has no values
        self.assertEqual(len(flagged), 4)
        for node in flagged:
            value = node.values[_keys(node).index("_do_not_populate")]
            self.assertIs(value.value, True)
        payload_keys = {
            key
            for node in ast.walk(self.helper)
            if isinstance(node, ast.Dict)
            for key in _keys(node)
        }
        # each keyed on its own parents: the WKB row on its numeric row (07), the source on its
        # two (09a), the GkSource on its set (09b), the policy data on its source
        for key in ("numeric_serial", "Tq_serial", "Tr_serial", "parents", "source"):
            self.assertIn(key, payload_keys)

    def test_the_stage_does_not_ask_for_a_reader_mode(self):
        for fn in (self.stage, self.helper):
            self.assertNotIn("labels_only", ast.dump(fn))

    def test_the_work_list_lines_are_unchanged(self):
        text = MAIN_PY.read_text()
        self.assertEqual(text.count(WORK_LIST), 1)
        self.assertEqual(text.count(BATCH_LINE), 1)
        self.assertLess(text.index(WORK_LIST), text.index(BATCH_LINE))


# ---------------------------------------------------------------------------------------------
# the pipeline's own logic, on a store
# ---------------------------------------------------------------------------------------------


class _WorkPool:
    """Stands in for RayWorkPool: runs the task builder over the items, in order, and keeps the
    results, which the stand-in pool returns directly rather than as references."""

    def __init__(self, pool, items, task_builder, **kwargs):
        self.items = list(items)
        self.task_builder = task_builder

    def run(self):
        self.results = [self.task_builder(item) for item in self.items]


class _Pool:
    """Stands in for ShardedPool's object_get and object_get_vectorized on one in-memory shard:
    the real factory's build() runs, and each call is recorded."""

    def __init__(self, store):
        self.store = store
        self.calls = []

    def object_get(self, cls_name, **payload):
        self.calls.append((cls_name, dict(payload)))
        return self._build(cls_name, payload)

    def object_get_vectorized(self, cls_name, shard_key, payload_data):
        out = []
        for payload in payload_data:
            payload = dict(payload, **shard_key)
            self.calls.append((cls_name, payload))
            out.append(self._build(cls_name, payload))
        return out

    def _build(self, cls_name, payload):
        s = self.store
        return _factories[cls_name].build(
            payload,
            s.conn,
            s.tables[cls_name],
            s.inserters[cls_name],
            s.tables,
            s.inserters,
        )


class _Proxy:
    """GkSourceProxy without Ray: what the GkSourcePolicyData lookup reads of its source."""

    def __init__(self, obj):
        self.store_id = obj.store_id if obj.available else None
        self.available = self.store_id is not None


Z_RESPONSE_ROW = redshift(store_id=2, z=10.0, is_source=False, is_response=True)
Q_EXIT, R_EXIT = K_EXIT, OTHER_K_EXIT
# rows of this store: Tk numeric 1 (at K_EXIT) and 5 (at OTHER_K_EXIT), each with a Tk WKB row
# starting from it, 7 and 8; QuadSource 61 from Tk numeric 1 and 5; GkSource 71 with its parent
# set, and GkSourcePolicyData 81 from it
TK_NUMERIC_R = 5
TK_WKB = {1: 7, TK_NUMERIC_R: 8}
QUADSOURCE = 61
GKSOURCE = 71
GK_POLICY_DATA = 81
GK_SET = ((1, None, 91),)
EXPECTED = dict(
    source_serial=QUADSOURCE,
    data_serial=GK_POLICY_DATA,
    Tq_numeric_serial=1,
    Tq_WKB_serial=7,
    Tr_numeric_serial=TK_NUMERIC_R,
    Tr_WKB_serial=8,
)


def _pipeline_store():
    """07's base store (Tk numeric 1 and 2 at K_EXIT, differing in their atol), with the rest of
    one QuadSourceIntegral's parents: a Tk numeric row at OTHER_K_EXIT, a Tk WKB row starting from
    each Tk numeric row the lookup finds, the QuadSource computed from the two, a GkSource and its
    GkSourcePolicyData. Every row carries tag 1."""
    from ComputeTargets import TkNumericIntegration

    s = _base_store()
    s.insert(
        "TkNumericIntegration",
        serial=TK_NUMERIC_R,
        label="tknum-r",
        wavenumber_exit_serial=OTHER_K_EXIT.store_id,
        model_serial=MODEL.store_id,
        atol_serial=1,
        rtol_serial=RTOL.store_id,
        break_point_kind=TkNumericIntegration.BREAK_POINT_KIND,
        solver_serial=1,
        z_init_serial=1,
        z_min_serial=3,
        z_samples=1,
        validated=True,
    )
    for numeric in (1, 2, TK_NUMERIC_R):
        s.insert("TkNumeric_tags", integration_serial=numeric, tag_serial=1)
    for numeric, wkb in TK_WKB.items():
        s.insert(
            "TkWKBIntegration",
            serial=wkb,
            label=f"tkwkb-{wkb}",
            wavenumber_exit_serial=(
                K_EXIT.store_id if numeric == 1 else OTHER_K_EXIT.store_id
            ),
            model_serial=MODEL.store_id,
            numeric_serial=numeric,
            rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
            phase_solver_serial=1,
            friction_solver_serial=1,
            z_min_serial=3,
            z_samples=1,
            z_init=100.0,
            T_init=1.0,
            Tprime_init=0.0,
            sin_coeff=0.0,
            cos_coeff=1.0,
            validated=True,
        )
        s.insert("TkWKB_tags", wkb_serial=wkb, tag_serial=1)
    s.insert(
        "QuadSource",
        serial=QUADSOURCE,
        label="quadsource",
        model_serial=MODEL.store_id,
        q_wavenumber_exit_serial=Q_EXIT.store_id,
        r_wavenumber_exit_serial=R_EXIT.store_id,
        Tq_serial=1,
        Tr_serial=TK_NUMERIC_R,
        z_samples=0,
        validated=True,
    )
    s.insert("QuadSource_tags", parent_serial=QUADSOURCE, tag_serial=1)
    s.insert(
        "GkSource",
        serial=GKSOURCE,
        label="gksource",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        z_response_serial=Z_RESPONSE_ROW.store_id,
        parents_digest=GkSource_parents_digest(GK_SET),
        z_max_serial=1,
        z_samples=0,
        validated=True,
    )
    for z_source, numeric, wkb in GK_SET:
        s.insert(
            "GkSource_parents",
            gk_source_serial=GKSOURCE,
            z_source_serial=z_source,
            numeric_serial=numeric,
            wkb_serial=wkb,
        )
    s.insert("GkSource_tags", parent_serial=GKSOURCE, tag_serial=1)
    s.insert(
        "GkSourcePolicyData",
        serial=GK_POLICY_DATA,
        source_serial=GKSOURCE,
        policy_serial=1,
        wavenumber_exit_serial=K_EXIT.store_id,
        type=0,
        quality=0,
    )
    s.conn.commit()
    return s


def _load_helper(pool):
    """main.py's own resolve_QuadSourceIntegral_parents, compiled from its source alone, its free
    names bound to stand-ins: the pool, the work pool, the run's model, tolerances, tags and
    policy, and GkSourceProxy."""
    node = _run_pipeline_function(HELPER)
    code = compile(ast.Module(body=[node], type_ignores=[]), str(MAIN_PY), "exec")
    namespace = {
        "pool": pool,
        "RayWorkPool": _WorkPool,
        "model_proxy": MODEL,
        "Tk_numeric_atol": TOL[1],
        "Tk_numeric_rtol": RTOL,
        "GkSource_policy_1pt5": _Ref(1),
        "GkSourceProxy": _Proxy,
    }
    for tag in (
        "TkProductionTag",
        "GkProductionTag",
        "RunLabelTag",
        "SourceGridConstructionTag",
        "SourceZGridSizeTag",
        "ResponseZGridSizeTag",
        "OutsideHorizonEfoldsTag",
        "LargestSourceZTag",
        "SmallestSourceZTag",
        "ResponseSparsenessZTag",
    ):
        # the store's rows carry one tag; main.py asks for several, which here are all it
        namespace[tag] = _Ref(1)
    exec(code, namespace)
    return namespace[HELPER]


class TestThePipelinesOwnLogic(unittest.TestCase):
    """
    The prompt's first stop condition: in a store written as the pipeline writes it, each work
    item's six parents are found by main.py's own resolution, keyed on their own parents, and the
    rows found satisfy the relations compute() asserts.
    """

    ITEM = (Z_RESPONSE_ROW, K_EXIT, Q_EXIT, R_EXIT)
    KEY = (Z_RESPONSE_ROW.store_id, K_EXIT.store_id, Q_EXIT.store_id, R_EXIT.store_id)
    GK = {(K_EXIT.store_id, Z_RESPONSE_ROW.store_id): GK_SET}

    def setUp(self):
        self.s = _pipeline_store()
        self.addCleanup(self.s.close)
        self.pool = _Pool(self.s)
        self.resolve = _load_helper(self.pool)

    def test_the_six_are_resolved_serial_only(self):
        parents = self.resolve([self.ITEM], self.GK)
        self.assertEqual(parents, {self.KEY: EXPECTED})
        self.assertEqual(
            [c for c, _ in self.pool.calls],
            [
                "TkNumericIntegration",
                "TkNumericIntegration",
                "TkWKBIntegration",
                "TkWKBIntegration",
                "QuadSource",
                "GkSource",
                "GkSourcePolicyData",
            ],
        )
        for cls_name, payload in self.pool.calls:
            if cls_name != "GkSourcePolicyData":
                self.assertIs(payload.get("_do_not_populate"), True, cls_name)

    def test_each_mode_and_pair_is_looked_up_once(self):
        other_z = redshift(store_id=3, z=5.0, is_source=False, is_response=True)
        items = [self.ITEM, (other_z, K_EXIT, Q_EXIT, R_EXIT)]
        gk = {**self.GK, (K_EXIT.store_id, other_z.store_id): GK_SET}
        parents = self.resolve(items, gk)
        # no GkSource is stored at the second z_response, so that item has no parents
        self.assertEqual(parents[self.KEY], EXPECTED)
        self.assertIsNone(
            parents[
                (other_z.store_id, K_EXIT.store_id, Q_EXIT.store_id, R_EXIT.store_id)
            ]
        )
        counts = {}
        for cls_name, _ in self.pool.calls:
            counts[cls_name] = counts.get(cls_name, 0) + 1
        self.assertEqual(
            counts,
            {
                "TkNumericIntegration": 2,
                "TkWKBIntegration": 2,
                "QuadSource": 1,
                "GkSource": 2,
                "GkSourcePolicyData": 1,
            },
        )

    def test_an_item_with_a_parent_missing_has_none(self):
        # the diagonal (q, q): its Tk rows are stored, but no QuadSource from Tk 1 twice
        item = (Z_RESPONSE_ROW, K_EXIT, Q_EXIT, Q_EXIT)
        parents = self.resolve([item, self.ITEM], self.GK)
        self.assertIsNone(
            parents[
                (
                    Z_RESPONSE_ROW.store_id,
                    K_EXIT.store_id,
                    Q_EXIT.store_id,
                    Q_EXIT.store_id,
                )
            ]
        )
        self.assertEqual(parents[self.KEY], EXPECTED)

    def test_nothing_wanted_reads_nothing(self):
        self.assertEqual(self.resolve([], {}), {})
        self.assertEqual(self.pool.calls, [])

    def test_the_rows_found_satisfy_the_relations_compute_asserts(self):
        parents = self.resolve([self.ITEM], self.GK)[self.KEY]
        # the rows as the stage's own lookups find them: each by its own parents
        lookup = self.pool.object_get
        tags = [_Ref(1)]
        Tk = {}
        for k_exit in (Q_EXIT, R_EXIT):
            numeric = lookup(
                "TkNumericIntegration",
                model=MODEL,
                z_sample=None,
                k=k_exit,
                z_init=None,
                atol=TOL[1],
                rtol=RTOL,
                tags=tags,
                _do_not_populate=True,
            )
            WKB = lookup(
                "TkWKBIntegration",
                solver_labels=[],
                model=MODEL,
                z_sample=None,
                k=k_exit,
                numeric_serial=numeric.store_id,
                tags=tags,
                _do_not_populate=True,
            )
            Tk[k_exit.store_id] = (numeric, WKB)
        (source,) = self.pool.object_get_vectorized(
            "QuadSource",
            {"q": Q_EXIT},
            [
                {
                    "model": MODEL,
                    "z_sample": None,
                    "r": R_EXIT,
                    "Tq_serial": Tk[Q_EXIT.store_id][0].store_id,
                    "Tr_serial": Tk[R_EXIT.store_id][0].store_id,
                    "tags": tags,
                    "_do_not_populate": True,
                }
            ],
        )
        GkPolicy = SimpleNamespace(store_id=parents["data_serial"])
        payload = {
            "Tq_numeric": Tk[Q_EXIT.store_id][0],
            "Tq_WKB": Tk[Q_EXIT.store_id][1],
            "Tr_numeric": Tk[R_EXIT.store_id][0],
            "Tr_WKB": Tk[R_EXIT.store_id][1],
        }
        integral = QuadSourceIntegral(
            None,
            model=MODEL,
            policy=_Ref(1),
            quad_policy=_Ref(1),
            z_response=Z_RESPONSE_ROW,
            z_source_max=Z_RESPONSE_ROW,
            k=K_EXIT,
            q=Q_EXIT,
            r=R_EXIT,
            atol=TOL[1],
            rtol=RTOL,
            **parents,
        )
        # raises if the six handed in are not the six resolved, or if they do not fit together
        integral._check_the_rows_it_is_keyed_on(source, GkPolicy, payload)
        self.assertEqual(source.Tq_serial, payload["Tq_numeric"].store_id)
        self.assertEqual(
            payload["Tr_WKB"].numeric_serial, payload["Tr_numeric"].store_id
        )


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
