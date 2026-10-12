"""
Tests for prompt 09a of ``prompts/datastore-integrity`` (board item K4), the parts that are not the
factory or the inventory (``ObjectFactories/tests/test_quadsource_parent_key.py`` holds those): a
``QuadSource`` is keyed on the two ``TkNumericIntegration`` rows it is computed from.

  4. ``TestTheAssertion`` -- a ``QuadSource`` made to be computed carries the two serials it is keyed
     under, from its construction, and ``compute()`` raises, naming both pairs, when it is handed
     other rows. Charter §2: a derived row is keyed on the rows its ``compute()`` receives, and
     nothing checked that. ``q == r`` is the diagonal, where one row is handed twice.
  8. ``TestMainPyResolvesTheParents`` -- ``main.py``'s two ``QuadSource`` stages, read with ``ast``
     and never imported (it parses ``sys.argv`` and opens Ray at module scope): the ``QuadSource``
     stage looks the two Tk rows up, serial-only, before its existence query, and passes their
     serials to it and to the construction; the ``QuadSourceIntegral`` stage looks the Tk rows up
     before its ``QuadSource`` lookup and keys that lookup on them.
  * ``TestTheReaders`` -- the three ``QuadSource`` readers hold only physical labels and ask for the
     reader mode, which refuses more than one stored match.

No Ray, nothing under ``var/``.
"""

import ast
import unittest
from pathlib import Path
from unittest import mock

import ray

from ComputeTargets import QuadSource
from ComputeTargets.tests.test_main_plumbing import MAIN_PY

REPO_ROOT = Path(__file__).resolve().parents[2]
READERS = (
    REPO_ROOT / "extract_tensor_source_data.py",
    REPO_ROOT / "extract_QuadSourceIntegral_data.py",
    REPO_ROOT
    / "docs"
    / "source-remediation-verification"
    / "run_quadsource_integrals.py",
)


class _Row:
    """Stands in for a TkNumericIntegration, as far as compute()'s assertion goes."""

    def __init__(self, store_id, available=True):
        self.store_id = store_id
        self.available = available
        self.model_proxy = object()


def _source(Tq_serial=5, Tr_serial=6, **kwargs):
    return QuadSource(
        payload=None,
        model=object(),
        z_sample=None,
        q=object(),
        r=object(),
        Tq_serial=Tq_serial,
        Tr_serial=Tr_serial,
        **kwargs,
    )


def _compute(source, Tq, Tr):
    """compute(), with the Ray task and the cosmology check stood in for."""
    task = mock.MagicMock()
    task.remote.return_value = "a Ray reference"
    with mock.patch("ComputeTargets.QuadSource.compute_quad_source", task):
        with mock.patch("ComputeTargets.QuadSource.check_cosmology"):
            return source.compute(payload={"Tq": Tq, "Tr": Tr}), task


class TestTheConstructor(unittest.TestCase):
    def test_it_carries_the_serials_it_is_keyed_under(self):
        source = _source(5, 6)
        self.assertEqual((source.Tq_serial, source.Tr_serial), (5, 6))

    def test_an_object_to_be_computed_names_both(self):
        for kwargs in ({}, {"Tq_serial": 5}, {"Tr_serial": 6}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as caught:
                    QuadSource(
                        payload=None,
                        model=object(),
                        z_sample=None,
                        q=object(),
                        r=object(),
                        **kwargs,
                    )
                self.assertIn("Tq_serial", str(caught.exception))

    def test_a_labels_only_miss_names_neither_and_cannot_be_computed(self):
        miss = QuadSource(
            payload=None,
            model=object(),
            z_sample=None,
            q=object(),
            r=object(),
            labels_only=True,
        )
        self.assertIsNone(miss.Tq_serial)
        self.assertIsNone(miss.Tr_serial)
        with self.assertRaises(RuntimeError) as caught:
            _compute(miss, _Row(5), _Row(6))
        self.assertIn("labels-only", str(caught.exception))

    def test_a_labels_only_miss_given_a_parent_is_an_error(self):
        with self.assertRaises(ValueError):
            _source(labels_only=True)

    def test_a_stored_object_carries_the_serials_the_row_records(self):
        stored = QuadSource(
            payload={
                "store_id": 9,
                "compute_time": 1.0,
                "values": [],
                "Tq_serial": 5,
                "Tr_serial": 6,
            },
            model=object(),
            z_sample=None,
            q=object(),
            r=object(),
        )
        self.assertEqual((stored.Tq_serial, stored.Tr_serial), (5, 6))


class TestTheAssertion(unittest.TestCase):
    """§3 test 4."""

    def test_the_rows_it_is_keyed_on_are_accepted(self):
        source = _source(5, 6)
        ref, task = _compute(source, _Row(5), _Row(6))
        self.assertEqual(ref, "a Ray reference")
        task.remote.assert_called_once()

    def test_the_serials_are_not_replaced_by_the_rows_it_is_handed(self):
        # compute() used to overwrite them with the rows' own serials, so that the row store()
        # wrote recorded whatever it was handed and not what it was keyed under
        source = _source(5, 6)
        _compute(source, _Row(5), _Row(6))
        self.assertEqual((source.Tq_serial, source.Tr_serial), (5, 6))

    def test_another_tq_raises_naming_both_serials(self):
        source = _source(5, 6)
        with self.assertRaises(RuntimeError) as caught:
            _compute(source, _Row(7), _Row(6))
        message = str(caught.exception)
        self.assertIn("Tq store_id=7 but Tq_serial=5", message)
        self.assertIn("Tr store_id=6 but Tr_serial=6", message)

    def test_another_tr_raises_naming_both_serials(self):
        source = _source(5, 6)
        with self.assertRaises(RuntimeError) as caught:
            _compute(source, _Row(5), _Row(8))
        message = str(caught.exception)
        self.assertIn("Tq store_id=5 but Tq_serial=5", message)
        self.assertIn("Tr store_id=8 but Tr_serial=6", message)

    def test_swapped_rows_raise(self):
        with self.assertRaises(RuntimeError):
            _compute(_source(5, 6), _Row(6), _Row(5))

    def test_nothing_is_computed_when_it_raises(self):
        source = _source(5, 6)
        task = mock.MagicMock()
        with mock.patch("ComputeTargets.QuadSource.compute_quad_source", task):
            with mock.patch("ComputeTargets.QuadSource.check_cosmology"):
                with self.assertRaises(RuntimeError):
                    source.compute(payload={"Tq": _Row(7), "Tr": _Row(6)})
        task.remote.assert_not_called()

    def test_an_unavailable_row_is_still_its_own_error(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_source(5, 6), _Row(5, available=False), _Row(6))
        self.assertIn("not available", str(caught.exception))


class TestQEqualsR(unittest.TestCase):
    """§3 test 5, the assertion's half. On the diagonal ``Tq`` and ``Tr`` are the same row, and
    ``main.py`` hands ``compute()`` that one object twice."""

    def test_the_one_row_handed_twice_is_accepted(self):
        row = _Row(5)
        ref, task = _compute(_source(5, 5), row, row)
        self.assertEqual(ref, "a Ray reference")

    def test_a_diagonal_source_refuses_a_different_row_for_r(self):
        with self.assertRaises(RuntimeError) as caught:
            _compute(_source(5, 5), _Row(5), _Row(6))
        self.assertIn("Tr store_id=6 but Tr_serial=5", str(caught.exception))

    def test_an_off_diagonal_source_refuses_the_diagonal(self):
        row = _Row(5)
        with self.assertRaises(RuntimeError):
            _compute(_source(5, 6), row, row)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree, name):
    (found,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _lookups(scope):
    """``(lineno, method, class name, call)`` for each ``pool.object_get`` and
    ``pool.object_get_vectorized`` in ``scope`` naming its class as a string, in source order.
    """
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


def _dicts_with_key(scope, key):
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == key for k in node.keys)
    ]


def _dict_value(node: ast.Dict, key):
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    raise KeyError(key)


class TestMainPyResolvesTheParents(unittest.TestCase):
    """§3 test 8."""

    @classmethod
    def setUpClass(cls):
        tree = _tree(MAIN_PY)
        cls.stage = _function(tree, "build_tensor_source_work")
        cls.qsi = _function(tree, "build_QuadSourceIntegral_batch")

    def test_the_quadsource_stage_resolves_the_tk_rows_before_its_existence_query(self):
        order = [(method, name) for _, method, name, _ in _lookups(self.stage)]
        self.assertEqual(
            order,
            [
                ("object_get", "TkNumericIntegration"),  # serial-only, for every (q, r)
                ("object_get_vectorized", "QuadSource"),  # the existence query
                (
                    "object_get",
                    "TkNumericIntegration",
                ),  # the full rows, missing items only
                ("object_get", "QuadSource"),  # the object to be computed
            ],
        )

    def test_the_first_tk_lookup_is_serial_only_and_the_second_is_not(self):
        (first, second) = [
            call
            for _, _, name, call in _lookups(self.stage)
            if name == "TkNumericIntegration"
        ]
        # each is dispatched over a batch of payload dicts: the first batch's dicts carry
        # _do_not_populate, and the second's do not
        with_flag = [
            d
            for d in _dicts_with_key(self.stage, "_do_not_populate")
            if isinstance(_dict_value(d, "_do_not_populate"), ast.Constant)
            and _dict_value(d, "_do_not_populate").value is True
            and any(isinstance(k, ast.Constant) and k.value == "k" for k in d.keys)
        ]
        self.assertEqual(len(with_flag), 1, "one Tk payload asks for serial-only")
        without_flag = [
            d
            for d in _dicts_with_key(self.stage, "k")
            if not any(
                isinstance(k, ast.Constant) and k.value == "_do_not_populate"
                for k in d.keys
            )
            and any(isinstance(k, ast.Constant) and k.value == "atol" for k in d.keys)
        ]
        self.assertEqual(len(without_flag), 1, "one Tk payload asks for the full row")
        self.assertLess(with_flag[0].lineno, without_flag[0].lineno)

    def test_the_existence_query_names_both_parents(self):
        (payload,) = _dicts_with_key(self.stage, "Tq_serial")
        self.assertIn(
            "Tr_serial",
            [k.value for k in payload.keys if isinstance(k, ast.Constant)],
        )
        # by the serial the Tk lookup resolved, never a literal or a missing value
        for key in ("Tq_serial", "Tr_serial"):
            self.assertIsInstance(_dict_value(payload, key), ast.Subscript)
        vectorized = [
            call
            for _, method, name, call in _lookups(self.stage)
            if name == "QuadSource" and method == "object_get_vectorized"
        ]
        self.assertEqual(len(vectorized), 1)
        self.assertLess(
            [c for _, _, n, c in _lookups(self.stage) if n == "TkNumericIntegration"][
                0
            ].lineno,
            payload.lineno,
        )

    def test_the_construction_names_both_parents(self):
        (construction,) = [
            call
            for _, method, name, call in _lookups(self.stage)
            if name == "QuadSource" and method == "object_get"
        ]
        keywords = {kw.arg: kw.value for kw in construction.keywords}
        for key in ("Tq_serial", "Tr_serial"):
            self.assertIn(key, keywords)
            self.assertIsInstance(keywords[key], ast.Attribute)
            self.assertEqual(keywords[key].attr, "store_id")

    def test_the_stage_does_not_ask_for_the_reader_mode(self):
        self.assertNotIn("labels_only", ast.dump(self.stage))

    def test_the_source_integral_stage_looks_the_tk_rows_up_before_its_quadsource_lookup(
        self,
    ):
        order = [
            (method, name)
            for _, method, name, _ in _lookups(self.qsi)
            if name in ("QuadSource", "TkNumericIntegration")
        ]
        self.assertEqual(
            order,
            [
                ("object_get", "TkNumericIntegration"),
                ("object_get_vectorized", "QuadSource"),
            ],
        )

    def test_the_source_integral_stage_keys_its_quadsource_lookup_on_them(self):
        (payload,) = _dicts_with_key(self.qsi, "Tq_serial")
        self.assertIn(
            "Tr_serial", [k.value for k in payload.keys if isinstance(k, ast.Constant)]
        )
        for key in ("Tq_serial", "Tr_serial"):
            self.assertIsInstance(_dict_value(payload, key), ast.Subscript)
        self.assertNotIn("labels_only", ast.dump(self.qsi))

    def test_the_source_integral_stage_still_loads_full_tk_rows(self):
        # TkSourceFunctions reads the stored sample values, so the lookup that moved is not
        # serial-only
        for d in _dicts_with_key(self.qsi, "atol"):
            if any(isinstance(k, ast.Constant) and k.value == "k" for k in d.keys):
                self.assertNotIn(
                    "_do_not_populate",
                    [k.value for k in d.keys if isinstance(k, ast.Constant)],
                )

    def test_the_source_integral_stage_resolves_the_serials_after_checking_availability(
        self,
    ):
        # an unavailable Tk row has no serial to key on, so the check that stops the stage for it
        # comes before the QuadSource lookup that would need it
        source = MAIN_PY.read_text().splitlines()
        lines = {
            "check": next(
                i
                for i, text in enumerate(source, start=1)
                if i >= self.qsi.lineno and "Tk_numeric_unavailable = [" in text
            ),
            "quadsource": next(
                i
                for i, text in enumerate(source, start=1)
                if i >= self.qsi.lineno and "QuadSource_lookup_batch = [" in text
            ),
        }
        self.assertLess(lines["check"], lines["quadsource"])


class TestTheReaders(unittest.TestCase):
    """The readers hold only physical labels: they ask for the reader mode."""

    def test_each_quadsource_lookup_asks_for_labels_only(self):
        for path in READERS:
            with self.subTest(path=path.name):
                calls = [
                    node
                    for node in ast.walk(_tree(path))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "object_get"
                    and len(node.args) > 0
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "QuadSource"
                ]
                self.assertEqual(len(calls), 1)
                keywords = {kw.arg: kw.value for kw in calls[0].keywords}
                self.assertIn("labels_only", keywords)
                self.assertIsInstance(keywords["labels_only"], ast.Constant)
                self.assertIs(keywords["labels_only"].value, True)
                self.assertNotIn("Tq_serial", keywords)
                self.assertNotIn("Tr_serial", keywords)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
