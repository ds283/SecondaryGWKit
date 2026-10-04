"""
The tensor-source reader asks for ``QuadSource`` in the stored order, and refuses a miss naming
its pair (prompts/a3-v2-readiness, prompt 07a, E1–E2).

A store holds ``QuadSource`` rows for q <= r only: main.py pairs its k with
``itertools.combinations_with_replacement`` over a ``wavenumber_exit_time_array``. The reader drew
its k with ``random.sample`` and paired them in the order drawn, so a q > r pair matched nothing,
and the unpopulated labels-only miss failed on ``values`` naming neither k
(``[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]``).

- ``TestQuadSourcePairs`` runs ``extract_common.quad_source_pairs`` on every permutation of three
  and of four k: one list, ``main.py``'s, q <= r, n(n+1)/2 pairs, a duplicate once.
- ``TestRequireQuadSource`` runs ``extract_common.require_quad_source`` on a labels-only miss and
  on a stored ``QuadSource``.
- ``TestTheReader`` reads ``extract_tensor_source_data.py`` with ``ast`` (it calls ``ray.init``
  at module scope and cannot be imported): ``run_pipeline`` builds ``work_grid`` from
  ``quad_source_pairs`` and calls no ``combinations_with_replacement``, and
  ``plot_tensor_source`` calls ``require_quad_source`` before it first reads ``source.values`` or
  ``source.functions``.

The exit times are real ``wavenumber_exit_time``s, built from a payload as the factory builds them,
so they order by ``k`` and hash by ``store_id`` exactly as the store's do; the cosmology is a stand-in
carrying only the units ``check_units`` compares. No Ray, no datastore, nothing under ``var/``.
"""

import ast
import itertools
import unittest
from pathlib import Path

import ray

from ComputeTargets import QuadSource
from CosmologyConcepts import wavenumber, wavenumber_exit_time
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
    wavenumber_exit_time_array,
)
from Units import Mpc_units
from extract_common import quad_source_pairs, require_quad_source

REPO_ROOT = Path(__file__).resolve().parents[2]
READER = REPO_ROOT / "extract_tensor_source_data.py"

UNITS = Mpc_units()

# k in 1/Mpc, deliberately not in store_id order, so that an ordering by store_id is not
# mistaken for an ordering by k
K_INV_MPC = {11: 1.0e6, 12: 1.0e5, 13: 3.1622776601683794e5, 14: 2.0e5}


class _Cosmology:
    """Stands in for a cosmology, as far as ``wavenumber_exit_time``'s ``check_units`` goes."""

    units = UNITS


def _exit_time(store_id: int, k_inv_Mpc: float) -> wavenumber_exit_time:
    """A stored ``wavenumber_exit_time``, built as the factory builds one from its row."""
    payload = {
        "store_id": store_id,
        "z_exit": 1.0e4,
        "compute_time": 0.0,
        "stepping": 0,
    }
    for offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{offset}"] = 1.0e4
    for offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{offset}"] = 1.0e4
    return wavenumber_exit_time(
        payload=payload,
        # the wavenumber's serial is offset from the exit time's, as in a store
        k=wavenumber(store_id + 100, k_inv_Mpc, UNITS, is_source=True),
        cosmology=_Cosmology(),
        atol=None,
        rtol=None,
    )


def _exit_times(n: int):
    return [_exit_time(sid, K_INV_MPC[sid]) for sid in list(K_INV_MPC)[:n]]


def _ks(pairs):
    return [(q.k.k_inv_Mpc, r.k.k_inv_Mpc) for q, r in pairs]


class TestQuadSourcePairs(unittest.TestCase):
    """E1, run."""

    def test_every_order_gives_mains_pairs(self):
        for n in (3, 4):
            items = _exit_times(n)
            expected = list(
                itertools.combinations_with_replacement(
                    wavenumber_exit_time_array(items), 2
                )
            )
            for order in itertools.permutations(items):
                with self.subTest(n=n, order=[x.k.k_inv_Mpc for x in order]):
                    pairs = quad_source_pairs(order)
                    self.assertEqual(pairs, expected)
                    self.assertEqual(_ks(pairs), _ks(expected))

    def test_every_pair_has_q_le_r_and_there_are_n_n_plus_1_over_2(self):
        for n in (3, 4):
            for order in itertools.permutations(_exit_times(n)):
                with self.subTest(n=n, order=[x.k.k_inv_Mpc for x in order]):
                    pairs = quad_source_pairs(order)
                    self.assertEqual(len(pairs), n * (n + 1) // 2)
                    for q, r in pairs:
                        self.assertLessEqual(q.k.k_inv_Mpc, r.k.k_inv_Mpc)

    def test_the_pairs_are_ascending_in_k(self):
        k = sorted(K_INV_MPC[sid] for sid in list(K_INV_MPC)[:3])
        self.assertEqual(
            _ks(quad_source_pairs(reversed(_exit_times(3)))),
            [
                (k[0], k[0]),
                (k[0], k[1]),
                (k[0], k[2]),
                (k[1], k[1]),
                (k[1], k[2]),
                (k[2], k[2]),
            ],
        )

    def test_a_duplicated_member_appears_once(self):
        items = _exit_times(3)
        # the same object twice, and a second object for the same stored row
        again = _exit_time(items[1].store_id, items[1].k.k_inv_Mpc)
        for duplicated in (items + [items[0]], items + [again]):
            with self.subTest(duplicated=duplicated[-1] is items[0]):
                pairs = quad_source_pairs(duplicated)
                self.assertEqual(pairs, quad_source_pairs(items))
                self.assertEqual(len(pairs), 6)

    def test_any_iterable(self):
        items = _exit_times(4)
        self.assertEqual(
            quad_source_pairs(x for x in reversed(items)), quad_source_pairs(items)
        )


def _miss(q, r) -> QuadSource:
    """The factory's labels-only miss, built as ``test_quadsource_parent_main`` builds one."""
    return QuadSource(
        payload=None,
        model=object(),
        z_sample=None,
        q=q,
        r=r,
        labels_only=True,
    )


def _stored(q, r) -> QuadSource:
    return QuadSource(
        payload={
            "store_id": 9,
            "compute_time": 1.0,
            "values": [],
            "Tq_serial": 5,
            "Tr_serial": 6,
        },
        model=object(),
        z_sample=None,
        q=q,
        r=r,
    )


class TestRequireQuadSource(unittest.TestCase):
    """E2, run."""

    def test_a_miss_names_both_k_and_says_no_stored_quadsource(self):
        q, r = _exit_time(13, 3.1622776601683794e5), _exit_time(12, 1.0e5)
        miss = _miss(q, r)
        self.assertFalse(miss.available)
        with self.assertRaises(RuntimeError) as caught:
            require_quad_source(miss)
        message = str(caught.exception)
        self.assertIn(str(q.k.k_inv_Mpc), message)
        self.assertIn(str(r.k.k_inv_Mpc), message)
        self.assertIn("store_id 13", message)
        self.assertIn("store_id 12", message)
        self.assertIn("no stored QuadSource", message)

    def test_a_miss_on_the_diagonal_names_its_k(self):
        q = _exit_time(11, 1.0e6)
        with self.assertRaises(RuntimeError) as caught:
            require_quad_source(_miss(q, q))
        self.assertIn(str(q.k.k_inv_Mpc), str(caught.exception))

    def test_a_stored_source_passes(self):
        q, r = _exit_time(12, 1.0e5), _exit_time(13, 3.1622776601683794e5)
        stored = _stored(q, r)
        self.assertTrue(stored.available)
        self.assertIsNone(require_quad_source(stored))


def _function(tree, name) -> ast.FunctionDef:
    (found,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _called(node) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _position(node):
    return (node.lineno, node.col_offset)


class TestTheReader(unittest.TestCase):
    """E1 and E2, read with ``ast`` from ``extract_tensor_source_data.py``."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(READER.read_text(), filename=str(READER))

    def test_run_pipeline_builds_work_grid_from_quad_source_pairs(self):
        scope = _function(self.tree, "run_pipeline")
        assigns = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "work_grid" for t in node.targets
            )
        ]
        self.assertEqual(len(assigns), 1)
        value = assigns[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertEqual(_called(value), "quad_source_pairs")
        self.assertEqual([ast.unparse(a) for a in value.args], ["k_subsample"])

    def test_run_pipeline_calls_no_combinations_with_replacement(self):
        scope = _function(self.tree, "run_pipeline")
        calls = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Call)
            and _called(node) == "combinations_with_replacement"
        ]
        self.assertEqual(calls, [])

    def test_plot_tensor_source_requires_the_source_before_reading_it(self):
        scope = _function(self.tree, "plot_tensor_source")
        checks = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Call)
            and _called(node) == "require_quad_source"
            and [ast.unparse(a) for a in node.args] == ["source"]
        ]
        reads = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "source"
            and node.attr in ("values", "functions")
        ]
        self.assertGreater(len(checks), 0)
        self.assertGreater(len(reads), 0)
        self.assertLess(
            min(_position(n) for n in checks), min(_position(n) for n in reads)
        )

    def test_the_helpers_are_imported_from_extract_common(self):
        imported = {
            alias.name
            for node in self.tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "extract_common"
            for alias in node.names
        }
        self.assertLessEqual({"quad_source_pairs", "require_quad_source"}, imported)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
