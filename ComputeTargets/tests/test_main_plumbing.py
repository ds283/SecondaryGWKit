"""
Structural tests for the payload plumbing of main.py: the QuadSourceIntegral stage, and (prompt 16
of prompts/GkTk-remedial) the unresolved-oscillation summary the two numeric work queues now print
in place of the per-object warning.

`main.py` is a script: importing it builds a `ShardedPool`, connects to Ray and starts running
the pipeline, so it cannot be imported from a test. The two module-level helpers of its
QuadSourceIntegral stage are therefore extracted with `ast` and compiled on their own, with the
handful of names their bodies need supplied as globals. Nothing in main.py's import list is
executed.

These tests live in `ComputeTargets/tests/` rather than in a new top-level `tests/` directory
because (i) the campaign's convention is one test root per package
(`python -m unittest discover -s <package>/tests -t .`, README section 5 item 7) and a second
root would need a second discover command that nothing else runs, and (ii) the contract being
checked belongs to this package: the payload keys are
`ComputeTargets.QuadSourceIntegral.QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS`, which the tests
below assert against directly.

This is a *dry* test. It checks that the right objects reach the right payload slots and that a
missing ingredient fails loudly; it does not exercise the pipeline, which needs Ray and a
datastore (prompt 12 of prompts/source-remediation).
"""

import ast
import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from ComputeTargets.QuadSourceIntegral import QuadSourceIntegral

MAIN_PY = Path(__file__).parents[2] / "main.py"


def load_main_py_functions(names: Iterable[str]) -> dict:
    """
    Compile the named top-level functions of main.py in isolation and return the namespace they
    were executed in. Only `datetime` and the names used in the extracted signatures'
    annotations are provided; if a function ever comes to need anything else from main.py's
    globals, this will fail with a NameError rather than silently test something else.
    """
    names = list(names)
    tree = ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY))

    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = [node.name for node in wanted]
    missing = [name for name in names if name not in found]
    if len(missing) > 0:
        raise RuntimeError(
            f"load_main_py_functions: {', '.join(missing)} is not a module-level function of main.py"
        )

    module = ast.Module(body=wanted, type_ignores=[])
    code = compile(module, str(MAIN_PY), "exec")

    # annotations in a signature are evaluated when the function object is created, so the
    # annotation names have to exist; their values are irrelevant here
    namespace = {
        "datetime": datetime,
        "List": List,
        "redshift": object,
        "wavenumber_exit_time": object,
        "BesselPhaseProxy": object,
        "GkSourcePolicyData": object,
        "QuadSource": object,
        "TkNumericIntegration": object,
        "TkWKBIntegration": object,
    }
    exec(code, namespace)

    return namespace


_main = load_main_py_functions(
    [
        "build_QuadSourceIntegral_payload",
        "record_unresolved_osc",
        "format_unresolved_osc_summary",
    ]
)
build_QuadSourceIntegral_payload = _main["build_QuadSourceIntegral_payload"]
record_unresolved_osc = _main["record_unresolved_osc"]
format_unresolved_osc_summary = _main["format_unresolved_osc_summary"]

# the two numeric work queues whose post_handler must feed the accumulator
NUMERIC_QUEUE_TITLES = (
    "CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS",
    "CALCULATE NUMERICAL PART OF MATTER TRANSFER FUNCTIONS",
)


class StubWavenumber:
    def __init__(self, store_id: int, k_inv_Mpc: float):
        self.store_id = store_id
        self.k_inv_Mpc = k_inv_Mpc


class StubExitTime:
    """Stands in for a wavenumber_exit_time: a store_id plus the wavenumber it carries."""

    def __init__(self, store_id: int, k_inv_Mpc: float):
        self.store_id = store_id
        self.k = StubWavenumber(store_id + 1000, k_inv_Mpc)


class StubRedshift:
    def __init__(self, store_id: int, z: float):
        self.store_id = store_id
        self.z = z


class StubIngredient:
    """Stands in for any of the six looked-up ingredient objects."""

    def __init__(self, name: str, available: bool = True):
        self.name = name
        self.available = available


class StubProxy:
    def __init__(self, name: str):
        self.name = name


class QuadSourceIntegralPayloadTestCase(unittest.TestCase):
    """
    The caches are populated with more than one entry each, and with different objects for q and
    for r, so that a payload assembled with a transposed or mis-keyed lookup fails.
    """

    def setUp(self):
        self.z_response = StubRedshift(11, 1.5)
        self.other_z_response = StubRedshift(12, 3.0)

        self.k = StubExitTime(21, 1.0e6)
        self.q = StubExitTime(22, 2.0e5)
        self.r = StubExitTime(23, 9.0e5)
        self.other_k = StubExitTime(24, 3.0e8)

        self.GkPolicy = StubIngredient("GkPolicy(k,z_response)")
        self.source = StubIngredient("QuadSource(q,r)")
        self.Tq_numeric = StubIngredient("Tk_numeric(q)")
        self.Tq_WKB = StubIngredient("Tk_WKB(q)")
        self.Tr_numeric = StubIngredient("Tk_numeric(r)")
        self.Tr_WKB = StubIngredient("Tk_WKB(r)")

        self.Gk_cache = {
            (self.k.store_id, self.z_response.store_id): self.GkPolicy,
            (self.k.store_id, self.other_z_response.store_id): StubIngredient("decoy"),
            (self.other_k.store_id, self.z_response.store_id): StubIngredient("decoy"),
        }
        self.source_cache = {
            (self.q.store_id, self.r.store_id): self.source,
            (self.q.store_id, self.q.store_id): StubIngredient("decoy"),
            (self.r.store_id, self.r.store_id): StubIngredient("decoy"),
        }
        self.Tk_numeric_cache = {
            self.q.store_id: self.Tq_numeric,
            self.r.store_id: self.Tr_numeric,
            self.other_k.store_id: StubIngredient("decoy"),
        }
        self.Tk_WKB_cache = {
            self.q.store_id: self.Tq_WKB,
            self.r.store_id: self.Tr_WKB,
            self.other_k.store_id: StubIngredient("decoy"),
        }

        self.b_value = 0.0
        self.Bessel_0pt5 = StubProxy("Bessel_0pt5")
        self.Bessel_2pt5 = StubProxy("Bessel_2pt5")

    def build(self, q=None, r=None):
        return build_QuadSourceIntegral_payload(
            self.z_response,
            self.k,
            self.q if q is None else q,
            self.r if r is None else r,
            self.Gk_cache,
            self.source_cache,
            self.Tk_numeric_cache,
            self.Tk_WKB_cache,
            self.b_value,
            self.Bessel_0pt5,
            self.Bessel_2pt5,
        )

    def test_payload_supplies_exactly_the_required_keys(self):
        payload = self.build()

        self.assertEqual(
            set(payload.keys()), set(QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS)
        )
        # the four transfer-function keys are the ones prompt 08 added and this stage now supplies
        for key in ("Tq_numeric", "Tq_WKB", "Tr_numeric", "Tr_WKB"):
            self.assertIn(key, payload)

    def test_payload_carries_the_objects_for_the_right_store_ids(self):
        payload = self.build()

        self.assertIs(payload["GkPolicy"], self.GkPolicy)
        self.assertIs(payload["source"], self.source)
        self.assertIs(payload["Tq_numeric"], self.Tq_numeric)
        self.assertIs(payload["Tq_WKB"], self.Tq_WKB)
        self.assertIs(payload["Tr_numeric"], self.Tr_numeric)
        self.assertIs(payload["Tr_WKB"], self.Tr_WKB)

        # q and r are distinct modes, so the numeric/WKB pairs must not be transposed
        self.assertIsNot(payload["Tq_numeric"], payload["Tr_numeric"])
        self.assertIsNot(payload["Tq_WKB"], payload["Tr_WKB"])

    def test_b_and_the_bessel_proxies_pass_straight_through(self):
        payload = self.build()

        self.assertEqual(payload["b"], self.b_value)
        self.assertIs(payload["Bessel_0pt5"], self.Bessel_0pt5)
        self.assertIs(payload["Bessel_2pt5"], self.Bessel_2pt5)

    def test_q_equals_r_uses_the_same_transfer_function_objects(self):
        payload = self.build(q=self.q, r=self.q)

        self.assertIs(payload["Tq_numeric"], payload["Tr_numeric"])
        self.assertIs(payload["Tq_WKB"], payload["Tr_WKB"])
        self.assertIs(payload["source"], self.source_cache[(22, 22)])

    def test_an_unavailable_ingredient_raises_and_is_reported(self):
        cases = {
            "GkPolicy": (self.Gk_cache, (self.k.store_id, self.z_response.store_id)),
            "source": (self.source_cache, (self.q.store_id, self.r.store_id)),
            "Tq_numeric": (self.Tk_numeric_cache, self.q.store_id),
            "Tq_WKB": (self.Tk_WKB_cache, self.q.store_id),
            "Tr_numeric": (self.Tk_numeric_cache, self.r.store_id),
            "Tr_WKB": (self.Tk_WKB_cache, self.r.store_id),
        }

        for label, (cache, key) in cases.items():
            with self.subTest(ingredient=label):
                held = cache[key]
                cache[key] = StubIngredient(label, available=False)
                try:
                    captured = io.StringIO()
                    with redirect_stdout(captured):
                        with self.assertRaises(RuntimeError) as raised:
                            self.build()
                finally:
                    cache[key] = held

                self.assertIn(
                    "missing or incomplete source data", str(raised.exception)
                )
                self.assertIn("!! MISSING DATA WARNING", captured.getvalue())

    def test_all_ingredients_available_prints_nothing(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.build()

        self.assertEqual(captured.getvalue(), "")

    def test_a_cache_miss_is_a_keyerror_not_a_silent_none(self):
        with self.assertRaises(KeyError):
            self.build(q=StubExitTime(99, 1.0e7))


if __name__ == "__main__":
    unittest.main()


# -------------------------------------------------------------------------------------------
# prompt 16 of prompts/GkTk-remedial: the unresolved-oscillation summary
#
# The per-object warning that numeric_with_phase_cut used to print is gated off in the two
# production integrators; main.py accumulates the flag over each numeric work queue through a
# post_handler and prints one block per sector instead (README section 7 decision D2, taken by
# the user 2026-09-11). Both halves are tested here: the two module-level functions on stand-ins,
# and -- with ast, because the wiring lives inside run_pipeline, which cannot be extracted -- the
# fact that the queues are wired to them at all. The failure mode this guards against is silent:
# drop the post_handler and the pipeline still runs, and simply never reports.
# -------------------------------------------------------------------------------------------


class StubNumericObject:
    """Stands in for a GkNumericIntegration / TkNumericIntegration object as post_handler sees it.

    ``populated=False`` reproduces the RuntimeError those classes raise from
    ``has_unresolved_osc`` before a payload has been read back
    (TkNumericIntegration.py:235-238 and the GkNumericIntegration twin).
    """

    def __init__(
        self,
        k_inv_Mpc: float,
        flagged: bool = False,
        efolds: float = None,
        populated: bool = True,
    ):
        self.k = StubWavenumber(1, k_inv_Mpc)
        self._flagged = flagged
        self._efolds = efolds
        self._populated = populated

    @property
    def has_unresolved_osc(self):
        if not self._populated:
            raise RuntimeError("has_unresolved_osc has not yet been populated")
        return self._flagged

    @property
    def unresolved_efolds_subh(self):
        if not self._populated:
            raise RuntimeError("has_unresolved_osc has not yet been populated")
        return self._efolds


class UnresolvedOscSummaryTestCase(unittest.TestCase):
    def _accumulate(self, objects):
        summary = {}
        for obj in objects:
            record_unresolved_osc(summary, obj)
        return summary

    def test_summary_names_both_wavenumbers_with_counts_and_efolds(self):
        summary = self._accumulate(
            [
                StubNumericObject(1.0e5, flagged=True, efolds=3.5),
                StubNumericObject(1.0e5, flagged=True, efolds=4.25),
                StubNumericObject(1.0e5, flagged=False),
                StubNumericObject(3.0e8, flagged=True, efolds=6.75),
            ]
        )

        self.assertEqual(summary[1.0e5]["recorded"], 3)
        self.assertEqual(summary[1.0e5]["flagged"], 2)
        self.assertEqual(summary[1.0e5]["efolds_min"], 3.5)
        self.assertEqual(summary[1.0e5]["efolds_max"], 4.25)
        self.assertEqual(summary[3.0e8]["recorded"], 1)
        self.assertEqual(summary[3.0e8]["flagged"], 1)

        lines = format_unresolved_osc_summary(summary, "tensor Green's functions")

        self.assertEqual(len(lines), 4)
        self.assertIn("tensor Green's functions", lines[0])

        # one line per flagged wavenumber, in ascending k
        self.assertIn("k = 1e+05/Mpc", lines[1])
        self.assertIn("2 of 3 objects flagged", lines[1])
        self.assertIn("3.5 to 4.25", lines[1])
        self.assertIn("k = 3e+08/Mpc", lines[2])
        self.assertIn("1 of 1 objects flagged", lines[2])

        # the closing totals line
        self.assertIn("TOTAL: 3 of 4 objects flagged", lines[3])
        self.assertIn("2 of 2 wavenumbers", lines[3])

    def test_a_wavenumber_that_never_flagged_is_not_listed_but_is_counted(self):
        summary = self._accumulate(
            [
                StubNumericObject(1.0e5, flagged=True, efolds=3.5),
                StubNumericObject(3.0e8, flagged=False),
                StubNumericObject(3.0e8, flagged=False),
            ]
        )
        lines = format_unresolved_osc_summary(summary, "tensor Green's functions")

        self.assertEqual(len(lines), 3)
        self.assertNotIn("3e+08", lines[1])
        self.assertIn("TOTAL: 1 of 3 objects flagged", lines[2])
        self.assertIn("1 of 2 wavenumbers", lines[2])

    def test_an_all_clear_accumulator_says_so_in_one_line(self):
        """A run that prints nothing is indistinguishable from a run whose wiring was dropped."""
        summary = self._accumulate(
            [
                StubNumericObject(1.0e5, flagged=False),
                StubNumericObject(3.0e8, flagged=False),
            ]
        )
        lines = format_unresolved_osc_summary(summary, "matter transfer functions")

        self.assertEqual(len(lines), 2)
        self.assertIn("matter transfer functions", lines[0])
        self.assertIn("no object reported unresolved oscillations", lines[1])
        self.assertIn("2 objects over 2 wavenumbers", lines[1])

    def test_an_empty_accumulator_prints_nothing(self):
        """The queue did not run: there is nothing to report and no block to print."""
        self.assertEqual(
            format_unresolved_osc_summary({}, "tensor Green's functions"), []
        )

    def test_an_unpopulated_object_does_not_raise_and_is_counted_separately(self):
        summary = self._accumulate(
            [
                StubNumericObject(1.0e5, flagged=True, efolds=3.5),
                StubNumericObject(1.0e5, populated=False),
                None,
            ]
        )

        self.assertEqual(summary[1.0e5]["recorded"], 2)
        self.assertEqual(summary[1.0e5]["flagged"], 1)
        self.assertEqual(summary[1.0e5]["unpopulated"], 1)

        lines = format_unresolved_osc_summary(summary, "tensor Green's functions")
        self.assertIn("1 of 2 objects flagged", lines[1])
        self.assertIn("without a populated has_unresolved_osc flag", lines[-1])

    def test_a_flagged_object_without_efolds_still_reports(self):
        summary = self._accumulate(
            [StubNumericObject(1.0e5, flagged=True, efolds=None)]
        )
        lines = format_unresolved_osc_summary(summary, "tensor Green's functions")
        self.assertIn("1 of 1 objects flagged", lines[1])
        self.assertIn("e-folds inside horizon unavailable", lines[1])


class UnresolvedOscWiringTestCase(unittest.TestCase):
    """Both numeric RayWorkPool constructions pass a post_handler, and both are followed by a
    call to the formatter. This is the test that catches the silent failure: if a later edit
    drops the post_handler, the pipeline still runs and simply never reports."""

    @staticmethod
    def _numeric_queue_blocks():
        """Return, per numeric queue title, the ast.If block of main.py that constructs it."""
        tree = ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY))

        blocks = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            for call in ast.walk(node):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "RayWorkPool"
                ):
                    continue
                for keyword in call.keywords:
                    if keyword.arg != "title":
                        continue
                    if not isinstance(keyword.value, ast.Constant):
                        continue
                    if keyword.value.value in NUMERIC_QUEUE_TITLES:
                        blocks[keyword.value.value] = (node, call)

        return blocks

    def test_both_numeric_queues_are_found(self):
        blocks = self._numeric_queue_blocks()
        self.assertEqual(set(blocks.keys()), set(NUMERIC_QUEUE_TITLES))

    def test_both_numeric_queues_pass_a_post_handler_that_records(self):
        for title, (_, call) in self._numeric_queue_blocks().items():
            keywords = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn("post_handler", keywords, msg=title)

            names = {
                node.id
                for node in ast.walk(keywords["post_handler"])
                if isinstance(node, ast.Name)
            }
            self.assertIn("record_unresolved_osc", names, msg=title)

    def test_both_numeric_queues_are_followed_by_the_formatter(self):
        for title, (block, _) in self._numeric_queue_blocks().items():
            calls = {
                node.func.id
                for node in ast.walk(block)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("format_unresolved_osc_summary", calls, msg=title)
