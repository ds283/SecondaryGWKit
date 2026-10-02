"""
Value pins for the production quadrature tolerance pair, `config.defaults.DEFAULT_QUADRATURE_ATOL`
and `DEFAULT_QUADRATURE_RTOL`.

**Why these two numbers.** The A3 v2 store is built at `(atol, rtol) = (1e-32, 1e-7)`. That is the
pair the 2026-09-23/24 tolerance sweep (`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` section 0,
section 7) found to be the loosest converged in all three response-redshift bands, and the user
fixed it as decision D1 of `prompts/a3-v2-readiness` (README section 6.1). Decision D2 was to set
the default rather than add a flag, so that `main.py`, `quadsource_atol_sweep.py --build` and
`extract_QuadSourceIntegral_data.py` share one constant and cannot drift: before prompt 01 of that
campaign `DEFAULT_QUADRATURE_RTOL` was `1e-8`, only the build knew about `1e-7` (by a textual
substitution into `main.py`), and the extractor looked every row up at the wrong tolerance.

**What a change here costs.** Both values are part of the key of every `QuadSourceIntegral` row,
through the `tolerance` rows `main.py` and the extractor look up. Changing either constant
therefore changes which rows every store's readers find, so a store built at the old pair becomes
unreadable by the extractor at the new one. That is a decision, not a refactor.

A value pin is weak on its own: it says nothing about whether the value is right, and anyone can
edit a pin to match a change. Its job is to make a change deliberate. These tests check what the
constants are, and that `extract_QuadSourceIntegral_data.py` looks its `QuadSourceIntegral` rows up
under them.

**Why the second check is here and not in `test_main_plumbing.py`.** That module's
`ReaderToleranceAgreementTestCase` compares the constants each reader passes to `object_get`. The
extractor reads `QuadSourceIntegral` with `object_read_batch`, which the comparison does not scan,
so for this extractor it sees only `wavenumber_exit_time` and would not notice the extractor
querying `QuadSourceIntegral` under another target's tolerance (prompt 01 of
`prompts/a3-v2-readiness`, deliberate breakage (iii)). The check below reads the extractor's own
binding of `quad_atol`/`quad_rtol` and the call that passes them, and does not edit that module.
"""

import ast
import unittest
from pathlib import Path

from ComputeTargets.tests.test_main_plumbing import _defaults_constant_bindings
from config.defaults import DEFAULT_QUADRATURE_ATOL, DEFAULT_QUADRATURE_RTOL

EXTRACTOR = Path(__file__).parents[2] / "extract_QuadSourceIntegral_data.py"


class TestProductionQuadratureTolerance(unittest.TestCase):
    def test_rtol_is_the_a3_v2_rung(self):
        """`rtol = 1e-7` is the loosest rung converged in all three bands (sweep section 0, D1);
        `1e-8`, the old default, is the rung one tighter, used to check a science output.
        """
        self.assertEqual(DEFAULT_QUADRATURE_RTOL, 1e-7)

    def test_atol_is_the_a3_v2_floor(self):
        """`atol = 1e-32` is `5255ac0`'s value, confirmed by the sweep to be the right one on the
        production geometry: relaxing it to `1e-28` or looser returns a wrong answer (sweep
        section 0, answer 1)."""
        self.assertEqual(DEFAULT_QUADRATURE_ATOL, 1e-32)


class TestExtractorQueriesUnderTheQuadraturePair(unittest.TestCase):
    """`extract_QuadSourceIntegral_data.py` must look the tolerance rows its reads are keyed on up
    under `DEFAULT_QUADRATURE_ATOL` / `DEFAULT_QUADRATURE_RTOL`, the constants `main.py` writes
    under. Read with `ast`; neither Ray nor a datastore is needed."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(EXTRACTOR.read_text(), filename=str(EXTRACTOR))

    def test_the_tolerance_lookups_are_under_the_quadrature_constants(self):
        bindings = _defaults_constant_bindings(self.tree)
        self.assertEqual(bindings.get("quad_atol"), "DEFAULT_QUADRATURE_ATOL")
        self.assertEqual(bindings.get("quad_rtol"), "DEFAULT_QUADRATURE_RTOL")

    def test_the_quadsource_integral_read_passes_those_names(self):
        """The names bound above are the ones the `QuadSourceIntegral` read passes as its
        tolerance, so the binding is not a dead local."""
        reads = [
            node
            for node in ast.walk(self.tree)
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
        self.assertTrue(reads, "no QuadSourceIntegral read_batch found")
        for call in reads:
            passed = {
                kw.arg: kw.value.id
                for kw in call.keywords
                if kw.arg in ("atol", "rtol") and isinstance(kw.value, ast.Name)
            }
            self.assertEqual(passed, {"atol": "quad_atol", "rtol": "quad_rtol"})


if __name__ == "__main__":
    unittest.main()
