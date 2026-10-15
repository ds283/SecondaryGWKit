"""The drivers' `run.finish(` calls (store-fingerprint prompt 04, test 12).

Moved by runregistry-extraction prompt 06 from the registry's `tests/test_store_fingerprint.py`
(`TestDrivers`, with `_HANDLERS` and `_finish_calls`), unchanged but for this module's own
`REPO_ROOT`. It reads the two drivers' source with `ast`, and imports neither.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVERS = (
    REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py",
    REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py",
)

# The drivers' signal handlers, by name: each driver's `terminal`, installed before `ray.init`,
# and `scoped_pipeline_run.make_killed_handler`'s `killed_handler`, installed after it
# (a3-v2-readiness prompt 04, §3 item 5).
_HANDLERS = ("terminal", "killed_handler")


def _finish_calls(path: Path):
    """``[(inside a signal handler, passes fingerprint=True)]`` for every ``run.finish(``."""
    tree = ast.parse(path.read_text())
    found = []

    def visit(node, inside):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside = inside or node.name in _HANDLERS
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "finish"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "run"
        ):
            passes = any(
                k.arg == "fingerprint"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in node.keywords
            )
            found.append((inside, passes))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return found


class TestDrivers(unittest.TestCase):
    """Test 12: every `run.finish(` after the pipeline passes `fingerprint=True`; the `terminal`
    signal handlers, which run with the pool open, do not."""

    def test_the_finish_calls(self):
        expected = {
            "scoped_pipeline_run.py": (4, 2),
            "quadsource_atol_sweep.py": (3, 2),
        }
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                calls = _finish_calls(path)
                outside = [passes for inside, passes in calls if not inside]
                inside = [passes for inside, passes in calls if inside]
                self.assertEqual((len(outside), len(inside)), expected[path.name])
                self.assertTrue(all(outside), outside)
                self.assertFalse(any(inside), inside)


if __name__ == "__main__":
    unittest.main()
