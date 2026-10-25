"""
The four readers that plot ``analytic_G_w`` plot only sensible points of it, and write the right
columns (prompts/handover prompt 09 §2 and §3).

**The defects guarded.**

- ``extract_GkWKB_data.py`` plotted |``analytic_G_w``| raw beside |G| on log-log axes and died with
  ``OverflowError`` on A3 v2, where the column reaches 7.57e+243
  (``[08-the-gkwkb-reader-fails-plotting-…]``). Every expression in the four readers that builds
  plot points from ``analytic_G_w`` or ``analytic_Gprime_w`` must pass through
  ``extract_common.sensible_log_points``, and ``extract_Gk_data.py``'s signed linear-y series
  through its signed form.
- With prompt 09 §1, new stores hold ``NULL`` in those columns, and ``fabs(None)`` raises
  ``TypeError``: ``extract_GkWKB_data.py`` must apply no bare ``fabs`` to an ``analytic_`` attribute.
- ``extract_Gk_data.py`` filled its CSV columns ``analytic_G_w`` and ``analytic_Gprime_w`` from the
  radiation-form attributes (``[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]``).

The readers call ``ray.init`` at module scope and cannot be imported, so they are read with ``ast``,
as ``test_reader_read_only`` reads them. An occurrence of either attribute is accepted when it is
(a) inside a call to ``sensible_log_points``; (b) in a statement that builds a CSV column, one
assigning or extending a name ending ``_column``; or (c) in a statement that builds a list whose
every later use is as the first argument of ``sensible_log_points`` or as the receiver of its own
``extend`` or ``append``. Anything else is a plot-points expression that bypasses the helper.
No Ray, no datastore.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

READERS = (
    "extract_GkWKB_data.py",
    "extract_Gk_data.py",
    "extract_GkSource_data.py",
    "extract_QuadSourceIntegral_data.py",
)

ATTRIBUTES = ("analytic_G_w", "analytic_Gprime_w")
HELPER = "sensible_log_points"


def _parse(name):
    tree = ast.parse((REPO_ROOT / name).read_text(), filename=name)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree


def _ancestors(node):
    while hasattr(node, "parent"):
        node = node.parent
        yield node


def _is_helper_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == HELPER
    )


def _statement(node):
    return next(a for a in _ancestors(node) if isinstance(a, ast.stmt))


def _function(node):
    return next((a for a in _ancestors(node) if isinstance(a, ast.FunctionDef)), None)


def _target(statement):
    """The name a statement builds: ``T = ...``, or ``T.extend(...)`` / ``T.append(...)``."""
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr in ("extend", "append")
        and isinstance(statement.value.func.value, ast.Name)
    ):
        return statement.value.func.value.id
    return None


def _is_routed(function, name) -> bool:
    """Every load of ``name`` in ``function`` is the helper's first argument or its own extend/append."""
    helper_uses = 0
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        parent = node.parent
        if _is_helper_call(parent) and parent.args and parent.args[0] is node:
            helper_uses += 1
            continue
        if isinstance(parent, ast.Attribute) and parent.attr in ("extend", "append"):
            continue
        return False
    return helper_uses > 0


def classify(tree):
    """``[(line, kind, detail)]`` for every occurrence of the two attributes, kind one of
    ``wrapped``, ``column``, ``routed`` or ``BYPASS``."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ATTRIBUTES):
            continue
        helper = next((a for a in _ancestors(node) if _is_helper_call(a)), None)
        statement = _statement(node)
        target = _target(statement)
        if helper is not None:
            signed = any(
                kw.arg == "signed"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in helper.keywords
            )
            found.append((node.lineno, "wrapped", (target, signed)))
        elif target is not None and target.endswith("_column"):
            found.append((node.lineno, "column", target))
        elif target is not None and _is_routed(_function(node), target):
            found.append((node.lineno, "routed", target))
        else:
            found.append((node.lineno, "BYPASS", ast.unparse(statement)[:120]))
    return found


class TestEveryPlottedSeriesPassesThroughTheHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = {name: _parse(name) for name in READERS}
        cls.found = {name: classify(tree) for name, tree in cls.trees.items()}

    def test_no_reader_bypasses_the_helper(self):
        for name in READERS:
            with self.subTest(reader=name):
                bypasses = [f for f in self.found[name] if f[1] == "BYPASS"]
                self.assertEqual(bypasses, [])

    def test_each_reader_builds_plot_points_through_the_helper(self):
        # the walk finds something in each reader: a test that matched nothing would pass vacuously
        for name in READERS:
            with self.subTest(reader=name):
                kinds = [kind for _, kind, _ in self.found[name]]
                self.assertTrue(
                    any(kind in ("wrapped", "routed") for kind in kinds), kinds
                )

    def test_each_reader_imports_the_helper_from_extract_common(self):
        for name in READERS:
            with self.subTest(reader=name):
                imported = {
                    alias.name
                    for node in ast.walk(self.trees[name])
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "extract_common"
                    for alias in node.names
                }
                self.assertIn(HELPER, imported)

    def test_the_gk_reader_filters_its_signed_series_with_the_signed_form(self):
        wrapped = {
            detail
            for _, kind, detail in self.found["extract_Gk_data.py"]
            if kind == "wrapped"
        }
        self.assertIn(("analytic_G_w_points", True), wrapped)
        self.assertIn(("abs_analytic_G_w_points", False), wrapped)

    def test_the_gkwkb_reader_routes_both_branches_to_one_filtered_series(self):
        routed = [
            detail
            for _, kind, detail in self.found["extract_GkWKB_data.py"]
            if kind == "routed"
        ]
        self.assertEqual(len(routed), 2)  # the numeric branch and the WKB branch
        self.assertEqual(len(set(routed)), 1)


class TestTheGkWKBReaderHasNoBareFabsOnAnAnalyticColumn(unittest.TestCase):
    """New stores hold None in the analytic columns, and ``fabs(None)`` raises ``TypeError``."""

    def test_no_bare_fabs_of_an_analytic_attribute(self):
        tree = _parse("extract_GkWKB_data.py")
        offenders = [
            (node.lineno, ast.unparse(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fabs"
            and any(
                isinstance(sub, ast.Attribute) and sub.attr.startswith("analytic_")
                for arg in node.args
                for sub in ast.walk(arg)
            )
        ]
        self.assertEqual(offenders, [])

    def test_every_analytic_attribute_under_a_modulus_uses_safe_fabs(self):
        tree = _parse("extract_GkWKB_data.py")
        safe = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "safe_fabs"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr.startswith("analytic_")
        ]
        # |analytic_G_rad| for the plot, twice, and for the CSV, twice; |analytic_G_w| for the CSV, twice
        self.assertEqual(len(safe), 6)


class TestTheGkReaderWritesTheWColumns(unittest.TestCase):
    """``extract_Gk_data.py``'s CSV columns ``analytic_G_w``/``analytic_Gprime_w`` hold the w(z) form."""

    def test_the_csv_maps_each_w_column_to_its_own_attribute(self):
        tree = _parse("extract_Gk_data.py")
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "plot_Gk"
        )
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        frames = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_dict"
        ]
        self.assertEqual(len(frames), 1)
        columns = frames[0].args[0]
        self.assertIsInstance(columns, ast.Dict)
        mapping = {
            key.value: value.id
            for key, value in zip(columns.keys, columns.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
        }
        for column in ATTRIBUTES:
            with self.subTest(column=column):
                self.assertEqual(
                    ast.unparse(assignments[mapping[column]]),
                    f"[value.{column} for value in values]",
                )


if __name__ == "__main__":
    unittest.main()
