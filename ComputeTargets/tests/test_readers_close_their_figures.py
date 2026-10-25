"""
Every figure a reader opens is closed before the figure is dropped.

**The defect guarded.** ``plot_Gk`` in ``extract_GkWKB_data.py`` closed its main figure and then
opened a second for the theta plot, which it never closed, so each plotting worker kept one figure
per pair; on A3 v2 matplotlib warned "More than 20 figures have been opened"
(``[10-extract-gkwkb-data-never-closes-its-theta-figure]``). ``extract_TkWKB_data.py`` left its
theta and friction figures open in the same way.

The readers call ``ray.init`` at module scope and cannot be imported, so they are read with ``ast``,
as ``test_analytic_g_w_readers`` reads them. A statement ``fig = plt.figure()`` is closed when a
``plt.close()`` statement follows it in its own block, with no other ``fig = plt.figure()`` between
them in that block; failing that, when one follows the compound statement that holds the block, in
the block that holds *it*, and so on out to the function. A loop ends the search, since a close
after a loop does not close each iteration's figure. No Ray, no datastore.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

READERS = tuple(sorted(p.name for p in REPO_ROOT.glob("extract_*_data.py")))

BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers")


def _is_plt_call(node, name) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "plt"
    )


def _opens(statement) -> bool:
    return isinstance(statement, ast.Assign) and _is_plt_call(statement.value, "figure")


def _closes(statement) -> bool:
    return isinstance(statement, ast.Expr) and _is_plt_call(statement.value, "close")


def _blocks(tree):
    """Yield (owner, block) for every statement list, where owner is the node holding it."""
    for node in ast.walk(tree):
        for field in BLOCK_FIELDS:
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield node, block


def _unclosed_figures(source, filename="<reader>"):
    """The line of every ``fig = plt.figure()`` in ``source`` that is not closed."""
    tree = ast.parse(source, filename=filename)
    # where each statement sits: its block, and the node that holds the block
    place = {}
    for owner, block in _blocks(tree):
        for statement in block:
            place[id(statement)] = (owner, block)

    unclosed = []
    for owner, block in _blocks(tree):
        for statement in block:
            if not _opens(statement):
                continue
            current = statement
            closed = False
            while id(current) in place:
                holder, holding_block = place[id(current)]
                for later in holding_block[holding_block.index(current) + 1 :]:
                    if _closes(later):
                        closed = True
                        break
                    if _opens(later):
                        break
                if closed or isinstance(
                    holder, (ast.FunctionDef, ast.Module, ast.For, ast.While)
                ):
                    break
                current = holder
            if not closed:
                unclosed.append(statement.lineno)
    return unclosed


class TestReadersCloseTheirFigures(unittest.TestCase):
    def test_the_readers_are_found(self):
        self.assertIn("extract_GkWKB_data.py", READERS)
        self.assertIn("extract_TkWKB_data.py", READERS)
        self.assertEqual(len(READERS), 6)

    def test_every_figure_is_closed(self):
        for name in READERS:
            with self.subTest(reader=name):
                source = (REPO_ROOT / name).read_text()
                self.assertEqual(_unclosed_figures(source, name), [])

    def test_the_rule_sees_an_unclosed_figure(self):
        source = (
            "def plot(theta):\n"
            "    fig = plt.figure()\n"
            "    fig.savefig('a.pdf')\n"
            "    plt.close()\n"
            "    if theta is not None:\n"
            "        fig = plt.figure()\n"
            "        fig.savefig('b.pdf')\n"
            "    write_csv()\n"
        )
        self.assertEqual(_unclosed_figures(source), [6])

    def test_the_rule_accepts_a_close_after_the_enclosing_block(self):
        # extract_QuadSourceIntegral_data.py's pattern: a nested figure closed after its if
        source = (
            "def plot(q, r):\n"
            "    fig = plt.figure()\n"
            "    if q != r:\n"
            "        plt.close()\n"
            "        fig = plt.figure()\n"
            "    plt.close()\n"
        )
        self.assertEqual(_unclosed_figures(source), [])

    def test_the_rule_rejects_a_close_after_a_loop(self):
        source = (
            "def plot(items):\n"
            "    for item in items:\n"
            "        fig = plt.figure()\n"
            "    plt.close()\n"
        )
        self.assertEqual(_unclosed_figures(source), [3])
