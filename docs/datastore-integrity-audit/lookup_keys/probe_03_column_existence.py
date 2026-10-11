"""
Column-existence sweep over every factory function -- the hole the read-back guard's docstring
lists as point 6 ("a SELECT naming a column the table does not have would be caught by SQLAlchemy
at query-build time, not here"). That is true only on a path that runs; on a path nothing runs, a
wrong column name is a latent AttributeError.

Statically resolves each ``<var>.c.<name>`` / ``<var>.c["<name>"]`` whose ``<var>`` is the
factory's own ``table``, ``tables["X"]``, or an ``.alias(...)`` of either, and checks ``<name>``
against the columns ``build_schema`` actually declares for that table. ``subquery.c.<name>`` and
dynamically named columns are skipped (they are not table columns).

Also reports ``insert`` payload keys that match no column (``Datastore._insert`` passes the dict
to ``sqla.insert(table)``, which drops unknown keys silently).
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import sqlalchemy as sqla  # noqa: E402

from config.datastore import factories as _factories  # noqa: E402
from datastorekit.SQL.schema import build_schema  # noqa: E402

FACTORY_DIR = ROOT / "Datastore" / "SQL" / "ObjectFactories"


def main():
    built = build_schema(sqla.MetaData(), _factories)
    columns = {name: set(t.c.keys()) for name, t in built.tables.items()}
    cls_to_table = {f.__name__: name for name, f in _factories.items()}

    problems = []
    checked = 0
    for path in sorted(FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            own = cls_to_table.get(cls.name)
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                env = {}
                if own is not None:
                    env["table"] = own

                def table_of(expr):
                    # tables["X"]
                    if (
                        isinstance(expr, ast.Subscript)
                        and isinstance(expr.value, ast.Name)
                        and expr.value.id == "tables"
                        and isinstance(expr.slice, ast.Constant)
                    ):
                        return expr.slice.value
                    if isinstance(expr, ast.Name):
                        return env.get(expr.id)
                    # <x>.alias(...)
                    if (
                        isinstance(expr, ast.Call)
                        and isinstance(expr.func, ast.Attribute)
                        and expr.func.attr == "alias"
                    ):
                        return table_of(expr.func.value)
                    return None

                for node in sorted(
                    (n for n in ast.walk(fn) if hasattr(n, "lineno")),
                    key=lambda n: (n.lineno, n.col_offset),
                ):
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        tgt = node.targets[0]
                        if isinstance(tgt, ast.Name):
                            t = table_of(node.value)
                            if t is not None:
                                env[tgt.id] = t
                    name = None
                    base = None
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "c"
                    ):
                        name, base = node.attr, node.value.value
                    elif (
                        isinstance(node, ast.Subscript)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "c"
                        and isinstance(node.slice, ast.Constant)
                    ):
                        name, base = node.slice.value, node.value.value
                    if name is None:
                        continue
                    t = table_of(base)
                    if t is None:
                        continue
                    checked += 1
                    if t not in columns:
                        problems.append(
                            (
                                path.name,
                                cls.name,
                                fn.name,
                                node.lineno,
                                f"{t} (no such table)",
                                name,
                            )
                        )
                    elif name not in columns[t]:
                        problems.append(
                            (path.name, cls.name, fn.name, node.lineno, t, name)
                        )

                # insert payload keys: inserter(conn, {...}) / value_inserter(conn, {...})
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id.endswith("inserter")
                        and len(node.args) == 2
                        and isinstance(node.args[1], ast.Dict)
                    ):
                        target = own
                        if node.func.id != "inserter":
                            # value_inserter = inserters["X"]
                            for a in ast.walk(fn):
                                if (
                                    isinstance(a, ast.Assign)
                                    and isinstance(a.targets[0], ast.Name)
                                    and a.targets[0].id == node.func.id
                                    and isinstance(a.value, ast.Subscript)
                                    and isinstance(a.value.slice, ast.Constant)
                                ):
                                    target = a.value.slice.value
                        if target not in columns:
                            continue
                        for k in node.args[1].keys:
                            if (
                                isinstance(k, ast.Constant)
                                and k.value not in columns[target]
                            ):
                                problems.append(
                                    (
                                        path.name,
                                        cls.name,
                                        fn.name,
                                        node.lineno,
                                        f"{target} [insert key]",
                                        k.value,
                                    )
                                )

    print(f"checked {checked} resolvable column references\n")
    for mod, cls, fn, line, table, name in problems:
        print(f"{mod}:{line}  {cls}.{fn}: '{name}' is not a column of {table}")
    if not problems:
        print("no problems")


if __name__ == "__main__":
    main()
