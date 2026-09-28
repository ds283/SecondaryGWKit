"""
Measurements behind the orchestrator's review of datastore-integrity prompt 04 (board §1), and the
expected figures prompt 04b is written against.

Read-only and static: it imports the guard module shipped by 04 (6a97d8a) and walks the factory
sources with ``ast``; it opens no engine and no store, and needs no Ray. Run from the repository
root:

    PYTHONPATH=. ./venv/bin/python prompts/datastore-integrity/orchestrator/review_04_measurements.py

Sections:
  A. every row binding in every function of every factory, keyed by function, and which of them
     04's analyser does not visit (review finding 1);
  B. whether dropping any one visited function name from the analyser fails the coverage pin
     (review finding 2);
  C. every ``<base>.c.<name>`` G2 cannot resolve to a table, per function and base, including
     module-level functions, and every ``tables["X"]`` / insert target naming no schema table
     (review finding 3);
  D. the function-qualified coverage set a guard visiting every function would pin, and its
     projection onto 04's "module:class.variable" identity (the target for prompt 04b).
"""

import ast
from collections import Counter, defaultdict

import sqlalchemy as sqla

import Datastore.tests.test_factory_select_columns as m


def functions(tree):
    """(class name, FunctionDef) for every class method and every module-level function."""
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef):
                yield cls.name, fn
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            yield "<module-level>", fn


def section_a():
    print("== A. row bindings by function (every function; read_batch inlined)")
    visited = {"build", "read_batch"} | set(m._ROW_HELPERS)
    per_fn = Counter()
    unvisited = []
    for path in sorted(m.FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls, fn in functions(tree):
            if fn.name == "read_batch":
                found = m.analyse_read_batch(fn, path.name, cls)
            else:
                inlined, _ = m.inline_nested_row_helper(fn)
                found = m.analyse_build(inlined, path.name, cls)
            for f in found:
                per_fn[fn.name] += 1
                status = "decidable" if f.decidable else "BLIND"
                missing = sorted(f.missing) if f.spec is not None else None
                line = f"   {path.name}:{cls}.{fn.name}.{f.row_var}  {status}  missing={missing}"
                if fn.name not in visited:
                    unvisited.append(line)
    print("   bindings per function name:", dict(sorted(per_fn.items())))
    print(f"   bindings in functions 04 does not visit: {len(unvisited)}")
    for line in unvisited:
        print(line)


def section_b():
    print("== B. drop one visited helper from _ROW_HELPERS: does anything fail?")
    for name in sorted(m._ROW_HELPERS):
        orig = m._ROW_HELPERS
        m._ROW_HELPERS = frozenset(orig - {name})
        try:
            found = m.analyse_factories()
        finally:
            m._ROW_HELPERS = orig
        covered = {f"{f.key}.{f.row_var}" for f in found if f.decidable}
        short = defaultdict(set)
        for f in found:
            if f.decidable and f.missing:
                short[f.key].update(f.missing)
        short = {k: sorted(v) for k, v in short.items()}
        print(
            f"   without {name}: coverage pin {'FAILS' if covered != set(m.EXPECTED_COVERAGE) else 'passes'}; "
            f"KNOWN_UNFIXED pin {'FAILS' if short != m.KNOWN_UNFIXED else 'passes'}"
        )


def section_c():
    print("== C. G2: references whose table does not resolve")
    built = m.build_schema(sqla.MetaData(), m._factories)
    columns = {n: set(t.c.keys()) for n, t in built.tables.items()}
    cls_to_table = {f.__name__: n for n, f in m._factories.items()}
    unresolved = Counter()
    unknown_tables = []
    total = 0
    for path in sorted(m.FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls, fn in functions(tree):
            own = cls_to_table.get(cls)
            env = {"table": own} if own else {}
            site = f"{path.name}:{cls}.{fn.name}"
            for node in sorted(
                (n for n in ast.walk(fn) if hasattr(n, "lineno")),
                key=lambda n: (n.lineno, n.col_offset),
            ):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "tables"
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value not in columns
                ):
                    unknown_tables.append(
                        f"{site}:{node.lineno} tables[{node.slice.value!r}]"
                    )
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    if isinstance(node.targets[0], ast.Name):
                        t = m._table_of(node.value, env)
                        if t is not None:
                            env[node.targets[0].id] = t
                ref = m._column_ref(node)
                if ref is None:
                    continue
                total += 1
                t = m._table_of(ref[0], env)
                if t is None:
                    unresolved[(site, ast.unparse(ref[0]))] += 1
    print(f"   .c. references, class methods and module-level functions: {total}")
    print(
        f"   unresolved: {sum(unresolved.values())} at {len(unresolved)} (function, base) pairs"
    )
    for (site, base), n in sorted(unresolved.items()):
        print(f"   {n:4d}  {site}  base={base}")
    print(f"   tables['X'] naming no schema table: {len(unknown_tables)}")
    for line in unknown_tables:
        print("   ", line)


def section_d():
    print("== D. function-qualified identity, every function visited")
    decidable, blind = set(), set()
    for path in sorted(m.FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls, fn in functions(tree):
            if fn.name == "read_batch":
                found = m.analyse_read_batch(fn, path.name, cls)
            else:
                found = m.analyse_build(
                    m.inline_nested_row_helper(fn)[0], path.name, cls
                )
            for f in found:
                key = f"{path.name}:{cls}.{fn.name}.{f.row_var}"
                (decidable if f.decidable else blind).add(key)
    projection = {k.rsplit(".", 2)[0] + "." + k.rsplit(".", 1)[1] for k in decidable}
    old = set(m.EXPECTED_COVERAGE)
    print(f"   decidable: {len(decidable)}; blind: {len(blind)} {sorted(blind)}")
    print(
        f"   04's EXPECTED_COVERAGE: {len(old)}; missing from the projection: {sorted(old - projection)}"
    )
    print(f"   in the projection but not in 04's: {sorted(projection - old)}")
    helpers = sorted(
        k for k in decidable if "._build_impl_" in k or ".read_batch." in k
    )
    print(
        f"   _build_impl_* and read_batch bindings: {len(helpers)} (probe_02.out lists 16)"
    )
    for k in sorted(decidable):
        print("   ", k)


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
