"""
Run the read-back guard's idea (Datastore/tests/test_factory_select_columns.py) by hand over the
functions the guard does not look at, for [01-read-batch-is-outside-the-guard].

The guard analyses only methods literally named ``build``. This script reuses its analyser
(``analyse_build``) unchanged and applies it to

  * every ``read_batch`` -- after inlining the nested ``make_obj(row)`` / ``make_object(row)``
    helper as ``for <param> in <row variable>: <helper body>`` at the end of the function, which is
    the one link the guard cannot make (a helper's row parameter to the query it is called with);
  * every ``_build_impl_serial`` / ``_build_impl_model`` helper, which the guard never visits
    because of their names, although ``build()`` delegates the whole lookup to them.

It also runs one check the guard does not make at all: a ``.label("..._is_source")`` taken from a
``.response`` column, or a ``.label("..._is_response")`` taken from a ``.source`` column. The guard
compares *names* read against *names* selected, so a label attached to the wrong column passes it.

Nothing is executed; the factory modules are only parsed.
"""

import ast
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Datastore.tests.test_factory_select_columns import (  # noqa: E402
    FACTORY_DIR,
    analyse_build,
)

HELPERS = {"_build_impl_serial", "_build_impl_model"}


def inline_nested_row_helper(fn: ast.FunctionDef):
    """Return a copy of ``fn`` with its nested one-argument helper inlined as a for-loop."""
    fn = copy.deepcopy(fn)
    nested = [n for n in fn.body if isinstance(n, ast.FunctionDef)]
    if not nested:
        return fn, None
    helper = nested[0]
    param = helper.args.args[0].arg
    # the row variable: the name the comprehension that calls the helper iterates over
    row_var = None
    for node in ast.walk(fn):
        if isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Call):
            if isinstance(node.elt.func, ast.Name) and node.elt.func.id == helper.name:
                row_var = node.generators[0].iter.id
    if row_var is None:
        return fn, None
    fn.body = [n for n in fn.body if n is not helper]
    loop = ast.For(
        target=ast.Name(id=param, ctx=ast.Store()),
        iter=ast.Name(id=row_var, ctx=ast.Load()),
        body=helper.body,
        orelse=[],
        lineno=helper.lineno,
    )
    fn.body.insert(len(fn.body) - 1, loop)
    ast.fix_missing_locations(fn)
    return fn, (helper.name, param, row_var)


def mislabelled_flags(fn: ast.FunctionDef):
    out = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "label"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.func.value, ast.Attribute)
        ):
            label = node.args[0].value
            column = node.func.value.attr
            if (label.endswith("is_response") and column == "source") or (
                label.endswith("is_source") and column == "response"
            ):
                out.append((node.lineno, ast.unparse(node)))
    return out


def main():
    print("Functions the guard does not visit, analysed with its own analyse_build()\n")
    for path in sorted(FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name not in HELPERS | {"read_batch", "build"}:
                    continue
                flags = mislabelled_flags(fn)
                if fn.name == "build" and not flags:
                    continue
                site = f"{path.name}:{cls.name}.{fn.name}@{fn.lineno}"
                if fn.name == "build":
                    for lineno, text in flags:
                        print(f"[label/column mismatch] {site} line {lineno}: {text}")
                    continue
                target, link = (
                    inline_nested_row_helper(fn)
                    if fn.name == "read_batch"
                    else (fn, None)
                )
                findings = analyse_build(target, path.name, cls.name)
                print(
                    site
                    + (
                        f"  (inlined {link[0]}({link[1]}) over {link[2]})"
                        if link
                        else ""
                    )
                )
                if not findings:
                    print("   no row reads the analyser can see")
                for f in findings:
                    if f.spec is None:
                        state = "UNDECIDABLE (query not resolved)"
                    elif not f.decidable:
                        state = (
                            f"UNDECIDABLE (dynamic columns: {len(f.spec.unresolved)})"
                        )
                    elif f.missing:
                        state = f"DEFECT: reads not selected = {sorted(f.missing)}"
                    else:
                        state = "clean"
                    print(
                        f"   row '{f.row_var}' bound at line {f.lineno}: {len(f.reads)} reads -> {state}"
                    )
                for lineno, text in flags:
                    print(f"   [label/column mismatch] line {lineno}: {text}")

    # the module-level OneLoopIntegral.read_batch is not a method of any factory
    path = FACTORY_DIR / "OneLoopIntegral.py"
    tree = ast.parse(path.read_text())
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "read_batch":
            target, link = inline_nested_row_helper(fn)
            findings = analyse_build(target, path.name, "<module>")
            print(f"{path.name}:<module-level>.read_batch@{fn.lineno}  link={link}")
            for f in findings:
                print(
                    f"   row '{f.row_var}': missing={sorted(f.missing)} decidable={f.decidable}"
                )
            if not findings:
                print(
                    "   no row reads resolved: make_object(row_data) is defined but never called, "
                    "and the function has no return statement"
                )


if __name__ == "__main__":
    main()
