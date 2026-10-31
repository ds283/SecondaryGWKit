"""levin-phase-floor prompt 06b section 5, Gate W0's first clause: the parent's production code is
06a's. New in 06b.

``git diff --stat e05c34f <code commit's parent> -- . ':!docs' ':!prompts'`` must show only:
- ``CLAUDE.md``;
- files whose ASTs without docstrings equal ``e05c34f``'s;
- test modules that the copied harness does not import, checked against every module the harness
  imported from the parent tree in a run at the parent (MODULES_JSON, written by ``w0.py check``
  from W0's parent workers' ``bye`` messages: every module in ``sys.modules`` whose file is in the
  parent worktree).

A file's AST "without docstrings" is ``ast.parse`` of its source at each SHA (``git show``), with
the leading string-constant expression statement removed from the body of the module and of every
class and (async) function, compared by ``ast.dump`` (no line or column attributes; comments are
not in the AST). Read-only git (``--no-optional-locks``).

Usage: python -B w0_clause1.py MODULES_JSON
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, P_SHA, TREES, header  # noqa: E402

E05 = "e05c34f2fcc6ef20fc3323cb3a6f81d130c5a01c"  # 06a's tree h


def git(*args):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(MAIN), *args], capture_output=True, text=True, check=True
    ).stdout


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


def ast_key(src, name):
    return ast.dump(strip_docstrings(ast.parse(src, filename=name)), include_attributes=False)


def main():
    modules_json = Path(sys.argv[1])
    mods = json.loads(modules_json.read_text())
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06b: Gate W0 clause 1", [modules_json]):
        print(l)
    rng = [E05, P_SHA, "--", ".", ":!docs", ":!prompts"]
    print(f"\n$ git diff --stat {E05[:7]} {P_SHA[:7]} -- . ':!docs' ':!prompts'")
    print(git("diff", "--stat", *rng))
    files = [l.split("\t") for l in git("diff", "--name-status", *rng).splitlines() if l.strip()]
    print(f"{len(files)} files")
    print(f"\nthe harness's imports from the parent tree ({TREES['p'][0]}): {len(mods)} modules")
    tests_imported = sorted(m for m in mods if ".tests" in m)
    print(f"  of them in a tests package: {tests_imported}")
    ok = True
    print("\n== each changed file")
    for st, path in files:
        if path == "CLAUDE.md":
            print(f"  {st} {path}: CLAUDE.md -- allowed")
            continue
        if not path.endswith(".py") or st not in ("M",):
            print(f"  {st} {path}: NOT A MODIFIED .py FILE -- not allowed")
            ok = False
            continue
        a = git("show", f"{E05}:{path}")
        b = git("show", f"{P_SHA}:{path}")
        same = ast_key(a, path) == ast_key(b, path)
        modname = path[:-3].replace("/", ".")
        imported = modname in mods
        is_test = "/tests/" in path and Path(path).name.startswith("test_")
        if same:
            print(f"  {st} {path}: AST without docstrings EQUAL to {E05[:7]}'s -- allowed (imported by the harness: {imported})")
        elif is_test and not imported:
            print(f"  {st} {path}: AST differs; a test module, {modname}, NOT imported by the harness -- allowed")
        else:
            print(f"  {st} {path}: AST differs; test module {is_test}; imported {imported} -- NOT ALLOWED")
            ok = False
    print(f"\n== W0 clause 1: {'HOLDS' if ok else 'FAILS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
