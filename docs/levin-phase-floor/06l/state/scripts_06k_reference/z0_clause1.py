"""levin-phase-floor prompt 06e section 5, Gate Z0's first clause: the parent's production code is
06c's census's. Copied from 06c's x0_clause1.py (kept unchanged in state/scripts_06c_reference/);
changed (prompt 06e) only in the two commits: 06c's Part B dispatch commit ``e5764a6``
(prov.G06C_SHA) and 06e's code commit's parent ``50277f3`` (prov.P_SHA), and the gate's name.
MODULES_JSON is state/Z0/p_tree_modules.json, written by ``z0.py check`` from Z0's parent workers.

(06c's text follows, unchanged; read its 92bc62f as e5764a6, its fef401c as 50277f3, its X0 as Z0.)

levin-phase-floor prompt 06c section 5, Gate X0's first clause: the parent's production code is
06b's census's. New in 06c; adapted from 06b's w0_clause1.py (a record, kept unchanged in
state/scripts_06b_reference/).

``git diff --stat <06b's Part B dispatch commit 92bc62f> <06c's code commit's parent fef401c> -- .
':!docs' ':!prompts'`` must show only ``CLAUDE.md`` and files whose ASTs without docstrings equal
92bc62f's (prompt 06c section 5). Unlike 06b's W0, the clause grants no exemption to test modules
the harness does not import; every changed file is also looked up in MODULES_JSON (every module
the harness imported from the parent tree in X0's parent workers, written by ``x0.py check``), and
reported.

A file's AST "without docstrings" is ``ast.parse`` of its source at each SHA (``git show``), with
the leading string-constant expression statement removed from the body of the module and of every
class and (async) function, compared by ``ast.dump`` (no line or column attributes; comments are
not in the AST). Read-only git (``--no-optional-locks``).

Usage: python -B x0_clause1.py MODULES_JSON
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import G06C_SHA as B06B_SHA, MAIN, P_SHA, TREES, header  # noqa: E402  prompt 06e: e5764a6 under 06c's name


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
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06e: Gate Z0 clause 1", [modules_json]):
        print(l)
    rng = [B06B_SHA, P_SHA, "--", ".", ":!docs", ":!prompts"]
    print(f"\n$ git diff --stat {B06B_SHA[:7]} {P_SHA[:7]} -- . ':!docs' ':!prompts'")
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
        a = git("show", f"{B06B_SHA}:{path}")
        b = git("show", f"{P_SHA}:{path}")
        same = ast_key(a, path) == ast_key(b, path)
        modname = path[:-3].replace("/", ".")
        imported = modname in mods
        if same:
            print(f"  {st} {path}: AST without docstrings EQUAL to {B06B_SHA[:7]}'s -- allowed (imported by the harness: {imported})")
        else:
            print(f"  {st} {path}: AST differs (imported by the harness: {imported}) -- NOT ALLOWED")
            ok = False
    print(f"\n== Z0 clause 1: {'HOLDS' if ok else 'FAILS'}")  # prompt 06e
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
