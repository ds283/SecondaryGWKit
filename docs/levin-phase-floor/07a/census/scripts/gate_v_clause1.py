"""levin-phase-floor prompt 06g section 5, Gate V's git clauses. New in 06g; adapted from 06h's
gate_h_clause1.py (kept unchanged in state/scripts_06h_reference/; its helpers ``git``,
``strip_docstrings`` and ``ast_key`` are copied unchanged). Read-only git (``--no-optional-locks``,
run against the main checkout's object store, which holds every commit named here).

Three clauses, each printed with its ``git diff --stat`` and its file list:

1. **"The integrand is 06h's"** (prompt 06g section 5, Gate V, second bullet): ``git diff --stat
   c1ec3b2 3ae6f3a -- . ':!docs' ':!prompts'`` (06h's census tree against 06g's code commit) touches
   only ``AdaptiveLevin/``, ``tools/``, test files and ``CLAUDE.md``. A path is allowed if it is
   ``CLAUDE.md``, starts with ``AdaptiveLevin/`` or ``tools/``, or is a test file (a ``.py`` file under
   a ``tests/`` directory). Each changed file is also looked up in MODULES_JSON (every module the
   harness imported from the parent tree in Gate V's parent workers, written by ``gate_v.py check``)
   and reported: the ones the census imports are the ones whose change reaches it.
2. **The parent's code is 06h's census's** (what makes "the parent reproduces 06h" a reproduction of
   06h's code): ``git diff --stat c1ec3b2 4f4f9ed -- . ':!docs' ':!prompts'`` shows only ``CLAUDE.md``
   and files whose ASTs without docstrings equal c1ec3b2's (06h's own rule for its parent), each
   looked up in MODULES_JSON.
3. **The census tree's code is the code commit's:** ``git diff --stat 3ae6f3a 1c5336e -- .
   ':!docs' ':!prompts'`` is empty (Part B's dispatch commit adds only records).

Usage: python -B gate_v_clause1.py MODULES_JSON
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CODE_SHA_06G_FULL, G06H_SHA, G_SHA, MAIN, P_SHA, TREES, header  # noqa: E402


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


def changed(a, b):
    rng = [a, b, "--", ".", ":!docs", ":!prompts"]
    print(f"\n$ git diff --stat {a[:7]} {b[:7]} -- . ':!docs' ':!prompts'")
    print(git("diff", "--stat", *rng))
    files = [l.split("\t") for l in git("diff", "--name-status", *rng).splitlines() if l.strip()]
    print(f"{len(files)} files")
    return files


def modname(path):
    return path[:-3].replace("/", ".") if path.endswith(".py") else None


def is_test_file(path):
    parts = path.split("/")
    return path.endswith(".py") and "tests" in parts[:-1]


def main():
    modules_json = Path(sys.argv[1])
    mods = json.loads(modules_json.read_text())
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06g: Gate V, the git clauses", [modules_json]):
        print(l)
    print(f"\nthe harness's imports from the parent tree ({TREES['p'][0]}): {len(mods)} modules")
    print(f"  of them in a tests package: {sorted(m for m in mods if '.tests' in m)}")
    print(f"  of them under AdaptiveLevin: {sorted(m for m in mods if m.startswith('AdaptiveLevin'))}")
    print(f"  of them under tools: {sorted(m for m in mods if m.startswith('tools'))}")

    print("\n== clause 1, the integrand is 06h's: 06h's census tree c1ec3b2 against 06g's code commit 3ae6f3a")
    ok1 = True
    for st, path in changed(G06H_SHA, CODE_SHA_06G_FULL):
        allowed = path == "CLAUDE.md" or path.startswith("AdaptiveLevin/") or path.startswith("tools/") or is_test_file(path)
        imported = modname(path) in mods if modname(path) else False
        print(f"  {st} {path}: {'allowed' if allowed else 'NOT ALLOWED'} (imported by the harness at the parent: {imported})")
        ok1 &= allowed
    print(f"   clause 1: {'HOLDS' if ok1 else 'FAILS'}")

    print("\n== clause 2, the parent's code is 06h's census's: c1ec3b2 against 06g's parent 4f4f9ed")
    ok2 = True
    for st, path in changed(G06H_SHA, P_SHA):
        if path == "CLAUDE.md":
            print(f"  {st} {path}: CLAUDE.md -- allowed")
            continue
        if not path.endswith(".py") or st not in ("M",):
            print(f"  {st} {path}: NOT A MODIFIED .py FILE -- not allowed")
            ok2 = False
            continue
        same = ast_key(git("show", f"{G06H_SHA}:{path}"), path) == ast_key(git("show", f"{P_SHA}:{path}"), path)
        print(f"  {st} {path}: AST without docstrings {'EQUAL -- allowed' if same else 'DIFFERS -- NOT ALLOWED'} "
              f"(imported by the harness: {modname(path) in mods})")
        ok2 &= same
    print(f"   clause 2: {'HOLDS' if ok2 else 'FAILS'}")

    print("\n== clause 3, the census tree's code is the code commit's: 3ae6f3a against 1c5336e")
    files3 = changed(CODE_SHA_06G_FULL, G_SHA)
    ok3 = len(files3) == 0
    print(f"   clause 3: {'HOLDS' if ok3 else 'FAILS'}")

    ok = ok1 and ok2 and ok3
    print(f"\n== Gate V, the git clauses: {'HOLD' if ok else 'DO NOT HOLD'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
