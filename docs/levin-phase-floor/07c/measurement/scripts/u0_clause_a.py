"""levin-phase-floor prompt 06 section 6, gate U0 clause (a). New in 06.

`git diff --stat 1fc349c <code commit's parent> -- . ':!docs' ':!prompts'` must show CLAUDE.md and
bd38e50's two comment-only files (ComputeTargets/primitive_phase.py,
ComputeTargets/tests/test_g_phase_reanchor.py), and those two files' ASTs, docstrings removed, must
equal 1fc349c's. Read-only git (diff --stat, show), run in the parent worktree with
--no-optional-locks. Prints the stat, each file's AST verdict, and the clause's verdict."""
import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import T5_SHA, T5_TREE, header  # noqa: E402

BASE = "1fc349c"
EXPECTED = {
    "CLAUDE.md",
    "ComputeTargets/primitive_phase.py",
    "ComputeTargets/tests/test_g_phase_reanchor.py",
}


def git(*args):
    r = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(T5_TREE), *args],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"git {args}: {r.stderr}")
    return r.stdout


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def main():
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06 section 6: gate U0, clause (a)"):
        print(l)
    stat = git("diff", "--stat", BASE, T5_SHA, "--", ".", ":!docs", ":!prompts")
    print(f"$ git diff --stat {BASE} {T5_SHA[:7]} -- . ':!docs' ':!prompts'")
    print(stat.rstrip())
    names = set(git("diff", "--name-only", BASE, T5_SHA, "--", ".", ":!docs", ":!prompts").split())
    print(f"files: {sorted(names)}; expected {sorted(EXPECTED)}: {names == EXPECTED}")
    ok = names == EXPECTED
    for path in sorted(EXPECTED - {"CLAUDE.md"}):
        a = ast.dump(strip_docstrings(ast.parse(git("show", f"{BASE}:{path}"))))
        b = ast.dump(strip_docstrings(ast.parse(git("show", f"{T5_SHA}:{path}"))))
        same = a == b
        ok &= same
        print(f"{path}: AST without docstrings equal to {BASE}'s: {same}")
    print(f"== U0 (a): {'HOLDS' if ok else 'FAILS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
