"""levin-phase-floor prompt 07c section 6, Gate Y0's first clause (the census agent; new in 07c, after 07b's
ast_check_07b.py, a record, not edited, whose form it keeps): is the parent's code 07b's (S0'), and with the parent's
AdaptiveLevin/levin_quadrature.py installed in tree h, is the code that executes 07b's? Read only.

1. S0': every file that differs between 07b's code commit 5e19c4b6 and 07c's code commit's parent cff4237f outside docs/
   and prompts/ (git diff --name-only), with what it is. Expected: CLAUDE.md alone (not code).
2. Each .py file of (1), if any, at 5e19c4b6 and at cff4237f (git show) parsed with ast, every docstring removed, and
   ast.dump of the two compared (07b's method). Expected: none to check.
3. Every file that differs between 07b's census tree 0a103b5a and h (cc3368ce) outside docs/ and prompts/, with what it
   is: levin_quadrature.py (replaced by the parent's file in the K0 run at the parent), a test module (never imported by
   the harness), CLAUDE.md, or one of (2)'s AST-checked modules.
4. Whether any code in h reads ``__doc__`` (a grep over the tree's .py files outside tests, docs and prompts).

Which of these files the K0 workers actually loaded is u0a_imports_07c.py's (after the K0 run).
Writes OUT_TXT (provenance header, then the findings). Exits 1 unless S0' lists only the expected files and every .py file
of (2) is AST-equal.

Usage: python -B ast_check_07c.py OUT_TXT
"""

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CODE_SHA_07B, H07B_SHA, H_SHA, H_TREE, PARENT_SHA_07C, header, sha256  # noqa: E402

EXPECTED_S0 = {"CLAUDE.md"}


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(H_TREE), *args], capture_output=True, text=True,
                          check=True).stdout


def strip_docstrings(tree):
    removed = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(
                    body[0].value.value, str):
                name = getattr(node, "name", "<module>")
                removed.append((type(node).__name__, name, body[0].value.value))
                node.body = body[1:] or [ast.Pass()]
    return tree, removed


def main():
    out_p = Path(sys.argv[1])
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07c: Gate Y0 (1), the parent's code is 07b's (S0', AST check)")
    ok = True
    s0 = git("diff", "--name-only", CODE_SHA_07B, PARENT_SHA_07C, "--", ".", ":!docs", ":!prompts").split()
    lines.append(f"# S0': git diff --name-only {CODE_SHA_07B[:8]} {PARENT_SHA_07C[:8]} -- . ':!docs' ':!prompts': {len(s0)} files")
    for n in s0:
        lines.append(f"  {n}: {'expected (CLAUDE.md, not code)' if n in EXPECTED_S0 else 'UNEXPECTED'}")
    ok &= set(s0) <= EXPECTED_S0
    lines.append(f"# S0' lists only CLAUDE.md: {set(s0) <= EXPECTED_S0}")
    lines.append("# the build imports none of S0''s files: " + str(all(not n.endswith('.py') for n in s0)))
    for m in sorted(n for n in s0 if n.endswith(".py")):
        src_a = git("show", f"{CODE_SHA_07B}:{m}")
        src_p = git("show", f"{PARENT_SHA_07C}:{m}")
        ta, _ra = strip_docstrings(ast.parse(src_a))
        tp, _rp = strip_docstrings(ast.parse(src_p))
        eq = ast.dump(ta, include_attributes=False) == ast.dump(tp, include_attributes=False)
        ok &= eq
        lines.append(f"# {m}: AST without docstrings equal: {eq}")
    names = git("diff", "--name-only", H07B_SHA, H_SHA, "--", ".", ":!docs", ":!prompts").split()
    lines.append(f"# files differing between 07b's census tree {H07B_SHA[:8]} and h {H_SHA[:8]} outside docs/ and prompts/: "
                 f"{len(names)}")
    for n in names:
        role = ("replaced in the K0 run at the parent by the parent's file (variant_loader); h's own at the code"
                if n == "AdaptiveLevin/levin_quadrature.py"
                else "a test module (not imported by the harness)" if "/tests/" in n
                else "not code" if n == "CLAUDE.md"
                else "docstrings only: AST checked above" if n in EXPECTED_S0 else "UNEXPECTED")
        ok &= role != "UNEXPECTED"
        lines.append(f"  {n}: {role}")
    lq_p = git("show", f"{PARENT_SHA_07C}:AdaptiveLevin/levin_quadrature.py")
    lq_b = git("show", f"{CODE_SHA_07B}:AdaptiveLevin/levin_quadrature.py")
    lines.append(f"# AdaptiveLevin/levin_quadrature.py at {PARENT_SHA_07C[:8]} sha256 {hashlib.sha256(lq_p.encode()).hexdigest()}, "
                 f"at {CODE_SHA_07B[:8]} {hashlib.sha256(lq_b.encode()).hexdigest()}: byte-identical {lq_p == lq_b}")
    ok &= lq_p == lq_b
    grep = subprocess.run(["grep", "-rn", "--include=*.py", "__doc__", "."], cwd=str(H_TREE), capture_output=True,
                          text=True).stdout.splitlines()
    grep = [g for g in grep if not g.startswith(("./docs/", "./prompts/")) and "/tests/" not in g]
    lines.append(f"# '__doc__' read in h's .py files outside docs/, prompts/ and tests: {len(grep)} lines")
    lines += [f"  {g}" for g in grep[:40]]
    lines.append(f"# verdict: {'HOLDS' if ok else 'FAILS'}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
