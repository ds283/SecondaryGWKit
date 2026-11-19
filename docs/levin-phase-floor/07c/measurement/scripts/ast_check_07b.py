"""levin-phase-floor prompt 07b section 6, Gate U0's first clause (the census agent; new in 07b, after 07a's
ast_lg_check_07a.py, a record, not edited): is the parent's code 07a's (S0'), and with the parent's
AdaptiveLevin/levin_quadrature.py installed in tree h, is the code that executes 07a's? Read only.

1. S0': every file that differs between 07a's code commit 1ec734d7 and 07b's code commit's parent 7b82a0cd outside docs/
   and prompts/ (git diff --name-only), with what it is: CLAUDE.md (not code) and 89c368f2's four files.
2. Each .py file of (1) at 1ec734d7 and at 7b82a0cd (git show) is parsed with ast, every docstring removed (the first
   statement of the module and of each class, function and async function body, when it is a string constant; a body
   left empty gets ``pass``), and ast.dump of the two compared. Equal means the two compile to the same code apart from
   ``__doc__``. The removed docstrings that differ are listed by their node.
3. Every file that differs between 07a's census tree 4bf50d34 and h (0a103b5a) outside docs/ and prompts/, with what it
   is: levin_quadrature.py (replaced by the parent's file in the K0 run at the parent), a test module (never imported by
   the harness), CLAUDE.md, or one of (2)'s AST-checked modules.
4. Whether any code in h reads ``__doc__`` (a grep over the tree's .py files outside tests, docs and prompts), since a
   docstring then could reach a computed value.

Which of these files the K0 workers actually loaded is u0a_imports_07b.py's (after the K0 run).
Writes OUT_TXT (provenance header, then the findings). Exits 1 unless S0' lists only the expected files and every .py file
of (2) is AST-equal.

Usage: python -B ast_check_07b.py OUT_TXT
"""

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CODE_SHA_07A, H07A_SHA, H_SHA, H_TREE, PARENT_SHA_07B, header, sha256  # noqa: E402

EXPECTED_S0 = {"CLAUDE.md", "ComputeTargets/tests/test_quadsource_integral.py", "LiouvilleGreen/bessel_phase.py",
               "LiouvilleGreen/tests/test_three_bessel.py", "LiouvilleGreen/three_bessel_integrals.py"}


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
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07b: Gate U0 (1), the parent's code is 07a's (S0', AST check)")
    ok = True
    s0 = git("diff", "--name-only", CODE_SHA_07A, PARENT_SHA_07B, "--", ".", ":!docs", ":!prompts").split()
    lines.append(f"# S0': git diff --name-only {CODE_SHA_07A[:8]} {PARENT_SHA_07B[:8]} -- . ':!docs' ':!prompts': {len(s0)} files")
    for n in s0:
        lines.append(f"  {n}: {'expected (CLAUDE.md, or one of 89c368f2s four files)' if n in EXPECTED_S0 else 'UNEXPECTED'}")
    ok &= set(s0) <= EXPECTED_S0
    lines.append(f"# S0' lists only CLAUDE.md and 89c368f2's four files: {set(s0) <= EXPECTED_S0}; "
                 f"all five present: {set(s0) == EXPECTED_S0}")
    lines.append("# 89c368f2: " + git("log", "-1", "--format=%H %s", "89c368f2").strip())
    lines.append("# 89c368f2's files: " + " ".join(git("show", "--name-only", "--format=", "89c368f2").split()))
    for m in sorted(n for n in s0 if n.endswith(".py")):
        src_a = git("show", f"{CODE_SHA_07A}:{m}")
        src_p = git("show", f"{PARENT_SHA_07B}:{m}")
        src_h = (H_TREE / m).read_text()
        ta, ra = strip_docstrings(ast.parse(src_a))
        tp, rp = strip_docstrings(ast.parse(src_p))
        eq = ast.dump(ta, include_attributes=False) == ast.dump(tp, include_attributes=False)
        ok &= eq
        lines.append(f"# {m}: sha256 at {CODE_SHA_07A[:8]} {hashlib.sha256(src_a.encode()).hexdigest()}, at "
                     f"{PARENT_SHA_07B[:8]} {hashlib.sha256(src_p.encode()).hexdigest()}, in h {sha256(H_TREE / m)} "
                     f"(h equal to {PARENT_SHA_07B[:8]}: {src_h == src_p}); AST without docstrings equal "
                     f"({CODE_SHA_07A[:8]} vs {PARENT_SHA_07B[:8]}): {eq}")
        da = {(k, n): s for k, n, s in ra}
        dp = {(k, n): s for k, n, s in rp}
        diff = sorted(k for k in set(da) | set(dp) if da.get(k) != dp.get(k))
        lines.append(f"  docstrings removed: {len(ra)} at {CODE_SHA_07A[:8]}, {len(rp)} at {PARENT_SHA_07B[:8]}; differing: {diff}")
    names = git("diff", "--name-only", H07A_SHA, H_SHA, "--", ".", ":!docs", ":!prompts").split()
    lines.append(f"# files differing between 07a's census tree {H07A_SHA[:8]} and h {H_SHA[:8]} outside docs/ and prompts/: "
                 f"{len(names)}")
    for n in names:
        role = ("replaced in the K0 run at the parent by the parent's file (variant_loader); h's own at the code"
                if n == "AdaptiveLevin/levin_quadrature.py"
                else "a test module (not imported by the harness)" if "/tests/" in n
                else "not code" if n == "CLAUDE.md"
                else "docstrings only: AST checked above" if n in EXPECTED_S0 else "UNEXPECTED")
        ok &= role != "UNEXPECTED"
        lines.append(f"  {n}: {role}")
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
