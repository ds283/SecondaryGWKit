"""levin-phase-floor prompt 07a section 6, Gate W0 (the census agent; new in 07a): with the parent's
AdaptiveLevin/levin_quadrature.py installed in tree h, is the code that executes the parent's? Read only.

1. Every file that differs between the parent dadd63a0 and h (4bf50d34) outside docs/ and prompts/ is listed
   (git diff --name-only), with what it is.
2. LiouvilleGreen/bessel_phase.py and LiouvilleGreen/three_bessel_integrals.py: the module at dadd63a0 (git show) and
   h's file are parsed with ast, every docstring removed (the first statement of the module and of each class,
   function and async function body, when it is a string constant; a body left empty gets ``pass``), and ast.dump of
   the two compared. Equal means the two compile to the same code apart from ``__doc__``. The removed docstrings that
   differ are listed by their node.
3. Whether any code in h reads ``__doc__`` (a grep over the tree's .py files outside tests, docs and prompts), since a
   docstring then could reach a computed value.

Writes OUT_TXT (provenance header, then the findings). Exits 1 unless both modules are AST-equal.

Usage: python -B ast_lg_check_07a.py OUT_TXT
"""

import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_SHA, H_TREE, PARENT_SHA_07A, header, sha256  # noqa: E402

MODULES = ("LiouvilleGreen/bessel_phase.py", "LiouvilleGreen/three_bessel_integrals.py")


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
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07a: Gate W0, the parent's code in tree h (AST check)",
                   [H_TREE / m for m in MODULES])
    ok = True
    names = git("diff", "--name-only", PARENT_SHA_07A, H_SHA, "--", ".", ":!docs", ":!prompts").split()
    lines.append(f"# files differing between {PARENT_SHA_07A[:8]} and {H_SHA[:8]} outside docs/ and prompts/: {len(names)}")
    for n in names:
        role = ("replaced in the K0 run by the parent's file (variant_loader)" if n == "AdaptiveLevin/levin_quadrature.py"
                else "a test module (not imported by the harness)" if "/tests/" in n
                else "docstrings only: AST checked below" if n in MODULES else "UNEXPECTED")
        ok &= role != "UNEXPECTED"
        lines.append(f"  {n}: {role}")
    for m in MODULES:
        src_p = git("show", f"{PARENT_SHA_07A}:{m}")
        src_h = (H_TREE / m).read_text()
        tp, rp = strip_docstrings(ast.parse(src_p))
        th, rh = strip_docstrings(ast.parse(src_h))
        eq = ast.dump(tp, include_attributes=False) == ast.dump(th, include_attributes=False)
        ok &= eq
        lines.append(f"# {m}: sha256 at {PARENT_SHA_07A[:8]} (git show) "
                     f"{__import__('hashlib').sha256(src_p.encode()).hexdigest()}, in h {sha256(H_TREE / m)}; "
                     f"text equal {src_p == src_h}; AST without docstrings equal: {eq}")
        dp = {(k, n): s for k, n, s in rp}
        dh = {(k, n): s for k, n, s in rh}
        diff = sorted(k for k in set(dp) | set(dh) if dp.get(k) != dh.get(k))
        lines.append(f"  docstrings removed: {len(rp)} at the parent, {len(rh)} in h; differing: {diff}")
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
