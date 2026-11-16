"""
levin-phase-floor prompt 07b, Part B, section 10.3: LiouvilleGreen/tests/test_three_bessel.py may differ from the
parent's only in TestPhaseGroups.test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy. Both modules are
parsed (the parent's from `git show c768c4ad:`), every module, class and function docstring is removed, and the
top-level statements and class bodies are compared node by node; the ones whose ASTs differ are listed.

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07b/code/ast_check_07b.py
Writes docs/levin-phase-floor/07b/code/out/ast_check_07b.txt. No Ray and no datastore.
"""

import ast
import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PATH = "LiouvilleGreen/tests/test_three_bessel.py"
PARENT = "c768c4ad3f249c2094040601b3b5d86c93659fc0"
OUT = Path(__file__).resolve().parent / "out" / "ast_check_07b.txt"


def strip(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def differing(a, b, path, out):
    if ast.dump(a) == ast.dump(b):
        return
    if (
        isinstance(a, (ast.Module, ast.ClassDef))
        and type(a) is type(b)
        and len(a.body) == len(b.body)
    ):
        for x, y in zip(a.body, b.body):
            differing(x, y, path + [getattr(x, "name", type(x).__name__)], out)
    else:
        out.append(".".join(path))


def main():
    parent = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{PARENT}:{PATH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    current = (REPO / PATH).read_text()
    out = []
    differing(strip(parent), strip(current), [], out)
    lines = [
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"{PATH}: parent {PARENT} sha256 {hashlib.sha256(parent.encode()).hexdigest()}, working tree sha256 "
        f"{hashlib.sha256(current.encode()).hexdigest()}",
        f"nodes whose AST without docstrings differs: {out}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
