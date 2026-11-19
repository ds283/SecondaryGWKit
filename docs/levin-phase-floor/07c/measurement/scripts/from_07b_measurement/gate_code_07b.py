"""levin-phase-floor 07b Part A, the Gate's code and state clauses. New in 07b.

1. ``git diff --stat 1ec734d7 <dispatch SHA> -- AdaptiveLevin ComputeTargets LiouvilleGreen tools`` lists only
   89c368f2's four files, and each is AST-equal to 1ec734d7's with docstrings removed (module, class and function
   docstrings: a first statement that is a string constant).
2. 07a's census checkpoint has SHA-256 461df2ec...; R30's primary e5f8b443... is not read.
3. The worktree is clean with --ignored; the main checkout is at 28ebaf07, clean.
4. AdaptiveLevin/levin_quadrature.py in the worktree is 4ea1948b... (07a's census tree h's).

Git is read with --no-optional-locks only. Writes OUT (argv[1]) as JSON with this script's SHA-256. Exit 0 iff all hold.
Usage: python -B gate_code_07b.py OUT
"""

import ast
import datetime
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
DISPATCH = "436ed5d4"
CODE07A = "1ec734d7"
DOC_FILES = {
    "ComputeTargets/tests/test_quadsource_integral.py",
    "LiouvilleGreen/bessel_phase.py",
    "LiouvilleGreen/tests/test_three_bessel.py",
    "LiouvilleGreen/three_bessel_integrals.py",
}
CK07A = MAIN / "var/runs/levin-phase-floor-07a-census-20261112T132744/checkpoint.jsonl"
CK07A_SHA = "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"
LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*a, cwd=W):
    return subprocess.run(["git", "--no-optional-locks", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def main():
    out = Path(sys.argv[1])
    res = {"header": {"script": __file__, "script_sha256": sha(__file__), "host": socket.gethostname(),
                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}}
    names = [l for l in git("diff", "--name-only", CODE07A, DISPATCH, "--", "AdaptiveLevin", "ComputeTargets",
                              "LiouvilleGreen", "tools").splitlines() if l]
    res["diff_stat"] = git("diff", "--stat", CODE07A, DISPATCH, "--", "AdaptiveLevin", "ComputeTargets", "LiouvilleGreen", "tools")
    ast_eq = {}
    for n in names:
        a = git("show", f"{CODE07A}:{n}")
        b = git("show", f"{DISPATCH}:{n}")
        ast_eq[n] = strip_docstrings(ast.parse(a)) == strip_docstrings(ast.parse(b))
    res["clause1"] = {"files": names, "only_89c368f2s_four": set(names) == DOC_FILES, "ast_equal_without_docstrings": ast_eq,
                      "holds": set(names) == DOC_FILES and all(ast_eq.values())}
    ck = sha(CK07A)
    res["clause2"] = {"checkpoint": str(CK07A), "sha256": ck, "holds": ck == CK07A_SHA}
    st_w = git("status", "--short", "--ignored")
    head_w = git("rev-parse", "HEAD").strip()
    st_m = git("status", "--short", cwd=MAIN)
    head_m = git("rev-parse", "HEAD", cwd=MAIN).strip()
    res["clause3"] = {"worktree_head": head_w, "worktree_status_ignored": st_w, "main_head": head_m, "main_status": st_m,
                      "holds": head_w.startswith(DISPATCH) and st_w == "" and head_m.startswith("28ebaf07") and st_m == ""}
    lq = sha(W / "AdaptiveLevin/levin_quadrature.py")
    res["clause4"] = {"levin_quadrature_sha256": lq, "holds": lq == LQ_SHA}
    res["holds"] = all(res[f"clause{i}"]["holds"] for i in (1, 2, 3, 4))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["holds"] if isinstance(v, dict) and "holds" in v else v) for k, v in res.items() if k != "header"}, indent=1))
    print("wrote", out, sha(out))
    return 0 if res["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
