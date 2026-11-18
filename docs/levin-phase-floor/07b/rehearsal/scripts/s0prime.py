"""levin-phase-floor 07b Part C, prompt section 5 item 1 (S0'), and the tree's code clauses. New in 07b's rehearsal.

S0' as written: ``git diff --stat 1ec734d7 <code commit's parent> -- . ':!docs' ':!prompts'`` lists only CLAUDE.md
and 89c368f2's four files, each AST-equal with docstrings removed. The code commit is 5e19c4b6; its parent is read
from git (expected 7b82a0cd). The docstring stripping is Part A's (07b measurement scripts/gate_code_07b.py,
strip_docstrings, copied verbatim): a first statement that is a string constant is removed from every module, class
and function body.

Also recorded, not part of S0': 89c368f2's file list; the code commit's diff outside docs and prompts; the dispatch
SHA's diff from the code commit outside docs and prompts (expected empty); the SHA-256 of the tree's
levin_quadrature.py (expected 8956c7b6...).

Git is read with --no-optional-locks only. Writes OUT (argv[1]) as JSON with this script's SHA-256 and the tree's
HEAD. Exit 0 iff S0' holds and the tree clauses hold.
Usage: python -B s0prime.py OUT
"""

import ast
import datetime
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5")
DISPATCH = "0a103b5ae0de32b568f44904a5c3dcbfca922a2e"
CODE07A = "1ec734d7"
CODE07B = "5e19c4b6"
PARENT_EXPECTED = "7b82a0cd"
FOUR = {
    "ComputeTargets/tests/test_quadsource_integral.py",
    "LiouvilleGreen/bessel_phase.py",
    "LiouvilleGreen/tests/test_three_bessel.py",
    "LiouvilleGreen/three_bessel_integrals.py",
}
LQ_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"
PATHSPEC = ["--", ".", ":!docs", ":!prompts"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(
        ["git", "--no-optional-locks", *a], cwd=W, capture_output=True, text=True, check=True
    ).stdout


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def main():
    out = Path(sys.argv[1])
    head = git("rev-parse", "HEAD").strip()
    res = {
        "header": {
            "script": __file__,
            "script_sha256": sha(__file__),
            "host": socket.gethostname(),
            "tree": str(W),
            "tree_head": head,
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }
    parent = git("rev-parse", f"{CODE07B}^").strip()
    full07a = git("rev-parse", CODE07A).strip()
    res["code_commit"] = git("rev-parse", CODE07B).strip()
    res["code_commit_parent"] = parent
    res["parent_is_expected"] = parent.startswith(PARENT_EXPECTED)
    res["c89c368f2_files"] = [l for l in git("show", "--name-only", "--format=", "89c368f2").splitlines() if l]
    # S0' exactly as section 5 item 1 writes it.
    res["s0prime_command"] = f"git diff --stat {CODE07A} {parent[:8]} -- . ':!docs' ':!prompts'"
    res["s0prime_stat"] = git("diff", "--stat", full07a, parent, *PATHSPEC)
    names = [l for l in git("diff", "--name-only", full07a, parent, *PATHSPEC).splitlines() if l]
    ast_eq = {}
    for n in sorted(set(names) & FOUR):
        a = git("show", f"{full07a}:{n}")
        b = git("show", f"{parent}:{n}")
        ast_eq[n] = strip_docstrings(ast.parse(a)) == strip_docstrings(ast.parse(b))
    only = set(names) == (FOUR | {"CLAUDE.md"})
    res["s0prime"] = {
        "files": names,
        "only_CLAUDE_md_and_89c368f2s_four": only,
        "ast_equal_without_docstrings": ast_eq,
        "holds": only and len(ast_eq) == 4 and all(ast_eq.values()),
    }
    # The tree's clauses (recorded beside S0').
    res["code_commit_stat"] = git("diff", "--stat", parent, CODE07B, *PATHSPEC)
    res["code_commit_files"] = [l for l in git("diff", "--name-only", parent, CODE07B, *PATHSPEC).splitlines() if l]
    res["after_code_commit_stat"] = git("diff", "--stat", CODE07B, DISPATCH, *PATHSPEC)
    lq = sha(W / "AdaptiveLevin/levin_quadrature.py")
    res["tree"] = {
        "head_is_dispatch": head == DISPATCH,
        "after_code_commit_empty": res["after_code_commit_stat"] == "",
        "levin_quadrature_sha256": lq,
        "levin_quadrature_expected": lq == LQ_SHA,
    }
    res["tree"]["holds"] = all(v for k, v in res["tree"].items() if k != "levin_quadrature_sha256")
    res["holds"] = res["parent_is_expected"] and res["s0prime"]["holds"] and res["tree"]["holds"]
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", out, sha(out))
    return 0 if res["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
