"""levin-phase-floor 07c Part C, prompt section 5 item 1 (S0'), and the tree's code clauses. After 07b's rehearsal's
s0prime.py (f7e33882), rewritten for 07c's S0', which has no AST clause.

S0' as written: ``git diff --stat 5e19c4b6 <code commit's parent> -- . ':!docs' ':!prompts'`` lists nothing the build
imports. The code commit is 67b5af74; its parent is read from git (expected cff4237f). "Nothing the build imports" is
scored as: no listed path is a Python source (``*.py``), ``requirements.txt``, a ``*.toml``, or under ``config/``; the
files listed are recorded, and the dispatch's expectation (``CLAUDE.md`` alone) is recorded beside the verdict.

Also recorded, not part of S0': the code commit's diff outside docs and prompts (expected: only
AdaptiveLevin/levin_quadrature.py and AdaptiveLevin/tests/); the dispatch SHA's diff from the code commit outside docs
and prompts (expected empty); R31's tree 0a103b5a to the dispatch under ComputeTargets, ObjectFactories,
LiouvilleGreen, RunDrivers, main.py, tools, config, CosmologyModels, requirements.txt (expected empty); the SHA-256 of
the tree's levin_quadrature.py (expected d141a80b...).

Git is read with --no-optional-locks only. Writes OUT (argv[1]) as JSON with this script's SHA-256 and the tree's
HEAD. Exit 0 iff S0' holds and the tree clauses hold.
Usage: python -B s0prime.py OUT
"""

import datetime
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c")
DISPATCH = "cc3368ce69d42fd6871d6264fd30efa65dd00cea"
CODE07B = "5e19c4b6"
CODE07C = "67b5af74"
PARENT_EXPECTED = "cff4237f"
R31_TREE = "0a103b5a"
LQ_SHA = "d141a80bf0484058a0d7f625926b2e283a4bb64114bc84ccfd81339a6f249be6"
PATHSPEC = ["--", ".", ":!docs", ":!prompts"]
BUILD_PATHS = ["ComputeTargets", "ObjectFactories", "LiouvilleGreen", "RunDrivers", "main.py", "tools", "config",
               "CosmologyModels", "requirements.txt"]


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


def imported_by_build(path):
    return path.endswith(".py") or path.endswith(".toml") or path == "requirements.txt" or path.startswith("config/")


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
    parent = git("rev-parse", f"{CODE07C}^").strip()
    full07b = git("rev-parse", CODE07B).strip()
    res["code_commit"] = git("rev-parse", CODE07C).strip()
    res["code_commit_parent"] = parent
    res["parent_is_expected"] = parent.startswith(PARENT_EXPECTED)
    # S0' exactly as section 5 item 1 writes it.
    res["s0prime_command"] = f"git diff --stat {CODE07B} {parent[:8]} -- . ':!docs' ':!prompts'"
    res["s0prime_stat"] = git("diff", "--stat", full07b, parent, *PATHSPEC)
    names = [l for l in git("diff", "--name-only", full07b, parent, *PATHSPEC).splitlines() if l]
    imported = [n for n in names if imported_by_build(n)]
    res["s0prime"] = {
        "files": names,
        "files_the_build_imports": imported,
        "only_CLAUDE_md_as_the_dispatch_expects": names == ["CLAUDE.md"],
        "holds": imported == [],
    }
    # The tree's clauses (recorded beside S0').
    res["code_commit_stat"] = git("diff", "--stat", parent, CODE07C, *PATHSPEC)
    cfiles = [l for l in git("diff", "--name-only", parent, CODE07C, *PATHSPEC).splitlines() if l]
    res["code_commit_files"] = cfiles
    res["after_code_commit_stat"] = git("diff", "--stat", CODE07C, DISPATCH, *PATHSPEC)
    res["r31_tree_to_dispatch_build_paths_stat"] = git("diff", "--stat", R31_TREE, DISPATCH, "--", *BUILD_PATHS)
    lq = sha(W / "AdaptiveLevin/levin_quadrature.py")
    res["tree"] = {
        "head_is_dispatch": head == DISPATCH,
        "code_commit_touches_only_levin_quadrature_and_its_tests": all(
            f == "AdaptiveLevin/levin_quadrature.py" or f.startswith("AdaptiveLevin/tests/") for f in cfiles
        ),
        "after_code_commit_empty": res["after_code_commit_stat"] == "",
        "r31_tree_to_dispatch_build_paths_empty": res["r31_tree_to_dispatch_build_paths_stat"] == "",
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
