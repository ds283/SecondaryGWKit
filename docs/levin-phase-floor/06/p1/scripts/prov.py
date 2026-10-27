"""levin-phase-floor 06, finding P1 (the two R13/R14 rows whose G+q+r group moved): provenance
shared by every p1 script. Copied from prompt 06's ``wider/scripts/prov.py`` (sha256
9a8beb6d..., a record, not edited). Changed here, each change marked "p1":
- ``HERE``/``P1``: this copy lives in ``var/levin-phase-floor-06/p1/scripts``;
- the trees are named for the stores they read: ``r13`` (the worktree at ``9f66622``, which built
  R13) and ``r14`` (the worktree at ``f5cee4c``, whose code is ``dc9d28c``'s, which built R14),
  in place of 06's ``t5`` and ``s``;
- the stores are the rehearsal stores R13 and R14 (``STORE_BY_TREE``), each read only by its own
  tree; A3 v2, its scratch copy and 06's runs are not named;
- ``tree_state`` is unchanged (``--no-optional-locks``: read-only git).

Every output header records the SHA-256 of the script that wrote it and both trees' HEADs and
``git status --short``.
"""

import datetime
import hashlib
import os
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1 = HERE.parent  # p1: var/levin-phase-floor-06/p1
OUT = P1 / "out"
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE

# p1: the trees, named for the store each built and reads
R13_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-9f66622")
R13_SHA = "9f66622d5ced82de9a6ec0c20baa9a2e715c6d27"
R14_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-f5cee4c")
R14_SHA = "f5cee4c7a76e755119d923341f204f139b29be7e"  # its code is dc9d28c's
TREES = {"r13": (R13_TREE, R13_SHA), "r14": (R14_TREE, R14_SHA)}

PYTHON = MAIN / "venv" / "bin" / "python"
STORE_DIR = MAIN / "var" / "datastores"
# p1: each tree reads its own rehearsal store, and only that one
STORE_STEM_BY_TREE = {"r13": "levin-phase-floor-06-r13", "r14": "levin-phase-floor-06-r14"}

# p1: the two rows, (shard, serial) in each store (the finding's table)
ROWS = {
    "row1": {"label": "(k, q, r) = (3.162e5, 3.162e5, 3.162e5)", "r13": (0, 10096), "r14": (0, 10193)},
    "row2": {"label": "(k, q, r) = (1e5, 3.162e5, 3.162e5)", "r13": (3, 70), "r14": (3, 270)},
}


def store_files(tree_name):
    stem = STORE_STEM_BY_TREE[tree_name]
    return [STORE_DIR / f"{stem}.sqlite"] + [
        STORE_DIR / f"{stem}-shard{i:04d}.sqlite" for i in range(4)
    ]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_state(tree):
    """(HEAD, git status --short) of a tree, read-only."""
    head = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(tree), "status", "--short"],
        capture_output=True,
        text=True,
    ).stdout
    return head, status


def header(script_path, title, inputs=()):
    """Provenance header lines (each starting with '# ')."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# {title}",
        f"# written {now} on {socket.gethostname()}",
        f"# script {os.path.basename(str(script_path))} sha256 {sha256(script_path)}",
    ]
    for name, (tree, _sha) in TREES.items():
        head, status = tree_state(tree)
        lines.append(
            f"# tree {name}: {tree} HEAD {head} (status --short: "
            f"{'empty' if status.strip() == '' else repr(status)})"
        )
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
