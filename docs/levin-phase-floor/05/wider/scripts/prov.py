"""levin-phase-floor prompt 05 section 6 (the wider re-measurement): provenance shared by every
script. Adapted from prompt 04's ``scripts/prov.py`` (docs/levin-phase-floor/04/scripts/prov.py,
a record, not edited).

Two trees take part (prompt 05 section 6):
- P_TREE, the parent of the code commit, ``e648951``: the git worktree
  ``/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951`` (detached). Nothing is written there.
- C_TREE, the code commit: the main checkout, at the dispatch SHA ``53084dc``, whose code is the
  code commit ``120fdfe``'s (``53084dc`` adds only records under ``prompts/``). It also holds the
  records, the store, ``var/`` and the registry.

Every output header records the SHA-256 of the script that wrote it and both trees' HEADs and
``git status --short``. Read-only git only (rev-parse, status).
"""

import datetime
import hashlib
import os
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDER = HERE.parent  # var/levin-phase-floor-05/wider
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE

P_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951")
P_SHA = "e6489519ee662434d1f8e430e1c172246678ac8c"
C_TREE = MAIN
C_SHA = "53084dcb14f7835e22fe8ffde8c0ac09c20a6fca"  # dispatch SHA; its code is 120fdfe's
CODE_COMMIT = "120fdfe3"
TREES = {"p": (P_TREE, P_SHA), "c": (C_TREE, C_SHA)}

PYTHON = MAIN / "venv" / "bin" / "python"
STORE_DIR = MAIN / "var" / "datastores"
PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"
CENSUS = MAIN / "var" / "levin-phase-floor-01" / "out" / "groups_01.csv"
A3V2_DIGEST = "0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_state(tree):
    """(HEAD, git status --short) of a tree, read-only."""
    head = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(tree), "status", "--short"], capture_output=True, text=True
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
