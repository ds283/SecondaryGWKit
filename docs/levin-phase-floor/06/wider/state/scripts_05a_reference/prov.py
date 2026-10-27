"""levin-phase-floor prompt 05a section 6 (the wider re-measurement, with section 9's T1): provenance
shared by every script. Copied from prompt 05's ``scripts/prov.py``
(docs/levin-phase-floor/05/wider/scripts/prov.py, sha256 e8579509..., a record, not edited); changed
here only in its trees and its paths (prompt 05a section 6: "the scripts name 05's paths, its trees
p and c ... Change only those"), each change marked "prompt 05a".

Two trees take part (prompt 05a section 6):
- C_TREE, the code commit's parent ``c4eaefd``, whose code is 05's c: the git worktree
  ``/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd`` (detached). Nothing is written there.
- T_TREE, the code commit: the main checkout, at Part B's dispatch SHA ``149c393``, whose code is
  the code commit ``1fc349c``'s up to ``bd38e50``'s comment-only changes (``2db3f9e`` and
  ``149c393`` add only records under ``prompts/``). It also holds the records, the store, ``var/``
  and the registry.

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
WIDER = HERE.parent  # var/levin-phase-floor-05a/wider  (prompt 05a)
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE

# prompt 05a: the trees c (the code commit's parent, 05's c) and t (the code commit)
C_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd")
C_SHA = "c4eaefda26ccb615fa5864cac877454ba364c304"
T_TREE = MAIN
T_SHA = "149c3938bb9e074df3682f8e511c7b57cd9c55c5"  # Part B's dispatch SHA; its code is 1fc349c's
CODE_COMMIT = "1fc349c"
TREES = {"c": (C_TREE, C_SHA), "t": (T_TREE, T_SHA)}

# prompt 05a: 05's run, T0's reference for c and convergence_kinds.py's --capped-run
RUN05 = MAIN / "var" / "runs" / "levin-phase-floor-05-wider-20261026T020005"
SAMPLE_SHA256 = "73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7"

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
