"""Y1 references (levin-phase-floor 06c, Mac Studio): provenance shared by every script.

Written for this task (dispatch.md); modelled on the census's scripts/prov.py (a record, read
only). Read-only git only: ``git --no-optional-locks rev-parse HEAD`` and ``status --short``.
"""

import datetime
import hashlib
import os
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
Y1 = HERE.parent  # var/levin-phase-floor-06c/y1-references
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE

T17 = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c")
T17_SHA = "fef401c2614da3dbbf1b4261edd31f5f7f80cdbf"
T18 = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6")
T18_SHA = "e5764a6708757cb11704b95bcd66825baab4ba29"
TREES = {"R17": (T17, T17_SHA), "R18": (T18, T18_SHA)}

PYTHON = MAIN / "venv" / "bin" / "python"
ORIG_DIR = MAIN / "var" / "datastores"
COPY_DIR = Y1 / "store_copies"
STORE_NAMES = {"R17": "levin-phase-floor-06c-r17", "R18": "levin-phase-floor-06c-r18"}


def store_files(store, directory):
    name = STORE_NAMES[store]
    return [Path(directory) / f"{name}.sqlite"] + [
        Path(directory) / f"{name}-shard{i:04d}.sqlite" for i in range(4)
    ]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_state(tree):
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


def which_tree():
    """The tree whose code this process imports (by AdaptiveLevin's file), or None."""
    try:
        import AdaptiveLevin.levin_quadrature as LQ
    except Exception:
        return None
    for name, (t, _s) in TREES.items():
        if LQ.__file__.startswith(str(t) + os.sep):
            return name
    return None


def header(script_path, title, inputs=()):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# {title}",
        f"# host {socket.gethostname()}",
        f"# written {now} (UTC)",
        f"# script {script_path} sha256 {sha256(script_path)}",
        f"# PYTHONPATH {os.environ.get('PYTHONPATH')}",
    ]
    for name, (tree, sha) in TREES.items():
        head, status = tree_state(tree)
        lines.append(
            f"# tree {name}: {tree} HEAD {head} (expected {sha}: {head == sha}) "
            f"git status --short: {'empty' if status.strip() == '' else repr(status)}"
        )
    mhead, mstatus = tree_state(MAIN)
    lines.append(
        f"# main checkout {MAIN} HEAD {mhead} git status --short: "
        f"{'empty' if mstatus.strip() == '' else repr(mstatus)}"
    )
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
