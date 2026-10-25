"""levin-phase-floor prompt 04: provenance shared by every script.

Two trees take part (prompt 04 section 4 step 3):
- MAIN, the main checkout, holds the records, the store, var/ and the registry;
- WORKTREE, a git worktree detached at 98e68d3, holds the code that runs.

``git_heads()`` reads both (03's ``git_head()`` read one tree only).  Every output header records
the SHA-256 of the script that wrote it and both HEADs.  Read-only git only.
"""

import datetime
import hashlib
import os
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]  # var/levin-phase-floor-04/scripts -> repository root
WORKTREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3")
WORKTREE_SHA = "98e68d3e1b60e6de4378b7d9aded3a120d6088a9"
DISPATCH_SHA = "c8ca1559fcc53fdb88998f170257635431fb85ca"
OUTDIR = HERE.parent  # var/levin-phase-floor-04
STORE = MAIN / "var" / "datastores" / "handover-A3-v2-lambdacdm.sqlite"
RUN03 = MAIN / "var" / "runs" / "levin-phase-floor-03-instrument-20261025T194950"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _head(tree):
    return subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _dirty(tree):
    out = subprocess.run(
        ["git", "-C", str(tree), "status", "--short"], capture_output=True, text=True
    ).stdout
    return "clean" if out.strip() == "" else "DIRTY"


def git_heads():
    """(worktree HEAD, worktree status, main HEAD, main status)."""
    return _head(WORKTREE), _dirty(WORKTREE), _head(MAIN), _dirty(MAIN)


def header(script_path, title, inputs=()):
    """Provenance header lines (each starting with '# ')."""
    wt, wts, mh, mhs = git_heads()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# {title}",
        f"# written {now} on {socket.gethostname()}",
        f"# script {os.path.basename(str(script_path))} sha256 {sha256(script_path)}",
        f"# code: worktree {WORKTREE} HEAD {wt} ({wts})",
        f"# records: main checkout {MAIN} HEAD {mh} ({mhs})",
    ]
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
