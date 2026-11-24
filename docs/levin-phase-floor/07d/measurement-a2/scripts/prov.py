"""levin-phase-floor prompt 07d Part A' (section 9.2): provenance for every output of this Part. New in Part A'.

Every output's header carries: the writing script's SHA-256, each input's SHA-256, the worktree's HEAD and status, the
main checkout's HEAD and status, the host and the time (README section 5 rule 5). Git is read with --no-optional-locks.
"""

import datetime
import hashlib
import socket
import subprocess
from pathlib import Path

TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3")
TREE_SHA = "c2c57a36"  # the amendment's SHA (prompt section 9.2), code 07c's (67b5af74)
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
ROOT = MAIN / "var" / "levin-phase-floor-07d" / "measurement-a2"
PARTA = MAIN / "var" / "levin-phase-floor-07d" / "measurement"  # Part A's records: read only
TREE_LQ = TREE / "AdaptiveLevin" / "levin_quadrature.py"
TREE_LQ_SHA = "d141a80bf0484058a0d7f625926b2e283a4bb64114bc84ccfd81339a6f249be6"
PY = str(MAIN / "venv" / "bin" / "python")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


def header(title, script, inputs=()):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"# {title}", f"# written {now} on {socket.gethostname()}",
             f"# script {Path(script).name} sha256 {sha256(script)}"]
    for repo, tag in ((TREE, "tree"), (MAIN, "main checkout")):
        st = _git(repo, "status", "--short")
        lines.append(f"# {tag}: {repo} HEAD {_git(repo, 'rev-parse', 'HEAD')} (status --short: {'empty' if not st else 'NOT EMPTY'})")
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
