"""depth20-tiling (levin-phase-floor 06c follow-up, Mac Studio): provenance shared by this task's
own scripts. Written for this task (dispatch.md); modelled on the census's scripts/prov.py (a
record, read only). Read-only git only: ``git --no-optional-locks rev-parse HEAD`` and
``status --short``."""

import datetime
import hashlib
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent  # var/levin-phase-floor-06c/depth20-tiling
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE
G_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6")
G_SHA = "e5764a6708757cb11704b95bcd66825baab4ba29"
RUN = MAIN / "var" / "runs" / "levin-phase-floor-06c-census-20261029T130214"
CHECKPOINT = RUN / "checkpoint.jsonl"
CENSUS = MAIN / "var" / "levin-phase-floor-06c" / "census"
DEEP_LINES = CENSUS / "out" / "deep_lines_06c.json"
STATE = TASK / "state"
OUT = TASK / "out"
BLOCKS = ["G+q+r", "G+q-r", "G-q+r", "G-q-r", "J1", "J2", "J3", "J4", "Y1", "Y2", "Y3", "Y4"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_state(tree=G_TREE):
    head = subprocess.run(["git", "--no-optional-locks", "-C", str(tree), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "--no-optional-locks", "-C", str(tree), "status", "--short"],
                            capture_output=True, text=True).stdout
    return head, status


def header(script_path, title, inputs=()):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head, status = tree_state()
    lines = [
        f"# {title}",
        f"# host {socket.gethostname()}",
        f"# utc {now}",
        f"# tree g {G_TREE} HEAD {head} (status --short: {'empty' if status.strip() == '' else repr(status)})",
        f"# script {Path(script_path).name} sha256 {sha256(script_path)}",
    ]
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines


def block_of(label):
    """The source block of a call from its notify_label: 'group G+q-r' -> 'G+q-r'; 'analytic J1' -> 'J1'."""
    if label.startswith("analytic "):
        return label.split()[1]
    return label.rsplit("group ", 1)[1]


def hx(x):
    return None if x is None else float(x).hex()
