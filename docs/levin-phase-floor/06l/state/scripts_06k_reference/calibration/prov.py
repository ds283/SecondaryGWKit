"""levin-phase-floor prompt 06k: 06g's calibration prov.py (docs/levin-phase-floor/06g/calibration/scripts/prov.py,
a record, not edited; its copy is in state/scripts_06g_calibration_reference/), changed only where marked
"prompt 06k": REPO is 06k's worktree (this copy is not inside a tree), OUT is
var/levin-phase-floor-06k/out/lg_draws/, every git command is read with --no-optional-locks, and the title
names 06k. 06g's text follows.

levin-phase-floor prompt 06g (the trusted leaf's value and the node sum): provenance shared by the
calibration scripts in this directory. 06h's prov.py, with its title and its parent changed: the
parent here is 06g's dispatch commit, and what is loaded from it is AdaptiveLevin/levin_quadrature.py.

Every output header records the SHA-256 of the script that wrote it, of this file, of
AdaptiveLevin/levin_quadrature.py and of each other input it reads; the git HEAD it ran at; and
whether the tree was dirty (``git status --short``). Read-only git only (rev-parse, status, show).
"""

import datetime
import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parent
OUT = HERE.parents[1] / "out" / "lg_draws"  # prompt 06k (06g: CALIBRATION / "out")
REPO = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060")  # prompt 06k (06g: HERE.parents[4])
assert (REPO / "AdaptiveLevin" / "levin_quadrature.py").is_file(), REPO

LEVIN = REPO / "AdaptiveLevin" / "levin_quadrature.py"

#: 06g's dispatch commit, whose AdaptiveLevin/levin_quadrature.py is 06e's code (f4d8e17).
PARENT_SHA = "4f4f9ed2a31fb7437fcfca01263d260eebaff842"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(
        # prompt 06k: --no-optional-locks (the first 06k edit put this comment on the argument line and
        # commented out check/capture_output/text; corrected after run k0partL's 70_lg_draws failed)
        ["git", "--no-optional-locks", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def header(script, title, inputs=()) -> list:
    """The provenance lines every output opens with, each prefixed '# '."""
    status = git("status", "--short")
    lines = [
        f"levin-phase-floor 06k: {title}",  # prompt 06k (06g: 06g)
        f"written {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"
        f" on {socket.gethostname()}",
        f"python {platform.python_version()} ({sys.executable})",
        f"script {Path(script).name} sha256 {sha256(script)}",
        f"prov.py sha256 {sha256(__file__)}",
        f"AdaptiveLevin/levin_quadrature.py sha256 {sha256(LEVIN)}",
    ]
    for path in inputs:
        lines.append(
            f"input {Path(path).resolve().relative_to(REPO)} sha256 {sha256(path)}"
        )
    lines.append(f"HEAD {git('rev-parse', 'HEAD').strip()}")
    if status.strip():
        lines.append("tree dirty: yes; git status --short:")
        lines += [f"    {line}" for line in status.rstrip("\n").split("\n")]
    else:
        lines.append("tree dirty: no (git status --short empty)")
    lines.append(f"uptime at start: {uptime()}")
    return [f"# {line}" for line in lines]


def load_parent():
    """The parent commit's AdaptiveLevin/levin_quadrature.py, read with `git show` into a temporary
    directory and imported under another name; returns (module, its sha256)."""
    import importlib.util
    import tempfile

    source = git("show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py")
    tmp = Path(tempfile.mkdtemp(prefix="lpf06g-parent-"))
    path = tmp / "levin_quadrature_parent_4f4f9ed.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("levin_quadrature_parent_4f4f9ed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(path)
