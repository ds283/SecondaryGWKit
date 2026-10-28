"""levin-phase-floor prompt 06b (the round-off-limited exit checks its premise): provenance shared
by the calibration scripts in this directory.

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
OUT = CALIBRATION / "out"
REPO = HERE.parents[4]
assert (REPO / "AdaptiveLevin" / "levin_quadrature.py").is_file(), REPO

LEVIN = REPO / "AdaptiveLevin" / "levin_quadrature.py"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def header(script, title, inputs=()) -> list:
    """The provenance lines every output opens with, each prefixed '# '."""
    status = git("status", "--short")
    lines = [
        f"levin-phase-floor 06b: {title}",
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
