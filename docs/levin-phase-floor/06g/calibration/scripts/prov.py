"""levin-phase-floor prompt 06g (the trusted leaf's value and the node sum): provenance shared by the
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
OUT = CALIBRATION / "out"
REPO = HERE.parents[4]
assert (REPO / "AdaptiveLevin" / "levin_quadrature.py").is_file(), REPO

LEVIN = REPO / "AdaptiveLevin" / "levin_quadrature.py"

#: 06g's dispatch commit, whose AdaptiveLevin/levin_quadrature.py is 06e's code (f4d8e17).
PARENT_SHA = "4f4f9ed2a31fb7437fcfca01263d260eebaff842"


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
        f"levin-phase-floor 06g: {title}",
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
