"""levin-phase-floor 07 section 2: provenance shared by every calibration script.

Every output written by a script in this directory starts with ``header(script)``: the SHA-256 of
the script itself, of every script beside it (its inputs, such as ``reference.py``), and of
``AdaptiveLevin/levin_quadrature.py`` as it was when the script ran, the
git ``HEAD``, whether the tree was dirty (with ``git status --short``), the host, the time, the
load average (``uptime``) and the library versions.
"""

import datetime
import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
OUT = CAL / "out"
MAIN = HERE.parents[4]
LEVIN = MAIN / "AdaptiveLevin" / "levin_quadrature.py"
assert LEVIN.is_file(), LEVIN


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=MAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def header(script, title):
    import mpmath
    import numpy
    import scipy

    status = _git("status", "--short")
    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    return {
        "title": title,
        "script": str(Path(script).resolve().relative_to(MAIN)),
        "script_sha256": sha256(script),
        "levin_quadrature_sha256": sha256(LEVIN),
        "scripts_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "git_head": _git("rev-parse", "HEAD").strip(),
        "tree_dirty": bool(status.strip()),
        "git_status_short": status.splitlines(),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uptime": uptime,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "pid": os.getpid(),
    }
