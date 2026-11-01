"""levin-phase-floor prompt 06i: the provenance header every output carries (README Sec 5 rule 5)."""

import hashlib
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06i-6d81fad")
SCRIPTS = Path(__file__).resolve().parent
INPUTS = MAIN / "var/levin-phase-floor-06i/inputs"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(path, *args):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(path), *args],
        capture_output=True,
        text=True,
    ).stdout.strip()


def header(inputs=()):
    import mpmath
    import numpy
    import scipy

    return {
        "host": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "main_checkout_head": _git(MAIN, "rev-parse", "HEAD"),
        "main_checkout_status_lines": len(_git(MAIN, "status", "--short").splitlines()),
        "integrand_tree": str(TREE),
        "integrand_tree_head": _git(TREE, "rev-parse", "HEAD"),
        "integrand_tree_status_lines": len(
            _git(TREE, "status", "--short").splitlines()
        ),
        "scripts_sha256": {
            p.name: sha256(p) for p in sorted(SCRIPTS.glob("*.py")) if p.is_file()
        },
        "inputs_sha256": {str(Path(p).name): sha256(p) for p in inputs},
        "argv": sys.argv,
    }
