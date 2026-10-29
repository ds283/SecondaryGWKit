"""levin-phase-floor prompt 06c (global error control in the Levin driver): provenance shared
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
        f"levin-phase-floor 06c: {title}",
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


PARENT_SHA = "3f2a36dcbcef234887b56a7f070889fc59f4c380"


def load_parent():
    """The parent commit's AdaptiveLevin/levin_quadrature.py, read with `git show` into a temporary
    directory and imported under another name; returns (module, its sha256)."""
    import importlib.util
    import tempfile

    source = git("show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py")
    tmp = Path(tempfile.mkdtemp(prefix="lpf06c-parent-"))
    path = tmp / "levin_quadrature_parent_3f2a36d.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(
        "levin_quadrature_parent_3f2a36d", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(path)


class DebugCapture:
    """Collects the driver's DEBUG summary lines (the global budget's 'running test ... returned
    converged' line), so a disagreement between the loop's running test and the returned flag can
    be counted."""

    def __init__(self):
        import logging

        self.records = []
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if "global budget" in message:
                    outer.records.append(message)

        self.handler = _Handler(level=logging.DEBUG)
        self.logger = logging.getLogger("AdaptiveLevin.levin_quadrature")

    def __enter__(self):
        import logging

        self.logger.addHandler(self.handler)
        self._old = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._old)

    def disagreements(self):
        return sum("(DISAGREE)" in m for m in self.records)

    def trust_counts(self):
        """Sums over the captured calls of (checks, passes, found untrusted, returned untrusted,
        returned unchecked), from the trust check's part of the summary line."""
        import re

        totals = [0, 0, 0, 0, 0]
        for m in self.records:
            hit = re.search(
                r"trust check: (\d+) checks in (\d+) passes, (\d+) found untrusted, (\d+)"
                r" returned untrusted, (\d+) returned unchecked",
                m,
            )
            if hit:
                for i in range(5):
                    totals[i] += int(hit.group(i + 1))
        return totals

    def resum_rejections(self):
        total = 0
        for m in self.records:
            tail = m.rsplit("did not ", 1)[-1]
            total += int(tail.split(" ")[0])
        return total
