"""levin-phase-floor prompt 06e, section 2: the deliberate breakages (i)-(iii), each applied to
AdaptiveLevin/levin_quadrature.py by an exact text replacement, AdaptiveLevin.tests.test_levin_end_nodes
run in a subprocess, then reverted. The module's SHA-256 is checked equal to the clean file's after
every revert and at the end.

  (i)   the end nodes unpinned: chebyshev_matrices() returns the affine map's nodes;
  (ii)  Delta_k set to zero: _chebyshev_node_displacements() returns zeros;
  (iii) theta'_k replaced by the region's minimum |theta'| at every fine node.

Run from the repository root, with the module clean (the working tree's version is the clean copy):
PYTHONPATH=. ./venv/bin/python <this script>. Writes ../out/breakages.txt. No Ray, no datastore.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

OUT = prov.OUT / "breakages.txt"
MODULE = prov.LEVIN
TEST = "AdaptiveLevin.tests.test_levin_end_nodes"

BREAKAGES = {
    "(i) the end nodes unpinned": ("    x[0] = b\n    x[-1] = a\n", ""),
    "(ii) Delta_k set to zero": (
        "    return delta, err_abs\n",
        "    return np.zeros_like(delta), err_abs\n",
    ),
    "(iii) theta'_k replaced by the region's minimum |theta'|": (
        "    theta_second_fine = np.matmul(D_fine, theta_prime_fine)\n",
        "    theta_prime_fine = np.full_like(\n"
        "        theta_prime_fine, np.min(np.fabs(theta_prime_fine))\n"
        "    )\n"
        "    theta_second_fine = np.matmul(D_fine, theta_prime_fine)\n",
    ),
}


def main():
    clean = MODULE.read_text()
    clean_sha = prov.sha256(MODULE)
    lines = prov.header(
        __file__,
        f"breakages (i)-(iii), each applied to {MODULE.relative_to(prov.REPO)}, {TEST} run, then"
        " reverted",
    )
    lines.append(f"# clean module sha256 {clean_sha}")
    env = dict(os.environ, PYTHONPATH=".")
    try:
        for name, (old, new) in BREAKAGES.items():
            assert clean.count(old) == 1, name
            MODULE.write_text(clean.replace(old, new))
            result = subprocess.run(
                [sys.executable, "-m", "unittest", TEST],
                cwd=prov.REPO,
                env=env,
                capture_output=True,
                text=True,
            )
            MODULE.write_text(clean)
            assert prov.sha256(MODULE) == clean_sha
            stderr = result.stderr.splitlines()
            lines.append(f"== {name}")
            lines.append(f"   {stderr[-1] if stderr else '(no output)'}")
            failing = [l for l in stderr if l.startswith(("FAIL:", "ERROR:"))]
            lines.append(f"   failing tests and subtests: {len(failing)}")
            for line in failing:
                lines.append(f"     {line}")
            messages = [
                l for l in stderr if l.startswith(("AssertionError", "AttributeError"))
            ]
            for line in messages[:12]:
                lines.append(f"       {line[:240]}")
            if len(messages) > 12:
                lines.append(f"       ... {len(messages) - 12} more")
    finally:
        MODULE.write_text(clean)
    assert prov.sha256(MODULE) == clean_sha
    lines.append(f"module restored, sha256 {prov.sha256(MODULE)}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
