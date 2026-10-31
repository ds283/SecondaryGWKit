"""levin-phase-floor prompt 06h, section 2: the deliberate breakages (i)-(iii), each applied to
LiouvilleGreen/three_bessel_integrals.py by an exact text replacement,
LiouvilleGreen.tests.test_group_phase_rounding run in a subprocess, then reverted. The module's
SHA-256 is checked equal to the clean file's after every revert and at the end. 06e's breakages.py,
with its module, test and breakages changed.

  (i)   p_lo dropped from sin_cos's small angle: test 1 must fail;
  (ii)  the abscissa term dropped from the declaration (eps_exp |dtheta/du| set to zero in
        _declared_error_parts, so in theta_abserr): test 3 must fail;
  (iii) the array path of _two_product_error returning zeros, unlike the scalar path: test 6 must
        fail.

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
MODULE = prov.THREE_BESSEL
TEST = "LiouvilleGreen.tests.test_group_phase_rounding"

BREAKAGES = {
    "(i) p_lo dropped from the small angle": (
        "        small = small + _two_product_error(self.K, x, big)\n",
        "",
    ),
    "(ii) the abscissa term dropped from the declaration": (
        "        abscissa = EXP_RELERR * ((K_hat_x + K_lo_x) + log_deriv_magnitude)\n",
        "        abscissa = 0.0 * ((K_hat_x + K_lo_x) + log_deriv_magnitude)\n",
    ),
    "(iii) the array path's p_lo set to zero": (
        "    b = np.asarray(b, dtype=float)\n    a_high, a_low = _two_product_split(a)\n",
        "    return np.zeros_like(np.asarray(product, dtype=float))\n"
        "    b = np.asarray(b, dtype=float)\n    a_high, a_low = _two_product_split(a)\n",
    ),
}


def main():
    clean = MODULE.read_text()
    clean_sha = prov.sha256(MODULE)
    lines = prov.header(
        __file__,
        f"breakages (i)-(iii), each applied to {MODULE.relative_to(prov.REPO)}, {TEST} run,"
        " then reverted",
        inputs=(MODULE, prov.REPO / "LiouvilleGreen" / "tests" / "test_group_phase_rounding.py"),
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
            lines.append(f"== {name}")
            summary = [
                line
                for line in result.stderr.split("\n")
                if line.startswith(("OK", "FAILED", "Ran "))
            ]
            lines += [f"   {line}" for line in summary]
            failing = [
                line
                for line in result.stderr.split("\n")
                if line.startswith(("FAIL:", "ERROR:"))
            ]
            lines.append(f"   failing tests and subtests: {len(failing)}")
            lines += [f"     {line}" for line in failing]
            assertions = [
                line for line in result.stderr.split("\n") if "Error:" in line
            ]
            lines += [f"       {line[:400]}" for line in assertions[:6]]
    finally:
        MODULE.write_text(clean)
    lines.append(f"# module sha256 at the end {prov.sha256(MODULE)} (clean: {clean_sha})")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
