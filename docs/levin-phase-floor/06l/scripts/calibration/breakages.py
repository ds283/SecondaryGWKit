"""levin-phase-floor prompt 06g, section 2: the deliberate breakages (i)-(v), each applied to its
module by exact text replacements, the test that must catch it run in a subprocess, then reverted.
Each module's SHA-256 is checked equal to the clean file's after every revert and at the end. 06h's
breakages.py, with its modules, tests and breakages changed.

  (i)   a trusted leaf returns V_P (its own value, though labelled "quarters"): test 1 must fail;
  (ii)  the call carries sum |E| again (a fallback leaf's error is its whole floor, and |sum E| is
        dropped from the budget, the stop and abserr): test 4 must fail;
  (iii) the declared term left in the premise's ratio (the halving test compares whole floors):
        test 7's first case must fail;
  (iv)  r_trust = 1: test 3 must fail;
  (v)   Route A with K rounded once (the group's own K in place of the exact sum): test 8's Ci case
        must fail.

Run from the repository root, with the modules clean (the working tree's versions are the clean
copies): PYTHONPATH=. ./venv/bin/python <this script>. Writes ../out/breakages.txt. No Ray, no
datastore.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

OUT = prov.OUT / "breakages.txt"
LEVIN = prov.LEVIN
FILON = prov.REPO / "tools" / "filon_reference.py"

BREAKAGES = [
    (
        "(i) a trusted leaf returns V_P",
        LEVIN,
        [
            (
                '            _set_value(leaf, quarters_sum, "quarters")\n',
                '            _set_value(leaf, own_value, "quarters")\n',
            )
        ],
        "AdaptiveLevin.tests.test_levin_leaf_values.TestTrustedLeafReturnsQuarters"
        ".test_trusted_leaf_returns_quarters",
    ),
    (
        "(ii) the call carries sum |E| again",
        LEVIN,
        [
            (
                "        if self._phase_err_less_node is None:\n"
                "            return self._combine(self._phase_err)\n",
                "        if True:\n            return self._combine(self._phase_err)\n",
            ),
            (
                "        return value_sum.total(), error_sum.total(), abs(node_sum.total())\n",
                "        return value_sum.total(), error_sum.total(), 0.0\n",
            ),
            (
                "    return val, abserr_total, math.fsum(node_errors)\n",
                "    return val, abserr_total, 0.0\n",
            ),
        ],
        "AdaptiveLevin.tests.test_levin_node_sum.TestSignedNodeSum.test_signed_node_sum",
    ),
    (
        "(iii) the declared term left in the premise's ratio",
        LEVIN,
        [
            (
                "                floor_limited = _roundoff_exit_premise_holds(\n"
                "                    region.phase_err_roundoff,\n"
                "                    _solve_floor_roundoff(leaf.dataL),\n"
                "                    _solve_floor_roundoff(leaf.dataR),\n",
                "                floor_limited = _roundoff_exit_premise_holds(\n"
                "                    region.phase_err,\n"
                '                    leaf.dataL.get("phase_err", 0.0) or 0.0,\n'
                '                    leaf.dataR.get("phase_err", 0.0) or 0.0,\n',
            )
        ],
        "AdaptiveLevin.tests.test_levin_roundoff_exit.TestRoundoffExitPremise06g"
        ".test_declared_term_excluded",
    ),
    (
        "(iv) r_trust = 1",
        LEVIN,
        [("_LEVIN_TRUST_MAX_RATIO = 0.25\n", "_LEVIN_TRUST_MAX_RATIO = 1.0\n")],
        "AdaptiveLevin.tests.test_levin_leaf_values.TestSlowLeaves",
    ),
    (
        "(v) Route A with K rounded once",
        FILON,
        [
            (
                "        self.K_exact = sum(\n"
                "            (Fraction(s) * Fraction(c) for s, c in zip(g.signs, g.coefficients)),\n"
                "            Fraction(0),\n"
                "        )\n",
                "        self.K_exact = Fraction(float(g.K))\n",
            )
        ],
        "tools.tests.test_filon_reference.TestFilonReference.test_ci_closed_form",
    ),
]


def main():
    clean = {path: path.read_text() for path in (LEVIN, FILON)}
    clean_sha = {path: prov.sha256(path) for path in (LEVIN, FILON)}
    lines = prov.header(
        __file__,
        "breakages (i)-(v), each applied, its catching test run, then reverted",
        inputs=(
            LEVIN,
            FILON,
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_leaf_values.py",
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_node_sum.py",
            prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_roundoff_exit.py",
            prov.REPO / "tools" / "tests" / "test_filon_reference.py",
        ),
    )
    for path in (LEVIN, FILON):
        lines.append(f"# clean {path.relative_to(prov.REPO)} sha256 {clean_sha[path]}")
    env = dict(os.environ, PYTHONPATH=".")
    caught = []
    try:
        for name, path, replacements, test in BREAKAGES:
            broken = clean[path]
            for old, new in replacements:
                assert broken.count(old) == 1, (name, old)
                broken = broken.replace(old, new)
            path.write_text(broken)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "-v", test],
                    cwd=prov.REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                )
            finally:
                path.write_text(clean[path])
            assert prov.sha256(path) == clean_sha[path]
            lines.append(f"== {name}: {path.relative_to(prov.REPO)}, {len(replacements)}"
                         f" replacement(s); test {test}")
            stderr = result.stderr.split("\n")
            lines += [f"   {line}" for line in stderr if line.startswith(("OK", "FAILED", "Ran "))]
            lines += [
                f"   {line}"
                for line in stderr
                if line.startswith(("FAIL:", "ERROR:", "AssertionError"))
            ]
            caught.append((name, result.returncode != 0))
    finally:
        for path in (LEVIN, FILON):
            path.write_text(clean[path])
            assert prov.sha256(path) == clean_sha[path]
    lines.append("")
    for name, was_caught in caught:
        lines.append(f"{name}: {'CAUGHT' if was_caught else 'NOT CAUGHT'}")
    lines.append(f"modules restored: {all(prov.sha256(p) == clean_sha[p] for p in clean_sha)}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
