"""
levin-phase-floor prompt 07d, Part B, section 3 as amended by section 9.4: the deliberate breakages (i)-(vi). Each is
applied to AdaptiveLevin/levin_quadrature.py as text replacements (each must match exactly once), the module holding
section 3's tests, AdaptiveLevin/tests/test_levin_cc_estimate.py, is run, the failing tests are recorded, and the
module's original bytes are written back (not `git checkout --`, which would discard the uncommitted change under
test); after each, the module's SHA-256 is checked against the one it had at the start, and `git status --short` of
the repository against what it was at the start.
Writes out/breakages_summary.txt (and out/breakages/breakage_<name>.txt, each run's unittest output).

  (i)   the noise term removed (_CC_OWN_NOISE_KAPPA = 0)                            -> (a)'s noise case or (b) must fail
  (ii)  the noise term added to abserr_direct only, not to the two floors          -> (b) must fail
  (iii) the pair restored as the estimate, |CC_{4N-3} - CC_{2N-1}|                 -> (a)'s jump case must fail
  (iv)  CC_{2N-1} returned in place of CC_{4N-3}                                    -> (a) must fail
  (v)   the threshold set back to 6 pi                                              -> (c) must fail
  (vi)  the window's count fed the old estimate |CC_{2N-1} - CC_N| and the
        (2N - 1)-point resasc                                                       -> (e) must fail

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07d/code/breakages_07d.py
No Ray and no datastore.
"""

import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LQ = REPO / "AdaptiveLevin/levin_quadrature.py"
OUT = Path(__file__).resolve().parent / "out"
(OUT / "breakages").mkdir(parents=True, exist_ok=True)
MODULE = "AdaptiveLevin.tests.test_levin_cc_estimate"
FINE_VALUE = "float(np.dot(_cc_weights(x_span, fine_order), integrand[::2]))"


def status():
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True
    ).stdout


B = {
    "i": [("_CC_OWN_NOISE_KAPPA = 4.0\n", "_CC_OWN_NOISE_KAPPA = 0.0\n")],
    "ii": [
        (
            "    round_off_err = floor + noise\n",
            "    round_off_err = floor\n",
        ),
        (
            "    round_off_err_less_node = floor_less_node + noise\n",
            "    round_off_err_less_node = floor_less_node\n",
        ),
    ],
    "iii": [
        (
            "            float(np.dot(_cc_weights(x_span, estimate_order), integrand_estimate))\n",
            f"            {FINE_VALUE}\n",
        )
    ],
    "iv": [('        "value": value,\n', f'        "value": {FINE_VALUE},\n')],
    "v": [
        (
            "CC_GATE_PHASE_VARIATION = 10.0 * np.pi\n",
            "CC_GATE_PHASE_VARIATION = 6.0 * np.pi\n",
        )
    ],
    "vi": [
        (
            '        "resasc": resasc,\n',
            '        "resasc": float(np.sum(np.fabs(_cc_weights(x_span, fine_order))'
            " * np.fabs(integrand[::2] - "
            + FINE_VALUE
            + " / float(np.sum(_cc_weights(x_span, fine_order)))))),\n"
            '        "cc_abserr_2n1": float(np.fabs('
            + FINE_VALUE
            + " - float(np.dot(_cc_weights(x_span,"
            " chebyshev_order), integrand[::4])))),\n",
        ),
        (
            "                _cc_leaf_unresolved(left.region.abserr, left.region.resasc),\n"
            "                _cc_leaf_unresolved(right.region.abserr, right.region.resasc),\n",
            '                _cc_leaf_unresolved(leaf.dataL.get("cc_abserr_2n1", left.region.abserr), left.region.resasc),\n'
            '                _cc_leaf_unresolved(leaf.dataR.get("cc_abserr_2n1", right.region.abserr), right.region.resasc),\n',
        ),
    ],
}
EXPECT = {
    "i": "(a)'s noise case or (b)",
    "ii": "(b)",
    "iii": "(a)'s jump case",
    "iv": "(a)",
    "v": "(c)",
    "vi": "(e)",
}


def main():
    original = LQ.read_bytes()
    sha0 = hashlib.sha256(original).hexdigest()
    status0 = status()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    lines = [
        f"script sha256 {script_sha}; git HEAD {head}; levin_quadrature.py sha256 {sha0} (with this commit's change)",
        f"tests: {MODULE}",
    ]
    for name, edits in B.items():
        text = original.decode()
        for old, new in edits:
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        try:
            LQ.write_text(text)
            proc = subprocess.run(
                ["./venv/bin/python", "-m", "unittest", "-v", MODULE],
                cwd=REPO,
                env={"PYTHONPATH": "."},
                capture_output=True,
                text=True,
            )
        finally:
            LQ.write_bytes(original)
        assert hashlib.sha256(LQ.read_bytes()).hexdigest() == sha0
        assert status() == status0
        (OUT / "breakages" / f"breakage_{name}.txt").write_text(proc.stderr)
        failed = sorted(
            set(
                m.group(1) + (f" [{m.group(2)}]" if m.group(2) else "")
                for m in re.finditer(
                    r"^(?:FAIL|ERROR): (test_\w+) \([\w.]+\)(?: \((.*)\))?$",
                    proc.stderr,
                    re.M,
                )
            )
        )
        summary = re.findall(r"^(Ran .*|OK.*|FAILED.*)$", proc.stderr, re.M)
        lines.append(
            f"({name}) expected to fail: {EXPECT[name]}; {' '.join(summary)}; failing: "
            + ("; ".join(failed) if failed else "NONE")
        )
    lines.append(
        f"module restored after each: sha256 {sha0}; git status --short as at the start after each"
    )
    (OUT / "breakages_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
