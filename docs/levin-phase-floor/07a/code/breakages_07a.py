"""
levin-phase-floor prompt 07a, Part B, section 3 and section 9.4: the deliberate breakages (i)-(ix). Each is applied
to AdaptiveLevin/levin_quadrature.py as a text replacement (each must match exactly once), the two test modules that
hold sections 3's and 9.4's tests are run, the failing tests are recorded, and the module's original bytes are
written back; at the end its SHA-256 is checked against the one it had at the start. Writes
out/breakages_summary.txt (and out/breakages/breakage_<name>.txt, each run's unittest output).

  (i)    the C2 gate back on the mean-of-nodes estimate               -> tests 2 and 3
  (ii)   Rule A back (skip below SIX_PI on the scan)                   -> test 5 (test 13 (b))
  (iii)  a phase skipped when its rate has no strict sign change       -> test 7
  (iv)   |theta(b) - theta(a)| on every region (no node-partition sum) -> test 4
  (v)    the LU choice left on the old estimate                        -> test 8
  (vi)   the value test back at a fixed 1e-5                           -> test 14 (a)
  (vii)  the skip removed                                              -> test 14 (a)
  (viii) the count on e' instead of the raw estimates (_own_error()
         returning budget_err, the leaf's e')                          -> test 14 (a)
  (ix)   the saturation factor 200 replaced by 1                       -> test 14 (b)
  (x)    a value built from theta["theta"] (section 10.3), in two forms -> the rewritten
         LiouvilleGreen test_the_raw_phase_enters_no_value_or_error:
         x-cc:    the Clenshaw-Curtis rule's sin and cos of the raw phase in place of theta_mod_2pi;
         x-levin: the Levin solve's endpoint basis w0, wk from the raw phase in place of theta_mod_2pi.
(i)-(ix) run AdaptiveLevin's test_levin_phase_variation and test_levin_stationary; (x) runs
LiouvilleGreen's TestLevinConsumption.

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07a/code/breakages_07a.py
No Ray and no datastore.
"""

import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
LQ = REPO / "AdaptiveLevin/levin_quadrature.py"
OUT = Path(__file__).resolve().parent / "out"
(OUT / "breakages").mkdir(parents=True, exist_ok=True)
OLD_EST = "float(np.mean(np.fabs(theta_prime_Cheb)) * np.fabs(grid[0] - grid[-1]))"
MODULES = {
    False: [
        "AdaptiveLevin.tests.test_levin_phase_variation",
        "AdaptiveLevin.tests.test_levin_stationary",
    ],
    True: ["LiouvilleGreen.tests.test_bessel_compatibility.TestLevinConsumption"],
}
B = {
    "x-cc": [
        (
            '        if hasattr(self, "_theta_mod_2pi"):\n            argument = self._theta_mod_2pi(x)\n',
            '        if hasattr(self, "_theta_mod_2pi"):\n            argument = self._theta(x)\n',
        )
    ],
    "x-levin": [
        (
            "            theta0_mod_2pi = self._theta_mod_2pi(grid[-1])\n            thetak_mod_2pi = self._theta_mod_2pi(grid[0])\n",
            "            theta0_mod_2pi = self._theta(grid[-1])\n            thetak_mod_2pi = self._theta(grid[0])\n",
        )
    ],
    "i": [("    if phase_span < SIX_PI:\n", f"    if {OLD_EST} < SIX_PI:\n")],
    "ii": [
        (
            '        if not np.any(finite & (d != 0.0)):\n            skipped_phases.append({"phase": phase_index, "reason": "zero_rate"})\n            continue\n',
            '        both_finite = finite[:-1] & finite[1:]\n        abs_rate = np.abs(np.where(finite, d, 0.0))\n        variation = float(np.sum(np.where(both_finite, 0.5 * (abs_rate[:-1] + abs_rate[1:]) * np.diff(xs), 0.0)))\n        if variation < SIX_PI:\n            skipped_phases.append({"phase": phase_index, "variation": variation})\n            continue\n',
        )
    ],
    "iii": [
        (
            "        if not np.any(finite & (d != 0.0)):\n",
            "        if not (np.any(finite & (d > 0.0)) and np.any(finite & (d < 0.0))):\n",
        )
    ],
    "iv": [
        (
            "        if _changes_sign(theta_prime_Cheb):\n            theta_nodes = theta_Cheb\n",
            "        if False:\n            theta_nodes = theta_Cheb\n",
        )
    ],
    "v": [
        (
            "    if phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN:\n",
            f"    if {OLD_EST} > _LEVIN_DIRECT_SOLVE_PHASE_SPAN:\n",
        )
    ],
    "vi": [
        (
            "    value_tolerance = min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL, rtol)\n",
            "    value_tolerance = _LEVIN_WINDOW_IROFF1_VALUE_RTOL\n",
        )
    ],
    "vii": [
        (
            "    if unresolved_left or unresolved_right:\n        return False\n    value_tolerance",
            "    value_tolerance",
        )
    ],
    "viii": [("    own = region.own_err\n", "    own = region.budget_err\n")],
    "ix": [("_DQK21_SATURATION_FACTOR = 200.0\n", "_DQK21_SATURATION_FACTOR = 1.0\n")],
}
original = LQ.read_bytes()
sha0 = hashlib.sha256(original).hexdigest()
report = []
for name, reps in B.items():
    text = original.decode()
    for a, b in reps:
        assert text.count(a) == 1, (name, a)
        text = text.replace(a, b)
    LQ.write_text(text)
    res = subprocess.run(
        [
            str(REPO / "venv/bin/python"),
            "-m",
            "unittest",
        ]
        + MODULES[name.startswith("x")],
        cwd=REPO,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    LQ.write_bytes(original)
    failing = re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", res.stderr, re.M)
    last = [
        l for l in res.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))
    ]
    line = f"({name}) failing: {sorted(set(f'{m}.{t}' for t, m in failing))} | {' | '.join(last)}"
    print(line, flush=True)
    report.append(line)
    (OUT / "breakages" / f"breakage_{name}.txt").write_text(res.stderr)
assert hashlib.sha256(LQ.read_bytes()).hexdigest() == sha0
print("restored, sha256", sha0)
head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
).stdout.strip()
provenance = [
    f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
    f"git HEAD {head} (levin_quadrature.py modified in the working tree: the code commit's)",
    f"levin_quadrature.py sha256 {sha0}, restored after every breakage",
]
(OUT / "breakages_summary.txt").write_text("\n".join(provenance + report) + "\n")
