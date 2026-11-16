"""
levin-phase-floor prompt 07b, Part B, section 3 as amended by section 9: the deliberate breakages (i)-(vi). Each is
applied to AdaptiveLevin/levin_quadrature.py as a text replacement (each must match exactly once), the module holding
section 3's tests, AdaptiveLevin/tests/test_levin_rate_rules.py, is run (for (ii) with the LiouvilleGreen test of
section 10 beside it), the failing tests are recorded, and the module's original bytes are written back; after each,
`git status --porcelain` of AdaptiveLevin/ and LiouvilleGreen/ is checked equal to what it was at the start, and at
the end the module's SHA-256 against the one it had at the start.
Writes out/breakages_summary.txt (and out/breakages/breakage_<name>.txt, each run's unittest output).

  (i)   _LEVIN_EDGE_RATE_RATIO_MAX set to infinity                          -> (a) and (b) must fail
  (ii)  _LEVIN_LEAF_RATE_RANGE_MAX set to infinity                          -> (a) and (c) must fail, and (section
        10.3) LiouvilleGreen's test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy, through its stall
        assertion
  (iii) the two rules swapped between touching and non-touching leaves      -> (a) must fail
  (iv)  the edge's neighbour taken as the grid's far end (grid[0] at lo,
        grid[-1] at hi)                                                     -> (a) must fail
  (v)   the rules applied to Clenshaw-Curtis leaves as well                 -> (e) must fail
  (vi)  a refused leaf at the depth cap counted as trusted (in converged,
        and so in its declaration and value)                                -> (d) must fail

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07b/code/breakages_07b.py
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
MODULE = "AdaptiveLevin.tests.test_levin_rate_rules"
LG_TEST = (
    "LiouvilleGreen.tests.test_three_bessel.TestPhaseGroups."
    "test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy"
)
EXTRA = {"ii": [LG_TEST]}


def status():
    return subprocess.run(
        ["git", "status", "--porcelain", "--", "AdaptiveLevin", "LiouvilleGreen"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout


B = {
    "i": [
        (
            "_LEVIN_EDGE_RATE_RATIO_MAX = 2.0\n",
            "_LEVIN_EDGE_RATE_RATIO_MAX = float('inf')\n",
        )
    ],
    "ii": [
        (
            "_LEVIN_LEAF_RATE_RANGE_MAX = 10.0\n",
            "_LEVIN_LEAF_RATE_RANGE_MAX = float('inf')\n",
        )
    ],
    "iii": [("    if window_edge_ends:\n", "    if not window_edge_ends:\n")],
    "iv": [
        (
            "            neighbour = grid[-2] if edge_x == lo else grid[1]\n",
            "            neighbour = grid[0] if edge_x == lo else grid[-1]\n",
        )
    ],
    "v": [
        (
            "        if trusted and not leaf.is_direct and rate_fns is not None:\n",
            "        if trusted and rate_fns is not None:\n",
        )
    ],
    "vi": [
        (
            "            if refused:\n                trusted = False\n",
            "            if refused and leaf.depth < depth_max:\n                trusted = False\n",
        )
    ],
}
original = LQ.read_bytes()
sha0 = hashlib.sha256(original).hexdigest()
status0 = status()
report = []
for name, reps in B.items():
    text = original.decode()
    for a, b in reps:
        assert text.count(a) == 1, (name, a)
        text = text.replace(a, b)
    LQ.write_text(text)
    res = subprocess.run(
        [str(REPO / "venv/bin/python"), "-m", "unittest", MODULE] + EXTRA.get(name, []),
        cwd=REPO,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    LQ.write_bytes(original)
    assert status() == status0, name
    assert hashlib.sha256(LQ.read_bytes()).hexdigest() == sha0, name
    failing = re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", res.stderr, re.M)
    last = [
        l for l in res.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))
    ]
    line = f"({name}) status restored; failing: {sorted(set(f'{m}.{t}' for t, m in failing))} | {' | '.join(last)}"
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
