"""
levin-phase-floor prompt 07c, Part B, section 3 as amended by section 9: the deliberate breakages (i)-(vii). Each is
applied to AdaptiveLevin/levin_quadrature.py as text replacements (each must match exactly once), the module holding
section 3's tests, AdaptiveLevin/tests/test_levin_chase_stop.py, is run, the failing tests are recorded, and the
module's original bytes are written back; after each, `git status --porcelain` of AdaptiveLevin/ and LiouvilleGreen/
is checked equal to what it was at the start, and the module's SHA-256 against the one it had at the start.
Writes out/breakages_summary.txt (and out/breakages/breakage_<name>.txt, each run's unittest output).

  (i)   _LEVIN_CHASE_STOP_LIMIT set to infinity                                  -> (b) and (c) must fail
  (ii)  _LEVIN_CHASE_NOISE_FACTOR set to infinity (every checked leaf at the
        noise level)                                                             -> (a) and (c) must fail
  (iii) the stop not reaching leaves made after it (the end of _check_leaf()
        no longer stops a leaf whose root has stopped)                           -> (c) must fail
  (iv)  children not taking their parent's root                                  -> (b) or (c) must fail
  (v)   the innermost refused leaf as the root (each refused leaf roots its own
        subtree)                                                                 -> (d) must fail
  (vi)  bisections counted below any untrusted leaf, not only a refused one
        (a leaf the trust test refuses roots a subtree too)                      -> (c) or (f) must fail
  (vii) a stopped untrusted leaf keeping its quarters' value and the trusted
        declaration (max(residual, delta1) / (1 - r) if r < 1; max(residual,
        delta1) if not, the trusted form without its factor)                     -> (e) must fail

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07c/code/breakages_07c.py
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
MODULE = "AdaptiveLevin.tests.test_levin_chase_stop"


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
            "_LEVIN_CHASE_STOP_LIMIT = 6\n",
            "_LEVIN_CHASE_STOP_LIMIT = float('inf')\n",
        )
    ],
    "ii": [
        (
            "_LEVIN_CHASE_NOISE_FACTOR = 4.0\n",
            "_LEVIN_CHASE_NOISE_FACTOR = float('inf')\n",
        )
    ],
    "iii": [
        (
            "            leaf.seq\n        ):\n            _stop_chase_leaf(leaf)\n",
            "            leaf.seq\n        ):\n            pass\n",
        )
    ],
    "iv": [
        (
            "            chase_root[left.seq] = chase_root_seq\n"
            "            chase_root[right.seq] = chase_root_seq\n",
            "            pass\n",
        )
    ],
    "v": [
        (
            "            chase_root.setdefault(leaf.seq, leaf.seq)\n",
            "            chase_root[leaf.seq] = leaf.seq\n",
        )
    ],
    "vi": [
        (
            "        if refused_by_rate_rule:\n            chase_root.setdefault(leaf.seq, leaf.seq)\n",
            "        if not trusted:\n            chase_root.setdefault(leaf.seq, leaf.seq)\n",
        )
    ],
    "vii": [
        (
            "        if leaf.region._phase_limited:\n            return\n"
            "        _floor_limit_stopped_window_leaf(leaf)\n        num_chase_stopped_leaves += 1\n",
            "        if leaf.region._phase_limited:\n            return\n"
            "        region = leaf.region\n"
            "        if region._trusted is False and leaf.grandchildren is not None:\n"
            "            (LL, LR), (RL, RR) = leaf.grandchildren\n"
            "            ratio = region._convergence_ratio\n"
            "            _remove_error(leaf)\n"
            "            region._declared_resolution = (\n"
            "                max(region.abserr, region._delta1) / (1.0 - ratio)\n"
            "                if ratio < 1.0\n"
            "                else max(region.abserr, region._delta1)\n"
            "            )\n"
            "            region._trusted = None\n"
            "            region._phase_limited = True\n"
            "            _restore_error(leaf)\n"
            "            _set_value(\n"
            "                leaf, (LL['value'] + LR['value']) + (RL['value'] + RR['value']), 'quarters'\n"
            "            )\n"
            "            leaf.dataL = None\n"
            "            leaf.dataR = None\n"
            "            leaf.grandchildren = None\n"
            "            _push_candidate(leaf)\n"
            "            num_chase_stopped_leaves += 1\n"
            "            return\n"
            "        _floor_limit_stopped_window_leaf(leaf)\n        num_chase_stopped_leaves += 1\n",
        )
    ],
}


def main():
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
        try:
            res = subprocess.run(
                [str(REPO / "venv/bin/python"), "-m", "unittest", MODULE],
                cwd=REPO,
                env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
            )
        finally:
            LQ.write_bytes(original)
        assert status() == status0, name
        assert hashlib.sha256(LQ.read_bytes()).hexdigest() == sha0, name
        failing = re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", res.stderr, re.M)
        last = [
            l for l in res.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))
        ]
        line = (
            f"({name}) status restored; failing: "
            f"{sorted(set(d.split('.')[-2] + '.' + m for m, d in failing))} | "
            f"{' | '.join(last)}"
        )
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


if __name__ == "__main__":
    main()
