"""
Single entry point that regenerates every table and figure in the
AdaptiveLevin performance campaign.

    python -m levin_bench.campaign all        # everything (~40 min)
    python -m levin_bench.campaign tierA      # synthetic problems only
    python -m levin_bench.campaign tierB      # Bessel-product tier only
    python -m levin_bench.campaign figures    # re-draw figures from existing CSVs

Timing fidelity: the frequency ladder (`ladder`) is the only experiment whose
wall-clock numbers are quoted as headline results, and it is always run alone
and serially.  Everything else is run for accuracy / region-structure data, so
concurrent execution is acceptable there.  If you re-run `ladder` for timing,
do not run anything else at the same time.

Requires PYTHONPATH to include the SecondaryGWKit checkout.
"""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def _write(rows, path):
    """
    Write `rows` (a list of dicts) to `path` as CSV.

    Columns are the union of all keys, ordered by first appearance, with
    private keys (leading underscore, e.g. the retained per-region
    diagnostics `_regions`) excluded so that bulky non-scalar payloads stay
    out of the tables.  Returns `path` so callers can `return _write(...)`.
    """
    cols = []
    for r in rows:
        for k in r:
            if not k.startswith("_") and k not in cols:
                cols.append(k)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k in cols})
    return path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

PY = sys.executable

# (module, argv, label, run_alone)
TIER_A = [
    ("levin_bench.problems", [], "validate oracles vs mpmath", False),
    ("levin_bench.sweeps", ["ladder"], "frequency ladder (TIMED - runs alone)", True),
    ("levin_bench.sweeps", ["pareto"], "accuracy-cost Pareto fronts", False),
    ("levin_bench.sweeps", ["tolerance"], "tolerance map", False),
    ("levin_bench.sweeps", ["order"], "Chebyshev order economics", False),
    ("levin_bench.sweeps", ["phase_floor"], "phase evaluation floor", False),
    ("levin_bench.sweeps", ["reduction_cost"], "range-reduction cost", False),
    ("levin_bench.sweeps", ["fidelity"], "estimator fidelity", False),
]

TIER_B = [
    ("levin_bench.bessel_tier", ["truncation"], "Bessel truncation scan", False),
    ("levin_bench.bessel_tier", ["production"], "Bessel production scan", False),
    ("levin_bench.bessel_tier", ["head"], "Bessel head-to-head vs quad", False),
]


def _run(mod, argv, label):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, "-u", "-m", mod, *argv], cwd=ROOT)
    dt = time.time() - t0
    status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"--- {label}: {status} in {dt:.1f}s", flush=True)
    return r.returncode == 0


def main(argv):
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    which = argv[0] if argv else "all"

    jobs = []
    if which in ("all", "tierA"):
        jobs += TIER_A
    if which in ("all", "tierB"):
        jobs += TIER_B

    failed = []
    for mod, args, label, _alone in jobs:
        if not _run(mod, args, label):
            failed.append(label)

    if which in ("all", "figures"):
        if not _run("levin_bench.figures", [], "figures"):
            failed.append("figures")

    print("\n" + "=" * 60)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("campaign complete; tables in results/, figures in figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
