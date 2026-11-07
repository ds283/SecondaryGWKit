"""levin-phase-floor 07 section 2: the proposed references (reference.py) against the prompt's,
on every case where the prompt's is practical, and the proposed references' cost on every case.

- Every case of ``reference.cases()`` is computed by the proposed reference (``closed_R`` for
  (R), ``contour_C`` for (C)), timed.
- Where the prompt's reference, made to converge (``reference.literal``: ``mpmath.quad`` at 30
  digits on pieces of at most pi of phase, split at every zero of theta' and at x_0), takes
  minutes rather than hours -- (R) at c = 1e4 (both x_s) and at c = 1e6 (x_s = 0.38...), and (C)
  at b = 1e4 (every lambda) -- it is computed too, and the difference recorded.

Writes ``out/validate_reference.json``.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import reference as R  # noqa: E402
from prov import OUT, header  # noqa: E402


def proposed(case):
    t0 = time.perf_counter()
    if case["family"] == "R":
        value, omitted = R.closed_R(case["c"], case["x_s"]), None
    else:
        value, omitted = R.contour_C(case["b"], case["a"])
    return value, omitted, time.perf_counter() - t0


def literal_practical(case):
    if case["family"] == "R":
        return case["c"] == 1.0e4 or (
            case["c"] == 1.0e6 and case["x_s"] == R.XS_FLOATS[0]
        )
    return case["b"] == 1.0e4


def main():
    out = {"header": header(Path(__file__), "07 section 2: validating the reference")}
    rows = []
    for case in R.cases():
        value, omitted, wall = proposed(case)
        row = dict(case)
        row["proposed"] = mp.nstr(value, R.REPORT_DPS)
        row["proposed_wall_s"] = wall
        if omitted is not None:
            row["contour_omitted_bound"] = mp.nstr(omitted, 3)
        if literal_practical(case):
            if case["family"] == "R":
                family = R.family_R(case["c"], case["x_s"])
            else:
                family = R.family_C(case["b"], case["a"])
            t0 = time.perf_counter()
            lit, pieces = R.literal(*family)
            row["literal"] = mp.nstr(lit, R.REPORT_DPS)
            row["literal_pieces"] = pieces
            row["literal_wall_s"] = time.perf_counter() - t0
            with mp.workdps(R.WORKING_DPS):
                diff = value - lit
                row["proposed_minus_literal"] = mp.nstr(diff, 5)
                row["relative"] = mp.nstr(abs(diff / lit), 5)
        print(json.dumps(row), flush=True)
        rows.append(row)
    out["rows"] = rows
    path = OUT / "validate_reference.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
