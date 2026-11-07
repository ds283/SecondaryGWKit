"""levin-phase-floor 07 section 2: the references (reference.py) against each other and against
the prompt's, on every case where the prompt's is practical, and their cost on every case
(prompt section 9 point 2).

- Every (R) case of ``reference.cases()``, omega = 0 and omega = 1, is computed twice, by
  ``closed_R`` (the error function, at 40 working digits, and again at 60 to show the erf is
  stable) and by ``contour_R``. The two must agree to well below the precision section 2 needs;
  their largest relative disagreement is recorded.
- Every (C) case is computed by ``contour_C``.
- Where the prompt's reference, made to converge (``reference.literal``: ``mpmath.quad`` at 30
  digits on pieces of at most pi of phase, split at every zero of theta' and at x_0), takes
  minutes rather than hours -- (R) at c = 1e4 (every case, omega = 1 included) and at c = 1e6,
  x_s = 0.38..., omega = 0, and (C) at b = 1e4 (every lambda) -- it is computed too, and the
  difference recorded.

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
    extra = {}
    if case["family"] == "R":
        value = R.closed_R(case["c"], case["x_s"], case["omega"])
        extra["closed_wall_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        contour = R.contour_R(case["c"], case["x_s"], case["omega"])
        extra["contour_wall_s"] = time.perf_counter() - t1
        saved = R.WORKING_DPS
        R.WORKING_DPS = 60
        closed60 = R.closed_R(case["c"], case["x_s"], case["omega"])
        R.WORKING_DPS = saved
        with mp.workdps(60):
            extra["contour"] = mp.nstr(contour, R.REPORT_DPS)
            extra["closed_minus_contour_relative"] = float(
                abs((value - contour) / value)
            )
            extra["closed40_minus_closed60_relative"] = float(
                abs((value - closed60) / closed60)
            )
        omitted = None
    else:
        value, omitted = R.contour_C(case["b"], case["a"])
    return value, omitted, time.perf_counter() - t0, extra


def literal_practical(case):
    if case["family"] == "R":
        return case["c"] == 1.0e4 or (
            case["c"] == 1.0e6
            and case["x_s"] == R.XS_FLOATS[0]
            and case["omega"] == 0.0
        )
    return case["b"] == 1.0e4


def main():
    out = {"header": header(Path(__file__), "07 section 2: validating the reference")}
    rows = []
    for case in R.cases():
        value, omitted, wall, extra = proposed(case)
        row = dict(case)
        row["proposed"] = mp.nstr(value, R.REPORT_DPS)
        row["proposed_wall_s"] = wall
        row.update(extra)
        if omitted is not None:
            row["contour_omitted_bound"] = mp.nstr(omitted, 3)
        if literal_practical(case):
            if case["family"] == "R":
                family = R.family_R(case["c"], case["x_s"], case["omega"])
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
    out["R_closed_minus_contour_largest_relative"] = max(
        r["closed_minus_contour_relative"] for r in rows if r["family"] == "R"
    )
    out["R_closed40_minus_closed60_largest_relative"] = max(
        r["closed40_minus_closed60_relative"] for r in rows if r["family"] == "R"
    )
    out["largest_relative_against_literal"] = max(
        float(r["relative"]) for r in rows if "relative" in r
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "R_closed_minus_contour_largest_relative",
                    "R_closed40_minus_closed60_largest_relative",
                    "largest_relative_against_literal",
                )
            }
        )
    )
    path = OUT / "validate_reference.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
