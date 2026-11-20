"""The Levin records whose phase_limited changed between R31 and R32 (descriptive; the orchestrator's note at stop 1).

levin-phase-floor 07c, rehearsal agent, R32. New. Reads only tabulate_qsi_r31_r32.py's records CSV (every Levin record
differing in a non-timing field, with both stores' fields) and prints, for each record whose phase_limited differs, its
row, record path and label, both stores' value, abserr, converged, phase_limited and regions, |dv| against each abserr
and against their sum (exactly, in Fractions of the stored doubles), and the four location counts. Writes the same as
JSON.

    python -B phase_limited_flips.py <records csv> <out json>
"""

import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    recs_p, out_p = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    print(f"# input {recs_p} sha256 {sha256(recs_p)}")
    out = []
    n = 0
    for x in csv.DictReader(open(recs_p)):
        n += 1
        if x["phase_limited_31"] == x["phase_limited_32"]:
            continue
        v31, v32 = Fraction(float.fromhex(x["value_31"])), Fraction(float.fromhex(x["value_32"]))
        e31, e32 = Fraction(float.fromhex(x["abserr_31"])), Fraction(float.fromhex(x["abserr_32"]))
        d = abs(v32 - v31)
        rec = {"key": [x["k"], x["q"], x["r"], x["z_response"]], "kind": x["kind"], "record": x["record"], "label": x["label"],
               "fields_differing": x["fields_differing"],
               "value_31": x["value_31"], "value_32": x["value_32"], "abserr_31": x["abserr_31"], "abserr_32": x["abserr_32"],
               "abserr_32_over_31": float(e32 / e31),
               "abs_dv_over_e31": float(d / e31), "abs_dv_over_e32": float(d / e32), "abs_dv_over_sum": float(d / (e31 + e32)),
               "beyond_sum": d > e31 + e32,
               "converged_31": x["converged_31"], "converged_32": x["converged_32"],
               "phase_limited_31": x["phase_limited_31"], "phase_limited_32": x["phase_limited_32"],
               "regions_31": x["regions_31"], "regions_32": x["regions_32"],
               "counts_31": [x[f"{g}_31"] for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")],
               "counts_32": [x[f"{g}_32"] for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")]}
        out.append(rec)
    print(f"records read {n}; phase_limited changed in {len(out)}")
    print("| (k, q, r; z) | record | label | phase_limited | converged | regions | value R31 -> R32 | abserr R31 -> R32 (ratio) | |dv|/e31 | |dv|/e32 | |dv|/(e31+e32) | counts sp/wp/gr/gdm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| ({r['key'][0]}, {r['key'][1]}, {r['key'][2]}; {r['key'][3]}) | {r['record']} | {r['label']} | "
              f"{r['phase_limited_31']} -> {r['phase_limited_32']} | {r['converged_31']} -> {r['converged_32']} | "
              f"{r['regions_31']} -> {r['regions_32']} | `{r['value_31']}` -> `{r['value_32']}` | "
              f"`{r['abserr_31']}` -> `{r['abserr_32']}` ({r['abserr_32_over_31']:.6f}) | {r['abs_dv_over_e31']:.3e} | "
              f"{r['abs_dv_over_e32']:.3e} | {r['abs_dv_over_sum']:.3e} | {'/'.join(r['counts_31'])} -> {'/'.join(r['counts_32'])} |")
    Path(out_p).write_text(json.dumps({"records": out}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
