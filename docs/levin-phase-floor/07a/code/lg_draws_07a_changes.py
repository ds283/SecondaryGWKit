"""
levin-phase-floor prompt 07a, Part B: the LiouvilleGreen draws' changes, call by call. A reader of lg_draws_07a.py's
outputs (out/parent/, out/commit/; that script is not changed after its run, so that its recorded SHA-256 still
names it), written after its report: every Levin call whose `converged` changes between the parent and the commit,
and every three-Bessel result whose honesty (|value - analytic| <= abserr) changes, with what the route
instrument recorded for the calls it holds. Writes out/lg_draws_changes.txt.

Run from the repository root:  ./venv/bin/python docs/levin-phase-floor/07a/code/lg_draws_07a_changes.py
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lg_draws_07a as draws  # noqa: E402

OUT = HERE / "out" / "lg_draws_changes.txt"


def main():
    parent, commit = draws.load("parent"), draws.load("commit")
    lines = [
        "levin-phase-floor 07a Part B: the LiouvilleGreen draws' changes, call by call",
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; "
        f"lg_draws_07a.py sha256 {draws.sha256_of(draws.__file__)}",
        "",
        "1. Levin calls whose converged flag changes (parent -> commit):",
    ]
    for key in sorted(commit):
        for p, c in zip(parent[key]["levin_calls"], commit[key]["levin_calls"]):
            if p["converged"] == c["converged"]:
                continue
            lines.append(
                f"   seed {key[0]} {key[1]} call {c['index']}: x_span {c['x_span']}, rtol {c['rtol']}: "
                f"converged {p['converged']} -> {c['converged']}; abserr {p['abserr']:.4g} -> {c['abserr']:.4g}; "
                f"value moved {abs(c['value'] - p['value']):.3g} ({abs(c['value'] - p['value']) / p['abserr']:.3g} "
                f"of the parent's abserr); regions {p['num_regions']} -> {c['num_regions']}, depth "
                f"{p['max_depth']} -> {c['max_depth']}; windows {p['num_windows']} / {c['num_windows']}; regions "
                f"whose route changed {c['route_changed']}, LU choice {c['lu_changed']}: "
                + "; ".join(
                    f"[{r['span'][0]:.6g}, {r['span'][1]:.6g}] V {r['V']:.5g} estimate {r['estimate']:.5g}"
                    for r in c["route_changes"][:4]
                )
            )
    lines.append("")
    lines.append("2. Three-Bessel results whose honesty changes (parent -> commit):")
    for key in sorted(commit):
        pc, cc = parent[key]["levin_calls"], commit[key]["levin_calls"]
        for p, c in zip(parent[key]["bessel_results"], commit[key]["bessel_results"]):
            if draws.honest(p) == draws.honest(c):
                continue
            lo, hi = c["levin_calls"]
            changed = sum(cl["route_changed"] + cl["lu_changed"] for cl in cc[lo:hi])
            windows = sum(
                pl["num_windows"] + cl["num_windows"]
                for pl, cl in zip(pc[lo:hi], cc[lo:hi])
            )
            lines.append(
                f"   seed {key[0]} {key[1]} {c['evaluator']} {c['mu_nu_sigma']} k,q,s {c['k_q_s']} rtol {c['rtol']}: "
                f"true error {abs(p['value'] - p['analytic']):.4g} -> {abs(c['value'] - c['analytic']):.4g}, "
                f"abserr {p['abserr']:.4g} -> {c['abserr']:.4g}, ratio "
                f"{abs(p['value'] - p['analytic']) / p['abserr']:.3g} -> {abs(c['value'] - c['analytic']) / c['abserr']:.3g}, "
                f"honest {draws.honest(p)} -> {draws.honest(c)}; its calls: regions route or LU changed {changed}, "
                f"windows {windows}"
            )
    lines.append("")
    lines.append(
        "3. Three-Bessel results whose |err| / abserr rises by more than 2x under the commit:"
    )
    for key in sorted(commit):
        for p, c in zip(parent[key]["bessel_results"], commit[key]["bessel_results"]):
            rp = abs(p["value"] - p["analytic"]) / p["abserr"]
            rc = abs(c["value"] - c["analytic"]) / c["abserr"]
            if rc > 2.0 * rp and rc > 0.01:
                lines.append(
                    f"   seed {key[0]} {key[1]} {c['evaluator']} {c['mu_nu_sigma']} rtol {c['rtol']}: ratio "
                    f"{rp:.3g} -> {rc:.3g} (abserr {p['abserr']:.4g} -> {c['abserr']:.4g}, true error "
                    f"{abs(p['value'] - p['analytic']):.4g} -> {abs(c['value'] - c['analytic']):.4g})"
                )
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
