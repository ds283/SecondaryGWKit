"""
levin-phase-floor prompt 03: before the registered run, check that a row pickle carries what
step 7 needs. Loads the pickle, takes every group Levin call, and for its deepest regions
recomputes the discrepancy the driver computed, from the unpickled PhaseGroup alone, through the
driver's own single-region routine AdaptiveLevin.levin_quadrature._adaptive_levin_subregion:
|v0 - vL - vR| for a Levin region and abserr_direct for a Clenshaw-Curtis one. The recomputed
value must equal the recorded region.abserr exactly.

Usage: PYTHONPATH=. ./venv/bin/python .../smoke_check_pickle.py PICKLE [PICKLE ...]
"""

import importlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import load_record  # noqa: E402

LQ = importlib.import_module("AdaptiveLevin.levin_quadrature")
QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")


def driver_discrepancy(group, a, b, order, rtol):
    f = [group.f_sin, group.f_cos]
    basis = LQ._Basis_SinCos(group.levin_theta(include_deriv=QSI.LEVIN_USE_THETA_DERIV))
    lazy = LQ._LazyUUID()
    data = LQ._adaptive_levin_subregion(
        (a, b), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    if data.get("is_direct", False):
        return "direct", data["abserr_direct"]
    c = (b + a) / 2.0
    dL = LQ._adaptive_levin_subregion(
        (a, c), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    dR = LQ._adaptive_levin_subregion(
        (c, b), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    return "Levin", float(np.fabs(data["value"] - (dL["value"] + dR["value"])))


def main():
    bad = 0
    for path in sys.argv[1:]:
        rec = load_record(path)
        print(
            f"{path}: {rec['name']} M0 {rec['m0_holds']}; {len(rec['levin_calls'])} Levin calls"
        )
        builds = {b["index"]: b for b in rec["build_calls"]}
        for call in rec["levin_calls"]:
            if call["group_label"] is None:
                continue
            group = [
                g
                for g in builds[call["build_index"]]["groups"]
                if g.label == call["group_label"]
            ][0]
            regions = call["result"]["regions"]
            deepest = sorted(regions, key=lambda r: -r.depth)[:3]
            for reg in deepest:
                kind, d = driver_discrepancy(
                    group, reg.start, reg.end, call["chebyshev_order"], call["rtol"]
                )
                same = d == reg.abserr and kind == reg.type
                bad += not same
                print(
                    f"   {call['parsed']['regime']:>4} {call['group_label']:<6} depth {reg.depth:2d} {reg.type:<6} recorded {reg.abserr!r} recomputed {d!r} ({kind}) {'EQUAL' if same else 'DIFFER'}"
                )
    print("ALL EQUAL" if bad == 0 else f"{bad} DIFFER")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
