"""levin-phase-floor prompt 06f, F4's violations attributed (descriptive). New in 06f. For every node of
item 1 where |delta theta| > theta_abserr, splits production's group phase error into: e_K, the
rounding of K = sum_i s_i c_i (K_double - K_exact) times eta_hat; e_Kx, the rounding of the product
fl(K_double * eta_hat) - K_double * eta_hat; and the remainder (constituents' residuals and the
angle addition), all exact (fractions) except the measured total. Reads out/replay/*/result.json and
nodes.json.gz and the captures (out/capture.json). Usage: PYTHONPATH=<tree g>:<scripts> python -B
f4_attribution_06f.py"""

import gzip
import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402
from replay06f import load_capture  # noqa: E402


def main():
    cap = ROOT / "out" / "capture.json"
    sample = json.loads((ROOT / "out" / "sample" / "sample.json").read_text())
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06f: F4's violations attributed",
        [cap],
    ):
        print(l)
    tot = {"n": 0, "end": 0, "interior": 0}
    for c in sample["M2"]:
        name = f"{c['unit']}_{c['index']}"
        r = json.loads((ROOT / "out" / "replay" / name / "result.json").read_text())
        nodes = json.loads(
            gzip.open(ROOT / "out" / "replay" / name / "nodes.json.gz").read()
        )
        viol = [n for n in nodes if n["ratio"] is not None and n["ratio"] > 1.0]
        if not viol:
            continue
        rec, _e = load_capture(cap, c["unit"], c["index"])
        g = rec["theta"]["theta_mod_2pi"].__self__
        K_exact = sum(Fr(s) * Fr(cc) for s, cc in zip(g.signs, g.coefficients))
        dK = Fr(g.K) - K_exact
        nn = {L["i"]: L["n_nodes"] for L in r["M2"]["per_leaf"]}
        br = {L["i"]: L["branch"] for L in r["M2"]["per_leaf"]}
        print(
            f"\n{name}: K {g.K!r} (K_double - K_exact = {float(dK):.3e}); {len(viol)} nodes violate"
        )
        for n in sorted(viol, key=lambda n: -n["ratio"]):
            u = float.fromhex(n["u"])
            eta = np.exp(u)
            eK = float(dK * Fr(float(eta)))
            eKx = float(Fr(float(g.K * eta)) - Fr(g.K) * Fr(float(eta)))
            rest = n["delta_theta"] - eK - eKx
            kind = "end" if n["k"] in (0, nn[n["leaf"]] - 1) else "interior"
            tot["n"] += 1
            tot[kind] += 1
            print(
                f"   leaf {n['leaf']} ({br[n['leaf']]}, {kind} node {n['k']}): K*eta {g.K * eta:.4e}; dtheta {n['delta_theta']:+.3e} = "
                f"K rounding {eK:+.3e} + K*eta rounding {eKx:+.3e} + rest {rest:+.3e}; theta_abserr {n['theta_abserr']:.3e} "
                f"(ratio {n['ratio']:.3g}; |rest|/theta_abserr {abs(rest) / n['theta_abserr']:.3g})"
            )
    print(
        f"\nviolating nodes {tot['n']}: at a leaf end {tot['end']}, interior {tot['interior']}"
    )


if __name__ == "__main__":
    main()
