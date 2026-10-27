"""levin-phase-floor prompt 06 §5.4, a read-only look at the outliers of tabulate_phase_classes.py.

1. Every GkWKBValue / TkWKBValue row whose |Δ theta_mod_2pi|/(ε|θ|) exceeds 4, with |θ| formed from
   the producer's own cycle count, beside the |θ| of the GkSourceValue rows of the same k that hold
   the same pair of remainders (GkSource's cycle count is the rectified one, anchored across
   source redshifts), and the ratio against that |θ|.
2. Every value row whose |Δ G_WKB| (or T_WKB) exceeds A·|Δ theta_mod_2pi|, with |Δθ|, |ΔG|, A,
   ε·A and |G|: whether the excess is the rounding of G itself.

Reads only tabulate_phase_classes.py's CSV for (1), and both stores through the tool's reader for
(2), as that script does.

    PYTHONPATH=<code worktree> python -B examine_outliers.py <phase_classes_rows.csv> <R13> <R14>
"""

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

EPS = 2.0**-52


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def kf(x):
    return float.fromhex(x) if x.startswith("0x") else float(x)


def main():
    csv_path, a_path, b_path = sys.argv[1:4]
    me = Path(__file__).resolve()
    print(f"# {me.name} sha256 {sha256(me)}")
    print(f"# input {csv_path} sha256 {sha256(csv_path)}")
    print()
    rows = list(csv.DictReader(open(csv_path)))
    src = defaultdict(list)
    for r in rows:
        if r["table"] == "GkSourceValue":
            src[(r["k"], r["mod_A"], r["mod_B"])].append(float(r["theta_abs"]))
    print("(1) rows with |Δ mod|/(eps|theta|) > 4, |theta| from the producer's own cycle count")
    n = 0
    for r in sorted(rows, key=lambda r: -float(r["dmod_over_eps_theta"])):
        if r["table"] == "GkSourceValue" or float(r["dmod_over_eps_theta"]) <= 4:
            continue
        n += 1
        ma, mb = float.fromhex(r["mod_A"]), float.fromhex(r["mod_B"])
        d = abs(mb - ma)
        g = src.get((r["k"], r["mod_A"], r["mod_B"]), [])
        gtxt = ", ".join(f"{t:.4e} (ratio {d / (EPS * t):.3f})" for t in sorted(set(g))[:3]) or "none"
        print(f"  {r['table']} k={kf(r['k']):.6g} div {r['div_B']} |theta| {float(r['theta_abs']):.4e} "
              f"|Δmod| {d:.3e} ratio {float(r['dmod_over_eps_theta']):.2f}; GkSource |theta| same k and "
              f"remainders: {gtxt}")
    print(f"  total: {n}")
    print()

    from config.datastore import factories
    from tools.store_value_diff import read_store

    A = read_store(a_path, factories, classes=["GkWKBIntegration", "TkWKBIntegration"])
    B = read_store(b_path, factories, classes=["GkWKBIntegration", "TkWKBIntegration"])
    print("(2) rows with |ΔG| > A |Δ theta_mod|")
    for cls, vt, gc in (("GkWKBIntegration", "GkWKBValue", "G_WKB"),
                        ("TkWKBIntegration", "TkWKBValue", "T_WKB")):
        ta, tb = A.classes[cls].values[vt], B.classes[cls].values[vt]
        over = []
        for key in ta.rows:
            va, vb = ta.rows[key][0].values, tb.rows[key][0].values
            prow = B.classes[cls].rows.rows[key[0]][0].values
            amp = math.sqrt(vb["H_ratio"] / math.sqrt(vb["omega_WKB_sq"])) * math.hypot(
                prow["sin_coeff"], prow["cos_coeff"])
            if vt == "TkWKBValue":
                amp *= math.exp(vb["friction"])
            dth = abs(vb["theta_mod_2pi"] - va["theta_mod_2pi"])
            dg = abs(vb[gc] - va[gc])
            if dg > amp * dth:
                over.append((dg / (amp * dth) if dth else math.inf, dth, dg, amp, abs(vb[gc])))
        over.sort(reverse=True)
        print(f"  {vt}: {len(over)} rows; of them |ΔG| <= A|Δθ| + 4 eps A: "
              f"{sum(1 for o in over if o[2] <= o[3] * o[1] + 4 * EPS * o[3])}; "
              f"|Δθ| <= 1e-14 rad: {sum(1 for o in over if o[1] <= 1e-14)}; "
              f"max |Δθ| among them {max((o[1] for o in over), default=0):.3e}")
        for ratio, dth, dg, amp, g in over[:8]:
            print(f"    ratio {ratio:.3f}: |Δθ| {dth:.3e}, |ΔG| {dg:.3e}, A {amp:.3e}, eps*A {EPS * amp:.3e}, "
                  f"|G| {g:.3e}, (|ΔG| - A|Δθ|)/(eps A) {(dg - amp * dth) / (EPS * amp):.2f}")


if __name__ == "__main__":
    main()
