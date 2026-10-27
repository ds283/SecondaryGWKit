"""levin-phase-floor 06, finding P1, step 4 (comparison half): the two trees' integrands at the
common points of samples.py. Needs numpy only (no tree).

For every recorded column, per row, over all points and over the dense window |x - x_s| <= 0.05:
  * phases (raw theta, phi): max |r14 - r13| in rad; in units of ulp(|raw theta|) pointwise
    (np.spacing of that factor's or group's raw phase at the point); and in units of
    ulp(max |raw theta|) over the points (the granularity of the largest unreduced phase);
  * remainders (theta_mod_2pi): max cyclic |r14 - r13| in rad, and in units of ulp(2pi) ~ 8.9e-16
    and of ulp(max |raw theta|);
  * everything else (rates, amplitudes, H, w): max |r14 - r13| over the column's max |value|
    (relative to its envelope, which f_sin/f_cos need because they pass through zero), and the
    max pointwise relative difference where |value| > 1e-3 of its max;
  * the number of points at which the two are bit-identical.
And, per factor, the smoothness of the phi spline's knots in each tree: r_i = phi_i minus the
cubic through knots i-2, i-1, i+1, i+2 (in u = log1p z), for every interior knot inside the
sub-interval. For a smooth phi this is O(h^4 phi''''); for noise of size s it is about 1.5 s.
Also in units of ulp(raw theta) at the knot.

Usage: python -B compare.py   (reads out/samples_r13.npz, out/samples_r14.npz)
Writes out/compare.json.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from prov import OUT, header, sha256  # noqa: E402

TWO_PI = 2 * np.pi
GROUPS = ("G+q+r", "G+q-r", "G-q+r", "G-q-r")


def e(x):
    return "None" if x is None else f"{x:.3e}"


def stats(key, a, b, raw, mask):
    d = b - a
    kind = key.split(".")[-1]
    out = {"n": int(mask.sum()), "n_bit_identical": int(np.sum((a == b)[mask]))}
    ulp_pt = np.spacing(np.abs(raw[mask]))
    ulp_max = float(np.spacing(np.max(np.abs(raw[mask]))))
    if kind in ("theta", "raw_theta", "phi"):
        ad = np.abs(d[mask])
        out["max_abs_diff_rad"] = float(ad.max())
        out["median_abs_diff_rad"] = float(np.median(ad))
        out["max_diff_in_pointwise_ulp_raw_theta"] = float((ad / ulp_pt).max())
        out["max_diff_in_ulp_of_max_raw_theta"] = float(ad.max() / ulp_max)
    elif kind == "theta_mod_2pi":
        cd = np.abs(np.mod(d + np.pi, TWO_PI) - np.pi)[mask]
        out["max_cyclic_diff_rad"] = float(cd.max())
        out["median_cyclic_diff_rad"] = float(np.median(cd))
        out["max_cyclic_diff_in_ulp_2pi"] = float(cd.max() / np.spacing(TWO_PI))
        out["max_cyclic_diff_in_ulp_of_max_raw_theta"] = float(cd.max() / ulp_max)
    else:
        am = a[mask]
        bm = b[mask]
        env = float(np.max(np.abs(am)))
        out["max_abs"] = env
        out["max_abs_diff"] = float(np.max(np.abs(bm - am)))
        out["max_diff_over_envelope"] = float(np.max(np.abs(bm - am)) / env) if env > 0 else None
        big = np.abs(am) > 1e-3 * env
        out["max_rel_diff_where_above_1e-3_envelope"] = (
            float(np.max(np.abs(bm[big] - am[big]) / np.abs(am[big]))) if big.any() else None
        )
    return out


def knot_smoothness(D, row, nm, z_lo, z_hi):
    zk = D[f"{row}|knots.{nm}.z"]
    ph = D[f"{row}|knots.{nm}.phi"]
    th = D[f"{row}|knots.{nm}.raw_theta"]
    u = np.log1p(zk)
    resid = []
    for i in range(2, len(u) - 2):
        idx = [i - 2, i - 1, i + 1, i + 2]
        cf = np.polyfit(u[idx] - u[i], ph[idx], 3)
        resid.append(ph[i] - np.polyval(cf, 0.0))
    resid = np.abs(np.array(resid))
    ulp = np.spacing(np.abs(th[2:-2]))
    inside = (zk[2:-2] >= z_lo) & (zk[2:-2] <= z_hi)
    ri, ui = resid[inside], ulp[inside]
    return {
        "n_knots": int(len(u)),
        "n_interior_knots_inside_subinterval": int(inside.sum()),
        "median_abs_resid_rad": float(np.median(ri)),
        "max_abs_resid_rad": float(ri.max()),
        "median_resid_in_ulp_raw_theta": float(np.median(ri / ui)),
        "max_resid_in_ulp_raw_theta": float((ri / ui).max()),
    }


def main():
    A = np.load(OUT / "samples_r13.npz")
    B = np.load(OUT / "samples_r14.npz")
    res = {
        "header": header(Path(__file__), "p1 compare", inputs=[OUT / "samples_r13.npz", OUT / "samples_r14.npz"]),
        "rows": {},
    }
    for row in ("row1", "row2"):
        xa, xb = A[f"{row}|x"], B[f"{row}|x"]
        assert np.array_equal(xa, xb), "points differ between trees"
        xs = float(A[f"{row}|x_s"][0])
        masks = {"all": np.ones_like(xa, dtype=bool), "dense_window": np.abs(xa - xs) <= 0.05}
        keys = sorted(
            k.split("|", 1)[1]
            for k in A.files
            if k.startswith(row + "|")
            and k.split("|", 1)[1] not in ("x", "x_s")
            and not k.split("|", 1)[1].startswith("knots.")
        )
        r = {"n_points": int(len(xa)), "x_s": xs, "columns": {}, "knot_smoothness": {}}
        print(f"== {row}: {len(xa)} points, x_s {xs!r}")
        for key in keys:
            a, b = A[f"{row}|{key}"], B[f"{row}|{key}"]
            owner = key.split(".")[0]
            if owner in ("G", "Tq", "Tr"):
                raw = A[f"{row}|{owner}.raw_theta"]
            elif owner in GROUPS:
                raw = A[f"{row}|{owner}.theta"]
            else:
                raw = a
            v = {m: stats(key, a, b, raw, mk) for m, mk in masks.items()}
            r["columns"][key] = v
            s, w = v["all"], v["dense_window"]
            if "max_abs_diff_rad" in s:
                print(f"   {key:22s} max |d| {s['max_abs_diff_rad']:.3e} rad = {s['max_diff_in_ulp_of_max_raw_theta']:.2f} "
                      f"ulp(max|theta|) (pointwise-ulp max {s['max_diff_in_pointwise_ulp_raw_theta']:.3g}); "
                      f"window {w['max_abs_diff_rad']:.3e}; identical {s['n_bit_identical']}/{s['n']}")
            elif "max_cyclic_diff_rad" in s:
                print(f"   {key:22s} max cyclic |d| {s['max_cyclic_diff_rad']:.3e} rad = "
                      f"{s['max_cyclic_diff_in_ulp_of_max_raw_theta']:.2f} ulp(max|theta|) = "
                      f"{s['max_cyclic_diff_in_ulp_2pi']:.3g} ulp(2pi); window {w['max_cyclic_diff_rad']:.3e}; "
                      f"identical {s['n_bit_identical']}/{s['n']}")
            else:
                print(f"   {key:22s} max |d|/env {e(s['max_diff_over_envelope'])}, max rel (>1e-3 env) "
                      f"{e(s['max_rel_diff_where_above_1e-3_envelope'])}; window {e(w['max_diff_over_envelope'])}; "
                      f"identical {s['n_bit_identical']}/{s['n']}")
        z_lo, z_hi = 250.0, float(np.expm1(xa.max()))
        for nm in ("G", "Tq", "Tr"):
            r["knot_smoothness"][nm] = {t: knot_smoothness(D, row, nm, z_lo, z_hi) for t, D in (("r13", A), ("r14", B))}
            a13, a14 = r["knot_smoothness"][nm]["r13"], r["knot_smoothness"][nm]["r14"]
            print(f"   knots {nm}: r13 median/max resid {a13['median_abs_resid_rad']:.3e}/{a13['max_abs_resid_rad']:.3e} rad "
                  f"({a13['median_resid_in_ulp_raw_theta']:.3g}/{a13['max_resid_in_ulp_raw_theta']:.3g} ulp); "
                  f"r14 {a14['median_abs_resid_rad']:.3e}/{a14['max_abs_resid_rad']:.3e} rad "
                  f"({a14['median_resid_in_ulp_raw_theta']:.3g}/{a14['max_resid_in_ulp_raw_theta']:.3g} ulp); "
                  f"{a13['n_interior_knots_inside_subinterval']} knots in the sub-interval")
        res["rows"][row] = r
    path = OUT / "compare.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
