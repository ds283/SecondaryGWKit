"""Question 3: what a residual or nested-pair estimate that does not fall under refinement is made of.

For case c1e6w0 (and, with argv, another case) the driver is re-run at the given (Phi_w, depth_max)
configurations to capture every non-window solve it makes (lc.REC["solves"]) with the driver's own
vectorize_cache, then:

1. **History of every Levin leaf ever made** outside the window (every recorded span whose two
   halves were also solved): its step-(4) residual |V_P - (V_L + V_R)|, its eq. (151) floor, the
   true error of V_P and of V_L + V_R against the closed form; summarised by depth.
2. **Decomposition** of chosen leaves. Each solve's error V - exact is split by re-solving the same
   scheme in mpmath (lc.mp_levin / lc.mp_cc, MP_DPS digits):
     resolution   = mp(exact nodes, exact theta) - exact integral
     node         = mp(double nodes, theta exact at them) - mp(exact nodes)
     endpoint/arg = mp(double nodes, the double theta the code hands to sin/cos) - mp(double nodes)
                    (Levin: theta at the two end nodes; CC: theta at every node)
     other        = double - mp(double nodes, double theta)
   For a Levin leaf, its two halves are decomposed too, so the residual V_P - V_L - V_R is split
   the same way. For a CC (direct) leaf, both CC_{2N-1} and CC_N are decomposed, so the nested
   pair is split. Each chosen leaf is also re-solved with a range-reduced phase theta_mod_2pi
   supplied (lc.theta_mod_2pi_exact: theta in mp at the double x, mod 2 pi, rounded once), and
   its halves too: the residual and the floor under it.
3. **The CC leaves' nested pair under further bisection**: each chosen direct leaf's halves and
   quarters solved as the driver would (same cache), their nested pairs and true errors.

Usage: q3_decompose.py CASE CONFIGS     CONFIGS like 2:6,2:9,8:12 (Phi_w/pi : depth_max)
Writes out/q3_<CASE>.json.
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import lc  # noqa: E402

lq = lc.lq


def mp_cc_args(case, nodes, args, half, N, coarse=False):
    """sum_k w_k (f1 sin a_k + f2 cos a_k) * half with exact weights, at the given nodes and args."""
    fine = 2 * N - 1
    with mp.workdps(lc.MP_DPS):
        _, _, f1, f2 = lc._mp_funcs(case)
        w = lc.mp_cc_weights_base(N if coarse else fine)
        xs = nodes[::2] if coarse else nodes
        ag = args[::2] if coarse else args
        s = mp.mpf(0)
        for wk, xk, ak in zip(w, xs, ag):
            s += wk * (f1(xk) * mp.sin(ak) + f2(xk) * mp.cos(ak))
        return +(s * half)


def solve(basis, cache, span):
    f = lc.amplitudes()
    return lq._adaptive_levin_subregion(
        span,
        f,
        basis,
        id_label=None,
        chebyshev_order=lc.ORDER,
        rtol=lc.RTOL,
        vectorize_cache=cache,
    )


def decompose_levin(case, basis, cache, span, data):
    a, b = span
    p_use = [bool(np.isfinite(r) and r > lc.RTOL) for r in data["p_ratios"]]
    grid, _ = lq.chebyshev_matrices(span, lc.ORDER)
    th_d = lq._sample_vectorized(basis._theta, grid, cache, ("theta", id(basis)))
    with mp.workdps(lc.MP_DPS):
        ex = lc.closed_leaf(case, a, b)
        v_ex = lc.mp_levin(case, a, b, p_use)
        dn = [mp.mpf(float(g)) for g in grid]
        v_dn = lc.mp_levin(case, a, b, p_use, nodes=dn)
        v_dn_end = lc.mp_levin(case, a, b, p_use, nodes=dn, end_args=(th_d[0], th_d[-1]))
        V = mp.mpf(data["value"])
        th_exact_b = lc._mp_funcs(case)[0](mp.mpf(float(grid[0])))
        th_exact_a = lc._mp_funcs(case)[0](mp.mpf(float(grid[-1])))
        return {
            "span": [a, b],
            "kind": "Levin",
            "value": data["value"],
            "floor": data["phase_err"],
            "truncation": data["abserr_truncation"],
            "p_use": p_use,
            "grid_ends_exact": bool(grid[0] == b and grid[-1] == a),
            "end_node_offsets": [float(grid[-1] - a), float(grid[0] - b)],
            "end_theta_rounding": [
                float(mp.mpf(float(th_d[-1])) - th_exact_a),
                float(mp.mpf(float(th_d[0])) - th_exact_b),
            ],
            "error": float(V - ex),
            "resolution": float(v_ex - ex),
            "node": float(v_dn - v_ex),
            "endpoint_theta": float(v_dn_end - v_dn),
            "other": float(V - v_dn_end),
            "_mp": (ex, v_ex, v_dn, v_dn_end, V),
        }


def decompose_cc(case, basis, cache, span, data):
    a, b = span
    N = lc.ORDER
    fine = 2 * N - 1
    fine_grid, _ = lq.chebyshev_matrices(span, fine)
    ba = [basis.eval_basis_and_arguments(x) for x in fine_grid]
    args = [float(g[0]) for _, g in ba]
    f = lc.amplitudes()
    f_fine = np.vstack([fn(fine_grid) for fn in f])
    basis_fine = np.array([v for v, _ in ba]).T
    integrand = np.sum(f_fine * basis_fine, axis=0)
    w_c = lq._cc_weights(span, N)
    v_coarse = float(np.dot(w_c, integrand[::2]))
    out = {"span": [a, b], "kind": "CC", "value": data["value"], "floor": data["phase_err"]}
    with mp.workdps(lc.MP_DPS):
        ex = lc.closed_leaf(case, a, b)
        half = (mp.mpf(b) - mp.mpf(a)) / 2
        dn = [mp.mpf(float(x)) for x in fine_grid]
        for name, coarse, V in (("fine", False, data["value"]), ("coarse", True, v_coarse)):
            v_ex = lc.mp_cc(case, a, b, coarse=coarse)
            v_dn = lc.mp_cc(case, a, b, nodes=dn, coarse=coarse)
            v_dn_arg = mp_cc_args(case, dn, [mp.mpf(g) for g in args], half, N, coarse=coarse)
            Vm = mp.mpf(V)
            out[name] = {
                "value": V,
                "error": float(Vm - ex),
                "resolution": float(v_ex - ex),
                "node": float(v_dn - v_ex),
                "argument": float(v_dn_arg - v_dn),
                "other": float(Vm - v_dn_arg),
            }
    out["nested_pair"] = abs(data["value"] - v_coarse)
    out["nested_pair_code"] = data.get("abserr_direct")
    return out


def leaf_with_halves(case, basis, cache, span, solves, tm_basis, tm_cache):
    a, b = span
    c = (b + a) / 2.0
    P = solves.get((a, b)) or solve(basis, cache, (a, b))
    L = solves.get((a, c)) or solve(basis, cache, (a, c))
    R = solves.get((c, b)) or solve(basis, cache, (c, b))
    rec = {"span": [a, b], "dtheta_near": None}
    xr = lc.x_root(case)
    near = b if b <= xr else a
    rec["dtheta_near"] = lc.theta_rel(case, near)
    rec["theta_prime_near"] = lc.theta_prime(case, near)
    if P["is_direct"]:
        rec["parent"] = decompose_cc(case, basis, cache, (a, b), P)
        # further bisection of a CC leaf: halves and quarters, as the driver would solve them
        rec["refine"] = []
        for depth_extra in (1, 2, 3):
            n = 2**depth_extra
            pieces = [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n) for i in range(n)]
            # use the driver's own bisection arithmetic: repeated (lo+hi)/2
            pieces = [(a, b)]
            for _ in range(depth_extra):
                nxt = []
                for lo, hi in pieces:
                    mid = (hi + lo) / 2.0
                    nxt += [(lo, mid), (mid, hi)]
                pieces = nxt
            npair = 0.0
            err = mp.mpf(0)
            abserr_sum = 0.0
            floor_sum = 0.0
            kinds = set()
            with mp.workdps(lc.MP_DPS):
                for sp in pieces:
                    d = solve(basis, cache, sp)
                    kinds.add("CC" if d["is_direct"] else "Levin")
                    if d["is_direct"]:
                        npair += d["abserr_direct"]
                    floor_sum += d["phase_err"]
                    e = mp.mpf(d["value"]) - lc.closed_leaf(case, *sp)
                    err += e
                    abserr_sum += abs(float(e))
            rec["refine"].append(
                {
                    "extra_depth": depth_extra,
                    "kinds": sorted(kinds),
                    "sum_nested_pair": npair,
                    "sum_floor": floor_sum,
                    "sum_true_error": float(err),
                    "sum_abs_true_error": abserr_sum,
                }
            )
        return rec
    parts = {}
    for name, sp, d in (("P", (a, b), P), ("L", (a, c), L), ("R", (c, b), R)):
        if d["is_direct"]:
            parts[name] = decompose_cc(case, basis, cache, sp, d)
        else:
            parts[name] = decompose_levin(case, basis, cache, sp, d)
    rec["parts"] = {k: {kk: vv for kk, vv in v.items() if kk != "_mp"} for k, v in parts.items()}
    rec["residual"] = abs(P["value"] - (L["value"] + R["value"]))
    rec["floor"] = P["phase_err"]
    # residual split: V_P - V_L - V_R = sum over components of (comp_P - comp_L - comp_R)
    split = {}
    for comp in ("resolution", "node", "endpoint_theta", "other"):
        vals = []
        for name in ("P", "L", "R"):
            part = parts[name]
            if part["kind"] == "Levin":
                vals.append(part[comp])
            else:
                key = {"endpoint_theta": "argument"}.get(comp, comp)
                vals.append(part["fine"][key])
        split[comp] = vals[0] - vals[1] - vals[2]
    rec["residual_signed"] = P["value"] - (L["value"] + R["value"])
    rec["residual_split"] = split
    # the same leaf with a range-reduced phase supplied
    Pt = solve(tm_basis, tm_cache, (a, b))
    Lt = solve(tm_basis, tm_cache, (a, c))
    Rt = solve(tm_basis, tm_cache, (c, b))
    with mp.workdps(lc.MP_DPS):
        exP = lc.closed_leaf(case, a, b)
        exL = lc.closed_leaf(case, a, c)
        exR = lc.closed_leaf(case, c, b)
        rec["theta_mod_2pi"] = {
            "residual": abs(Pt["value"] - (Lt["value"] + Rt["value"])),
            "floor": Pt["phase_err"],
            "P_error": float(mp.mpf(Pt["value"]) - exP),
            "L_error": float(mp.mpf(Lt["value"]) - exL),
            "R_error": float(mp.mpf(Rt["value"]) - exR),
            "is_direct": [Pt["is_direct"], Lt["is_direct"], Rt["is_direct"]],
        }
    return rec


def history(case, solves):
    """Every recorded Levin span whose halves were also recorded."""
    rows = []
    for (a, b), d in solves.items():
        if d["is_direct"]:
            continue
        c = (b + a) / 2.0
        L = solves.get((a, c))
        R = solves.get((c, b))
        if L is None or R is None:
            continue
        with mp.workdps(lc.MP_DPS):
            exP = lc.closed_leaf(case, a, b)
            exL = lc.closed_leaf(case, a, c)
            exR = lc.closed_leaf(case, c, b)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "depth": round(math.log2(1.0 / (b - a))) if b > a else None,
                    "width": b - a,
                    "residual": abs(d["value"] - (L["value"] + R["value"])),
                    "floor": d["phase_err"],
                    "P_error": float(mp.mpf(d["value"]) - exP),
                    "halves_error": float(mp.mpf(L["value"]) + mp.mpf(R["value"]) - exL - exR),
                    "halves_direct": [L["is_direct"], R["is_direct"]],
                    "dtheta_near": lc.theta_rel(case, b if b <= lc.x_root(case) else a),
                }
            )
    return rows


def main():
    case_key = sys.argv[1]
    configs = [tuple(float(v) for v in s.split(":")) for s in sys.argv[2].split(",")]
    case = lc.CASES[case_key]
    with mp.workdps(lc.MP_DPS):
        ref = float(lc.closed_leaf(case, 0.0, 1.0))
    atol = 1e-14 * abs(ref)
    out = {"header": lc.header(__file__), "case": case, "configs": {}}
    path = lc.OUT / f"q3_{case_key}.json"
    tm_basis = lq._Basis_SinCos(
        dict(
            lc.phase(case["c"], case["x_s"], case["omega"]),
            theta_mod_2pi=lc.theta_mod_2pi_exact(case),
        )
    )
    tm_cache = {}
    for k, dm in configs:
        dm = int(dm)
        t0 = time.perf_counter()
        res, basis = lc.run_driver(case, k, dm, atol=atol)
        solves = dict(lc.REC["solves"])
        cache = lc.REC["cache"]
        other = [g for g in res["regions"] if g.type != "window"]
        hist = history(case, solves)
        # chosen leaves: the top 3 returned Levin leaves by residual, the top 3 direct by nested
        # pair, the leaf adjacent to the window on each side, and the far-left control leaf
        lev = sorted([g for g in other if g.type == "Levin"], key=lambda g: -g.abserr)
        cc = sorted([g for g in other if g.type == "direct"], key=lambda g: -g.abserr)
        pts = res["stationary_points"]
        lo, hi = pts[0]["window"]
        adj = [g for g in other if g.end == lo or g.start == hi]
        chosen = []
        for g in lev[:3] + cc[:3] + adj + [other[0]]:
            if (g.start, g.end) not in [(c.start, c.end) for c in chosen]:
                chosen.append(g)
        recs = []
        for g in chosen:
            rec = leaf_with_halves(
                case, basis, cache, (g.start, g.end), solves, tm_basis, tm_cache
            )
            rec["returned"] = {
                "type": g.type,
                "depth": g.depth,
                "abserr": g.abserr,
                "floor": g.phase_err,
                "total_err": g.total_err,
                "phase_limited": g.phase_limited,
            }
            recs.append(rec)
            print(f"  {k:g}pi d{dm} leaf {g.start:.10f},{g.end:.10f} {g.type} done", flush=True)
        out["configs"][f"{k:g}pi_d{dm}"] = {
            "wall_s": time.perf_counter() - t0,
            "history": hist,
            "chosen": recs,
        }
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"{case_key} {k:g}pi d{dm}: {len(hist)} history rows, {len(recs)} chosen", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
