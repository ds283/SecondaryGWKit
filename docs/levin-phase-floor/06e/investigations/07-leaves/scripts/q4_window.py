"""Question 4: inside the window. For each window leaf of a run: |CC_31 - CC_16| (the leaf's abserr),
the three terms of 06d's floor C1 (n+4)uS, C2 (argument), C3 (node), recomputed from the same
arrays as the code (checked against the leaf's phase_err), its declared error and its true error
against the closed form. Aggregated over the window; and for sampled depth-0 pieces (the one
holding x_r, one halfway out, the outermost on each side) at two forced depths, the leaves inside
each: declared against true, so that refinement can be read off the same x-range. For a few leaves
the fine rule's true error is split in mpmath into node rounding / argument rounding / other.

Usage: q4_window.py CASE PHI D1,D2      (Phi_w / pi, forced depth_max values)
Writes out/q4_<CASE>_<PHI>pi.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import lc  # noqa: E402
from q3_decompose import mp_cc_args  # noqa: E402

lq = lc.lq


def terms(basis, cache, span):
    N = lc.ORDER
    f = lc.amplitudes()
    a, b = span
    grid, Dmat = lq.chebyshev_matrices(span, N)
    _, thp, _, _, _ = basis.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache=cache)
    fine = 2 * N - 1
    fine_grid, D_fine = lq.chebyshev_matrices(span, fine)
    f_fine = np.vstack(
        [lq._sample_vectorized(fn, fine_grid, cache, ("f", id(fn))) for fn in f]
    )
    ba = [basis.eval_basis_and_arguments(x) for x in fine_grid]
    bas = np.array([v for v, _ in ba]).T
    args = np.array([g for _, g in ba], dtype=float).T
    w = lq._cc_weights(span, fine)
    aw = np.fabs(w)
    n = aw.shape[0]
    S = float(np.dot(aw, np.sum(np.fabs(f_fine * bas), axis=0)))
    C1 = (n + lq._CC_ROUNDOFF_PER_TERM_COUNT) * lq._UNIT_ROUNDOFF * S
    C2 = lq._UNIT_ROUNDOFF * float(np.dot(aw, np.sum(np.fabs(f_fine) * np.fabs(args), axis=0)))
    fp = np.vstack([np.matmul(D_fine, row) for row in f_fine])
    delta = lq._cc_node_error_bound(span)
    tpb = float(np.max(np.fabs(thp)))
    C3 = delta * float(np.dot(aw, np.sum(tpb * np.fabs(f_fine) + np.fabs(fp), axis=0)))
    # the actual node displacement of this leaf, in units of u|x|, for the record
    with mp.workdps(lc.MP_DPS):
        t, _ = lc.mp_base(fine)
        A, B = mp.mpf(a), mp.mpf(b)
        mid, half = (A + B) / 2, (B - A) / 2
        dx = [float(mp.mpf(float(fine_grid[k])) - (mid + half * t[k])) for k in range(fine)]
    return {
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "delta": delta,
        "theta_prime_bound": tpb,
        "max_abs_dx": max(abs(v) for v in dx),
        "fine_grid": fine_grid,
        "args": args[0],
    }


def main():
    case_key = sys.argv[1]
    k = float(sys.argv[2])
    depths = [int(v) for v in sys.argv[3].split(",")]
    case = lc.CASES[case_key]
    with mp.workdps(lc.MP_DPS):
        ref = float(lc.closed_leaf(case, 0.0, 1.0))
    atol = 1e-14 * abs(ref)
    out = {"header": lc.header(__file__), "case": case, "phi_w_over_pi": k, "runs": {}}
    path = lc.OUT / f"q4_{case_key}_{k:g}pi.json"
    xr = lc.x_root(case)
    pieces0 = None
    for dm in depths:
        t0 = time.perf_counter()
        res, basis = lc.run_driver(case, k, dm, atol=atol)
        cache = lc.REC["cache"]
        win = [g for g in res["regions"] if g.type == "window"]
        cuts = None
        pts = res["stationary_points"]
        rows = []
        agg = {
            "C1": 0.0,
            "C2": 0.0,
            "C3": 0.0,
            "nested": 0.0,
            "declared": 0.0,
            "true_signed": mp.mpf(0),
            "true_abs": 0.0,
            "recon_max_rel_diff": 0.0,
        }
        with mp.workdps(lc.MP_DPS):
            for g in win:
                tm = terms(basis, cache, (g.start, g.end))
                floor = tm["C1"] + tm["C2"] + tm["C3"]
                agg["recon_max_rel_diff"] = max(
                    agg["recon_max_rel_diff"], abs(floor - g.phase_err) / g.phase_err
                )
                e = mp.mpf(g._rec_value) - lc.closed_leaf(case, g.start, g.end)
                row = {
                    "a": g.start,
                    "b": g.end,
                    "depth": g.depth,
                    "nested": g.abserr,
                    "C1": tm["C1"],
                    "C2": tm["C2"],
                    "C3": tm["C3"],
                    "floor": g.phase_err,
                    "declared": g.total_err,
                    "true": float(e),
                    "trusted": g.trusted,
                    "declared_resolution": g.declared_resolution,
                    "dtheta_mid": lc.theta_rel(case, 0.5 * (g.start + g.end)),
                    "theta_prime_bound": tm["theta_prime_bound"],
                    "delta_over_u_x": tm["delta"] / (lq._UNIT_ROUNDOFF * max(abs(g.start), abs(g.end))),
                    "max_abs_dx_over_u_x": tm["max_abs_dx"] / (lq._UNIT_ROUNDOFF * max(abs(g.start), abs(g.end))),
                }
                rows.append(row)
                for key in ("C1", "C2", "C3", "nested", "declared"):
                    agg[key] += row[key]
                agg["true_signed"] += e
                agg["true_abs"] += abs(float(e))
        agg["true_signed"] = float(agg["true_signed"])
        # the depth-0 pieces: from the window's cuts, reconstructed from the leaves of depth 0's
        # ancestry -- every leaf's depth-0 piece is the one of the location's cuts containing it
        lo, hi = pts[0]["window"]
        # depth-0 pieces: group leaves by their ancestor at depth 0 = maximal runs of 2^depth leaves
        if pieces0 is None:
            # take them from a depth-0-only call of the location (no solving needed)
            loc = lq._locate_stationary_points(
                (0.0, 1.0),
                basis.independent_phases(),
                basis.independent_phase_rates(),
                {},
                window_half_phase=k * np.pi,
                coalescence_lambda=lq._LEVIN_COALESCENCE_LAMBDA,
                piece_phase=lq._LEVIN_WINDOW_PIECE_PHASE,
            )
            cuts = loc["windows"][0]["cuts"]
            pieces0 = list(zip(cuts[:-1], cuts[1:]))
        # sampled depth-0 pieces: holding x_r, halfway out on the left, the outermost each side
        idx_r = next(i for i, (u, v) in enumerate(pieces0) if u <= xr <= v)
        sample = sorted({0, idx_r // 2, idx_r, len(pieces0) - 1})
        per_piece = []
        with mp.workdps(lc.MP_DPS):
            for i in sample:
                u, v = pieces0[i]
                inside = [r for r in rows if r["a"] >= u and r["b"] <= v]
                per_piece.append(
                    {
                        "piece": i,
                        "span": [u, v],
                        "dtheta": [lc.theta_rel(case, u), lc.theta_rel(case, v)],
                        "leaves": len(inside),
                        "nested_sum": sum(r["nested"] for r in inside),
                        "C1_sum": sum(r["C1"] for r in inside),
                        "C2_sum": sum(r["C2"] for r in inside),
                        "C3_sum": sum(r["C3"] for r in inside),
                        "declared_sum": sum(r["declared"] for r in inside),
                        "true_sum": sum(r["true"] for r in inside),
                        "true_abs_sum": sum(abs(r["true"]) for r in inside),
                        "max_abs_true": max(abs(r["true"]) for r in inside),
                        "piece_exact": mp.nstr(lc.closed_leaf(case, u, v), 20),
                    }
                )
        # mp split of the fine rule's error for three leaves: the one holding x_r, one at the
        # middle of the sampled half-way piece, and the outermost
        split = []
        picks = []
        for target in (xr, 0.5 * sum(pieces0[idx_r // 2]), pieces0[0][0]):
            g = min(win, key=lambda g: 0.0 if g.start <= target <= g.end else min(abs(g.start - target), abs(g.end - target)))
            picks.append(g)
        with mp.workdps(lc.MP_DPS):
            for g in picks:
                a, b = g.start, g.end
                N = lc.ORDER
                fine_grid, _ = lq.chebyshev_matrices((a, b), 2 * N - 1)
                ba = [basis.eval_basis_and_arguments(x) for x in fine_grid]
                args = [mp.mpf(float(x[1][0])) for x in ba]
                dn = [mp.mpf(float(x)) for x in fine_grid]
                half = (mp.mpf(b) - mp.mpf(a)) / 2
                ex = lc.closed_leaf(case, a, b)
                v_ex = lc.mp_cc(case, a, b)
                v_dn = lc.mp_cc(case, a, b, nodes=dn)
                v_arg = mp_cc_args(case, dn, args, half, N)
                V = mp.mpf(g._rec_value)
                split.append(
                    {
                        "span": [a, b],
                        "depth": g.depth,
                        "dtheta_mid": lc.theta_rel(case, 0.5 * (a + b)),
                        "true": float(V - ex),
                        "resolution": float(v_ex - ex),
                        "node": float(v_dn - v_ex),
                        "argument": float(v_arg - v_dn),
                        "other": float(V - v_arg),
                        "C3": next(r["C3"] for r in rows if r["a"] == a and r["b"] == b),
                        "nested": g.abserr,
                    }
                )
        out["runs"][f"d{dm}"] = {
            "wall_s": time.perf_counter() - t0,
            "window": [lo, hi],
            "window_leaves": len(win),
            "aggregate": agg,
            "pieces": per_piece,
            "split": split,
            "leaves_sample": [r for r in rows if any(r["a"] >= pieces0[i][0] and r["b"] <= pieces0[i][1] for i in sample)][:4000],
        }
        path.write_text(json.dumps(out, indent=1, default=str))
        print(
            f"{case_key} {k:g}pi d{dm}: window {len(win)} leaves, declared {agg['declared']:.3g} "
            f"C3 {agg['C3']:.3g} C1 {agg['C1']:.3g} C2 {agg['C2']:.3g} nested {agg['nested']:.3g} "
            f"true sum {agg['true_signed']:.3g} sum|true| {agg['true_abs']:.3g} recon {agg['recon_max_rel_diff']:.2g} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
