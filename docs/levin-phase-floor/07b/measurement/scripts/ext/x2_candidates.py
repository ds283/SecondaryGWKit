"""levin-phase-floor prompt 07b extension, section 2.7.1 (X2): candidate quantities that the code can compute from a
leaf's own solves, on every trusted Levin leaf Part A referenced. New in the extension.

**The leaves:** Part A's mechanism rows (out/mech/mech_all.json) at 07a's code (``tree``; item 3's at 07's, ``07``),
trusted Levin leaves, with their true errors err(Q)/e' against their references. Part A's rate-range set is these with
reference accuracy at most e'/100 (out/tables/tables_07b.md, 73 leaves); the others are computed and flagged.

**The candidates**, each from solves of the leaf's own span made exactly as the driver makes them
(``_adaptive_levin_subregion`` on ``_Basis_SinCos(theta)``, the call's f, rtol and the leaf's ``force_direct``), with the
order N the call's (24 in every call here):
- **(C1) another collocation order** (the prompt's first candidate): the leaf P and its four quarters re-solved at
  N' = N - 8, N + 8, N + 16 and 2N (16, 32, 40, 48); D_P(N') = |V_P(N') - V_P(N)| / e', D_Q(N') = |Q(N') - Q(N)| / e'; and Q(N')'s own true error
  against the leaf's reference, err(Q(N'))/e'. If the error common to P, halves and quarters is the collocation's
  (p not resolved across the leaf at order N), a higher order moves Q by about that error.
- **(C2) the eighths** (variant (c)'s check on every trusted leaf): S3, delta3 = |Q - S3|, r3 = delta3/delta2, and
  |S3 - Q|/e'.
- **(C3) the tail of p's Chebyshev coefficients** (a quantity the solve already has): p at order N on P's grid
  (build_p_sample), its Chebyshev coefficients c_k (the type-I DCT on the Lobatto nodes), and
  T_P = sum over components of |c_{N-2}| + |c_{N-1}|, in the value's units (|w| = 1); T_P/e'; and the same summed over
  the four quarters, T_Q/e'. Also |c_{N-1}| / max_k |c_k| (the relative tail).
- **(C4) the conditioning** of P's Levin operator D + i diag(theta') (the complexified form the solve uses), its
  2-norm condition number.
- **(C5) the rate range** max|theta'|/min|theta'| over P's nodes (variant (f)'s statistic), for comparison.
Each leaf's V_P at order N is checked against its probe by hex.

Writes OUT_JSON. Usage: PYTHONPATH=<tree>:<scripts> python -B x2_candidates.py OUT_JSON VARIANT (tree | 07)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

mp.dps = 50
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
TREE_LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"
MECH = ROOT / "out" / "mech" / "mech_all.json"
ORDER_OFFSETS = (-8, 8, 16, None)  # N - 8, N + 8, N + 16 and 2N (None); N the call's order


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cheb_coeffs(vals):
    """Chebyshev coefficients of the interpolant through vals at the Lobatto nodes y_j = cos(pi j/(n-1)), j = 0..n-1."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    j = np.arange(n)
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    c = np.array([2.0 / (n - 1) * np.sum(w * v * np.cos(np.pi * k * j / (n - 1))) for k in range(n)])
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


def main():
    out_p, variant = Path(sys.argv[1]), sys.argv[2]
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if os.environ.get("PYTHONPATH", "").split(":")[0] != str(TREE):
        raise SystemExit("PYTHONPATH must start with the tree")
    sys.path.insert(1, str(HERE.parent))
    if variant == "tree":
        import AdaptiveLevin.levin_quadrature as LQ

        msha = sha256(LQ.__file__)
        if msha != TREE_LQ_SHA:
            raise SystemExit("the tree's module has another SHA-256")
    else:
        import variant_loader as VL

        vp = HERE.parent / "variants" / "levin_quadrature_07.py"
        msha = "4e5da371f51e6e3dc20e73579eb45e0e985e2a3b18571ee29eb86b3d3e9c93fc"
        LQ = VL.install(vp, msha, "07b_07")
    import ray.cloudpickle as cp

    mech = json.loads(MECH.read_text())["rows"]
    rows = [r for r in mech if r["variant"] == variant and r["type"] == "Levin" and r["trusted"]]
    out = []
    cache_calls = {}
    for r in rows:
        t0 = time.perf_counter()
        pr = json.loads(Path(r["probe"]).read_text())
        c = pr["header"]["capture"]
        key = (c["path"], c["offset"], c["sha256"], json.dumps(c.get("override"), sort_keys=True))
        if key not in cache_calls:
            with open(c["path"], "rb") as f:
                f.seek(c["offset"])
                blob = f.read(c["length"])
            if hashlib.sha256(blob).hexdigest() != c["sha256"]:
                raise SystemExit("capture blob has another SHA-256")
            rec = cp.loads(blob)
            kw = dict(rec["kw"])
            if c.get("override"):
                kw["rtol"], kw["atol"] = c["override"]["rtol"], c["override"]["atol"]
            cache_calls[key] = (rec, kw)
        rec, kw = cache_calls[key]
        N = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
        rtol = kw["rtol"]
        A = LQ._Basis_SinCos(rec["theta"])
        vcache = {}
        leaf = next(e for e in pr["leaves"] if e["i"] == r["i"])
        L = leaf["leaf"]
        a_, b_ = float.fromhex(L["start"]), float.fromhex(L["end"])
        force = bool(L["window"])

        def solve(span, order, p_sample=False):
            return LQ._adaptive_levin_subregion(span, rec["f"], A, id_label="x2-07b", chebyshev_order=order, rtol=rtol,
                                               notify_label=rec["notify_label"], vectorize_cache=vcache,
                                               force_direct=force, build_p_sample=p_sample)

        c_ = (b_ + a_) / 2.0
        cL, cR = (c_ + a_) / 2.0, (b_ + c_) / 2.0
        quarters = [(a_, cL), (cL, c_), (c_, cR), (cR, b_)]
        eighths = [(q0, (q0 + q1) / 2.0) for q0, q1 in quarters] + [((q0 + q1) / 2.0, q1) for q0, q1 in quarters]
        eighths.sort()
        dP = solve((a_, b_), N, p_sample=True)
        dq = [solve(q, N, p_sample=True) for q in quarters]
        VP = dP["value"]
        Q = (dq[0]["value"] + dq[1]["value"]) + (dq[2]["value"] + dq[3]["value"])
        dL, dR = solve((a_, c_), N), solve((c_, b_), N)
        S1 = dL["value"] + dR["value"]
        S3 = sum(solve(e, N)["value"] for e in eighths)
        eprime = r["e_prime"]
        ref = mpf(r["ref"])
        delta2 = abs(S1 - Q)
        delta3 = abs(Q - S3)
        res = {"sid": r["sid"], "variant": variant, "i": r["i"], "depth": r["depth"], "lo": a_, "hi": b_,
               "touches_window": bool(r["edge_lo"] or r["edge_hi"]), "e_prime": eprime,
               "err_Q_over_eprime": r["err_over_eprime"]["Q"], "ref_acc_over_eprime": r["ref_acc_over_eprime"],
               "rate_end_to_end": max(abs(r["theta_prime_lo"]), abs(r["theta_prime_hi"])) /
               min(abs(r["theta_prime_lo"]), abs(r["theta_prime_hi"])),
               "V_P_reproduces": float(VP).hex() == leaf["probe"]["V_P"], "Q_reproduces": float(Q).hex() == leaf["probe"]["Q"],
               "order": N, "r": r["r"], "delta2": delta2, "delta3": delta3,
               "r3": (delta3 / delta2) if delta2 > 0 else float("inf"), "S3_minus_Q_over_eprime": delta3 / eprime,
               "orders": {}}
        for Np in [N + k if k is not None else 2 * N for k in ORDER_OFFSETS]:
            dPp = solve((a_, b_), Np)
            Qp = sum(solve(q, Np)["value"] for q in quarters)
            res["orders"][str(Np)] = {"D_P": abs(dPp["value"] - VP) / eprime, "D_Q": abs(Qp - Q) / eprime,
                                      "err_Q_over_eprime": float((mpf(Qp) - ref) / eprime),
                                      "err_VP_over_eprime": float((mpf(dPp["value"]) - ref) / eprime),
                                      "P_direct": bool(dPp.get("is_direct", False))}
        # (C3) the tail of p's Chebyshev coefficients, P and the quarters
        def tail(d):
            if d.get("p_sample") is None:  # a Clenshaw-Curtis piece (the C2 gate's fallback): no p
                return 0.0, 0.0
            P = np.array([[float(v) for v in ps] for _, ps in d["p_sample"]])  # nodes x components
            co = [cheb_coeffs(P[:, k]) for k in range(P.shape[1])]
            t_abs = float(sum(abs(cc[-1]) + abs(cc[-2]) for cc in co))
            t_rel = float(max(abs(cc[-1]) / np.max(np.abs(cc)) for cc in co if np.max(np.abs(cc)) > 0))
            return t_abs, t_rel
        tP, tPrel = tail(dP)
        tq = [tail(d) for d in dq]
        res |= {"tail_P_over_eprime": tP / eprime, "tail_P_rel": tPrel,
                "tail_Q_over_eprime": sum(t for t, _ in tq) / eprime, "tail_Q_rel_max": max(t for _, t in tq)}
        # (C4) conditioning, (C5) rate range over P's nodes
        grid, Dmat = LQ.chebyshev_matrices((a_, b_), N)
        _, tpc, _, _, _ = A.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache={})
        Lop = Dmat.astype(complex)
        Lop[np.diag_indices(N)] += 1j * np.asarray(tpc)
        res["cond_P"] = float(np.linalg.cond(Lop))
        res["rate_range_nodes"] = float(np.max(np.abs(tpc)) / np.min(np.abs(tpc)))
        res["wall_s"] = time.perf_counter() - t0
        out.append(res)
        o = res["orders"]
        print(f"{r['sid']:16s} {r['i']:4d} err/e' {res['err_Q_over_eprime']:+9.3g} rr {res['rate_end_to_end']:9.3g} | "
              f"D_Q " + " ".join(f"{k} {v['D_Q']:8.3g}" for k, v in o.items()) + f" | err Q(2N) {o[str(2 * N)]['err_Q_over_eprime']:+9.3g} | "
              f"r3 {res['r3']:8.3g} |S3-Q|/e' {res['S3_minus_Q_over_eprime']:8.3g} | tailP {tP / eprime:8.3g} tailQ {res['tail_Q_over_eprime']:8.3g} "
              f"| cond {res['cond_P']:8.3g} | repro {res['V_P_reproduces']} {res['Q_reproduces']}", flush=True)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "module_sha256": msha,
                      "variant": variant, "mech": str(MECH), "mech_sha256": sha256(MECH), "order_offsets": ORDER_OFFSETS,
                      "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "leaves": out}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    main()
