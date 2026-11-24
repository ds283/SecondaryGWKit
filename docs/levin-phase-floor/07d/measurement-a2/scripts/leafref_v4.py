"""levin-phase-floor prompt 07d Part A' (section 9.2 item 0): the leaf reference v4, by the rule fixed in
state/refs_rule_leaf_v4.txt, and what it needs from a captured call. New in Part A'.

The integrand, the knots and sigma_hat are Part A's v3 code (Part A's scripts/leafcc_07d.py, a record, SHA-256 in
LEAFCC_07D_SHA), copied here unchanged except where marked "Part A'":
- ``Integrand``: f by _sample_vectorized, the basis by A.eval_basis_and_arguments (v3's g_on);
- ``call_knots``: production_edges for a group call, else v3's analytic reader;
- ``v3_sigma``: v3's piecewise Chebyshev least squares (starting degrees, convergence rule), returning sigma_hat (the
  pooled residual RMS of the d_hi fits) and v3's own ref, eps_LS and pieces, so that a Part A leaf can be checked.
``v4_reference`` is new: composite 24-point Gauss-Legendre on the pieces, doubled piece by piece from 8 panels (rule items
2-5).
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from prov import PARTA

LEAFCC_07D_SHA = hashlib.sha256((PARTA / "scripts" / "leafcc_07d.py").read_bytes()).hexdigest()
LS_M = 4096
PIECE_MIN_M = 96
CONV_RATIO = 1.5
GL_ORDER = 24
LEVELS = [8 * 2 ** k for k in range(7)]  # 8 .. 512 panels a piece
SETTLE_K = 3.0
_GX, _GW = np.polynomial.legendre.leggauss(GL_ORDER)


def manifest():
    man = {}
    for p in sorted((PARTA / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    return man


def load_capture(c):
    import ray.cloudpickle as cp

    with open(c["path"], "rb") as fh:
        fh.seek(c["offset"])
        blob = fh.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    return cp.loads(blob)


class Integrand:
    """g = sum_i f_i b_i(theta) as Part A's v3 evaluates it, with an evaluation count."""

    def __init__(self, LQ, rec):
        self.LQ = LQ
        self.f = rec["f"]
        self.A = LQ._Basis_SinCos(rec["theta"])
        self.cache = {}
        self.n_eval = 0

    def f_on(self, grid):
        return np.vstack([self.LQ._sample_vectorized(fn, grid, self.cache, ("f", id(fn))) for fn in self.f])

    def g_on(self, grid, with_parts=False):
        grid = np.asarray(grid, dtype=float)
        self.n_eval += len(grid)
        fs = self.f_on(grid)
        ba = [self.A.eval_basis_and_arguments(x) for x in grid]
        bs = np.array([v for v, _ in ba]).T
        g = np.sum(fs * bs, axis=0)
        return (g, fs) if with_parts else g


def call_knots(rec, xs0, xs1):
    """(edges_all or None, n_tables, note): Part A's v3 reading of the call's knots, unchanged."""
    from tools.filon_reference import production_edges

    f = rec["f"]
    th = rec["theta"]
    try:
        edges_all, n_tables = production_edges(rec)
        return edges_all, n_tables, None
    except Exception as e:  # noqa: BLE001
        edges_all, n_tables, edges_err = None, None, repr(e)[:300]
    try:
        grp = th["theta"].__self__
        cl0 = dict(zip(f[0].__code__.co_freevars, [c_.cell_contents for c_ in f[0].__closure__]))
        objs = list(zip(grp.phases, grp.coefficients)) + [
            (cl0["mod_Gk"], cl0["k"].k), (cl0["mod_Tk"], cl0["q"].k * cl0["cs"]), (cl0["mod_Tk"], cl0["r"].k * cl0["cs"])]
        ks, n_t = [], 0
        lo_, hi_ = min(xs0, xs1), max(xs0, xs1)
        for o, c_ in objs:
            co_ = o._corrections
            xst = co_.x_star
            if xst > 0:
                ks.append(math.log(xst / c_))
            reached = co_.near is not None and c_ * math.exp(lo_) < xst
            if reached:
                n_t += 1
                for interp in (co_._r_interp, co_._ell_interp):
                    if interp is not None:
                        ks.extend(float(B) - math.log(c_) for B in np.asarray(interp.breakpoints).ravel())
        edges_all = sorted(set([lo_, hi_] + [k_ for k_ in ks if lo_ < k_ < hi_]))
        return edges_all, n_t, ("analytic block: near-region interpolants reached by " + str(n_t) +
                                " objects; edges are their breakpoints and the x_star crossings")
    except Exception as e:  # noqa: BLE001
        return None, None, (edges_err or "") + f"; analytic reader: {repr(e)[:200]}"


def interior(edges_all, lo, hi):
    return None if edges_all is None else [e for e in edges_all if lo < e < hi]


def _ls_fit(lo, hi, vals, tt, deg):
    co = np.polynomial.chebyshev.chebfit(tt, vals, deg)
    kk = np.arange(deg + 1)
    qk = np.zeros(deg + 1)
    qk[::2] = 2.0 / (1.0 - kk[::2].astype(float) ** 2)
    integ = 0.5 * (hi - lo) * math.fsum(co * qk)
    resid = vals - np.polynomial.chebyshev.chebval(tt, co)
    return co, integ, resid, qk


def v3_sigma(G, lo, hi, knots, V):
    """Part A's v3 step 2 (leafcc_07d.py), unchanged: pooled sigma_hat, and v3's ref and eps_LS (no GL check)."""
    width = hi - lo
    brk = [lo] + (knots or []) + [hi]
    pieces = []
    for pa, pb in zip(brk[:-1], brk[1:]):
        lp_ = pb - pa
        if len(brk) == 2:
            Mp, dh = LS_M, max(32, int(math.ceil(V / 2.0)) + 32)
            dl = dh - 8
        else:
            Mp = max(PIECE_MIN_M, int(round(LS_M * lp_ / width)))
            dh = max(16, int(math.ceil(V * lp_ / width / 2.0)) + 16)
            dh = min(dh, Mp // 3)
            dl = dh - 4
        ttp = np.cos(np.pi * (np.arange(Mp) + 0.5) / Mp)
        xsp = 0.5 * (pa + pb) + 0.5 * lp_ * ttp
        gsp = G.g_on(xsp)
        step = 8 if len(brk) == 2 else 4
        cap = Mp // 4 if len(brk) == 2 else Mp // 3
        while True:
            _, Ilo_p, rlo_p, _ = _ls_fit(pa, pb, gsp, ttp, dl)
            _, Ihi_p, rhi_p, qk_p = _ls_fit(pa, pb, gsp, ttp, dh)
            s_lo_, s_hi_ = float(np.sqrt(np.mean(rlo_p ** 2))), float(np.sqrt(np.mean(rhi_p ** 2)))
            if s_lo_ <= CONV_RATIO * s_hi_ or dh + step > cap:
                break
            dl, dh = dl + step, dh + step
        Av = np.polynomial.chebyshev.chebvander(ttp, dh)
        try:
            vq = max(float(qk_p @ np.linalg.solve(Av.T @ Av, qk_p)), 0.0)
        except np.linalg.LinAlgError:
            vq = math.inf
        pieces.append({"a": pa, "b": pb, "M": Mp, "d": (dl, dh), "Ilo": Ilo_p, "Ihi": Ihi_p, "rhi": rhi_p, "vq": vq})
    res_hi = np.concatenate([p["rhi"] for p in pieces])
    dof = sum(p["M"] - p["d"][1] - 1 for p in pieces)
    sig_hat = float(np.sqrt(np.sum(res_hi ** 2) / max(dof, 1)))
    s_LS = sig_hat * math.sqrt(sum((0.5 * (p["b"] - p["a"])) ** 2 * p["vq"] for p in pieces))
    dI = sum(abs(p["Ihi"] - p["Ilo"]) for p in pieces)
    return {"sigma_hat": sig_hat, "ref_v3_ls": math.fsum(p["Ihi"] for p in pieces), "eps_LS": dI + 2.0 * s_LS,
            "n_pieces": len(pieces)}


def _gl_piece(G, pa, pb, n):
    h = (pb - pa) / n
    terms, w2 = [], 0.0
    for j in range(n):
        x0 = pa + j * h
        x1 = pa + (j + 1) * h if j < n - 1 else pb
        ww = 0.5 * (x1 - x0) * _GW
        terms.append(ww * G.g_on(0.5 * (x0 + x1) + 0.5 * (x1 - x0) * _GX))
        w2 += float(np.sum(ww * ww))
    return math.fsum(np.concatenate(terms)), w2


def v4_reference(G, lo, hi, knots, sigma_hat):
    """The rule's items 2-5. Returns ref, eps_ref, settled, and each piece's history."""
    brk = [lo] + (knots or []) + [hi]
    pieces = []
    for pa, pb in zip(brk[:-1], brk[1:]):
        hist = []  # (n, I, s)
        settled_at = None
        for k, n in enumerate(LEVELS):
            I, w2 = _gl_piece(G, pa, pb, n)
            hist.append((n, I, sigma_hat * math.sqrt(w2)))
            if k >= 2:
                D1 = abs(hist[k][1] - hist[k - 1][1])
                c1 = math.sqrt(hist[k][2] ** 2 + hist[k - 1][2] ** 2)
                D0 = abs(hist[k - 1][1] - hist[k - 2][1])
                c0 = math.sqrt(hist[k - 1][2] ** 2 + hist[k - 2][2] ** 2)
                if D1 <= SETTLE_K * c1 and D0 <= SETTLE_K * c0:
                    settled_at = k
                    break
        k = len(hist) - 1
        pieces.append({"a": pa, "b": pb, "settled": settled_at is not None, "n": hist[k][0], "I": hist[k][1],
                       "D": abs(hist[k][1] - hist[k - 1][1]), "c": math.sqrt(hist[k][2] ** 2 + hist[k - 1][2] ** 2),
                       "s": hist[k][2], "history": [(n_, I_.hex(), s_) for n_, I_, s_ in hist]})
    ref = math.fsum(p["I"] for p in pieces)
    eps = sum(p["D"] for p in pieces) + 2.0 * math.sqrt(sum(p["s"] ** 2 for p in pieces))
    return {"ref": ref, "eps_ref": eps, "settled": all(p["settled"] for p in pieces), "n_pieces": len(pieces),
            "n_unsettled_pieces": sum(1 for p in pieces if not p["settled"]),
            "panels": [p["n"] for p in pieces], "pieces": pieces}
