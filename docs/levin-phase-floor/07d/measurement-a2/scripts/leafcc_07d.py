"""levin-phase-floor prompt 07d Part A, sections 2.2-2.4 (D-N, D-E, D-T): one call's sampled leaves, each solved both ways
on its own span and measured against a reference. New in 07d; a stated extension of 07c Part A's leafcc_07c.py
(scripts/from_07c_measurement/, a record). Run at the tree's module (07c's code, imported unchanged); nothing is written
but OUT.

For each leaf [a, b] of a captured call (A = ``_Basis_SinCos(theta)``, N the call's order, 24 here):
- **V** as the driver computes it (``A.build_Levin_data`` on the leaf's order-N nodes), and the amplitude-table and
  phase-spline knots strictly inside the leaf (``tools.filon_reference.production_edges``; None where its closure walk
  does not fit the call, e.g. an analytic block);
- **the Clenshaw-Curtis route, as the driver solves it:** ``_adaptive_levin_subregion((a, b), ..., force_direct=True)``
  (on an ascending span exactly ``_adaptive_levin_subregion_cc``, the gate's solve): CC_{2N-1} (its value), the pair
  |CC_{2N-1} - CC_N|, the floor ``phase_err`` (and its parts), ``resasc``; CC_N from the same fine samples. Where the spec
  gives the leaf's own value from the replay (a Clenshaw-Curtis leaf), the re-solve must equal it in hex (``reproduces``);
- **the Levin route,** ``_adaptive_levin_subregion`` with the gate disabled (the module attribute ``SIX_PI`` set to -inf
  for these solves only, then restored; the gate reads it at call time): its value V_P, its floor and truncation term,
  and the step-(4) residual against its halves |V_P - (V_L + V_R)|, so its own declaration max(residual, floor) +
  truncation (``own_err``'s form); the trust check's delta1, delta2 and r from its halves and quarters; whether any
  solve was guarded. Where the spec gives a Levin leaf's own value, V_P must equal it in hex;
- **the reference** (the rule fixed in state/refs_rule_leaf_07d.txt before any reading): the integrand as the driver
  evaluates it (f by ``_sample_vectorized``, the basis by ``A.eval_basis_and_arguments``) sampled at LS_M Chebyshev
  points of the first kind, fitted by Chebyshev least squares at degrees d_lo and d_hi = max(32, ceil(V/2) + 32),
  d_lo = d_hi - 8, each fit integrated exactly (``math.fsum``). ref = the d_hi integral. eps_LS = |I_hi - I_lo| + 2 s_LS,
  s_LS = sigma_hat (b - a)/2 sqrt(q^T (A^T A)^-1 q) (the exact propagation of independent sample noise of RMS sigma_hat,
  the d_hi fit's residual RMS with M/(M - d_hi - 1) correction; q the Chebyshev integrals). **Its check, a second rule:**
  composite Gauss-Legendre, GL_POINTS points a panel, on the leaf split at its interior knots and then halved until it
  has at least GL_MIN_PANELS panels (fsum), noise s_GL = sigma_hat sqrt(sum w^2). The check passes if
  |ref - GL| <= |I_hi - I_lo| + 3 sqrt(s_LS^2 + s_GL^2); then eps_ref = eps_LS; if not, eps_ref = eps_LS + |ref - GL|
  (``ref_check`` False). Also CC_{4N-3} and CC_{8N-7} (fsum);
- **true errors:** |CC_{2N-1} - ref|, |CC_N - ref| (signed errors kept), |V_P - ref|;
- **D-N, the noise:** sigma (residual RMS) of the d_lo and d_hi fits, against mean |g|; the plateau of the fine
  interpolant's Chebyshev coefficients (DCT-I of the 2N - 1 samples) and of CC_{8N-7}'s, by Chebfun's standardChop
  (Aurentz and Trefethen 2017; state/standardChop_rule.md) at tol = 2^-52: sigma_plateau = rms(c_k, k > cutoff)
  sqrt((n - 1)/2), None if not chopped; the same least-squares residual on each component on the same points: each
  amplitude f_i, the raw phase, theta_mod_2pi unwrapped against it (raw + rem(mod - raw)); u |theta_raw| beside the
  phase's sigma; the lag-1..3 autocorrelation of g's and the phase's residuals in node order; the spread of CC_{2N-1}
  over N_SHIFT spans shifted by fractions (seeded, SHIFTS) of the mean fine-node spacing, each against a least-squares
  fit on the widened span, against sigma sqrt(sum w^2); the declared ``theta_abserr`` at the ends, where supplied;
- **the candidates' ingredients** (formed into declarations by d_e_table_07d.py, under the definitions fixed after D-N):
  the fine coefficients, their standardChop cutoff, sum w_j^2 on the fine rule, sum w_j^2 (u |theta_raw,j| |f_j|)^2 and
  sum |w_j| u |theta_raw,j| |f_j| (|f_j| = sqrt(sum_i f_ij^2)), nest3 = |CC_{4N-3} - CC_{2N-1}|, tail_sum4, tail_max4x2;
- **the cause tests** (section 2.3), on every leaf: the composite CC_{2N-1} over the leaf split at its interior knots and
  its true error; the signed errors of CC_N and CC_{2N-1} (an error common to both); the noise ratio.

Spec JSON: {"capture": {path, offset, length, sha256}, "rtol", "atol", "call_value" (hex), "x_span" (hex pair),
"leaves": [{"id", "start", "end" (hex), "type", "value_own" (hex or null)}]}.
Usage: PYTHONPATH=<tree>:<scripts> python -B leafcc_07d.py SPEC OUT
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TREE_LQ_SHA = "d141a80bf0484058a0d7f625926b2e283a4bb64114bc84ccfd81339a6f249be6"
LS_M = 4096
PIECE_MIN_M = 96  # v2: least squares on each knot interval, at least this many points a piece
CONV_RATIO = 1.5  # v3: a fit has converged when sigma(d_lo) <= CONV_RATIO sigma(d_hi)
GL_POINTS = 32
GL_MIN_PANELS = 32
N_SHIFT = 8
SHIFT_SEED = 707406
U = 2.0 ** -53
TOL_CHOP = 2.0 ** -52


def H(x):
    return None if x is None else float(x).hex()


def standard_chop(coeffs, tol):
    """Chebfun's standardChop, transcribed from state/standardChop_rule.md (1-based MATLAB indices kept as such)."""
    if tol >= 1:
        return 1
    n = len(coeffs)
    cutoff = n
    if n < 17:
        return cutoff
    b = np.abs(np.asarray(coeffs, dtype=float))
    m = np.maximum.accumulate(b[::-1])[::-1].copy()
    if m[0] == 0:
        return 1
    env = m / m[0]
    plateau_point = None
    j2 = None
    for j in range(2, n + 1):
        j2 = int(round(1.25 * j + 5))
        if j2 > n:
            return cutoff
        e1, e2 = env[j - 1], env[j2 - 1]
        r = 3 * (1 - math.log(e1) / math.log(tol)) if e1 > 0 else 0.0
        if e1 == 0 or e2 / e1 > r:
            plateau_point = j - 1
            break
    if env[plateau_point - 1] == 0:
        return plateau_point
    j3 = int(np.sum(env >= tol ** (7 / 6)))
    if j3 < j2:
        j2 = j3 + 1
        env[j2 - 1] = tol ** (7 / 6)
    cc = np.log10(env[:j2]) + np.linspace(0, (-1 / 3) * math.log10(tol), j2)
    d = int(np.argmin(cc)) + 1
    return max(d - 1, 1)


def dct1_coeffs(g):
    """Chebyshev coefficients of the interpolant through g at the n extremal (Lobatto) points, ascending in x."""
    n = len(g)
    j = np.arange(n)
    wts = np.ones(n)
    wts[0] = wts[-1] = 0.5
    # the driver's grids run x from a to b; cos(pi j/(n-1)) runs from 1 to -1, so reverse to pair node j with cos
    gj = np.asarray(g, dtype=float)[::-1]
    ck = np.array([(2.0 / (n - 1)) * np.sum(wts * gj * np.cos(np.pi * j * k / (n - 1))) for k in range(n)])
    ck[0] *= 0.5
    ck[-1] *= 0.5
    return ck


def plateau_sigma(ck):
    n = len(ck)
    cut = standard_chop(ck, TOL_CHOP)
    if cut >= n:
        return cut, None
    tail = ck[cut:]
    return cut, float(np.sqrt(np.mean(tail * tail)) * math.sqrt((n - 1) / 2.0))


def autocorr(r, lags=(1, 2, 3)):
    r = np.asarray(r, dtype=float) - np.mean(r)
    v = float(np.dot(r, r))
    return [None if v == 0 else float(np.dot(r[:-k], r[k:]) / v) for k in lags]


def load_capture(c):
    import ray.cloudpickle as cp

    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    return cp.loads(blob)


def main():
    spec_p, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    if hashlib.sha256(Path(LQ.__file__).read_bytes()).hexdigest() != TREE_LQ_SHA:
        raise SystemExit("the module is not the tree's")
    from tools.filon_reference import production_edges

    spec = json.loads(spec_p.read_text())
    rec = load_capture(spec["capture"])
    f = rec["f"]
    th = rec["theta"]
    A = LQ._Basis_SinCos(th)
    N = int(rec["kw"].get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
    rtol = float(spec["rtol"])
    atol = float(spec["atol"])
    call_value = float.fromhex(spec["call_value"])
    xs0, xs1 = (float.fromhex(v) for v in spec["x_span"])
    request = max(atol, rtol * abs(call_value))
    cache = {}
    idl = LQ._LazyUUID()
    raw = th["theta"]
    tmod = th.get("theta_mod_2pi")
    try:
        edges_all, n_tables = production_edges(rec)
        edges_err = None
    except Exception as e:  # noqa: BLE001  an integrand the closure walk does not fit
        edges_all, n_tables, edges_err = None, None, repr(e)[:300]
    if edges_all is None:
        # v2: an analytic block (Bessel phase group). If none of its three phase residuals and three amplitudes has a
        # near-region interpolant (each a closed-form tail), it has no tables: its only edges are where an argument
        # c x crosses an object's x_star. Otherwise its knots are unknown (None).
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
                    ks.append(math.log(xst / c_))  # where c x crosses x_star
                reached = co_.near is not None and c_ * math.exp(lo_) < xst
                if reached:
                    n_t += 1
                    for interp in (co_._r_interp, co_._ell_interp):
                        if interp is not None:
                            # PiecewiseChebyshev over log of the object's argument: a breakpoint B is at u = B - log c
                            ks.extend(float(B) - math.log(c_) for B in np.asarray(interp.breakpoints).ravel())
            edges_all = sorted(set([lo_, hi_] + [k_ for k_ in ks if lo_ < k_ < hi_]))
            n_tables = n_t
            edges_err = ("analytic block: near-region interpolants reached by " + str(n_t) + " objects; edges are their "
                         "breakpoints and the x_star crossings")
        except Exception as e:  # noqa: BLE001
            edges_err = (edges_err or "") + f"; analytic reader: {repr(e)[:200]}"

    def f_on(grid):
        return np.vstack([LQ._sample_vectorized(fn, grid, cache, ("f", id(fn))) for fn in f])

    def g_on(grid, with_parts=False):
        fs = f_on(grid)
        ba = [A.eval_basis_and_arguments(x) for x in grid]
        bs = np.array([v for v, _ in ba]).T
        g = np.sum(fs * bs, axis=0)
        if with_parts:
            return g, fs
        return g

    def cc_rule(span, M):
        grid, _ = LQ.chebyshev_matrices(span, M)
        g = g_on(grid)
        w = LQ._cc_weights(span, M)
        return math.fsum(w * g), grid, g, w

    def ls_fit(lo, hi, vals, tt, deg):
        co = np.polynomial.chebyshev.chebfit(tt, vals, deg)
        kk = np.arange(deg + 1)
        qk = np.zeros(deg + 1)
        qk[::2] = 2.0 / (1.0 - kk[::2].astype(float) ** 2)
        integ = 0.5 * (hi - lo) * math.fsum(co * qk)
        resid = vals - np.polynomial.chebyshev.chebval(tt, co)
        return co, integ, resid, qk

    def interior_knots(lo, hi):
        if edges_all is None:
            return None
        return [e for e in edges_all if lo < e < hi]

    gl_x, gl_w = np.polynomial.legendre.leggauss(GL_POINTS)
    rng = np.random.default_rng(SHIFT_SEED)
    out = []
    t_all = time.perf_counter()
    for lf in spec["leaves"]:
      try:  # v3: one leaf's failure is recorded and does not end the spec
        t0 = time.perf_counter()
        a, b = float.fromhex(lf["start"]), float.fromhex(lf["end"])
        if not a < b:
            out.append({"id": lf["id"], "error": "not an ascending span"})
            continue
        lo, hi = a, b
        width = hi - lo
        grid_N, D_N = LQ.chebyshev_matrices((lo, hi), N)
        V = float(A.build_Levin_data(grid_N, D_N, need_AmatT=False, vectorize_cache=cache)[4])
        knots = interior_knots(lo, hi)
        # --- the Clenshaw-Curtis route, as the driver solves it
        d = LQ._adaptive_levin_subregion((lo, hi), f, A, id_label=idl, chebyshev_order=N, rtol=rtol,
                                         vectorize_cache=cache, force_direct=True)
        fine = 2 * N - 1
        grid_f, _ = LQ.chebyshev_matrices((lo, hi), fine)
        g_f, f_f = g_on(grid_f, with_parts=True)
        w_f = LQ._cc_weights((lo, hi), fine)
        w_c = LQ._cc_weights((lo, hi), N)
        cc47 = float(d["value"])
        cc47_check = float(np.dot(w_f, g_f))
        cc24 = float(np.dot(w_c, g_f[::2]))
        pair = float(d["abserr_direct"])
        floor = float(d["phase_err"])
        # --- the Levin route, the gate disabled
        six = LQ.SIX_PI
        try:
            LQ.SIX_PI = -math.inf
            c_ = (lo + hi) / 2.0
            lp = LQ._adaptive_levin_subregion((lo, hi), f, A, id_label=idl, chebyshev_order=N, rtol=rtol, vectorize_cache=cache)
            lL = LQ._adaptive_levin_subregion((lo, c_), f, A, id_label=idl, chebyshev_order=N, rtol=rtol, vectorize_cache=cache)
            lR = LQ._adaptive_levin_subregion((c_, hi), f, A, id_label=idl, chebyshev_order=N, rtol=rtol, vectorize_cache=cache)
            cL, cR = (lo + c_) / 2.0, (c_ + hi) / 2.0
            q = [LQ._adaptive_levin_subregion(s, f, A, id_label=idl, chebyshev_order=N, rtol=rtol, vectorize_cache=cache)
                 for s in ((lo, cL), (cL, c_), (c_, cR), (cR, hi))]
        finally:
            LQ.SIX_PI = six
        if any(x.get("is_direct") for x in [lp, lL, lR] + q):
            out.append({"id": lf["id"], "error": "a Levin-route solve took the Clenshaw-Curtis branch"})
            continue
        vP = float(lp["value"])
        s1 = float(lL["value"]) + float(lR["value"])
        s2 = sum(float(x["value"]) for x in q)
        lev_res = abs(vP - s1)
        lev_floor = float(lp.get("phase_err") or 0.0)
        lev_trunc = float(lp.get("abserr_truncation") or 0.0)
        delta1, delta2 = abs(vP - s1), abs(s1 - s2)
        r_ = (0.0 if delta2 == 0 else math.inf) if delta1 == 0 else delta2 / delta1
        lev_guarded = [bool(x.get("guarded", False)) for x in [lp, lL, lR] + q]
        # --- the reference (v2): least squares on each knot interval (one piece if none, or if the knots are unknown)
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
            gsp, fsp = g_on(xsp, with_parts=True)
            # v3: raise the degrees until the fit has converged: while the d_lo fit's residual RMS exceeds CONV_RATIO
            # times the d_hi fit's, both step up by the step (8 for one piece, 4 for several), up to M_p // 4 (one piece)
            # or M_p // 3 (several)
            step = 8 if len(brk) == 2 else 4
            cap = Mp // 4 if len(brk) == 2 else Mp // 3
            while True:
                _, Ilo_p, rlo_p, _ = ls_fit(pa, pb, gsp, ttp, dl)
                _, Ihi_p, rhi_p, qk_p = ls_fit(pa, pb, gsp, ttp, dh)
                s_lo_, s_hi_ = float(np.sqrt(np.mean(rlo_p ** 2))), float(np.sqrt(np.mean(rhi_p ** 2)))
                if s_lo_ <= CONV_RATIO * s_hi_ or dh + step > cap:
                    break
                dl, dh = dl + step, dh + step
            Av = np.polynomial.chebyshev.chebvander(ttp, dh)
            try:
                vq = max(float(qk_p @ np.linalg.solve(Av.T @ Av, qk_p)), 0.0)
            except np.linalg.LinAlgError:
                vq = math.inf
            pieces.append({"a": pa, "b": pb, "M": Mp, "d": (dl, dh), "tt": ttp, "xs": xsp, "gs": gsp, "fs": fsp,
                           "Ilo": Ilo_p, "Ihi": Ihi_p, "rlo": rlo_p, "rhi": rhi_p, "vq": vq})
        xs = np.concatenate([p["xs"] for p in pieces])
        gs = np.concatenate([p["gs"] for p in pieces])
        fs = np.hstack([p["fs"] for p in pieces])
        res_lo = np.concatenate([p["rlo"] for p in pieces])
        res_hi = np.concatenate([p["rhi"] for p in pieces])
        dof = sum(p["M"] - p["d"][1] - 1 for p in pieces)
        I_hi = math.fsum(p["Ihi"] for p in pieces)
        I_lo = math.fsum(p["Ilo"] for p in pieces)
        d_lo, d_hi = pieces[0]["d"] if len(pieces) == 1 else (min(p["d"][0] for p in pieces), max(p["d"][1] for p in pieces))
        sig_lo = float(np.sqrt(np.mean(res_lo ** 2)))
        sig_hi = float(np.sqrt(np.mean(res_hi ** 2)))
        sig_hat = float(np.sqrt(np.sum(res_hi ** 2) / max(dof, 1)))
        s_LS = sig_hat * math.sqrt(sum((0.5 * (p["b"] - p["a"])) ** 2 * p["vq"] for p in pieces))
        dI = sum(abs(p["Ihi"] - p["Ilo"]) for p in pieces)
        eps_LS = dI + 2.0 * s_LS

        def piecewise_resid(values):
            out_, k_ = [], 0
            for p in pieces:
                v_ = values[k_:k_ + p["M"]]
                k_ += p["M"]
                out_.append(ls_fit(p["a"], p["b"], v_, p["tt"], p["d"][1])[2])
            return np.concatenate(out_)

        # the second rule: composite Gauss-Legendre split at the knots
        pan = list(zip(brk[:-1], brk[1:]))
        while len(pan) < GL_MIN_PANELS:
            pan = [p for (pa, pb) in pan for p in ((pa, (pa + pb) / 2.0), ((pa + pb) / 2.0, pb))]
        gl_terms, gl_w2 = [], 0.0
        for pa, pb in pan:
            xg = 0.5 * (pa + pb) + 0.5 * (pb - pa) * gl_x
            wg = 0.5 * (pb - pa) * gl_w
            gl_terms.append(wg * g_on(xg))
            gl_w2 += float(np.sum(wg * wg))
        GL = math.fsum(np.concatenate(gl_terms))
        s_GL = sig_hat * math.sqrt(gl_w2)
        ref = I_hi
        check_ok = abs(ref - GL) <= dI + 3.0 * math.sqrt(s_LS ** 2 + s_GL ** 2)
        eps_ref = eps_LS if check_ok else eps_LS + abs(ref - GL)
        cc93, grid93, g93, w93 = cc_rule((lo, hi), 4 * N - 3)
        cc185, grid185, g185, w185 = cc_rule((lo, hi), 8 * N - 7)
        # --- true errors
        e47, e24, eL = cc47 - ref, cc24 - ref, vP - ref
        # --- D-N: plateaus, components, structure, shifts, declared
        ck_f = dct1_coeffs(g_f)
        cut_f, sig_pl_f = plateau_sigma(ck_f)
        ck_185 = dct1_coeffs(g185)
        cut_185, sig_pl_185 = plateau_sigma(ck_185)
        comp = {}
        for i in range(fs.shape[0]):
            rr = piecewise_resid(fs[i])
            comp[f"f{i}"] = {"sigma": float(np.sqrt(np.mean(rr ** 2))), "rms": float(np.sqrt(np.mean(fs[i] ** 2))),
                             "acf": autocorr(rr)}
        th_raw = np.array([float(raw(x)) for x in xs])
        th0 = float(raw(0.5 * (lo + hi)))  # v2: the fit is made on theta - theta0 (exact by Sterbenz where |theta| >> its variation)
        rr_raw = piecewise_resid(th_raw - th0)
        comp["theta_raw"] = {"sigma": float(np.sqrt(np.mean(rr_raw ** 2))), "mean_abs": float(np.mean(np.abs(th_raw))),
                             "u_mean_abs": U * float(np.mean(np.abs(th_raw))),
                             "ulp_mean": float(np.mean(np.spacing(np.abs(th_raw)))), "acf": autocorr(rr_raw)}
        if tmod is not None:
            th_m = np.array([float(tmod(x)) for x in xs])
            # v2: unwrap exactly. k = round((theta_raw - theta_mod)/2pi) is an integer; theta_mod + 2 pi k - theta0 is formed
            # as theta_mod + fl(2 pi k - theta0), the second term in 40 digits, so that theta_mod's own precision is kept
            # (theta_raw + (theta_mod - theta_raw) in double would carry theta_raw's rounding back in).
            from mpmath import mp as _mp
            _mp.dps = 40
            kk_ = np.round((th_raw - th_m) / (2 * np.pi)).astype(np.int64)
            offs = {int(k_): float(2 * _mp.pi * int(k_) - _mp.mpf(th0)) for k_ in set(kk_.tolist())}
            th_rel = th_m + np.array([offs[int(k_)] for k_ in kk_])
            rr_m = piecewise_resid(th_rel)
            rr_d = piecewise_resid(th_rel - (th_raw - th0))
            comp["theta_mod_2pi"] = {"sigma": float(np.sqrt(np.mean(rr_m ** 2))), "acf": autocorr(rr_m),
                                     "sigma_of_difference_from_raw": float(np.sqrt(np.mean(rr_d ** 2)))}
        fmag = np.sqrt(np.sum(fs * fs, axis=0))
        comp["g"] = {"sigma": sig_hi, "acf": autocorr(res_hi), "mean_abs": float(np.mean(np.abs(gs))),
                     "rms_fmag": float(np.sqrt(np.mean(fmag ** 2)))}
        # shifts (v2): CC_{2N-1} on spans shifted by s h, against ref - int_a^{a+sh} g + int_b^{b+sh} g, each end piece by
        # 32-point Gauss-Legendre split at any knot inside it
        h = width / (fine - 1)

        def gl_small(x0, x1):
            kn = [e for e in (edges_all or []) if min(x0, x1) < e < max(x0, x1)]
            bb = sorted([x0, x1] + kn)
            tot = math.fsum(math.fsum(0.5 * (pb - pa) * gl_w * g_on(0.5 * (pa + pb) + 0.5 * (pb - pa) * gl_x))
                            for pa, pb in zip(bb[:-1], bb[1:]))
            return tot if x1 >= x0 else -tot

        shifts = rng.uniform(-0.5, 0.5, N_SHIFT)
        sh_err = []
        clo, chi = min(xs0, xs1), max(xs0, xs1)
        for s in shifts:
            # v3: a shifted span stays inside the call's span (the integrand is not defined beyond it): a shift that
            # would leave it is reflected; if both directions would, the shift is skipped (None)
            if lo + s * h < clo or hi + s * h > chi:
                s = -s
            if lo + s * h < clo or hi + s * h > chi:
                sh_err.append(None)
                continue
            sp = (lo + s * h, hi + s * h)
            try:
                gsg, _ = LQ.chebyshev_matrices(sp, fine)
                sh_err.append(float(np.dot(LQ._cc_weights(sp, fine), g_on(gsg))) - (ref - gl_small(lo, sp[0]) + gl_small(hi, sp[1])))
            except Exception:  # noqa: BLE001  outside the integrand's domain after all
                sh_err.append(None)
        sw2 = float(np.sum(w_f * w_f))
        s_cc = sig_hat * math.sqrt(sw2)
        th_f = np.array([float(raw(x)) for x in grid_f])
        fmag_f = np.sqrt(np.sum(f_f * f_f, axis=0))
        eta_f = U * np.abs(th_f) * fmag_f
        dec = None
        if A.theta_abserr_at(lo) is not None:
            dec = {"theta_abserr_lo": A.theta_abserr_at(lo), "theta_abserr_hi": A.theta_abserr_at(hi)}
        # cause test: split at the knots
        split = None
        if knots:
            kb = [lo] + knots + [hi]
            vals = []
            for pa, pb in zip(kb[:-1], kb[1:]):
                gq, _ = LQ.chebyshev_matrices((pa, pb), fine)
                vals.append(float(np.dot(LQ._cc_weights((pa, pb), fine), g_on(gq))))
            sv = math.fsum(vals)
            split = {"n_pieces": len(vals), "cc_split": H(sv), "err_split": H(sv - ref)}
        own = lf.get("value_own")
        rep_cc = rep_lev = None
        if own is not None:
            if lf.get("type") == "direct":
                rep_cc = H(cc47) == own
            else:
                rep_lev = H(vP) == own
        out.append({
            "id": lf["id"], "start": lf["start"], "end": lf["end"], "type": lf.get("type"), "N": N, "V": V,
            "V_over_6pi": V / (6 * math.pi), "width": width, "n_knots": None if knots is None else len(knots),
            "reproduces_cc": rep_cc, "reproduces_levin": rep_lev,
            "cc47": H(cc47), "cc47_check": H(cc47_check), "cc24": H(cc24), "pair": H(pair), "floor": H(floor),
            "floor_less_node": H(d.get("phase_err_less_node")), "node_E": H(d.get("node_error_signed")),
            "floor_declared": H(d.get("phase_err_declared")), "resasc": H(d.get("resasc")), "phase_span": H(d.get("phase_span")),
            "levin": {"vP": H(vP), "s1": H(s1), "s2": H(s2), "residual": H(lev_res), "floor": H(lev_floor),
                      "trunc": H(lev_trunc), "own_decl": H(max(lev_res, lev_floor) + lev_trunc),
                      "delta1": H(delta1), "delta2": H(delta2), "r": r_, "guarded": lev_guarded,
                      "lstsq": bool(V >= LQ._LEVIN_DIRECT_SOLVE_PHASE_SPAN)},
            "ref": H(ref), "eps_ref": H(eps_ref), "eps_LS": H(eps_LS), "I_lo": H(I_lo), "n_pieces": len(pieces), "s_LS": H(s_LS), "GL": H(GL),
            "s_GL": H(s_GL), "ref_check": bool(check_ok), "deg": [d_lo, d_hi], "deg_pieces": [list(p["d"]) for p in pieces], "cc93": H(cc93), "cc185": H(cc185),
            "err47": H(e47), "err24": H(e24), "err_levin": H(eL),
            "request_share": request * width / abs(xs1 - xs0), "request": request,
            "noise": {"sigma_lo": sig_lo, "sigma_hi": sig_hi, "sigma_hat": sig_hat,
                      "sigma_over_mean_abs_g": sig_hi / max(float(np.mean(np.abs(gs))), 1e-300),
                      "plateau_fine": {"cutoff": cut_f, "sigma": sig_pl_f},
                      "plateau_185": {"cutoff": cut_185, "sigma": sig_pl_185},
                      "components": comp, "shift_errors": [H(x) for x in sh_err], "shifts": shifts.tolist(),
                      "s_cc": s_cc, "declared": dec},
            "ingredients": {"ck_fine": [H(x) for x in ck_f], "chop_fine": cut_f, "sum_w2": sw2,
                            "sum_w2_eta2": float(np.sum(w_f * w_f * eta_f * eta_f)),
                            "sum_absw_eta": float(np.sum(np.abs(w_f) * eta_f)),
                            "nest3": H(abs(cc93 - cc47)),
                            "tail_sum4": H(0.5 * width * float(np.sum(np.abs(ck_f[-4:])))),
                            "tail_max4x2": H(width * float(np.max(np.abs(ck_f[-4:])))),
                            "scale": H(float(np.dot(np.abs(w_f), np.abs(g_f))))},
            "split_at_knots": split, "wall_s": time.perf_counter() - t0,
        })
      except Exception as e:  # noqa: BLE001
        import traceback
        out.append({"id": lf["id"], "error": "exception: " + repr(e)[:300], "traceback": traceback.format_exc()[-1500:]})
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "module": LQ.__file__, "module_sha256": TREE_LQ_SHA, "spec": str(spec_p),
                      "spec_sha256": hashlib.sha256(spec_p.read_bytes()).hexdigest(),
                      "constants": {"LS_M": LS_M, "GL_POINTS": GL_POINTS, "GL_MIN_PANELS": GL_MIN_PANELS, "N_SHIFT": N_SHIFT,
                                    "SHIFT_SEED": SHIFT_SEED, "TOL_CHOP": TOL_CHOP}},
           "sid": spec.get("sid"), "rtol": rtol, "N": N, "edges_error": edges_err, "n_tables": n_tables,
           "wall_s": time.perf_counter() - t_all, "leaves": out}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc))
    print(f"{spec_p.name}: {len(out)} leaves, {sum(1 for x in out if 'error' in x)} errors, "
          f"reproduce cc {sum(1 for x in out if x.get('reproduces_cc'))}/{sum(1 for x in out if x.get('reproduces_cc') is not None)}, "
          f"levin {sum(1 for x in out if x.get('reproduces_levin'))}/{sum(1 for x in out if x.get('reproduces_levin') is not None)}, "
          f"{doc['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
