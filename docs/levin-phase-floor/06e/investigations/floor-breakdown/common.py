"""Shared code for the floor-breakdown measurements (levin-phase-floor 06d's floor on 07's (R) cases).

Run from the throwaway worktree at e5764a6 with PYTHONPATH=. so that ``AdaptiveLevin`` is 06d's.

* cases: prompt 07's (R) family, theta = omega x + (c/2)(x - x_s)^2 on [0, 1], with
  f_sin = exp(-x), f_cos = cos(3x), the phase handed to the driver as {"theta", "theta_deriv"}
  exactly as docs/levin-phase-floor/07/calibration/scripts/calibrate.py does (copied, not imported).
* closed_leaf: the exact integral over [x_a, x_b], reference.py's closed_R() with u_0 = x_a - x_s,
  u_1 = x_b - x_s, at MP_DPS digits.
* floor_terms: each term of 06d's floor as the code forms it, recomputed from the same arrays.
* mp_levin / mp_cc: the same discrete scheme as the double solve (same order, same Chebyshev
  nodes, same basis, same linear algebra: the complexified collocation system and the
  nested CC rule) carried out in mpmath at MP_DPS digits with the exact nodes and exact
  matrices/weights. double - mp is the leaf's round-off; mp - exact its truncation.
"""

import datetime
import hashlib
import math
import platform
import socket
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

import AdaptiveLevin.levin_quadrature as lq

MP_DPS = 60
ORDER = 16
RTOL = 1.0e-10
ATOL_FACTOR = 1.0e-14
X_S = 0.7236067977499790

CASES = [
    {"name": "c=1e6 omega=0", "c": 1.0e6, "x_s": X_S, "omega": 0.0},
    {"name": "c=1e8 omega=0", "c": 1.0e8, "x_s": X_S, "omega": 0.0},
    {"name": "c=1e8 omega=1", "c": 1.0e8, "x_s": X_S, "omega": 1.0},
]

HERE = Path(__file__).resolve().parent
SCRIPTS_07 = Path(
    "/Users/ds283/Documents/Code/SecondaryGWKit/docs/levin-phase-floor/07/calibration/scripts"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header(script):
    import scipy

    wt = Path(lq.__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", "--no-optional-locks", *a], cwd=wt, capture_output=True, text=True
    ).stdout
    status = git("status", "--short")
    return {
        "script": str(Path(script).resolve()),
        "script_sha256": sha256(script),
        "scripts_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "scripts_07_sha256": {
            p.name: sha256(p) for p in sorted(SCRIPTS_07.glob("*.py"))
        },
        "levin_quadrature": str(Path(lq.__file__).resolve()),
        "levin_quadrature_sha256": sha256(lq.__file__),
        "worktree": str(wt),
        "git_head": git("rev-parse", "HEAD").strip(),
        "tree_dirty": bool(status.strip()),
        "git_status_short": status.splitlines(),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mp.__version__,
        "mp_dps": MP_DPS,
    }


# ---------------------------------------------------------------- the problem, as calibrate.py


def amplitudes():
    return [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]


def phase(case):
    c, x_s, omega = case["c"], case["x_s"], case["omega"]
    if omega == 0.0:
        return {
            "theta": lambda x: 0.5 * c * (x - x_s) ** 2,
            "theta_deriv": lambda x: c * (x - x_s),
        }
    return {
        "theta": lambda x: omega * x + 0.5 * c * (x - x_s) ** 2,
        "theta_deriv": lambda x: omega + c * (x - x_s),
    }


def x_root(case):
    with mp.workdps(40):
        return float(
            mp.mpf(case["x_s"]) - mp.mpf(case["omega"]) / mp.mpf(case["c"])
        )


# ---------------------------------------------------------------- exact leaf integral


def closed_leaf(case, xa, xb):
    """reference.closed_R() with u_0 = x_a - x_s, u_1 = x_b - x_s (an mpf at MP_DPS)."""
    with mp.workdps(MP_DPS):
        c = mp.mpf(case["c"])
        x_s = mp.mpf(case["x_s"])
        omega = mp.mpf(case["omega"])
        beta = mp.mpc(0, -1) * c / 2
        sb = mp.sqrt(beta)
        u0, u1 = mp.mpf(xa) - x_s, mp.mpf(xb) - x_s

        def J0(g):
            shift = mp.mpc(0, 1) * g / c
            pref = (
                mp.exp(g * x_s)
                * mp.exp(mp.mpc(0, 1) * g**2 / (2 * c))
                * mp.sqrt(mp.pi)
                / (2 * sb)
            )
            return pref * (mp.erf(sb * (u1 - shift)) - mp.erf(sb * (u0 - shift)))

        def J(alpha):
            return J0(alpha + mp.mpc(0, 1) * omega)

        return +(
            mp.im(J(mp.mpf(-1))) + mp.re(J(mp.mpc(0, 3)) + J(mp.mpc(0, -3))) / 2
        )


# ---------------------------------------------------------------- exact base matrices

_BASE = {}


def mp_base(N):
    """Exact (MP_DPS) Chebyshev extremal nodes t_i = cos(i pi/(N-1)) (descending, as
    chebyshev_matrices) and their differentiation matrix (barycentric form)."""
    if N in _BASE:
        return _BASE[N]
    with mp.workdps(MP_DPS + 10):
        n = N - 1
        t = [mp.cos(mp.pi * i / n) for i in range(N)]
        w = [(-1) ** i * (mp.mpf(1) / 2 if i in (0, n) else 1) for i in range(N)]
        D = mp.matrix(N, N)
        for i in range(N):
            s = mp.mpf(0)
            for j in range(N):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (t[i] - t[j])
                    s += D[i, j]
            D[i, i] = -s
    _BASE[N] = (t, D)
    return _BASE[N]


_CCW = {}


def mp_cc_weights_base(N):
    """_cc_weights_base(N) carried out exactly."""
    if N in _CCW:
        return _CCW[N]
    with mp.workdps(MP_DPS + 10):
        n = N - 1
        th = [mp.pi * i / n for i in range(N)]
        w = [mp.mpf(0)] * N
        w0 = mp.mpf(1) / (n * n - 1) if n % 2 == 0 else mp.mpf(1) / (n * n)
        w[0] = w0
        w[-1] = w0
        for i in range(1, n):
            v = mp.mpf(1)
            if n % 2 == 0:
                for k in range(1, n // 2):
                    v -= 2 * mp.cos(2 * k * th[i]) / (4 * k * k - 1)
                v -= mp.cos(n * th[i]) / (n * n - 1)
            else:
                for k in range(1, (n - 1) // 2 + 1):
                    v -= 2 * mp.cos(2 * k * th[i]) / (4 * k * k - 1)
            w[i] = 2 * v / n
    _CCW[N] = w
    return w


def _mp_funcs(case):
    c = mp.mpf(case["c"])
    x_s = mp.mpf(case["x_s"])
    omega = mp.mpf(case["omega"])
    return (
        lambda x: omega * x + c / 2 * (x - x_s) ** 2,
        lambda x: omega + c * (x - x_s),
        lambda x: mp.exp(-x),
        lambda x: mp.cos(3 * x),
    )


def mp_levin(case, xa, xb, p_use, N=ORDER, nodes=None, end_args=None):
    """The complexified Levin collocation solve (D + i diag theta') q = f_1 + i f_2 on
    chebyshev_matrices((xa, xb), N)'s nodes, exactly; value = sum_i P_i(b) w_i(b) - P_i(a) w_i(a)
    with the components the double solve dropped (p_use False) dropped here too."""
    with mp.workdps(MP_DPS):
        th, thp, f1, f2 = _mp_funcs(case)
        t, Db = mp_base(N)
        a, b = mp.mpf(xa), mp.mpf(xb)
        mid, half = (a + b) / 2, (b - a) / 2
        x = [mid + half * ti for ti in t] if nodes is None else nodes
        L = mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                L[i, j] = Db[i, j] / half
            L[i, i] += mp.mpc(0, 1) * thp(x[i])
        rhs = mp.matrix([f1(xi) + mp.mpc(0, 1) * f2(xi) for xi in x])
        q = mp.lu_solve(L, rhs)
        P = [[mp.re(q[i]) for i in range(N)], [mp.im(q[i]) for i in range(N)]]
        # node 0 is the upper endpoint b, node N-1 the lower a
        if end_args is None:
            ab, aa = th(x[0]), th(x[-1])
        else:
            # the basis at the endpoints from the double-computed arguments (exactly as mpf)
            ab, aa = mp.mpf(end_args[0]), mp.mpf(end_args[1])
        wb = [mp.sin(ab), mp.cos(ab)]
        wa = [mp.sin(aa), mp.cos(aa)]
        upper = sum(P[i][0] * wb[i] for i in range(2) if p_use[i])
        lower = sum(P[i][-1] * wa[i] for i in range(2) if p_use[i])
        return +(upper - lower)


def mp_cc(case, xa, xb, N=ORDER, nodes=None):
    """CC_{2N-1} on chebyshev_matrices((xa, xb), 2N-1)'s nodes, exactly (exact weights). With
    nodes given (e.g. the double nodes, as mpf), the rule is evaluated there instead."""
    fine = 2 * N - 1
    with mp.workdps(MP_DPS):
        th, thp, f1, f2 = _mp_funcs(case)
        t = [mp.cos(mp.pi * i / (fine - 1)) for i in range(fine)]
        a, b = mp.mpf(xa), mp.mpf(xb)
        mid, half = (a + b) / 2, (b - a) / 2
        x = [mid + half * ti for ti in t] if nodes is None else nodes
        w = mp_cc_weights_base(fine)
        s = mp.mpf(0)
        for wk, xk in zip(w, x):
            tk = th(xk)
            s += wk * (f1(xk) * mp.sin(tk) + f2(xk) * mp.cos(tk))
        return +(s * half)


# ---------------------------------------------------------------- the floor, term by term


def floor_terms(case, span, data, N=ORDER):
    """Every term of 06d's floor for the leaf `span`, recomputed from the same arrays the solve
    used, as the code forms them. data is _adaptive_levin_subregion's return for span."""
    f, BasisData, cache = context(case)
    smp = lambda fn, g, tag: lq._sample_vectorized(fn, g, cache, (tag, id(BasisData) if tag != "f" else id(fn)))  # noqa: E731
    a, b = span
    width = float(np.fabs(b - a))
    if not data["is_direct"]:
        grid, Dmat = lq.chebyshev_matrices(span, N)
        thp = smp(BasisData._theta_deriv, grid, "theta_deriv")
        F = np.vstack([smp(fn, grid, "f") for fn in f])
        f_scale = float(np.max(np.sqrt(np.sum(F * F, axis=0))))
        G0, G1 = lq._levin_G0_G1(thp, width)
        k2 = float(N) * float(N)
        if not (G0 > 0.0) or G0 <= lq._LEVIN_ROUNDOFF_G0_NOISE_FLOOR * G1:
            T1 = T2 = T3 = float("inf")
        else:
            pref = lq._LEVIN_ROUNDOFF_SAFETY * lq.MACHINE_EPSILON * f_scale * 0.5 * width
            T1 = pref
            T2 = pref * G1 / G0
            T3 = pref * max(G1, k2) / G0
        recon = lq._roundoff_floor(f_scale, width, G0, G1, N)
        return {
            "kind": "Levin",
            "terms": {"L1_eps_f_h2": T1, "L2_G1_over_G0": T2, "L3_maxG1k2_over_G0": T3},
            "G0": G0,
            "G1": G1,
            "G1_over_G0": G1 / G0 if G0 > 0 else float("inf"),
            "k2_dominates": bool(k2 > G1),
            "f_scale": f_scale,
            "theta_prime_min": float(np.min(np.fabs(thp))),
            "theta_prime_max": float(np.max(np.fabs(thp))),
            "declared_term": 0.0,
            "floor_recomputed": recon,
            "floor_code": data["phase_err"],
            "phase_span": data["phase_span"],
            "p_ratios": data["p_ratios"],
            "abserr_truncation": data["abserr_truncation"],
        }
    # fallback (CC) leaf
    grid, _ = lq.chebyshev_matrices(span, N)
    thp_coarse = smp(BasisData._theta_deriv, grid, "theta_deriv")
    fine = 2 * N - 1
    fine_grid, D_fine = lq.chebyshev_matrices(span, fine)
    f_fine = np.vstack([smp(fn, fine_grid, "f") for fn in f])
    ba = [BasisData.eval_basis_and_arguments(x) for x in fine_grid]
    basis = np.array([v for v, _ in ba]).T
    args = np.array([g for _, g in ba], dtype=float).T
    w = lq._cc_weights(span, fine)
    abs_w = np.fabs(w)
    n = abs_w.shape[0]
    abs_f = np.fabs(f_fine)
    S = float(np.dot(abs_w, np.sum(np.fabs(f_fine * basis), axis=0)))
    arg_term = float(np.dot(abs_w, np.sum(abs_f * np.fabs(args), axis=0)))
    f_prime = np.vstack([np.matmul(D_fine, row) for row in f_fine])
    delta = lq._cc_node_error_bound(span)
    tpb = float(np.max(np.fabs(thp_coarse)))
    node_theta = delta * float(np.dot(abs_w, np.sum(tpb * abs_f, axis=0)))
    node_fprime = delta * float(np.dot(abs_w, np.sum(np.fabs(f_prime), axis=0)))
    node_total = delta * float(
        np.dot(abs_w, np.sum(tpb * abs_f + np.fabs(f_prime), axis=0))
    )
    C1 = (n + lq._CC_ROUNDOFF_PER_TERM_COUNT) * lq._UNIT_ROUNDOFF * S
    C2 = lq._UNIT_ROUNDOFF * arg_term
    recon = lq._cc_roundoff_floor(
        w, f_fine, basis, args, delta, tpb, f_prime
    )
    return {
        "kind": "CC",
        "terms": {
            "C1_n4_uS": C1,
            "C2_argument": C2,
            "C3_node": node_total,
        },
        "C3_node_theta_part": node_theta,
        "C3_node_fprime_part": node_fprime,
        "node_delta": delta,
        "theta_prime_bound": tpb,
        "declared_term": 0.0,
        "floor_recomputed": recon,
        "floor_code": data["phase_err"],
        "phase_span": data["phase_span"],
        "fine_nodes": [float(x) for x in fine_grid],
    }


_CTX = {}


def context(case):
    """(f, BasisData, vectorize_cache) for this case, with the cache primed as 06d's driver primes
    it: each callable probed once (_detect_vectorized) on the first grid it meets, the root
    region's. The driver's own decision is then reproduced on every leaf (it matters: theta's
    array and scalar evaluations differ by an ulp here, numpy's x**2 against the scalar pow(),
    so the driver samples theta by its scalar loop)."""
    key = (case["c"], case["x_s"], case["omega"])
    if key not in _CTX:
        f = amplitudes()
        B = lq._Basis_SinCos(phase(case))
        cache = {}
        root, _ = lq.chebyshev_matrices((0.0, 1.0), ORDER)
        lq._sample_vectorized(B._theta, root, cache, ("theta", id(B)))
        lq._sample_vectorized(B._theta_deriv, root, cache, ("theta_deriv", id(B)))
        for fn in f:
            lq._sample_vectorized(fn, root, cache, ("f", id(fn)))
        _CTX[key] = (f, B, cache)
    return _CTX[key]


def solve_leaf(case, span):
    f, B, cache = context(case)
    return lq._adaptive_levin_subregion(
        span,
        f,
        B,
        id_label=None,
        chebyshev_order=ORDER,
        rtol=RTOL,
        vectorize_cache=cache,
    )


def analyse_leaf(case, span, exact=True, split_cc_nodes=True):
    """Double solve, floor terms, same-scheme mp value, and (optionally) the exact integral."""
    data = solve_leaf(case, span)
    ft = floor_terms(case, span, data)
    v = data["value"]
    out = {"a": span[0], "b": span[1], "h": span[1] - span[0], "value": v}
    out.update(ft)
    if ft["kind"] == "Levin":
        p_use = [np.isfinite(r) and r > RTOL for r in data["p_ratios"]]
        out["p_use"] = [bool(u) for u in p_use]
        mv = mp_levin(case, span[0], span[1], p_use)
        grid, _ = lq.chebyshev_matrices(span, ORDER)
        out["grid_ends_exact"] = bool(grid[0] == span[1] and grid[-1] == span[0])
        _, B, cache = context(case)
        th_d = lq._sample_vectorized(B._theta, grid, cache, ("theta", id(B)))
        out["theta_vectorized"] = bool(cache[("theta", id(B))])
        with mp.workdps(MP_DPS):
            mv_end = mp_levin(
                case, span[0], span[1], p_use, end_args=(th_d[0], th_d[-1])
            )
            # the part of the round-off due to the rounding of theta at the two endpoints
            # (common-mode between neighbours: it telescopes in the sum where p is continuous)
            out["roundoff_endpoint_phase_part"] = float(mv_end - mv)
            out["roundoff_rest_part"] = float(mp.mpf(v) - mv_end)
    else:
        mv = mp_cc(case, span[0], span[1])
        if split_cc_nodes:
            with mp.workdps(MP_DPS):
                mv_dn = mp_cc(
                    case, span[0], span[1], nodes=[mp.mpf(x) for x in ft["fine_nodes"]]
                )
                out["roundoff_node_part"] = float(mv_dn - mv)
                out["roundoff_rest_part"] = float(mp.mpf(v) - mv_dn)
    out.pop("fine_nodes", None)
    with mp.workdps(MP_DPS):
        out["mp_value"] = mp.nstr(mv, 30)
        out["roundoff"] = float(mp.mpf(v) - mv)
        if exact:
            ex = closed_leaf(case, span[0], span[1])
            out["exact"] = mp.nstr(ex, 30)
            out["truncation"] = float(mv - ex)
            out["actual_error"] = float(mp.mpf(v) - ex)
    out["floor_sum_terms"] = float(sum(ft["terms"].values()))
    return out
