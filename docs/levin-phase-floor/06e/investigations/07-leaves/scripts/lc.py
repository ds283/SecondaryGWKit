"""Shared code for the 07-leaves measurement: where the declared error of 07's driver sits, leaf by
leaf, on prompt 07's (R) stuck case and its relatives.

Run from the throwaway worktree ../SecondaryGWKit-07leaves (e5764a6 + 07's staged patch) with
PYTHONPATH=. so that ``AdaptiveLevin`` is 07's code. Nothing here changes the driver: the driver is
run as it is, and two module-level names are wrapped *only to record*, never to alter a value:

* ``lq._levin_leaf``: a subclass whose __init__ stores the leaf's value on its used_interval as
  ``region._rec_value`` (used_interval does not carry the value);
* ``lq._adaptive_levin_subregion``: records each non-window solve (span -> value, floor, truncation,
  p_ratios, is_direct, nested-pair estimate) and the driver's own vectorize_cache;
* ``lq._adaptive_levin_window_piece``: records the driver's vectorize_cache only.

The problem, the amplitudes and the phase are copied from 07's stop4_diagnose.py; the closed form
of a leaf and the mpmath re-solves of the scheme are copied from
var/levin-phase-floor-floor-breakdown/common.py (closed_leaf, mp_base, mp_cc_weights_base, mp_levin,
mp_cc), unchanged in substance.
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

RECORDS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07-leaves")
SCRIPTS = RECORDS / "scripts"
OUT = RECORDS / "out"
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
PATCH = RECORDS / "07-staged.patch"
BASE_SHA = "e5764a6708757cb11704b95bcd66825baab4ba29"

MP_DPS = 50
ORDER = 16
RTOL = 1.0e-10
X_S = 0.7236067977499790

CASES = {
    "c1e6w0": {"c": 1.0e6, "x_s": X_S, "omega": 0.0},
    "c1e8w0": {"c": 1.0e8, "x_s": X_S, "omega": 0.0},
    "c1e8w1": {"c": 1.0e8, "x_s": X_S, "omega": 1.0},
}


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
    return {
        "script": str(Path(script).resolve()),
        "script_sha256": sha256(script),
        "scripts_sha256": {p.name: sha256(p) for p in sorted(SCRIPTS.glob("*.py"))},
        "base_sha": BASE_SHA,
        "worktree_head": git("rev-parse", "HEAD").strip(),
        "worktree_status": git("status", "--short").splitlines(),
        "patch_sha256": sha256(PATCH),
        "levin_quadrature": str(Path(lq.__file__).resolve()),
        "levin_quadrature_sha256": sha256(lq.__file__),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mp.__version__,
        "mp_dps": MP_DPS,
    }


# ---------------------------------------------------------------- the problem, as stop4_diagnose.py


def amplitudes():
    return [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]


def phase(c, x_s, omega):
    return {
        "theta": lambda x: omega * x + 0.5 * c * (x - x_s) ** 2,
        "theta_deriv": lambda x: omega + c * (x - x_s),
    }


def x_root(case):
    with mp.workdps(40):
        return float(mp.mpf(case["x_s"]) - mp.mpf(case["omega"]) / mp.mpf(case["c"]))


def theta_rel(case, x):
    """theta(x) - theta(x_r), exactly (mp), as a float: (c/2)(x - x_r)^2."""
    with mp.workdps(MP_DPS):
        c = mp.mpf(case["c"])
        xr = mp.mpf(case["x_s"]) - mp.mpf(case["omega"]) / c
        return float(c / 2 * (mp.mpf(x) - xr) ** 2)


def theta_prime(case, x):
    with mp.workdps(MP_DPS):
        return float(
            mp.mpf(case["omega"]) + mp.mpf(case["c"]) * (mp.mpf(x) - mp.mpf(case["x_s"]))
        )


def theta_mod_2pi_exact(case):
    """A range-reduced phase computed exactly from the double x it is handed: (theta(x) mod 2 pi),
    theta evaluated in mp at the double x, rounded once."""
    c = case["c"]
    x_s = case["x_s"]
    omega = case["omega"]

    def tm(x):
        with mp.workdps(40):
            X = mp.mpf(float(x))
            t = mp.mpf(omega) * X + mp.mpf(c) / 2 * (X - mp.mpf(x_s)) ** 2
            return float(mp.fmod(t, 2 * mp.pi))

    return tm


# ---------------------------------------------------------------- recording hooks

REC = {"solves": {}, "cache": None, "record_solves": True}

_orig_leaf = lq._levin_leaf
_orig_sub = lq._adaptive_levin_subregion
_orig_wp = lq._adaptive_levin_window_piece


class _RecLeaf(_orig_leaf):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.region._rec_value = self.value
        self.region._rec_kind = self.kind


def _rec_sub(x_span, f, BasisData, *args, **kwargs):
    data = _orig_sub(x_span, f, BasisData, *args, **kwargs)
    REC["cache"] = kwargs.get("vectorize_cache")
    if REC["record_solves"]:
        REC["solves"][(float(x_span[0]), float(x_span[1]))] = {
            "value": data["value"],
            "phase_err": data.get("phase_err"),
            "abserr_truncation": data.get("abserr_truncation"),
            "p_ratios": data.get("p_ratios"),
            "is_direct": bool(data.get("is_direct", False)),
            "abserr_direct": data.get("abserr_direct"),
            "phase_span": data.get("phase_span"),
            "order": data["metadata"].get("chebyshev_order"),
        }
    return data


def _rec_wp(x_span, f, BasisData, chebyshev_order, notify_label, id_label, vectorize_cache):
    REC["cache"] = vectorize_cache
    return _orig_wp(
        x_span, f, BasisData, chebyshev_order, notify_label, id_label, vectorize_cache
    )


def install_hooks():
    lq._levin_leaf = _RecLeaf
    lq._adaptive_levin_subregion = _rec_sub
    lq._adaptive_levin_window_piece = _rec_wp


def reset_rec(record_solves=True):
    REC["solves"] = {}
    REC["cache"] = None
    REC["record_solves"] = record_solves


def run_driver(case, phi_w_over_pi, depth_max, theta_mod=False, atol=None):
    """07's driver on the case, as stop4_diagnose.py runs it, with Phi_w set; returns (res, basis)."""
    install_hooks()
    reset_rec()
    ph = phase(case["c"], case["x_s"], case["omega"])
    if theta_mod:
        ph["theta_mod_2pi"] = theta_mod_2pi_exact(case)
    basis = lq._Basis_SinCos(ph)
    saved = lq._LEVIN_WINDOW_HALF_PHASE
    lq._LEVIN_WINDOW_HALF_PHASE = phi_w_over_pi * math.pi
    try:
        res = lq._adaptive_levin(
            (0.0, 1.0),
            amplitudes(),
            basis,
            atol=atol,
            rtol=RTOL,
            depth_max=depth_max,
        )
    finally:
        lq._LEVIN_WINDOW_HALF_PHASE = saved
    return res, basis


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

        return +(mp.im(J(mp.mpf(-1))) + mp.re(J(mp.mpc(0, 3)) + J(mp.mpc(0, -3))) / 2)


# ---------------------------------------------------------------- the scheme in mpmath

_BASE = {}


def mp_base(N):
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
    """The complexified Levin collocation solve, exactly; nodes default to the exact
    chebyshev_matrices nodes; end_args, when given, are the (double) arguments of the basis at
    (b, a), taken exactly as mpf."""
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
        if end_args is None:
            ab, aa = th(x[0]), th(x[-1])
        else:
            ab, aa = mp.mpf(end_args[0]), mp.mpf(end_args[1])
        wb = [mp.sin(ab), mp.cos(ab)]
        wa = [mp.sin(aa), mp.cos(aa)]
        upper = sum(P[i][0] * wb[i] for i in range(2) if p_use[i])
        lower = sum(P[i][-1] * wa[i] for i in range(2) if p_use[i])
        return +(upper - lower)


def mp_cc(case, xa, xb, N=ORDER, nodes=None, coarse=False):
    """CC_{2N-1} (or, coarse=True, CC_N on every other node) on chebyshev_matrices((xa, xb),
    2N-1)'s nodes, exactly; with nodes given (e.g. the double nodes as mpf), there instead."""
    fine = 2 * N - 1
    with mp.workdps(MP_DPS):
        th, thp, f1, f2 = _mp_funcs(case)
        t = [mp.cos(mp.pi * i / (fine - 1)) for i in range(fine)]
        a, b = mp.mpf(xa), mp.mpf(xb)
        mid, half = (a + b) / 2, (b - a) / 2
        x = [mid + half * ti for ti in t] if nodes is None else nodes
        if coarse:
            w = mp_cc_weights_base(N)
            x = x[::2]
        else:
            w = mp_cc_weights_base(fine)
        s = mp.mpf(0)
        for wk, xk in zip(w, x):
            tk = th(xk)
            s += wk * (f1(xk) * mp.sin(tk) + f2(xk) * mp.cos(tk))
        return +(s * half)


# ---------------------------------------------------------------- per-leaf attribution


def component(region):
    """Which component sets the leaf's declared error total_err = max(resolution, floor) + trunc,
    with resolution = declared_resolution if the trust check set it, else abserr."""
    floor = region.phase_err if region.phase_err is not None else 0.0
    trunc = region.abserr_truncation or 0.0
    if region.declared_resolution is not None:
        resolution = region.declared_resolution
        res_name = "trust_widening" if region.trusted else "cap_untrusted_d1+d2"
    else:
        resolution = region.abserr
        res_name = "residual" if region.type == "Levin" else "nested_pair"
    if region.phase_limited:
        base_name = "floor(phase_limited)"
    else:
        base_name = res_name if resolution >= floor else "floor"
    base = max(resolution, floor)
    if trunc > base:
        return "truncation", base, trunc
    return base_name, base, trunc


def outside_leaf_record(case, region, exact=True):
    a, b = region.start, region.end
    xr = x_root(case)
    near = b if b <= xr else a
    rec = {
        "a": a,
        "b": b,
        "side": "left" if b <= xr else "right",
        "depth": region.depth,
        "type": region.type,
        "value": region._rec_value,
        "abserr": region.abserr,
        "floor": region.phase_err,
        "truncation": region.abserr_truncation,
        "delta1": region.delta1,
        "delta2": region.delta2,
        "r": region.convergence_ratio,
        "trusted": region.trusted,
        "declared_resolution": region.declared_resolution,
        "phase_limited": region.phase_limited,
        "total_err": region.total_err,
        "dtheta_near": theta_rel(case, near),
        "theta_prime_near": theta_prime(case, near),
        "dtheta_far": theta_rel(case, a if near == b else b),
    }
    comp, base, trunc = component(region)
    rec["component"] = comp
    if exact:
        with mp.workdps(MP_DPS):
            ex = closed_leaf(case, a, b)
            rec["true_error"] = float(mp.mpf(region._rec_value) - ex)
    return rec
