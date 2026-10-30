"""depth20-tiling: shared code for the replay, the counterfactual and the re-solves.

* ``load_capture``: a capture written by x1_capture.py / worker.py (the census route), read back by
  offset and checked against its SHA-256 (the census pickles name ``harness``: this scripts
  directory, holding the census's harness.py, must be on PYTHONPATH).
* ``pinned``: a context manager that swaps the module attribute
  ``AdaptiveLevin.levin_quadrature.chebyshev_matrices`` for a wrapper that calls the original and
  then sets x[0] := b and x[-1] := a (x_span = (a, b)) on a copy of x; D and every other node are
  the original's. Every caller in the module (_adaptive_levin_subregion_impl, the Clenshaw-Curtis
  branch) looks the name up in the module at call time, so the swap reaches them. The tree is not
  edited; the swap lives only in this process.
* ``mp_base``, ``mp_cc_weights_base``: the exact Chebyshev differentiation matrix and
  Clenshaw-Curtis weights on [-1, 1], copied from var/levin-phase-floor-floor-breakdown/common.py
  (a record, read only) unchanged in substance.
* ``solve_double``: the module's own _adaptive_levin_subregion on a span, computed or pinned ends.
* ``solve_mp``: the same scheme with the linear algebra exact (mpmath, MP_DPS digits), on the
  double data the code samples: f and theta' at the grid's double nodes, the endpoint phase
  theta_mod_2pi at the grid's two double end nodes -- each evaluated in double by the call's own
  callables -- promoted exactly to mp; the differentiation matrix exact (mp_base / half-width, the
  half-width (b - a)/2 exact in mp), as the floor-breakdown scheme's mp_levin; sin and cos in mp of
  the double phase. The rtol gate on p-modes is the double solve's (p_use from its p_ratios). The
  Clenshaw-Curtis branch: exact weights times (b - a)/2 in mp, the double samples of f and of
  theta_mod_2pi at the double fine-grid nodes.
"""

import contextlib
import hashlib
import pickle
from pathlib import Path

import mpmath as mp
import numpy as np

import AdaptiveLevin.levin_quadrature as LQ

MP_DPS = 50
_ORIG_CHEB = LQ.chebyshev_matrices


def load_capture(entry):
    with open(Path(entry["dir"]) / entry["pickle"], "rb") as f:
        f.seek(entry["offset"])
        blob = f.read(entry["length"])
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise RuntimeError("capture blob sha256 differs")
    rec = pickle.loads(blob)
    if rec["unit"] != entry["unit"] or rec["index"] != entry["index"]:
        raise RuntimeError(f"record is {rec['unit']} {rec['index']}")
    return rec


def pinned_chebyshev_matrices(x_span, N):
    x, D = _ORIG_CHEB(x_span, N)
    x = np.array(x, copy=True)
    a, b = x_span
    x[0] = b
    x[-1] = a
    return x, D


@contextlib.contextmanager
def pinned():
    LQ.chebyshev_matrices = pinned_chebyshev_matrices
    try:
        yield
    finally:
        LQ.chebyshev_matrices = _ORIG_CHEB


def grid_of(span, N, pin):
    return (pinned_chebyshev_matrices if pin else _ORIG_CHEB)(span, N)


def patch_selfcheck(spans, orders=(24, 47)):
    """The patch's output against the original's, element by element, on each span and order."""
    rows = []
    for s in spans:
        for N in orders:
            x0, D0 = _ORIG_CHEB(s, N)
            x1, D1 = pinned_chebyshev_matrices(s, N)
            rows.append({
                "span": [float(s[0]).hex(), float(s[1]).hex()], "N": N,
                "interior_nodes_bit_identical": bool(np.array_equal(x0[1:-1].view(np.uint64), x1[1:-1].view(np.uint64))),
                "D_bit_identical": bool(np.array_equal(D0.view(np.uint64), D1.view(np.uint64))),
                "x0_is_b": bool(x1[0] == s[1]), "xN_is_a": bool(x1[-1] == s[0]),
                "original_x0_was_b": bool(x0[0] == s[1]), "original_xN_was_a": bool(x0[-1] == s[0]),
                "original_untouched": bool(np.array_equal(x0.view(np.uint64), _ORIG_CHEB(s, N)[0].view(np.uint64))),
            })
    return rows


# ---------------------------------------------------------------- exact base matrices (floor-breakdown)
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


# ---------------------------------------------------------------- solves
def solve_double(span, f, basis, order, rtol, pin, vcache=None):
    ctx = pinned() if pin else contextlib.nullcontext()
    with ctx:
        d = LQ._adaptive_levin_subregion(
            span, f, basis, id_label=LQ._LazyUUID(), chebyshev_order=order, rtol=rtol,
            notify_label=None, build_p_sample=False, vectorize_cache=vcache if vcache is not None else {},
        )
    return d


def g_double(f, basis, x):
    """The integrand's two products at x, in double data: f_sin(x), f_cos(x) and theta_mod_2pi(x)
    (each a double from the call's callables); returned as mp: f_sin sin(theta) + f_cos cos(theta)."""
    fs = float(np.asarray(f[0](x)))
    fc = float(np.asarray(f[1](x)))
    th = float(basis._theta_mod_2pi(x)) if hasattr(basis, "_theta_mod_2pi") else float(basis._theta(x))
    with mp.workdps(MP_DPS):
        return +(mp.mpf(fs) * mp.sin(mp.mpf(th)) + mp.mpf(fc) * mp.cos(mp.mpf(th)))


def solve_mp(span, f, basis, order, rtol, pin, d_double):
    """The exact-arithmetic re-solve of the solve d_double (its branch and p-mode gate), at the
    computed (pin False) or pinned (pin True) end nodes. Returns (value as mpf, end nodes)."""
    a, b = span
    vcache = {}
    if d_double.get("is_direct", False):
        fine = 2 * order - 1
        grid, _ = grid_of(span, fine, pin)
        F = np.vstack([LQ._sample_vectorized(fn, grid, vcache, ("f", id(fn))) for fn in f])
        args = [float(basis._theta_mod_2pi(x)) if hasattr(basis, "_theta_mod_2pi") else float(basis._theta(x)) for x in grid]
        with mp.workdps(MP_DPS):
            w = mp_cc_weights_base(fine)
            half = (mp.mpf(b) - mp.mpf(a)) / 2
            s = mp.mpf(0)
            for k in range(fine):
                s += w[k] * (mp.mpf(float(F[0, k])) * mp.sin(mp.mpf(args[k])) + mp.mpf(float(F[1, k])) * mp.cos(mp.mpf(args[k])))
            return +(s * abs(half)), (float(grid[-1]), float(grid[0]))
    grid, Dmat = grid_of(span, order, pin)
    _AmatT, thp, _w0, _wk, _ps = basis.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache=vcache)
    F = np.vstack([LQ._sample_vectorized(fn, grid, vcache, ("f", id(fn))) for fn in f])
    if hasattr(basis, "_theta_mod_2pi"):
        th_b, th_a = float(basis._theta_mod_2pi(grid[0])), float(basis._theta_mod_2pi(grid[-1]))
    else:
        th_b, th_a = float(basis._theta(grid[0])), float(basis._theta(grid[-1]))
    p_use = [bool(np.isfinite(r) and r > rtol) for r in d_double["p_ratios"]]
    N = order
    with mp.workdps(MP_DPS):
        _t, Db = mp_base(N)
        half = (mp.mpf(b) - mp.mpf(a)) / 2
        L = mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                L[i, j] = Db[i, j] / half
            L[i, i] += mp.mpc(0, 1) * mp.mpf(float(thp[i]))
        rhs = mp.matrix([mp.mpf(float(F[0, i])) + mp.mpc(0, 1) * mp.mpf(float(F[1, i])) for i in range(N)])
        q = mp.lu_solve(L, rhs)
        P = [[mp.re(q[i]) for i in range(N)], [mp.im(q[i]) for i in range(N)]]
        wb = [mp.sin(mp.mpf(th_b)), mp.cos(mp.mpf(th_b))]
        wa = [mp.sin(mp.mpf(th_a)), mp.cos(mp.mpf(th_a))]
        upper = sum((P[i][0] * wb[i] for i in range(2) if p_use[i]), mp.mpf(0))
        lower = sum((P[i][-1] * wa[i] for i in range(2) if p_use[i]), mp.mpf(0))
        return +(upper - lower), (float(grid[-1]), float(grid[0]))


def offset_term(span, f, basis, order, d_double):
    """The end-node offset's predicted effect on a solve's value: V(computed ends) - V(pinned ends)
    ~ g(b) (x^_0 - b) - g(a) (x^_{N-1} - a) for a Levin solve (its value integrates the interval
    between the computed end nodes); for a Clenshaw-Curtis solve, whose weights stay put while a node
    moves, w_0 (g(x^_0) - g(b)) + w_{N-1} (g(x^_{N-1}) - g(a)), exactly what moving the end samples
    does to the rule. g = f_sin sin(theta_mod) + f_cos cos(theta_mod) in double data, as mp."""
    a, b = span
    if d_double.get("is_direct", False):
        fine = 2 * order - 1
        grid, _ = _ORIG_CHEB(span, fine)
        lo, hi = float(grid[-1]), float(grid[0])
        with mp.workdps(MP_DPS):
            w = mp_cc_weights_base(fine)
            half = abs((mp.mpf(b) - mp.mpf(a)) / 2)
            t = mp.mpf(0)
            if hi != b:
                t += w[0] * half * (g_double(f, basis, hi) - g_double(f, basis, b))
            if lo != a:
                t += w[-1] * half * (g_double(f, basis, lo) - g_double(f, basis, a))
            return +t
    grid, _ = _ORIG_CHEB(span, order)
    lo, hi = float(grid[-1]), float(grid[0])
    with mp.workdps(MP_DPS):
        t = mp.mpf(0)
        if hi != b:
            t += g_double(f, basis, b) * (mp.mpf(hi) - mp.mpf(b))
        if lo != a:
            t -= g_double(f, basis, a) * (mp.mpf(lo) - mp.mpf(a))
        return +t
