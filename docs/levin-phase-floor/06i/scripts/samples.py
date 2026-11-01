"""levin-phase-floor prompt 06i: loading Z3's captured calls, and each route's integrand. New in 06i.

Run with the census tree (6d81fad) as the working directory and
PYTHONPATH=.:<main>/var/levin-phase-floor-06e/census/scripts:<this directory>, as d3_probe.py loads
the pickles (ray.cloudpickle.loads).

The captured call is  I = int_{u_a}^{u_b} [f_sin(u) sin theta(u) + f_cos(u) cos theta(u)] du
                        = Re int F(u) exp(i theta(u)) du,   F = f_cos - i f_sin.

Route A (analytic BesselPhaseGroup; u = log eta, theta = K eta + C + R(eta)):
    I = Re int_{e^{u_a}}^{e^{u_b}} g(eta) exp(i K eta) d eta,  g = F(log eta) exp(i (C + R(eta))) / eta,
with K = sum_i e_i m_i exactly (a Fraction of the float coefficients), C = sum_i e_i (pi/4 - pi nu_i/2)
exactly in mpmath, R the group's own residual (evaluated in double at the double eta), and the
limits exact.

Route B (cosmological groups, x = log(1+z)): see route_b.py.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
INPUTS = MAIN / "var/levin-phase-floor-06i/inputs"
Z3 = MAIN / "docs/levin-phase-floor/06e/census/references/out/z3_scores.json"

ROUTE_A = (8, 9, 10, 22, 23, 25, 27)
ROUTE_B = (1, 2, 3, 13, 14, 15)


def pickle_path(n):
    return INPUTS / f"s{n:02d}.pkl"


def load(n):
    import ray.cloudpickle as cp

    return cp.loads(pickle_path(n).read_bytes())


def z3_row(n):
    for r in json.loads(Z3.read_text())["rows"]:
        if r["n"] == n:
            return r
    raise KeyError(n)


# ------------------------------------------------------------------------------------------
# Route A


class RouteA:
    def __init__(self, rec):
        self.rec = rec
        th = rec["theta"]
        self.group = th["theta"].__self__
        g = self.group
        self.f_sin, self.f_cos = rec["f"]
        self.u_a, self.u_b = (float(v) for v in rec["x_span"])
        # K exactly, from the float coefficients and signs (each term exact)
        self.K_exact = sum(
            (Fraction(s) * Fraction(c) for s, c in zip(g.signs, g.coefficients)),
            Fraction(0),
        )
        self.K_lo = self.K_exact - Fraction(g.K)
        self.K_lo_fsum = math.fsum(
            [s * c for s, c in zip(g.signs, g.coefficients)] + [-g.K]
        )
        self.nus = [p.nu for p in g.phases]

    def K_mp(self):
        return mp.mpf(self.K_exact.numerator) / self.K_exact.denominator

    def C_mp(self):
        g = self.group
        return mp.fsum(
            s * (mp.pi / 4 - mp.pi * mp.mpf(p.nu) / 2)
            for s, p in zip(g.signs, g.phases)
        )

    def F(self, u):
        return complex(self.f_cos(u), -self.f_sin(u))

    def g_double(self, C_d):
        grp = self.group
        F = self.F

        def gd(eta):
            return (
                F(math.log(eta))
                * complex(
                    math.cos(C_d + grp.residual(eta)), math.sin(C_d + grp.residual(eta))
                )
                / eta
            )

        return gd

    def g_mp(self):
        """g at an mp eta: amplitude and residual evaluated in double at fl(eta), the rest in mp."""
        grp = self.group
        C = self.C_mp()
        F = self.F

        def gm(eta):
            ed = float(eta)
            ud = float(mp.log(eta))
            Fv = F(ud)
            return mp.mpc(Fv.real, Fv.imag) * mp.expj(C + grp.residual(ed)) / eta

        return gm

    def u_breaks(self):
        """u at which a constituent's representation switches: near-region panel edges and x_star."""
        out = set()
        for p, m in zip(self.group.phases, self.group.coefficients):
            c = p._corrections
            lm = math.log(m)
            cand = [math.log(c.x_star) - lm]
            if c.near is not None:
                cand += [float(b) - lm for b in c.near.r_interp.breakpoints]
                cand += [float(b) - lm for b in c.near.log_a_interp.breakpoints]
            for v in cand:
                if self.u_a < v < self.u_b:
                    out.add(v)
        return sorted(out)

    def u_segments(self, du_max):
        """u edges: the span, the breaks, and uniform subdivision to du_max."""
        pts = [self.u_a] + self.u_breaks() + [self.u_b]
        edges = [mp.mpf(self.u_a)]
        for l, r in zip(pts[:-1], pts[1:]):
            n = max(1, math.ceil((r - l) / du_max))
            L, R = mp.mpf(l), mp.mpf(r)
            for j in range(1, n + 1):
                edges.append(L + (R - L) * j / n)
        return edges


# ------------------------------------------------------------------------------------------
# Route B helpers


def rem2pi(v):
    """v reduced into (-pi, pi], in mpmath."""
    two_pi = 2 * mp.pi
    r = v - two_pi * mp.floor(v / two_pi + mp.mpf(1) / 2)
    if r <= -mp.pi:
        r += two_pi
    if r > mp.pi:
        r -= two_pi
    return r


def route_b_closure(rec):
    """(terms_at's closure dict, active constituents) of a cosmological group call."""
    ta = rec["f"][0].__closure__[2].cell_contents
    cl = dict(
        zip(
            ta.__wrapped__.__code__.co_freevars,
            [c.cell_contents for c in ta.__wrapped__.__closure__],
        )
    )
    active = rec["theta"]["theta"].__closure__[0].cell_contents
    return cl, active


def route_b_edges(rec):
    """
    x edges inside the span at which some ingredient of the integrand changes representation:
    the phi spline knots of each constituent, the Green's function amplitude spline's knots, and
    for every CumulativeTable reached (each constituent's leading primitive, each T factor's
    friction table) its nodes, the midpoints where its nearest-node choice switches, and its
    declared breaks.
    """
    a, b = (float(v) for v in rec["x_span"])
    cl, active = route_b_closure(rec)
    xs = set()
    tables = []
    for _, fac in active:
        ph = fac.phase
        xs.update(float(t) for t in ph._spline.t)
        tables.append(ph._leading._table)
    for name in ("q_factor", "r_factor"):
        f = cl[name]
        if hasattr(f, "_f") and hasattr(f._f, "_friction_F"):
            tables.append(f._f._friction_F._table)
    G = cl["G_factor"]
    if hasattr(G, "_Gk"):
        xs.update(float(t) for t in G._Gk.sin_amplitude._spline.t)
    seen = set()
    for t in tables:
        if id(t) in seen:
            continue
        seen.add(id(t))
        asc = np.asarray(t._u_asc, dtype=float)
        xs.update(asc.tolist())
        xs.update((asc[:-1] + (asc[1:] - asc[:-1]) / 2).tolist())
        if t._breaks is not None:
            xs.update(float(v) for v in np.asarray(t._breaks).ravel())
    inner = sorted(v for v in xs if a < v < b)
    edges = [a]
    for v in inner:
        if v - edges[-1] > 1e-9:
            edges.append(v)
    if b - edges[-1] <= 1e-9:
        edges.pop()
    edges.append(b)
    return edges, len(seen)


def cheb_eval(c, y):
    """sum_k c_k T_k(y) (plain sum, not sum''), Clenshaw; works for float or mpf."""
    b1 = b2 = 0 * y
    for ck in reversed(c[1:]):
        b1, b2 = 2 * y * b1 - b2 + ck, b1
    return y * b1 - b2 + c[0]


def cheb_der(c):
    """coefficients of d/dy sum_k c_k T_k."""
    n = len(c) - 1
    if n == 0:
        return [0 * c[0]]
    d = [0 * c[0]] * (n + 1)
    for k in range(n, 0, -1):
        d[k - 1] = (d[k + 1] if k + 1 <= n else 0 * c[0]) + 2 * k * c[k]
    d[0] = d[0] / 2
    return d[:n]


class Segment:
    """
    One x segment [X_l, X_r] of a Route B call, with the map t = p(x): a polynomial of degree M in
    y = (x - mid)/half, fitted by least squares to the unwrapped captured phase at K points (doubles
    x_k distributed as Chebyshev-Lobatto points):

        theta~_k = raw_k + rem(theta_mod_2pi_k - raw_k)      (exact, in mpmath)

    raw theta only chooses the 2 pi branch; theta_mod_2pi supplies the value. theta_mod_2pi carries
    a white jitter (about 0.6 eps |theta'|, the rounding of z = expm1(x): primitive_phase.py's
    "04's I1"), so p is the least-squares smooth phase, and its residual rms is the jitter's measure.
    p = b(y) + q(y): b is the mp line through theta~ at the two ends, q is in double.

    With `ends` = (E_l, E_r) given (mp), p is refitted with p(-1) = E_l and p(+1) = E_r exactly, so
    that adjacent segments share their edge value and t(x) is continuous.
    """

    def __init__(self, rec, X_l, X_r, M=20, K=241, ends=None, samples=None):
        th = rec["theta"]
        self.X_l, self.X_r, self.M, self.K = X_l, X_r, M, K
        self.mid = (mp.mpf(X_l) + mp.mpf(X_r)) / 2
        self.half = (mp.mpf(X_r) - mp.mpf(X_l)) / 2
        if samples is None:
            xs = [
                float(self.mid + self.half * mp.cos(mp.pi * i / (K - 1)))
                for i in range(K)
            ]
            xs[0], xs[-1] = X_r, X_l
            vals = []
            branch = 0.0
            for x in xs:
                raw = th["theta"](x)
                mod = th["theta_mod_2pi"](x)
                d = rem2pi(mp.mpf(mod) - mp.mpf(raw))
                branch = max(branch, float(abs(d)))
                vals.append(mp.mpf(raw) + d)
            samples = (xs, vals, branch)
        self.samples = samples
        xs, vals, self.branch_margin = samples
        ys = np.array([float((mp.mpf(x) - self.mid) / self.half) for x in xs])
        # baseline: the mp line through the end samples (index 0 is X_r, y = +1; last is X_l)
        self.b0 = (vals[0] + vals[-1]) / 2
        self.b1 = (vals[0] - vals[-1]) / 2
        resid = np.array(
            [float(v - (self.b0 + self.b1 * mp.mpf(y))) for v, y in zip(vals, ys)]
        )
        V = np.polynomial.chebyshev.chebvander(ys, M)
        if ends is None:
            q, *_ = np.linalg.lstsq(V, resid, rcond=None)
        else:
            E_l, E_r = ends
            qr = float(E_r - (self.b0 + self.b1))
            ql = float(E_l - (self.b0 - self.b1))
            A0, B0 = (qr + ql) / 2, (qr - ql) / 2
            cols = []
            for k in range(2, M + 1):
                cols.append(V[:, k] - (V[:, 0] if k % 2 == 0 else V[:, 1]))
            Vc = np.stack(cols, axis=1)
            rc, *_ = np.linalg.lstsq(Vc, resid - (A0 + B0 * ys), rcond=None)
            q = np.zeros(M + 1)
            q[2:] = rc
            q[0] = A0 - sum(rc[k - 2] for k in range(2, M + 1) if k % 2 == 0)
            q[1] = B0 - sum(rc[k - 2] for k in range(3, M + 1) if k % 2 == 1)
        fit = V @ q
        r = resid - fit
        self.sigma = float(np.sqrt(np.sum(r * r) / max(1, K - M - 1)))
        self.max_resid = float(np.max(np.abs(r)))
        self.design = V
        self.q = [float(v) for v in q]
        # p's Chebyshev coefficients in mp: baseline + q
        self.c = [mp.mpf(v) for v in self.q]
        self.c[0] += self.b0
        self.c[1] += self.b1
        self.dc = cheb_der(self.c)
        self.ddc = cheb_der(self.dc)
        self.T_l = cheb_eval(self.c, mp.mpf(-1))
        self.T_r = cheb_eval(self.c, mp.mpf(1))
        self.c_d = [float(v) for v in self.c]
        self.dc_d = [float(v) for v in self.dc]
        self.ddc_d = [float(v) for v in self.ddc]
        self.mid_d = float(self.mid)
        self.half_d = float(self.half)
        # monotonicity: p' at 161 points has one sign, and it is theta_deriv's sign
        sgn = set()
        dmin = math.inf
        for i in range(161):
            y = -1 + 2 * i / 160
            dp = cheb_eval(self.dc_d, y)
            sgn.add(dp > 0)
            dmin = min(dmin, abs(dp) / self.half_d)
        td = [th["theta_deriv"](x) for x in (X_l, float(self.mid), X_r)]
        sgn.update(v > 0 for v in td)
        self.monotone = len(sgn) == 1 and all(v != 0 for v in td)
        self.min_abs_pprime = dmin

    def value_at_end(self, which):
        return self.T_l if which < 0 else self.T_r

    def q_shift(self, shift):
        """p - shift as double Chebyshev coefficients (shift an mp value near p's range)."""
        c = list(self.c_d)
        c[0] = float(self.c[0] - shift)
        return c

    def y_of(self, x_mp):
        return (x_mp - self.mid) / self.half

    def p(self, y):
        return cheb_eval(self.c, y)

    def pprime_x(self, y):
        """dp/dx at y (mp)."""
        return cheb_eval(self.dc, y) / self.half

    def solve_y(self, t, shift=None):
        """
        y in [-1, 1] with p(y) = t: Newton in double on p(y) - t with the constant shifted
        (shift = an mp value near t, default t), from a linear guess, then one Newton step in
        mpmath. Returns y as mpf.
        """
        if shift is None:
            shift = t
        c = list(self.c_d)
        c[0] = float(self.c[0] - shift)
        target = float(t - shift)
        span = float(self.T_r - self.T_l)
        y = -1 + 2 * float(t - self.T_l) / span if span != 0 else 0.0
        y = min(1.0, max(-1.0, y))
        for _ in range(30):
            f = cheb_eval(c, y) - target
            d = cheb_eval(self.dc_d, y)
            step = f / d
            yn = min(1.0, max(-1.0, y - step))
            if abs(yn - y) <= 1e-16 * max(1.0, abs(y)):
                y = yn
                break
            y = yn
        ym = mp.mpf(y)
        ym = ym - (self.p(ym) - t) / cheb_eval(self.dc, ym)
        return ym
