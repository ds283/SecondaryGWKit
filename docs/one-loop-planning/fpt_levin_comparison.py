"""Same RD integrals as trap_test.py, evaluated with AdaptiveLevin as QuadSourceIntegral would:
partition at each leg's hand-over (q eta = X_T), factors below their hand-over evaluated as
smooth closed forms, factors above as amplitude x sin/cos of a linear phase, all products
expanded into phase groups, each group one adaptive_levin_sincos call in log(eta)."""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpt_rd_quadrature import Phi, dPhi_dx, integrand, reference, trap, k, cs, eta_c
from AdaptiveLevin import adaptive_levin_sincos

X_T = 15.5
ATOL, RTOL, ORDER = 1e-32, 1e-8, 24


# ---- oscillatory-polynomial algebra: {omega: (S, C)} means sum S(eta) sin(omega eta) + C(eta) cos(omega eta)
def _add(d, w, S, C):
    if w < 0:
        w, S = -w, (lambda e, S=S: -S(e))
    if w in d:
        S0, C0 = d[w]
        d[w] = ((lambda e, a=S0, b=S: a(e) + b(e)), (lambda e, a=C0, b=C: a(e) + b(e)))
    else:
        d[w] = (S, C)


def add(p, q):
    d = dict(p)
    for w, (S, C) in q.items():
        _add(d, w, S, C)
    return d


def mul(p, q):
    d = {}
    for a, (Sa, Ca) in p.items():
        for b, (Sb, Cb) in q.items():
            h = lambda f, g: (lambda e: f(e) * g(e))
            # sin a sin b = [cos(a-b) - cos(a+b)]/2 ; cos a cos b = [cos(a+b) + cos(a-b)]/2
            # sin a cos b = [sin(a+b) + sin(a-b)]/2 ; cos a sin b = [sin(a+b) - sin(a-b)]/2
            SS, CC, SC, CS = h(Sa, Sb), h(Ca, Cb), h(Sa, Cb), h(Ca, Sb)
            _add(
                d,
                a + b,
                lambda e, f=SC, g=CS: 0.5 * (f(e) + g(e)),
                lambda e, f=CC, g=SS: 0.5 * (f(e) - g(e)),
            )
            _add(
                d,
                a - b,
                lambda e, f=SC, g=CS: 0.5 * (f(e) - g(e)),
                lambda e, f=CC, g=SS: 0.5 * (f(e) + g(e)),
            )
    return d


def scal(c, p):
    return {
        w: ((lambda e, S=S: c * S(e)), (lambda e, C=C: c * C(e)))
        for w, (S, C) in p.items()
    }


zero = lambda e: np.zeros_like(e)


def smooth(F):
    return {0.0: (zero, F)}


def leg(q, oscillatory):
    if not oscillatory:
        return smooth(lambda e: Phi(q * e)), smooth(lambda e: q * dPhi_dx(q * e))
    w = cs * q
    As = lambda e: 9.0 / (cs * q**3 * e**3)
    Ac = lambda e: -9.0 / (q**2 * e**2)
    dAs = lambda e: -27.0 / (cs * q**3 * e**4)
    dAc = lambda e: 18.0 / (q**2 * e**3)
    P = {w: (As, Ac)}
    dP = {w: ((lambda e: dAs(e) - w * Ac(e)), (lambda e: dAc(e) + w * As(e)))}
    return P, dP


def build(u, v, eta_l, osc1, osc2, partner):
    q1, q2 = u * k, v * k
    P1, D1 = leg(q1, osc1)
    P2, D2 = leg(q2, osc2)
    eta1 = smooth(lambda e: e)
    eta2 = smooth(lambda e: e**2)
    f = add(
        add(
            scal(4 / 3, mul(P1, P2)),
            scal(4 / 9, mul(eta1, add(mul(D1, P2), mul(P1, D2)))),
        ),
        scal(4 / 9, mul(eta2, mul(D1, D2))),
    )
    c0, s0 = np.cos(k * eta_l), np.sin(k * eta_l)
    G = (
        {k: ((lambda e: s0 * np.ones_like(e)), (lambda e: c0 * np.ones_like(e)))}
        if partner
        else {k: ((lambda e: c0 * np.ones_like(e)), (lambda e: -s0 * np.ones_like(e)))}
    )
    return mul(eta1, mul(f, G))


def levin_J(u, v, eta_l, partner):
    q1, q2 = u * k, v * k
    cuts = sorted(
        {
            eta_l,
            eta_c,
            min(max(X_T / q1, eta_l), eta_c),
            min(max(X_T / q2, eta_l), eta_c),
        }
    )
    total, abserr, solves, regions = 0.0, 0.0, 0, 0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi <= lo * (1 + 1e-12):
            continue
        mid = np.sqrt(lo * hi)
        groups = build(u, v, eta_l, q1 * mid > X_T, q2 * mid > X_T, partner)
        for w, (S, C) in groups.items():
            fs = lambda x, S=S: np.exp(x) * S(np.exp(x))  # d eta = eta dx
            fc = lambda x, C=C: np.exp(x) * C(np.exp(x))
            th = {
                "theta": (lambda x, w=w: w * np.exp(x)),
                "theta_deriv": (lambda x, w=w: w * np.exp(x)),
            }
            d = adaptive_levin_sincos(
                (np.log(lo), np.log(hi)),
                [fs, fc],
                theta=th,
                atol=ATOL,
                rtol=RTOL,
                chebyshev_order=ORDER,
            )
            total += d["value"]
            abserr += d["abserr"]
            solves += d["num_subregion_solves"]
            regions += d["num_regions"]
    return total, abserr, solves, regions


print(
    f"{'t':>7} {'s':>4} | {'trap n':>7} {'trap ms':>8} {'trap err R':>10} | {'Levin ms':>8} {'solves':>6} {'regions':>7} {'Levin err R':>11} {'Levin err Js':>12} {'Levin err Jc':>12} {'claimed':>9}"
)
for s in (0.0, 0.7):
    for t in (0.1, 0.3, 0.732, 1.0, 1.5, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
        u, v = (t + s + 1) / 2, (t - s + 1) / 2
        eta_l = 1e-2 / (max(u, v, 1.0) * k)
        Rs, Rc = reference(u, v, eta_l)
        Rref = Rs**2 + Rc**2
        t0 = time.perf_counter()
        Js, Jc, n, _ = trap(u, v)
        t_trap = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ls, es, n1, r1 = levin_J(u, v, eta_l, False)
        Lc, ec, n2, r2 = levin_J(u, v, eta_l, True)
        t_lev = time.perf_counter() - t0
        print(
            f"{t:7.3f} {s:4.1f} | {n:7d} {1e3*t_trap:8.1f} {abs(Js**2+Jc**2-Rref)/Rref:10.2e} | {1e3*t_lev:8.0f} {n1+n2:6d} {r1+r2:7d} {abs(Ls**2+Lc**2-Rref)/Rref:11.2e} {abs(Ls-Rs)/abs(Rs):12.2e} {abs(Lc-Rc)/abs(Rc):12.2e} {(es+ec)/np.sqrt(Rref):9.1e}"
        )
