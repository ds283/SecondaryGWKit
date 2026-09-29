"""Score Franciolini-Pritchard-Tada's trapezoid time quadrature in exact RD.

Their rule (core/kernel.py + export_kernels.py): eta grid = geomspace(eta_l, eta_c, n),
eta_l = 1e-2/(max(u,v,1) k), eta_c = 400/k, n = clip(45 * max(u,v) * k*eta_c/(2pi), 800, 2e5),
J_i = trapezoid(g_i * a * f, eta).  Reference: composite Gauss-Legendre with 8 panels per period
of the fastest leg and 16 nodes per panel.  Scored quantity: R = Js^2 + Jc^2 (basis-invariant,
this is what enters the oscillation-averaged kernel in RD).
"""

import numpy as np

k = 1.0
cs = 1 / np.sqrt(3.0)
eta_c = 400.0 / k


def Phi(x):  # RD transfer function, Phi -> 1
    y = x * cs
    small = y < 1e-3
    out = np.empty_like(x)
    ys = y[~small]
    out[~small] = 9.0 * (np.sin(ys) / ys - np.cos(ys)) / x[~small] ** 2
    out[small] = 1.0 - y[small] ** 2 / 10.0
    return out


def dPhi_dx(x):
    y = x * cs
    small = y < 1e-3
    out = np.empty_like(x)
    ys = y[~small]
    xs = x[~small]
    # d/dx [9 (sin y/y - cos y)/x^2],  y = cs x
    g = np.sin(ys) / ys - np.cos(ys)
    dg = cs * (np.cos(ys) / ys - np.sin(ys) / ys**2 + np.sin(ys))
    out[~small] = 9.0 * (dg / xs**2 - 2.0 * g / xs**3)
    out[small] = -(cs**2) * x[small] / 5.0
    return out


def integrand(eta, u, v, eta_l):
    q1, q2 = u * k, v * k
    P1, P2 = Phi(q1 * eta), Phi(q2 * eta)
    D1, D2 = q1 * dPhi_dx(q1 * eta), q2 * dPhi_dx(q2 * eta)  # dPhi/deta
    f = (
        (4 / 3) * P1 * P2
        + (4 / 9) * eta * (D1 * P2 + P1 * D2)
        + (4 / 9) * eta**2 * D1 * D2
    )  # RD: 1/H = eta
    ph = k * (eta - eta_l)
    return eta * f * np.sin(ph), eta * f * np.cos(ph)  # a ~ eta


def their_grid(u, v):
    umax = max(u, v, 1.0)
    eta_l = 1e-2 / (umax * k)
    n = int(np.clip(45.0 * max(u, v) * k * eta_c / (2 * np.pi), 800, 200000))
    return np.geomspace(eta_l, eta_c, n), eta_l, n


def trap(u, v):
    eta, eta_l, n = their_grid(u, v)
    s, c = integrand(eta, u, v, eta_l)
    return np.trapezoid(s, eta), np.trapezoid(c, eta), n, eta_l


def reference(u, v, eta_l, panels_per_period=8, nodes=16):
    period = 2 * np.pi / (cs * max(u, v, 1.0) * k)
    h = min(period, 2 * np.pi / k) / panels_per_period
    edges = np.arange(eta_l, eta_c, h)
    edges = np.append(edges, eta_c)
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    a, b = edges[:-1], edges[1:]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    tot_s = tot_c = 0.0
    for j in range(0, len(mid), 200000):
        m, hh = mid[j : j + 200000], half[j : j + 200000]
        x = (m[:, None] + hh[:, None] * xg[None, :]).ravel()
        s, c = integrand(x, u, v, eta_l)
        w = (hh[:, None] * wg[None, :]).ravel()
        tot_s += np.dot(s, w)
        tot_c += np.dot(c, w)
    return tot_s, tot_c


if __name__ == "__main__":
    print(
        f"{'t':>7} {'s':>4} {'u':>8} {'n_their':>8} {'n_needed':>9} {'R_ref':>12} {'relerr R':>10} {'relerr Js':>10} {'relerr Jc':>10} {'ref chk':>9}"
    )
    for s in (0.0, 0.7):
        for t in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
            u, v = (t + s + 1) / 2, (t - s + 1) / 2
            Js, Jc, n, eta_l = trap(u, v)
            Rs, Rc = reference(u, v, eta_l)
            Rs2, Rc2 = reference(u, v, eta_l, panels_per_period=12, nodes=16)
            R, Rref, Rref2 = Js**2 + Jc**2, Rs**2 + Rc**2, Rs2**2 + Rc2**2
            need = int(45.0 * max(u, v) * k * eta_c / (2 * np.pi))
            print(
                f"{t:7.1f} {s:4.1f} {u:8.2f} {n:8d} {need:9d} {Rref:12.4e} {abs(R-Rref)/Rref:10.2e} {abs(Js-Rs)/abs(Rs):10.2e} {abs(Jc-Rc)/abs(Rc):10.2e} {abs(Rref-Rref2)/Rref:9.1e}"
            )
