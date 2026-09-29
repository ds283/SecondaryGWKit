"""Einstein tensor for ds^2 = a^2(eta)[-e^{2 phi} d eta^2 + e^{-2 phi} (delta_ij + h_ij) dx^i dx^j]
to second order in phi (bookkeeping e) and first order in h (bookkeeping d). Mixed components G^mu_nu.
Signature (-,+,+,+).

Run from the repository root:  ./venv/bin/python docs/derivation/scripts/einstein_tensor.py
Supports Sec. 2.1, 3.1 and 3.2 of docs/derivation/one-loop-sigw-derivation.tex."""

import sympy as sp

eta, x, y, zc, e, d = sp.symbols("eta x y z varepsilon delta")
X = [eta, x, y, zc]
a = sp.Function("a")(eta)
phi = sp.Function("phi")(eta, x, y, zc)
h = sp.Function("h")(
    eta, zc
)  # h_xy = h_yx = h(eta,z): transverse-traceless plus mode along z


def trunc(expr):
    expr = sp.expand(expr)
    P = sp.Poly(expr, e, d)
    out = 0
    for (i, j), c in P.terms():
        if i <= 2 and j <= 1:
            out += c * e**i * d**j
    return sp.expand(out)


E2p = 1 + 2 * e * phi + 2 * e**2 * phi**2  # e^{2 e phi} to O(e^2)
E2m = 1 - 2 * e * phi + 2 * e**2 * phi**2  # e^{-2 e phi}
gm = sp.zeros(4, 4)
gm[0, 0] = -(a**2) * E2p
for i in range(1, 4):
    gm[i, i] = a**2 * E2m
gm[1, 2] = gm[2, 1] = a**2 * E2m * d * h
gm = gm.applyfunc(trunc)
# inverse: g = g0 + dg, g^{-1} = g0^{-1} - g0^{-1} dg g0^{-1} + g0^{-1} dg g0^{-1} dg g0^{-1}
g0 = sp.diag(-(a**2), a**2, a**2, a**2)
g0i = g0.inv()
dg = gm - g0
M = g0i * dg
ginv = (g0i - M * g0i + M * M * g0i - M * M * M * g0i).applyfunc(trunc)
assert (gm * ginv).applyfunc(trunc) == sp.eye(4), "inverse check failed"
Gam = [[[0] * 4 for _ in range(4)] for _ in range(4)]
for l in range(4):
    for m in range(4):
        for n in range(m, 4):
            s = sum(
                ginv[l, k]
                * (
                    sp.diff(gm[k, m], X[n])
                    + sp.diff(gm[k, n], X[m])
                    - sp.diff(gm[m, n], X[k])
                )
                for k in range(4)
            )
            Gam[l][m][n] = Gam[l][n][m] = trunc(s / 2)
Ric = sp.zeros(4, 4)
for m in range(4):
    for n in range(m, 4):
        s = 0
        for l in range(4):
            s += sp.diff(Gam[l][m][n], X[l]) - sp.diff(Gam[l][m][l], X[n])
            for k in range(4):
                s += Gam[l][l][k] * Gam[k][m][n] - Gam[l][n][k] * Gam[k][m][l]
        Ric[m, n] = Ric[n, m] = trunc(s)
Rs = trunc(sum(ginv[m, n] * Ric[m, n] for m in range(4) for n in range(4)))
G = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        G[m, n] = trunc(
            sum(ginv[m, l] * Ric[l, n] for l in range(4))
            - sp.Rational(1, 2) * (1 if m == n else 0) * Rs
        )
calH = sp.diff(a, eta) / a


def coeff(expr, oe, od):
    return sp.expand(sp.expand(expr).coeff(e, oe).coeff(d, od))


lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, zc, 2)
print(
    "background a^2 G^0_0 + 3H^2:",
    sp.simplify(a**2 * coeff(G[0, 0], 0, 0) + 3 * calH**2),
)
print(
    "background a^2 G^x_x + (2H'+H^2):",
    sp.simplify(a**2 * coeff(G[1, 1], 0, 0) + (2 * sp.diff(calH, eta) + calH**2)),
)
g00 = a**2 * coeff(G[0, 0], 1, 0)
print(
    "O(phi) a^2 G^0_0 - [6H phi' + 6H^2 phi - 2 lap phi] =",
    sp.simplify(
        g00 - (6 * calH * sp.diff(phi, eta) + 6 * calH**2 * phi - 2 * lap(phi))
    ),
)
g0x = a**2 * coeff(G[0, 1], 1, 0)
print(
    "O(phi) a^2 G^0_x - [-2 d_x(phi' + H phi)] =",
    sp.simplify(g0x - (-2 * sp.diff(sp.diff(phi, eta) + calH * phi, x))),
)
gxx = a**2 * coeff(G[1, 1], 1, 0)
print(
    "O(phi) a^2 G^x_x - [6H phi' + 2phi'' + 2phi(2H'+H^2)] (should be pure gradient, d_x^2 - lap type) =",
    sp.simplify(
        gxx
        - (
            6 * calH * sp.diff(phi, eta)
            + 2 * sp.diff(phi, eta, 2)
            + 2 * phi * (2 * sp.diff(calH, eta) + calH**2)
        )
    ),
)
gxy1 = a**2 * coeff(G[1, 2], 1, 0)
print("O(phi) a^2 G^x_y =", gxy1)
gxy2 = a**2 * coeff(G[1, 2], 2, 0)
print("O(phi^2) a^2 G^x_y =", gxy2)
gh = a**2 * coeff(G[1, 2], 0, 1)
print(
    "O(h) a^2 G^x_y - (1/2)[h'' + 2H h' - d_z^2 h] =",
    sp.simplify(
        gh
        - sp.Rational(1, 2)
        * (sp.diff(h, eta, 2) + 2 * calH * sp.diff(h, eta) - sp.diff(h, zc, 2))
    ),
)
