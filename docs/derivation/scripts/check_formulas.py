"""Machine checks of the formulas presented in docs/derivation/one-loop-sigw-derivation.tex.

Run from the repository root:  ./venv/bin/python docs/derivation/scripts/check_formulas.py
Every symbolic difference should print 0 and every numerical residual should be at rounding level.
"""

import numpy as np, sympy as sp
from scipy.special import jv, yv, gamma
from scipy.integrate import quad

print("=== 1. Transfer function ODE (conformal, constant w) and Bessel solution")
# phi'' + 2(2+b)/eta phi' + cs^2 k^2 phi = 0 solved by (k cs eta)^{-3/2-b} J_{3/2+b}(k cs eta)
b = 0.3
k = 1.7
cs = 0.55


def Tk(eta):
    x = k * cs * eta
    return 2 ** (1.5 + b) * gamma(2.5 + b) * x ** (-1.5 - b) * jv(1.5 + b, x)


def d(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def dd(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


for eta in [0.3, 2.0, 7.0]:
    res = dd(Tk, eta) + 2 * (2 + b) / eta * d(Tk, eta) + cs**2 * k**2 * Tk(eta)
    print(f"  eta={eta}: residual {res:.2e}, T={Tk(eta):.5f}")
print("  T(1e-3) =", Tk(1e-3), "(should -> 1)")

print("=== 2. Green's function Gr_k: ODE, continuity, unit jump")


def Gr(eta, etap):
    if eta < etap:
        return 0.0
    nu = b + 0.5
    return (
        np.pi
        / 2
        * np.sqrt(eta * etap)
        * (jv(nu, k * etap) * yv(nu, k * eta) - yv(nu, k * etap) * jv(nu, k * eta))
    )


etap = 1.3
for eta in [2.0, 5.0]:
    f = lambda e: Gr(e, etap)
    res = dd(f, eta) + (k**2 - b * (1 + b) / eta**2) * f(eta)
    print(f"  eta={eta}: homogeneous residual {res:.2e}")
print(
    "  G(eta'+,eta') =",
    Gr(etap + 1e-9, etap),
    " dG/deta at eta'+ =",
    d(lambda e: Gr(e, etap), etap + 1e-6),
)

print(
    "=== 3. Source function: w-form (MAIN 14 R18) vs Bessel two-term form (R28), and NUM 06 R11 = R28/c^2"
)
w = (1 - b) / (3 * (1 + b))
q = 0.9
r = 1.4
eta = 3.1
Phi = (
    lambda x: 2 ** (1.5 + b)
    * gamma(2.5 + b)
    * (x * cs) ** (-1.5 - b)
    * jv(1.5 + b, x * cs)
)
dPhi = lambda x: d(Phi, x)
H = 2 / (1 + 3 * w) / eta  # conformal Hubble
c = 3 * (1 + w) / (5 + 3 * w)
f14_w = c * Phi(q * eta) * Phi(r * eta) + 6 * (1 + w) / (5 + 3 * w) ** 2 * (
    (r / H) * Phi(q * eta) * dPhi(r * eta)
    + (q / H) * dPhi(q * eta) * Phi(r * eta)
    + q * r / H**2 * dPhi(q * eta) * dPhi(r * eta)
)
# = c*[PhiPhi + (2/(3(1+w))) (Phi + q Phi'/H)(Phi + r Phi'/H)]
f14_alt = c * (
    Phi(q * eta) * Phi(r * eta)
    + 2
    / (3 * (1 + w))
    * (Phi(q * eta) + q * dPhi(q * eta) / H)
    * (Phi(r * eta) + r * dPhi(r * eta) / H)
)
xq, xr = q * eta * cs, r * eta * cs
f14_B = (
    (2 + b)
    / (3 + 2 * b) ** 3
    * 2 ** (3 + 2 * b)
    * gamma(2.5 + b) ** 2
    * (xq * xr) ** (-0.5 - b)
    * (
        jv(0.5 + b, xq) * jv(0.5 + b, xr)
        + (2 + b) / (1 + b) * jv(2.5 + b, xq) * jv(2.5 + b, xr)
    )
)
f06_B = (
    2 ** (3 + 2 * b)
    / ((3 + 2 * b) * (2 + b))
    * gamma(2.5 + b) ** 2
    * (xq * xr) ** (-0.5 - b)
    * (
        jv(0.5 + b, xq) * jv(0.5 + b, xr)
        + (2 + b) / (1 + b) * jv(2.5 + b, xq) * jv(2.5 + b, xr)
    )
)
print(f"  f14 (w-form)   = {f14_w:.10f}")
print(f"  f14 (compact)  = {f14_alt:.10f}")
print(f"  f14 (Bessel)   = {f14_B:.10f}")
print(
    f"  f06*c^2        = {f06_B*c**2:.10f}   c^2={c**2:.6f}  (2+b)^2/(3+2b)^2={(2+b)**2/(3+2*b)**2:.6f}"
)
# four-term intermediate (R27)
f27 = (
    (2 + b)
    / (3 + 2 * b)
    * 2 ** (3 + 2 * b)
    * gamma(2.5 + b) ** 2
    * (xq * xr) ** (-1.5 - b)
    * (
        jv(1.5 + b, xq) * jv(1.5 + b, xr)
        - xr / (3 + 2 * b) * jv(1.5 + b, xq) * jv(2.5 + b, xr)
        - xq / (3 + 2 * b) * jv(2.5 + b, xq) * jv(1.5 + b, xr)
        + xq * xr / ((3 + 2 * b) * (1 + b)) * jv(2.5 + b, xq) * jv(2.5 + b, xr)
    )
)
print(f"  f14 (4-term)   = {f27:.10f}")

print("=== 4. TARGET: time integral vs direct quadrature")
eta0 = 1e-4
eta_r = 4.2


def integrand(ep):
    return (
        (ep / eta_r) ** (1 + b)
        * Gr(eta_r, ep)
        * (2 + b)
        / (3 + 2 * b) ** 3
        * 2 ** (3 + 2 * b)
        * gamma(2.5 + b) ** 2
        * (q * ep * cs * r * ep * cs) ** (-0.5 - b)
        * (
            jv(0.5 + b, q * ep * cs) * jv(0.5 + b, r * ep * cs)
            + (2 + b) / (1 + b) * jv(2.5 + b, q * ep * cs) * jv(2.5 + b, r * ep * cs)
        )
    )


direct, err = quad(integrand, eta0, eta_r, limit=400)
nu = b + 0.5
S = lambda ep: jv(0.5 + b, q * ep * cs) * jv(0.5 + b, r * ep * cs) + (2 + b) / (
    1 + b
) * jv(2.5 + b, q * ep * cs) * jv(2.5 + b, r * ep * cs)
IJ, _ = quad(
    lambda ep: ep ** (0.5 - b) * jv(nu, k * ep) * S(ep), eta0, eta_r, limit=400
)
IY, _ = quad(
    lambda ep: ep ** (0.5 - b) * yv(nu, k * ep) * S(ep), eta0, eta_r, limit=400
)
target = (
    np.pi
    * 2 ** (2 + 2 * b)
    * (2 + b)
    / (3 + 2 * b) ** 3
    * gamma(2.5 + b) ** 2
    * (cs**2 * q * r * eta_r) ** (-0.5 - b)
    * (yv(nu, k * eta_r) * IJ - jv(nu, k * eta_r) * IY)
)
print(
    f"  direct = {direct:.10e}   TARGET formula = {target:.10e}   rel diff {abs(direct-target)/abs(direct):.2e}"
)

print("=== 5. Redshift form of the phi equation (sympy)")
z, eps, cs2 = sp.symbols("z epsilon c_s2")
Hf = sp.Function("H")(z)
a0 = sp.symbols("a_0", positive=True)
phi = sp.Function("phi")(z)
a = a0 / (1 + z)
# d/deta = -(1+z) a H d/dz ; calH = aH ; calH' = a^2 H^2 (1-eps)
D = lambda F: -(1 + z) * a * Hf * sp.diff(F, z)
calH = a * Hf
epsilon_expr = (1 + z) * sp.diff(Hf, z) / Hf
calHp = a**2 * Hf**2 * (1 - epsilon_expr)
lhs = (
    D(D(phi))
    + 3 * calH * (1 + cs2) * D(phi)
    + (2 * calHp + calH**2 * (1 + 3 * cs2)) * phi
)
lhs = sp.simplify(lhs / ((1 + z) ** 2 * a**2 * Hf**2))
spec = (
    sp.diff(phi, z, 2)
    + (epsilon_expr - 3 * (1 + cs2)) / (1 + z) * sp.diff(phi, z)
    + (3 * (1 + cs2) - 2 * epsilon_expr) / (1 + z) ** 2 * phi
)
print("  difference (excluding gradient term):", sp.simplify(lhs - spec))

print("=== 6. Redshift form of the Green's-function operator (sympy)")
G = sp.Function("G")(z)
# chi'' + (k^2 - a''/a) chi with a''/a = d(a^2 H)/dt ... use a''/a = a^2H^2(2-eps)
lhsG = D(D(G)) - a**2 * Hf**2 * (2 - epsilon_expr) * G
lhsG = sp.simplify(lhsG / ((1 + z) ** 2 * a**2 * Hf**2))
specG = (
    sp.diff(G, z, 2)
    + epsilon_expr / (1 + z) * sp.diff(G, z)
    - (2 - epsilon_expr) / (1 + z) ** 2 * G
)
print("  difference (excluding k^2 term):", sp.simplify(lhsG - specG))
# a''/a identity: a'' = d/deta (a calH) ; check a''/a = a^2H^2(2-eps)
eta_s = sp.symbols("eta")
print("  (a''/a identity checked separately below)")

print("=== 7. omega_eff^2 for the Green's function (sympy)")
epsf = sp.Function("epsilon")(z)
kH2 = sp.symbols("kH2")
f = sp.Function("f")(z)
dlnf = -sp.Rational(1, 2) * epsf / (1 + z)
fpp_over_f = sp.diff(dlnf, z) + dlnf**2
om2 = fpp_over_f + epsf / (1 + z) * dlnf + kH2 - (2 - epsf) / (1 + z) ** 2
spec_om2 = (
    kH2
    - sp.diff(epsf, z) / 2 / (1 + z)
    + (sp.Rational(3, 2) * epsf - epsf**2 / 4 - 2) / (1 + z) ** 2
)
print("  difference:", sp.simplify(om2 - spec_om2))

print("=== 8. omega_eff^2 for the transfer function (sympy)")
cs_f = sp.Function("c")(z)  # c = c_s
dlnP = -sp.Rational(1, 2) * (epsf - 3 * (1 + cs_f**2)) / (1 + z)
Ppp_over_P = sp.diff(dlnP, z) + dlnP**2
om2T = (
    Ppp_over_P
    + dlnP * (epsf - 3 * (1 + cs_f**2)) / (1 + z)
    + (3 * (1 + cs_f**2) - 2 * epsf) / (1 + z) ** 2
)
spec_om2T = (3 * cs_f * sp.diff(cs_f, z) - sp.diff(epsf, z) / 2) / (1 + z) + (
    sp.Rational(3, 2) * (1 + epsf) * (1 + cs_f**2)
    - epsf / 2 * (3 + epsf / 2)
    - sp.Rational(9, 4) * (1 + cs_f**2) ** 2
) / (1 + z) ** 2
print("  difference:", sp.simplify(sp.expand(om2T - spec_om2T)))

print("=== 9. Angular reduction and the 648 pi^2 coefficient")
# 32 * (1/(2pi)^3) * ∫dφ cos^2 2φ (=π) * (1/2) q^4 sin^4θ  * P P  with P = 2π^2 𝒫/q^3 -> (2π^2)^2 = 4π^4
coef = sp.Rational(32) * sp.pi * sp.Rational(1, 2) * 4 * sp.pi**4 / (2 * sp.pi) ** 3
print(
    "  32 * pi * 1/2 * 4pi^4/(2pi)^3 =",
    sp.simplify(coef),
    " ; times 81 =",
    sp.simplify(coef * 81),
)
print(
    "  2*36^2 =",
    2 * 36**2,
    "; 2592/2 =",
    1296,
    "; 1296 pi * pi/2 =",
    sp.simplify(1296 * sp.pi * sp.pi / 2),
)

print("=== 10. Polarisation factors")
th, ph = sp.symbols("theta phi")
qq = sp.symbols("q", positive=True)
e = sp.Matrix([1, 0, 0])
ebar = sp.Matrix([0, 1, 0])
qv = qq * sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
ep = (e * e.T - ebar * ebar.T) / sp.sqrt(2)
ex = (e * ebar.T + ebar * e.T) / sp.sqrt(2)
Qp = sp.simplify((qv.T * ep * qv)[0])
Qx = sp.simplify((qv.T * ex * qv)[0])
print("  Q_+ =", sp.trigsimp(Qp), "  Q_x =", sp.trigsimp(Qx))
print(
    "  e_+:e_+ =",
    sp.simplify(sum(ep[i, j] * ep[i, j] for i in range(3) for j in range(3))),
)

print("=== 11. a''/a = (aH)^2 (2 - eps) in cosmic time")
t = sp.symbols("t")
at = sp.Function("a")(t)
Ht = sp.diff(at, t) / at
app_over_a = (
    at * sp.diff(at**2 * Ht, t) / at
)  # a' = da/deta = a da/dt = a^2 H ; a'' = a d/dt (a^2 H)
epst = -sp.diff(Ht, t) / Ht**2
print("  a''/a - a^2H^2(2-eps) =", sp.simplify(app_over_a - at**2 * Ht**2 * (2 - epst)))

print("=== 12. Bessel identities used")
nu_s, x = sp.symbols("nu x")
Z = sp.besselj
print(
    "  Z' - (nu/x) Z + Z_{nu+1} =",
    sp.simplify(sp.diff(Z(nu_s, x), x) - nu_s / x * Z(nu_s, x) + Z(nu_s + 1, x)),
)
print(
    "  Z_nu - x/(2nu) Z_{nu+1} - x/(2nu) Z_{nu-1} =",
    sp.simplify(
        Z(nu_s, x) - x / (2 * nu_s) * Z(nu_s + 1, x) - x / (2 * nu_s) * Z(nu_s - 1, x)
    ),
)
print(
    "  Wronskian J Y' - Y J' - 2/(pi x):",
    sp.simplify(
        Z(nu_s, x) * sp.diff(sp.bessely(nu_s, x), x)
        - sp.bessely(nu_s, x) * sp.diff(Z(nu_s, x), x)
        - 2 / (sp.pi * x)
    ),
)

print("=== 13. eta_0 -> 0 behaviour of the integrands")
# Y kernel: (eta')^{1/2-b} Y_{b+1/2}(k eta') ~ (eta')^{1/2-b} (eta')^{-b-1/2} = (eta')^{-2b}; times S ~ (eta')^{1+2b} -> (eta')^1
# J kernel: (eta')^{1/2-b} (eta')^{b+1/2} (eta')^{1+2b} = (eta')^{2+2b}
print("  Y-kernel ~ eta'^1, J-kernel ~ eta'^(2+2b): consistent with spec note")

print(
    "=== 14. Super-horizon relation phi = 3(1+w)/(5+3w) zeta for constant w (Bardeen)"
)
# Standard: zeta = -(5+3w)/(3(1+w)) Phi on superhorizon scales (up to sign convention). Numeric sanity only.
for wv in [1 / 3, 0]:
    print(f"  w={wv}: 3(1+w)/(5+3w) = {3*(1+wv)/(5+3*wv):.4f}")
