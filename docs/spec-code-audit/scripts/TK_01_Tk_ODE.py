import os
"""
TK_01: derive spec 01 R14/R21 (redshift-space phi equation) from spec 01 R5 (conformal-time
equation) by the variable change of R12/R13, and compare with the RHS coded in
ComputeTargets/TkNumericIntegration.py:73-80.
"""

import sympy as sp

z, k, a0 = sp.symbols("z k a0", positive=True)
H = sp.Function("H")(z)
cs2 = sp.Function("c")(z)  # c_s^2 = wPerturbations(z)
f = sp.Function("f")(z)  # phi(z)

a = a0 / (1 + z)
calH = a * H  # R13: script-H = aH

# R15: eps = (1+z)/H dH/dz  ->  dH/dz = H eps/(1+z)
eps = sp.Symbol("epsilon")
subs_dH = {sp.Derivative(H, z): H * eps / (1 + z)}


def d_deta(expr):
    """R12: d/deta = -(1+z) a H d/dz"""
    return sp.expand(-(1 + z) * a * H * sp.diff(expr, z))


# --- check R13:  calH' = a^2 H^2 (1-eps) -------------------------------------
calH_prime = d_deta(calH).subs(subs_dH)
r13 = sp.simplify(calH_prime - a**2 * H**2 * (1 - eps))
print("R13 check  calH' - a^2H^2(1-eps) =", r13)

# --- R5 in conformal time ----------------------------------------------------
phi_p = d_deta(f)
phi_pp = sp.expand(-(1 + z) * a * H * sp.diff(phi_p, z))

R5 = (
    phi_pp
    + 3 * calH * (1 + cs2) * phi_p
    + (2 * calH_prime + calH**2 * (1 + 3 * cs2)) * f
    + cs2 * k**2 * f
)
R5 = sp.expand(R5.subs(subs_dH).doit())

# normalise: divide by the coefficient of f''
c2 = sp.simplify(R5.coeff(sp.Derivative(f, z, 2)))
R14 = sp.expand(sp.simplify(R5 / c2))
print("\ncoefficient of f'' before normalising:", sp.simplify(c2))

coeff_f1 = sp.simplify(R14.coeff(sp.Derivative(f, z, 1)))
coeff_f0 = sp.simplify(sp.expand(R14 - coeff_f1 * sp.Derivative(f, z, 1)).coeff(f))

# spec R14/R21 (boxed):
spec_f1 = (eps - 3 * (1 + cs2)) / (1 + z)
spec_f0 = (3 * (1 + cs2) - 2 * eps) / (1 + z) ** 2 + cs2 * k**2 / (a0**2 * H**2)

print("\nR14 friction coefficient, derived :", sp.simplify(coeff_f1))
print("R14 friction coefficient, spec    :", sp.simplify(spec_f1))
print("difference                        :", sp.simplify(coeff_f1 - spec_f1))

print("\nR14 mass coefficient, derived     :", sp.simplify(coeff_f0))
print("R14 mass coefficient, spec        :", sp.simplify(spec_f0))
print("difference                        :", sp.simplify(coeff_f0 - spec_f0))

# --- the code's RHS ----------------------------------------------------------
# TkNumericIntegration.RHS, lines 73-80, with k_over_H = k_code/H and k_code = k/a0
T = sp.Function("T")(z)
Tprime = sp.Derivative(T, z)
k_code = k / a0
code_rhs = -(eps - 3 * (1 + cs2)) * Tprime / (1 + z) - (
    (3 * (1 + cs2) - 2 * eps) / (1 + z) ** 2 + cs2 * (k_code / H) ** 2
) * T

# spec form solved for T''
spec_rhs = -spec_f1 * Tprime - spec_f0 * T
print("\ncode RHS - spec RHS =", sp.simplify(sp.expand(code_rhs - spec_rhs)))
