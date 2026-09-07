import os
"""GK_01: derive the redshift-space Green's-function ODE (spec 02 R11->R17) with sympy and
compare term-by-term with the RHS coded in ComputeTargets/GkNumericIntegration.py:64-66.

Chain: G_{,eta eta} + (k^2 - a''/a) G = 0, with a = a0/(1+z), tau = a0*eta, dtau/dz = -1/H.
"""

import sympy as sp

z, a0, kcom = sp.symbols("z a0 k_com", positive=True)
H = sp.Function("H")(z)
G = sp.Function("G")(z)

one_plus_z = 1 + z
eps_expr = one_plus_z * sp.diff(sp.log(H), z)  # spec R21: eps = (1+z) dlnH/dz

# d/dtau = -H d/dz   (BackgroundModel.py:56  d tau/dz = -1/H)
d_dtau = lambda f: -H * sp.diff(f, z)
# d/deta = a0 d/dtau
d_deta = lambda f: a0 * d_dtau(f)

a = a0 / one_plus_z
a_pp_over_a = sp.simplify(d_deta(d_deta(a)) / a)

# express through eps
eps = sp.Symbol("epsilon")
Hp = sp.Symbol("Hprime")
subs_deriv = {sp.diff(H, z): Hp}
a_pp_over_a_e = sp.simplify(a_pp_over_a.subs(subs_deriv).subs(Hp, eps * H / one_plus_z))
target = a0**2 * H**2 * (2 - eps) / one_plus_z**2
print("a''/a  =", sp.simplify(a_pp_over_a_e))
print("check a''/a == a0^2 H^2 (2-eps)/(1+z)^2 :", sp.simplify(a_pp_over_a_e - target) == 0)

# full equation, divided by a0^2 H^2
eqn = d_deta(d_deta(G)) + (kcom**2 - a_pp_over_a) * G
eqn = eqn.subs(subs_deriv).subs(Hp, eps * H / one_plus_z)
eqn = sp.expand(sp.simplify(eqn / (a0**2 * H**2)))

k_phys = sp.Symbol("k_phys", positive=True)
eqn_phys = sp.expand(eqn.subs(kcom, a0 * k_phys))

spec_R17 = sp.expand(
    sp.diff(G, z, 2)
    + eps / one_plus_z * sp.diff(G, z)
    + (k_phys**2 / H**2 - (2 - eps) / one_plus_z**2) * G
)
print("\nderived (in k_phys):", eqn_phys)
print("spec 02 R17 LHS   :", spec_R17)
print("difference        :", sp.simplify(eqn_phys - spec_R17))

# code RHS: GkNumericIntegration.py:64-66
Gp = sp.Symbol("Gprime")
k_over_H_2 = (k_phys / H) ** 2
code_dGprime_dz = -eps * Gp / one_plus_z - (k_over_H_2 + (eps - 2) / one_plus_z**2) * G
# rewrite spec R17 (homogeneous) as G'' = ...
Gs = sp.Symbol("G")
spec_dGprime_dz = -(
    eps / one_plus_z * Gp + (k_phys**2 / H**2 - (2 - eps) / one_plus_z**2) * Gs
)
code_dGprime_dz = code_dGprime_dz.subs(G, Gs)
print("\ncode  G'' =", sp.expand(code_dGprime_dz))
print("spec  G'' =", sp.expand(spec_dGprime_dz))
print("difference:", sp.simplify(code_dGprime_dz - spec_dGprime_dz))
