import os
"""GK_02: reproduce Gk_omegaEff_sq (spec 02 R20-R22 / R32) and Gk_d_ln_omegaEff_dz
(spec 02 R31, the NUM 10 fix) symbolically, and compare with ComputeTargets/WKB_Gk.py.
"""

import sympy as sp

z, k = sp.symbols("z k_phys", positive=True)
H = sp.Function("H")(z)
one_plus_z = 1 + z

eps = one_plus_z * sp.diff(sp.log(H), z)  # spec R21
epsp = sp.diff(eps, z)

# --- omega_eff^2 built from scratch (R20): f''/f + eps/(1+z) f'/f + k^2/H^2 + (eps-2)/(1+z)^2
# with dln f/dz = -eps/2/(1+z)
lnf = sp.Function("lnf")(z)
dlnf = -eps / 2 / one_plus_z
f_p_over_f = dlnf
f_pp_over_f = sp.diff(dlnf, z) + dlnf**2  # (f'/f)' + (f'/f)^2
omega_sq_derived = sp.simplify(
    f_pp_over_f + eps / one_plus_z * f_p_over_f + k**2 / H**2 + (eps - 2) / one_plus_z**2
)

# --- spec R22 / R32 closed form
E, Ep = sp.symbols("E Ep")  # stand-ins for eps, eps'
spec_omega_sq = k**2 / H**2 - Ep / 2 / one_plus_z + (3 * E / 2 - E**2 / 4 - 2) / one_plus_z**2

# --- code: ComputeTargets/WKB_Gk.py:4-19
code_omega_sq = (
    (k / H) ** 2
    + (-Ep / 2 / one_plus_z)
    + (3 * E / 2 - E * E / 4 - 2) / one_plus_z**2
)

print("R22 vs code omegaEff^2 difference:", sp.simplify(spec_omega_sq - code_omega_sq))
sub = {E: eps, Ep: epsp}
print(
    "derived-from-R20 vs R22 difference:",
    sp.simplify(omega_sq_derived - spec_omega_sq.subs(sub)),
)

# --- d omega_eff/dz.  R31 gives 2 omega omega'
spec_2ww = (
    (-sp.Symbol("Epp") / 2 - 2 * E * k**2 / H**2) / one_plus_z
    + (2 * Ep - E * Ep / 2) / one_plus_z**2
    - 2 * (3 * E / 2 - E**2 / 4 - 2) / one_plus_z**3
)
Epp = sp.Symbol("Epp")
sub2 = {E: eps, Ep: epsp, Epp: sp.diff(eps, z, 2)}
true_2ww = sp.diff(spec_omega_sq.subs(sub2), z)
print(
    "R31 vs d(omega^2)/dz difference:",
    sp.simplify(spec_2ww.subs(sub2) - true_2ww),
)

# --- code: ComputeTargets/WKB_Gk.py:22-43 numerator
code_num = (
    (-Epp / 2 - 2 * E * (k / H) ** 2) / one_plus_z
    + (2 * Ep - E * Ep / 2) / one_plus_z**2
    - (3 * E - E * E / 2 - 4) / one_plus_z**3
)
print("R31 vs code numerator difference:", sp.simplify(spec_2ww - code_num))
print(
    "code d_ln_omegaEff_dz == omega'/omega ?",
    sp.simplify(
        code_num.subs(sub2) / (2 * code_omega_sq.subs(sub2))
        - true_2ww / (2 * spec_omega_sq.subs(sub2))
    )
    == 0,
)
