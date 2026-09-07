import os
"""GK_07: verify spec 02 R24 / R29 (d ln H/dz, d^2 ln H/dz^2, d^3 ln H/dz^3 in LambdaCDM)
against the closed forms coded in CosmologyModels/LambdaCDM/LambdaCDM.py:131-197.
"""

import sympy as sp

z, om, orr, occ = sp.symbols("z Omega_m Omega_r Omega_cc", positive=True)
opz = 1 + z
E2 = om * opz**3 + orr * opz**4 + occ  # 3H^2 M_P^2 / rho_crit0, spec R24
lnH = sp.log(sp.sqrt(E2))

d1_true = sp.diff(lnH, z)
d2_true = sp.diff(lnH, z, 2)
d3_true = sp.diff(lnH, z, 3)

den = 2 * E2
d1_code = (3 * om * opz**2 + 4 * orr * opz**3) / den
d2_code = (6 * om * opz + 12 * orr * opz**2) / den - 2 * d1_code**2
d3_code = (6 * om + 24 * orr * opz) / den - 6 * d1_code * d2_code - 4 * d1_code**3

for name, a, b in (
    ("d lnH/dz   (R24)", d1_code, d1_true),
    ("d2 lnH/dz2 (R24)", d2_code, d2_true),
    ("d3 lnH/dz3 (R29)", d3_code, d3_true),
):
    print(f"{name}: difference = {sp.simplify(a - b)}")
