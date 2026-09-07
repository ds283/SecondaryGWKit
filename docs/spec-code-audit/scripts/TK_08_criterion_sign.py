import os
"""
TK_08: sign of Tk_d_ln_omegaEff_dz, and the consequences for the guard at
ComputeTargets/TkWKBIntegration.py:356-362 (which omits fabs, unlike lines 442, 490 and
Quadrature/integrators/WKB_phase_function.py:662).
"""
import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import numpy as np

from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq
from TK_03_numeric_vs_analytic import FakeModel

print(f"{'w':>6} {'k':>9} {'z':>12} {'k/aH':>9} {'omega_eff^2':>13} "
      f"{'d_ln_omega_dz':>15} {'criterion (no fabs)':>20}")
for w in [1.0 / 3.0, 0.2]:
    m = FakeModel(w)
    for k in [1.0e4]:
        for z in [1.0e3, 1.0e2, 1.0e1, 1.0]:
            o2 = Tk_omegaEff_sq(m, k, z)
            if o2 <= 0:
                continue
            d = Tk_d_ln_omegaEff_dz(m, k, z)
            print(f"{w:6.3f} {k:9.3g} {z:12.5g} {(1+z)*k/m.Hubble(z):9.4g} "
                  f"{o2:13.5g} {d:15.6g} {d/np.sqrt(o2):20.6g}")
print("\n-> d ln omega_eff / dz is negative throughout the sub-horizon regime, so")
print("   'if WKB_criterion_init > 1.0' at TkWKBIntegration.py:357 can never fire.")
