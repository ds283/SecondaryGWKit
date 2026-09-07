import os
"""GK_05: confirm the phase (div_2pi, mod_2pi) carry used by the Green's-function pipeline is
consistent:
  (a) WKB_mod_2pi / WKB_product_mod_2pi round-trip: div*2pi + mod == theta, mod in (-2pi, 0]
  (b) shift_theta_sample (used at GkWKBIntegration.py:414) preserves theta up to a constant
      2pi offset, so sin/cos are unchanged
  (c) phase_spline reassembles a known analytic phase, and theta_mod_2pi gives the right sin
"""

import sys
from math import sin, cos, fabs, pi, log

import numpy as np

sys.path.insert(0, os.getcwd())  # run from the repository root

from LiouvilleGreen.WKBtools import WKB_mod_2pi, WKB_product_mod_2pi, shift_theta_sample
from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.phase_spline import phase_spline

rng = np.random.default_rng(7)

print("=== (a) WKB_mod_2pi round trip")
worst = 0.0
worst_range = 0.0
for _ in range(20000):
    theta = rng.normal() * 10 ** rng.uniform(-2, 6)
    d, m = WKB_mod_2pi(theta)
    worst = max(worst, fabs((d * TWO_PI + m) - theta) / max(fabs(theta), 1.0))
    if m > 0.0 or m <= -TWO_PI - 1e-12:
        worst_range = max(worst_range, fabs(m))
print(f"    worst |div*2pi+mod - theta|/|theta| = {worst:.3e}; range violations metric {worst_range:.3e}")

print("=== (a') WKB_product_mod_2pi round trip (omega_init*(1+u) * Q + mod_init)")
worst = 0.0
for _ in range(20000):
    big = 10 ** rng.uniform(0, 7)
    small = -rng.uniform(0.0, 2.0)
    mod_init = -rng.uniform(0.0, TWO_PI)
    d, m = WKB_product_mod_2pi(big, small, mod_init)
    exact = big * small + mod_init
    worst = max(worst, fabs((d * TWO_PI + m) - exact) / max(fabs(exact), 1.0))
print(f"    worst relative reassembly error = {worst:.3e}")

print("=== (b) shift_theta_sample")
theta_true = -np.cumsum(rng.uniform(0.05, 0.6, 300))
pairs = [WKB_mod_2pi(t) for t in theta_true]
div = [p[0] for p in pairs]
mod = [p[1] for p in pairs]
shift = 0.7351
new_div, new_mod = shift_theta_sample(div, mod, shift)
recon = np.array([d * TWO_PI + m for d, m in zip(new_div, new_mod)])
offset = recon - (theta_true + shift)
print(f"    reconstructed theta+shift up to constant offset: spread of offset/2pi = "
      f"{np.ptp(offset)/TWO_PI:.3e}, offset/2pi = {offset[0]/TWO_PI:.6g}")
print(f"    max |sin(new_mod) - sin(theta+shift)| = "
      f"{np.max(np.abs(np.sin(new_mod) - np.sin(theta_true + shift))):.3e}")

print("=== (c) phase_spline reassembly of an analytic phase")
# analytic phase theta(z) = -A [ (1+z0)^-1 - (1+z)^-1 ] style: monotone decreasing in decreasing z
zs = np.geomspace(1.0, 400.0, 240)[::-1]  # descending, like GkSource z_sample
A = 5.0e3
theta_ex = -A * (1.0 / (1.0 + zs[0]) - 1.0 / (1.0 + zs)) * -1.0
theta_ex = -np.abs(theta_ex)  # keep negative, monotone
pairs = [WKB_mod_2pi(t) for t in theta_ex]
# phase_spline needs ascending x
order = np.argsort(np.log(1.0 + zs))
logx = [log(1.0 + zs[i]) for i in order]
dv = [pairs[i][0] for i in order]
md = [pairs[i][1] for i in order]
ps = phase_spline(logx, dv, md, x_is_log=True, x_is_redshift=True,
                  chunk_step=None, chunk_logstep=None, increasing=False)
err = []
for i in order[5:-5]:
    z = zs[i]
    err.append(fabs(sin(ps.theta_mod_2pi(z)) - sin(theta_ex[i])))
print(f"    max |sin(spline mod 2pi) - sin(theta_exact)| at nodes = {max(err):.3e}")
