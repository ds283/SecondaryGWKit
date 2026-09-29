import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpt_rd_quadrature import integrand, reference, k, eta_c, cs


def trap_var(u, v, qfast):
    umax = max(u, v, 1.0)
    eta_l = 1e-2 / (umax * k)
    n = int(np.clip(45.0 * qfast * k * eta_c / (2 * np.pi), 800, 200000))
    eta = np.geomspace(eta_l, eta_c, n)
    s, c = integrand(eta, u, v, eta_l)
    # points per period of the fastest leg inside the last decade eta in [40, 400]
    last = np.sum(eta > 40.0)
    periods_last = (
        360.0 * cs * max(u, v, 1.0) * k / (2 * np.pi)
        if max(u, v) * cs > 1
        else 360.0 * k / (2 * np.pi)
    )
    return np.trapezoid(s, eta), np.trapezoid(c, eta), n, eta_l, last / periods_last


print(f"{'t':>6} {'u':>6} | {'q_fast=max(u,v)':^28} | {'q_fast=max(u,v,1)':^28}")
print(
    f"{'':>6} {'':>6} | {'n':>7} {'pts/per(last)':>13} {'relerr R':>9} | {'n':>7} {'pts/per(last)':>13} {'relerr R':>9}"
)
s = 0.0
for t in (0.1, 0.3, 0.5, 0.6, 0.7, 0.732, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0):
    u, v = (t + s + 1) / 2, (t - s + 1) / 2
    Rs, Rc = reference(u, v, 1e-2 / (max(u, v, 1.0) * k))
    Rref = Rs**2 + Rc**2
    row = f"{t:6.3f} {u:6.3f}"
    for qf in (max(u, v), max(u, v, 1.0)):
        Js, Jc, n, eta_l, ppp = trap_var(u, v, qf)
        row += f" | {n:7d} {ppp:13.1f} {abs(Js**2+Jc**2-Rref)/Rref:9.2e}"
    print(row)
