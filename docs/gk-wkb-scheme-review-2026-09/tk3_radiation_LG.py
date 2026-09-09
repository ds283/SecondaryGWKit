"""T_k LG reconstruction (store() algebra) vs the exact radiation transfer function: LG truncation floor vs x_init."""
import numpy as np
from math import sqrt, log, exp, sin, cos, atan2, pi
from scipy.integrate import quad
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
k = 1.0e6; cs = 1/sqrt(3.0)
# normalized radiation: H = s^2, tau = 1/s, x = k c_s tau = k c_s / s ; omega^2 = k^2/(3 s^4) - 2/s^2
om2 = lambda s: k*k/(3*s**4) - 2.0/s**2
dlnom = lambda s: (-4*k*k/(3*s**5) + 4.0/s**3)/(2*om2(s))   # d ln omega / dz
env = lambda x: 3*sqrt(1+x*x)/x**3
print(f"{'x_init':>6} {'x range':>14} {'max|dT|/env':>12} {'|dA/A| at x=1e4':>16} {'phase err at x=1e4':>19} {'rho_T = theta-(x_i-x) at 1e4 (~1/x_i)':>34}")
for x_i in [24.0, 50.0, 100.0, 400.0]:
    s_i = k*cs/x_i
    T_i = compute_analytic_T(k, 1/3, 1/s_i); Tp_i = compute_analytic_Tprime(k, 1/3, 1/s_i, s_i**2)
    w_i = sqrt(om2(s_i))
    raw_cos = sqrt(w_i)*T_i; raw_sin = (Tp_i + (T_i/2)*(dlnom(s_i) + (2 - 4)/s_i))/sqrt(w_i)
    delta = atan2(raw_cos, raw_sin); B = sqrt(raw_cos**2 + raw_sin**2)
    xs = np.geomspace(x_i, 1e4, 400)
    errs = []; 
    for x in xs:
        s = k*cs/x
        th = -quad(lambda v: sqrt(om2(exp(v)))*exp(v), log(s), log(s_i), epsrel=1e-13, epsabs=0, limit=400)[0]
        f = 2*log(s/s_i)
        M = sqrt(s_i**2/(s**2*sqrt(om2(s))))*exp(f)*B
        T_wkb = M*sin(th + delta); T_ex = compute_analytic_T(k, 1/3, 1/s)
        errs.append((x, (T_wkb - T_ex)/env(x), M/env(x) - 1, th, T_wkb, T_ex))
    E = max(abs(e[1]) for e in errs)
    x, _, dA, th, Tw, Te = errs[-1]
    # phase error at the far end from the pair (T, T') would need T'; estimate from residual after amplitude: use asin
    ph_wkb = th + delta; ph_ex_mod = atan2(Te, sqrt(max(env(x)**2 - Te**2, 0)))  # |sin| only; report envelope-relative instead
    print(f"{x_i:6.0f} {f'{x_i:.0f}..1e4':>14} {E:12.2e} {abs(dA):16.2e} {'(see dT/env)':>19} {th - (x_i - x):34.5e}")
