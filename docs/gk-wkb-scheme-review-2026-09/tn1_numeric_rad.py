"""GkNumericIntegration path (numeric_with_phase_cut, stop mode) on exact radiation: accuracy vs envelope, cost, stop point."""
import sys, time, numpy as np
from math import exp, pi, sin, cos
from types import SimpleNamespace as NS
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import RadModel, grid
from CosmologyConcepts import redshift, redshift_array
import ComputeTargets.GkNumericIntegration; GN = sys.modules["ComputeTargets.GkNumericIntegration"]
import Quadrature.integrators.numeric_with_phase_cut as N
N.check_units = lambda *a, **k: None
run = N.numeric_with_phase_cut._function
k = 1.0e6; s_exit = k; s_e3, s_e4, s_e6 = k/exp(3), k/exp(4), k/exp(6)
kmock = NS(k=NS(k=k, k_inv_Mpc=k, store_id=0), z_exit=s_exit-1)
proxy = NS(get=lambda: RadModel())
def exact(s_s, s):
    th = k*(1/s_s - 1/s); env = s_s**2/k
    return env*sin(th), cos(th)*s_s**2/s**2, th, env
print(f"{'x_source':>8} {'(atol,rtol)':>14} {'nfev':>6} {'t[s]':>5} {'max|dG|/env':>12} {'max|dG_p|/env_p':>15} {'stop x':>7} {'G_stop/env':>10} {'|dG_stop|/env':>13} {'|dGp_stop|/env_p':>16} {'stop phase err':>14}")
for x_s in [exp(-5), exp(-2), 1.0, exp(2), exp(3.9)]:
    s_s = k/x_s
    zs = grid(s_s/10**(12/100)-1, 0.85*s_e6-1, 100/12)
    zsamp = redshift_array([redshift(i+1, float(z)) for i, z in enumerate(zs)])
    for atol, rtol in [(1e-10, 1e-8), (1e-13, 1e-11)]:
        t0 = time.perf_counter()
        out = run(proxy, kmock, redshift(0, s_s-1), zsamp, 0.0, 1.0, GN.RHS, atol=atol, rtol=rtol, delta_logz=0.01, mode="stop",
                  stop_search_window_z_begin=min(s_e3, s_s)-1, stop_search_window_z_end=s_e6-1, task_label="tn1", object_label="G")
        dt = time.perf_counter()-t0
        G, Gp = np.array(out["value_sample"]), np.array(out["deriv_sample"]); zz = zs[:len(G)]
        ex = [exact(s_s, 1+z) for z in zz]
        eG = max(abs(g - e[0])/e[3] for g, e in zip(G, ex)); eGp = max(abs(gp - e[1])/(s_s**2/(1+z)**2) for gp, e, z in zip(Gp, ex, zz))
        s_stop = 1 + (s_exit-1) - out["stop_deltaz_subh"]
        Gs, Gps, ths, env = exact(s_s, s_stop)
        # nearest exact extremum: cos(theta)=0
        th_ext = -(pi/2) - pi*round((-ths - pi/2)/pi)
        print(f"{x_s:8.3g} {f'({atol:.0e},{rtol:.0e})':>14} {out['data'].RHS_evaluations:6d} {dt:5.2f} {eG:12.2e} {eGp:15.2e} {k/s_stop:7.1f} {out['stop_value']/env:10.6f} {abs(out['stop_value']-Gs)/env:13.2e} {abs(out['stop_deriv']-Gps)/(s_s**2/s_stop**2):16.2e} {ths-th_ext:14.2e}")
