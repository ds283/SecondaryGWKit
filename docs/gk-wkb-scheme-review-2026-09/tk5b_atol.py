"""TkNumericIntegration: is the 1e-5 error at production tolerances the absolute tolerance? Exact initial data, vary atol."""
import sys, numpy as np
from math import exp, sqrt
from types import SimpleNamespace as NS
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import grid
from CosmologyConcepts import redshift, redshift_array
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
import ComputeTargets.TkNumericIntegration; TN = sys.modules["ComputeTargets.TkNumericIntegration"]
import Quadrature.integrators.numeric_with_phase_cut as N
N.check_units = lambda *a, **kw: None
run = N.numeric_with_phase_cut._function
class Rad:
    def __init__(self): self.functions = NS(Hubble=lambda z: (1+z)**2, epsilon=lambda z: 2.0, d_epsilon_dz=lambda z: 0.0, d2_epsilon_dz2=lambda z: 0.0,
                                            wPerturbations=lambda z: 1/3, d_wPerturbations_dz=lambda z: 0.0, d2_wPerturbations_dz2=lambda z: 0.0)
k = 1e6; cs = 1/sqrt(3); s_exit = k; s_e3, s_e6 = k/exp(3), k/exp(6); s_i = k*exp(5)
kmock = NS(k=NS(k=k, k_inv_Mpc=k, store_id=0), z_exit=s_exit-1); proxy = NS(get=lambda: Rad())
zs = grid(s_i/10**0.01-1, 0.85*s_e6-1, 100); zsamp = redshift_array([redshift(i+1, float(z)) for i, z in enumerate(zs)])
env = lambda x: 3*sqrt(1+x*x)/x**3 if x > 1e-3 else 1.0
T0, Tp0 = compute_analytic_T(k, 1/3, 1/s_i), compute_analytic_Tprime(k, 1/3, 1/s_i, s_i**2)
for atol, rtol in [(1e-10, 1e-8), (1e-13, 1e-8), (1e-16, 1e-8), (1e-20, 1e-8)]:
    out = run(proxy, kmock, redshift(0, s_i-1), zsamp, T0, Tp0, TN.RHS, atol=atol, rtol=rtol, delta_logz=0.01, mode="stop",
              stop_search_window_z_begin=s_e3-1, stop_search_window_z_end=s_e6-1, task_label="tk5b", object_label="T")
    T = np.array(out["value_sample"]); zz = zs[:len(T)]
    e = [abs(t - compute_analytic_T(k, 1/3, 1/(1+z)))/env(k*cs/(1+z)) for t, z in zip(T, zz)]
    print(f"exact init, (atol,rtol)=({atol:.0e},{rtol:.0e}): nfev={out['data'].RHS_evaluations:6d}, max|dT|/env = {max(e):.2e}; |T| at stop = {abs(out['stop_value']):.2e}")
