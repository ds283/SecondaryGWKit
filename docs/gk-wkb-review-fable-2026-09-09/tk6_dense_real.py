"""Is the low-z jump in the T_k stage-2 phase error dense-output amplification? Re-solve the stage-2 Q ODE on LambdaCDM,
compare error at accepted step endpoints vs at the production 100/decade grid points (dense output)."""
import sys, numpy as np, mpmath as mp
from math import log, exp, sqrt
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import cosmo, model, H_mp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
k = 1e5; H = cosmo.Hubble
Om, Or = mp.mpf(cosmo.omega_m), mp.mpf(cosmo.omega_r)
w_mp = lambda z: (lambda s: (Or*s/3)/(Om + Or*s))(1+mp.mpf(z))
zi = brentq(lambda z: Tk_omegaEff_sq(model, k, z) - 1e6, 0.1, 1e9)   # stage-2 start: omega^2 = 1e6
om = lambda z: sqrt(Tk_omegaEff_sq(model, k, z)); om_i = om(zi)
def rhs(u, y): z = zi - u; return [-om(z)/om_i/(1+u) - y[0]/(1+u)]
sol = solve_ivp(rhs, (0.0, zi - 0.1), [0.0], method="DOP853", dense_output=True, rtol=1e-8, atol=1e-10)
def ref(z):
    lead = mp.quad(lambda v: mp.mpf(k)*mp.sqrt(w_mp(mp.exp(v)-1))/H_mp(mp.exp(v)-1)*mp.exp(v), [mp.log(1+mp.mpf(z)), mp.log(1+mp.mpf(zi))])
    def f(v):
        zz = exp(v)-1; kcH = k*sqrt(cosmo.wPerturbations(zz))/H(zz); o2 = Tk_omegaEff_sq(model, k, zz)
        return (o2-kcH*kcH)/(sqrt(o2)+kcH)*exp(v)
    return -(lead + mp.mpf(quad(f, log(1+z), log(1+zi), limit=800, epsrel=1e-10, epsabs=1e-14)[0]))
print(f"k={k:.0e}: stage-2 start z={zi:.4g}, {len(sol.t)-1} accepted steps, nfev={sol.nfev}")
print(f"{'u':>10} {'z':>10} {'kind':>9} {'phase err [rad]':>16}")
# step endpoints in the last decade of z, and grid points (100/dec) in between
ends = [(u, zi-u) for u in sol.t if zi-u < 3.0]
for u, z in ends:
    e = float(mp.mpf(om_i*(1+u))*mp.mpf(float(sol.sol(u)[0])) - ref(z)); print(f"{u:10.6f} {z:10.4g} {'step end':>9} {e:16.3e}")
zg = np.geomspace(1.1, 3.0, 5) - 1
for z in zg:
    u = zi - z; e = float(mp.mpf(om_i*(1+u))*mp.mpf(float(sol.sol(u)[0])) - ref(z)); print(f"{u:10.6f} {z:10.4g} {'grid pt':>9} {e:16.3e}")
# same but direct theta integration for comparison at the grid points
st = solve_ivp(lambda u, y: [-om(zi-u)], (0.0, zi-0.1), [0.0], method="DOP853", dense_output=True, rtol=1e-8, atol=1e-10)
for z in zg[[0, -1]]:
    u = zi - z; e = float(mp.mpf(float(st.sol(u)[0])) - ref(z)); print(f"{u:10.6f} {z:10.4g} {'theta-ODE':>9} {e:16.3e}   (direct theta, dense output; nfev={st.nfev})")
