"""Stage-2 Q formulation: error at accepted step endpoints vs at dense-output (t_eval) points."""
import sys, numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from scipy.integrate import solve_ivp
def q_solve(omega, zi, rtol, atol, u_end):
    om_i = omega(zi)
    def rhs(u, y):
        Q = y[0]; z = zi - u
        return [-omega(z)/om_i/(1+u) - Q/(1+u)]
    sol = solve_ivp(rhs, (0.0, u_end), [0.0], method="DOP853", dense_output=True, rtol=rtol, atol=atol)
    return sol, om_i
def theta_solve(omega, zi, rtol, atol, u_end):
    def rhs(u, y): return [-omega(zi - u)]
    return solve_ivp(rhs, (0.0, u_end), [0.0], method="DOP853", dense_output=True, rtol=rtol, atol=atol)
for label, omega, exact in [
    ("constant omega=1e7", lambda z: 1e7, lambda zi, u: -mp.mpf(1e7)*mp.mpf(u)),
    ("radiation k=1e7,  z_i=99", lambda z: 1e7/(1+z)**2, lambda zi, u: mp.mpf(1e7)*(1/(1+mp.mpf(zi)) - 1/(1+mp.mpf(zi)-mp.mpf(u)))),
]:
    zi = 99.0; u_end = 98.9
    for rtol, atol in [(1e-8, 1e-10), (1e-12, 1e-14)]:
        sol, om_i = q_solve(omega, zi, rtol, atol, u_end)
        ends = np.array([float(om_i*(1+u)*Q - exact(zi, u)) for u, Q in zip(sol.t, sol.y[0])])
        mids = 0.5*(sol.t[:-1] + sol.t[1:]); Qm = sol.sol(mids)[0]
        mid = np.array([float(om_i*(1+u)*Q - exact(zi, u)) for u, Q in zip(mids, Qm)])
        # fine dense grid
        uf = np.linspace(0.0, u_end, 2000)[1:]; Qf = sol.sol(uf)[0]
        fine = np.array([float(om_i*(1+u)*Q - exact(zi, u)) for u, Q in zip(uf, Qf)])
        st = theta_solve(omega, zi, rtol, atol, u_end)
        tf = st.sol(uf)[0]; tfine = np.array([float(t - exact(zi, u)) for u, t in zip(uf, tf)])
        print(f"{label:>26} rtol={rtol:.0e}: Q-path steps={len(sol.t)-1:4d} nfev={sol.nfev:5d} | max|err| at step ends={np.max(np.abs(ends)):.2e}, at step midpoints={np.max(np.abs(mid)):.2e}, on fine grid={np.max(np.abs(fine)):.2e} | direct theta: steps={len(st.t)-1:4d} nfev={st.nfev:5d} fine-grid max|err|={np.max(np.abs(tfine)):.2e}")
