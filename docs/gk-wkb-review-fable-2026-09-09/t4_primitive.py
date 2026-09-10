"""Shared tau-primitive construction on the real LambdaCDM background, vs mpmath."""
import sys, time, math, numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import *
from scipy.interpolate import make_interp_spline
H = cosmo.Hubble
# production-like grid: 100/decade in 1+z from z_e3(k=1e5)=2.31e9 down to 0.1
z_hi, z_lo, per_decade = 2.31e9, 0.1, 100
n = int(np.ceil(np.log10((1+z_hi)/(1+z_lo))*per_decade))+1
u_nodes = np.linspace(np.log(1+z_hi), np.log(1+z_lo), n)  # decreasing z
def gl(order):
    x, w = np.polynomial.legendre.leggauss(order); return x, w
def increments(order):
    x, w = gl(order); inc = np.empty(n-1)
    for i in range(n-1):
        a, b = u_nodes[i], u_nodes[i+1]; um = 0.5*(a+b)*np.ones(order) + 0.5*(b-a)*x
        inc[i] = 0.5*(b-a)*np.sum(w*np.array([np.exp(u)/H(np.exp(u)-1) for u in um]))
    return inc  # negative (z decreasing): tau increases as z decreases? dtau = dz/H, z decreasing -> tau decreasing; sign irrelevant
# reference at a handful of nodes: cumulative from z_hi
check = [1, 5, 50, 200, 400, 600, 800, 900, n-1]
ref = {}
t0=time.perf_counter()
for j in check:
    ref[j] = tau_increment_mp(np.exp(u_nodes[j])-1, z_hi)
print(f"mpmath reference for {len(check)} nodes: {time.perf_counter()-t0:.1f}s")
for order in [4, 8, 12]:
    t0=time.perf_counter(); inc = increments(order); dt=time.perf_counter()-t0
    cum = [math.fsum(inc[:j]) for j in check]  # tau(z_j) - tau(z_hi) (sign: negative of int dz/H over decreasing z)
    rel = max(abs(float((-c - r)/r)) for c, r in zip(cum, ref.values()))
    tau_end = -cum[-1]
    print(f"Gauss-Legendre order {order:2d}: {order*(n-1):6d} H-evals, {dt:.2f}s, max relative error of cumulative tau at checked nodes = {rel:.2e}  -> x k*tau at z=0.1, k=3e8: {3e8*float(tau_end):.4g} rad, abs err {rel*3e8*float(tau_end):.2e} rad")
# off-grid: anchor + local Gauss(8) vs spline of nodes
inc = increments(12); tau_nodes = np.array([0.0] + [ -math.fsum(inc[:j]) for j in range(1, n)])  # tau relative to z_hi (positive, increasing toward low z)
rng = np.random.default_rng(1); picks = rng.integers(3, n-4, 25); fr = rng.uniform(0.05, 0.95, 25)
x8, w8 = gl(8)
errs_local, errs_cub, errs_quin = [], [], []
cub = make_interp_spline(u_nodes[::-1], tau_nodes[::-1], k=3); quin = make_interp_spline(u_nodes[::-1], tau_nodes[::-1], k=5)
for j, f in zip(picks, fr):
    a = u_nodes[j]; b = u_nodes[j+1]; u = a + f*(b-a)
    um = 0.5*(a+u) + 0.5*(u-a)*x8
    loc = tau_nodes[j] - 0.5*(u-a)*np.sum(w8*np.array([np.exp(v)/H(np.exp(v)-1) for v in um]))
    r = float(tau_increment_mp(np.exp(u)-1, z_hi))
    errs_local.append(abs(loc - r)/r); errs_cub.append(abs(float(cub(u)) - r)/r); errs_quin.append(abs(float(quin(u)) - r)/r)
print(f"off-grid tau, 25 random points: anchor+Gauss8 max rel err {max(errs_local):.1e}; cubic spline of nodes {max(errs_cub):.1e}; quintic spline {max(errs_quin):.1e}")
print(f"  as phase error at k=1e5 (theta~1.4e9): {max(errs_local)*1.4e9:.1e} / {max(errs_cub)*1.4e9:.1e} / {max(errs_quin)*1.4e9:.1e} rad ; at k=3e8 (theta~4e12): {max(errs_local)*4e12:.1e} / {max(errs_cub)*4e12:.1e} / {max(errs_quin)*4e12:.1e} rad")
