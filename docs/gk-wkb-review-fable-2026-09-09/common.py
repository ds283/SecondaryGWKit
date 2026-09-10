import numpy as np, mpmath as mp
from types import SimpleNamespace as NS
from CosmologyConcepts import redshift, redshift_array
mp.mp.dps = 40
class RadModel:  # exact radiation: H=(1+z)^2, eps=2 -> omega=k/(1+z)^2, theta=k(1/(1+zi)-1/(1+z))
    def __init__(self):
        self.functions = NS(Hubble=lambda z: (1.0+z)**2, epsilon=lambda z: 2.0,
                            d_epsilon_dz=lambda z: 0.0, d2_epsilon_dz2=lambda z: 0.0)
def key(k): return NS(k=NS(k=k, k_inv_Mpc=k, store_id=0))
def grid(z_hi, z_lo, per_decade):
    n = int(np.ceil(np.log10((1+z_hi)/(1+z_lo))*per_decade))+1
    return np.geomspace(1+z_hi, 1+z_lo, n) - 1
def zarr(zs): return redshift_array([redshift(i, float(z)) for i, z in enumerate(zs)])
def exact_theta(k, zi, z): return mp.mpf(k)*(1/(1+mp.mpf(zi)) - 1/(1+mp.mpf(float(z))))
def phase_err(div, mod, k, zi, zs):
    return np.array([float(mp.mpf(int(d))*2*mp.pi + mp.mpf(float(m)) - exact_theta(k, zi, z)) for d, m, z in zip(div, mod, zs)])
