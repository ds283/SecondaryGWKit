import mpmath as mp
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units
from ComputeTargets.BackgroundModel import ModelFunctions
mp.mp.dps = 40
cosmo = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
class M: pass
model = M()
model.functions = ModelFunctions(
    Hubble=cosmo.Hubble,
    epsilon=lambda z: (1.0 + z) * cosmo.d_lnH_dz(z),
    d_epsilon_dz=lambda z: cosmo.d_lnH_dz(z) + (1.0 + z) * cosmo.d2_lnH_dz2(z),
    d2_epsilon_dz2=lambda z: 2.0 * cosmo.d2_lnH_dz2(z) + (1.0 + z) * cosmo.d3_lnH_dz3(z),
    wBackground=cosmo.wBackground, wPerturbations=cosmo.wPerturbations, tau=None,
    T_photon=cosmo.T_photon, d_lnH_dz=cosmo.d_lnH_dz, d2_lnH_dz2=cosmo.d2_lnH_dz2,
    d3_lnH_dz3=cosmo.d3_lnH_dz3, d_wPerturbations_dz=cosmo.d_wPerturbations_dz,
    d2_wPerturbations_dz2=cosmo.d2_wPerturbations_dz2)
_rm, _rr, _rc, _M2 = mp.mpf(cosmo.rho_m0), mp.mpf(cosmo.rho_r0), mp.mpf(cosmo.rho_cc), mp.mpf(cosmo.Mpsq)
def H_mp(z):
    s = 1 + mp.mpf(z)
    return mp.sqrt((_rm*s**3 + _rr*s**4 + _rc)/(3*_M2))
def tau_increment_mp(z_lo, z_hi):
    """int_{z_lo}^{z_hi} dz/H  via u=log(1+z), mpmath"""
    f = lambda u: mp.exp(u)/H_mp(mp.exp(u)-1)
    return mp.quad(f, [mp.log(1+mp.mpf(z_lo)), mp.log(1+mp.mpf(z_hi))])
