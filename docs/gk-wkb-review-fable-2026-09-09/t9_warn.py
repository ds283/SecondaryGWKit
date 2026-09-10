import sys, warnings; sys.path.insert(0, __file__.rsplit('/',1)[0])
warnings.simplefilter("error", DeprecationWarning)
from common import *
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
import Quadrature.integrators.WKB_phase_function as W
k=1.1e6; zi=k/30-1; zs=grid((1+zi)/10**0.12-1, 0.1, 100/12); meta={}
try:
    W.integrate_phase_function(RadModel(), key(k), zi, zarr(zs), om2, dlom, om2(RadModel(),k,zi), 1e-10, 1e-8, meta, "t9", "G")
    print("no DeprecationWarning raised; resets =", meta.get("phase_cycle_events"))
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:200])
