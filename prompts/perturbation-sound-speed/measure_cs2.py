"""
Measurements behind AUDIT.md: the adiabatic sound speed of the QCD plasma against the p/rho that
`LambdaCDM_GenericEOS.wPerturbations` feeds to the transfer-function sector.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python prompts/perturbation-sound-speed/measure_cs2.py

Needs neither Ray nor a datastore. Everything in AUDIT.md's tables comes from this script.

Definitions. For a plasma in thermal equilibrium at zero chemical potential, p = T s - rho and
d rho = T ds, so dp/dT = s and d rho/dT = T ds/dT. With s = (2 pi^2/45) g_s T^3,

    c_s^2 = dp/d rho = s / (T ds/dT) = 1 / (3 + d ln g_s / d ln T),
    w     = p / rho  = 4 g_s / (3 g_rho) - 1.

The code's `QCD_EOS.w` is the second line; nothing in the tree computes the first.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from CosmologyModels.LambdaCDM import Planck2018
from Units.Mpc_units import Mpc_units

UNITS = Mpc_units()
GeV = UNITS.GeV
eos = QCD_EOS(UNITS)


def w_of_T(T_GeV: float) -> float:
    return eos.w(T_GeV * GeV)


def dlnGs_dlnT(T_GeV: float, h: float = 1e-5) -> float:
    return (
        np.log(eos.Gs(T_GeV * np.exp(h) * GeV))
        - np.log(eos.Gs(T_GeV * np.exp(-h) * GeV))
    ) / (2 * h)


def cs2_of_T(T_GeV: float, h: float = 1e-5) -> float:
    return 1.0 / (3.0 + dlnGs_dlnT(T_GeV, h))


# ---------------------------------------------------------------------------------------------
# 1. w against c_s^2 across the Standard Model thermal history
# ---------------------------------------------------------------------------------------------
print(
    "## 1. w = p/rho against c_s^2 = dp/drho, from the code's own g_rho and g_s fits\n"
)
print(f"| T [GeV] | w | c_s^2 | c_s^2 - w | (c_s^2 - w)/c_s^2 |\n|---|---|---|---|---|")
for T in (
    3e-3,
    1e-2,
    3e-2,
    0.06,
    0.09,
    0.11,
    0.13,
    0.14,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.5,
    1.0,
    3.0,
    30.0,
    100.0,
    300.0,
):
    w, c = w_of_T(T), cs2_of_T(T)
    print(f"| {T:g} | {w:.4f} | {c:.4f} | {c - w:+.4f} | {(c - w) / c:+.3f} |")

Ts = np.geomspace(3e-3, 300.0, 6000)
ws = np.array([w_of_T(T) for T in Ts])
cs = np.array([cs2_of_T(T) for T in Ts])
i, j, m = np.argmin(ws), np.argmin(cs), np.argmax(np.abs(cs - ws))
print(f"\nmin w      = {ws[i]:.4f} at T = {Ts[i]:.4f} GeV")
print(f"min c_s^2  = {cs[j]:.4f} at T = {Ts[j]:.4f} GeV")
print(
    f"max |c_s^2 - w| = {abs(cs - ws)[m]:.4f} at T = {Ts[m]:.4f} GeV, i.e. {abs(cs - ws)[m] / cs[m]:.1%} of c_s^2"
)
print(
    f"max (c_s - c_s,w)/c_s (sound-speed, not squared) = {np.max(np.abs(np.sqrt(cs) - np.sqrt(ws)) / np.sqrt(cs)):.1%}"
)

# ---------------------------------------------------------------------------------------------
# 2. Behaviour of the derivative at the fit's branch joins
# ---------------------------------------------------------------------------------------------
print("\n## 2. c_s^2 at the parametrisation's branch joins (one-sided derivatives)\n")
print(
    "| join | T [GeV] | c_s^2 just below | c_s^2 just above | w there | centred difference straddling the join |\n|---|---|---|---|---|---|"
)
for name, T0 in (("EOS_T_LO", QCD_EOS.EOS_T_LO), ("T_120_MEV", QCD_EOS.T_120_MEV)):
    d = 1e-4
    below = cs2_of_T(T0 * (1 - 3 * d), h=d)
    above = cs2_of_T(T0 * (1 + 3 * d), h=d)
    straddle = cs2_of_T(T0, h=d)
    print(
        f"| {name} | {T0:g} | {below:.4f} | {above:.4f} | {w_of_T(T0):.4f} | {straddle:.4f} |"
    )

# ---------------------------------------------------------------------------------------------
# 3. Sound-phase error on the production background
# ---------------------------------------------------------------------------------------------
print("\n## 3. Accumulated sound-phase error after horizon entry, on QCD_Cosmology\n")
cosmo = QCD_Cosmology(store_id=0, units=UNITS, params=Planck2018(), max_z=1e20)


def T_GeV_of_z(z: float) -> float:
    return cosmo._rho_fluid(z)["T"] / GeV


def dc_s(z: float) -> float:
    """c_s(adiabatic) - c_s(p/rho) for the plasma at redshift z; the matter admixture is ignored (it is below 1e-9 here)."""
    T = T_GeV_of_z(z)
    return np.sqrt(cs2_of_T(T)) - np.sqrt(w_of_T(T))


def z_entry(k: float) -> float:
    # horizon entry: k (the code's k/a_0) = H(z)/(1+z)
    return np.expm1(
        brentq(
            lambda lz: np.log(cosmo.Hubble(np.expm1(lz)) / np.exp(lz)) - np.log(k),
            np.log(1e5),
            np.log(1e18),
        )
    )


def z_of_T(T_GeV: float) -> float:
    return np.expm1(
        brentq(
            lambda lz: np.log(T_GeV_of_z(np.expm1(lz))) - np.log(T_GeV),
            np.log(1e7),
            np.log(1e16),
        )
    )


def dc_s_clamped(z: float) -> float:
    """As dc_s, but with c_s^2 clamped below EOS_T_LO exactly as QCD_EOS.w clamps w: the equilibrium
    formula is invalid once the neutrinos have decoupled, for c_s^2 as much as for w."""
    T = max(T_GeV_of_z(z), QCD_EOS.EOS_T_LO)
    return np.sqrt(cs2_of_T(T)) - np.sqrt(w_of_T(T))


z_hi = z_of_T(10.0)
for label, T_lo in (
    ("everything above the clamp, T >= EOS_T_LO", QCD_EOS.EOS_T_LO),
    ("the QCD crossover alone, T >= 0.02 GeV", 0.02),
):
    z_lo = z_of_T(T_lo)
    print(
        f"\n### Window: {label}; z in [{z_lo:.3g}, {z_hi:.3g}], T in [{T_lo:g}, 10] GeV\n"
    )
    print(
        f"| k [1/Mpc] | T at entry [GeV] | sound phase from entry to window bottom [rad] | phase error from c_s^2 = w [rad] |\n|---|---|---|---|"
    )
    for k in (1e5, 3e5, 1e6, 2e6, 5e6, 1e7, 3e7, 1e8, 3e8):
        ze = z_entry(k)
        top = min(ze, z_hi)
        if top <= z_lo:
            print(f"| {k:g} | {T_GeV_of_z(ze):.3g} | (enters below the window) | 0 |")
            continue
        pts = [
            z for z in (4.25e7, 8.64e11) if z_lo < z < top
        ]  # H(z) jumps (QCD_EOS.discontinuity_temperatures_GeV)
        dtheta, _ = quad(
            lambda z: dc_s_clamped(z) * k / cosmo.Hubble(z),
            z_lo,
            top,
            points=pts or None,
            limit=400,
        )
        theta, _ = quad(
            lambda z: np.sqrt(w_of_T(max(T_GeV_of_z(z), QCD_EOS.EOS_T_LO)))
            * k
            / cosmo.Hubble(z),
            z_lo,
            top,
            points=pts or None,
            limit=400,
        )
        print(f"| {k:g} | {T_GeV_of_z(ze):.3g} | {theta:.3g} | {dtheta:+.3g} |")

# ---------------------------------------------------------------------------------------------
# 4. Size of the mass term 3 H^2 (c_s^2 - w) against c_s^2 k^2
# ---------------------------------------------------------------------------------------------
print(
    "\n## 4. The term 3 H^2 (c_s^2 - w) that the closure sets to zero, relative to c_s^2 k^2\n"
)
Td = Ts[m]
print(
    f"at the largest gap (T = {Td:.3f} GeV): 3 (c_s^2 - w)/c_s^2 = {3 * (cs[m] - ws[m]) / cs[m]:+.3f}, times (aH/k)^2:"
)
for x in (1.0, 3.0, 10.0, 30.0):
    print(f"    k/(aH) = {x:5.1f}: {3 * (cs[m] - ws[m]) / cs[m] / x**2:+.2e}")
