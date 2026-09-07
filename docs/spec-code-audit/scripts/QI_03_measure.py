import os
"""QI_03: measure / Jacobian check on the three numerical branches of
ComputeTargets/QuadSourceIntegral.py.

Feeds numeric_quad_integral / WKB_quad_integral / WKB_Levin_integral synthetic smooth
G(z'), f(z'), H(z') through stand-in GkPolicy / source / model objects and compares the
returned value with an independent scipy.quad of the spec 03 R28 / spec 04 R1 measure

    (1+z) * int_{z_response}^{z_source_max} dz'  G(z,z') * f(z') / ( (1+z') H(z')^2 )

i.e. int dz' G * (1+z)/(1+z') * f / H^2.  Also reported: the value of the same integral
*without* the 1/(1+z') factor, to show which one the code implements.
"""

import importlib
import sys
from math import exp, log, sin, cos, sqrt, pi

from scipy.integrate import quad

sys.path.insert(0, os.getcwd())  # run from the repository root
QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")
from ComputeTargets.GkSourcePolicyData import GkSourceFunctions
from ComputeTargets.QuadSource import QuadSourceFunctions

Z_RESPONSE = 100.0
Z_MAX = 5000.0


def G_of_z(z):
    return 0.37 * (1.0 + z) ** -1.3 * (1.0 - (1.0 + Z_RESPONSE) / (1.0 + z))


def f_of_z(z):
    return 1.7 + 0.4 * log(1.0 + z)


def H_of_z(z):
    return 0.13 * (1.0 + z) ** 2


# --- stand-ins ----------------------------------------------------------------------------
class Spline:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, z, z_is_log=False):
        zz = exp(z) - 1.0 if z_is_log else z
        return self.fn(zz)


class FakeK:
    def __init__(self, kv):
        self.k = kv
        self.k_inv_Mpc = kv
        self.store_id = 0


class FakeZ:
    def __init__(self, z):
        self.z = z
        self.store_id = 0


class FakeModelFunctions:
    def Hubble(self, z):
        return H_of_z(z)

    def tau(self, z):
        return 1.0 / (1.0 + z)


class FakeModel:
    functions = FakeModelFunctions()


class FakeSource:
    store_id = 0

    def __init__(self):
        self.functions = QuadSourceFunctions(source=Spline(f_of_z), **_zero_extras())


def _zero_extras():
    # fill any other fields of QuadSourceFunctions with None
    return {k: None for k in QuadSourceFunctions._fields if k != "source"}


class FakePolicy:
    store_id = 0
    quality = "good"

    def __init__(self, typ, **kw):
        self.type = typ
        base = dict(
            numeric_region=None,
            WKB_region=None,
            numeric_Gk=None,
            WKB_Gk=None,
            phase=None,
            sin_amplitude=None,
            type=typ,
            quality="good",
            crossover_z=None,
        )
        base.update(kw)
        self.functions = GkSourceFunctions(**base)
        self.crossover_z = base["crossover_z"]
        self.Levin_z = None


def reference():
    """spec 03 R28 measure, evaluated independently."""
    with_jac = quad(
        lambda zp: G_of_z(zp) * (1.0 + Z_RESPONSE) / (1.0 + zp) * f_of_z(zp) / H_of_z(zp) ** 2,
        Z_RESPONSE,
        Z_MAX,
        limit=400,
        epsabs=1e-30,
        epsrel=1e-13,
    )[0]
    without_jac = quad(
        lambda zp: G_of_z(zp) * (1.0 + Z_RESPONSE) * f_of_z(zp) / H_of_z(zp) ** 2,
        Z_RESPONSE,
        Z_MAX,
        limit=400,
        epsabs=1e-30,
        epsrel=1e-13,
    )[0]
    no_Hsq = quad(
        lambda zp: G_of_z(zp) * (1.0 + Z_RESPONSE) / (1.0 + zp) * f_of_z(zp),
        Z_RESPONSE,
        Z_MAX,
        limit=400,
        epsabs=1e-30,
        epsrel=1e-13,
    )[0]
    return with_jac, without_jac, no_Hsq


def main():
    src = FakeSource()
    model = FakeModel()
    kk = FakeK(1.0)
    zr = FakeZ(Z_RESPONSE)

    with_jac, without_jac, no_Hsq = reference()
    print(f"reference  int dz' G (1+z)/(1+z') f / H^2 = {with_jac:.12e}")
    print(f"           int dz' G (1+z)        f / H^2 = {without_jac:.12e}")
    print(f"           int dz' G (1+z)/(1+z') f       = {no_Hsq:.12e}")

    # --- numeric branch
    pol = FakePolicy(
        "numeric",
        numeric_Gk=Spline(G_of_z),
        numeric_region=(Z_MAX, Z_RESPONSE),
    )
    out = QSI.numeric_quad_integral(
        model, kk, kk, kk, src, pol, zr, max_z=Z_MAX, min_z=Z_RESPONSE,
        atol=1e-30, rtol=1e-12,
    )
    print(f"\nnumeric_quad_integral  = {out['value']:.12e}   rel vs with_jac = "
          f"{abs(out['value'] - with_jac) / abs(with_jac):.3e}")
    print(f"                             rel vs without_jac = "
          f"{abs(out['value'] - without_jac) / abs(without_jac):.3e}")

    # --- WKB quad branch (same integrand, different G spline slot)
    pol2 = FakePolicy("WKB", WKB_Gk=Spline(G_of_z), WKB_region=(Z_MAX, Z_RESPONSE))
    out2 = QSI.WKB_quad_integral(
        model, kk, kk, kk, src, pol2, zr, max_z=Z_MAX, min_z=Z_RESPONSE,
        atol=1e-30, rtol=1e-12,
    )
    print(f"WKB_quad_integral      = {out2['value']:.12e}   rel vs numeric branch = "
          f"{abs(out2['value'] - out['value']) / abs(out['value']):.3e}")

    # --- WKB Levin branch: G = sin_amplitude(z) * sin(theta(z)).  Build a synthetic phase
    #     spline-like object with an analytic theta so the two can be compared exactly.
    AMPL = 0.37

    class Phase:
        def raw_theta(self, x, x_is_log=False):
            z = exp(x) - 1.0 if x_is_log else x
            return 400.0 * log(1.0 + z)

        def theta_mod_2pi(self, x, x_is_log=False):
            return self.raw_theta(x, x_is_log) % (2.0 * pi)

        def theta_deriv(self, x, x_is_log=False, log_derivative=False):
            return 400.0

    def ampl(z):
        return AMPL * (1.0 + z) ** -1.3

    def G_osc(z):
        return ampl(z) * sin(400.0 * log(1.0 + z))

    pol3 = FakePolicy(
        "WKB",
        sin_amplitude=Spline(ampl),
        phase=Phase(),
        WKB_region=(Z_MAX, Z_RESPONSE),
    )
    out3 = QSI.WKB_Levin_integral(
        model, kk, kk, kk, src, pol3, zr, max_z=Z_MAX, min_z=Z_RESPONSE,
        atol=1e-30, rtol=1e-12,
    )
    ref3 = quad(
        lambda zp: G_osc(zp) * (1.0 + Z_RESPONSE) / (1.0 + zp) * f_of_z(zp) / H_of_z(zp) ** 2,
        Z_RESPONSE,
        Z_MAX,
        limit=4000,
        epsabs=1e-30,
        epsrel=1e-13,
    )[0]
    print(f"\nWKB_Levin_integral     = {out3['value']:.12e}")
    print(f"reference (oscillatory G, same R28 measure) = {ref3:.12e}")
    print(f"   rel diff = {abs(out3['value'] - ref3) / abs(ref3):.3e}")

    # region-boundary consistency: same integrand from the quad and Levin routes
    pol4 = FakePolicy("WKB", WKB_Gk=Spline(G_osc), WKB_region=(Z_MAX, Z_RESPONSE))
    out4 = QSI.WKB_quad_integral(
        model, kk, kk, kk, src, pol4, zr, max_z=Z_MAX, min_z=Z_RESPONSE,
        atol=1e-30, rtol=1e-12,
    )
    print(f"WKB_quad with same osc G = {out4['value']:.12e}   rel vs Levin = "
          f"{abs(out4['value'] - out3['value']) / abs(out3['value']):.3e}")


if __name__ == "__main__":
    main()
