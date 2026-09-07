import os
"""QI_02: numerical check of ComputeTargets/QuadSourceIntegral.py::analytic_integral (b=0)
against a direct scipy.quad of the spec 04 R14 integrand.

analytic_integral is called with stand-in model/wavenumber/redshift objects (it only uses
model.functions.tau(z), k.k, q.k, r.k and the *_inv_Mpc/store_id fields for log labels), and with
real bessel_phase() splines for nu = 0.5 and nu = 2.5, exactly as main.py builds them.
"""

import importlib
import sys
from math import pi, sqrt, gamma

import numpy as np
from scipy.integrate import quad
from scipy.special import jv, yv

sys.path.insert(0, os.getcwd())  # run from the repository root

QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")
from LiouvilleGreen.bessel_phase import bessel_phase


class FakeK:
    def __init__(self, kval):
        self.k = kval
        self.k_inv_Mpc = kval
        self.store_id = 0


class FakeZ:
    def __init__(self, z):
        self.z = z
        self.store_id = 0


class FakeFunctions:
    """tau(z) for a radiation era: a0*eta = A/(1+z).  z -> eta monotone decreasing."""

    def __init__(self, A):
        self.A = A

    def tau(self, z):
        return self.A / (1.0 + z)


class FakeModel:
    def __init__(self, A):
        self.functions = FakeFunctions(A)


def run(kv, qv, rv, eta_init, eta_resp, b=0.0):
    A = 1.0
    model = FakeModel(A)
    # z such that tau = A/(1+z) = eta   =>  z = A/eta - 1
    z_max = FakeZ(A / eta_init - 1.0)  # earliest time  (largest z, smallest eta)
    z_resp = FakeZ(A / eta_resp - 1.0)

    xmax = 1.10 * max(kv, qv, rv) * eta_resp
    B05 = bessel_phase(0.5 + b, xmax, atol=1e-25, rtol=5e-14)
    B25 = bessel_phase(2.5 + b, xmax, atol=1e-25, rtol=5e-14)

    out = QSI.analytic_integral(
        model,
        FakeK(kv),
        FakeK(qv),
        FakeK(rv),
        z_response=z_resp,
        max_z=z_max,
        min_z=z_resp,
        b=b,
        Bessel_0pt5=B05,
        Bessel_2pt5=B25,
        rtol=1e-10,
        atol=1e-25,
    )

    # ---- independent implementation of spec 04 R14 (a_0 absorbed) -------------------------
    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    A_coeff = (2.0 + b) / (1.0 + b)

    def src(e):
        return jv(0.5 + b, qv * cs * e) * jv(0.5 + b, rv * cs * e) + A_coeff * jv(
            2.5 + b, qv * cs * e
        ) * jv(2.5 + b, rv * cs * e)

    IJ = quad(
        lambda e: e ** (0.5 - b) * jv(0.5 + b, kv * e) * src(e),
        eta_init,
        eta_resp,
        limit=800,
        epsabs=1e-25,
        epsrel=1e-12,
    )[0]
    IY = quad(
        lambda e: e ** (0.5 - b) * yv(0.5 + b, kv * e) * src(e),
        eta_init,
        eta_resp,
        limit=800,
        epsabs=1e-25,
        epsrel=1e-12,
    )[0]

    pref = (
        -(pi / 2.0)
        * 2.0 ** (3.0 + 2.0 * b)
        / ((3.0 + 2.0 * b) * (2.0 + b))
        * gamma(2.5 + b) ** 2
        * (qv * rv * cs * cs * eta_resp) ** (-0.5 - b)
    )
    spec = pref * (
        yv(0.5 + b, kv * eta_resp) * IJ - jv(0.5 + b, kv * eta_resp) * IY
    )
    return out["value"], spec, IJ, IY


if __name__ == "__main__":
    cases = [
        # (k, q, r, eta_init, eta_response) -- eta_init < eta_response
        (1.0, 1.3, 0.8, 1e-3, 5.0),
        (10.0, 7.0, 5.0, 1e-3, 3.0),
        (50.0, 30.0, 25.0, 1e-3, 2.0),
        (200.0, 120.0, 90.0, 1e-3, 1.0),
    ]
    print(f"{'k':>7} {'q':>7} {'r':>7} {'code':>16} {'spec R14':>16} {'rel diff':>11}")
    for (kv, qv, rv, e0, e1) in cases:
        code, spec, IJ, IY = run(kv, qv, rv, e0, e1)
        rel = abs(code - spec) / abs(spec)
        print(f"{kv:7.4g} {qv:7.4g} {rv:7.4g} {code:16.9e} {spec:16.9e} {rel:11.3e}")
