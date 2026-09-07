import os
"""QI_05: is the f used by the numerical branches (QuadSource.source_function) the same f
whose fixed-w Bessel reduction the analytic branch implements (spec 04 R11 final form)?

If yes, `total` and `analytic_rad` in the QuadSourceIntegral payload are like-for-like and the
ratio to spec 05 R31 is 1/c^2 for BOTH.

Constant-w background, a0*eta = A/(1+z)^{...}: for a(eta) ~ eta^{1+b} and 1+z = a0/a we have
   1+z = (eta_0/eta)^{1+b},  and calH = (1+b)/eta.
Transfer function T from spec 04 R4, primes d/dz obtained analytically.
"""

import sys
from math import gamma, pi, sqrt

from scipy.special import jv

sys.path.insert(0, os.getcwd())  # run from the repository root
from ComputeTargets.QuadSource import source_function


def T_and_dTdz(kv, eta, b, cs, eta0):
    """T(eta) and dT/dz with 1+z = (eta0/eta)^(1+b)."""
    x = kv * cs * eta
    pref = 2.0 ** (1.5 + b) * gamma(2.5 + b)
    T = pref * x ** (-1.5 - b) * jv(1.5 + b, x)
    # dT/deta = k cs dT/dx = -pref x^{-3/2-b} k cs J_{5/2+b}(x)   (spec 04 R9)
    dT_deta = -pref * x ** (-1.5 - b) * kv * cs * jv(2.5 + b, x)
    # 1+z = (eta0/eta)^{1+b}  =>  d(1+z)/deta = -(1+b)(1+z)/eta
    # dT/dz = dT/deta * deta/dz = dT/deta * ( -eta / ((1+b)(1+z)) )
    one_plus_z = (eta0 / eta) ** (1.0 + b)
    dT_dz = dT_deta * (-eta / ((1.0 + b) * one_plus_z))
    return T, dT_dz, one_plus_z


def f_spec_R11(qv, rv, eta, b, cs):
    """spec 04 R11 final boxed form."""
    xq = qv * cs * eta
    xr = rv * cs * eta
    pref = (
        2.0 ** (3.0 + 2.0 * b)
        / ((3.0 + 2.0 * b) * (2.0 + b))
        * gamma(2.5 + b) ** 2
        * xq ** (-0.5 - b)
        * xr ** (-0.5 - b)
    )
    brace = jv(0.5 + b, xq) * jv(0.5 + b, xr) + (2.0 + b) / (1.0 + b) * jv(
        2.5 + b, xq
    ) * jv(2.5 + b, xr)
    return pref * brace


def f_spec_R28_main14(qv, rv, eta, b, cs):
    """spec 05 R28 (MAIN 14) boxed form."""
    xq = qv * cs * eta
    xr = rv * cs * eta
    pref = (
        (2.0 + b) / (3.0 + 2.0 * b) ** 3
        * 2.0 ** (3.0 + 2.0 * b)
        * gamma(2.5 + b) ** 2
        * xq ** (-0.5 - b)
        * xr ** (-0.5 - b)
    )
    brace = jv(0.5 + b, xq) * jv(0.5 + b, xr) + (2.0 + b) / (1.0 + b) * jv(
        2.5 + b, xq
    ) * jv(2.5 + b, xr)
    return pref * brace


if __name__ == "__main__":
    eta0 = 1.0e-3  # eta at which 1+z = 1, irrelevant to the ratio
    print(f"{'b':>5} {'w':>8} {'eta':>7} {'f_code':>15} {'f_R11':>15} {'reldiff':>10}"
          f" {'f_code/f_R28':>14} {'1/c^2':>10}")
    for b in (0.0, 0.2, -0.15):
        w = (1.0 - b) / (3.0 * (1.0 + b))
        cs = sqrt(w)  # cs^2 = w for constant w (script QI_01 item (d))
        c2 = ((2.0 + b) / (3.0 + 2.0 * b)) ** 2
        for eta in (0.4, 3.0):
            qv, rv = 7.0, 5.0
            Tq, dTq, opz = T_and_dTdz(qv, eta, b, cs, eta0)
            Tr, dTr, _ = T_and_dTdz(rv, eta, b, cs, eta0)
            z = opz - 1.0
            out = source_function(Tq, Tr, dTq, dTr, z, w)
            fc = out["source"]
            f11 = f_spec_R11(qv, rv, eta, b, cs)
            f28 = f_spec_R28_main14(qv, rv, eta, b, cs)
            print(
                f"{b:5.2f} {w:8.4f} {eta:7.3g} {fc:15.7e} {f11:15.7e}"
                f" {abs(fc-f11)/abs(f11):10.2e} {fc/f28:14.7f} {1.0/c2:10.7f}"
            )
