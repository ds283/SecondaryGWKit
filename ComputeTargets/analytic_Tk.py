from math import sqrt, pow
from scipy.special import gamma, jv


def compute_analytic_T(k, w, tau):
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    cs = sqrt(w)
    k_cs_tau = k * cs * tau

    n1 = 1.5 + b
    A = pow(2.0, n1)
    B = gamma(2.5 + b)
    C = pow(k_cs_tau, -n1)
    D = jv(n1, k_cs_tau)

    return A * B * C * D


def compute_analytic_Tprime(k, w, tau, H):
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    cs = sqrt(w)
    k_cs_tau = k * cs * tau

    n0 = 0.5 + b
    n1 = 1.5 + b
    n2 = 2.5 + b

    A = pow(2.0, n0) * cs * k
    B = gamma(2.5 + b)
    C = pow(k_cs_tau, -n2)
    D1 = k_cs_tau * jv(n0, k_cs_tau)
    D2 = -(3.0 + 2.0 * b) * jv(n1, k_cs_tau)
    D3 = -k_cs_tau * jv(n2, k_cs_tau)
    D = D1 + D2 + D3

    return -(1.0 / H) * A * B * C * D
