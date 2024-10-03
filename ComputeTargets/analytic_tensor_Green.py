from math import pi, sqrt
from scipy.special import jv, yv


def compute_analytic_G(k, w, tau_source, tau, H_source):
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source

    # The main numerical calculation obtains G_us(z,z') defined in DS' analytical calculation, which is
    # for the source -delta(z-z') expressed in redshift.
    # The analytic result is expressed in terms of conformal time, as G_them(eta, eta') with source
    # delta(eta-eta').
    # The delta function transforms with the (absolute value of) the Jacobian of the transformation.
    # This means we get G_us(z,z') = -H(z') G_them(eta, eta').
    # The multiplication of -H_source here accounts for this minus sign and the coordinate Jacobian
    n0 = 0.5 + b

    A = -H_source * pi / 2.0
    B = sqrt(tau * tau_source)
    C = jv(n0, k_tau_source) * yv(n0, k_tau)
    D = jv(n0, k_tau) * yv(n0, k_tau_source)

    return A * B * (C - D)


def compute_analytic_Gprime(k, w, tau_source, tau, H_source, H):
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source

    n0 = 0.5 + b
    n1 = 1.5 + b

    A = pi / 2.0 * (H_source / H)
    B = sqrt(tau_source / tau)
    C1 = k_tau * jv(n1, k_tau) - (b + 1.0) * jv(n0, k_tau)
    C2 = yv(n0, k_tau_source)
    D1 = (b + 1.0) * yv(n0, k_tau) - k_tau * yv(n1, k_tau)
    D2 = jv(n0, k_tau_source)

    return A * B * (C1 * C2 + D1 * D2)
