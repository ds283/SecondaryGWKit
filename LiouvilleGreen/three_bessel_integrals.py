import numpy as np
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from Quadrature.simple_quadrature import simple_quadrature

# The Levin rule integrates against a (sin, cos) basis carrying a single phase, so a product of three
# Bessel functions has to be decomposed into sum-and-difference phases first. Writing each Bessel factor
# in Liouville-Green form as (modulus) x (sine of phase), the triple product expands into four terms
# whose phases are theta_mu + e_nu*theta_nu + e_sigma*theta_sigma. These are the four sign pairs.
_PHASE_GROUP_SIGNS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

# Chebyshev spectral order used for the three-Bessel Levin evaluations. The cost of each Levin solve is
# dominated by evaluating the phase and modulus splines at the collocation points, so total cost scales
# roughly as (order) x (number of solves): a high order buys fewer subintervals but more than pays for
# it per solve.
#
# This was previously 64. Measured against the analytic oracles (J000, J231, Y022 at k,q,s = 1.3,1.7,2.1
# and max_x = 1e5), the relative error is *identical* at orders 12 through 64 -- it is set by the
# accuracy of the phase and modulus splines, not by the spectral order -- while the runtime rises by a
# factor of 3 to 6.6 over that range. So the order was costing several times the necessary work for no
# accuracy at all.
#
# Note the SVD-failure recovery in adaptive_levin_sincos() steps the order down in twos to a floor of 8,
# so 12 leaves it two steps of headroom.
DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12


def quad_JJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    min_x_mu = mu_phase["min_x"]
    min_x_nu = nu_phase["min_x"]
    min_x_sigma = sigma_phase["min_x"]

    # the integral from 0 up to max_x needs to be cut into two: numerical quadrature for low x,
    # where the Bessel phase representation is not reliable, and Levin integration
    # for the rest.
    # The first value of x where phase information is available for all three Bessel functions
    # min_cut = max(lowest x value) / min(k, q, s)
    x_cut = max(min_x_mu, min_x_nu, min_x_sigma)
    min_kqs = min(k, q, s)
    min_cut = x_cut / min_kqs

    numeric = _direct_JJJ(mu, nu, sigma, k, q, s, 0.0, min_cut, atol, rtol)
    Levin = _Levin_JJJ(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_cut,
        max_x,
        atol,
        rtol,
        chebyshev_order=chebyshev_order,
    )

    # print(f">> numeric on (0, {min_cut}) = {numeric}")
    # print(f">> Levin on ({min_cut}, {max_x}) = {Levin}")

    return numeric + Levin


def _direct_JJJ(
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
):
    def integrand(x):
        return (
            np.sqrt(x)
            * jv(mu + 0.5, k * x)
            * jv(nu + 0.5, q * x)
            * jv(sigma + 0.5, s * x)
        )

    data = simple_quadrature(
        integrand,
        a=min_x,
        b=max_x,
        atol=atol,
        rtol=rtol,
        label="numeric part",
        method="quad",
    )

    return np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) * data["value"]


def _phase_group(phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma):
    """
    Build the phase-function dictionary for one sum-and-difference group, in the form expected by
    adaptive_levin_sincos(). The integration variable is log(x).

    We supply theta' explicitly rather than letting the Levin driver obtain it by spectral
    differentiation of theta. At large argument the raw phase is a large float whose absolute resolution
    is ~eps*theta, so differentiating the sampled values loses precision in proportion to
    theta/(phase change across the subinterval); the phase spline can supply theta' directly without
    that loss.
    """

    def theta(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            + e_nu * phase_nu.raw_theta(q * x)
            + e_sigma * phase_sigma.raw_theta(s * x)
        )

    def theta_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            + e_nu * phase_nu.theta_mod_2pi(q * x)
            + e_sigma * phase_sigma.theta_mod_2pi(s * x)
        )

    def theta_deriv(log_x: float):
        # each term is theta_X(m*x) with x = exp(log_x), and log(m*x) = log(m) + log(x), so
        # d/d(log x) theta_X(m*x) is the log-derivative of theta_X evaluated at m*x
        x = np.exp(log_x)

        return (
            phase_mu.theta_deriv(k * x, log_derivative=True)
            + e_nu * phase_nu.theta_deriv(q * x, log_derivative=True)
            + e_sigma * phase_sigma.theta_deriv(s * x, log_derivative=True)
        )

    return {
        "theta": theta,
        "theta_mod_2pi": theta_mod_2pi,
        "theta_deriv": theta_deriv,
    }


def _Levin_3bessel(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    weights,
    combination,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    """
    Shared driver for the JJJ and YJJ three-Bessel integrals. These differ only in which slot of the
    (sin, cos) basis the slowly varying amplitude occupies, and in the signs with which the four
    sum-and-difference groups are recombined.

    :param weights: callable mapping the amplitude to the Levin f-vector
    :param combination: signs with which the four group values are combined
    """
    phase_mu = mu_phase["phase"]
    phase_nu = nu_phase["phase"]
    phase_sigma = sigma_phase["phase"]

    m_mu = mu_phase["mod"]
    m_nu = nu_phase["mod"]
    m_sigma = sigma_phase["mod"]

    x_span = (np.log(min_x), np.log(max_x))

    def Levin_f(log_x: float):
        x = np.exp(log_x)

        return np.pow(x, 3.0 / 2.0) * m_mu(k * x) * m_nu(q * x) * m_sigma(s * x)

    total = 0.0
    for index, (e_nu, e_sigma) in enumerate(_PHASE_GROUP_SIGNS):
        data = adaptive_levin_sincos(
            x_span,
            f=weights(Levin_f),
            theta=_phase_group(phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma),
            atol=atol,
            rtol=rtol,
            chebyshev_order=chebyshev_order,
            notify_label=f"phase{index + 1}",
        )
        total = total + combination[index] * data["value"]

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) / 4.0

    return norm_factor * total


def _Levin_JJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    # J_nu = m sin(theta), so the amplitude sits in the sine slot
    return _Levin_3bessel(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_x,
        max_x,
        atol,
        rtol,
        weights=lambda amplitude: [amplitude, lambda x: 0.0],
        combination=(-1.0, 1.0, 1.0, -1.0),
        chebyshev_order=chebyshev_order,
    )


def quad_YJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    min_x_mu = mu_phase["min_x"]
    min_x_nu = nu_phase["min_x"]
    min_x_sigma = sigma_phase["min_x"]

    # cut integral in two, as in quad_JJJ: see documentation there
    x_cut = max(min_x_mu, min_x_nu, min_x_sigma)
    min_kqs = min(k, q, s)
    min_cut = x_cut / min_kqs

    numeric = _direct_YJJ(mu, nu, sigma, k, q, s, 0.0, min_cut, atol, rtol)
    Levin = _Levin_YJJ(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_cut,
        max_x,
        atol,
        rtol,
        chebyshev_order=chebyshev_order,
    )

    return numeric + Levin


def _direct_YJJ(
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
):
    def integrand(x):
        return (
            np.sqrt(x)
            * yv(mu + 0.5, k * x)
            * jv(nu + 0.5, q * x)
            * jv(sigma + 0.5, s * x)
        )

    data = simple_quadrature(
        integrand,
        a=min_x,
        b=max_x,
        atol=atol,
        rtol=rtol,
        label="numeric part",
        method="quad",
    )

    return np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) * data["value"]


def _Levin_YJJ(
    mu_phase,
    nu_phase,
    sigma_phase,
    k: float,
    q: float,
    s: float,
    min_x: float,
    max_x: float,
    atol: float,
    rtol: float,
    chebyshev_order: int = DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
):
    # Y_nu = -m cos(theta), so relative to JJJ the amplitude moves to the cosine slot and the four
    # groups recombine with the opposite signs
    return _Levin_3bessel(
        mu_phase,
        nu_phase,
        sigma_phase,
        k,
        q,
        s,
        min_x,
        max_x,
        atol,
        rtol,
        weights=lambda amplitude: [lambda x: 0.0, amplitude],
        combination=(1.0, -1.0, -1.0, 1.0),
        chebyshev_order=chebyshev_order,
    )
