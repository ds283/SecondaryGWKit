import numpy as np
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from Quadrature.simple_quadrature import simple_quadrature


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
        mu_phase, nu_phase, sigma_phase, k, q, s, min_cut, max_x, atol, rtol
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
):
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

    def phase1(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            + phase_nu.raw_theta(q * x)
            + phase_sigma.raw_theta(s * x)
        )

    def phase1_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            + phase_nu.theta_mod_2pi(q * x)
            + phase_sigma.theta_mod_2pi(s * x)
        )

    group1_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase1, "theta_mod_2pi": phase1_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase1",
    )

    def phase2(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            + phase_nu.raw_theta(q * x)
            - phase_sigma.raw_theta(s * x)
        )

    def phase2_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            + phase_nu.theta_mod_2pi(q * x)
            - phase_sigma.theta_mod_2pi(s * x)
        )

    group2_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase2, "theta_mod_2pi": phase2_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase2",
    )

    def phase3(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            - phase_nu.raw_theta(q * x)
            + phase_sigma.raw_theta(s * x)
        )

    def phase3_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            - phase_nu.theta_mod_2pi(q * x)
            + phase_sigma.theta_mod_2pi(s * x)
        )

    group3_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase3, "theta_mod_2pi": phase3_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase3",
    )

    def phase4(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            - phase_nu.raw_theta(q * x)
            - phase_sigma.raw_theta(s * x)
        )

    def phase4_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            - phase_nu.theta_mod_2pi(q * x)
            - phase_sigma.theta_mod_2pi(s * x)
        )

    group4_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta={"theta": phase4, "theta_mod_2pi": phase4_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase4",
    )

    group1_value = group1_data["value"]
    group2_value = group2_data["value"]
    group3_value = group3_data["value"]
    group4_value = group4_data["value"]

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) / 4.0
    Levin = norm_factor * (-group1_value + group2_value + group3_value - group4_value)

    return Levin


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
        mu_phase, nu_phase, sigma_phase, k, q, s, min_cut, max_x, atol, rtol
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
):
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

    def phase1(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            + phase_nu.raw_theta(q * x)
            + phase_sigma.raw_theta(s * x)
        )

    def phase1_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            + phase_nu.theta_mod_2pi(q * x)
            + phase_sigma.theta_mod_2pi(s * x)
        )

    group1_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta={"theta": phase1, "theta_mod_2pi": phase1_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase1",
    )

    def phase2(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            + phase_nu.raw_theta(q * x)
            - phase_sigma.raw_theta(s * x)
        )

    def phase2_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            + phase_nu.theta_mod_2pi(q * x)
            - phase_sigma.theta_mod_2pi(s * x)
        )

    group2_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta={"theta": phase2, "theta_mod_2pi": phase2_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase2",
    )

    def phase3(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            - phase_nu.raw_theta(q * x)
            + phase_sigma.raw_theta(s * x)
        )

    def phase3_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            - phase_nu.theta_mod_2pi(q * x)
            + phase_sigma.theta_mod_2pi(s * x)
        )

    group3_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta={"theta": phase3, "theta_mod_2pi": phase3_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase3",
    )

    def phase4(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.raw_theta(k * x)
            - phase_nu.raw_theta(q * x)
            - phase_sigma.raw_theta(s * x)
        )

    def phase4_mod_2pi(log_x: float):
        x = np.exp(log_x)

        return (
            phase_mu.theta_mod_2pi(k * x)
            - phase_nu.theta_mod_2pi(q * x)
            - phase_sigma.theta_mod_2pi(s * x)
        )

    group4_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta={"theta": phase4, "theta_mod_2pi": phase4_mod_2pi},
        atol=atol,
        rtol=rtol,
        chebyshev_order=64,
        notify_label="phase4",
    )

    group1_value = group1_data["value"]
    group2_value = group2_data["value"]
    group3_value = group3_data["value"]
    group4_value = group4_data["value"]

    norm_factor = np.pow(np.pi / 2.0, 3.0 / 2.0) / np.sqrt(k * q * s) / 4.0
    Levin = norm_factor * (group1_value - group2_value - group3_value + group4_value)

    return Levin
