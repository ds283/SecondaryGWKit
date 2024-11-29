import numpy as np
from scipy.special import jv

from matplotlib import pyplot as plt
import seaborn as sns

from LiouvilleGreen.bessel_phase import bessel_phase
from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos
from Quadrature.simple_quadrature import simple_quadrature


def eval_3bessel(
    mu: float, nu: float, sigma: float, k: float, q: float, s: float, max_x: float
):
    mu_phase = bessel_phase(mu + 0.5, 1.075 * max_x, atol=1e-25, rtol=5e-14)
    nu_phase = bessel_phase(nu + 0.5, 1.075 * max_x, atol=1e-25, rtol=5e-14)
    sigma_phase = bessel_phase(sigma + 0.5, 1.075 * max_x, atol=1e-25, rtol=5e-14)

    x_grid = np.logspace(np.log10(100.0), np.log10(max_x), 250)
    y_grid = [
        quad_3bessel(
            mu_phase,
            nu_phase,
            sigma_phase,
            mu,
            nu,
            sigma,
            k,
            q,
            s,
            x,
            atol=1e-14,
            rtol=1e-10,
        )
        for x in x_grid
    ]

    sns.set_theme()

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(x_grid, y_grid, color="b", label="Numeric + Levin")


def quad_3bessel(
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

    numeric = numeric_3bessel(mu, nu, sigma, k, q, s, 0.0, min_cut, atol, rtol)
    Levin = Levin_3bessel(
        mu_phase, nu_phase, sigma_phase, k, q, s, min_cut, max_x, atol, rtol
    )

    return numeric + Levin


def numeric_3bessel(
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


def Levin_3bessel(
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

        return np.sqrt(x) * m_mu(k * x) * m_nu(q * x) * m_sigma(s * x)

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

    return norm_factor * (-group1_value + group2_value + group3_value - group4_value)
