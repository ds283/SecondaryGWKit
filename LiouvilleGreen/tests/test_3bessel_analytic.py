import unittest
from datetime import datetime
from pathlib import Path
from random import uniform

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import jv

from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos
from LiouvilleGreen.bessel_phase import bessel_phase
from Quadrature.simple_quadrature import simple_quadrature


def eval_3bessel(
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    analytic_result: float,
    timestamp,
):
    mu_phase = bessel_phase(mu + 0.5, 1.075 * k * max_x, atol=1e-25, rtol=5e-14)
    nu_phase = bessel_phase(nu + 0.5, 1.075 * q * max_x, atol=1e-25, rtol=5e-14)
    sigma_phase = bessel_phase(sigma + 0.5, 1.075 * s * max_x, atol=1e-25, rtol=5e-14)

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

    ax.axhline(analytic_result, color="r", linestyle=(0, (1, 1)), label="Analytic")

    ax.text(0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.0, 1.08, f"$\\mu$={mu:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(0.3, 1.08, f"$\\nu$={nu:.5g}", transform=ax.transAxes, fontsize="x-small")
    ax.text(
        0.6, 1.08, f"$\\sigma$={sigma:.5g}", transform=ax.transAxes, fontsize="x-small"
    )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = Path(
        f"test_3bessel_analytic/{timestamp.isoformat()}/mu={mu:.3f}_nu={nu:.3f}_sigma={sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
    ).resolve()
    fig_path.parents[0].mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"))

    plt.close()

    value = quad_3bessel(
        mu_phase,
        nu_phase,
        sigma_phase,
        mu,
        nu,
        sigma,
        k,
        q,
        s,
        max_x,
        atol=1e-14,
        rtol=1e-10,
    )
    return value


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

    # print(f">> numeric on (0, {min_cut}) = {numeric}")
    # print(f">> Levin on ({min_cut}, {max_x}) = {Levin}")

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

    # print(
    #     f">> Levin groups: group1: {group1_value}, group2 {group2_value}, group3 {group3_value}, group4 {group4_value}"
    # )
    # print(
    #     f">> Levin result = {norm_factor} * ( {-group1_value} + {group2_value} + {group3_value} + {-group4_value} = {norm_factor * (-group1_value + group2_value + group3_value - group4_value)}"
    # )

    return norm_factor * (-group1_value + group2_value + group3_value - group4_value)


class J000:
    mu = 0.0
    nu = 0.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        return (np.pi / 4.0) / (k * q * s)


class J110:
    mu = 1.0
    nu = 1.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        q_sq = q * q
        s_sq = s * s
        return (np.pi / 8.0) * (k_sq + q_sq - s_sq) / (k_sq * q_sq * s)


class J220:
    mu = 2.0
    nu = 2.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        k3 = k_sq * k
        k4 = k_sq * k_sq
        q_sq = q * q
        q3 = q_sq * q
        s_sq = s * s

        pre_factor = np.pi / 32.0
        numerator = (
            3.0 * k4
            + 2.0 * k_sq * (q_sq - 3.0 * s_sq)
            + 3.0 * (q_sq - s_sq) * (q_sq - s_sq)
        )
        denominator = k3 * q3 * s

        return pre_factor * numerator / denominator


class J222:
    mu = 2.0
    nu = 2.0
    sigma = 2.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        k3 = k_sq * k
        q3 = q_sq * q
        s3 = s_sq * s

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        s6 = s4 * s_sq

        pre_factor = np.pi / 64.0

        numerator = (
            (3.0 * k4 + 2.0 * k_sq * q_sq + 3.0 * q4) * s_sq
            + 3.0 * (k_sq + q_sq) * s4
            - 3.0 * (k_sq - q_sq) * (k_sq - q_sq) * (k_sq + q_sq)
            - 3.0 * s6
        )
        denominator = k3 * q3 * s3

        return pre_factor * numerator / denominator


class J231:
    mu = 2.0
    nu = 3.0
    sigma = 1.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        k3 = k_sq * k

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        k6 = s4 * s_sq

        pre_factor = np.pi / 64.0

        numerator = (
            3.0 * k4 * (q_sq + 5.0 * s_sq)
            + (q_sq - s_sq) * (q_sq - s_sq) * (q_sq + 5.0 * s_sq)
            + k_sq * (q4 + 6.0 * q_sq * s_sq - 15.0 * s4)
            - 5.0 * k6
        )
        denominator = k3 * q4 * s_sq

        return pre_factor * numerator / denominator


Jintegrals = [J000, J110, J220, J222, J231]


class Test3BesselAnalytic(unittest.TestCase):

    def test_3Bessel(self):
        timestamp = datetime.now().replace(microsecond=0)

        for J in Jintegrals:
            k = uniform(1.0, 5.0)
            q = uniform(1.0, 5.0)
            s = uniform(1.0, 5.0)

            analytic = J.analytic(k, q, s)
            numeric = eval_3bessel(
                J.mu,
                J.nu,
                J.sigma,
                k,
                q,
                s,
                max_x=1e5,
                analytic_result=analytic,
                timestamp=timestamp,
            )

            abserr = np.fabs(numeric - analytic)
            relerr = abserr / analytic

            def intify(x: float):
                return int(round(x, 0))

            print(f"@@ ({intify(J.mu)},{intify(J.nu)},{intify(J.sigma)}):")
            print(f"   k={k}, q={q}, s=s{s}")
            print(f"   quadrature result = {numeric}")
            print(f"   analytic result = {analytic}")
            print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
            self.assertTrue(relerr < 1e-5 or abserr < 1e-6)
