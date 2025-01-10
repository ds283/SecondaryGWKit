import unittest
from datetime import datetime
from pathlib import Path
from random import uniform

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.three_bessel_integrals import quad_JJJ, quad_YJJ

ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5

MAX_X = 1e8


def is_triangle(k: float, q: float, s: float):
    return np.fabs(k - q) < s < k + q


def plot_and_compute_3Bessel(
    evaluator,
    mu: float,
    nu: float,
    sigma: float,
    k: float,
    q: float,
    s: float,
    max_x: float,
    analytic_result: float,
    label: str,
    timestamp,
):
    mu_phase = bessel_phase(mu + 0.5, 1.075 * k * max_x, atol=1e-25, rtol=5e-14)
    nu_phase = bessel_phase(nu + 0.5, 1.075 * q * max_x, atol=1e-25, rtol=5e-14)
    sigma_phase = bessel_phase(sigma + 0.5, 1.075 * s * max_x, atol=1e-25, rtol=5e-14)

    x_grid = np.logspace(np.log10(100.0), np.log10(max_x), 250)
    y_grid = [
        evaluator(
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

    if analytic_result is not None:
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
        f"test_3bessel_analytic/{timestamp.isoformat()}/{label}_mu={mu:.3f}_nu={nu:.3f}_sigma={sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
    ).resolve()
    fig_path.parents[0].mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"))

    plt.close()

    value = evaluator(
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


class J000:
    mu = 0.0
    nu = 0.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

        return (np.pi / 4.0) / (k * q * s)


class J110:
    mu = 1.0
    nu = 1.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        if not is_triangle(k, q, s):
            return 0.0

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
        if not is_triangle(k, q, s):
            return 0.0

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
        if not is_triangle(k, q, s):
            return 0.0

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
        if not is_triangle(k, q, s):
            return 0.0

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


class Y000:
    mu = 0.0
    nu = 0.0
    sigma = 0.0

    @staticmethod
    def analytic(k, q, s):
        prefactor = (1.0 / 4.0) / (k * q * s)

        numerator = (k - q + s) * (k + q - s)
        denominator = (k + q + s) * (k - q - s)

        return prefactor * np.log(np.abs(numerator / denominator))


class Y022:
    mu = 0.0
    nu = 2.0
    sigma = 2.0

    @staticmethod
    def analytic(k, q, s):
        k_sq = k * k
        q_sq = q * q
        s_sq = s * s

        q3 = q * q_sq
        s3 = s * s_sq

        k4 = k_sq * k_sq
        q4 = q_sq * q_sq
        s4 = s_sq * s_sq

        A = 3.0 * (q_sq + s_sq - k_sq) / 8.0 / (k * q_sq * s_sq)

        B = 3.0 * (k4 + q4 + s4) - 6.0 * (k_sq * q_sq + k_sq * s_sq) + 2.0 * q_sq * s_sq
        C = 32.0 * k * q3 * s3

        numerator = (k - q + s) * (k + q - s)
        denominator = (k + q + s) * (k - q - s)

        return A + (B / C) * np.log(np.abs(numerator / denominator))


Jintegrals = [J000, J110, J220, J222, J231]
Yintegrals = [Y000, Y022]


def intify(x: float):
    return int(round(x, 0))


class Test3BesselAnalytic(unittest.TestCase):

    def test_JJJ(self):
        timestamp = datetime.now().replace(microsecond=0)

        for J in Jintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)
            s = uniform(0.1, 5.0)

            analytic = J.analytic(k, q, s)
            numeric = plot_and_compute_3Bessel(
                quad_JJJ,
                J.mu,
                J.nu,
                J.sigma,
                k,
                q,
                s,
                max_x=MAX_X,
                analytic_result=analytic,
                label="JJJ",
                timestamp=timestamp,
            )

            abserr = np.fabs(numeric - analytic)

            if is_triangle(k, q, s):
                relerr = abserr / analytic

                print(f"@@ (J{intify(J.mu)},J{intify(J.nu)},J{intify(J.sigma)}):")
                print(f"   k={k}, q={q}, s={s} satisfies the triangle inequality")
                print(f"   quadrature result = {numeric}")
                print(f"   analytic result = {analytic}")
                print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
                self.assertTrue(relerr < REL_TOLERANCE or abserr < ABS_TOLERANCE)

            else:
                print(f"@@ (J{intify(J.mu)},J{intify(J.nu)},J{intify(J.sigma)}):")
                print(
                    f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                )
                print(f"   quadrature result = {numeric}")
                print(f"   analytic result = {analytic}")
                print(f"   abserr={abserr:.5g}")
                self.assertTrue(abserr < ABS_TOLERANCE)

    def test_YJJ(self):
        timestamp = datetime.now().replace(microsecond=0)

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)
            s = uniform(0.1, 5.0)

            analytic = Y.analytic(k, q, s)
            numeric = plot_and_compute_3Bessel(
                quad_YJJ,
                Y.mu,
                Y.nu,
                Y.sigma,
                k,
                q,
                s,
                max_x=MAX_X,
                analytic_result=analytic,
                label="YJJ",
                timestamp=timestamp,
            )

            abserr = np.fabs(numeric - analytic)
            relerr = abserr / analytic

            print(f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}):")
            if is_triangle(k, q, s):
                print(f"   k={k}, q={q}, s={s} satisfies the triangle inequality")
            else:
                print(
                    f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                )

            print(f"   quadrature result = {numeric}")
            print(f"   analytic result = {analytic}")
            print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
            self.assertTrue(relerr < REL_TOLERANCE or abserr < ABS_TOLERANCE)

    def test_J110_notriangle(self):
        J = J110
        k = 4.870969802124623
        q = 1.3338040554375765
        s = 2.0319743925393587

        analytic = J.analytic(k, q, s)

        max_x = MAX_X
        mu_phase = bessel_phase(J.mu + 0.5, 1.075 * k * max_x, atol=1e-25, rtol=5e-14)
        nu_phase = bessel_phase(J.nu + 0.5, 1.075 * q * max_x, atol=1e-25, rtol=5e-14)
        sigma_phase = bessel_phase(
            J.sigma + 0.5, 1.075 * s * max_x, atol=1e-25, rtol=5e-14
        )

        numeric = quad_JJJ(
            mu_phase,
            nu_phase,
            sigma_phase,
            J.mu,
            J.nu,
            J.sigma,
            k,
            q,
            s,
            max_x,
            atol=1e-14,
            rtol=1e-10,
        )

        abserr = np.fabs(numeric - analytic)

        print(f"@@ ({intify(J.mu)},{intify(J.nu)},{intify(J.sigma)}):")
        print(f"   k={k}, q={q}, s={s}, is_triangle={is_triangle(k, q, s)}")
        print(f"   quadrature result = {numeric}")
        print(f"   analytic result = {analytic}")
        print(f"   abserr={abserr:.5g}")
        self.assertTrue(abserr < ABS_TOLERANCE)
