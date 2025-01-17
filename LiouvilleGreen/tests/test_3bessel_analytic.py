import time
import unittest
from datetime import datetime
from pathlib import Path
from random import uniform

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.three_bessel_integrals import quad_JJJ, quad_YJJ
from utilities import format_time

ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5

SINGULARITY_ABS_TOLERANCE = 1e-3
SINGULARITY_REL_TOLERANCE = 1e-2

# with max eta around 10500 and a largest k of order 10^7/Mpc or so, the largest argument we would
# appear to need is about 1E4 * 1E7 ~ 1E11
MAX_X = 1e12


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
    phase_atol=1e-25,
    phase_rtol=5e-14,
    quad_atol=1e-14,
    quad_rtol=1e-10,
):
    mu_phase = bessel_phase(
        mu + 0.5, 1.075 * k * max_x, atol=phase_atol, rtol=phase_rtol
    )
    nu_phase = bessel_phase(
        nu + 0.5, 1.075 * q * max_x, atol=phase_atol, rtol=phase_rtol
    )
    sigma_phase = bessel_phase(
        sigma + 0.5, 1.075 * s * max_x, atol=phase_atol, rtol=phase_rtol
    )

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
            atol=quad_atol,
            rtol=quad_rtol,
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
        atol=quad_atol,
        rtol=quad_rtol,
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
singularity_eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


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

    def test_YJJ_log_singularity(self):
        timestamp = datetime.now().replace(microsecond=0)

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)

            for eps in singularity_eps:
                s_lo = np.abs(k - q) + eps
                s_hi = k + q - eps

                values = [("lo", s_lo), ("hi", s_hi)]

                for label, s in values:
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
                        label=f"YJJ_log_eps={eps:.2g}_{label}",
                        timestamp=timestamp,
                        quad_atol=1e-10,
                        quad_rtol=1e-8,
                    )

                    abserr = np.fabs(numeric - analytic)
                    relerr = abserr / analytic

                    print(
                        f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}) eps={eps:.2g} {label}:"
                    )
                    if is_triangle(k, q, s):
                        print(
                            f"   k={k}, q={q}, s={s} satisfies the triangle inequality"
                        )
                    else:
                        print(
                            f"   k={k}, q={q}, s={s} does not satisfy the triangle inequality"
                        )

                    print(f"   quadrature result = {numeric}")
                    print(f"   analytic result = {analytic}")
                    print(f"   relerr={relerr:.5g}, abserr={abserr:.5g}")
                    self.assertTrue(
                        relerr < SINGULARITY_REL_TOLERANCE
                        or abserr < SINGULARITY_ABS_TOLERANCE
                    )

    def test_YJJ_log_scaling(
        self,
        max_x=MAX_X,
        phase_atol=1e-25,
        phase_rtol=5e-14,
        quad_atol=1e-10,
        quad_rtol=1e-8,
    ):
        timestamp = datetime.now().replace(microsecond=0)

        sns.set_theme()

        for Y in Yintegrals:
            k = uniform(0.1, 5.0)
            q = uniform(0.1, 5.0)

            mu_phase = bessel_phase(
                Y.mu + 0.5, 1.075 * k * max_x, atol=phase_atol, rtol=phase_rtol
            )
            nu_phase = bessel_phase(
                Y.nu + 0.5, 1.075 * q * max_x, atol=phase_atol, rtol=phase_rtol
            )

            configs = [
                ("lo", lambda k, q, eps: np.abs(k - q) + eps),
                ("hi", lambda k, q, eps: k + q - eps),
            ]

            for label, s_evaluator in configs:
                print(
                    f"@@ (Y{intify(Y.mu)},J{intify(Y.nu)},J{intify(Y.sigma)}) {label}:"
                )

                total_start = time.perf_counter()
                last_notify = total_start

                analytic_grid = []
                numeric_grid = []
                elapsed_grid = []

                for eps in singularity_eps:
                    start = time.perf_counter()

                    s = s_evaluator(k, q, eps)

                    sigma_phase = bessel_phase(
                        Y.sigma + 0.5,
                        1.075 * s * max_x,
                        atol=phase_atol,
                        rtol=phase_rtol,
                    )

                    analytic = Y.analytic(k, q, s)
                    numeric = quad_YJJ(
                        mu_phase,
                        nu_phase,
                        sigma_phase,
                        Y.mu,
                        Y.nu,
                        Y.sigma,
                        k,
                        q,
                        s,
                        max_x,
                        atol=quad_atol,
                        rtol=quad_rtol,
                    )

                    stop = time.perf_counter()

                    analytic_grid.append(analytic)
                    numeric_grid.append(numeric)

                    elapsed = stop - start
                    elapsed_grid.append(elapsed)

                    last_notify_elapsed = stop - last_notify
                    total_elapsed = stop - total_start
                    if elapsed > 2 * 60 or last_notify_elapsed > 5 * 60:
                        print(
                            f"   -- evaluated eps={eps:.3g} (value={numeric:.5g}) in time {format_time(elapsed)} (total time for this YJJ {format_time(total_elapsed)})"
                        )
                        last_notify = stop

                fig = plt.figure()
                ax = plt.gca()

                ax.plot(
                    singularity_eps, numeric_grid, color="b", label="Numeric + Levin"
                )
                ax.plot(
                    singularity_eps,
                    analytic_grid,
                    color="r",
                    linestyle=(0, (1, 1)),
                    label="Analytic",
                )

                ax.text(
                    0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.0,
                    1.08,
                    f"$\\mu$={Y.mu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.3,
                    1.08,
                    f"$\\nu$={Y.nu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.6,
                    1.08,
                    f"$\\sigma$={Y.sigma:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )

                ax.set_xscale("log")
                ax.set_yscale("linear")
                ax.xaxis.set_inverted(True)
                ax.set_xlabel("$\epsilon$")
                ax.legend(loc="best")
                ax.grid(True)

                fig_path = Path(
                    f"test_3bessel_analytic/{timestamp.isoformat()}/eps_dependence_{label}_mu={Y.mu:.3f}_nu={Y.nu:.3f}_sigma={Y.sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
                ).resolve()
                fig_path.parents[0].mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)
                fig.savefig(fig_path.with_suffix(".png"))

                plt.close()

                fig = plt.figure()
                ax = plt.gca()

                ax.plot(
                    singularity_eps, elapsed_grid, color="b", label="Integration time"
                )

                ax.text(
                    0.0, 1.03, f"k={k:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.3, 1.03, f"q={q:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.6, 1.03, f"s={s:.5g}", transform=ax.transAxes, fontsize="x-small"
                )
                ax.text(
                    0.0,
                    1.08,
                    f"$\\mu$={Y.mu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.3,
                    1.08,
                    f"$\\nu$={Y.nu:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.text(
                    0.6,
                    1.08,
                    f"$\\sigma$={Y.sigma:.5g}",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.xaxis.set_inverted(True)
                ax.set_xlabel("$\epsilon$")
                ax.legend(loc="best")
                ax.grid(True)

                fig_path = Path(
                    f"test_3bessel_analytic/{timestamp.isoformat()}/integration_time_{label}_mu={Y.mu:.3f}_nu={Y.nu:.3f}_sigma={Y.sigma:.3f}_k={k:.3f}_q={q:.3f}_s={s:.3f}_maxx={max_x:.5g}.pdf"
                ).resolve()
                fig_path.parents[0].mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)
                fig.savefig(fig_path.with_suffix(".png"))

                total_stop = time.perf_counter()
                total_elapsed = total_stop - total_start

                print(f"   >> completed plot in time {format_time(total_elapsed)}")
