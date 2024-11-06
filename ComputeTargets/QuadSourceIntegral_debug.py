from pathlib import Path
from typing import Optional

import numpy as np
import seaborn as sns
from math import sqrt, pow, pi
from matplotlib import pyplot as plt
from scipy.special import jv, yv

from AdaptiveLevin.bessel_phase import MINIMUM_X
from thirdparty.autoscale import autoscale


def _safe_fabs(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None

    return np.fabs(x)


def set_loglog_axes(ax, inverted: bool = True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True)

    if inverted:
        ax.xaxis.set_inverted(True)


def set_loglinear_axes(ax, inverted: bool = True):
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    if inverted:
        ax.xaxis.set_inverted(True)


def _bessel_function_plot(phase_data, b, timestamp):
    nu_types = {"0pt5": 0.5, "2pt5": 2.5}
    base_path = Path("three_bessel_debug").resolve() / f"{timestamp.isoformat()}"

    for nu_type, data in phase_data.items():
        nu = nu_types[nu_type]

        max_x = data["max_x"]
        x_grid = np.logspace(np.log10(1e-10), np.log10(max_x), num=250)

        bessel_j = data["bessel_j"]
        bessel_y = data["bessel_y"]

        phase = data["phase"]
        dphase = data["dphase"]
        ddphase = data["ddphase"]

        j_grid = [_safe_fabs(jv(nu + b, x)) for x in x_grid]
        j_approx_grid = [_safe_fabs(bessel_j(x)) for x in x_grid]

        y_grid = [_safe_fabs(yv(nu + b, x)) for x in x_grid]
        y_approx_grid = [_safe_fabs(bessel_y(x)) for x in x_grid]

        theta_grid = [_safe_fabs(phase(x)) for x in x_grid]
        dphase_grid = [_safe_fabs(dphase(x)) for x in x_grid]
        ddphase_grid = [_safe_fabs(ddphase(x)) for x in x_grid]

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(x_grid, j_grid, linestyle="solid", color="r", label="$J_{\\nu}$ direct")
        ax.plot(
            x_grid,
            j_approx_grid,
            linestyle="dashed",
            color="b",
            label="$J_{\\nu}$ Liouville-Green",
        )

        set_loglog_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "BesselJ.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(x_grid, y_grid, linestyle="solid", color="r", label="$Y_{\\nu}$ direct")
        ax.plot(
            x_grid,
            y_approx_grid,
            linestyle="dashed",
            color="b",
            label="$Y_{\\nu}$ Liouville-Green",
        )

        set_loglog_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "BesselY.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            x_grid, theta_grid, linestyle="solid", color="r", label="$\\theta$ phase"
        )

        set_loglog_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "Bessel_phase.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        set_loglinear_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "Bessel_phase_linear.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        ax.set_xlim(MINIMUM_X, 5e-6)
        autoscale(ax=ax, axis="y")
        fig_path = base_path / nu_type / "Bessel_phase_linear_zoom.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            x_grid,
            dphase_grid,
            linestyle="solid",
            color="g",
            label="$\\beta = d\\theta/dx$",
        )
        ax.plot(
            x_grid,
            ddphase_grid,
            linestyle="solid",
            color="b",
            label="$\\alpha = d^2\\theta/dx^2$",
        )

        set_loglog_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "Bessel_alpha_beta.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        set_loglinear_axes(ax, inverted=False)

        fig_path = base_path / nu_type / "Bessel_alpha_beta_linear.pdf"
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()


def _three_bessel_plot(k, q, r, min_eta, max_eta, b, phase_data, nu_type, timestamp):
    base_path = Path("three_bessel_debug").resolve() / f"{timestamp.isoformat()}"
    grid = np.logspace(np.log10(min_eta), np.log10(max_eta), num=250)

    nu_types = {"0pt5": 0.5, "2pt5": 2.5}
    nu = nu_types[nu_type]

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)

    phase_data_A = phase_data["0pt5"]
    phase_data_B = phase_data[nu_type]

    phase_A = phase_data_A["phase"]
    phase_B = phase_data_B["phase"]

    dphase_A = phase_data_A["dphase"]
    dphase_B = phase_data_B["dphase"]

    def j_integrand(eta):
        A = pow(eta, 1.5 - b)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        B = jv(nu + b, x2) * jv(nu + b, x3)
        C = jv(0.5 + b, x1)

        return A * B * C

    def y_integrand(eta):
        A = pow(eta, 1.5 - b)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        B = jv(nu + b, x2) * jv(nu + b, x3)
        C = yv(0.5 + b, x1)

        return A * B * C

    def j_integrand_approx(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        beta1 = dphase_A(x1)
        beta2 = dphase_B(x2)
        beta3 = dphase_B(x3)

        A = pow(eta, -b)
        B = 1.0 / sqrt(beta1 * beta2 * beta3)

        gamma1 = phase_A(x1)
        gamma2 = phase_B(x2)
        gamma3 = phase_B(x3)

        # if np.fabs(gamma1) < 1e-2 and np.fabs(gamma2) < 1e-2 and np.fabs(gamma3) < 1e-2:
        #     C = -4.0 * gamma2 * gamma3
        # else:
        P = gamma1 + gamma2 + gamma3
        Q = gamma1 + gamma2 - gamma3
        R = gamma1 - gamma2 + gamma3
        S = gamma1 - gamma2 - gamma3

        C = -np.sin(P) + np.sin(Q) + np.sin(R) - np.sin(S)

        norm_factor = pow(2.0 / pi, 3.0 / 2.0) / sqrt(k.k * q.k * r.k) / cs / 4.0

        return norm_factor * A * B * C

    def y_integrand_approx(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        beta1 = dphase_A(x1)
        beta2 = dphase_B(x2)
        beta3 = dphase_B(x3)

        A = pow(eta, -b)
        B = 1.0 / sqrt(beta1 * beta2 * beta3)

        gamma1 = phase_A(x1)
        gamma2 = phase_B(x2)
        gamma3 = phase_B(x3)

        # if np.fabs(gamma1) < 1e-2 and np.fabs(gamma2) < 1e-2 and np.fabs(gamma3) < 1e-2:
        #     C = -4.0 * gamma1 * gamma2 * gamma3
        # else:
        P = gamma1 + gamma2 + gamma3
        Q = gamma1 + gamma2 - gamma3
        R = gamma1 - gamma2 + gamma3
        S = gamma1 - gamma2 - gamma3

        C = -np.cos(P) + np.cos(Q) + np.cos(R) - np.cos(S)

        norm_factor = pow(2.0 / pi, 3.0 / 2.0) / sqrt(k.k * q.k * r.k) / cs / 4.0

        return norm_factor * A * B * C

    j_line = [np.fabs(j_integrand(eta)) for eta in grid]
    y_line = [np.fabs(y_integrand(eta)) for eta in grid]

    j_approx_line = [np.fabs(j_integrand_approx(eta)) for eta in grid]
    y_approx_line = [np.fabs(y_integrand_approx(eta)) for eta in grid]

    sns.set_theme()

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(grid, j_line, linestyle="solid", color="r", label="$J_{\\nu}$ direct")
    ax.plot(
        grid,
        j_approx_line,
        linestyle="dashed",
        color="g",
        label="$J_{\\nu}$ Liouville-Green",
    )

    set_loglog_axes(ax)

    fig_path = base_path / nu_type / "ThreeBessel-J.pdf"
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(grid, y_line, linestyle="solid", color="b", label="$Y_{\\nu}$ direct")
    ax.plot(
        grid,
        y_approx_line,
        linestyle="dashed",
        color="m",
        label="$Y_{\\nu}$ Liouville-Green",
    )

    set_loglog_axes(ax)

    fig_path = base_path / nu_type / "ThreeBessel-Y.pdf"
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()

    def phaseP(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) + phase_B(x2) + phase_B(x3)

    def phaseQ(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) + phase_B(x2) - phase_B(x3)

    def phaseR(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) - phase_B(x2) + phase_B(x3)

    def phaseS(eta):
        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) - phase_B(x2) - phase_B(x3)

    phaseP_line = [phaseP(eta) for eta in grid]
    phaseQ_line = [phaseQ(eta) for eta in grid]
    phaseR_line = [phaseR(eta) for eta in grid]
    phaseS_line = [phaseS(eta) for eta in grid]

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(grid, phaseP_line, linestyle="solid", color="r", label="$P$")
    ax.plot(grid, phaseQ_line, linestyle="solid", color="b", label="$Q$")
    ax.plot(grid, phaseR_line, linestyle="solid", color="g", label="$R$")
    ax.plot(grid, phaseS_line, linestyle="solid", color="m", label="$S$")

    set_loglinear_axes(ax, inverted=False)

    fig_path = base_path / nu_type / "phases.pdf"
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()
