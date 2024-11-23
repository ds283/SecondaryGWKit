from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import jv, yv

from LiouvilleGreen.bessel_phase import bessel_phase

sns.set_theme()


def plot_bessel_phase(nu: float):
    data = bessel_phase(nu, 100.0)
    x_cut = data["x_cut"]
    min_x = max(0.1, x_cut)
    grid_J = np.linspace(min_x, 100.0, 250)
    grid_Y = np.linspace(min_x, 100.0, 250)

    phase = data["phase"]
    bessel_j = data["bessel_j"]
    bessel_y = data["bessel_y"]

    our_j_points = [bessel_j(x) for x in grid_J]
    their_j_points = [jv(nu, x) for x in grid_J]

    our_y_points = [bessel_y(x) for x in grid_Y]
    their_y_points = [yv(nu, x) for x in grid_Y]

    phase_points = [phase(x) for x in grid_J]

    # BESSEL PLOTS

    sns.set_theme()

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(grid_J, our_j_points, linestyle="solid", color="b", label="Our $J_{\\nu}$")
    ax.plot(
        grid_J, their_j_points, linestyle="dashed", color="g", label="Their $J_{\\nu}$"
    )

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = Path(f"bessel_J_plot_nu={nu:.3g}.pdf").resolve()
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(grid_Y, our_y_points, linestyle="solid", color="b", label="Our $Y_{\\nu}$")
    ax.plot(
        grid_Y, their_y_points, linestyle="dashed", color="g", label="Their $Y_{\\nu}$"
    )

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = Path(f"bessel_Y_plot_nu={nu:.3g}.pdf").resolve()
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()

    # PHASE_PLOTS PLOTS

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(
        grid_J, phase_points, linestyle="solid", color="r", label="Phase $\\gamma(x)$"
    )

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = Path(f"phase_plot_nu={nu:.3g}.pdf").resolve()
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()


plot_bessel_phase(nu=1.0 / 2.0)
plot_bessel_phase(nu=3.0 / 2.0)
plot_bessel_phase(nu=5.0 / 2.0)
