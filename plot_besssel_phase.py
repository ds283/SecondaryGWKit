from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import jv, yv

from LiouvilleGreen.bessel_phase import bessel_phase

sns.set_theme()


def plot_bessel_phase(nu: float):
    data = bessel_phase(nu, 100.0)

    # "min_x", not "x_min": bessel_phase() has never returned a key by the latter name, and this
    # script read it -- so it raised KeyError on the first line of its body and had been dead for
    # at least two interface changes before the transfer-remedial campaign repaired it.
    min_x = max(0.1, data["min_x"])
    grid_J = np.linspace(min_x, 100.0, 250)
    grid_Y = np.linspace(min_x, 100.0, 250)

    phase = data["phase"]
    bessel_j = data["bessel_j"]
    bessel_y = data["bessel_y"]

    our_j_points = [bessel_j(x) for x in grid_J]
    their_j_points = [jv(nu, x) for x in grid_J]

    our_y_points = [bessel_y(x) for x in grid_Y]
    their_y_points = [yv(nu, x) for x in grid_Y]

    # the phase object is not callable, and never has been -- neither the phase_spline this script
    # was written against nor the two-region BesselPhaseFunction that replaced it defines
    # __call__. The accessor is raw_theta().
    phase_points = [phase.raw_theta(x) for x in grid_J]

    # the residual r_nu = theta - x - c_nu, in place of the old "Q" panel. Q was theta/x, the
    # pre-offset state of a phase ODE that no longer exists; the residual is the smooth quantity
    # the two-region construction represents.
    residual_points = [phase.residual(x) for x in grid_J]

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

    # PHASE PLOTS

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

    # RESIDUAL PLOT

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(
        grid_J,
        residual_points,
        linestyle="solid",
        color="b",
        label="Residual $r_{\\nu}(x)$",
    )

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)

    fig_path = Path(f"residual_plot_nu={nu:.3g}.pdf").resolve()
    fig_path.parents[0].mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path)

    plt.close()


plot_bessel_phase(nu=1.0 / 2.0)
plot_bessel_phase(nu=3.0 / 2.0)
plot_bessel_phase(nu=5.0 / 2.0)
