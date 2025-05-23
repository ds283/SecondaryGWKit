from math import fabs
from typing import Optional

from ComputeTargets import GkSource, GkSourcePolicyData
from ComputeTargets.GkSourcePolicyData import GkSourceFunctions
from CosmologyConcepts import wavenumber_exit_time, redshift

TEXT_DISPLACEMENT_MULTIPLIER = 0.85

TOP_ROW = 1.12
MIDDLE_ROW = 1.07
BOTTOM_ROW = 1.02

LEFT_COLUMN = 0.0
MIDDLE_COLUMN = 0.4
RIGHT_COLUMN = 0.85

BOTTOM_MARGIN = 0.0


def safe_fabs(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None

    return fabs(x)


def safe_div(x: Optional[float], y: float) -> Optional[float]:
    if x is None or y is None:
        return None

    try:
        return x / y
    except ZeroDivisionError:
        pass

    return None


def set_loglinear_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


def set_loglog_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


def set_linear_axes(ax):
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


# Matplotlib line style from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),
#
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

LOOSE_DASHED = (0, (4, 4))
LOOSE_DOTTED = (0, (1, 2))


def add_zexit_lines(ax, k_exit: wavenumber_exit_time, col1: str = "b", col2: str = "r"):
    ax.axvline(k_exit.z_exit_subh_e3, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(k_exit.z_exit_subh_e5, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(k_exit.z_exit_suph_e3, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(
        k_exit.z_exit, linestyle=(0, (3, 1, 1, 1)), color=col2
    )  # densely dashdotted
    trans = ax.get_xaxis_transform()
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_suph_e3,
        0.75,
        "$-3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e3,
        0.85,
        "$+3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e5,
        0.75,
        "$+5$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit,
        0.92,
        "re-entry",
        transform=trans,
        fontsize="x-small",
        color="r",
    )


def add_GkSourcePolicyData_lines(ax, GkPolicy):
    trans = ax.get_xaxis_transform()
    if GkPolicy.type == "mixed" and GkPolicy.crossover_z is not None:
        ax.axvline(
            GkPolicy.crossover_z, linestyle=(5, (10, 3)), color="m"
        )  # long dash with offset
        ax.text(
            TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.crossover_z,
            0.15,
            "crossover_z",
            transform=trans,
            fontsize="x-small",
            color="m",
        )
    if (
        GkPolicy.type == "mixed" or GkPolicy.type == "WKB"
    ) and GkPolicy.Levin_z is not None:
        ax.axvline(
            GkPolicy.Levin_z, linestyle=(0, (5, 10)), color="m"
        )  # loosely dashed
        ax.text(
            TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.Levin_z,
            0.05,
            "Levin boundary",
            transform=trans,
            fontsize="x-small",
            color="m",
        )


def add_k_labels(
    ax,
    k_exit: Optional[wavenumber_exit_time],
    q_exit: Optional[wavenumber_exit_time],
    r_exit: Optional[wavenumber_exit_time],
):
    if k_exit is not None:
        ax.text(
            LEFT_COLUMN,
            BOTTOM_ROW,
            f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if q_exit is not None:
        ax.text(
            MIDDLE_COLUMN,
            BOTTOM_ROW,
            f"$q$ =  {q_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if r_exit is not None:
        ax.text(
            RIGHT_COLUMN,
            BOTTOM_ROW,
            f"$r$ = {r_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def add_simple_plot_labels(
    ax,
    z_source: Optional[redshift] = None,
    z_response: Optional[redshift] = None,
    k_exit: Optional[wavenumber_exit_time] = None,
    q_exit: Optional[wavenumber_exit_time] = None,
    r_exit: Optional[wavenumber_exit_time] = None,
    model_label: str = "LambdaCDM",
):
    add_k_labels(ax, k_exit, q_exit, r_exit)

    if z_source is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"z-source: {z_source.z:.5g}",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    elif z_response is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"z-response: {z_response.z:.5g}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if model_label is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"Model: {model_label}",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def add_GkSource_plot_labels(
    ax, Gk: GkSource, GkPolicy: GkSourcePolicyData, model_label: str = "LambdaCDM"
):
    k_exit: wavenumber_exit_time = Gk._k_exit
    fns: GkSourceFunctions = GkPolicy.functions

    # TOP ROW
    if fns.WKB_region is not None:
        ax.text(
            LEFT_COLUMN,
            TOP_ROW,
            f"WKB region: ({fns.WKB_region[0]:.3g}, {fns.WKB_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.numeric_region is not None:
        ax.text(
            MIDDLE_COLUMN,
            TOP_ROW,
            f"Numeric region: ({fns.numeric_region[0]:.3g}, {fns.numeric_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    ax.text(
        RIGHT_COLUMN,
        TOP_ROW,
        f"Model: {model_label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )

    # MIDDLE ROW
    if fns.type is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"Type: {fns.type}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.quality is not None:
        ax.text(
            MIDDLE_COLUMN,
            MIDDLE_ROW,
            f"Quality: {fns.quality}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.phase is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"Chunks: {fns.phase.num_chunks}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    # BOTTOM ROW
    ax.text(
        LEFT_COLUMN,
        BOTTOM_ROW,
        f"z-response: {Gk.z_response.z:.5g}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        MIDDLE_COLUMN,
        BOTTOM_ROW,
        f"z-exit: {k_exit.z_exit:.5g}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        RIGHT_COLUMN,
        BOTTOM_ROW,
        f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
        transform=ax.transAxes,
        fontsize="x-small",
    )

    # BOTTOM MARGIN
    ax.text(
        LEFT_COLUMN,
        BOTTOM_MARGIN,
        f"policy: {GkPolicy.policy.label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )


def add_region_labels(
    ax,
    z_min_quad,
    z_max_quad,
    z_min_WKB_quad,
    z_max_WKB_quad,
    z_min_Levin,
    z_max_Levin,
    model_label: str = "LambdaCDM",
):
    if z_min_quad is not None and z_max_quad is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"numeric: [{z_min_quad.z:.5g}, {z_max_quad.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    if z_min_WKB_quad is not None and z_max_WKB_quad is not None:
        ax.text(
            MIDDLE_COLUMN,
            MIDDLE_ROW,
            f"WKB numeric: [{z_min_WKB_quad.z:.5g}, {z_max_WKB_quad.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    if z_min_Levin is not None and z_max_Levin is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"WKB Levin: [{z_min_Levin.z:.5g}, {z_max_Levin.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    ax.text(
        RIGHT_COLUMN,
        TOP_ROW,
        f"Model: {model_label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
