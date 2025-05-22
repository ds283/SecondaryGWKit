import argparse
import sys
from datetime import datetime
from itertools import product
from math import pi, sqrt
from pathlib import Path
from random import sample
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns
from ray import ObjectRef

from ComputeTargets import (
    BackgroundModel,
    GkSource,
    GkSourceValue,
    GkSourcePolicyData,
    GkSourceProxy,
)
from ComputeTargets.BackgroundModel import ModelProxy
from ComputeTargets.GkSourcePolicyData import GkSourceFunctions
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
    redshift_array,
    redshift,
)
from CosmologyConcepts.wavenumber import wavenumber_exit_time_array
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.ShardedPool import ShardedPool
from MetadataConcepts import tolerance
from RayTools.RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from extract_common import (
    TEXT_DISPLACEMENT_MULTIPLIER,
    TOP_ROW,
    MIDDLE_ROW,
    BOTTOM_ROW,
    LEFT_COLUMN,
    MIDDLE_COLUMN,
    RIGHT_COLUMN,
    set_loglinear_axes,
    set_loglog_axes,
    set_linear_axes,
    add_zexit_lines,
    safe_div,
    safe_fabs,
    LOOSE_DASHED,
    LOOSE_DOTTED,
)
from model_list import build_model_list

DEFAULT_TIMEOUT = 60

two_pi = 2.0 * pi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    default=None,
    help="read/write work items using the specified database cache",
)
parser.add_argument(
    "--db-timeout",
    default=DEFAULT_TIMEOUT,
    type=int,
    help="specify connection timeout for database layer",
)
parser.add_argument(
    "--profile-db",
    type=str,
    default=None,
    help="write profiling and performance data to the specified database",
)
parser.add_argument(
    "--ray-address", default="auto", type=str, help="specify address of Ray cluster"
)
parser.add_argument(
    "--output", default="GkSource-out", type=str, help="specify folder for output files"
)
args = parser.parse_args()

if args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

VERSION_LABEL = "2025.1.1"

# instantiate a Datastore actor: this runs on its own node, and acts as a broker between
# ourselves and the dataabase.
# For performance reasons, we want all database activity to run on this node.
# For one thing, this lets us use transactions efficiently.

profile_agent = None
if args.profile_db is not None:
    label = f'{VERSION_LABEL}-jobname-extract_GkSource_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )


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
def add_z_labels(ax, Gk, GkPolicy: GkSourcePolicyData):
    k_exit: wavenumber_exit_time = Gk._k_exit
    add_zexit_lines(ax, k_exit)

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


def add_GkSource_plot_labels(
    ax, Gk: GkSource, GkPolicy: GkSourcePolicyData, model_label: str = "LambdaCDM"
):
    k_exit: wavenumber_exit_time = Gk._k_exit
    fns: GkSourceFunctions = GkPolicy.functions

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

    # TOP ROW

    if fns.WKB_region is not None:
        ax.text(
            LEFT_COLUMN,
            TOP_ROW,
            f"WKB region: ({fns.WKB_region[0]:.3g}, {fns.WKB_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.numerical_region is not None:
        ax.text(
            MIDDLE_COLUMN,
            TOP_ROW,
            f"Numeric region: ({fns.numerical_region[0]:.3g}, {fns.numerical_region[1]:.3g})",
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


@ray.remote
def plot_Gk(
    model_label: str,
    model_proxy: ModelProxy,
    GkPolicy: GkSourcePolicyData,
    z_response_sample_max_z: redshift,
):
    if not GkPolicy.available:
        print(
            f"** [FATAL] GkSourcePolicyData object is not marked as available. This GkSource object may not have been processed, or be incomplete."
        )
        return

    Gk: GkSource = GkPolicy._source_proxy.get()
    if not Gk.available:
        print(
            f"** [FATAL] GkSource object is not marked as available. The datastore is likely incomplete."
        )
        return

    model: BackgroundModel = model_proxy.get()

    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

    values: List[GkSourceValue] = Gk.values

    functions: GkSourceFunctions = GkPolicy.functions

    num_max_z = None
    num_min_z = None
    WKB_max_z = None
    WKB_min_z = None

    num_region = functions.numerical_region
    if num_region is not None:
        num_max_z, num_min_z = num_region

    WKB_region = functions.WKB_region
    if WKB_region is not None:
        WKB_max_z, WKB_min_z = WKB_region

    if num_region is not None:
        numerical_points = [
            value for value in values if num_max_z >= value.z_source.z >= num_min_z
        ]
    else:
        numerical_points = []

    if WKB_region is not None:
        WKB_points = [
            value for value in values if WKB_max_z >= value.z_source.z >= WKB_min_z
        ]
    else:
        WKB_points = []

    abs_G_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(value.numeric.G),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in values
    ]
    abs_G_WKB_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(value.WKB.G_WKB),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in values
    ]
    abs_G_spline_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(functions.numerical_Gk(value.z_source.z)),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in numerical_points
    ]
    abs_G_WKB_spline_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(functions.WKB_Gk(value.z_source.z)),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in WKB_points
    ]
    abs_analytic_G_rad_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(value.analytic_G_rad),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in values
    ]
    abs_analytic_G_w_points = [
        (
            value.z_source.z,
            safe_div(
                safe_fabs(value.analytic_G_w),
                (1.0 + value.z_source.z)
                * model.functions.Hubble(value.z_source.z) ** 2,
            ),
        )
        for value in values
    ]

    abs_theta_points = [
        (value.z_source.z, safe_fabs(value.WKB.theta)) for value in values
    ]
    abs_raw_theta_points = [
        (value.z_source.z, safe_fabs(value.WKB.raw_theta)) for value in values
    ]
    abs_theta_spline_points = [
        (
            value.z_source.z,
            safe_fabs(functions.phase.raw_theta(value.z_source.z)),
        )
        for value in WKB_points
    ]

    theta_points = [(value.z_source.z, value.WKB.theta) for value in values]
    raw_theta_points = [(value.z_source.z, value.WKB.raw_theta) for value in values]
    theta_spline_points = [
        (
            value.z_source.z,
            functions.phase.raw_theta(value.z_source.z),
        )
        for value in WKB_points
    ]

    theta_spline_deriv_points = [
        (value.z_source.z, functions.phase.theta_deriv(value.z_source.z))
        for value in WKB_points
    ]

    abs_amplitude_sin_points = [
        (
            value.z_source.z,
            safe_fabs(
                value.WKB.sin_coeff * sqrt(value.WKB.H_ratio / sqrt(value.omega_WKB_sq))
            ),
        )
        for value in WKB_points
    ]
    abs_amplitude_cos_points = [
        (
            value.z_source.z,
            safe_fabs(
                value.WKB.cos_coeff * sqrt(value.WKB.H_ratio / sqrt(value.omega_WKB_sq))
            ),
        )
        for value in WKB_points
    ]

    abs_G_x, abs_G_y = zip(*abs_G_points)
    abs_G_WKB_x, abs_G_WKB_y = zip(*abs_G_WKB_points)
    abs_analytic_G_rad_x, abs_analytic_G_rad_y = zip(*abs_analytic_G_rad_points)
    abs_analytic_G_w_x, abs_analytic_G_w_y = zip(*abs_analytic_G_w_points)

    abs_G_spline_x, abs_G_spline_y = (
        zip(*abs_G_spline_points) if len(abs_G_spline_points) > 0 else ([], [])
    )
    abs_G_WKB_spline_x, abs_G_WKB_spline_y = (
        zip(*abs_G_WKB_spline_points) if len(abs_G_WKB_spline_points) > 0 else ([], [])
    )

    abs_theta_x, abs_theta_y = zip(*abs_theta_points)
    abs_raw_theta_x, abs_raw_theta_y = zip(*abs_raw_theta_points)

    theta_x, theta_y = zip(*theta_points)
    raw_theta_x, raw_theta_y = zip(*raw_theta_points)

    abs_theta_spline_x, abs_theta_spline_y = (
        zip(*abs_theta_spline_points) if len(abs_theta_spline_points) > 0 else ([], [])
    )
    theta_spline_x, theta_spline_y = (
        zip(*theta_spline_points) if len(theta_spline_points) > 0 else ([], [])
    )

    theta_spline_deriv_x, theta_spline_deriv_y = (
        zip(*theta_spline_deriv_points)
        if len(theta_spline_deriv_points) > 0
        else ([], [])
    )

    abs_amplitude_sin_x, abs_amplitude_sin_y = (
        zip(*abs_amplitude_sin_points)
        if len(abs_amplitude_sin_points) > 0
        else ([], [])
    )
    abs_amplitude_cos_x, abs_amplitude_cos_y = (
        zip(*abs_amplitude_cos_points)
        if len(abs_amplitude_cos_points) > 0
        else ([], [])
    )

    k_exit = Gk._k_exit
    z_response = Gk.z_response

    sns.set_theme()

    if len(abs_G_x) > 0 and (
        any(y is not None and y > 0 for y in abs_G_y)
        or any(y is not None and y > 0 for y in abs_G_WKB_y)
        or any(y is not None and y > 0 for y in abs_analytic_G_rad_y)
        or any(y is not None and y > 0 for y in abs_analytic_G_w_y)
        or any(y is not None and y > 0 for y in abs_G_spline_y)
        or any(y is not None and y > 0 for y in abs_G_WKB_spline_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            abs_G_x,
            abs_G_y,
            label="Numeric $G_k$",
            color="r",
            linestyle="solid",
        )
        ax.plot(
            abs_G_WKB_x,
            abs_G_WKB_y,
            label="WKB $G_k$",
            color="b",
            linestyle="solid",
        )
        ax.plot(
            abs_analytic_G_rad_x,
            abs_analytic_G_rad_y,
            label="Analytic $G_k$ [radiation]",
            color="y",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            abs_analytic_G_rad_x,
            abs_analytic_G_rad_y,
            label="Analytic $G_k$ [$w=w(z)$]",
            color="g",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            abs_G_spline_x,
            abs_G_spline_y,
            label="Spline numeric $G_k$",
            color="c",
            linestyle=LOOSE_DOTTED,
        )
        ax.plot(
            abs_G_WKB_spline_x,
            abs_G_WKB_spline_y,
            label="Spline WKB $G_k$",
            color="m",
            linestyle=LOOSE_DOTTED,
        )

        add_z_labels(ax, Gk, GkPolicy)
        add_GkSource_plot_labels(ax, Gk, GkPolicy, model_label)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$G_k(z, z') / [(1+z') H(z')^2 ]$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/Gk/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        if z_response.z <= k_exit.z_exit_suph_e3:
            ax.set_xlim(
                int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/Gk-reentry-zoom/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

        ax.set_xlim(
            min(int(round(z_response.z * 100.0 + 0.5, 0)), z_response_sample_max_z.z),
            max(int(round(z_response.z * 0.9 - 0.5, 0)), 0.01),
        )

        fig_path = (
            base_path
            / f"plots/Gk-zresponse-zoom/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        plt.close()

    if len(abs_theta_x) > 0 and (
        any(y is not None and y > 0 for y in abs_theta_y)
        or any(y is not None and y > 0 for y in abs_raw_theta_y)
        or any(y is not None and y > 0 for y in abs_theta_spline_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            abs_theta_x,
            abs_theta_y,
            label="WKB phase $\\theta$",
            color="r",
            linestyle="solid",
        )
        ax.plot(
            abs_raw_theta_x,
            abs_raw_theta_y,
            label="Raw WKB phase $\\theta$",
            color="b",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            abs_theta_spline_x,
            abs_theta_spline_y,
            label="Spline WKB phase $\\theta$",
            color="g",
            linestyle=(0, (1, 1)),
        )

        add_z_labels(ax, Gk, GkPolicy)
        add_GkSource_plot_labels(ax, Gk, GkPolicy, model_label)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("WKB phase $\\theta$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/theta-log/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))
        plt.close()

    if len(theta_x) > 0 and (
        any(y is not None for y in theta_y)
        or any(y is not None for y in raw_theta_y)
        or any(y is not None for y in theta_spline_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            theta_x,
            theta_y,
            label="WKB phase $\\theta$",
            color="r",
            linestyle="solid",
        )
        ax.plot(
            raw_theta_x,
            raw_theta_y,
            label="Raw WKB phase $\\theta$",
            color="b",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            theta_spline_x,
            theta_spline_y,
            label="Spline WKB phase $\\theta$",
            color="g",
            linestyle=(0, (1, 1)),
        )

        add_z_labels(ax, Gk, GkPolicy)
        add_GkSource_plot_labels(ax, Gk, GkPolicy, model_label)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("WKB phase $\\theta$")

        set_loglinear_axes(ax)

        fig_path = (
            base_path
            / f"plots/theta-linear/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        if GkPolicy.Levin_z is not None:
            theta_at_Levin_z = functions.phase.raw_theta(GkPolicy.Levin_z)
            ax.set_xlim(
                int(round(GkPolicy.Levin_z - 50.0 + 0.5, 0)),
                int(round(GkPolicy.Levin_z + 50.0 + 0.5, 0)),
            )
            ax.set_ylim(
                int(round(theta_at_Levin_z - 100.0 + 0.5, 0)),
                int(round(theta_at_Levin_z + 100.0 + 0.5, 0)),
            )

            set_linear_axes(ax)

            fig_path = (
                base_path
                / f"plots/theta-linear-Levin-zoom/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

        plt.close()

    if len(theta_spline_deriv_x) > 0 and (
        any(y is not None for y in theta_spline_deriv_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            theta_spline_deriv_x,
            theta_spline_deriv_y,
            label="derivative of WKB phase $\\theta$",
        )

        add_z_labels(ax, Gk, GkPolicy)
        add_GkSource_plot_labels(ax, Gk, GkPolicy, model_label)

        ax.set_xlabel("source redshift $z$")

        set_loglinear_axes(ax)

        fig_path = (
            base_path
            / f"plots/theta-linear-deriv/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))
        plt.close()

    if len(abs_amplitude_sin_x) > 0 and (
        any(y is not None and y > 0 for y in abs_amplitude_sin_y)
        or any(y is not None and y > 0 for y in abs_amplitude_cos_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(abs_amplitude_sin_x, abs_amplitude_sin_y, label="$\\sin$ amplitude")
        ax.plot(abs_amplitude_cos_x, abs_amplitude_cos_y, label="$\\cos$ amplitude")

        add_GkSource_plot_labels(ax, Gk, GkPolicy, model_label)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("amplitude")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/amplitude/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))
        plt.close()

    z_source_column = [value.z_source.z for value in values]
    G_column = [value.numeric.G for value in values]
    Gprime_column = [value.numeric.Gprime for value in values]
    G_WKB_column = [value.WKB.G_WKB for value in values]
    theta_div_2pi_column = [value.WKB.theta_div_2pi for value in values]
    theta_mod_2pi_column = [value.WKB.theta_mod_2pi for value in values]
    theta_column = [value.WKB.theta for value in values]
    raw_theta_div_2pi_column = [value.WKB.raw_theta_div_2pi for value in values]
    raw_theta_column = [value.WKB.raw_theta for value in values]
    H_ratio_column = [value.WKB.H_ratio for value in values]
    sin_coeff_column = [value.WKB.sin_coeff for value in values]
    cos_coeff_column = [value.WKB.cos_coeff for value in values]
    omega_WKB_sq_column = [value.omega_WKB_sq for value in values]
    WKB_criterion_column = [value.WKB_criterion for value in values]
    analytic_G_column = [value.analytic_G_rad for value in values]
    analytic_Gprime_column = [value.analytic_Gprime_rad for value in values]

    csv_path = (
        base_path
        / f"csv/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zresponse={z_response.z:.5g}-z-serial={z_response.store_id}.csv"
    )
    csv_path.parents[0].mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame.from_dict(
        {
            "z_source": z_source_column,
            "G": G_column,
            "Gprime": Gprime_column,
            "G_WKB": G_WKB_column,
            "theta_div_2pi": theta_div_2pi_column,
            "theta_mod_2pi": theta_mod_2pi_column,
            "theta": theta_column,
            "raw_theta_div_2pi": raw_theta_div_2pi_column,
            "raw_theta": raw_theta_column,
            "H_ratio": H_ratio_column,
            "sin_coeff": sin_coeff_column,
            "cos_coeff": cos_coeff_column,
            "omega_WKB_sq": omega_WKB_sq_column,
            "WKB_criterion": WKB_criterion_column,
            "analytic_G_rad": analytic_G_column,
            "analytic_Gprime_rad": analytic_Gprime_column,
        }
    )
    df.sort_values(by="z_source", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(csv_path, header=True, index=False)


def run_pipeline(model_data):
    model_label = model_data["label"]
    model_cosmology = model_data["cosmology"]

    print(f"\n>> RUNNING PIPELINE FOR MODEL {model_label}")

    # build absolute and relative tolerances
    atol, rtol = ray.get(
        [
            pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
        ]
    )

    # read in the model instance, which will tell us which z-sample points to use
    model: BackgroundModel = ray.get(
        pool.object_get(
            BackgroundModel,
            solver_labels=[],
            cosmology=model_cosmology,
            z_sample=None,
            atol=atol,
            rtol=rtol,
        )
    )
    if not model.available:
        raise RuntimeError(
            "Could not locate suitable background model instance in the datastore"
        )

    # set up a proxy object to avoid having to repeatedly serialize the model instance and ship it out
    model_proxy = ModelProxy(model)

    # set up/read in array of k-modes matching the SOURCE and RESPONSE k-grids
    # for now, we assume data is available for all k-modes in the database
    source_k_array = ray.get(pool.read_wavenumber_table(units=units, is_source=True))
    response_k_array = ray.get(
        pool.read_wavenumber_table(units=units, is_response=True)
    )

    def create_k_exit_work(k: wavenumber):
        return pool.object_get(
            wavenumber_exit_time,
            k=k,
            cosmology=model_cosmology,
            atol=atol,
            rtol=rtol,
        )

    # query wavenumber_exit_time objects corresponding to these k modes
    source_k_exit_queue = RayWorkPool(
        pool,
        source_k_array,
        task_builder=create_k_exit_work,
        compute_handler=None,
        store_handler=None,
        store_results=True,
        title="QUERY K_EXIT VALUES",
    )
    source_k_exit_queue.run()
    source_k_exit_times = wavenumber_exit_time_array(source_k_exit_queue.results)

    response_k_exit_queue = RayWorkPool(
        pool,
        response_k_array,
        task_builder=create_k_exit_work,
        compute_handler=None,
        store_handler=None,
        store_results=True,
        title="QUERY K_EXIT VALUES",
    )
    response_k_exit_queue.run()
    response_k_exit_times = wavenumber_exit_time_array(response_k_exit_queue.results)

    # choose a subsample of about 30% of the RESPONSE k modes
    k_subsample: List[wavenumber_exit_time] = sample(
        list(response_k_exit_times),
        k=int(round(0.3 * len(response_k_exit_times) + 0.5, 0)),
    )

    z_source_array = ray.get(
        pool.read_redshift_table(is_source=True, model_proxy=model_proxy)
    )
    z_source_sample = redshift_array(z_array=z_source_array)

    z_response_array = ray.get(
        pool.read_redshift_table(is_response=True, model_proxy=model_proxy)
    )
    z_response_sample = redshift_array(z_array=z_response_array)

    # choose a subsample of about 10% of the RESPONSE redshifts
    z_subsample: List[redshift] = sample(
        list(z_response_sample), k=int(round(0.1 * len(z_response_sample) + 0.5, 0))
    )
    z_response_sample_max_z = max(z_subsample)

    GkSource_policy_1pt5, GkSource_policy_5pt0 = ray.get(
        [
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="1.5"',
                Levin_threshold=1.5,
                numeric_policy="maximize_Levin",
            ),
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize_Levin",
            ),
        ]
    )

    def build_plot_GkSource_work(item):
        k_exit, z_response = item
        k_exit: wavenumber_exit_time
        z_response: redshift

        GkSource_ref: ObjectRef = pool.object_get(
            "GkSource",
            model=model_proxy,
            k=k_exit,
            z_response=z_response,
            z_sample=None,
            atol=atol,
            rtol=rtol,
        )
        GkSource = ray.get(GkSource_ref)
        GkSource_proxy = GkSourceProxy(GkSource)

        Policy_ref: ObjectRef = pool.object_get(
            "GkSourcePolicyData",
            source=GkSource_proxy,
            policy=GkSource_policy_1pt5,
            k=k_exit,
        )

        return plot_Gk.remote(
            model_label, model_proxy, Policy_ref, z_response_sample_max_z
        )

    work_grid = product(k_subsample, z_subsample)

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_plot_GkSource_work,
        compute_handler=None,
        store_handler=None,
        available_handler=None,
        validation_handler=None,
        post_handler=None,
        label_builder=None,
        create_batch_size=10,
        process_batch_size=10,
        notify_batch_size=50,
        notify_time_interval=120,
        title="GENERATING GkSource DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()


# establish a ShardedPool to orchestrate database access
with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_GkSource_data",
    prune_unvalidated=False,
) as pool:

    # get list of models we want to extract transfer functions for
    units = Mpc_units()
    model_list = build_model_list(pool, units)

    for model_data in model_list:
        run_pipeline(model_data)
