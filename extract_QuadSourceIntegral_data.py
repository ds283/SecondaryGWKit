import argparse
import itertools
import json
import sys
from datetime import datetime
from math import fabs, pi, sqrt
from pathlib import Path
from random import sample
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns
from ray import ObjectRef

from ComputeTargets import (
    BackgroundModel,
    ModelProxy,
    QuadSourceIntegral,
    GkSourceProxy,
    GkSourcePolicyData,
    QuadSource,
    GkSource,
    GkSourceValue,
    QuadSourceValue,
    QuadSourceFunctions,
)
from ComputeTargets.BackgroundModel import ModelFunctions
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
from RayTools.RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_QUADRATURE_ATOL,
    DEFAULT_QUADRATURE_RTOL,
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
    "--output",
    default="QuadSourceIntegral-out",
    type=str,
    help="specify folder for output files",
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
    label = f'{VERSION_LABEL}-jobname-extract_QuadSourceIntegral_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )


def set_loglog_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


TEXT_DISPLACEMENT_MULTIPLIER = 0.85


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
def add_z_labels(
    ax, GkPolicy: GkSourcePolicyData, k_exit: wavenumber_exit_time, label: str
):
    ax.axvline(k_exit.z_exit_subh_e3, linestyle=(0, (1, 1)), color="b")  # dotted
    ax.axvline(k_exit.z_exit_subh_e5, linestyle=(0, (1, 1)), color="b")  # dotted
    ax.axvline(k_exit.z_exit_suph_e3, linestyle=(0, (1, 1)), color="b")  # dotted
    ax.axvline(
        k_exit.z_exit, linestyle=(0, (3, 1, 1, 1)), color="r"
    )  # densely dashdotted
    trans = ax.get_xaxis_transform()
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_suph_e3,
        0.75,
        f"{label}$-3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e3,
        0.85,
        f"{label}$+3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e5,
        0.75,
        f"{label}$+5$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit,
        0.92,
        f"{label} re-entry",
        transform=trans,
        fontsize="x-small",
        color="r",
    )

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


def add_k_labels(ax, k_exit, q_exit, r_exit):
    ax.text(
        0.0,
        1.1,
        f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        0.33,
        1.1,
        f"$q$ =  {q_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        0.66,
        1.1,
        f"$r$ = {r_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
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
):
    if z_min_quad is not None and z_max_quad is not None:
        ax.text(
            0.0,
            1.01,
            f"numeric: [{z_min_quad.z:.5g}, {z_max_quad.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    if z_min_WKB_quad is not None and z_max_WKB_quad is not None:
        ax.text(
            0.33,
            1.05,
            f"WKB numeric: [{z_min_WKB_quad.z:.5g}, {z_max_WKB_quad.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    if z_min_Levin is not None and z_max_Levin is not None:
        ax.text(
            0.66,
            1.01,
            f"WKB Levin: [{z_min_Levin.z:.5g}, {z_max_Levin.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def add_Gk_labels(ax, obj: GkSourcePolicyData):
    fns: GkSourceFunctions = obj.functions
    if fns.type is not None:
        ax.text(
            0.0,
            1.05,
            f"Type: {fns.type}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.quality is not None:
        ax.text(
            0.5,
            1.05,
            f"Quality: {fns.quality}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.WKB_region is not None:
        ax.text(
            0.0,
            1.1,
            f"WKB region: ({fns.WKB_region[0]:.3g}, {fns.WKB_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.numerical_region is not None:
        ax.text(
            0.5,
            1.1,
            f"Numeric region: ({fns.numerical_region[0]:.3g}, {fns.numerical_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.phase is not None:
        ax.text(
            0.8,
            1.05,
            f"Chunks: {fns.phase.num_chunks}",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def safe_fabs(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None

    return fabs(x)


def safe_div(x: Optional[float], y: float) -> Optional[float]:
    if x is None:
        return None

    return x / y


@ray.remote
def plot_QuadSourceIntegral(
    model_label: str,
    k_exit: wavenumber_exit_time,
    q_exit: wavenumber_exit_time,
    r_exit: wavenumber_exit_time,
    policy: GkSourcePolicyData,
    integral_data: List[QuadSourceIntegral],
):
    if len(integral_data) <= 1:
        return

    integral_data.sort(key=lambda x: x.z_response.z, reverse=True)

    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

    z_min_quad = None
    z_max_quad = None
    z_min_WKB_quad = None
    z_max_WKB_quad = None
    z_min_Levin = None
    z_max_Levin = None

    abs_points = []
    abs_analytic_points = []

    for obj in integral_data:
        obj: QuadSourceIntegral

        abs_points.append(
            (
                obj.z_response.z,
                safe_fabs(obj.total),
            )
        )
        abs_analytic_points.append((obj.z_response.z, safe_fabs(obj.analytic_rad)))

        if obj.numeric_quad is not None and fabs(obj.numeric_quad) > 1e-25:
            if z_min_quad is None or obj.z_response < z_min_quad:
                z_min_quad = obj.z_response
            if z_max_quad is None or obj.z_response > z_max_quad:
                z_max_quad = obj.z_response

        if obj.WKB_quad is not None and fabs(obj.WKB_quad) > 1e-25:
            if z_min_WKB_quad is None or obj.z_response < z_min_WKB_quad:
                z_min_WKB_quad = obj.z_response
            if z_max_WKB_quad is None or obj.z_response > z_max_WKB_quad:
                z_max_WKB_quad = obj.z_response

        if obj.WKB_Levin is not None and fabs(obj.WKB_Levin) > 1e-25:
            if z_min_Levin is None or obj.z_response < z_min_Levin:
                z_min_Levin = obj.z_response
            if z_max_Levin is None or obj.z_response > z_max_Levin:
                z_max_Levin = obj.z_response

    abs_x, abs_y = zip(*abs_points)
    abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)

    sns.set_theme()
    if len(abs_x) > 0 and (
        any(y is not None and y > 0 for y in abs_y)
        or any(y is not None and y > 0 for y in abs_analytic_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        if z_min_quad is not None and z_max_quad is not None:
            ax.axvspan(xmin=z_min_quad.z, xmax=z_max_quad.z, color="b", alpha=0.15)

        if z_min_WKB_quad is not None and z_max_WKB_quad is not None:
            ax.axvspan(
                xmin=z_min_WKB_quad.z,
                xmax=z_max_WKB_quad.z,
                color="r",
                alpha=0.15,
            )

        if z_min_Levin is not None and z_max_Levin is not None:
            ax.axvspan(xmin=z_min_Levin.z, xmax=z_max_Levin.z, color="g", alpha=0.15)

        ax.plot(abs_x, abs_y, label="Numeric", color="r", linestyle="solid")
        ax.plot(
            abs_analytic_x,
            abs_analytic_y,
            label="Analytic",
            color="g",
            linestyle="dashed",
        )

        add_z_labels(ax, policy, k_exit, "$k$")
        add_k_labels(ax, k_exit, q_exit, r_exit)
        add_region_labels(
            ax,
            z_min_quad,
            z_max_quad,
            z_min_WKB_quad,
            z_max_WKB_quad,
            z_min_Levin,
            z_max_Levin,
        )

        ax.set_xlabel("response redshift $z$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/integral.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

    csv_folder = (
        base_path
        / f"csv/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}"
    )

    payload = {
        "k_inv_Mpc": k_exit.k.k_inv_Mpc,
        "q_inv_Mpc": q_exit.k.k_inv_Mpc,
        "r_inv_Mpc": r_exit.k.k_inv_Mpc,
        "k_z_exit": k_exit.z_exit,
        "q_z_exit": q_exit.z_exit,
        "r_z_exit": r_exit.z_exit,
    }

    payload_path = csv_folder / "integral_metadata.csv"
    payload_path.parents[0].mkdir(exist_ok=True, parents=True)
    with open(payload_path, "w", newline="") as f:
        json.dump(payload, f, indent=4, sort_keys=True)

    z_response_column = [obj.z_response.z for obj in integral_data]
    z_source_max_column = [obj.z_source_max.z for obj in integral_data]
    eta_response_column = [obj.eta_response for obj in integral_data]
    eta_source_max_column = [obj.eta_source_max for obj in integral_data]
    total_column = [obj.total for obj in integral_data]
    numeric_quad_column = [obj.numeric_quad for obj in integral_data]
    WKB_quad_column = [obj.WKB_quad for obj in integral_data]
    WKB_Levin_column = [obj.WKB_Levin for obj in integral_data]
    analytic_rad_column = [obj.analytic_rad for obj in integral_data]
    metadata = [obj.metadata for obj in integral_data]

    csv_path = csv_folder / "integral_data.csv"
    csv_path.parents[0].mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame.from_dict(
        {
            "z_response": z_response_column,
            "z_source_max": z_source_max_column,
            "eta_response": eta_response_column,
            "eta_source_max": eta_source_max_column,
            "total": total_column,
            "numeric_quad": numeric_quad_column,
            "WKB_quad": WKB_quad_column,
            "WKB_Levin": WKB_Levin_column,
            "analytic_rad": analytic_rad_column,
            "metadata": metadata,
        }
    )
    df.sort_values(by="z_response", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(csv_path, header=True, index=False)


@ray.remote
def plot_QuadSourceIntegrand(
    model_label: str,
    k_exit: wavenumber_exit_time,
    q_exit: wavenumber_exit_time,
    r_exit: wavenumber_exit_time,
    model_proxy: ModelProxy,
    policy: GkSourcePolicyData,
    source: QuadSource,
    z_source_max: redshift,
):
    if not policy.available:
        print(f"** GkPolicy not available")
        return

    Gk: GkSource = policy._source_proxy.get()
    if not Gk.available:
        print(f"** GkSource not available")
        return

    if source.z_sample.max.z < z_source_max.z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"supplied quadratic source term has maximum z_source={source.z_sample.max.z:.5g}, but required value is at least z_source={z_source_max.z:.5g}"
        )

    if Gk.z_sample.max.z < z_source_max.z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"supplied Gk has maximum z_source={Gk.z_sample.max.z:.5g}, but required value is at least z_source={z_source_max.z:.5g}"
        )

    if Gk.k.store_id != k_exit.k.store_id:
        raise RuntimeError(
            f"supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={k_exit.k.store_id}])"
        )

    if source.q.store_id != q_exit.k.store_id:
        raise RuntimeError(
            f"supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={q_exit.k.store_id}])"
        )

    if source.r.store_id != r_exit.k.store_id:
        raise RuntimeError(
            f"supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={r_exit.k.store_id}])"
        )

    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

    model: BackgroundModel = model_proxy.get()
    model_f: ModelFunctions = model.functions

    Gk_values: List[GkSourceValue] = Gk.values
    source_values: List[QuadSourceValue] = source.values

    Gk_value_dict = {
        value.z_source.store_id: value
        for value in Gk_values
        if value.z_source <= z_source_max
    }
    source_value_dict = {
        value.z.store_id: value for value in source_values if value.z <= z_source_max
    }

    Gk_value_ids = set(Gk_value_dict.keys())
    source_value_ids = set(source_value_dict.keys())

    common_ids = Gk_value_ids.intersection(source_value_ids)

    def integrand(store_id: int):
        Gk_value = Gk_value_dict[store_id]
        source_value = source_value_dict[store_id]

        Green_analytic = Gk_value.analytic_G_rad
        if Gk_value.has_numeric:
            Green = Gk_value.numeric.G
        elif Gk_value.has_WKB:
            Green = Gk_value.WKB.G_WKB
        else:
            raise RuntimeError("GkSource value has neither numeric nor WKB content")

        source = source_value.source
        source_analytic = source_value.analytic_source_rad

        H = model_f.Hubble(Gk_value.z_source.z)
        H_sq = H * H

        payload = {
            "z": Gk_value.z_source,
            "integrand": Green * source / H_sq,
            "analytic_integrand": Green_analytic * source_analytic / H_sq,
        }

        if Gk_value.has_WKB:
            sin_ampl = Gk_value.WKB.sin_coeff * sqrt(
                Gk_value.WKB.H_ratio / sqrt(Gk_value.omega_WKB_sq)
            )
            payload["Levin_f"] = sin_ampl * source / H_sq
            payload["analytic_Levin_f"] = sin_ampl * source_analytic / H_sq

        return payload

    integrand_points = [integrand(z_id) for z_id in common_ids]
    integrand_points.sort(key=lambda x: x["z"].z, reverse=True)

    abs_integrand_points = [
        (x["z"].z, safe_fabs(x["integrand"])) for x in integrand_points
    ]
    abs_analytic_points = [
        (x["z"].z, safe_fabs(x["analytic_integrand"])) for x in integrand_points
    ]
    abs_Levin_f_points = [
        (x["z"].z, safe_fabs(x["Levin_f"])) for x in integrand_points if "Levin_f" in x
    ]
    abs_analytic_Levin_f_points = [
        (x["z"].z, safe_fabs(x["analytic_Levin_f"]))
        for x in integrand_points
        if "analytic_Levin_f" in x
    ]

    abs_integrand_x, abs_integrand_y = zip(*abs_integrand_points)
    abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)

    abs_Levin_f_x, abs_Levin_f_y = zip(*abs_Levin_f_points)
    abs_analytic_Levin_f_x, abs_analytic_Levin_f_y = zip(*abs_analytic_Levin_f_points)

    sns.set_theme()

    if len(abs_integrand_x) > 0 and (
        any(y is not None and y > 0 for y in abs_integrand_y)
        or any(y is not None and y > 0 for y in abs_analytic_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            abs_integrand_x,
            abs_integrand_y,
            label="Numeric",
            color="r",
            linestyle="solid",
        )
        ax.plot(
            abs_analytic_x,
            abs_analytic_y,
            label="Analytic",
            color="b",
            linestyle="dashed",
        )

        add_z_labels(ax, policy, k_exit, "$k$")
        add_Gk_labels(ax, policy)

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/integrand.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        ax.set_xlim(
            int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
            int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
        )

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/integrand_reentry_zoom.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

    if len(abs_Levin_f_x) > 0 and (
        any(y is not None and y > 0 for y in abs_Levin_f_y)
        or any(y is not None and y > 0 for y in abs_analytic_Levin_f_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(
            abs_Levin_f_x,
            abs_Levin_f_y,
            label="Numeric",
            color="r",
            linestyle="solid",
        )
        ax.plot(
            abs_analytic_Levin_f_x,
            abs_analytic_Levin_f_y,
            label="Analytic",
            color="b",
            linestyle="dashed",
        )

        add_z_labels(ax, policy, k_exit, "$k$")
        add_Gk_labels(ax, policy)

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/Levin_function.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

    z_source_column = [x["z"].z for x in integrand_points]
    integrand_column = [x["integrand"] for x in integrand_points]
    analytic_column = [x["analytic_integrand"] for x in integrand_points]
    Levin_f_column = [
        x["Levin_f"] if hasattr(x, "Levin_f") else None for x in integrand_points
    ]
    analytic_Levin_f_column = [
        x["analytic_Levin_f"] if hasattr(x, "analytic_Levin_f") else None
        for x in integrand_points
    ]

    csv_path = (
        base_path
        / f"csv/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/integrand.csv"
    )
    csv_path.parents[0].mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame.from_dict(
        {
            "z_source": z_source_column,
            "integrand": integrand_column,
            "analytic_integrand": analytic_column,
            "Levin_f": Levin_f_column,
            "analytic_Levin_f": analytic_Levin_f_column,
        }
    )
    df.sort_values(by="z_source", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(csv_path, header=True, index=False)


@ray.remote
def plot_Gk(
    model_label: str,
    k_exit: wavenumber_exit_time,
    q_exit: wavenumber_exit_time,
    r_exit: wavenumber_exit_time,
    model_proxy: ModelProxy,
    GkPolicy: GkSourcePolicyData,
):
    if not GkPolicy.available:
        print(f"** GkPolicy not available")
        return

    Gk: GkSource = GkPolicy._source_proxy.get()
    if not Gk.available:
        print(f"** GkSource not available")
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
            color="g",
            linestyle="dashed",
        )
        ax.plot(
            abs_analytic_G_rad_x,
            abs_analytic_G_rad_y,
            label="Analytic $G_k$ [$w=w(z)$]",
            color="b",
            linestyle="dashed",
        )
        ax.plot(
            abs_G_spline_x,
            abs_G_spline_y,
            label="Spline $G_k$",
            color="r",
            linestyle="dashdot",
        )
        ax.plot(
            abs_G_WKB_spline_x,
            abs_G_WKB_spline_y,
            label="Spline WKB $G_k$",
            color="b",
            linestyle="dashdot",
        )

        add_z_labels(ax, GkPolicy, k_exit, "$k$")
        add_Gk_labels(ax, GkPolicy)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$G_k(z, z') / [(1+z') H(z')^2 ]$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/GkSource.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        if z_response.z <= k_exit.z_exit_suph_e3:
            ax.set_xlim(
                int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/GkSource-reentry-zoom.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

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
        / f"csv/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/GkSource.csv"
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


@ray.remote
def plot_tensor_source(
    model_label: str,
    k_exit: wavenumber_exit_time,
    q_exit: wavenumber_exit_time,
    r_exit: wavenumber_exit_time,
    policy: GkSourcePolicyData,
    source: QuadSource,
):
    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

    sns.set_theme()

    values: List[QuadSourceValue] = source.values
    functions: QuadSourceFunctions = source.functions

    abs_source_points = [(value.z.z, safe_fabs(value.source)) for value in values]
    abs_analytic_rad_points = [
        (value.z.z, safe_fabs(value.analytic_source_rad)) for value in values
    ]

    abs_source_x, abs_source_y = zip(*abs_source_points)
    abs_analytic_rad_x, abs_analytic_rad_y = zip(*abs_analytic_rad_points)

    abs_spline_y = [fabs(functions.source(z)) for z in abs_source_x]

    if len(abs_source_x) > 0 and (
        any(y is not None and y > 0 for y in abs_source_y)
        or any(y is not None and y > 0 for y in abs_analytic_rad_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(abs_source_x, abs_source_y, label="Numeric")
        ax.plot(
            abs_analytic_rad_x,
            abs_analytic_rad_y,
            label="Analytic [radiation]",
            linestyle="dashed",
        )
        ax.plot(abs_source_x, abs_spline_y, label="Spline")

        add_z_labels(ax, policy, q_exit, "$q$")
        add_k_labels(ax, k_exit, q_exit, r_exit)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$T_k(z)$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/quad_source.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        ax.set_xlim(
            int(round(q_exit.z_exit_suph_e3 + 0.5, 0)),
            int(round(0.85 * q_exit.z_exit_subh_e5 + 0.5, 0)),
        )

        fig_path = (
            base_path
            / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/quad_source_q_reentry_zoom.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        if q_exit.store_id != r_exit.store_id:
            plt.close()

            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_source_x, abs_source_y, label="Numeric")
            ax.plot(
                abs_analytic_rad_x,
                abs_analytic_rad_y,
                label="Analytic [radiation]",
                linestyle="dashed",
            )
            ax.plot(abs_source_x, abs_spline_y, label="Spline")

            ax.set_xlim(
                int(round(r_exit.z_exit_suph_e3 + 0.5, 0)),
                int(round(0.85 * r_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            add_z_labels(ax, policy, r_exit, "$r$")
            add_k_labels(ax, k_exit, q_exit, r_exit)

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("$T_k(z)$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/quad_source_r_reentry_zoom.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

        plt.close()

    z_column = [value.z.z for value in values]
    source_column = [value.source for value in values]
    undiff_column = [value.undiff for value in values]
    diff_column = [value.diff for value in values]
    analytic_source_rad_column = [value.analytic_source_rad for value in values]
    analytic_undiff_rad_column = [value.analytic_undiff_rad for value in values]
    analytic_diff_rad_column = [value.analytic_diff_rad for value in values]
    analytic_source_w_column = [value.analytic_source_w for value in values]
    analytic_undiff_w_column = [value.analytic_undiff_w for value in values]
    analytic_diff_w_column = [value.analytic_diff_w for value in values]
    source_spline = [functions.source(z) for z in z_column]
    spline_diff = [source_spline[i] - source_column[i] for i in range(len(z_column))]
    spline_err = [spline_diff[i] / source_column[i] for i in range(len(z_column))]

    csv_path = (
        base_path
        / f"csv/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}/quad_source.csv"
    )
    csv_path.parents[0].mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame.from_dict(
        {
            "z": z_column,
            "source": source_column,
            "undiff": undiff_column,
            "diff": diff_column,
            "analytic_source_rad": analytic_source_rad_column,
            "analytic_undiff_rad": analytic_undiff_rad_column,
            "analytic_diff_rad": analytic_diff_rad_column,
            "analytic_source_w": analytic_source_w_column,
            "analytic_undiff_w": analytic_undiff_w_column,
            "analytic_diff_w": analytic_diff_w_column,
            "source_spline": source_spline,
            "spline_diff": spline_diff,
            "spline_err": spline_err,
        }
    )
    df.sort_values(by="z", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(csv_path, header=True, index=False)


def run_pipeline(model_data):
    model_label = model_data["label"]
    model_cosmology = model_data["cosmology"]

    print(f"\n>> RUNNING PIPELINE FOR MODEL {model_label}")

    # build absolute and relative tolerances
    atol, rtol, quad_atol, quad_rtol = ray.get(
        [
            pool.object_get("tolerance", tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_REL_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_ATOL),
            pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_RTOL),
        ]
    )

    # array of k-modes matching the SOURCE and RESPONSE k-grids
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

    full_k_exit_times = source_k_exit_times + response_k_exit_times

    k_exit_earliest: wavenumber_exit_time = full_k_exit_times.max

    # choose a subsample of the RESPONSE k modes
    k_subsample: List[wavenumber_exit_time] = sample(
        list(response_k_exit_times),
        k=int(round(0.4 * len(response_k_exit_times) + 0.5, 0)),
    )

    qs_range = list(itertools.combinations_with_replacement(source_k_exit_times, 2))
    qs_subsample: List[Tuple[wavenumber_exit_time, wavenumber_exit_time]] = sample(
        qs_range,
        k=int(round(0.4 * len(qs_range) + 0.5, 0)),
    )

    z_source_array = ray.get(pool.read_redshift_table(is_source=True))
    z_source_sample = redshift_array(z_array=z_source_array)

    z_response_array = ray.get(pool.read_redshift_table(is_response=True))
    z_response_sample = redshift_array(z_array=z_response_array)

    # choose a subsample of RESPONSE redshifts
    # z_subsample: List[redshift] = sample(
    #     list(z_response_sample), k=int(round(0.12 * len(z_response_sample) + 0.5, 0))
    # )

    z_source_integral_max_z = z_source_sample[10]
    z_source_integral_response_sample = z_response_sample.truncate(
        z_source_integral_max_z, keep="lower-strict"
    )

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

    GkSource_policy_1pt5, GkSource_policy_5pt0 = ray.get(
        [
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-Levin"-Levin-threshold="1.5"',
                Levin_threshold=1.5,
                numeric_policy="maximize_Levin",
            ),
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-Levin"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize_Levin",
            ),
        ]
    )

    def build_plot_QuadSourceIntegral_work(item):
        k_exit, qr_pair = item
        q_exit, r_exit = qr_pair

        k_exit: wavenumber_exit_time
        q_exit: wavenumber_exit_time
        r_exit: wavenumber_exit_time

        Integral_ref: ObjectRef = pool.object_read_batch(
            ObjectClass="QuadSourceIntegral",
            shard_key={"k": k_exit},
            model=model_proxy,
            policy=GkSource_policy_1pt5,
            k=k_exit,
            q=q_exit,
            r=r_exit,
            z_response=None,
            z_source_max=None,
            atol=quad_atol,
            rtol=quad_rtol,
        )

        GkSource_ref: ObjectRef = pool.object_get(
            "GkSource",
            model=model_proxy,
            k=k_exit,
            z_response=z_source_integral_response_sample.min,
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

        Tsource_ref = pool.object_get(
            "QuadSource",
            model=model_proxy,
            z_sample=None,
            q=q_exit,
            r=r_exit,
        )

        return [
            plot_QuadSourceIntegral.remote(
                model_label, k_exit, q_exit, r_exit, Policy_ref, Integral_ref
            ),
            plot_QuadSourceIntegrand.remote(
                k_exit,
                q_exit,
                r_exit,
                model_proxy,
                Policy_ref,
                Tsource_ref,
                z_source_integral_max_z,
            ),
            plot_Gk.remote(
                model_label, k_exit, q_exit, r_exit, model_proxy, Policy_ref
            ),
            plot_tensor_source.remote(
                model_label, k_exit, q_exit, r_exit, Policy_ref, Tsource_ref
            ),
        ]

    work_grid = itertools.product(k_subsample, qs_subsample)

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_plot_QuadSourceIntegral_work,
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
        title="GENERATING QuadSourceIntegral DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()


# establish a ShardedPool to orchestrate database access
with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_QuadSourceIntegral_data",
    prune_unvalidated=False,
) as pool:

    # get list of models we want to extract transfer functions for
    units = Mpc_units()
    model_list = build_model_list(pool, units)

    for model_data in model_list:
        run_pipeline(model_data)
