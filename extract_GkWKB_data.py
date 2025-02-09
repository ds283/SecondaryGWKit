import argparse
import sys
from datetime import datetime
from itertools import product
from math import fabs
from pathlib import Path
from random import sample
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns

from ComputeTargets import (
    GkWKBIntegration,
    BackgroundModel,
    GkNumericalIntegration,
    GkNumericalValue,
    GkWKBValue,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
    redshift_array,
    redshift,
)
from CosmologyConcepts.wavenumber import wavenumber_exit_time_array
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.ShardedPool import ShardedPool
from MetadataConcepts import tolerance
from RayTools.RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_TIMEOUT = 60

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
    "--output", default="GkWKB-out", type=str, help="specify folder for output files"
)
args = parser.parse_args()

if args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

VERSION_LABEL = "2024.1.1"

# instantiate a Datastore actor: this runs on its own node, and acts as a broker between
# ourselves and the dataabase.
# For performance reasons, we want all database activity to run on this node.
# For one thing, this lets us use transactions efficiently.

profile_agent = None
if args.profile_db is not None:
    label = f'{VERSION_LABEL}-jobname-extract_GkWKB_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )

# establish a ShardedPool to orchestrate database access
with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_GkWKB_data",
    prune_unvalidated=False,
) as pool:

    # set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
    units = Mpc_units()
    params = Planck2018()
    LambdaCDM_Planck2018 = ray.get(
        pool.object_get(LambdaCDM, params=params, units=units)
    )

    # build absolute and relative tolerances
    atol, rtol = ray.get(
        [
            pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
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
            cosmology=LambdaCDM_Planck2018,
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
        k=int(round(0.9 * len(response_k_exit_times) + 0.5, 0)),
    )

    z_source_array = ray.get(pool.read_redshift_table(is_source=True))
    z_source_sample = redshift_array(z_array=z_source_array)

    z_response_array = ray.get(pool.read_redshift_table(is_response=True))
    z_response_sample = redshift_array(z_array=z_response_array)

    # choose a subsample of SOURCE redshifts
    z_subsample: List[redshift] = sample(
        list(z_source_sample), k=int(round(0.15 * len(z_source_sample) + 0.5, 0))
    )

    model = ray.get(
        pool.object_get(
            BackgroundModel,
            solver_labels=[],
            cosmology=LambdaCDM_Planck2018,
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

    def set_loglinear_axes(ax):
        ax.set_xscale("log")
        ax.legend(loc="best")
        ax.grid(True)
        ax.xaxis.set_inverted(True)

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
    def add_z_labels(ax, Gk: GkWKBIntegration, k_exit: wavenumber_exit_time):
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

    @ray.remote
    def plot_Gk(Gk_numerical: GkNumericalIntegration, Gk_WKB: GkWKBIntegration):
        k_exit = Gk_numerical._k_exit
        z_source = Gk_numerical.z_source

        sns.set_theme()
        fig = plt.figure()
        ax = plt.gca()

        z_column = []
        G_column = []
        abs_G_column = []
        analytic_G_rad_column = []
        abs_analytic_G_rad_column = []
        analytic_G_w_column = []
        abs_analytic_G_w_column = []
        theta_column = []
        H_ratio_column = []
        omega_WKB_sq_column = []
        WKB_criterion_column = []
        type_column = []

        if Gk_numerical.available:
            values: List[GkNumericalValue] = Gk_numerical.values

            numerical_points = [(value.z.z, fabs(value.G)) for value in values]
            analytic_rad_points = [
                (value.z.z, fabs(value.analytic_G_rad)) for value in values
            ]
            analytic_w_points = [
                (value.z.z, fabs(value.analytic_G_w)) for value in values
            ]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_rad_x, analytic_rad_y = zip(*analytic_rad_points)
            analytic_w_x, analytic_w_y = zip(*analytic_w_points)

            if len(numerical_x) > 0 and (
                any(y is not None and y > 0 for y in numerical_y)
                or any(y is not None and y > 0 for y in analytic_rad_y)
                or any(y is not None and y > 0 for y in analytic_w_y)
            ):
                ax.plot(numerical_x, numerical_y, label="Numerical $G_k$")
                ax.plot(
                    analytic_rad_x,
                    analytic_rad_y,
                    label="Analytic $G_k$ [numeric, radiation]",
                    linestyle="dashed",
                )
                ax.plot(
                    analytic_w_x,
                    analytic_w_y,
                    label="Analytic $G_k$ [numeric, $w=w(z)$]",
                    linestyle="dashdot",
                )

            z_column.extend(value.z.z for value in values)
            G_column.extend(value.G for value in values)
            abs_G_column.extend(fabs(value.G) for value in values)
            analytic_G_rad_column.extend(value.analytic_G_rad for value in values)
            abs_analytic_G_rad_column.extend(
                fabs(value.analytic_G_rad) for value in values
            )
            analytic_G_w_column.extend(value.analytic_G_w for value in values)
            abs_analytic_G_w_column.extend(fabs(value.analytic_G_w) for value in values)
            theta_column.extend(None for _ in range(len(values)))
            H_ratio_column.extend(None for _ in range(len(values)))
            omega_WKB_sq_column.extend(value.omega_WKB_sq for value in values)
            WKB_criterion_column.extend(value.WKB_criterion for value in values)
            type_column.extend(0 for _ in range(len(values)))

        theta_x = None
        theta_y = None
        if Gk_WKB.available:
            values: List[GkWKBValue] = Gk_WKB.values

            numerical_points = [(value.z.z, fabs(value.G_WKB)) for value in values]
            analytic_rad_points = [
                (value.z.z, fabs(value.analytic_G_rad)) for value in values
            ]
            analytic_w_points = [
                (value.z.z, fabs(value.analytic_G_w)) for value in values
            ]
            theta_points = [(value.z.z, fabs(value.theta)) for value in values]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_rad_x, analytic_rad_y = zip(*analytic_rad_points)
            analytic_w_x, analytic_w_y = zip(*analytic_w_points)
            theta_x, theta_y = zip(*theta_points)

            if len(numerical_x) > 0 and (
                any(y is not None and y > 0 for y in numerical_y)
                or any(y is not None and y > 0 for y in analytic_rad_y)
                or any(y is not None and y > 0 for y in analytic_w_y)
            ):
                ax.plot(numerical_x, numerical_y, label="WKB $G_k$")
                ax.plot(
                    analytic_rad_x,
                    analytic_rad_y,
                    label="Analytic $G_k$ [WKB, radiation]]",
                    linestyle="dashed",
                )
                ax.plot(
                    analytic_w_x,
                    analytic_w_y,
                    label="Analytic $G_k$ [WKB, $w=w(z)$]",
                    linestyle="dashdot",
                )

            z_column.extend(value.z.z for value in values)
            G_column.extend(value.G_WKB for value in values)
            abs_G_column.extend(fabs(value.G_WKB) for value in values)
            analytic_G_rad_column.extend(value.analytic_G_rad for value in values)
            abs_analytic_G_rad_column.extend(
                fabs(value.analytic_G_rad) for value in values
            )
            analytic_G_w_column.extend(value.analytic_G_w for value in values)
            abs_analytic_G_w_column.extend(fabs(value.analytic_G_w) for value in values)
            theta_column.extend(value.theta for value in values)
            H_ratio_column.extend(value.H_ratio for value in values)
            omega_WKB_sq_column.extend(value.omega_WKB_sq for value in values)
            WKB_criterion_column.extend(value.WKB_criterion for value in values)
            type_column.extend(1 for _ in range(len(values)))

        add_z_labels(ax, Gk_WKB, k_exit)

        ax.set_xlabel("response redshift $z$")
        ax.set_ylabel("$G_k(z_{\\text{source}}, z_{\\text{response}})$")

        set_loglog_axes(ax)

        base_path = Path(args.output).resolve()
        fig_path = (
            base_path
            / f"plots/full-range/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        ax.set_xlim(
            int(round(k_exit.z_exit_suph_e3 + 0.5, 0)),
            int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
        )
        fig_path = (
            base_path
            / f"plots/zoom-matching/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        plt.close()

        if theta_x is not None and theta_y is not None:
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(theta_x, theta_y, label="WKB phase $\\theta$")

            add_z_labels(ax, Gk_WKB, k_exit)

            ax.set_xlabel("response redshift $z$")
            ax.set_ylabel("WKB phase $\\theta$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/theta/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

        csv_path = (
            base_path
            / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "redshift": z_column,
                "G": G_column,
                "abs_G": abs_G_column,
                "analytic_G_rad": analytic_G_rad_column,
                "abs_analytic_G_rad": abs_analytic_G_rad_column,
                "analytic_G_w": analytic_G_w_column,
                "abs_analytic_G_w": abs_analytic_G_w_column,
                "theta": theta_column,
                "H_ratio": H_ratio_column,
                "omega_WKB_sq": omega_WKB_sq_column,
                "WKB_criterion": WKB_criterion_column,
                "type": type_column,
            }
        )
        df.sort_values(by="redshift", ascending=False, inplace=True, ignore_index=True)
        df.to_csv(csv_path, header=True, index=False)

    def build_plot_Gk_work(item):
        k_exit, z_source = item
        k_exit: wavenumber_exit_time
        z_source: redshift

        # query the list of GkNumberical and GkWKB data for this k, z_source combination
        query_payload = {
            "solver_labels": [],
            "model": model_proxy,
            "k": k_exit,
            "z_source": z_source,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }

        GkN_ref = pool.object_get("GkNumericalIntegration", **query_payload)
        GkW_ref = pool.object_get("GkWKBIntegration", **query_payload)

        return plot_Gk.remote(GkN_ref, GkW_ref)

    work_grid = product(k_subsample, z_subsample)

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_plot_Gk_work,
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
        title="GENERATING GkWKB DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
