import argparse
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from random import sample
from typing import List, Optional

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from math import fabs
from matplotlib import pyplot as plt

from ComputeTargets import (
    GkNumericalIntegration,
    BackgroundModel,
    ModelProxy,
    GkNumericalValue,
)
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
    "--output", default="Gk-out", type=str, help="specify folder for output files"
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
    label = f'{VERSION_LABEL}-jobname-extract_Gk_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )

with ShardedPool(
    version_label="2024.1.1",
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_Gk_data",
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

    ## UTILITY FUNCTIONS

    def convert_to_wavenumbers(k_sample_set):
        return pool.object_get(
            "wavenumber",
            payload_data=[{"k_inv_Mpc": k, "units": units} for k in k_sample_set],
        )

    def convert_to_redshifts(z_sample_set):
        return pool.object_get(
            "redshift", payload_data=[{"z": z} for z in z_sample_set]
        )

    # array of k-modes matching the SOURCE k-grid
    source_k_array = ray.get(
        convert_to_wavenumbers(np.logspace(np.log10(1e3), np.log10(5e7), 10))
    )

    # array of k-modes matching the RESPONSE k-grid
    response_k_array = ray.get(
        convert_to_wavenumbers(np.logspace(np.log10(1e3), np.log10(5e7), 10))
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

    DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100
    DEFAULT_RESPONSE_SAMPLES_PER_LOG10_Z = 8
    DEFAULT_ZEND = 0.1

    source_z_grid = k_exit_earliest.populate_z_sample(
        outside_horizon_efolds=5,
        samples_per_log10z=DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z,
        z_end=DEFAULT_ZEND,
    )
    response_z_grid = k_exit_earliest.populate_z_sample(
        outside_horizon_efolds=5,
        samples_per_log10z=DEFAULT_RESPONSE_SAMPLES_PER_LOG10_Z,
        z_end=DEFAULT_ZEND,
    )

    # embed these redshift list into the database
    z_source_array = ray.get(convert_to_redshifts(source_z_grid))
    z_response_array = ray.get(convert_to_redshifts(response_z_grid))

    z_source_sample = redshift_array(z_array=z_source_array)
    z_response_sample = redshift_array(z_array=z_response_array)

    # choose a subsample of SOURCE redshifts
    z_subsample: List[redshift] = sample(
        list(z_source_sample), k=int(round(0.12 * len(z_source_sample) + 0.5, 0))
    )

    model: BackgroundModel = ray.get(
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
    def add_z_labels(ax, Gk: GkNumericalIntegration, k_exit: wavenumber_exit_time):
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
    def plot_Gk(Gk: GkNumericalIntegration):
        if not Gk.available:
            return

        values: List[GkNumericalValue] = Gk.values
        base_path = Path(args.output).resolve()

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def safe_div(x: Optional[float], y: float) -> Optional[float]:
            if x is None:
                return None

            return x / y

        abs_G_points = [(value.z, safe_fabs(value.G)) for value in values]
        abs_analytic_G_rad_points = [
            (value.z, safe_fabs(value.analytic_G_rad)) for value in values
        ]
        abs_analytic_G_w_points = [
            (value.z, safe_fabs(value.analytic_G_w)) for value in values
        ]

        G_points = [(value.z, value.G) for value in values]
        analytic_G_rad_points = [(value.z, value.analytic_G_rad) for value in values]
        analytic_G_w_points = [(value.z, value.analytic_G_w) for value in values]

        abs_G_x, abs_G_y = zip(*abs_G_points)
        abs_analytic_G_rad_x, abs_analytic_G_rad_y = zip(*abs_analytic_G_rad_points)
        abs_analytic_G_w_x, abs_analytic_G_w_y = zip(*abs_analytic_G_w_points)

        G_x, G_y = zip(*G_points)
        analytic_G_rad_x, analytic_G_rad_y = zip(*analytic_G_rad_points)
        analytic_G_w_x, analytic_G_w_y = zip(*analytic_G_w_points)

        k_exit = Gk._k_exit
        z_source = Gk.z_source

        sns.set_theme()

        if len(abs_G_x) > 0 and (
            any(y is not None and y > 0 for y in abs_G_y)
            or any(y is not None and y > 0 for y in abs_analytic_G_rad_y)
            or any(y is not None and y > 0 for y in abs_analytic_G_w_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(
                abs_G_x, abs_G_y, label="Numerical $G_k$", color="r", linestyle="solid"
            )
            ax.plot(
                abs_analytic_G_rad_x,
                abs_analytic_G_rad_y,
                label="Analytic $G_k$ [radiation]",
                color="g",
                linestyle="dashed",
            )
            ax.plot(
                abs_analytic_G_w_x,
                abs_analytic_G_w_y,
                label="Analytic $G_k$ [$w=w(z)$]",
                color="b",
                linestyle="dashdot",
            )

            add_z_labels(ax, Gk, k_exit)

            ax.set_xlabel("response redshift $z$")
            ax.set_ylabel("$G_k(z, z')$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/Gk/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            if z_source.z >= k_exit.z_exit_subh_e5:
                ax.set_xlim(
                    int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                    int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
                )

                fig_path = (
                    base_path
                    / f"plots/Gk-reentry-zoom/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
                )
                fig_path.parents[0].mkdir(exist_ok=True, parents=True)
                fig.savefig(fig_path)

            plt.close()

        if len(G_x) > 0 and (
            any(y is not None for y in G_y)
            or any(y is not None for y in analytic_G_rad_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(G_x, G_y, label="Numerical $G_k$", color="r", linestyle="solid")
            ax.plot(
                analytic_G_rad_x,
                analytic_G_rad_y,
                label="Analytic $G_k$ [radiation]",
                color="g",
                linestyle="dashed",
            )
            ax.plot(
                analytic_G_w_x,
                analytic_G_w_y,
                label="Analytic $G_k$ [$w=w(z)$]",
                color="b",
                linestyle="dashdot",
            )

            add_z_labels(ax, Gk, k_exit)

            ax.set_xlabel("response redshift $z$")
            ax.set_ylabel("$G_k(z, z')$")

            set_loglinear_axes(ax)

            fig_path = (
                base_path
                / f"plots/Gk-linear/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            if z_source.z >= k_exit.z_exit_subh_e5:
                ax.set_xlim(
                    int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                    int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
                )

                fig_path = (
                    base_path
                    / f"plots/Gk-linear-reentry-zoom/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
                )
                fig_path.parents[0].mkdir(exist_ok=True, parents=True)
                fig.savefig(fig_path)

            plt.close()

        z_response_column = [value.z for value in values]
        G_column = [value.G for value in values]
        Gprime_column = [value.Gprime for value in values]
        analytic_G_rad_column = [value.analytic_G_rad for value in values]
        analytic_Gprime_rad_column = [value.analytic_Gprime_rad for value in values]
        analytic_G_w_column = [value.analytic_G_rad for value in values]
        analytic_Gprime_w_column = [value.analytic_Gprime_rad for value in values]
        omega_WKB_sq_column = [value.omega_WKB_sq for value in values]
        WKB_criterion_column = [value.WKB_criterion for value in values]

        csv_path = (
            base_path
            / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "z_response": z_response_column,
                "G": G_column,
                "Gprime": Gprime_column,
                "analytic_G_rad": analytic_G_rad_column,
                "analytic_Gprime_rad": analytic_Gprime_rad_column,
                "analytic_G_w": analytic_G_w_column,
                "analytic_Gprime_w": analytic_Gprime_w_column,
                "omega_WKB_sq": omega_WKB_sq_column,
                "WKB_criterion": WKB_criterion_column,
            }
        )
        df.sort_values(
            by="z_response", ascending=False, inplace=True, ignore_index=True
        )
        df.to_csv(csv_path, header=True, index=False)

    def build_plot_Gk_work(item):
        k_exit, z_source = item
        k_exit: wavenumber_exit_time
        z_source: redshift

        query_payload = {
            "solver_labels": [],
            "model": model_proxy,
            "k": k_exit,
            "z_source": z_source,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }

        GkS_ref = pool.object_get("GkNumericalIntegration", **query_payload)

        return plot_Gk.remote(GkS_ref)

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
        title="GENERATING GkNumerical DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
