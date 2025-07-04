import argparse
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from random import sample
from typing import List

import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from ComputeTargets import (
    GkNumericIntegration,
    BackgroundModel,
    ModelProxy,
    GkNumericValue,
)
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
    add_zexit_lines,
    set_loglog_axes,
    set_loglinear_axes,
    safe_fabs,
    add_simple_plot_labels,
    LOOSE_DASHED,
)
from model_list import build_model_list

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

VERSION_LABEL = "2025.1.1"

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


@ray.remote
def plot_Gk(model_label: str, Gk: GkNumericIntegration):
    if not Gk.available:
        return

    values: List[GkNumericValue] = Gk.values
    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

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

        ax.plot(abs_G_x, abs_G_y, label="Numeric $G_k$", color="r", linestyle="solid")
        ax.plot(
            abs_analytic_G_rad_x,
            abs_analytic_G_rad_y,
            label="Analytic $G_k$ [radiation]",
            color="g",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            abs_analytic_G_w_x,
            abs_analytic_G_w_y,
            label="Analytic $G_k$ [$w=w(z)$]",
            color="b",
            linestyle=LOOSE_DASHED,
        )

        add_zexit_lines(ax, k_exit)
        add_simple_plot_labels(
            ax, z_source=z_source, k_exit=k_exit, model_label=model_label
        )

        ax.set_xlabel("response redshift $z$")
        ax.set_ylabel("$G_k(z, z')$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/Gk/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zsource={z_source.z:.5g}-z-serial={z_source.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        if z_source.z >= k_exit.z_exit_subh_e5:
            ax.set_xlim(
                int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/Gk-reentry-zoom/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zsource={z_source.z:.5g}-z-serial={z_source.store_id}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

        plt.close()

    if len(G_x) > 0 and (
        any(y is not None for y in G_y) or any(y is not None for y in analytic_G_rad_y)
    ):
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(G_x, G_y, label="Numeric $G_k$", color="r", linestyle="solid")
        ax.plot(
            analytic_G_rad_x,
            analytic_G_rad_y,
            label="Analytic $G_k$ [radiation]",
            color="g",
            linestyle=LOOSE_DASHED,
        )
        ax.plot(
            analytic_G_w_x,
            analytic_G_w_y,
            label="Analytic $G_k$ [$w=w(z)$]",
            color="b",
            linestyle=LOOSE_DASHED,
        )

        add_zexit_lines(ax, k_exit)
        add_simple_plot_labels(
            ax, z_source=z_source, k_exit=k_exit, model_label=model_label
        )

        ax.set_xlabel("response redshift $z$")
        ax.set_ylabel("$G_k(z, z')$")

        set_loglinear_axes(ax)

        fig_path = (
            base_path
            / f"plots/Gk-linear/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zsource={z_source.z:.5g}-z-serial={z_source.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        if z_source.z >= k_exit.z_exit_subh_e5:
            ax.set_xlim(
                int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/Gk-linear-reentry-zoom/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zsource={z_source.z:.5g}-z-serial={z_source.store_id}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

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
        / f"csv/k={k_exit.k.k_inv_Mpc:.5g}-k-serial={k_exit.store_id}/zsource={z_source.z:.5g}-z-serial={z_source.store_id}.csv"
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
    df.sort_values(by="z_response", ascending=False, inplace=True, ignore_index=True)
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

    model_proxy = ModelProxy(model)

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

    # choose a subsample of about 10% of the SOURCE redshifts
    z_subsample: List[redshift] = sample(
        list(z_source_sample), k=int(round(0.1 * len(z_source_sample) + 0.5, 0))
    )

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

        GkS_ref = pool.object_get("GkNumericIntegration", **query_payload)

        return plot_Gk.remote(model_label, GkS_ref)

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
        title="GENERATING GkNumeric DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()


with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_Gk_data",
    prune_unvalidated=False,
) as pool:

    # get list of models we want to extract transfer functions for
    units = Mpc_units()
    model_list = build_model_list(pool, units)

    for model_data in model_list:
        run_pipeline(model_data)
