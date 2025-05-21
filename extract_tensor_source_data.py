import argparse
import itertools
import sys
from datetime import datetime
from math import fabs
from pathlib import Path
from random import sample
from typing import List

import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from ComputeTargets import (
    BackgroundModel,
    QuadSource,
    QuadSourceValue,
    QuadSourceFunctions,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
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
    safe_fabs,
    set_loglog_axes,
    add_simple_plot_labels,
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
    "--output",
    default="tensor-source-out",
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
    label = f'{VERSION_LABEL}-jobname-extract_tensor_source_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )


def add_z_labels(ax, q_exit: wavenumber_exit_time, r_exit: wavenumber_exit_time):
    add_zexit_lines(ax, q_exit, "r", "b")
    if q_exit.store_id != r_exit.store_id:
        add_zexit_lines(ax, r_exit, "m", "c")


@ray.remote
def plot_tensor_source(model_label: str, source: QuadSource):
    q_exit: wavenumber_exit_time = source._q_exit
    r_exit: wavenumber_exit_time = source._r_exit

    base_path = Path(args.output).resolve()
    base_path = base_path / f"{model_label}"

    sns.set_theme()

    values: List[QuadSourceValue] = source.values
    functions: QuadSourceFunctions = source.functions

    abs_source_points = [(value.z.z, safe_fabs(value.source)) for value in values]
    abs_undiff_points = [(value.z.z, safe_fabs(value.undiff)) for value in values]
    abs_diff_points = [(value.z.z, safe_fabs(value.diff)) for value in values]
    abs_analytic_rad_points = [
        (value.z.z, safe_fabs(value.analytic_source_rad)) for value in values
    ]
    abs_analytic_w_points = [
        (value.z.z, safe_fabs(value.analytic_source_w)) for value in values
    ]

    abs_source_x, abs_source_y = zip(*abs_source_points)
    abs_undiff_x, abs_undiff_y = zip(*abs_undiff_points)
    abs_diff_x, abs_diff_y = zip(*abs_diff_points)

    abs_analytic_rad_x, abs_analytic_rad_y = zip(*abs_analytic_rad_points)
    abs_analytic_w_x, abs_analytic_w_y = zip(*abs_analytic_w_points)

    abs_spline_y = [fabs(functions.source(z)) for z in abs_source_x]

    if len(abs_source_x) > 0 and (
        any(y is not None and y > 0 for y in abs_source_y)
        or any(y is not None and y > 0 for y in abs_undiff_y)
        or any(y is not None and y > 0 for y in abs_diff_y)
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
        ax.plot(
            abs_analytic_w_x,
            abs_analytic_w_y,
            label="Analytic [$w=w(z)$]",
            linestyle="dashdot",
        )
        ax.plot(abs_source_x, abs_spline_y, label="Spline")

        add_z_labels(ax, q_exit=q_exit, r_exit=r_exit)
        add_simple_plot_labels(
            ax, q_exit=q_exit, r_exit=r_exit, model_label=model_label
        )

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$T_k(z)$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/full/q={q_exit.k.k_inv_Mpc:.5g}-r={r_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}-r-serial={r_exit.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        ax.set_xlim(
            int(round(q_exit.z_exit_suph_e3 + 0.5, 0)),
            int(round(0.85 * q_exit.z_exit_subh_e5 + 0.5, 0)),
        )

        fig_path = (
            base_path
            / f"plots/q-zoom/q={q_exit.k.k_inv_Mpc:.5g}-r={r_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}-r-serial={r_exit.store_id}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)

        if q_exit.store_id != r_exit.store_id:
            ax.set_xlim(
                int(round(r_exit.z_exit_suph_e3 + 0.5, 0)),
                int(round(0.85 * r_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/r-zoom/q={q_exit.k.k_inv_Mpc:.5g}-r={r_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}-r-serial={r_exit.store_id}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

        plt.close()

    if len(abs_undiff_x) > 0:
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(abs_source_x, abs_source_y, label="Numerical")
        ax.plot(abs_undiff_x, abs_undiff_y, label="$T_{k}$ part", linestyle="dashed")
        ax.plot(abs_diff_x, abs_diff_y, label="$dT_{k}/dz$ part", linestyle="dashdot")

        add_z_labels(ax, q_exit=q_exit, r_exit=r_exit)
        add_simple_plot_labels(
            ax, q_exit=q_exit, r_exit=r_exit, model_label=model_label
        )

        ax.set_xlabel("redshift $z$")
        ax.set_ylabel("$T_k(z)$")

        set_loglog_axes(ax)

        fig_path = (
            base_path
            / f"plots/parts/q={q_exit.k.k_inv_Mpc:.5g}-r={r_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}-r-serial={r_exit.store_id}.pdf"
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
        / f"csv/q={q_exit.k.k_inv_Mpc:.5g}-r={r_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}-r-serial={r_exit.store_id}.csv"
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
    atol, rtol = ray.get(
        [
            pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
        ]
    )

    # array of k-modes matching the SOURCE k-grid
    source_k_array = ray.get(pool.read_wavenumber_table(units=units, is_source=True))

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

    # choose a subsample of the SOURCE k modes
    k_subsample: List[wavenumber_exit_time] = sample(
        list(source_k_exit_times),
        k=int(round(0.9 * len(source_k_exit_times) + 0.5, 0)),
    )

    model = ray.get(
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

    def build_tensor_source_work(item):
        q, r = item
        q: wavenumber_exit_time
        r: wavenumber_exit_time

        Tsource_ref = pool.object_get(
            "QuadSource",
            model=model_proxy,
            z_sample=None,
            q=q,
            r=r,
        )

        return plot_tensor_source.remote(model_label, Tsource_ref)

    work_grid = list(itertools.combinations_with_replacement(k_subsample, 2))

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_tensor_source_work,
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
        title="EXTRACT TENSOR SOURCE DATA",
        store_results=False,
    )
    work_queue.run()


# establish a ShardedPool to orchestrate database access
with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_tensor_source_data",
    prune_unvalidated=False,
) as pool:

    # get list of models we want to extract transfer functions for
    units = Mpc_units()
    model_list = build_model_list(pool, units)

    for model_data in model_list:
        run_pipeline(model_data)
