import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import ray
import seaborn as sns
from math import fabs
from matplotlib import pyplot as plt

from ComputeTargets import (
    BackgroundModel,
    QuadSource,
    QuadSourceValue,
    QuadSourceFunctions,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
)
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.ShardedPool import ShardedPool
from MetadataConcepts import tolerance
from RayWorkPool import RayWorkPool
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

VERSION_LABEL = "2024.1.1"

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

# establish a ShardedPool to orchestrate database access
with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_GkSource_data",
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

    k_array = ray.get(pool.read_wavenumber_table(units=units))

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

    def create_k_exit_work(k: wavenumber):
        return pool.object_get(
            wavenumber_exit_time,
            k=k,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        )

    # query wavenumber_exit_time objects corresponding to these k modes
    k_exit_queue = RayWorkPool(
        pool,
        k_array,
        task_builder=create_k_exit_work,
        compute_handler=None,
        store_handler=None,
        store_results=True,
        title="QUERY K_EXIT VALUES",
    )
    k_exit_queue.run()
    k_exit_times = k_exit_queue.results

    @ray.remote
    def plot_tensor_source(source: QuadSource):
        q_exit: wavenumber_exit_time = source._q_exit
        r_exit: wavenumber_exit_time = source._r_exit
        base_path = Path(args.output).resolve()

        sns.set_theme()

        values: List[QuadSourceValue] = source.values
        functions: QuadSourceFunctions = source.functions

        abs_source_points = [(value.z.z, fabs(value.source_term)) for value in values]
        abs_undiff_points = [(value.z.z, fabs(value.undiff_part)) for value in values]
        abs_diff_points = [(value.z.z, fabs(value.diff_part)) for value in values]
        abs_analytic_points = [
            (value.z.z, fabs(value.analytic_source_term)) for value in values
        ]

        abs_source_x, abs_source_y = zip(*abs_source_points)
        abs_undiff_x, abs_undiff_y = zip(*abs_undiff_points)
        abs_diff_x, abs_diff_y = zip(*abs_diff_points)
        abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)
        abs_spline_y = [fabs(functions.source(z)) for z in abs_source_x]

        if len(abs_source_x) > 0 and (
            any(y is not None and y > 0 for y in abs_source_y)
            or any(y is not None and y > 0 for y in abs_undiff_y)
            or any(y is not None and y > 0 for y in abs_diff_y)
            or any(y is not None and y > 0 for y in abs_analytic_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_source_x, abs_source_y, label="Numerical")
            ax.plot(abs_analytic_x, abs_analytic_y, label="Analytic", linestyle="--")
            ax.plot(abs_source_x, abs_spline_y, label="Spline")

            ax.axvline(q_exit.z_exit_subh_e3, linestyle="--", color="r")
            ax.axvline(q_exit.z_exit_subh_e5, linestyle="--", color="b")

            trans = ax.get_xaxis_transform()
            ax.text(
                q_exit.z_exit_subh_e3,
                0.05,
                "$q+3$ e-folds",
                transform=trans,
                fontsize="small",
            )
            ax.text(
                q_exit.z_exit_subh_e5,
                0.2,
                "$q+5$ e-folds",
                transform=trans,
                fontsize="small",
            )

            if r_exit.store_id != q_exit.store_id:
                ax.axvline(r_exit.z_exit_subh_e3, linestyle="--", color="g")
                ax.axvline(r_exit.z_exit_subh_e5, linestyle="--", color="m")

                ax.text(
                    r_exit.z_exit_subh_e3,
                    0.05,
                    "$r+3$ e-folds",
                    transform=trans,
                    fontsize="small",
                )
                ax.text(
                    r_exit.z_exit_subh_e5,
                    0.2,
                    "$r+5$ e-folds",
                    transform=trans,
                    fontsize="small",
                )

            ax.set_xlabel("redshift $z$")
            ax.set_ylabel("$T_k(z)$")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/full/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            ax.set_xlim(
                int(round(q_exit.z_exit_suph_e3 + 0.5, 0)),
                int(round(0.85 * q_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/q-zoom/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.pdf"
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
                    / f"plots/r-zoom/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.pdf"
                )
                fig_path.parents[0].mkdir(exist_ok=True, parents=True)
                fig.savefig(fig_path)

            plt.close()

        if len(abs_undiff_x) > 0:
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_source_x, abs_source_y, label="Numerical")
            ax.plot(abs_undiff_x, abs_undiff_y, label="$T_{k}$ part")
            ax.plot(abs_diff_x, abs_diff_y, label="$dT_{k}/dz$ part")

            trans = ax.get_xaxis_transform()
            ax.text(
                q_exit.z_exit_subh_e3,
                0.05,
                "$q+3$ e-folds",
                transform=trans,
                fontsize="small",
            )
            ax.text(
                q_exit.z_exit_subh_e5,
                0.2,
                "$q+5$ e-folds",
                transform=trans,
                fontsize="small",
            )

            if r_exit.store_id != q_exit.store_id:
                ax.axvline(r_exit.z_exit_subh_e3, linestyle="--", color="g")
                ax.axvline(r_exit.z_exit_subh_e5, linestyle="--", color="m")

                ax.text(
                    r_exit.z_exit_subh_e3,
                    0.05,
                    "$r+3$ e-folds",
                    transform=trans,
                    fontsize="small",
                )
                ax.text(
                    r_exit.z_exit_subh_e5,
                    0.2,
                    "$r+5$ e-folds",
                    transform=trans,
                    fontsize="small",
                )

            ax.set_xlabel("redshift $z$")
            ax.set_ylabel("$T_k(z)$")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/parts/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            plt.close()

        z_column = [value.z.z for value in values]
        source_column = [value.source_term for value in values]
        undiff_column = [value.undiff_part for value in values]
        diff_column = [value.diff_part for value in values]
        analytic_source_column = [value.analytic_source_term for value in values]
        analytic_undiff_column = [value.analytic_undiff_part for value in values]
        analytic_diff_column = [value.analytic_diff_part for value in values]
        source_spline = [functions.source(z) for z in z_column]
        spline_diff = [
            source_spline[i] - source_column[i] for i in range(len(z_column))
        ]
        spline_err = [spline_diff[i] / source_column[i] for i in range(len(z_column))]

        csv_path = (
            base_path
            / f"csv/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "z": z_column,
                "source_term": source_column,
                "undiff": undiff_column,
                "diff": diff_column,
                "analytic_source_term": analytic_source_column,
                "analytic_undiff": analytic_undiff_column,
                "analytic_diff": analytic_diff_column,
                "source_spline": source_spline,
                "spline_diff": spline_diff,
                "spline_err": spline_err,
            }
        )
        df.sort_values(by="z", ascending=False, inplace=True, ignore_index=True)
        df.to_csv(csv_path, header=True, index=False)

    def build_tensor_source_work(item):
        q, r = item
        q: wavenumber_exit_time
        r: wavenumber_exit_time

        payload_data = [
            {
                "solver_labels": [],
                "model": model,
                "k": k_mode,
                "z_sample": None,
                "z_init": None,
                "atol": atol,
                "rtol": rtol,
            }
            for k_mode in [q, r]
        ]

        Tq_ref, Tr_ref = [
            pool.object_get("TkNumericalIntegration", **payload)
            for payload in payload_data
        ]

        Tsource_ref = pool.object_get(
            "QuadSource", z_sample=None, q=q, Tq=Tq_ref, Tr=Tr_ref
        )

        return plot_tensor_source.remote(Tsource_ref)

    work_grid = list(itertools.combinations_with_replacement(k_exit_times, 2))

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
