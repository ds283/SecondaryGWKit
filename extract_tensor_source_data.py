import argparse
import itertools
import sys
from datetime import datetime
from math import fabs
from pathlib import Path
from random import sample
from typing import List, Optional

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
    label = f'{VERSION_LABEL}-jobname-extract_tensor_source_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

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
    job_name="extract_tensor_source_data",
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

    # array of k-modes matching the SOURCE k-grid
    source_k_array = ray.get(pool.read_wavenumber_table(units=units, is_source=True))

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

    # choose a subsample of the SOURCE k modes
    k_subsample: List[wavenumber_exit_time] = sample(
        list(source_k_exit_times),
        k=int(round(0.9 * len(source_k_exit_times) + 0.5, 0)),
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
    def add_z_labels(
        ax,
        source: QuadSource,
        q_exit: wavenumber_exit_time,
        r_exit: wavenumber_exit_time,
    ):
        def add_z_lines(k_exit: wavenumber_exit_time, col1: str, col2: str):
            ax.axvline(
                k_exit.z_exit_subh_e3, linestyle=(0, (1, 1)), color=col1
            )  # dotted
            ax.axvline(
                k_exit.z_exit_subh_e5, linestyle=(0, (1, 1)), color=col1
            )  # dotted
            ax.axvline(
                k_exit.z_exit_suph_e3, linestyle=(0, (1, 1)), color=col1
            )  # dotted
            ax.axvline(
                k_exit.z_exit, linestyle=(0, (3, 1, 1, 1)), color=col2
            )  # densely dashdotted

            trans = ax.get_xaxis_transform()
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_suph_e3,
                0.75,
                "$q-3$ e-folds",
                transform=trans,
                fontsize="x-small",
                color="b",
            )
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e3,
                0.85,
                "$q+3$ e-folds",
                transform=trans,
                fontsize="x-small",
                color="b",
            )
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e5,
                0.75,
                "$q+5$ e-folds",
                transform=trans,
                fontsize="x-small",
                color="b",
            )
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit,
                0.92,
                "$q$ re-entry",
                transform=trans,
                fontsize="x-small",
                color="r",
            )

        add_z_lines(q_exit, "r", "b")
        if q_exit.store_id != r_exit.store_id:
            add_z_lines(r_exit, "m", "c")

    @ray.remote
    def plot_tensor_source(source: QuadSource):
        q_exit: wavenumber_exit_time = source._q_exit
        r_exit: wavenumber_exit_time = source._r_exit

        base_path = Path(args.output).resolve()

        sns.set_theme()

        values: List[QuadSourceValue] = source.values
        functions: QuadSourceFunctions = source.functions

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def safe_div(x: Optional[float], y: float) -> Optional[float]:
            if x is None:
                return None

            return x / y

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

            add_z_labels(ax, source=q_exit, q_exit=q_exit, r_exit=r_exit)

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("$T_k(z)$")

            set_loglog_axes(ax)

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
            ax.plot(
                abs_undiff_x, abs_undiff_y, label="$T_{k}$ part", linestyle="dashed"
            )
            ax.plot(
                abs_diff_x, abs_diff_y, label="$dT_{k}/dz$ part", linestyle="dashdot"
            )

            add_z_labels(ax, source=q_exit, q_exit=q_exit, r_exit=r_exit)

            ax.set_xlabel("redshift $z$")
            ax.set_ylabel("$T_k(z)$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/parts/q-serial={source.q.store_id}-q={q_exit.k.k_inv_Mpc:.5g}-r-serial={source.r.store_id}-r={r_exit.k.k_inv_Mpc:.5g}.pdf"
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

        return plot_tensor_source.remote(Tsource_ref)

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
