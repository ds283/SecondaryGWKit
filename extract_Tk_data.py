import argparse
import sys
from datetime import datetime
from pathlib import Path
from random import sample
from typing import List, Optional

import pandas as pd
import ray
import seaborn as sns
from math import fabs
from matplotlib import pyplot as plt

from ComputeTargets import (
    TkNumericalIntegration,
    BackgroundModel,
    TkNumericalValue,
    ModelProxy,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
)
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
    "--output", default="Tk-out", type=str, help="specify folder for output files"
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

with ShardedPool(
    version_label="2024.1.1",
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_Tk_data",
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

    model_proxy = ModelProxy(model)

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

    # choose a subsample of k modes
    k_subsample: List[wavenumber_exit_time] = sample(
        list(k_exit_times), k=int(round(0.9 * len(k_exit_times) + 0.5, 0))
    )

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
    def add_z_labels(ax, Tk: TkNumericalIntegration, k_exit: wavenumber_exit_time):
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
    def plot_Tk(Tk: TkNumericalIntegration):
        if not Tk.available:
            return

        values: List[TkNumericalValue] = Tk.values
        base_path = Path(args.output).resolve()

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def safe_div(x: Optional[float], y: float) -> Optional[float]:
            if x is None:
                return None

            return x / y

        abs_T_points = [(value.z, safe_fabs(value.T)) for value in values]
        abs_analytic_T_rad_points = [
            (value.z, safe_fabs(value.analytic_T_rad)) for value in values
        ]
        abs_analytic_T_w_points = [
            (value.z, safe_fabs(value.analytic_T_w)) for value in values
        ]

        T_points = [(value.z, value.T) for value in values]
        analytic_T_rad_points = [(value.z, value.analytic_T_rad) for value in values]
        analytic_T_w_points = [(value.z, value.analytic_T_w) for value in values]

        abs_T_x, abs_T_y = zip(*abs_T_points)
        abs_analytic_T_rad_x, abs_analytic_T_rad_y = zip(*abs_analytic_T_rad_points)
        abs_analytic_T_w_x, abs_analytic_T_w_y = zip(*abs_analytic_T_w_points)

        T_x, T_y = zip(*T_points)
        analytic_T_rad_x, analytic_T_rad_y = zip(*analytic_T_rad_points)
        analytic_T_w_x, analytic_T_w_y = zip(*analytic_T_w_points)

        k_exit = Tk._k_exit

        sns.set_theme()

        if len(abs_T_x) > 0 and (
            any(y is not None and y > 0 for y in abs_T_y)
            or any(y is not None and y > 0 for y in abs_analytic_T_rad_y)
            or any(y is not None and y > 0 for y in abs_analytic_T_w_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(
                abs_T_x, abs_T_y, label="Numerical $T_k$", color="r", linestyle="solid"
            )
            ax.plot(
                abs_analytic_T_rad_x,
                abs_analytic_T_rad_y,
                label="Analytic $T_k$ [radiation]",
                color="g",
                linestyle="dashed",
            )
            ax.plot(
                abs_analytic_T_w_x,
                abs_analytic_T_w_y,
                label="Analytic $T_k$ [$w=w(z)$]",
                color="b",
                linestyle="dashdot",
            )

            add_z_labels(ax, Tk, k_exit)

            ax.set_xlabel("response redshift $z$")
            ax.set_ylabel("$G_k(z, z')$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/Tk/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            plt.close()

        if len(T_x) > 0 and (
            any(y is not None for y in T_y)
            or any(y is not None for y in analytic_T_rad_y)
            or any(y is not None for y in analytic_T_w_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(T_x, T_y, label="Numerical $T_k$", color="r", linestyle="solid")
            ax.plot(
                analytic_T_rad_x,
                analytic_T_rad_y,
                label="Analytic $T_k$ [radiation]",
                color="g",
                linestyle="dashed",
            )
            ax.plot(
                analytic_T_w_x,
                analytic_T_w_y,
                label="Analytic $T_k$ [$w=w(z)$]",
                color="b",
                linestyle="dashdot",
            )

            add_z_labels(ax, Tk, k_exit)

            ax.set_xlabel("response redshift $z$")
            ax.set_ylabel("$G_k(z, z')$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/Tk-linear/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            plt.close()

        z_response_column = [value.z for value in values]
        T_column = [value.T for value in values]
        Tprime_column = [value.Tprime for value in values]
        analytic_T_rad_column = [value.analytic_T_rad for value in values]
        analytic_Tprime_rad_column = [value.analytic_Tprime_rad for value in values]
        analytic_T_w_column = [value.analytic_T_rad for value in values]
        analytic_Tprime_w_column = [value.analytic_Tprime_rad for value in values]

        csv_path = (
            base_path / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "z_response": z_response_column,
                "T": T_column,
                "Tprime": Tprime_column,
                "analytic_T_rad": analytic_T_rad_column,
                "analytic_Tprime_rad": analytic_Tprime_rad_column,
                "analytic_T_w": analytic_T_w_column,
                "analytic_Tprime_w": analytic_Tprime_w_column,
            }
        )
        df.sort_values(
            by="z_response", ascending=False, inplace=True, ignore_index=True
        )
        df.to_csv(csv_path, header=True, index=False)

    def build_plot_Tk_work(k_exit: wavenumber_exit_time):
        query_payload = {
            "solver_labels": [],
            "model": model_proxy,
            "k": k_exit,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }

        GkS_ref = pool.object_get("TkNumericalIntegration", **query_payload)

        return plot_Tk.remote(GkS_ref)

    work_queue = RayWorkPool(
        pool,
        k_subsample,
        task_builder=build_plot_Tk_work,
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
        title="GENERATING TkNumerical DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
