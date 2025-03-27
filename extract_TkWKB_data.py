import argparse
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
    TkNumericalIntegration,
    BackgroundModel,
    TkNumericalValue,
    ModelProxy,
    TkWKBIntegration,
    TkWKBValue,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
)
from CosmologyConcepts.wavenumber import wavenumber_exit_time_array
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.ShardedPool import ShardedPool
from RayTools.RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
)

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
    "--output", default="TkWKB-out", type=str, help="specify folder for output files"
)
args = parser.parse_args()

if args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

VERSION_LABEL = "2025.1.1"

# instantiate a Datastore actor: this runs on its own node, and acts as a broker between
# ourselves and the database.
# For performance reasons, we want all database activity to run on this node.
# For one thing, this lets us use transactions efficiently.

profile_agent = None
if args.profile_db is not None:
    label = f'{VERSION_LABEL}-jobname-extract_Tk_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

    profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
        db_name=args.profile_db,
        timeout=args.db_timeout,
        label=label,
    )

with ShardedPool(
    version_label=VERSION_LABEL,
    db_name=args.database,
    timeout=args.db_timeout,
    profile_agent=profile_agent,
    job_name="extract_TkWKB_data",
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
            pool.object_get("tolerance", tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_REL_TOLERANCE),
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
    if len(source_k_exit_times) > 20:
        k_subsample: List[wavenumber_exit_time] = sample(
            list(source_k_exit_times),
            k=int(round(0.5 * len(source_k_exit_times) + 0.5, 0)),
        )
    else:
        k_subsample: List[wavenumber_exit_time] = list(source_k_exit_times)

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

    def add_k_labels(ax, k_exit):
        ax.text(
            0.0,
            1.1,
            f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    @ray.remote
    def plot_Tk(Tk_numerical: TkNumericalIntegration, Tk_WKB: TkWKBIntegration):
        k_exit = Tk_numerical._k_exit

        sns.set_theme()
        fig = plt.figure()
        ax = plt.gca()

        z_column = []
        T_column = []
        abs_T_column = []
        analytic_T_rad_column = []
        abs_analytic_T_rad_column = []
        analytic_T_w_column = []
        abs_analytic_T_w_column = []
        theta_column = []
        friction_column = []
        H_ratio_column = []
        omega_WKB_sq_column = []
        WKB_criterion_column = []
        type_column = []

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def safe_div(x: Optional[float], y: float) -> Optional[float]:
            if x is None:
                return None

            return x / y

        if Tk_numerical.available:
            values: List[TkNumericalValue] = Tk_numerical.values

            numerical_points = [(value.z.z, safe_fabs(value.T)) for value in values]
            analytic_rad_points = [
                (value.z, safe_fabs(value.analytic_T_rad)) for value in values
            ]
            analytic_w_points = [
                (value.z, safe_fabs(value.analytic_T_w)) for value in values
            ]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_rad_x, analytic_rad_y = zip(*analytic_rad_points)
            analytic_w_x, analytic_w_y = zip(*analytic_w_points)

            if len(numerical_x) > 0 and (
                any(y is not None and y > 0 for y in numerical_y)
                or any(y is not None and y > 0 for y in analytic_rad_y)
                or any(y is not None and y > 0 for y in analytic_w_y)
            ):
                ax.plot(
                    numerical_x,
                    numerical_y,
                    label="Numerical $T_k$",
                    linestyle="solid",
                )
                ax.plot(
                    analytic_rad_x,
                    analytic_rad_y,
                    label="Analytic $T_k$ [radiation]",
                    linestyle="dashed",
                )
                ax.plot(
                    analytic_w_x,
                    analytic_w_y,
                    label="Analytic $T_k$ [$w=w(z)$]",
                    linestyle="dashdot",
                )

            z_column.extend(value.z.z for value in values)
            T_column.extend(value.T for value in values)
            abs_T_column.extend(safe_fabs(value.T) for value in values)
            analytic_T_rad_column.extend(value.analytic_T_rad for value in values)
            abs_analytic_T_rad_column.extend(
                safe_fabs(value.analytic_T_rad) for value in values
            )
            analytic_T_w_column.extend(value.analytic_T_w for value in values)
            abs_analytic_T_w_column.extend(value.analytic_T_w for value in values)
            theta_column.extend(None for _ in range(len(values)))
            friction_column.extend(None for _ in range(len(values)))
            H_ratio_column.extend(None for _ in range(len(values)))
            omega_WKB_sq_column.extend(value.omega_WKB_sq for value in values)
            WKB_criterion_column.extend(value.WKB_criterion for value in values)
            type_column.extend(0 for _ in range(len(values)))

        theta_x = None
        theta_y = None
        friction_x = None
        friction_y = None
        if Tk_WKB.available:
            values: List[TkWKBValue] = Tk_WKB.values

            numerical_points = [(value.z.z, safe_fabs(value.T_WKB)) for value in values]
            analytic_rad_points = [
                (value.z, safe_fabs(value.analytic_T_rad)) for value in values
            ]
            analytic_w_points = [
                (value.z, safe_fabs(value.analytic_T_w)) for value in values
            ]
            theta_points = [(value.z.z, safe_fabs(value.theta)) for value in values]
            friction_points = [
                (value.z.z, safe_fabs(value.friction)) for value in values
            ]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_rad_x, analytic_rad_y = zip(*analytic_rad_points)
            analytic_w_x, analytic_w_y = zip(*analytic_w_points)
            theta_x, theta_y = zip(*theta_points)
            friction_x, friction_y = zip(*friction_points)

            if len(numerical_x) > 0 and (
                any(y is not None and y > 0 for y in numerical_y)
                or any(y is not None and y > 0 for y in analytic_rad_y)
                or any(y is not None and y > 0 for y in analytic_w_y)
            ):
                ax.plot(
                    numerical_x,
                    numerical_y,
                    label="WKB $T_k$",
                    linestyle="solid",
                )
                ax.plot(
                    analytic_rad_x,
                    analytic_rad_y,
                    label="Analytic $T_k$ [radiation]",
                    linestyle="dashed",
                )
                ax.plot(
                    analytic_w_x,
                    analytic_w_y,
                    label="Analytic $T_k$ [$w=w(z)$]",
                    linestyle="dashdot",
                )

            z_column.extend(value.z.z for value in values)
            T_column.extend(value.T_WKB for value in values)
            abs_T_column.extend(safe_fabs(value.T_WKB) for value in values)
            analytic_T_rad_column.extend(value.analytic_T_rad for value in values)
            abs_analytic_T_rad_column.extend(
                safe_fabs(value.analytic_T_rad) for value in values
            )
            analytic_T_w_column.extend(value.analytic_T_w for value in values)
            abs_analytic_T_w_column.extend(value.analytic_T_w for value in values)
            theta_column.extend(value.theta for value in values)
            friction_column.extend(value.friction for value in values)
            H_ratio_column.extend(value.H_ratio for value in values)
            omega_WKB_sq_column.extend(value.omega_WKB_sq for value in values)
            WKB_criterion_column.extend(value.WKB_criterion for value in values)
            type_column.extend(1 for _ in range(len(values)))

        add_z_labels(ax, Tk_WKB, k_exit)
        add_k_labels(ax, k_exit)

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$T_k(z)$")

        set_loglog_axes(ax)

        base_path = Path(args.output).resolve()
        fig_path = (
            base_path
            / f"plots/full-range/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        ax.set_xlim(
            int(round(k_exit.z_exit_suph_e3 + 0.5, 0)),
            int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
        )
        fig_path = (
            base_path
            / f"plots/zoom-matching/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        min_z = min(numerical_x)
        max_z = max(numerical_x)
        ax.set_xlim(
            min(int(round(min_z * 100.0 + 0.5, 0)), max_z),
            max(int(round(min_z * 0.9 - 0.5, 0)), 0.01),
        )

        fig_path = (
            base_path
            / f"plots/zoom-lowz/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))

        plt.close()

        if theta_x is not None and theta_y is not None:
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(theta_x, theta_y, label="WKB phase $\\theta$")

            add_z_labels(ax, Tk_WKB, k_exit)

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("WKB phase $\\theta$")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/theta/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

        if friction_x is not None and friction_y is not None:
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(friction_x, friction_y, label="WKB friction")

            add_z_labels(ax, Tk_WKB, k_exit)

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("WKB friction")

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/friction/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

        csv_path = (
            base_path / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "redshift": z_column,
                "T": T_column,
                "abs_T": abs_T_column,
                "analytic_T_rad": analytic_T_rad_column,
                "abs_analytic_T_rad": abs_analytic_T_rad_column,
                "analytic_T_w": analytic_T_w_column,
                "abs_analytic_T_w": abs_analytic_T_w_column,
                "theta": theta_column,
                "H_ratio": H_ratio_column,
                "omega_WKB_sq": omega_WKB_sq_column,
                "WKB_criterion": WKB_criterion_column,
                "type": type_column,
            }
        )
        df.sort_values(by="redshift", ascending=False, inplace=True, ignore_index=True)
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

        TkNumerical_ref = pool.object_get("TkNumericalIntegration", **query_payload)
        TkWKB_ref = pool.object_get("TkWKBIntegration", **query_payload)

        return plot_Tk.remote(TkNumerical_ref, TkWKB_ref)

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
        title="GENERATING TkNumerical/TkWKB DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
