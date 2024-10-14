import argparse
import sys
from itertools import product
from pathlib import Path
from random import sample
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns
from math import fabs

from ComputeTargets import (
    BackgroundModel,
    GkSource,
    GkSourceValue,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
    redshift_array,
    redshift,
)
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
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
    "--output", default="GkSource-out", type=str, help="specify folder for output files"
)
args = parser.parse_args()

if args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

# instantiate a Datastore actor: this runs on its own node, and acts as a broker between
# ourselves and the dataabase.
# For performance reasons, we want all database activity to run on this node.
# For one thing, this lets us use transactions efficiently.
with ShardedPool(
    version_label="2024.1.1",
    db_name=args.database,
    timeout=args.db_timeout,
    profile_db=args.profile_db,
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

    k_array = ray.get(pool.read_wavenumber_table(units=units))
    z_array = ray.get(pool.read_redshift_table())
    z_sample = redshift_array(z_array=z_array)

    model: BackgroundModel = ray.get(
        pool.object_get(
            BackgroundModel,
            solver_labels=[],
            cosmology=LambdaCDM_Planck2018,
            z_sample=z_sample,
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

    # choose a subsample of source redshifts
    z_subsample: List[redshift] = sample(
        list(z_sample), k=int(round(0.12 * len(z_sample) + 0.5, 0))
    )

    @ray.remote
    def plot_Gk(Gk: GkSource):
        if not Gk.available:
            return

        k_exit: wavenumber_exit_time = Gk._k_exit
        values: List[GkSourceValue] = Gk.values
        base_path = Path(args.output).resolve()

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def safe_div(x: Optional[float], y: float) -> Optional[float]:
            if x is None:
                return None

            return x / y

        abs_G_points = [
            (
                value.z_source.z,
                safe_div(
                    safe_fabs(value.numeric.G),
                    (
                        (1.0 + value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                    ),
                ),
            )
            for value in values
        ]
        abs_G_WKB_points = [
            (
                value.z_source.z,
                safe_div(
                    safe_fabs(value.WKB.G_WKB),
                    (
                        (1.0 + value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                    ),
                ),
            )
            for value in values
        ]
        abs_analytic_points = [
            (
                value.z_source.z,
                safe_div(
                    safe_fabs(value.analytic_G),
                    (
                        (1.0 + value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                        # * model.functions.Hubble(value.z_source.z)
                    ),
                ),
            )
            for value in values
        ]

        theta_points = [(value.z_source.z, value.WKB.theta) for value in values]
        raw_theta_points = [(value.z_source.z, value.WKB.raw_theta) for value in values]
        abs_theta_points = [
            (value.z_source.z, safe_fabs(value.WKB.theta)) for value in values
        ]
        abs_raw_theta_points = [
            (value.z_source.z, safe_fabs(value.WKB.raw_theta)) for value in values
        ]
        abs_sin_coeff_points = [
            (value.z_source.z, safe_fabs(value.WKB.sin_coeff)) for value in values
        ]
        abs_cos_coeff_points = [
            (value.z_source.z, safe_fabs(value.WKB.cos_coeff)) for value in values
        ]

        abs_G_x, abs_G_y = zip(*abs_G_points)
        abs_G_WKB_x, abs_G_WKB_y = zip(*abs_G_WKB_points)
        abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)

        theta_x, theta_y = zip(*theta_points)
        raw_theta_x, raw_theta_y = zip(*raw_theta_points)
        abs_theta_x, abs_theta_y = zip(*abs_theta_points)
        abs_raw_theta_x, abs_raw_theta_y = zip(*abs_raw_theta_points)
        abs_sin_coeff_x, abs_sin_coeff_y = zip(*abs_sin_coeff_points)
        abs_cos_coeff_x, abs_cos_coeff_y = zip(*abs_cos_coeff_points)

        k_exit = Gk._k_exit
        z_response = Gk.z_response

        sns.set_theme()

        if len(abs_G_x) > 0 and (
            any(y is not None and y > 0 for y in abs_G_y)
            or any(y is not None and y > 0 for y in abs_G_WKB_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_G_x, abs_G_y, label="Numerical $G_k$")
            ax.plot(abs_G_WKB_x, abs_G_WKB_y, label="WKB $G_k$")
            ax.plot(
                abs_analytic_x,
                abs_analytic_y,
                label="Analytic $G_k$",
                linestyle="--",
            )

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("$G_k(z_{\\text{source}}, z_{\\text{response}}) / (1+z')$")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/Gk/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zresponse={z_response.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            plt.close()

        if len(abs_theta_x) > 0 and (
            any(y is not None and y > 0 for y in abs_theta_y)
            or any(y is not None and y > 0 for y in abs_raw_theta_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_theta_x, abs_theta_y, label="WKB phase $\\theta$")
            ax.plot(
                abs_raw_theta_x,
                abs_raw_theta_y,
                label="Raw WKB phase $\\theta$",
            )

            ax.axvline(k_exit.z_exit_subh_e3, linestyle="--", color="r")
            ax.axvline(k_exit.z_exit_subh_e5, linestyle="--", color="b")

            trans = ax.get_xaxis_transform()
            ax.text(
                k_exit.z_exit_subh_e3,
                0.9,
                "$+3$ e-folds",
                transform=trans,
                fontsize="small",
            )
            ax.text(
                k_exit.z_exit_subh_e5,
                0.75,
                "$+5$ e-folds",
                transform=trans,
                fontsize="small",
            )

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("WKB phase $\\theta$")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/theta-log/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zsource={z_response.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            plt.close()

        if len(theta_x) > 0 and (
            any(y is not None for y in theta_y)
            or any(y is not None for y in raw_theta_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(theta_x, theta_y, label="WKB phase $\\theta$")
            ax.plot(raw_theta_x, raw_theta_y, label="Raw WKB phase $\\theta$")

            ax.axvline(k_exit.z_exit_subh_e3, linestyle="--", color="r")
            ax.axvline(k_exit.z_exit_subh_e5, linestyle="--", color="b")

            trans = ax.get_xaxis_transform()
            ax.text(
                k_exit.z_exit_subh_e3,
                0.9,
                "$+3$ e-folds",
                transform=trans,
                fontsize="small",
            )
            ax.text(
                k_exit.z_exit_subh_e5,
                0.75,
                "$+5$ e-folds",
                transform=trans,
                fontsize="small",
            )

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("WKB phase $\\theta$")

            ax.set_xscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/theta-linear/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zsource={z_response.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            plt.close()

        if len(abs_sin_coeff_x) > 0 and (
            any(y is not None and y > 0 for y in abs_sin_coeff_y)
            or any(y is not None and y > 0 for y in abs_cos_coeff_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(abs_sin_coeff_x, abs_sin_coeff_y, label="$\\sin$ coefficient")
            ax.plot(abs_cos_coeff_x, abs_cos_coeff_y, label="$\\cos coefficient")

            ax.set_xlabel("source redshift $z$")
            ax.set_ylabel("coefficient")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend(loc="best")
            ax.grid(True)
            ax.xaxis.set_inverted(True)

            fig_path = (
                base_path
                / f"plots/coeffs/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zsource={z_response.z:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
            plt.close()

        z_source_column = [value.z_source.z for value in values]
        G_column = [value.numeric.G for value in values]
        Gprime_column = [value.numeric.Gprime for value in values]
        G_WKB_column = [value.WKB.G_WKB for value in values]
        theta_column = [value.WKB.theta for value in values]
        raw_theta_column = [value.WKB.raw_theta for value in values]
        H_ratio_column = [value.WKB.H_ratio for value in values]
        sin_coeff_column = [value.WKB.sin_coeff for value in values]
        cos_coeff_column = [value.WKB.cos_coeff for value in values]
        omega_WKB_sq_column = [value.omega_WKB_sq for value in values]
        WKB_criterion_column = [value.WKB_criterion for value in values]
        analytic_G_column = [value.analytic_G for value in values]
        analytic_Gprime_column = [value.analytic_Gprime for value in values]

        csv_path = (
            base_path
            / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zsource={z_response.z:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "z_source": z_source_column,
                "G": G_column,
                "Gprime": Gprime_column,
                "G_WKB": G_WKB_column,
                "theta": theta_column,
                "raw_theta": raw_theta_column,
                "H_ratio": H_ratio_column,
                "sin_coeff": sin_coeff_column,
                "cos_coeff": cos_coeff_column,
                "omega_WKB_sq": omega_WKB_sq_column,
                "WKB_criterion": WKB_criterion_column,
                "analytic_G": analytic_G_column,
                "analytic_Gprime": analytic_Gprime_column,
            }
        )
        df.sort_values(by="z_source", ascending=False, inplace=True, ignore_index=True)
        df.to_csv(csv_path, header=True, index=False)

    def build_plot_GkSource_work(item):
        k_exit, z_response = item
        k_exit: wavenumber_exit_time
        z_response: redshift

        query_payload = {
            "model": model,
            "k": k_exit,
            "z_response": z_response,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }

        GkS_ref = pool.object_get("GkSource", **query_payload)

        return plot_Gk.remote(GkS_ref)

    work_grid = product(k_exit_times, z_subsample)

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
