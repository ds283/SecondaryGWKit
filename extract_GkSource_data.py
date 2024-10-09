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

    model = ray.get(
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

    work_grid = product(k_exit_times, z_subsample)

    payload_data = [
        {
            "model": model,
            "k": k_exit,
            "z_response": z_response,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }
        for k_exit, z_response in work_grid
    ]

    print("\n-- QUERYING GkSource CACHE")
    Gk_source_grid = ray.get(pool.object_get(GkSource, payload_data=payload_data))
    print(f"   >> GkSource cache populated with length {len(Gk_source_grid)}")

    @ray.remote
    def plot_Gk(Gk: GkSource):
        if not Gk.available:
            return

        values: List[GkSourceValue] = Gk.values

        def my_fabs(x: Optional[float]):
            if x is None:
                return None

            return fabs(x)

        G_points = [(value.z_source.z, my_fabs(value.numeric.G)) for value in values]
        G_WKB_points = [
            (value.z_source.z, my_fabs(value.WKB.G_WKB)) for value in values
        ]
        analytic_points = [
            (value.z_source.z, my_fabs(value.analytic_G)) for value in values
        ]

        theta_points = [(value.z_source.z, value.WKB.theta) for value in values]
        sin_coeff_points = [(value.z_source.z, value.WKB.sin_coeff) for value in values]
        cos_coeff_points = [(value.z_source.z, value.WKB.cos_coeff) for value in values]

        G_x, G_y = zip(*G_points)
        G_WKB_x, G_WKB_y = zip(*G_WKB_points)
        analytic_x, analytic_y = zip(*analytic_points)

        theta_x, theta_y = zip(*theta_points)
        sin_coeff_x, sin_coeff_y = zip(*sin_coeff_points)
        cos_coeff_x, cos_coeff_y = zip(*cos_coeff_points)

        k_exit = Gk._k_exit
        z_response = Gk.z_response

        sns.set_theme()
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(G_x, G_y, label="Numerical $G_k$")
        ax.plot(G_WKB_x, G_WKB_y, label="WKB $G_k$")
        ax.plot(
            analytic_x,
            analytic_y,
            label="Analytic $G_k$",
            linestyle="--",
        )

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("$G_k(z_{\\text{source}}, z_{\\text{response}})$")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.grid(True)
        ax.xaxis.set_inverted(True)

        base_path = Path(args.output).resolve()
        fig_path = (
            base_path
            / f"plots/Gk/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zresponse={z_response.z:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        plt.close()

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(theta_x, theta_y, label="WKB phase $\theta$")

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("WKB phase $\\theta$")

        ax.set_xscale("log")

        ax.grid(True)
        ax.xaxis.set_inverted(True)

        fig_path = (
            base_path
            / f"plots/theta/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_response.store_id}-zsource={z_response.z:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        plt.close()

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(sin_coeff_x, sin_coeff_y, label="$\\sin$ coefficient")
        ax.plot(cos_coeff_x, cos_coeff_y, label="$\\cos coefficient")

        ax.set_xlabel("source redshift $z$")
        ax.set_ylabel("coefficient")

        ax.set_xscale("log")

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
        H_ratio_column = [value.WKB.H_ratio for value in values]
        sin_coeff_column = [value.WKB.sin_coeff for value in values]
        cos_coeff_column = [value.WKB.cos_coeff for value in values]
        omega_WKB_sq_column = [value.omega_WKB_sq for value in values]
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
                "H_ratio": H_ratio_column,
                "sin_coeff": sin_coeff_column,
                "cos_coeff": cos_coeff_column,
                "omega_WKB_sq": omega_WKB_sq_column,
                "analytic_G": analytic_G_column,
                "analytic_Gprime": analytic_Gprime_column,
            }
        )
        df.to_csv(csv_path, header=True, index=False)

    work_refs = [plot_Gk.remote(Gk) for Gk in Gk_source_grid]
    ray.get(work_refs)
