import argparse
import sys
from itertools import product
from pathlib import Path
from random import sample
from typing import List

import matplotlib.pyplot as plt
import ray
import seaborn as sns
from math import fabs

from ComputeTargets import (
    GkWKBIntegration,
    BackgroundModel,
    GkNumericalIntegration,
    GkNumericalValue,
    GkWKBValue,
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
    "--output", default="Gk-WKB-out", type=str, help="specify folder for output files"
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
        list(z_sample), k=int(round(0.07 * len(z_sample) + 0.5, 0))
    )

    work_grid = product(k_exit_times, z_subsample)

    payload_data = [
        {
            "solver_labels": [],
            "model": model,
            "k": k_exit,
            "z_source": z_source,
            "z_sample": None,
            "atol": atol,
            "rtol": rtol,
        }
        for k_exit, z_source in work_grid
    ]

    print("\n-- QUERYING NUMERICAL INTEGRATIONS")
    Gk_numerical_grid = ray.get(
        pool.object_get(GkNumericalIntegration, payload_data=payload_data)
    )
    print(
        f"   >> numerical integrations populated with length {len(Gk_numerical_grid)}"
    )

    print("\n-- QUERYING WKB INTEGRATIONS")
    Gk_WKB_grid = ray.get(pool.object_get(GkWKBIntegration, payload_data=payload_data))
    print(f"   >> WKB integrations populated with length {len(Gk_WKB_grid)}")

    work_items = zip(Gk_numerical_grid, Gk_WKB_grid)
    print(f"\n-- BUILT WORK GRID")

    @ray.remote
    def plot_Gk(Gk_numerical: GkNumericalIntegration, Gk_WKB: GkWKBIntegration):
        k_exit = Gk_numerical._k_exit
        z_source = Gk_numerical.z_source

        sns.set_theme()
        fig = plt.figure()
        ax = plt.gca()

        if Gk_numerical.available:
            values: List[GkNumericalValue] = Gk_numerical.values

            numerical_points = [(value.z.z, fabs(value.G)) for value in values]
            analytic_points = [(value.z.z, fabs(value.analytic_G)) for value in values]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_x, analytic_y = zip(*analytic_points)

            ax.plot(numerical_x, numerical_y, label="Numerical $G_k$")
            ax.plot(
                analytic_x,
                analytic_y,
                label="Analytic $G_k$ (numerical region)",
                linestyle="--",
            )

        if Gk_WKB.available:
            values: List[GkWKBValue] = Gk_WKB.values

            numerical_points = [(value.z.z, fabs(value.G_WKB)) for value in values]
            analytic_points = [(value.z.z, fabs(value.analytic_G)) for value in values]

            numerical_x, numerical_y = zip(*numerical_points)
            analytic_x, analytic_y = zip(*analytic_points)

            ax.plot(numerical_x, numerical_y, label="WKB $G_k$")
            ax.plot(
                analytic_x,
                analytic_y,
                label="Analytic $G_k$ (WKB region)",
                linestyle="--",
            )

        ax.set_xlabel("response redshift $z$")
        ax.set_ylabel("$G_k(z_{\\text{source}}, z_{\\text{response}})$")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.grid(True)
        ax.xaxis.set_inverted(True)

        base_path = Path(args.output).resolve()
        fig_path = (
            base_path
            / f"figures/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}/z-serial={z_source.store_id}-zsource={z_source.z:.5g}.pdf"
        )
        fig_path.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path)
        plt.close()

    work_refs = [
        plot_Gk.remote(Gk_numerical, Gk_WKB) for Gk_numerical, Gk_WKB in work_items
    ]
    ray.get(work_refs)
