import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from random import sample
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns
from math import fabs, pi
from ray import ObjectRef

from ComputeTargets import (
    BackgroundModel,
    ModelProxy,
    QuadSourceIntegral,
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

two_pi = 2.0 * pi

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
    default="QuadSourceIntegral-out",
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
        k=int(round(0.4 * len(response_k_exit_times) + 0.5, 0)),
    )

    qs_range = itertools.combinations_with_replacement(source_k_exit_times, 2)
    qs_subsample: List[Tuple[wavenumber_exit_time, wavenumber_exit_time]] = sample(
        qs_range,
        k=int(round(0.4 * len(qs_range) + 0.5, 0)),
    )

    DEFAULT_SAMPLES_PER_LOG10_Z = 150
    DEFAULT_ZEND = 0.1

    universal_z_grid = k_exit_earliest.populate_z_sample(
        outside_horizon_efolds=5,
        samples_per_log10z=DEFAULT_SAMPLES_PER_LOG10_Z,
        z_end=DEFAULT_ZEND,
    )

    z_array = ray.get(convert_to_redshifts(universal_z_grid))
    z_global_sample = redshift_array(z_array=z_array)

    z_source_sample = z_global_sample
    z_response_sample = z_global_sample

    # choose a subsample of RESPONSE redshifts
    z_subsample: List[redshift] = sample(
        list(z_response_sample), k=int(round(0.2 * len(z_response_sample) + 0.5, 0))
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

    # set up a proxy object to avoid having to repeatedly serialize the model instance and ship it out
    model_proxy = ModelProxy(model)

    GkSource_policy_2pt5, GkSource_policy_5pt0 = ray.get(
        [
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-numeric"-Levin-threshold="2.5"',
                Levin_threshold=2.5,
                numeric_policy="maximize_numeric",
            ),
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-numeric"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize_numeric",
            ),
        ]
    )

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
    def add_z_labels(ax, k_exit: wavenumber_exit_time):
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
    def plot_QuadSourceIntegral(
        k_exit: wavenumber_exit_time,
        q_exit: wavenumber_exit_time,
        r_exit: wavenumber_exit_time,
        z_response: redshift,
        data: List[QuadSourceIntegral],
    ):
        if len(data) <= 1:
            return

        data.sort(key=lambda x: x.z_response.z, reverse=True)

        base_path = Path(args.output).resolve()

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        z_min_quad = None
        z_max_quad = None
        z_min_WKB_quad = None
        z_max_WKB_quad = None
        z_min_Levin = None
        z_max_Levin = None

        abs_points = []
        abs_analytic_points = []

        for obj in data:
            obj: QuadSourceIntegral

            abs_points.append(
                (
                    obj.z_response.z,
                    safe_fabs(obj.total),
                )
            )
            abs_analytic_points.append((obj.z_response.z, safe_fabs(obj.analytic_rad)))

            if obj.numeric_quad is not None:
                if z_min_quad is None or obj.z_response < z_min_quad:
                    z_min_quad = obj.z_response
                if z_max_quad is None or obj.z_response > z_max_quad:
                    z_max_quad = obj.z_response

            if obj.WKB_quad is not None:
                if z_min_WKB_quad is None or obj.z_response < z_min_WKB_quad:
                    z_min_WKB_quad = obj.z_response
                if z_max_WKB_quad is None or obj.z_response > z_max_WKB_quad:
                    z_max_WKB_quad = obj.z_response

            if obj.WKB_Levin is not None:
                if z_min_Levin is None or obj.z_response < z_min_Levin:
                    z_min_Levin = obj.z_response
                if z_max_Levin is None or obj.z_response > z_max_Levin:
                    z_max_Levin = obj.z_response

            abs_x, abs_y = zip(*abs_points)
            abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)

            sns.set_theme()
            if len(abs_x) > 0 and any(y is not None and y > 0 for y in abs_y):
                fig = plt.figure()
                ax = plt.gca()

                ax.plot(abs_x, abs_y, label="Numerical", color="r", linestyle="solid")
                ax.plot(
                    abs_analytic_x,
                    abs_analytic_y,
                    label="Analytic",
                    color="g",
                    linestyle="dashed",
                )

                add_z_labels(ax, k_exit)

                if z_min_quad is not None and z_max_quad is not None:
                    ax.axvspan(
                        xmin=z_min_quad.z, xmax=z_max_quad.z, color="b", alpha=0.3
                    )

                if z_min_WKB_quad is not None and z_max_WKB_quad is not None:
                    ax.axvspan(
                        xmin=z_min_WKB_quad.z,
                        xmax=z_max_WKB_quad.z,
                        color="r",
                        alpha=0.3,
                    )

                if z_min_Levin is not None and z_max_Levin is not None:
                    ax.axvspan(
                        xmin=z_min_Levin.z, xmax=z_max_Levin.z, color="g", alpha=0.3
                    )

                ax.set_xlabel("response redshift $z$")

                set_loglog_axes(ax)

                fig_path = (
                    base_path
                    / f"plots/k-serial={k_exit.store_id}={k_exit.k.k_inv_Mpc:.5g}-q-serial={q_exit.store_id}={q_exit.k.k_inv_Mpc:.5g}-r-serial={r_exit.store_id}={r_exit.k.k_inv_Mpc:.5g}.pdf"
                )
                fig_path.parents[0].mkdir(exist_ok=True, parents=True)
                fig.savefig(fig_path)

                plt.close()

    def build_plot_QuadSourceIntegral_work(item):
        k_exit, qr_pair, z_response = item
        q_exit, r_exit = qr_pair

        k_exit: wavenumber_exit_time
        q_exit: wavenumber_exit_time
        r_exit: wavenumber_exit_time
        z_response: redshift

        data_ref: ObjectRef = pool.object_read_batch(
            ObjectClass="QuadSourceIntegral",
            shard_key={"k": k_exit},
            model=model_proxy,
            policy=GkSource_policy_2pt5,
            k=k_exit,
            q=q_exit,
            r=r_exit,
            z_response=None,
            z_source_max=None,
        )

        return plot_QuadSourceIntegral.remote(
            k_exit, q_exit, r_exit, z_response, data_ref
        )

    work_grid = itertools.product(k_subsample, qs_subsample, z_subsample)

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_plot_QuadSourceIntegral_work,
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
        title="GENERATING QuadSourceIntegral DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
