import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from random import sample
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from math import pi, fabs, sqrt
from matplotlib import pyplot as plt
from ray import ObjectRef

from ComputeTargets import (
    BackgroundModel,
    GkSourcePolicyData,
    GkSourceProxy,
    QuadSource,
    GkSource,
    GkSourceValue,
    QuadSourceValue,
)
from ComputeTargets.BackgroundModel import ModelProxy, ModelFunctions
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
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)

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
    default="QuadSourceIntegrand-out",
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
    label = f'{VERSION_LABEL}-jobname-extract_QuadSourceIntegrand_data-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

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
    job_name="extract_QuadSourceIntegrand_data",
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
        k=int(round(0.9 * len(response_k_exit_times) + 0.5, 0)),
    )

    qs_range = list(itertools.combinations_with_replacement(source_k_exit_times, 2))
    qs_subsample: List[Tuple[wavenumber_exit_time, wavenumber_exit_time]] = sample(
        qs_range,
        k=int(round(0.4 * len(qs_range) + 0.5, 0)),
    )

    DEFAULT_SAMPLES_PER_LOG10_Z = 150
    DEFAULT_ZEND = 0.1

    # array of z-sample points matching the SOURCE GRID
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
        list(z_response_sample), k=int(round(0.12 * len(z_response_sample) + 0.5, 0))
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
    def add_z_labels(ax, GkPolicy: GkSourcePolicyData, k_exit: wavenumber_exit_time):
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

        if GkPolicy.type == "mixed" and GkPolicy.crossover_z is not None:
            ax.axvline(
                GkPolicy.crossover_z, linestyle=(5, (10, 3)), color="m"
            )  # long dash with offset
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.crossover_z,
                0.15,
                "crossover_z",
                transform=trans,
                fontsize="x-small",
                color="m",
            )

        if (
            GkPolicy.type == "mixed" or GkPolicy.type == "WKB"
        ) and GkPolicy.Levin_z is not None:
            ax.axvline(
                GkPolicy.Levin_z, linestyle=(0, (5, 10)), color="m"
            )  # loosely dashed
            ax.text(
                TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.Levin_z,
                0.05,
                "Levin boundary",
                transform=trans,
                fontsize="x-small",
                color="m",
            )

    def add_Gk_labels(ax, obj: GkSourcePolicyData):
        fns = obj.functions
        if fns.type is not None:
            ax.text(
                0.0,
                1.05,
                f"Type: {fns.type}",
                transform=ax.transAxes,
                fontsize="x-small",
            )

        if fns.quality is not None:
            ax.text(
                0.5,
                1.05,
                f"Quality: {fns.quality}",
                transform=ax.transAxes,
                fontsize="x-small",
            )

        if fns.WKB_region is not None:
            ax.text(
                0.0,
                1.1,
                f"WKB region: ({fns.WKB_region[0]:.3g}, {fns.WKB_region[1]:.3g})",
                transform=ax.transAxes,
                fontsize="x-small",
            )

        if fns.numerical_region is not None:
            ax.text(
                0.5,
                1.1,
                f"Numeric region: ({fns.numerical_region[0]:.3g}, {fns.numerical_region[1]:.3g})",
                transform=ax.transAxes,
                fontsize="x-small",
            )

    @ray.remote
    def plot_QuadSourceIntegrand(
        k_exit: wavenumber_exit_time,
        q_exit: wavenumber_exit_time,
        r_exit: wavenumber_exit_time,
        policy: GkSourcePolicyData,
        source: QuadSource,
        z_source_max: redshift,
    ):
        if not policy.available:
            print(f"** GkPolicy not available")
            return

        Gk: GkSource = policy._source_proxy.get()
        if not Gk.available:
            print(f"** GkSource not available")
            return

        if source.z_sample.max.z < z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"supplied quadratic source term has maximum z_source={source.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.z_sample.max.z < z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"supplied Gk has maximum z_source={Gk.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.k.store_id != k_exit.k.store_id:
            raise RuntimeError(
                f"supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={k_exit.k.store_id}])"
            )

        if source.q.store_id != q_exit.k.store_id:
            raise RuntimeError(
                f"supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={q_exit.k.store_id}])"
            )

        if source.r.store_id != r_exit.k.store_id:
            raise RuntimeError(
                f"supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={r_exit.k.store_id}])"
            )

        base_path = Path(args.output).resolve()

        model_f: ModelFunctions = model.functions

        Gk_values: List[GkSourceValue] = Gk.values
        source_values: List[QuadSourceValue] = source.values

        Gk_value_dict = {
            value.z_source.store_id: value
            for value in Gk_values
            if value.z_source <= z_source_max
        }
        source_value_dict = {
            value.z.store_id: value
            for value in source_values
            if value.z <= z_source_max
        }

        Gk_value_ids = set(Gk_value_dict.keys())
        source_value_ids = set(source_value_dict.keys())

        common_ids = Gk_value_ids.intersection(source_value_ids)

        def safe_fabs(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None

            return fabs(x)

        def integrand(store_id: int):
            Gk_value = Gk_value_dict[store_id]
            source_value = source_value_dict[store_id]

            Green_analytic = Gk_value.analytic_G_rad
            if Gk_value.has_numeric:
                Green = Gk_value.numeric.G
            elif Gk_value.has_WKB:
                Green = Gk_value.WKB.G_WKB
            else:
                raise RuntimeError("GkSource value has neither numeric nor WKB content")

            source = source_value.source
            source_analytic = source_value.analytic_source_rad

            H = model_f.Hubble(Gk_value.z_source.z)
            H_sq = H * H

            payload = {
                "z": Gk_value.z_source,
                "integrand": Green * source / H_sq,
                "analytic_integrand": Green_analytic * source_analytic / H_sq,
            }

            if Gk_value.has_WKB:
                sin_ampl = Gk_value.WKB.sin_coeff * sqrt(
                    Gk_value.WKB.H_ratio / sqrt(Gk_value.omega_WKB_sq)
                )
                payload["Levin_f"] = sin_ampl * source / H_sq
                payload["analytic_Levin_f"] = sin_ampl * source_analytic / H_sq

            return payload

        integrand_points = [integrand(z_id) for z_id in common_ids]
        integrand_points.sort(key=lambda x: x["z"].z, reverse=True)

        abs_integrand_points = [
            (x["z"].z, safe_fabs(x["integrand"])) for x in integrand_points
        ]
        abs_analytic_points = [
            (x["z"].z, safe_fabs(x["analytic_integrand"])) for x in integrand_points
        ]
        abs_Levin_f_points = [
            (x["z"].z, safe_fabs(x["Levin_f"]))
            for x in integrand_points
            if "Levin_f" in x
        ]
        abs_analytic_Levin_f_points = [
            (x["z"].z, safe_fabs(x["analytic_Levin_f"]))
            for x in integrand_points
            if "analytic_Levin_f" in x
        ]

        abs_integrand_x, abs_integrand_y = zip(*abs_integrand_points)
        abs_analytic_x, abs_analytic_y = zip(*abs_analytic_points)

        abs_Levin_f_x, abs_Levin_f_y = zip(*abs_Levin_f_points)
        abs_analytic_Levin_f_x, abs_analytic_Levin_f_y = zip(
            *abs_analytic_Levin_f_points
        )

        sns.set_theme()

        if len(abs_integrand_x) > 0 and (
            any(y is not None and y > 0 for y in abs_integrand_y)
            or any(y is not None and y > 0 for y in abs_analytic_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(
                abs_integrand_x,
                abs_integrand_y,
                label="Numeric",
                color="r",
                linestyle="solid",
            )
            ax.plot(
                abs_analytic_x,
                abs_analytic_y,
                label="Analytic",
                color="b",
                linestyle="dashed",
            )

            add_z_labels(ax, policy, k_exit)
            add_Gk_labels(ax, policy)

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/integrand/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            ax.set_xlim(
                int(round(k_exit.z_exit_suph_e5 + 0.5, 0)),
                int(round(0.85 * k_exit.z_exit_subh_e5 + 0.5, 0)),
            )

            fig_path = (
                base_path
                / f"plots/reentry-zoom/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            plt.close()

        if len(abs_Levin_f_x) > 0 and (
            any(y is not None and y > 0 for y in abs_Levin_f_y)
            or any(y is not None and y > 0 for y in abs_analytic_Levin_f_y)
        ):
            fig = plt.figure()
            ax = plt.gca()

            ax.plot(
                abs_Levin_f_x,
                abs_Levin_f_y,
                label="Numeric",
                color="r",
                linestyle="solid",
            )
            ax.plot(
                abs_analytic_Levin_f_x,
                abs_analytic_Levin_f_y,
                label="Analytic",
                color="b",
                linestyle="dashed",
            )

            add_z_labels(ax, policy, k_exit)
            add_Gk_labels(ax, policy)

            set_loglog_axes(ax)

            fig_path = (
                base_path
                / f"plots/Levin_f/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.pdf"
            )
            fig_path.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

            plt.close()

        z_source_column = [x["z"].z for x in integrand_points]
        integrand_column = [x["integrand"] for x in integrand_points]
        analytic_column = [x["analytic_integrand"] for x in integrand_points]
        Levin_f_column = [
            x["Levin_f"] if hasattr(x, "Levin_f") else None for x in integrand_points
        ]
        analytic_Levin_f_column = [
            x["analytic_Levin_f"] if hasattr(x, "analytic_Levin_f") else None
            for x in integrand_points
        ]

        csv_path = (
            base_path / f"csv/k-serial={k_exit.store_id}-k={k_exit.k.k_inv_Mpc:.5g}.csv"
        )
        csv_path.parents[0].mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame.from_dict(
            {
                "z_source": z_source_column,
                "integrand": integrand_column,
                "analytic_integrand": analytic_column,
                "Levin_f": Levin_f_column,
                "analytic_Levin_f": analytic_Levin_f_column,
            }
        )
        df.sort_values(by="z_source", ascending=False, inplace=True, ignore_index=True)
        df.to_csv(csv_path, header=True, index=False)

    z_source_integral_max_z = z_source_sample[10]
    z_source_integral_response_sample = z_response_sample.truncate(
        z_source_integral_max_z, keep="lower-strict"
    )

    def build_plot_QuadSourceIntegrand_work(item):
        k_exit, qr_pair = item
        q_exit, r_exit = qr_pair

        k_exit: wavenumber_exit_time
        q_exit: wavenumber_exit_time
        r_exit: wavenumber_exit_time

        GkSource_ref: ObjectRef = pool.object_get(
            "GkSource",
            model=model_proxy,
            k=k_exit,
            z_response=z_source_integral_response_sample.min,
            z_sample=None,
            atol=atol,
            rtol=rtol,
        )
        GkSource = ray.get(GkSource_ref)
        GkSource_proxy = GkSourceProxy(GkSource)

        Policy_ref: ObjectRef = pool.object_get(
            "GkSourcePolicyData",
            source=GkSource_proxy,
            policy=GkSource_policy_2pt5,
            k=k_exit,
        )

        Tsource_ref = pool.object_get(
            "QuadSource",
            model=model_proxy,
            z_sample=None,
            q=q_exit,
            r=r_exit,
        )
        return plot_QuadSourceIntegrand.remote(
            k_exit, q_exit, r_exit, Policy_ref, Tsource_ref, z_source_integral_max_z
        )

    work_grid = itertools.product(k_subsample, qs_subsample)

    work_queue = RayWorkPool(
        pool,
        work_grid,
        task_builder=build_plot_QuadSourceIntegrand_work,
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
        title="GENERATING QuadSourceIntegrand DATA PRODUCTS",
        store_results=False,
    )
    work_queue.run()
