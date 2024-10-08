import argparse
import itertools
import sys
from datetime import datetime
from typing import List

import numpy as np
import ray

from ComputeTargets import (
    TkNumericalIntegration,
    GkNumericalIntegration,
    GkWKBIntegration,
    TensorSource,
    GkNumericalValue,
    GkWKBValue,
    GkSource,
)
from CosmologyConcepts import (
    wavenumber,
    redshift,
    wavenumber_array,
    wavenumber_exit_time,
    redshift_array,
)
from CosmologyModels.LambdaCDM import Planck2018
from Datastore.SQL.ShardedPool import ShardedPool
from RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)
from utilities import grouper

DEFAULT_LABEL = "SecondaryGWKit-test"
DEFAULT_TIMEOUT = 60
DEFAULT_SHARDS = 20
DEFAULT_RAY_ADDRESS = "auto"
DEFAULT_SAMPLES_PER_LOG10_Z = 150
DEFAULT_ZEND = 0.1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    type=str,
    default=None,
    help="read/write work items using the specified database cache",
)
parser.add_argument(
    "--job-name",
    default=DEFAULT_LABEL,
    help="specify a label for this job (used to identify integrations and other numerical products)",
)
parser.add_argument(
    "--shards",
    type=int,
    default=DEFAULT_SHARDS,
    help="specify number of shards to be used when creating a new datastore",
)
parser.add_argument(
    "--db-timeout",
    type=int,
    default=DEFAULT_TIMEOUT,
    help="specify connection timeout for database layer",
)
parser.add_argument(
    "--profile-db",
    type=str,
    default=None,
    help="write profiling and performance data to the specified database",
)
parser.add_argument(
    "--samples-log10z",
    type=int,
    default=DEFAULT_SAMPLES_PER_LOG10_Z,
    help="specify number of z-sample points per log10(z)",
)
parser.add_argument(
    "--transfer-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the transfer function work queue",
)
parser.add_argument(
    "--numerical-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the GkNumerical work queue",
)
parser.add_argument(
    "--WKB-queue",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="run the GkWKB work queue",
)
parser.add_argument(
    "--source-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the GkSource builder work queue",
)
parser.add_argument(
    "--zend",
    type=float,
    default=DEFAULT_ZEND,
    help="specify final redshift",
)
parser.add_argument(
    "--prune-unvalidated",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="prune unvalidated data from the datastore during startup",
)
parser.add_argument(
    "--ray-address",
    default=DEFAULT_RAY_ADDRESS,
    type=str,
    help="specify address of Ray cluster",
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
    shards=args.shards,
    profile_db=args.profile_db,
    job_name=args.job_name,
    prune_unvalidated=args.prune_unvalidated,
) as pool:

    # set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
    units = Mpc_units()
    params = Planck2018()
    LambdaCDM_Planck2018 = ray.get(
        pool.object_get("LambdaCDM", params=params, units=units)
    )

    zend = args.zend
    samples_per_log10z = args.samples_log10z

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

    ## DATASTORE OBJECTS

    # build absolute and relative tolerances
    atol, rtol = ray.get(
        [
            pool.object_get("tolerance", tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_REL_TOLERANCE),
        ]
    )

    # build stepper labels; we have to query these up-front from the pool in order to be
    # certain that they get the same serial number in each database shard.
    # So we can no longer construct these on-the-fly in the integration classes, as used to be done
    (
        solve_ivp_RK45,
        solve_ivp_DOP852,
        solve_ivp_Radau,
        solve_ivp_BDF,
        solve_icp_LSODA,
    ) = ray.get(
        [
            pool.object_get("IntegrationSolver", label="solve_ivp+RK45", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+DOP853", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+Radau", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+BDF", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+LSODA", stepping=0),
        ]
    )
    solvers = {
        "solve_ivp+RK45-stepping0": solve_ivp_RK45,
        "solve_ivp+DOP853-stepping0": solve_ivp_DOP852,
        "solve_ivp+Radau-stepping0": solve_ivp_Radau,
        "solve_ivp+BDF-stepping0": solve_ivp_BDF,
        "solve_ivp+LSODA-stepping0": solve_icp_LSODA,
    }

    ## STEP 1
    ## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES

    # build array of k-sample points
    print("\n** BUILDING ARRAY OF WAVENUMBERS AT WHICH TO SAMPLE")
    k_array = ray.get(
        convert_to_wavenumbers(np.logspace(np.log10(0.1), np.log10(5e7), 10))
    )
    k_sample = wavenumber_array(k_array=k_array)

    def build_k_exit_work(k: wavenumber):
        return pool.object_get(
            "wavenumber_exit_time",
            k=k,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        )

    # for each k mode we sample, determine its horizon exit point
    k_exit_queue = RayWorkPool(
        pool,
        k_sample,
        task_builder=build_k_exit_work,
        title="CALCULATE HORIZON EXIT TIMES",
        store_results=True,
    )
    k_exit_queue.run()
    k_exit_times = k_exit_queue.results

    ## STEP 2
    ## BUILD A UNIVERSAL GRID OF Z-VALUES AT WHICH TO SAMPLE

    print("\n** BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE")
    k_exit_earliest: wavenumber_exit_time = k_exit_times[-1]
    print(
        f"   @@ earliest horizon exit time is {k_exit_earliest.k.k_inv_Mpc:.5g}/Mpc with z_exit={k_exit_earliest.z_exit:.5g}"
    )

    # build a log-spaced universal grid of wavenumbers

    universal_z_grid = k_exit_earliest.populate_z_sample(
        outside_horizon_efolds="suph_e3",
        samples_per_log10z=samples_per_log10z,
        z_end=zend,
    )

    # embed this universal redshift list into the database
    z_array = ray.get(convert_to_redshifts(universal_z_grid))
    z_sample = redshift_array(z_array=z_array)

    # build tags and other labels
    (
        TkProductionTag,
        GkProductionTag,
        GlobalZGridTag,
        OutsideHorizonEfoldsTag,
        LargestZTag,
        SamplesPerLog10ZTag,
    ) = ray.get(
        [
            pool.object_get("store_tag", label="TkOneLoopDensity"),
            pool.object_get("store_tag", label="GkOneLoopDensity"),
            pool.object_get("store_tag", label=f"GlobalRedshiftGrid_{len(z_sample)}"),
            pool.object_get("store_tag", label=f"OutsideHorizonEfolds_e3"),
            pool.object_get("store_tag", label=f"LargestRedshift_{z_sample.max.z:.5g}"),
            pool.object_get(
                "store_tag", label=f"SamplesPerLog10Z_{samples_per_log10z}"
            ),
        ]
    )

    ## STEP 1a
    ## BAKE THE BACKGROUND COSMOLOGY INTO A BACKGROUND MODEL OBJECT
    model = ray.get(
        pool.object_get(
            "BackgroundModel",
            solver_labels=solvers,
            cosmology=LambdaCDM_Planck2018,
            z_sample=z_sample,
            atol=atol,
            rtol=rtol,
            tags=[GlobalZGridTag, LargestZTag, SamplesPerLog10ZTag],
        )
    )
    if not model.available:
        print("\n** CALCULATING BACKGROUND MODEL")
        data = ray.get(model.compute(label=LambdaCDM_Planck2018.name))
        model.store()
        model = ray.get(pool.object_store(model))
        outcome = ray.get(pool.object_validate(model))
    else:
        print(
            f'\n** FOUND EXISTING BACKGROUND MODEL "{model.label}" (store_id={model.store_id})'
        )

    ## STEP 2
    ## COMPUTE MATTER TRANSFER FUNCTIONS

    def build_Tk_work(k_exit: wavenumber_exit_time):
        my_sample = z_sample.truncate(k_exit.z_exit_suph_e5)
        return pool.object_get(
            "TkNumericalIntegration",
            solver_labels=solvers,
            model=model,
            k=k_exit,
            z_sample=my_sample,
            z_init=my_sample.max,
            atol=atol,
            rtol=rtol,
            tags=[
                TkProductionTag,
                GlobalZGridTag,
                OutsideHorizonEfoldsTag,
                LargestZTag,
                SamplesPerLog10ZTag,
            ],
            delta_logz=1.0 / float(samples_per_log10z),
        )

    def build_Tk_work_label(Tk: TkNumericalIntegration):
        return f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def validate_Tk_work(Tk: TkNumericalIntegration):
        return pool.object_validate(Tk)

    def post_Tk_work(Tk: TkNumericalIntegration):
        return ray.put(Tk)

    if args.transfer_queue:
        Tk_queue = RayWorkPool(
            pool,
            k_exit_times,
            task_builder=build_Tk_work,
            validation_handler=validate_Tk_work,
            post_handler=post_Tk_work,
            label_builder=build_Tk_work_label,
            title="CALCULATE MATTER TRANSFER FUNCTIONS",
            store_results=True,
        )
        Tk_queue.run()
        Tks = Tk_queue.results

        tensor_source_grid = list(
            itertools.combinations_with_replacement(range(len(Tks)), 2)
        )

        def build_tensor_source_work(grid_idx):
            idx_i, idx_j = grid_idx

            q = k_sample[idx_i]
            Tq = Tks[idx_i]
            Tr = Tks[idx_j]

            # q is not used by the TensorSource constructor, but is accepted because it functions as the shard key
            return pool.object_get(
                "TensorSource",
                z_sample=z_sample,
                q=q,
                Tq=Tq,
                Tr=Tr,
                tags=[
                    TkProductionTag,
                    GlobalZGridTag,
                    OutsideHorizonEfoldsTag,
                    LargestZTag,
                    SamplesPerLog10ZTag,
                ],
            )

        def validate_tensor_source_work(calc: TensorSource):
            return pool.object_validate(calc)

        def build_tensor_source_work_label(calc: TensorSource):
            q = calc.q
            r = calc.r
            return f"{args.job_name}-tensor-src-q{q.k_inv_Mpc:.3g}-r{r.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"

        TensorSource_queue = RayWorkPool(
            pool,
            tensor_source_grid,
            task_builder=build_tensor_source_work,
            validation_handler=validate_tensor_source_work,
            label_builder=build_tensor_source_work_label,
            title="CALCULATE TENSOR SOURCE TERMS",
            store_results=False,
        )
        TensorSource_queue.run()

    ## STEP 3
    ## COMPUTE TENSOR GREEN'S FUNCTIONS NUMERICALLY FOR RESPONSE TIMES NOT TOO FAR INSIDE THE HORIZON

    def build_Gk_numerical_work(z_source: redshift):
        work_refs = []

        for k_exit in k_exit_times:
            # find redshift where this k-mode is at least 3 efolds inside the horizon
            # we won't calculate numerical Green's functions with the response redshift later than this, because
            # the oscillations become rapid, and we are better switching to a WKB approximation
            if z_source.z > k_exit.z_exit_subh_e3 - DEFAULT_FLOAT_PRECISION:
                # cut down response zs to those that are (1) later than the source, and (2) earlier than the 5-efolds-inside-the-horizon point
                # (here with a 10% tolerance)
                response_zs = z_sample.truncate(z_source, keep="lower").truncate(
                    0.9 * k_exit.z_exit_subh_e5, keep="higher"
                )

                if len(response_zs) > 1:
                    work_refs.append(
                        pool.object_get(
                            "GkNumericalIntegration",
                            solver_labels=solvers,
                            model=model,
                            k=k_exit,
                            z_source=z_source,
                            z_sample=response_zs,
                            atol=atol,
                            rtol=rtol,
                            tags=[
                                GkProductionTag,
                                GlobalZGridTag,
                                OutsideHorizonEfoldsTag,
                                LargestZTag,
                                SamplesPerLog10ZTag,
                            ],
                            delta_logz=1.0 / float(samples_per_log10z),
                            mode="stop",
                        )
                    )

        return work_refs

    def build_Gk_numerical_work_label(Gk: GkNumericalIntegration):
        return f"{args.job_name}-GkNumerical-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def validate_Gk_numerical_work(Gk: GkNumericalIntegration):
        return pool.object_validate(Gk)

    if args.numerical_queue:
        Gk_numerical_queue = RayWorkPool(
            pool,
            z_sample,
            task_builder=build_Gk_numerical_work,
            validation_handler=validate_Gk_numerical_work,
            label_builder=build_Gk_numerical_work_label,
            title="CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS",
            store_results=False,
            create_batch_size=5,
            notify_batch_size=2000,
            process_batch_size=50,
        )
        Gk_numerical_queue.run()

    ## STEP 4
    ## COMPUTE TENSOR GREEN'S FUNCTIONS IN THE WKB APPROXIMATION FOR RESPONSE TIMES INSIDE THE HORIZON

    def build_Gk_WKB_work(batch: List[redshift]):
        # build GKWKBIntegration work items for a batch of z_source times

        # Query for a GkNumericalIntegration instances for each combination of model, k_exit, and z_source.
        # If one exists, it will be used to set initial conditions for the WKB part of the evolution.
        # We want to do this in a vectorized way, pulling an entire batch of GkNumericalIntegration instances
        # for a range of z-source values from the same shard. This is more efficient than paying the Ray and database
        # overhead for multiple lookups
        payload_batch = [
            {
                "k": k_exit,
                "payload": [
                    {
                        "solver_labels": solvers,
                        "model": model,
                        "k": k_exit,
                        "z_source": z_source,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [GkProductionTag, GlobalZGridTag, SamplesPerLog10ZTag],
                    }
                    for z_source in batch
                ],
            }
            for k_exit in k_exit_times
        ]

        lookup_queue = RayWorkPool(
            pool,
            payload_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkNumericalIntegration", {"k": x["k"]}, payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=5,  # may need to tweak relative to number of shards
            process_batch_size=1,
        )
        lookup_queue.run()

        work_refs = []

        response_pools = {
            z_source.store_id: z_sample.truncate(z_source, keep="lower")
            for z_source in batch
        }

        for i, k_exit in enumerate(k_exit_times):
            k_exit: wavenumber_exit_time
            Gk_data = lookup_queue.results[i]

            for z_source, Gk in zip(batch, Gk_data):
                # be defensive about ensuring provenance for our data products
                Gk: GkNumericalIntegration
                assert Gk._k_exit.store_id == k_exit.store_id
                assert Gk._z_source.store_id == z_source.store_id

                response_pool = response_pools[z_source.store_id]

                if (
                    len(response_pool) == 0
                    or response_pool.max.z <= response_pool.min.z
                ):
                    continue

                # query whether there is a pre-computed GkNumericalIntegration for this source redshift.
                # typically, this will be the case provided the source redshift is sufficiently early
                if Gk.available:
                    G_init = Gk.stop_G
                    Gprime_init = Gk.stop_Gprime
                    z_init = k_exit.z_exit - Gk.stop_efolds_subh

                    # find redshift where this k-mode is at least 3 e-folds inside the horizon
                    # we don't calculate WKB Green's functions with the response redshift earlier than this
                    max_response = min(k_exit.z_exit_subh_e3, z_init)
                    response_zs = response_pool.truncate(max_response, keep="lower")

                    if len(response_zs) > 0:
                        work_refs.append(
                            pool.object_get(
                                "GkWKBIntegration",
                                solver_labels=solvers,
                                model=model,
                                k=k_exit,
                                z_source=z_source,
                                z_init=z_init,
                                G_init=G_init,
                                Gprime_init=Gprime_init,
                                z_sample=response_zs,
                                atol=atol,
                                rtol=rtol,
                                tags=[
                                    GkProductionTag,
                                    GlobalZGridTag,
                                    OutsideHorizonEfoldsTag,
                                    LargestZTag,
                                    SamplesPerLog10ZTag,
                                ],
                            )
                        )
                else:
                    # no pre-computed initial condition available.
                    # This is OK and expected if the z_source is sufficiently far after horizon re-entry,
                    # but we should warn if we expect an initial condition to be available
                    if z_source.z > k_exit.z_exit_subh_e3 - DEFAULT_FLOAT_PRECISION:
                        print(
                            f"!! WARNING: no numerically-computed initial conditions ia available for k={k_exit.k.k_inv_Mpc:.5g}/Mpc, z_source={z_source.z:.5g}"
                        )
                        print(
                            f"|    This may indicate an edge-effect, or possibly that not all initial conditions are present in the datastore."
                        )

                    if z_source.z >= k_exit.z_exit:
                        print(
                            f"!! ERROR: attempt to compute WKB solution for k={k_exit.k.k_inv_Mpc:.5g}/Mpc, z_source={z_source.z:.5g} without a pre-computed initial condition"
                        )
                        print(
                            f"|    For this k-mode, horizon entry occurs at z_entry={k_exit.z_exit:.5g}"
                        )
                        raise RuntimeError(
                            f"Cannot compute WKB solution outside the horizon"
                        )

                    work_refs.append(
                        pool.object_get(
                            "GkWKBIntegration",
                            solver_labels=solvers,
                            model=model,
                            k=k_exit,
                            z_source=z_source,
                            G_init=0.0,
                            Gprime_init=1.0,
                            z_sample=response_pool,
                            atol=atol,
                            rtol=rtol,
                            tags=[
                                GkProductionTag,
                                GlobalZGridTag,
                                OutsideHorizonEfoldsTag,
                                LargestZTag,
                                SamplesPerLog10ZTag,
                            ],
                        )
                    )

        return work_refs

    def build_Gk_WKB_work_label(Gk: GkWKBIntegration):
        return f"{args.job_name}-GkWKB-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-zinit{Gk.z_init:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def validate_Gk_WKB_work(Gk: GkWKBIntegration):
        return pool.object_validate(Gk)

    GkWKB_work_batches = list(grouper(z_sample, n=20, incomplete="ignore"))

    if args.WKB_queue:
        Gk_WKB_queue = RayWorkPool(
            pool,
            GkWKB_work_batches,
            task_builder=build_Gk_WKB_work,
            validation_handler=validate_Gk_WKB_work,
            label_builder=build_Gk_WKB_work_label,
            title="CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS",
            store_results=False,
            create_batch_size=5,
            notify_batch_size=2000,
            process_batch_size=50,
        )
        Gk_WKB_queue.run()

    # STEP 5
    # REBUILD TENSOR GREEN'S FUNCTIONS AS FUNCTIONS OF THE SOURCE REDSHIFT, RATHER THAN THE RESPONSE REDSHIFT

    # (In theory we could construct this on-the-fly when we need it. Storing the rebuilt tensor Green's functions in the datastore is,
    # strictly, redundant. But it is likely much faster because we need to perform complicated table lookups with joins in order to build
    # these directly from GkNumericalValue and GkWKValue rows. Also, there is the reproducibility angle of keeping a record of what
    # rebuilt data products we used.)

    def build_GkSource_batch(batch):
        # try to do parallel lookup of the GkNumericalIntegration/GkWKBIntegration records needed
        # to process this batch

        z_sources_batch = [
            z_sample.truncate(z_response, keep="higher") for z_response, _ in batch
        ]

        batch_data = [
            (data[0], data[1], z_sources_batch[i]) for i, data in enumerate(batch)
        ]

        # all the z-source values for a single k-mode will be held on the same database shard.
        # It's critical that we vectorize the lookup for these. There will typically be hundreds or
        # thousands of z values, and we cannot afford the overhead of looking up each GkNumericalValue or
        # GkWKBValue individually. This entails a database overhead (lookups are much more efficient if wrapped
        # in a single transaction) and a multiprocessing overhead (Ray has to serialize, schedule, deserialize, ...)
        payload_batch = [
            {
                # note we don't specify the k mode as part of the main payload
                # instead we specify it separately to object_get_vectorized()
                "k": k_exit,
                "payload": [
                    {
                        "model": model,
                        "z_source": z_source,
                        "z": z_response,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            GlobalZGridTag,
                            OutsideHorizonEfoldsTag,
                            LargestZTag,
                            SamplesPerLog10ZTag,
                        ],
                    }
                    for z_source in z_sources
                ],
            }
            for z_response, k_exit, z_sources in batch_data
        ]

        labelled_payloads = [
            (label, payload)
            for payload in payload_batch
            for label in ["GkNumericalValue", "GkWKBValue"]
        ]

        lookup_queue = RayWorkPool(
            pool,
            labelled_payloads,
            task_builder=lambda x: pool.object_get_vectorized(
                x[0], {"k": x[1]["k"]}, payload_data=x[1]["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=5,  # may need to tweak relative to number of shards
            process_batch_size=10,
        )
        lookup_queue.run()

        work_refs = []

        for i, data in enumerate(batch_data):
            z_response, k_exit, z_sources = data
            z_response: redshift
            k_exit: wavenumber_exit_time
            z_sources: redshift_array

            numerics = lookup_queue.results[2 * i]
            WKBs = lookup_queue.results[2 * i + 1]

            numeric_data = {}
            WKB_data = {}

            for j, z_source in enumerate(z_sources):
                z_source: redshift

                numeric_value = numerics[j]
                WKB_value = WKBs[j]

                # be defensive about ensuring the provenance for our data
                assert isinstance(numeric_value, GkNumericalValue)
                assert isinstance(WKB_value, GkWKBValue)

                # the _k_exit and _z_source fields are not part of the public API for these
                # objects, but they are set by the data store after deserialization

                if not numeric_value.available and not WKB_value.available:
                    raise RuntimeError(
                        f"GkSource builder: no data are available for k={k_exit.k.k_inv_Mpc}/Mpc, z_source={z_source.z:.5g}, z_response={z_response.z:.5g}"
                    )

                if numeric_value.available:
                    assert numeric_value._k_exit.store_id == k_exit.store_id
                    assert numeric_value._z_source.store_id == z_source.store_id

                    numeric_data[z_source.store_id] = numeric_value

                if WKB_value.available:
                    assert WKB_value._k_exit.store_id == k_exit.store_id
                    assert WKB_value._z_source.store_id == z_source.store_id

                    WKB_data[z_source.store_id] = WKB_value

            work_refs.append(
                pool.object_get(
                    "GkSource",
                    payload={
                        "numeric": numeric_data,
                        "WKB": WKB_data,
                    },
                    model=model,
                    k=k_exit,
                    atol=atol,
                    rtol=rtol,
                    z_response=z_response,
                    z_sample=z_sources,
                    tags=[
                        GkProductionTag,
                        GlobalZGridTag,
                        OutsideHorizonEfoldsTag,
                        LargestZTag,
                        SamplesPerLog10ZTag,
                    ],
                )
            )

        return work_refs

    def build_GkSource_work_label(Gk: GkSource):
        return f"{args.job_name}-GkSource-k{Gk.k.k_inv_Mpc:.3g}-responsez{Gk.z_response.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def validate_GkSource_work(Gk: GkSource):
        return pool.object_validate(Gk)

    if args.source_queue:
        # THERE IS A LOT OF WORK TO DO!
        # For each wavenumber in the k-sample (here k_exit_times), and each value of r_response,
        # we need to build the Green's function for all possible values of z_source.
        # Even with 10 wavenumbers and 2,000 z-samples points, that is 20,000 items.
        # With 500 wavenumbers and 2,000 z-samples, it is 1 million items.
        # To process these efficiently, we break the queue up into batches, and try to
        # run a sub-queue that queries that needed data in parallel.

        # we could set all this up insider the task builder, but to prevent spawning too many Ray tasks in one
        # go, it seems preferable to break down the problem a bit more
        GkSource_work_items = itertools.product(z_sample, k_exit_times)
        GkSource_work_batches = list(
            grouper(GkSource_work_items, n=100, incomplete="ignore")
        )

        GkSource_queue = RayWorkPool(
            pool,
            GkSource_work_batches,
            task_builder=build_GkSource_batch,
            validation_handler=validate_GkSource_work,
            label_builder=build_GkSource_work_label,
            title="REBUILD GREENS FUNCTIONS FOR SOURCE REDSHIFT",
            store_results=False,
            create_batch_size=1,  # we have batched the work queue into chunks ourselves
            notify_batch_size=2000,
            process_batch_size=50,
        )
        GkSource_queue.run()
