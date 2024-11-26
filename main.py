import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import List, Optional

import numpy as np
import ray
from pandas import DataFrame

from ComputeTargets import (
    TkNumericalIntegration,
    GkNumericalIntegration,
    GkWKBIntegration,
    QuadSource,
    GkNumericalValue,
    GkWKBValue,
    GkSource,
    QuadSourceIntegral,
    GkSourceProxy,
    GkSourcePolicyData,
    ModelProxy,
    BackgroundModel,
    BesselPhaseProxy,
)
from CosmologyConcepts import (
    wavenumber,
    redshift,
    wavenumber_array,
    wavenumber_exit_time,
    wavenumber_exit_time_array,
    redshift_array,
)
from CosmologyModels.LambdaCDM import Planck2018
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.ShardedPool import ShardedPool
from LiouvilleGreen.bessel_phase import bessel_phase
from MetadataConcepts import GkSourcePolicy
from RayTools.RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_QUADRATURE_RTOL,
    DEFAULT_QUADRATURE_ATOL,
)
from utilities import grouper, format_time

DEFAULT_LABEL = "SecondaryGWKit-test"
DEFAULT_TIMEOUT = 60
DEFAULT_SHARDS = 20
DEFAULT_RAY_ADDRESS = "auto"
DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100
DEFAULT_RESPONSE_SAMPLES_SPARSENESS = 12
DEFAULT_ZEND = 0.1

allowed_drop_actions = [
    "tk-numeric",
    "quad-source",
    "gk-numeric",
    "gk-wkb",
    "gk-source",
    "gk-source-policy-records",
    "gk-source-policy",
    "quad-source-integral",
    "1loop-integral",
]

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
    "--source-samples-log10z",
    type=int,
    default=DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z,
    help="specify number of z-sample points per log10(z) for the source term",
)
parser.add_argument(
    "--response-sparseness",
    type=int,
    default=DEFAULT_RESPONSE_SAMPLES_SPARSENESS,
    help="specify sparseness of response z-sample points relative to source sample",
)
parser.add_argument(
    "--Tk-numerical-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the transfer function work queue",
)
parser.add_argument(
    "--Gk-numerical-queue",
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
    "--Gk-source-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the GkSource builder work queue",
)
parser.add_argument(
    "--quad-source-integral-queue",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run the QuadSourceIntegral work queue",
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
    "--drop",
    type=str,
    nargs="+",
    default=[],
    choices=allowed_drop_actions,
    help="drop one or more data categories",
    action="extend",
)
parser.add_argument(
    "--dump-incomplete",
    type=str,
    default=None,
    help="directory to be used to dump details of incomplete GkSource objects for diagnostic purposes (only dumps computed objects, not those deserialized from the datastore)",
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

VERSION_LABEL = "2024.1.1"

specified_drop_actions = [x.lower() for x in args.drop]
drop_actions = [x for x in specified_drop_actions if x in allowed_drop_actions]

profile_agent = None
if args.profile_db is not None:
    if args.job_name is not None:
        label = f'{VERSION_LABEL}-jobname-"{args.job_name}"-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'
    else:
        label = f'{VERSION_LABEL}-primarydb-"{args.database}"-shards-{args.shards}-{datetime.now().replace(microsecond=0).isoformat()}'

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
    shards=args.shards,
    profile_agent=profile_agent,
    job_name=args.job_name,
    prune_unvalidated=args.prune_unvalidated,
    drop_actions=drop_actions,
) as pool:

    # set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
    units = Mpc_units()
    params = Planck2018()
    LambdaCDM_Planck2018 = ray.get(
        pool.object_get("LambdaCDM", params=params, units=units)
    )

    zend = args.zend
    source_samples_per_log10z = args.source_samples_log10z
    response_sparseness = args.response_sparseness

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
    atol, rtol, quad_atol, quad_rtol = ray.get(
        [
            pool.object_get("tolerance", tol=DEFAULT_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_REL_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_ATOL),
            pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_RTOL),
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

    # create the GkSource policies that we will apply later
    GkSource_policy_1pt5, GkSource_policy_5pt0 = ray.get(
        [
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-numeric"-Levin-threshold="1.5"',
                Levin_threshold=1.5,
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

    ## STEP 1
    ## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES

    print("\n** BUILDING ARRAY OF WAVENUMBERS AT WHICH TO SAMPLE")

    # Build array of k-sample points covering the region of the primoridal power spectrum that we want to include
    # in the one-loop integral. We will eventually evaluate the one-loop integral on a square grid
    # of source_k_array x source_k_array

    # For a spike of power to produce a PBH, we want the peak to be around k = 1E6/Mpc.
    # We choose a mesh of k-values around this point (with a longer tail to the IR)
    source_k_array = ray.get(
        convert_to_wavenumbers(np.logspace(np.log10(1e3), np.log10(5e7), 10))
    )
    source_k_sample = wavenumber_array(k_array=source_k_array)

    # build array of k-sample points covering the region of the target power spectrum where we want to evaluate
    # the one-loop integral
    response_k_array = ray.get(
        convert_to_wavenumbers(np.logspace(np.log10(1e3), np.log10(5e7), 10))
    )
    response_k_sample = wavenumber_array(k_array=response_k_array)

    def build_k_exit_work(k: wavenumber):
        return pool.object_get(
            "wavenumber_exit_time",
            k=k,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        )

    # for each k mode we sample, determine its horizon exit point
    source_k_exit_queue = RayWorkPool(
        pool,
        source_k_sample,
        task_builder=build_k_exit_work,
        title="CALCULATE HORIZON EXIT TIMES FOR SOURCE K-SAMPLE",
        store_results=True,
    )
    source_k_exit_queue.run()
    source_k_exit_times = wavenumber_exit_time_array(source_k_exit_queue.results)

    response_k_exit_queue = RayWorkPool(
        pool,
        response_k_sample,
        task_builder=build_k_exit_work,
        title="CALCULATE HORIZON EXIT TIMES FOR RESPONSE K-SAMPLE",
        store_results=True,
    )
    response_k_exit_queue.run()
    response_k_exit_times = wavenumber_exit_time_array(response_k_exit_queue.results)

    ## STEP 2
    ## BUILD A UNIVERSAL GRID OF Z-VALUES AT WHICH TO SAMPLE

    print("\n** BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE")

    full_k_exit_times = source_k_exit_times + response_k_exit_times
    k_exit_earliest: wavenumber_exit_time = full_k_exit_times.max
    print(
        f"   @@ earliest horizon exit/re-entry time is {k_exit_earliest.k.k_inv_Mpc:.5g}/Mpc with z_exit={k_exit_earliest.z_exit:.5g}"
    )

    # build a log-spaced universal grid of source sample times

    source_z_grid = k_exit_earliest.populate_z_sample(
        outside_horizon_efolds=5,
        samples_per_log10z=source_samples_per_log10z,
        z_end=zend,
    )

    # embed these redshift list into the database
    z_source_array = ray.get(convert_to_redshifts(source_z_grid))
    z_source_sample = redshift_array(z_array=z_source_array)

    # we need the response sample to be a subset of the source sample, otherwise we won't have
    # source data for the Green's functions right the way down to the response time
    z_response_sample = z_source_sample.winnow(sparseness=response_sparseness)

    # build tags and other labels
    (
        TkProductionTag,
        GkProductionTag,
        SourceZGridSizeTag,  # labels size of the z_source sample grid
        ResponseZGridSizeTag,  # labels size of the z_response sample grid
        OutsideHorizonEfoldsTag,  # labels number of e-folds outside the horizon at which we begin Tk numerical integrations
        LargestSourceZTag,  # labels largest z in the global grid
        SourceSamplesPerLog10ZTag,  # labels number of redshifts per log10 interval of 1+z in the source grid
        ResponseSparsenessZTag,  # labels number of redshifts per log10 interval of 1+z in the response grid
    ) = ray.get(
        [
            pool.object_get("store_tag", label="TkOneLoopDensity"),
            pool.object_get("store_tag", label="GkOneLoopDensity"),
            pool.object_get(
                "store_tag", label=f"SourceRedshiftGrid_{len(z_source_sample)}"
            ),
            pool.object_get(
                "store_tag", label=f"ReponseRedshiftGrid_{len(z_response_sample)}"
            ),
            pool.object_get("store_tag", label=f"OutsideHorizonEfolds_e3"),
            pool.object_get(
                "store_tag", label=f"LargestSourceRedshift_{z_source_sample.max.z:.5g}"
            ),
            pool.object_get(
                "store_tag", label=f"SourceSamplesPerLog10Z_{source_samples_per_log10z}"
            ),
            pool.object_get(
                "store_tag",
                label=f"ResponseSparsenessZ_{response_sparseness}",
            ),
        ]
    )

    ## STEP 1a
    ## BAKE THE BACKGROUND COSMOLOGY INTO A BACKGROUND MODEL OBJECT
    model: BackgroundModel = ray.get(
        pool.object_get(
            "BackgroundModel",
            solver_labels=solvers,
            cosmology=LambdaCDM_Planck2018,
            z_sample=z_source_sample,
            atol=atol,
            rtol=rtol,
            tags=[LargestSourceZTag, SourceSamplesPerLog10ZTag],
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

    # set up a proxy object to avoid having to repeatedly serialize the model instance and ship it out
    model_proxy = ModelProxy(model)

    print("\n** BUILDING BESSEL FUNCTION SPLINES")
    phase_start = time.perf_counter()

    # find largest argument of the Bessel functions J_nu(k c_s \tau), Y_nu(k c_s \tau) that we will need
    # these are
    largest_source_k = source_k_exit_times.max.k
    mod_f = model.functions
    largest_tau = mod_f.tau(zend)
    largest_x = largest_source_k.k * largest_tau
    largest_x_with_clearance = 1.075 * largest_x

    b_value = 0.0

    print(
        f"   @@ largest source k = {largest_source_k.k_inv_Mpc:.5g}/Mpc, latest tau = {largest_tau:.5g} (for z={zend:.5g}), largest x={largest_x:.5g}, largest x with 7.5% clearance={largest_x_with_clearance:.5g}"
    )

    # tight tolerances are needed to compute the Liouville-Green phase function to good accuracy up to large values
    # of the Bessel function argument x. We compute the phase in the form x Q where Q -> 1 at large x, so getting the phase
    # accurately means keeping Q very accurately close to 1 as the integration proceeds.
    # Internally, the bessel_phase() function uses the Dormand-Prince 8,5(3) stepper to compute a high accuracy solution.
    Bessel_0pt5 = bessel_phase(
        0.5 + b_value, largest_x_with_clearance, atol=1e-25, rtol=1e-16
    )
    Bessel_0pt5_proxy = BesselPhaseProxy(Bessel_0pt5)

    Bessel_2pt5 = bessel_phase(
        2.5 + b_value, largest_x_with_clearance, atol=1e-25, rtol=1e-16
    )
    Bessel_2pt5_proxy = BesselPhaseProxy(Bessel_2pt5)

    phase_end = time.perf_counter()
    print(f"   @@ constructed splines in time {format_time(phase_end - phase_start)}")

    ## STEP 2
    ## COMPUTE MATTER TRANSFER FUNCTIONS

    def build_Tk_work(batch: List[wavenumber_exit_time]):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        query_batch = [
            {
                "solver_labels": [],
                "model": model_proxy,
                "k": k_exit,
                "z_sample": None,
                "z_init": None,
                "atol": atol,
                "rtol": rtol,
                "tags": [
                    TkProductionTag,
                    SourceZGridSizeTag,
                    OutsideHorizonEfoldsTag,
                    LargestSourceZTag,
                    SourceSamplesPerLog10ZTag,
                ],
                "_do_not_populate": True,
            }
            for k_exit in batch
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get("TkNumericalIntegration", **x),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        query_queue.run()

        missing = [
            k_exit
            for k_exit, obj in zip(batch, query_queue.results)
            if not obj.available
        ]

        if len(missing) == 0:
            return []

        work_refs = []

        for k_exit in missing:
            my_sample = z_source_sample.truncate(k_exit.z_exit_suph_e5, keep="lower")
            work_refs.append(
                pool.object_get(
                    "TkNumericalIntegration",
                    solver_labels=solvers,
                    model=model_proxy,
                    k=k_exit,
                    z_sample=my_sample,
                    z_init=my_sample.max,
                    atol=atol,
                    rtol=rtol,
                    tags=[
                        TkProductionTag,
                        SourceZGridSizeTag,
                        OutsideHorizonEfoldsTag,
                        LargestSourceZTag,
                        SourceSamplesPerLog10ZTag,
                    ],
                    delta_logz=1.0 / float(source_samples_per_log10z),
                )
            )

        return work_refs

    def build_Tk_work_label(Tk: TkNumericalIntegration):
        return f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def compute_Tk_work(Tk: TkNumericalIntegration, label: Optional[str] = None):
        return Tk.compute(label=label)

    def validate_Tk_work(Tk: TkNumericalIntegration):
        if not Tk.available:
            raise RuntimeError(
                "TkNumericalIntegration object passed for validation, but is not yet available"
            )

        return pool.object_validate(Tk)

    def build_tensor_source_work(batch):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        binned_batch = {}
        for q, r in batch:
            if q not in binned_batch:
                binned_batch[q] = []
            binned_batch[q].append(r)

        # freeze q labels in a defined order, so we are guaranteed to know what order they come back in
        q_keys = list(binned_batch.keys())

        query_batch = [
            {
                "shard_key": {"q": q},
                "payload": [
                    {
                        "model": model_proxy,
                        "r": r,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            TkProductionTag,
                            SourceZGridSizeTag,
                            OutsideHorizonEfoldsTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                        ],
                        "_do_not_populate": True,
                    }
                    for r in binned_batch[q]
                ],
            }
            for q in q_keys
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "QuadSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        query_queue.run()

        missing_labels = [
            (q, r)
            for q, query_outcomes in zip(q_keys, query_queue.results)
            for r, obj in zip(binned_batch[q], query_outcomes)
            if not obj.available
        ]

        if len(missing_labels) == 0:
            return []

        missing_set = set(q for q, r in missing_labels)
        missing_set.update(r for q, r in missing_labels)
        missing_Tk = list(missing_set)

        Tk_lookup_batch = [
            {
                "model": model_proxy,
                "z_sample": None,
                "k": k,
                "z_init": None,
                "atol": atol,
                "rtol": rtol,
                "tags": [
                    TkProductionTag,
                    SourceZGridSizeTag,
                    OutsideHorizonEfoldsTag,
                    LargestSourceZTag,
                    SourceSamplesPerLog10ZTag,
                ],
            }
            for k in missing_Tk
        ]

        Tk_lookup_queue = RayWorkPool(
            pool,
            Tk_lookup_batch,
            task_builder=lambda x: pool.object_get("TkNumericalIntegration", **x),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        Tk_lookup_queue.run()

        Tk_cache = {
            k.store_id: obj for k, obj in zip(missing_Tk, Tk_lookup_queue.results)
        }

        work_refs = []

        for q, r in missing_labels:
            q: wavenumber_exit_time
            r: wavenumber_exit_time

            # ensure that q, r are ordered respecting the triangular condition
            assert q.k <= r.k

            Tq: TkNumericalIntegration = Tk_cache[q.store_id]
            Tr: TkNumericalIntegration = Tk_cache[r.store_id]

            missing_data = False
            if not Tq.available:
                missing_data = True
                print(
                    f"!! MISSING DATA WARNING ({datetime.now().replace(microsecond=0).isoformat()}) TkNumericalIntegration for q={q.k.k_inv_Mpc}/Mpc"
                )

            if not Tr.available:
                missing_data = True
                print(
                    f"!! MISSING DATA WARNING ({datetime.now().replace(microsecond=0).isoformat()}) TkNumericalIntegration for r={r.k.k_inv_Mpc}/Mpc"
                )

            if missing_data:
                raise RuntimeError(
                    f"QuadSource builder: missing or incomplete source data for q={q.k.k_inv_Mpc}/Mpc, r={r.k.k_inv_Mpc}/Mpc"
                )

            work_refs.append(
                {
                    "ref": pool.object_get(
                        "QuadSource",
                        model=model_proxy,
                        z_sample=z_source_sample,
                        q=q,
                        r=r,
                        tags=[
                            TkProductionTag,
                            SourceZGridSizeTag,
                            OutsideHorizonEfoldsTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                        ],
                    ),
                    "compute_payload": {"Tq": Tq, "Tr": Tr},
                }
            )

        return work_refs

    def compute_tensor_source_work(
        calc: QuadSource, payload, label: Optional[str] = None
    ):
        return calc.compute(payload=payload, label=label)

    def validate_tensor_source_work(calc: QuadSource):
        if not calc.available:
            raise RuntimeError(
                "QuadSource object passed for validation, but is not yet available"
            )

        return pool.object_validate(calc)

    def build_tensor_source_work_label(calc: QuadSource):
        q = calc.q
        r = calc.r
        return f"{args.job_name}-tensor-src-q{q.k_inv_Mpc:.3g}-r{r.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"

    if args.Tk_numerical_queue:
        Tk_numerical_work_batch = list(
            grouper(source_k_exit_times, n=50, incomplete="fill")
        )

        Tk_queue = RayWorkPool(
            pool,
            Tk_numerical_work_batch,
            task_builder=build_Tk_work,
            compute_handler=compute_Tk_work,
            validation_handler=validate_Tk_work,
            label_builder=build_Tk_work_label,
            title="CALCULATE MATTER TRANSFER FUNCTIONS",
            store_results=False,
            create_batch_size=1,
            max_task_queue=300,
            process_batch_size=20,
        )
        Tk_queue.run()

        # the source term is obviously symmetric, so we do not need to compute QuadSource on
        # a square q x r grid; it will suffice to do so on an upper or lower triangle where q <= r.
        # Then we can obtain the other values by reflection.

        # The ordering of pairs should be stable, because source_k_exit_times is ordered.
        tensor_source_grid = list(
            grouper(
                itertools.combinations_with_replacement(source_k_exit_times, 2),
                n=50,
                incomplete="fill",
            )
        )

        QuadSource_queue = RayWorkPool(
            pool,
            tensor_source_grid,
            task_builder=build_tensor_source_work,
            validation_handler=validate_tensor_source_work,
            label_builder=build_tensor_source_work_label,
            title="CALCULATE QUADRATIC SOURCE TERMS",
            store_results=False,
            create_batch_size=1,
            max_task_queue=300,
            process_batch_size=20,
        )
        QuadSource_queue.run()

    ## STEP 3
    ## COMPUTE TENSOR GREEN'S FUNCTIONS NUMERICALLY FOR RESPONSE TIMES NOT TOO FAR INSIDE THE HORIZON

    def build_Gk_numerical_work(batch: List[redshift]):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        query_batch = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "solver_labels": [],
                        "model": model_proxy,
                        "z_source": z_source,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,  # restrict query to integrations with the correct source grid size
                            ResponseZGridSizeTag,  # restrict query to integrations with the correct response grid size
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                        "_do_not_populate": True,
                    }
                    for z_source in batch
                ],
            }
            for k_exit in response_k_exit_times
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkNumericalIntegration", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        query_queue.run()

        missing = {
            k_exit.store_id: [
                z_source
                for obj, z_source in zip(query_outcomes, batch)
                if not obj.available
            ]
            for query_outcomes, k_exit in zip(
                query_queue.results, response_k_exit_times
            )
        }

        work_refs = []

        for k_exit in response_k_exit_times:
            k_exit: wavenumber_exit_time

            # find redshift where this k-mode is at least 3 efolds inside the horizon

            # we aim not to calculate numerical Green's functions with the response redshift
            # (and hence the source redshift) later than this, because
            # the oscillations become rapid, and we are better switching to a WKB approximation.
            # Typically we cannot continue the numerical integratioan more than about 6 e-folds
            # inside the horizon.
            # However, we do need a bit of leeway. We sometimes cannot get WKB values for
            # z_source/z_response combinations that are both close to the 3-efold crossover point.
            # This is because we don't find a boundary conditions for the WKB evolution until
            # some time after z_source.
            # The upshot is that we don't find enough overlap between the numerical and WKB regions
            # to allow a smooth handover. To deal with this, we allow z_source to go as far as
            # 4 e-folds inside the horizon.
            for z_source in missing[k_exit.store_id]:
                z_source: redshift

                if z_source.z > k_exit.z_exit_subh_e4 - DEFAULT_FLOAT_PRECISION:
                    # cut down response zs to those that are (1) later than the source, and (2) earlier than the 6-efolds-inside-the-horizon point
                    # (here with a 10% tolerance)
                    response_zs = z_response_sample.truncate(
                        z_source, keep="lower"
                    ).truncate(0.85 * k_exit.z_exit_subh_e6, keep="higher-include")

                    if len(response_zs) > 0:
                        work_refs.append(
                            pool.object_get(
                                "GkNumericalIntegration",
                                solver_labels=solvers,
                                model=model_proxy,
                                k=k_exit,
                                z_source=z_source,
                                z_sample=response_zs,
                                atol=atol,
                                rtol=rtol,
                                tags=[
                                    GkProductionTag,
                                    SourceZGridSizeTag,
                                    ResponseZGridSizeTag,
                                    LargestSourceZTag,
                                    SourceSamplesPerLog10ZTag,
                                    ResponseSparsenessZTag,
                                ],
                                delta_logz=1.0 / float(source_samples_per_log10z),
                                mode="stop",
                                _do_not_populate=True,  # instructs the datastore not to read in the contents of this object
                            )
                        )

        return work_refs

    def build_Gk_numerical_work_label(Gk: GkNumericalIntegration):
        return f"{args.job_name}-GkNumerical-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def compute_Gk_numerical_work(
        Gk: GkNumericalIntegration, label: Optional[str] = None
    ):
        return Gk.compute(label=label)

    def validate_Gk_numerical_work(Gk: GkNumericalIntegration):
        if not Gk.available:
            raise RuntimeError(
                "GkNumericalIntegration object passed for validation, but is not yet available"
            )

        return pool.object_validate(Gk)

    if args.Gk_numerical_queue:
        GkNumerical_work_batches = list(
            grouper(z_source_sample, n=50, incomplete="fill")
        )

        Gk_numerical_queue = RayWorkPool(
            pool,
            GkNumerical_work_batches,
            task_builder=build_Gk_numerical_work,
            validation_handler=validate_Gk_numerical_work,
            label_builder=build_Gk_numerical_work_label,
            title="CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS",
            store_results=False,
            create_batch_size=2,
            notify_batch_size=2000,
            max_task_queue=100,
            process_batch_size=50,
        )
        Gk_numerical_queue.run()

    ## STEP 4
    ## COMPUTE TENSOR GREEN'S FUNCTIONS IN THE WKB APPROXIMATION FOR RESPONSE TIMES INSIDE THE HORIZON

    def build_Gk_WKB_work(batch: List[redshift]):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        # build GKWKBIntegration work items for a batch of z_source times

        # first, we query for existing objects in the datastore matching these z_source times.
        # if they are present (and validated) we assume them to be complete.
        # We don't attempt to recompute objects that are already in the store.
        query_batch = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "solver_labels": [],
                        "model": model_proxy,
                        "z_source": z_source,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,  # restrict query to integrations with the correct source grid size
                            ResponseZGridSizeTag,  # restrict query to integrations with the correct response grid size
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                        "_do_not_populate": True,
                    }
                    for z_source in batch
                ],
            }
            for k_exit in response_k_exit_times
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkWKBIntegration", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        query_queue.run()

        # determine which objects are missing
        missing = {
            k_exit.store_id: [
                z_source
                for obj, z_source in zip(query_outcomes, batch)
                if not obj.available
            ]
            for query_outcomes, k_exit in zip(
                query_queue.results, response_k_exit_times
            )
        }

        # Query for a GkNumericalIntegration instances for each combination of model, k_exit, and z_source.
        # If one exists, it will be used to set initial conditions for the WKB part of the evolution.
        # We want to do this in a vectorized way, pulling an entire batch of GkNumericalIntegration instances
        # for a range of z-source values from the same shard. This is more efficient than paying the Ray and database
        # overhead for multiple lookups
        payload_batch = [
            {
                # note we don't specify the k mode as part of the main payload
                # instead we specify it separately to object_get_vectorized()
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "solver_labels": [],
                        "model": model_proxy,
                        "z_source": z_source,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    }
                    for z_source in missing[k_exit.store_id]
                ],
            }
            for k_exit in response_k_exit_times
        ]

        lookup_queue = RayWorkPool(
            pool,
            payload_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkNumericalIntegration", x["shard_key"], payload_data=x["payload"]
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
            z_source.store_id: z_response_sample.truncate(z_source, keep="lower")
            for z_source in batch
        }

        for k_exit, Gk_data in zip(response_k_exit_times, lookup_queue.results):
            k_exit: wavenumber_exit_time
            Gk_data: List[GkNumericalIntegration]

            # use numerical initial conditions up to geometric mean of 3- and 4-efold points (inside the horizon)
            # see long comment below
            z_source_limit = sqrt(k_exit.z_exit_subh_e3 * k_exit.z_exit_subh_e4)

            for z_source, Gk in zip(missing[k_exit.store_id], Gk_data):
                # be defensive about ensuring provenance for our data products
                Gk: GkNumericalIntegration
                assert Gk._k_exit.store_id == k_exit.store_id
                assert Gk._z_source.store_id == z_source.store_id

                response_sample = response_pools[z_source.store_id]

                if len(response_sample) == 0:
                    continue

                # query whether there is a pre-computed GkNumericalIntegration for this source redshift.
                # typically, this will be the case provided the source redshift is no more than
                # 4 e-folds inside the horizon, because that is where we cut the GkNumericalIntegration work queue.

                # Notice that, if a GkNumericalIntegration object is available, we don't get to decide where the
                # initial condition is; we just have to use the initial condition that is available.
                # That might be some time later than z_source. Then, we will not get WKB solutions for
                # z_response between z_source and the initial condition. The can lead to gaps in the WKB data
                # when we later assemble GkSource objects.

                # The idea is to switch to a WKB solution fairly soon after 3 e-folds inside the horizon.
                # In particular, we shouldn't keep using the GkNumericalIntegration boundary condition
                # right up to the boundary where we stop computing it, because that risks the first available z_response
                # for such very late z_source bouncing around erratically in a region where we depend on having it.
                # Instead, we should use the numerical boundary condition for z_source up to some point between the 3 and 4 e-fold points.
                # After that we should use a fully WKB analysis. Then we should have safely overlapping WKB and
                # numerical solutions around the 4-efold point.
                if z_source.z >= z_source_limit and Gk.available:
                    # obtain initial data and initial time from the GkNumericalIntegration object
                    G_init = Gk.stop_G
                    Gprime_init = Gk.stop_Gprime
                    z_init = k_exit.z_exit - Gk.stop_deltaz_subh

                    # Cut the response times so that we only extract responses later than 3-efolds inside the horizon,
                    # or after z_init, whichever is later
                    #
                    # Note that this means there may be z_source/z_response combinations that we do not get WKB values for.
                    # These can cause incompleteness when building GkSource objects (see below), if we are not
                    # careful to allow sufficient overlap with the purely numerical results.
                    max_response = min(k_exit.z_exit_subh_e3, z_init)
                    response_sample = response_sample.truncate(
                        max_response, keep="lower"
                    )

                    if len(response_sample) > 0:
                        work_refs.append(
                            pool.object_get(
                                "GkWKBIntegration",
                                solver_labels=solvers,
                                model=model_proxy,
                                k=k_exit,
                                z_source=z_source,
                                z_init=z_init,
                                G_init=G_init,
                                Gprime_init=Gprime_init,
                                z_sample=response_sample,
                                atol=atol,
                                rtol=rtol,
                                tags=[
                                    GkProductionTag,
                                    SourceZGridSizeTag,
                                    ResponseZGridSizeTag,
                                    LargestSourceZTag,
                                    SourceSamplesPerLog10ZTag,
                                    ResponseSparsenessZTag,
                                ],
                                _do_not_populate=True,  # can specify this, but we know every object we query will not exist in the datastore, so redundant really
                            )
                        )
                else:
                    # no pre-computed initial condition available.
                    # This is OK and expected if the z_source is sufficiently far after horizon re-entry,
                    # but we should warn if we expect an initial condition to be available
                    if z_source.z > k_exit.z_exit_subh_e3 - DEFAULT_FLOAT_PRECISION:
                        print(
                            f"!! WARNING: no numerically-computed initial conditions is available for k={k_exit.k.k_inv_Mpc:.5g}/Mpc, z_source={z_source.z:.5g}"
                        )
                        print(
                            f"|    This may indicate that GkNumericalIntegration records for all necessary initial conditions are not present in the datastore."
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
                            model=model_proxy,
                            k=k_exit,
                            z_source=z_source,
                            G_init=0.0,
                            Gprime_init=1.0,
                            z_sample=response_sample,
                            atol=atol,
                            rtol=rtol,
                            tags=[
                                GkProductionTag,
                                SourceZGridSizeTag,
                                ResponseZGridSizeTag,
                                LargestSourceZTag,
                                SourceSamplesPerLog10ZTag,
                                ResponseSparsenessZTag,
                            ],
                        )
                    )

        return work_refs

    def build_Gk_WKB_work_label(Gk: GkWKBIntegration):
        return f"{args.job_name}-GkWKB-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-zinit{Gk.z_init:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def compute_Gk_WKB_work(Gk: GkWKBIntegration, label: Optional[str] = None):
        return Gk.compute(label=label)

    def validate_Gk_WKB_work(Gk: GkWKBIntegration):
        if not Gk.available:
            raise RuntimeError(
                "GkWKBIntegration object passed for validation, but is not yet available"
            )

        return pool.object_validate(Gk)

    if args.WKB_queue:
        GkWKB_work_batches = list(grouper(z_source_sample, n=20, incomplete="fill"))

        Gk_WKB_queue = RayWorkPool(
            pool,
            GkWKB_work_batches,
            task_builder=build_Gk_WKB_work,
            compute_handler=compute_Gk_WKB_work,
            validation_handler=validate_Gk_WKB_work,
            label_builder=build_Gk_WKB_work_label,
            title="CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS",
            store_results=False,
            create_batch_size=2,
            notify_batch_size=2000,
            max_task_queue=100,
            process_batch_size=50,
        )
        Gk_WKB_queue.run()

    # STEP 5
    # REBUILD TENSOR GREEN'S FUNCTIONS AS FUNCTIONS OF THE SOURCE REDSHIFT, RATHER THAN THE RESPONSE REDSHIFT

    # (In theory we could construct G_k as functions of the source redshift z' on-the-fly when we need them.
    # Storing the rebuilt tensor Green's functions in the datastore is, in this sense, redundant. But it is likely much faster
    # to pre-build and cache, because we need to perform complicated table lookups with joins in order to extract the relevant
    # solution points from GkNumericalValue and GkWKValue rows. Also, there is the reproducibility angle of keeping a record of what
    # rebuilt data products we used.)

    def build_GkSource_batch(batch: List[redshift]):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        # try to do parallel lookup of the GkNumericalIntegration/GkWKBIntegration records needed
        # to process this batch
        z_source_pool = {
            z_response.store_id: z_source_sample.truncate(z_response, keep="higher")
            for z_response in batch
        }

        # first, determine which GkSource objects are missing from the datastore. We do not rebuild (or even re-query) any that are already
        # present (although we do not attempt to validate them either). Even querying has an impact: scheduling a remote database lookup,
        # serialization/deserialization of the output product, etc.
        query_batch = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_response": z_response,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                        "_do_not_populate": True,
                    }
                    for z_response in batch
                ],
            }
            for k_exit in response_k_exit_times
        ]

        # we use object_get_vectorized() to query a whole batch of GkSource objects on the same shard. This helps amortize the overheads
        # of the remote database lookup.
        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        query_queue.run()

        # determine which z_response values are missing for each k_exit value
        missing = {
            k_exit.store_id: [
                z_response
                for obj, z_response in zip(query_outcomes, batch)
                if not obj.available
            ]
            for query_outcomes, k_exit in zip(
                query_queue.results, response_k_exit_times
            )
        }

        work_refs = build_missing_GkSource(missing, z_source_pool)

        return work_refs

    def build_missing_GkSource(missing, z_source_pool):
        # all the z-source values for a single k-mode will be held on the same database shard.
        # It's critical that we look these up efficiently. We use the object_read_batch() API
        # so that we can collect an entire set of z_source values for a given z_response value,
        # in a single database query
        payload_batch = [
            {
                # note we don't specify the k mode as part of the main payload
                # instead we specify it separately to object_read_batch()
                "shard_key": {"k": k_exit},
                "object": cls_name,
                "payload": {
                    "model": model_proxy,
                    "z": z_response,  # no specification of z_source; means we read all available z_source values
                    "atol": atol,
                    "rtol": rtol,
                    "tags": [
                        GkProductionTag,
                        SourceZGridSizeTag,
                        ResponseZGridSizeTag,
                        LargestSourceZTag,
                        SourceSamplesPerLog10ZTag,
                        ResponseSparsenessZTag,
                    ],
                },
            }
            for k_exit in response_k_exit_times
            for z_response in missing[k_exit.store_id]
            for cls_name in ["GkNumericalValue", "GkWKBValue"]
        ]
        lookup_queue = RayWorkPool(
            pool,
            payload_batch,
            task_builder=lambda x: pool.object_read_batch(
                x["object"], x["shard_key"], **x["payload"]
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

        i = 0
        for k_exit in response_k_exit_times:
            k_exit: wavenumber_exit_time

            for z_response in missing[k_exit.store_id]:
                z_response: redshift
                GkNs = lookup_queue.results[i]
                GkWs = lookup_queue.results[i + 1]

                i += 2

                numeric_data = {}
                WKB_data = {}

                for GkN in GkNs:
                    GkN: GkNumericalValue
                    assert isinstance(GkN, GkNumericalValue)

                    # the _k_exit and _z_source fields are not part of the public API for these
                    # objects, but they are set by the data store after deserialization
                    assert GkN._k_exit.store_id == k_exit.store_id
                    assert GkN.z.store_id == z_response.store_id

                    numeric_data[GkN._z_source.store_id] = GkN

                for GkW in GkWs:
                    GkW: GkWKBValue
                    assert isinstance(GkW, GkWKBValue)

                    assert GkW._k_exit.store_id == k_exit.store_id
                    assert GkW.z.store_id == z_response.store_id

                    WKB_data[GkW._z_source.store_id] = GkW

                # ensure there is at least one numeric or WKB data point for every source redshift we need
                missing_sources = [
                    z_source
                    for z_source in z_source_pool[z_response.store_id]
                    if z_source.store_id not in numeric_data
                    and z_source.store_id not in WKB_data
                ]
                if len(missing_sources) > 0:
                    print(
                        f"!! MISSING DATA WARNING ({datetime.now().replace(microsecond=0).isoformat()}) when building GkSource for k={k_exit.k.k_inv_Mpc:.5g}/Mpc, z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                    )
                    for missing_zsource in missing_sources[:20]:
                        print(
                            f"|  -- z_source={missing_zsource.z:.5g} (store_id={missing_zsource.store_id})"
                        )
                    expected = len(z_source_pool[z_response.store_id])
                    missing = len(missing_sources)
                    if missing > 15:
                        print("|  ... (more missing redshifts skipped)")
                    print(
                        f"|     total missing = {missing} out of {expected} = {missing / expected:.2%}"
                    )
                    raise RuntimeError(
                        f"GkSource builder: missing or incomplete source data for k={k_exit.k.k_inv_Mpc:.5g}/Mpc, z_response={z_response.z:.5g}"
                    )

                # note that the payload for each GkSource workload can be quite large, so we do not want to overload the queue
                # with pending compute/store/validate tasks. These all involve passing around the (possibly large) resulting object.
                work_refs.append(
                    {
                        "ref": pool.object_get(
                            "GkSource",
                            model=model_proxy,
                            k=k_exit,
                            atol=atol,
                            rtol=rtol,
                            z_response=z_response,
                            z_sample=z_source_pool[z_response.store_id],
                            tags=[
                                GkProductionTag,
                                SourceZGridSizeTag,
                                ResponseZGridSizeTag,
                                LargestSourceZTag,
                                SourceSamplesPerLog10ZTag,
                                ResponseSparsenessZTag,
                            ],
                        ),
                        "compute_payload": {
                            "numeric": numeric_data,
                            "WKB": WKB_data,
                        },
                    }
                )

        return work_refs

    def build_GkSource_work_label(Gk: GkSource):
        return f"{args.job_name}-GkSource-k{Gk.k.k_inv_Mpc:.3g}-responsez{Gk.z_response.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def compute_GkSource_work(Gk: GkSource, payload, label: Optional[str] = None):
        return Gk.compute(payload=payload, label=label)

    def validate_GkSource_work(Gk: GkSource):
        if not Gk.available:
            raise RuntimeError(
                "GkSource object passed for validation, but is not yet available"
            )

        return pool.object_validate(Gk)

    dump_incomplete = args.dump_incomplete is not None
    dump_incomplete_path = (
        Path(args.dump_incomplete).resolve() if dump_incomplete else None
    )

    if dump_incomplete:
        print(
            f'\n** INCOMPLETE GkSourcePolicyData OBJECTS WILL BE DUMPED TO "{dump_incomplete_path}"'
        )

    @ray.remote
    def dump_incomplete_GkSourcePolicy(obj: GkSourcePolicyData):
        output_folder = (
            dump_incomplete_path
            / f"store_id={obj.store_id}_k={obj.k.k_inv_Mpc:.5g}_zresponse={obj.z_response.z:.5g}"
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        source = obj._source_proxy.get()

        data = {
            "policy_store_id": obj.store_id,
            "type": obj.type,
            "quality": obj.quality,
            "crossover_z": obj.crossover_z,
            "metadata": obj.metadata,
            "source_store_id": source.store_id,
            "model_store_id": source.model_proxy.store_id,
            "k_inv_Mpc": source.k.k_inv_Mpc,
            "k_store_id": source.k.store_id,
            "k_exit_store_id": source._k_exit.store_id,
            "z_exit": source._k_exit.z_exit,
            "z_exit_subh_e3": source._k_exit.z_exit_subh_e3,
            "z_exit_subh_e4": source._k_exit.z_exit_subh_e4,
            "z_exit_subh_e5": source._k_exit.z_exit_subh_e5,
            "z_exit_subh_e6": source._k_exit.z_exit_subh_e6,
            "z_response": source.z_response.z,
            "z_response_store_id": source.z_response.store_id,
            "z_response_efolds_subh": model.efolds_subh(source.k, source.z_response),
            "numerical_z_source_limit": sqrt(
                source._k_exit.z_exit_subh_e3 * source._k_exit.z_exit_subh_e4
            ),
            "numerical_z_source_limit_efolds": model.efolds_subh(
                source.k,
                sqrt(source._k_exit.z_exit_subh_e3 * source._k_exit.z_exit_subh_e4),
            ),
            "z_sample": [z.z for z in source.z_sample],
            "z_sample_size": len(source.z_sample),
            "values_size": len(source.values) if source.values is not None else None,
            "numerical_smallest_z": (
                source.numerical_smallest_z.z
                if source.numerical_smallest_z is not None
                else None
            ),
            "numerical_smallest_z_store_id": (
                source.numerical_smallest_z.store_id
                if source.numerical_smallest_z is not None
                else None
            ),
            "primary_WKB_largest_z": (
                source.primary_WKB_largest_z.z
                if source.primary_WKB_largest_z is not None
                else None
            ),
            "primary_WKB_largest_z_store_id": (
                source.primary_WKB_largest_z.store_id
                if source.primary_WKB_largest_z is not None
                else None
            ),
            "label": source.label,
            "tags": [tag.label for tag in source.tags],
        }

        data_file = output_folder / "data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        df: DataFrame = source.values_as_DataFrame()
        csv_file = output_folder / "values.csv"
        df.to_csv(csv_file, header=True, index=False)

    GkSourcePolicy_statistics = {}

    def postprocess_GkSourcePolicy(obj: GkSourcePolicyData):
        global GkSourcePolicy_total

        policy: GkSourcePolicy = obj.policy
        policy_label = policy.label
        key = policy_label if policy_label is not None else policy.store_id

        if key not in GkSourcePolicy_statistics:
            GkSourcePolicy_statistics[key] = {"total": 0, "statistics": {}}
        policy_statistics = GkSourcePolicy_statistics[key]

        type: str = obj.type
        quality: str = obj.quality

        statistics = policy_statistics["statistics"]
        if type not in statistics:
            statistics[type] = {}
        type_stats = statistics[type]

        if quality not in type_stats:
            type_stats[quality] = 0
        type_stats[quality] += 1

        policy_statistics["total"] = policy_statistics["total"] + 1

    def build_incomplete_GkSourcePolicy(incomplete):
        GkSource_lookup = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_response": z_response,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    }
                    for z_response in incomplete[k_exit.store_id]
                ],
            }
            for k_exit in response_k_exit_times
        ]

        GkSource_lookup_queue = RayWorkPool(
            pool,
            GkSource_lookup,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        GkSource_lookup_queue.run()
        GkSource_proxies = [
            [GkSourceProxy(obj) for obj in lookup_data]
            for k_exit, lookup_data in zip(
                response_k_exit_times, GkSource_lookup_queue.results
            )
        ]

        query_batch = [
            {
                # note we don't specify the k mode as part of the main payload
                # instead we specify it separately to object_read_batch()
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "source": source_proxy,
                        "policy": GkSource_policy_1pt5,
                        "k": k_exit,
                    }
                    for z_response, source_proxy in zip(
                        incomplete[k_exit.store_id], proxy_data
                    )
                ],
            }
            for k_exit, proxy_data in zip(response_k_exit_times, GkSource_proxies)
        ]

        def dump_Gk_handler(missing_Gk: List[GkSource]):
            for Gk in missing_Gk:
                dump_incomplete_GkSourcePolicy.remote(Gk)

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSourcePolicyData", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            post_handler=dump_Gk_handler,
            label_builder=None,
            title=None,
            store_results=False,
            create_batch_size=20,
            process_batch_size=20,
        )
        query_queue.run()

    def build_GkSourcePolicy_batch(batch: List[redshift]):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        GkSource_lookup = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_response": z_response,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                        "_do_not_populate": True,
                    }
                    for z_response in batch
                ],
            }
            for k_exit in response_k_exit_times
        ]

        GkSource_lookup_queue = RayWorkPool(
            pool,
            GkSource_lookup,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        GkSource_lookup_queue.run()
        GkSource_proxies = [
            [GkSourceProxy(obj) for obj in lookup_data]
            for k_exit, lookup_data in zip(
                response_k_exit_times, GkSource_lookup_queue.results
            )
        ]

        query_batch = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "source": source_proxy,
                        "policy": GkSource_policy_1pt5,
                        "k": k_exit,
                    }
                    for z_response, source_proxy in zip(batch, proxy_data)
                ],
            }
            for k_exit, proxy_data in zip(response_k_exit_times, GkSource_proxies)
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSourcePolicyData", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        query_queue.run()

        # apply the postprocess handler to any available objects, so that we can include them in our summary statistics
        # the work queue never sees these objects (they are never returned as ObjectRef instances from the task builder),
        # so this is our only chance to count them
        available = [
            obj
            for query_outcomes in query_queue.results
            for obj in query_outcomes
            if obj.available
        ]
        for obj in available:
            postprocess_GkSourcePolicy(obj)

        missing = {
            k_exit.store_id: [
                z_response
                for z_response, obj in zip(batch, query_outcomes)
                if not obj.available
            ]
            for k_exit, query_outcomes in zip(
                response_k_exit_times, query_queue.results
            )
        }

        incomplete = {
            k_exit.store_id: [
                z_response
                for obj, z_response in zip(query_outcomes, batch)
                if obj.available and obj.quality == "incomplete"
            ]
            for k_exit, query_outcomes in zip(
                response_k_exit_times, query_queue.results
            )
        }

        work_refs = build_missing_GkSourcePolicy(missing)
        if dump_incomplete:
            build_incomplete_GkSourcePolicy(incomplete)

        return work_refs

    def build_missing_GkSourcePolicy(missing):
        GkSource_lookup = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_response": z_response,
                        "z_sample": None,
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    }
                    for z_response in missing[k_exit.store_id]
                ],
            }
            for k_exit in response_k_exit_times
        ]

        GkSource_lookup_queue = RayWorkPool(
            pool,
            GkSource_lookup,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,  # may need to tweak relative to number of shards
            process_batch_size=20,
        )
        GkSource_lookup_queue.run()
        GkSource_proxies = [
            [GkSourceProxy(obj) for obj in lookup_data]
            for k_exit, lookup_data in zip(
                response_k_exit_times, GkSource_lookup_queue.results
            )
        ]

        work_refs = []

        for k_exit, proxy_data in zip(response_k_exit_times, GkSource_proxies):
            for z_response, source_proxy in zip(missing[k_exit.store_id], proxy_data):
                work_refs.append(
                    pool.object_get(
                        "GkSourcePolicyData",
                        source=source_proxy,
                        policy=GkSource_policy_1pt5,
                        k=k_exit,
                    )
                )

        return work_refs

    def compute_GkSourcePolicy_work(obj: GkSourcePolicyData):
        return obj.compute()

    def validate_GkSourcePolicy_work(obj: GkSourcePolicyData):
        if not obj.available:
            raise RuntimeError(
                "GkSourcePolicyData object passed for validation, but is not yet available"
            )

        if obj.quality == "incomplete":
            dump_incomplete_GkSourcePolicy.remote(obj)

    if args.Gk_source_queue:
        # For each wavenumber in the k-sample (here k_exit_times), and each value of z_response,
        # we need to build the Green's function for all possible values of z_source.
        # Even with 10 wavenumbers and 2,000 z-samples points, that is 20,000 items.
        # With 500 wavenumbers and 2,000 z-samples, it is 1 million items.
        # To process these efficiently, we break the queue up into batches, and try to
        # run a sub-queue that queries that needed data in parallel.

        GkSource_work_batches = list(
            grouper(z_response_sample, n=20, incomplete="fill")
        )

        # need to prune compute/store/validate tasks from the system very rapidly in order to prevent memory-consuming payloads
        # for a number of compute workloads from piling up.
        GkSource_queue = RayWorkPool(
            pool,
            GkSource_work_batches,
            task_builder=build_GkSource_batch,
            compute_handler=compute_GkSource_work,
            validation_handler=validate_GkSource_work,
            post_handler=None,
            label_builder=build_GkSource_work_label,
            title="CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT",
            store_results=False,
            create_batch_size=1,  # we have batched the work queue into chunks ourselves, so don't process too many of these chunks at once
            notify_batch_size=2000,
            max_task_queue=100,
            process_batch_size=50,
        )
        GkSource_queue.run()

        GkSourcePolicy_queue = RayWorkPool(
            pool,
            GkSource_work_batches,
            task_builder=build_GkSourcePolicy_batch,
            compute_handler=compute_GkSourcePolicy_work,
            validation_handler=None,
            post_handler=postprocess_GkSourcePolicy,
            label_builder=None,
            title="APPLY GKSOURCE POLICIES",
            store_results=False,
            create_batch_size=1,
            notify_batch_size=2000,
            max_task_queue=100,
            process_batch_size=50,
        )
        GkSourcePolicy_queue.run()

        print("\n** GkSourcePolicyData SUMMARY STATISTICS")
        for policy_id, policy_data in GkSourcePolicy_statistics.items():
            policy_total = policy_data["total"]
            print(f"\n     * Policy: {policy_id}")
            print(f"        Total instances: {policy_data['total']}")
            for Gk_type, type_stats in policy_data["statistics"].items():
                type_total = sum(type_stats.values())
                print(
                    f"\n        >> {Gk_type}: {type_total} instances = {type_total / policy_total:.2%}"
                )
                for quality, quality_total in type_stats.items():
                    print(
                        f"           {quality}: {quality_total} instances = {quality_total/type_total:.2%} of this type, {quality_total / policy_total:.2%} of total"
                    )

    # STEP 6
    # PERFORM CALCULATION OF SOURCE INTEGRALS

    # strip off a number of z-sample values so that we do not need to begin the source integration right at the upper limit, where we cannot construct a spline
    if len(z_source_sample) < 11:
        raise RuntimeError(
            "z_source_sample is too small. At least O(10) sample points (but most likely very many more) are required to effectively construct the source integral."
        )

    z_source_integral_max_z = z_source_sample[10]
    z_source_integral_response_sample = z_response_sample.truncate(
        z_source_integral_max_z, keep="lower-strict"
    )

    def build_QuadSourceIntegral_batch(batch):
        # grouper may fill with None values, so we need to strip those out
        batch = [x for x in batch if x is not None]

        # batch is a list of (z_response, k, q, r) pairs to be queried, and work scheduled to produce them if they are missing

        # to allow vectorized object_gets to the datastore, we need to bin these labels by shards. Then we can pull all (z, q, r)
        # combinations together from the same shard
        binned_batch = {}
        for z_response, k, q, r in batch:
            if k not in binned_batch:
                binned_batch[k] = []
            binned_batch[k].append((z_response, q, r))

        # freeze k labels in a defined order, so we are guaranteed to know what order they come back in
        k_keys = list(binned_batch.keys())

        # find which instances are missing from the datastore

        query_batch = [
            {
                "shard_key": {"k": k},
                "payload": [
                    {
                        "model": model_proxy,
                        "policy": GkSource_policy_1pt5,
                        "q": q,
                        "r": r,
                        "z_response": z_response,
                        "z_source_max": z_source_integral_max_z,
                        "atol": quad_atol,
                        "rtol": quad_rtol,
                        "tags": [
                            GkProductionTag,
                            TkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    }
                    for z_response, q, r in binned_batch[k]
                ],
            }
            for k in k_keys
        ]

        query_queue = RayWorkPool(
            pool,
            query_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "QuadSourceIntegral", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        query_queue.run()

        # for each (z_response, k, q, r) combination, we need to pull the associated Gk and QuadSource instances.
        # These are needed to assemble the quadratic source integral.

        # we try to do this in a vectorized way, based on the existing binning.
        # however, some of the GkSource and QuadSource instances will be duplicates.
        # There is no point repeatedly pulling multiple copies of them.
        # Instead we should pull just one copy, which we place in the object store.

        # Note we have to covert everything we will iterate over into a list, so that it has
        # a stable iteration order
        missing_labels = [
            (z_response, k, q, r)
            for k, query_outcomes in zip(k_keys, query_queue.results)
            for obj, (z_response, q, r) in zip(query_outcomes, binned_batch[k])
            if not obj.available
        ]

        # if no work to do, bail out now
        if len(missing_labels) == 0:
            return []

        missing_Gk = {
            k: list(
                set(
                    z_response
                    for obj, (z_response, q, r) in zip(query_outcomes, binned_batch[k])
                    if not obj.available
                )
            )
            for k, query_outcomes in zip(k_keys, query_queue.results)
        }
        missing_source_a = [
            (q, r)
            for k, query_outcomes in zip(k_keys, query_queue.results)
            for obj, (z_response, q, r) in zip(query_outcomes, binned_batch[k])
            if not obj.available
        ]
        missing_source_b = {}
        for q, r in missing_source_a:
            if q not in missing_source_b:
                missing_source_b[q] = set()
            missing_source_b[q].add(r)
        missing_source = {q: list(data) for q, data in missing_source_b.items()}
        q_keys = list(missing_source.keys())

        GkSource_lookup_batch = [
            {
                "shard_key": {"k": k},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_response": z_response,
                        "z_sample": None,  # need to specify, but not queried against; we pick up whatever z_source sample is stored
                        "atol": atol,
                        "rtol": rtol,
                        "tags": [
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    }
                    for z_response in missing_Gk[k]
                ],
            }
            for k in k_keys
        ]

        GkSource_lookup_queue = RayWorkPool(
            pool,
            GkSource_lookup_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            # post_handler=lambda objlist: [
            #     ray.put(obj) for obj in objlist
            # ],  # replace list with Ray ObjectRefs to object store version
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        GkSource_lookup_queue.run()
        GkSource_proxies = [
            [GkSourceProxy(obj) for obj in lookup_data]
            for k_exit, lookup_data in zip(k_keys, GkSource_lookup_queue.results)
        ]

        GkSourcePolicy_lookup_batch = [
            {
                "shard_key": {"k": k_exit},
                "payload": [
                    {
                        "source": source_proxy,
                        "policy": GkSource_policy_1pt5,
                        "k": k_exit,
                    }
                    for z_response, source_proxy in zip(missing_Gk[k_exit], proxy_data)
                ],
            }
            for k_exit, proxy_data in zip(k_keys, GkSource_proxies)
        ]

        GkSourcePolicy_lookup_queue = RayWorkPool(
            pool,
            GkSourcePolicy_lookup_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "GkSourcePolicyData", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        GkSourcePolicy_lookup_queue.run()

        QuadSource_lookup_batch = [
            {
                "shard_key": {"q": q},
                "payload": [
                    {
                        "model": model_proxy,
                        "z_sample": None,
                        "r": r,
                        "tags": [
                            TkProductionTag,
                            SourceZGridSizeTag,
                            OutsideHorizonEfoldsTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                        ],
                    }
                    for r in missing_source_b[q]
                ],
            }
            for q in q_keys
        ]

        QuadSource_lookup_queue = RayWorkPool(
            pool,
            QuadSource_lookup_batch,
            task_builder=lambda x: pool.object_get_vectorized(
                "QuadSource", x["shard_key"], payload_data=x["payload"]
            ),
            available_handler=None,
            compute_handler=None,
            store_handler=None,
            validation_handler=None,
            label_builder=None,
            title=None,
            store_results=True,
            create_batch_size=20,
            process_batch_size=20,
        )
        QuadSource_lookup_queue.run()

        # process these retrieved items into simple look-up tables
        Gk_cache = {
            (k.store_id, z_response.store_id): obj
            for k, query_outcomes in zip(k_keys, GkSourcePolicy_lookup_queue.results)
            for z_response, obj in zip(missing_Gk[k], query_outcomes)
        }
        source_cache = {
            (q.store_id, r.store_id): obj
            for q, query_outcomes in zip(q_keys, QuadSource_lookup_queue.results)
            for r, obj in zip(missing_source[q], query_outcomes)
        }

        work_refs = []

        for z_response, k, q, r in missing_labels:
            z_response: redshift
            k: wavenumber_exit_time
            q: wavenumber_exit_time
            r: wavenumber_exit_time

            # TODO: would be preferable to pack Gk and source into the object store and pass around an ObjectRef for them.
            #  Then we would not have to repeatedly seralize them when shipping out payloads for computation.
            #  However, the QuadSourceIntegral object has to inspect each Gk and source, so this isn't completely trivial.
            #  Maybe we need to retain a double cache.
            GkPolicy: GkSourcePolicyData = Gk_cache[(k.store_id, z_response.store_id)]
            source: QuadSource = source_cache[(q.store_id, r.store_id)]

            missing_data = False
            if not GkPolicy.available:
                missing_data = True
                print(
                    f"!! MISSING DATA WARNING ({datetime.now().replace(microsecond=0).isoformat()}) when building QuadSourceIntegral: GkSource for for k={k.k.k_inv_Mpc:.5g}/Mpc, z_response={z_response.z:.5g}"
                )
            if not source.available:
                missing_data = True
                print(
                    f"!! MISSING DATA WARNING ({datetime.now().replace(microsecond=0).isoformat()}) when building QuadSourceIntegral: QuadSource for for q={q.k.k_inv_Mpc:.5g}/Mpc, r={r.k.k_inv_Mpc:.5g}/Mpc"
                )

            if missing_data:
                raise RuntimeError(
                    f"QuadSourceIntegral builder: missing or incomplete source data for k={k.k.k_inv_Mpc:.5g}/Mpc, q={q.k.k_inv_Mpc:.5g}/Mpc, r={r.k.k_inv_Mpc:.5g}/Mpc"
                )

            work_refs.append(
                {
                    "ref": pool.object_get(
                        "QuadSourceIntegral",
                        model=model_proxy,
                        policy=GkSource_policy_1pt5,
                        k=k,
                        q=q,
                        r=r,
                        z_response=z_response,
                        z_source_max=z_source_integral_max_z,
                        atol=quad_atol,
                        rtol=quad_rtol,
                        tags=[
                            TkProductionTag,
                            GkProductionTag,
                            SourceZGridSizeTag,
                            ResponseZGridSizeTag,
                            LargestSourceZTag,
                            SourceSamplesPerLog10ZTag,
                            ResponseSparsenessZTag,
                        ],
                    ),
                    "compute_payload": {
                        "GkPolicy": GkPolicy,
                        "source": source,
                        "b": b_value,
                        "Bessel_0pt5": Bessel_0pt5_proxy,
                        "Bessel_2pt5": Bessel_2pt5_proxy,
                    },
                }
            )

        return work_refs

    def build_QuadSourceIntegral_label(qsi: QuadSourceIntegral):
        return f"{args.job_name}-QuadSourceIntegral-k{qsi.k.k_inv_Mpc:.3g}-q{qsi.q.k_inv_Mpc:.3g}-r{qsi.r.k_inv_Mpc:.3g}-zresponse{qsi.z_response.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"

    def compute_QuadSourceIntegral_work(
        qsi: QuadSourceIntegral, payload, label: Optional[str] = None
    ):
        return qsi.compute(payload=payload, label=label)

    if args.quad_source_integral_queue:
        # by placing the shard key last in the product, the idea is to balance out how many shards get used in each
        # batch of work. The intention is to balance the load on the datastore, to help maximize throughput.
        qsi_work_items = itertools.product(
            z_source_integral_response_sample,
            itertools.combinations_with_replacement(source_k_exit_times, 2),
            response_k_exit_times,
        )
        qsi_work_items = [(z, k, q, r) for z, (q, r), k in qsi_work_items]

        # chunk size may need tuning for the cluster in use.
        # GkSource objects with about 2k z-sample points occupy about 1 Mb.
        # If we end up submitting N jobs based on a work batch, this will require about (1 + 1) N Mb ~ 2 N Mb in storage,
        # assuming the QuadSource object needed is also about 1 Mb.
        # So we with 250 items we could lock up about 500 Mb or probably just a bit more to account for overheads.
        # On a small machine, this might need to be reduced.
        # On a larger machine, there could be efficiencies in making the chunk size bigger.
        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))

        QuadSourceIntegral_queue = RayWorkPool(
            pool,
            qsi_work_batches,
            task_builder=build_QuadSourceIntegral_batch,
            validation_handler=None,
            label_builder=build_QuadSourceIntegral_label,
            title="CALCULATE QUADRATIC SOURCE INTEGRALS",
            store_results=False,
            create_batch_size=1,  # we have batched the work queue into chunks ourselves, so don't process too many of these chunks at once!
            notify_batch_size=2000,
            notify_min_time_interval=5 * 60,
            max_task_queue=1000,  # don't allow too many tasks to build up, because the payload storage requirenments could max out the object store
            process_batch_size=100,
        )
        QuadSourceIntegral_queue.run()
