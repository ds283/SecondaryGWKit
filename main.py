import argparse
import sys
from datetime import datetime
from math import exp

import numpy as np
import ray

from ComputeTargets import (
    MatterTransferFunctionIntegration,
    TensorGreenFunctionIntegration,
)
from CosmologyConcepts import (
    wavenumber,
    redshift,
    wavenumber_array,
    wavenumber_exit_time,
    redshift_array,
)
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Datastore.SQL.sqla_impl import ShardedPool
from MetadataConcepts import tolerance, store_tag
from RayWorkQueue import RayWorkQueue
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

default_label = "SecondaryGWKit-test"
default_timeout = 60

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    default=None,
    help="read/write work items using the specified database cache",
)
parser.add_argument(
    "--compute",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="enable/disable computation of work items (use in conjunction with --create-database",
)
parser.add_argument(
    "--job-name",
    default=default_label,
    help="specify a label for this job (used to identify integrations and other numerical products)",
)
parser.add_argument(
    "--db-actors",
    default=5,
    help="specify number of Ray actors used to provide database services",
)
parser.add_argument(
    "--db-timeout",
    default=default_timeout,
    help="specify connection timeout for database layer",
)
parser.add_argument(
    "--ray-address", default="auto", type=str, help="specify address of Ray cluster"
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
pool: ShardedPool = ShardedPool(
    version_label="2024.1.1", db_name=args.database, timeout=args.db_timeout, shards=20
)

# set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
LambdaCDM_Planck2018 = ray.get(pool.object_get(LambdaCDM, params=params, units=units))


## UTILITY FUNCTIONS


def convert_to_wavenumbers(k_sample_set):
    return pool.object_get(
        wavenumber,
        payload_data=[{"k_inv_Mpc": k, "units": units} for k in k_sample_set],
    )


def convert_to_redshifts(z_sample_set):
    return pool.object_get(redshift, payload_data=[{"z": z} for z in z_sample_set])


## DATASTORE OBJECTS


# build absolute and relative tolerances
atol, rtol = ray.get(
    [
        pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)


## STEP 1
## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES

# build array of k-sample points
print("\n** BUILDING ARRAY OF WAVENUMBERS AT WHICH TO SAMPLE")
k_array = ray.get(
    convert_to_wavenumbers(np.logspace(np.log10(0.001), np.log10(1.0), 500))
)
k_sample = wavenumber_array(k_array=k_array)


def create_k_exit_work(k: wavenumber):
    return pool.object_get(
        wavenumber_exit_time,
        k=k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
    )


# for each k mode we sample, determine its horizon exit point
k_exit_queue = RayWorkQueue(
    pool,
    k_sample,
    create_k_exit_work,
    title="CALCULATE HORIZON EXIT TIMES",
    store_results=True,
)
k_exit_queue.run()
k_exit_times = k_exit_queue.results


## STEP 2
## BUILD A UNIVERSAL GRID OF Z-VALUES AT WHICH TO SAMPLE
print("\n** BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE")
k_exit_earliest: wavenumber_exit_time = k_exit_times[-1]

# build a log-spaced universal grid of wavenumbers
LATEST_Z = 0.1
OUTSIDE_HORIZON_EFOLDS = 10.0
SAMPLES_PER_LOG10_Z = 150

universal_z_grid = k_exit_earliest.populate_z_sample(
    outside_horizon_efolds=OUTSIDE_HORIZON_EFOLDS,
    samples_per_log10z=SAMPLES_PER_LOG10_Z,
    z_end=LATEST_Z,
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
        pool.object_get(store_tag, label="TkOneLoopDensity"),
        pool.object_get(store_tag, label="GkOneLoopDensity"),
        pool.object_get(store_tag, label=f"GlobalRedshiftGrid_{len(z_sample)}"),
        pool.object_get(
            store_tag, label=f"OutsideHorizonEfolds_{OUTSIDE_HORIZON_EFOLDS:.2f}"
        ),
        pool.object_get(store_tag, label=f"LargestRedshift_{z_sample.max.z:.5g}"),
        pool.object_get(store_tag, label=f"SamplesPerLog10Z_{SAMPLES_PER_LOG10_Z}"),
    ]
)


## STEP 2
## COMPUTE MATTER TRANSFER FUNCTIONS


def create_Tk_work(k_exit: wavenumber_exit_time):
    my_sample = z_sample.truncate(exp(OUTSIDE_HORIZON_EFOLDS) * k_exit.z_exit)
    return pool.object_get(
        MatterTransferFunctionIntegration,
        k=k_exit.k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
        z_sample=my_sample,
        z_init=my_sample.max,
        tags=[
            TkProductionTag,
            GlobalZGridTag,
            OutsideHorizonEfoldsTag,
            LargestZTag,
            SamplesPerLog10ZTag,
        ],
    )


def create_Tk_work_label(Tk: MatterTransferFunctionIntegration):
    return f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"


def validate_Tk_work(Tk: MatterTransferFunctionIntegration):
    return pool.object_validate(Tk)


Tk_queue = RayWorkQueue(
    pool,
    k_exit_times,
    create_Tk_work,
    validation_maker=validate_Tk_work,
    label_maker=create_Tk_work_label,
    title="CALCULATE MATTER TRANSFER FUNCTIONS",
    store_results=True,
)
Tk_queue.run()
Tks = Tk_queue.results


## STEP 3
## COMPUTE TENOR GREEN'S FUNCTIONS ON A GRID OF SOURCE/RESPONSE REDSHIFT SAMPLES

G_LATEST_RESPONSE_Z = 0.5


def create_Gk_work(k_exit: wavenumber_exit_time):
    source_zs = z_sample.truncate(exp(OUTSIDE_HORIZON_EFOLDS) * k_exit.z_exit)

    work_refs = []

    for source_z in source_zs:
        response_zs = z_sample.truncate(source_z)
        if len(response_zs) >= 3:

            work_refs.append(
                pool.object_get(
                    TensorGreenFunctionIntegration,
                    k=k_exit.k,
                    cosmology=LambdaCDM_Planck2018,
                    atol=atol,
                    rtol=rtol,
                    z_source=source_z,
                    z_sample=response_zs,
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


def create_Gk_exit_work_label(Gk: TensorGreenFunctionIntegration):
    return f"{args.job_name}-Gk-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"


def validate_Gk_work(Gk: TensorGreenFunctionIntegration):
    return pool.object_validate(Gk)


Gk_queue = RayWorkQueue(
    pool,
    k_exit_times,
    create_Gk_work,
    validation_maker=validate_Gk_work,
    label_maker=create_Gk_exit_work_label,
    title="CALCULATE TENSOR GREEN FUNCTIONS",
    notify_batch_size=1000,
    notify_time_interval=5 * 60,
    notify_min_time_interval=30,
    store_results=False,
)
Gk_queue.run()
