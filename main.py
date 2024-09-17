import argparse
import sys
from datetime import datetime
from math import log10

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
from Datastore.SQL import Datastore
from MetadataConcepts import tolerance, store_tag
from RayWorkQueue import RayWorkQueue
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

default_label = "SecondaryGWKit-test"
default_timeout = 60

parser = argparse.ArgumentParser()
parser.add_argument(
    "--create-database",
    default=None,
    help="create a database cache in the specified file",
)
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

if args.create_database is None and args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

# instantiate a Datastore actor: this runs on its own node, and acts as a broker between
# ourselves and the dataabase.
# For performance reasons, we want all database activity to run on this node.
# For one thing, this lets us use transactions efficiently.
store: Datastore = Datastore.remote("2024.1.1")

if args.create_database is not None:
    ray.get(
        store.create_datastore.remote(args.create_database, timeout=args.db_timeout)
    )
else:
    ray.get(store.open_datastore.remote(args.database, timeout=args.db_timeout))

# set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
LambdaCDM_Planck2018 = ray.get(
    store.object_get.remote(LambdaCDM, params=params, units=units)
)


def convert_to_wavenumbers(k_sample_set):
    return store.object_get.remote(
        wavenumber,
        payload_data=[{"k_inv_Mpc": k, "units": units} for k in k_sample_set],
    )


def convert_to_redshifts(z_sample_set):
    return [redshift(z=z) for z in z_sample_set]


## STEP 1
## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES

# build array of k-sample points
print("-- building array of k-sample wavenumbers")
k_array = ray.get(
    convert_to_wavenumbers(np.logspace(np.log10(0.001), np.log10(1.0), 500))
)
k_sample = wavenumber_array(k_array=k_array)


# build absolute and relative tolerances
print("-- building tolerance objects")
atol, rtol = ray.get(
    [
        store.object_get.remote(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        store.object_get.remote(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)

# build tags and other labels
print("-- building database tags")
Tk_production_tag, Gk_production_tag = ray.get(
    [
        store.object_get.remote(store_tag, label="TkOneLoopDensity"),
        store.object_get.remote(store_tag, label="GkOneLoopDensity"),
    ]
)


def create_k_exit_work(k: wavenumber):
    return store.object_get.remote(
        wavenumber_exit_time,
        k=k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
    )


# for each k mode we sample, determine its horizon exit point
k_exit_queue = RayWorkQueue(
    store,
    k_sample,
    create_k_exit_work,
    title="CALCULATE HORIZON EXIT TIMES",
    store_results=True,
)
k_exit_queue.run()
k_exit_times = k_exit_queue.results


## STEP 2
## COMPUTE MATTER TRANSFER FUNCTIONS


def create_Tk_work(k_exit: wavenumber_exit_time):
    z_sample = k_exit.populate_z_sample(outside_horizon_efolds=10.0, z_end=0.5)
    return store.object_get.remote(
        MatterTransferFunctionIntegration,
        k=k_exit.k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
        z_sample=z_sample,
        z_init=z_sample.max,
        tags=[Tk_production_tag],
    )


def create_Tk_work_label(Tk: MatterTransferFunctionIntegration):
    return f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"


Tk_queue = RayWorkQueue(
    store,
    k_exit_times,
    create_Tk_work,
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
    source_zs = k_exit.populate_z_sample(
        outside_horizon_efolds=10.0, z_end=G_LATEST_RESPONSE_Z
    )

    work_refs = []

    for source_z in source_zs:
        num_sample = int(
            round(60 * (log10(source_z.z) - log10(G_LATEST_RESPONSE_Z)) + 0.5, 0)
        )
        if num_sample >= 3:
            response_zs = np.logspace(
                log10(source_z.z), log10(G_LATEST_RESPONSE_Z), num_sample
            )
            response_array = redshift_array(z_array=convert_to_redshifts(response_zs))

            work_refs.append(
                store.object_get.remote(
                    TensorGreenFunctionIntegration,
                    k=k_exit.k,
                    cosmology=LambdaCDM_Planck2018,
                    atol=atol,
                    rtol=rtol,
                    z_source=source_z,
                    z_sample=response_array,
                    tags=[Gk_production_tag],
                )
            )

    return work_refs


def create_Gk_exit_work_label(Gk: TensorGreenFunctionIntegration):
    return f"{args.job_name}-Gk-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"


Gk_queue = RayWorkQueue(
    store,
    k_exit_times,
    create_Gk_work,
    label_maker=create_Gk_exit_work_label,
    title="CALCULATE TENSOR GREEN FUNCTIONS",
    notify_batch_size=1000,
    notify_time_interval=5 * 60,
    notify_min_time_interval=30,
    store_results=False,
)
Gk_queue.run()
