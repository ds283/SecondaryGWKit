import argparse
import sys
from datetime import datetime

import numpy as np
import ray
from ray import ObjectRef

from ComputeTargets import (
    MatterTransferFunctionContainer,
    MatterTransferFunctionIntegration,
)
from CosmologyConcepts import (
    tolerance,
    wavenumber,
    redshift,
    redshift_array,
    wavenumber_array,
    wavenumber_exit_time,
)
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Datastore.SQL import Datastore
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

# set up LambdaCDM object representing basic Planck2018 cosmology in Mpc units
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
    return store.object_get.remote(
        redshift,
        payload_data=[{"z": z} for z in z_sample_set],
    )


## STEP 1
## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES


# build array of k-sample points
k_array = ray.get(
    convert_to_wavenumbers(np.logspace(np.log10(0.001), np.log10(0.5), 500))
)
k_sample = wavenumber_array(k_array=k_array)


# build absolute and relative tolerances
atol, rtol = ray.get(
    [
        store.object_get.remote(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        store.object_get.remote(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)

# for each k mode we sample, determine its horizon exit point
# (in principle this is a quick operation so we do not mind blocking on ray.get() while
# the exit time objects are constructed; if any calculations are outstanding, we just get back
# empty objects)
k_exit_times = ray.get(
    [
        store.object_get.remote(
            wavenumber_exit_time,
            k=k,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        )
        for k in k_array
    ]
)

exit_time_work_refs = []
exit_time_work_lookup = {}

exit_time_store_refs = []
exit_time_store_lookup = {}

# determine whether any exit time objects require computation;
# if so, initiate the corresponding compute tasks and store the ObjectRefs in a queue
for i, obj in enumerate(k_exit_times):
    if obj.available:
        continue

    ref: ObjectRef = obj.compute()
    exit_time_work_refs.append(ref)
    exit_time_work_lookup[ref.hex] = (i, obj)

# while compute tasks are still being processed, initiate store tasks
# to make efficient use of the Datastore actor
# Each store task produces another ObjectRef that needs to be held in a second queue
while len(exit_time_work_refs) > 0:
    done_ref, exit_time_work_refs = ray.wait(exit_time_work_refs, num_returns=1)
    if len(done_ref) > 0:
        i, obj = exit_time_work_lookup[done_ref[0].hex]
        obj.store()

        ref: ObjectRef = store.object_store.remote(obj)
        exit_time_store_refs.append(ref)
        exit_time_store_lookup[ref.hex] = i

# while store tasks are still being processed, collect the populated stored
# objects (which are now complete with a store_id field) and replace them in
# the k_exit_times array
while len(exit_time_store_refs) > 0:
    done_ref, exit_time_store_refs = ray.wait(exit_time_store_refs, num_returns=1)
    if len(done_ref) > 0:
        i = exit_time_store_lookup[done_ref[0].hex]
        k_exit_times[i] = ray.get(done_ref[0])


## STEP 2
## FOR EACH K-MODE, OBTAIN A MESH OF Z-SAMPLE POINTS AT WHICH TO COMPUTE ITS TRANSFER FUNCTION


# build a set of z-sample points for each k-mode
# These run from 10 e-folds before horizon exit, up to the final redshift z_end
z_sample_values = ray.get(
    [
        convert_to_redshifts(
            k_exit.populate_z_sample(outside_horizon_efolds=10.0, z_end=0.1)
        )
        for k_exit in k_exit_times
    ]
)

# finally, wrap all of these lists into redshift_array containers
z_sample = [redshift_array(z_array=zs) for zs in z_sample_values]


## STEP 3
## USING THESE Z-SAMPLE MESHES, COMPUTE THE TRANSFER FUNCTION FOR EACH K-MODE


def build_Tks():
    return ray.get(
        [
            store.object_get.remote(
                MatterTransferFunctionContainer,
                k=k_sample[i],
                cosmology=LambdaCDM_Planck2018,
                atol=atol,
                rtol=rtol,
                z_sample=z_sample[i],
                z_init=z_sample[i].max,
            )
            for i in range(len(k_sample))
        ]
    )


Tks = build_Tks()
cycle = 1


while any(not Tk.available for Tk in Tks):
    # schedule integrations to populate any z-sample points that are not already held in the container

    label = f"{args.job_name}-cycle={cycle}-{datetime.now().replace(microsecond=0).isoformat()}"

    Tk_work_refs = []
    Tk_work_lookup = {}

    Tk_store_refs = []
    Tk_store_lookup = []

    for i, Tk in enumerate(Tks):
        if not Tk.available:
            obj = ray.get(
                store.object_get.remote(
                    MatterTransferFunctionIntegration,
                    k=Tk.k,
                    cosmology=LambdaCDM_Planck2018,
                    atol=atol,
                    rtol=rtol,
                    z_sample=Tk.missing_z_sample,
                    z_init=Tk.z_init,
                    label=label,
                )
            )
            ref: ObjectRef = obj.compute()

            Tk_work_refs.append(ref)
            Tk_work_lookup[ref.hex] = obj

    while len(Tk_work_refs) > 0:
        done_ref, Tk_work_refs = ray.wait(Tk_work_refs, num_returns=1)
        if len(done_ref) > 0:
            obj = Tk_work_lookup[done_ref[0].hex]
            obj.store()

            ref: ObjectRef = store.object_store.remote(obj)
            Tk_store_refs.append(ref)

    while len(Tk_store_refs) > 0:
        done_ref, Tk_store_refs = ray.wait(Tk_store_refs, num_returns=1)

    Tks = build_Tks()
    cycle += 1
