import argparse
import sys
from datetime import datetime
from math import log10

import numpy as np
import ray
from ray import ObjectRef

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
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import WallclockTimer

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


# for each k mode we sample, determine its horizon exit point
# (in principle this is a quick operation so we do not mind blocking on ray.get() while
# the exit time objects are constructed; if any calculations are outstanding, we just get back
# empty objects)
print("-- populating array of k-exit objects")
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
print("-- queueing tasks to compute horizon exit times")
for i, obj in enumerate(k_exit_times):
    if obj.available:
        continue

    ref: ObjectRef = obj.compute()
    exit_time_work_refs.append(ref)
    exit_time_work_lookup[ref.hex] = (i, obj)

# while compute tasks are still being processed, initiate store tasks
# to make efficient use of the Datastore actor
# Each store task produces another ObjectRef that needs to be held in a second queue
total_done = 0
batch_done = 0
queue_size = len(exit_time_work_refs)
print(f"--   waiting for {queue_size} horizon exit tasks to complete")
while len(exit_time_work_refs) > 0:
    done_ref, exit_time_work_refs = ray.wait(exit_time_work_refs, num_returns=1)
    if len(done_ref) > 0:
        i, obj = exit_time_work_lookup[done_ref[0].hex]
        obj.store()

        ref: ObjectRef = store.object_store.remote(obj)
        exit_time_store_refs.append(ref)
        exit_time_store_lookup[ref.hex] = i

        batch_size = len(done_ref)
        total_done += batch_size
        batch_done += batch_size
        if batch_done >= 20:
            print(
                f"--   {total_done}/{queue_size} = {100.0 * float(total_done) / float(queue_size):.2f}% horizon exit computations complete, {len(exit_time_work_refs)} still queued"
            )
            batch_done = 0

# while store tasks are still being processed, collect the populated stored
# objects (which are now complete with a store_id field) and replace them in
# the k_exit_times array
total_done = 0
batch_done = 0
queue_size = len(exit_time_store_refs)
print(f"--   waiting for {queue_size} horizon exit storage requests to complete")
while len(exit_time_store_refs) > 0:
    done_ref, exit_time_store_refs = ray.wait(exit_time_store_refs, num_returns=1)
    if len(done_ref) > 0:
        i = exit_time_store_lookup[done_ref[0].hex]
        k_exit_times[i] = ray.get(done_ref[0])

        batch_size = len(done_ref)
        total_done += batch_size
        batch_done += batch_size
        if batch_done >= 20:
            print(
                f"--    {total_done}/{queue_size} = {100.0*float(total_done)/float(queue_size):.2f}% stores complete, {len(exit_time_store_refs)} still queued"
            )
            batch_done = 0


## STEP 2
## FOR EACH K-MODE, OBTAIN A MESH OF Z-SAMPLE POINTS AT WHICH TO COMPUTE ITS TRANSFER FUNCTION


# build a set of z-sample points for each k-mode
# These run from 10 e-folds before horizon exit, up to the final redshift z_end
print("-- populating array of z-sample times for each k-mode")
with WallclockTimer() as timer:
    Tk_z_sample = [
        k_exit.populate_z_sample(outside_horizon_efolds=10.0, z_end=10.0)
        for k_exit in k_exit_times
    ]
print(f"--   redshift sample arrays constructed in time {timer.elapsed:.3g}sec")


## STEP 3
## USING THESE Z-SAMPLE MESHES, COMPUTE THE TRANSFER FUNCTION FOR EACH K-MODE

print("-- querying matter transfer function values")
Tks = ray.get(
    [
        store.object_get.remote(
            MatterTransferFunctionIntegration,
            k=k_sample[i],
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
            z_sample=Tk_z_sample[i],
            z_init=Tk_z_sample[i].max,
            tags=[Tk_production_tag],
        )
        for i in range(len(k_sample))
    ]
)


Tk_work_refs = []
Tk_work_lookup = {}

Tk_store_refs = []

print(f"-- scheduling work to populate missing matter transfer function values")
for Tk in Tks:
    if Tk.available:
        continue

    ref: ObjectRef = Tk.compute(
        label=f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"
    )

    Tk_work_refs.append(ref)
    Tk_work_lookup[ref.hex] = Tk

total_done = 0
batch_done = 0
queue_size = len(Tk_work_refs)
print(
    f"--   waiting for {queue_size} matter transfer function integrations to complete"
)
while len(Tk_work_refs) > 0:
    done_ref, Tk_work_refs = ray.wait(Tk_work_refs, num_returns=1)
    if len(done_ref) > 0:
        Tk = Tk_work_lookup[done_ref[0].hex]
        Tk.store()

        ref: ObjectRef = store.object_store.remote(Tk)
        Tk_store_refs.append(ref)

        batch_size = len(done_ref)
        total_done += batch_size
        batch_done += batch_size
        if batch_done >= 20:
            print(
                f"--   {total_done}/{queue_size} = {100.0*float(total_done)/float(queue_size):.2f}% integrations complete, {len(Tk_work_refs)} still queued"
            )
            batch_done = 0

total_done = 0
queue_size = len(Tk_store_refs)
print(
    f"--   waiting for {queue_size} matter transfer function storage requests to complete"
)
while len(Tk_store_refs) > 0:
    num_remaining = len(Tk_store_refs)
    done_ref, Tk_store_refs = ray.wait(
        Tk_store_refs, num_returns=min(20, num_remaining)
    )

    total_done += len(done_ref)
    print(
        f"--    {total_done}/{queue_size} = {100.0*float(total_done)/float(queue_size):.2f}% stores complete, {len(Tk_store_refs)} still queued"
    )


## STEP 4
## BUILD A GRID OF SOURCE/RESPONSE REDSHIFT SAMPLE REDSHIFTS

print("-- QUERYING TENSOR GREEN'S FUNCTION VALUES")

print(
    "-- building grid of z_source/z_response sample times for Green's function calculation"
)

G_LATEST_RESPONSE_Z = 0.5

todo = len(k_exit_times)

for counter, k_exit in enumerate(k_exit_times):
    k_exit: wavenumber_exit_time

    k = k_exit.k

    Gk_ref_list = []
    z_response_arrays = []

    with WallclockTimer() as timer:
        source_zs = k_exit.populate_z_sample(
            outside_horizon_efolds=10.0, z_end=G_LATEST_RESPONSE_Z
        )

        print(
            f"** constructing source/response redshift arrays for k = {k.k_inv_Mpc:.3g}/Mpc ({counter}/{todo} = {100.0*float(counter)/float(todo):.2f}%)"
        )

        for z_source in source_zs:
            num_response_sample = int(
                round(60 * (log10(z_source) - log10(G_LATEST_RESPONSE_Z)) + 0.5, 0)
            )
            if num_response_sample > 1:
                response_redshifts = np.logspace(
                    log10(z_source), log10(G_LATEST_RESPONSE_Z), num_response_sample
                )
                z_response_array = redshift_array(
                    z_array=convert_to_redshifts(response_redshifts)
                )
                z_response_arrays.append(z_response_array)

                # query for an integration with this z sample list
                Gk_ref_list.append(
                    store.object_get.remote(
                        TensorGreenFunctionIntegration,
                        k=k,
                        cosmology=LambdaCDM_Planck2018,
                        atol=atol,
                        rtol=rtol,
                        z_source=z_source,
                        z_sample=z_response_array,
                        tags=[Gk_production_tag],
                    )
                )

    print(
        f"**    {len(z_response_arrays)} source/response redshift arrays for k = {k.k_inv_Mpc:.3g}/Mpc constructed in time {timer.elapsed:.3g} sec"
    )

    Gks = ray.get(Gk_ref_list)

    Gk_work_refs = []
    Gk_work_lookup = {}

    Gk_store_refs = []

    print(
        f"-- scheduling work to populate missing tensor Green's function values for k = {k.k_inv_Mpc:.3g}/Mpc"
    )
    for Gk in Gks:
        if Gk.available:
            continue

        ref: ObjectRef = Gk.compute(
            label=f"{args.job_name}-Gk-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"
        )

        Gk_work_refs.append(ref)
        Gk_work_lookup[ref.hex] = Gk

    total_done = 0
    batch_done = 0
    queue_size = len(Gk_work_refs)
    print(
        f"--   waiting for {queue_size} tensor Green's function integrations to complete for k = {k.k_inv_Mpc:.3g}/Mpc"
    )
    while len(Gk_work_refs) > 0:
        done_ref, Gk_work_refs = ray.wait(Gk_work_refs, num_returns=1)
        if len(done_ref) > 0:
            Gk = Gk_work_lookup[done_ref[0].hex]
            Gk.store()

            ref: ObjectRef = store.object_store.remote(Gk)
            Gk_store_refs.append(ref)

            batch_size = len(done_ref)
            total_done += batch_size
            batch_done += batch_size
            if batch_done >= 20:
                # print(
                #     f"--   {total_done}/{queue_size} = {100.0 * float(total_done) / float(queue_size):.2f}% integrations complete, {len(Gk_work_refs)} still queued"
                # )
                batch_done = 0

    total_done = 0
    queue_size = len(Gk_store_refs)
    print(
        f"--   waiting for {queue_size} tensor Green's function storage requests to complete"
    )
    while len(Gk_store_refs) > 0:
        num_remaining = len(Gk_store_refs)
        done_ref, Gk_store_refs = ray.wait(
            Gk_store_refs, num_returns=min(20, num_remaining)
        )

        total_done += len(done_ref)
        # print(
        #     f"--    {total_done}/{queue_size} = {100.0 * float(total_done) / float(queue_size):.2f}% stores complete, {len(Gk_store_refs)} still queued"
        # )
