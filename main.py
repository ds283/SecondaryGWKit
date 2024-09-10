import argparse
import sys

import ray

import numpy as np

from Datastore.SQL import Datastore
from Units import Mpc_units
from CosmologyConcepts import (
    tolerance,
    wavenumber,
    redshift,
    redshift_array,
    wavenumber_array,
    wavenumber_exit_time,
)
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from ComputeTargets import (
    MatterTransferFunction,
    IntegrationSolver,
    MatterTransferFunctionIntegration,
    MatterTransferFunctionValue,
    MatterTransferFunctionContainer,
)

from pyinstrument import Profiler

from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

parser = argparse.ArgumentParser()
parser.add_argument("--Trad-final", nargs=2, help="specify final radiation temperature")
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
    ray.get(store.create_datastore.remote(args.create_database))
else:
    ray.get(store.open_datastore.remote(args.database))

# set up LambdaCDM object representing basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
LambdaCDM_Planck2018 = ray.get(
    store.object_factory.remote(LambdaCDM, params=params, units=units)
)


def convert_numbers_to_wavenumbers(k_sample_set):
    return store.object_factory.remote(
        wavenumber,
        payload_data=[{"k_inv_Mpc": k, "units": units} for k in k_sample_set],
    )


def convert_numbers_to_redshifts(z_sample_set):
    return store.object_factory.remote(
        redshift,
        payload_data=[{"z": z} for z in z_sample_set],
    )


k_array = ray.get(
    convert_numbers_to_wavenumbers(np.logspace(np.log10(0.001), np.log10(0.5), 5000))
)
k_sample = wavenumber_array(k_array=k_array)

# build some sample transfer function histories
k1: wavenumber = k_sample[0]
k2: wavenumber = k_sample[1000]
k3: wavenumber = k_sample[4000]

atol, rtol = ray.get(
    [
        store.object_factory.remote(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        store.object_factory.remote(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)

k1_exit, k2_exit, k3_exit = ray.get(
    [
        store.object_factory.remote(
            wavenumber_exit_time,
            k=k1,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        ),
        store.object_factory.remote(
            wavenumber_exit_time,
            k=k2,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        ),
        store.object_factory.remote(
            wavenumber_exit_time,
            k=k3,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
        ),
    ]
)

k1_z_array, k2_z_array, k3_z_array = ray.get(
    [
        convert_numbers_to_redshifts(
            k1_exit.populate_z_samples(outside_horizon_efolds=10.0, z_end=0.1)
        ),
        convert_numbers_to_redshifts(
            k2_exit.populate_z_samples(outside_horizon_efolds=10.0, z_end=0.1)
        ),
        convert_numbers_to_redshifts(
            k3_exit.populate_z_samples(outside_horizon_efolds=10.0, z_end=0.1)
        ),
    ]
)

k1_z_sample = redshift_array(z_array=k1_z_array)
k2_z_sample = redshift_array(z_array=k2_z_array)
k3_z_sample = redshift_array(z_array=k3_z_array)

T_k1_container, T_k2_container, T_k3_container = ray.get(
    [
        store.object_factory.remote(
            MatterTransferFunctionContainer,
            k=k1,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
            z_sample=k1_z_sample,
            z_init=max(k1_z_sample.as_list()),
        ),
        store.object_factory.remote(
            MatterTransferFunctionContainer,
            k=k2,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
            z_sample=k2_z_sample,
            z_init=max(k2_z_sample.as_list()),
        ),
        store.object_factory.remote(
            MatterTransferFunctionContainer,
            k=k3,
            cosmology=LambdaCDM_Planck2018,
            atol=atol,
            rtol=rtol,
            z_sample=k3_z_sample,
            z_init=max(k3_z_sample.as_list()),
        ),
    ]
)

# Tk1_future = store.object_factory(
#     MatterTransferFunction, cosmology=LambdaCDM_Planck2018, k=k_samples[0]
# )
# Tk2_future = store.object_factory(
#     MatterTransferFunction, cosmology=LambdaCDM_Planck2018, k=k_samples[1000]
# )
# Tk3_future = store.object_factory(
#     MatterTransferFunction, cosmology=LambdaCDM_Planck2018, k=k_samples[4000]
# )
#
# ray.get([Tk1_future, Tk2_future, Tk3_future])
