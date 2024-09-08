import argparse
import sys

import ray

import numpy as np

from Datastore import Datastore
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
)

from pyinstrument import Profiler

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
    obj = store.create_datastore.remote(args.create_database)
    output = ray.get(obj)
else:
    obj = store.open_datastore.remote(args.database)
    output = ray.get(obj)

# set up LambdaCDM object representing basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
cosmology = LambdaCDM(store, params, units)

k_sample_set = np.logspace(np.log10(0.001), np.log10(0.5), 5000)
k_samples = wavenumber_array(store, k_sample_set, units)

MatterTransferFunction.populate_z_samples(store, cosmology, k_samples[0])
MatterTransferFunction.populate_z_samples(store, cosmology, k_samples[1000])
MatterTransferFunction.populate_z_samples(store, cosmology, k_samples[4000])
