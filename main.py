import argparse
import sys

import ray

import numpy as np

from CosmologyConcepts.wavenumber import wavenumber_exit_time
from Datastore import Datastore
from Units import Mpc_units
from CosmologyConcepts import wavenumber, redshift, redshift_array, wavenumber_array
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from ComputeTargets import MatterTransferFunction

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

store: Datastore = Datastore.remote("2024.1.1")

# register storable classes
ray.get(
    store.register_storable_classes.remote(
        {redshift, wavenumber, wavenumber_exit_time, LambdaCDM}
    )
)

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
