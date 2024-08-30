import argparse
import sys

import ray

from Datastore import Datastore
from Units import Mpc_units
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018

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
ray.get(store.register_storable_class.remote(LambdaCDM))

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
