import argparse
import sys

import pyarrow as pa
import ray
from pyarrow.csv import CSVWriter

from ComputeTargets import (
    MatterTransferFunctionIntegration,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
)
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
from Datastore.SQL.sqla_impl import ShardedPool
from MetadataConcepts import tolerance
from RayWorkQueue import RayWorkQueue
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_TIMEOUT = 60
OUTSIDE_HORIZON_EFOLDS = 3.5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    default=None,
    help="read/write work items using the specified database cache",
)
parser.add_argument(
    "--db-timeout",
    default=DEFAULT_TIMEOUT,
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
    version_label="2024.1.1",
    db_name=args.database,
    timeout=args.db_timeout,
)

# set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
LambdaCDM_Planck2018 = ray.get(pool.object_get(LambdaCDM, params=params, units=units))

# build absolute and relative tolerances
atol, rtol = ray.get(
    [
        pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)

k_array = ray.get(pool.read_wavenumber_table(units=units))


def create_k_exit_work(k: wavenumber):
    return pool.object_get(
        wavenumber_exit_time,
        k=k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
    )


# query wavenumber_exit_time objects corresponding to these k modes
k_exit_queue = RayWorkQueue(
    pool,
    k_array,
    task_maker=create_k_exit_work,
    compute_maker=None,
    store_maker=None,
    store_results=True,
)
k_exit_queue.run()
k_exit_times = k_exit_queue.results

time_series_schema = pa.schema(
    [
        ("k_serial", pa.int32()),
        ("k_exit_serial", pa.int32()),
        ("k_inv_Mpc", pa.float64()),
        ("z_init_serial", pa.int32()),
        ("z_init", pa.float64()),
        ("z_exit", pa.float64()),
        ("z_serial", pa.int32()),
        ("z", pa.float64()),
        ("value", pa.float64()),
    ]
)
metadata_schema = pa.schema(
    [
        ("integration_serial", pa.int32()),
        ("k_serial", pa.int32()),
        ("k_exit_serial", pa.int32()),
        ("k_inv_Mpc", pa.float64()),
        ("z_init_serial", pa.int32()),
        ("z_init", pa.float64()),
        ("z_exit", pa.float64()),
        ("compute_time", pa.float64()),
        ("compute_steps", pa.int32()),
        ("RHS_evaluations", pa.int32()),
        ("mean_RHS_time", pa.float64()),
        ("min_RHS_time", pa.float64()),
        ("max_RHS_time", pa.float64()),
    ]
)

# read in integrations, one by one, and write CSV files for the time series data and the integration metadata
metadata_rows = []

with CSVWriter("Tk_time_series.csv", schema=time_series_schema) as writer:
    for k_exit in k_exit_times:
        if k_exit.available:
            obj: MatterTransferFunctionIntegration = ray.get(
                pool.object_get(
                    MatterTransferFunctionIntegration,
                    solver_labels=[],
                    cosmology=LambdaCDM_Planck2018,
                    k=k_exit,
                    z_sample=None,
                    z_init=None,  # will query for any sample time
                    atol=atol,
                    rtol=rtol,
                )
            )

            if obj.available:
                # print(
                #     f'>> recovered matter transfer function integration "{obj.label}" for k={k_exit.k.k_inv_Mpc:.5g}/Mpc and z_init={obj.z_init.z:.5g} with {len(obj.z_sample)} z-samples'
                # )
                metadata_rows.append(
                    {
                        "integration_serial": obj.store_id,
                        "k_serial": obj.k.store_id,
                        "k_exit_serial": obj._k_exit.store_id,
                        "k_inv_Mpc": obj.k.k_inv_Mpc,
                        "z_init_serial": obj.z_init.store_id,
                        "z_init": obj.z_init.z,
                        "z_exit": obj.z_exit,
                        "compute_time": obj.compute_time,
                        "compute_steps": obj.compute_steps,
                        "RHS_evaluations": obj.RHS_evaluations,
                        "mean_RHS_time": obj.mean_RHS_time,
                        "min_RHS_time": obj.min_RHS_time,
                        "max_RHS_time": obj.max_RHS_time,
                    }
                )

                time_series_rows = [
                    {
                        "k_serial": obj.k.store_id,
                        "k_exit_serial": obj._k_exit.store_id,
                        "k_inv_Mpc": obj.k.k_inv_Mpc,
                        "z_init_serial": obj.z_init.store_id,
                        "z_init": obj.z_init.z,
                        "z_exit": obj.z_exit,
                        "z_serial": value.z.store_id,
                        "z": value.z.z,
                        "value": value.value,
                    }
                    for value in obj.values
                ]
                batch = pa.RecordBatch.from_pylist(
                    time_series_rows, schema=time_series_schema
                )
                writer.write(batch)

with CSVWriter("Tk_metadata.csv", schema=metadata_schema) as writer:
    batch = pa.RecordBatch.from_pylist(metadata_rows, schema=metadata_schema)
    writer.write(batch)
