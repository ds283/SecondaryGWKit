import argparse
import sys
from pathlib import Path

import pyarrow as pa
import ray
from pyarrow import dataset
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
    type=int,
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
    task_builder=create_k_exit_work,
    compute_handler=None,
    store_handler=None,
    store_results=True,
    title="QUERY K_EXIT VALUES",
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


@ray.remote
def write_CSV_content(Tk: MatterTransferFunctionIntegration):
    time_series_path = Path(
        f"Tk-csv/time-series/storeid{Tk.store_id}-kid{Tk.k.store_id}.csv"
    ).resolve()
    time_series_path.parents[0].mkdir(exist_ok=True, parents=True)

    with CSVWriter(time_series_path, schema=time_series_schema) as writer:
        time_series_rows = [
            {
                "k_serial": Tk.k.store_id,
                "k_exit_serial": Tk._k_exit.store_id,
                "k_inv_Mpc": Tk.k.k_inv_Mpc,
                "z_init_serial": Tk.z_init.store_id,
                "z_init": Tk.z_init.z,
                "z_exit": Tk.z_exit,
                "z_serial": value.z.store_id,
                "z": value.z.z,
                "value": value.value,
            }
            for value in Tk.values
        ]
        batch = pa.RecordBatch.from_pylist(time_series_rows, schema=time_series_schema)
        writer.write(batch)

    metadata_path = Path(
        f"Tk-csv/metadata/storeid{Tk.store_id}-kid{Tk.k.store_id}.csv"
    ).resolve()
    metadata_path.parents[0].mkdir(exist_ok=True, parents=True)

    with CSVWriter(metadata_path, schema=metadata_schema) as writer:
        metadata_rows = [
            {
                "integration_serial": Tk.store_id,
                "k_serial": Tk.k.store_id,
                "k_exit_serial": Tk._k_exit.store_id,
                "k_inv_Mpc": Tk.k.k_inv_Mpc,
                "z_init_serial": Tk.z_init.store_id,
                "z_init": Tk.z_init.z,
                "z_exit": Tk.z_exit,
                "compute_time": Tk.compute_time,
                "compute_steps": Tk.compute_steps,
                "RHS_evaluations": Tk.RHS_evaluations,
                "mean_RHS_time": Tk.mean_RHS_time,
                "min_RHS_time": Tk.min_RHS_time,
                "max_RHS_time": Tk.max_RHS_time,
            }
        ]
        batch = pa.RecordBatch.from_pylist(metadata_rows, schema=metadata_schema)
        writer.write(batch)


def build_Tk_work(k_exit: wavenumber_exit_time):
    if not k_exit.available:
        raise RuntimeError(f"k_exit object (store_id={k_exit.store_id}) is not ready")

    return pool.object_get(
        MatterTransferFunctionIntegration,
        solver_labels=[],
        cosmology=LambdaCDM_Planck2018,
        k=k_exit,
        z_sample=None,
        z_init=None,  # will query for any sample time
        atol=atol,
        rtol=rtol,
    )


def Tk_available_map(Tk: MatterTransferFunctionIntegration):
    return write_CSV_content.remote(Tk)


build_csv_queue = RayWorkQueue(
    pool,
    k_exit_times,
    task_builder=build_Tk_work,
    available_handler=Tk_available_map,
    compute_handler=None,
    store_handler=None,
    store_results=False,
    title="EXTRACT MATTER TRANSFER FUNCTION DATA",
)
build_csv_queue.run()


# use PyArrow to ingest all created CSV files into a dataaset, and then re-emit them as a single consolidated CSV
base_path = Path("Tk-csv").resolve()
time_series_path = base_path / "time-series"
metadata_path = base_path / "metadata"

time_series_data = dataset.dataset(
    time_series_path, format="csv", schema=time_series_schema
)
time_series_sorted = time_series_data.sort_by(
    [("k_inv_Mpc", "ascending"), ("z", "descending")]
)
dataset.write_dataset(
    time_series_sorted,
    base_dir=base_path,
    basename_template="time-series-{i}.csv",
    format="csv",
    schema=time_series_schema,
    existing_data_behavior="overwrite_or_ignore",
)

metadata_data = dataset.dataset(metadata_path, format="csv", schema=metadata_schema)
dataset.write_dataset(
    metadata_data,
    base_dir=base_path,
    basename_template="metadata-{i}.csv",
    format="csv",
    schema=metadata_schema,
    existing_data_behavior="overwrite_or_ignore",
)
