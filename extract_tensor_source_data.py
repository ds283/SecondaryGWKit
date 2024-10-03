import argparse
import itertools
import sys
from pathlib import Path

import pyarrow as pa
import ray
from pyarrow import dataset
from pyarrow.csv import CSVWriter

from ComputeTargets import (
    MatterTransferFunctionIntegration,
)
from ComputeTargets.TensorSource import TensorSource
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
)
from CosmologyModels.LambdaCDM import Planck2018, LambdaCDM
from Datastore.SQL.ShardedPool import ShardedPool
from MetadataConcepts import tolerance
from RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_TIMEOUT = 60

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
    "--profile-db",
    type=str,
    default=None,
    help="write profiling and performance data to the specified database",
)
parser.add_argument(
    "--ray-address", default="auto", type=str, help="specify address of Ray cluster"
)
parser.add_argument(
    "--output",
    default="tensor-source-csv",
    type=str,
    help="specify folder for output files",
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
with ShardedPool(
    version_label="2024.1.1",
    db_name=args.database,
    timeout=args.db_timeout,
    profile_db=args.profile_db,
) as pool:

    # set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
    units = Mpc_units()
    params = Planck2018()
    LambdaCDM_Planck2018 = ray.get(
        pool.object_get(LambdaCDM, params=params, units=units)
    )

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
    k_exit_queue = RayWorkPool(
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

    def build_Tk_work(k_exit: wavenumber_exit_time):
        if not k_exit.available:
            raise RuntimeError(
                f"k_exit object (store_id={k_exit.store_id}) is not ready"
            )

        return pool.object_get(
            MatterTransferFunctionIntegration,
            solver_labels=[],
            cosmology=LambdaCDM_Planck2018,
            k=k_exit,
            z_sample=None,
            z_init=None,  # will query for any init time
            atol=atol,
            rtol=rtol,
        )

    Tk_queue = RayWorkPool(
        pool,
        k_exit_times,
        task_builder=build_Tk_work,
        compute_handler=None,
        store_handler=None,
        store_results=True,
        title="QUERY MATTER TRANSFER FUNCTION VALUES",
    )
    Tk_queue.run()
    Tks = Tk_queue.results

    time_series_schema = pa.schema(
        [
            ("q_serial", pa.int32()),
            ("q_exit_serial", pa.int32()),
            ("q_inv_Mpc", pa.float64()),
            ("q_exit", pa.float64()),
            ("r_serial", pa.int32()),
            ("r_exit_serial", pa.int32()),
            ("r_inv_Mpc", pa.float64()),
            ("r_exit", pa.float64()),
            ("z_serial", pa.int32()),
            ("z", pa.float64()),
            ("source_term", pa.float64()),
            ("undiff_part", pa.float64()),
            ("diff_part", pa.float64()),
            ("analytic_source_term", pa.float64()),
            ("analytic_undiff_part", pa.float64()),
            ("analytic_diff_part", pa.float64()),
        ]
    )

    @ray.remote
    def write_CSV_content(source: TensorSource):
        base_path = Path(args.output).resolve()
        time_series_path = (
            base_path
            / f"time-series/storeid{source.store_id}-qid{source.q.store_id}-rid{source.r.store_id}.csv"
        )
        time_series_path.parents[0].mkdir(exist_ok=True, parents=True)

        with CSVWriter(time_series_path, schema=time_series_schema) as writer:
            time_series_rows = [
                {
                    "q_serial": source.q.store_id,
                    "q_exit_serial": source._q_exit.store_id,
                    "q_inv_Mpc": source.q.k_inv_Mpc,
                    "q_exit": source._q_exit.z_exit,
                    "r_serial": source.r.store_id,
                    "r_exit_serial": source._r_exit.store_id,
                    "r_inv_Mpc": source.r.k_inv_Mpc,
                    "r_exit": source._r_exit.z_exit,
                    "z_serial": value.z.store_id,
                    "z": value.z.z,
                    "source_term": value.source_term,
                    "undiff_part": value.undiff_part,
                    "diff_part": value.diff_part,
                    "analytic_source_term": value.analytic_source_term,
                    "analytic_undiff_part": value.analytic_undiff_part,
                    "analytic_diff_part": value.analytic_diff_part,
                }
                for value in source.values
            ]
            batch = pa.RecordBatch.from_pylist(
                time_series_rows, schema=time_series_schema
            )
            writer.write(batch)

    def build_tensor_source_work(grid_idx):
        idx_i, idx_j = grid_idx

        q = k_array[idx_i]
        Tq = Tks[idx_i]
        Tr = Tks[idx_j]

        return pool.object_get(TensorSource, z_sample=None, q=q, Tq=Tq, Tr=Tr)

    def tensor_source_available_map(Tk: MatterTransferFunctionIntegration):
        return write_CSV_content.remote(Tk)

    tensor_source_grid = list(
        itertools.combinations_with_replacement(range(len(k_exit_times)), 2)
    )

    build_csv_queue = RayWorkPool(
        pool,
        tensor_source_grid,
        task_builder=build_tensor_source_work,
        available_handler=tensor_source_available_map,
        compute_handler=None,
        store_handler=None,
        store_results=False,
        title="EXTRACT TENSOR SOURCE DATA",
    )
    build_csv_queue.run()

    # use PyArrow to ingest all created CSV files into a dataaset, and then re-emit them as a single consolidated CSV
    base_path = Path(args.output).resolve()
    time_series_path = base_path / "time-series"
    metadata_path = base_path / "metadata"

    time_series_data = dataset.dataset(
        time_series_path, format="csv", schema=time_series_schema
    )
    time_series_sorted = time_series_data.sort_by(
        [("q_inv_Mpc", "ascending"), ("r_inv_Mpc", "ascending"), ("z", "descending")]
    )
    dataset.write_dataset(
        time_series_sorted,
        base_dir=base_path,
        basename_template="time-series-{i}.csv",
        format="csv",
        schema=time_series_schema,
        existing_data_behavior="overwrite_or_ignore",
    )
