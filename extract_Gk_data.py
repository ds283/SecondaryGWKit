import argparse
import sys
from pathlib import Path

import pyarrow as pa
import ray
from pyarrow import dataset
from pyarrow.csv import CSVWriter

from ComputeTargets import (
    GkNumericalIntegration,
    BackgroundModel,
)
from CosmologyConcepts import (
    wavenumber,
    wavenumber_exit_time,
    redshift_array,
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
    "--output", default="Gk-out", type=str, help="specify folder for output files"
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
    job_name="extract_Gk_data",
    prune_unvalidated=False,
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
    z_array = ray.get(pool.read_redshift_table())
    z_sample = redshift_array(z_array=z_array)

    model = ray.get(
        pool.object_get(
            BackgroundModel,
            solver_labels=[],
            cosmology=LambdaCDM_Planck2018,
            z_sample=z_sample,
            atol=atol,
            rtol=rtol,
        )
    )
    if not model.available:
        raise RuntimeError(
            "Could not locate suitable background model instance in the datastore"
        )

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

    time_series_schema = pa.schema(
        [
            ("k_serial", pa.int32()),
            ("k_exit_serial", pa.int32()),
            ("k_inv_Mpc", pa.float64()),
            ("z_exit", pa.float64()),
            ("z_source_serial", pa.int32()),
            ("z_source", pa.float64()),
            ("z_response_serial", pa.int32()),
            ("z_response", pa.float64()),
            ("G", pa.float64()),
            ("Gprime", pa.float64()),
            ("analytic_G", pa.float64()),
            ("analytic_Gprime", pa.float64()),
            ("omega_WKB_sq", pa.float64()),
            ("WKB_criterion", pa.float64()),
        ]
    )

    metadata_schema = pa.schema(
        [
            ("integration_serial", pa.int32()),
            ("k_serial", pa.int32()),
            ("k_exit_serial", pa.int32()),
            ("k_inv_Mpc", pa.float64()),
            ("z_exit", pa.float64()),
            ("z_source_serial", pa.int32()),
            ("z_source", pa.float64()),
            ("compute_time", pa.float64()),
            ("compute_steps", pa.int32()),
            ("RHS_evaluations", pa.int32()),
            ("mean_RHS_time", pa.float64()),
            ("min_RHS_time", pa.float64()),
            ("max_RHS_time", pa.float64()),
            ("has_unresolved_osc", pa.bool_()),
            ("unresolved_z", pa.float64()),
            ("unresolved_efolds_subh", pa.float64()),
            ("init_efolds_suph", pa.float64()),
            ("stop_efolds_subh", pa.float64()),
            ("stop_G", pa.float64()),
            ("stop_Gprime", pa.float64()),
        ]
    )

    @ray.remote
    def write_CSV_content(Gk: GkNumericalIntegration):
        base_path = Path(args.output).resolve()
        time_series_path = (
            base_path
            / f"time-series/storeid{Gk.store_id}-kid{Gk.k.store_id}-zsource{Gk.z_source.store_id}.csv"
        )
        time_series_path.parents[0].mkdir(exist_ok=True, parents=True)

        with CSVWriter(time_series_path, schema=time_series_schema) as writer:
            time_series_rows = [
                {
                    "k_serial": Gk.k.store_id,
                    "k_exit_serial": Gk._k_exit.store_id,
                    "k_inv_Mpc": Gk.k.k_inv_Mpc,
                    "z_exit": Gk.z_exit,
                    "z_source_serial": Gk.z_source.store_id,
                    "z_source": Gk.z_source.z,
                    "z_response_serial": value.z.store_id,
                    "z_response": value.z.z,
                    "G": value.G,
                    "Gprime": value.Gprime,
                    "analytic_G": value.analytic_G,
                    "analytic_Gprime": value.analytic_Gprime,
                    "omega_WKB_sq": value.omega_WKB_sq,
                    "WKB_criterion": value.WKB_criterion,
                }
                for value in Gk.values
            ]
            batch = pa.RecordBatch.from_pylist(
                time_series_rows, schema=time_series_schema
            )
            writer.write(batch)

        metadata_path = (
            base_path
            / f"metadata/storeid{Gk.store_id}-kid{Gk.k.store_id}-zsource{Gk.z_source.store_id}.csv"
        )
        metadata_path.parents[0].mkdir(exist_ok=True, parents=True)

        with CSVWriter(metadata_path, schema=metadata_schema) as writer:
            metadata_rows = [
                {
                    "integration_serial": Gk.store_id,
                    "k_serial": Gk.k.store_id,
                    "k_exit_serial": Gk._k_exit.store_id,
                    "k_inv_Mpc": Gk.k.k_inv_Mpc,
                    "z_exit": Gk.z_exit,
                    "z_source_serial": Gk.z_source.store_id,
                    "z_source": Gk.z_source.z,
                    "compute_time": Gk.compute_time,
                    "compute_steps": Gk.compute_steps,
                    "RHS_evaluations": Gk.RHS_evaluations,
                    "mean_RHS_time": Gk.mean_RHS_time,
                    "min_RHS_time": Gk.min_RHS_time,
                    "max_RHS_time": Gk.max_RHS_time,
                    "has_unresolved_osc": Gk.has_unresolved_osc,
                    "unresolved_z": Gk.unresolved_z,
                    "unresolved_efolds_subh": Gk.unresolved_efolds_subh,
                    "init_efolds_suph": Gk.init_efolds_suph,
                    "stop_efolds_subh": Gk.stop_efolds_subh,
                    "stop_G": Gk.stop_G,
                    "stop_Gprime": Gk.stop_Gprime,
                }
            ]
            batch = pa.RecordBatch.from_pylist(metadata_rows, schema=metadata_schema)
            writer.write(batch)

    def build_Gk_work(k_exit: wavenumber_exit_time):
        if not k_exit.available:
            raise RuntimeError(
                f"k_exit object (store_id={k_exit.store_id}) is not ready"
            )

        source_zs = z_sample.truncate(k_exit.z_exit_subh_e3, keep="higher")

        return [
            pool.object_get(
                GkNumericalIntegration,
                solver_labels=[],
                model=model,
                k=k_exit,
                z_sample=None,
                z_source=source_z,
                atol=atol,
                rtol=rtol,
            )
            for source_z in source_zs
        ]

    def Gk_available_map(Gk: GkNumericalIntegration):
        return write_CSV_content.remote(Gk)

    build_csv_queue = RayWorkPool(
        pool,
        k_exit_times,
        task_builder=build_Gk_work,
        available_handler=Gk_available_map,
        compute_handler=None,
        store_handler=None,
        store_results=False,
        title="EXTRACT TENSOR GREEN FUNCTION TIME SERIES DATA",
        notify_time_interval=60,
        notify_batch_size=5,
    )
    build_csv_queue.run()

    # use PyArrow to ingest all created CSV files into a dataaset, and then re-emit them as a single consolidated CSV
    base_path = Path(args.output).resolve()
    time_series_path = base_path / "time-series"
    metadata_path = base_path / "metadata"

    time_series_data = dataset.dataset(
        time_series_path, format="csv", schema=time_series_schema
    )
    # TODO: sorting turns out to be pointless. PyArrow does not guarantee sort order when writing out a dataset.
    #  See: https://github.com/apache/arrow/issues/26818, https://github.com/apache/arrow/issues/39030
    time_series_sorted = time_series_data.sort_by(
        [
            ("k_inv_Mpc", "ascending"),
            ("z_source", "descending"),
            ("z_response", "descending"),
        ]
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
