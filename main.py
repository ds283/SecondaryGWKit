import argparse
import itertools
import sys
from datetime import datetime

import numpy as np
import ray

from ComputeTargets import (
    MatterTransferFunctionIntegration,
    TensorGreenFunctionIntegration,
    IntegrationSolver,
)
from ComputeTargets.TensorSource import TensorSource
from CosmologyConcepts import (
    wavenumber,
    redshift,
    wavenumber_array,
    wavenumber_exit_time,
    redshift_array,
)
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Datastore.SQL.sqla_impl import ShardedPool
from MetadataConcepts import tolerance, store_tag
from RayWorkPool import RayWorkPool
from Units import Mpc_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_LABEL = "SecondaryGWKit-test"
DEFAULT_TIMEOUT = 60
DEFAULT_SHARDS = 20
DEFAULT_RAY_ADDRESS = "auto"
DEFAULT_SAMPLES_PER_LOG10_Z = 150
DEFAULT_ZEND = 0.1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    type=str,
    default=None,
    help="read/write work items using the specified database cache",
)
parser.add_argument(
    "--job-name",
    default=DEFAULT_LABEL,
    help="specify a label for this job (used to identify integrations and other numerical products)",
)
parser.add_argument(
    "--shards",
    type=int,
    default=DEFAULT_SHARDS,
    help="specify number of shards to be used when creating a new datastore",
)
parser.add_argument(
    "--db-timeout",
    type=int,
    default=DEFAULT_TIMEOUT,
    help="specify connection timeout for database layer",
)
parser.add_argument(
    "--profile-db",
    type=str,
    default=None,
    help="write profiling and performance data to the specified database",
)
parser.add_argument(
    "--samples-log10z",
    type=int,
    default=DEFAULT_SAMPLES_PER_LOG10_Z,
    help="specify number of z-sample points per log10(z)",
)
parser.add_argument(
    "--zend",
    type=float,
    default=DEFAULT_ZEND,
    help="specify final redshift",
)
parser.add_argument(
    "--ray-address",
    default=DEFAULT_RAY_ADDRESS,
    type=str,
    help="specify address of Ray cluster",
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
    shards=args.shards,
    profile_db=args.profile_db,
)

# set up LambdaCDM object representing a basic Planck2018 cosmology in Mpc units
units = Mpc_units()
params = Planck2018()
LambdaCDM_Planck2018 = ray.get(pool.object_get(LambdaCDM, params=params, units=units))

zend = args.zend
samples_per_log10z = args.samples_log10z


## UTILITY FUNCTIONS


def convert_to_wavenumbers(k_sample_set):
    return pool.object_get(
        wavenumber,
        payload_data=[{"k_inv_Mpc": k, "units": units} for k in k_sample_set],
    )


def convert_to_redshifts(z_sample_set):
    return pool.object_get(redshift, payload_data=[{"z": z} for z in z_sample_set])


## DATASTORE OBJECTS


# build absolute and relative tolerances
atol, rtol = ray.get(
    [
        pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
        pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
    ]
)


# build stepper labels; we have to query these up-front from the pool in order to be
# certain that they get the same serial number in each database shard.
# So we can no longer construct these on-the-fly in the integration classes, as used to be done
(
    solve_ivp_RK45,
    solve_ivp_DOP852,
    solve_ivp_Radau,
    solve_ivp_BDF,
    solve_icp_LSODA,
) = ray.get(
    [
        pool.object_get(IntegrationSolver, label="solve_ivp+RK45", stepping=0),
        pool.object_get(IntegrationSolver, label="solve_ivp+DOP853", stepping=0),
        pool.object_get(IntegrationSolver, label="solve_ivp+Radau", stepping=0),
        pool.object_get(IntegrationSolver, label="solve_ivp+BDF", stepping=0),
        pool.object_get(IntegrationSolver, label="solve_ivp+LSODA", stepping=0),
    ]
)
solvers = {
    "solve_ivp+RK45-stepping0": solve_ivp_RK45,
    "solve_ivp+DOP853-stepping0": solve_ivp_DOP852,
    "solve_ivp+Radau-stepping0": solve_ivp_Radau,
    "solve_ivp+BDF-stepping0": solve_ivp_BDF,
    "solve_ivp+LSODA-stepping0": solve_icp_LSODA,
}


## STEP 1
## BUILD SAMPLE OF K-WAVENUMBERS AND OBTAIN THEIR CORRESPONDING HORIZON EXIT TIMES

# build array of k-sample points
print("\n** BUILDING ARRAY OF WAVENUMBERS AT WHICH TO SAMPLE")
k_array = ray.get(
    convert_to_wavenumbers(np.logspace(np.log10(0.1), np.log10(5e7), 500))
)
k_sample = wavenumber_array(k_array=k_array)


def build_k_exit_work(k: wavenumber):
    return pool.object_get(
        wavenumber_exit_time,
        k=k,
        cosmology=LambdaCDM_Planck2018,
        atol=atol,
        rtol=rtol,
    )


# for each k mode we sample, determine its horizon exit point
k_exit_queue = RayWorkPool(
    pool,
    k_sample,
    task_builder=build_k_exit_work,
    title="CALCULATE HORIZON EXIT TIMES",
    store_results=True,
)
k_exit_queue.run()
k_exit_times = k_exit_queue.results


## STEP 2
## BUILD A UNIVERSAL GRID OF Z-VALUES AT WHICH TO SAMPLE
print("\n** BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE")
k_exit_earliest: wavenumber_exit_time = k_exit_times[-1]
print(
    f"   @@ earliest horizon exit time is {k_exit_earliest.k.k_inv_Mpc:.5g}/Mpc with z_exit={k_exit_earliest.z_exit:.5g}"
)


# build a log-spaced universal grid of wavenumbers

universal_z_grid = k_exit_earliest.populate_z_sample(
    outside_horizon_efolds="suph_e3",
    samples_per_log10z=samples_per_log10z,
    z_end=zend,
)

# embed this universal redshift list into the database
z_array = ray.get(convert_to_redshifts(universal_z_grid))
z_sample = redshift_array(z_array=z_array)


# build tags and other labels
(
    TkProductionTag,
    GkProductionTag,
    GlobalZGridTag,
    OutsideHorizonEfoldsTag,
    LargestZTag,
    SamplesPerLog10ZTag,
) = ray.get(
    [
        pool.object_get(store_tag, label="TkOneLoopDensity"),
        pool.object_get(store_tag, label="GkOneLoopDensity"),
        pool.object_get(store_tag, label=f"GlobalRedshiftGrid_{len(z_sample)}"),
        pool.object_get(store_tag, label=f"OutsideHorizonEfolds_e3"),
        pool.object_get(store_tag, label=f"LargestRedshift_{z_sample.max.z:.5g}"),
        pool.object_get(store_tag, label=f"SamplesPerLog10Z_{samples_per_log10z}"),
    ]
)


## STEP 2
## COMPUTE MATTER TRANSFER FUNCTIONS


def build_Tk_work(k_exit: wavenumber_exit_time):
    my_sample = z_sample.truncate(k_exit.z_exit_suph_e3)
    return pool.object_get(
        MatterTransferFunctionIntegration,
        solver_labels=solvers,
        cosmology=LambdaCDM_Planck2018,
        k=k_exit,
        z_sample=my_sample,
        z_init=my_sample.max,
        atol=atol,
        rtol=rtol,
        tags=[
            TkProductionTag,
            GlobalZGridTag,
            OutsideHorizonEfoldsTag,
            LargestZTag,
            SamplesPerLog10ZTag,
        ],
        delta_logz=1.0 / float(samples_per_log10z),
    )


def build_Tk_work_label(Tk: MatterTransferFunctionIntegration):
    return f"{args.job_name}-Tk-k{Tk.k.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"


def validate_Tk_work(Tk: MatterTransferFunctionIntegration):
    return pool.object_validate(Tk)


def post_Tk_work(Tk: MatterTransferFunctionIntegration):
    return ray.put(Tk)


Tk_queue = RayWorkPool(
    pool,
    k_exit_times,
    task_builder=build_Tk_work,
    validation_handler=validate_Tk_work,
    post_handler=post_Tk_work,
    label_builder=build_Tk_work_label,
    title="CALCULATE MATTER TRANSFER FUNCTIONS",
    store_results=True,
)
Tk_queue.run()
Tks = Tk_queue.results

tensor_source_grid = list(itertools.combinations_with_replacement(range(len(Tks)), 2))


def build_tensor_source_work(grid_idx):
    idx_i, idx_j = grid_idx

    q = k_sample[idx_i]
    Tq = Tks[idx_i]
    Tr = Tks[idx_j]

    # q is not used by the TensorSource constructor, but is accepted because it functions as the shard key
    return pool.object_get(
        TensorSource,
        z_sample=z_sample,
        q=q,
        Tq=Tq,
        Tr=Tr,
        tags=[
            TkProductionTag,
            GlobalZGridTag,
            OutsideHorizonEfoldsTag,
            LargestZTag,
            SamplesPerLog10ZTag,
        ],
    )


def validate_tensor_source_work(calc: TensorSource):
    return pool.object_validate(calc)


def build_tensor_source_work_label(calc: TensorSource):
    q = calc.q
    r = calc.r
    return f"{args.job_name}-tensor-src-q{q.k_inv_Mpc:.3g}-r{r.k_inv_Mpc:.3g}-{datetime.now().replace(microsecond=0).isoformat()}"


TensorSource_queue = RayWorkPool(
    pool,
    tensor_source_grid,
    task_builder=build_tensor_source_work,
    validation_handler=validate_tensor_source_work,
    label_builder=build_tensor_source_work_label,
    title="CALCULATE TENSOR SOURCE TERMS",
    store_results=False,
)
TensorSource_queue.run()


## STEP 3
## COMPUTE TENOR GREEN'S FUNCTIONS ON A GRID OF SOURCE/RESPONSE REDSHIFT SAMPLES

G_LATEST_RESPONSE_Z = 0.5


def build_Gk_work(k_exit: wavenumber_exit_time):
    if not k_exit.available:
        raise RuntimeError(f"k_exit object (store_id={k_exit.store_id}) is not ready")

    # find redshift where this k-mode is at least 3 efolds inside the horizon
    # we won't calculate Green's functions with the source redshift later than this, because
    # the oscillations become rapid, and we are better switching to a WKB approximation
    source_zs = z_sample.truncate(k_exit.z_exit_subh_e3, keep="higher")

    work_refs = []

    for source_z in source_zs:
        response_zs = z_sample.truncate(source_z)
        if len(response_zs) >= 3:
            work_refs.append(
                pool.object_get(
                    TensorGreenFunctionIntegration,
                    solver_labels=solvers,
                    cosmology=LambdaCDM_Planck2018,
                    k=k_exit,
                    z_source=source_z,
                    z_sample=response_zs,
                    atol=atol,
                    rtol=rtol,
                    tags=[
                        GkProductionTag,
                        GlobalZGridTag,
                        OutsideHorizonEfoldsTag,
                        LargestZTag,
                        SamplesPerLog10ZTag,
                    ],
                    delta_logz=1.0 / float(samples_per_log10z),
                    mode="stop",
                )
            )

    return work_refs


def build_Gk_exit_work_label(Gk: TensorGreenFunctionIntegration):
    return f"{args.job_name}-Gk-k{Gk.k.k_inv_Mpc:.3g}-sourcez{Gk.z_source.z:.5g}-{datetime.now().replace(microsecond=0).isoformat()}"


def validate_Gk_work(Gk: TensorGreenFunctionIntegration):
    return pool.object_validate(Gk)


Gk_queue = RayWorkPool(
    pool,
    k_exit_times,
    task_builder=build_Gk_work,
    validation_handler=validate_Gk_work,
    label_builder=build_Gk_exit_work_label,
    title="CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS",
    store_results=False,
)
Gk_queue.run()
