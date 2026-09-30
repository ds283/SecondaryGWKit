"""Schedule QuadSourceIntegral work items against a datastore that already holds every
ingredient, and record the results as JSON lines.

This exists because main.py's `--quad-source-integral-queue` stage aborts the whole run on the
first failing work item, and on a production-scale redshift grid about 44 % of the work items
fail in `build_partition`'s region guard (issue [12-region-check-absolute-tolerance] on the
campaign board). The compute path itself is untouched: this driver calls the real
`@ray.remote compute_QuadSource_integral` with the same nine-element payload
`main.py:build_QuadSourceIntegral_payload` assembles, and records both the successes and the
failures, instead of letting the first failure end the run.

Nothing is written to the datastore: the results are not stored, only reported.

    PYTHONPATH=. ./venv/bin/python docs/source-remediation-verification/run_quadsource_integrals.py \
        --database <primary sqlite> --shards 4 --cpus 10 --zend 1e7 \
        --samples-per-log10z 100 --z-stride 4 --out results.jsonl
"""

import argparse
import itertools
import json
import time
from types import SimpleNamespace
from math import log, exp

import ray

from ComputeTargets import (
    BackgroundModel,
    BesselPhaseProxy,
    GkSource,
    GkSourceProxy,
)
from ComputeTargets.BackgroundModel import ModelProxy
from ComputeTargets.QuadSourceIntegral import compute_QuadSource_integral
from CosmologyConcepts import (
    wavenumber_exit_time,
    wavenumber_array,
    redshift_array,
)
from Datastore.SQL.ShardedPool import ShardedPool
from LiouvilleGreen.bessel_phase import bessel_phase
from MetadataConcepts import tolerance
from Units import Mpc_units
from config.defaults import (
    DEFAULT_HEXIT_ABS_TOLERANCE,
    DEFAULT_HEXIT_REL_TOLERANCE,
    DEFAULT_TK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_TK_NUMERIC_REL_TOLERANCE,
    DEFAULT_QUADRATURE_ATOL,
    DEFAULT_QUADRATURE_RTOL,
    DEFAULT_FLOAT_PRECISION,
)
from config.model_list import build_model_list
from config.sharding import (
    replicated_tables,
    sharded_tables,
    shard_key_wavenumber_store_id,
    shard_key_type,
    read_table_config,
)

VERSION_LABEL = "2025.1.1"


def closes_triangle(k, q, r) -> bool:
    """main.py:closes_triangle, verbatim (prompt 04)."""
    k_val = k.k.k
    q_val = q.k.k
    r_val = r.k.k
    tol = DEFAULT_FLOAT_PRECISION * max(k_val, q_val, r_val)
    return (abs(q_val - r_val) - tol) <= k_val <= (q_val + r_val + tol)


def round_trip_trips_guard(z: float) -> bool:
    """True if log(1+z) -> exp() - 1 moves z down by more than DEFAULT_FLOAT_PRECISION, which is
    what `_check_region_covers` compares against an exactly-equal region boundary."""
    return (exp(log(1.0 + z)) - 1.0) < z - DEFAULT_FLOAT_PRECISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument("--zend", type=float, required=True)
    parser.add_argument("--samples-per-log10z", type=int, default=100)
    parser.add_argument("--model-label", default="LambdaCDM")
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument(
        "--z-stride",
        type=int,
        default=1,
        help="take every N-th response redshift of the eligible ones",
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="also schedule the work items whose z_response trips the region guard",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_QUADRATURE_ATOL,
        help="quadrature atol handed to the task (default: the pipeline value)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_QUADRATURE_RTOL,
        help="quadrature rtol handed to the task (default: the pipeline value)",
    )
    parser.add_argument(
        "--z-values",
        type=float,
        nargs="+",
        default=None,
        help="restrict to the response redshifts nearest these values",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ray.init(num_cpus=args.cpus, include_dashboard=False)
    units = Mpc_units()

    with ShardedPool(
        version_label=VERSION_LABEL,
        db_name=args.database,
        ShardKeyType=shard_key_type,
        ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
        replicated_tables=replicated_tables,
        sharded_tables=sharded_tables,
        timeout=60,
        shards=args.shards,
        profile_agent=None,
        job_name="source-remediation-verification-qsi",
        prune_unvalidated=False,
        drop_actions=[],
        read_table_config=read_table_config,
    ) as pool:
        # each keyed target is looked up under its own pair, the one main.py writes it with
        # (prompt 05a of prompts/tolerance-convergence). BackgroundModel, TkWKBIntegration and
        # GkSource are keyed on no tolerance, so their lookups pass none. A lookup under another
        # target's pair does not raise; it misses the row main.py wrote
        # ([05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair])
        hexit_atol, hexit_rtol, Tk_numeric_atol, Tk_numeric_rtol = ray.get(
            [
                pool.object_get(tolerance, tol=DEFAULT_HEXIT_ABS_TOLERANCE),
                pool.object_get(tolerance, tol=DEFAULT_HEXIT_REL_TOLERANCE),
                pool.object_get(tolerance, tol=DEFAULT_TK_NUMERIC_ABS_TOLERANCE),
                pool.object_get(tolerance, tol=DEFAULT_TK_NUMERIC_REL_TOLERANCE),
            ]
        )
        models = build_model_list(pool, units)
        model_data = [m for m in models if m["label"] == args.model_label][0]
        cosmology = model_data["cosmology"]

        background: BackgroundModel = ray.get(
            pool.object_get(
                BackgroundModel,
                solver_labels=[],
                cosmology=cosmology,
                z_sample=None,
            )
        )
        if not background.available:
            raise RuntimeError("BackgroundModel not found")
        model_proxy = ModelProxy(background)
        mod_f = background.functions

        source_k = wavenumber_array(
            k_array=ray.get(pool.read_table("wavenumber", units=units, is_source=True))
        )
        response_k = wavenumber_array(
            k_array=ray.get(
                pool.read_table("wavenumber", units=units, is_response=True)
            )
        )
        z_source_sample = redshift_array(
            z_array=ray.get(
                pool.read_table("redshift", is_source=True, model_proxy=model_proxy)
            )
        )
        z_response_sample = redshift_array(
            z_array=ray.get(
                pool.read_table("redshift", is_response=True, model_proxy=model_proxy)
            )
        )
        print(
            f"** {len(source_k)} source k, {len(response_k)} response k, "
            f"{len(z_source_sample)} source z, {len(z_response_sample)} response z"
        )

        def k_exit_of(k):
            return pool.object_get(
                "wavenumber_exit_time",
                k=k,
                cosmology=cosmology,
                atol=hexit_atol,
                rtol=hexit_rtol,
            )

        source_k_exit = ray.get([k_exit_of(k) for k in source_k])
        response_k_exit = ray.get([k_exit_of(k) for k in response_k])

        # the six production tags main.py builds from the grid (main.py:430-465)
        tag_labels = [
            "TkOneLoopDensity",
            f"SourceRedshiftGrid_{len(z_source_sample)}",
            "OutsideHorizonEfolds_e3",
            f"LargestSourceRedshift_{z_source_sample.max.z:.5g}",
            f"SmallestSourceRedshift_{z_source_sample.min.z:.5g}",
            f"SourceSamplesPerLog10Z_{args.samples_per_log10z}",
        ]
        print(f"** tags: {tag_labels}")
        tags = ray.get([pool.object_get("store_tag", label=lbl) for lbl in tag_labels])

        # Bessel phase splines, exactly as main.py:510-530
        largest_source_k = max(source_k_exit, key=lambda x: x.k.k).k
        largest_x = largest_source_k.k * mod_f.tau(args.zend)
        largest_x_with_clearance = 1.075 * largest_x
        print(f"** largest x = {largest_x:.5g} (+7.5% clearance)")
        Bessel_0pt5_proxy = BesselPhaseProxy(
            bessel_phase(0.5 + args.b, largest_x_with_clearance, atol=1e-25, rtol=5e-14)
        )
        Bessel_2pt5_proxy = BesselPhaseProxy(
            bessel_phase(2.5 + args.b, largest_x_with_clearance, atol=1e-25, rtol=5e-14)
        )

        GkSource_policy = ray.get(
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="1.5"',
                Levin_threshold=1.5,
                numeric_policy="maximize-WKB",
            )
        )

        # main.py:2245-2253
        z_source_integral_max_z = z_source_sample[10]
        z_response_pool = [
            z
            for z in z_response_sample
            if z.z < z_source_integral_max_z.z - DEFAULT_FLOAT_PRECISION
        ]
        blocked = [z for z in z_response_pool if round_trip_trips_guard(z.z)]
        eligible = [z for z in z_response_pool if not round_trip_trips_guard(z.z)]
        print(
            f"** response redshifts below z_source_max: {len(z_response_pool)}; "
            f"{len(blocked)} trip the region guard, {len(eligible)} do not"
        )
        chosen = (z_response_pool if args.include_blocked else eligible)[
            :: args.z_stride
        ]
        if args.z_values is not None:
            pool_z = z_response_pool if args.include_blocked else eligible
            chosen = [
                min(pool_z, key=lambda z: abs(log(1.0 + z.z) - log(1.0 + target)))
                for target in args.z_values
            ]
        print(f"** scheduling {len(chosen)} response redshifts")

        # triangle-closing triples, main.py:2677-2694
        triples = [
            (k, q, r)
            for (q, r) in itertools.combinations_with_replacement(source_k_exit, 2)
            for k in response_k_exit
            if closes_triangle(k, q, r)
        ]
        print(f"** {len(triples)} triangle-closing (k,q,r) triples")

        # an exit time the store does not hold at main.py's pair comes back unpopulated, with no
        # store_id to key the caches below on. A triple that needs one is set aside and counted
        # as a failure at every response redshift, rather than raising on that store_id
        missing_exit = [kqr for kqr in triples if not all(x.available for x in kqr)]
        triples = [kqr for kqr in triples if all(x.available for x in kqr)]
        if missing_exit:
            print(
                f"** {len(missing_exit)} triples set aside: an exit time is not available"
            )

        # ------------------------------------------------------------ ingredients
        distinct_k = {k.store_id: k for k, _, _ in triples}
        pairs = {(q.store_id, r.store_id): (q, r) for _, q, r in triples}

        print("** looking up QuadSource rows")
        source_cache = {}
        for (q_id, r_id), (q, r) in pairs.items():
            # a QuadSource is keyed on the two TkNumericIntegrations it is computed from
            # (prompts/datastore-integrity, prompt 09a). This lookup precedes the transfer-function
            # lookups below and holds only physical labels, so it asks for the reader mode, which
            # refuses more than one stored match
            obj = ray.get(
                pool.object_get(
                    "QuadSource",
                    model=model_proxy,
                    z_sample=None,
                    q=q,
                    r=r,
                    labels_only=True,
                    tags=tags,
                )
            )
            source_cache[(q_id, r_id)] = obj
        missing = [key for key, obj in source_cache.items() if not obj.available]
        print(f"   {len(source_cache)} QuadSource rows, {len(missing)} unavailable")

        print("** looking up transfer functions")
        Tk_numeric_cache = {}
        Tk_WKB_cache = {}
        for k_exit in {x.store_id: x for _, q, r in triples for x in (q, r)}.values():
            Tk_numeric_cache[k_exit.store_id] = ray.get(
                pool.object_get(
                    "TkNumericIntegration",
                    model=model_proxy,
                    z_sample=None,
                    k=k_exit,
                    z_init=None,
                    atol=Tk_numeric_atol,
                    rtol=Tk_numeric_rtol,
                    tags=tags,
                )
            )
            # a TkWKBIntegration is keyed on the TkNumericIntegration it starts from
            # (prompts/datastore-integrity, prompt 07), so the WKB lookup names the numeric row
            # just looked up. A mode with no numeric row has no WKB row that can be keyed: it is
            # recorded as unavailable, as a missing WKB row always was, rather than looked up
            Tk_numeric = Tk_numeric_cache[k_exit.store_id]
            if not Tk_numeric.available:
                Tk_WKB_cache[k_exit.store_id] = SimpleNamespace(available=False)
                continue
            Tk_WKB_cache[k_exit.store_id] = ray.get(
                pool.object_get(
                    "TkWKBIntegration",
                    solver_labels=[],
                    model=model_proxy,
                    z_sample=None,
                    k=k_exit,
                    numeric_serial=Tk_numeric.store_id,
                    tags=tags,
                )
            )
        print(
            f"   {len(Tk_numeric_cache)} numeric, {len(Tk_WKB_cache)} WKB; "
            f"unavailable: "
            f"{sum(1 for o in Tk_numeric_cache.values() if not o.available)}/"
            f"{sum(1 for o in Tk_WKB_cache.values() if not o.available)}"
        )

        # ------------------------------------------------------------ work items
        n_ok = n_fail = 0
        failures = {}
        if missing_exit:
            failures["exit time unavailable"] = len(missing_exit) * len(chosen)
            n_fail = failures["exit time unavailable"]
        with open(args.out, "w") as out:
            for z_response in chosen:
                print(f"** z_response = {z_response.z:.6g}", flush=True)
                # GkSourcePolicyData for each response k at this z_response
                Gk_cache = {}
                for k_exit in distinct_k.values():
                    src: GkSource = ray.get(
                        pool.object_get(
                            "GkSource",
                            model=model_proxy,
                            k=k_exit,
                            z_response=z_response,
                            z_sample=None,
                        )
                    )
                    Gk_cache[k_exit.store_id] = ray.get(
                        pool.object_get(
                            "GkSourcePolicyData",
                            source=GkSourceProxy(src),
                            policy=GkSource_policy,
                            k=k_exit,
                        )
                    )

                refs = {}
                for k, q, r in triples:
                    payload_ok = (
                        Gk_cache[k.store_id].available
                        and source_cache[(q.store_id, r.store_id)].available
                        and Tk_numeric_cache[q.store_id].available
                        and Tk_WKB_cache[q.store_id].available
                        and Tk_numeric_cache[r.store_id].available
                        and Tk_WKB_cache[r.store_id].available
                    )
                    if not payload_ok:
                        failures.setdefault("ingredient unavailable", 0)
                        failures["ingredient unavailable"] += 1
                        n_fail += 1
                        continue
                    ref = compute_QuadSource_integral.remote(
                        model_proxy,
                        k,
                        q,
                        r,
                        source_cache[(q.store_id, r.store_id)],
                        Gk_cache[k.store_id],
                        z_response,
                        z_source_integral_max_z,
                        args.b,
                        Bessel_0pt5_proxy,
                        Bessel_2pt5_proxy,
                        Tk_numeric_cache[q.store_id],
                        Tk_WKB_cache[q.store_id],
                        Tk_numeric_cache[r.store_id],
                        Tk_WKB_cache[r.store_id],
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    refs[ref] = (k, q, r)

                for ref, (k, q, r) in refs.items():
                    try:
                        data = ray.get(ref)
                    except Exception as exc:  # noqa: BLE001 - diagnostic driver
                        message = str(exc).strip().split("\n")[-1]
                        key = message.split(":")[-1][:80]
                        failures.setdefault(key, 0)
                        failures[key] += 1
                        n_fail += 1
                        out.write(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "k": k.k.k_inv_Mpc,
                                    "q": q.k.k_inv_Mpc,
                                    "r": r.k.k_inv_Mpc,
                                    "z_response": z_response.z,
                                    "error": message,
                                }
                            )
                            + "\n"
                        )
                        continue
                    n_ok += 1
                    record = {
                        "status": "ok",
                        "k": k.k.k_inv_Mpc,
                        "q": q.k.k_inv_Mpc,
                        "r": r.k.k_inv_Mpc,
                        "k_z_exit": k.z_exit,
                        "z_response": z_response.z,
                        "rtol": args.rtol,
                        "atol": args.atol,
                    }
                    for field in (
                        "total",
                        "total_abserr",
                        "total_converged",
                        "total_phase_limited",
                        "b",
                        "numeric_quad",
                        "WKB_Levin",
                        "analytic_rad",
                        "compute_time",
                        "analytic_compute_time",
                        "WKB_phase_spline_chunks",
                    ):
                        record[field] = data[field]
                    levin = data["WKB_Levin_data"]
                    record["Levin_regions"] = (
                        levin.num_regions if levin is not None else None
                    )
                    record["Levin_evaluations"] = (
                        levin.evaluations if levin is not None else None
                    )
                    record["Levin_simple_regions"] = (
                        levin.num_simple_regions if levin is not None else None
                    )
                    record["Levin_SVD_errors"] = (
                        levin.num_SVD_errors if levin is not None else None
                    )
                    record["Levin_elapsed"] = (
                        levin.elapsed if levin is not None else None
                    )
                    record["metadata"] = data["metadata"]
                    out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"   cumulative: {n_ok} ok, {n_fail} failed", flush=True)

        print(f"\n** {n_ok} work items completed, {n_fail} failed")
        for key, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"   {count:>6}  {key}")


if __name__ == "__main__":
    main()
