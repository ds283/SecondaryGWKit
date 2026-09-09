"""Second half of the prompt-12 Layer 2 analysis: the checks that need the stored Green's
functions and source terms rather than the QuadSourceIntegral table.

  * audit section 4.1 -- continuity of G at `crossover_z`: for every `mixed` GkSourcePolicyData,
    evaluate `functions.numeric_Gk` and `functions.WKB_Gk` there and report the distribution of
    the relative difference;
  * audit section 4.2 -- reachability of A6: the type/quality census of GkSourcePolicyData;
  * audit section 4.3 -- `has_WKB_violation` on TkWKBIntegration/GkWKBIntegration, with the
    (k, WKB_violation_efolds_subh) pairs and, for one violating mode, T_WKB against
    analytic_T_rad just past the violation point (report only, no policy change);
  * prompt 06 section 3.4 -- the QuadSource spline residual near the hand-over, measured on the
    real stored rows against the exact radiation source.

The Green's-function part opens a real ShardedPool (as extract_GkSource_data.py does) on a
locally bootstrapped Ray instance; everything else reads the shard files directly in `mode=ro`.
Nothing is written back: the pool is opened with prune_unvalidated=False and no object_store or
object_validate call is made.
"""

import argparse
import glob
import sqlite3
from collections import Counter
from math import log, log10, sqrt, fabs

import numpy as np
import ray
from scipy.interpolate import make_interp_spline

from ComputeTargets import BackgroundModel, GkSource, GkSourceProxy
from ComputeTargets.BackgroundModel import ModelProxy
from ComputeTargets.QuadSource import source_function
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore.SQL.ShardedPool import ShardedPool
from MetadataConcepts import tolerance
from Units import Mpc_units
from config.defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from config.model_list import build_model_list
from config.sharding import (
    replicated_tables,
    sharded_tables,
    shard_key_wavenumber_store_id,
    shard_key_type,
    read_table_config,
    inventory_config,
)

VERSION_LABEL = "2025.1.1"

# Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:13-35
QUALITY = {0: "complete", 1: "acceptable", 2: "marginal", 3: "minimal", 4: "incomplete"}
TYPE = {0: "numeric", 1: "WKB", 2: "mixed", 99: "fail"}


def quantiles(values):
    if not values:
        return "n/a"
    s = sorted(values)
    n = len(s)
    pick = lambda p: s[min(n - 1, int(p * n))]
    return (
        f"min={s[0]:.3e}  p25={pick(0.25):.3e}  median={pick(0.5):.3e}  "
        f"p75={pick(0.75):.3e}  p90={pick(0.90):.3e}  max={s[-1]:.3e}"
    )


# --------------------------------------------------------------------------- SQL-only sections


def sql_sections(shard_glob):
    print("\n## audit section 4.2 -- GkSourcePolicyData census (reachability of A6)")
    census = Counter()
    crossovers = []
    for db in sorted(glob.glob(shard_glob)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for row in conn.execute(
            "select type, quality, count(*) from GkSourcePolicyData group by type, quality"
        ):
            census[(row[0], row[1])] += row[2]
        for row in conn.execute(
            "select crossover_z from GkSourcePolicyData where type='mixed'"
        ):
            crossovers.append(row[0])
        conn.close()
    total = sum(census.values())
    print(f"   rows: {total}")
    for (t, q), n in sorted(census.items()):
        print(
            f"     type={TYPE.get(t, t):<10} quality={QUALITY.get(q, q):<12} {n:>7}  "
            f"({100.0*n/total:5.2f}%)"
        )
    minimal = sum(n for (t, q), n in census.items() if QUALITY.get(q) == "minimal")
    failed = sum(n for (t, q), n in census.items() if TYPE.get(t) == "fail")
    print(f"   rows classified 'minimal' (audit section 4.2, defect A6): {minimal}")
    print(f"   rows classified type 'fail': {failed}")

    print("\n## audit section 4.3 -- has_WKB_violation")
    for table, label_col in [
        ("TkWKBIntegration", None),
        ("GkWKBIntegration", "z_source_serial"),
    ]:
        n_all = n_viol = 0
        pairs = []
        for db in sorted(glob.glob(shard_glob)):
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            wavenumbers = {
                r[0]: r[1]
                for r in conn.execute("select serial, k_inv_Mpc from wavenumber")
            }
            exits = {
                r[0]: r[1]
                for r in conn.execute(
                    "select serial, wavenumber_serial from wavenumber_exit_time"
                )
            }
            n_all += conn.execute(f"select count(*) from {table}").fetchone()[0]
            for r in conn.execute(
                f"select wavenumber_exit_serial, WKB_violation_z, "
                f"WKB_violation_efolds_subh from {table} where has_WKB_violation = 1"
            ):
                n_viol += 1
                pairs.append((wavenumbers[exits[r[0]]], r[1], r[2]))
            conn.close()
        print(f"   {table}: {n_viol} of {n_all} rows have has_WKB_violation set")
        seen = {}
        for k, z, efolds in pairs:
            if k not in seen or efolds > seen[k][1]:
                seen[k] = (z, efolds)
        for k in sorted(seen):
            z, efolds = seen[k]
            print(
                f"     k={k:.5g}/Mpc  worst WKB_violation_z={z:.5g}  efolds_subh={efolds:.4g}"
            )


def tk_wkb_against_oracle(shard_glob):
    """For every TkWKBIntegration, compare the stored T_WKB against the stored analytic_T_rad
    oracle (both are columns of TkWKBValue), overall and past any violation point."""
    print("\n## audit section 4.3 (continued) -- T_WKB against analytic_T_rad")
    for db in sorted(glob.glob(shard_glob)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        wavenumbers = {
            r[0]: r[1] for r in conn.execute("select serial, k_inv_Mpc from wavenumber")
        }
        exits = {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                "select serial, wavenumber_serial, z_exit from wavenumber_exit_time"
            )
        }
        redshifts = {r[0]: r[1] for r in conn.execute("select serial, z from redshift")}
        for wkb_serial, k_exit_serial, violation, violation_z in conn.execute(
            "select serial, wavenumber_exit_serial, has_WKB_violation, WKB_violation_z "
            "from TkWKBIntegration"
        ):
            k = wavenumbers[exits[k_exit_serial][0]]
            z_exit = exits[k_exit_serial][1]
            rows = [
                (redshifts[r[0]], r[1], r[2])
                for r in conn.execute(
                    "select z_serial, T_WKB, analytic_T_rad from TkWKBValue "
                    "where wkb_serial = ?",
                    (wkb_serial,),
                )
            ]
            rows.sort(key=lambda t: -t[0])
            if not rows:
                continue
            env = max(abs(t[2]) for t in rows)
            errs = [(abs(t[1] - t[2]) / max(abs(t[2]), 1e-3 * env), t[0]) for t in rows]
            worst = max(errs)
            print(
                f"   k={k:.5g}/Mpc  {len(rows)} WKB samples, z in "
                f"[{rows[-1][0]:.5g}, {rows[0][0]:.5g}], violation={bool(violation)}"
                + (f" at z={violation_z:.5g}" if violation else "")
            )
            print(
                f"      |T_WKB - analytic_T_rad| / max(|analytic|, 1e-3 envelope): "
                f"median={sorted(e[0] for e in errs)[len(errs)//2]:.3e} "
                f"worst={worst[0]:.3e} at z={worst[1]:.5g}"
            )
            if violation and violation_z is not None:
                past = [e for e, z in errs if z < violation_z]
                if past:
                    print(
                        f"      restricted to z < WKB_violation_z: n={len(past)} "
                        f"median={sorted(past)[len(past)//2]:.3e} max={max(past):.3e}"
                    )
        conn.close()


# ------------------------------------------------------------------ pool-backed sections


def gk_continuity(pool, model_proxy, cosmology, atol, rtol, shard_glob, max_cases):
    print("\n## audit section 4.1 -- continuity of G at crossover_z")

    # find the (k, z_response) pairs whose policy came out "mixed"
    cases = []
    for db in sorted(glob.glob(shard_glob)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        wavenumbers = {
            r[0]: r[1] for r in conn.execute("select serial, k_inv_Mpc from wavenumber")
        }
        exits = {
            r[0]: r[1]
            for r in conn.execute(
                "select serial, wavenumber_serial from wavenumber_exit_time"
            )
        }
        redshifts = {r[0]: r[1] for r in conn.execute("select serial, z from redshift")}
        for k_exit_serial, source_serial, crossover_z, quality in conn.execute(
            "select wavenumber_exit_serial, source_serial, crossover_z, quality "
            "from GkSourcePolicyData where type = 2"  # 2 = "mixed"
        ):
            z_response_serial = conn.execute(
                "select z_response_serial from GkSource where serial = ?",
                (source_serial,),
            ).fetchone()[0]
            cases.append(
                (
                    wavenumbers[exits[k_exit_serial]],
                    redshifts[z_response_serial],
                    crossover_z,
                    quality,
                )
            )
        conn.close()

    print(f"   'mixed' policy rows: {len(cases)}")
    if not cases:
        return

    step = max(1, len(cases) // max_cases)
    sampled = cases[::step][:max_cases]
    print(f"   evaluating {len(sampled)} of them")

    # the response k-modes and redshifts, read back from the datastore
    units = Mpc_units()
    k_objs = ray.get(pool.read_table("wavenumber", units=units, is_response=True))
    z_objs = ray.get(
        pool.read_table("redshift", is_response=True, model_proxy=model_proxy)
    )
    k_exits = ray.get(
        [
            pool.object_get(
                "wavenumber_exit_time",
                k=k,
                cosmology=cosmology,
                atol=atol,
                rtol=rtol,
            )
            for k in k_objs
        ]
    )
    k_exit_map = {k.k.k_inv_Mpc: k for k in k_exits}
    z_map = {z.z: z for z in z_objs}

    policy = ray.get(
        pool.object_get(
            "GkSourcePolicy",
            label='policy="maximize-WKB"-Levin-threshold="1.5"',
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
    )

    residuals = []
    failures = []
    for k_val, z_val, crossover_z, quality in sampled:
        k_exit = k_exit_map[k_val]
        z_resp = z_map[z_val]
        source: GkSource = ray.get(
            pool.object_get(
                "GkSource",
                model=model_proxy,
                k=k_exit,
                z_response=z_resp,
                z_sample=None,
                atol=atol,
                rtol=rtol,
            )
        )
        if not source.available:
            failures.append((k_val, z_val, "GkSource not available"))
            continue
        policy_data = ray.get(
            pool.object_get(
                "GkSourcePolicyData",
                source=GkSourceProxy(source),
                policy=policy,
                k=k_exit,
            )
        )
        functions = policy_data.functions
        try:
            numeric_value = functions.numeric_Gk(crossover_z)
            WKB_value = functions.WKB_Gk(crossover_z)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            failures.append((k_val, z_val, f"{type(exc).__name__}: {exc}"))
            continue
        scale = max(fabs(numeric_value), fabs(WKB_value))
        rel = fabs(numeric_value - WKB_value) / scale if scale > 0.0 else 0.0
        residuals.append((rel, k_val, z_val, crossover_z, numeric_value, WKB_value))

    print(f"   evaluated {len(residuals)}, failed {len(failures)}")
    for f in failures[:5]:
        print(f"     FAILED k={f[0]:.5g} z_response={f[1]:.5g}: {f[2]}")
    if residuals:
        print(f"   |G_num - G_WKB| / max(|G_num|,|G_WKB|) at crossover_z:")
        print(f"     {quantiles([r[0] for r in residuals])}")
        print("   worst 5:")
        for rel, k_val, z_val, cz, gn, gw in sorted(residuals, reverse=True)[:5]:
            print(
                f"     k={k_val:.5g}/Mpc z_response={z_val:.5g} crossover_z={cz:.5g}: "
                f"numeric={gn:+.6e} WKB={gw:+.6e} rel={rel:.3e}"
            )


def quadsource_residual(pool, model_proxy, shard_glob, max_pairs):
    print("\n## prompt 06 section 3.4 -- QuadSource spline residual near the hand-over")
    model = model_proxy.get()
    functions = model.functions

    pairs = []
    for db in sorted(glob.glob(shard_glob)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        wavenumbers = {
            r[0]: r[1] for r in conn.execute("select serial, k_inv_Mpc from wavenumber")
        }
        exits = {
            r[0]: r[1]
            for r in conn.execute(
                "select serial, wavenumber_serial from wavenumber_exit_time"
            )
        }
        redshifts = {r[0]: r[1] for r in conn.execute("select serial, z from redshift")}
        for serial, q_serial, r_serial, z_samples in conn.execute(
            "select serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, z_samples "
            "from QuadSource"
        ):
            values = [
                (redshifts[row[0]], row[1], row[2])
                for row in conn.execute(
                    "select z_serial, source, analytic_source_rad from QuadSourceValue "
                    "where parent_serial = ?",
                    (serial,),
                )
            ]
            values.sort(key=lambda t: -t[0])
            pairs.append(
                (
                    wavenumbers[exits[q_serial]],
                    wavenumbers[exits[r_serial]],
                    z_samples,
                    values,
                )
            )
        conn.close()

    print(f"   QuadSource rows: {len(pairs)}")
    pairs.sort(key=lambda t: (t[0], t[1]))
    step = max(1, len(pairs) // max_pairs)
    W_RAD = 1.0 / 3.0

    print(
        "        q          r      n_z   z region                 "
        "node |src-oracle|/env   midpoint |spline-oracle|/env"
    )
    worst_all = 0.0
    for q_val, r_val, n_z, values in pairs[::step][:max_pairs]:
        if len(values) < 10:
            continue
        z_nodes = np.array([v[0] for v in values])
        src = np.array([v[1] for v in values])
        oracle_nodes = np.array([v[2] for v in values])

        # the spline QuadSource._create_functions builds
        x = np.log(1.0 + z_nodes[::-1])
        spline = make_interp_spline(x, src[::-1])

        envelope = np.max(np.abs(oracle_nodes))
        node_err = np.max(np.abs(src - oracle_nodes)) / envelope

        # midpoints of the lowest 25% of the region (nearest the hand-over)
        x_sorted = np.sort(x)
        n_low = max(5, len(x_sorted) // 4)
        mids = 0.5 * (x_sorted[:n_low] + x_sorted[1 : n_low + 1])
        errs = []
        for xm in mids:
            z = float(np.expm1(xm))
            tau = functions.tau(z)
            H = functions.Hubble(z)
            wb = functions.wBackground(z)
            Tq = compute_analytic_T(q_val, W_RAD, tau)
            Tr = compute_analytic_T(r_val, W_RAD, tau)
            Tq_prime = compute_analytic_Tprime(q_val, W_RAD, tau, H)
            Tr_prime = compute_analytic_Tprime(r_val, W_RAD, tau, H)
            oracle = source_function(Tq, Tr, Tq_prime, Tr_prime, z, wb)["source"]
            errs.append(abs(float(spline(xm)) - oracle))
        mid_err = max(errs) / envelope
        worst_all = max(worst_all, mid_err)
        print(
            f"   {q_val:>10.4g} {r_val:>10.4g}  {len(values):>5}  "
            f"[{z_nodes[0]:.4g}, {z_nodes[-1]:.4g}]   {node_err:>18.3e}   {mid_err:>18.3e}"
        )
    print(f"   worst midpoint residual over the sampled pairs: {worst_all:.3e}")
    print(
        "   (q, r are in 1/Mpc; 'env' is the largest |analytic_source_rad| on the row; the "
        "midpoints are the lowest quarter of the region, i.e. nearest the hand-over)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--shards-glob", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--model-label", default="LambdaCDM")
    parser.add_argument("--max-gk-cases", type=int, default=40)
    parser.add_argument("--max-quadsource-pairs", type=int, default=10)
    args = parser.parse_args()

    sql_sections(args.shards_glob)
    tk_wkb_against_oracle(args.shards_glob)

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
        job_name="source-remediation-verification-analysis",
        prune_unvalidated=False,
        drop_actions=[],
        read_table_config=read_table_config,
        inventory_config=inventory_config,
    ) as pool:
        atol, rtol = ray.get(
            [
                pool.object_get(tolerance, tol=DEFAULT_ABS_TOLERANCE),
                pool.object_get(tolerance, tol=DEFAULT_REL_TOLERANCE),
            ]
        )
        models = build_model_list(pool, units)
        model_data = [m for m in models if m["label"] == args.model_label]
        if not model_data:
            raise RuntimeError(f"no model labelled {args.model_label}")
        cosmology = model_data[0]["cosmology"]
        background: BackgroundModel = ray.get(
            pool.object_get(
                BackgroundModel,
                solver_labels=[],
                cosmology=cosmology,
                z_sample=None,
                atol=atol,
                rtol=rtol,
            )
        )
        if not background.available:
            raise RuntimeError("BackgroundModel not found in the datastore")
        model_proxy = ModelProxy(background)

        gk_continuity(
            pool,
            model_proxy,
            cosmology,
            atol,
            rtol,
            args.shards_glob,
            args.max_gk_cases,
        )
        quadsource_residual(
            pool, model_proxy, args.shards_glob, args.max_quadsource_pairs
        )


if __name__ == "__main__":
    main()
