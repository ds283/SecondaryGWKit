"""Read the QuadSourceIntegral rows of a datastore produced by scoped_pipeline_run.py and report
what prompt 12 of prompts/source-remediation asks for in its section 2:

  * `total` against the fixed-w oracle `analytic_rad`, and the ratio of that residual to the
    stored quadrature bound `total_abserr` (item 1);
  * the regime mix of the partition recorded in `metadata["partition"]`, against the six rows of
    the campaign README's phase-group table (item 2);
  * the hand-over clamp gaps that board issue [08-handover-clamp-error] predicts, and the
    `skipped` records that prompt 09 added (items 1 and 2, and the board's standing notes);
  * convergence and phase-limited flags (board [09-abserr-is-a-quadrature-bound]);
  * cost per row, split by regime, and the aggregated Levin counters (item 8).

Read-only: the datastore is opened through sqlite3 in `mode=ro` and nothing is written.
"""

import argparse
import glob
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict

# the README's phase-group table, keyed by the (G_osc, q_osc, r_osc) regime tuple
REGIME_ROW = {
    (False, False, False): "none (ordinary quadrature)",
    (True, False, False): "G only",
    (False, True, False): "T_q only",
    (False, False, True): "T_r only",
    (True, True, False): "G and T_q",
    (True, False, True): "G and T_r",
    (False, True, True): "T_q and T_r, G smooth",
    (True, True, True): "all three",
}
N_GROUPS = {0: 0, 1: 1, 2: 2, 3: 4}


def regime_row(regime):
    return REGIME_ROW[tuple(bool(x) for x in regime)]


def load_rows(pattern):
    rows = []
    for db in sorted(glob.glob(pattern)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        wavenumbers = {
            r[0]: r[1] for r in conn.execute("select serial, k_inv_Mpc from wavenumber")
        }
        exit_times = {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                "select serial, wavenumber_serial, z_exit from wavenumber_exit_time"
            )
        }
        redshifts = {r[0]: r[1] for r in conn.execute("select serial, z from redshift")}

        for r in conn.execute(
            "select k_wavenumber_exit_serial, q_wavenumber_exit_serial, "
            "r_wavenumber_exit_serial, z_response_serial, b, total, total_abserr, "
            "total_converged, total_phase_limited, numeric_quad, WKB_quad, WKB_Levin, "
            "analytic_rad, compute_time, analytic_compute_time, WKB_Levin_num_regions, "
            "WKB_Levin_evaluations, WKB_Levin_simple_regions, WKB_Levin_SVD_errors, "
            "WKB_Levin_elapsed, WKB_phase_spline_chunks, metadata "
            "from QuadSourceIntegral"
        ):
            rows.append(
                {
                    "k": wavenumbers[exit_times[r[0]][0]],
                    "q": wavenumbers[exit_times[r[1]][0]],
                    "r": wavenumbers[exit_times[r[2]][0]],
                    "k_z_exit": exit_times[r[0]][1],
                    "z_response": redshifts[r[3]],
                    "b": r[4],
                    "total": r[5],
                    "total_abserr": r[6],
                    "converged": r[7],
                    "phase_limited": r[8],
                    "numeric_quad": r[9],
                    "WKB_quad": r[10],
                    "WKB_Levin": r[11],
                    "analytic_rad": r[12],
                    "compute_time": r[13],
                    "analytic_compute_time": r[14],
                    "Levin_regions": r[15],
                    "Levin_evaluations": r[16],
                    "Levin_simple_regions": r[17],
                    "Levin_SVD_errors": r[18],
                    "Levin_elapsed": r[19],
                    "phase_spline_chunks": r[20],
                    "metadata_raw": r[21],
                    "metadata": json.loads(r[21]) if r[21] else {},
                }
            )
        conn.close()
    return rows


def load_jsonl(path):
    """Rows written by run_quadsource_integrals.py, which schedules the same work items
    main.py's stage would and records the failures instead of aborting."""
    rows = []
    failed = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("status") != "ok":
                failed.append(record)
                continue
            record["metadata_raw"] = json.dumps(record["metadata"])
            record["converged"] = record["total_converged"]
            record["phase_limited"] = record["total_phase_limited"]
            rows.append(record)
    return rows, failed


def shape_of(row):
    """Classify a triple into the three shapes prompt 08 tested."""
    k, q, r = row["k"], row["q"], row["r"]
    lo, hi = min(q, r), max(q, r)
    if hi / lo >= 3.0 and abs(math.log10(max(k, hi) / min(k, hi))) < 0.4:
        return "q << k ~ r"
    if lo / k >= 3.0 and hi / lo < 3.0:
        return "q ~ r >> k"
    if max(k, q, r) / min(k, q, r) < 3.0:
        return "q ~ r ~ k"
    return "other"


def quantiles(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * n))]

    return {
        "min": s[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": s[-1],
    }


def fmt_q(qs, fmt="{:.3e}"):
    if qs is None:
        return "n/a"
    return "  ".join(f"{k}={fmt.format(v)}" for k, v in qs.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", help="glob for the shard sqlite files")
    parser.add_argument("--jsonl", help="output of run_quadsource_integrals.py")
    parser.add_argument(
        "--radiation-z",
        type=float,
        default=None,
        help="only compare against analytic_rad for z_response above this",
    )
    args = parser.parse_args()

    if args.jsonl:
        rows, failed = load_jsonl(args.jsonl)
        print(f"# work items recorded: {len(rows) + len(failed)}")
        print(f"#   completed: {len(rows)}, failed: {len(failed)}")
        reasons = Counter(
            f.get("error", "?").split(" is out-of-bounds")[0][-90:] for f in failed
        )
        for reason, n in reasons.most_common(5):
            print(f"#     {n:>6}  {reason}")
        if failed:
            zs = sorted(set(f["z_response"] for f in failed))
            print(
                f"#   distinct z_response among failures: {len(zs)} "
                f"({min(zs):.5g} ... {max(zs):.5g})"
            )
    else:
        rows = load_rows(args.shards)
    print(f"# QuadSourceIntegral rows: {len(rows)}")
    if len(rows) == 0:
        return

    print(f"  b values stored: {sorted(set(r['b'] for r in rows))}")
    print(
        f"  distinct (k,q,r): {len(set((r['k'], r['q'], r['r']) for r in rows))}, "
        f"distinct z_response: {len(set(r['z_response'] for r in rows))}"
    )
    print(
        f"  z_response range: {min(r['z_response'] for r in rows):.5g} "
        f"... {max(r['z_response'] for r in rows):.5g}"
    )
    print(
        f"  metadata JSON length: max {max(len(r['metadata_raw']) for r in rows)} chars, "
        f"median {sorted(len(r['metadata_raw']) for r in rows)[len(rows)//2]}"
    )

    # ---------------------------------------------------------------- item 1
    print("\n## 1. total vs analytic_rad")
    sel = [
        r
        for r in rows
        if r["analytic_rad"] not in (None, 0.0)
        and (args.radiation_z is None or r["z_response"] >= args.radiation_z)
    ]
    print(f"   rows compared: {len(sel)} of {len(rows)}")
    rel = [abs(r["total"] - r["analytic_rad"]) / abs(r["analytic_rad"]) for r in sel]
    print(f"   |total-analytic|/|analytic|      {fmt_q(quantiles(rel))}")
    # normalised by the scale prompt 08 used, which is insensitive to cancellation
    scaled = [
        abs(r["total"] - r["analytic_rad"])
        / max(abs(r["numeric_quad"]), abs(r["WKB_Levin"]), abs(r["analytic_rad"]))
        for r in sel
    ]
    print(f"   same, / max(|nq|,|Levin|,|an|)   {fmt_q(quantiles(scaled))}")
    ratio = [
        abs(r["total"] - r["analytic_rad"]) / r["total_abserr"]
        for r in sel
        if r["total_abserr"]
    ]
    print(f"   |total-analytic|/total_abserr    {fmt_q(quantiles(ratio))}")
    ab = [abs(r["total_abserr"] / r["total"]) for r in sel if r["total"]]
    print(f"   total_abserr/|total|             {fmt_q(quantiles(ab))}")
    canc = [
        max(abs(r["numeric_quad"]), abs(r["WKB_Levin"])) / abs(r["total"])
        for r in sel
        if r["total"]
    ]
    print(f"   cancellation max(|part|)/|total| {fmt_q(quantiles(canc), '{:.3g}')}")

    print("\n   by shape:")
    by_shape = defaultdict(list)
    for r, x, y in zip(sel, rel, scaled):
        by_shape[shape_of(r)].append((x, y))
    for shape, vals in sorted(by_shape.items()):
        print(
            f"     {shape:<14} n={len(vals):<5} "
            f"rel median={sorted(v[0] for v in vals)[len(vals)//2]:.3e} "
            f"max={max(v[0] for v in vals):.3e} | "
            f"scaled median={sorted(v[1] for v in vals)[len(vals)//2]:.3e} "
            f"max={max(v[1] for v in vals):.3e}"
        )

    print("\n   worst 8 rows by |total-analytic|/scale:")
    order = sorted(range(len(sel)), key=lambda i: -scaled[i])[:8]
    for i in order:
        r = sel[i]
        print(
            f"     k={r['k']:.4g} q={r['q']:.4g} r={r['r']:.4g} z={r['z_response']:.5g} "
            f"[{shape_of(r):<12}] total={r['total']:+.6e} analytic={r['analytic_rad']:+.6e} "
            f"rel={rel[i]:.3e} scaled={scaled[i]:.3e} "
            f"gap={r['metadata'].get('partition', {}).get('max_clamp_gap_log1pz', float('nan')):.3e}"
        )

    # ---------------------------------------------------------------- item 2
    print("\n## 2. regime mix (sub-intervals over all rows)")
    regime_counter = Counter()
    method_counter = Counter()
    rows_with_regime = defaultdict(set)
    for r in rows:
        for sub in r["metadata"].get("partition", {}).get("subintervals", []):
            row_label = regime_row(sub["regime"])
            regime_counter[row_label] += 1
            method_counter[(row_label, sub["method"])] += 1
            rows_with_regime[row_label].add(shape_of(r))
    total_subs = sum(regime_counter.values())
    print(f"   sub-intervals: {total_subs} over {len(rows)} rows")
    for label in REGIME_ROW.values():
        n = regime_counter.get(label, 0)
        if n == 0:
            print(f"     {label:<28} 0")
        else:
            methods = {m: c for (lbl, m), c in method_counter.items() if lbl == label}
            print(
                f"     {label:<28} {n:<7} ({100.0*n/total_subs:5.2f}%)  methods={methods}  "
                f"shapes={sorted(rows_with_regime[label])}"
            )
    per_row = Counter(
        len(r["metadata"].get("partition", {}).get("subintervals", [])) for r in rows
    )
    print(f"   sub-intervals per row: {dict(sorted(per_row.items()))}")
    gtype = Counter(r["metadata"].get("partition", {}).get("G_type") for r in rows)
    print(f"   GkSourcePolicyData type of the Green's function: {dict(gtype)}")

    # ---------------------------------------------------------------- clamp
    print("\n## 3. hand-over clamp gaps and skipped breakpoints")
    gaps = [
        r["metadata"].get("partition", {}).get("max_clamp_gap_log1pz", 0.0)
        for r in rows
    ]
    print(f"   max_clamp_gap_log1pz             {fmt_q(quantiles(gaps))}")
    per_factor = defaultdict(list)
    for r in rows:
        for sub in r["metadata"].get("partition", {}).get("subintervals", []):
            for factor, gap in sub.get("clamp_gaps_log1pz", {}).items():
                if gap > 0.0:
                    per_factor[factor].append(gap)
    for factor, vals in sorted(per_factor.items()):
        print(f"     {factor:<8} n={len(vals):<6} {fmt_q(quantiles(vals))}")
    if not per_factor:
        print("     no sub-interval carried a non-zero clamp gap")
    skipped = [
        s for r in rows for s in r["metadata"].get("partition", {}).get("skipped", [])
    ]
    print(f"   skipped breakpoints (prompt 09): {len(skipped)}")
    for s in skipped[:5]:
        print(f"     {s}")

    # ------------------------------------------------------------ diagnostics
    print("\n## 4. convergence flags and error bound")
    print(
        f"   total_converged False: {sum(1 for r in rows if not r['converged'])} of {len(rows)}"
    )
    print(
        f"   total_phase_limited True: {sum(1 for r in rows if r['phase_limited'])} of {len(rows)}"
    )
    print(
        f"   WKB_quad non-zero: {sum(1 for r in rows if r['WKB_quad'])} of {len(rows)}"
    )

    # ------------------------------------------------------------------ cost
    print("\n## 5. cost")
    ct = [r["compute_time"] for r in rows if r["compute_time"]]
    at = [r["analytic_compute_time"] for r in rows if r["analytic_compute_time"]]
    print(f"   compute_time (s)                 {fmt_q(quantiles(ct), '{:.3g}')}")
    print(f"   analytic_compute_time (s)        {fmt_q(quantiles(at), '{:.3g}')}")
    print(f"   sum compute_time = {sum(ct):.1f} s, sum analytic = {sum(at):.1f} s")
    lev = [r["Levin_elapsed"] for r in rows if r["Levin_elapsed"]]
    print(f"   WKB_Levin_elapsed (s)            {fmt_q(quantiles(lev), '{:.3g}')}")
    print(
        f"   Levin regions   {fmt_q(quantiles([r['Levin_regions'] for r in rows if r['Levin_regions'] is not None]), '{:.0f}')}"
    )
    print(
        f"   Levin evaluations {fmt_q(quantiles([r['Levin_evaluations'] for r in rows if r['Levin_evaluations'] is not None]), '{:.0f}')}"
    )
    print(
        f"   Levin simple (Clenshaw-Curtis) regions {fmt_q(quantiles([r['Levin_simple_regions'] for r in rows if r['Levin_simple_regions'] is not None]), '{:.0f}')}"
    )
    print(
        f"   Levin SVD errors, total: {sum(r['Levin_SVD_errors'] or 0 for r in rows)}"
    )
    print(
        "\n   cost by number of oscillatory factors in the row's richest sub-interval:"
    )
    by_n = defaultdict(list)
    for r in rows:
        subs = r["metadata"].get("partition", {}).get("subintervals", [])
        if not subs or not r["compute_time"]:
            continue
        n = max(sum(1 for x in s["regime"] if x) for s in subs)
        by_n[n].append(r["compute_time"])
    for n, vals in sorted(by_n.items()):
        print(
            f"     {n} oscillatory factors ({N_GROUPS[n]} phase groups): n={len(vals):<5} "
            f"median={sorted(vals)[len(vals)//2]:.3g} s  max={max(vals):.3g} s"
        )


if __name__ == "__main__":
    main()
