"""Prompt 06b §4.3: QuadSourceIntegral, R15 (parent da5c24d) against R16 (code 4ddc3c2), per row.

Reads every store file with sqlite `mode=ro`, and store_value_diff's JSON (for a cross-check
only). Rows are paired by their physical (k, q, r, z_response): k, q and r are each
`wavenumber.k_inv_Mpc` through `wavenumber_exit_time`, z_response is `redshift.z`. The pairing is
checked against the tool's identity-matched report: the same key set, and the same `float.hex` of
`total` wherever the tool reports `total` as differing.

**Where the phase-limited flags are stored** (ComputeTargets/QuadSourceIntegral.py,
ObjectFactories/QuadSourceIntegral.py), in the shard table `QuadSourceIntegral`:
  (a) the column `total_phase_limited` (SQLAlchemy Boolean, SQLite integer 0/1, nullable): the OR
      of every phase group's flag of every Levin sub-interval (evaluate_QuadSource_integral);
  (b) the column `metadata` (a JSON string, json.dumps), at
      `WKB_Levin.subintervals[i].groups[j].phase_limited` (JSON true/false): each phase group's
      adaptive_levin_sincos flag (phase_group_Levin_integral). `WKB_Levin.phase_limited` and
      `WKB_Levin.subintervals[i].phase_limited` are ORs of these, and are checked to be;
  (c) the same `metadata` column, at
      `analytic.{0pt5,2pt5}.{J,Y}{1..4}_data.phase_limited` when the three-Bessel integral is
      Levin only, or `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.phase_limited` when it is split
      between quad and Levin (_three_bessel_integrals, _three_bessel_Levin). Their ORs are not
      stored anywhere.
"Phase-limited in R15" is: (a) true, or any (b) true, or any (c) true, in R15's row. Every other
metadata leaf named `phase_limited` is also collected, and must be one of the above or an OR of
(b).

**Field kinds.** Every column and every metadata leaf is put in one kind:
  - timing: a wall clock. Columns ending `compute_time`, `RHS_time` or `elapsed`; metadata leaves
    named `elapsed` or `compute_time`. Not compared.
  - label: the `label` column (it carries main.py's datetime.now()). Not compared.
  - identity: store bookkeeping, not a computed value: the column `serial`, every column ending
    `_serial` (a row's own and its parents' serials, which each store assigns for itself; rows
    are paired by physical key instead), and `timestamp` (the write time). Not compared.
    (Added after attempt 1, which counted these as values; see the log.)
  - counter: a work counter, a count of what the integrator did rather than a computed value.
    Columns `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
    `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
    `WKB_Levin_max_depth`, `numeric_quad_compute_steps`, `numeric_quad_RHS_evaluations`; metadata
    leaves named `regions`, `simple_regions` or `evaluations`. Compared and reported, but a
    difference in one is not a value difference.
  - value: everything else (total, total_abserr, total_converged, total_phase_limited,
    numeric_quad, WKB_Levin, analytic_rad, b, eta_*, every other metadata leaf, including each
    group record's label, value, abserr, converged and phase_limited, and each analytic block's
    abserr, converged and phase_limited).
Bit-identity is by `float.hex` for floats and by == for bool/int/str/None, with the type also
required equal. Never isclose.

Prompt 06b §4.3's rule: a row with no phase-limited flag in R15 is bit-identical in `total`,
`total_abserr`, the analytic columns (`analytic_rad`, and the `analytic` metadata) and every group
record (`WKB_Levin.subintervals[i].groups[j]`). The script checks that set (the "named" set), then
every value field of such rows, then their counters, and tabulates every other row.

    python -I tabulate_qsi_phase_limited.py <R15 primary> <R16 primary> <store_value_diff.json> <out.csv>
"""

import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

TIMING_COLS = re.compile(r"(compute_time|RHS_time|elapsed)$")
COUNTER_COLS = {
    "WKB_Levin_num_regions",
    "WKB_Levin_evaluations",
    "WKB_Levin_simple_regions",
    "WKB_Levin_SVD_errors",
    "WKB_Levin_order_changes",
    "WKB_Levin_chebyshev_min_order",
    "WKB_Levin_max_depth",
    "numeric_quad_compute_steps",
    "numeric_quad_RHS_evaluations",
}
TIMING_LEAVES = {"elapsed", "compute_time"}
COUNTER_LEAVES = {"regions", "simple_regions", "evaluations"}

GROUP_PL = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.phase_limited$")
ANALYTIC_PL = re.compile(r"^analytic\.(0pt5|2pt5)(\.Levin)?\.[JY][1-4]_data\.phase_limited$")
AGG_PL = re.compile(r"^WKB_Levin(\.subintervals\[\d+\])?\.phase_limited$")
GROUP_REC = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.")
GROUP_REGIONS = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.regions$")
ANALYTIC_REGIONS = re.compile(r"^analytic\.(0pt5|2pt5)(\.Levin)?\.[JY][1-4]_data\.regions$")

KEY_COLS = ("k", "q", "r", "z_response")


def read(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    cols = None
    pl_types = Counter()
    for name in shards:
        con = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        if cols is None:
            cols = [r[1] for r in con.execute("PRAGMA table_info(QuadSourceIntegral)")]
        kq = (
            "(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
            "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})"
        )
        sql = (
            f"SELECT {kq.format('k_wavenumber_exit_serial')}, "
            f"{kq.format('q_wavenumber_exit_serial')}, {kq.format('r_wavenumber_exit_serial')}, "
            "(SELECT z FROM redshift WHERE serial = q.z_response_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.atol_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial), "
            "typeof(q.total_phase_limited), "
            + ", ".join(f'q."{c}"' for c in cols)
            + " FROM QuadSourceIntegral q"
        )
        for row in con.execute(sql):
            key = row[:4]
            assert key not in out, f"duplicate {key}"
            pl_types[row[6]] += 1
            rec = dict(zip(cols, row[7:]))
            rec["_atol"], rec["_rtol"] = row[4], row[5]
            out[key] = rec
        con.close()
    return out, cols, pl_types


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        if not obj:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def bits(x):
    if isinstance(x, float):
        return ("float", x.hex())
    if isinstance(x, list):
        return ("list", tuple(x))
    return (type(x).__name__, x)


def col_kind(c):
    if c == "label":
        return "label"
    if c == "serial" or c.endswith("_serial") or c == "timestamp":
        return "identity"
    if TIMING_COLS.search(c):
        return "timing"
    if c in COUNTER_COLS:
        return "counter"
    return "value"


def leaf_kind(path):
    leaf = re.split(r"[.\]]", path)[-1] if not path.endswith("]") else path
    if leaf in TIMING_LEAVES:
        return "timing"
    if leaf in COUNTER_LEAVES:
        return "counter"
    return "value"


def named(field):
    """Prompt §4.3's named set: total, total_abserr, the analytic columns, every group record."""
    if field in ("col:total", "col:total_abserr", "col:analytic_rad"):
        return True
    if field.startswith("meta:analytic."):
        return True
    if field.startswith("meta:") and GROUP_REC.match(field[5:]):
        return True
    return False


def fields(rec):
    out = {}
    for c, v in rec.items():
        if c.startswith("_") or c == "metadata":
            continue
        out[f"col:{c}"] = (col_kind(c), v)
    meta = json.loads(rec["metadata"]) if rec["metadata"] is not None else {}
    for p, v in flatten(meta).items():
        out[f"meta:{p}"] = (leaf_kind(p), v)
    return out, meta


def pl_info(rec, flat):
    total_pl = rec["total_phase_limited"]
    groups = {p: v for p, v in flat.items() if GROUP_PL.match(p)}
    analytic = {p: v for p, v in flat.items() if ANALYTIC_PL.match(p)}
    agg = {p: v for p, v in flat.items() if AGG_PL.match(p)}
    other = {
        p: v
        for p, v in flat.items()
        if p.split(".")[-1] == "phase_limited" and p not in groups and p not in analytic
        and p not in agg
    }
    for v in list(groups.values()) + list(analytic.values()) + list(agg.values()):
        assert isinstance(v, bool), (v, type(v))
    # the aggregates are ORs of the group flags
    gor = any(groups.values())
    assert bool(total_pl) == gor, ("total_phase_limited is not the OR of the groups", total_pl, gor)
    if "WKB_Levin.phase_limited" in agg:
        assert agg["WKB_Levin.phase_limited"] == gor
    for p, v in agg.items():
        m = re.match(r"^WKB_Levin\.subintervals\[(\d+)\]\.phase_limited$", p)
        if m:
            sub = [
                g for q, g in groups.items() if q.startswith(f"WKB_Levin.subintervals[{m[1]}].")
            ]
            assert v == any(sub), p
    return {
        "total_pl": total_pl,
        "groups_pl": sum(groups.values()),
        "groups_n": len(groups),
        "analytic_pl": sum(analytic.values()),
        "analytic_n": len(analytic),
        "analytic_pl_paths": sorted(p for p, v in analytic.items() if v),
        "groups_pl_paths": sorted(p for p, v in groups.items() if v),
        "other": other,
        "any": bool(total_pl) or any(groups.values()) or any(analytic.values()),
    }


def hx(x):
    return None if x is None else float(x).hex()


def main():
    a_path, b_path, report_path, out_csv = sys.argv[1:5]
    A, cols_a, types_a = read(a_path)
    B, cols_b, types_b = read(b_path)
    print(f"rows: R15 {len(A)}, R16 {len(B)}; same (k,q,r,z_response) set: {set(A) == set(B)}")
    print(f"columns equal: {cols_a == cols_b}; {len(cols_a)} columns")
    print(f"typeof(total_phase_limited): R15 {dict(types_a)}; R16 {dict(types_b)}")
    print(
        f"tolerances R15: {sorted(set((v['_atol'], v['_rtol']) for v in A.values()))}; "
        f"R16: {sorted(set((v['_atol'], v['_rtol']) for v in B.values()))}"
    )
    print(
        "column kinds: "
        + "; ".join(
            f"{k}: {sorted(c for c in cols_a if col_kind(c) == k and c != 'metadata')}"
            for k in ("timing", "label", "identity", "counter")
        )
    )

    rep = json.load(open(report_path))["classes"]["QuadSourceIntegral"]["rows"]
    tool = {}
    for e in rep["rows"]:
        k = e["key"]
        key = (
            k["k"]["k"]["k_inv_Mpc"],
            k["q"]["k"]["k_inv_Mpc"],
            k["r"]["k"]["k_inv_Mpc"],
            k["z_response"]["z"],
        )
        assert key not in tool
        tool[key] = e["columns"]
    print(
        f"tool: {len(tool)} rows differ, {rep['identical']} identical; "
        f"its differing keys within the stores' set: {set(tool) <= set(A)}"
    )
    mism = 0
    for key in A:
        t = tool.get(key, {}).get("total")
        if t is None:
            mism += hx(A[key]["total"]) != hx(B[key]["total"])
        else:
            mism += (t["a"], t["b"]) != (hx(A[key]["total"]), hx(B[key]["total"]))
    print(f"cross-check of `total` against the tool: {mism} mismatches")
    print()

    rows = []
    viol_named, viol_value, viol_counter = Counter(), Counter(), Counter()
    viol_rows_named, viol_rows_value, viol_rows_counter = [], [], []
    other_pl = Counter()
    field_sets_differ = 0
    for key in sorted(A):
        a, b = A[key], B[key]
        fa, ma = fields(a)
        fb, mb = fields(b)
        flat_a = {p[5:]: v for p, (_, v) in fa.items() if p.startswith("meta:")}
        flat_b = {p[5:]: v for p, (_, v) in fb.items() if p.startswith("meta:")}
        ia, ib = pl_info(a, flat_a), pl_info(b, flat_b)
        for p in ia["other"]:
            other_pl[p] += 1
        diffs = {}
        for f in set(fa) | set(fb):
            kind = (fa.get(f) or fb.get(f))[0]
            if f not in fa or f not in fb:
                diffs[f] = (kind, "missing")
                continue
            if kind in ("timing", "label", "identity"):
                continue
            if bits(fa[f][1]) != bits(fb[f][1]):
                diffs[f] = (kind, "differs")
        if set(fa) != set(fb):
            field_sets_differ += 1
        d_named = sorted(f for f in diffs if named(f) and diffs[f][0] != "counter")
        d_value = sorted(f for f, (kd, _) in diffs.items() if kd == "value")
        d_counter = sorted(f for f, (kd, _) in diffs.items() if kd == "counter")
        if not ia["any"]:
            if d_named:
                viol_rows_named.append((key, d_named))
                viol_named.update(d_named)
            if d_value:
                viol_rows_value.append((key, d_value))
                viol_value.update(d_value)
            if d_counter:
                viol_rows_counter.append((key, d_counter))
                viol_counter.update(d_counter)
        dt = b["total"] - a["total"]
        g_reg_a = sum(v for p, v in flat_a.items() if GROUP_REGIONS.match(p))
        g_reg_b = sum(v for p, v in flat_b.items() if GROUP_REGIONS.match(p))
        an_reg_a = sum(v for p, v in flat_a.items() if ANALYTIC_REGIONS.match(p))
        an_reg_b = sum(v for p, v in flat_b.items() if ANALYTIC_REGIONS.match(p))
        an_err_a = ma.get("analytic", {}).get("abserr")
        an_err_b = mb.get("analytic", {}).get("abserr")
        rows.append(
            {
                "k": key[0],
                "q": key[1],
                "r": key[2],
                "z_response": key[3],
                "pl_R15": ia["any"],
                "pl_R16": ib["any"],
                "total_pl_R15": a["total_phase_limited"],
                "total_pl_R16": b["total_phase_limited"],
                "group_pl_R15": ia["groups_pl"],
                "group_pl_R16": ib["groups_pl"],
                "groups_n": ia["groups_n"],
                "analytic_pl_R15": ia["analytic_pl"],
                "analytic_pl_R16": ib["analytic_pl"],
                "analytic_blocks_n": ia["analytic_n"],
                "analytic_pl_paths_R15": ";".join(ia["analytic_pl_paths"]),
                "group_pl_paths_R15": ";".join(ia["groups_pl_paths"]),
                "total_R15": hx(a["total"]),
                "total_R16": hx(b["total"]),
                "total_bit_identical": hx(a["total"]) == hx(b["total"]),
                "abs_dtotal": abs(dt),
                "rel_dtotal": abs(dt) / abs(a["total"]) if a["total"] else math.inf,
                "abserr_R15": a["total_abserr"],
                "abserr_R16": b["total_abserr"],
                "abserr_ratio_R16_R15": (
                    b["total_abserr"] / a["total_abserr"] if a["total_abserr"] else None
                ),
                "dtotal_over_abserr_R15": abs(dt) / a["total_abserr"] if a["total_abserr"] else None,
                "dtotal_over_abserr_R16": abs(dt) / b["total_abserr"] if b["total_abserr"] else None,
                "converged_R15": a["total_converged"],
                "converged_R16": b["total_converged"],
                "WKB_Levin_num_regions_R15": a["WKB_Levin_num_regions"],
                "WKB_Levin_num_regions_R16": b["WKB_Levin_num_regions"],
                "WKB_Levin_max_depth_R15": a["WKB_Levin_max_depth"],
                "WKB_Levin_max_depth_R16": b["WKB_Levin_max_depth"],
                "WKB_Levin_evaluations_R15": a["WKB_Levin_evaluations"],
                "WKB_Levin_evaluations_R16": b["WKB_Levin_evaluations"],
                "group_regions_R15": g_reg_a,
                "group_regions_R16": g_reg_b,
                "analytic_regions_R15": an_reg_a,
                "analytic_regions_R16": an_reg_b,
                "analytic_rad_R15": hx(a["analytic_rad"]),
                "analytic_rad_R16": hx(b["analytic_rad"]),
                "analytic_rad_bit_identical": hx(a["analytic_rad"]) == hx(b["analytic_rad"]),
                "analytic_abserr_R15": an_err_a,
                "analytic_abserr_R16": an_err_b,
                "d_analytic_rad_over_analytic_abserr_R15": (
                    abs(b["analytic_rad"] - a["analytic_rad"]) / an_err_a
                    if an_err_a
                    else None
                ),
                "value_fields_differing": len(d_value),
                "counter_fields_differing": len(d_counter),
            }
        )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    pl = [r for r in rows if r["pl_R15"]]
    npl = [r for r in rows if not r["pl_R15"]]
    print(f"rows whose metadata field sets differ between R15 and R16: {field_sets_differ}")
    print(f"other metadata leaves named phase_limited: {dict(other_pl) or 'none'}")
    print()
    print("== phase-limited in R15 (any of (a) total_phase_limited, (b) a group's flag, (c) an analytic block's flag)")
    print(f"rows: {len(pl)} of {n}; not phase-limited: {len(npl)}")
    print(
        f"  by source: (a) total_phase_limited true {sum(1 for r in rows if r['total_pl_R15'])}; "
        f"(b) any group flag {sum(1 for r in rows if r['group_pl_R15'])}; "
        f"(c) any analytic-block flag {sum(1 for r in rows if r['analytic_pl_R15'])}; "
        f"(a or b) only {sum(1 for r in pl if r['group_pl_R15'] and not r['analytic_pl_R15'])}, "
        f"(c) only {sum(1 for r in pl if r['analytic_pl_R15'] and not r['group_pl_R15'])}, "
        f"both {sum(1 for r in pl if r['analytic_pl_R15'] and r['group_pl_R15'])}"
    )
    print(
        f"  phase-limited in R16: {sum(1 for r in rows if r['pl_R16'])} rows; "
        f"(a) {sum(1 for r in rows if r['total_pl_R16'])}, (b) {sum(1 for r in rows if r['group_pl_R16'])}, "
        f"(c) {sum(1 for r in rows if r['analytic_pl_R16'])}; "
        f"rows phase-limited in R16 but not in R15: {sum(1 for r in rows if r['pl_R16'] and not r['pl_R15'])}"
    )
    print(
        f"  flags: group flags set R15 {sum(r['group_pl_R15'] for r in rows)}, R16 "
        f"{sum(r['group_pl_R16'] for r in rows)} (of {sum(r['groups_n'] for r in rows)} group records); "
        f"analytic-block flags set R15 {sum(r['analytic_pl_R15'] for r in rows)}, R16 "
        f"{sum(r['analytic_pl_R16'] for r in rows)} (of {sum(r['analytic_blocks_n'] for r in rows)} blocks)"
    )
    print()
    print("== §4.3 rule: rows with no phase-limited flag in R15")
    print(
        f"  named set (total, total_abserr, analytic_rad, analytic metadata, every group record; "
        f"timings, labels and counters excluded): rows not bit-identical {len(viol_rows_named)}"
        + (f"; fields {dict(viol_named)}" if viol_named else "")
    )
    print(
        f"  every value field of the row (columns and metadata): rows not bit-identical "
        f"{len(viol_rows_value)}" + (f"; fields {dict(viol_value)}" if viol_value else "")
    )
    print(
        f"  work counters: rows differing {len(viol_rows_counter)}"
        + (f"; fields {dict(viol_counter)}" if viol_counter else "")
    )
    for key, d in viol_rows_named[:20]:
        print(f"    NAMED VIOLATION {key}: {d}")
    for key, d in viol_rows_value[:20]:
        print(f"    VALUE VIOLATION {key}: {d}")
    for key, d in viol_rows_counter[:20]:
        print(f"    counter difference {key}: {d}")
    print(
        f"  check: in these rows total bit-identical {sum(r['total_bit_identical'] for r in npl)}/{len(npl)}, "
        f"analytic_rad bit-identical {sum(r['analytic_rad_bit_identical'] for r in npl)}/{len(npl)}"
    )
    print()
    print("== rows phase-limited in R15, tabulated")
    print(
        f"  total bit-identical {sum(r['total_bit_identical'] for r in pl)}/{len(pl)}; "
        f"analytic_rad bit-identical {sum(r['analytic_rad_bit_identical'] for r in pl)}/{len(pl)}; "
        f"rows with every value field identical {sum(1 for r in pl if r['value_fields_differing'] == 0)}"
    )

    def g(x, f="{:.3e}"):
        return "-" if x is None else f.format(x)

    hdr = (
        "k        q        r        z_resp       | PL15 a/b/c  | PL16 a/b/c  | conv15>16 | "
        "abserr15   abserr16   16/15     | |dT|/err15 |dT|/err16 | rel dT    | "
        "Levin reg 15>16 | depth 15>16 | grp reg 15>16 | an reg 15>16 | an_rad id | d an/an_err15"
    )
    print("  " + hdr)
    for r in sorted(pl, key=lambda r: (r["k"], r["q"], r["r"], r["z_response"])):
        print(
            f"  {r['k']:<8.3g} {r['q']:<8.3g} {r['r']:<8.3g} {r['z_response']:<12.6g} | "
            f"{int(bool(r['total_pl_R15']))}/{r['group_pl_R15']}/{r['analytic_pl_R15']:<6} | "
            f"{int(bool(r['total_pl_R16']))}/{r['group_pl_R16']}/{r['analytic_pl_R16']:<6} | "
            f"{int(bool(r['converged_R15']))}>{int(bool(r['converged_R16']))}       | "
            f"{g(r['abserr_R15'])}  {g(r['abserr_R16'])}  {g(r['abserr_ratio_R16_R15'], '{:.3g}'):<9} | "
            f"{g(r['dtotal_over_abserr_R15'])}  {g(r['dtotal_over_abserr_R16'])}  {g(r['rel_dtotal'])} | "
            f"{r['WKB_Levin_num_regions_R15']}>{r['WKB_Levin_num_regions_R16']:<12} | "
            f"{r['WKB_Levin_max_depth_R15']}>{r['WKB_Levin_max_depth_R16']:<9} | "
            f"{r['group_regions_R15']}>{r['group_regions_R16']:<10} | "
            f"{r['analytic_regions_R15']}>{r['analytic_regions_R16']:<9} | "
            f"{str(r['analytic_rad_bit_identical']):<9} | {g(r['d_analytic_rad_over_analytic_abserr_R15'])}"
        )
    print()
    print("  summaries over the phase-limited rows:")
    moved = [r for r in pl if not r["total_bit_identical"]]
    if moved:
        ratios = sorted(r["abserr_ratio_R16_R15"] for r in moved if r["abserr_ratio_R16_R15"])
        print(
            f"    total moved in {len(moved)}: abserr16/abserr15 min {ratios[0]:.3g}, "
            f"median {ratios[len(ratios) // 2]:.3g}, max {ratios[-1]:.3g}; "
            f"max |dT|/abserr15 {max(r['dtotal_over_abserr_R15'] for r in moved):.3g}, "
            f"max |dT|/abserr16 {max(r['dtotal_over_abserr_R16'] for r in moved):.3g}"
        )
    am = [r for r in pl if not r["analytic_rad_bit_identical"]]
    if am:
        v = [r["d_analytic_rad_over_analytic_abserr_R15"] for r in am if r["d_analytic_rad_over_analytic_abserr_R15"] is not None]
        print(f"    analytic_rad moved in {len(am)}: max |d analytic_rad|/analytic abserr15 {max(v):.3g}")
    c = Counter((bool(r["converged_R15"]), bool(r["converged_R16"])) for r in pl)
    print(f"    total_converged R15->R16: {dict(c)}")
    c = Counter((r["WKB_Levin_max_depth_R15"], r["WKB_Levin_max_depth_R16"]) for r in pl)
    print(f"    WKB_Levin_max_depth R15->R16: {dict(sorted(c.items(), key=str))}")
    print(
        f"    Levin regions R15 {sum(r['WKB_Levin_num_regions_R15'] or 0 for r in pl)}, R16 "
        f"{sum(r['WKB_Levin_num_regions_R16'] or 0 for r in pl)}; analytic-block regions R15 "
        f"{sum(r['analytic_regions_R15'] for r in pl)}, R16 {sum(r['analytic_regions_R16'] for r in pl)}"
    )
    bad = len(viol_rows_named) + len(viol_rows_value)
    print()
    print(f"VERDICT: rows with no phase-limited flag in R15 that are not bit-identical in values: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
