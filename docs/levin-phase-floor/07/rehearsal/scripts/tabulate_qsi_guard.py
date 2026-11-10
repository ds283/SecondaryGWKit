"""R28 against R27 inside QuadSourceIntegral: classify every row by R28's four guard counts, check
the untouched rows bit for bit, and tabulate the rest.

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.4 ("R28 against R27", inside
QuadSourceIntegral). New. The store reader (`read`), `flatten`, `bits`, `col_kind`, `leaf_kind` and
`fields` are copied verbatim from 06m's rehearsal qsi_identity_r24_r25.py (e63aae0c...), which copied
them verbatim from 06g's tabulate_qsi_leafvalues.py (3268a0bc...); so the field classification is
06g's (and 06m's, 06h's, 06e's, 06c's, 06b's):
- every column of table QuadSourceIntegral, in every shard, read with sqlite `mode=ro`, and every
  leaf of its `metadata` JSON (json.loads, flattened to a dotted path with [i] list indices);
- rows matched by their physical key (k, q, r, z_response) through the store's own wavenumber and
  redshift tables, with the tolerances (log10 atol, log10 rtol) carried and compared;
- a field is `timing` (a column ending compute_time / RHS_time / elapsed, or a metadata leaf
  `elapsed` / `compute_time`), `label`, `identity` (serial, *_serial, timestamp), `counter` (the
  work counters: the columns WKB_Levin_num_regions, _evaluations, _simple_regions, _SVD_errors,
  _order_changes, _chebyshev_min_order, _max_depth, numeric_quad_compute_steps,
  numeric_quad_RHS_evaluations, and the metadata leaves regions, simple_regions, evaluations) or
  `value` (everything else).

Prompt 07's four counts (section 1.6, ComputeTargets/QuadSourceIntegral.py at 74600350, 33cb161e):
leaves `stationary_points` (number located, an int), `window_pieces`, `guarded_regions`,
`guard_at_depth_max` of
- every group record `WKB_Levin.subintervals[i].groups[j]` (phase_group_Levin_integral), and
- every three-Bessel block record `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data` (_three_bessel_Levin).
They are asserted present in every such record of R28, absent everywhere in R27, and absent from
every other path of R28. They are set aside from every comparison (their kind here is `guard`).

A row is UNTOUCHED iff every count in every one of its Levin records (group and analytic) is 0.
Every untouched row must be bit-identical (`float.hex` for floats; type and value otherwise) to
R27's in every value field (total, total_abserr, total_converged, total_phase_limited, analytic_rad,
every Levin record's value / abserr / converged / phase_limited, the analytic metadata, ...) and,
reported separately, every counter field. An untouched row with a value difference is a stop
(prompt 07 section 8); the script lists such rows first and exits 2.

Every other row is tabulated into the CSV, with a summary on stdout.

    ./venv/bin/python -I -B tabulate_qsi_guard.py <R27 primary> <R28 primary> <store_value_diff json> <out csv>
"""

import csv
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

TREES = (
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db",  # R27's tree, 91472dbe
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035",  # R28's tree, 74600350
)
RUNS = (
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-r27-parent-20261108T000634",
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-r28-guard-20261108T001916",
)
GUARD_LEAVES = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
GROUP_RE = re.compile(r"^WKB_Levin\.subintervals\[(\d+)\]\.groups\[(\d+)\]$")
BLOCK_RE = re.compile(r"^analytic\.(0pt5|2pt5)\.Levin\.([JY][1-4])_data$")
DEPTH_MAX = 20  # DEFAULT_LEVIN_MAX_DEPTH, the same in both trees (checked in the log)

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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- copied verbatim from 06m's qsi_identity_r24_r25.py (06g's) ----
def read(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    cols = None
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
            + ", ".join(f'q."{c}"' for c in cols)
            + " FROM QuadSourceIntegral q"
        )
        for row in con.execute(sql):
            key = row[:4]
            assert key not in out, f"duplicate {key}"
            rec = dict(zip(cols, row[6:]))
            rec["_atol"], rec["_rtol"] = row[4], row[5]
            out[key] = rec
        con.close()
    return out, cols


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


def fields(rec):
    out = {}
    for c, v in rec.items():
        if c.startswith("_") or c == "metadata":
            continue
        out[f"col:{c}"] = (col_kind(c), v)
    meta = json.loads(rec["metadata"]) if rec["metadata"] is not None else {}
    flat = flatten(meta)
    for p, v in flat.items():
        out[f"meta:{p}"] = (leaf_kind(p), v)
    return out, meta, flat


# ---- end of the verbatim copy ----


def hx(x):
    return x.hex() if isinstance(x, float) else repr(x)


def levin_records(meta):
    """(kind, name, record) for every Levin record of a row: group records and analytic blocks."""
    out = []
    for i, s in enumerate((meta.get("WKB_Levin") or {}).get("subintervals", [])):
        for j, g in enumerate(s.get("groups", [])):
            out.append(("group", f"WKB_Levin.subintervals[{i}].groups[{j}]", g, s))
    for rad in ("0pt5", "2pt5"):
        lev = ((meta.get("analytic") or {}).get(rad) or {}).get("Levin") or {}
        for blk, rec in lev.items():
            out.append(("analytic", f"analytic.{rad}.Levin.{blk}", rec, None))
    return out


def guard_paths(flat):
    return {p for p in flat if re.split(r"[.\]]", p)[-1] in GUARD_LEAVES}


def ratio(d, e):
    if e is None or not isinstance(e, float) or e == 0.0 or not math.isfinite(e):
        return float("nan")
    return d / e


def main():
    a_path, b_path, json_path, csv_path = sys.argv[1:5]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    for p in (a_path, b_path):
        prim = Path(p)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# input {f} sha256 {sha256(f)}")
    print(f"# input {json_path} sha256 {sha256(json_path)}")
    print(f"# cwd {Path.cwd()}")
    for t in TREES:
        head = subprocess.run(["git", "--no-optional-locks", "-C", t, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "--no-optional-locks", "-C", t, "status", "--short"],
                            capture_output=True, text=True).stdout
        print(f"# tree {t} HEAD {head} status --short {st!r}")
    for run in RUNS:
        s = json.load(open(Path(run) / "status.json"))
        print(f"# run {Path(run).name}: state {s['state']} exit {s['exit_code']} fingerprint "
              f"{s['fingerprint']['digest']} taken at {s['fingerprint']['taken']['git_head']}")
    print("# this script imports no tree code; it reads the stores with sqlite3 mode=ro and the tool's JSON")

    A, cols_a = read(a_path)
    B, cols_b = read(b_path)
    print(f"A = R27 {a_path}: {len(A)} rows, {len(cols_a)} columns")
    print(f"B = R28 {b_path}: {len(B)} rows, {len(cols_b)} columns")
    assert cols_a == cols_b, "columns differ"
    assert set(A) == set(B), "key sets differ"
    print("columns equal: True; key sets equal: True (0 only in A, 0 only in B)")

    tool = json.load(open(json_path))
    q = tool["classes"]["QuadSourceIntegral"]["rows"]
    print(f"tool: QuadSourceIntegral identical {q['identical']}, differ {q['differ']}, only_in_a "
          f"{len(q['only_in_a'])}, only_in_b {len(q['only_in_b'])}, unmatched {len(q['unmatched'])}, "
          f"columns only in a/b {q['columns_only_in_a']}/{q['columns_only_in_b']}")

    keys = sorted(A)
    stop_rows = []
    rows_out = []
    tol_mismatch = 0
    n_untouched = n_untouched_value_id = n_untouched_counter_id = 0
    n_touched = 0
    untouched_counter_diffs = Counter()
    diff_fields_touched = Counter()
    total_bitid = abserr_bitid = rad_bitid = radabserr_bitid = conv_bitid = 0
    total_bitid_untouched = total_bitid_touched = 0
    rad_bitid_untouched = rad_bitid_touched = 0
    guard_totals = {"group": Counter(), "analytic": Counter()}
    rec_counts = {"group": Counter(), "analytic": Counter()}
    rows_located_group = rows_located_analytic = rows_guarded_group = rows_guarded_analytic = 0
    conv_change_up, conv_change_down = [], []
    regions_sum = [0, 0]
    unconv_calls = [0, 0]
    rows_at_depth_max = [0, 0]
    for key in keys:
        a, b = A[key], B[key]
        if (a["_atol"], a["_rtol"]) != (b["_atol"], b["_rtol"]):
            tol_mismatch += 1
        fa, meta_a, flat_a = fields(a)
        fb, meta_b, flat_b = fields(b)
        # the four counts: absent in R27, present exactly in the Levin records of R28
        assert not guard_paths(flat_a), (key, sorted(guard_paths(flat_a))[:3])
        recs_b = levin_records(meta_b)
        recs_a = levin_records(meta_a)
        assert [r[1] for r in recs_a] == [r[1] for r in recs_b], key
        expected = set()
        counts = {"group": Counter(), "analytic": Counter()}
        located = []
        for kind, name, rec, sub in recs_b:
            for g in GUARD_LEAVES:
                assert g in rec and isinstance(rec[g], int) and not isinstance(rec[g], bool), (key, name, g)
                expected.add(f"{name}.{g}")
                counts[kind][g] += rec[g]
                guard_totals[kind][g] += rec[g]
            rec_counts[kind]["records"] += 1
            if any(rec[g] for g in GUARD_LEAVES):
                rec_counts[kind]["touched records"] += 1
                located.append((kind, name, rec))
        assert guard_paths(flat_b) == expected, (key, sorted(guard_paths(flat_b) ^ expected)[:5])
        for p in expected:
            fb.pop(f"meta:{p}")
        if set(fa) != set(fb):
            raise SystemExit(f"field sets differ at {key}: {sorted(set(fa) ^ set(fb))[:10]}")
        vdiff, cdiff = [], []
        for f in fa:
            ka, va = fa[f]
            kb, vb = fb[f]
            assert ka == kb, (key, f)
            if bits(va) != bits(vb):
                if ka == "value":
                    vdiff.append(f)
                elif ka == "counter":
                    cdiff.append(f)
        untouched = all(v == 0 for c in counts.values() for v in c.values())
        t_same = bits(a["total"]) == bits(b["total"])
        r_same = bits(a["analytic_rad"]) == bits(b["analytic_rad"])
        total_bitid += t_same
        abserr_bitid += bits(a["total_abserr"]) == bits(b["total_abserr"])
        rad_bitid += r_same
        ea_a = (meta_a.get("analytic") or {}).get("abserr")
        ea_b = (meta_b.get("analytic") or {}).get("abserr")
        radabserr_bitid += bits(ea_a) == bits(ea_b)
        conv_bitid += bits(a["total_converged"]) == bits(b["total_converged"])
        # consistency checks on the stored row (06m's): region sum and the AND of group flags
        for st, meta, rec in ((0, meta_a, a), (1, meta_b, b)):
            groups = [r[2] for r in levin_records(meta) if r[0] == "group"]
            if rec["WKB_Levin_num_regions"] is None:
                assert not groups, key
            else:
                assert sum(g["regions"] for g in groups) == rec["WKB_Levin_num_regions"], key
                regions_sum[st] += rec["WKB_Levin_num_regions"]
            if groups:
                assert bool(rec["total_converged"]) == all(g["converged"] for g in groups), key
            unconv_calls[st] += sum(1 for g in groups if not g["converged"])
            rows_at_depth_max[st] += rec["WKB_Levin_max_depth"] == DEPTH_MAX
        ca, cb = bool(a["total_converged"]), bool(b["total_converged"])
        if ca and not cb:
            conv_change_down.append(key)
        if cb and not ca:
            conv_change_up.append(key)
        if untouched:
            n_untouched += 1
            total_bitid_untouched += t_same
            rad_bitid_untouched += r_same
            if vdiff:
                stop_rows.append((key, vdiff))
            else:
                n_untouched_value_id += 1
            if cdiff:
                for f in cdiff:
                    untouched_counter_diffs[re.sub(r"\[\d+\]", "[i]", f)] += 1
            else:
                n_untouched_counter_id += 1
            continue
        n_touched += 1
        total_bitid_touched += t_same
        rad_bitid_touched += r_same
        for f in vdiff + cdiff:
            diff_fields_touched[re.sub(r"\[\d+\]", "[i]", f)] += 1
        g_loc = counts["group"]["stationary_points"] > 0
        an_loc = counts["analytic"]["stationary_points"] > 0
        rows_located_group += g_loc
        rows_located_analytic += an_loc
        rows_guarded_group += counts["group"]["guarded_regions"] > 0
        rows_guarded_analytic += counts["analytic"]["guarded_regions"] > 0
        dT = abs(b["total"] - a["total"])
        dR = abs(b["analytic_rad"] - a["analytic_rad"]) if a["analytic_rad"] is not None else float("nan")
        groups_a = {r[1]: r[2] for r in recs_a}
        loc_desc = []
        for kind, name, rec in located:
            ra = groups_a[name]
            lab = rec.get("label", name.split(".")[-1])
            loc_desc.append(
                f"{kind}:{name}:{lab}:sp={rec['stationary_points']},wp={rec['window_pieces']},"
                f"gr={rec['guarded_regions']},gd={rec['guard_at_depth_max']};"
                f"v27={hx(ra.get('value'))},v28={hx(rec.get('value'))},"
                f"e27={hx(ra['abserr'])},e28={hx(rec['abserr'])},"
                f"c27={ra['converged']},c28={rec['converged']},"
                f"reg27={ra['regions']},reg28={rec['regions']}"
            )
        rows_out.append({
            "k": key[0], "q": key[1], "r": key[2], "z_response": key[3],
            "total_27": hx(a["total"]), "total_28": hx(b["total"]),
            "total_abserr_27": hx(a["total_abserr"]), "total_abserr_28": hx(b["total_abserr"]),
            "abs_dtotal": repr(dT),
            "abs_dtotal_over_abserr27": repr(ratio(dT, a["total_abserr"])),
            "abs_dtotal_over_abserr28": repr(ratio(dT, b["total_abserr"])),
            "total_bit_identical": t_same,
            "total_abserr_bit_identical": bits(a["total_abserr"]) == bits(b["total_abserr"]),
            "total_converged_27": ca, "total_converged_28": cb,
            "group_calls": sum(1 for r in recs_b if r[0] == "group"),
            "group_calls_unconverged_27": sum(1 for r in recs_a if r[0] == "group" and not r[2]["converged"]),
            "group_calls_unconverged_28": sum(1 for r in recs_b if r[0] == "group" and not r[2]["converged"]),
            "group_stationary_points": counts["group"]["stationary_points"],
            "group_window_pieces": counts["group"]["window_pieces"],
            "group_guarded_regions": counts["group"]["guarded_regions"],
            "group_guard_at_depth_max": counts["group"]["guard_at_depth_max"],
            "analytic_stationary_points": counts["analytic"]["stationary_points"],
            "analytic_window_pieces": counts["analytic"]["window_pieces"],
            "analytic_guarded_regions": counts["analytic"]["guarded_regions"],
            "analytic_guard_at_depth_max": counts["analytic"]["guard_at_depth_max"],
            "WKB_Levin_num_regions_27": a["WKB_Levin_num_regions"],
            "WKB_Levin_num_regions_28": b["WKB_Levin_num_regions"],
            "WKB_Levin_max_depth_27": a["WKB_Levin_max_depth"],
            "WKB_Levin_max_depth_28": b["WKB_Levin_max_depth"],
            "analytic_rad_27": hx(a["analytic_rad"]), "analytic_rad_28": hx(b["analytic_rad"]),
            "analytic_abserr_27": hx(ea_a), "analytic_abserr_28": hx(ea_b),
            "abs_drad_over_abserr27": repr(ratio(dR, ea_a)),
            "abs_drad_over_abserr28": repr(ratio(dR, ea_b)),
            "analytic_rad_bit_identical": r_same,
            "analytic_converged_27": (meta_a.get("analytic") or {}).get("converged"),
            "analytic_converged_28": (meta_b.get("analytic") or {}).get("converged"),
            "value_fields_differing": len(vdiff),
            "counter_fields_differing": len(cdiff),
            "located_records": " | ".join(loc_desc),
        })

    print(f"rows compared: {len(keys)}; tolerance mismatches {tol_mismatch}")
    print(f"Levin records in R28: group {rec_counts['group']['records']} "
          f"(with a nonzero count {rec_counts['group']['touched records']}); analytic blocks "
          f"{rec_counts['analytic']['records']} (with a nonzero count {rec_counts['analytic']['touched records']})")
    print(f"sums of the four counts, group records: {dict(guard_totals['group'])}")
    print(f"sums of the four counts, analytic blocks: {dict(guard_totals['analytic'])}")
    print()
    print(f"UNTOUCHED rows (every count of every Levin record 0): {n_untouched}")
    print(f"  bit-identical in every value field (the four counts set aside): {n_untouched_value_id} / {n_untouched}")
    print(f"  bit-identical in every counter field too: {n_untouched_counter_id} / {n_untouched}")
    if untouched_counter_diffs:
        print(f"  counter differences in untouched rows (field: rows): {dict(untouched_counter_diffs)}")
    print(f"  total bit-identical: {total_bitid_untouched}; analytic_rad bit-identical: {rad_bitid_untouched}")
    if stop_rows:
        print("STOP: UNTOUCHED ROWS NOT BIT-IDENTICAL (prompt 07 section 8):")
        for key, vd in stop_rows[:50]:
            a, b = A[key], B[key]
            print(f"  {key}: {len(vd)} value fields differ, first {vd[:8]}; total {hx(a['total'])} -> "
                  f"{hx(b['total'])}; total_abserr {hx(a['total_abserr'])} -> {hx(b['total_abserr'])}")
    print()
    print(f"OTHER rows (some count nonzero): {n_touched}")
    print(f"  rows in which a group call located a point: {rows_located_group}; an analytic block: {rows_located_analytic}")
    print(f"  rows in which a group call guarded a leaf: {rows_guarded_group}; an analytic block: {rows_guarded_analytic}")
    print(f"  total bit-identical: {total_bitid_touched} / {n_touched}; analytic_rad bit-identical: {rad_bitid_touched} / {n_touched}")
    print("  fields differing in these rows (path, indices folded: rows):")
    for f, c in sorted(diff_fields_touched.items()):
        print(f"    {f}: {c}")
    if rows_out:
        def fl(s):
            return float(s)
        big27 = max(rows_out, key=lambda r: (fl(r["abs_dtotal_over_abserr27"]) if not math.isnan(fl(r["abs_dtotal_over_abserr27"])) else -1))
        big28 = max(rows_out, key=lambda r: (fl(r["abs_dtotal_over_abserr28"]) if not math.isnan(fl(r["abs_dtotal_over_abserr28"])) else -1))
        print(f"  largest |dtotal|/abserr27: {big27['abs_dtotal_over_abserr27']} at "
              f"({big27['k']}, {big27['q']}, {big27['r']}, {big27['z_response']}); there |dtotal|/abserr28 {big27['abs_dtotal_over_abserr28']}")
        print(f"  largest |dtotal|/abserr28: {big28['abs_dtotal_over_abserr28']} at "
              f"({big28['k']}, {big28['q']}, {big28['r']}, {big28['z_response']}); there |dtotal|/abserr27 {big28['abs_dtotal_over_abserr27']}")
        an = [r for r in rows_out if r["analytic_stationary_points"] > 0]
        if an:
            bA = max(an, key=lambda r: max(fl(r["abs_drad_over_abserr27"]), fl(r["abs_drad_over_abserr28"])))
            print(f"  analytic rows that located: {len(an)}; largest |drad|/abserr: 27 {bA['abs_drad_over_abserr27']}, "
                  f"28 {bA['abs_drad_over_abserr28']} at ({bA['k']}, {bA['q']}, {bA['r']}, {bA['z_response']})")
        print(f"  total_converged among them: R27 {sum(r['total_converged_27'] for r in rows_out)}, "
              f"R28 {sum(r['total_converged_28'] for r in rows_out)} of {n_touched}")
        print(f"  unconverged group calls among them: R27 {sum(r['group_calls_unconverged_27'] for r in rows_out)}, "
              f"R28 {sum(r['group_calls_unconverged_28'] for r in rows_out)}")
        print(f"  rows at WKB_Levin_max_depth {DEPTH_MAX}: R27 {sum(r['WKB_Levin_max_depth_27'] == DEPTH_MAX for r in rows_out)}, "
              f"R28 {sum(r['WKB_Levin_max_depth_28'] == DEPTH_MAX for r in rows_out)}")
    print()
    print(f"ALL ROWS: total bit-identical {total_bitid} / {len(keys)}; total_abserr {abserr_bitid}; "
          f"analytic_rad {rad_bitid}; analytic.abserr {radabserr_bitid}; total_converged {conv_bitid}")
    print(f"WKB_Levin_num_regions summed: R27 {regions_sum[0]}, R28 {regions_sum[1]}; unconverged group calls: "
          f"R27 {unconv_calls[0]}, R28 {unconv_calls[1]}; rows at max depth {DEPTH_MAX}: R27 {rows_at_depth_max[0]}, R28 {rows_at_depth_max[1]}")
    print(f"total_converged: R27 {sum(bool(A[k]['total_converged']) for k in keys)}, R28 {sum(bool(B[k]['total_converged']) for k in keys)}")
    print(f"total_converged changes: unconverged -> converged {len(conv_change_up)}; converged -> unconverged {len(conv_change_down)}")
    for lab, lst in (("unconverged -> converged", conv_change_up), ("converged -> unconverged", conv_change_down)):
        for key in lst:
            a, b = A[key], B[key]
            print(f"  {lab}: {key} touched={'no' if key not in {(r['k'], r['q'], r['r'], r['z_response']) for r in rows_out} else 'yes'} "
                  f"total {hx(a['total'])} -> {hx(b['total'])}, abserr {hx(a['total_abserr'])} -> {hx(b['total_abserr'])}, "
                  f"max depth {a['WKB_Levin_max_depth']} -> {b['WKB_Levin_max_depth']}")
    with open(csv_path, "w", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    print(f"# wrote {len(rows_out)} rows to {csv_path} sha256 {sha256(csv_path)}")
    if stop_rows:
        print("VERDICT: STOP -- untouched rows differ")
        sys.exit(2)
    print("VERDICT: every untouched row bit-identical in every value field; the other rows tabulated")


if __name__ == "__main__":
    main()
