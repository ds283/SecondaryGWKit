"""R30 against R29 inside QuadSourceIntegral: tabulate every row that moved.

levin-phase-floor 07a, rehearsal agent, prompt 07a section 5 item 3 ("inside it, every moved row
tabulated"). New. The store reader (`read`), `flatten`, `bits`, `col_kind`, `leaf_kind` and
`fields` are copied verbatim (lines 88-180) from 07's R29 tabulate_qsi_guard_r29.py (47e18cbe...),
which copied them verbatim from 06m's qsi_identity_r24_r25.py (e63aae0c...), 06g's; so the field
classification is 06g's:
- every column of table QuadSourceIntegral, in every shard, read with sqlite `mode=ro`, and every
  leaf of its `metadata` JSON (json.loads, flattened to a dotted path with [i] list indices);
- rows matched by their physical key (k, q, r, z_response) through the store's own wavenumber and
  redshift tables, with the tolerances (log10 atol, log10 rtol) carried and compared;
- a field is `timing`, `label`, `identity`, `counter` (WKB_Levin_num_regions, _evaluations,
  _simple_regions, _SVD_errors, _order_changes, _chebyshev_min_order, _max_depth,
  numeric_quad_compute_steps, numeric_quad_RHS_evaluations, and the metadata leaves regions,
  simple_regions, evaluations) or `value` (everything else).
Here one more kind is split out of `value`: `guard`, the four counts of prompt 07 section 1.6
(stationary_points, window_pieces, guarded_regions, guard_at_depth_max), present in every Levin
record (group and analytic block) of BOTH stores (both codes write them; ComputeTargets is the
same at e182bc7a and 4bf50d34). Their presence in exactly the Levin records is asserted in both.

A row is MOVED iff some value, counter or guard field differs (bit for bit: float.hex, else type
and value). Rows whose only differences are timing and label are UNMOVED. Every moved row is
tabulated into <rows csv>: total, total_abserr, |dtotal| against R29's and R30's total_abserr,
regions (WKB_Levin_num_regions), maximum depth (WKB_Levin_max_depth), total_converged, and the
four counts summed over the group records and over the analytic blocks, in each store; and
analytic_rad's change against analytic.abserr in each. Every Levin record that differs in any
non-timing field is written to <records csv> with both stores' fields. The rows whose
total_converged changes are listed in each direction, and so are group records and analytic
blocks whose `converged` changes.

Imports no tree code; reads the stores with sqlite3 mode=ro and store_value_diff's JSON (whose
QuadSourceIntegral row counts are checked against this script's).

    ./venv/bin/python -I -B tabulate_qsi_r29_r30.py <R29 primary> <R30 primary> <store_value_diff json> <rows csv> <records csv>
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
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3",  # R30's tree, 4bf50d34 (code 1ec734d7)
)
RUNS = (
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-r29-guard-rule-a-20261108T123429",
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07a-r30-07a-20261112T113806",
)
GUARD_LEAVES = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
DEPTH_MAX = 20  # DEFAULT_LEVIN_MAX_DEPTH (unchanged by 1ec734d7: prompt 07a section 8)

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
            out.append(("group", f"WKB_Levin.subintervals[{i}].groups[{j}]", g))
    for rad in ("0pt5", "2pt5"):
        lev = ((meta.get("analytic") or {}).get(rad) or {}).get("Levin") or {}
        for blk, rec in lev.items():
            out.append(("analytic", f"analytic.{rad}.Levin.{blk}", rec))
    return out


def guard_paths(flat):
    return {p for p in flat if re.split(r"[.\]]", p)[-1] in GUARD_LEAVES}


def ratio(d, e):
    if e is None or not isinstance(e, float) or e == 0.0 or not math.isfinite(e):
        return float("nan")
    return d / e


def fnum(s):
    x = float(s)
    return -1.0 if math.isnan(x) else x


def main():
    a_path, b_path, json_path, csv_path, rec_csv_path = sys.argv[1:6]
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
        st = subprocess.run(["git", "--no-optional-locks", "-C", t, "status", "--short", "--ignored"],
                            capture_output=True, text=True).stdout
        print(f"# tree {t} HEAD {head} status --short --ignored {st!r}")
    for run in RUNS:
        s = json.load(open(Path(run) / "status.json"))
        print(f"# run {Path(run).name}: state {s['state']} exit {s['exit_code']} fingerprint "
              f"{s['fingerprint']['digest']} taken at {s['fingerprint']['taken']['git_head']}")
    print("# this script imports no tree code; it reads the stores with sqlite3 mode=ro and the tool's JSON")

    A, cols_a = read(a_path)
    B, cols_b = read(b_path)
    print(f"A = R29 {a_path}: {len(A)} rows, {len(cols_a)} columns")
    print(f"B = R30 {b_path}: {len(B)} rows, {len(cols_b)} columns")
    assert cols_a == cols_b, "columns differ"
    assert set(A) == set(B), "key sets differ"
    print("columns equal: True; key sets equal: True (0 only in A, 0 only in B)")

    tool = json.load(open(json_path))
    q = tool["classes"]["QuadSourceIntegral"]["rows"]
    print(f"tool: QuadSourceIntegral identical {q['identical']}, differ {q['differ']}, only_in_a "
          f"{len(q['only_in_a'])}, only_in_b {len(q['only_in_b'])}, unmatched {len(q['unmatched'])}, "
          f"columns only in a/b {q['columns_only_in_a']}/{q['columns_only_in_b']}")
    tool_cols = q["columns"]
    for c in ("total", "total_abserr", "analytic_rad", "WKB_Levin", "WKB_Levin_num_regions", "WKB_Levin_max_depth"):
        print(f"tool: column {c}: {tool_cols.get(c, {'count': 0})}")

    keys = sorted(A)
    rows_out, recs_out = [], []
    tol_mismatch = 0
    n_unmoved = 0
    moved_kinds = Counter()
    diff_fields = Counter()
    total_bitid = abserr_bitid = rad_bitid = radabserr_bitid = conv_bitid = 0
    conv_up, conv_down = [], []
    grp_conv_up, grp_conv_down, blk_conv_up, blk_conv_down = [], [], [], []
    an_conv_change = []
    regions_sum = [0, 0]
    unconv_groups = [0, 0]
    rows_at_depth_max = [0, 0]
    guard_totals = [{"group": Counter(), "analytic": Counter()}, {"group": Counter(), "analytic": Counter()}]
    rows_with = [Counter(), Counter()]
    rec_changed = Counter()
    rec_changed_fields = Counter()
    for key in keys:
        a, b = A[key], B[key]
        if (a["_atol"], a["_rtol"]) != (b["_atol"], b["_rtol"]):
            tol_mismatch += 1
        fa, meta_a, flat_a = fields(a)
        fb, meta_b, flat_b = fields(b)
        recs_a, recs_b = levin_records(meta_a), levin_records(meta_b)
        assert [r[1] for r in recs_a] == [r[1] for r in recs_b], key
        cnt = []
        for st, recs, flat, ff in ((0, recs_a, flat_a, fa), (1, recs_b, flat_b, fb)):
            expected = set()
            c = {"group": Counter(), "analytic": Counter()}
            for kind, name, rec in recs:
                for g in GUARD_LEAVES:
                    assert g in rec and isinstance(rec[g], int) and not isinstance(rec[g], bool), (st, key, name, g)
                    expected.add(f"{name}.{g}")
                    c[kind][g] += rec[g]
                    guard_totals[st][kind][g] += rec[g]
            assert guard_paths(flat) == expected, (st, key, sorted(guard_paths(flat) ^ expected)[:5])
            for p in expected:
                k_, v_ = ff[f"meta:{p}"]
                ff[f"meta:{p}"] = ("guard", v_)
            for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"):
                if c["group"][g] + c["analytic"][g] > 0:
                    rows_with[st][g] += 1
            cnt.append(c)
        if set(fa) != set(fb):
            raise SystemExit(f"field sets differ at {key}: {sorted(set(fa) ^ set(fb))[:10]}")
        kinds = Counter()
        for f in fa:
            ka, va = fa[f]
            kb, vb = fb[f]
            assert ka == kb, (key, f)
            if bits(va) != bits(vb) and ka in ("value", "counter", "guard"):
                kinds[ka] += 1
                diff_fields[(ka, re.sub(r"\[\d+\]", "[i]", f))] += 1
        t_same = bits(a["total"]) == bits(b["total"])
        e_same = bits(a["total_abserr"]) == bits(b["total_abserr"])
        r_same = bits(a["analytic_rad"]) == bits(b["analytic_rad"])
        ea_a = (meta_a.get("analytic") or {}).get("abserr")
        ea_b = (meta_b.get("analytic") or {}).get("abserr")
        total_bitid += t_same
        abserr_bitid += e_same
        rad_bitid += r_same
        radabserr_bitid += bits(ea_a) == bits(ea_b)
        conv_bitid += bits(a["total_converged"]) == bits(b["total_converged"])
        for st, meta, rec in ((0, meta_a, a), (1, meta_b, b)):
            groups = [r[2] for r in levin_records(meta) if r[0] == "group"]
            if rec["WKB_Levin_num_regions"] is None:
                assert not groups, key
            else:
                assert sum(g["regions"] for g in groups) == rec["WKB_Levin_num_regions"], key
                regions_sum[st] += rec["WKB_Levin_num_regions"]
            if groups:
                assert bool(rec["total_converged"]) == all(g["converged"] for g in groups), key
            unconv_groups[st] += sum(1 for g in groups if not g["converged"])
            rows_at_depth_max[st] += rec["WKB_Levin_max_depth"] == DEPTH_MAX
        ca, cb = bool(a["total_converged"]), bool(b["total_converged"])
        if ca and not cb:
            conv_down.append(key)
        if cb and not ca:
            conv_up.append(key)
        an_ca = (meta_a.get("analytic") or {}).get("converged")
        an_cb = (meta_b.get("analytic") or {}).get("converged")
        if an_ca != an_cb:
            an_conv_change.append((key, an_ca, an_cb))
        # Levin records
        changed_desc = []
        for (kind, name, ra), (_, _, rb) in zip(recs_a, recs_b):
            fl_a, fl_b = flatten(ra, name), flatten(rb, name)
            assert set(fl_a) == set(fl_b), (key, name)
            dif = [p for p in fl_a if bits(fl_a[p]) != bits(fl_b[p])
                   and leaf_kind(p) != "timing" and not p.endswith(".label")]
            if ra.get("converged") and not rb.get("converged"):
                (grp_conv_down if kind == "group" else blk_conv_down).append((key, name))
            if rb.get("converged") and not ra.get("converged"):
                (grp_conv_up if kind == "group" else blk_conv_up).append((key, name))
            if not dif:
                continue
            lab = ra.get("label", name.split(".")[-1])
            rec_changed[(kind, lab if kind == "group" else name.split(".", 1)[1])] += 1
            for p in dif:
                rec_changed_fields[(kind, p.rsplit(".", 1)[-1])] += 1
            recs_out.append({
                "k": key[0], "q": key[1], "r": key[2], "z_response": key[3], "kind": kind, "record": name,
                "label": lab,
                "fields_differing": " ".join(p.rsplit(".", 1)[-1] for p in dif),
                "value_29": hx(ra.get("value")), "value_30": hx(rb.get("value")),
                "abserr_29": hx(ra.get("abserr")), "abserr_30": hx(rb.get("abserr")),
                "abs_dvalue_over_abserr29": repr(ratio(abs(rb["value"] - ra["value"]), ra["abserr"]))
                if isinstance(ra.get("value"), float) and isinstance(rb.get("value"), float) else "",
                "abs_dvalue_over_abserr30": repr(ratio(abs(rb["value"] - ra["value"]), rb["abserr"]))
                if isinstance(ra.get("value"), float) and isinstance(rb.get("value"), float) else "",
                "converged_29": ra.get("converged"), "converged_30": rb.get("converged"),
                "phase_limited_29": ra.get("phase_limited"), "phase_limited_30": rb.get("phase_limited"),
                "regions_29": ra.get("regions"), "regions_30": rb.get("regions"),
                **{f"{g}_29": ra[g] for g in GUARD_LEAVES},
                **{f"{g}_30": rb[g] for g in GUARD_LEAVES},
            })
            changed_desc.append(f"{kind}:{name}:{lab}")
        if not kinds:
            n_unmoved += 1
            assert t_same and e_same and r_same, key
            continue
        for k_ in kinds:
            moved_kinds[k_] += 1
        dT = abs(b["total"] - a["total"])
        dR = abs(b["analytic_rad"] - a["analytic_rad"]) if a["analytic_rad"] is not None else float("nan")
        row = {
            "k": key[0], "q": key[1], "r": key[2], "z_response": key[3],
            "kinds_differing": " ".join(f"{k_}:{v_}" for k_, v_ in sorted(kinds.items())),
            "total_29": hx(a["total"]), "total_30": hx(b["total"]),
            "total_abserr_29": hx(a["total_abserr"]), "total_abserr_30": hx(b["total_abserr"]),
            "abs_dtotal": repr(dT),
            "abs_dtotal_over_abserr29": repr(ratio(dT, a["total_abserr"])),
            "abs_dtotal_over_abserr30": repr(ratio(dT, b["total_abserr"])),
            "s3_abs_dtotal_over_sum_abserr": repr(ratio(dT, a["total_abserr"] + b["total_abserr"]))
            if isinstance(a["total_abserr"], float) and isinstance(b["total_abserr"], float) else "nan",
            "total_bit_identical": t_same, "total_abserr_bit_identical": e_same,
            "total_converged_29": ca, "total_converged_30": cb,
            "WKB_Levin_num_regions_29": a["WKB_Levin_num_regions"], "WKB_Levin_num_regions_30": b["WKB_Levin_num_regions"],
            "WKB_Levin_max_depth_29": a["WKB_Levin_max_depth"], "WKB_Levin_max_depth_30": b["WKB_Levin_max_depth"],
            "group_calls_unconverged_29": sum(1 for r in recs_a if r[0] == "group" and not r[2]["converged"]),
            "group_calls_unconverged_30": sum(1 for r in recs_b if r[0] == "group" and not r[2]["converged"]),
        }
        for st, tag in ((0, "29"), (1, "30")):
            for kind in ("group", "analytic"):
                for g in GUARD_LEAVES:
                    row[f"{kind}_{g}_{tag}"] = cnt[st][kind][g]
        row.update({
            "analytic_rad_29": hx(a["analytic_rad"]), "analytic_rad_30": hx(b["analytic_rad"]),
            "analytic_abserr_29": hx(ea_a), "analytic_abserr_30": hx(ea_b),
            "abs_drad_over_abserr29": repr(ratio(dR, ea_a)), "abs_drad_over_abserr30": repr(ratio(dR, ea_b)),
            "analytic_rad_bit_identical": r_same,
            "analytic_converged_29": an_ca, "analytic_converged_30": an_cb,
            "levin_records_changed": " | ".join(changed_desc),
        })
        rows_out.append(row)

    n = len(keys)
    print(f"rows compared: {n}; tolerance mismatches {tol_mismatch}")
    for st, tag in ((0, "R29"), (1, "R30")):
        print(f"{tag}: sums of the four counts, group records {dict(guard_totals[st]['group'])}; "
              f"analytic blocks {dict(guard_totals[st]['analytic'])}")
        print(f"{tag}: rows with a located point {rows_with[st]['stationary_points']}, a window piece "
              f"{rows_with[st]['window_pieces']}, a guarded region {rows_with[st]['guarded_regions']}, "
              f"a guard at depth_max {rows_with[st]['guard_at_depth_max']}")
    print()
    print(f"UNMOVED rows (only timing and label differ): {n_unmoved}")
    print(f"MOVED rows (a value, counter or guard field differs): {len(rows_out)}; by kind present: {dict(moved_kinds)}")
    print("  fields differing (kind, path with indices folded: rows):")
    for (k_, f), c in sorted(diff_fields.items()):
        print(f"    {k_} {f}: {c}")
    print(f"  Levin records differing in a non-timing field, by (kind, label/block): {dict(sorted(rec_changed.items()))}")
    print(f"  ... by (kind, field): {dict(sorted(rec_changed_fields.items()))}")
    print()
    print(f"ALL ROWS: total bit-identical {total_bitid} / {n}; total_abserr {abserr_bitid}; "
          f"analytic_rad {rad_bitid}; analytic.abserr {radabserr_bitid}; total_converged {conv_bitid}")
    print(f"WKB_Levin_num_regions summed: R29 {regions_sum[0]}, R30 {regions_sum[1]}; unconverged group calls: "
          f"R29 {unconv_groups[0]}, R30 {unconv_groups[1]}; rows at max depth {DEPTH_MAX}: R29 {rows_at_depth_max[0]}, R30 {rows_at_depth_max[1]}")
    if rows_out:
        tot_moved = [r for r in rows_out if not r["total_bit_identical"]]
        print(f"moved rows whose total differs: {len(tot_moved)}; whose total_abserr differs: "
              f"{sum(not r['total_abserr_bit_identical'] for r in rows_out)}; whose analytic_rad differs: "
              f"{sum(not r['analytic_rad_bit_identical'] for r in rows_out)}")
        for col in ("abs_dtotal_over_abserr29", "abs_dtotal_over_abserr30", "s3_abs_dtotal_over_sum_abserr",
                    "abs_drad_over_abserr29", "abs_drad_over_abserr30"):
            big = max(rows_out, key=lambda r: fnum(r[col]))
            print(f"  largest {col}: {big[col]} at ({big['k']}, {big['q']}, {big['r']}, {big['z_response']}); "
                  f"total {big['total_29']} -> {big['total_30']}, abserr {big['total_abserr_29']} -> {big['total_abserr_30']}")
        print(f"  moved rows with |dtotal|/abserr29 > 1: {sum(fnum(r['abs_dtotal_over_abserr29']) > 1 for r in rows_out)}; "
              f"> abserr30: {sum(fnum(r['abs_dtotal_over_abserr30']) > 1 for r in rows_out)}; "
              f"> abserr29+abserr30: {sum(fnum(r['s3_abs_dtotal_over_sum_abserr']) > 1 for r in rows_out)}")
        print(f"  moved rows with regions changed: {sum(r['WKB_Levin_num_regions_29'] != r['WKB_Levin_num_regions_30'] for r in rows_out)}; "
              f"max depth changed: {sum(r['WKB_Levin_max_depth_29'] != r['WKB_Levin_max_depth_30'] for r in rows_out)}")
        for st, tag in (("29", "R29"), ("30", "R30")):
            print(f"  among moved rows, {tag}: with a located point "
                  f"{sum(r[f'group_stationary_points_{st}'] + r[f'analytic_stationary_points_{st}'] > 0 for r in rows_out)}, "
                  f"a window piece {sum(r[f'group_window_pieces_{st}'] + r[f'analytic_window_pieces_{st}'] > 0 for r in rows_out)}, "
                  f"a guarded region {sum(r[f'group_guarded_regions_{st}'] + r[f'analytic_guarded_regions_{st}'] > 0 for r in rows_out)}, "
                  f"total_converged {sum(r[f'total_converged_{st}'] for r in rows_out)} of {len(rows_out)}")
    print()
    print(f"total_converged: R29 {sum(bool(A[k]['total_converged']) for k in keys)}, R30 {sum(bool(B[k]['total_converged']) for k in keys)}")
    print(f"total_converged changes: unconverged -> converged {len(conv_up)}; converged -> unconverged {len(conv_down)}")
    for lab, lst in (("unconverged -> converged", conv_up), ("converged -> unconverged", conv_down)):
        for key in lst:
            a, b = A[key], B[key]
            print(f"  {lab}: {key} total {hx(a['total'])} -> {hx(b['total'])}, abserr {hx(a['total_abserr'])} -> "
                  f"{hx(b['total_abserr'])}, regions {a['WKB_Levin_num_regions']} -> {b['WKB_Levin_num_regions']}, "
                  f"max depth {a['WKB_Levin_max_depth']} -> {b['WKB_Levin_max_depth']}")
    print(f"group-record converged changes: up {len(grp_conv_up)}, down {len(grp_conv_down)}")
    for lab, lst in (("up", grp_conv_up), ("down", grp_conv_down)):
        for key, name in lst:
            print(f"  group {lab}: {key} {name}")
    print(f"analytic-block converged changes: up {len(blk_conv_up)}, down {len(blk_conv_down)}")
    for lab, lst in (("up", blk_conv_up), ("down", blk_conv_down)):
        for key, name in lst:
            print(f"  block {lab}: {key} {name}")
    print(f"analytic.converged changes: {len(an_conv_change)}")
    for key, x, y in an_conv_change:
        print(f"  analytic {key}: {x} -> {y}")
    for path, out in ((csv_path, rows_out), (rec_csv_path, recs_out)):
        with open(path, "w", newline="") as f:
            if out:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader()
                w.writerows(out)
        print(f"# wrote {len(out)} rows to {path} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
