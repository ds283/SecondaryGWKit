"""Prompt 06m §4: QuadSourceIntegral, R25 (parent 5ff8a5e4, production code R24's, 06g's) against
R26 (38f6a828, 06m's code commit 48cf87c0, route B's signed trust yardstick), every row, and R1 on
`total` and on the analytic oracle.

(06g's `tabulate_qsi_leafvalues.py` (3268a0bc...), copied and adapted for 06m: the store names
R23/R24 -> R25/R26 and their suffixes 23/24 -> 25/26 (by sed and one regex over suffixes), V1 -> R1;
this header and the counters' paragraph restated for 06m; a provenance header printed (this
script's SHA-256, every store file's SHA-256, the JSON's SHA-256, the code worktree's HEAD and both
stores' fingerprints from their runs' status.json); and the rows whose `total_converged` changes
listed in each direction. Nothing else in the method. The diff against 06g's copy is saved as
out/tabulate_qsi_signedyardstick.diff.
06g's own header: 06h's `tabulate_qsi_groupphase.py`, copied and adapted for 06g: the store names
R21/R22 -> R23/R24 and their suffixes 21/22 -> 23/24, H0 -> V1; the counters' paragraph restated
for 06g; and V1's second half, the same rule on the oracle (`analytic_rad` against
`analytic.abserr`), scored with the same exclusions. 06h's own header: 06e's
`tabulate_qsi_endnodes.py`, adapted for 06h: R19/R20 -> R21/R22, Z1 -> H0. 06e's own header: 06c's
`tabulate_qsi_global.py`, adapted for 06e: R17/R18 -> R19/R20, Y1 -> Z1, and 06a's 720
stationary-point rows identified, tabulated and excluded from the rule's scoring.)

Reads every store file with sqlite `mode=ro`, and store_value_diff's JSON (for a cross-check
only). Rows are paired by their physical (k, q, r, z_response): k, q and r are each
`wavenumber.k_inv_Mpc` through `wavenumber_exit_time`, z_response is `redshift.z`; the tolerances
(`tolerance.log10_tol` through `atol_serial`/`rtol_serial`) are read and reported. The pairing is
checked against the tool's identity-matched report: the same key set, and the same `float.hex` of
`total` in both stores wherever the tool reports `total` as differing, and equal hex where it
does not. (Written from 06b's `tabulate_qsi_phase_limited.py`, whose reader and field kinds it
keeps; its §4.3 rule is replaced by 06c's tabulation and R1.)

**Where the region counts and maximum depths are stored** (ComputeTargets/QuadSourceIntegral.py,
ObjectFactories/QuadSourceIntegral.py), in each shard's table `QuadSourceIntegral`:
  - column `WKB_Levin_num_regions` (SQLAlchemy Integer, nullable): the sum over every
    adaptive_levin_sincos call of the row's WKB_Levin sub-intervals (every phase group of every
    sub-interval) of the call's `num_regions` (_aggregate_LevinData). NULL when the row made no
    WKB Levin call;
  - column `WKB_Levin_max_depth` (Integer, nullable): the maximum over the same calls of the
    call's `max_depth`; NULL as above;
  - column `metadata` (a JSON string, json.dumps), at
    `WKB_Levin.subintervals[i].groups[j].regions` (JSON int): each call's own `num_regions`
    (phase_group_Levin_integral). Their sum is checked equal to `WKB_Levin_num_regions`;
  - the same `metadata`, at `analytic.{0pt5,2pt5}[.Levin].{J,Y}{1..4}_data.regions` (JSON int):
    `len(regions)` of each three-Bessel Levin block of the analytic oracle (_three_bessel_Levin),
    which feeds `analytic_rad`, not `total`. **No maximum depth of the analytic blocks is stored.**
  - `total_converged` is the column (Boolean, SQLite integer 0/1): the AND of every WKB Levin
    call's `converged` (evaluate_QuadSource_integral). `total_abserr` is the column (Float(64)),
    the linear sum of numeric_quad's and every WKB Levin call's `abserr`.
`num_untrusted_regions` and the trust check's per-leaf fields (06c §8) are not persisted by
QuadSourceIntegral; they are not in the store.

**Field kinds** (06b's):
  - timing: a wall clock. Columns ending `compute_time`, `RHS_time` or `elapsed`; metadata leaves
    named `elapsed` or `compute_time`. Not compared.
  - label: the `label` column (it carries main.py's datetime.now()). Not compared.
  - identity: store bookkeeping: `serial`, every `*_serial`, `timestamp`. Not compared (rows are
    paired by physical key).
  - counter: a work counter, a count of what the integrator did rather than a computed value of
    the integral. Columns `WKB_Levin_num_regions`, `WKB_Levin_evaluations`,
    `WKB_Levin_simple_regions`, `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`,
    `WKB_Levin_chebyshev_min_order`, `WKB_Levin_max_depth`, `numeric_quad_compute_steps`,
    `numeric_quad_RHS_evaluations`; metadata leaves named `regions`, `simple_regions` or
    `evaluations`. They describe the partition the driver chose and the work it did. 06m moves
    the partition (06m §0.3: values, errors, partitions and convergence move only in calls where
    some leaf is trusted under nu_B and not under phi_P) without changing what any counter means:
    its diff to AdaptiveLevin/levin_quadrature.py (5ff8a5e4 -> 38f6a828) adds
    `_trust_noise_yardstick` and changes the trust test, and leaves the result keys
    `num_regions`, `num_simple_regions`, `evaluations`, `max_depth`, `num_SVD_errors`,
    `num_order_changes` and `chebyshev_min_order` and their assignments untouched (no +/- line of
    the diff names any of them), and ComputeTargets/ is unchanged (06m §0.3).
    Compared and tabulated, but a difference in one is not a value difference.
  - value: everything else (total, total_abserr, total_converged, total_phase_limited,
    numeric_quad, WKB_Levin, analytic_rad, b, eta_*, every other metadata leaf).
Bit-identity is by `float.hex` for floats and by == for bool/int/str/None, with the type also
required equal. Never isclose.

**06a's 720 stationary-point rows** are rows of A3 v2, named `<shard>-<serial>` there. They are
identified as `analyse06c.py`'s `scan06a` identifies them: the 1e-7 records of 06a's census
checkpoint (`var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl`) whose `scan`
has a call with `n_sign_changes > 0`; their number is asserted to be 720. Each is mapped to its
physical key (k, q, r, z_response) through 06a's `state/rows.json` (the A3 v2 row's `k`, `q`, `r`
and `z_response`, parsed from their repr strings with `float`). A rehearsal row is "among them" iff
its own physical key, read from the store, is equal as floats to one of theirs. Both input files
are read only, and their SHA-256 printed.

**R1** (prompt 06m §4): every row outside 06a's 720 stationary-point rows with finite
`total_abserr` in both stores has |total_26 - total_25| <= total_abserr_25 + total_abserr_26,
computed in float64 as written (`abs(b - a) <= ea + eb`). Rows whose `total_abserr` is infinite (or
not finite, or NULL) in either store are counted separately, by store. The stationary-point rows
are tabulated, not scored. A violating row is printed with both totals and both `total_abserr` in
`float.hex`.
**R1 on the oracle**, the same rule with `analytic_rad` (column, Float(64)) for the value and
`analytic.abserr` (the `metadata` JSON's top-level `analytic` object's `abserr`, a JSON number,
json.dumps of a Python float) for the error: |rad_26 - rad_25| <= abserr_25 + abserr_26 on every
row outside the 720 with both errors finite (and both values finite); non-finite or absent in
either counted separately; the stationary-point rows tabulated, not scored.

    python -I tabulate_qsi_signedyardstick.py <R25 primary> <R26 primary> <store_value_diff.json> <out.csv>
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

GROUP_REC = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.")
GROUP_REGIONS = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.regions$")
GROUP_CONV = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.converged$")
GROUP_PL = re.compile(r"^WKB_Levin\.subintervals\[\d+\]\.groups\[\d+\]\.phase_limited$")
ANALYTIC_REGIONS = re.compile(r"^analytic\.(0pt5|2pt5)(\.Levin)?\.[JY][1-4]_data\.regions$")
ANALYTIC_PL = re.compile(r"^analytic\.(0pt5|2pt5)(\.Levin)?\.[JY][1-4]_data\.phase_limited$")
ANALYTIC_CONV = re.compile(r"^analytic\.(0pt5|2pt5)(\.Levin)?\.[JY][1-4]_data\.converged$")

DEPTH_MAX = 20  # AdaptiveLevin DEFAULT_LEVIN_MAX_DEPTH, unchanged by 06c, 06d, 06e, 06h, 06g and 06m

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
CHECKPOINT_06A = MAIN / "var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
ROWS_JSON_06A = MAIN / "var/levin-phase-floor-06a/state/rows.json"
TREE_PARENT = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-5ff8a5e"
TREE_CODE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-38f6a82"
RUN_R25 = "levin-phase-floor-06m-r25-parent-20261106T092420"
RUN_R26 = "levin-phase-floor-06m-r26-signedyardstick-20261106T093937"


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stationary_point_keys():
    """06a's stationary-point rows (scan06a's rule), as A3 v2 names and physical keys."""
    sp = set()
    with open(CHECKPOINT_06A) as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            rk = f"{d['shard']}-{d['serial']}"
            sc = {s["index"]: s["n_sign_changes"] for s in (d.get("scan") or []) if s}
            if any(v > 0 for v in sc.values()):
                sp.add(rk)
    assert len(sp) == 720, len(sp)
    rows = json.load(open(ROWS_JSON_06A))["rows"]
    by_name = {r["name"]: r for r in rows}
    keys = {}
    for name in sp:
        r = by_name[name]
        keys[(float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"]))] = name
    assert len(keys) == 720
    a3_keys = {(float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"])) for r in rows}
    return keys, a3_keys, len(rows)


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


def hx(x):
    return None if x is None else float(x).hex()


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def per_row(rec, flat):
    g_reg = [v for p, v in flat.items() if GROUP_REGIONS.match(p)]
    g_conv = [v for p, v in flat.items() if GROUP_CONV.match(p)]
    g_pl = [v for p, v in flat.items() if GROUP_PL.match(p)]
    an_reg = [v for p, v in flat.items() if ANALYTIC_REGIONS.match(p)]
    an_pl = [v for p, v in flat.items() if ANALYTIC_PL.match(p)]
    an_conv = [v for p, v in flat.items() if ANALYTIC_CONV.match(p)]
    col_reg = rec["WKB_Levin_num_regions"]
    # the column is the sum of the groups' own counts
    if col_reg is None:
        assert not g_reg, ("groups without a WKB_Levin_num_regions", g_reg)
    else:
        assert col_reg == sum(g_reg), ("WKB_Levin_num_regions != sum of group regions", col_reg)
    if g_conv:
        assert bool(rec["total_converged"]) == all(g_conv), "total_converged != AND of groups"
    return {
        "calls": len(g_reg),
        "group_regions": sum(g_reg),
        "group_regions_max": max(g_reg) if g_reg else None,
        "groups_unconverged": sum(1 for v in g_conv if not v),
        "groups_pl": sum(1 for v in g_pl if v),
        "analytic_blocks": len(an_reg),
        "analytic_regions": sum(an_reg),
        "analytic_pl": sum(1 for v in an_pl if v),
        "analytic_unconverged": sum(1 for v in an_conv if not v),
    }


SP_KEYS = {}


def main():
    a_path, b_path, report_path, out_csv = sys.argv[1:5]
    import subprocess

    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    for pth in (a_path, b_path):
        prim = Path(pth)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# store file {f} sha256 {sha256(f)}")
    print(f"# store_value_diff JSON {report_path} sha256 {sha256(report_path)}")
    print(f"# cwd {Path.cwd()}")
    for tree in (TREE_PARENT, TREE_CODE):
        head = subprocess.run(["git", "--no-optional-locks", "-C", tree, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        print(f"# tree {tree} HEAD {head}")
    for tag, run in (("R25", RUN_R25), ("R26", RUN_R26)):
        st = json.load(open(MAIN / "var/runs" / run / "status.json"))
        print(f"# {tag} run {run}: state {st['state']}, exit {st['exit_code']}, fingerprint {st['fingerprint']['digest']}")
    print()
    A, cols_a = read(a_path)
    B, cols_b = read(b_path)
    print(f"rows: R25 {len(A)}, R26 {len(B)}; same (k,q,r,z_response) set: {set(A) == set(B)}")
    assert set(A) == set(B)
    print(f"columns equal: {cols_a == cols_b}; {len(cols_a)} columns")
    print(
        f"tolerances (log10 atol, log10 rtol) R25: "
        f"{sorted(Counter((v['_atol'], v['_rtol']) for v in A.values()).items())}; "
        f"R26: {sorted(Counter((v['_atol'], v['_rtol']) for v in B.values()).items())}"
    )
    tol_same = all((A[k]["_atol"], A[k]["_rtol"]) == (B[k]["_atol"], B[k]["_rtol"]) for k in A)
    print(f"tolerances equal row by row: {tol_same}")
    print(
        "column kinds: "
        + "; ".join(
            f"{k}: {sorted(c for c in cols_a if col_kind(c) == k and c != 'metadata')}"
            for k in ("timing", "label", "identity", "counter", "value")
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
        f"tool: {len(tool)} rows differ, {rep['identical']} identical, only_in_a "
        f"{len(rep['only_in_a'])}, only_in_b {len(rep['only_in_b'])}, unmatched "
        f"{len(rep['unmatched'])}; its keys equal the stores' set: {set(tool) == set(A)}"
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

    global SP_KEYS
    print(f"06a census checkpoint {CHECKPOINT_06A} sha256 {sha256(CHECKPOINT_06A)}")
    print(f"06a rows.json {ROWS_JSON_06A} sha256 {sha256(ROWS_JSON_06A)}")
    sp_all, a3_keys, n_a3 = stationary_point_keys()
    SP_KEYS = {k: v for k, v in sp_all.items() if k in A}
    print(
        f"06a's stationary-point rows: {len(sp_all)} (of A3 v2's {n_a3}); rehearsal rows among them "
        f"(physical key equal as floats): {len(SP_KEYS)}; rehearsal rows equal to any A3 v2 row: "
        f"{len(set(A) & a3_keys)}"
    )
    print(
        f"  rehearsal k {sorted({k[0] for k in A})}; k shared with A3 v2 "
        f"{sorted({k[0] for k in A} & {k[0] for k in a3_keys})}; z_response values: rehearsal "
        f"{len({k[3] for k in A})}, shared with A3 v2 {len({k[3] for k in A} & {k[3] for k in a3_keys})}"
    )
    print()

    rows = []
    field_sets_differ = 0
    value_diff_fields = Counter()
    for key in sorted(A):
        a, b = A[key], B[key]
        fa, ma, flat_a = fields(a)
        fb, mb, flat_b = fields(b)
        if set(fa) != set(fb):
            field_sets_differ += 1
        d_value, d_counter = [], []
        for f in set(fa) | set(fb):
            kind = (fa.get(f) or fb.get(f))[0]
            if kind in ("timing", "label", "identity"):
                continue
            if f not in fa or f not in fb or bits(fa[f][1]) != bits(fb[f][1]):
                (d_counter if kind == "counter" else d_value).append(f)
        value_diff_fields.update(re.sub(r"\[\d+\]", "[]", f) for f in d_value)
        pa, pb = per_row(a, flat_a), per_row(b, flat_b)
        ta, tb = a["total"], b["total"]
        ea, eb = a["total_abserr"], b["total_abserr"]
        dt = abs(tb - ta)
        both_finite = finite(ea) and finite(eb)
        in_sp = key in SP_KEYS
        if in_sp:
            y1 = "stationary-point-row (not scored)"
            y1_ratio = dt / (ea + eb) if both_finite and (ea + eb) > 0 else None
        elif both_finite:
            y1 = "holds" if dt <= ea + eb else "VIOLATES"
            y1_ratio = dt / (ea + eb) if (ea + eb) > 0 else (0.0 if dt == 0 else math.inf)
        else:
            y1 = "nonfinite:" + ("25" if not finite(ea) else "") + ("26" if not finite(eb) else "")
            y1_ratio = None
        an_a = ma.get("analytic", {}).get("abserr")
        an_b = mb.get("analytic", {}).get("abserr")
        ra, rb = a["analytic_rad"], b["analytic_rad"]
        o_fin = finite(an_a) and finite(an_b) and finite(ra) and finite(rb)
        o_dt = abs(rb - ra) if finite(ra) and finite(rb) else None
        if in_sp:
            v1o = "stationary-point-row (not scored)"
            v1o_ratio = o_dt / (an_a + an_b) if o_fin and (an_a + an_b) > 0 else None
        elif o_fin:
            v1o = "holds" if o_dt <= an_a + an_b else "VIOLATES"
            v1o_ratio = o_dt / (an_a + an_b) if (an_a + an_b) > 0 else (0.0 if o_dt == 0 else math.inf)
        else:
            v1o = (
                "nonfinite:"
                + ("25" if not (finite(an_a) and finite(ra)) else "")
                + ("26" if not (finite(an_b) and finite(rb)) else "")
            )
            v1o_ratio = None
        rows.append(
            {
                "k": key[0],
                "q": key[1],
                "r": key[2],
                "z_response": key[3],
                "log10_atol": a["_atol"],
                "log10_rtol": a["_rtol"],
                "levin_calls_R25": pa["calls"],
                "levin_calls_R26": pb["calls"],
                "total_R25": hx(ta),
                "total_R26": hx(tb),
                "total_bit_identical": hx(ta) == hx(tb),
                "all_value_fields_bit_identical": not d_value,
                "value_fields_differing": len(d_value),
                "counter_fields_differing": len(d_counter),
                "abs_dtotal": dt,
                "rel_dtotal": dt / abs(ta) if ta else (0.0 if dt == 0 else math.inf),
                "total_abserr_R25": ea,
                "total_abserr_R26": eb,
                "total_abserr_R25_hex": hx(ea),
                "total_abserr_R26_hex": hx(eb),
                "abserr_ratio_R26_R25": (eb / ea) if (finite(ea) and finite(eb) and ea) else None,
                "dtotal_over_abserr_R25": (dt / ea) if (finite(ea) and ea) else None,
                "dtotal_over_abserr_R26": (dt / eb) if (finite(eb) and eb) else None,
                "stationary_point_row_06a": SP_KEYS.get(key, ""),
                "R1": y1,
                "R1_ratio_dtotal_over_sum": y1_ratio,
                "total_converged_R25": a["total_converged"],
                "total_converged_R26": b["total_converged"],
                "total_phase_limited_R25": a["total_phase_limited"],
                "total_phase_limited_R26": b["total_phase_limited"],
                "WKB_Levin_num_regions_R25": a["WKB_Levin_num_regions"],
                "WKB_Levin_num_regions_R26": b["WKB_Levin_num_regions"],
                "WKB_Levin_max_depth_R25": a["WKB_Levin_max_depth"],
                "WKB_Levin_max_depth_R26": b["WKB_Levin_max_depth"],
                "WKB_Levin_evaluations_R25": a["WKB_Levin_evaluations"],
                "WKB_Levin_evaluations_R26": b["WKB_Levin_evaluations"],
                "group_regions_max_R25": pa["group_regions_max"],
                "group_regions_max_R26": pb["group_regions_max"],
                "groups_unconverged_R25": pa["groups_unconverged"],
                "groups_unconverged_R26": pb["groups_unconverged"],
                "groups_phase_limited_R25": pa["groups_pl"],
                "groups_phase_limited_R26": pb["groups_pl"],
                "analytic_blocks": pa["analytic_blocks"],
                "analytic_regions_R25": pa["analytic_regions"],
                "analytic_regions_R26": pb["analytic_regions"],
                "analytic_blocks_phase_limited_R25": pa["analytic_pl"],
                "analytic_blocks_phase_limited_R26": pb["analytic_pl"],
                "analytic_blocks_unconverged_R25": pa["analytic_unconverged"],
                "analytic_blocks_unconverged_R26": pb["analytic_unconverged"],
                "analytic_rad_R25": hx(a["analytic_rad"]),
                "analytic_rad_R26": hx(b["analytic_rad"]),
                "analytic_rad_bit_identical": hx(a["analytic_rad"]) == hx(b["analytic_rad"]),
                "analytic_abserr_R25": an_a,
                "analytic_abserr_R26": an_b,
                "analytic_abserr_R25_hex": hx(an_a) if finite(an_a) else repr(an_a),
                "analytic_abserr_R26_hex": hx(an_b) if finite(an_b) else repr(an_b),
                "analytic_abserr_bit_identical": (
                    finite(an_a) and finite(an_b) and hx(an_a) == hx(an_b)
                ),
                "abs_d_analytic_rad": o_dt,
                "d_analytic_rad_over_abserr_R25": (o_dt / an_a) if o_dt is not None and finite(an_a) and an_a else None,
                "d_analytic_rad_over_abserr_R26": (o_dt / an_b) if o_dt is not None and finite(an_b) and an_b else None,
                "d_analytic_rad_over_sum_analytic_abserr": (
                    abs(b["analytic_rad"] - a["analytic_rad"]) / (an_a + an_b)
                    if finite(an_a) and finite(an_b) and (an_a + an_b) > 0
                    else None
                ),
                "R1_oracle": v1o,
                "R1_oracle_ratio_d_over_sum": v1o_ratio,
                "analytic_converged_R25": ma.get("analytic", {}).get("converged"),
                "analytic_converged_R26": mb.get("analytic", {}).get("converged"),
            }
        )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)

    def g(x, f="{:.3e}"):
        return "-" if x is None else f.format(x)

    def q(vals, ps=(0.0, 0.5, 0.9, 0.99, 1.0)):
        v = sorted(vals)
        if not v:
            return "none"
        return ", ".join(f"p{int(p * 100)} {v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]:.3g}" for p in ps)

    lev = [r for r in rows if r["levin_calls_R25"] or r["levin_calls_R26"]]
    nolev = [r for r in rows if not (r["levin_calls_R25"] or r["levin_calls_R26"])]
    print(f"rows whose column/metadata field sets differ between R25 and R26: {field_sets_differ}")
    print(
        f"rows with a WKB Levin call: {len(lev)} (calls R25 {sum(r['levin_calls_R25'] for r in rows)}, "
        f"R26 {sum(r['levin_calls_R26'] for r in rows)}; rows whose call count differs: "
        f"{sum(1 for r in rows if r['levin_calls_R25'] != r['levin_calls_R26'])}); with none: {len(nolev)}"
    )
    print()
    print("== bit-identity (float.hex), every row")
    print(
        f"  total bit-identical: {sum(r['total_bit_identical'] for r in rows)}/{n} "
        f"(rows with a Levin call {sum(r['total_bit_identical'] for r in lev)}/{len(lev)}, "
        f"rows with none {sum(r['total_bit_identical'] for r in nolev)}/{len(nolev)})"
    )
    print(
        f"  total_abserr bit-identical: "
        f"{sum(r['total_abserr_R25_hex'] == r['total_abserr_R26_hex'] for r in rows)}/{n}"
    )
    print(
        f"  every value field (columns and metadata, timings/labels/identity/counters excluded) "
        f"bit-identical: {sum(r['all_value_fields_bit_identical'] for r in rows)}/{n} "
        f"(rows with a Levin call {sum(r['all_value_fields_bit_identical'] for r in lev)}, "
        f"rows with none {sum(r['all_value_fields_bit_identical'] for r in nolev)})"
    )
    print(
        f"  rows with no Levin call whose value fields differ: "
        f"{sum(1 for r in nolev if not r['all_value_fields_bit_identical'])}"
    )
    print(
        f"  analytic_rad bit-identical: {sum(r['analytic_rad_bit_identical'] for r in rows)}/{n}; "
        f"max |d analytic_rad|/(analytic abserr25+26) "
        f"{g(max((r['d_analytic_rad_over_sum_analytic_abserr'] for r in rows if r['d_analytic_rad_over_sum_analytic_abserr'] is not None), default=None))}"
    )
    print(f"  value fields differing, by field (indices folded), rows: {dict(value_diff_fields.most_common())}")
    print()
    print(
        "== R1: |total_26 - total_25| <= total_abserr_25 + total_abserr_26, rows outside 06a's 720 "
        "stationary-point rows with both finite"
    )
    sprows = [r for r in rows if r["stationary_point_row_06a"]]
    fin = [r for r in rows if r["R1"] in ("holds", "VIOLATES")]
    viol = [r for r in fin if r["R1"] == "VIOLATES"]
    nonf = Counter(r["R1"] for r in rows if r["R1"].startswith("nonfinite"))
    assert len(sprows) + len(fin) + sum(nonf.values()) == n
    print(f"  rows among 06a's stationary-point rows (tabulated, not scored): {len(sprows)}")
    print(f"  rows scored (outside them, finite total_abserr in both): {len(fin)}; violations: {len(viol)}")
    print(f"  rows outside them with a non-finite total_abserr, counted separately: {dict(nonf) or 'none'}")
    for r in sprows:
        print(
            f"    stationary-point row {r['stationary_point_row_06a']}: k={r['k']:.6g} q={r['q']:.6g} "
            f"r={r['r']:.6g} z={r['z_response']:.8g}: total25 {r['total_R25']} total26 {r['total_R26']} "
            f"abserr25 {r['total_abserr_R25_hex']} abserr26 {r['total_abserr_R26_hex']}"
        )
    for tag in ("R25", "R26"):
        cand = [r for r in fin if r[f"dtotal_over_abserr_{tag}"] is not None]
        if cand:
            w = max(cand, key=lambda r: r[f"dtotal_over_abserr_{tag}"])
            print(
                f"  largest |dT|/total_abserr_{tag[1:]} over the scored rows: "
                f"{w[f'dtotal_over_abserr_{tag}']:.6g} at k={w['k']:.6g} q={w['q']:.6g} r={w['r']:.6g} "
                f"z={w['z_response']:.8g} (total25 {w['total_R25']} total26 {w['total_R26']} abserr25 "
                f"{w['total_abserr_R25_hex']} abserr26 {w['total_abserr_R26_hex']})"
            )
    print(
        f"  |dT|/(abserr25+abserr26) over the finite rows: {q([r['R1_ratio_dtotal_over_sum'] for r in fin])}"
    )
    print(
        f"  over the finite rows whose total moved: "
        f"{q([r['R1_ratio_dtotal_over_sum'] for r in fin if not r['total_bit_identical']])}"
    )
    for r in viol:
        print(
            f"    VIOLATION k={r['k']:.6g} q={r['q']:.6g} r={r['r']:.6g} z={r['z_response']:.8g}: "
            f"total25 {r['total_R25']} total26 {r['total_R26']} abserr25 {r['total_abserr_R25_hex']} "
            f"abserr26 {r['total_abserr_R26_hex']} ratio {r['R1_ratio_dtotal_over_sum']:.6g}"
        )
    for r in rows:
        if r["R1"].startswith("nonfinite"):
            print(
                f"    non-finite {r['R1']}: k={r['k']:.6g} q={r['q']:.6g} r={r['r']:.6g} "
                f"z={r['z_response']:.8g}: total25 {r['total_R25']} total26 {r['total_R26']} "
                f"abserr25 {r['total_abserr_R25_hex']} abserr26 {r['total_abserr_R26_hex']} "
                f"conv {r['total_converged_R25']}>{r['total_converged_R26']} "
                f"depth {r['WKB_Levin_max_depth_R25']}>{r['WKB_Levin_max_depth_R26']} "
                f"regions {r['WKB_Levin_num_regions_R25']}>{r['WKB_Levin_num_regions_R26']}"
            )
    print()
    print("== total_abserr and the totals, rows with a Levin call")
    print(f"  abserr26/abserr25 (both finite, nonzero): {q([r['abserr_ratio_R26_R25'] for r in lev if r['abserr_ratio_R26_R25'] is not None])}")
    print(f"  |dT|/abserr25: {q([r['dtotal_over_abserr_R25'] for r in lev if r['dtotal_over_abserr_R25'] is not None])}")
    print(f"  |dT|/abserr26: {q([r['dtotal_over_abserr_R26'] for r in lev if r['dtotal_over_abserr_R26'] is not None])}")
    print(f"  |dT|/|T25|: {q([r['rel_dtotal'] for r in lev])}")
    print(
        f"  abserr26 > abserr25: {sum(1 for r in lev if finite(r['total_abserr_R25']) and finite(r['total_abserr_R26']) and r['total_abserr_R26'] > r['total_abserr_R25'])}; "
        f"abserr26 < abserr25: {sum(1 for r in lev if finite(r['total_abserr_R25']) and finite(r['total_abserr_R26']) and r['total_abserr_R26'] < r['total_abserr_R25'])}"
    )
    print()
    print("== total_converged, R25 -> R26 (every row)")
    print(f"  {dict(Counter((r['total_converged_R25'], r['total_converged_R26']) for r in rows))}")
    print(
        f"  converged R25 {sum(1 for r in rows if r['total_converged_R25'])}, "
        f"R26 {sum(1 for r in rows if r['total_converged_R26'])} of {n}; "
        f"Levin calls unconverged R25 {sum(r['groups_unconverged_R25'] for r in rows)}, "
        f"R26 {sum(r['groups_unconverged_R26'] for r in rows)}"
    )
    for frm, to, label in ((1, 0, "converged -> unconverged"), (0, 1, "unconverged -> converged")):
        ch = [r for r in rows if r["total_converged_R25"] == frm and r["total_converged_R26"] == to]
        print(f"  rows whose total_converged went {label}: {len(ch)}")
        for r in ch:
            print(
                f"    k={r['k']:.6g} q={r['q']:.6g} r={r['r']:.6g} z={r['z_response']:.8g} "
                f"(log10 atol, rtol) ({r['log10_atol']}, {r['log10_rtol']}): Levin calls unconverged "
                f"{r['groups_unconverged_R25']} -> {r['groups_unconverged_R26']}; regions "
                f"{r['WKB_Levin_num_regions_R25']} -> {r['WKB_Levin_num_regions_R26']}; max depth "
                f"{r['WKB_Levin_max_depth_R25']} -> {r['WKB_Levin_max_depth_R26']}; total "
                f"{r['total_R25']} -> {r['total_R26']}; abserr {r['total_abserr_R25_hex']} -> "
                f"{r['total_abserr_R26_hex']}; R1 {r['R1']} ratio {g(r['R1_ratio_dtotal_over_sum'], '{:.4g}')}"
            )
    print(f"  total_phase_limited R25 -> R26: {dict(Counter((r['total_phase_limited_R25'], r['total_phase_limited_R26']) for r in rows))}")
    print(
        f"  Levin calls phase-limited R25 {sum(r['groups_phase_limited_R25'] for r in rows)}, "
        f"R26 {sum(r['groups_phase_limited_R26'] for r in rows)}"
    )
    print()
    print("== region counts and maximum depths (rows with a Levin call)")
    print(
        f"  WKB_Levin_num_regions summed: R25 {sum(r['WKB_Levin_num_regions_R25'] or 0 for r in rows)}, "
        f"R26 {sum(r['WKB_Levin_num_regions_R26'] or 0 for r in rows)}; "
        f"evaluations R25 {sum(r['WKB_Levin_evaluations_R25'] or 0 for r in rows)}, "
        f"R26 {sum(r['WKB_Levin_evaluations_R26'] or 0 for r in rows)}"
    )
    print(
        f"  rows: regions equal {sum(1 for r in lev if r['WKB_Levin_num_regions_R25'] == r['WKB_Levin_num_regions_R26'])}, "
        f"more in R26 {sum(1 for r in lev if (r['WKB_Levin_num_regions_R26'] or 0) > (r['WKB_Levin_num_regions_R25'] or 0))}, "
        f"fewer in R26 {sum(1 for r in lev if (r['WKB_Levin_num_regions_R26'] or 0) < (r['WKB_Levin_num_regions_R25'] or 0))}"
    )
    print(f"  regions R26/R25 per row: {q([r['WKB_Levin_num_regions_R26'] / r['WKB_Levin_num_regions_R25'] for r in lev if r['WKB_Levin_num_regions_R25']])}")
    print(f"  WKB_Levin_max_depth R25: {dict(sorted(Counter(r['WKB_Levin_max_depth_R25'] for r in lev).items()))}")
    print(f"  WKB_Levin_max_depth R26: {dict(sorted(Counter(r['WKB_Levin_max_depth_R26'] for r in lev).items()))}")
    print(
        f"  rows at depth_max ({DEPTH_MAX}): R25 {sum(1 for r in lev if r['WKB_Levin_max_depth_R25'] == DEPTH_MAX)}, "
        f"R26 {sum(1 for r in lev if r['WKB_Levin_max_depth_R26'] == DEPTH_MAX)}"
    )
    print(
        f"  max depth: equal {sum(1 for r in lev if r['WKB_Levin_max_depth_R25'] == r['WKB_Levin_max_depth_R26'])}, "
        f"deeper in R26 {sum(1 for r in lev if (r['WKB_Levin_max_depth_R26'] or 0) > (r['WKB_Levin_max_depth_R25'] or 0))}, "
        f"shallower {sum(1 for r in lev if (r['WKB_Levin_max_depth_R26'] or 0) < (r['WKB_Levin_max_depth_R25'] or 0))}"
    )
    print(
        f"  analytic-block regions summed: R25 {sum(r['analytic_regions_R25'] for r in rows)}, "
        f"R26 {sum(r['analytic_regions_R26'] for r in rows)} (of {sum(r['analytic_blocks'] for r in rows)} blocks); "
        f"blocks phase-limited R25 {sum(r['analytic_blocks_phase_limited_R25'] for r in rows)}, "
        f"R26 {sum(r['analytic_blocks_phase_limited_R26'] for r in rows)}; "
        f"blocks unconverged R25 {sum(r['analytic_blocks_unconverged_R25'] for r in rows)}, "
        f"R26 {sum(r['analytic_blocks_unconverged_R26'] for r in rows)}"
    )
    print()
    print("== by tolerance, rows with a Levin call")
    for tol in sorted(set((r["log10_atol"], r["log10_rtol"]) for r in lev)):
        s = [r for r in lev if (r["log10_atol"], r["log10_rtol"]) == tol]
        print(
            f"  (log10 atol, log10 rtol) {tol}: rows {len(s)}; total moved {sum(1 for r in s if not r['total_bit_identical'])}; "
            f"conv R25 {sum(1 for r in s if r['total_converged_R25'])} R26 {sum(1 for r in s if r['total_converged_R26'])}; "
            f"R1 violations {sum(1 for r in s if r['R1'] == 'VIOLATES')}; non-finite {sum(1 for r in s if r['R1'].startswith('nonfinite'))}; "
            f"regions R25 {sum(r['WKB_Levin_num_regions_R25'] or 0 for r in s)} R26 {sum(r['WKB_Levin_num_regions_R26'] or 0 for r in s)}; "
            f"depth20 R25 {sum(1 for r in s if r['WKB_Levin_max_depth_R25'] == DEPTH_MAX)} R26 {sum(1 for r in s if r['WKB_Levin_max_depth_R26'] == DEPTH_MAX)}"
        )
    print()
    print("== R1 on the oracle: |rad_26 - rad_25| <= analytic.abserr_25 + analytic.abserr_26, rows outside 06a's 720 stationary-point rows with both finite")
    o_fin = [r for r in rows if r["R1_oracle"] in ("holds", "VIOLATES")]
    o_viol = [r for r in o_fin if r["R1_oracle"] == "VIOLATES"]
    o_nonf = Counter(r["R1_oracle"] for r in rows if r["R1_oracle"].startswith("nonfinite"))
    o_sp = [r for r in rows if r["R1_oracle"].startswith("stationary")]
    assert len(o_sp) + len(o_fin) + sum(o_nonf.values()) == n
    print(f"  rows among 06a's stationary-point rows (tabulated, not scored): {len(o_sp)}")
    print(f"  rows scored: {len(o_fin)}; violations: {len(o_viol)}")
    print(f"  rows outside them with a non-finite or absent oracle value or error, counted separately: {dict(o_nonf) or 'none'}")
    print(f"  analytic_rad bit-identical over the scored rows: {sum(r['analytic_rad_bit_identical'] for r in o_fin)}/{len(o_fin)}; analytic.abserr bit-identical: {sum(r['analytic_abserr_bit_identical'] for r in o_fin)}/{len(o_fin)}")
    for r in o_sp:
        print(
            f"    stationary-point row {r['stationary_point_row_06a']}: rad25 {r['analytic_rad_R25']} rad26 {r['analytic_rad_R26']} "
            f"abserr25 {r['analytic_abserr_R25_hex']} abserr26 {r['analytic_abserr_R26_hex']} ratio {r['R1_oracle_ratio_d_over_sum']}"
        )
    if o_fin:
        w = max(o_fin, key=lambda r: r["R1_oracle_ratio_d_over_sum"])
        print(
            f"  largest |d rad|/(abserr25+abserr26) over the scored rows: {w['R1_oracle_ratio_d_over_sum']:.6g} at k={w['k']:.6g} "
            f"q={w['q']:.6g} r={w['r']:.6g} z={w['z_response']:.8g} (rad25 {w['analytic_rad_R25']} rad26 {w['analytic_rad_R26']} "
            f"abserr25 {w['analytic_abserr_R25_hex']} abserr26 {w['analytic_abserr_R26_hex']})"
        )
        for tag in ("R25", "R26"):
            cand = [r for r in o_fin if r[f"d_analytic_rad_over_abserr_{tag}"] is not None]
            if cand:
                w = max(cand, key=lambda r: r[f"d_analytic_rad_over_abserr_{tag}"])
                print(f"  largest |d rad|/analytic.abserr_{tag[1:]} over the scored rows: {w[f'd_analytic_rad_over_abserr_{tag}']:.6g}")
    print(f"  |d rad|/(abserr25+abserr26) over the scored rows: {q([r['R1_oracle_ratio_d_over_sum'] for r in o_fin])}")
    print(f"  over the scored rows whose analytic_rad moved: {q([r['R1_oracle_ratio_d_over_sum'] for r in o_fin if not r['analytic_rad_bit_identical']])}")
    for r in o_viol:
        print(
            f"    VIOLATION (oracle) k={r['k']:.6g} q={r['q']:.6g} r={r['r']:.6g} z={r['z_response']:.8g}: "
            f"rad25 {r['analytic_rad_R25']} rad26 {r['analytic_rad_R26']} abserr25 {r['analytic_abserr_R25_hex']} "
            f"abserr26 {r['analytic_abserr_R26_hex']} ratio {r['R1_oracle_ratio_d_over_sum']:.6g}"
        )
    for r in rows:
        if r["R1_oracle"].startswith("nonfinite"):
            print(
                f"    non-finite (oracle) {r['R1_oracle']}: k={r['k']:.6g} q={r['q']:.6g} r={r['r']:.6g} z={r['z_response']:.8g}: "
                f"rad25 {r['analytic_rad_R25']} rad26 {r['analytic_rad_R26']} abserr25 {r['analytic_abserr_R25_hex']} abserr26 {r['analytic_abserr_R26_hex']}"
            )
    print(f"  analytic.converged R25 -> R26: {dict(Counter((str(r['analytic_converged_R25']), str(r['analytic_converged_R26'])) for r in rows))}")
    print()
    print(f"VERDICT R1 (oracle): {'holds' if not o_viol else 'VIOLATED'} ({len(o_viol)} violations of {len(o_fin)} scored rows; {sum(o_nonf.values())} rows non-finite in either, counted separately; {len(o_sp)} stationary-point rows tabulated, not scored)")
    print(f"VERDICT R1 (total): {'holds' if not viol else 'VIOLATED'} ({len(viol)} violations of {len(fin)} scored rows (outside 06a's stationary-point rows, finite total_abserr in both); {sum(nonf.values())} rows non-finite in either, counted separately; {len(sprows)} stationary-point rows tabulated, not scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
