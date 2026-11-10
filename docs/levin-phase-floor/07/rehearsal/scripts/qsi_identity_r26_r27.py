"""R27 against R26 inside QuadSourceIntegral: identity by bit pattern over every row and field.

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.3 (S0, "R27 against 06m's R26").
06m's rehearsal agent's qsi_identity_r24_r25.py (e63aae0c...), with R24 -> R26 and R25 -> R27 by
sed, this paragraph, and a provenance header naming both worktrees' HEAD and status. In 06m: new; the
store reader (`read`), `flatten`, `bits`, `col_kind`, `leaf_kind` and `fields` are copied verbatim from
06g's docs/levin-phase-floor/06g/rehearsal/scripts/tabulate_qsi_leafvalues.py (3268a0bc...), so
the field classification is 06g's (and 06h's, 06e's, 06c's, 06b's):
- every column of table QuadSourceIntegral, in every shard, read with sqlite `mode=ro`, and every
  leaf of its `metadata` JSON (flattened to a dotted path);
- rows matched by their physical key (k, q, r, z_response) through the store's own wavenumber and
  redshift tables, with the tolerances (log10 atol, log10 rtol) carried and compared;
- a field is `timing` (a column ending compute_time / RHS_time / elapsed, or a metadata leaf
  `elapsed` / `compute_time`), `label`, `identity` (serial, *_serial, timestamp), `counter` (06g's
  work counters) or `value` (everything else).

Identity is by `bits`: a float compares by `float.hex`, never isclose; anything else by type and
value. Every value field and every counter field must be bit-identical in every row, and both
stores must have the same rows and the same field sets. Named checks: `total`, `total_abserr`,
`analytic_rad` (the oracle's value), `metadata` `analytic.abserr` (the oracle's error),
`total_converged`.

    ./venv/bin/python -I -B qsi_identity_r24_r25.py <R26 primary> <R27 primary>
"""

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
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035",  # code worktree, 74600350
)
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


def main():
    a_path, b_path = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    for p in (a_path, b_path):
        prim = Path(p)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# input {f} sha256 {sha256(f)}")
    print(f"# cwd {Path.cwd()}")
    for t in TREES:
        head = subprocess.run(["git", "--no-optional-locks", "-C", t, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "--no-optional-locks", "-C", t, "status", "--short"],
                            capture_output=True, text=True).stdout
        print(f"# tree {t} HEAD {head} status --short {st!r}")
    print("# this script imports no tree code; it reads the stores with sqlite3 mode=ro")
    A, cols_a = read(a_path)
    B, cols_b = read(b_path)
    print(f"A = R26 {a_path}: {len(A)} rows, {len(cols_a)} columns")
    print(f"B = R27 {b_path}: {len(B)} rows, {len(cols_b)} columns")
    print(f"columns equal: {cols_a == cols_b}")
    print(f"key sets equal: {set(A) == set(B)}; only in A {len(set(A) - set(B))}, only in B {len(set(B) - set(A))}")
    diff_by_kind = Counter()
    diff_fields = Counter()
    rows_value_identical = 0
    rows_counter_identical = 0
    named = {"col:total": 0, "col:total_abserr": 0, "col:analytic_rad": 0, "meta:analytic.abserr": 0,
             "col:total_converged": 0}
    named_present = Counter()
    fieldset_mismatch = 0
    tol_mismatch = 0
    nonfloat_named = Counter()
    first_value_diffs = []
    n_fields_compared = Counter()
    for key in sorted(set(A) & set(B)):
        a, b = A[key], B[key]
        if (a["_atol"], a["_rtol"]) != (b["_atol"], b["_rtol"]):
            tol_mismatch += 1
        fa, _, _ = fields(a)
        fb, _, _ = fields(b)
        if set(fa) != set(fb):
            fieldset_mismatch += 1
        vbad = cbad = False
        for f in set(fa) | set(fb):
            ka, va = fa.get(f, (None, "<absent>"))
            kb, vb = fb.get(f, (None, "<absent>"))
            kind = ka or kb
            n_fields_compared[kind] += 1
            same = bits(va) == bits(vb)
            if f in named:
                named_present[f] += 1
                if same:
                    named[f] += 1
                if va is not None and not isinstance(va, (float, bool, int)):
                    nonfloat_named[f] += 1
            if not same:
                diff_by_kind[kind] += 1
                diff_fields[(kind, re.sub(r"\[\d+\]", "[i]", f))] += 1
                if kind == "value":
                    vbad = True
                    if len(first_value_diffs) < 20:
                        first_value_diffs.append((key, f, va, vb))
                if kind == "counter":
                    cbad = True
        rows_value_identical += not vbad
        rows_counter_identical += not cbad
    n = len(set(A) & set(B))
    print(f"rows compared: {n}; tolerance mismatches {tol_mismatch}; field-set mismatches {fieldset_mismatch}")
    print(f"fields compared, by kind: {dict(n_fields_compared)}")
    print(f"field differences, by kind: {dict(diff_by_kind)}")
    print("differing fields (kind, path with indices folded): count of row-field differences")
    for (k, f), c in sorted(diff_fields.items()):
        print(f"  {k:8s} {f}: {c}")
    print(f"rows with every value field bit-identical: {rows_value_identical} / {n}")
    print(f"rows with every counter field bit-identical: {rows_counter_identical} / {n}")
    for f, c in named.items():
        print(f"{f} bit-identical: {c} / {named_present[f]} rows where present (non-numeric type: {nonfloat_named[f]})")
    if first_value_diffs:
        print("FIRST VALUE DIFFERENCES:")
        for key, f, va, vb in first_value_diffs:
            hx = lambda x: x.hex() if isinstance(x, float) else repr(x)
            print(f"  {key} {f}: R26 {hx(va)} R27 {hx(vb)}")
    ok = (cols_a == cols_b and set(A) == set(B) and tol_mismatch == 0 and fieldset_mismatch == 0
          and diff_by_kind.get("value", 0) == 0 and diff_by_kind.get("counter", 0) == 0)
    print(f"VERDICT: R27 {'IS' if ok else 'IS NOT'} value-identical to R26 in QuadSourceIntegral "
          f"(every value and counter field, every row, by bit pattern)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
