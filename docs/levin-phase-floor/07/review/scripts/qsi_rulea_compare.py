"""Orchestrator's own read of QuadSourceIntegral: R29 (Rule A) against R28 and R27.

Usage: python -I -B qsi_rulea_compare.py R27_PRIMARY R28_PRIMARY R29_PRIMARY OUT_CSV

levin-phase-floor 07 Part B, section B.7 item 2 (planned stop 1'). Read only (sqlite mode=ro),
through qsi_full_compare.read (06m's orchestrator reader, imported from this directory): rows keyed
by (k, q, r, z_response, log10 atol, log10 rtol), every column but store bookkeeping and timing,
`metadata` flattened leaf by leaf with timing leaves dropped.

A Levin record is the set of leaves under a prefix that carries the four counts in R28 (a group
record `.groups[j]` or an analytic `*_data` record under `.Levin.`). The rule tested (B.7 item 2):
  - a Levin record of R29 may differ from R28's (any leaf, the four counts included) only if R28's
    stationary_points > 0 and every one of R29's four counts is 0; such a record must equal R27's
    bit for bit in value, abserr, converged, phase_limited and regions (and is also compared with
    R27 on every other leaf R27 has);
  - every other Levin record of R29 is bit-identical to R28's, the four counts included;
  - a row whose fields outside its Levin records differ between R29 and R28 must hold a changed
    record.
"""

import collections
import csv
import hashlib
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qsi_full_compare import read, same  # noqa: E402

COUNTS = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
MUST_EQUAL_R27 = ("value", "abserr", "converged", "phase_limited", "regions")


def lastkey(path):
    return re.split(r"[.\[]", path)[-1].rstrip("]")


def record_prefixes(row):
    return sorted(
        {f.rsplit(".", 1)[0] for f in row if f.startswith("metadata.") and lastkey(f) in COUNTS}
    )


def leaves(row, pfx):
    return {f[len(pfx) + 1 :]: v for f, v in row.items() if f.startswith(pfx + ".")}


def main():
    p27, p28, p29, out_csv = sys.argv[1:5]
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"# qsi_rulea_compare.py sha256 {me}")
    for p in (p27, p28, p29):
        print(f"# {p} sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()}")
    a, b, c = read(p27), read(p28), read(p29)
    print(f"rows R27 {len(a)} R28 {len(b)} R29 {len(c)}; keys equal {set(a) == set(b) == set(c)}")
    if not (set(a) == set(b) == set(c)):
        print("KEY SETS DIFFER; stopping")
        return 1

    n_rec = n_rec_identical = 0
    changed = []  # (key, pfx, label, sp28, r27_equal_on_must, r27_equal_all, diffs)
    violations = []
    layout_problems = 0
    rows_out = []
    row_violations = []
    for key in sorted(b):
        x, y, z = a[key], b[key], c[key]
        pf28, pf29 = record_prefixes(y), record_prefixes(z)
        if pf28 != pf29:
            layout_problems += 1
            print("RECORD LAYOUT DIFFERS", key, sorted(set(pf28) ^ set(pf29))[:4])
        row_changed = []
        in_records = set()
        for pfx in pf28:
            n_rec += 1
            ly, lz, lx = leaves(y, pfx), leaves(z, pfx), leaves(x, pfx)
            in_records |= {pfx + "." + k for k in set(ly) | set(lz)}
            diffs = sorted(k for k in set(ly) | set(lz) if not same(ly.get(k, "<absent>"), lz.get(k, "<absent>")))
            if not diffs:
                n_rec_identical += 1
                continue
            sp28 = ly.get("stationary_points")
            counts29 = [lz.get(k) for k in COUNTS]
            ok_shape = isinstance(sp28, int) and sp28 > 0 and all(v == 0 for v in counts29)
            eq_must = all(same(lx.get(k, "<absent>"), lz.get(k, "<absent>")) for k in MUST_EQUAL_R27)
            r27_diff_all = sorted(
                k for k in lx if not same(lx[k], lz.get(k, "<absent>"))
            )
            extra29 = sorted(k for k in lz if k not in lx and k not in COUNTS)
            rec = dict(
                key=key, pfx=pfx, label=ly.get("label"), sp28=sp28, wp28=ly.get("window_pieces"),
                regions27=lx.get("regions"), regions28=ly.get("regions"), regions29=lz.get("regions"),
                eq_r27_must=eq_must, r27_diff_all=r27_diff_all, extra29=extra29, diffs=diffs,
            )
            changed.append(rec)
            row_changed.append(rec)
            if not (ok_shape and eq_must):
                violations.append(rec)
        # fields outside the Levin records
        outside = sorted(
            f for f in set(y) | set(z)
            if f not in in_records and not same(y.get(f, "<absent>"), z.get(f, "<absent>"))
        )
        if outside and not row_changed:
            row_violations.append((key, outside[:8]))
        if row_changed or outside:
            t27, t28, t29 = x["total"], y["total"], z["total"]
            e27, e28, e29 = x["total_abserr"], y["total_abserr"], z["total_abserr"]
            d28 = abs(Fraction(t29) - Fraction(t28))
            d27 = abs(Fraction(t29) - Fraction(t27))
            rows_out.append(
                dict(
                    k=key[0], q=key[1], r=key[2], z=key[3],
                    changed_records=";".join(r["pfx"].replace("metadata.", "") for r in row_changed),
                    total_27=t27.hex(), total_28=t28.hex(), total_29=t29.hex(),
                    abserr_27=e27.hex(), abserr_28=e28.hex(), abserr_29=e29.hex(),
                    total_29_eq_27=same(t29, t27), total_29_eq_28=same(t29, t28),
                    d29_28_over_abserr28=float(d28 / Fraction(e28)) if e28 else float("inf"),
                    d29_28_over_abserr29=float(d28 / Fraction(e29)) if e29 else float("inf"),
                    d29_27_over_abserr27=float(d27 / Fraction(e27)) if e27 else float("inf"),
                    d29_27_over_abserr29=float(d27 / Fraction(e29)) if e29 else float("inf"),
                    conv_27=x["total_converged"], conv_28=y["total_converged"], conv_29=z["total_converged"],
                    nregions_27=x.get("WKB_Levin_num_regions"), nregions_28=y.get("WKB_Levin_num_regions"),
                    nregions_29=z.get("WKB_Levin_num_regions"),
                    outside_fields_differing=len(outside),
                )
            )

    print(f"Levin records {n_rec}; bit-identical to R28 (counts included) {n_rec_identical}; changed {len(changed)}")
    print(f"record-layout problems {layout_problems}")
    lab = collections.Counter((r["label"], r["sp28"]) for r in changed)
    print("changed records by (label, R28 stationary_points):", dict(lab))
    qr_eq = [r for r in changed if r["key"][1] == r["key"][2] and r["label"] == "q-r"]
    print(f"  of which q-r with q = r: {len(qr_eq)}; other: {len(changed) - len(qr_eq)}")
    for r in changed:
        if not (r["key"][1] == r["key"][2] and r["label"] == "q-r"):
            print("  OTHER CHANGED", r["key"][:4], r["pfx"], "sp28", r["sp28"], "eq R27 (must)", r["eq_r27_must"])
    print(f"changed records equal to R27 on {MUST_EQUAL_R27}: {sum(r['eq_r27_must'] for r in changed)} of {len(changed)}")
    print(f"changed records equal to R27 on every leaf R27 has: {sum(not r['r27_diff_all'] for r in changed)} of {len(changed)}")
    other_leaf = collections.Counter(k for r in changed for k in r["r27_diff_all"])
    if other_leaf:
        print("  leaves differing from R27 among changed records:", dict(other_leaf))
    extra = collections.Counter(k for r in changed for k in r["extra29"])
    if extra:
        print("  leaves in R29 not in R27 (counts aside):", dict(extra))
    print(f"VIOLATIONS of the record rule: {len(violations)}")
    for r in violations[:10]:
        print("  VIOLATION", r["key"][:4], r["pfx"], "sp28", r["sp28"], "eq_r27_must", r["eq_r27_must"], "diffs", r["diffs"][:8])
    print(f"rows differing outside their Levin records with no changed record: {len(row_violations)}")
    for rv in row_violations[:10]:
        print("  ROW VIOLATION", rv)
    print(f"rows tabulated {len(rows_out)}")
    if rows_out:
        print(f"  total equal to R27 {sum(r['total_29_eq_27'] for r in rows_out)}; equal to R28 {sum(r['total_29_eq_28'] for r in rows_out)}")
        m = max(rows_out, key=lambda r: r["d29_27_over_abserr29"])
        print(f"  largest |t29 - t27|/abserr29 {m['d29_27_over_abserr29']:.6g} at {(m['k'], m['q'], m['r'], m['z'])}")
        tc = collections.Counter((r["conv_28"], r["conv_29"]) for r in rows_out)
        print(f"  total_converged (R28, R29) among them: {dict(tc)}")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    for tag, s in (("R27", a), ("R28", b), ("R29", c)):
        nr = sum(v.get("WKB_Levin_num_regions") or 0 for v in s.values())
        conv = sum(bool(v["total_converged"]) for v in s.values())
        print(f"{tag}: summed WKB_Levin_num_regions {nr}; total_converged {conv} of {len(s)}")
    return 1 if (violations or row_violations or layout_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
