"""R29 against R28 (and R27) inside QuadSourceIntegral, Levin record by Levin record.

levin-phase-floor 07, rehearsal agent, R29 (prompt 07 section 10, Rule A). New. The reader and field
classification are tabulate_qsi_guard.py's (copied unchanged into this directory, a1eaba31), imported.

For every row (physical key) and every Levin record (group ``WKB_Levin.subintervals[i].groups[j]`` and analytic
block ``analytic.<rad>.Levin.<B>_data``) of R28 and R29 (and R27's record at the same path):
- a record is **unchanged** if every leaf of it other than ``elapsed`` (a timing) is bit-identical between R28 and
  R29 (``float.hex`` for floats; type and value otherwise), the four counts included;
- a **changed** record is **allowed** (the orchestrator's rule for R29) iff R28's ``stationary_points`` > 0, R29's
  four counts are all 0, and R29's record equals R27's bit for bit in value (groups), abserr, converged,
  phase_limited and regions; simple_regions and evaluations (groups) are compared too and reported;
- every changed record is listed with its key, path, label and the three stores' fields; a changed record that is
  not allowed is a STOP, listed first.
Rows: a row whose R28 and R29 differ in any value or counter field (not timing, label or identity) must hold a
changed record; those rows are tabulated (total against both total_abserr, total_converged in each). The script
also checks every row with no changed record is bit-identical between R28 and R29 in every value and counter
field (four counts included), and reports summed regions in the three stores.

    ./venv/bin/python -I -B compare_r29_r28.py <R27 primary> <R28 primary> <R29 primary> <out csv>
"""

import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabulate_qsi_guard as T  # noqa: E402

FIELDS = ("value", "abserr", "converged", "phase_limited", "regions")
EXTRA = ("simple_regions", "evaluations")


def rec_leaves(rec):
    return {k: v for k, v in rec.items() if k != "elapsed"}


def same(a, b):
    return T.bits(a) == T.bits(b)


def main():
    p27, p28, p29, out_csv = sys.argv[1:5]
    print(f"# script {Path(__file__).resolve()} sha256 {T.sha256(__file__)}")
    print(f"# imports tabulate_qsi_guard.py sha256 {T.sha256(T.__file__)}")
    for p in (p27, p28, p29):
        prim = Path(p)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# input {f} sha256 {T.sha256(f)}")
    A, ca = T.read(p27)
    B, cb = T.read(p28)
    C, cc = T.read(p29)
    assert ca == cb == cc and set(A) == set(B) == set(C)
    print(f"rows {len(C)} in each store; columns equal; key sets equal")
    changed, stops, rows_out = [], [], []
    n_rec = n_unchanged = 0
    rows_with_changes = set()
    rows_differing = set()
    regions = {"R27": [0, 0], "R28": [0, 0], "R29": [0, 0]}
    for key in sorted(C):
        a, b, c = A[key], B[key], C[key]
        fa, ma, _ = T.fields(a)
        fb, mb, _ = T.fields(b)
        fc, mc, _ = T.fields(c)
        for tag, rec_, m_ in (("R27", a, ma), ("R28", b, mb), ("R29", c, mc)):
            regions[tag][0] += rec_["WKB_Levin_num_regions"] or 0
            regions[tag][1] += sum(r[2]["regions"] for r in T.levin_records(m_) if r[0] == "group")
        # row-level difference R28 vs R29 (values and counters, the counts included)
        diff_fields = [f for f in fc if fb.get(f, (None, "<absent>"))[0] in ("value", "counter")
                       and not same(fb[f][1], fc[f][1])]
        assert set(fb) == set(fc), key
        if diff_fields:
            rows_differing.add(key)
        r27 = {r[1]: r[2] for r in T.levin_records(ma)}
        r28 = {r[1]: (r[0], r[2]) for r in T.levin_records(mb)}
        for kind, name, rc, sub in T.levin_records(mc):
            n_rec += 1
            k28, rb = r28[name]
            ra = r27[name]
            lb, lc = rec_leaves(rb), rec_leaves(rc)
            assert set(lb) == set(lc), (key, name)
            if all(same(lb[k], lc[k]) for k in lb):
                n_unchanged += 1
                continue
            rows_with_changes.add(key)
            eq27 = {f: same(ra.get(f), rc.get(f)) for f in FIELDS if f in rc or f in ra}
            eq27x = {f: same(ra.get(f), rc.get(f)) for f in EXTRA if f in rc}
            allowed = (rb["stationary_points"] > 0 and all(rc[g] == 0 for g in T.GUARD_LEAVES) and all(eq27.values()))
            e = {"k": key[0], "q": key[1], "r": key[2], "z_response": key[3], "kind": kind, "record": name,
                 "label": rc.get("label", name.split(".")[-1]), "allowed": allowed,
                 "R28_counts": {g: rb[g] for g in T.GUARD_LEAVES}, "R29_counts": {g: rc[g] for g in T.GUARD_LEAVES},
                 "R29_equals_R27": eq27, "R29_equals_R27_extra": eq27x,
                 "differing_leaves_R28_R29": sorted(k for k in lb if not same(lb[k], lc[k]))}
            for tag, x in (("R27", ra), ("R28", rb), ("R29", rc)):
                for f in FIELDS + EXTRA:
                    if f in x:
                        e[f"{f}_{tag}"] = T.hx(x[f]) if isinstance(x[f], float) else x[f]
            changed.append(e)
            if not allowed:
                stops.append(e)
        if diff_fields:
            dT = abs(c["total"] - b["total"])
            rows_out.append({"k": key[0], "q": key[1], "r": key[2], "z_response": key[3],
                             "total_28": T.hx(b["total"]), "total_29": T.hx(c["total"]),
                             "total_abserr_28": T.hx(b["total_abserr"]), "total_abserr_29": T.hx(c["total_abserr"]),
                             "abs_dtotal_over_abserr28": repr(T.ratio(dT, b["total_abserr"])),
                             "abs_dtotal_over_abserr29": repr(T.ratio(dT, c["total_abserr"])),
                             "total_converged_28": bool(b["total_converged"]), "total_converged_29": bool(c["total_converged"]),
                             "total_equals_R27": same(a["total"], c["total"]), "total_abserr_equals_R27": same(a["total_abserr"], c["total_abserr"]),
                             "analytic_rad_equals_R28": same(b["analytic_rad"], c["analytic_rad"]),
                             "analytic_rad_equals_R27": same(a["analytic_rad"], c["analytic_rad"]),
                             "row_all_values_equal_R27": all(same(fa[f][1], fc[f][1]) for f in fa if fa[f][0] == "value"),
                             "n_differing_fields": len(diff_fields),
                             "changed_records": ";".join(f"{x['record']}({x['label']})" for x in changed if (x['k'], x['q'], x['r'], x['z_response']) == key)})
    print(f"Levin records compared: {n_rec}; unchanged (bit-identical, counts included): {n_unchanged}; changed: {len(changed)}")
    print(f"rows differing R28 vs R29 (value or counter fields): {len(rows_differing)}; rows holding a changed record: {len(rows_with_changes)}; "
          f"differing rows without a changed record: {len(rows_differing - rows_with_changes)}")
    if stops:
        print(f"\nSTOP: {len(stops)} CHANGED RECORDS BREAK THE RULE (R28 stationary_points > 0, R29 four counts 0, R29 == R27):")
        for e in stops:
            print(f"  ({e['k']:.6g}, {e['q']:.6g}, {e['r']:.6g}, z {e['z_response']!r}) {e['record']} {e['label']}: "
                  f"R28 counts {e['R28_counts']}, R29 counts {e['R29_counts']}; R29 == R27 {e['R29_equals_R27']}; "
                  f"abserr R27 {e.get('abserr_R27')} R28 {e.get('abserr_R28')} R29 {e.get('abserr_R29')}; regions {e.get('regions_R27')}/{e.get('regions_R28')}/{e.get('regions_R29')}; "
                  f"converged {e.get('converged_R27')}/{e.get('converged_R28')}/{e.get('converged_R29')}; phase_limited {e.get('phase_limited_R27')}/{e.get('phase_limited_R28')}/{e.get('phase_limited_R29')}")
    from collections import Counter
    print("\nchanged records by (kind, label, allowed):", dict(Counter((e["kind"], e["label"], e["allowed"]) for e in changed)))
    print("changed records, every one:")
    for e in changed:
        print(f"  {'ok  ' if e['allowed'] else 'STOP'} ({e['k']:.6g}, {e['q']:.6g}, {e['r']:.6g}, z {e['z_response']:.8g}) {e['record']} {e['label']}: "
              f"R28 sp {e['R28_counts']['stationary_points']} wp {e['R28_counts']['window_pieces']}; R29 counts {list(e['R29_counts'].values())}; "
              f"R29 == R27 in {','.join(f for f, v in e['R29_equals_R27'].items() if v)}"
              + (f"; NOT in {','.join(f for f, v in e['R29_equals_R27'].items() if not v)}" if not all(e['R29_equals_R27'].values()) else "")
              + f"; extra {e['R29_equals_R27_extra']}")
    print(f"\nrows differing, tabulated: {len(rows_out)}")
    for r in rows_out:
        print(f"  ({r['k']:.6g}, {r['q']:.6g}, {r['r']:.6g}, z {r['z_response']:.8g}): |dT|/e28 {float(r['abs_dtotal_over_abserr28']):.4g}, "
              f"|dT|/e29 {float(r['abs_dtotal_over_abserr29']):.4g}; converged {r['total_converged_28']}->{r['total_converged_29']}; "
              f"total == R27 {r['total_equals_R27']}, abserr == R27 {r['total_abserr_equals_R27']}; analytic_rad == R28 {r['analytic_rad_equals_R28']}, == R27 {r['analytic_rad_equals_R27']}; "
              f"all values == R27 {r['row_all_values_equal_R27']}; records {r['changed_records']}")
    print("\nsummed regions (WKB_Levin_num_regions; group records' regions):", {k: v for k, v in regions.items()})
    with open(out_csv, "w", newline="") as f:
        cols = []
        for e in changed:
            for k in e:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(changed)
    rows_csv = Path(out_csv).with_name(Path(out_csv).stem + "_rows.csv")
    with open(rows_csv, "w", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    print(f"# wrote {out_csv} sha256 {T.sha256(out_csv)} and {rows_csv} sha256 {T.sha256(rows_csv)}")
    print("VERDICT:", "STOP" if stops else "every changed record allowed (R28 located, R29 counts 0, R29 == R27); every other record bit-identical to R28")
    sys.exit(2 if stops else 0)


if __name__ == "__main__":
    main()
