"""levin-phase-floor prompt 06 §5.4: tabulate, per class, the differences between R13 (parent
9f66622) and R14 (code dc9d28c) inside GkWKBIntegration, TkWKBIntegration, GkSource and
GkSourcePolicyData. QuadSourceIntegral is 05's tabulate_qsi.py's, run separately and unchanged.

Rows are paired by tools.store_value_diff's own reader (`read_store`, the code worktree's, with
config.datastore's factories), so the pairing is the tool's identity matching, and every compared
column of both sides is in hand, not only the differing ones the JSON carries. R13 is read without
`theta_residual`, which its files lack (the tool's §1.5 extension). The stores are opened by the
tool's read-only reader; this script writes nothing but stdout and the CSV named.

Per value table (GkWKBValue, TkWKBValue, GkSourceValue):
- `theta_residual`: present in which store; in R14, null count and max |r|;
- `theta_mod_2pi`: |Δ| against ε|θ|, ε = 2**-52, where |θ| = |div·2π + mod| is formed from R14's
  stored cycle count and remainder (float; it is a scale, not a value compared);
- `theta_div_2pi` (and GkSource's `raw_theta_div_2pi`) changes: each listed with whether R14's
  remainder lies within 4ε|θ| of 0 or of −2π;
- `G_WKB`/`T_WKB` (and GkSource's `new_G_WKB`): relative change |Δ|/|A-side|, and |Δ| over the
  Liouville-Green amplitude A·|Δθ_mod|, A = sqrt(H_ratio/ω)·hypot(sin_coeff, cos_coeff) (·exp(friction)
  for T), which is ≤ 1 (to rounding) if the change is the remainder's change alone;
- every other value column that differs, by count.
GkSourcePolicyData: each row column that differs, and for `metadata`, each JSON key that differs
with the changes of its numeric values.
It cross-checks its differing-row counts per column against the tool's JSON column summaries.

    PYTHONPATH=<code worktree> python -B tabulate_phase_classes.py <R13 primary> <R14 primary>
        <store_value_diff.json> <out.csv> <R13 status.json> <R14 status.json>
"""

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

EPS = 2.0**-52
TWO_PI = 2.0 * math.pi
CLASSES = ("GkWKBIntegration", "TkWKBIntegration", "GkSource", "GkSourcePolicyData")
VALUE_TABLES = {"GkWKBIntegration": "GkWKBValue", "TkWKBIntegration": "TkWKBValue",
                "GkSource": "GkSourceValue"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def q(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else None


def fmt(x):
    return "None" if x is None else f"{x:.3e}"


def stats(name, vals, out):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        out.append(f"    {name}: no values")
        return
    out.append(f"    {name}: n {len(vals)}, median {fmt(q(vals, .5))}, 90% {fmt(q(vals, .9))}, "
               f"99% {fmt(q(vals, .99))}, max {fmt(max(vals))}")


def bits_equal(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return float(a).hex() == float(b).hex()
    return a == b


def main():
    a_path, b_path, json_path, out_csv, st_a, st_b = sys.argv[1:7]
    me = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# {me.name} sha256 {sha256(me)}")
    print(f"# git HEAD (cwd) {head}")
    for p in (a_path, b_path):
        for f in sorted(Path(p).parent.glob(Path(p).stem + "*.sqlite")):
            print(f"# input {f} sha256 {sha256(f)}")
    print(f"# input {json_path} sha256 {sha256(json_path)}")
    for label, st in (("A=R13", st_a), ("B=R14", st_b)):
        fp = json.load(open(st))["fingerprint"]
        print(f"# {label} fingerprint {fp['digest']} (taken {fp['taken']['git_head']}, {st})")
    print()

    from config.datastore import factories
    from tools.store_value_diff import read_store

    A = read_store(a_path, factories, classes=list(CLASSES))
    B = read_store(b_path, factories, classes=list(CLASSES))
    print(f"absent columns, read without: A {A.absent_columns}; B {B.absent_columns}")

    def k_of(store, parent_key):
        try:
            d = json.loads(parent_key)["key"]["k"]
            m, _ = store.by_digest[("wavenumber_exit_time", d)]
            m2, _ = store.by_digest[("wavenumber", m["k"])]
            return m2["k_inv_Mpc"]
        except Exception:
            return None

    rep = json.load(open(json_path))["classes"]
    csv_rows = []
    lines = []
    for cls in CLASSES:
        ca, cb = A.classes[cls], B.classes[cls]
        lines.append(f"== {cls}")
        # row columns
        ra, rb = ca.rows, cb.rows
        common = sorted(set(ra.rows) & set(rb.rows))
        lines.append(f"  rows: A {len(ra.rows)}, B {len(rb.rows)}, paired {len(common)}, "
                     f"only A {len(set(ra.rows) - set(rb.rows))}, only B {len(set(rb.rows) - set(ra.rows))}")
        coldiff = Counter()
        meta_keys = defaultdict(list)
        for key in common:
            va, vb = ra.rows[key][0].values, rb.rows[key][0].values
            for col in sorted(set(ra.columns) & set(rb.columns)):
                x, y = va.get(col), vb.get(col)
                if col == "metadata" and x is not None and y is not None:
                    jx, jy = json.loads(x), json.loads(y)
                    if jx != jy:
                        coldiff[col] += 1
                        for mk in sorted(set(jx) | set(jy)):
                            if jx.get(mk) != jy.get(mk):
                                meta_keys[mk].append((jx.get(mk), jy.get(mk), key))
                    continue
                if not (repr(x) == repr(y) or bits_equal(x, y)):
                    coldiff[col] += 1
        lines.append(f"  row columns differing (rows): {dict(coldiff) or 'none'}")
        for mk, ch in sorted(meta_keys.items()):
            lines.append(f"    metadata key {mk}: {len(ch)} rows")
            num = [(x, y) for x, y, _ in ch if isinstance(x, (int, float)) and isinstance(y, (int, float))
                   and not isinstance(x, bool)]
            if num and cls == "GkSourcePolicyData":
                stats("|Δ|", [abs(y - x) for x, y in num], lines)
                stats("|Δ|/|A|", [abs(y - x) / abs(x) for x, y in num if x != 0], lines)
                nulls = [(x, y) for x, y, _ in ch if (x is None) != (y is None)]
                lines.append(f"      rows where one side is null: {len(nulls)}")
                for x, y, key in sorted(ch, key=lambda t: -abs((t[1] or 0) - (t[0] or 0)))[:5]:
                    lines.append(f"      largest: A {x!r} B {y!r}")
        if cls == "GkSourcePolicyData":
            for col in ("Levin_z", "crossover_z", "type", "quality"):
                lines.append(f"    {col}: {coldiff.get(col, 0)} rows differ")

        vt = VALUE_TABLES.get(cls)
        if vt is None:
            continue
        ta, tb = ca.values[vt], cb.values[vt]
        lines.append(f"  {vt}: columns only in A {sorted(set(ta.columns) - set(tb.columns))}, "
                     f"only in B {sorted(set(tb.columns) - set(ta.columns))}")
        keys = sorted(set(ta.rows) & set(tb.rows))
        lines.append(f"  {vt}: rows A {len(ta.rows)}, B {len(tb.rows)}, paired {len(keys)}, "
                     f"only A {len(set(ta.rows) - set(tb.rows))}, only B {len(set(tb.rows) - set(ta.rows))}, "
                     f"unmatched A {len(ta.unmatched)} B {len(tb.unmatched)}")
        cols = sorted(set(ta.columns) & set(tb.columns))
        vdiff = Counter()
        r_null = 0
        r_abs = []
        mod_ratio = []
        mod_abs = []
        theta_abs = []
        div_changes = []
        raw_div_changes = 0
        rel_G = []
        lg_ratio = []
        out_of_range_B = 0
        Gcol = "T_WKB" if vt == "TkWKBValue" else "G_WKB"
        Gcols = [Gcol] + (["new_G_WKB"] if vt == "GkSourceValue" else [])
        rel_new = []
        lg_new = []
        per_k = defaultdict(list)
        for key in keys:
            va, vb = ta.rows[key][0].values, tb.rows[key][0].values
            for col in cols:
                if not bits_equal(va.get(col), vb.get(col)) and not (
                    hasattr(va.get(col), "digest") and va.get(col) == vb.get(col)
                ):
                    vdiff[col] += 1
            r = vb.get("theta_residual")
            if vb.get("theta_mod_2pi") is None:
                continue
            if r is None:
                r_null += 1
            else:
                r_abs.append(abs(r))
            ma, mb = va["theta_mod_2pi"], vb["theta_mod_2pi"]
            da, db = va["theta_div_2pi"], vb["theta_div_2pi"]
            if not (-TWO_PI < mb <= 0.0):
                out_of_range_B += 1
            th = abs(db * TWO_PI + mb)
            theta_abs.append(th)
            dm = abs(mb - ma)
            mod_abs.append(dm)
            ratio = dm / (EPS * th) if th > 0 else (0.0 if dm == 0 else math.inf)
            if da == db:
                mod_ratio.append(ratio)
            else:
                near0 = abs(mb) <= 4 * EPS * th
                near2pi = abs(mb + TWO_PI) <= 4 * EPS * th
                div_changes.append((key, da, db, ma, mb, th, near0, near2pi))
            if vt == "GkSourceValue" and not bits_equal(va.get("raw_theta_div_2pi"), vb.get("raw_theta_div_2pi")):
                raw_div_changes += 1
            # amplitude
            if vt == "GkSourceValue":
                sc, cc = vb.get("sin_coeff"), vb.get("cos_coeff")
            else:
                prow = (cb.rows.rows.get(key[0]) or [None])[0]
                sc = prow.values.get("sin_coeff") if prow else None
                cc = prow.values.get("cos_coeff") if prow else None
            amp = None
            if None not in (sc, cc, vb.get("H_ratio"), vb.get("omega_WKB_sq")) and vb["omega_WKB_sq"] > 0:
                amp = math.sqrt(vb["H_ratio"] / math.sqrt(vb["omega_WKB_sq"])) * math.hypot(sc, cc)
                if vt == "TkWKBValue":
                    amp *= math.exp(vb["friction"])
            dth = abs(mb - ma) if da == db else None
            for gc, rl, lr in ((Gcols[0], rel_G, lg_ratio),) + ((("new_G_WKB", rel_new, lg_new),) if len(Gcols) > 1 else ()):
                ga, gb = va.get(gc), vb.get(gc)
                if ga is None or gb is None:
                    continue
                if ga != 0:
                    rl.append(abs(gb - ga) / abs(ga))
                if amp and dth:
                    lr.append(abs(gb - ga) / (amp * dth))
                elif amp and dth == 0 and gb != ga:
                    lr.append(math.inf)
            k = k_of(B, key[0])
            per_k[k].append(ratio if da == db else None)
            csv_rows.append({"table": vt, "k": k, "parent": key[0][:120], "z": key[1],
                             "div_A": da, "div_B": db, "mod_A": ma.hex(), "mod_B": mb.hex(),
                             "theta_abs": th, "dmod_over_eps_theta": ratio, "residual_B": r,
                             f"{Gcol}_A": va.get(Gcol), f"{Gcol}_B": vb.get(Gcol)})

        lines.append(f"  {vt}: value columns differing (rows): {dict(vdiff)}")
        tool = rep[cls]["values"][vt]["columns"]
        tool_counts = {c: e["count"] for c, e in tool.items()}
        lines.append(f"  tool's JSON column counts: {tool_counts}; equal to this read: "
                     f"{tool_counts == dict(vdiff)}")
        lines.append(f"  theta_residual (R14 only): non-null {len(r_abs)}, null {r_null} (of rows with a "
                     f"stored remainder); max |r| {fmt(max(r_abs) if r_abs else None)}; "
                     f"|r| > pi + 2e-3: {sum(1 for x in r_abs if x > math.pi + 2e-3)}")
        lines.append(f"  R14 theta_mod_2pi outside (-2pi, 0]: {out_of_range_B}")
        stats("|theta| (from R14's div and mod)", theta_abs, lines)
        stats("|Δ theta_mod_2pi| (rad)", mod_abs, lines)
        stats("|Δ theta_mod_2pi| / (eps |theta|), rows with div unchanged", mod_ratio, lines)
        for thr in (0.5, 1.0, 2.0, 4.0):
            lines.append(f"      > {thr}: {sum(1 for x in mod_ratio if x > thr)}")
        lines.append(f"      bit-identical theta_mod_2pi: {sum(1 for x in mod_abs if x == 0)}")
        lines.append(f"  theta_div_2pi changes: {len(div_changes)}")
        for key, da, db, ma, mb, th, n0, n2 in div_changes:
            lines.append(f"    k={k_of(B, key[0])} z={key[1][:90]}: div {da}->{db}, mod {ma!r}->{mb!r}, "
                         f"|theta| {th:.6e}, 4eps|theta| {4 * EPS * th:.3e}, R14 mod within 4eps|theta| of 0: {n0}, of -2pi: {n2}"
                         + ("" if (n0 or n2) else "  <-- OTHER CHANGE"))
        if vt == "GkSourceValue":
            lines.append(f"  raw_theta_div_2pi changes: {raw_div_changes}")
        stats(f"|Δ {Gcols[0]}|/|{Gcols[0]}_A|", rel_G, lines)
        stats(f"|Δ {Gcols[0]}|/(A |Δ theta_mod|) (≤ 1 if the remainder alone moved it)", lg_ratio, lines)
        if len(Gcols) > 1:
            stats("|Δ new_G_WKB|/|new_G_WKB_A|", rel_new, lines)
            stats("|Δ new_G_WKB|/(A |Δ theta_mod|)", lg_new, lines)
        lines.append("  per k: rows, median and max |Δ mod|/(eps|theta|) (div unchanged)")
        for k in sorted(per_k, key=lambda x: (x is None, x)):
            v = [x for x in per_k[k] if x is not None]
            lines.append(f"    k={k}: {len(per_k[k])} rows, median {fmt(q(v, .5))}, max {fmt(max(v) if v else None)}")
    print("\n".join(lines))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({f for r in csv_rows for f in r}))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV: {len(csv_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
