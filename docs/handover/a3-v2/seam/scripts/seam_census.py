"""The seam census of a store's QuadSourceIntegral rows (handover prompt 11 §1 step 4).

Reads every QuadSourceIntegral row at (atol, rtol) = (1e-32, 1e-7) and its metadata["partition"], and
reports (a) the population, (b) one CSV record per sub-interval, (c) the clamp-gap census for Tq, Tr
and source, set beside the 2026-09 figures, and (d) the gap's effect on `total` against
`analytic_rad` in the seam band.

Read-only and Ray-free, following docs/handover/a3-v2/after/scripts/depth_census.py: the primary and
every shard are opened `mode=ro`; sqlite3 and json only; nothing from the pipeline is imported. The
maps `redshift`, `wavenumber_exit_time`/`wavenumber` and `tolerance` come from shard 0.

    ./venv/bin/python -I var/handover-11/scripts/seam_census.py <primary.sqlite> [--expect 9280]

The census goes to stdout. The CSV goes to var/handover-11/out/subintervals_11.csv (next to this
script's `scripts/`). Exit status 0; 1 if the population is not --expect rows, a row has no
partition, or a gap exceeds its row's allowance.

The field read is partition["subintervals"][i]["clamp_gaps_log1pz"]. `max_clamp_gap_log1pz` is the
ALLOWANCE (HANDOVER_CLAMP_MAX_GRID_STEPS x the mean source-grid step), one number per row.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

PAIR = (1e-32, 1e-7)
FACTORS = ("Tq", "Tr", "source")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "out")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_constant():
    src = open(os.path.join(REPO, "ComputeTargets/QuadSourceIntegral.py")).read()
    m = re.search(r"^HANDOVER_CLAMP_MAX_GRID_STEPS\s*=\s*([0-9.eE+-]+)", src, re.M)
    return float(m.group(1))


def band_of(z):
    if z < 9:
        return "low"
    if z <= 1e5:
        return "mid"
    return "seam"


def quart(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return None
    return [float(np.percentile(a, p)) for p in (0, 25, 50, 75, 100)]


def fq(q):
    return "-" if q is None else "  ".join(f"{v:9.3e}" for v in q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary")
    ap.add_argument("--expect", type=int, default=9280)
    a = ap.parse_args()
    out = print
    primary = os.path.abspath(a.primary)
    store_dir = os.path.dirname(primary)
    ncs = read_constant()

    # ---- header
    out("SEAM CENSUS (handover prompt 11)")
    out(f"script           {os.path.abspath(__file__)}")
    out(f"script sha256    {sha256(__file__)}")
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out(f"git HEAD         {head}")
    p = ro(primary)
    shards = [
        os.path.join(store_dir, os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    out(f"store            {primary}")
    out(f"  {sha256(primary)}  {os.path.basename(primary)}")
    for s in shards:
        out(f"  {sha256(s)}  {os.path.basename(s)}")
    sidecar = primary[: -len(".sqlite")] + ".manifest.json"
    out(f"  {sha256(sidecar)}  {os.path.basename(sidecar)}")
    fp = json.load(open(sidecar))["fingerprint"]["digest"]
    out(f"store fingerprint (sidecar's) {fp}")
    out(f"HANDOVER_CLAMP_MAX_GRID_STEPS (read from ComputeTargets/QuadSourceIntegral.py) = {ncs:g}")

    # ---- maps from shard 0
    c0 = ro(shards[0])
    zmap = dict(c0.execute("select serial, z from redshift"))
    kmap = dict(
        c0.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on w.serial = e.wavenumber_serial"
        )
    )
    tolmap = {s: 10.0**l for s, l in c0.execute("select serial, log10_tol from tolerance")}
    c0.close()

    cols = (
        "serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, "
        "z_response_serial, atol_serial, rtol_serial, total, analytic_rad, b, metadata"
    )
    rows, offpair, nopart, bad = [], 0, [], []
    for i, sh in enumerate(shards):
        c = ro(sh)
        for t in c.execute(f"select {cols} from QuadSourceIntegral"):
            serial, ks, qs, rs, zs, at, rt, tot, arad, b, md = t
            if not (
                math.isclose(tolmap[at], PAIR[0], rel_tol=1e-9)
                and math.isclose(tolmap[rt], PAIR[1], rel_tol=1e-9)
            ):
                offpair += 1
                continue
            part = None
            try:
                part = json.loads(md)["partition"]
                subs = part["subintervals"]
                allow = float(part["max_clamp_gap_log1pz"])
                assert isinstance(subs, list) and len(subs) >= 1
                for s in subs:
                    s["clamp_gaps_log1pz"], s["z_max"], s["z_min"], s["regime"], s["method"]
            except Exception as e:  # noqa
                nopart.append((i, serial, repr(e)))
                continue
            rows.append(
                dict(
                    shard=i,
                    serial=serial,
                    k=kmap[ks],
                    q=kmap[qs],
                    r=kmap[rs],
                    z=zmap[zs],
                    band=band_of(zmap[zs]),
                    total=tot,
                    analytic_rad=arad,
                    b=b,
                    part=part,
                    subs=subs,
                    allow=allow,
                    step=allow / ncs,
                )
            )
        c.close()

    # ---- (a) population
    out("\n" + "=" * 100)
    out("(a) THE POPULATION")
    out(f"QuadSourceIntegral rows at (atol, rtol) = {PAIR}: {len(rows) + len(nopart)}")
    out(f"  rows at any other pair: {offpair}")
    out(f"  expected {a.expect}: {len(rows) + len(nopart) == a.expect}")
    out(f"  rows whose metadata['partition'] does not parse (or has no subintervals): {len(nopart)}")
    for r in nopart[:20]:
        out(f"    shard {r[0]} serial {r[1]}: {r[2]}")
    out(f"  by shard: {[sum(1 for r in rows if r['shard'] == i) for i in range(len(shards))]}")
    out(f"  by band:  {dict(Counter(r['band'] for r in rows))}")
    out(f"  by k:     {dict(sorted(Counter(r['k'] for r in rows).items()))}")
    out(f"  distinct b values: {sorted({r['b'] for r in rows})}")
    nsub = sum(len(r["subs"]) for r in rows)
    out(f"  sub-intervals in all: {nsub}; per row {Counter(len(r['subs']) for r in rows)}")
    out(f"  methods: {dict(Counter(s['method'] for r in rows for s in r['subs']))}")
    out(
        f"  allowance (max_clamp_gap_log1pz) over rows: min {min(r['allow'] for r in rows):.4e}, "
        f"median {float(np.median([r['allow'] for r in rows])):.4e}, max {max(r['allow'] for r in rows):.4e}; "
        f"implied mean source-grid step: min {min(r['step'] for r in rows):.4e}, max {max(r['step'] for r in rows):.4e}"
    )
    out(f"  distinct allowances: {len({round(r['allow'], 15) for r in rows})}")

    # ---- (b) the CSV
    os.makedirs(OUTDIR, exist_ok=True)
    csvpath = os.path.join(OUTDIR, "subintervals_11.csv")
    over = []
    with open(csvpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shard", "serial", "k_inv_Mpc", "q_inv_Mpc", "r_inv_Mpc", "z_response", "band",
                "sub_index", "z_max", "z_min", "regime_G_q_r", "n_osc", "method",
                "gap_Tq", "gap_Tr", "gap_source",
                "max_clamp_gap_log1pz", "mean_source_grid_step", "HANDOVER_CLAMP_MAX_GRID_STEPS",
                "total", "analytic_rad",
            ]
        )
        for r in rows:
            for j, s in enumerate(r["subs"]):
                g = s["clamp_gaps_log1pz"]
                for lab, v in g.items():
                    if v > r["allow"]:
                        over.append((r["shard"], r["serial"], j, lab, v, r["allow"]))
                w.writerow(
                    [
                        r["shard"], r["serial"], repr(r["k"]), repr(r["q"]), repr(r["r"]),
                        repr(r["z"]), r["band"], j, repr(s["z_max"]), repr(s["z_min"]),
                        "".join("1" if x else "0" for x in s["regime"]), sum(bool(x) for x in s["regime"]),
                        s["method"],
                        *[repr(g[lab]) if lab in g else "" for lab in FACTORS],
                        repr(r["allow"]), repr(r["step"]), ncs, repr(r["total"]), repr(r["analytic_rad"]),
                    ]
                )
    out("\n" + "=" * 100)
    out("(b) ONE RECORD PER SUB-INTERVAL")
    out(f"written {csvpath}: {nsub} records, {os.path.getsize(csvpath)} bytes, sha256 {sha256(csvpath)}")
    out(f"gaps above the row's allowance (stop condition): {len(over)}")
    for o in over[:20]:
        out(f"   shard {o[0]} serial {o[1]} sub {o[2]} {o[3]}: gap {o[4]:.6e} > allowance {o[5]:.6e}")

    # ---- (c) the gap census
    out("\n" + "=" * 100)
    out("(c) THE GAP CENSUS")
    out(
        "Field: partition['subintervals'][i]['clamp_gaps_log1pz'][factor]: the shortfall, in log(1+z), of the\n"
        "factor's representation against the sub-interval end; 0.0 where there is none. 'present' = the factor has\n"
        "an entry on that sub-interval (Tq and Tr on every sub-interval; 'source' only on all-smooth, quad ones).\n"
        "'steps' = gap / (the row's allowance / HANDOVER_CLAMP_MAX_GRID_STEPS) = in mean source-grid steps.\n"
        "'near' = gap >= 0.99 x the row's allowance (at or within 1 % of it).\n"
    )

    def census_block(label, rs):
        out(f"\n-- {label}: {len(rs)} rows, {sum(len(r['subs']) for r in rs)} sub-intervals")
        out(
            f"   {'factor':>6} {'present':>8} {'nonzero':>8} {'rows>=1':>8} | quartiles of the non-zero gap in log(1+z): "
            f"min / Q1 / median / Q3 / max   | in steps: median / max | near-allowance"
        )
        for lab in FACTORS:
            present = nz = 0
            rows_any = set()
            gaps, steps = [], []
            near = 0
            for r in rs:
                for s in r["subs"]:
                    g = s["clamp_gaps_log1pz"]
                    if lab in g:
                        present += 1
                        if g[lab] > 0.0:
                            nz += 1
                            rows_any.add((r["shard"], r["serial"]))
                            gaps.append(g[lab])
                            steps.append(g[lab] / r["step"])
                            if g[lab] >= 0.99 * r["allow"]:
                                near += 1
            q = quart(gaps)
            st = quart(steps)
            out(
                f"   {lab:>6} {present:8d} {nz:8d} {len(rows_any):8d} | {fq(q)} | "
                + ("-" if st is None else f"{st[2]:7.4f} {st[4]:7.4f}")
                + f" | {near}"
            )
        anyrow = sum(
            1
            for r in rs
            if any(v > 0.0 for s in r["subs"] for v in s["clamp_gaps_log1pz"].values())
        )
        out(f"   rows with at least one non-zero gap (any of the three): {anyrow} of {len(rs)}")
        sk = [r for r in rs if r["part"]["skipped"]]
        out(f"   rows with any 'skipped' breakpoint: {len(sk)} of {len(rs)}")

    census_block("ALL ROWS", rows)

    out("\n== ALL ROWS, the number of non-zero gaps per sub-interval, and clamped factors per row")
    per_sub = Counter(
        sum(1 for v in s["clamp_gaps_log1pz"].values() if v > 0.0)
        for r in rows
        for s in r["subs"]
    )
    out(f"   sub-intervals by number of clamped factors (0..3): {dict(sorted(per_sub.items()))}")

    def clamped_in_row(r, labs=FACTORS):
        return {
            lab
            for lab in labs
            for s in r["subs"]
            if s["clamp_gaps_log1pz"].get(lab, 0.0) > 0.0
        }

    out(
        f"   rows by number of clamped factors in the row (any sub-interval; of Tq, Tr, source): "
        f"{dict(sorted(Counter(len(clamped_in_row(r)) for r in rows).items()))}"
    )
    out(
        f"   rows by number of clamped factors among Tq, Tr alone: "
        f"{dict(sorted(Counter(len(clamped_in_row(r, ('Tq', 'Tr'))) for r in rows).items()))}"
    )
    out(f"   rows by which factors are clamped: {dict(Counter(tuple(sorted(clamped_in_row(r))) for r in rows))}")

    out("\n== THE SKIPPED BREAKPOINTS (rows with any 'skipped' entry)")
    sk_labels = Counter()
    sk_rows = 0
    for r in rows:
        if r["part"]["skipped"]:
            sk_rows += 1
            for e in r["part"]["skipped"]:
                sk_labels[(e["factor"], e["reason"])] += 1
    out(f"   rows: {sk_rows} of {len(rows)}; entries by (factor, reason):")
    for (f_, rsn), n in sorted(sk_labels.items(), key=lambda x: -x[1]):
        out(f"     {n:6d}  factor={f_}  {rsn}")
    out(f"   rows with a skipped entry, by band: {dict(Counter(r['band'] for r in rows if r['part']['skipped']))}")
    out(
        f"   largest skipped log_width: "
        f"{max((e['log_width'] for r in rows for e in r['part']['skipped']), default=None)}"
    )

    out("\n== THE 2026-09 FIGURES, for comparison (prompts/source-remediation board [12-handover-clamp-error-in-production])")
    out(
        "   one production run: Tq non-zero on 1666 sub-intervals, Tr on 1798; median gap 1.2e-02, max 2.2e-02\n"
        "   (about 1.0 mean source-grid step against the 1.5-step allowance); the QuadSource spline clamped on 108.\n"
        "   A3 v2 is the whole 9280-row store, so the counts are not like-for-like; the medians and maxima are."
    )

    out("\n== BY BAND")
    for bnd in ("low", "mid", "seam"):
        census_block(f"band {bnd}", [r for r in rows if r["band"] == bnd])
    out("\n== BY k")
    for k in sorted({r["k"] for r in rows}):
        census_block(f"k = {k:.6e} /Mpc", [r for r in rows if r["k"] == k])

    # ---- (d) the oracle
    out("\n" + "=" * 100)
    out("(d) THE GAP'S EFFECT ON total, AGAINST analytic_rad")
    out(
        "WHAT THIS DOES NOT SEPARATE. analytic_rad is a constant-w oracle built at b = 0. Its own deficit on\n"
        "LambdaCDM grows towards matter-radiation equality, so |total/analytic_rad - 1| is the clamp's effect\n"
        "PLUS the oracle's own error, and the second cannot be told from the first here. The 2026-09 comparison\n"
        "was taken at z_response = 7.63e9 and 1.45e9 for that reason. Decades below about 1e8 are REPORTED below\n"
        "(marked '*') but are NOT read as the clamp's error. The measure also contains the seam's other term\n"
        "(the LG phase representation, 4.08e-05 - 2.68e-03 in the 2x2 of prompt 02), which this does not isolate,\n"
        "and the quadrature's own error. Strata are tabulated, not fitted, and nothing here is causal.\n"
        "Population: seam-band rows, z_response > 1e5 (radiation-dominated). Rows with analytic_rad zero or\n"
        "non-finite (or total non-finite) are counted and excluded."
    )
    seam = [r for r in rows if r["band"] == "seam"]
    good, excl = [], []
    for r in seam:
        ar, tt = r["analytic_rad"], r["total"]
        if ar is None or not math.isfinite(ar) or ar == 0.0 or tt is None or not math.isfinite(tt):
            excl.append(r)
        else:
            r["dev"] = abs(tt / ar - 1.0)
            good.append(r)
    out(f"seam-band rows: {len(seam)}; used: {len(good)}; excluded (analytic_rad zero/non-finite or total non-finite): {len(excl)}")
    for r in excl[:20]:
        out(f"   excluded shard {r['shard']} serial {r['serial']}: total={r['total']!r} analytic_rad={r['analytic_rad']!r}")
    if good:
        dv = [r["dev"] for r in good]
        out(f"all used rows: median {np.median(dv):.3e}, max {np.max(dv):.3e}")
    zs_used = sorted({r["z"] for r in good})
    out(f"distinct z_response values in the seam band: {len(zs_used)}; min {zs_used[0]:.4e}, max {zs_used[-1]:.4e}")

    def last_levin_nosc(r):
        lev = [s for s in r["subs"] if s["method"] == "Levin"]
        if not lev:
            return 0  # no Levin sub-interval: every factor smooth
        return sum(bool(x) for x in lev[-1]["regime"])

    def tabulate(title, key, keys, rs):
        out(f"\n-- {title}")
        decades = sorted({math.floor(math.log10(r["z"])) for r in rs})
        hdr = f"   {'decade':>10} |"
        for kk in keys:
            hdr += f" {'[' + str(kk) + '] n':>8} {'median':>9} {'max':>9} |"
        out(hdr)
        for d in decades:
            sub = [r for r in rs if math.floor(math.log10(r["z"])) == d]
            line = f"   {'1e' + str(d):>10}{'*' if d < 8 else ' '}|"
            for kk in keys:
                v = [r["dev"] for r in sub if key(r) == kk]
                if v:
                    line += f" {len(v):8d} {np.median(v):9.2e} {np.max(v):9.2e} |"
                else:
                    line += f" {0:8d} {'-':>9} {'-':>9} |"
            out(line)
        line = f"   {'all':>10} |"
        for kk in keys:
            v = [r["dev"] for r in rs if key(r) == kk]
            line += (
                f" {len(v):8d} {np.median(v):9.2e} {np.max(v):9.2e} |"
                if v
                else f" {0:8d} {'-':>9} {'-':>9} |"
            )
        out(line)
        out("   ('*' = decade below 1e8, where the oracle's own deficit is not negligible: reported, not read as the clamp's error)")

    nclamp = lambda r: len(clamped_in_row(r))  # noqa
    nclamp_qr = lambda r: len(clamped_in_row(r, ("Tq", "Tr")))  # noqa
    tabulate(
        "Stratum 1: clamped factors in the row (any sub-interval; of Tq, Tr, source)",
        nclamp,
        sorted({nclamp(r) for r in good}),
        good,
    )
    tabulate(
        "Stratum 1b: clamped factors among Tq and Tr alone",
        nclamp_qr,
        sorted({nclamp_qr(r) for r in good}),
        good,
    )
    tabulate(
        "Stratum 2: oscillatory factors in the row's last Levin sub-interval (0 = the row has no Levin sub-interval)",
        last_levin_nosc,
        sorted({last_levin_nosc(r) for r in good}),
        good,
    )
    out("\n-- The two strata crossed (all seam-band decades together; n / median / max)")
    for nc in sorted({nclamp(r) for r in good}):
        for no in sorted({last_levin_nosc(r) for r in good}):
            v = [r["dev"] for r in good if nclamp(r) == nc and last_levin_nosc(r) == no]
            if v:
                out(f"   clamped={nc} osc_last_Levin={no}: n={len(v):5d} median {np.median(v):.2e} max {np.max(v):.2e}")

    out("\n-- At the response redshifts nearest the two 2026-09 comparison points (7.63e9 and 1.45e9), by clamped factors in the row")
    for target in (7.63e9, 1.45e9):
        zn = min(zs_used, key=lambda z: abs(math.log(z / target)))
        sub = [r for r in good if r["z"] == zn]
        out(f"   z_response = {zn:.5e} (nearest to {target:.3g}): {len(sub)} rows")
        for nc in sorted({nclamp(r) for r in sub}):
            v = [r["dev"] for r in sub if nclamp(r) == nc]
            out(f"      clamped={nc}: n={len(v):4d} median {np.median(v):.2e} max {np.max(v):.2e} min {np.min(v):.2e}")
        for no in sorted({last_levin_nosc(r) for r in sub}):
            v = [r["dev"] for r in sub if last_levin_nosc(r) == no]
            out(f"      osc_last_Levin={no}: n={len(v):4d} median {np.median(v):.2e} max {np.max(v):.2e} min {np.min(v):.2e}")
    out(
        "\n   2026-09 for reference (z_response = 7.63e9): rows with no clamp gap 6.2e-05; with a gap 6.6e-02 - 4.6e-01;\n"
        "   by regime 0 / 2 / 3 oscillatory factors 6.2e-05 / 7.8e-03 / 2.7e-01."
    )

    ok = len(rows) + len(nopart) == a.expect and not nopart and not over
    out(f"\nPOPULATION {len(rows) + len(nopart)}; partitions parsed {len(rows)}; gaps over allowance {len(over)}; OK={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
