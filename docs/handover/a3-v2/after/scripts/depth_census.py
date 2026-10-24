"""The depth census of a store's QuadSourceIntegral rows (handover prompt 08 §4).

Reproduces the A3 baseline's ad-hoc census (handover board, the head of
[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]) on any store.
Read-only and Ray-free: the primary and every shard are opened `mode=ro`, following
`RunDrivers/quadsource_atol_sweep.py` `report()` -- the `redshift`, `wavenumber_exit_time`/`wavenumber`
and `tolerance` maps come from shard 0, and `tolerance.log10_tol` is log10 of the value.

    ./venv/bin/python var/handover-08/scripts/depth_census.py <primary.sqlite> [--expect N]

The shards are the primary's own `shards` table, in serial order. Exit status 0; 1 if the population
at (1e-32, 1e-7) is not `--expect` rows (when given).
"""

import argparse
import math
import os
import re
import sqlite3
import statistics
import sys

PAIR = (1e-32, 1e-7)
BASELINE = """A3 baseline (handover board; QUADSOURCE-TOLERANCE-SWEEP.md §1), 7600 rows at (1e-32, 1e-8), tree 704a12e:
  NULL 2489 rows, mean 0.062 s | 1-19 3054 rows, mean 0.776 s | 20: 2057 rows, mean 170.6 s
  98.2 CPU-hr in all, 97.5 of it at depth 20; 889 unconverged at depth 20 (43.2 %)
  195 rows over 60 s, 74.4 % unconverged; WKB_Levin_num_regions up to 1,530,190
  SUM(WKB_Levin_elapsed) 351,296 s of SUM(compute_time) 353,489 s; SUM(numeric_quad_compute_time) 9 s"""


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def levin_max_depth_constant():
    src = open("AdaptiveLevin/levin_quadrature.py").read()
    m = re.search(r"^DEFAULT_LEVIN_MAX_DEPTH\s*=\s*(\d+)", src, re.M)
    return int(m.group(1))


def fmt_s(x):
    return f"{x:.3f}" if x is not None else "-"


def summarise(label, rows, out):
    n = len(rows)
    if n == 0:
        out(f"  {label}: 0 rows")
        return
    ct = [r["compute_time"] or 0.0 for r in rows]
    unconv = sum(1 for r in rows if not r["total_converged"])
    out(
        f"  {label}: {n} rows; mean compute_time {statistics.fmean(ct):.3f} s; total {sum(ct):.1f} s "
        f"= {sum(ct) / 3600:.3f} CPU-hr; total_converged = 0: {unconv} ({100 * unconv / n:.1f} %)"
    )


def block(title, rows, maxdepth, out):
    out(f"\n== {title}: {len(rows)} rows")
    if not rows:
        return
    bins = {
        "NULL (no Levin call)": [r for r in rows if r["depth"] is None],
        f"1-{maxdepth - 1}": [
            r for r in rows if r["depth"] is not None and 1 <= r["depth"] < maxdepth
        ],
        f"exactly {maxdepth} (DEFAULT_LEVIN_MAX_DEPTH)": [
            r for r in rows if r["depth"] == maxdepth
        ],
    }
    other = [
        r for r in rows if r["depth"] is not None and not (1 <= r["depth"] <= maxdepth)
    ]
    out("-- bins on WKB_Levin_max_depth")
    for k, v in bins.items():
        summarise(k, v, out)
    if other:
        summarise(
            f"OUTSIDE 1-{maxdepth} (depths {sorted({r['depth'] for r in other})})",
            other,
            out,
        )
    ct = sum(r["compute_time"] or 0.0 for r in rows)
    out(f"-- CPU-hours (SUM(compute_time)/3600): {ct / 3600:.3f}")
    depths = [r["depth"] for r in rows if r["depth"] is not None]
    regions = [r["regions"] for r in rows if r["regions"] is not None]
    out(
        f"-- max WKB_Levin_max_depth {max(depths) if depths else None}; max WKB_Levin_num_regions {max(regions) if regions else None}"
    )
    le = sum(r["levin_elapsed"] or 0.0 for r in rows)
    nq = sum(r["nq_time"] or 0.0 for r in rows)
    out(
        f"-- SUM(WKB_Levin_elapsed) {le:.1f} s; SUM(compute_time) {ct:.1f} s; SUM(numeric_quad_compute_time) {nq:.1f} s"
    )
    unconv = [r for r in rows if not r["total_converged"]]
    d20 = bins[f"exactly {maxdepth} (DEFAULT_LEVIN_MAX_DEPTH)"]
    out(
        f"-- unconverged (total_converged = 0): {len(unconv)} store-wide; "
        f"{sum(1 for r in d20 if not r['total_converged'])} within depth {maxdepth}; "
        f"total_phase_limited = 1: {sum(1 for r in rows if r['phase_limited'])}"
    )
    over60 = [r for r in rows if (r["compute_time"] or 0.0) > 60.0]
    u60 = sum(1 for r in over60 if not r["total_converged"])
    out(
        f"-- rows over 60 s: {len(over60)}"
        + (f"; unconverged {u60} ({100 * u60 / len(over60):.1f} %)" if over60 else "")
    )


def table(title, rows, out):
    out(f"\n-- {title}: {len(rows)} rows")
    out(
        f"   {'k /Mpc':>11} {'q /Mpc':>11} {'r /Mpc':>11} {'z_response':>12} {'time (s)':>10} {'depth':>5} {'regions':>9} {'conv':>4} {'abserr/|total|':>14} {'shard':>5} {'serial':>7}"
    )
    for r in rows:
        rel = (
            (r["abserr"] / abs(r["total"]))
            if (r["abserr"] is not None and r["total"])
            else float("nan")
        )
        out(
            f"   {r['k']:11.4e} {r['q']:11.4e} {r['r']:11.4e} {r['z']:12.6g} {r['compute_time'] or 0.0:10.2f} "
            f"{r['depth'] if r['depth'] is not None else -1:5d} {r['regions'] if r['regions'] is not None else -1:9d} "
            f"{int(bool(r['total_converged'])):4d} {rel:14.3e} {r['shard']:5d} {r['serial']:7d}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary")
    ap.add_argument("--expect", type=int)
    a = ap.parse_args()
    out = print
    maxdepth = levin_max_depth_constant()
    out(f"store {os.path.abspath(a.primary)}")
    out(f"DEFAULT_LEVIN_MAX_DEPTH (AdaptiveLevin/levin_quadrature.py) = {maxdepth}")
    p = ro(a.primary)
    shards = [
        os.path.join(os.path.dirname(os.path.abspath(a.primary)), os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    out(f"shards (the primary's shards table): {[os.path.basename(s) for s in shards]}")
    c0 = ro(shards[0])
    zmap = dict(c0.execute("select serial, z from redshift"))
    kmap = dict(
        c0.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on w.serial = e.wavenumber_serial"
        )
    )
    tolmap = {
        s: 10.0**l for s, l in c0.execute("select serial, log10_tol from tolerance")
    }
    c0.close()
    cols = (
        "serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, z_response_serial, "
        "atol_serial, rtol_serial, total, total_abserr, total_converged, total_phase_limited, compute_time, "
        '"WKB_Levin_max_depth", "WKB_Levin_num_regions", "WKB_Levin_elapsed", numeric_quad_compute_time'
    )
    allrows, offpair = [], []
    for i, sh in enumerate(shards):
        c = ro(sh)
        for t in c.execute(f"select {cols} from QuadSourceIntegral"):
            (
                serial,
                ks,
                qs,
                rs,
                zs,
                at,
                rt,
                tot,
                abserr,
                conv,
                plim,
                ct,
                depth,
                regions,
                le,
                nq,
            ) = t
            row = dict(
                shard=i,
                serial=serial,
                k=kmap[ks],
                q=kmap[qs],
                r=kmap[rs],
                z=zmap[zs],
                atol=tolmap[at],
                rtol=tolmap[rt],
                total=tot,
                abserr=abserr,
                total_converged=conv,
                phase_limited=plim,
                compute_time=ct,
                depth=depth,
                regions=regions,
                levin_elapsed=le,
                nq_time=nq,
            )
            pair = (row["atol"], row["rtol"])
            if math.isclose(pair[0], PAIR[0], rel_tol=1e-9) and math.isclose(
                pair[1], PAIR[1], rel_tol=1e-9
            ):
                allrows.append(row)
            else:
                offpair.append(row)
        c.close()
    out(
        f"\n== population: QuadSourceIntegral rows at (atol, rtol) = {PAIR}: {len(allrows)}"
        + (
            f" (expected {a.expect}: {len(allrows) == a.expect})"
            if a.expect is not None
            else ""
        )
    )
    out(f"   rows at any other pair: {len(offpair)}")
    for r in offpair:
        out(
            f"   OTHER PAIR shard {r['shard']} serial {r['serial']}: ({r['atol']:g}, {r['rtol']:g})"
        )
    out(
        f"   by shard: {[sum(1 for r in allrows if r['shard'] == i) for i in range(len(shards))]}"
    )

    block("store-wide", allrows, maxdepth, out)
    bands = [
        (
            "seam band, z_response > 1e5 (6080 expected)",
            [r for r in allrows if r["z"] > 1e5],
        ),
        (
            "mid band, 9 <= z_response <= 1e5 (2112 expected)",
            [r for r in allrows if 9 <= r["z"] <= 1e5],
        ),
        (
            "low band, z_response < 9 (1088 expected)",
            [r for r in allrows if r["z"] < 9],
        ),
    ]
    for t, rs in bands:
        block(t, rs, maxdepth, out)
    block(
        "z_response > 174.1 (the population the A3 baseline held completely; no baseline figure was recorded for it)",
        [r for r in allrows if r["z"] > 174.1],
        maxdepth,
        out,
    )

    out("\n== the rows")
    d20 = sorted(
        [r for r in allrows if r["depth"] == maxdepth],
        key=lambda r: -(r["compute_time"] or 0.0),
    )
    table(f"every row at depth {maxdepth}, by compute_time", d20, out)
    top = sorted(allrows, key=lambda r: -(r["compute_time"] or 0.0))[:20]
    table("the 20 most expensive rows", top, out)

    out("\n== the comparison (NOT like-for-like)")
    out(BASELINE)
    out(
        "  Not like-for-like: (i) the pair differs, rtol 1e-8 (baseline) -> 1e-7 (v2), which is the change under test;\n"
        "  (ii) the baseline lacked 1680 items, all at z_response <= 174.1, including every low-band item;\n"
        "  (iii) no per-band baseline figure exists. The closest like-for-like population is z_response > 174.1\n"
        "  (above), which both stores hold completely; no baseline number for it was recorded."
    )
    return 0 if (a.expect is None or len(allrows) == a.expect) else 1


if __name__ == "__main__":
    sys.exit(main())
