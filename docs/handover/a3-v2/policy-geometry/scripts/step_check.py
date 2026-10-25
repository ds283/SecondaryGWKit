"""Read-only: the mean source-grid step on the numeric and the WKB side of every mixed row, and where the grid
is finer than its usual step. A small supplement to policy_geometry.py (read at the same tree, mode=ro).

    ./venv/bin/python -I var/handover-03/scripts/step_check.py <primary.sqlite>
"""

import math
import os
import sqlite3
import sys

import numpy as np


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


primary = os.path.abspath(sys.argv[1])
store_dir = os.path.dirname(primary)
p = ro(primary)
shards = [
    os.path.join(store_dir, os.path.basename(f))
    for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
]
p.close()
c0 = ro(shards[0])
zmap = dict(c0.execute("select serial, z from redshift"))
c0.close()
order = sorted(zmap, key=lambda s: zmap[s])
lz = np.array([math.log(1.0 + zmap[s]) for s in order])
gidx = {s: i for i, s in enumerate(order)}
steps = np.diff(lz)
usual = float(np.max(steps))
fine = np.where(steps < usual * (1 - 1e-6))[0]
print(
    f"usual (maximum) step {usual:.6e}; intervals finer than that: {len(fine)} of {len(steps)}"
)
if len(fine):
    zf = [zmap[order[i]] for i in fine]
    print(f"  they lie between z = {min(zf):.4e} and z = {max(zf):.4e}")
    # contiguous runs of fine intervals
    runs, start = [], fine[0]
    for a, b in zip(fine[:-1], fine[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, fine[-1]))
    for a, b in runs:
        print(
            f"  run of {b - a + 1} intervals from z = {zmap[order[a]]:.4e} to z = {zmap[order[b + 1]]:.4e}, step {steps[a:b + 1].min():.4e} .. {steps[a:b + 1].max():.4e}"
        )
rows = []
for si, sh in enumerate(shards):
    c = ro(sh)
    for ser, xz, ns, wl in c.execute(
        "select d.serial, d.crossover_z, s.numeric_smallest_z_serial, s.primary_WKB_largest_z_serial "
        "from GkSourcePolicyData d join GkSource s on s.serial = d.source_serial where d.type = 2"
    ):
        i_n, i_w = gidx[ns], gidx[wl]
        i_x = next(i for i in range(i_n, i_w + 1) if zmap[order[i]] == xz)
        sn = (lz[i_x] - lz[i_n]) / (i_x - i_n)
        sw = (lz[i_w] - lz[i_x]) / (i_w - i_x)
        nfine_num = int(np.sum(steps[i_n:i_x] < usual * (1 - 1e-6)))
        nfine_wkb = int(np.sum(steps[i_x:i_w] < usual * (1 - 1e-6)))
        rows.append((si, ser, sn, sw, nfine_num, nfine_wkb))
    c.close()
print(f"mixed rows: {len(rows)}")
print(
    f"mean step on the numeric side: min {min(r[2] for r in rows):.6e}, max {max(r[2] for r in rows):.6e}"
)
print(
    f"mean step on the WKB side: min {min(r[3] for r in rows):.6e}, max {max(r[3] for r in rows):.6e}"
)
print(
    f"rows with any finer-than-usual interval on the numeric side: {sum(1 for r in rows if r[4])}; "
    f"on the WKB side: {sum(1 for r in rows if r[5])}; "
    f"most finer intervals on a numeric side: {max(r[4] for r in rows)}, on a WKB side: {max(r[5] for r in rows)}"
)
