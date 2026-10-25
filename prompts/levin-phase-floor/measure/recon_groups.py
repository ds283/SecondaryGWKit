"""Which phase group carries A3 v2's depth-20 cost? A reconnaissance, not a prompt's measurement.

Run by the orchestrator on 2026-10-25 while planning `prompts/levin-phase-floor`, on the Mac
Studio, against `var/datastores/handover-A3-v2-lambdacdm.sqlite` (fingerprint `0e32f24a...`):

    ssh macstudio-tunnel '~/Documents/Code/SecondaryGWKit/venv/bin/python -I -' < recon_groups.py

Read-only and Ray-free: every shard is opened `mode=ro`, as `depth_census.py` (handover 08) does.
It reads the per-group records `phase_group_Levin_integral` stores in each row's
`metadata["WKB_Levin"]["subintervals"][i]["groups"]`. Prompt 01 re-measures all of this at full
resolution; where the two differ, 01 governs.
"""

import collections
import json
import os
import sqlite3

BASE = os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/datastores/")
STORE = "handover-A3-v2-lambdacdm"
MIN_TIME = 100.0  # seconds; the population is depth-20 rows over this


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def band(z):
    return "low" if z < 9.0 else ("mid" if z <= 1.0e5 else "seam")


c0 = ro(BASE + f"{STORE}-shard0000.sqlite")
zmap = dict(c0.execute("select serial, z from redshift"))
kmap = dict(
    c0.execute(
        "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
        "join wavenumber w on w.serial = e.wavenumber_serial"
    )
)
c0.close()

rows = []
for i in range(4):
    c = ro(BASE + f"{STORE}-shard{i:04d}.sqlite")
    for s, ks, qs, rs, zs, ct, md in c.execute(
        "select serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, "
        "r_wavenumber_exit_serial, z_response_serial, compute_time, metadata "
        'from QuadSourceIntegral where "WKB_Levin_max_depth" = 20 and compute_time > ?',
        (MIN_TIME,),
    ):
        rows.append((i, s, kmap[ks], kmap[qs], kmap[rs], zmap[zs], ct, json.loads(md)))
    c.close()

print(f"store {STORE}; depth-20 rows with compute_time > {MIN_TIME:g} s: {len(rows)}")
print(f"sum of their compute_time: {sum(r[6] for r in rows) / 3600:.2f} CPU-hr")

elapsed = collections.Counter()
regions = collections.Counter()
top = collections.Counter()
band_el = collections.Counter()
for i, s, k, q, r, z, ct, m in rows:
    best = None
    for sub in m["WKB_Levin"]["subintervals"]:
        regime = "".join(n for n, f in zip("Gqr", sub["regime"]) if f)
        for g in sub["groups"]:
            key = (regime, g["label"])
            elapsed[key] += g["elapsed"]
            regions[key] += g["regions"]
            band_el[(band(z), key)] += g["elapsed"]
            if best is None or g["elapsed"] > best[1]:
                best = (key, g["elapsed"])
    geometry = "r==k" if r == k else "r!=k"
    top[(geometry, best[0])] += 1

total_el = sum(elapsed.values())
print(f"sum of per-group Levin elapsed: {total_el / 3600:.2f} CPU-hr\n")
print(
    "per-group Levin elapsed, summed over the population: (regime, group) CPU-hr share regions"
)
for key, v in elapsed.most_common():
    print(f"   {key!s:22} {v / 3600:8.2f} {v / total_el:7.1%} {regions[key]:>12,}")
print("\nthe most expensive group of each row: (geometry, (regime, group)) rows")
for key, v in top.most_common():
    print(f"   {key!s:36} {v:5d}")
print("\nper-group Levin elapsed by z_response band: (band, (regime, group)) CPU-hr")
for key, v in sorted(band_el.items(), key=lambda t: (t[0][0], -t[1])):
    if v > 360.0:
        print(f"   {key!s:36} {v / 3600:8.2f}")
