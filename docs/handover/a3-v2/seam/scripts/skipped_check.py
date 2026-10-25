"""Extra read-only check (log 11, IMPLEMENTATION CHOICE): which rows carry a 'skipped' breakpoint, against q == r.
    ./venv/bin/python -I var/handover-11/scripts/skipped_check.py <primary.sqlite>
"""
import json, os, sqlite3, sys
from collections import Counter

def ro(p): return sqlite3.connect(f"file:{os.path.abspath(p)}?mode=ro", uri=True)
primary = sys.argv[1]
p = ro(primary)
shards = [os.path.join(os.path.dirname(os.path.abspath(primary)), os.path.basename(f)) for (f,) in p.execute("select filename from shards order by serial")]
c0 = ro(shards[0])
kmap = dict(c0.execute("select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on w.serial = e.wavenumber_serial"))
cnt = Counter()
for sh in shards:
    c = ro(sh)
    for q, r, md in c.execute("select q_wavenumber_exit_serial, r_wavenumber_exit_serial, metadata from QuadSourceIntegral"):
        part = json.loads(md)["partition"]
        sk = bool(part["skipped"])
        eq = kmap[q] == kmap[r]
        cnt[(("q==r" if eq else "q!=r"), ("skipped" if sk else "no skipped"))] += 1
        if sk:
            cnt[("skipped factors", tuple(e["factor"] for e in part["skipped"]), "crossover_z_q==crossover_z_r", part["crossover_z_q"] == part["crossover_z_r"])] += 1
for k, v in sorted(cnt.items(), key=str):
    print(v, k)
