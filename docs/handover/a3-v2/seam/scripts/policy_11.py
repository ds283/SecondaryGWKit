"""Step 3: the GkSourcePolicyData summary, from stdout.log verbatim and from the shards (mode=ro).

    ./venv/bin/python -I var/handover-11/scripts/policy_11.py <primary.sqlite> <stdout.log>

Exit 1 if any (type, quality) cell disagrees with the log's block, or the block is missing.
"""

import collections
import os
import re
import sqlite3
import sys

TYPE = {0: "numeric", 1: "WKB", 2: "mixed", 99: "fail"}
QUALITY = {0: "complete", 1: "acceptable", 2: "marginal", 3: "minimal", 4: "incomplete"}


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


primary, logpath = sys.argv[1], sys.argv[2]
p = ro(primary)
shards = [
    os.path.join(os.path.dirname(os.path.abspath(primary)), os.path.basename(f))
    for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
]
p.close()

lines = open(logpath).read().split("\n")
start = next(
    (i for i, l in enumerate(lines) if "GkSourcePolicyData SUMMARY STATISTICS" in l), None
)
if start is None:
    print("!! SUMMARY BLOCK MISSING")
    sys.exit(1)
# the block runs to the first line after it that is not blank / policy / counts
end = start
for i in range(start + 1, len(lines)):
    if re.match(r"^\s*(\* Policy|Total instances|>> |\w+: \d+ instances)", lines[i]) or not lines[i].strip():
        end = i
    else:
        break
while not lines[end].strip():
    end -= 1
print(f"== {os.path.basename(logpath)} lines {start + 1}-{end + 1}, verbatim")
for i in range(start, end + 1):
    print(f"{i + 1:5d}: {lines[i]}")

# parse the log
log_total = None
log_cells = {}
cur_type = None
for l in lines[start : end + 1]:
    m = re.search(r"Total instances: (\d+)", l)
    if m:
        log_total = int(m.group(1))
    m = re.search(r">> (\w+): (\d+) instances", l)
    if m:
        cur_type = m.group(1)
        continue
    m = re.match(r"\s+(\w+): (\d+) instances = [\d.]+% of this type", l)
    if m and cur_type:
        log_cells[(cur_type, m.group(1))] = int(m.group(2))

c0 = ro(shards[0])
zmap = dict(c0.execute("select serial, z from redshift"))
kmap = dict(
    c0.execute(
        "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on w.serial = e.wavenumber_serial"
    )
)
c0.close()

table_cells = collections.Counter()
policies = collections.Counter()
per_shard = []
nonc = []
n = 0
for i, sh in enumerate(shards):
    c = ro(sh)
    cnt = 0
    for serial, pol, kser, ty, ql, xz, zr in c.execute(
        "select d.serial, d.policy_serial, d.wavenumber_exit_serial, d.type, d.quality, d.crossover_z, s.z_response_serial "
        "from GkSourcePolicyData d join GkSource s on s.serial = d.source_serial"
    ):
        cnt += 1
        n += 1
        t, q = TYPE[ty], QUALITY[ql]
        table_cells[(t, q)] += 1
        policies[pol] += 1
        if q != "complete" or t == "fail":
            nonc.append((i, serial, kmap[kser], zmap[zr], t, q, xz))
    per_shard.append(cnt)
    c.close()

print(f"\n== GkSourcePolicyData rows read from the shards (mode=ro): {n}; by shard {per_shard}")
print(f"   policy_serial values: {dict(policies)}")
print(f"\n== cell by cell: log block vs table")
cells = sorted(set(log_cells) | set(table_cells))
ok = True
print(f"   {'type':>8} {'quality':>10} {'log':>6} {'table':>6}  agree")
for c in cells:
    a, b = log_cells.get(c, 0), table_cells.get(c, 0)
    ok &= a == b
    print(f"   {c[0]:>8} {c[1]:>10} {a:6d} {b:6d}  {a == b}")
ok &= log_total == n == sum(log_cells.values())
print(f"   total: log header {log_total}, sum of log cells {sum(log_cells.values())}, table {n}")
print(f"\nEVERY CELL AGREES: {ok}")

print(f"\n== rows that are not 'complete' (or are 'fail'): {len(nonc)}")
print(f"   {'shard':>5} {'serial':>7} {'k /Mpc':>12} {'z_response':>14} {'type':>8} {'quality':>10} {'crossover_z':>14}")
for sh, serial, k, z, t, q, xz in nonc:
    print(f"   {sh:5d} {serial:7d} {k:12.6e} {z:14.8g} {t:>8} {q:>10} {xz if xz is None else format(xz, '14.8g')}")
fails = [r for r in nonc if r[4] == "fail" or r[5] == "incomplete"]
print(f"\n'fail' or 'incomplete' rows: {len(fails)}")
sys.exit(0 if ok else 1)
