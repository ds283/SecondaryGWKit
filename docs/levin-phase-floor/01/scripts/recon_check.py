"""Re-derive README 0.2's reconnaissance table (measure/out/recon_groups.txt) from groups_01.csv
(levin-phase-floor prompt 01, step 5). Same population: depth-20 rows with compute_time > 100 s.
Prints in recon_groups.py's own format so the two outputs can be diffed line by line.

    ./venv/bin/python -I var/levin-phase-floor-01/scripts/recon_check.py var/levin-phase-floor-01/out/groups_01.csv
"""
import collections, csv, hashlib, subprocess, sys

MIN_TIME = 100.0
path = sys.argv[1]
print(f"# recon_check.py sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
print(f"# input {path} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
print(f"# git HEAD {subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
with open(path) as fh:
    body = [l for l in fh if not l.startswith("#")]
recs = [r for r in csv.DictReader(body)
        if r["WKB_Levin_max_depth"] == "20" and float(r["compute_time"]) > MIN_TIME]

rows = {}
for r in recs:
    rows[(r["shard"], r["serial"])] = r
print(f"store handover-A3-v2-lambdacdm; depth-20 rows with compute_time > {MIN_TIME:g} s: {len(rows)}")
print(f"sum of their compute_time: {sum(float(r['compute_time']) for r in rows.values()) / 3600:.2f} CPU-hr")

elapsed = collections.Counter(); regions = collections.Counter()
top = collections.Counter(); band_el = collections.Counter()
best = {}
for r in recs:
    key = (r["regime"], r["label"])
    el = float(r["elapsed"])
    elapsed[key] += el
    regions[key] += int(r["regions"])
    band_el[(r["band"], key)] += el
    rk = (r["shard"], r["serial"])
    if rk not in best or el > best[rk][1]:
        best[rk] = (key, el, r)
for rk, (key, el, r) in best.items():
    geometry = "r==k" if r["r"] == r["k"] else "r!=k"
    top[(geometry, key)] += 1

total_el = sum(elapsed.values())
print(f"sum of per-group Levin elapsed: {total_el / 3600:.2f} CPU-hr\n")
print("per-group Levin elapsed, summed over the population: (regime, group) CPU-hr share regions")
for key, v in elapsed.most_common():
    print(f"   {key!s:22} {v / 3600:8.2f} {v / total_el:7.1%} {regions[key]:>12,}")
print("\nthe most expensive group of each row: (geometry, (regime, group)) rows")
for key, v in top.most_common():
    print(f"   {key!s:36} {v:5d}")
print("\nper-group Levin elapsed by z_response band: (band, (regime, group)) CPU-hr")
for key, v in sorted(band_el.items(), key=lambda t: (t[0][0], -t[1])):
    if v > 360.0:
        print(f"   {key!s:36} {v / 3600:8.2f}")

# the README 0.2 shares by family
fam = collections.Counter()
for (reg, lab), v in elapsed.items():
    fam[lab if reg != "Gqr" else "Gqr (four groups)"] += v
print("\nREADME 0.2 families: " + "; ".join(f"{k}: {v / total_el:.1%}" for k, v in fam.most_common()))
print(f"low band share of CPU-hr: {sum(v for (b, _), v in band_el.items() if b == 'low') / 3600:.2f} of {total_el / 3600:.2f}")
