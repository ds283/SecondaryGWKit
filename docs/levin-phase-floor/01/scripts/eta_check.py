"""Which rows give the negative eta(z)/eta_response ratio in census_01.txt (a)? Read-only, mode=ro.
    ./venv/bin/python -I var/levin-phase-floor-01/scripts/eta_check.py var/levin-phase-floor-01/out/groups_01.csv
"""
import csv, hashlib, sqlite3, sys, importlib.util
print(f"# eta_check.py sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
spec = importlib.util.spec_from_file_location("f", "docs/handover/a3-v2/findings/depth20_floor.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
body = [l for l in open(sys.argv[1]) if not l.startswith("#")]
Fneg = sum(1 for r in csv.DictReader(body) if float(r["F"]) <= 0)
print(f"group records with F <= 0 in groups_01.csv: {Fneg}")
c = sqlite3.connect("file:var/datastores/handover-A3-v2-lambdacdm-shard0000.sqlite?mode=ro", uri=True)
zmap = dict(c.execute("select serial, z from redshift"))
seen = {}
for i in range(4):
    c = sqlite3.connect(f"file:var/datastores/handover-A3-v2-lambdacdm-shard{i:04d}.sqlite?mode=ro", uri=True)
    for zs, er in c.execute("select z_response_serial, eta_response from QuadSourceIntegral"):
        seen.setdefault(zmap[zs], set()).add(er)
bad = [(z, sorted(v)) for z, v in sorted(seen.items()) if any(f.eta(z) / e < 0.99 or f.eta(z) / e > 1.01 for e in v)]
print(f"z_response values with eta ratio outside [0.99, 1.01]: {len(bad)} of {len(seen)}")
for z, v in bad:
    print(f"  z_response {z:.6g}: depth20_floor eta {f.eta(z):.6g} Mpc; stored eta_response {v}")
