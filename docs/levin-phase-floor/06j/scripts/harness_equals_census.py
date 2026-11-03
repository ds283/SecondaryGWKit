import json, hashlib, sys
from pathlib import Path
run = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06g-census-20261101T114315")
here = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/scripts")
seen = set(); n = 0
with open(run / "checkpoint.jsonl") as f:
    for line in f:
        d = json.loads(line)["data"]; n += 1
        seen.add(json.dumps(d.get("scripts_sha256"), sort_keys=True))
print("records", n, "distinct scripts_sha256 sets", len(seen))
rec = json.loads(next(iter(seen)))
ok = True
for name, h in sorted(rec.items()):
    mine = hashlib.sha256((here / name).read_bytes()).hexdigest()
    print(f"{name}: census {h} mine {mine} {'EQUAL' if mine == h else 'DIFFERS'}"); ok &= mine == h
print("ALL EQUAL" if ok and len(seen) == 1 else "NOT ALL EQUAL")
