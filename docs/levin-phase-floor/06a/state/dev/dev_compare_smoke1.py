# dev only (not a gate): compare the smoke test's unit with 06's s unit
import json, sys
sys.path.insert(0, "scripts")
import c0
d = [json.loads(l)["data"] for l in open("state/dev/smoke1/ledger.jsonl")][0]
rec = c0.load_census_record("state/dev/smoke1", d)
sd, srec = c0.load_06("s-1e-7-low-G+r-1")
ok, lines = c0.compare_to_06(d, rec, sd, srec)
print(ok); print("\n".join(lines))
