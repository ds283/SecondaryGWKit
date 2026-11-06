"""levin-phase-floor 06m census agent: the units whose cache_built (an excluded key) differs from route B's, with both
sides, read only. Prints them."""
import json, sys
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06m/census/scripts")
import compare_06m as C
doc = json.load(open("out/gate_m/compare_m.json"))
units = [u["unit"] for u in doc["units"] if any(p.startswith("data.cache_built") for p in u["excl_differing"])]
idx = json.loads(C.INDEX.read_text())["units"]
R = __import__("pathlib").Path(doc["run_dir"])
mine = {}
with open(R / "checkpoint.jsonl") as f:
    for line in f:
        if any(f'"{u}"' in line[:200] for u in units):
            r = json.loads(line); mine[r["unit"]] = r["data"]
for u in units:
    e = idx[u]
    with open(C.RUN06L_B / "checkpoint.jsonl", "rb") as f:
        f.seek(e["offset"]); g = json.loads(f.read(e["length"]))["data"]
    a, b = mine[u]["cache_built"], g["cache_built"]
    print(u, "worker", mine[u]["worker"])
    for x, y in zip(a, b):
        if x != y:
            print("   m:", x); print("   B:", y)
    print("   lengths", len(a), len(b))
