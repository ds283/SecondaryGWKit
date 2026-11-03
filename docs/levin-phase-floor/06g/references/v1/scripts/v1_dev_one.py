"""levin-phase-floor 06g addendum 1, development only: one V1 unit through the harness (pass 1 of
v1_capture.py for one row and tree), to see that it runs and to time it. Writes ../state/dev/."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1_capture as V
wid, tree = sys.argv[1], sys.argv[2]
rows = {r["wid"]: r for r in json.loads((V.ROOT / "out" / "v1_rows.json").read_text())}
rec = rows[wid][V.STORE_TAG[tree]]
t0 = time.time()
out = V.run_workers([(f"DEV-{wid}-{tree}", tree, V.ROOT / "state" / "dev" / f"{wid}-{tree}", V.unit_spec(tree, rec))])
rc, unit, bye = out[f"DEV-{wid}-{tree}"]
print("rc", rc, "status", None if unit is None else unit["status"], "wall", round(time.time() - t0, 1), "bye ray", None if bye is None else bye["ray_initialized"])
if unit and unit.get("result"):
    print("total", unit["result"]["total"]["hex"], "store", rec["total"], unit["result"]["total"]["hex"] == rec["total"])
    want = rec["group"]
    for c in unit["calls"]:
        print(c["index"], c["notify_label"][-50:], c["value"]["hex"], c["abserr"]["hex"], c["value"]["hex"] == want["value"] and c["abserr"]["hex"] == want["abserr"])
