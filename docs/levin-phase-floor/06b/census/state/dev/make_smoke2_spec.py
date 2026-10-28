"""06b dev: a spec for a smoke test of W3's recomputation: b, 1e-7, the first three rows of k 1:102
(rows.json order) with a call phase-limited in 06a and a sign change of theta' in 06a's scan. Not a
record of the run."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06b/census/scripts")
import w0
from prov import RUN06A, sha256
doc, by05, by_name = w0.load_rows()
cap = json.loads(w0.CAPTURE_JSON.read_text())["capture"]
want = set()
for line in open(RUN06A / "checkpoint.jsonl"):
    d = json.loads(line)["data"]
    if d["rtol"] != "1e-7" or d["k_key"] != "1:102":
        continue
    scan = {s["index"]: s for s in d["scan"] if s}
    if any(c["phase_limited"] and scan[c["index"]]["n_sign_changes"] > 0 for c in d["calls"]):
        want.add(d["row"])
rows = [r for r in doc["rows"] if r["name"] in want][:3]
out = Path(sys.argv[1])
spec = {"tree": "b", "name": "S2", "units": [w0.unit(r, "1e-7", "b", cap) for r in rows], "out_dir": str(out), "emit_fd": None,
        "emit_file": str(out / "emit.jsonl"), "rows_sha256": sha256(w0.ROWS_JSON), "w2_capture_sha256": sha256(w0.CAPTURE_JSON)}
(out / "spec.json").write_text(json.dumps(spec, indent=1))
print([u["uid"] for u in spec["units"]])
