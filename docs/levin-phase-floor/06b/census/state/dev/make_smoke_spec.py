"""06b dev: a one-unit spec for a smoke test of the changed worker (b, 1e-7-1-34599). Not a record of the run."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06b/census/scripts")
import w0
from prov import sha256
doc, by05, by_name = w0.load_rows()
cap = json.loads(w0.CAPTURE_JSON.read_text())["capture"]
D = by_name["1-34599"]
out = Path(sys.argv[1])
spec = {"tree": "b", "name": "S1", "units": [w0.unit(D, "1e-7", "b", cap)], "out_dir": str(out), "emit_fd": None,
        "emit_file": str(out / "emit.jsonl"), "rows_sha256": sha256(w0.ROWS_JSON), "w2_capture_sha256": sha256(w0.CAPTURE_JSON)}
(out / "spec.json").write_text(json.dumps(spec, indent=1))
print(json.dumps(spec["units"]))
