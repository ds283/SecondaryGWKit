"""levin-phase-floor prompt 06e section 5, Z3: the captured inputs of the samples, copied out of
depth20-tiling's captures (var/levin-phase-floor-06c/depth20-tiling/state/capture/, read only) into
references/inputs/sNN.pkl, each sliced at captured.json's offset and length and checked against its
SHA-256; where the orchestrator's re-check already holds a copy (orchestrator_check/inputs/sNN.pkl),
it is compared byte for byte. New in 06e. Read-only outside references/inputs/.

Usage: python -B slice_inputs.py  (no tree needed; nothing is unpickled)"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D20T = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/depth20-tiling")
SAMPLES = (1, 2, 3, 8, 9, 10, 13, 14, 15, 22, 23, 25, 27)  # prompt 06e section 5, Z3


def main():
    print("script", Path(__file__).name, "sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cj = D20T / "state" / "captured.json"
    print("captured.json sha256", hashlib.sha256(cj.read_bytes()).hexdigest())
    cap = {e["sample_no"]: e for e in json.loads(cj.read_text())}
    ok = True
    out = {}
    for n in SAMPLES:
        e = cap[n]
        src = D20T / e["dir"] / e["pickle"]
        with open(src, "rb") as f:
            f.seek(e["offset"])
            blob = f.read(e["length"])
        h = hashlib.sha256(blob).hexdigest()
        good = h == e["sha256"] and len(blob) == e["length"] and e["capture_matches_census"] and e["capture_error"] is None
        dst = HERE / "inputs" / f"s{n:02d}.pkl"
        if dst.exists():
            raise SystemExit(f"{dst} exists; not overwritten")
        dst.write_bytes(blob)
        orch = D20T / "orchestrator_check" / "inputs" / f"s{n:02d}.pkl"
        same_orch = None if not orch.exists() else orch.read_bytes() == blob
        ok &= good and same_orch is not False
        out[n] = {"unit": e["unit"], "index": e["index"], "notify_label": e["notify_label"], "rtol": e["rtol"],
                  "block": e["block"], "x_span": e["x_span"], "census06c_value": e["census_value"],
                  "census06c_abserr": e["census_abserr"], "pickle": dst.name, "sha256": h, "bytes": len(blob)}
        print(f"s{n:02d} {e['unit']} #{e['index']} {e['block']}: {len(blob)} bytes sha256 {h} "
              f"{'EQUALS' if h == e['sha256'] else 'DIFFERS FROM'} captured.json; capture_matches_census {e['capture_matches_census']}; "
              f"orchestrator's copy: {'absent' if same_orch is None else ('identical' if same_orch else 'DIFFERS')}")
    (HERE / "inputs" / "inputs.json").write_text(json.dumps(out, indent=1))
    print("ALL GOOD" if ok else "SOME NOT GOOD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
