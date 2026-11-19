"""levin-phase-floor prompt 07 section 6, V4: each of the 24 captures (and z12's calibration capture) as a file of its
own, for tools.filon_reference (one captured call per pickle). After 06g's slice_refs.py (a record, not edited). Each
call's blob is copied from its capture worker's inputs pickle, at the offset and length x1_capture.py recorded, and
checked against the recorded SHA-256; only captures that reproduce the census (out/v4/check_captures.json) are
sliced. Nothing is unpickled. Writes state/v4/inputs/<sid>.pkl (refusing to overwrite) and out/v4/inputs.json.
Usage: python -B slice_v4_07.py"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

CAPS = WIDER / "out" / "v4" / "captures.json"
CHECK = WIDER / "out" / "v4" / "check_captures.json"


def main():
    out_dir = WIDER / "state" / "v4" / "inputs"
    out_json = WIDER / "out" / "v4" / "inputs.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V4's captured calls, one file each", [CAPS, CHECK])
    chk = {r["sid"]: r for r in json.loads(CHECK.read_text())}
    doc = {"header": hdr, "inputs": {}}
    for e in json.loads(CAPS.read_text()):
        sid = e["sid"]
        if not (e["capture_matches_census"] and chk[sid]["reproduces_census"]):
            raise SystemExit(f"{sid}: the capture does not reproduce the census")
        with open(Path(e["dir"]) / e["pickle"], "rb") as f:
            f.seek(e["offset"])
            blob = f.read(e["length"])
        if hashlib.sha256(blob).hexdigest() != e["sha256"]:
            raise SystemExit(f"{sid}: sha256 differs")
        p = out_dir / f"{sid}.pkl"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_bytes(blob)
        doc["inputs"][sid] = {"path": str(p), "sha256": sha256(p), "bytes": len(blob), "unit": e["unit"], "index": e["index"],
                              "rtol": e["unit"].split("-")[1] + "-" + e["unit"].split("-")[2]}
        print(f"{sid}: {p} sha256 {doc['inputs'][sid]['sha256']} ({len(blob)} bytes)")
    out_json.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out_json} sha256 {sha256(out_json)}")


if __name__ == "__main__":
    main()
