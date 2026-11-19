"""levin-phase-floor prompt 06j: each capture as a file of its own. New in 06j, after 06g's slice_refs.py (which
also checks 06i's inputs; 06j has none). Copies each call's blob, at the offset and length x1_capture.py
recorded, into state/inputs/<sid>.pkl and checks it against the recorded SHA-256. Only captures that reproduce
the census (x1_capture.py's ``capture_matches_census`` and check_captures.py's ``reproduces_census``) are sliced.
Nothing is unpickled. Writes state/inputs/<sid>.pkl (refusing to overwrite) and out/inputs.json.

Usage: python -B slice_06j.py CAPTURE_JSON CHECK_JSON SELECTION_JSON [OUT_JSON]

prompt 06l: the optional OUT_JSON (default out/inputs.json) names the output, so that a second slicing does not
overwrite the first run's record.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def main():
    cap_p, chk_p, sel_p = (Path(a) for a in sys.argv[1:4])
    out_dir = RC.ROOT / "state" / "inputs"
    out_json = Path(sys.argv[4]) if len(sys.argv) > 4 else RC.ROOT / "out" / "inputs.json"  # prompt 06l: OUT_JSON
    if out_json.exists():
        raise SystemExit(f"{out_json} exists; not overwritten")
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06j: the captured calls, one file each",
                    [cap_p, chk_p, sel_p]):
        print(l)
    sel = {(e["unit"], e["index"]): e for e in json.loads(sel_p.read_text())}
    chk = {(r["unit"], r["index"]): r for r in json.loads(chk_p.read_text())}
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"captures": {}, "not_sliced": []}
    for c in json.loads(cap_p.read_text()):
        k = (c["unit"], c["index"])
        e = sel[k]
        good = bool(c.get("capture_matches_census")) and bool(chk.get(k, {}).get("reproduces_census"))
        if not good:
            print(f"{e['sid']}: not sliced (does not reproduce the census)")
            doc["not_sliced"].append(e["sid"])
            continue
        with open(Path(c["dir"]) / c["pickle"], "rb") as f:
            f.seek(c["offset"])
            blob = f.read(c["length"])
        h = hashlib.sha256(blob).hexdigest()
        if h != c["sha256"] or len(blob) != c["length"]:
            raise SystemExit(f"{e['sid']}: blob sha256 or length differs from the capture record")
        dst = out_dir / f"{e['sid']}.pkl"
        if dst.exists():
            raise SystemExit(f"{dst} exists; not overwritten")
        dst.write_bytes(blob)
        doc["captures"][e["sid"]] = {"unit": c["unit"], "index": c["index"], "kind": e["kind"], "pickle": str(dst),
                                     "sha256": h, "bytes": len(blob), "source": f"{c['dir']}/{c['pickle']}@{c['offset']}"}
        print(f"{e['sid']} {c['unit']} #{c['index']}: {len(blob)} bytes sha256 {h} -> {dst.name}")
    out_json.write_text(json.dumps(doc, indent=1))
    print(f"{len(doc['captures'])} captures sliced, {len(doc['not_sliced'])} not; wrote {out_json} sha256 {RC.sha256(out_json)}")
    return 0 if not doc["not_sliced"] else 1


if __name__ == "__main__":
    sys.exit(main())
