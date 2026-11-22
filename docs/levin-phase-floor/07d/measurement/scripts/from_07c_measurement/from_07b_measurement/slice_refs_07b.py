"""levin-phase-floor prompt 07b Part A, section 2.4: each reference call's captured blob as a file of its own, for
tools.filon_reference (one captured call per pickle), after 07a's refs_slice_07a.py (a record). New in 07b.

One file per integrand, sid ``ref-<selection sid>``: R1 from 07's cross-check capture of the row (the blob 07a Part A's
manifest names, x###); R2 and R3 but the pairs from this Part's captures at h (out/captures/captures_h.json); the pairs
from 07a Part A's 1e-7 capture (p14501_7, p35006_7). A blob is sliced only once the replays show it reproduces 07a's
census at its rtol and, where the manifest has it, at the other (out/replays/tree/<sid>.json and <sid>~o / R1's -7 and
-8, ``reproduces_census``). Each blob is copied at its recorded offset and length and checked against its SHA-256; nothing
is unpickled. Writes state/refs/inputs/<sid>.pkl (refusing to overwrite) and out/refs/inputs.json.
Usage: python -B slice_refs_07b.py
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

MAN = WIDER / "state" / "calls_07b.json"
SEL = WIDER / "state" / "selection_07b.json"
REP = WIDER / "out" / "replays" / "tree"


def main():
    out_dir = WIDER / "state" / "refs" / "inputs"
    out_json = WIDER / "out" / "refs" / "inputs.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text())["calls"]
    sel = json.loads(SEL.read_text())["calls"]
    doc = {"header": header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: the reference calls, one file each",
                            [MAN, SEL]), "inputs": {}}
    for e in sel:
        s = e["sid"]
        if e["set"] == "R1":
            reps = [f"{s}-8", f"{s}-7"]
            src = man[f"{s}-8"]
        elif e["set"] == "R3-pair":
            row = s.split("-", 1)[1]
            tag = "p" + row.split("-")[1]
            reps = [f"{tag}_7", f"{tag}_8"]
            src = man[f"{tag}_7"]
        else:
            reps = [s, f"{s}~o"]
            src = man[s]
        checked = {}
        for r in reps:
            p = REP / f"{r}.json"
            checked[r] = json.loads(p.read_text())["reproduces_census"] if p.exists() else None
        if not checked[reps[0]]:
            raise SystemExit(f"{s}: its replay at its own rtol does not reproduce 07a's census: {checked}")
        with open(src["path"], "rb") as f:
            f.seek(src["offset"])
            blob = f.read(src["length"])
        if hashlib.sha256(blob).hexdigest() != src["sha256"]:
            raise SystemExit(f"{s}: sha256 differs")
        sid = f"ref-{s}"
        p = out_dir / f"{sid}.pkl"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_bytes(blob)
        doc["inputs"][sid] = {"path": str(p), "sha256": sha256(p), "bytes": len(blob), "selection_sid": s, "set": e["set"],
                              "kind": e["kind"], "unit": e["unit"], "index": e["index"], "rtol": e["rtol"],
                              "replays": reps, "replays_reproduce": checked}
        print(f"{sid}: {e['kind']} {e['unit']} #{e['index']} replays {checked}")
    out_json.write_text(json.dumps(doc, indent=1))
    print(f"{len(doc['inputs'])} inputs; wrote {out_json} sha256 {sha256(out_json)}")


if __name__ == "__main__":
    main()
