"""levin-phase-floor prompt 06g section 5, the references: each capture as a file of its own. New in
06g, after 06e's references/slice_inputs.py (kept in state/scripts_06e_references_reference/).

``tools.filon_reference`` takes one captured call per pickle file (``python -m tools.filon_reference
PICKLE ...``). The captures' workers append every call to one ``inputs_<worker>.pkl``; this script
copies each call's blob, at the offset and length x1_capture.py recorded, into state/inputs/<sid>.pkl
and checks it against the recorded SHA-256. Only captures that reproduce the census (x1_capture.py's
``capture_matches_census`` and check_captures.py's ``reproduces_census``) are sliced. It also checks
06i's 13 inputs on this host (var/levin-phase-floor-06e/census/references/inputs/, read in place,
never copied or written) against 06i's inputs_SHA256SUMS.txt. Nothing is unpickled.

Writes state/inputs/<sid>.pkl (refusing to overwrite) and out/inputs.json.

Usage: python -B slice_refs.py CAPTURE_JSON CHECK_JSON SELECTION_JSON
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
    out_json = RC.ROOT / "out" / "inputs.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists; not overwritten")
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references' captured calls, one file each; 06i's inputs checked",
        [cap_p, chk_p, sel_p, RC.I06_SUMS],
    ):
        print(l)
    sel = {(e["unit"], e["index"]): e for e in json.loads(sel_p.read_text())}
    chk = {(r["unit"], r["index"]): r for r in json.loads(chk_p.read_text())}
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"captures": {}, "inputs_06i": {}}
    ok = True
    for c in json.loads(cap_p.read_text()):
        k = (c["unit"], c["index"])
        e = sel[k]
        good = bool(c.get("capture_matches_census")) and bool(
            chk.get(k, {}).get("reproduces_census")
        )
        if not good:
            print(f"{e['sid']}: not sliced (does not reproduce the census)")
            ok = False
            continue
        with open(Path(c["dir"]) / c["pickle"], "rb") as f:
            f.seek(c["offset"])
            blob = f.read(c["length"])
        h = hashlib.sha256(blob).hexdigest()
        if h != c["sha256"] or len(blob) != c["length"]:
            raise SystemExit(
                f"{e['sid']}: blob sha256 or length differs from the capture record"
            )
        dst = out_dir / f"{e['sid']}.pkl"
        if dst.exists():
            raise SystemExit(f"{dst} exists; not overwritten")
        dst.write_bytes(blob)
        doc["captures"][e["sid"]] = {
            "unit": c["unit"],
            "index": c["index"],
            "kind": e["kind"],
            "pickle": str(dst),
            "sha256": h,
            "bytes": len(blob),
            "source": f"{c['dir']}/{c['pickle']}@{c['offset']}",
        }
        print(
            f"{e['sid']} {c['unit']} #{c['index']}: {len(blob)} bytes sha256 {h} -> {dst.name}"
        )
    sums = {}
    for line in RC.I06_SUMS.read_text().splitlines():
        h, p = line.split()
        sums[Path(p).name] = h
    for n in RC.I06_SAMPLES:
        p = RC.I06_INPUTS / f"s{n:02d}.pkl"
        h = RC.sha256(p)
        same = h == sums[p.name]
        ok &= same
        doc["inputs_06i"][f"s{n:02d}"] = {
            "path": str(p),
            "sha256": h,
            "equals_06i_sums": same,
        }
        print(
            f"06i input {p}: sha256 {h} {'EQUALS' if same else 'DIFFERS FROM'} 06i's inputs_SHA256SUMS.txt"
        )
    out_json.write_text(json.dumps(doc, indent=1))
    print(
        f"{len(doc['captures'])} captures sliced; wrote {out_json} sha256 {RC.sha256(out_json)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
