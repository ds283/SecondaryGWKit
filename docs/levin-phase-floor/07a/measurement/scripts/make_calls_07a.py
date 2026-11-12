"""levin-phase-floor prompt 07a Part A, section 2.1: the manifest of every call replayed, with its capture. New in 07a.

For each call: its set (``v4``: V4's 25; ``pairs``: the two cross-check pairs at both rtols; ``nowindow``: five calls
of the pairs' units with no window, for section 2.3's check; ``x396``: the class-(c) rows at 1e-8), its capture blob
(path, offset, length, SHA-256 -- 07's out/v4/inputs.json for V4, 07a's out/pairs_capture/captures.json for the pairs
and the no-window calls, 07's out/xcheck/captures.json for the 396), and the census record it is compared with
(state/census_refs.pkl's key; for the no-window calls, 07a's capture check). **The 396:** 07 captured 390 of them at
1e-7 and 6 at 1e-8, and section 2.1 item 3 names the calls at 1e-8; a capture taken at 1e-7 is replayed with the
keyword arguments of 07's census call at 1e-8 (``rtol`` and ``atol`` from that call's record; ``override``), and the
replay at 07's code must reproduce that 1e-8 record bit for bit, which shows the integrand is the same.

Writes state/calls_07a.json (refuses to overwrite). Usage: python -B make_calls_07a.py
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
C07 = MAIN / "var" / "levin-phase-floor-07" / "census"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = ROOT / "state" / "calls_07a.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    refs = json.loads((ROOT / "out" / "census_refs.json").read_text())["calls"]
    calls = {}
    v4in = json.loads((C07 / "out" / "v4" / "inputs.json").read_text())["inputs"]
    for sid, e in sorted(v4in.items()):
        p = Path(e["path"])
        calls[sid] = {"set": "v4", "path": str(p), "offset": 0, "length": p.stat().st_size, "sha256": e["sha256"],
                      "census": sid, "override": None}
    for e in json.loads((ROOT / "out" / "pairs_capture" / "captures.json").read_text()):
        if not e["capture_matches_census"]:
            raise SystemExit(f"{e['sid']}: the capture does not match")
        p = Path(e["dir"]) / e["pickle"]
        st = "pairs" if e["sid"].startswith("p") else "nowindow"
        calls[e["sid"]] = {"set": st, "path": str(p), "offset": e["offset"], "length": e["length"],
                           "sha256": e["sha256"], "census": e["sid"] if st == "pairs" else None, "override": None,
                           "census_unit": e["unit"], "census_index": e["index"], "census_value": e["census_value"],
                           "census_abserr": e["census_abserr"], "census_num_regions": e["census_num_regions"],
                           "census_max_depth": e["census_max_depth"], "census_converged": e["census_converged"]}
    for e in json.loads((C07 / "out" / "xcheck" / "captures.json").read_text()):
        if not e["capture_matches_census"]:
            raise SystemExit(f"{e['sid']}: the capture does not match")
        p = C07 / e["dir"] / e["pickle"]
        r = refs[e["sid"]]
        ov = None
        if e["unit"] != r["unit"]:
            ov = {"rtol": float(r["rtol"]), "atol": float(r["atol"]), "from_unit": e["unit"], "to_unit": r["unit"]}
        calls[e["sid"]] = {"set": "x396", "path": str(p), "offset": e["offset"], "length": e["length"],
                           "sha256": e["sha256"], "census": e["sid"], "override": ov}
    # every blob's SHA-256, now
    for sid, c in calls.items():
        with open(c["path"], "rb") as f:
            f.seek(c["offset"])
            b = f.read(c["length"])
        if hashlib.sha256(b).hexdigest() != c["sha256"]:
            raise SystemExit(f"{sid}: capture blob has another SHA-256")
    doc = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "census_refs_json_sha256": sha256(ROOT / "out" / "census_refs.json"), "calls": calls}
    out.write_text(json.dumps(doc, indent=1))
    from collections import Counter

    print(Counter(c["set"] for c in calls.values()), "override:", sum(1 for c in calls.values() if c["override"]))
    print("wrote", out, sha256(out))


if __name__ == "__main__":
    sys.exit(main())
