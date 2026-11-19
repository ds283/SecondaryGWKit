"""levin-phase-floor prompt 07b extension, section 2.7.1 (X1): the leaf-reference specs for (e)'s remaining calls. New in
the extension.

For each of (e)'s 11 calls converged and dishonest (out/tables/scores_final.json ``qualification.e``), the replay under
(e) (out/replays/e/<sid>.json) is compared with the tree's (``regions_sha256``). Where they are equal, the call runs
the tree's code under (e), every leaf is unchanged, and Part A's leaf references serve (named, not re-read). Where they
differ, or Part A has no reference on the call's leaves (the other rtol's calls ``~o`` and the pairs, which section 2.2
did not attribute), a spec is written for leafref_07b.py:
- **spans:** every returned leaf that is not a window leaf (``window`` false): the Levin leaves and the Clenshaw-Curtis
  leaves between windows. The window leaves are not read: they hold the stationary point, whose segment the tool
  integrates directly at a cost proportional to its phase (Part A's deviation 16). Their summed true error is found by
  subtraction, from the call's reference (section 2.4's) less the non-window leaves' references.
- **points:** every leaf's ends, and on each trusted Levin leaf that is not a window leaf its halves', quarters' and
  eighths' break points (``_levin_leaf.break_point``'s expressions, as leafspec_07b.py), so that every carrying leaf's
  section 2.2 records are read in the same reading.
Writes state/ext/x1/<sid>.json (refuses to overwrite) and state/ext/x1/plan.json.
Usage: python -B x1_spec.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "state" / "ext" / "x1"
PART_A_REFS = {  # Part A's leaf references on these calls' tree leaves (out/leafrefs/)
    "v09": "carrier_v09.json",
    "W2-0-12977-5": "attr_W2-0-12977-5.json",
    "W2-1-2174-2": "attr_W2-1-2174-2.json",
    "W2-2-14345-3": "attr_W2-2-14345-3.json",
    "W2-2-34473-3": "attr_W2-2-34473-3.json",
}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    scores = ROOT / "out" / "tables" / "scores_final.json"
    q = json.loads(scores.read_text())["qualification"]["e"]["converged_dishonest"]
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"scores": str(scores), "scores_sha256": sha256(scores), "script_sha256": sha256(__file__), "calls": {}}
    for sid in q:
        re_ = json.loads((ROOT / "out" / "replays" / "e" / f"{sid}.json").read_text())
        rt = json.loads((ROOT / "out" / "replays" / "tree" / f"{sid}.json").read_text())
        same = re_["regions_sha256"] == rt["regions_sha256"] and re_["replay"] == rt["replay"]
        entry = {"e_equals_tree": same, "regions_e": re_["replay"]["num_regions"], "n_windows": re_["n_windows"]}
        if same and sid in PART_A_REFS:
            entry["reference"] = str(ROOT / "out" / "leafrefs" / PART_A_REFS[sid])
            plan["calls"][sid] = entry
            print(f"{sid}: (e) runs the tree's code; Part A's {PART_A_REFS[sid]} serves")
            continue
        points, spans, named = set(), [], []
        for i, L in enumerate(re_["leaves"]):
            lo, hi = float.fromhex(L["start"]), float.fromhex(L["end"])
            points |= {lo, hi}
            if L["window"]:
                continue
            spans.append((lo, hi))
            if L["type"] == "Levin" and L["trusted"]:
                cm = (hi + lo) / 2.0
                cL, cR = (cm + lo) / 2.0, (hi + cm) / 2.0
                pts = [cm, cL, cR] + [(q1 + q0) / 2.0 for q0, q1 in ((lo, cL), (cL, cm), (cm, cR), (cR, hi))]
                points |= set(pts)
                named.append({"name": f"e:{i}:8", "variant": "e", "i": i, "lo": L["start"], "hi": L["end"],
                              "depth": L["depth"], "type": L["type"], "eighths": True})
        spans = sorted(set(spans))
        spec = {"sid": sid, "capture": re_["header"]["capture"], "spans": [[a.hex(), b.hex()] for a, b in spans],
                "points": [p.hex() for p in sorted(points)], "named": named, "all_leaves": None,
                "replay": str(ROOT / "out" / "replays" / "e" / f"{sid}.json"),
                "replay_sha256": sha256(ROOT / "out" / "replays" / "e" / f"{sid}.json"),
                "window_leaves_not_read": [i for i, L in enumerate(re_["leaves"]) if L["window"]]}
        p = OUT / f"{sid}.json"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_text(json.dumps(spec, indent=1))
        entry |= {"spec": str(p), "spec_sha256": sha256(p), "n_spans": len(spans), "n_points": len(points),
                  "n_named": len(named), "n_window_leaves": len(spec["window_leaves_not_read"])}
        plan["calls"][sid] = entry
        print(f"{sid}: (e) {'equals' if same else 'differs from'} the tree; {len(spans)} spans, {len(points)} points, "
              f"{len(named)} trusted Levin leaves, {len(spec['window_leaves_not_read'])} window leaves not read")
    pp = OUT / "plan.json"
    if pp.exists():
        raise SystemExit(f"{pp} exists")
    pp.write_text(json.dumps(plan, indent=1))
    print("wrote", pp, sha256(pp))


if __name__ == "__main__":
    main()
