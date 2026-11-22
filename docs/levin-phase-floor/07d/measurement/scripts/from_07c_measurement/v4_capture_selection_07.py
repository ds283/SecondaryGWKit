"""levin-phase-floor prompt 07 section 6, V4: the capture selection (x1_capture.py's input) for V4's 24 calls
(state/v4_selection.json, fixed in state/predictions.md), plus z12's call as a calibration capture for the references'
cost projection only (it is not one of the 24 and is not referenced): each entry with its census g record's value,
abserr (hex), num_regions, max_depth, converged and unit budget, read from 07's census checkpoint (SHA-256 checked),
and its 06m record's value and abserr. New in 07. Read only.
Usage: python -B v4_capture_selection_07.py OUT_JSON
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

SEL = HERE.parent / "state" / "v4_selection.json"
SEL_SHA256 = "15a4391b8e49c547a0557bc20829faf5c2323f0e826420ef0e21d694b134d6c8"
CAL = [{"sid": "cal_z12", "unit": "g-1e-8-1-2174", "index": 1, "kind": "group", "why": "calibration only (z12)"}]


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    ck = RC.CENSUS07 / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V4's capture selection", [SEL, ck])
    if RC.sha256(SEL) != SEL_SHA256 or RC.sha256(ck) != RC.CENSUS07_CHECKPOINT_SHA256:
        raise SystemExit("the selection or the census checkpoint has another SHA-256")
    sel = json.loads(SEL.read_text())["calls"] + CAL
    want = {e["unit"] for e in sel}
    recs = {}
    with open(ck) as f:
        for line in f:
            if not any(f'"{u}"' in line[:200] for u in want):
                continue
            r = json.loads(line)
            if r["unit"] in want:
                if r["unit"] in recs:
                    raise SystemExit(f"{r['unit']} twice")
                recs[r["unit"]] = r["data"]
    out = []
    for e in sel:
        d = recs[e["unit"]]
        if d["status"] != "done":
            raise SystemExit(f"{e['unit']}: {d['status']}")
        key = "calls" if e["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == e["index"]]
        if len(c) != 1:
            raise SystemExit(f"{e['unit']} #{e['index']}: not one call")
        c = c[0]
        if "label" in e and c["notify_label"] != e["label"]:
            raise SystemExit(f"{e['unit']} #{e['index']}: label differs from the selection's")
        out.append({"sid": e["sid"], "unit": e["unit"], "g_unit": e["unit"], "index": e["index"], "kind": e["kind"],
                    "notify_label": c["notify_label"], "census_value": c["value"]["hex"],
                    "census_abserr": c["abserr"]["hex"], "census_num_regions": c["num_regions"],
                    "census_max_depth": c["max_depth"], "census_converged": c["converged"],
                    "census_unit_budget_s": d["budget_s"], "value_06m": e.get("value_06m"),
                    "abserr_06m": e.get("abserr_06m"), "why": e.get("why"),
                    "guard_07_points": c["guard_07"].get("n_points_by_kind"),
                    "guard_07_skipped": c["guard_07"].get("location_skipped_phases")})
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    for l in hdr:
        print(l)
    for e in out:
        print(f"{e['sid']} {e['unit']} #{e['index']}: g value {e['census_value']} abserr {e['census_abserr']} regions "
              f"{e['census_num_regions']} depth {e['census_max_depth']} conv {e['census_converged']} points {e['guard_07_points']}")
    print(f"wrote {out_p} sha256 {RC.sha256(out_p)}")


if __name__ == "__main__":
    main()
