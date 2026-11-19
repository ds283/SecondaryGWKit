"""levin-phase-floor prompt 07c Part A (section 2.1 items 1 and 4): the selection x1_capture_07c.py captures, from the census
records of state/census_refs_07c.pkl (07b's census). New in 07c.

Usage: python -B capture_selection_07c.py SET OUT_JSON [SIDS_JSON]
  SET is "chase" (every chase record) or "r5" (the sids listed in SIDS_JSON's "sids", each an m-... record).
Each entry carries the unit, index, notify_label and 07b's census record (``census_*``: value and abserr by hex, num_regions,
max_depth, converged) and the unit's census budget (``census_unit_budget_s``), which x1_capture_07c.py gives the worker.
"""
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402


def main():
    which, out = sys.argv[1], Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists")
    pk = WIDER / "state" / "census_refs_07c.pkl"
    calls = pickle.loads(pk.read_bytes())["calls"]
    if which == "chase":
        sids = sorted(s for s, v in calls.items() if v["set"] == "chase")
    elif which == "r5":
        sids = json.loads(Path(sys.argv[3]).read_text())["sids"]
    else:
        raise SystemExit("SET is chase or r5")
    sel = []
    for s in sids:
        v = calls[s]
        c = v["call"]
        sel.append({"sid": s, "set": v["set"], "unit": v["unit"], "index": v["index"], "kind": v["kind"],
                    "notify_label": c["notify_label"], "rtol": v["unit"].split("-")[1] + "-" + v["unit"].split("-")[2],
                    "census_value": c["value"]["hex"], "census_abserr": c["abserr"]["hex"],
                    "census_num_regions": c["num_regions"], "census_max_depth": c["max_depth"],
                    "census_converged": c["converged"], "census_unit_budget_s": v["budget_s"],
                    "census_num_edge_rule_refusals": c.get("num_edge_rule_refusals"),
                    "census_num_range_rule_refusals": c.get("num_range_rule_refusals")})
    if any(e["kind"] != "group" for e in sel):
        raise SystemExit("x1 capture takes group calls only; an analytic call needs another route")
    out.write_text(json.dumps(sel, indent=1))
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07c Part A: capture selection {which}", [pk])
    (out.parent / (out.stem + "_header.txt")).write_text("\n".join(hdr) + "\n")
    for l in hdr:
        print(l)
    print(f"{len(sel)} calls on {len({e['unit'] for e in sel})} units; wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
