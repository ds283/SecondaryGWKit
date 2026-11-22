"""levin-phase-floor prompt 07d Part A, section 2.1 item 1's top-up: every analytic call of each unit a top-up round drew
(state/selection_calls_07d_rN.json), captured with it. New in 07d (log deviation 8).

A capture re-runs its whole unit whichever of its calls it lists, so listing all of a unit's analytic calls (16 blocks, J1
.. Y4 of the unit's two subintervals) costs no more evaluation than listing the one drawn. The analytic strata's leaves are
rare per call (four gate leaves above 3/4 of 6 pi in 80 analytic calls at each rtol in the first sample), so the round's
drawn calls are kept as drawn, and the other analytic calls of their units ride along: a sample of units by the rule,
with all their analytic calls. The ride-along calls are marked ``ridealong`` in the selection.
Writes state/selection_calls_07d_rNx.json and state/capture_selection_07d_rNx.json.
Usage: PYTHONPATH=<tree>:<scripts> python -B ridealong_07d.py N
"""

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402


def main():
    n = int(sys.argv[1])
    src = WIDER / "state" / f"selection_calls_07d_r{n}.json"
    sel_p = WIDER / "state" / f"selection_calls_07d_r{n}x.json"
    cap_p = WIDER / "state" / f"capture_selection_07d_r{n}x.json"
    for p in (sel_p, cap_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    drawn = json.loads(src.read_text())["selection"]
    idx_p = WIDER / "state" / "census_index_07d.pkl"
    rows = pickle.loads(idx_p.read_bytes())["rows"]
    taken = set()
    for prev in sorted((WIDER / "state").glob("selection_calls_07d*.json")):
        if prev.name in (src.name, sel_p.name):
            continue
        taken |= {e["sid"] for e in json.loads(prev.read_text())["selection"]}
    units = {e["unit"] for e in drawn}
    dsids = {e["sid"] for e in drawn}
    sel = [dict(e, ridealong=False) for e in drawn]
    for r in rows:
        if r["unit"] in units and r["kind"] == "analytic" and r["sid"] not in dsids and r["sid"] not in taken:
            sel.append(dict(r, cells=[], ridealong=True))
    sel.sort(key=lambda r: r["sid"])
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07d Part A: top-up round {n}, with its units' analytic calls", [src, idx_p])
    sel_p.write_text(json.dumps({"header": hdr, "round": n, "from": str(src), "selection": sel}, indent=1))
    cap = [{"unit": r["unit"], "index": r["index"], "notify_label": r["notify_label"], "census_value": r["value"],
            "census_abserr": r["abserr"], "census_num_regions": r["num_regions"], "census_max_depth": r["max_depth"],
            "census_converged": r["converged"], "census_unit_budget_s": r["budget_s"], "sid": r["sid"]} for r in sel]
    cap_p.write_text(json.dumps(cap, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(drawn)} drawn, {sum(r['ridealong'] for r in sel)} ride along: {len(sel)} calls in {len(units)} units")
    print(f"wrote {sel_p} {sha256(sel_p)}; {cap_p} {sha256(cap_p)}")


if __name__ == "__main__":
    main()
