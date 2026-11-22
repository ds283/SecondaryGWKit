"""levin-phase-floor prompt 07c section 6, Gate Y0's third and fourth clauses: the units, chosen by a rule fixed before
any is run (this docstring; new in 07c, after 07b's select_u0_units_07b.py, a record, not edited, whose form it keeps).
Read only, except the two units files it writes.

- **The unit where the fix acts** (clauses 3 and 4): row 0-33929 at 1e-8, ``g-1e-8-0-33929`` (run at h as
  ``h-1e-8-0-33929``), fixed by name. Its group calls #2 and #3 are Part A's chase calls ``c-1e-8-0-33929-2`` and
  ``c-1e-8-0-33929-3``: two of 07b's 198 newly unconverged calls, both made different from 07b's census by (S)'s replays
  (07c Part A, out/replays/S_summary_vplain.json: ``reproduces_census`` false), so the stop is expected to act in them.
  Chosen as the cheapest unit at 07b's census (6.3 s) holding two chase calls (S)'s replays touched. That the fix acts in
  it at h (some call's num_chase_stops > 0) is read from the clause runs' own records (compare_07c.py prints each unit's
  stops); if it does not act, the clause is not met and the agent stops.
- **The unit where the fix does not act** (clause 3): from Gate Y0's K0 run at h (K0_H_DIR/units.jsonl), among the units
  recorded ``done`` with at least one Levin call (group or analytic), those every one of whose Levin calls has
  num_chase_stops 0; of these the one with the most Levin calls in which a bisection was counted against a root
  (num_chase_noise_bisections > 0: the stop's count ran, and the instrument's pass-through with it, but no root reached
  the limit), then the most Levin calls, then the most group calls, then the first in the order of 06k's K0 list
  (docs/levin-phase-floor/06l/out/k0/units.json). If none, the script exits 1 and writes nothing (the agent stops and
  asks).

Writes state/y0c3_units.json (the two units) and state/y0c4_units.json (the unit where the fix acts), in run_k0.py's form
({"units": [{"uid": "g-..."}]}, the K0 list's names), each with this rule's basis; prints the choice.

Usage: python -B select_y0_units_07c.py K0_H_DIR
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_TREE, WIDER, header, sha256  # noqa: E402

K0_UNITS = H_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"
ACTS_UNIT = "g-1e-8-0-33929"


def main():
    k0_dir = Path(sys.argv[1])
    src = k0_dir / "units.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: Gate Y0's clause 3 and 4 units", [K0_UNITS, src])
    for l in hdr:
        print(l)
    order = [u["uid"] for u in json.loads(K0_UNITS.read_text())["units"]]
    recs = {}
    for line in open(src):
        r = json.loads(line)
        recs["g-" + r["unit"].split("-", 1)[1]] = r["data"]
    cands = []
    n_acted_units = 0
    for pos, uid in enumerate(order):
        if uid == ACTS_UNIT or uid not in recs or recs[uid]["status"] != "done":
            continue
        d = recs[uid]
        cs = (d.get("calls") or []) + (d.get("other_calls") or [])
        if not cs:
            continue
        counts = [(c.get("num_chase_stops"), c.get("num_chase_noise_bisections")) for c in cs]
        if any(a is None or b is None for a, b in counts):
            raise SystemExit(f"{uid}: a call has no counts")
        if all(a == 0 for a, _b in counts):
            n_counted = sum(1 for _a, b in counts if b > 0)
            cands.append((-n_counted, -len(cs), -len(d.get("calls") or []), pos, uid))
        else:
            n_acted_units += 1
    cands.sort()
    print(f"K0 units at h with a Levin call: {len(cands) + n_acted_units}; the fix acts in {n_acted_units}; "
          f"untouched in every call: {len(cands)}; with a counted bisection: {sum(1 for c in cands if c[0] < 0)}")
    if not cands:
        print("no K0 unit at h is untouched in every Levin call: nothing written")
        return 1
    quiet_unit = cands[0][4]
    d = recs[quiet_unit]
    basis = {"rule": __doc__.split("Writes ")[0].strip(), "header": hdr, "acts_unit": ACTS_UNIT, "quiet_unit": quiet_unit,
             "quiet_unit_calls": [(c["index"], c["notify_label"], c.get("num_edge_rule_refusals"),
                                   c.get("num_range_rule_refusals"), c.get("num_chase_stops"),
                                   c.get("num_chase_stopped_leaves"), c.get("num_chase_noise_bisections"),
                                   c.get("num_regions"))
                                  for c in (d.get("calls") or []) + (d.get("other_calls") or [])],
             "candidates": [c[4] for c in cands],
             "acted_units": sorted(u for u in recs if any((c.get("num_chase_stops") or 0) > 0 for c in
                                                         (recs[u].get("calls") or []) + (recs[u].get("other_calls") or [])))}
    p3 = WIDER / "state" / "y0c3_units.json"
    p4 = WIDER / "state" / "y0c4_units.json"
    for p in (p3, p4):
        if p.exists():
            raise SystemExit(f"{p} exists; not overwritten")
    p3.write_text(json.dumps({"basis": basis, "units": [{"uid": ACTS_UNIT}, {"uid": quiet_unit}]}, indent=1))
    p4.write_text(json.dumps({"basis": basis, "units": [{"uid": ACTS_UNIT}]}, indent=1))
    print(f"acts unit {ACTS_UNIT}; quiet unit {quiet_unit} ({len(basis['quiet_unit_calls'])} Levin calls, all stops 0, "
          f"{-cands[0][0]} with a counted bisection)")
    print(f"K0 units at h where the fix acts: {basis['acted_units']}")
    print(f"wrote {p3} sha256 {sha256(p3)}")
    print(f"wrote {p4} sha256 {sha256(p4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
