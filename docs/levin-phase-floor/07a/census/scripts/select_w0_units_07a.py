"""levin-phase-floor prompt 07a section 6, Gate W0's third and fourth clauses: the units, chosen by a rule fixed before
any is run (this docstring; new in 07a). Read only, except the two units files it writes.

- **The unit with a window** (clauses 3 and 4): z12's row at 1e-8, ``g-1e-8-1-2174`` (07's own V0 (c) unit, whose
  ``G+q+r`` call #1 locates one root and integrates its window at 07; run at h as ``h-1e-8-1-2174``). Fixed by name. That it
  has a windowed call at h is read from the clause runs' own records (compare_07a.py prints each unit's windowed calls).
- **The unit with a route change** (clause 3): from Gate W0's K0 run at h (K0_H_DIR/units.jsonl), in the order of 06k's
  K0 list (docs/levin-phase-floor/06l/out/k0/units.json), the first unit, other than the window unit, that has no
  windowed call at h and has a call one of whose ``own`` solves (its span a region the driver made) changed its C2 route
  (route_07a ``n_route_differs_own`` > 0). If none: the first such unit with an own solve whose LU choice changed
  (``n_lu_differs_own`` > 0); if none, the first with any solve whose route or LU choice changed. If none at all, the
  script exits 1 and writes nothing.

Writes state/w0c3_units.json (the two units) and state/w0c4_units.json (the window unit), in run_k0.py's form
({"units": [{"uid": "g-..."}]}, 07's census names), each with this rule's basis; prints the choice.

Usage: python -B select_w0_units_07a.py K0_H_DIR
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_TREE, WIDER, header, sha256  # noqa: E402

K0_UNITS = H_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"
WINDOW_UNIT = "g-1e-8-1-2174"


def main():
    k0_dir = Path(sys.argv[1])
    src = k0_dir / "units.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: Gate W0's clause 3 and 4 units", [K0_UNITS, src])
    for l in hdr:
        print(l)
    order = [u["uid"] for u in json.loads(K0_UNITS.read_text())["units"]]
    recs = {}
    for line in open(src):
        r = json.loads(line)
        recs["g-" + r["unit"].split("-", 1)[1]] = r["data"]
    tiers = {1: None, 2: None, 3: None}
    for uid in order:
        if uid == WINDOW_UNIT or uid not in recs:
            continue
        d = recs[uid]
        cs = (d.get("calls") or []) + (d.get("other_calls") or [])
        if any((c.get("guard_07") or {}).get("n_windows") for c in cs):
            continue
        rs = [c.get("route_07a") for c in cs if isinstance(c.get("route_07a"), dict) and "error" not in c["route_07a"]]
        if tiers[1] is None and any(r_["n_route_differs_own"] for r_ in rs):
            tiers[1] = uid
        if tiers[2] is None and any(r_["n_lu_differs_own"] for r_ in rs):
            tiers[2] = uid
        if tiers[3] is None and any(r_["n_route_differs"] or r_["n_lu_differs"] for r_ in rs):
            tiers[3] = uid
    tier = next((t for t in (1, 2, 3) if tiers[t] is not None), None)
    print(f"candidates by tier (1: an own solve's C2 route; 2: an own solve's LU choice; 3: any solve): {tiers}")
    if tier is None:
        print("no K0 unit at h without a window has a route or LU change: nothing written")
        return 1
    route_unit = tiers[tier]
    d = recs[route_unit]
    detail = [(c["index"], c["notify_label"], {k: c["route_07a"][k] for k in ("n_route_differs_own", "n_lu_differs_own",
                                                                              "n_route_differs_aux", "n_lu_differs_aux")})
              for c in (d.get("calls") or []) + (d.get("other_calls") or [])
              if isinstance(c.get("route_07a"), dict) and (c["route_07a"].get("n_route_differs") or c["route_07a"].get("n_lu_differs"))]
    basis = {"rule": __doc__.split("Writes ")[0].strip(), "header": hdr, "window_unit": WINDOW_UNIT,
             "route_unit": route_unit, "route_unit_tier": tier, "route_unit_calls_changed": detail}
    p3 = WIDER / "state" / "w0c3_units.json"
    p4 = WIDER / "state" / "w0c4_units.json"
    for p in (p3, p4):
        if p.exists():
            raise SystemExit(f"{p} exists; not overwritten")
    p3.write_text(json.dumps({"basis": basis, "units": [{"uid": WINDOW_UNIT}, {"uid": route_unit}]}, indent=1))
    p4.write_text(json.dumps({"basis": basis, "units": [{"uid": WINDOW_UNIT}]}, indent=1))
    print(f"window unit {WINDOW_UNIT}; route unit {route_unit} (tier {tier}); its changed calls {detail}")
    print(f"wrote {p3} sha256 {sha256(p3)}")
    print(f"wrote {p4} sha256 {sha256(p4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
