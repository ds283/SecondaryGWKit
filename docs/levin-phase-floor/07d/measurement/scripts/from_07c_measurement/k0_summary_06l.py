"""levin-phase-floor prompt 06l, Gate K0: the two comparisons' verdicts, and what route B did on K0's 100 units
(descriptive; nothing here is a prediction's score). New in 06l, after 06k's k0_summary.py (which compared each
route with a tree-module control; 06l has none, and compares with 06g's census records, through
state/index_06g.json, instead). Read only.

Reads out/k0/compare_identity.json and out/k0/compare_routeB.json (compare_k0.py), state/k0/routeB/units.jsonl
(run_k0.py) and 06g's census checkpoint lines for the same units. Reports:
- the gate: clause 1 (identity, every unit whole) and clause 2 (route B, every call with no firing);
- route B's records: firings, withdrawals (expected none) and the untrusted leaves recorded, calls with a firing;
- the firings by depth, by the parent's and the halves' kinds, by delta1/phi_P (against 5.8), delta1/nu_B and r;
- per group call, converged, regions and max depth against 06g's census record of the same call.
Writes out/k0/summary.json and prints it. Usage: PYTHONPATH=<tree>:<scripts> python -B k0_summary_06l.py
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

K1_LIMIT = 5.8


def band(x, edges):
    for e in edges:
        if x < e:
            return f"<{e:g}"
    return f">={edges[-1]:g}"


def kind(direct):
    return "CC" if direct else "Levin"


def main():
    files = [RC.ROOT / "out" / "k0" / "compare_identity.json", RC.ROOT / "out" / "k0" / "compare_routeB.json",
             RC.ROOT / "state" / "k0" / "routeB" / "units.jsonl", RC.ROOT / "state" / "index_06g.json"]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: Gate K0 summarised", files)
    for l in hdr:
        print(l)
    doc = {"header": hdr, "gate": {}}
    for v, f in (("identity", files[0]), ("routeB", files[1])):
        c = json.loads(f.read_text())
        doc["gate"][v] = {k: c[k] for k in ("mode", "n_units", "n_fail", "n_calls_compared_clause2", "n_calls_fired",
                                            "n_units_compared_whole")}
        doc["gate"][v]["excluded_keys_differing_count"] = c["excluded_keys_differing_count"]
    doc["gate"]["clause1_identity_holds"] = doc["gate"]["identity"]["n_fail"] == 0 and doc["gate"]["identity"]["n_units_compared_whole"] == 100
    doc["gate"]["clause2_routeB_holds"] = doc["gate"]["routeB"]["n_fail"] == 0
    idx = json.loads(files[3].read_text())["units"]
    ck = RC.census_run_06g() / "checkpoint.jsonl"
    recs = []
    calls_cmp = collections.Counter()
    conv_changes = []
    with open(ck, "rb") as fck:
        for line in open(files[2]):
            m = json.loads(line)
            uid, d = m["unit"], m["data"]
            ent = idx[uid]
            fck.seek(ent["offset"])
            gl = fck.read(ent["length"])
            if hashlib.sha256(gl).hexdigest() != ent["line_sha256"]:
                raise SystemExit(f"{uid}: 06g line sha256 differs")
            g = json.loads(gl)["data"]
            gcalls = {c["index"]: c for c in (g.get("calls") or []) + (g.get("other_calls") or [])}
            for c in (d.get("calls") or []) + (d.get("other_calls") or []):
                rb = c.get("route_06l")
                if rb is None:
                    raise SystemExit(f"{uid} #{c['index']}: no route_06l record")
                gc = gcalls[c["index"]]
                fired = rb["n_firings"] > 0
                calls_cmp["calls"] += 1
                calls_cmp["calls_with_firing"] += fired
                calls_cmp["withdrawn"] += rb["n_withdrawn"]
                if c.get("group") is not None:
                    calls_cmp["group_calls"] += 1
                    calls_cmp["group_regions_06g"] += gc["num_regions"]
                    calls_cmp["group_regions_B"] += c["num_regions"]
                    if bool(gc["converged"]) != bool(c["converged"]):
                        conv_changes.append({"unit": uid, "index": c["index"], "group": c.get("group"),
                                             "g": [gc["converged"], gc["num_regions"], gc["max_depth"], gc["abserr"]],
                                             "B": [c["converged"], c["num_regions"], c["max_depth"], c["abserr"]],
                                             "node_limited_B": c.get("node_limited"), "firings": rb["n_firings"]})
                    calls_cmp["group_at_depth_max_06g"] += (gc.get("max_depth") or 0) >= (gc.get("depth_max") or 20)
                    calls_cmp["group_at_depth_max_B"] += (c.get("max_depth") or 0) >= (c.get("depth_max") or 20)
                for r in rb["records"]:
                    r = dict(r, unit=uid, index=c["index"], group=c.get("group"))
                    recs.append(r)
    fires = [r for r in recs if r["kind"] == "fired"]
    doc["routeB"] = {
        "calls": dict(calls_cmp),
        "n_records": len(recs),
        "by_kind": dict(collections.Counter(r["kind"] for r in recs)),
        "firings_by_depth": dict(sorted(collections.Counter(r["depth"] for r in fires).items())),
        "firings_by_kinds": dict(collections.Counter(
            f"{kind(r['is_direct'])} parent / {kind(r['is_direct_L'])}+{kind(r['is_direct_R'])}" for r in fires)),
        "firings_by_d1_over_phiP": dict(collections.Counter(band(r["delta1"] / r["phi_P"], (1, 2, K1_LIMIT, 10, 100)) for r in fires)),
        "firings_by_d1_over_nuB": dict(collections.Counter(band(r["delta1"] / r["nuB"], (0.01, 0.1, 0.5, 1.0000000001)) for r in fires)),
        "firings_by_r": dict(collections.Counter(band(r["r"], (0.5, 0.75, 1.0)) for r in fires)),
        "levin_parent_firings": sum(1 for r in fires if not r["is_direct"]),
        "levin_parent_firings_at_d1_over_phiP_ge_5.8": sum(1 for r in fires if not r["is_direct"] and r["delta1"] / r["phi_P"] >= K1_LIMIT),
        "max_d1_over_phiP_levin_parent": max((r["delta1"] / r["phi_P"] for r in fires if not r["is_direct"]), default=None),
        "max_d1_over_phiP_cc_parent": max((r["delta1"] / r["phi_P"] for r in fires if r["is_direct"]), default=None),
        "untrusted_records_trusted_by_route1": sum(1 for r in recs if r["kind"] == "untrusted" and r["trusted_r1"]),
        "untrusted_records_trusted_by_route1_by_kinds": dict(collections.Counter(
            f"{kind(r['is_direct'])} parent / {kind(r['is_direct_L'])}+{kind(r['is_direct_R'])}"
            for r in recs if r["kind"] == "untrusted" and r["trusted_r1"])),
        "nuB_gt_nu1_records": sum(1 for r in recs if r["nuB"] > r["nu1"]),
        "nuB_gt_nu1_rel_max": max(((r["nuB"] - r["nu1"]) / r["nu1"] for r in recs if r["nuB"] > r["nu1"]), default=None),
        "fired_not_trusted_by_route1": sum(1 for r in fires if not r["trusted_r1"]),
        "convergence_changes_against_06g": conv_changes,
    }
    out = RC.ROOT / "out" / "k0" / "summary.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: v for k, v in doc.items() if k != "header"}, indent=1, default=str))
    print(f"wrote {out} sha256 {RC.sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
