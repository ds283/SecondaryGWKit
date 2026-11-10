"""levin-phase-floor prompt 06k, Gate K0: the comparisons' verdicts, and what each route did on K0's 100
units (descriptive; nothing here is a prediction's score). New in 06k. Read only.

Reads out/k0/compare_<variant>.json and state/k0/<variant>/units.jsonl. Per route: calls with and without
a firing; firings by depth, by delta1/phi_P (against c07's 5.8) and by r; and, per group call, its
converged flag, regions and max depth against the tree control's on the same unit.
Writes out/k0/summary.json and prints it. Usage: python -B k0_summary.py
"""

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

VARIANTS = ("tree", "identity", "route1", "route2", "route3", "route2lit")  # route2lit after stop 1


def num(x):
    return float(x) if not isinstance(x, str) else float(x)


def load(v):
    out = {}
    for line in open(RC.ROOT / "state" / "k0" / v / "units.jsonl"):
        m = json.loads(line)
        out[m["unit"]] = m["data"]
    return out


def band(x, edges):
    for e in edges:
        if x < e:
            return f"<{e:g}"
    return f">={edges[-1]:g}"


def main():
    files = [RC.ROOT / "out" / "k0" / f"compare_{v}.json" for v in VARIANTS]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06k: Gate K0 summarised", files)
    for l in hdr:
        print(l)
    doc = {"header": hdr, "gate": {}, "routes": {}}
    for v, f in zip(VARIANTS, files):
        c = json.loads(f.read_text())
        doc["gate"][v] = {k: c[k] for k in ("mode", "n_units", "n_fail", "n_calls_compared_clause2", "n_calls_fired",
                                            "n_units_compared_whole")}
        doc["gate"][v]["excluded_non_provenance_keys_differing"] = {
            k: n for k, n in c["excluded_keys_differing_count"].items() if "cache_built" in k}
    tree = load("tree")
    for v in ("route1", "route2", "route3", "route2lit"):
        units = load(v)
        fires = []
        conv = collections.Counter()
        regions = {"group_tree": 0, "group_route": 0}
        for uid, d in units.items():
            t = {c["index"]: c for c in tree[uid]["calls"]}
            for c in d["calls"] + d["other_calls"]:
                rr = c["route_06k"]
                for fi in rr["firings"]:
                    fires.append((fi, "group" if c in d["calls"] else "other"))
            for c in d["calls"]:
                tc = t[c["index"]]
                conv[(tc["converged"], c["converged"])] += 1
                regions["group_tree"] += tc["num_regions"]
                regions["group_route"] += c["num_regions"]
        def ratio(fi):
            phi = num(fi["phi_P"])
            return num(fi["delta1"]) / phi if phi > 0 else math.inf
        doc["routes"][v] = {
            "n_firings": len(fires),
            "n_firings_group_calls": sum(1 for _, k in fires if k == "group"),
            "by_depth": dict(sorted(collections.Counter(fi["depth"] for fi, _ in fires).items())),
            "at_depth_max": sum(1 for fi, _ in fires if fi["at_depth_max"]),
            "by_d1_over_phiP": dict(collections.Counter(band(ratio(fi), (1, 2, 5.8, 25, 1e3, 1e7)) for fi, _ in fires)),
            "n_d1_over_phiP_ge_5.8": sum(1 for fi, _ in fires if ratio(fi) >= 5.8),
            "max_d1_over_phiP": max((ratio(fi) for fi, _ in fires), default=None),
            "by_r": dict(collections.Counter(band(num(fi["r"]), (0.25, 0.5, 0.75, 1.0)) for fi, _ in fires)),
            "group_calls_converged_tree_to_route": {f"{a}->{b}": n for (a, b), n in sorted(conv.items())},
            "group_call_regions": regions,
        }
        if v == "route2lit":  # after stop 1: the literal reading's firings by kind, and by the parent's branch
            doc["routes"][v]["by_kind"] = dict(collections.Counter(fi["kind"] for fi, _ in fires))
            doc["routes"][v]["by_kind_and_parent"] = dict(collections.Counter(
                f"{fi['kind']}|{'CC parent' if fi['is_direct'] else 'Levin parent'}" for fi, _ in fires))
            doc["routes"][v]["withdrawn_r_lt_quarter"] = sum(
                1 for fi, _ in fires if fi["kind"] == "trust_withdrawn" and num(fi["r"]) < 0.25)
            doc["routes"][v]["trust_added_d1_over_phiP_ge_5.8"] = sum(
                1 for fi, _ in fires if fi["kind"] == "trust_added" and ratio(fi) >= 5.8)
            doc["routes"][v]["trust_added_max_d1_over_phiP"] = max(
                (ratio(fi) for fi, _ in fires if fi["kind"] == "trust_added"), default=None)
        if v == "route3":
            doc["routes"][v]["by_D_over_cap"] = dict(collections.Counter(
                band(num(fi["D"]) / num(fi["cap"]), (1e-6, 1e-3, 1e-1, 1.0001)) for fi, _ in fires))
    out = RC.ROOT / "out" / "k0" / "summary.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: v for k, v in doc.items() if k != "header"}, indent=1, default=str))
    print(f"wrote {out} sha256 {RC.sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
