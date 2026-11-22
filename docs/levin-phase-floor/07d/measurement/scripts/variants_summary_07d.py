"""levin-phase-floor prompt 07d Part A, section 2.5: the variants' replays of the 4,220 section-2.1 calls, summarised against
the tree's replays and 07c's census record. New in 07d.

Per variant: calls written; calls the variant leaves untouched (no route change and no Clenshaw-Curtis solve re-declared,
by the variant's own counters ``num_07d_route_changed`` and ``num_07d_cc_new_estimate``) and how many of them reproduce their
census record bit for bit (all must); calls touched; route changes; Clenshaw-Curtis solves re-declared; calls whose value,
region count, depth or converged flag differ from the tree's replay; the abserr ratio to the tree's; subregion solves against
the tree's (the cost); wall time against the tree's.
Writes out/tables/variants_replay.json and .md. Usage: PYTHONPATH=<tree>:<scripts> python -B variants_summary_07d.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

fx = float.fromhex
VARIANTS = ["Eg", "Ea", "ET20", "ET12", "Ep", "T20"]


def main():
    rd = ROOT / "out" / "replays"
    tree = {}
    for p in (rd / "tree").glob("*.json"):
        d = json.loads(p.read_text())
        tree[d["sid"]] = d
    vdoc = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())
    out = {}
    for v in VARIANTS:
        o = {"written": 0, "untouched": 0, "untouched_reproduce": 0, "untouched_not_reproducing": [], "touched": 0,
             "route_changes": 0, "calls_route_changed": 0, "cc_redeclared": 0, "value_differs_tree": 0,
             "regions_differ_tree": 0, "depth_differs_tree": 0, "converged_differs_tree": 0, "abserr_ratio": [],
             "solves": 0, "solves_tree": 0, "wall": 0.0, "wall_tree": 0.0, "converged_false": 0, "converged_false_tree": 0}
        for p in (rd / v).glob("*.json"):
            d = json.loads(p.read_text())
            t = tree[d["sid"]]
            o["written"] += 1
            x = d["extra_07d"]
            touched = x["num_07d_route_changed"] > 0 or x["num_07d_cc_new_estimate"] > 0
            if touched:
                o["touched"] += 1
            else:
                o["untouched"] += 1
                if d["reproduces_census"]:
                    o["untouched_reproduce"] += 1
                else:
                    o["untouched_not_reproducing"].append(d["sid"])
            o["route_changes"] += x["num_07d_route_changed"]
            o["calls_route_changed"] += x["num_07d_route_changed"] > 0
            o["cc_redeclared"] += x["num_07d_cc_new_estimate"]
            r, tr = d["replay"], t["replay"]
            o["value_differs_tree"] += r["value"] != tr["value"]
            o["regions_differ_tree"] += r["num_regions"] != tr["num_regions"]
            o["depth_differs_tree"] += r["max_depth"] != tr["max_depth"]
            o["converged_differs_tree"] += r["converged"] != tr["converged"]
            o["converged_false"] += not r["converged"]
            o["converged_false_tree"] += not tr["converged"]
            if fx(tr["abserr"]) > 0:
                o["abserr_ratio"].append(fx(r["abserr"]) / fx(tr["abserr"]))
            o["solves"] += d["num_subregion_solves"]
            o["solves_tree"] += t["num_subregion_solves"]
            o["wall"] += d["wall_s"]
            o["wall_tree"] += t["wall_s"]
        a = np.array(o.pop("abserr_ratio"))
        o["abserr_ratio_tree"] = {"min": float(a.min()), "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
                                  "p90": float(np.percentile(a, 90)), "max": float(a.max())}
        out[v] = o
    srcs = [ROOT / "out" / "variants" / "variants.json"]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: the variants' replays of the 4,220 calls", srcs)
    od = ROOT / "out" / "tables"
    (od / "variants_replay.json").write_text(json.dumps({"header": hdr, "variants_json": vdoc, "table": out}, indent=1))
    L = [l.replace("# ", "", 1) for l in hdr] + [""]
    L += ["| variant | written | untouched | of which reproduce census | touched | calls route changed | route changes | CC solves re-declared | value differs from tree | regions differ | depth differs | converged differs | not converged (tree) | abserr / tree min, median, p90, max | solves / tree | wall / tree |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, o in out.items():
        a = o["abserr_ratio_tree"]
        L.append(f"| {v} | {o['written']} | {o['untouched']} | {o['untouched_reproduce']} | {o['touched']} | {o['calls_route_changed']} | "
                 f"{o['route_changes']} | {o['cc_redeclared']} | {o['value_differs_tree']} | {o['regions_differ_tree']} | "
                 f"{o['depth_differs_tree']} | {o['converged_differs_tree']} | {o['converged_false']} ({o['converged_false_tree']}) | "
                 f"{a['min']:.3g}, {a['median']:.3g}, {a['p90']:.3g}, {a['max']:.3g} | {o['solves'] / o['solves_tree']:.3f} | {o['wall'] / o['wall_tree']:.2f} |")
    (od / "variants_replay.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    for v, o in out.items():
        if o["untouched_not_reproducing"]:
            print(v, "UNTOUCHED NOT REPRODUCING", o["untouched_not_reproducing"][:10])
    print("wrote", od / "variants_replay.json", sha256(od / "variants_replay.json"))


if __name__ == "__main__":
    main()
