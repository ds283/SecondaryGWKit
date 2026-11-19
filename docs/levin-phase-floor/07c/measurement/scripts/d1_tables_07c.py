"""levin-phase-floor prompt 07c Part A, section 2.6 table 1 (D1), from out/d1/d1_descent.json and out/d1/d1_cc_leaves.json.
New in 07c. Writes out/d1/d1_tables.md and .json:
1. the descent analysis against the census agent's (07b census summary.json ``newly_unconverged``), per item;
2. the bisections below refused leaves by route and cause (the round-off exit: residual above the floor so not considered;
   considered and refused by the halving test; by the budget clause; chosen and found floor-limited, not bisected), by why
   chosen, by the trust clause that failed, by depth; with delta1/nu_B, delta2/nu_B and delta1/request quantiles;
3. the chase's Clenshaw-Curtis leaves' true errors (d2_table_07c.py d1's summary, copied);
4. the calls with no depth-20 leaf, leaf by leaf: every returned leaf's e', floor, residual, trusted, phase_limited, against
   the call's request.
Usage: python -B d1_tables_07c.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)])
    return "-" if not len(a) else " / ".join(f"{np.quantile(a, p):.3g}" for p in (0, 0.5, 0.9, 0.99, 1))


def main():
    d = json.loads((ROOT / "out" / "d1" / "d1_descent.json").read_text())
    cc = json.loads((ROOT / "out" / "d1" / "d1_cc_leaves.json").read_text())
    C = d["calls"]
    chase = [s for s in C if s.startswith("c-")]
    md = ["# D1: the chase, measured (table 1)", "", "## 1. The descent analysis against the census agent's (the 198 chase calls)", ""]
    agree = Counter()
    for s in chase:
        for k, v in C[s]["agrees_with_census_agent"].items():
            agree[k] += int(v)
    tot = {"calls": len(chase), "all_depth20_below_refused": sum(C[s]["all_depth20_below_refused"] for s in chase),
           "no_depth20": sum(C[s]["n_depth20"] == 0 for s in chase), "depth20_leaves": sum(C[s]["n_depth20"] for s in chase),
           "depth20_routes": dict(sum((Counter(C[s]["routes"]) for s in chase), Counter())),
           "shallowest_refused_depth_of_depth20_leaves": dict(sum((Counter(C[s]["shallowest_refused_depths"]) for s in chase), Counter())),
           "nearest_refused_depth_of_depth20_leaves": dict(sum((Counter(C[s]["nearest_refused_depths"]) for s in chase), Counter())),
           "calls_by_min_refused_depth": dict(Counter(str(C[s]["min_refused_ancestor_depth_of_depth20"]) for s in chase))}
    md += [f"- calls {tot['calls']}; every depth-20 leaf below a refused leaf: **{tot['all_depth20_below_refused']}** (census agent: 192); "
           f"no depth-20 leaf: {tot['no_depth20']} (census agent: 6); depth-20 leaves {tot['depth20_leaves']} (census agent: 668), by route "
           f"{tot['depth20_routes']}",
           f"- per call, agreeing with the census agent's record: " + ", ".join(f"{k} {v}/198" for k, v in agree.items()),
           f"- refused-ancestor depth of the depth-20 leaves, shallowest: {tot['shallowest_refused_depth_of_depth20_leaves']}; nearest: "
           f"{tot['nearest_refused_depth_of_depth20_leaves']}; calls by their shallowest: {tot['calls_by_min_refused_depth']}",
           "- item 2: " + "; ".join(f"{s} depth-20 {C[s]['n_depth20']} {C[s]['routes']}, all below a refused leaf {C[s]['all_depth20_below_refused']}, "
                                    f"refused leaves {C[s]['n_refused']}, regions {C[s]['num_regions']}"
                                    for s in ("a:R2-28", "a:R2-28~o", "r31-3-20363-1")), ""]
    below = d["below"]
    md += ["## 2. Every leaf chosen below a refused leaf (the 198 chase calls, R2-28 at both rtols, R31's record)", "",
           "Cause: `residual_above_floor` (the round-off exit not considered), `halving_test` (considered; the halving test refused),",
           "`budget_clause`, `floor_limited` (chosen and found floor-limited: not bisected). The refused leaves' own bisections are",
           f"apart: {len(d['own'])}.", "",
           "| route | cause | n | why untrusted / budget | clause r>=1 / noise (r<1, delta1>nu_B) / rate rule | delta1/nu_B min/med/90%/99%/max | delta2/nu_B | delta1/request |",
           "|---|---|---|---|---|---|---|---|"]
    for (rt, ca), n in sorted(Counter((x["route"], x["cause"]) for x in below).items()):
        xs = [x for x in below if x["route"] == rt and x["cause"] == ca]
        why = Counter(x["why"] for x in xs)
        cl = Counter(x.get("clause_failed") for x in xs)
        md.append(f"| {rt} | {ca} | {n} | {why.get('untrusted', 0)} / {why.get('budget', 0)} | {cl.get('r>=1', 0)} / {cl.get('noise', 0)} / "
                  f"{sum(v for k, v in cl.items() if k and k.startswith('rate'))} | {q([x.get('delta1_over_nuB') for x in xs])} | "
                  f"{q([x['delta2'] / x['nu_B'] for x in xs if x.get('nu_B')])} | {q([x.get('delta1_over_request') for x in xs])} |")
    bis = [x for x in below if x["outcome"] == "split"]
    md += ["", f"Bisections (not floor-limited): {len(bis)}, by route {dict(Counter(x['route'] for x in bis))}; by depth (CC) "
           f"{dict(sorted(Counter(x['depth'] for x in bis if x['route'] == 'cc_gate').items()))}; (Levin) "
           f"{dict(sorted(Counter(x['depth'] for x in bis if x['route'] == 'levin').items()))}", ""]
    md += ["## 3. The chase's Clenshaw-Curtis leaves: CC_47's true error against |CC_47 - CC_24| and the floor", "",
           "From leafcc_07c.py (reference: the better of CC_185/GL and a 600-point Chebyshev least-squares fit; out/d1/d1_cc_leaves.md).", ""]
    md += (ROOT / "out" / "d1" / "d1_cc_leaves.md").read_text().splitlines()
    md += ["", "## 4. The calls with no depth-20 leaf, leaf by leaf", ""]
    for s in chase:
        nd = C[s]["no_depth20"]
        if not nd:
            continue
        ls = sorted(nd["leaves"], key=lambda l: -l["eprime"])
        tot_e = sum(l["eprime"] for l in ls)
        rp = json.loads((ROOT / "out" / "replays" / "tree" / f"{s}.json").read_text())
        req = max(float.fromhex(rp["atol"]), float.fromhex(rp["rtol"]) * abs(float.fromhex(rp["replay"]["value"])))
        md.append(f"- **{s}**: abserr {nd['abserr']:.4g} = {nd['abserr'] / req:.4g} of the request {req:.4g}, converged {nd['converged']}, {len(ls)} leaves, sum e' {tot_e:.4g}; refused "
                  f"leaves {C[s]['n_refused']} at depths {C[s]['refused_depths']}. Largest e' first:")
        for l in ls[:8]:
            md.append(f"  - depth {l['depth']} {l['route']}: e' {l['eprime']:.3g} ({l['eprime'] / tot_e:.1%}), floor {l['floor']:.3g}, residual "
                      f"{l['residual']:.3g}, trusted {l['trusted']}, phase_limited {l['phase_limited']}")
    (ROOT / "out" / "d1" / "d1_tables.md").write_text("\n".join(md) + "\n")
    (ROOT / "out" / "d1" / "d1_tables.json").write_text(json.dumps({"descent": tot, "agree": agree}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
