"""levin-phase-floor prompt 07d Part A: the leaf verdicts' transitions from reference v2 (out/leaves_v2_superseded/) to v3
(out/leaves/), per stratum and route, on every leaf measured under both (the same stratum and rank). New in 07d (the
orchestrator's condition on v3, log deviation 7).
Routes: ``cc`` (the tree's declaration max(pair, phase_err) on CC_{2N-1}) and ``levin`` (the Levin solve's own declaration
on V_P), each by state/refs_rule_leaf_07d.txt's verdict rule against the version's own reference. Every leaf that moves
between bounded and understates, in either direction, is named.
Writes out/tables/v2_v3_transitions.json and .md. Usage: PYTHONPATH=<tree>:<scripts> python -B v2_v3_transitions_07d.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

fx = float.fromhex


def verdict(D, v, ref, eps):
    te = abs(v - ref)
    return "bounded" if D >= te + eps else ("understates" if D < te - eps else "undecided")


def leaves(base):
    st = json.loads((base / "fill_state.json").read_text())
    out, cache = {}, {}
    for s, S in st["strata"].items():
        for r in S["results"]:
            if not r["file"]:
                continue
            if r["file"] not in cache:
                cache = {r["file"]: {l["id"]: l for l in json.loads((base / r["file"]).read_text())["leaves"]}}
            l = cache[r["file"]].get(f"{s}#{r['rank']}")
            if l and "error" not in l:
                out[(s, r["rank"])] = l
    return out


def verdicts(l):
    ref, eps = fx(l["ref"]), fx(l["eps_ref"])
    return {"cc": verdict(max(fx(l["pair"]), fx(l["floor"])), fx(l["cc47"]), ref, eps),
            "levin": verdict(fx(l["levin"]["own_decl"]), fx(l["levin"]["vP"]), ref, eps)}


def main():
    v2 = leaves(ROOT / "out" / "leaves_v2_superseded")
    v3 = leaves(ROOT / "out" / "leaves")
    both = sorted(set(v2) & set(v3))
    tab, named = {}, []
    for k in both:
        a, b = verdicts(v2[k]), verdicts(v3[k])
        for route in ("cc", "levin"):
            tab.setdefault(f"{k[0]}|{route}", Counter())[f"{a[route]}->{b[route]}"] += 1
            if {a[route], b[route]} == {"bounded", "understates"}:
                named.append({"stratum": k[0], "rank": k[1], "route": route, "v2": a[route], "v3": b[route],
                              "start": v3[k]["start"], "end": v3[k]["end"], "eps_v2": fx(v2[k]["eps_ref"]),
                              "eps_v3": fx(v3[k]["eps_ref"]), "ref_v2": v2[k]["ref"], "ref_v3": v3[k]["ref"]})
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: leaf verdicts, reference v2 to v3",
                 [ROOT / "out" / "leaves_v2_superseded" / "fill_state.json", ROOT / "out" / "leaves" / "fill_state.json"])
    od = ROOT / "out" / "tables"
    od.mkdir(parents=True, exist_ok=True)
    tot = Counter()
    for c in tab.values():
        tot.update(c)
    (od / "v2_v3_transitions.json").write_text(json.dumps({"header": hdr, "n_both": len(both), "only_v2": len(set(v2) - set(v3)),
                                                           "only_v3": len(set(v3) - set(v2)), "total": dict(tot),
                                                           "by_stratum_route": {k: dict(v) for k, v in sorted(tab.items())},
                                                           "bounded_understates_moves": named}, indent=1))
    keys = ["bounded->bounded", "understates->understates", "undecided->undecided", "undecided->bounded", "undecided->understates",
            "bounded->undecided", "understates->undecided", "bounded->understates", "understates->bounded"]
    L = [l.replace("# ", "", 1) for l in hdr] + ["", f"Leaves measured under both: {len(both)} (only v2: {len(set(v2) - set(v3))}, "
                                                 f"only v3: {len(set(v3) - set(v2))}).", "",
                                                 "| stratum|route | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for k, c in sorted(tab.items()):
        L.append(f"| {k} | " + " | ".join(str(c.get(x, 0)) for x in keys) + " |")
    L.append(f"| **all** | " + " | ".join(str(tot.get(x, 0)) for x in keys) + " |")
    L += ["", f"## Leaves moving between bounded and understates ({len(named)})", "",
          "| stratum | rank | route | v2 | v3 | start | end | eps v2 | eps v3 |", "|---|---|---|---|---|---|---|---|---|"]
    for n in named:
        L.append(f"| {n['stratum']} | {n['rank']} | {n['route']} | {n['v2']} | {n['v3']} | {n['start']} | {n['end']} | "
                 f"{n['eps_v2']:.3g} | {n['eps_v3']:.3g} |")
    (od / "v2_v3_transitions.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-(len(named) + 8):]))
    print("totals:", dict(tot))
    print("wrote", od / "v2_v3_transitions.json", sha256(od / "v2_v3_transitions.json"))


if __name__ == "__main__":
    main()
