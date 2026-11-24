"""levin-phase-floor 07d Part A' addendum, item A.1: the near-edge selection, fixed before any reference is read.

**Levin side:** every check of this Part's table 4 (out/tables/t4_dt.json ``per_leaf``) at T in {6, 8, 10, 12} pi, under the
CC_93 and the CC_47 fallback, that is trusted and undecided on v4 (``"<T>|levin|<fb>"`` with ``trusted`` and ``v ==
"undecided"``). Its value is the check's Q, its declaration D = total_err_trusted (the leaf record's item 1 entry for that
T and fallback; checked equal to table 4's D).
**CC side:** every table-4 CC-side verdict at those T, declarations pair93 and tail93 (``"<T>|cc|cc93:<decl>"``), that is
undecided. Its value is CC_93; D is table 4's (and table 3's) max(E, floor_47) + kappa s_93, kappa = 4; beside it the
V93-family variant's own declaration (out/repro/decl93_V93.json for pair93, decl93_V93tail.json for tail93).
**Sets:** "returned" (every stratum but band_20_32v) and "visited" (band_20_32v), kept apart.
Writes addendum/state/selection_add.json (the checks, and the distinct leaves with their record file and piece count).
Usage: python3 -B select_add.py
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2")
ADD = ROOT / "addendum"
TS = (6, 8, 10, 12)
KAPPA = 4.0
fx = float.fromhex


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = ADD / "state" / "selection_add.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    t4p = ROOT / "out" / "tables" / "t4_dt.json"
    t4 = json.loads(t4p.read_text())
    recs, files = {}, {}
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if d.is_dir() and (d.name.startswith("partA_") or d.name.startswith("nb")):
            for p in sorted(d.glob("*.json")):
                for L in json.loads(p.read_text())["leaves"]:
                    recs[L["id"]] = L
                    files[L["id"]] = str(p)
    own = {}
    for decl, v in (("pair93", "V93"), ("tail93", "V93tail")):
        for r in json.loads((ROOT / "out" / "repro" / f"decl93_{v}.json").read_text())["rows"]:
            own[(decl, r["id"])] = r["D_variant"]
    checks = []
    for L4 in t4["per_leaf"]:
        i = L4["id"]
        st = "visited" if i.startswith("band_20_32v") else "returned"
        L = recs[i]
        for k, v in L4["by_T"].items():
            Tp, route, fb = k.split("|")
            Tp = int(Tp)
            if Tp not in TS:
                continue
            if route == "levin":
                if not (isinstance(v, dict) and v.get("trusted") and v.get("v") == "undecided"):
                    continue
                c = next(c for c in L["item1"] if c["T_pi"] == Tp and c["fallback"] == fb and c.get("checked"))
                D = fx(c["total_err_trusted"])
                if D != v["D"]:
                    raise SystemExit(f"{i} {k}: D differs from table 4's")
                checks.append({"id": i, "set": st, "T_pi": Tp, "route": "levin", "what": f"Q, fallback {fb}",
                               "value": c["Q"], "D": D, "D_variant": None, "te_v4": v["te"], "eps_v4": v["eps"]})
            elif fb in ("cc93:pair93", "cc93:tail93") and v == "undecided":
                decl = fb.split(":")[1]
                floor = fx(L["cc"]["floor"])
                c93 = fx(L["cc"]["cc93"])
                s93 = L["cc"]["c93"]["s_own"]
                E = abs(fx(L["cc"]["cc185"]) - c93) if decl == "pair93" else fx(L["cc"]["c93"]["tail_sum4"])
                D = max(E, floor) + KAPPA * s93
                ref, eps = fx(L["v4"]["ref"]), fx(L["v4"]["eps_ref"])
                checks.append({"id": i, "set": st, "T_pi": Tp, "route": "cc", "what": f"CC_93, {decl}",
                               "value": L["cc"]["cc93"], "D": D, "D_variant": own.get((decl, i)),
                               "te_v4": abs(c93 - ref), "eps_v4": eps})
    leaves = {}
    for c in checks:
        L = recs[c["id"]]
        leaves[c["id"]] = {"file": files[c["id"]], "start": L["start"], "end": L["end"], "n_knots": L["n_knots"],
                           "n_pieces": 1 + (L["n_knots"] or 0), "V_over_pi": L["V_over_pi"],
                           "sigma_hat": L["v3_sigma"]["sigma_hat"], "kind": "group" if "|group|" in c["id"] else "analytic",
                           "set": c["set"]}
    counts = Counter(f"{c['set']}|{c['route']}|{c['what']}|T{c['T_pi']}" for c in checks)
    doc = {"rule": __doc__, "inputs": {str(t4p): sha(t4p)}, "counts": dict(sorted(counts.items())),
           "n_checks": len(checks), "n_leaves": len(leaves),
           "leaves_by_set_kind": dict(Counter(f"{v['set']}|{v['kind']}" for v in leaves.values())),
           "pieces_total": sum(v["n_pieces"] for v in leaves.values()), "checks": checks, "leaves": leaves}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("counts", "n_checks", "n_leaves", "leaves_by_set_kind", "pieces_total")}, indent=1))


if __name__ == "__main__":
    main()
