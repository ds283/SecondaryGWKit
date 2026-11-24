"""levin-phase-floor prompt 07d Part A' (the orchestrator's go-ahead, item 2: "V93's floor and node term on the 93-point
grid"): the declaration a V93-family variant itself makes on a Clenshaw-Curtis leaf, scored on v4. New in Part A'.

Table 3 (tables_a2.py decls93) scored each CC_93 candidate as max(E, floor_47) + kappa s_93, with floor_47 the tree's
CC_47 solve's phase_err (round-off floor + declared endpoint term on the 47-point grid). The variant declares
max(abserr_direct, phase_err) with abserr_direct = E + noise and phase_err = (round-off floor on the 93 grid + noise) +
the declared endpoint term on the 93 grid. This script takes the variant's own declaration, on every settled leaf of the
leaf records, from the variant's ``_adaptive_levin_subregion(..., force_direct=True)``, and scores it on v4 with table 3's
verdict (bounded: D >= te + eps; understates: D < te - eps; else undecided), te = |CC_93 - ref|.
Also: its value against the record's cc93 (bit for bit), and its declaration over table 3's for the same candidate.
Usage: PYTHONPATH=<tree>:<scripts> python -B decl93_a2.py VARIANT OUT [I N]   (leaf records i mod N == I)
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, header  # noqa: E402

fx = float.fromhex


def verdict(D, te, eps):
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def main():
    variant, out = sys.argv[1], Path(sys.argv[2])
    part, nparts = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    if out.exists():
        raise SystemExit(f"{out} exists")
    import variant_loader as VL

    vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
    LQ = VL.install(Path(vj["path"]), vj["sha256"], f"a2_{variant}")
    est = vj["est"]
    kappa = json.loads((ROOT / "out" / "tables" / "t3_cc93.json").read_text())["kappa93"]
    import checkrepro_a2 as CR
    import leafref_v4 as R4

    files = []
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if d.is_dir() and (d.name.startswith("partA_") or d.name.startswith("nb")):
            files += sorted(d.glob("*.json"))
    rows = []
    for j, p in enumerate(files):
        if j % nparts != part:
            continue
        doc = json.loads(p.read_text())
        leaves = [L for L in doc["leaves"] if "error" not in L and L["v4"]["settled"]]
        if not leaves:
            continue
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        rec = R4.load_capture(spec["capture"])
        kw = dict(rec["kw"])
        kw["rtol"] = float(spec["rtol"])
        C = CR.CallContext(LQ, rec, kw)
        for L in leaves:
            a, b = fx(L["start"]), fx(L["end"])
            d = LQ._adaptive_levin_subregion((a, b), rec["f"], C.A, id_label=C.id_label, chebyshev_order=C.N,
                                             rtol=C.rtol, vectorize_cache=C.cache, force_direct=True)
            D = max(float(d["abserr_direct"]), float(d.get("phase_err") or 0.0))
            ref, eps = fx(L["v4"]["ref"]), fx(L["v4"]["eps_ref"])
            c93 = fx(L["cc"]["cc93"])
            te = abs(c93 - ref)
            floor47 = fx(L["cc"]["floor"])
            s93 = L["cc"]["c93"]["s_own"]
            E = {"pair93": abs(fx(L["cc"]["cc185"]) - c93), "tail93": fx(L["cc"]["c93"]["tail_sum4"]),
                 "nest93": abs(c93 - fx(L["cc"]["cc47"])), "chop93": fx(L["cc"]["c93"]["chop_tail"])}.get(est)
            D3 = None if E is None else max(E, floor47) + kappa * s93
            rows.append({"id": L["id"], "value_equal_cc93": float(d["value"]).hex() == L["cc"]["cc93"], "D_variant": D,
                         "abserr_direct": float(d["abserr_direct"]), "phase_err_93": float(d.get("phase_err") or 0.0),
                         "floor_47": floor47, "D_table3": D3, "te93": te, "eps": eps, "verdict": verdict(D, te, eps),
                         "verdict_table3": None if D3 is None else verdict(D3, te, eps)})
    summ = {"n": len(rows), "value_equal": sum(r["value_equal_cc93"] for r in rows),
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "verdicts_table3": dict(Counter(r["verdict_table3"] for r in rows)),
            "verdict_changes": dict(Counter(f"{r['verdict_table3']}->{r['verdict']}" for r in rows
                                            if r["verdict"] != r["verdict_table3"])),
            "understates_ids": [r["id"] for r in rows if r["verdict"] == "understates"],
            "D_over_table3": (lambda xs: {"min": min(xs), "median": sorted(xs)[len(xs) // 2], "max": max(xs)} if xs else None)(
                [r["D_variant"] / r["D_table3"] for r in rows if r["D_table3"]]),
            "floor93_over_floor47": (lambda xs: {"min": min(xs), "median": sorted(xs)[len(xs) // 2], "max": max(xs)} if xs
                                     else None)([r["phase_err_93"] / r["floor_47"] for r in rows if r["floor_47"] > 0])}
    out.write_text(json.dumps({"header": header(f"levin-phase-floor 07d Part A': {variant}'s own CC declaration on v4", __file__,
                                                [Path(vj["path"])]), "variant": variant, "est": est, "kappa": kappa,
                               "part": [part, nparts], "summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
