"""levin-phase-floor prompt 07d Part A' (section 9.3): tables 1-6 from the leaf run's records, by the rules fixed in
state/rules_tables_a2.md. New in Part A'.

Inputs: this Part's leaf records (out/leaves_a2/partA_round*/, nb*/), Part A's leaf records (Part A's out/leaves/round*/),
Part A's candidates (state/candidates_07d.json) and its tables 2 and 3 (out/tables/d_e.json, d_t.json), all read only.
Part A's declaration formulas (d_e_table_07d.py: s_own, chop_tail, estimate, declaration) are transcribed below and checked
against its recorded D on every leaf of its table 2.
Writes out/tables/t1_v4_v3, t2_de_v4, t3_cc93, t4_dt, t5_knots, t6_cost (.json and .md), and out/tables/kappa93.json.
Usage: PYTHONPATH=<tree>:<scripts> python -B tables_a2.py
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, ROOT, header  # noqa: E402

fx = float.fromhex
OUT = ROOT / "out" / "tables"
T_GRID = [4, 6, 8, 10, 12, 16, 20]
BANDS = [("band_6_8", 6, 8), ("band_8_10", 8, 10), ("band_10_12", 10, 12), ("band_12_16", 12, 16), ("band_16_20", 16, 20),
         ("band_20_32", 20, 32)]
KAPPA_CHOICES = [3, 4, 5, 6, 8]


def fxn(x):
    return None if x is None else fx(x)


def verdict(D, te, eps):
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def kclass(n):
    return "unknown" if n is None else "none" if n == 0 else "1-4" if n <= 4 else "5+"


def kind_of(i):
    return "group" if "|group|" in i else "analytic"


def stratum_of(i):
    return i.split("#")[0]


def family(i):
    return stratum_of(i).split("|")[0]


# --- Part A's formulas (d_e_table_07d.py), transcribed
def pa_s_own(l, tail_k):
    ck = np.array([fx(x) for x in l["ingredients"]["ck_fine"]])
    n = len(ck)
    cut = l["ingredients"]["chop_fine"]
    tail = ck[cut:] if cut < n else ck[-tail_k:]
    return float(np.sqrt(np.mean(tail * tail)) * math.sqrt((n - 1) / 2.0)) * math.sqrt(l["ingredients"]["sum_w2"])


def pa_chop_tail(l):
    ck = np.array([fx(x) for x in l["ingredients"]["ck_fine"]])
    cut = l["ingredients"]["chop_fine"]
    if cut >= len(ck):
        return fx(l["ingredients"]["tail_sum4"])
    return 0.5 * l["width"] * float(np.sum(np.abs(ck[max(cut - 4, 0):cut])))


def pa_estimate(l, name):
    g = l["ingredients"]
    return {"pair": fx(l["pair"]), "nest3": fx(g["nest3"]), "tail_sum4": fx(g["tail_sum4"]),
            "tail_max4x2": fx(g["tail_max4x2"]), "chop_tail": pa_chop_tail(l),
            "pair_or_tail4": max(fx(l["pair"]), fx(g["tail_sum4"]))}[name]


def pa_declaration(l, cand, consts):
    floor = fx(l["floor"])
    E = pa_estimate(l, cand["estimate"])
    noise = 0.0
    if cand.get("noise") == "s_own":
        noise = cand["kappa"] * pa_s_own(l, consts["TAIL_K"])
    elif cand.get("noise") == "s_decl":
        dec = l["noise"].get("declared")
        if not dec:
            return None
        noise = (cand["kappa"] * max(dec["theta_abserr_lo"], dec["theta_abserr_hi"]) * l["noise"]["components"]["g"]["rms_fmag"]
                 * math.sqrt(l["ingredients"]["sum_w2"]))
    return max(E, floor) + noise


def load():
    mine, docs = {}, {}
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if not d.is_dir() or not (d.name.startswith("partA_") or d.name.startswith("nb")):
            continue
        for p in sorted(d.glob("*.json")):
            doc = json.loads(p.read_text())
            for L in doc["leaves"]:
                if L["id"] in mine:
                    raise SystemExit(f"{L['id']} twice")
                mine[L["id"]] = L
                docs[L["id"]] = doc
    pa = {}
    for rd in sorted((PARTA / "out" / "leaves").glob("round*")):
        for p in sorted(rd.glob("*.json")):
            for L in json.loads(p.read_text())["leaves"]:
                pa[L["id"]] = L
    return mine, docs, pa


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def g3(x):
    return "—" if x is None else f"{x:.3g}"


def ratio_stats(r):
    return (min(r), float(np.median(r)), max(r)) if r else (None, None, None)


def write(name, doc, lines, inputs):
    hdr = header(f"levin-phase-floor 07d Part A': {name}", __file__, inputs)
    (OUT / f"{name}.json").write_text(json.dumps({"header": hdr, **doc}, indent=1))
    (OUT / f"{name}.md").write_text("\n".join(hdr + [""] + lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mine, docs, pa = load()
    cdoc = json.loads((PARTA / "state" / "candidates_07d.json").read_text())
    cands, consts = cdoc["candidates"], cdoc["constants"]
    de = json.loads((PARTA / "out" / "tables" / "d_e.json").read_text())
    dt = json.loads((PARTA / "out" / "tables" / "d_t.json").read_text())
    inputs = [PARTA / "state" / "candidates_07d.json", PARTA / "out" / "tables" / "d_e.json",
              PARTA / "out" / "tables" / "d_t.json", ROOT / "state" / "rules_tables_a2.md"]
    ok = {i: L for i, L in mine.items() if "error" not in L}
    errs = {i: L["error"] for i, L in mine.items() if "error" in L}
    settled = {i for i, L in ok.items() if L["v4"]["settled"]}

    def R4(i):
        L = ok[i]
        return fx(L["v4"]["ref"]), fx(L["v4"]["eps_ref"])

    # ================= table 1: v4 against v3
    t1 = {"errors": errs, "n_leaves": len(mine), "n_ok": len(ok), "n_settled": len(settled), "checks": {}, "changes": [],
          "change_counts": {}, "ref_agreement": {}, "unknown_knots": {}, "unsettled_by_stratum": {}}
    # checks: my CC_47 and V_P equal Part A's; my s_own_47 equals Part A's; Part A's D recomputed equals its record
    c_cc = c_sig = c_sown = c_D = n_pa = 0
    for i, L in ok.items():
        if i not in pa or "error" in pa[i]:
            continue
        n_pa += 1
        c_cc += L["cc"]["cc47"] == pa[i]["cc47"]
        c_sig += L["v3_sigma"]["sigma_hat"] == pa[i]["noise"]["sigma_hat"]
        c_sown += abs(L["cc"]["c47"]["s_own"] - pa_s_own(pa[i], consts["TAIL_K"])) <= 1e-12 * max(L["cc"]["c47"]["s_own"], 1e-300)
    de_leaves = {f"{r['stratum']}#{r['rank']}": r for r in de["leaves"]}
    for i, r in de_leaves.items():
        for nm, D in r["D"].items():
            mine_D = pa_declaration(pa[i], cands[nm], consts)
            c_D += mine_D is not None and abs(mine_D - D) <= 1e-12 * abs(D)
    t1["checks"] = {"part_a_leaves": n_pa, "cc47_equal": c_cc, "sigma_hat_equal": c_sig, "s_own47_equal": c_sown,
                    "part_a_D_recomputed_equal": c_D, "part_a_D_total": sum(len(r["D"]) for r in de_leaves.values())}
    chg = Counter()
    # every Part A leaf measured here: CC strata (table 2) take every candidate on CC_47; gate and band strata (table 3)
    # take every candidate on CC_47 and the Levin route's own declaration on V_P. v3's verdict from Part A's ref and
    # eps_ref, v4's from this Part's; D from Part A's leaf record by Part A's formulas.
    v3_mismatch = 0
    for i in sorted(ok):
        if i not in pa or "error" in pa[i]:
            continue
        fam = family(i)
        tabs = (["table2"] if fam in ("chase", "window") or fam.startswith("gate_q") else []) + \
               (["table3"] if fam.startswith("gate_q") or fam.startswith("band_") else [])
        if i not in settled:
            for tb in tabs:
                chg[(stratum_of(i), tb, "v4 unsettled")] += 1
            continue
        ref, eps = R4(i)
        ref3, eps3 = fx(pa[i]["ref"]), fx(pa[i]["eps_ref"])
        cc47 = fx(pa[i]["cc47"])
        items = []
        for nm, cand in cands.items():
            D = pa_declaration(pa[i], cand, consts)
            if D is not None:
                items.append(("cc " + nm, D, cc47))
        lv = pa[i]["levin"]
        items.append(("levin own", fx(lv["own_decl"]), fx(lv["vP"])))
        for tb in tabs:
            for nm, D, val in items:
                if tb == "table2" and nm == "levin own":
                    continue
                v3 = verdict(D, abs(val - ref3), eps3)
                v4 = verdict(D, abs(val - ref), eps)
                if tb == "table2" and i in de_leaves and nm[3:] in de_leaves[i]["verdicts"]:
                    v3_mismatch += de_leaves[i]["verdicts"][nm[3:]] != v3
                if v3 != v4:
                    chg[(stratum_of(i), tb, f"{nm}: {v3} -> {v4}")] += 1
                    t1["changes"].append({"id": i, "table": tb, "candidate": nm, "v3": v3, "v4": v4, "D": D,
                                          "te_v4": abs(val - ref), "eps_v4": eps, "te_v3": abs(val - ref3), "eps_v3": eps3,
                                          "n_knots": pa[i]["n_knots"]})
    t1["checks"]["v3_verdict_recomputed_differs_from_part_a_d_e"] = v3_mismatch
    t1["change_counts"] = [{"stratum": k[0], "table": k[1], "change": k[2], "n": n} for k, n in sorted(chg.items())]
    # reference agreement by knot class
    agr = defaultdict(lambda: {"n": 0, "disagree": 0, "ratio": []})
    for i, L in ok.items():
        if i not in pa or i not in settled or "error" in pa[i]:
            continue
        ref, eps = R4(i)
        d = abs(ref - fx(pa[i]["ref"]))
        e = eps + fx(pa[i]["eps_ref"])
        A = agr[(kind_of(i), kclass(L["n_knots"]))]
        A["n"] += 1
        A["disagree"] += d > e
        if fx(pa[i]["eps_ref"]) > 0:
            A["ratio"].append(d / fx(pa[i]["eps_ref"]))
    t1["ref_agreement"] = {f"{k[0]}|{k[1]}": {"n": v["n"], "disagree_beyond_both_eps": v["disagree"],
                                               "median_diff_over_eps_v3": float(np.median(v["ratio"])) if v["ratio"] else None,
                                               "max_diff_over_eps_v3": max(v["ratio"]) if v["ratio"] else None}
                           for k, v in sorted(agr.items())}
    # unsettled by stratum, and the unknown-knots class
    for i, L in ok.items():
        s = stratum_of(i)
        U = t1["unsettled_by_stratum"].setdefault(s, {"drawn": 0, "unsettled": 0})
        U["drawn"] += 1
        U["unsettled"] += i not in settled
    unk = [i for i, L in ok.items() if L["n_knots"] is None]
    why = Counter((docs[i].get("edges_note") or "")[:160] for i in unk)
    t1["unknown_knots"] = {
        "n": len(unk), "by_kind": dict(Counter(kind_of(i) for i in unk)),
        "by_stratum": dict(sorted(Counter(stratum_of(i) for i in unk).items())),
        "why": dict(why.most_common()), "settled": sum(1 for i in unk if i in settled),
        "unsettled": sum(1 for i in unk if i not in settled),
        "others_settled": sum(1 for i in ok if ok[i]["n_knots"] is not None and i in settled),
        "others_unsettled": sum(1 for i in ok if ok[i]["n_knots"] is not None and i not in settled),
        "verdict_changes_on_them": sum(1 for c in t1["changes"] if c["id"] in set(unk)),
        "panels_hist": dict(Counter(ok[i]["v4"]["max_panels"] for i in unk)),
        "panels_hist_others": dict(Counter(ok[i]["v4"]["max_panels"] for i in ok if ok[i]["n_knots"] is not None))}
    stop_strata = {s: v for s, v in t1["unsettled_by_stratum"].items() if v["unsettled"] * 10 > v["drawn"]}
    t1["stop_more_than_a_tenth_unsettled"] = stop_strata
    L1 = [f"leaves {len(mine)}, errors {len(errs)}, settled {len(settled)} of {len(ok)}",
          f"checks: {json.dumps(t1['checks'])}", "",
          "**Strata with more than a tenth unsettled (prompt section 9.3 stop):** " + (json.dumps(stop_strata) or "none"), ""]
    L1 += ["### Verdict changes, v3 -> v4", ""] + md_table(["stratum", "table", "change", "n"],
                                                         [(c["stratum"], c["table"], c["change"], c["n"]) for c in t1["change_counts"]])
    L1 += ["", "### |ref_v4 - ref_v3| against eps_v3 + eps_v4, by kind and knots", ""] + md_table(
        ["kind|knots", "n", "disagree", "median diff/eps_v3", "max"],
        [(k, v["n"], v["disagree_beyond_both_eps"], g3(v["median_diff_over_eps_v3"]), g3(v["max_diff_over_eps_v3"]))
         for k, v in t1["ref_agreement"].items()])
    L1 += ["", "### Leaves whose knots are unknown (the orchestrator's item 3)", "", "```", json.dumps(t1["unknown_knots"], indent=1), "```"]
    write("t1_v4_v3", t1, L1, inputs)

    # ================= kappa_93 (rules: before table 3 is scored)
    rat = {"group": [], "analytic": []}
    for i, L in ok.items():
        c = L["cc"]["c93"]
        if c["chop"] < c["n"] and L["v3_sigma"]["sigma_hat"] > 0:
            rat[kind_of(i)].append(c["sigma_own"] / L["v3_sigma"]["sigma_hat"])
    q = float(np.percentile(rat["group"], 5))
    kappa = next((k for k in KAPPA_CHOICES if k * q >= 2.8), KAPPA_CHOICES[-1])
    kap = {"q_p05_group": q, "n_group": len(rat["group"]), "median_group": float(np.median(rat["group"])),
           "p05_analytic": float(np.percentile(rat["analytic"], 5)) if rat["analytic"] else None,
           "median_analytic": float(np.median(rat["analytic"])) if rat["analytic"] else None, "n_analytic": len(rat["analytic"]),
           "kappa93": kappa, "rule": "smallest of {3,4,5,6,8} with kappa * p05(sigma_own93/sigma_hat, chopped group) >= 2.8"}
    write("kappa93", kap, ["```", json.dumps(kap, indent=1), "```"], inputs)

    # ================= declarations on my leaves
    def decls47(L):
        floor = fx(L["cc"]["floor"])
        return {"tree": max(fx(L["cc"]["pair"]), floor),
                "nest3N4": max(abs(fx(L["cc"]["cc93"]) - fx(L["cc"]["cc47"])), floor) + 4 * L["cc"]["c47"]["s_own"]}

    def decls93(L):
        floor = fx(L["cc"]["floor"])
        c93, c47, c185 = fx(L["cc"]["cc93"]), fx(L["cc"]["cc47"]), fx(L["cc"]["cc185"])
        s93 = L["cc"]["c93"]["s_own"]
        return {"nest3N4": max(abs(c93 - c47), floor) + 4 * L["cc"]["c47"]["s_own"],
                "pair93": max(abs(c185 - c93), floor) + kappa * s93,
                "chop93": max(fx(L["cc"]["c93"]["chop_tail"]), floor) + kappa * s93,
                "nest93": max(abs(c93 - c47), floor) + kappa * s93,
                "tail93": max(fx(L["cc"]["c93"]["tail_sum4"]), floor) + kappa * s93}

    # ================= table 2: D-E on v4 (Part A's candidates on CC_47)
    t2 = {"table": {}, "qualifies": {}, "tree_understates": []}
    for i in sorted(settled):
        fam = family(i)
        if i not in pa or "error" in pa[i] or not (fam in ("chase", "window") or fam.startswith("gate_q")):
            continue
        Dd = {nm: pa_declaration(pa[i], cand, consts) for nm, cand in cands.items()}
        Dd = {k: v for k, v in Dd.items() if v is not None}
        ref, eps = R4(i)
        te = abs(fx(pa[i]["cc47"]) - ref)
        st = stratum_of(i)
        T = t2["table"].setdefault(st, {"drawn": 0, "scored": 0, "c": {}})
        T["drawn"] += 1
        vt = verdict(Dd["tree"], te, eps)
        if vt == "undecided":
            continue
        T["scored"] += 1
        for nm, D in Dd.items():
            C = T["c"].setdefault(nm, {"bounded": 0, "understates": 0, "undecided": 0, "ratios": []})
            C[verdict(D, te, eps)] += 1
            if te > 0:
                C["ratios"].append(D / te)
        if vt == "understates":
            L = ok[i]
            split = (L["item4"] or {}).get("split", {}).get("all")
            te_sp = None if not split else abs(fx(split["cc47"]) - ref)
            e24, e47 = fx(pa[i]["cc24"]) - ref, fx(pa[i]["cc47"]) - ref
            s_cc = L["v3_sigma"]["sigma_hat"] * math.sqrt(L["cc"]["c47"]["sum_w2"])
            if te_sp is not None and te_sp + eps < 0.5 * te:
                cause = "knots"
            elif e24 * e47 > 0 and abs(e47) >= 0.5 * abs(e24):
                cause = "common"
            elif te <= 4 * s_cc:
                cause = "noise"
            else:
                cause = "unexplained"
            t2["tree_understates"].append({"id": i, "te": te, "eps": eps, "D_tree": Dd["tree"], "n_knots": L["n_knots"],
                                           "te_split_all": te_sp, "noise_ratio": te / s_cc if s_cc else None,
                                           "common": e24 * e47 > 0 and abs(e47) >= 0.5 * abs(e24), "cause": cause})
    for nm in cands:
        t2["qualifies"][nm] = all(T["c"].get(nm, {}).get("understates", 0) == 0 for T in t2["table"].values())
    L2 = ["Part A's candidates on CC_47, re-scored on v4 (every measured leaf; scored = tree verdict not undecided)", ""]
    rows = []
    for st, T in sorted(t2["table"].items()):
        for nm, C in T["c"].items():
            mn, md, mx = ratio_stats(C["ratios"])
            rows.append((st, T["drawn"], T["scored"], nm, C["bounded"], C["understates"], C["undecided"], g3(mn), g3(md), g3(mx)))
    L2 += md_table(["stratum", "drawn (settled)", "scored", "candidate", "bounded", "understates", "undecided", "D/te min",
                    "median", "max"], rows)
    L2 += ["", f"qualifies: {json.dumps(t2['qualifies'])}", "",
           f"tree understates on {len(t2['tree_understates'])}; causes {dict(Counter(u['cause'] for u in t2['tree_understates']))}"]
    write("t2_de_v4", t2, L2, inputs)

    # ================= table 3: CC_93
    t3 = {"kappa93": kappa, "table": {}, "qualifies": {}, "closer": {}, "dot_fsum_max_rel": 0.0, "understatements": []}
    for i in sorted(settled):
        L = ok[i]
        ref, eps = R4(i)
        st = stratum_of(i)
        c93, c47 = fx(L["cc"]["cc93"]), fx(L["cc"]["cc47"])
        if c93 != 0:
            t3["dot_fsum_max_rel"] = max(t3["dot_fsum_max_rel"], abs(c93 - fx(L["cc"]["cc93_fsum"])) / abs(c93))
        te93, te47 = abs(c93 - ref), abs(c47 - ref)
        cl = "93 closer" if te47 - te93 > 2 * eps else "47 closer" if te93 - te47 > 2 * eps else "undecided"
        key = f"{family(i)}|{kind_of(i)}|{kclass(L['n_knots'])}"
        C = t3["closer"].setdefault(key, Counter())
        C[cl] += 1
        T = t3["table"].setdefault(st, {"settled": 0, "c": {}})
        T["settled"] += 1
        for nm, D in decls93(L).items():
            v = verdict(D, te93, eps)
            X = T["c"].setdefault(nm, {"bounded": 0, "understates": 0, "undecided": 0, "ratios": []})
            X[v] += 1
            if te93 > 0 and v != "undecided":
                X["ratios"].append(D / te93)
            if v == "understates":
                t3["understatements"].append({"id": i, "candidate": nm, "D": D, "te93": te93, "eps": eps,
                                              "n_knots": L["n_knots"]})
    for nm in ("nest3N4", "pair93", "chop93", "nest93", "tail93"):
        t3["qualifies"][nm] = {
            "returned": all(T["c"][nm]["understates"] == 0 for s, T in t3["table"].items() if not s.startswith("band_20_32v")),
            "with_visited": all(T["c"][nm]["understates"] == 0 for T in t3["table"].values())}
    t3["closer"] = {k: dict(v) for k, v in sorted(t3["closer"].items())}
    L3 = [f"kappa_93 = {kappa} (out/tables/kappa93.md); max |dot - fsum|/|CC_93| = {t3['dot_fsum_max_rel']:.3g}", ""]
    rows = []
    for st, T in sorted(t3["table"].items()):
        for nm, X in T["c"].items():
            mn, md, mx = ratio_stats(X["ratios"])
            rows.append((st, T["settled"], nm, X["bounded"], X["understates"], X["undecided"], g3(mn), g3(md), g3(mx)))
    L3 += md_table(["stratum", "settled", "candidate", "bounded", "understates", "undecided", "D/te min", "median", "max"], rows)
    L3 += ["", "qualifies (understates on no leaf; returned strata, and with band_20_32v): " + json.dumps(t3["qualifies"]), "",
           "### CC_93 against CC_47 (closer where v4 can tell), by family, kind and knots", ""]
    L3 += md_table(["class", "93 closer", "47 closer", "undecided"],
                   [(k, v.get("93 closer", 0), v.get("47 closer", 0), v.get("undecided", 0)) for k, v in t3["closer"].items()])
    write("t3_cc93", t3, L3, inputs)

    # ================= table 4: D-T as the tree runs it
    dt_ids = [i for i in settled if family(i).startswith("band_") or family(i).startswith("gate_q")]
    qual93 = [nm for nm, q in t3["qualifies"].items() if q["returned"]]
    agg = defaultdict(lambda: {"scored": 0, "bounded": 0, "understates": 0, "undecided": 0, "ratios": [], "te_req": [],
                               "untrusted": 0, "own_cc": 0, "own_guarded": 0, "r_untrusted": [], "levin_own": Counter()})
    per_leaf = []
    for i in dt_ids:
        L = ok[i]
        ref, eps = R4(i)
        fam, kd, kc = family(i), kind_of(i), kclass(L["n_knots"])
        visited = fam == "band_20_32v"
        req = L["request_share"]
        checks = {(c["T_pi"], c["fallback"]): c for c in L["item1"] if "error" not in c}
        rec = {"id": i, "V_over_pi": L["V_over_pi"], "n_knots": L["n_knots"], "fate": (L.get("fate") or {}).get("fate"),
               "by_T": {}}
        for Tp in T_GRID:
            if L["V"] >= Tp * math.pi:
                for fb in ("cc47", "cc93"):
                    c = checks.get((Tp, fb))
                    key = (Tp, "levin", fb, kd, fam, kc)
                    A = agg[key]
                    if c is None:
                        continue
                    if not c.get("checked"):
                        A["own_cc" if c["own_route"] == "cc" else "own_guarded"] += 1
                        rec["by_T"][f"{Tp}|levin|{fb}"] = "own solve " + c["own_route"] + (" guarded" if c["own_guarded"] else "")
                        continue
                    te_own = abs(fx(c["V_P"]) - ref)
                    A["levin_own"][verdict(fx(c["own_decl"]), te_own, eps)] += 1
                    if not c["trusted"]:
                        A["untrusted"] += 1
                        A["r_untrusted"].append(c["r"])
                        rec["by_T"][f"{Tp}|levin|{fb}"] = {"trusted": False, "r": c["r"], "delta1": c["delta1"],
                                                           "refused": c["refused_by_rate_rule"]}
                        continue
                    te = abs(fx(c["Q"]) - ref)
                    D = fx(c["total_err_trusted"])
                    v = verdict(D, te, eps)
                    rec["by_T"][f"{Tp}|levin|{fb}"] = {"trusted": True, "v": v, "D": D, "te": te, "eps": eps}
                    if v != "undecided":
                        A["scored"] += 1
                        A[v] += 1
                        if te > 0:
                            A["ratios"].append(D / te)
                        A["te_req"].append(te / req)
                    else:
                        A["undecided"] += 1
            else:
                c47, c93 = fx(L["cc"]["cc47"]), fx(L["cc"]["cc93"])
                for dn, D, val in [(f"cc47:{k}", v, c47) for k, v in decls47(L).items()] + \
                                  [(f"cc93:{k}", v, c93) for k, v in decls93(L).items()]:
                    te = abs(val - ref)
                    v = verdict(D, te, eps)
                    A = agg[(Tp, "cc", dn, kd, fam, kc)]
                    rec["by_T"][f"{Tp}|cc|{dn}"] = v
                    if v != "undecided":
                        A["scored"] += 1
                        A[v] += 1
                        if te > 0:
                            A["ratios"].append(D / te)
                        A["te_req"].append(te / req)
                    else:
                        A["undecided"] += 1
        per_leaf.append(rec)
    # verdict per T, kind, pairing (returned leaves only; band_20_32v apart)
    pairings = {"tree": ("cc47", "cc47:tree"), "nest3N4 on CC_47": ("cc47", "cc47:nest3N4")}
    for nm in qual93:
        pairings[f"CC_93 {nm}"] = ("cc93", f"cc93:{nm}")
    verd = {}
    for Tp in T_GRID:
        for kd in ("group", "analytic"):
            for pn, (fb, dn) in pairings.items():
                def side(route, which, kc=None):
                    s = u = 0
                    for k, A in agg.items():
                        if k[0] == Tp and k[1] == route and k[2] == which and k[3] == kd and k[4] != "band_20_32v" \
                                and (kc is None or k[5] == kc):
                            s += A["scored"]
                            u += A["understates"]
                    return ("vacuous" if s == 0 else "fails" if u else "bounded"), s, u

                lv, cc = side("levin", fb), side("cc", dn)
                byk = {kc: {"levin": side("levin", fb, kc), "cc": side("cc", dn, kc)} for kc in ("none", "1-4", "5+", "unknown")}
                verd[f"{Tp}|{kd}|{pn}"] = {"T_pi": Tp, "kind": kd, "pairing": pn, "levin": lv, "cc": cc, "by_knots": byk}
    t4 = {"pairings": pairings, "qualifying_cc93": qual93,
          "rows": [{"T_pi": k[0], "route": k[1], "decl": k[2], "kind": k[3], "family": k[4], "knots": k[5],
                    **{x: A[x] for x in ("scored", "bounded", "understates", "undecided", "untrusted", "own_cc", "own_guarded")},
                    "ratio": ratio_stats(A["ratios"]), "te_over_request_median": float(np.median(A["te_req"])) if A["te_req"] else None,
                    "te_over_request_le1": sum(1 for x in A["te_req"] if x <= 1), "r_untrusted_median":
                        float(np.median(A["r_untrusted"])) if A["r_untrusted"] else None, "levin_own": dict(A["levin_own"])}
                   for k, A in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0]))],
          "verdicts": verd, "per_leaf": per_leaf}
    L4 = ["Per T, call kind and pairing (Levin side's fallback, CC side's declaration); returned leaves only "
          "(band_20_32v, visited not returned, is in the rows, never pooled here). Each side: verdict, scored, understates.", ""]
    L4 += md_table(["T/pi", "kind", "pairing", "Levin side (V >= T)", "CC side (V < T)"],
                   [(v["T_pi"], v["kind"], v["pairing"], f"{v['levin'][0]} ({v['levin'][1]}, {v['levin'][2]})",
                     f"{v['cc'][0]} ({v['cc'][1]}, {v['cc'][2]})") for v in verd.values()])
    L4 += ["", "### Rows by T, route, declaration, kind, family and knots", ""]
    L4 += md_table(["T/pi", "route", "decl/fallback", "kind", "family", "knots", "scored", "bounded", "understates", "undecided",
                    "untrusted", "own CC/guarded", "D/te min, median, max", "te/request median", "te<=request", "Levin own decl"],
                   [(r["T_pi"], r["route"], r["decl"], r["kind"], r["family"], r["knots"], r["scored"], r["bounded"],
                     r["understates"], r["undecided"], r["untrusted"], f"{r['own_cc']}/{r['own_guarded']}",
                     ", ".join(g3(x) for x in r["ratio"]), g3(r["te_over_request_median"]), r["te_over_request_le1"],
                     json.dumps(r["levin_own"]) if r["route"] == "levin" else "") for r in t4["rows"]])
    write("t4_dt", t4, L4, inputs)

    # ================= item 3: routes by kind and knots (accuracy against the request; which is closer)
    acc = defaultdict(lambda: {"n": 0, "VP": [], "Q": [], "CC47": [], "CC93": [], "VP_vs_93": Counter(), "Q_vs_93": Counter()})
    for i in dt_ids:
        L = ok[i]
        if not family(i).startswith("band_"):
            continue
        ref, eps = R4(i)
        req = L["request_share"]
        A = acc[(family(i), kind_of(i), kclass(L["n_knots"]))]
        A["n"] += 1
        e47, e93 = abs(fx(L["cc"]["cc47"]) - ref), abs(fx(L["cc"]["cc93"]) - ref)
        A["CC47"].append(e47 / req)
        A["CC93"].append(e93 / req)
        if "V_P" in L["levin_one"]:
            ev = abs(fx(L["levin_one"]["V_P"]) - ref)
            A["VP"].append(ev / req)
            A["VP_vs_93"]["VP closer" if e93 - ev > 2 * eps else "93 closer" if ev - e93 > 2 * eps else "undecided"] += 1
        c6 = next((c for c in L["item1"] if c.get("T_pi") == 6 and c.get("fallback") == "cc47" and c.get("checked")), None)
        if c6 is not None:
            eq = abs(fx(c6["Q"]) - ref)
            A["Q"].append(eq / req)
            A["Q_vs_93"]["Q closer" if e93 - eq > 2 * eps else "93 closer" if eq - e93 > 2 * eps else "undecided"] += 1
    t4b = {f"{k[0]}|{k[1]}|{k[2]}": {"n": A["n"], **{f"{x}_te_over_req_median": float(np.median(A[x])) if A[x] else None
                                                    for x in ("VP", "Q", "CC47", "CC93")},
                                    **{f"{x}_te_le_req": sum(1 for y in A[x] if y <= 1) for x in ("VP", "Q", "CC47", "CC93")},
                                    "VP_vs_93": dict(A["VP_vs_93"]), "Q_vs_93": dict(A["Q_vs_93"])}
           for k, A in sorted(acc.items())}
    L4b = md_table(["family|kind|knots", "n", "te/req median VP, Q(6pi), CC47, CC93", "te<=req VP, Q, CC47, CC93",
                    "VP vs CC93", "Q vs CC93"],
                   [(k, v["n"], ", ".join(g3(v[f"{x}_te_over_req_median"]) for x in ("VP", "Q", "CC47", "CC93")),
                     ", ".join(str(v[f"{x}_te_le_req"]) for x in ("VP", "Q", "CC47", "CC93")), json.dumps(v["VP_vs_93"]),
                     json.dumps(v["Q_vs_93"])) for k, v in t4b.items()])
    write("t4b_routes_by_kind_knots", {"rows": t4b}, L4b, inputs)

    # ================= table 5: knots by kind
    kinds = ["phase", "G", "node", "mid", "break", "all"]
    cnt = defaultdict(list)
    jumps = defaultdict(lambda: defaultdict(list))
    carry = defaultdict(lambda: defaultdict(Counter))
    for i, L in ok.items():
        if L.get("item4") is None or i not in settled:
            continue
        fam = family(i)
        ref, eps = R4(i)
        for k in kinds:
            cnt[(fam, k)].append(L["item4"]["counts"].get(k, 0))
        for k, row in L["item4"]["jumps"].items():
            for ing in ("f0", "f1", "theta_raw"):
                x = row.get(ing) or {}
                ctl = x.get("control_same_h") or x.get("control_h_control")
                if not x.get("jump") or not ctl:
                    continue
                for m in range(4):
                    jumps[(k, ing)][m].append(x["jump"][m] / ctl[m] if ctl[m] > 0 else math.inf)
        unsplit = {"cc47": abs(fx(L["cc"]["cc47"]) - ref), "cc93": abs(fx(L["cc"]["cc93"]) - ref),
                   "V_P": abs(fx(L["levin_one"]["V_P"]) - ref) if "V_P" in L["levin_one"] else None}
        for k, sp in L["item4"]["split"].items():
            for rl in ("cc47", "cc93", "V_P"):
                if sp.get(rl) is None or unsplit[rl] is None:
                    continue
                te_s = abs(fx(sp[rl]) - ref)
                te = unsplit[rl]
                carry[k][rl]["lowers" if te_s + eps < 0.5 * te else "raises" if te_s > 2 * te + eps else "neither"] += 1
    t5 = {"counts": {f"{f}|{k}": {"leaves": len(v), "median": float(np.median(v)), "max": max(v),
                                  "with_any": sum(1 for x in v if x > 0)} for (f, k), v in sorted(cnt.items())},
          "jumps": {f"{k}|{ing}": {f"d{m}": {"n": len(v), "median_ratio": float(np.median(v)),
                                             "above_10x": sum(1 for x in v if x > 10)} for m, v in d.items()}
                    for (k, ing), d in sorted(jumps.items())},
          "carries": {k: {rl: dict(c) for rl, c in d.items()} for k, d in sorted(carry.items())}}
    L5 = ["### Counts inside a leaf, by family and kind (group leaves with knots read)", ""]
    L5 += md_table(["family|kind", "leaves", "median", "max", "leaves with any"],
                   [(k, v["leaves"], g3(v["median"]), v["max"], v["with_any"]) for k, v in t5["counts"].items()])
    L5 += ["", "### Continuity: jump / control (same h), at the kind's knot nearest the leaf's midpoint", ""]
    L5 += md_table(["kind|ingredient", "n", "d0 median, >10x", "d1", "d2", "d3"],
                   [(k, v["d0"]["n"], *[f"{g3(v[f'd{m}']['median_ratio'])}, {v[f'd{m}']['above_10x']}" for m in range(4)])
                    for k, v in t5["jumps"].items()])
    L5 += ["", "### The error it carries: true error after splitting at that kind's knots alone, against unsplit", ""]
    L5 += md_table(["kind", "CC47 lowers/raises/neither", "CC93", "V_P"],
                   [(k, *[f"{d.get(rl, {}).get('lowers', 0)}/{d.get(rl, {}).get('raises', 0)}/{d.get(rl, {}).get('neither', 0)}"
                          for rl in ("cc47", "cc93", "V_P")]) for k, d in t5["carries"].items()])
    write("t5_knots", t5, L5, inputs)

    # ================= table 6: cost
    cost = defaultdict(lambda: defaultdict(list))
    for i, L in ok.items():
        fam = family(i)
        c = L["cost"]
        for k in ("cc47_solve_s", "cc93_s", "cc185_s", "levin_solve_s"):
            cost[fam][k].append(c[k])
        for ch in L["item1"]:
            if ch.get("checked") and ch["fallback"] == "cc47":
                cost[fam][f"check_T{ch['T_pi']}_s"].append(ch["wall_s"])
                cost[fam][f"check_T{ch['T_pi']}_solves"].append(ch["solves"])
                cost[fam][f"check_T{ch['T_pi']}_cc_subsolves"].append(sum(1 for x in ch["sub_routes"] if x == "cc"))
    t6 = {f: {k: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in sorted(d.items())}
          for f, d in sorted(cost.items())}
    L6 = ["Points sampled: CC_47 47, CC_93 93, CC_185 185, one Levin solve 24 (N); a checked Levin leaf is 7 solves.", ""]
    L6 += md_table(["family", "quantity", "n", "median", "p90"],
                   [(f, k, v["n"], g3(v["median"]), g3(v["p90"])) for f, d in t6.items() for k, v in d.items()])
    write("t6_cost", t6, L6, inputs)
    print(json.dumps({"t1": {k: t1[k] for k in ("n_leaves", "n_ok", "n_settled", "checks")},
                      "stop_strata": stop_strata, "kappa": kap, "t2_qualifies": t2["qualifies"],
                      "t3_qualifies": t3["qualifies"]}, indent=1))


if __name__ == "__main__":
    main()
