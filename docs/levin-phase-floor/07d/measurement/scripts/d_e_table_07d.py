"""levin-phase-floor prompt 07d Part A, section 2.3 (D-E): every candidate declaration of CC_{2N-1}'s error, scored on the
leaf sample against the leaf references. New in 07d.

The candidates are defined in state/candidates_07d.json, written after D-N (section 2.2) and before this script scores
anything; each is a formula over the leaf's own quantities (leafcc_07d.py ``ingredients``), with its constants, basis and
cost in extra evaluations:
- ``s_own`` = sigma_own sqrt(sum w_j^2) on the fine rule, sigma_own the fine coefficients' plateau by standardChop
  (rms of the coefficients past the cutoff, x sqrt((n - 1)/2)) where it chops, else the rms of the last TAIL_K
  coefficients x sqrt((n - 1)/2) (an upper reading, truncation included);
- ``s_decl`` = max(theta_abserr at the leaf's ends) x rms|f| x sqrt(sum w^2), the scatter the caller's declared phase
  error would give (only where the call declares theta_abserr: the analytic blocks);
- forms: ``max(pair, floor)`` (tree); ``max(E, floor) + kappa s`` with E one of pair, nest3, tail_sum4, tail_max4x2,
  chop_tail (the standardChop-read decaying part: (b - a)/2 times the sum of |c_k| over the four coefficients up to the
  cutoff, or tail_sum4 if it does not chop), max(pair, tail_sum4); s one of s_own, s_decl, or none.
Verdicts as state/refs_rule_leaf_07d.txt. A leaf is scored if the tree's verdict is not undecided; the table counts each
stratum's fill prefix (up to and including its 40th scored leaf). Per candidate and stratum: bounded, understates,
undecided (on scored leaves; the undecided column counts the candidate's own undecided verdicts among them), D/true
error min, median, max (on scored leaves with true error > 0), how often the estimate sits at or below the floor, extra
evaluations. A candidate **qualifies** if it understates on no scored leaf of any stratum.
Also: every leaf where the tree's declaration understates, with the cause tests (knots and the split, an error common
to CC_N and CC_{2N-1}, noise, the reference's check).
Writes out/tables/d_e.json and out/tables/d_e.md. Usage: PYTHONPATH=<tree>:<scripts> python -B d_e_table_07d.py
"""

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402
from d_n_table_07d import prefix_leaves  # noqa: E402

fx = float.fromhex


def s_own(l, tail_k):
    ck = np.array([fx(x) for x in l["ingredients"]["ck_fine"]])
    n = len(ck)
    cut = l["ingredients"]["chop_fine"]
    tail = ck[cut:] if cut < n else ck[-tail_k:]
    return float(np.sqrt(np.mean(tail * tail)) * math.sqrt((n - 1) / 2.0)) * math.sqrt(l["ingredients"]["sum_w2"])


def chop_tail(l):
    ck = np.array([fx(x) for x in l["ingredients"]["ck_fine"]])
    cut = l["ingredients"]["chop_fine"]
    if cut >= len(ck):
        return fx(l["ingredients"]["tail_sum4"])
    return 0.5 * l["width"] * float(np.sum(np.abs(ck[max(cut - 4, 0):cut])))


def estimate(l, name):
    g = l["ingredients"]
    return {"pair": fx(l["pair"]), "nest3": fx(g["nest3"]), "tail_sum4": fx(g["tail_sum4"]),
            "tail_max4x2": fx(g["tail_max4x2"]), "chop_tail": chop_tail(l),
            "pair_or_tail4": max(fx(l["pair"]), fx(g["tail_sum4"]))}[name]


def declaration(l, cand, consts):
    floor = fx(l["floor"])
    E = estimate(l, cand["estimate"])
    noise = 0.0
    if cand.get("noise") == "s_own":
        noise = cand["kappa"] * s_own(l, consts["TAIL_K"])
    elif cand.get("noise") == "s_decl":
        dec = l["noise"].get("declared")
        if not dec:
            return None, None  # the candidate does not apply: the call declares no theta_abserr
        noise = (cand["kappa"] * max(dec["theta_abserr_lo"], dec["theta_abserr_hi"]) * l["noise"]["components"]["g"]["rms_fmag"]
                 * math.sqrt(l["ingredients"]["sum_w2"]))
    return max(E, floor) + noise, E <= floor


def verdict(D, te, eps):
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def main():
    cands_p = ROOT / "state" / "candidates_07d.json"
    cdoc = json.loads(cands_p.read_text())
    cands, consts = cdoc["candidates"], cdoc["constants"]
    sel = json.loads((ROOT / "state" / "selection_leaves_07d.json").read_text())["selection"]
    state = json.loads((ROOT / "out" / "leaves" / "fill_state.json").read_text())
    tab, under, per_leaf = {}, [], []
    for st, rk, l, v_tree in prefix_leaves(sel, state, {"gate_q", "chase", "window", "guarded"}):
        if l is None or "error" in l:
            continue
        te = abs(fx(l["cc47"]) - fx(l["ref"]))
        eps = fx(l["eps_ref"])
        T = tab.setdefault(st, {"drawn": 0, "scored": 0, "c": {}})
        T["drawn"] += 1
        scored = v_tree in ("bounded", "understates")
        T["scored"] += scored
        rec = {"stratum": st, "rank": rk, "start": l["start"], "end": l["end"], "true_err": te, "eps_ref": eps,
               "ref_check": l["ref_check"], "scored": scored, "verdicts": {}, "D": {}}
        for name, cand in cands.items():
            D, at_floor = declaration(l, cand, consts)
            if D is None:
                continue
            vv = verdict(D, te, eps)
            rec["verdicts"][name], rec["D"][name] = vv, D
            if not scored:
                continue
            C = T["c"].setdefault(name, {"bounded": 0, "understates": 0, "undecided": 0, "ratios": [], "at_floor": 0})
            C[vv] += 1
            C["at_floor"] += at_floor
            if te > 0:
                C["ratios"].append(D / te)
        per_leaf.append(rec)
        if scored and v_tree == "understates":
            s_cc = l["noise"]["s_cc"]
            e24, e47 = fx(l["err24"]), fx(l["err47"])
            sp = l.get("split_at_knots")
            under.append({"stratum": st, "rank": rk, "start": l["start"], "end": l["end"], "V_over_6pi": l["V_over_6pi"],
                          "true_err": te, "D_tree": rec["D"]["tree"], "eps_ref": eps, "ref_check": l["ref_check"],
                          "n_knots": l["n_knots"], "err_split_at_knots": None if not sp else fx(sp["err_split"]),
                          "err24": e24, "err47": e47, "common": (e24 * e47 > 0 and abs(e47) >= 0.5 * abs(e24)),
                          "noise_ratio": te / s_cc if s_cc > 0 else None, "s_cc": s_cc,
                          "pair": fx(l["pair"]), "floor": fx(l["floor"])})
    out = {}
    for st, T in sorted(tab.items()):
        out[st] = {"drawn": T["drawn"], "scored": T["scored"], "candidates": {}}
        for name, C in T["c"].items():
            r = C["ratios"]
            out[st]["candidates"][name] = {k: C[k] for k in ("bounded", "understates", "undecided", "at_floor")} | {
                "ratio_min": min(r) if r else None, "ratio_median": float(np.median(r)) if r else None,
                "ratio_max": max(r) if r else None, "extra_evals": cands[name]["extra_evals"]}
    qual = {name: all(out[st]["candidates"].get(name, {}).get("understates", 0) == 0 for st in out) for name in cands}
    applies = {name: sorted(st for st in out if name in out[st]["candidates"]) for name in cands}
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: D-E, the candidate estimates scored",
                 [cands_p, ROOT / "out" / "leaves" / "fill_state.json", ROOT / "state" / "selection_leaves_07d.json"])
    od = ROOT / "out" / "tables"
    od.mkdir(parents=True, exist_ok=True)
    (od / "d_e.json").write_text(json.dumps({"header": hdr, "candidates": cands, "constants": consts, "table": out,
                                             "qualifies": qual, "applies_to": applies, "tree_understates": under, "leaves": per_leaf}, indent=1))
    L = [l.replace("# ", "", 1) for l in hdr] + [""]
    L += ["| stratum | drawn | scored | candidate | bounded | understates | undecided | D/true min | median | max | est <= floor | extra evals |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, o in out.items():
        for name, c in o["candidates"].items():
            g = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
            L.append(f"| {st} | {o['drawn']} | {o['scored']} | {name} | {c['bounded']} | {c['understates']} | {c['undecided']} | "
                     f"{g(c['ratio_min'])} | {g(c['ratio_median'])} | {g(c['ratio_max'])} | {c['at_floor']} | {c['extra_evals']} |")
    L += ["", "**Qualifies** (understates on no scored leaf of any stratum): " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in qual.items()), "",
          f"## The {len(under)} scored leaves where the tree's declaration understates", "",
          "| stratum | rank | V/6pi | true err | D tree | eps_ref | ref check | knots | err split at knots | err CC_N | err CC_2N-1 | common | true/s_CC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for u in under:
        g = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
        L.append(f"| {u['stratum']} | {u['rank']} | {u['V_over_6pi']:.3f} | {u['true_err']:.3g} | {u['D_tree']:.3g} | {u['eps_ref']:.2g} | "
                 f"{u['ref_check']} | {u['n_knots']} | {g(u['err_split_at_knots'])} | {u['err24']:.3g} | {u['err47']:.3g} | {u['common']} | {g(u['noise_ratio'])} |")
    (od / "d_e.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("qualifies:", qual)
    print("wrote", od / "d_e.json", sha256(od / "d_e.json"))


if __name__ == "__main__":
    main()
