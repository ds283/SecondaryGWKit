"""
levin-phase-floor prompt 05 section 6: score the run from its checkpoints and region pickles.

Reads the run's checkpoint.jsonl and unit_*.pkl, state/sample.json, and (descriptive only) prompt
04's checkpoint for its arm A on R1-R7. Runs nothing; imports nothing from either tree.

Definitions (prompt 05 section 6, as written):
- a row's **regions** are WKB_Levin_data.num_regions of the unit (04's "X regions");
- a **group** is a Levin group call, identified by (regime, group label); p's and c's calls on the
  same row are checked to be the same sequence of (regime, label, x_span) bit for bit;
- a group's **depth-18 count** is its regions at depth >= 18 (04's n_depth_ge_18);
- **p's scored set** on a row: the groups with at least 10 regions at depth >= 18 under p (03's
  definition, "here under p");
- **L1 / L1q, a row holds** iff c@1e-7 is in budget, every group of p's scored set has fewer than
  10 regions at depth >= 18 under c@1e-7, and regions(c@1e-7) <= regions(p) / 10. L1 is over the
  depth-20 rows of the sample whose (stored) top group is G+r, G-r or lone G; L1q over those whose
  top group is Gqr. Holds iff true on every such row.
- **L2, a control holds** iff |total_c - total_p| <= total_abserr_p (c and p at 1e-7).
- **L3, a row holds** iff |total_c(1e-7) - total_c(1e-8)| <= total_abserr_c(1e-7), on every
  depth-20 row with both c units in budget.
The inequalities are evaluated exactly on the doubles (fractions.Fraction); an infinite abserr
makes a rule true, a NaN false. The numbers printed are doubles.

Usage: python -I -B analyse.py RUN_DIR OUT_JSON   (text to stdout)
"""

import collections
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ray.cloudpickle as cloudpickle  # noqa: E402

from prov import MAIN, WIDER, header  # noqa: E402

RUN04 = MAIN / "var" / "runs" / "levin-phase-floor-04-remove-20261025T231110"
R03 = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def F(d):
    return None if d is None else float.fromhex(d["hex"])


def le_exact(lhs_a, lhs_b, rhs):
    """|a - b| <= rhs, exactly on the doubles. Returns (bool, |a-b| as float, ratio)."""
    if any(x is None for x in (lhs_a, lhs_b, rhs)):
        return None, None, None
    if any(math.isnan(x) for x in (lhs_a, lhs_b, rhs)):
        return False, float("nan"), float("nan")
    if math.isinf(lhs_a) or math.isinf(lhs_b):
        return False, float("inf"), float("inf")
    d = abs(Fraction(lhs_a) - Fraction(lhs_b))
    df = float(d)
    if math.isinf(rhs):
        return True, df, 0.0
    ok = d <= Fraction(rhs)
    ratio = (df / rhs) if rhs != 0 else (0.0 if df == 0 else float("inf"))
    return ok, df, ratio


def g(x, fmt=".3g"):
    if x is None:
        return "-"
    if isinstance(x, float):
        return format(x, fmt)
    return str(x)


def main():
    run_dir = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    lines = header(
        Path(__file__).resolve(),
        "levin-phase-floor 05 section 6: the wider re-measurement, scored",
        [run_dir / "checkpoint.jsonl", WIDER / "state" / "sample.json"],
    )
    out = lines.append
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    rows = sample["rows"]
    by_name = {r["name"]: r for r in rows}

    recs = [json.loads(l) for l in (run_dir / "checkpoint.jsonl").read_text().splitlines()]
    units = {r["unit"]: r for r in recs}
    out(f"# run {run_dir.name}: {len(recs)} checkpoint records")

    # provenance of every record
    prov_bad = []
    for r in recs:
        d = r["data"]
        want = {"p": "e6489519ee662434d1f8e430e1c172246678ac8c",
                "c": "53084dcb14f7835e22fe8ffde8c0ac09c20a6fca"}[d["tree"]]
        if d["tree_head"] != want or d["tree_status_short"].strip() != "":
            prov_bad.append(r["unit"])
    shas = collections.Counter(r["script_sha256"] for r in recs)
    out(f"# every record's script_sha256: {dict(shas)}; tree HEAD and status as expected on "
        f"every record: {not prov_bad} {prov_bad}")
    flags = collections.Counter()
    for r in recs:
        d = r["data"]
        for k, v in (d.get("dispatcher_flags") or {}).items():
            flags[(d["tree"], k)] += v
    out(f"# theta_mod_2pi calls by tree and phase flag: {dict(flags)}")
    qs_bad = [r["unit"] for r in recs if not r["data"]["qs_check"]["L0a_holds"]]
    out(f"# QuadSource recomputed equals the stored row at its nodes and adds B1's 3 nodes "
        f"(L0 clause (a)) in every unit: {not qs_bad} {qs_bad}")

    def unit(tree, rtol, name):
        r = units.get(f"{tree}-{rtol}-{name}")
        return None if r is None else r["data"]

    def regions(d):
        if d is None or d["status"] != "done":
            return None
        w = d["result"]["WKB_Levin_data"]
        return None if w is None else w["num_regions"]

    def maxdepth(d):
        if d is None or d["status"] != "done":
            return None
        w = d["result"]["WKB_Levin_data"]
        return None if w is None else w["max_depth"]

    def groups(d):
        return [((c["regime"], c["group"]), c) for c in d["calls"]]

    def load_pkl(d):
        return cloudpickle.loads((run_dir / d["pickle"]).read_bytes())

    over = [r["unit"] for r in recs if r["data"]["status"] == "over budget"]
    errs = [r["unit"] for r in recs if r["data"]["status"] == "error"]
    out(f"# over-budget units: {over if over else 'none'}; error units: {errs if errs else 'none'}")

    # ------------------------------------------------------------------ per-row table
    res = {}
    out("")
    out("== per row: p and c at rtol 1e-7, c at 1e-8 (regions = WKB_Levin_data.num_regions; "
        "d = max depth; wall in s)")
    out("   name           cls  band  stored(d,reg)  p(d,reg,wall)        c7(d,reg,wall,budget)        "
        "c8(d,reg,wall)    conv p/c7/c8")
    for r in rows:
        name = r["name"]
        p, c7, c8 = unit("p", "1e-7", name), unit("c", "1e-7", name), unit("c", "1e-8", name)
        st = p["stored"]
        sw = st["WKB_Levin_data"] or {}
        e = {
            "name": name,
            "role": r["role"],
            "cls": r["cls"],
            "band": r["band"],
            "k": float(r["k"]),
            "z_response": r["z_response"],
            "stored_depth": sw.get("max_depth"),
            "stored_regions": sw.get("num_regions"),
            "p_depth": maxdepth(p),
            "p_regions": regions(p),
            "p_wall": p["wall_s"],
            "c7_depth": maxdepth(c7),
            "c7_regions": regions(c7),
            "c7_wall": c7["wall_s"],
            "c7_budget": c7["budget_s"],
            "c7_status": c7["status"],
            "p_conv": p["result"]["total_converged"] if p["status"] == "done" else None,
            "c7_conv": c7["result"]["total_converged"] if c7["status"] == "done" else None,
        }
        if c8 is not None:
            e.update(
                c8_depth=maxdepth(c8),
                c8_regions=regions(c8),
                c8_wall=c8["wall_s"],
                c8_status=c8["status"],
                c8_conv=c8["result"]["total_converged"] if c8["status"] == "done" else None,
            )
        # totals
        tp, ap = F(p["result"]["total"]), F(p["result"]["total_abserr"])
        tc, ac = F(c7["result"]["total"]), F(c7["result"]["total_abserr"])
        e.update(total_p=tp, abserr_p=ap, total_c7=tc, abserr_c7=ac)
        if c8 is not None and c8["status"] == "done":
            e.update(total_c8=F(c8["result"]["total"]), abserr_c8=F(c8["result"]["total_abserr"]))
        e["total_stored"] = F(st["total"])
        e["abserr_stored"] = F(st["total_abserr"])
        res[name] = e
        out(
            f"   {name:<14s} {str(r['cls']):4s} {r['band']:4s}  "
            f"({g(e['stored_depth'])},{g(e['stored_regions'])})".ljust(44)
            + f"({g(e['p_depth'])},{g(e['p_regions'])},{e['p_wall']:.2f})".ljust(21)
            + f"({g(e['c7_depth'])},{g(e['c7_regions'])},{e['c7_wall']:.2f},{g(e['c7_budget'])} {e['c7_status']})".ljust(29)
            + (f"({g(e.get('c8_depth'))},{g(e.get('c8_regions'))},{e['c8_wall']:.2f})".ljust(18) if c8 is not None else "-".ljust(18))
            + f"{e['p_conv']}/{e['c7_conv']}/{e.get('c8_conv', '-')}"
        )

    # ------------------------------------------------------------------ groups, p vs c
    group_tables = {}
    seq_ok = {}
    for r in rows:
        name = r["name"]
        p, c7 = unit("p", "1e-7", name), unit("c", "1e-7", name)
        gp = groups(p)
        gc = groups(c7) if c7["status"] == "done" else []
        same = [
            (k, a["x_span"][0]["hex"], a["x_span"][1]["hex"]) for k, a in gp
        ] == [(k, a["x_span"][0]["hex"], a["x_span"][1]["hex"]) for k, a in gc]
        seq_ok[name] = same
        cmap = {k: c for k, c in gc}
        tab = []
        for k, a in gp:
            c = cmap.get(k)
            tab.append(
                {
                    "regime": k[0],
                    "group": k[1],
                    "p_n18": a["n_depth_ge_18"],
                    "p_regions": a["num_regions"],
                    "p_maxd": a["max_depth"],
                    "c_n18": None if c is None else c["n_depth_ge_18"],
                    "c_regions": None if c is None else c["num_regions"],
                    "c_maxd": None if c is None else c["max_depth"],
                    "scored": a["n_depth_ge_18"] >= 10,
                }
            )
        group_tables[name] = tab
        res[name]["scored_groups"] = [f"{t['regime']}/{t['group']}" for t in tab if t["scored"]]
    out("")
    out(f"# the group calls of p and c@1e-7 are the same sequence of (regime, label, x_span) on "
        f"every row: {all(seq_ok.values())} {[n for n, v in seq_ok.items() if not v]}")

    def l1_row(name):
        e = res[name]
        tab = group_tables[name]
        if e["c7_status"] != "done":
            return False, "over budget" if e["c7_status"] == "over budget" else e["c7_status"], None
        scored = [t for t in tab if t["scored"]]
        worst = max((t["c_n18"] for t in scored), default=None)
        clause1 = all(t["c_n18"] is not None and t["c_n18"] < 10 for t in scored)
        pr, cr = e["p_regions"], e["c7_regions"]
        clause2 = pr is not None and cr is not None and Fraction(cr) <= Fraction(pr, 10)
        detail = {
            "n_scored": len(scored),
            "worst_c_n18_in_scored": worst,
            "clause1": clause1,
            "p_regions": pr,
            "c_regions": cr,
            "ratio": None if not pr else cr / pr,
            "clause2": clause2,
        }
        return clause1 and clause2, None, detail

    def l1_table(names, title):
        out(f"   {'row':<14s} band  p(d,regions)  c7(d,regions)  c/p      scored groups under p (n>=18: p -> c)        clause1 clause2 row")
        verdicts = []
        for n in names:
            ok, why, det = l1_row(n)
            e = res[n]
            sg = [
                f"{t['regime']}/{t['group']} {t['p_n18']}->{t['c_n18']}"
                for t in group_tables[n]
                if t["scored"]
            ]
            out(
                f"   {n:<14s} {e['band']:4s}  ({g(e['p_depth'])},{g(e['p_regions'])})".ljust(36)
                + f"({g(e['c7_depth'])},{g(e['c7_regions'])})".ljust(15)
                + f"{g(det['ratio'] if det else None, '.4f')}".ljust(9)
                + f"{'; '.join(sg) if sg else '(none)'}".ljust(45)
                + f"{det['clause1'] if det else '-'}   {det['clause2'] if det else '-'}   "
                + ("holds" if ok else f"FAILS{' (' + why + ')' if why else ''}")
            )
            verdicts.append((n, ok, det))
        return verdicts

    # ------------------------------------------------------------------ L1, L1q as written
    d20 = [r["name"] for r in rows if r["role"] == "d20"]
    l1_rows = [n for n in d20 if res[n]["cls"] in ("G+r", "G-r", "G")]
    l1q_rows = [n for n in d20 if res[n]["cls"] == "Gqr"]
    scores = {}
    out("")
    out("== L1, as written: every depth-20 row of the sample whose top group is G+r, G-r or lone G "
        f"({len(l1_rows)} rows)")
    v = l1_table(l1_rows, "L1")
    nh = sum(1 for _n, ok, _d in v if ok)
    scores["L1"] = {"rows": len(v), "hold": nh, "fail": [n for n, ok, _ in v if not ok],
                    "verdict": "holds" if nh == len(v) else "fails"}
    out(f"   L1: true on {nh} of {len(v)} rows -> {scores['L1']['verdict'].upper()}; "
        f"false on {scores['L1']['fail']}")
    out("")
    out(f"== L1q, as written: every depth-20 row of the sample whose top group is Gqr ({len(l1q_rows)} rows)")
    v = l1_table(l1q_rows, "L1q")
    nh = sum(1 for _n, ok, _d in v if ok)
    scores["L1q"] = {"rows": len(v), "hold": nh, "fail": [n for n, ok, _ in v if not ok],
                     "verdict": "holds" if nh == len(v) else "fails"}
    out(f"   L1q: true on {nh} of {len(v)} rows -> {scores['L1q']['verdict'].upper()}; "
        f"false on {scores['L1q']['fail']}")

    # depth-18 regions against z_response on failing Gqr rows (the rule's own report)
    for n in scores["L1q"]["fail"]:
        out(f"   L1q failure {n}: regions at depth >= 18 under c@1e-7 (and p), x = log(1+z'), "
            f"z_response = {res[n]['z_response']:.6g}")
        for tree in ("c", "p"):
            d = unit(tree, "1e-7", n)
            pk = load_pkl(d)
            n_deep = sum(int((c["depth"] >= 18).sum()) for c in pk["regions"])
            out(f"      {tree}: {n_deep} regions at depth >= 18 in this row's group calls; max "
                f"depth {max((int(c['depth'].max()) for c in pk['regions'] if len(c['depth'])), default=None)}")
            for c in pk["regions"]:
                m = c["depth"] >= 18
                if m.any():
                    zs = np.expm1(0.5 * (c["start"][m] + c["end"][m]))
                    out(f"      {tree} {c['regime']}/{c['group']}: {int(m.sum())} regions; z' "
                        f"from {zs.min():.6g} to {zs.max():.6g}; (1+z')/(1+z_resp) "
                        f"{(1 + zs.min()) / (1 + res[n]['z_response']):.6g} to "
                        f"{(1 + zs.max()) / (1 + res[n]['z_response']):.6g}")

    # ------------------------------------------------------------------ the second reading
    capped = [n for n in d20 if res[n]["p_depth"] == 20]
    out("")
    out("== THE SECOND READING (not scored; the user's question): the rows at depth 20 under p, "
        "i.e. still capped after B1 and before 05's change")
    out(f"   {len(capped)} of {len(d20)} depth-20 rows of the sample reach depth 20 under p: {capped}")
    out(f"   not capped under p: {[n for n in d20 if res[n]['p_depth'] != 20]}")
    out("   rows whose top group is G+r, G-r or lone G (L1's rule on them):")
    v1 = l1_table([n for n in capped if res[n]["cls"] in ("G+r", "G-r", "G")], "L1-capped")
    out("   rows whose top group is Gqr (L1q's rule on them):")
    v2 = l1_table([n for n in capped if res[n]["cls"] == "Gqr"], "L1q-capped")
    scores["L1_capped"] = {"rows": len(v1), "hold": sum(1 for _n, ok, _ in v1 if ok),
                           "fail": [n for n, ok, _ in v1 if not ok]}
    scores["L1q_capped"] = {"rows": len(v2), "hold": sum(1 for _n, ok, _ in v2 if ok),
                            "fail": [n for n, ok, _ in v2 if not ok]}
    out(f"   L1's rule on the capped rows: true on {scores['L1_capped']['hold']} of "
        f"{scores['L1_capped']['rows']}; false on {scores['L1_capped']['fail']}")
    out(f"   L1q's rule on the capped rows: true on {scores['L1q_capped']['hold']} of "
        f"{scores['L1q_capped']['rows']}; false on {scores['L1q_capped']['fail']}")
    out("   their groups, p -> c@1e-7 (regions; n at depth >= 18; max depth), every group:")
    for n in capped:
        out(f"   {n}:")
        for t in group_tables[n]:
            out(f"      {t['regime']:>4s}/{t['group']:<6s} regions {t['p_regions']:>6d} -> {g(t['c_regions']):>4s}; "
                f"n>=18 {t['p_n18']:>6d} -> {g(t['c_n18']):>3s}; max depth {t['p_maxd']:>2d} -> {g(t['c_maxd'])}"
                + ("   [scored under p]" if t["scored"] else ""))

    # ------------------------------------------------------------------ L2
    out("")
    out("== L2: every control, |total_c - total_p| <= total_abserr_p (both at 1e-7)")
    ctrl = [r["name"] for r in rows if r["role"] == "control"]
    l2 = []
    for n in ctrl:
        e = res[n]
        ok, df, ratio = le_exact(e["total_c7"], e["total_p"], e["abserr_p"])
        bit = float(e["total_c7"]).hex() == float(e["total_p"]).hex()
        out(f"   {n:<12s} band {e['band']:4s} p {e['total_p']!r} c {e['total_c7']!r} |dt| {g(df)} "
            f"abserr_p {g(e['abserr_p'])} ratio {g(ratio)} bit-identical {bit} -> {'holds' if ok else 'FAILS'}")
        l2.append((n, ok, ratio, bit))
    scores["L2"] = {"rows": len(l2), "hold": sum(1 for x in l2 if x[1]),
                    "fail": [x[0] for x in l2 if not x[1]],
                    "max_ratio": max(x[2] for x in l2),
                    "verdict": "holds" if all(x[1] for x in l2) else "fails"}
    out(f"   L2: true on {scores['L2']['hold']} of {len(l2)} -> {scores['L2']['verdict'].upper()}; "
        f"largest ratio {g(scores['L2']['max_ratio'])}")

    # ------------------------------------------------------------------ L3
    out("")
    out("== L3: every depth-20 row with both c units in budget, |total_c(1e-7) - total_c(1e-8)| "
        "<= total_abserr_c(1e-7)")
    l3 = []
    for n in d20:
        e = res[n]
        if e["c7_status"] != "done" or e.get("c8_status") != "done":
            out(f"   {n:<14s} excluded: c7 {e['c7_status']}, c8 {e.get('c8_status')}")
            continue
        ok, df, ratio = le_exact(e["total_c7"], e["total_c8"], e["abserr_c7"])
        q5 = df / abs(e["total_c8"]) if e["total_c8"] else None
        out(f"   {n:<14s} {e['band']:4s} {str(e['cls']):4s} c7 {e['total_c7']!r} c8 {e['total_c8']!r} "
            f"|dt| {g(df)} abserr_c7 {g(e['abserr_c7'])} ratio {g(ratio)} |dt|/|c8| {g(q5)} -> "
            f"{'holds' if ok else 'FAILS'}")
        l3.append((n, ok, ratio, q5))
    finite = [x[2] for x in l3 if x[2] is not None and not math.isnan(x[2])]
    scores["L3"] = {"rows": len(l3), "hold": sum(1 for x in l3 if x[1]),
                    "fail": [x[0] for x in l3 if not x[1]],
                    "max_ratio": max(finite) if finite else None,
                    "verdict": "holds" if all(x[1] for x in l3) else "fails"}
    out(f"   L3: true on {scores['L3']['hold']} of {len(l3)} -> {scores['L3']['verdict'].upper()}; "
        f"largest ratio {g(scores['L3']['max_ratio'])}; false on {scores['L3']['fail']}")
    l3c = [x for x in l3 if x[0] in capped]
    out(f"   (on the {len(l3c)} rows capped under p: true on {sum(1 for x in l3c if x[1])}; "
        f"ratios {[g(x[2]) for x in l3c]})")

    # ------------------------------------------------------------------ descriptive
    out("")
    out("== descriptive: |total_c - total_p| / total_abserr_p on the depth-20 rows (04's N4, wider; "
        "c includes the modulus correction)")
    n4 = []
    for n in d20:
        e = res[n]
        ok, df, ratio = le_exact(e["total_c7"], e["total_p"], e["abserr_p"])
        rel = df / abs(e["total_p"]) if e["total_p"] else None
        out(f"   {n:<14s} {e['band']:4s} {str(e['cls']):4s} capped-under-p {e['p_depth'] == 20!s:5s} "
            f"ratio {g(ratio)} |dt|/|p| {g(rel)} abserr_p {g(e['abserr_p'])}")
        n4.append((n, ratio))
    over1 = [(n, r) for n, r in n4 if r is not None and r > 1]
    out(f"   ratio > 1 on {len(over1)} of {len(n4)}: {[(n, g(r)) for n, r in over1]}")

    out("")
    out("== descriptive: cost and regions by class and band (sums over the depth-20 rows; wall in s)")
    out("   cell          rows  capped(p)  p regions  c7 regions  c8 regions   p wall   c7 wall   c8 wall")
    cells = collections.OrderedDict()
    for n in d20:
        e = res[n]
        key = ("03" if n in R03 else e["band"], e["cls"])
        cells.setdefault(key, []).append(e)
    for (b, cls), es in cells.items():
        out(f"   {b:4s} {str(cls):4s}     {len(es):4d}  {sum(1 for e in es if e['p_depth'] == 20):9d}  "
            f"{sum(e['p_regions'] or 0 for e in es):9d}  {sum(e['c7_regions'] or 0 for e in es):10d}  "
            f"{sum(e.get('c8_regions') or 0 for e in es):10d}  {sum(e['p_wall'] for e in es):7.2f}  "
            f"{sum(e['c7_wall'] for e in es):8.2f}  {sum(e.get('c8_wall') or 0 for e in es):8.2f}")
    for lab, sel in (("low band (drawn)", lambda n: res[n]["band"] == "low"),
                     ("mid band (drawn)", lambda n: res[n]["band"] == "mid" and n not in R03),
                     ("03's R1-R7 (mid)", lambda n: n in R03)):
        es = [res[n] for n in d20 if sel(n)]
        out(f"   {lab:<18s}: rows {len(es)}, capped under p {sum(1 for e in es if e['p_depth'] == 20)}, "
            f"regions p {sum(e['p_regions'] or 0 for e in es)} -> c7 {sum(e['c7_regions'] or 0 for e in es)}, "
            f"wall p {sum(e['p_wall'] for e in es):.1f} s -> c7 {sum(e['c7_wall'] for e in es):.1f} s, "
            f"max depth p {max(e['p_depth'] or 0 for e in es)} -> c7 {max(e['c7_depth'] or 0 for e in es)} "
            f"-> c8 {max(e.get('c8_depth') or 0 for e in es)}")
    ctr = [res[n] for n in ctrl]
    out(f"   controls: regions p {sum(e['p_regions'] or 0 for e in ctr)} -> c7 "
        f"{sum(e['c7_regions'] or 0 for e in ctr)}; wall p {sum(e['p_wall'] for e in ctr):.2f} -> "
        f"c7 {sum(e['c7_wall'] for e in ctr):.2f}")

    out("")
    out("== descriptive: total_converged, p -> c@1e-7 -> c@1e-8")
    tr = collections.Counter()
    for r in rows:
        e = res[r["name"]]
        tr[(r["role"], e["p_conv"], e["c7_conv"], e.get("c8_conv"))] += 1
    for k, v in sorted(tr.items(), key=str):
        out(f"   {k[0]:<8s} p {k[1]!s:5s} -> c7 {k[2]!s:5s} -> c8 {k[3]!s:5s}: {v} rows")
    out(f"   capped under p: " + "; ".join(
        f"{n} {res[n]['p_conv']}->{res[n]['c7_conv']}->{res[n].get('c8_conv')}" for n in capped))

    out("")
    out("== descriptive: 03's R1-R7 against 04's arm A (layer 1(a) at the pre-B1 tree 98e68d3, "
        "against TWO_PI, anchored at z_top) and X")
    try:
        r04 = [json.loads(l) for l in (RUN04 / "checkpoint.jsonl").read_text().splitlines()]
        a04 = {}
        for rr in r04:
            dd = rr["data"]
            if dd["status"] == "done":
                a04[(dd["arm"], dd["rtol"], dd["row"])] = dd["result"]["WKB_Levin_data"]["num_regions"]
        for n in R03:
            e = res[n]
            out(f"   {n}: regions 04 P {a04.get(('P', '1e-7', n))}, A {a04.get(('A', '1e-7', n))}, "
                f"X {a04.get(('X', '1e-7', n))}; 05 stored {e['stored_regions']}, p {e['p_regions']} "
                f"(depth {e['p_depth']}), c7 {e['c7_regions']} (depth {e['c7_depth']})")
    except FileNotFoundError:
        out("   04's run directory not found")

    out("")
    out("== for the board: every row, |total_p - total_stored| / total_abserr_stored "
        "(B1's effect on A3 v2's stored totals; p is B1's code)")
    b1 = []
    for r in rows:
        e = res[r["name"]]
        ok, df, ratio = le_exact(e["total_p"], e["total_stored"], e["abserr_stored"])
        rel = df / abs(e["total_stored"]) if e["total_stored"] else None
        out(f"   {r['name']:<14s} {r['role']:<7s} {r['band']:4s} stored {e['total_stored']!r} "
            f"(abserr {g(e['abserr_stored'])}) p {e['total_p']!r} |dt| {g(df)} ratio {g(ratio)} "
            f"rel {g(rel)} within {ok}")
        b1.append((r["name"], ratio, ok))
    beyond = [(n, g(x)) for n, x, ok in b1 if not ok]
    out(f"   beyond the stored error on {len(beyond)} of {len(b1)} rows: {beyond}")
    scores["B1_beyond_stored_error"] = [n for n, x, ok in b1 if not ok]

    out("")
    out("== VERDICTS")
    for k in ("L1", "L1q", "L2", "L3"):
        s = scores[k]
        out(f"   {k}: {s['verdict'].upper()} ({s['hold']} of {s['rows']} rows true; false on {s['fail']})")
    out(f"   (second reading, unscored) L1's rule on the rows capped under p: {scores['L1_capped']}")
    out(f"   (second reading, unscored) L1q's rule on the rows capped under p: {scores['L1q_capped']}")
    out(f"   over-budget units: {over if over else 'none'}")

    out_json.write_text(json.dumps({"rows": res, "groups": group_tables, "scores": scores,
                                    "capped_under_p": capped, "over_budget": over}, indent=1,
                                   default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
