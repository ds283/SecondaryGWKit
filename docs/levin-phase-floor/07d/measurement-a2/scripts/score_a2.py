"""levin-phase-floor prompt 07d Part A' (section 9.3 table 7): section 2.7's tables 4 (honesty) and 5 (convergence, reach
and cost) for this Part's variants, beside Part A's, and section 9.3's qualification. New in Part A', after Part A's
score_07d.py (a record), changed in its inputs and its qualification.

**References:** Part A's referenced calls and their references, as Part A scored them (Part A's out/tables/scores_07d.json
``calls``: ``ref`` to 25 digits, ``spread``, ``set``), read only; no call reference is re-read and none is new (log).
**Scored:** spread at most min(abserr)/100 over the tree and this Part's variants given (section 2.6's rule over the codes
scored here). **Verdicts:** converged and |v - ref| <= abserr: honest; converged and above: dishonest; else unconverged.
The tree's replays are Part A's (out/replays/tree/), the variants' this Part's.
**Table 5,** per variant against the tree over all 4,220 calls: converged changes each way, regions, wall, subregion
solves, route changes (``num_a2_route_changed``), knot gates (``num_a2_gate_knots``), knot breaks (``num_a2_knot_breaks``),
the chase stop's three counts and its stopped leaves' summed budget_err, calls whose stopped-window leaves differ (the set
of spans of window leaves that are phase_limited; the replays here record leaves, not windows), calls touched.
**Qualification** (section 9.3): (a) no referenced call converged and dishonest under it that is not at the tree; (b) its
declaration qualifies on table 2 or 3 against the value it returns (state/variants_a2.json ``est``; out/tables/t3_cc93.json
``qualifies``); (c) its threshold bounded, not vacuous, on both sides on table 4 (out/tables/t4_dt.json ``verdicts``,
pairing "CC_93 <est>", both kinds). (G) and (K) are judged on (c) at their V threshold (6 pi); a knot gate has no table-4
row of its own, which the output says.
Usage: PYTHONPATH=<tree>:<scripts> python -B score_a2.py VARIANT [VARIANT ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, ROOT, header  # noqa: E402

fx = float.fromhex


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def phase_limited_windows(doc):
    return sorted((l["start"], l["end"]) for l in doc["leaves"] if l["window"] and l["phase_limited"])


def main():
    variants = sys.argv[1:]
    codes = ["tree"] + variants
    sc = json.loads((PARTA / "out" / "tables" / "scores_07d.json").read_text())
    vspec = json.loads((ROOT / "state" / "variants_a2.json").read_text())["variants"]
    t3 = json.loads((ROOT / "out" / "tables" / "t3_cc93.json").read_text())
    t4 = json.loads((ROOT / "out" / "tables" / "t4_dt.json").read_text())
    sids = [s for s in (ROOT / "state" / "sids_all_a2.txt").read_text().split() if s]
    rep = {}
    for s in sids:
        rep[s] = {"tree": json.loads((PARTA / "out" / "replays" / "tree" / f"{fname(s)}.json").read_text())}
        for v in variants:
            rep[s][v] = json.loads((ROOT / "out" / "replays" / v / f"{fname(s)}.json").read_text())
    calls, t7 = {}, {c: Counter() for c in codes}
    t7_set = {}
    for sid, c in sorted(sc["calls"].items()):
        ref, spread = mpf(c["ref"]), float(c["spread"])
        ab = {k: fx(rep[sid][k]["replay"]["abserr"]) for k in codes}
        scored = spread <= min(ab.values()) / 100
        by = {}
        for k in codes:
            r = rep[sid][k]["replay"]
            ratio = float(abs(mpf(fx(r["value"])) - ref) / mpf(ab[k])) if ab[k] > 0 else float("inf")
            conv = r["converged"]
            verdict = ("honest" if ratio <= 1 else "dishonest") if conv else "unconverged"
            by[k] = {"ratio": ratio, "converged": conv, "verdict": verdict}
            if scored:
                t7[k][verdict] += 1
                t7_set.setdefault(c["set"].split("~")[0], {}).setdefault(k, Counter())[verdict] += 1
        calls[sid] = {"set": c["set"], "spread": spread, "scored": scored, "by": by}
    t5 = {}
    for v in variants:
        T = Counter()
        touched, up, down = [], [], []
        for sid in sids:
            a, b = rep[sid]["tree"], rep[sid][v]
            ra, rb = a["replay"], b["replay"]
            T["calls"] += 1
            T["regions_tree"] += ra["num_regions"]
            T["regions_var"] += rb["num_regions"]
            T["wall_tree"] += a["wall_s"]
            T["wall_var"] += b["wall_s"]
            T["solves_tree"] += a.get("num_subregion_solves") or 0
            T["solves_var"] += b.get("num_subregion_solves") or 0
            ex = b.get("extra_a2") or {}
            for k in ("num_a2_route_changed", "num_a2_gate_knots", "num_a2_knot_breaks", "num_a2_cc93", "num_a2_resampled"):
                T[k] += int(ex.get(k, 0))
            T["calls_route_changed"] += int(ex.get("num_a2_route_changed", 0) > 0)
            T["calls_gate_knots"] += int(ex.get("num_a2_gate_knots", 0) > 0)
            T["calls_knot_breaks"] += int(ex.get("num_a2_knot_breaks", 0) > 0)
            for k in ("num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections"):
                T[f"{k}_tree"] += ra[k]
                T[f"{k}_var"] += rb[k]
            T["stopped_decl_tree"] += sum(fx(l["budget_err"]) for l in a["leaves"] if l["chase_stopped"])
            T["stopped_decl_var"] += sum(fx(l["budget_err"]) for l in b["leaves"] if l["chase_stopped"])
            if phase_limited_windows(a) != phase_limited_windows(b):
                T["calls_stopped_window_leaves_differ"] += 1
            if not b["equals_tree"]:
                touched.append(sid)
            if not ra["converged"] and rb["converged"]:
                up.append(sid)
            if ra["converged"] and not rb["converged"]:
                down.append(sid)
            T["unconverged_var"] += not rb["converged"]
            T["unconverged_tree"] += not ra["converged"]
        t5[v] = dict(T, calls_touched=len(touched), calls_untouched=len(sids) - len(touched), converged_gained=len(up),
                     converged_lost=len(down), converged_gained_sids=up[:50], converged_lost_sids=down[:200],
                     regions_ratio=T["regions_var"] / T["regions_tree"], wall_ratio=T["wall_var"] / T["wall_tree"],
                     solves_ratio=T["solves_var"] / max(T["solves_tree"], 1))
    qual = {}
    for v in variants:
        s = vspec[v]
        new_dis = [sid for sid, c in calls.items() if c["scored"] and c["by"][v]["verdict"] == "dishonest"
                   and c["by"]["tree"]["verdict"] != "dishonest"]
        est = s["est"]
        est_ok = bool(t3["qualifies"].get(est, {}).get("returned"))
        Tp = int(s["T_pi"])
        thr = {kd: t4["verdicts"].get(f"{Tp}|{kd}|CC_93 {est}") for kd in ("group", "analytic")}
        thr_ok = all(x is not None and x["levin"][0] == "bounded" and x["cc"][0] == "bounded" for x in thr.values())
        qual[v] = {"new_converged_dishonest": new_dis, "declaration": est, "declaration_qualifies": est_ok,
                   "threshold_pi": Tp, "threshold_verdicts": {kd: (x["levin"], x["cc"]) if x else None for kd, x in thr.items()},
                   "threshold_bounded_both_sides": thr_ok,
                   "note": ("the knot gate (G) has no table-4 row of its own; judged at its V threshold" if s.get("gate_knots")
                            else ("knot bisection (K) leaves the gate at its V threshold" if s.get("knots") else None)),
                   "qualifies": (not new_dis) and est_ok and thr_ok}
    hdr = header("levin-phase-floor 07d Part A': table 7 (honesty and convergence) and the qualification", __file__,
                 [PARTA / "out" / "tables" / "scores_07d.json", ROOT / "state" / "variants_a2.json",
                  ROOT / "out" / "tables" / "t3_cc93.json", ROOT / "out" / "tables" / "t4_dt.json"])
    doc = {"header": hdr, "codes": codes, "table7_honesty": {k: dict(v) for k, v in t7.items()},
           "table7_honesty_by_set": {s: {k: dict(v) for k, v in d.items()} for s, d in t7_set.items()},
           "part_a_table4_beside": sc["table4"], "table7_convergence": t5, "qualification": qual, "calls": calls}
    out = ROOT / "out" / "tables" / "t7_scores.json"
    out.write_text(json.dumps(doc, indent=1))
    L = hdr + ["", "### Honesty (referenced calls, scored)", "", "| code | honest | dishonest | unconverged |", "|---|---|---|---|"]
    for k in codes:
        L.append(f"| {k} | {t7[k]['honest']} | {t7[k]['dishonest']} | {t7[k]['unconverged']} |")
    L += ["", "Part A's (its own codes, its scoring): " + json.dumps(sc["table4"]), "", "### By set", ""]
    for s_, d in sorted(t7_set.items()):
        L.append(f"- {s_}: " + "; ".join(f"{k} {dict(v)}" for k, v in d.items()))
    L += ["", "### Converged and dishonest calls", ""]
    for sid, c in sorted(calls.items()):
        if c["scored"] and any(c["by"][k]["verdict"] == "dishonest" for k in codes):
            L.append(f"- {sid} ({c['set']}): " + ", ".join(f"{k} {c['by'][k]['verdict']} {c['by'][k]['ratio']:.3g}" for k in codes))
    L += ["", "### Convergence, reach and cost (all 4,220 calls)", "", "| variant | touched | untouched | conv lost | conv gained | "
          "unconverged (tree) | regions/tree | solves/tree | wall/tree | route changes (calls) | knot gates (calls) | "
          "knot breaks (calls) | chase stops tree/var | stopped leaves tree/var | stopped decl tree/var | stopped-window leaves differ |",
          "|" + "---|" * 16]
    for v, T in t5.items():
        L.append(f"| {v} | {T['calls_touched']} | {T['calls_untouched']} | {T['converged_lost']} | {T['converged_gained']} | "
                 f"{T['unconverged_var']} ({T['unconverged_tree']}) | {T['regions_ratio']:.3f} | {T['solves_ratio']:.3f} | "
                 f"{T['wall_ratio']:.2f} | {T['num_a2_route_changed']} ({T['calls_route_changed']}) | {T['num_a2_gate_knots']} "
                 f"({T['calls_gate_knots']}) | {T['num_a2_knot_breaks']} ({T['calls_knot_breaks']}) | {T['num_chase_stops_tree']}/"
                 f"{T['num_chase_stops_var']} | {T['num_chase_stopped_leaves_tree']}/{T['num_chase_stopped_leaves_var']} | "
                 f"{T['stopped_decl_tree']:.3g}/{T['stopped_decl_var']:.3g} | {T['calls_stopped_window_leaves_differ']} |")
    L += ["", "### Qualification (section 9.3)", ""]
    for v, q in qual.items():
        L.append(f"- **{v}**: {'QUALIFIES' if q['qualifies'] else 'does not qualify'}; new converged-dishonest "
                 f"{len(q['new_converged_dishonest'])} {q['new_converged_dishonest']}; declaration {q['declaration']} "
                 f"qualifies {q['declaration_qualifies']}; threshold {q['threshold_pi']} pi bounded both sides "
                 f"{q['threshold_bounded_both_sides']} {json.dumps(q['threshold_verdicts'])}" + (f"; {q['note']}" if q["note"] else ""))
    (ROOT / "out" / "tables" / "t7_scores.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[len(hdr):]))


if __name__ == "__main__":
    main()
