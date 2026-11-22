"""levin-phase-floor prompt 07d Part A, section 2.7 tables 4 and 5, and the qualification. New in 07d.

**References** (each a reading of the integrand alone, so it scores every code):
- 07c Part A's regression set (``R:`` sids): the reference and stated error 07c Part A scored each call with
  (var/levin-phase-floor-07c/measurement/out/tables/scores_07c.json ``scored``: ``ref``, ``spread``, ``ref_key``, read only),
  under the call's 07c Part A sid (R2-28 is ``R:a:R2-28``, ``R:a:R2-28~o``, ``R:c-1e-7-0-13757-5`` and
  ``R:c-1e-8-0-13757-5``; R31's record ``R:r31``);
- (Ra) and (Rt) (``d:`` sids of their drawn calls, and the counterparts at the other rtol): this Part's readings,
  out/refs/readings/<key>-*.json, by state/refs_rule_calls_07d.txt (FCC or split FCC resolution 2, the spread of every
  reading).
**Scored:** the reference's stated error at most min(abserr)/100 over the codes scored (the tree and every variant given).
**Table 4 (honesty):** per code, converged and honest, converged and dishonest, unconverged (each call's ratio
|v - ref|/abserr recorded), by set. **Table 5 (convergence, reach and cost),** per variant against the tree, over every call
replayed by both: converged changes each way; regions and wall; route changes (the variant's ``num_07d_route_changed``);
fallback solves re-declared; window stops changed (the stopped windows' (lo, hi, stopped_after_split) sets compared);
the chase stop's three counts and the stopped leaves' summed declaration (budget_err); Rule A's changed skips (0 by
construction, out/cmd/01_gate.txt); calls touched (value, abserr, converged, num_regions or max_depth differing).
**Qualification** (section 2.7): a variant qualifies if (a) no referenced call is converged and dishonest under it that is
not so at the tree, (b) its estimate qualifies on table 2 (out/tables/d_e.json ``qualifies``), and (c) its threshold is
bounded on both sides on table 3 (out/tables/d_t.json ``thresholds``, under the variant's estimate).
(Ra)'s verdicts at 07c's code are reported beside, not folded in.
Usage: PYTHONPATH=<tree>:<scripts> python -B score_07d.py VARIANT [VARIANT ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import MEAS07C, header, sha256  # noqa: E402

fx = float.fromhex
EST_OF = {"tree": "tree", "Eg": "nest3N4", "Ea": "nest3N4", "ET20": "nest3N4", "ET12": "nest3N4", "Ep": "pairN4", "T20": "tree"}
T_OF = {"tree": "6pi", "Eg": "6pi", "Ea": "6pi", "ET20": "20pi", "ET12": "12pi", "Ep": "6pi", "T20": "20pi"}


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def other(sid):
    unit, rest = sid.split("|")
    p = unit.split("-")
    rt = "-".join(p[1:3])
    return "-".join([p[0], "1e-8" if rt == "1e-7" else "1e-7", p[3], p[4]]) + "|" + rest


def my_refs():
    out = {}
    rd = ROOT / "out" / "refs" / "readings"
    if not rd.exists():
        return out
    for inp in sorted((ROOT / "out" / "refs").glob("inputs_*.json")):
        for key, e in json.loads(inp.read_text())["inputs"].items():
            vals = {}
            for f in sorted(rd.glob(f"{key}-*.json")):
                d = json.loads(f.read_text())
                if "pieces" in str(d.get("readings", ""))[:20] or "readings" in d:
                    for res, r in d["readings"].items():
                        vals[f"fccsplit-r{res}"] = mp.fsum(mpf(x["value_30"]) for x in r["pieces"])
                elif d.get("value_30") is not None:
                    vals[f"{d['impl']}-r{d['res']}"] = mpf(d["value_30"])
            key2 = "fccsplit-r2" if "fccsplit-r2" in vals else ("fcc-r2" if "fcc-r2" in vals else None)
            if key2 is None:
                out[e["sid"]] = {"key": key, "ref": None, "readings": sorted(vals)}
                continue
            v = list(vals.values())
            r = {"key": key, "ref": vals[key2], "spread": float(max(v) - min(v)), "basis": "+".join(sorted(vals)),
                 "set": inp.stem.split("_")[1]}
            out[e["sid"]] = r
            out[other(e["sid"])] = dict(r, counterpart=True)
    return out


def main():
    codes = ["tree"] + sys.argv[1:]
    sc7 = json.loads((MEAS07C / "out" / "tables" / "scores_07c.json").read_text())["scored"]
    mine = my_refs()
    man = {}
    for p in sorted((ROOT / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    ra = {e["sid"] for e in json.loads((ROOT / "state" / "selection_ra_07d.json").read_text())["selection"]}
    rt_p = ROOT / "state" / "selection_rt_07d.json"
    rt = {e["sid"] for e in json.loads(rt_p.read_text())["selection"]} if rt_p.exists() else set()
    rep = {}
    for c in codes:
        for f in (ROOT / "out" / "replays" / c).glob("*.json"):
            d = json.loads(f.read_text())
            rep.setdefault(d["sid"], {})[c] = d
    t4 = {c: Counter() for c in codes}
    t4_set = {}
    calls = {}
    for sid, c in sorted(man.items()):
        if sid not in rep or any(k not in rep[sid] for k in codes):
            continue
        if sid.startswith("R:"):
            s7 = c.get("census_07c_sid") or ("r31-3-20363-1" if sid == "R:r31" else None)
            r7 = sc7.get(s7)
            if not r7:
                continue
            ref, spread, rset = mpf(r7["ref"]), float(r7["spread"]), "regression" + (" R2-28" if "R2-28" in r7["ref_key"] else "")
        else:
            base = sid[2:]
            r = mine.get(base)
            if not r or r["ref"] is None:
                continue
            ref, spread = r["ref"], r["spread"]
            orig = base if not r.get("counterpart") else other(base)
            rset = ("Ra" if orig in ra else ("Rt" if orig in rt else r["set"])) + ("~o" if r.get("counterpart") else "")
        ab = {k: fx(rep[sid][k]["replay"]["abserr"]) for k in codes}
        scored = spread <= min(ab.values()) / 100
        by = {}
        for k in codes:
            v = mpf(fx(rep[sid][k]["replay"]["value"]))
            ratio = float(abs(v - ref) / mpf(ab[k])) if ab[k] > 0 else float("inf")
            conv = rep[sid][k]["replay"]["converged"]
            verdict = ("honest" if ratio <= 1 else "dishonest") if conv else "unconverged"
            by[k] = {"ratio": ratio, "converged": conv, "verdict": verdict}
            if scored:
                t4[k][verdict] += 1
                t4_set.setdefault(rset.split("~")[0], {}).setdefault(k, Counter())[verdict] += 1
        calls[sid] = {"set": rset, "spread": spread, "scored": scored, "ref": mp.nstr(ref, 25), "by": by}
    t5 = {}
    for v in codes[1:]:
        T = Counter()
        touched, conv_up, conv_down = [], [], []
        for sid, d in rep.items():
            if "tree" not in d or v not in d:
                continue
            a, b = d["tree"], d[v]
            ra_, rb_ = a["replay"], b["replay"]
            T["calls"] += 1
            T["regions_tree"] += ra_["num_regions"]
            T["regions_var"] += rb_["num_regions"]
            T["wall_tree"] += a["wall_s"]
            T["wall_var"] += b["wall_s"]
            for k in ("num_07d_route_changed", "num_07d_cc_new_estimate", "num_07d_noise_term_larger", "num_07d_nest_resampled"):
                T[k] += int(b["extra_07d"].get(k, 0))
            T["calls_route_changed"] += int(b["extra_07d"].get("num_07d_route_changed", 0) > 0)
            for k in ("num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections"):
                T[f"{k}_tree"] += ra_[k]
                T[f"{k}_var"] += rb_[k]
            T["stopped_decl_tree"] += sum(fx(l["budget_err"]) for l in a["leaves"] if l["chase_stopped"])
            T["stopped_decl_var"] += sum(fx(l["budget_err"]) for l in b["leaves"] if l["chase_stopped"])
            ws_a = {(w["lo"], w["hi"], w.get("stopped_after_split")) for w in a["windows"] if w.get("stopped")}
            ws_b = {(w["lo"], w["hi"], w.get("stopped_after_split")) for w in b["windows"] if w.get("stopped")}
            if ws_a != ws_b:
                T["calls_window_stops_changed"] += 1
                T["window_stops_tree_only"] += len(ws_a - ws_b)
                T["window_stops_var_only"] += len(ws_b - ws_a)
            if any(ra_[k] != rb_[k] for k in ("value", "abserr", "converged", "num_regions", "max_depth")):
                touched.append(sid)
            if not ra_["converged"] and rb_["converged"]:
                conv_up.append(sid)
            if ra_["converged"] and not rb_["converged"]:
                conv_down.append(sid)
        t5[v] = dict(T, calls_touched=len(touched), converged_gained=len(conv_up), converged_lost=len(conv_down),
                     converged_gained_sids=conv_up[:50], converged_lost_sids=conv_down[:200], rule_a_changed_skips=0,
                     untouched_example=next((s for s in sorted(rep) if s not in touched and v in rep[s]), None))
    de = json.loads((ROOT / "out" / "tables" / "d_e.json").read_text())
    dt = json.loads((ROOT / "out" / "tables" / "d_t.json").read_text())
    qual = {}
    for v in codes[1:]:
        new_dis = [s for s, c in calls.items() if c["scored"] and c["by"][v]["verdict"] == "dishonest" and c["by"]["tree"]["verdict"] != "dishonest"]
        est_ok = bool(de["qualifies"].get(EST_OF[v]))
        thr = dt["thresholds"].get(T_OF[v], {})
        thr_ok = bool(thr.get("cc_below", {}).get(EST_OF[v], {}).get("bounded_both_sides"))
        # the Levin side is vacuous where table 3 holds no Levin leaf above the threshold (the bands stop at 20 pi)
        vac = int(thr.get("levin_above", {}).get("n", 0)) == 0
        qual[v] = {"new_converged_dishonest": new_dis, "estimate": EST_OF[v], "estimate_qualifies": est_ok,
                   "threshold": T_OF[v], "threshold_bounded_both_sides": thr_ok, "levin_side_vacuous": vac,
                   "qualifies": (not new_dis) and est_ok and thr_ok}
    ra_tree = {s: c for s, c in calls.items() if c["set"].startswith("Ra")}
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: tables 4 and 5, and the qualification",
                 [MEAS07C / "out" / "tables" / "scores_07c.json", ROOT / "out" / "tables" / "d_e.json", ROOT / "out" / "tables" / "d_t.json"])
    od = ROOT / "out" / "tables"
    doc = {"header": hdr, "codes": codes, "table4": {k: dict(v) for k, v in t4.items()},
           "table4_by_set": {s: {k: dict(v) for k, v in d.items()} for s, d in t4_set.items()},
           "table5": t5, "qualification": qual, "calls": calls,
           "ra_at_07c": {s: {"scored": c["scored"], "spread": c["spread"], **c["by"]["tree"]} for s, c in ra_tree.items()}}
    (od / "scores_07d.json").write_text(json.dumps(doc, indent=1))
    g = lambda x: "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))  # noqa: E731
    L = [l.replace("# ", "", 1) for l in hdr] + ["", f"Referenced calls: {len(calls)}, scored {sum(c['scored'] for c in calls.values())}.", "",
         "## Table 4 (honesty), every scored call", "", "| code | converged & honest | converged & dishonest | unconverged |", "|---|---|---|---|"]
    for k in codes:
        L.append(f"| {k} | {t4[k]['honest']} | {t4[k]['dishonest']} | {t4[k]['unconverged']} |")
    L += ["", "### by set", "", "| set | code | honest | dishonest | unconverged |", "|---|---|---|---|---|"]
    for s, d in sorted(t4_set.items()):
        for k in codes:
            c = d.get(k, Counter())
            L.append(f"| {s} | {k} | {c['honest']} | {c['dishonest']} | {c['unconverged']} |")
    L += ["", "### every converged-and-dishonest or unconverged scored call", "", "| sid | set | " + " | ".join(codes) + " |", "|---|---|" + "---|" * len(codes)]
    for s, c in sorted(calls.items()):
        if c["scored"] and any(b["verdict"] != "honest" for b in c["by"].values()):
            L.append(f"| {s} | {c['set']} | " + " | ".join(f"{b['verdict'][:5]} {b['ratio']:.3g}" for b in c["by"].values()) + " |")
    L += ["", "## Table 5 (convergence, reach, cost) against the tree", "",
          "| variant | calls | touched | conv gained | conv lost | regions / tree | wall / tree | route-changed solves (calls) | CC solves re-declared (noise term larger) | window-stop calls changed | chase stops / leaves / bisections (tree) | stopped leaves' declaration / tree | Rule A changes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, T in t5.items():
        L.append(f"| {v} | {T['calls']} | {T['calls_touched']} | {T['converged_gained']} | {T['converged_lost']} | {T['regions_var'] / max(T['regions_tree'], 1):.3f} | "
                 f"{T['wall_var'] / max(T['wall_tree'], 1e-9):.3f} | {T['num_07d_route_changed']} ({T['calls_route_changed']}) | {T['num_07d_cc_new_estimate']} ({T['num_07d_noise_term_larger']}) | "
                 f"{T['calls_window_stops_changed']} | {T['num_chase_stops_var']}/{T['num_chase_stopped_leaves_var']}/{T['num_chase_noise_bisections_var']} "
                 f"({T['num_chase_stops_tree']}/{T['num_chase_stopped_leaves_tree']}/{T['num_chase_noise_bisections_tree']}) | "
                 f"{T['stopped_decl_var'] / T['stopped_decl_tree'] if T['stopped_decl_tree'] else float('nan'):.3g} | 0 |")
    L += ["", "## Qualification", "", "| variant | new converged & dishonest | estimate (qualifies on table 2) | threshold (bounded both sides on table 3) | qualifies |", "|---|---|---|---|---|"]
    for v, q in qual.items():
        L.append(f"| {v} | {len(q['new_converged_dishonest'])} {q['new_converged_dishonest'][:4]} | {q['estimate']} ({'yes' if q['estimate_qualifies'] else 'no'}) | "
                 f"{q['threshold']} ({'yes' if q['threshold_bounded_both_sides'] else 'no'}{', Levin side vacuous: no leaf above it' if q['levin_side_vacuous'] else ''}) | "
                 f"{'yes' if q['qualifies'] else 'no'}{' (vacuously on its threshold)' if q['qualifies'] and q['levin_side_vacuous'] else ''} |")
    L += ["", "## (Ra) at 07c's code (beside the table, not folded in)", "", "| sid | set | scored | spread | converged | ratio | verdict |", "|---|---|---|---|---|---|---|"]
    for s, c in sorted(ra_tree.items()):
        b = c["by"]["tree"]
        L.append(f"| {s} | {c['set']} | {c['scored']} | {c['spread']:.2g} | {b['converged']} | {b['ratio']:.3g} | {b['verdict']} |")
    (od / "scores_07d.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print("wrote", od / "scores_07d.json", sha256(od / "scores_07d.json"))


if __name__ == "__main__":
    main()
