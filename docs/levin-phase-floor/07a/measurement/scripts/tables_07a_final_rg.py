"""levin-phase-floor prompt 07a Part A, section 2.4: the final tables, scored on r_g. A copy of tables_07a_final.py
(left as run), changed only for the user's ruling of 2026-11-11 on the pairs' formula, each change marked "07a r_g":
items 2 and 4 are scored on the cross-check's r_g = |v(1e-7) - v(1e-8)| / abserr(1e-7) (07's xcheck_rule_20261109.md);
the prompt's r is kept beside it, labelled. tables_07a_final.py's text follows.

levin-phase-floor prompt 07a Part A, section 2.4: the final tables. A copy of tables_07a.py (which wrote the interim
tables at stop 1, and is left as run), changed only for the user's rulings at stop 1, each change marked "07a final":
(b) and (g) run as b2 and g2 (log deviation 10) on items 1-2 and the no-window calls only, not on the 396; item 3 is
therefore undecided by construction for every variant (no (b) reading of the 396), and qualification is judged on
items 1-2 with item 3 undecided; and the pairs carry the cross-check's r_g = |v_7 - v_8| / e_7 beside the prompt's r,
labelled. tables_07a.py's text follows.

levin-phase-floor prompt 07a Part A, section 2.4: the tables, fixed before the run. New in 07a. Read only.

From the replays (OUT_DIR/<variant>/<sid>.json, replay_07a.py) at the tree and at variants a-g:
- **per call and variant:** value, abserr, converged, regions, maximum depth, windows stopped, leaves falling at the
  stop, wall time; a call whose replay is missing or timed out is listed as such, never dropped;
- **1. honesty on V4's 25:** |v - ref| / abserr, ref the call's FCC resolution-2 reading (07's rule: v00-v23 from
  07's out/v4/readings/<sid>-fcc-r2.json ``value_30``; z12 06g's ``ref_30``, scores_stage3/scores.json), all 25 of
  whose readings agree by 07's rule (its v4_scores.json); exact, doubles as Fractions of their hex and readings of
  their 30-digit strings;
- **2. the pairs:** r = |v_7 - v_8| / (e_7 + e_8), for 3-14501 and 3-35006;
- **3. the 396:** |v - v_b| / abserr, scored only where (b) converges with abserr_b <= abserr / 10; the rest undecided;
- **4. qualification:** a variant qualifies if no call is converged and dishonest (items 1 and 3 above 1) and each
  pair has r <= 1 or an unconverged member; for each, the cost against (a): calls unconverged, regions, wall;
- the no-window check (section 2.3): the five calls, every variant against the tree, bit for bit (value, abserr,
  converged, regions, depth and the region arrays' digest).
The attribution (item 5) is attr_tables_07a.py's.

Writes OUT_JSON and OUT_MD. Usage: python -B tables_07a.py OUT_DIR OUT_JSON OUT_MD [--variants tree a b c d e f g]
"""

import argparse
import datetime
import hashlib
import json
import socket
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
V4OUT = TREE / "docs" / "levin-phase-floor" / "07" / "census" / "out" / "v4"
SCORES06G = TREE / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "scores_stage3" / "scores.json"
FAIL = ("v08", "v09", "v12", "v17", "v22")
PAIRS = (("p14501_7", "p14501_8"), ("p35006_7", "p35006_8"))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def FH(h):
    return Fraction(float.fromhex(h))


def refs_v4():
    sc = json.loads((V4OUT / "v4_scores.json").read_text())
    out = {}
    for c in sc["calls"]:
        rd = json.loads((V4OUT / "readings" / f"{c['sid']}-fcc-r2.json").read_text())
        if c["ref_kind"] != "fcc_r2" or rd["sample"] != c["sid"]:
            raise SystemExit(f"{c['sid']}: reference is not its FCC r2 reading")
        out[c["sid"]] = {"ref_30": rd["value_30"], "basis": c["basis"], "verdict_07": c["verdict"],
                         "ratio_07": c["ratio_g"], "spread_over_tol": c["spread_over_tol"],
                         "source": str(V4OUT / "readings" / f"{c['sid']}-fcc-r2.json"),
                         "source_sha256": sha256(V4OUT / "readings" / f"{c['sid']}-fcc-r2.json")}
    z = [e for e in sc["z"] if e["sid"] == "z12"][0]
    r06 = [r for r in json.loads(SCORES06G.read_text())["rows"] if r["sid"] == "z12"][0]
    out["cal_z12"] = {"ref_30": r06["ref_30"], "basis": z["basis"], "verdict_07": z["verdict"], "ratio_07": z["ratio_g"],
                      "spread_over_tol": z["spread_over_tol"], "source": str(SCORES06G), "source_sha256": sha256(SCORES06G)}
    return out


def load(out_dir, variant, sid):
    p = Path(out_dir) / variant / f"{sid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def row(d):
    if d is None:
        return None
    r = d["replay"]
    return {"value": r["value"], "abserr": r["abserr"], "converged": r["converged"], "regions": r["num_regions"],
            "max_depth": r["max_depth"], "windows_stopped": d["n_windows_stopped"],
            "leaves_falling_at_stop": d["leaves_falling_at_stop"], "wall_s": d["wall_s"],
            "reproduces_census": d["reproduces_census"], "ier2": d.get("ier2_windows"),
            "regions_sha256": d["regions_sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("out_json")
    ap.add_argument("out_md")
    ap.add_argument("--variants", nargs="+", default=["tree", "a", "b2", "c", "d", "e", "f", "g2"])
    a = ap.parse_args()
    calls = json.loads((ROOT / "state" / "calls_07a.json").read_text())["calls"]
    status = {}
    for p in sorted(ROOT.glob("out/*_done.json")):
        for r in json.loads(p.read_text())["results"]:
            if r["status"] != "skipped (exists)":
                status[(r["sid"], r["variant"])] = r["status"]
    refs = refs_v4()
    V = a.variants
    sets = {s: sorted(k for k, c in calls.items() if c["set"] == s) for s in ("v4", "pairs", "nowindow", "x396")}
    per = {v: {} for v in V}
    # 07a final: b2 and g2 are not run on the 396 (the user's ruling at stop 1)
    NOT_RUN_396 = {"b2", "g2"}
    for v in V:
        for sid in calls:
            if v in NOT_RUN_396 and calls[sid]["set"] == "x396":
                per[v][sid] = None
                continue
            per[v][sid] = row(load(a.out_dir, v, sid))
    res = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "host": socket.gethostname(), "out_dir": str(Path(a.out_dir).resolve()),
                      "calls_manifest_sha256": sha256(ROOT / "state" / "calls_07a.json"),
                      "variants_json_sha256": sha256(ROOT / "out" / "variants" / "variants.json")},
           "variants": V, "per_call": per, "status": {f"{k[0]}|{k[1]}": s for k, s in status.items()}}
    # 1. honesty on V4's 25
    hon = {v: {} for v in V}
    for v in V:
        for sid in sets["v4"]:
            r = per[v][sid]
            if r is None:
                hon[v][sid] = None
                continue
            ref = Fraction(refs[sid]["ref_30"])
            ratio = abs(FH(r["value"]) - ref) / FH(r["abserr"]) if FH(r["abserr"]) > 0 else None
            hon[v][sid] = {"ratio": None if ratio is None else float(ratio), "honest": ratio is not None and ratio <= 1,
                           "converged": r["converged"],
                           "converged_dishonest": bool(r["converged"] and (ratio is None or ratio > 1))}
    res["refs_v4"] = refs
    res["honesty_v4"] = hon
    # 2. the pairs
    pr = {v: {} for v in V}
    for v in V:
        for s7, s8 in PAIRS:
            r7, r8 = per[v][s7], per[v][s8]
            if r7 is None or r8 is None:
                pr[v][s7[:6]] = None
                continue
            r = abs(FH(r7["value"]) - FH(r8["value"])) / (FH(r7["abserr"]) + FH(r8["abserr"]))
            r_g = abs(FH(r7["value"]) - FH(r8["value"])) / FH(r7["abserr"])  # 07a final: the cross-check's r_g, beside r
            pr[v][s7[:6]] = {"r": float(r), "r_g": float(r_g), "converged": [r7["converged"], r8["converged"]],
                             # 07a r_g: scored on r_g; the prompt's r kept for the record
                             "ok_r_prompt": bool(r <= 1 or not (r7["converged"] and r8["converged"])),
                             "ok": bool(r_g <= 1 or not (r7["converged"] and r8["converged"]))}
    res["pairs"] = pr
    # 3. the 396 against (b)
    x = {v: {} for v in V}
    for v in V:
        for sid in sets["x396"]:
            r = per[v][sid]
            # 07a final: (b) is not run on the 396, so no call is scored (undecided by construction)
            x[v][sid] = {"scored": False, "why": "undecided by construction: (b) not run on the 396 (user's ruling at stop 1)",
                         "converged": None if r is None else r["converged"], "replayed": r is not None}
            continue
            b = None
            if not (b["converged"] and FH(b["abserr"]) <= FH(r["abserr"]) / 10):
                x[v][sid] = {"scored": False, "why": "undecided: (b) unconverged or abserr_b > abserr/10",
                             "converged": r["converged"]}
                continue
            ratio = abs(FH(r["value"]) - FH(b["value"])) / FH(r["abserr"])
            x[v][sid] = {"scored": True, "ratio": float(ratio), "converged": r["converged"],
                         "converged_dishonest": bool(r["converged"] and ratio > 1)}
    res["x396"] = x
    # 4. qualification and cost
    q = {}
    for v in V:
        missing = [s for s in calls if per[v][s] is None and not (v in NOT_RUN_396 and calls[s]["set"] == "x396")]
        cd_v4 = [s for s in sets["v4"] if hon[v][s] and hon[v][s]["converged_dishonest"]]
        cd_x = [s for s in sets["x396"] if x[v][s].get("converged_dishonest")]
        bad_pairs = [p for p, e in pr[v].items() if e is None or not e["ok"]]
        main_sets = sets["v4"] + sets["pairs"] + sets["x396"]
        items12 = sets["v4"] + sets["pairs"]
        q[v] = {"qualifies": not cd_v4 and not cd_x and not bad_pairs and not missing,
                "qualifies_meaning": "on items 1-2 only; item 3 undecided by construction (07a final)",
                "regions_items12": sum(per[v][s]["regions"] for s in items12 if per[v][s] is not None),
                "wall_s_items12": sum(per[v][s]["wall_s"] for s in items12 if per[v][s] is not None),
                "x396_not_run": v in NOT_RUN_396,
                "missing": missing, "converged_dishonest_v4": cd_v4, "converged_dishonest_x396": cd_x,
                "pairs_failing": bad_pairs,
                "x396_scored": sum(1 for s in sets["x396"] if x[v][s]["scored"]),
                "x396_undecided": sum(1 for s in sets["x396"] if not x[v][s]["scored"]),
                "unconverged": {k: sum(1 for s in sets[k] if per[v][s] is not None and not per[v][s]["converged"])
                                for k in ("v4", "pairs", "x396")},
                "regions": sum(per[v][s]["regions"] for s in main_sets if per[v][s] is not None),
                "wall_s": sum(per[v][s]["wall_s"] for s in main_sets if per[v][s] is not None),
                "windows_stopped": sum(per[v][s]["windows_stopped"] for s in main_sets if per[v][s] is not None),
                "leaves_falling_at_stop": sum(per[v][s]["leaves_falling_at_stop"] for s in main_sets if per[v][s] is not None)}
    res["qualification"] = q
    # the no-window check
    nw = {}
    for v in V:
        if v == "tree":
            continue
        nw[v] = {}
        for sid in sets["nowindow"]:
            t, r = per["tree"][sid], per[v][sid]
            keys = ("value", "abserr", "converged", "regions", "max_depth", "regions_sha256", "windows_stopped")
            nw[v][sid] = None if (t is None or r is None) else all(t[k] == r[k] for k in keys)
    res["nowindow_check"] = nw
    # replay at 07's code reproduces the census (section 2.1's rule)
    res["reproduction"] = {v: {s: (per[v][s] or {}).get("reproduces_census") for s in calls} for v in ("tree", "a") if v in V}
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    # markdown
    L = [f"# levin-phase-floor 07a Part A: section 2.4's tables (final, items 2 and 4 scored on r_g)", "",
         "Items 2 and 4 are scored on the cross-check's r_g = |v_7 - v_8| / e_7 (the user's ruling of 2026-11-11); the "
         "prompt's r = |v_7 - v_8| / (e_7 + e_8) is kept beside it, labelled, for the record.", "",
         "(b) and (g) are run as b2 and g2 (log deviation 10), on items 1-2 and the no-window calls only (the user's ruling "
         "at stop 1). Item 3 is undecided by construction for every variant; 'qualifies' is judged on items 1-2 with item 3 "
         "undecided. Columns 'regions' and 'wall' sum V4's 25, the pairs and (where run) the 396; '... 1-2' columns sum V4 "
         "and the pairs only, comparable across all variants.", "",
         f"written {res['header']['written']} on {res['header']['host']}; script tables_07a_final_rg.py sha256 {res['header']['script_sha256']}; "
         f"calls manifest sha256 {res['header']['calls_manifest_sha256']}; variants.json sha256 {res['header']['variants_json_sha256']}", ""]
    for v in ("tree", "a"):
        if v in V:
            rr = res["reproduction"][v]
            bad = sorted(s for s, ok in rr.items() if ok is not True)
            L.append(f"Replays at {v} reproducing 07's census record bit for bit: {sum(1 for ok in rr.values() if ok is True)} of {len(rr)}; not: {bad}")
    L.append("")
    L.append("No-window check (five calls, every variant against the tree, bit for bit): " +
             "; ".join(f"{v}: {sum(1 for ok in d.values() if ok)}/{len(d)}" for v, d in nw.items()))
    L += ["", "## 4. Qualification and cost", "",
          "| variant | qualifies (items 1-2) | conv. & dishonest V4 | conv. & dishonest 396 | pairs failing | 396 scored / undecided | unconverged V4 / pairs / 396 | regions | wall s | regions 1-2 | wall s 1-2 | windows stopped | leaves falling at stop | missing |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        e = q[v]
        u396 = "not run" if e["x396_not_run"] else e["unconverged"]["x396"]
        L.append(f"| {v} | {'yes' if e['qualifies'] else 'no'} | {', '.join(e['converged_dishonest_v4']) or '-'} | "
                 f"{'n/a' if e['x396_not_run'] else len(e['converged_dishonest_x396'])} | {', '.join(e['pairs_failing']) or '-'} | {e['x396_scored']} / {e['x396_undecided']} | "
                 f"{e['unconverged']['v4']} / {e['unconverged']['pairs']} / {u396} | {e['regions']} | {e['wall_s']:.0f} | "
                 f"{e['regions_items12']} | {e['wall_s_items12']:.0f} | "
                 f"{e['windows_stopped']} | {e['leaves_falling_at_stop']} | {len(e['missing'])} |")
    L += ["", "## 1. Honesty on V4's 25: |v - ref| / abserr (* converged and dishonest; u unconverged)", "",
          "| call | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for sid in sets["v4"]:
        cells = []
        for v in V:
            h = hon[v][sid]
            if h is None:
                cells.append("missing")
            else:
                cells.append(f"{h['ratio']:.3g}" + ("*" if h["converged_dishonest"] else "") + ("" if h["converged"] else " u"))
        L.append(f"| {sid}{' (failure)' if sid in FAIL else ''} | " + " | ".join(cells) + " |")
    L += ["", "## 2. The pairs, scored on r_g = |v_7 - v_8| / e_7, the cross-check's; in brackets the prompt's r = |v_7 - v_8| / (e_7 + e_8), for the record (u: a member unconverged; * converged and r_g > 1)", "",
          "| pair | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for p in ("p14501", "p35006"):
        cells = []
        for v in V:
            e = pr[v][p]
            cells.append("missing" if e is None else f"r_g {e['r_g']:.3g} (r {e['r']:.3g})" + ("" if all(e["converged"]) else " u") + ("" if e["ok"] else "*"))
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "## 3. The 396 against (b): undecided by construction ((b) not run on the 396, the user's ruling at stop 1; "
          "(b2) converged on none of the 29 window calls of items 1-2)", "",
          "| variant | scored | undecided | honest | converged & dishonest | unconverged & dishonest | max ratio |", "|---|---|---|---|---|---|---|"]
    for v in V:
        sc = [x[v][s] for s in sets["x396"] if x[v][s]["scored"]]
        L.append(f"| {v} | {len(sc)} | {len(sets['x396']) - len(sc)} | {sum(1 for e in sc if e['ratio'] <= 1)} | "
                 f"{sum(1 for e in sc if e['converged_dishonest'])} | {sum(1 for e in sc if e['ratio'] > 1 and not e['converged'])} | "
                 f"{max((e['ratio'] for e in sc), default=float('nan')):.3g} |")
    L += ["", "## Per call and variant", "",
          "value, abserr, converged, regions, max depth, windows stopped, leaves falling at the stop, wall (s)", ""]
    for st in ("v4", "pairs", "nowindow"):
        L += [f"### {st}", "", "| call | variant | value | abserr | conv | regions | depth | stopped | falling | wall s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sid in sets[st]:
            for v in V:
                r = per[v][sid]
                if r is None:
                    L.append(f"| {sid} | {v} | missing ({status.get((sid, v), 'not run')}) | | | | | | | |")
                    continue
                L.append(f"| {sid} | {v} | {float.fromhex(r['value']):.15e} | {float.fromhex(r['abserr']):.3e} | {r['converged']} | "
                         f"{r['regions']} | {r['max_depth']} | {r['windows_stopped']} | {r['leaves_falling_at_stop']} | {r['wall_s']:.2f} |")
        L.append("")
    L += ["The 396's per-call rows are in the JSON (``per_call``)."]
    Path(a.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print("wrote", a.out_json, sha256(a.out_json), a.out_md, sha256(a.out_md))


if __name__ == "__main__":
    sys.exit(main())
