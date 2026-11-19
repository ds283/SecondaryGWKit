"""levin-phase-floor prompt 07b Part A, section 2.5: the four tables, fixed before the run, assembled. New in 07b.

Reads, each by SHA-256 in the header: out/tables/scores_final.json (score_07b.py), out/mech/mech_all.json (mech_07b.py),
out/m3/theta_prime_check.json (m3_theta_07b.py), out/w2/w2_rescore.json (w2_rescore_07b.py), state/selection_07b.json,
out/leafrefs/attr_*.json with the call references (closure), and 07a Part A's attr_tables_final.json in the tree (V4's
leaf references against 07a's).

Writes out/tables/tables_07b.md (tables 1, 2 and 4, and the closure checks), out/tables/table3_variants.md (table 3, every
call and variant) and out/tables/tables_07b.json. Usage: PYTHONPATH=<tree>:<scripts> python -B tables_07b.py
"""

import hashlib
import json
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
SCORES = ROOT / "out/tables/scores_final.json"
MECH = ROOT / "out/mech/mech_all.json"
M3 = ROOT / "out/m3/theta_prime_check.json"
W2 = ROOT / "out/w2/w2_rescore.json"
SEL = ROOT / "state/selection_07b.json"
ATTR07A = TREE / "docs/levin-phase-floor/07a/measurement/out/attr/attr_tables_final.json"
W2R = TREE / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
I3R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/census/out/refs/scores.json")
VARIANTS = ["tree", "b4", "b8", "c", "d", "e", "f"]
FIVE = ["v08", "v09", "v12", "v17", "v22"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def g(v, f=".3g"):
    return "-" if v is None else format(v, f)


def main():
    sc = json.loads(SCORES.read_text())
    T, pairs, r1, qual = sc["table"], sc["pairs"], sc["r1"], sc["qualification"]
    mech = json.loads(MECH.read_text())["rows"]
    m3 = {(x["sid"], x["variant"], x["leaf"]): x for x in json.loads(M3.read_text())["results"]}
    w2 = json.loads(W2.read_text())["rows"]
    sel = {e["sid"]: e for e in json.loads(SEL.read_text())["calls"]}
    md, out = [], {}

    def mark(r):
        if r is None or "ratio" not in r:
            return "-"
        m = "n" if r["spread_over_abserr"] > 0.01 else ("*" if (r["ratio"] > 1 and r["converged"]) else ("" if r["converged"] else "u"))
        return f"{r['ratio']:.3g}{m}"

    # ---------------------------------------------------------------- table 1
    md += ["# levin-phase-floor 07b Part A: section 2.5's tables", "",
           "Ratios are |v - ref|/abserr; `*` converged and dishonest; `u` unconverged; `n` not scored (the reference's "
           "stated error above min(abserr)/100). (a) is the tree, 07a's code.", "",
           "## Table 1. Honesty at 07a's code", "", "### V4's 25 (07's readings; z12's 06g's)", "",
           "| call | 07a ratio | converged | regions | spread/abserr |", "|---|---|---|---|---|"]
    t1 = {"v4": {}, "r1": {}, "r2": {}, "r3": {}, "w2": []}
    for sid in [f"v{i:02d}" for i in range(24)] + ["cal_z12"]:
        r = T[sid]["tree"]
        t1["v4"][sid] = r
        md.append(f"| {sid}{' (five)' if sid in FIVE else ''} | {mark(r)} | {r['converged']} | {r['regions']} | {r['spread_over_abserr']:.2g} |")
    nb = sum(1 for r in t1["v4"].values() if r["converged"] and r["ratio"] > 1)
    md += ["", f"Converged and dishonest: {nb} of 25 ({', '.join(s for s, r in t1['v4'].items() if r['converged'] and r['ratio'] > 1)}).", ""]
    md += ["### R1, class (c): 30 of 07's 396 cross-check rows, each at both rtols (one reference per row)", "",
           "| row | band | k | 07a 1e-8 | 07a 1e-7 | 07 1e-8 | 07 1e-7 | 06m 1e-8 | 06m 1e-7 | nearer (1e-8, 1e-7) |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    def fm(v, c):
        return "-" if v is None else f"{v:.3g}{'' if c else 'u'}"

    n_bad_r1 = 0
    for sid, x in sorted(r1.items()):
        a, b = x["by_rtol"]["1e-8"], x["by_rtol"]["1e-7"]
        r8, r7 = T[f"{sid}-8"]["tree"], T[f"{sid}-7"]["tree"]
        n_bad_r1 += (r8["converged"] and r8["ratio"] > 1) or (r7["converged"] and r7["ratio"] > 1)
        md.append(f"| {x['row']} | {x['band']} | {x['k_key']} | {mark(r8)} | {mark(r7)} | {fm(a['ratio_07'], a['converged_07'])} | "
                  f"{fm(b['ratio_07'], b['converged_07'])} | {fm(a['ratio_06m'], a['converged_06m'])} | "
                  f"{fm(b['ratio_06m'], b['converged_06m'])} | {a['nearer']}, {b['nearer']} |")
        t1["r1"][sid] = {"1e-8": r8, "1e-7": r7, "xcheck": x}
    near07 = sum(1 for x in r1.values() for t in ("1e-8", "1e-7") if x["by_rtol"][t]["nearer"] == "07")
    md += ["", f"Rows converged and dishonest at 07a's code (either rtol): {n_bad_r1} of {len(r1)}. 07 is nearer the reference "
           f"than 06m in {near07} of {2 * len(r1)} (row, rtol).", ""]
    md += ["### R2: 30 calls drawn uniformly from every Levin call of 07a's census (each scored at its own rtol and, `~o`, "
           "the other)", "", "| call | kind | band | k | rtol | class | 07a own | 07a other | 07 own |", "|---|---|---|---|---|---|---|---|---|"]
    for i in range(30):
        sid = f"R2-{i:02d}"
        e, r, ro = sel[sid], T[sid]["tree"], T.get(f"{sid}~o", {}).get("tree")
        st = e["stratum"]
        md.append(f"| {sid} | {st['kind']} ({st['label']}) | {st['band']} | {st['k_key']} | {st['rtol']} | {st['class']} | {mark(r)} | "
                  f"{mark(ro)} | {g(r.get('ratio_07'))} |")
        t1["r2"][sid] = {"own": r, "other": ro, "stratum": st}
    import collections
    for key in ("kind", "band", "rtol", "class"):
        cnt = collections.Counter(sel[f"R2-{i:02d}"]["stratum"][key] for i in range(30))
        md.append(f"- strata by {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    worst = max((T[s]["tree"]["ratio"], s) for s in T if s.startswith("R2-") and "ratio" in T[s]["tree"])
    md += ["", f"No R2 call is converged and dishonest at 07a's code; the largest ratio is {worst[0]:.3g} ({worst[1]}).", ""]
    md += ["### R3, targeted (each at 07a's code and at 07's)", "",
           "| call | set | unit, call | 07a own | 07a other | 07 own (converged) |", "|---|---|---|---|---|---|"]
    for sid in sorted(s for s in sel if sel[s]["set"].startswith("R3") and sel[s]["set"] != "R3-pair"):
        e, r, ro = sel[sid], T[sid]["tree"], T.get(f"{sid}~o", {}).get("tree")
        md.append(f"| {sid} | {e['set']} | {e['unit']} #{e['index']} | {mark(r)} | {mark(ro)} | {g(r.get('ratio_07'))} "
                  f"({e['census07_converged']}) |")
        t1["r3"][sid] = {"own": r, "other": ro}
    md += ["", "| pair | 07a 1e-7 | 07a 1e-8 | r_g at 07a |", "|---|---|---|---|"]
    for tag in ("14501", "35006"):
        md.append(f"| 3-{tag} #1 | {mark(T[f'p{tag}_7']['tree'])} | {mark(T[f'p{tag}_8']['tree'])} | {pairs[tag]['tree']['r_g']:.3g} |")
    md += ["", "### 06b's 24 W2 calls against S_10 (06b's rule): 07a's census beside 06b's", "",
           "| unit, call | band | 07a ratio | 07a converged | 06b ratio |", "|---|---|---|---|---|"]
    for r in sorted(w2, key=lambda r: (r["unit_h"], r["index"])):
        md.append(f"| {r['unit_h']} #{r['index']} | {r['band']} | {r['ratio_h']:.3g}{'*' if r['dishonest_h'] and r['converged_h'] else ''} | "
                  f"{r['converged_h']} | {r['ratio_b']:.3g} |")
        t1["w2"].append({k: r[k] for k in ("unit_h", "index", "ratio_h", "converged_h", "ratio_b")})
    md += ["", f"Dishonest at 07a's code: {sum(1 for r in w2 if r['dishonest_h'])} of 24, all converged; at 06b's: "
           f"{sum(1 for r in w2 if r['ratio_b'] > 1)}.", ""]
    out["table1"] = t1

    # ---------------------------------------------------------------- table 2
    md += ["## Table 2. The mechanism, per leaf (section 2.2)", "",
           "Columns: err(X)/e' is X's true error, signed, against the leaf's reference over its declared e'. M1: the spread "
           "|max - min| of err(S1), err(Q), err(S3) over |err(Q)| and whether they share sign; the piece carrying most of "
           "err(Q) (its own solve's error as a share of err(Q); eighths where solved, else quarters). M2: the observed falls "
           "beside r. M3: the change of Q with theta' x (1 + 2^-10) and with theta_mod_2pi moved by the phase's ulp at the end, "
           "each over |err(Q)|; delta_int is theta' against the captured phase's own derivative (m3_theta_07b.py, the "
           "integrated test, largest over the leaf's segments), delta* the uniform mismatch that would account for err(Q). "
           "The end: w a window edge; else the end the grid resolves worst. rate ratio: |theta'| across the grid's first "
           "spacing there; spacing/l with l = |theta'/theta''|; eps = theta''/theta'^2.", ""]
    hdr = ("| call | var | leaf | depth | span | clause | r | e' | V_P | S1 | Q | S3 | M1 spread | same sign | carrying piece (share) | "
           "S1/V_P | Q/S1 | S3/Q | M3 theta' | M3 theta mod | delta_int / delta* | end | rate ratio | spacing/l | eps | ref acc/e' |")
    sep = "|" + "---|" * 26
    rows_by = {}
    for r in mech:
        rows_by.setdefault(r["sid"], []).append(r)

    def leafrow(r):
        eo = r["err_over_eprime"]
        pe = r["pieces_err"]
        keys = [k for k in pe if len(k) == 3] or [k for k in pe if len(k) == 2]
        cp = max(keys, key=lambda k: abs(pe[k]["share_of_errQ"] or 0.0)) if keys else None
        side = "lo" if r["edge_lo"] else ("hi" if r["edge_hi"] else max(("lo", "hi"), key=lambda k: r["edge"][k]["spacing_over_l"] or 0.0))
        ed = r["edge"].get(side) or {}
        M = r.get("M3") or {}
        tp = M.get("theta_prime_+", {}).get("over_errQ")
        tm = M.get("theta_mod_2pi_edge_+", {}).get("over_errQ")
        mm = m3.get((r["sid"], r["variant"], r["i"]))
        dd = f"{mm['delta_int_max']:.1e} / {g(mm['delta_star'], '.1e')}" if mm else "-"
        fa = r["M2_fall"]
        return (f"| {r['sid']} | {r['variant']} | {r['i']} | {r['depth']} | [{r['lo']:.6g}, {r['hi']:.6g}] | {r['clause']} | {r['r']:.3g} | "
                f"{r['e_prime']:.2e} | {eo.get('V_P', float('nan')):+.3g} | {eo.get('S1', float('nan')):+.3g} | {eo.get('Q', float('nan')):+.3g} | "
                f"{eo.get('S3', float('nan')):+.3g} | {g(r.get('M1_spread_over_errQ'))} | {'yes' if r.get('M1_same_sign') else 'no'} | "
                f"{cp + ' (' + format(pe[cp]['share_of_errQ'], '.3g') + ')' if cp else '-'} | {g(fa.get('S1/V_P'))} | {g(fa.get('Q/S1'))} | "
                f"{g(fa.get('S3/Q'))} | {g(tp)} | {g(tm)} | {dd} | {side}{'(w)' if (r['edge_lo'] if side == 'lo' else r['edge_hi']) else ''} | "
                f"{g(ed.get('rate_ratio_first_spacing'))} | {g(ed.get('spacing_over_l'))} | {g(ed.get('eps_end'))} | {r['ref_acc_over_eprime']:.2g} |")

    blocks = [("The five failures (every Levin leaf touching a window, the carrying leaf first; v09's carrier has no window)",
               FIVE, ("tree",)),
              ("Item 3: 1-34599 #2 at 1e-7, 07's route (07's module), every Levin leaf", ["i3-34599-07"], ("07",)),
              ("Item 4: 06b's W2 calls dishonest at 07a's code, every Levin leaf", sorted(s for s in rows_by if s.startswith("W2-")), ("tree",)),
              ("The controls: the 20 honest V4 calls and z12, every Levin leaf touching a window",
               [s for s in [f"v{i:02d}" for i in range(24)] + ["cal_z12"] if s not in FIVE], ("tree",))]
    for title, sids, vs in blocks:
        md += [f"### {title}", "", hdr, sep]
        for sid in sids:
            rs = [r for r in rows_by.get(sid, []) if r["variant"] in vs]
            rs.sort(key=lambda r: -abs(r["err_over_eprime"].get("Q") or 0.0))
            md += [leafrow(r) for r in rs]
        md.append("")
    # M1 window scan: per call and window side, the touching leaf's true error at 2 pi, 4 pi, 8 pi
    md += ["### M1: the touching leaf's true error as the window widens (Phi_w = 2 pi, 4 pi, 8 pi; b4 and b8 replays)", "",
           "Per call and side of the window (`above`: the leaf starts at the window's upper edge). |err(Q)| absolute, and the "
           "fall per doubling; Phi_w^(-3/2) would give 2.83 per doubling.", "",
           "| call | side | |err| 2pi (err/e') | |err| 4pi (err/e') | |err| 8pi (err/e') | fall 2->4 | fall 4->8 |", "|---|---|---|---|---|---|---|"]
    scan = {}
    for sid in [f"v{i:02d}" for i in range(24)] + ["cal_z12"]:
        for side in ("above", "below"):
            row = {}
            for v in ("tree", "b4", "b8"):
                c = [r for r in rows_by.get(sid, []) if r["variant"] == v and (r["edge_lo"] if side == "above" else r["edge_hi"])]
                if c:
                    r = max(c, key=lambda r: abs(r["err"]["Q"]))
                    row[v] = (abs(r["err"]["Q"]), r["err_over_eprime"]["Q"])
            if len(row) == 3:
                f1, f2 = row["tree"][0] / row["b4"][0], row["b4"][0] / row["b8"][0]
                scan[f"{sid}:{side}"] = {"err": {k: v[0] for k, v in row.items()}, "err_over_eprime": {k: v[1] for k, v in row.items()},
                                         "fall": [f1, f2]}
                md.append(f"| {sid}{' (five)' if sid in FIVE else ''} | {side} | {row['tree'][0]:.2e} ({row['tree'][1]:+.3g}) | "
                          f"{row['b4'][0]:.2e} ({row['b4'][1]:+.3g}) | {row['b8'][0]:.2e} ({row['b8'][1]:+.3g}) | {f1:.3g} | {f2:.3g} |")
    md.append("")
    # the rate-range finding: trusted Levin leaves whose reference meets the 100x rule (ref acc <= e'/100) are scored;
    # the others are listed, and counted dishonest only where |err(Q)| exceeds ten times the reference's accuracy
    lev = [r for r in mech if r["variant"] in ("tree", "07") and r["type"] == "Levin" and r["trusted"]]
    rr, unsc = [], []
    for r in lev:
        a, b = abs(r["theta_prime_lo"]), abs(r["theta_prime_hi"])
        x = (max(a, b) / min(a, b), abs(r["err_over_eprime"].get("Q") or 0.0), r["sid"], r["i"], r["ref_acc_over_eprime"])
        (rr if r["ref_acc_over_eprime"] <= 0.01 else unsc).append(x)
    bad = [x for x in rr if x[1] > 1]
    md += [f"### The rate range across a leaf: |theta'| end to end, every trusted Levin leaf referenced (07a's code; item 3 "
           f"at 07's)", "",
           f"- scored (reference accuracy at most e'/100): {len(rr)} leaves; with |err(Q)| > e': {len(bad)}; the smallest "
           f"rate ratio among those {min(x[0] for x in bad):.3g}; the largest among the honest {max(x[0] for x in rr if x[1] <= 1):.3g}.",
           "- by rate-ratio band (scored leaves; dishonest): " + "; ".join(
               f"({lo:g}, {hi:g}]: {sum(1 for x in rr if lo < x[0] <= hi)}; {sum(1 for x in rr if lo < x[0] <= hi and x[1] > 1)}"
               for lo, hi in ((1, 10), (10, 30), (30, 100), (100, 1e3), (1e3, 1e12))),
           "- the dishonest: " + ", ".join(f"{x[2]} leaf {x[3]} (ratio {x[0]:.3g}, err/e' {x[1]:.3g})" for x in sorted(bad, key=lambda x: x[0])),
           f"- not scored (reference accuracy above e'/100): {len(unsc)} leaves, all of W2-2-34473-3's deep leaves"
           if all(x[2] == "W2-2-34473-3" for x in unsc) else f"- not scored: {len(unsc)} leaves",
           "  - of them with |err(Q)| > e' and > 10x the reference's accuracy: " + (", ".join(
               f"{x[2]} leaf {x[3]} (ratio {x[0]:.3g}, err/e' {x[1]:.3g}, ref acc/e' {x[4]:.2g})" for x in unsc if x[1] > 1 and x[1] > 10 * x[4]) or "none"),
           ""]
    out["table2"] = {"window_scan": scan, "rate_range": [list(x) for x in rr]}

    # ---------------------------------------------------------------- closure
    md += ["## Closure", "", "Item 3's and item 4's leaf references (every leaf of the call, resolution 2) summed against the "
           "call's reference; V4's leaf references against 07a Part A's on the same span.", "",
           "| call | sum of leaf refs - call ref | over abserr | call ref basis |", "|---|---|---|---|"]
    w2ref = {}
    for r in json.loads(W2R.read_text())["results"]:
        w2ref[f"W2-{'-'.join(r['unit'].split('-')[3:])}-{r['index']}"] = mpf(fx(r["S10"]))
    i3 = [mpf(r["ref_30"]) for r in json.loads(I3R.read_text())["calls"] if r["sid"] == "r1-34599-1e-7-2"][0]
    clos = {}
    for p in sorted((ROOT / "out/leafrefs").glob("attr_*.json")):
        d = json.loads(p.read_text())
        sid = d["sid"]
        s = mp.fsum(mpf(x["value_30"]) for x in d["readings"]["2"]["pieces"])
        ref = i3 if sid.startswith("i3") else w2ref[sid]
        ab = fx(T[sid if not sid.startswith("i3") else "i3-34599-h"]["tree"]["abserr"]) if not sid.startswith("i3") else None
        if sid.startswith("i3"):
            ab = fx(json.loads((ROOT / "out/replays/07/i3-34599-07.json").read_text())["replay"]["abserr"])
        clos[sid] = {"diff": float(s - ref), "over_abserr": float(abs(s - ref) / ab)}
        md.append(f"| {sid} | {float(s - ref):.3e} | {clos[sid]['over_abserr']:.2e} | {'07a census FCC r2' if sid.startswith('i3') else 'S10 (06b)'} |")
    a7 = json.loads(ATTR07A.read_text())["calls"]
    dv = []
    for r in mech:
        if r["variant"] != "tree" or r["sid"] not in a7:
            continue
        for L in a7[r["sid"]]["leaves"]:
            if fx(L["start"]) == r["lo"] and fx(L["end"]) == r["hi"]:
                dv.append((abs(float(mpf(r["ref"]) - mpf(L["ref_30"]))) / r["e_prime"], r["sid"], r["i"]))
    md += ["", f"V4's leaf references against 07a Part A's on the same span: {len(dv)} leaves, the largest difference "
           f"{max(dv)[0]:.2g} e' ({max(dv)[1]} leaf {max(dv)[2]}).", ""]
    out["closure"] = {"items34": clos, "v4_vs_07a": [list(x) for x in dv]}

    # ---------------------------------------------------------------- table 4
    md += ["## Table 4. Qualification, and cost against (a)", "",
           "A variant qualifies if no scored call is converged and dishonest under it and each pair has r_g <= 1 or an "
           "unconverged member. Costs over all 187 replays (V4 25, R1 60, R2 60, R3 22, W2 8, item 3 1, pairs 4, no-window 5).", "",
           "| variant | qualifies | converged & dishonest (scored) | pairs failing | unconverged | regions (x a) | wall s (x a) | "
           "bit-identical to (a) | leaves it changes |", "|---|---|---|---|---|---|---|---|---|"]
    changes = {"tree": "-", "b4": "every call with a window (Phi_w 2 pi -> 4 pi)", "b8": "every call with a window (Phi_w -> 8 pi)",
               "c": "trusted Levin leaves touching a window (eighths' check)",
               "d": "trusted Levin leaves touching a window (CC check over 4 pi beside it)",
               "e": "trusted Levin leaves touching a window (rate ratio across the first spacing > 2)",
               "f": "every trusted Levin leaf whose |theta'| ranges > 10 over its nodes (not tied to windows)"}
    q0 = qual["tree"]
    for v in VARIANTS:
        q = qual[v]
        same = sum(1 for r in T.values() if v in r and r[v]["value"] == r["tree"]["value"] and r[v]["abserr"] == r["tree"]["abserr"])
        md.append(f"| {v} | {'**yes**' if q['qualifies'] else 'no'} | {len(q['converged_dishonest'])} | {', '.join(q['pairs_failing']) or '-'} | "
                  f"{q['unconverged']} | {q['regions']} ({q['regions'] / q0['regions']:.2f}) | {q['wall_s']:.0f} ({q['wall_s'] / q0['wall_s']:.2f}) | "
                  f"{same} | {changes[v]} |")
    md += ["", "No-window calls n0-n4: (b4), (b8), (c), (d), (e) bit-identical to (a) on all five; (f) on n0 only.", ""]
    nwin = {v: [n for n in ("n0", "n1", "n2", "n3", "n4") if T[n][v]["value"] == T[n]["tree"]["value"] and T[n][v]["abserr"] == T[n]["tree"]["abserr"]]
            for v in VARIANTS}
    out["table4"] = {"qualification": qual, "no_window_identical": nwin}
    import collections as _c
    md += ["Unconverged calls by set under each variant:", ""]
    for v in VARIANTS:
        cnt = _c.Counter(r[v]["set"].split("~")[0] for r in T.values() if v in r and not r[v]["converged"])
        md.append(f"- {v}: " + (", ".join(f"{k} {n}" for k, n in sorted(cnt.items())) or "none"))
    md.append("")

    # ---------------------------------------------------------------- table 3 (its own file)
    t3 = ["# Table 3. The variants: every call and variant", "",
          "value and abserr as returned (hex in tables_07b.json via scores_final.json); ratio |v - ref|/abserr where referenced "
          "(`*` converged and dishonest, `n` not scored).", ""]
    for sid in sorted(T):
        t3 += [f"### {sid} ({next(iter(T[sid].values()))['set']})", "",
               "| variant | value | abserr | converged | regions | max depth | wall s | ratio |", "|---|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            r = T[sid].get(v)
            if r:
                t3.append(f"| {v} | {fx(r['value']):.15e} | {fx(r['abserr']):.4e} | {r['converged']} | {r['regions']} | {r['max_depth']} | "
                          f"{r['wall_s']:.2f} | {mark(r)} |")
        t3.append("")
    t3 += ["## The pairs: r_g = |v_7 - v_8| / e_7 (and r = |v_7 - v_8| / (e_7 + e_8))", "", "| pair | " + " | ".join(VARIANTS) + " |",
           "|---|" + "---|" * len(VARIANTS)]
    for tag, pv in pairs.items():
        t3.append(f"| 3-{tag} #1 | " + " | ".join(f"{pv[v]['r_g']:.3g} ({pv[v]['r']:.3g}){' u' if pv[v]['unconverged_member'] else ''}"
                                                  for v in VARIANTS) + " |")
    ins = [SCORES, MECH, M3, W2, SEL, ATTR07A, W2R, I3R] + sorted((ROOT / "out/leafrefs").glob("attr_*.json"))
    header = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "inputs": {str(p): sha256(p) for p in ins}}
    foot = ["", "---", f"script {header['script']} sha256 {header['script_sha256']}"] + [f"input {k} sha256 {v}" for k, v in header["inputs"].items()]
    p_md, p_t3, p_js = ROOT / "out/tables/tables_07b.md", ROOT / "out/tables/table3_variants.md", ROOT / "out/tables/tables_07b.json"
    p_md.write_text("\n".join(md + foot) + "\n")
    p_t3.write_text("\n".join(t3 + foot) + "\n")
    p_js.write_text(json.dumps({"header": header, **out}, indent=1))
    for p in (p_md, p_t3, p_js):
        print("wrote", p, sha256(p))


if __name__ == "__main__":
    main()
