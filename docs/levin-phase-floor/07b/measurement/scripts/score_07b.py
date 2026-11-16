"""levin-phase-floor prompt 07b Part A, section 2.5 tables 1, 3 and 4: honesty and the variants. New in 07b.

**References.**
- V4's 25: 07's readings (docs/levin-phase-floor/07/census/out/v4/v4_scores.json in the tree: ``calls`` and ``z``),
  the reference FCC r2 as 07 took it.
- section 2.4's calls: this Part's readings (out/refs/readings/: <ref sid>-fccsplit.json, leafref_07b.py's FCC r1 and
  r2 on split edges, for Route B; <ref sid>-<impl>-r<res>.json, one_reading.py's, for Route A's FCC and every QAWO), the
  reference FCC r2, the stated error the readings' spread (largest minus smallest of those taken).
- 06b's W2: S_10 (docs/levin-phase-floor/06b/census/out/w2_reference.json), stated error |S_9 - S_10|.
- item 3 (``i3-*``): 07a's census reference of 1-34599 (07a census out/refs/scores.json).
A call is scored if its reference's stated error is at most min(abserr over the codes compared)/100 (07a's W4 rule);
otherwise it is reported, not scored. |v - ref|/abserr above 1 is dishonest; ``*`` marks converged and dishonest.

**Calls scored.** Every replay of out/replays/<variant>/ (tree = (a)), each against its integrand's reference: the
sid's own (V4, W2, i3) or its selection's (R1 ``-7``/``-8``, R2/R3 and ``~o``, the pairs ``p*_7``/``_8``). For R3's W7,
aux and own calls, also 07's census value and abserr (from the selection) against the same reference.

**The pairs:** r_g = |v_7 - v_8| / e_7 per variant (and r = |v_7 - v_8|/(e_7 + e_8)).

**Qualification (table 4):** a variant qualifies if no scored call is converged and dishonest under it, and each pair has
r_g <= 1 or a member unconverged; its cost against (a): calls unconverged, regions and wall time.

Writes OUT_JSON and OUT_MD. Usage: PYTHONPATH=<tree>:<scripts> python -B score_07b.py OUT_JSON OUT_MD VARIANT [VARIANT ...]
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
V4S = TREE / "docs/levin-phase-floor/07/census/out/v4/v4_scores.json"
W2R = TREE / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
XCHECK = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/out/xcheck/xcheck.json")
I3R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/census/out/refs/scores.json")
SEL = ROOT / "state" / "selection_07b.json"
MAN = ROOT / "state" / "calls_07b.json"
READ = ROOT / "out" / "refs" / "readings"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def refs():
    out = {}
    v4 = json.loads(V4S.read_text())
    for c in v4["calls"]:
        rd = [mpf(v) for v in c["readings_30"].values() if v is not None]
        out[c["sid"]] = {"ref": mpf(c["readings_30"]["fcc_r2"]), "spread": float(max(rd) - min(rd)), "basis": c.get("basis", "07")}
    for c in v4["z"]:
        if c["sid"] == "z12":  # cal_z12's integrand (07's calibration capture), 06g's readings
            out["cal_z12"] = {"ref": mpf(fx(c["ref"])), "spread": c["spread"], "basis": c.get("basis", "06g")}
    for r in json.loads(W2R.read_text())["results"]:
        key = f"W2-{'-'.join(r['unit'].split('-')[3:])}-{r['index']}"
        out[key] = {"ref": mpf(fx(r["S10"])), "spread": abs(fx(r["S9"]) - fx(r["S10"])), "basis": "S10 (06b)"}
    for r in json.loads(I3R.read_text())["calls"]:
        if r["sid"] == "r1-34599-1e-7-2":
            out["i3"] = {"ref": mpf(r["ref_30"]), "spread": r["stated_error_spread"], "basis": "07a census FCC r2"}
    sids = {p.name.split("-fccsplit")[0] for p in READ.glob("ref-*-fccsplit.json")} | {
        p.name[: -len("-fcc-r2.json")] for p in READ.glob("ref-*-fcc-r2.json")}
    for sid in sorted(sids):
        rd = {}
        fs = READ / f"{sid}-fccsplit.json"
        if fs.exists():  # leafref_07b.py on the whole span: one piece per resolution
            d = json.loads(fs.read_text())
            for res, r in d["readings"].items():
                rd[f"fccsplit-r{res}"] = mp.fsum(mpf(x["value_30"]) for x in r["pieces"])
        for impl in ("fcc", "qawo"):
            for res in (1, 2):
                q = READ / f"{sid}-{impl}-r{res}.json"
                if q.exists():
                    d = json.loads(q.read_text())
                    if d.get("value_30") is not None:
                        rd[f"{impl}-r{res}"] = mpf(d["value_30"])
        key = "fccsplit-r2" if "fccsplit-r2" in rd else ("fcc-r2" if "fcc-r2" in rd else None)
        if key is None:
            continue
        vals = list(rd.values())
        out[sid] = {"ref": rd[key], "spread": float(max(vals) - min(vals)), "basis": "+".join(sorted(rd)), "n": len(vals)}
    return out


def ref_key(sid, man, sel_by):
    if sid.startswith(("v", "cal_")):
        return sid
    if sid.startswith("W2-"):
        return sid.split("~")[0]
    if sid.startswith("i3-"):
        return "i3"
    if sid.startswith("p"):
        return f"ref-R3p-3-{sid[1:].split('_')[0]}"
    if sid.startswith("R1-"):
        return "ref-" + sid.rsplit("-", 1)[0]
    return "ref-" + sid.split("~")[0]


def main():
    out_j, out_m, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    R = refs()
    man = json.loads(MAN.read_text())["calls"]
    sel = {e["sid"]: e for e in json.loads(SEL.read_text())["calls"]}
    table = {}
    for v in variants:
        for p in sorted((ROOT / "out" / "replays" / v).glob("*.json")):
            if p.name.endswith(".instrument.json"):
                continue
            d = json.loads(p.read_text())
            sid = d["sid"]
            rk = ref_key(sid, man, sel)
            v_, e_ = fx(d["replay"]["value"]), fx(d["replay"]["abserr"])
            row = {"sid": sid, "set": d["set"], "variant": v, "value": d["replay"]["value"], "abserr": d["replay"]["abserr"],
                   "converged": d["replay"]["converged"], "regions": d["replay"]["num_regions"], "max_depth": d["replay"]["max_depth"],
                   "wall_s": d["wall_s"], "reproduces_census": d["reproduces_census"], "ref_key": rk}
            if rk in R:
                rr = R[rk]
                row["ratio"] = float(abs(mpf(v_) - rr["ref"]) / e_)
                row["spread_over_abserr"] = rr["spread"] / e_
                row["ref_basis"] = rr["basis"]
            table.setdefault(sid, {})[v] = row
    # 07's values for R3 (W7, aux, own) and R1
    for sid, rows in table.items():
        base = sid.split("~")[0]
        e = sel.get(base)
        if e and e["set"] in ("R3-W7", "R3-aux", "R3-own", "R1", "R2") and "~" not in sid and ref_key(sid, man, sel) in R:
            rr = R[ref_key(sid, man, sel)]
            v7, e7 = fx(e["census07_value"]), fx(e["census07_abserr"])
            for row in rows.values():
                row["ratio_07"] = float(abs(mpf(v7) - rr["ref"]) / e7)
                row["converged_07"] = e["census07_converged"]
                row["spread_scored"] = rr["spread"] <= min(fx(row["abserr"]), e7) / 100
    # R1: 07 against 06m at 1e-8 and 1e-7 (06m's values from 07's cross-check, var/levin-phase-floor-07/census/out/xcheck/
    # xcheck.json ``pairs``; 07's from the selection), each against the row's reference, and which is nearer
    xc = {(x["row"], x["kind"], x["index"]): x for x in json.loads(XCHECK.read_text())["pairs"]}
    r1 = {}
    for e in sel.values():
        if e["set"] != "R1" or "ref-" + e["sid"] not in R:
            continue
        row = "-".join(e["unit"].split("-")[3:])
        x = xc[(row, e["kind"], e["index"])]
        ref = R["ref-" + e["sid"]]["ref"]
        d = {}
        for rt, tag in (("8", "1e-8"), ("7", "1e-7")):
            v7, e7 = fx(x["values"][f"g{rt}"]), fx(x["abserrs"][f"g{rt}"])
            vm, em = fx(x["values"][f"m{rt}"]), fx(x["abserrs"][f"m{rt}"])
            a = table.get(f"{e['sid']}-{rt}", {}).get("tree")
            d[tag] = {"ratio_07": float(abs(mpf(v7) - ref) / e7), "ratio_06m": float(abs(mpf(vm) - ref) / em),
                      "err_07": float(abs(mpf(v7) - ref)), "err_06m": float(abs(mpf(vm) - ref)),
                      "nearer": "07" if abs(mpf(v7) - ref) < abs(mpf(vm) - ref) else "06m",
                      "ratio_07a": a.get("ratio") if a else None, "converged_07a": a.get("converged") if a else None,
                      "converged_07": x["converged"][f"g{rt}"], "converged_06m": x["converged"][f"m{rt}"],
                      "listed": x["listed"][tag]}
        r1[e["sid"]] = {"row": row, "band": e["stratum"]["band"], "k_key": e["stratum"]["k_key"], "by_rtol": d}
    # the pairs
    pairs = {}
    for v in variants:
        for tag in ("14501", "35006"):
            a, b = table.get(f"p{tag}_7", {}).get(v), table.get(f"p{tag}_8", {}).get(v)
            if a and b:
                v7, v8, e7, e8 = fx(a["value"]), fx(b["value"]), fx(a["abserr"]), fx(b["abserr"])
                pairs.setdefault(tag, {})[v] = {"r_g": abs(v7 - v8) / e7, "r": abs(v7 - v8) / (e7 + e8),
                                               "unconverged_member": not (a["converged"] and b["converged"])}
    # qualification
    qual = {}
    for v in variants:
        rows = [r[v] for r in table.values() if v in r]
        scored = [r for r in rows if "ratio" in r and r["spread_over_abserr"] <= 0.01]
        bad = sorted(r["sid"] for r in scored if r["converged"] and r["ratio"] > 1)
        pair_bad = sorted(t for t, pv in pairs.items() if v in pv and pv[v]["r_g"] > 1 and not pv[v]["unconverged_member"])
        qual[v] = {"n_calls": len(rows), "n_scored": len(scored), "converged_dishonest": bad, "pairs_failing": pair_bad,
                   "qualifies": not bad and not pair_bad, "unconverged": sum(1 for r in rows if not r["converged"]),
                   "regions": sum(r["regions"] for r in rows), "wall_s": sum(r["wall_s"] for r in rows),
                   "not_reproducing": sum(1 for r in rows if not r["reproduces_census"])}
    hdr = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "inputs": {str(p): sha256(p) for p in (V4S, W2R, SEL, MAN)}}
    out_j.write_text(json.dumps({"header": hdr, "refs": {k: {"ref": mp.nstr(v["ref"], 25), "spread": v["spread"], "basis": v["basis"]}
                                                          for k, v in R.items()},
                                 "table": table, "pairs": pairs, "r1": r1, "qualification": qual}, indent=1))
    md = ["## Qualification and cost", "", "| variant | qualifies | converged & dishonest | pairs failing | scored | unconverged | regions | wall s |",
          "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        q = qual[v]
        md.append(f"| {v} | {'yes' if q['qualifies'] else 'no'} | {', '.join(q['converged_dishonest']) or '-'} | "
                  f"{', '.join(q['pairs_failing']) or '-'} | {q['n_scored']} / {q['n_calls']} | {q['unconverged']} | {q['regions']} | {q['wall_s']:.0f} |")
    md += ["", "## Honesty per call (|v - ref|/abserr; * converged and dishonest; u unconverged; n not scored)", "",
           "| call | set | " + " | ".join(variants) + " | 07 |", "|---|---|" + "---|" * (len(variants) + 1)]
    for sid in sorted(table):
        rows = table[sid]
        cells = []
        for v in variants:
            r = rows.get(v)
            if not r or "ratio" not in r:
                cells.append("-")
                continue
            mark = "n" if r["spread_over_abserr"] > 0.01 else ("*" if (r["ratio"] > 1 and r["converged"]) else ("u" if not r["converged"] else ""))
            cells.append(f"{r['ratio']:.3g}{mark}")
        any_r = next(iter(rows.values()))
        c07 = f"{any_r['ratio_07']:.3g}{'' if any_r.get('converged_07', True) else 'u'}" if "ratio_07" in any_r else "-"
        md.append(f"| {sid} | {any_r['set']} | " + " | ".join(cells) + f" | {c07} |")
    md += ["", "## The pairs (r_g; u a member unconverged)", "", "| pair | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for t, pv in pairs.items():
        md.append(f"| {t} | " + " | ".join((f"{pv[v]['r_g']:.3g}{' u' if pv[v]['unconverged_member'] else ''}" if v in pv else "-") for v in variants) + " |")
    md += ["", "## R1: 07a, 07 and 06m against the reference (|v - ref|/abserr; nearer of 07 and 06m)", "",
           "| row | band | k | 07a 1e-8 | 07 1e-8 | 06m 1e-8 | nearer 1e-8 | 07a 1e-7 | 07 1e-7 | 06m 1e-7 | nearer 1e-7 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid, x in sorted(r1.items()):
        a, b = x["by_rtol"]["1e-8"], x["by_rtol"]["1e-7"]

        def fm(v, c):
            return "-" if v is None else f"{v:.3g}{'' if c else 'u'}"
        md.append(f"| {x['row']} | {x['band']} | {x['k_key']} | {fm(a['ratio_07a'], a['converged_07a'])} | {fm(a['ratio_07'], a['converged_07'])} | "
                  f"{fm(a['ratio_06m'], a['converged_06m'])} | {a['nearer']} | {fm(b['ratio_07a'], b['converged_07a'])} | "
                  f"{fm(b['ratio_07'], b['converged_07'])} | {fm(b['ratio_06m'], b['converged_06m'])} | {b['nearer']} |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md[:6 + len(variants)]))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
