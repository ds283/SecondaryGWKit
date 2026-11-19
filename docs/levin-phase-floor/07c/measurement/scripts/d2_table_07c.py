"""levin-phase-floor prompt 07c Part A, section 2.6 table 2 (D2) and table 1's third part (the chase's Clenshaw-Curtis leaves'
true errors). New in 07c. Reads leafcc_07c.py's outputs.

Usage: python -B d2_table_07c.py d2 LEAFCC_DIR OUT_JSON OUT_MD
       python -B d2_table_07c.py d1 LEAFCC_DIR D1_JSON OUT_JSON OUT_MD

**D2 (table 2):** per stratum (state/selection_d2.json) and candidate (leafcc's ``candidates``): over the leaves whose true
error is resolved (|CC_{2N-1} - ref| above the reference's stated error), the ratio estimate / true error, its minimum,
median and maximum, and how many leaves have it below 1; over every leaf, how often the estimate is at or below the leaf's
floor (``phase_err``); and how often the true error exceeds max(estimate, floor) (the declaration the leaf would carry).

**D1 (table 1, third part):** for the chase's bisected Clenshaw-Curtis leaves (D1_JSON ``below``, route cc_gate), joined
by span to leafcc's records: CC_{2N-1}'s true error against |CC_{2N-1} - CC_N| and against the floor, by cause; and how
many of the residual-above-floor bisections (07b's 4,009-type) would pass the residual test (residual <= floor) with the
true error, or with each candidate, in place of the pair.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def fx(h):
    return None if h is None else float.fromhex(h)


def load(dirp):
    out = {}
    for p in sorted(Path(dirp).glob("*.json")):
        d = json.loads(p.read_text())
        sid = p.stem
        for l in d["leaves"]:
            out[(sid, l["start"], l["end"])] = l
    return out


def stats(xs):
    if not xs:
        return None
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs), "n_below_1": sum(1 for x in xs if x < 1)}


def main():
    mode = sys.argv[1]
    leaves = load(sys.argv[2])
    if mode == "d2":
        out_j, out_m = Path(sys.argv[3]), Path(sys.argv[4])
        by = {}
        for (sid, a, b), l in leaves.items():
            st = l["id"].split("|")[0] + "|" + l["id"].split("|")[1]
            by.setdefault(st, []).append(l)
        res = {}
        cands = None
        for st, ls in sorted(by.items()):
            ok = [l for l in ls if "error" not in l]
            cands = cands or list(ok[0]["candidates"])
            r = {"n": len(ls), "n_reproduce": sum(1 for l in ok if l["reproduces"]), "n_resolved": sum(1 for l in ok if l["resolved"]),
                 "V_over_6pi": [min(l["V_over_6pi"] for l in ok), max(l["V_over_6pi"] for l in ok)],
                 "true_err_over_floor": stats([fx(l["true_err"]) / fx(l["phase_err"]) for l in ok]), "by_candidate": {}}
            for c in cands:
                ratios = [fx(l["candidates"][c]) / fx(l["true_err"]) for l in ok if l["resolved"] and fx(l["true_err"]) > 0 and l["candidates"][c] is not None]
                r["by_candidate"][c] = {"ratio_est_over_true": stats(ratios),
                                        "n_at_or_below_floor": sum(1 for l in ok if l["candidates"][c] is not None and fx(l["candidates"][c]) <= fx(l["phase_err"])),
                                        "n_true_above_max_est_floor": sum(1 for l in ok if l["candidates"][c] is not None and fx(l["true_err"]) > max(fx(l["candidates"][c]), fx(l["phase_err"])))}
            res[st] = r
        out_j.write_text(json.dumps({"strata": res}, indent=1))
        md = ["| stratum | n | resolved | V/6pi | candidate | ratio min | median | max | below 1 | est <= floor | true > max(est, floor) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for st, r in res.items():
            for c, x in r["by_candidate"].items():
                s = x["ratio_est_over_true"]
                md.append(f"| {st} | {r['n']} | {r['n_resolved']} | {r['V_over_6pi'][0]:.3g}-{r['V_over_6pi'][1]:.3g} | {c} | "
                          + (f"{s['min']:.3g} | {s['median']:.3g} | {s['max']:.3g} | {s['n_below_1']}" if s else "- | - | - | -")
                          + f" | {x['n_at_or_below_floor']} | {x['n_true_above_max_est_floor']} |")
        out_m.write_text("\n".join(md) + "\n")
        print("\n".join(md))
    elif mode == "d1":
        d1 = json.loads(Path(sys.argv[3]).read_text())
        out_j, out_m = Path(sys.argv[4]), Path(sys.argv[5])
        rows = []
        for x in d1["below"]:
            if x["route"] != "cc_gate":
                continue
            l = leaves.get((x["sid"].replace(":", "_"), x["start"], x["end"]))
            if l is None or "error" in l:
                rows.append({"sid": x["sid"], "cause": x["cause"], "missing": True})
                continue
            fl = fx(l["phase_err"])
            te = fx(l["true_err"])
            rows.append({"sid": x["sid"], "cause": x["cause"], "why": x["why"], "reproduces": l["reproduces"], "resolved": l["resolved"],
                         "pair": fx(l["candidates"]["pair"]), "true_err": te, "ref_err": fx(l["ref_err"]), "floor": fl,
                         "residual_recorded": x["residual"], "pair_equals_recorded": fx(l["candidates"]["pair"]) == x["residual"],
                         "true_le_floor": te <= fl, "true_or_referr_le_floor": max(te, fx(l["ref_err"])) <= fl,
                         "cands_le_floor": {c: (v is not None and fx(v) <= fl) for c, v in l["candidates"].items()},
                         "true_over_pair": te / fx(l["candidates"]["pair"]) if fx(l["candidates"]["pair"]) else None,
                         "true_over_floor": te / fl if fl else None})
        ok = [r for r in rows if not r.get("missing")]
        summ = {}
        for cause in sorted({r["cause"] for r in ok}):
            rs = [r for r in ok if r["cause"] == cause]
            summ[cause] = {"n": len(rs), "reproduce": sum(1 for r in rs if r["reproduces"]), "resolved": sum(1 for r in rs if r["resolved"]),
                           "pair_equals_recorded_residual": sum(1 for r in rs if r["pair_equals_recorded"]),
                           "true_le_floor": sum(1 for r in rs if r["true_le_floor"]),
                           "true_or_referr_le_floor": sum(1 for r in rs if r["true_or_referr_le_floor"]),
                           "cand_le_floor": dict(Counter(c for r in rs for c, v in r["cands_le_floor"].items() if v)),
                           "true_over_pair": stats([r["true_over_pair"] for r in rs if r["true_over_pair"] is not None]),
                           "true_over_floor": stats([r["true_over_floor"] for r in rs if r["true_over_floor"] is not None])}
        out_j.write_text(json.dumps({"summary": summ, "n_missing": sum(1 for r in rows if r.get("missing")), "rows": rows}))
        md = ["| cause | n | reproduce | resolved | true <= floor | max(true, ref err) <= floor | candidate <= floor | true/pair min-med-max | true/floor min-med-max |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c, s in summ.items():
            tp, tf = s["true_over_pair"], s["true_over_floor"]
            md.append(f"| {c} | {s['n']} | {s['reproduce']} | {s['resolved']} | {s['true_le_floor']} | {s['true_or_referr_le_floor']} | "
                      f"{', '.join(f'{k} {v}' for k, v in sorted(s['cand_le_floor'].items()))} | "
                      f"{tp['min']:.2g}-{tp['median']:.2g}-{tp['max']:.2g} | {tf['min']:.2g}-{tf['median']:.2g}-{tf['max']:.2g} |")
        out_m.write_text("\n".join(md) + "\n")
        print("\n".join(md))
        print("missing:", sum(1 for r in rows if r.get("missing")))


if __name__ == "__main__":
    main()
