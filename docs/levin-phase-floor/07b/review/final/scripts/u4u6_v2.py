"""Orchestrator's own re-score of U4 and U6 (levin-phase-floor 07b Part C, §C.2). Read only; prints and writes OUT_JSON.
Independent of the census agent's scripts; adapted from 07a's review/final/scripts/w4_rescore.py and refs_check.py.

usage: python -I -B u4u6_v2.py OUT_JSON  (v2 of u4u6.py: Part A's set "pairs" added to U4)

The census values (value, abserr, converged) are read from 07b's census checkpoint at my own index
(review/stop3/index07b.json, file SHA-256 asserted, each line's SHA-256 checked); 07a's census beside it (index07a.json).
U4's calls and references:
  (a) Part A (measurement/out/tables/scores_final.json): every table id whose set is v4 (v00-v23, cal_z12), R1, R2, R2~o,
      R3-*, item3; its census call from state/census_wanted_07b.json (unit, kind, index); its reference
      refs[ref_key].ref (a decimal string), its spread refs[ref_key].spread.
  (b) z00, z07, z16, z18, z21, z24, z28, z29 (07a's census's other eight V4 calls): 06g's scores_stage3/scores.json rows,
      06g's `unusable` left out, reference ref_30 (asserted equal to fcc_r2's value_30); spread max-min of the readings.
  (c) 07a's references for rows 1-34599 and 3-678, group indices 2 and 3, both rtols: the readings
      07a/census/out/refs/readings/r<row>-1e-7-<i>-<impl>-r<res>.json, accepted only if tool d37e633b..., input pickle
      SHA-256 as recorded, route B, value_30 rounds to value_hex; reference FCC r2; spread max-min.
  Rule (07's): honest iff |v - ref| <= abserr; an unconverged call passes. Agreement of readings reported two ways:
  spread <= min(abserr_07b, abserr_07a)/100 and spread <= abserr_07b/100. Holds iff every call is honest or unconverged.
U6: 06b's 24 (docs/levin-phase-floor/06b/census/out/w2_reference.json); unit b-1e-7-X-N -> row X-N @1e-7, same index;
  reference S10 (hex). Holds iff |v - S10|/abserr <= 1 or unconverged.
Exact Fractions throughout.
"""
import glob, hashlib, json, sys
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
WT = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5")
RB = M / "var/runs/levin-phase-floor-07b-census-20261117T010200"
RA = M / "var/runs/levin-phase-floor-07a-census-20261112T132744"
IX = M / "var/levin-phase-floor-07b/review/stop3"
MEAS = M / "var/levin-phase-floor-07b/measurement"
S06G = WT / "docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json"
W2REF = WT / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
RD7A = M / "var/levin-phase-floor-07a/census/out/refs/readings"
TOOL = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
ZS = ["z00", "z07", "z16", "z18", "z21", "z24", "z28", "z29"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
fx = lambda v: F(float.fromhex(v["hex"] if isinstance(v, dict) else v))

ib = json.load(open(IX / "index07b.json")); ia = json.load(open(IX / "index07a.json"))
assert ib["sha256"] == "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253"
assert ia["sha256"] == "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"


def line(run, ix, key):
    e = ix["units"][key]
    with open(run / "checkpoint.jsonl", "rb") as f:
        f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    return json.loads(raw)["data"]


def call(run, ix, key, i, kind=None):
    d = line(run, ix, key)
    assert d["status"] == "done"
    lists = {"group": d["calls"], "analytic": d["other_calls"], None: d["calls"] + d["other_calls"]}[kind]
    m = [c for c in lists if c["index"] == i]
    assert len(m) == 1, (key, i, kind)
    return d["unit"], m[0]


def score(sid, src, key, i, kind, ref, spread, basis):
    ub, cb = call(RB, ib, key, i, kind)
    ua, ca = call(RA, ia, key, i, kind)
    assert cb["notify_label"] == ca["notify_label"] and json.dumps(cb["x_span"]) == json.dumps(ca["x_span"])
    vb, eb, va, ea = fx(cb["value"]), fx(cb["abserr"]), fx(ca["value"]), fx(ca["abserr"])
    rb = abs(vb - ref) / eb; ra = abs(va - ref) / ea
    honest = rb <= 1
    return dict(sid=sid, src=src, unit=ub, index=i, kind=kind, label=cb["notify_label"].split("group ")[-1],
                basis=basis, converged=cb["converged"], converged_07a=ca["converged"],
                edge=cb["num_edge_rule_refusals"], range=cb["num_range_rule_refusals"],
                ratio=float(rb), ratio_07a=float(ra), abserr_over_07a=float(eb / ea),
                spread_over_min_abserr=float(spread / min(eb, ea)) if spread is not None else None,
                agree_min=(spread <= min(eb, ea) / 100) if spread is not None else None,
                agree_own=(spread <= eb / 100) if spread is not None else None,
                verdict="honest" if honest else ("unconverged" if not cb["converged"] else "NOT honest"),
                value_hex=cb["value"]["hex"], abserr_hex=cb["abserr"]["hex"])


def main():
    out = Path(sys.argv[1])
    wanted = json.load(open(MEAS / "state/census_wanted_07b.json"))["wanted"]
    sf = json.load(open(MEAS / "out/tables/scores_final.json"))
    rows = []
    sets = {"v4", "R1", "R2", "R2~o", "R3-aux", "R3-aux~o", "R3-own", "R3-own~o", "R3-W7", "R3-W7~o", "item3", "pairs"}  # v2: Part A's four pair calls added, as the census agent reads R3
    for sid, v in sorted(sf["table"].items()):
        t = v["tree"]
        if t["set"] not in sets:
            continue
        w = wanted[sid]; assert w["which"] == "h"
        u = w["unit"]; rt, row = u[2:6], u[7:]
        r = sf["refs"][t["ref_key"]]
        rows.append(score(sid, "A:" + t["set"], f"{row}@{rt}", w["index"], w["kind"], F(Decimal(r["ref"])),
                          F(r["spread"]) if r.get("spread") is not None else None, r["basis"]))
    s06g = {r["sid"]: r for r in json.load(open(S06G))["rows"]}
    for z in ZS:
        r6 = s06g[z]
        rd = {k: F(Decimal(v["value_30"])) for k, v in r6["readings"].items() if k not in r6["unusable"]}
        assert F(Decimal(r6["ref_30"])) == rd["fcc_r2"]
        u = r6["unit"]; rt, row = u[2:6], u[7:]
        assert u.startswith("g-") and rt == r6["rtol"]
        rows.append(score(z, "06g", f"{row}@{rt}", r6["index"], r6["kind"] if r6["kind"] in ("group", "analytic") else None,
                          rd["fcc_r2"], max(rd.values()) - min(rd.values()), "06g " + "+".join(sorted(rd))))
    probs = []
    for row in ("1-34599", "3-678"):
        for i in (2, 3):
            rd = {}
            for p in sorted(glob.glob(str(RD7A / f"r{row}-1e-7-{i}-*.json"))):
                r = json.load(open(p))
                ok = (r["tool_sha256"] == TOOL and sha(r["pickle"]) == r["pickle_sha256"] and r["route"] == "B"
                      and float(r["value_30"]).hex() == r["value_hex"] and r["index"] == i)
                if ok:
                    rd[f"{r['impl']}_r{r['res']}"] = F(Decimal(r["value_30"]))
                else:
                    probs.append(p)
            for rt in ("1e-7", "1e-8"):
                rows.append(score(f"r{row}-{i}-{rt}", "07a refs", f"{row}@{rt}", i, "group", rd["fcc_r2"],
                                  max(rd.values()) - min(rd.values()), "07a " + "+".join(sorted(rd))))
    u4_fail = [r["sid"] for r in rows if r["verdict"] == "NOT honest"]
    u4_unconv = [r["sid"] for r in rows if r["verdict"] == "unconverged"]
    # U6
    u6 = []
    for x in json.load(open(W2REF))["results"]:
        u = x["unit"]; assert u.startswith("b-1e-7-")
        row = u[7:]
        ub, cb = call(RB, ib, f"{row}@1e-7", x["index"])
        ua, ca = call(RA, ia, f"{row}@1e-7", x["index"])
        assert cb["notify_label"] == x["notify_label"] == ca["notify_label"]
        s10 = F(float.fromhex(x["S10"]))
        rb = abs(fx(cb["value"]) - s10) / fx(cb["abserr"]); ra = abs(fx(ca["value"]) - s10) / fx(ca["abserr"])
        u6.append(dict(unit=ub, index=x["index"], label=cb["notify_label"].split("group ")[-1], converged=cb["converged"],
                       converged_07a=ca["converged"], ratio=float(rb), ratio_07a=float(ra),
                       edge=cb["num_edge_rule_refusals"], range=cb["num_range_rule_refusals"],
                       verdict="honest" if rb <= 1 else ("unconverged" if not cb["converged"] else "NOT honest")))
    res = dict(script_sha256=sha(__file__), inputs={str(p): sha(p) for p in (MEAS / "state/census_wanted_07b.json",
               MEAS / "out/tables/scores_final.json", S06G, W2REF)}, reading_problems=probs,
               U4=dict(n=len(rows), not_honest=u4_fail, unconverged=u4_unconv,
                       holds=not u4_fail, n_by_src={s: sum(r["src"] == s for r in rows) for s in sorted({r["src"] for r in rows})},
                       not_agree_min=[r["sid"] for r in rows if r["agree_min"] is False],
                       not_agree_own=[r["sid"] for r in rows if r["agree_own"] is False],
                       max_ratio_converged=max((r for r in rows if r["converged"]), key=lambda r: r["ratio"]),
                       rows=rows),
               U6=dict(n=len(u6), not_honest=[(r["unit"], r["index"]) for r in u6 if r["verdict"] == "NOT honest"],
                       unconverged=[(r["unit"], r["index"]) for r in u6 if r["verdict"] == "unconverged"],
                       holds=all(r["verdict"] != "NOT honest" for r in u6), rows=u6))
    out.write_text(json.dumps(res, indent=1, default=str))
    print("script", res["script_sha256"]); print("inputs", json.dumps(res["inputs"], indent=1)); print("reading problems", probs)
    print(f"U4: {len(rows)} calls {res['U4']['n_by_src']}")
    print(f"{'sid':22} {'src':12} {'conv':5} {'e/r':7} {'ratio':>10} {'ratio07a':>10} {'e/e07a':>8} {'spr/min':>9} verdict")
    for r in rows:
        print(f"{r['sid']:22} {r['src']:12} {str(r['converged'])[0]}/{str(r['converged_07a'])[0]:3} {r['edge']}/{r['range']:<5} "
              f"{r['ratio']:10.4g} {r['ratio_07a']:10.4g} {r['abserr_over_07a']:8.3g} {r['spread_over_min_abserr'] or 0:9.2g} {r['verdict']}")
    print("U4 not honest:", u4_fail, " unconverged:", u4_unconv, " holds:", res["U4"]["holds"])
    print("U4 readings not agreeing (min tol):", res["U4"]["not_agree_min"], " (own tol):", res["U4"]["not_agree_own"])
    print(f"U6: {len(u6)} calls")
    for r in u6:
        print(f"  {r['unit']:18} #{r['index']:<3} {r['label']:8} conv {str(r['converged'])[0]}/{str(r['converged_07a'])[0]} "
              f"e/r {r['edge']}/{r['range']:<4} ratio {r['ratio']:.4g} (07a {r['ratio_07a']:.4g}) {r['verdict']}")
    print("U6 not honest:", res["U6"]["not_honest"], " unconverged:", res["U6"]["unconverged"], " holds:", res["U6"]["holds"])


if __name__ == "__main__":
    main()
