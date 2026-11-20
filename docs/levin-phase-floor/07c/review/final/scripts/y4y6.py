"""Final review (levin-phase-floor 07c Part C, §C.2): my own re-score of Y4 and Y6. Read only but for OUT_JSON.
Independent of the census agent's scripts; adapted from 07b's review/final/scripts/u4u6_v2.py (U4, U6).

usage: python -I -B y4y6.py OUT_JSON

Census calls (value, abserr, converged) are read from 07c's census checkpoint ("c", the after) and 07b's ("b", the
before) through my own indices (final/index07c.json, index07b.json: each file's SHA-256 asserted, each line's SHA-256
checked on read).

Y4's calls and references:
 (1) 07b's U4's 190, built exactly as u4u6_v2.py built them:
   (a) 07b Part A (07b measurement out/tables/scores_final.json): every table id whose tree set is v4, R1, R2, R2~o,
       R3-*, item3 or pairs; its census call from 07b's state/census_wanted_07b.json; reference refs[ref_key].ref,
       spread refs[ref_key].spread;
   (b) z00, z07, z16, z18, z21, z24, z28, z29: 06g's scores_stage3/scores.json rows (worktree docs), 06g's `unusable`
       left out, reference ref_30 (asserted equal to fcc_r2's value_30), spread max-min of the readings;
   (c) 07a's references for rows 1-34599 and 3-678, group indices 2 and 3, both rtols: 07a census readings accepted only
       if tool d37e633b..., input pickle SHA-256 as recorded, route B, value_30 rounding to value_hex; reference fcc r2,
       spread max-min.
 (2) 07c Part A's R4, R5 and R2-28 (07c measurement out/tables/scores_07c.json `scored`, sets chase, chase~o, r5, r5~o);
     census call from state/census_wanted_07c*.json. R4's and R5's references are recomputed here from the readings
     (out/refs/readings/<ref_key>-{fccsplit,qawo-r1,qawo-r2}.json; tool SHA-256 d37e633b..., capture/pickle SHA-256
     checked against the input file): ref = split FCC r2 (sum of its pieces' value_30), spread = max - min over every
     reading taken (fccsplit r1, r2, qawo r1, r2 where present); each asserted equal to scores_07c.json refs[ref_key]
     (ref to the 25 significant digits it is written with there, spread to the float); my unrounded ref is used. R2-28's reference (ref-R2-28) is 07b Part A's, taken from
     scores_07c.json refs and asserted equal to 07b's scores_final.json refs["ref-R2-28"].
 Rule (07's): honest iff |v - ref| <= abserr, exact; an unconverged call passes. Y4 holds iff every referenced call other
 than R2-28's census call (h-<rtol>-0-13757 group #5, at both rtols, in either set) is honest or unconverged. The
 verdict is "honest" if ratio <= 1, else "unconverged" if converged is false, else "NOT honest" (u4u6_v2.py's); every
 call with converged false is also listed with its ratio. Agreement of
 readings is reported (spread <= min(abserr_c, abserr_b)/100), not used to drop any call.
Y6: 06b's 24 (worktree docs/levin-phase-floor/06b/census/out/w2_reference.json); unit b-1e-7-X-N -> row X-N @1e-7, same
 index; reference S10 (hex). Holds iff |v - S10| / abserr <= 1 or unconverged. S9-S10 agreement reported.
"""
import glob, hashlib, json, sys
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
WT = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c")
FIN = M / "var/levin-phase-floor-07c/review/final"
RC = M / "var/runs/levin-phase-floor-07c-census-20261119T201419"
RB = M / "var/runs/levin-phase-floor-07b-census-20261117T010200"
MEASB = M / "var/levin-phase-floor-07b/measurement"
MEASC = M / "var/levin-phase-floor-07c/measurement"
S06G = WT / "docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json"
W2REF = WT / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
RD7A = M / "var/levin-phase-floor-07a/census/out/refs/readings"
TOOL = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
SHA_C = "33d94ce21077cd23f29da65ef967b64676db6722c081b999946bbc91c822a38e"
SHA_B = "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253"
ZS = ["z00", "z07", "z16", "z18", "z21", "z24", "z28", "z29"]
R228 = {("0-13757@1e-7", "group", 5), ("0-13757@1e-8", "group", 5)}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
fx = lambda v: F(float.fromhex(v["hex"] if isinstance(v, dict) else v))

IC = json.load(open(FIN / "index07c.json")); IB = json.load(open(FIN / "index07b.json"))
assert IC["sha256"] == SHA_C and IB["sha256"] == SHA_B
FH = {"c": open(RC / "checkpoint.jsonl", "rb"), "b": open(RB / "checkpoint.jsonl", "rb")}
CACHE = {}


def line(which, key):
    if (which, key) in CACHE:
        return CACHE[(which, key)]
    e = (IC if which == "c" else IB)["units"][key]
    f = FH[which]; f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    d = json.loads(raw)["data"]
    assert d["status"] == "done"
    CACHE[(which, key)] = d
    return d


def call(which, key, i, kind):
    d = line(which, key)
    lists = {"group": d["calls"], "analytic": d["other_calls"], None: d["calls"] + d["other_calls"]}[kind]
    m = [c for c in lists if c["index"] == i]
    assert len(m) == 1, (key, i, kind)
    k = kind or ("group" if any(c is m[0] for c in d["calls"]) else "analytic")
    return d["unit"], m[0], k


def score(sid, src, key, i, kind, ref, spread, basis):
    uc, cc, k = call("c", key, i, kind)
    ub, cb, _ = call("b", key, i, kind)
    assert cc["notify_label"] == cb["notify_label"] and json.dumps(cc["x_span"]) == json.dumps(cb["x_span"])
    vc, ec, vb, eb = fx(cc["value"]), fx(cc["abserr"]), fx(cb["value"]), fx(cb["abserr"])
    rc = abs(vc - ref) / ec; rb = abs(vb - ref) / eb
    return dict(sid=sid, src=src, unit=uc, key=key, index=i, kind=k, label=cc["notify_label"].split("group ")[-1],
                basis=basis, converged=cc["converged"], converged_07b=cb["converged"],
                chase_stops=cc["num_chase_stops"], range=cc["num_range_rule_refusals"], edge=cc["num_edge_rule_refusals"],
                equal_07b=(cc["value"]["hex"], cc["abserr"]["hex"]) == (cb["value"]["hex"], cb["abserr"]["hex"]),
                ratio=float(rc), ratio_07b=float(rb), abserr_over_07b=float(ec / eb),
                spread_over_min_abserr=float(spread / min(ec, eb)) if spread is not None else None,
                agree_min=(spread <= min(ec, eb) / 100) if spread is not None else None,
                r228=(key, k, i) in R228,
                verdict="honest" if rc <= 1 else ("unconverged" if not cc["converged"] else "NOT honest"),
                verdict_07b="honest" if rb <= 1 else ("unconverged" if not cb["converged"] else "NOT honest"),
                value_hex=cc["value"]["hex"], abserr_hex=cc["abserr"]["hex"])


def keyof(unit):  # h-1e-7-0-13757 -> 0-13757@1e-7
    p = unit.split("-")
    return f"{p[3]}-{p[4]}@{p[1]}-{p[2]}"


def refs_07c(ref_key, problems):
    """R4/R5 reference from the readings: split FCC r2 summed over pieces; spread over all readings."""
    rd = {}
    pf = MEASC / "out/refs/readings" / f"{ref_key}-fccsplit.json"
    r = json.load(open(pf)); h = r["header"]
    ok = h["tool_sha256"] == TOOL and r["sid"] == ref_key and sha(h["capture"]["path"]) == h["capture"]["sha256"]
    if not ok:
        problems.append(str(pf))
    for res in ("1", "2"):
        rd[f"fccsplit-r{res}"] = sum((F(Decimal(p["value_30"])) for p in r["readings"][res]["pieces"]), F(0))
    for res in (1, 2):
        pq = MEASC / "out/refs/readings" / f"{ref_key}-qawo-r{res}.json"
        if not pq.exists():
            continue
        q = json.load(open(pq))
        ok = (q["tool_sha256"] == TOOL and sha(q["pickle"]) == q["pickle_sha256"] and q["route"] == "B" and q["res"] == res
              and q["impl"] == "qawo" and float(q["value_30"]).hex() == q["value_hex"] and q["sample"] == ref_key)
        if ok:
            rd[f"qawo-r{res}"] = F(Decimal(q["value_30"]))
        else:
            problems.append(str(pq))
    return rd["fccsplit-r2"], max(rd.values()) - min(rd.values()), "+".join(sorted(rd)), sorted(rd)


def main():
    out = Path(sys.argv[1])
    problems = []
    rows = []
    # (1a) 07b Part A
    wanted = json.load(open(MEASB / "state/census_wanted_07b.json"))["wanted"]
    sf = json.load(open(MEASB / "out/tables/scores_final.json"))
    sets = {"v4", "R1", "R2", "R2~o", "R3-aux", "R3-aux~o", "R3-own", "R3-own~o", "R3-W7", "R3-W7~o", "item3", "pairs"}
    for sid, v in sorted(sf["table"].items()):
        t = v["tree"]
        if t["set"] not in sets:
            continue
        w = wanted[sid]; assert w["which"] == "h"
        r = sf["refs"][t["ref_key"]]
        rows.append(score(sid, "07b-A:" + t["set"], keyof(w["unit"]), w["index"], w["kind"], F(Decimal(r["ref"])),
                          F(r["spread"]) if r.get("spread") is not None else None, r["basis"]))
    # (1b) 06g
    s06g = {r["sid"]: r for r in json.load(open(S06G))["rows"]}
    for z in ZS:
        r6 = s06g[z]
        rd = {k: F(Decimal(v["value_30"])) for k, v in r6["readings"].items() if k not in r6["unusable"]}
        assert F(Decimal(r6["ref_30"])) == rd["fcc_r2"]
        u = r6["unit"]; rt, row = u[2:6], u[7:]
        assert u.startswith("g-") and rt == r6["rtol"]
        rows.append(score(z, "06g", f"{row}@{rt}", r6["index"], r6["kind"] if r6["kind"] in ("group", "analytic") else None,
                          rd["fcc_r2"], max(rd.values()) - min(rd.values()), "06g " + "+".join(sorted(rd))))
    # (1c) 07a's references
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
                    problems.append(p)
            for rt in ("1e-7", "1e-8"):
                rows.append(score(f"r{row}-{i}-{rt}", "07a refs", f"{row}@{rt}", i, "group", rd["fcc_r2"],
                                  max(rd.values()) - min(rd.values()), "07a " + "+".join(sorted(rd))))
    n190 = len(rows)
    assert n190 == 190, n190
    # (2) 07c Part A: R4, R5, R2-28
    sc = json.load(open(MEASC / "out/tables/scores_07c.json"))
    W = {}
    for f in ("census_wanted_07c.json", "census_wanted_07c_chase.json", "census_wanted_07c_r5.json"):
        for k, v in json.load(open(MEASC / "state" / f))["wanted"].items():
            assert k not in W or W[k] == v
            W[k] = v
    ref_check = []
    for sid, v in sorted(sc["scored"].items()):
        if v["set"] not in ("chase", "chase~o", "r5", "r5~o"):
            continue
        rk = v["ref_key"]
        w = W[sid]
        if rk == "ref-R2-28":
            src = "07c-A:R2-28"
            r = sc["refs"][rk]
            rb_ = sf["refs"][rk]
            assert F(Decimal(r["ref"])) == F(Decimal(rb_["ref"])) and r["spread"] == rb_["spread"]
            ref, spread, basis = F(Decimal(r["ref"])), F(r["spread"]), r["basis"]
        else:
            src = "07c-A:R4" if rk.startswith("ref-R4-") else ("07c-A:R5" if rk.startswith("ref-R5-") else None)
            assert src is not None, rk
            ref, spread, basis, _ = refs_07c(rk, problems)
            r = sc["refs"][rk]
            # scores_07c.json writes the reference to 25 significant digits: agreement to that rounding is checked
            ref_check.append((rk, abs(ref - F(Decimal(r["ref"]))) <= abs(ref) * F(1, 10**24), float(spread) == r["spread"],
                              basis == r["basis"], F(Decimal(v["ref"])) == F(Decimal(r["ref"]))))
        rows.append(score(sid, src, keyof(w["unit"]), w["index"], w["kind"], ref, spread, basis) | {"set": v["set"]})
    ref_mismatch = sorted({x for x in ref_check if not all(x[1:])})
    fail = [r["sid"] for r in rows if r["verdict"] == "NOT honest" and not r["r228"]]
    r228 = [r for r in rows if r["r228"]]
    unconv = [r["sid"] for r in rows if r["verdict"] == "unconverged"]
    conv = [r for r in rows if r["converged"] and not r["r228"]]
    mx = max(conv, key=lambda r: r["ratio"])
    nsrc = {}
    for r in rows:
        nsrc[r["src"]] = nsrc.get(r["src"], 0) + 1
    distinct = len({(r["key"], r["kind"], r["index"]) for r in rows})
    # Y6
    y6 = []
    for x in json.load(open(W2REF))["results"]:
        u = x["unit"]; assert u.startswith("b-1e-7-")
        key = f"{u[7:]}@1e-7"
        uc, cc, k = call("c", key, x["index"], None)
        ub, cb, _ = call("b", key, x["index"], None)
        assert cc["notify_label"] == x["notify_label"] == cb["notify_label"] and k == "group"
        s10, s9 = F(float.fromhex(x["S10"])), F(float.fromhex(x["S9"]))
        ec = fx(cc["abserr"])
        rc = abs(fx(cc["value"]) - s10) / ec; rb = abs(fx(cb["value"]) - s10) / fx(cb["abserr"])
        y6.append(dict(unit=uc, index=x["index"], label=cc["notify_label"].split("group ")[-1], converged=cc["converged"],
                       converged_07b=cb["converged"], ratio=float(rc), ratio_07b=float(rb), chase_stops=cc["num_chase_stops"],
                       equal_07b=(cc["value"]["hex"], cc["abserr"]["hex"]) == (cb["value"]["hex"], cb["abserr"]["hex"]),
                       s9_s10_le_abserr_over_100=abs(s9 - s10) <= ec / 100,
                       verdict="honest" if rc <= 1 else ("unconverged" if not cc["converged"] else "NOT honest")))
    res = dict(
        script_sha256=sha(__file__),
        inputs={str(p): sha(p) for p in (FIN / "index07c.json", FIN / "index07b.json", MEASB / "state/census_wanted_07b.json",
                                         MEASB / "out/tables/scores_final.json", S06G, W2REF,
                                         MEASC / "out/tables/scores_07c.json", MEASC / "state/census_wanted_07c.json",
                                         MEASC / "state/census_wanted_07c_chase.json", MEASC / "state/census_wanted_07c_r5.json")},
        checkpoints={"07c": SHA_C, "07b": SHA_B}, reading_problems=problems, ref_recomputed_mismatch=ref_mismatch,
        n_refs_recomputed=len(ref_check),
        Y4=dict(n=len(rows), n_distinct_calls=distinct, n_by_src=nsrc, not_honest_excluding_R2_28=fail, unconverged=unconv,
                holds=not fail,
                R2_28=[{k: r[k] for k in ("sid", "src", "unit", "index", "converged", "ratio", "ratio_07b", "verdict",
                                          "chase_stops", "value_hex", "abserr_hex")} for r in r228],
                max_ratio_converged_excl_R2_28={k: mx[k] for k in ("sid", "src", "unit", "index", "ratio", "ratio_07b")},
                not_agree_min=[r["sid"] for r in rows if r["agree_min"] is False],
                converged_false=[(r["sid"], r["ratio"]) for r in rows if not r["converged"]],
                converged_false_07b=[(r["sid"], r["ratio_07b"]) for r in rows if not r["converged_07b"]],
                max_ratio_converged_07b_excl_R2_28=max(((r["sid"], r["ratio_07b"]) for r in rows
                                                        if r["converged_07b"] and not r["r228"]), key=lambda t: t[1]),
                n_equal_07b=sum(r["equal_07b"] for r in rows), n_chase_stops_pos=sum(r["chase_stops"] > 0 for r in rows),
                not_honest_at_07b=[r["sid"] for r in rows if r["verdict_07b"] == "NOT honest"],
                unconverged_at_07b=[r["sid"] for r in rows if r["verdict_07b"] == "unconverged"],
                rows=rows),
        Y6=dict(n=len(y6), not_honest=[(r["unit"], r["index"]) for r in y6 if r["verdict"] == "NOT honest"],
                unconverged=[(r["unit"], r["index"]) for r in y6 if r["verdict"] == "unconverged"],
                holds=all(r["verdict"] != "NOT honest" for r in y6), max_ratio=max(r["ratio"] for r in y6), rows=y6))
    out.write_text(json.dumps(res, indent=1, default=str))
    print("script", res["script_sha256"]); print("reading problems", problems); print("recomputed refs", len(ref_check),
                                                                                  "mismatching", ref_mismatch)
    print(f"Y4: {len(rows)} scorings, {distinct} distinct calls; {nsrc}")
    print(f"{'sid':24} {'src':14} {'c/b':5} {'stp':>3} {'ratio':>10} {'ratio07b':>10} {'e/e07b':>8} {'spr/min':>9} verdict")
    for r in rows:
        print(f"{r['sid']:24} {r['src']:14} {str(r['converged'])[0]}/{str(r['converged_07b'])[0]:3} {r['chase_stops']:>3} "
              f"{r['ratio']:10.4g} {r['ratio_07b']:10.4g} {r['abserr_over_07b']:8.3g} {r['spread_over_min_abserr'] or 0:9.2g} "
              f"{r['verdict']}{'  [R2-28]' if r['r228'] else ''}")
    print("Y4 not honest (excl. R2-28):", fail, " unconverged:", unconv, " holds:", res["Y4"]["holds"])
    print("R2-28:", json.dumps(res["Y4"]["R2_28"], indent=0))
    print("max ratio converged (excl. R2-28):", res["Y4"]["max_ratio_converged_excl_R2_28"])
    print("converged false at 07c (sid, ratio):", res["Y4"]["converged_false"])
    print("converged false at 07b (sid, ratio_07b):", res["Y4"]["converged_false_07b"])
    print("max ratio converged at 07b (excl. R2-28):", res["Y4"]["max_ratio_converged_07b_excl_R2_28"])
    print("scorings equal to 07b's call:", res["Y4"]["n_equal_07b"], " with num_chase_stops > 0:", res["Y4"]["n_chase_stops_pos"])
    print("not agreeing within min(abserr)/100:", res["Y4"]["not_agree_min"])
    print("at 07b: not honest", res["Y4"]["not_honest_at_07b"], " unconverged", res["Y4"]["unconverged_at_07b"])
    print(f"Y6: {len(y6)} calls")
    for r in y6:
        print(f"  {r['unit']:18} #{r['index']:<3} {r['label']:8} conv {str(r['converged'])[0]}/{str(r['converged_07b'])[0]} "
              f"stops {r['chase_stops']} eq07b {str(r['equal_07b'])[0]} ratio {r['ratio']:.4g} (07b {r['ratio_07b']:.4g}) "
              f"S9~S10 {str(r['s9_s10_le_abserr_over_100'])[0]} {r['verdict']}")
    print("Y6 not honest:", res["Y6"]["not_honest"], " unconverged:", res["Y6"]["unconverged"], " holds:", res["Y6"]["holds"])


if __name__ == "__main__":
    main()
