"""levin-phase-floor prompt 07c section 6: Y4 (a), 07b's U4 set from 07b's Part A and 07a's two rows, and Y6 (06b's 24),
scored from 07c's census on the same references by the same rule. Copied from 07b's census score_refs_07b.py (a record,
not edited), changed only where marked "prompt 07c": the run given is 07c's census, and "07a" (the other code: the
"scored" test's min(abserr) and the columns ``*_07a``) reads 07b's census (state/index_07b.json); each Part A record
also carries the call's three chase counts. The references, the sets and the rule are 07b's, unchanged. 07b's text
follows.

levin-phase-floor prompt 07b section 6: U4 (Part A's referenced calls and 07a's two rows) and U6 (06b's 24), scored
from 07b's census against their references, with 07a's census beside. New in 07b. Read only.

**U4, Part A's calls** (state/predictions.md part 4): every id of Part A's state/census_wanted_07b.json whose ``which``
is ``h`` and whose set is R1, R2, R3 (aux, own, W7, the pairs) or item 3, with ``~o`` (the other rtol's call on the same
integrand) included. Each id's reference is Part A's: its ``refs()`` and ``ref_key()`` (Part A's
scripts/score_07b.py, copied below unchanged in their logic, with the two committed inputs read from the tree h, whose
SHA-256 equal Part A's), from Part A's readings (out/refs/readings/), 07's V4 readings and 07a's census reference of
1-34599 (item 3). V4's 25 and 07a's other eight V4 calls are score_v4_07b.py's. **07a's two rows:** 07a's census
references (var/levin-phase-floor-07a/census/out/refs/scores.json): 1-34599 and 3-678, group calls 2 and 3, both
rtols. **The rule** (07's, as Part A and 07a's W4 applied it): a call is scored if its reference's stated error (the
readings' spread) is at most min(abserr_07b, abserr_07a)/100; otherwise reported, not scored; honest iff
|v - ref| <= abserr. A call unconverged at 07b counts under U4 as "unconverged under section 9's rule".

**U6:** 06b's 24 (docs/levin-phase-floor/06b/census/out/w2_reference.json; unit b-1e-7-X-N is this census's
h-1e-7-X-N, the group call of the same index), |v - S10|/abserr by 06b's rule: scored if |S9 - S10| <= abserr/100;
honest iff |v - S10| <= abserr; or unconverged.

Every comparison in mpmath at 40 digits (references as their 30-digit strings or hex). Writes OUT_JSON and prints the
summary. Usage: python -B score_refs_07b.py RUN_DIR OUT_JSON
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07A_SHA256, H_TREE, MAIN, RUN07A, WIDER, header, sha256  # noqa: E402

PART_A = MAIN / "var" / "levin-phase-floor-07b" / "measurement"
V4S = H_TREE / "docs/levin-phase-floor/07/census/out/v4/v4_scores.json"
W2R = H_TREE / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
I3R = MAIN / "var" / "levin-phase-floor-07a" / "census" / "out" / "refs" / "scores.json"
READ = PART_A / "out" / "refs" / "readings"
WANTED = PART_A / "state" / "census_wanted_07b.json"
CALLS = PART_A / "state" / "calls_07b.json"
INDEX07A = WIDER / "state" / "index_07a.json"
PART_A_INPUT_SHA = {V4S.name: "2d0268e607e0cb17b4d4247831a7b933c2df0db61c248b8959ecdf9f091649b8",
                    W2R.name: "4178395291167c5960859d8b7fe785f79039dfbc61bee09ab663abf7bf766140"}
INDEX07A = WIDER / "state" / "index_07b.json"  # prompt 07c: the other code is 07b's census (the name kept)
from prov import CHECKPOINT07B_SHA256, RUN07B  # noqa: E402  prompt 07c
RUN07A, CHECKPOINT07A_SHA256 = RUN07B, CHECKPOINT07B_SHA256  # prompt 07c
U4_SETS = ("R1", "R2", "R2~o", "R3-aux", "R3-aux~o", "R3-own", "R3-own~o", "R3-W7", "R3-W7~o", "pairs", "item3")


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def refs():  # Part A's (scripts/score_07b.py), unchanged in its logic
    out = {}
    v4 = json.loads(V4S.read_text())
    for c in v4["calls"]:
        rd = [mpf(v) for v in c["readings_30"].values() if v is not None]
        out[c["sid"]] = {"ref": mpf(c["readings_30"]["fcc_r2"]), "spread": float(max(rd) - min(rd)), "basis": c.get("basis", "07")}
    for c in v4["z"]:
        if c["sid"] == "z12":
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
        if fs.exists():
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


def ref_key(sid):  # Part A's (scripts/score_07b.py), unchanged in its logic
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


def reader(ck, units):
    """{unit: data} for the named units, read from a checkpoint by its unit-name prefix."""
    out = {}
    with open(ck) as f:
        for line in f:
            u = line[10:60].split('"', 1)[0]
            if u in units:
                out[u] = json.loads(line)["data"]
    return out


def call_of(d, kind, index):
    key = "calls" if kind == "group" else "other_calls"
    cs = [c for c in d.get(key) or [] if c["index"] == index]
    if len(cs) != 1:
        raise SystemExit(f"{d['unit']} {kind} #{index}: {len(cs)} calls")
    return cs[0]


def score(c, c7, ref, spread):
    v, e, v7, e7 = fx(c["value"]["hex"]), fx(c["abserr"]["hex"]), fx(c7["value"]["hex"]), fx(c7["abserr"]["hex"])
    r = {"value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "converged": c["converged"], "regions": c["num_regions"],
         "max_depth": c["max_depth"], "refusals": [c.get("num_edge_rule_refusals"), c.get("num_range_rule_refusals")],
         "chase_counts": [c.get("num_chase_stops"), c.get("num_chase_stopped_leaves"), c.get("num_chase_noise_bisections")],  # prompt 07c
         "value_07a": c7["value"]["hex"], "abserr_07a": c7["abserr"]["hex"], "converged_07a": c7["converged"],
         "regions_07a": c7["num_regions"], "ref": mp.nstr(ref, 20), "spread": spread,
         "ratio": float(abs(mpf(v) - ref) / e), "ratio_07a": float(abs(mpf(v7) - ref) / e7),
         "scored": spread <= min(e, e7) / 100, "spread_over_abserr": spread / e}
    r["verdict"] = ("reported, not scored" if not r["scored"] else "unconverged" if not c["converged"]
                    else "honest" if r["ratio"] <= 1 else "converged and NOT honest")
    r["verdict_07a"] = ("unconverged" if not c7["converged"] else "honest" if r["ratio_07a"] <= 1 else "converged and NOT honest")
    return r


def main():
    run, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    ck = run / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: Y4 (a) (07b Part A's calls and 07a's two rows) and Y6, scored",
                 [V4S, W2R, I3R, WANTED, CALLS, ck])
    for n, s in PART_A_INPUT_SHA.items():
        p = V4S if n == V4S.name else W2R
        if sha256(p) != s:
            raise SystemExit(f"{p} is not Part A's input")
    idx = json.loads(INDEX07A.read_text())
    if idx["checkpoint_sha256"] != CHECKPOINT07A_SHA256:
        raise SystemExit("the index is not of 07a's checkpoint")
    R = refs()
    wanted = {k: v for k, v in json.loads(WANTED.read_text())["wanted"].items() if v["which"] == "h"}
    sets = {k: v["set"] for k, v in json.loads(CALLS.read_text())["calls"].items()}
    u4 = {k: v for k, v in wanted.items() if sets[k] in U4_SETS}
    i3r = json.loads(I3R.read_text())["calls"]
    w2 = json.loads(W2R.read_text())["results"]
    units = {v["unit"] for v in u4.values()} | {c["unit"] for c in i3r} | {"h-" + r["unit"].split("-", 1)[1] for r in w2}
    mine = reader(ck, units)

    def theirs(u):
        e = idx["units"][u]
        with open(RUN07A / "checkpoint.jsonl", "rb") as f:
            f.seek(e["offset"])
            line = f.read(e["length"])
        if hashlib.sha256(line).hexdigest() != e["line_sha256"]:
            raise SystemExit(f"{u}: 07a's line sha256 differs")
        return json.loads(line)["data"]

    doc = {"header": hdr, "rule": __doc__, "u4_partA": {}, "u4_07a_rows": {}, "u6": {}}
    for sid in sorted(u4):
        w = u4[sid]
        rk = ref_key(sid)
        if rk not in R:
            doc["u4_partA"][sid] = {"set": sets[sid], "unit": w["unit"], "verdict": "no reference", "ref_key": rk}
            continue
        c, c7 = call_of(mine[w["unit"]], w["kind"], w["index"]), call_of(theirs(w["unit"]), w["kind"], w["index"])
        doc["u4_partA"][sid] = {"set": sets[sid], "unit": w["unit"], "kind": w["kind"], "index": w["index"], "ref_key": rk,
                                "ref_basis": R[rk]["basis"], **score(c, c7, R[rk]["ref"], R[rk]["spread"])}
    for r in i3r:
        c, c7 = call_of(mine[r["unit"]], "group", r["index"]), call_of(theirs(r["unit"]), "group", r["index"])
        doc["u4_07a_rows"][r["sid"]] = {"unit": r["unit"], "index": r["index"], "class_07a": r.get("class"),
                                        **score(c, c7, mpf(r["ref_30"]), r["stated_error_spread"])}
    for r in w2:
        u = "h-" + r["unit"].split("-", 1)[1]
        c, c7 = call_of(mine[u], "group", r["index"]), call_of(theirs(u), "group", r["index"])
        v, e = fx(c["value"]["hex"]), fx(c["abserr"]["hex"])
        S10, S9 = mpf(fx(r["S10"])), mpf(fx(r["S9"]))
        sp = float(abs(S9 - S10))
        x = {"unit": u, "index": r["index"], "band": r["band"], "k_key": r["k_key"], "value": c["value"]["hex"],
             "abserr": c["abserr"]["hex"], "converged": c["converged"], "regions": c["num_regions"],
             "refusals": [c.get("num_edge_rule_refusals"), c.get("num_range_rule_refusals")],
             "ratio": float(abs(mpf(v) - S10) / e), "S9_S10_over_abserr": sp / e, "scored": sp <= e / 100,
             "ratio_07a": float(abs(mpf(fx(c7["value"]["hex"])) - S10) / fx(c7["abserr"]["hex"])),
             "converged_07a": c7["converged"], "ratio_06b": r["err_b_over_abserr_b"], "regions_07a": c7["num_regions"]}
        x["verdict"] = ("reported, not scored" if not x["scored"] else "unconverged" if not c["converged"]
                        else "honest" if x["ratio"] <= 1 else "converged and NOT honest")
        doc["u6"][f"{r['unit']}#{r['index']}"] = x

    def tally(rows):
        t = {}
        for x in rows:
            t[x["verdict"]] = t.get(x["verdict"], 0) + 1
        return t

    pa, ar, u6 = list(doc["u4_partA"].values()), list(doc["u4_07a_rows"].values()), list(doc["u6"].values())
    doc["summary"] = {
        "u4_partA_by_set": {s: tally([x for x in pa if x["set"] == s]) for s in sorted({x["set"] for x in pa})},
        "u4_partA": tally(pa), "u4_07a_rows": tally(ar),
        "u4_partA_unconverged": sorted(k for k, x in doc["u4_partA"].items() if x["verdict"] == "unconverged"),
        "u4_partA_dishonest_converged": sorted(k for k, x in doc["u4_partA"].items() if x["verdict"] == "converged and NOT honest"),
        "u4_partA_not_scored": sorted(k for k, x in doc["u4_partA"].items() if x["verdict"] in ("reported, not scored", "no reference")),
        "u4_partA_unconverged_ratios": {k: x["ratio"] for k, x in doc["u4_partA"].items() if x.get("verdict") == "unconverged"},
        "u4_07a_rows_ratios": {k: [x["ratio"], x["ratio_07a"]] for k, x in doc["u4_07a_rows"].items()},
        "u4_partA_max_ratio_scored_converged": max((x["ratio"] for x in pa if x.get("verdict") == "honest"), default=None),
        "u4_partA_07a_dishonest_converged": sorted(k for k, x in doc["u4_partA"].items()
                                                   if x.get("scored") and x.get("verdict_07a") == "converged and NOT honest"),
        "u4_partA_distinct_census_calls": len({(x["unit"], x.get("kind"), x.get("index")) for x in pa}),
        "u6": tally(u6), "u6_ratios": {k: [x["ratio"], x["ratio_07a"], x["ratio_06b"], x["converged"]] for k, x in doc["u6"].items()},
        "u6_07a_dishonest": sorted(k for k, x in doc["u6"].items() if x["ratio_07a"] > 1 and x["converged_07a"]),
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc["summary"], indent=1, default=str))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
