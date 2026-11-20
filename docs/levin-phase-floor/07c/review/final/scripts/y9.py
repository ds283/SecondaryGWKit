"""Final review (levin-phase-floor 07c Part C, §C.2): my own re-score of Y9, the chase ends. Read only but for OUT_JSON.
Independent of the census agent's scripts.

usage: python -I -B y9.py OUT_JSON

Inputs: my extracts final/extract_{07c,07b}.jsonl (headers' checkpoint SHA-256s asserted), y7_conv.json (07b's newly
unconverged calls, 07a -> 07b, from my extracts), my indices final/index{07c,07b}.json, and the two censuses' region
pickles (each unit's record accepted only against its checkpoint line's pickle_record_sha256; the line itself checked
against my index's line SHA-256).

My reading of "every depth-20 leaf is below a leaf a rate rule refused", from the region records:
- a call's depth-20 leaves are its returned regions (the pickle's ``regions`` for a group call, ``other_regions`` for an
  analytic call, matched by index and asserted to be the call's: count equal to num_regions, max depth equal to max_depth)
  with depth >= the call's depth_max (20 in every call read; asserted);
- the leaves a rate rule refused are the call's rate-rule evaluations (the pickle's ``rate_rule_07b`` record) with
  ``refused`` true; the entry is matched to the call by index and asserted to be the call's by its refusal counts
  (refused by rule equal to num_edge_rule_refusals and num_range_rule_refusals) and n_unrecorded 0;
- a leaf is below a refused leaf iff some refused evaluation at a depth strictly smaller than the leaf's has a span
  containing the leaf's span, compared exactly (Fractions of the floats, each span ordered). A refused evaluation of the
  leaf's own span at its own depth is counted apart (self-refused), not as an ancestor. Evaluations with depth < 0 (depth
  unknown) are counted and reported; none is used.
  As a check that a containing refused span is an ancestor in the bisection tree, the ratio of its width to the leaf's
  over 2**(leaf depth - refused depth) is reported (1 for an exact bisection line).
This is the census instrument's test (07b's score_07b.r07b_unit, copied into 07c's harness) and, for the nesting of
nodes in one bisection tree, the same as Part A's d1_descent_07c.py's test by the split records (an ancestor that has a
check record with rate_rule.refused), which needs the replay instrument and cannot be run from the census records.
Clause (i): no unconverged call with at least one depth-20 leaf has every depth-20 leaf below a refused leaf. A call
with no depth-20 leaf is reported apart (the vacuous reading).
Clause (ii): of 07b's 198 newly unconverged calls, exactly the five named stay unconverged at 07c, each with
num_chase_stops 0 and equal to 07b's call in value, abserr (hex), converged, num_regions and max_depth.
The same reader is run on 07b's census for every call unconverged there (calibration against 07b's records: 192 of the
198 with every depth-20 leaf below a refused leaf, 668 leaves, shallowest refused depths 0 (129), 1 (31), 2 (30), 3 (2)).
"""
import hashlib, json, pickle, sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
FIN = M / "var/levin-phase-floor-07c/review/final"
RUNS = {"07c": (M / "var/runs/levin-phase-floor-07c-census-20261119T201419",
                "33d94ce21077cd23f29da65ef967b64676db6722c081b999946bbc91c822a38e"),
        "07b": (M / "var/runs/levin-phase-floor-07b-census-20261117T010200",
                "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253")}
FIVE = ["c-1e-8-1-2229-4", "c-1e-8-3-13490-2", "c-1e-8-3-13490-3", "c-1e-8-3-13615-2", "c-1e-8-3-13615-3"]
OUTS = ("value", "abserr", "converged", "num_regions", "max_depth")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_extract(which):
    p = FIN / f"extract_{which}.jsonl"
    out = {}
    with open(p) as f:
        h = json.loads(f.readline())
        assert h["checkpoint_sha256"] == RUNS[which][1]
        for l in f:
            d = json.loads(l)
            d["c"] = {(c["kind"], c["index"]): c for c in d["calls"]}
            out[d["key"]] = d
    return out


def cid(key, kind, i):
    row, rt = key.split("@")
    return f"c-{rt}-{row}-{i}" + ("" if kind == "group" else "-analytic")


class Reader:
    def __init__(self, which):
        self.run, want = RUNS[which]
        self.ix = json.load(open(FIN / f"index{which}.json"))
        assert self.ix["sha256"] == want
        self.f = open(self.run / "checkpoint.jsonl", "rb")

    def record(self, key):
        e = self.ix["units"][key]
        self.f.seek(e[0]); raw = self.f.read(e[1])
        assert hashlib.sha256(raw).hexdigest() == e[2]
        d = json.loads(raw)["data"]
        with open(self.run / d["pickle"], "rb") as g:
            g.seek(d["pickle_offset"]); b = g.read(d["pickle_length"])
        assert hashlib.sha256(b).hexdigest() == d["pickle_record_sha256"], key
        return d, pickle.loads(b)


def span(a, b):
    a, b = F(float(a)), F(float(b))
    return (a, b) if a <= b else (b, a)


def analyse(d, p, kind, i):
    lst = d["calls"] if kind == "group" else d["other_calls"]
    c = [x for x in lst if x["index"] == i]; assert len(c) == 1
    c = c[0]
    regs = [r for r in (p["regions"] if kind == "group" else p["other_regions"]) if r["index"] == i]
    assert len(regs) == 1
    r = regs[0]
    n = len(r["start"])
    assert n == c["num_regions"] and int(max(r["depth"])) == c["max_depth"], (d["unit"], i)
    dmax = c["depth_max"]; assert dmax == 20, dmax
    # the call's rate-rule record; the list is matched by index, and the entry checked against the call's counts
    rrs = [e for e in (p["rate_rule_07b"] or []) if e["index"] == i
           and e["num_edge_rule_refusals"] == c["num_edge_rule_refusals"]
           and e["num_range_rule_refusals"] == c["num_range_rule_refusals"]]
    assert len(rrs) >= 1, (d["unit"], i)
    rec = rrs[0]["record"]
    if len(rrs) > 1:  # same index, same counts: they must be one record
        assert all(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x["record"].items()}) ==
                   json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rec.items()}) for x in rrs)
    assert rec["n_unrecorded"] == 0
    ref = [(span(lo, hi), int(dp), int(ru)) for lo, hi, dp, ru, rf in
           zip(rec["lo"], rec["hi"], rec["depth"], rec["rule"], rec["refused"]) if rf]
    assert sum(1 for x in ref if x[2] == 0) == c["num_edge_rule_refusals"]
    assert sum(1 for x in ref if x[2] == 1) == c["num_range_rule_refusals"]
    n_unknown = sum(1 for x in ref if x[1] < 0)
    leaves = []
    for j in range(n):
        dj = int(r["depth"][j])
        if dj < dmax:
            continue
        s = span(r["start"][j], r["end"][j])
        anc = [(dp, (h - l) / (s[1] - s[0]) / 2 ** (dj - dp) if s[1] > s[0] else None)
               for (l, h), dp, ru in ref if 0 <= dp < dj and l <= s[0] and s[1] <= h]
        selfref = any(dp == dj and (l, h) == s for (l, h), dp, ru in ref)
        route = ("window" if int(r["window_07"][j]) == 1 else "guarded" if int(r["guarded_07"][j]) == 1
                 else "Levin" if int(r["type"][j]) == 0 else "CC")
        leaves.append(dict(j=j, depth=dj, route=route, below=bool(anc), self_refused=selfref,
                           shallowest=min(a[0] for a in anc) if anc else None,
                           nearest=max(a[0] for a in anc) if anc else None,
                           width_ratios=sorted({float(a[1]) for a in anc if a[1] is not None})))
    nb = sum(x["below"] for x in leaves)
    return dict(id=cid(f"{d['row']}@{d['rtol']}", kind, i), unit=d["unit"], kind=kind, index=i,
                label=c["notify_label"].split("group ")[-1], converged=c["converged"],
                num_regions=c["num_regions"], max_depth=c["max_depth"],
                edge=c["num_edge_rule_refusals"], range=c["num_range_rule_refusals"], n_refused=len(ref),
                n_refused_depth_unknown=n_unknown,
                chase_stops=c.get("num_chase_stops"), chase_stopped_leaves=c.get("num_chase_stopped_leaves"),
                chase_noise_bisections=c.get("num_chase_noise_bisections"),
                n_depth20=len(leaves), n_depth20_below_refused=nb, n_self_refused=sum(x["self_refused"] for x in leaves),
                all_depth20_below=bool(leaves) and nb == len(leaves),
                routes=dict(Counter(x["route"] for x in leaves)),
                routes_below=dict(Counter(x["route"] for x in leaves if x["below"])),
                shallowest_refused_depth=min((x["shallowest"] for x in leaves if x["below"]), default=None),
                shallowest_by_leaf=dict(Counter(str(x["shallowest"]) for x in leaves if x["below"])),
                width_ratios=sorted({w for x in leaves for w in x["width_ratios"]}))


def run(which, X, targets):
    rd = Reader(which)
    out = {}
    byunit = {}
    for key, kind, i in targets:
        byunit.setdefault(key, []).append((kind, i))
    for key in sorted(byunit):
        d, p = rd.record(key)
        for kind, i in byunit[key]:
            a = analyse(d, p, kind, i)
            xc = X[key]["c"][(kind, i)]
            assert all(xc[k] == a[k] for k in ("converged", "num_regions", "max_depth"))
            out[a["id"]] = a
    return out


def summary(res):
    vals = list(res.values())
    return dict(n=len(vals), by_rtol_kind=dict(Counter(v["unit"].split("-")[1] + "-" + v["unit"].split("-")[2] + " " + v["kind"]
                                                        for v in vals)),
                n_with_depth20=sum(v["n_depth20"] > 0 for v in vals), n_without_depth20=sum(v["n_depth20"] == 0 for v in vals),
                n_all_depth20_below=sum(v["all_depth20_below"] for v in vals),
                n_some_depth20_below=sum(0 < v["n_depth20_below_refused"] < v["n_depth20"] for v in vals),
                depth20_leaves=sum(v["n_depth20"] for v in vals),
                depth20_below=sum(v["n_depth20_below_refused"] for v in vals),
                self_refused=sum(v["n_self_refused"] for v in vals),
                refused_depth_unknown=sum(v["n_refused_depth_unknown"] for v in vals),
                stop_acted=sum(bool(v["chase_stops"]) for v in vals),
                all_below_shallowest_depths=dict(Counter(str(v["shallowest_refused_depth"]) for v in vals if v["all_depth20_below"])),
                width_ratio_min_max_n=(lambda w: [w[0], w[-1], len(w)] if w else None)(
                    sorted(w for v in vals for w in v["width_ratios"])))


def main():
    out = Path(sys.argv[1])
    XC, XB = load_extract("07c"), load_extract("07b")
    yc = json.load(open(FIN / "y7_conv.json"))
    assert yc["checkpoints"]["07c"] == RUNS["07c"][1]
    nu = yc["newly_unconverged_07b"]["calls"]
    assert len(nu) == 198
    unc_c = [(k, kd, i) for k, d in XC.items() for (kd, i), c in d["c"].items() if not c["converged"]]
    unc_b = [(k, kd, i) for k, d in XB.items() for (kd, i), c in d["c"].items() if not c["converged"]]
    RC_ = run("07c", XC, unc_c)
    RB_ = run("07b", XB, unc_b)
    # clause (i)
    chase = [v["id"] for v in RC_.values() if v["all_depth20_below"]]
    vacuous = [v["id"] for v in RC_.values() if v["n_depth20"] == 0]
    # clause (ii)
    ids198 = {cid(x["key"], x["kind"], x["index"]): x for x in nu}
    stay = sorted(i for i in ids198 if i in RC_)
    conv198 = sorted(i for i in ids198 if i not in RC_)
    five = {}
    for sid in FIVE:
        x = ids198.get(sid)
        if x is None:
            five[sid] = {"in_198": False}; continue
        cc = XC[x["key"]]["c"][(x["kind"], x["index"])]; cb = XB[x["key"]]["c"][(x["kind"], x["index"])]
        five[sid] = {"in_198": True, "unconverged_07c": not cc["converged"], "num_chase_stops": cc["num_chase_stops"],
                     "equal_07b": all(cc[k] == cb[k] for k in OUTS),
                     "differs_in": [k for k in OUTS if cc[k] != cb[k]]}
    clause2 = (set(stay) == set(FIVE) and all(f_["in_198"] and f_["unconverged_07c"] and f_["num_chase_stops"] == 0
                                               and f_["equal_07b"] for f_ in five.values()))
    # 07b calibration on the 198
    b198 = [RB_[i] for i in ids198]
    calib = dict(n=len(b198), all_below=sum(v["all_depth20_below"] for v in b198),
                 depth20_leaves_of_all_below=sum(v["n_depth20"] for v in b198 if v["all_depth20_below"]),
                 no_depth20=sum(v["n_depth20"] == 0 for v in b198),
                 shallowest=dict(Counter(str(v["shallowest_refused_depth"]) for v in b198 if v["all_depth20_below"])),
                 depth20_leaves=sum(v["n_depth20"] for v in b198))
    # the still-unconverged at 07c, descriptive
    desc = sorted(({k: v[k] for k in ("id", "label", "edge", "range", "num_regions", "max_depth", "n_depth20",
                                       "n_depth20_below_refused", "n_self_refused", "all_depth20_below", "chase_stops",
                                       "chase_stopped_leaves", "routes", "shallowest_refused_depth")}
                   | {"in_198": v["id"] in ids198,
                      "equal_07b": all(XC[f"{v['unit'][7:]}@{v['unit'][2:6]}"]["c"][(v["kind"], v["index"])][k] ==
                                       XB[f"{v['unit'][7:]}@{v['unit'][2:6]}"]["c"][(v["kind"], v["index"])][k] for k in OUTS)}
                   for v in RC_.values()), key=lambda t: t["id"])
    res = dict(
        script_sha256=sha(__file__),
        inputs={str(p): sha(p) for p in (FIN / "extract_07c.jsonl", FIN / "extract_07b.jsonl", FIN / "y7_conv.json",
                                         FIN / "index07c.json", FIN / "index07b.json")},
        checkpoints={k: v[1] for k, v in RUNS.items()},
        clause_i=dict(holds=not chase, chase_calls=chase, unconverged_without_depth20=vacuous,
                      summary_07c=summary(RC_), summary_07b=summary(RB_)),
        clause_ii=dict(holds=clause2, n_198=len(ids198), converge_at_07c=len(conv198), stay_unconverged=stay,
                       five=five, extra_beyond_five=sorted(set(stay) - set(FIVE)), five_missing=sorted(set(FIVE) - set(stay))),
        holds=(not chase) and clause2,
        calibration_07b_198=calib,
        still_unconverged_07c=desc,
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    s = {k: v for k, v in res.items() if k != "still_unconverged_07c"}
    print(json.dumps(s, indent=1, default=str))
    print(f"{'id':26} {'label':6} {'e/r':6} {'regs':>5} {'md':>3} {'d20':>4} {'below':>5} {'self':>4} {'stops':>5} {'in198':5} eq07b")
    for t in desc:
        print(f"{t['id']:26} {t['label'][:6]:6} {t['edge']}/{t['range']:<4} {t['num_regions']:5} {t['max_depth']:3} "
              f"{t['n_depth20']:4} {t['n_depth20_below_refused']:5} {t['n_self_refused']:4} {t['chase_stops']:5} "
              f"{str(t['in_198'])[0]:5} {str(t['equal_07b'])[0]}")


if __name__ == "__main__":
    main()
