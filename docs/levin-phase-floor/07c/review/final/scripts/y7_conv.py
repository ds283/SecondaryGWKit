"""Final review (levin-phase-floor 07c Part C, §C.2): Y7, and convergence against 07b's and 07a's censuses
(descriptive), from my own extracts. Read only but for OUT_JSON. Adapted from 07b's review/final/scripts/u_rescore.py (U7
and the convergence tables); independent of the census agent's scripts.

usage: python -I -B y7_conv.py OUT_JSON

Inputs: final/extract_{07c,07b,07a,07}.jsonl (extract.py's; each header's checkpoint SHA-256 asserted).
- Y7: population = 07's calls with guard_07.locates or .guards (either rtol), paired across rtols by (row, kind, index),
  x_span asserted equal across the pair; r_g = |v_7 - v_8| / e_7 at 07c, exact Fractions from hex; also at 07b and 07a.
  Holds iff every pair has r_g <= 1 or a member unconverged at 07c.
- Convergence (descriptive): call-level `converged` before -> after, 07a -> 07c and 07b -> 07c, by rtol, kind, 07c's
  rate-rule class (untouched / range / edge, by the two refusal counts) and whether the stop acted (num_chase_stops > 0).
  Units' (rows at one rtol) result.total_converged gained and lost, 07b -> 07c and 07a -> 07c, by rtol.
- 07b's newly unconverged calls (07a converged, 07b not), listed for y9.py, with counts by rtol and kind.
"""
import hashlib, json, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
FIN = M / "var/levin-phase-floor-07c/review/final"
SHAS = {"07c": "33d94ce21077cd23f29da65ef967b64676db6722c081b999946bbc91c822a38e",
        "07b": "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253",
        "07a": "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b",
        "07": "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"}


def load(which):
    p = FIN / f"extract_{which}.jsonl"
    out = {}
    with open(p) as f:
        h = json.loads(f.readline())
        assert h["header"] and h["checkpoint_sha256"] == SHAS[which] and h["census"] == which
        for l in f:
            d = json.loads(l)
            d["c"] = {(c["kind"], c["index"]): c for c in d["calls"]}
            assert len(d["c"]) == len(d["calls"])
            out[d["key"]] = d
    return out, hashlib.sha256(p.read_bytes()).hexdigest()


fh = lambda h: Fraction(float.fromhex(h))


def klass(c):
    e, r = c["num_edge_rule_refusals"], c["num_range_rule_refusals"]
    assert isinstance(e, int) and isinstance(r, int)
    return "untouched" if e + r == 0 else ("edge" if e else "range")


def main():
    out = Path(sys.argv[1])
    D, S = {}, {}
    for w in SHAS:
        D[w], S[w] = load(w)
    keys = set(D["07c"])
    assert all(set(D[w]) == keys for w in D)
    assert all(D["07c"][k]["status"] == "done" for k in keys)
    conv_ac, conv_bc, conv_ab = Counter(), Counter(), Counter()
    newly_unconv_b = []
    lists_ac = {"lost": [], "gained": []}
    rows = {"07b->07c": Counter(), "07a->07c": Counter(), "07a->07b": Counter()}
    rows_list = {"07b->07c": {"lost": [], "gained": []}, "07a->07c": {"lost": [], "gained": []}}
    pop = {}
    for k in sorted(keys):
        c_, b_, a_, s_ = D["07c"][k], D["07b"][k], D["07a"][k], D["07"][k]
        rt = c_["rtol"]
        assert set(c_["c"]) == set(b_["c"]) == set(a_["c"]) == set(s_["c"])
        for ck, cc in c_["c"].items():
            cb, ca, c7 = b_["c"][ck], a_["c"][ck], s_["c"][ck]
            assert cc["label"] == cb["label"] == ca["label"] == c7["label"]
            kd = ck[0]
            cls = klass(cc); st = "stopped" if cc["num_chase_stops"] else "not stopped"
            assert isinstance(cc["num_chase_stops"], int)
            conv_ac[(rt, kd, cls, st, bool(ca["converged"]), bool(cc["converged"]))] += 1
            conv_bc[(rt, kd, cls, st, bool(cb["converged"]), bool(cc["converged"]))] += 1
            conv_ab[(rt, kd, bool(ca["converged"]), bool(cb["converged"]))] += 1
            if ca["converged"] and not cb["converged"]:
                newly_unconv_b.append(dict(key=k, unit=b_["unit"], kind=kd, index=ck[1], label=cc["label"]))
            if ca["converged"] != cc["converged"]:
                lists_ac["lost" if ca["converged"] else "gained"].append((k, kd, ck[1], cls, st))
            if c7["locates"] or c7["guards"]:
                pop.setdefault((c_["row"], kd, ck[1]), {})[rt] = (cc, cb, ca)
        for lab, before in (("07b->07c", b_), ("07a->07c", a_), ("07a->07b", None)):
            x, y = (a_, b_) if before is None else (before, c_)
            tx, ty = bool(x["total_converged"]), bool(y["total_converged"])
            rows[lab][(rt, tx, ty)] += 1
            if before is not None and tx != ty:
                rows_list[lab]["lost" if tx else "gained"].append(k)
    y7 = []; unpaired = []
    for (row, kd, i), d in sorted(pop.items()):
        if set(d) != {"1e-7", "1e-8"}:
            unpaired.append((row, kd, i)); continue
        (c7, b7, a7), (c8, b8, a8) = d["1e-7"], d["1e-8"]
        assert json.dumps(c7["x_span"]) == json.dumps(c8["x_span"])
        rg = lambda u, v: abs(fh(u["value"]) - fh(v["value"])) / fh(u["abserr"])
        y7.append(dict(row=row, kind=kd, index=i, rg_c=float(rg(c7, c8)), rg_b=float(rg(b7, b8)), rg_a=float(rg(a7, a8)),
                       rg_c_gt1=rg(c7, c8) > 1, conv=[c7["converged"], c8["converged"]],
                       stops=[c7["num_chase_stops"], c8["num_chase_stops"]]))
    fail = [x for x in y7 if x["rg_c_gt1"] and x["conv"][0] and x["conv"][1]]
    summ = lambda cnt: {" ".join(map(str, kk)): v for kk, v in sorted(cnt.items(), key=lambda t: str(t[0]))}
    nb = Counter((x["key"].split("@")[1], x["kind"]) for x in newly_unconv_b)
    res = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), extracts_sha256=S, checkpoints=SHAS,
        Y7=dict(pairs=len(y7), unpaired=unpaired, holds=not fail, failing=fail,
                n_rg_le1_07c=sum(not x["rg_c_gt1"] for x in y7), gt1_07c=[x for x in y7 if x["rg_c_gt1"]],
                gt1_07b=[x for x in y7 if x["rg_b"] > 1], gt1_07a=[x for x in y7 if x["rg_a"] > 1],
                max_rg_07c=max(y7, key=lambda x: x["rg_c"]), max_rg_07b=max(y7, key=lambda x: x["rg_b"]),
                top5_07c=sorted(y7, key=lambda x: -x["rg_c"])[:5],
                n_pairs_stopped=sum(1 for x in y7 if x["stops"][0] or x["stops"][1])),
        converged_07a_to_07c=summ(conv_ac), converged_07b_to_07c=summ(conv_bc), converged_07a_to_07b=summ(conv_ab),
        calls_07a_to_07c=dict(lost=len(lists_ac["lost"]), gained=len(lists_ac["gained"]),
                              lost_by=summ(Counter((k.split("@")[1], kd, cls, st) for k, kd, i, cls, st in lists_ac["lost"])),
                              gained_by=summ(Counter((k.split("@")[1], kd, cls, st) for k, kd, i, cls, st in lists_ac["gained"])),
                              lost_list=lists_ac["lost"], gained_list=lists_ac["gained"]),
        rows_total_converged={lab: summ(v) for lab, v in rows.items()},
        rows_changed={lab: {"lost": len(v["lost"]), "gained": len(v["gained"]),
                            "lost_by_rtol": dict(Counter(k.split("@")[1] for k in v["lost"])),
                            "gained_by_rtol": dict(Counter(k.split("@")[1] for k in v["gained"])),
                            "lost_list": v["lost"], "gained_list": v["gained"]} for lab, v in rows_list.items()},
        newly_unconverged_07b=dict(n=len(newly_unconv_b), by_rtol_kind={" ".join(k): v for k, v in nb.items()},
                                   calls=newly_unconv_b),
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    short = {k: v for k, v in res.items() if k not in ("newly_unconverged_07b",)}
    short["calls_07a_to_07c"] = {k: v for k, v in res["calls_07a_to_07c"].items() if not k.endswith("_list")}
    short["rows_changed"] = {lab: {k: v for k, v in x.items() if not k.endswith("_list")} for lab, x in res["rows_changed"].items()}
    print(json.dumps(short, indent=1, default=str))
    print("07b newly unconverged:", res["newly_unconverged_07b"]["n"], res["newly_unconverged_07b"]["by_rtol_kind"])


if __name__ == "__main__":
    main()
