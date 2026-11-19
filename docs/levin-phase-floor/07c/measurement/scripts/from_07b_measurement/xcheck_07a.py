"""levin-phase-floor prompt 07a section 6: W6 (the cross-check's r_g re-computed at h over 07's 2,484 pairs) and the
descriptive "disagreement list: 07's cross-check list re-made at h against 06m and against 07, by class". Copied from 07's
census xcheck_07.py (a record, not edited), changed only where marked "prompt 07a": "g" is now h's census (RUN_H, its units
h-..., read by offset as 07's g-...); "m" (the before) is 07's census (BEFORE 07, by state/index_07.json) or 06m's
(BEFORE 06m, by 07's state/index_06m.json); the population is the rows 07 marked (POP 07: a call 07's census records as
locating or guarding, 07's 2,484 pairs) or the rows h marks (POP h). The rule is 07's state/xcheck_rule_20261109.md
(read in 07's records, SHA-256 checked), with g read as h. Checkpoints are SHA-256 checked before and after.
Usage: python -B xcheck_07a.py RUN_H BEFORE(07|06m) POP(07|h) OUT_DIR

07's text follows.

levin-phase-floor prompt 07, beyond section 6 (the user's request at stop 4, relayed verbatim: "Yes, please do the
cross-check on `rtol`s and the list of calls where `06m` and `07` differ by more than the stated error. We can decide on a
sample of references to sort of what these difference really show, later."): the cross-rtol check (part A) and the
disagreement list (part B), by the rule fixed in state/xcheck_rule_20261109.md (SHA-256 checked) before any result. New
in 07. Read only: 07's and 06m's census checkpoints (each SHA-256 checked before and after), state/rows.json,
state/v4_selection.json, 06g's scores. One process. Exact comparisons by Fraction.

Usage: python -B xcheck_07.py OUT_DIR
"""

import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import CHECKPOINT06M_SHA256, RUN06M, WIDER, header, sha256  # noqa: E402
from prov import CHECKPOINT07_SHA256, REC07, RUN07  # noqa: E402  prompt 07a
from prov import H_TREE as H_TREE_07A  # noqa: E402  prompt 07a

RULE = REC07 / "state" / "xcheck_rule_20261109.md"  # prompt 07a: 07's
RULE_SHA256 = "e06a105856046714465fdac0e89d282ee2ed167faca0885b840dadb2f3db8504"
CK07 = RC.CENSUS07 / "checkpoint.jsonl"
CK06 = RUN06M / "checkpoint.jsonl"
INDEX06M = WIDER / "state" / "index_06m.json"
ROWS = WIDER / "state" / "rows.json"
SEL = REC07 / "state" / "v4_selection.json"  # prompt 07a
SCORES06G = H_TREE_07A / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "scores_stage3" / "scores.json"
INF = float("inf")


def Fx(x):
    """A census float ({repr, hex}) as an exact Fraction; None stays None; a non-finite as a float."""
    if x is None:
        return None
    v = float.fromhex(x["hex"] if isinstance(x, dict) else x)
    return Fraction(v) if math.isfinite(v) else v


def ratio(num, den):
    """num/den as a double, num and den exact (Fraction) or non-finite; den <= 0 or non-finite -> inf or 0."""
    if num is None or den is None:
        return None
    if isinstance(num, float) or isinstance(den, float) or den <= 0:
        if isinstance(num, Fraction) and num == 0:
            return 0.0
        if isinstance(den, float) and math.isinf(den) and isinstance(num, Fraction):
            return 0.0
        return INF
    return float(num / den)


def le(num, den):
    """num <= den exactly (num, den Fractions or non-finite floats)."""
    if num is None or den is None:
        return None
    if isinstance(num, float) or isinstance(den, float):
        return float(num) <= float(den)
    return num <= den


def absdiff(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b))
    return abs(a - b)


def qtiles(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    def q(p):
        return xs[min(len(xs) - 1, int(math.floor(p * (len(xs) - 1) + 0.5)))]
    return {"n": len(xs), "q0": xs[0], "q10": q(0.1), "q50": q(0.5), "q90": q(0.9), "q99": q(0.99), "q100": xs[-1],
            "bins": {"<=0.01": sum(1 for x in xs if x <= 0.01), "(0.01,0.1]": sum(1 for x in xs if 0.01 < x <= 0.1),
                     "(0.1,1]": sum(1 for x in xs if 0.1 < x <= 1), "(1,10]": sum(1 for x in xs if 1 < x <= 10),
                     ">10": sum(1 for x in xs if x > 10)}}


def hx(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def main():
    # prompt 07a: the arguments; CK07 is h's census, CK06 and INDEX06M the before's, CKPOP the population's
    run_h, bef, pop, out_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    CK07 = run_h / "checkpoint.jsonl"
    if bef == "07":
        CK06, INDEX06M, CKB_SHA = RUN07 / "checkpoint.jsonl", WIDER / "state" / "index_07.json", CHECKPOINT07_SHA256
    elif bef == "06m":
        CK06, INDEX06M, CKB_SHA = RUN06M / "checkpoint.jsonl", REC07 / "state" / "index_06m.json", CHECKPOINT06M_SHA256
    else:
        raise SystemExit("BEFORE is 07 or 06m")
    if pop not in ("07", "h"):
        raise SystemExit("POP is 07 or h")
    CKPOP = RUN07 / "checkpoint.jsonl" if pop == "07" else CK07
    out_dir.mkdir(parents=True, exist_ok=True)
    if sha256(RULE) != RULE_SHA256:
        raise SystemExit("the rule file has another SHA-256")
    before = {"h": sha256(CK07), bef: sha256(CK06)}
    if before[bef] != CKB_SHA or json.loads(INDEX06M.read_text())["checkpoint_sha256"] != CKB_SHA:
        raise SystemExit(f"a checkpoint has another SHA-256: {before}")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07a: the cross-rtol check at h, before {bef}, population {pop}",
                 [RULE, INDEX06M, ROWS, SEL, SCORES06G])
    hdr.append(f"# input {CK07} sha256 {before['h']} (h's census, read before)")  # prompt 07a
    hdr.append(f"# input {CK06} sha256 {before[bef]} (the before, {bef}, checked before)")  # prompt 07a
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    idx06 = json.loads(INDEX06M.read_text())["units"]
    # V4's set
    v4 = {("1e-7", "1-2174", "group", 1): "z00", ("1e-8", "1-2174", "group", 1): "z12"}
    for r in json.loads(SCORES06G.read_text())["rows"]:
        if r["sid"] in ("z07", "z16", "z18", "z21", "z24", "z28", "z29"):
            p = r["unit"].split("-")
            v4[(r["rtol"], f"{p[3]}-{p[4]}", r["kind"], r["index"])] = r["sid"]
    for c in json.loads(SEL.read_text())["calls"]:
        v4[(c["rtol"], c["row"], c["kind"], c["index"])] = c["sid"]
    # 07: offsets, and the rows with a locating or guarding call
    off07, marked, off = {}, set(), 0
    with open(CK07, "rb") as f:  # prompt 07a: h's offsets, keyed by 07's names g-...
        for line in f:
            uid = line[10:60].decode().split('"', 1)[0]
            off07["g-" + uid.split("-", 1)[1]] = (off, len(line))
            off += len(line)
    marked_by = {}  # prompt 07a: the population's checkpoint marks the rows, and the calls
    n_marked_calls = 0  # prompt 07a: marked calls, counted at each rtol, as 07 counted its population
    with open(CKPOP, "rb") as f:
        for line in f:
            if b'"locates": true' in line or b'"guards": true' in line:
                uid = line[10:60].decode().split('"', 1)[0]
                p = uid.split("-")
                marked.add(f"{p[3]}-{p[4]}")
                d_ = json.loads(line)["data"]
                for k_, key_ in (("group", "calls"), ("analytic", "other_calls")):
                    for c_ in d_.get(key_) or []:
                        g_ = c_.get("guard_07") or {}
                        if g_.get("locates") or g_.get("guards"):
                            marked_by.setdefault(f"{p[3]}-{p[4]}", set()).add((k_, c_["index"]))
                            n_marked_calls += 1

    def load07(uid):
        o, n = off07[uid]
        with open(CK07, "rb") as f:
            f.seek(o)
            r = json.loads(f.read(n))
        assert r["unit"] == "h-" + uid.split("-", 1)[1]  # prompt 07a
        return r["data"]

    def load06(uid):
        e = idx06[uid]
        with open(CK06, "rb") as f:
            f.seek(e["offset"])
            line = f.read(e["length"])
        import hashlib

        if hashlib.sha256(line).hexdigest() != e["line_sha256"]:
            raise SystemExit(f"{uid}: the before's line sha256 differs")
        return json.loads(line)["data"]

    def calls_of(d):
        return {("group", c["index"]): c for c in d.get("calls") or []} | {("analytic", c["index"]): c for c in d.get("other_calls") or []}

    pairs, unpaired, pop_calls = [], [], 0
    units = {}
    for row in sorted(marked):
        U = {}
        for rt in ("1e-7", "1e-8"):
            uid = f"g-{rt}-{row}"
            U[rt] = (load07(uid), load06(uid))
        units[row] = {rt: {"total": [hx((U[rt][0].get("result") or {}).get("total")), hx((U[rt][1].get("result") or {}).get("total"))],
                           "total_abserr": [hx((U[rt][0].get("result") or {}).get("total_abserr")),
                                            hx((U[rt][1].get("result") or {}).get("total_abserr"))],
                           "status": [U[rt][0]["status"], U[rt][1]["status"]]} for rt in U}
        C = {rt: (calls_of(U[rt][0]), calls_of(U[rt][1])) for rt in U}
        keys = set(marked_by.get(row, set()))  # prompt 07a: the population's marked calls (either rtol)
        pop_calls = n_marked_calls
        for k in sorted(keys):
            g7c, g8c = C["1e-7"][0].get(k), C["1e-8"][0].get(k)
            m7c, m8c = C["1e-7"][1].get(k), C["1e-8"][1].get(k)
            ok = all(x is not None for x in (g7c, g8c, m7c, m8c))
            if ok:
                ids = {(x["notify_label"], tuple(e["hex"] for e in x["x_span"])) for x in (g7c, g8c, m7c, m8c)}
                ok = len(ids) == 1
            if not ok:
                unpaired.append({"row": row, "kind": k[0], "index": k[1]})
                continue
            g7, g8, m7, m8 = Fx(g7c["value"]), Fx(g8c["value"]), Fx(m7c["value"]), Fx(m8c["value"])
            eg7, eg8, em7, em8 = Fx(g7c["abserr"]), Fx(g8c["abserr"]), Fx(m7c["abserr"]), Fx(m8c["abserr"])
            d8 = absdiff(g8, m8)
            R = rows[row]
            pr = {"row": row, "kind": k[0], "index": k[1], "label": g7c["notify_label"],
                  "group": g7c.get("group") if k[0] == "group" else g7c["notify_label"],
                  "band": R["band"], "k": R["k"], "k_key": R["k_key"],
                  "r_g": ratio(absdiff(g7, g8), eg7), "r_m": ratio(absdiff(m7, m8), em7),
                  "d8_g": ratio(d8, eg8), "d8_m": ratio(d8, em8),
                  "g_self": le(absdiff(g7, g8), eg7), "m_self": le(absdiff(m7, m8), em7),
                  "agree8": le(d8, max(eg8, em8)),
                  "locates": [bool((g7c.get("guard_07") or {}).get("locates")), bool((g8c.get("guard_07") or {}).get("locates"))],
                  "guards": [bool((g7c.get("guard_07") or {}).get("guards")), bool((g8c.get("guard_07") or {}).get("guards"))],
                  "values": {"g7": hx(g7c["value"]), "g8": hx(g8c["value"]), "m7": hx(m7c["value"]), "m8": hx(m8c["value"])},
                  "abserrs": {"g7": hx(g7c["abserr"]), "g8": hx(g8c["abserr"]), "m7": hx(m7c["abserr"]), "m8": hx(m8c["abserr"])},
                  "depth": {"g7": g7c["max_depth"], "g8": g8c["max_depth"], "m7": m7c["max_depth"], "m8": m8c["max_depth"]},
                  "regions": {"g7": g7c["num_regions"], "g8": g8c["num_regions"], "m7": m7c["num_regions"], "m8": m8c["num_regions"]},
                  "converged": {"g7": g7c["converged"], "g8": g8c["converged"], "m7": m7c["converged"], "m8": m8c["converged"]}}
            if pr["g_self"] and not pr["m_self"] and pr["agree8"]:
                cls = "a"
            elif pr["m_self"] and not pr["g_self"] and pr["agree8"]:
                cls = "b"
            elif pr["g_self"] and pr["m_self"]:
                cls = "c"
            elif not pr["g_self"] and not pr["m_self"]:
                cls = "d"
            else:
                cls = "e"
            pr["class"] = cls
            pr["listed"] = {}
            for rt, (g, m, eg, em) in (("1e-7", (g7, m7, eg7, em7)), ("1e-8", (g8, m8, eg8, em8))):
                dd = absdiff(g, m)
                pr["listed"][rt] = not le(dd, max(eg, em))
                pr[f"dev_{rt}"] = {"over_eg": ratio(dd, eg), "over_em": ratio(dd, em)}
            pr["v4"] = {rt: v4.get((rt, row, k[0], k[1])) for rt in ("1e-7", "1e-8")}
            pairs.append(pr)
    # part A tables
    def tab(sel):
        return {"pairs": len(sel), "r_g<=1": sum(1 for p in sel if p["g_self"]), "r_m<=1": sum(1 for p in sel if p["m_self"])}

    A = {"population_calls": pop_calls, "pairs": len(pairs), "unpaired": unpaired, "overall": tab(pairs),
         "by_kind": {k: tab([p for p in pairs if (p["group"] if p["kind"] == "group" else p["label"]) == k])
                     for k in sorted({p["group"] if p["kind"] == "group" else p["label"] for p in pairs})},
         "by_band": {b: tab([p for p in pairs if p["band"] == b]) for b in sorted({p["band"] for p in pairs})},
         "by_k": {k: tab([p for p in pairs if p["k"] == k]) for k in sorted({p["k"] for p in pairs}, key=float)},
         "dist": {n: qtiles([p[n] for p in pairs]) for n in ("r_g", "r_m", "d8_g", "d8_m")},
         "r_g_gt_1": [p for p in pairs if not p["g_self"]]}
    # part B
    listed = [(rt, p) for p in pairs for rt in ("1e-7", "1e-8") if p["listed"][rt]]
    B = {"n_listed_calls": len(listed),
         "by_class": dict(Counter(p["class"] for _, p in listed)),
         "by_class_rtol": {f"{c}|{rt}": n for (c, rt), n in sorted(Counter((p["class"], rt) for rt, p in listed).items())},
         "by_class_band_k": {f"{c}|{b}|{k}": n for (c, b, k), n in sorted(Counter((p["class"], p["band"], p["k"]) for _, p in listed).items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2])))},
         "by_class_kind": {f"{c}|{g}": n for (c, g), n in sorted(Counter((p["class"], p["group"] if p["kind"] == "group" else p["label"]) for _, p in listed).items())},
         "in_v4": [{"rtol": rt, "row": p["row"], "index": p["index"], "sid": p["v4"][rt], "class": p["class"]} for rt, p in listed if p["v4"][rt]],
         "calls": [{"unit": f"g-{rt}-{p['row']}", "rtol": rt, "index": p["index"], "kind": p["kind"], "group": p["group"],
                    "band": p["band"], "k": p["k"], "class": p["class"],
                    "value_g": p["values"]["g7" if rt == "1e-7" else "g8"], "value_06m": p["values"]["m7" if rt == "1e-7" else "m8"],
                    "abserr_g": p["abserrs"]["g7" if rt == "1e-7" else "g8"], "abserr_06m": p["abserrs"]["m7" if rt == "1e-7" else "m8"],
                    "dev_over_abserr_g": p[f"dev_{rt}"]["over_eg"], "dev_over_abserr_06m": p[f"dev_{rt}"]["over_em"],
                    "depth_g": p["depth"]["g7" if rt == "1e-7" else "g8"], "depth_06m": p["depth"]["m7" if rt == "1e-7" else "m8"],
                    "regions_g": p["regions"]["g7" if rt == "1e-7" else "g8"], "regions_06m": p["regions"]["m7" if rt == "1e-7" else "m8"],
                    "r_g": p["r_g"], "r_m": p["r_m"], "d8_g": p["d8_g"], "d8_m": p["d8_m"], "v4": p["v4"][rt]}
                   for rt, p in listed]}
    # the row totals of the rows the listed calls move
    tot = []
    for rt, p in listed:
        u = units[p["row"]][rt]
        tg, tm = Fx(u["total"][0]), Fx(u["total"][1])
        eg, em = Fx(u["total_abserr"][0]), Fx(u["total_abserr"][1])
        dd = absdiff(tg, tm)
        tot.append({"unit": f"g-{rt}-{p['row']}", "index": p["index"], "class": p["class"], "total_g": u["total"][0],
                    "total_06m": u["total"][1], "total_abserr_g": u["total_abserr"][0], "total_abserr_06m": u["total_abserr"][1],
                    "dtotal_over_abserr_g": ratio(dd, eg), "dtotal_over_abserr_06m": ratio(dd, em),
                    "dtotal_rel": ratio(dd, abs(tm)) if isinstance(tm, Fraction) else None})
    B["row_totals"] = tot
    B["row_totals_summary"] = {"rows": len({t["unit"] for t in tot}),
                               "dtotal_over_abserr_06m": qtiles([t["dtotal_over_abserr_06m"] for t in tot]),
                               "dtotal_over_abserr_g": qtiles([t["dtotal_over_abserr_g"] for t in tot])}
    after = {"h": sha256(CK07), bef: sha256(CK06)}  # prompt 07a
    doc = {"header": hdr, "rule": str(RULE), "rule_sha256": RULE_SHA256, "checkpoints_before": before,
           "checkpoints_after": after, "checkpoints_unchanged": before == after, "A": A, "B": B, "pairs": pairs}
    (out_dir / "xcheck.json").write_text(json.dumps(doc, indent=1, default=str))
    for l in hdr:
        print(l)
    print(json.dumps({"A": {k: v for k, v in A.items() if k not in ("r_g_gt_1",)}, "B": {k: v for k, v in B.items() if k not in ("calls", "row_totals")},
                      "unchanged": before == after}, indent=1, default=str))
    print(f"wrote {out_dir / 'xcheck.json'} sha256 {sha256(out_dir / 'xcheck.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
