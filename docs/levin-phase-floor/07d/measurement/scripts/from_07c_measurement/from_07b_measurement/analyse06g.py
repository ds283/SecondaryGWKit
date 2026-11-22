"""levin-phase-floor prompt 06g section 5: the census's scores V2, V5, V6 and V7 and the descriptive items (V8 is
v8_06g.py's, out/v8.md, and is not re-scored here). New in 06g; its loaders follow 06h's analyse06h.py and
nearsingular06h.py (kept in state/scripts_06h_reference/). Every rule is state/predictions.md's (Part A the
prompt's text, Part B its readings). Reads, read only: this census's checkpoint and region pickles (RUN_DIR),
06h's census checkpoint and region pickles (prov.RUN06H; checkpoint SHA-256 checked), 06a's run's 1e-7
checkpoint lines (the 720) and state/rows.json. The pickles are read by at most six processes (spans per call,
06h's returned-leaf trust numbers, 06h's E per fallback leaf, g's b per row); every record's SHA-256 is checked.
Writes out/analyse06g.txt (via stdout), out/v2_rows.csv.gz (every scored row, both readings),
out/v5_depth_max_calls.csv (every g call outside the 720 at depth_max) and out/descriptive_06g.json.
Usage: python -B analyse06g.py RUN_DIR"""
import collections
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import pickle
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06H_SHA256, RUN06A, RUN06H, WIDER, header, sha256  # noqa: E402

RTOLS = ("1e-7", "1e-8")
DEPTH_MAX = 20
BANDS = ("r<1/16", "1/16<=r<1/4", "1/4<=r<1", "r>=1")


def H(x):
    if x is None:
        return None
    return float.fromhex(x["hex"] if isinstance(x, dict) else x)


def hexof(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def q(vals, ps=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = sorted(vals)
    if not v:
        return "none"
    return ", ".join(f"p{p*100:g} {v[min(len(v)-1, int(p*(len(v)-1)))]:.3g}" for p in ps)


CALL_KEYS = ("index", "notify_label", "group", "converged", "max_depth", "num_regions", "num_subregion_solves", "atol",
             "rtol", "n_at_depth_max", "num_untrusted_regions", "n_phase_limited_regions", "premise_checks",
             "premise_failed", "node_limited", "has_cc_leaf", "premise_06g", "values_06g")


def slim_call(kind, c):
    out = {k: c.get(k) for k in CALL_KEYS}
    out["kind"] = kind
    out["value"] = hexof(c["value"])
    out["abserr"] = hexof(c["abserr"])
    out["x_span"] = [hexof(c["x_span"][0]), hexof(c["x_span"][1])]
    for k in ("node_error_signed_sum", "node_error_abs_sum", "abserr_node_signed", "abserr_eprime_sum"):
        out[k] = hexof(c.get(k))
    ck = c.get("checked_06g")
    if isinstance(ck, dict) and "classes_all" in ck:
        out["checked"] = {k: ck[k] for k in ("n_checked", "n_checked_returned", "n_b_untrusted", "n_b_untrusted_returned",
                                             "classes_all", "classes_returned", "n_noise_clause_trusted")}
    else:
        out["checked"] = None
    f = c.get("cc_floor") or {}
    out["cc_abs_E_06h"] = hexof((f.get("sums_all_cc_leaves") or {}).get("abs_E")) if "sums_all_cc_leaves" in f else None
    out["cc_n"] = f.get("n_cc_leaves")
    return out


def load(run_dir):
    out = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            res = d.get("result")
            an = (d.get("analytic_06h") or {}).get("h") or {}
            post = sum((d.get(k_) or 0.0) for k_ in ("scan_s", "cc_floor_s", "declared_06h_s", "checked_06g_s"))
            post += sum(((d.get(k_) or {}).get("seconds") or 0.0) for k_ in ("w3", "deep_lines", "analytic_06h"))
            out[(d["rtol"], rowkey(r["unit"]))] = {
                "unit": r["unit"], "status": d["status"], "wall_s": d["wall_s"], "post_s": post, "row": d["row"],
                "total": None if res is None else hexof(res["total"]),
                "total_abserr": None if res is None else hexof(res["total_abserr"]),
                "total_converged": None if res is None else res["total_converged"],
                "oracle": hexof(an.get("value")), "oracle_abserr": hexof(an.get("abserr")), "oracle_converged": an.get("converged"),
                "pickle": (d.get("pickle"), d.get("pickle_offset"), d.get("pickle_length"), d.get("pickle_record_sha256")),
                "calls": [slim_call("group", c) for c in d["calls"]] + [slim_call("analytic", c) for c in d["other_calls"]],
            }
    return out


def scan06a_sp():
    sp = set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    return sp


def match(ug, uh):
    bi = {(c["kind"], c["index"]): c for c in uh["calls"]}
    out, probs = [], 0
    for cg in ug["calls"]:
        ch = bi.get((cg["kind"], cg["index"]))
        if ch is None or ch["notify_label"] != cg["notify_label"] or ch["x_span"] != cg["x_span"]:
            probs += 1
            continue
        out.append((cg, ch))
    probs += len(uh["calls"]) - len(out)
    return out, probs


def r_class(r, d1, phi):
    if r < 1.0 / 16.0:
        band = "r<1/16"
    elif r < 0.25:
        band = "1/16<=r<1/4"
    elif r < 1.0:
        band = "1/4<=r<1"
    else:
        band = "r>=1"
    return f"{band}|{'d1<=phi' if d1 <= phi else 'd1>phi'}"


def read_pickle_file(args):
    """One region pickle: per unit, per call (kind, index) a digest of the returned leaves' spans; for 06h, the
    returned leaves' trust classes and each fallback leaf's E (reproduced); for g at 1e-7, b."""
    path, entries, which = args
    res = {}
    with open(path, "rb") as fh:
        for uid, off, ln, sha in entries:
            fh.seek(off)
            blob = fh.read(ln)
            if hashlib.sha256(blob).hexdigest() != sha:
                raise RuntimeError(f"{uid}: pickle record sha256 differs")
            rec = pickle.loads(blob)
            u = {"spans": {}, "classes_ret": collections.Counter(), "E": {}, "b": None}
            for kind, key in (("group", "regions"), ("analytic", "other_regions")):
                for a in rec[key]:
                    hh = hashlib.sha1(a["start"].tobytes() + a["end"].tobytes()).hexdigest()
                    u["spans"][(kind, int(a["index"]))] = hh
                    if which == "06h":
                        for r_, d1, phi in zip(a["convergence_ratio"], a["delta1"], a["phase_err"]):
                            if not math.isnan(d1):
                                u["classes_ret"][r_class(float(r_), float(d1), float(phi))] += 1
            if which == "06h":
                for t in rec.get("cc_floor_terms") or []:
                    ok = t["captured"] & t["reproduces"]
                    u["E"][int(t["index"])] = (math.fsum(t["E"][ok].tolist()), int(ok.sum()), int(len(ok)))
            if which == "g" and rec.get("result_full") is not None:
                u["b"] = float.fromhex(rec["result_full"]["b"]["hex"])
            u["classes_ret"] = dict(u["classes_ret"])
            res[uid] = u
    return res


def read_pickles(run_dir, units, which, procs=6):
    by_file = collections.defaultdict(list)
    for u in units.values():
        p, off, ln, sha = u["pickle"]
        if p is not None:
            by_file[p].append((u["unit"], off, ln, sha))
    jobs = [(str(Path(run_dir) / p), ents, which) for p, ents in sorted(by_file.items())]
    out = {}
    with multiprocessing.get_context("spawn").Pool(processes=procs) as pool:
        for r in pool.imap_unordered(read_pickle_file, jobs):
            out.update(r)
    return out


def verdict(fail, undecided=False):
    return "fails" if fail else ("undecided" if undecided else "holds")


def main():
    run = Path(sys.argv[1])
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06g: V2, V5, V6, V7 and the descriptive items",
                    [run / "checkpoint.jsonl", WIDER / "state" / "predictions.md", WIDER / "state" / "rows.json"]):
        print(l)
    ck_h = sha256(RUN06H / "checkpoint.jsonl")
    print(f"# 06h's checkpoint sha256 {ck_h} (expected {CHECKPOINT06H_SHA256}: {ck_h == CHECKPOINT06H_SHA256})")
    assert ck_h == CHECKPOINT06H_SHA256
    sp = scan06a_sp()
    print(f"06a's stationary-point rows: {len(sp)} (must be 720)")
    assert len(sp) == 720
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]}
    G = load(run)
    Hh = load(RUN06H)
    print(f"units g {len(G)}, 06h {len(Hh)}; status g {dict(collections.Counter(u['status'] for u in G.values()))}, "
          f"06h {dict(collections.Counter(u['status'] for u in Hh.values()))}")
    PG = read_pickles(run, G, "g")
    PH = read_pickles(RUN06H, Hh, "06h")
    print(f"pickle records read: g {len(PG)}, 06h {len(PH)} (every SHA-256 checked)")
    desc = {}

    # ------------------------------------------------------------------ V2
    print("\n== V2 — the rows agree (Part A V2; Part B rule 4)")
    v2rows = []
    v2 = {}
    for rtol in RTOLS:
        for reading, vk, ek in (("total", "total", "total_abserr"), ("oracle", "oracle", "oracle_abserr")):
            n = nonfin = noresult = 0
            viol = []
            ratios = []
            moved = 0
            for (rt, rk), ug in G.items():
                if rt != rtol or rk in sp:
                    continue
                uh = Hh[(rt, rk)]
                if ug["status"] != "done" or uh["status"] != "done" or ug[vk] is None or uh[vk] is None:
                    noresult += 1
                    continue
                eg, eh = H(ug[ek]), H(uh[ek])
                if eg is None or eh is None or not (math.isfinite(eg) and math.isfinite(eh)):
                    nonfin += 1
                    continue
                n += 1
                vg, vh = H(ug[vk]), H(uh[vk])
                moved += ug[vk] != uh[vk]
                d = abs(Fraction(vg) - Fraction(vh))
                bound = Fraction(eh) + Fraction(eg)
                ratio = float(d / bound) if bound else (0.0 if d == 0 else math.inf)
                ratios.append(ratio)
                if d > bound:
                    viol.append((ratio, rk, ug[vk], uh[vk], ug[ek], uh[ek]))
                v2rows.append([rtol, reading, rk, ug[vk], uh[vk], ug[ek], uh[ek], ratio, d > bound])
            viol.sort(reverse=True)
            v2[(rtol, reading)] = {"scored": n, "nonfinite": nonfin, "no_result": noresult, "violations": len(viol),
                                   "moved": moved, "max_ratio": max(ratios) if ratios else None}
            print(f"   {rtol} {reading}: scored {n} rows, non-finite {nonfin}, no result {noresult}; moved {moved}; "
                  f"violations {len(viol)}; ratio |d|/(e_06h+e_g): {q(ratios)}")
            for ratio, rk, a, b, c, d_ in viol[:100]:
                print(f"      VIOLATION {rk}: ratio {ratio:.4g}; g {a} (abserr {c}); 06h {b} (abserr {d_})")
    v2_fail = any(v["violations"] for v in v2.values())
    print(f"   V2: {verdict(v2_fail).upper()} (1e-7: {verdict(v2[('1e-7','total')]['violations'] or v2[('1e-7','oracle')]['violations'])}; "
          f"1e-8: {verdict(v2[('1e-8','total')]['violations'] or v2[('1e-8','oracle')]['violations'])})")
    with gzip.open(WIDER / "out" / "v2_rows.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rtol", "reading", "row", "value_g", "value_06h", "abserr_g", "abserr_06h", "ratio", "violates"])
        w.writerows(v2rows)
    desc["V2"] = {f"{k[0]} {k[1]}": v for k, v in v2.items()}

    # ------------------------------------------------------------------ V5
    print("\n== V5 — the cap (Part A V5; Part B rule 5)")
    v5 = {}
    with open(WIDER / "out" / "v5_depth_max_calls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rtol", "row", "kind", "index", "block_or_group", "max_depth", "n_at_depth_max", "converged", "abserr",
                    "num_regions", "num_untrusted_regions", "06h_max_depth", "06h_converged"])
        for rtol in RTOLS:
            calls_cap = []
            calls_cap_sp = 0
            over = [u["unit"] for (rt, rk), u in G.items() if rt == rtol and u["status"] == "over budget" and rk not in sp]
            for (rt, rk), ug in sorted(G.items()):
                if rt != rtol:
                    continue
                hb = {(c["kind"], c["index"]): c for c in Hh[(rt, rk)]["calls"]}
                for c in ug["calls"]:
                    if (c["n_at_depth_max"] or 0) > 0:
                        if rk in sp:
                            calls_cap_sp += 1
                            continue
                        calls_cap.append((rk, c))
                        ch = hb.get((c["kind"], c["index"])) or {}
                        w.writerow([rtol, rk, c["kind"], c["index"], c["group"] if c["kind"] == "group" else c["notify_label"],
                                    c["max_depth"], c["n_at_depth_max"], c["converged"], c["abserr"], c["num_regions"],
                                    c["num_untrusted_regions"], ch.get("max_depth"), ch.get("converged")])
            h_cap = sum(1 for (rt, rk), u in Hh.items() if rt == rtol and rk not in sp for c in u["calls"] if (c.get("max_depth") or 0) >= DEPTH_MAX)
            v5[rtol] = {"calls_at_cap": len(calls_cap), "rows_at_cap": len({rk for rk, _ in calls_cap}), "over_budget": len(over),
                        "calls_at_cap_in_720": calls_cap_sp, "06h_calls_at_cap": h_cap,
                        "by_kind": dict(collections.Counter(c["kind"] for _, c in calls_cap)),
                        "converged_at_cap": sum(1 for _, c in calls_cap if c["converged"]),
                        "by_group": dict(collections.Counter(c["group"] if c["kind"] == "group" else c["notify_label"] for _, c in calls_cap).most_common())}
            print(f"   {rtol}: calls outside the 720 at depth_max under g {len(calls_cap)} in {v5[rtol]['rows_at_cap']} rows "
                  f"(by kind {v5[rtol]['by_kind']}; converged {v5[rtol]['converged_at_cap']}; by group {v5[rtol]['by_group']}); "
                  f"over-budget units outside the 720 {len(over)}; inside the 720: {calls_cap_sp} calls at the cap; 06h outside the 720: {h_cap} calls at the cap")
    v5_fail = any(v["calls_at_cap"] or v["over_budget"] for v in v5.values())
    print(f"   V5: {verdict(v5_fail).upper()} (1e-7: {verdict(v5['1e-7']['calls_at_cap'] or v5['1e-7']['over_budget'])}; "
          f"1e-8: {verdict(v5['1e-8']['calls_at_cap'] or v5['1e-8']['over_budget'])}); every call in out/v5_depth_max_calls.csv")
    desc["V5"] = v5

    # ------------------------------------------------------------------ V6, V7
    def find(rtol, rk, index):
        ug, uh = G[(rtol, rk)], Hh[(rtol, rk)]
        pairs, _ = match(ug, uh)
        for cg, ch in pairs:
            if cg["index"] == index:
                return ug, uh, cg, ch
        raise KeyError((rtol, rk, index))

    print("\n== V6 — Z2's call, g-1e-8-0-33995 call 5 (Part A V6; Part B rule 6)")
    ug, uh, cg, ch = find("1e-8", "0-33995", 5)
    ending = "converged" if cg["converged"] else ("node_limited" if cg["node_limited"] else "neither")
    v6_holds = cg["max_depth"] < DEPTH_MAX and ending != "neither"
    e06h = PH[uh["unit"]]["E"].get(5)
    print(f"   label {cg['notify_label']} group {cg['group']} span {cg['x_span']}")
    print(f"   g: ending {ending}; converged {cg['converged']}; node_limited {cg['node_limited']}; depth {cg['max_depth']}; regions {cg['num_regions']}; "
          f"solves {cg['num_subregion_solves']}; abserr {H(cg['abserr']):.6e}; request {max(float(cg['atol']), float(cg['rtol']) * abs(H(cg['value']))):.6e}; "
          f"sum E {H(cg['node_error_signed_sum']):.6e}; sum|E| {H(cg['node_error_abs_sum']):.6e}; |sum E| {H(cg['abserr_node_signed']):.6e}; e' sum {H(cg['abserr_eprime_sum']):.6e}; "
          f"untrusted returned {cg['num_untrusted_regions']}; at cap {cg['n_at_depth_max']}; checked {cg['checked'] and {k: cg['checked'][k] for k in ('n_checked', 'n_b_untrusted')}}")
    print(f"   06h: converged {ch['converged']}; depth {ch['max_depth']}; regions {ch['num_regions']}; solves {ch['num_subregion_solves']}; abserr {H(ch['abserr']):.6e}; "
          f"request {max(float(ch['atol']), float(ch['rtol']) * abs(H(ch['value']))):.6e}; sum|E| (cc_floor, reproduced leaves) {H(ch['cc_abs_E_06h']):.6e}; "
          f"sum E (pickle fsum, reproduced/n) {e06h}")
    print(f"   V6: {verdict(not v6_holds).upper()}")
    desc["V6"] = {"holds": v6_holds, "ending": ending, "g": {k: cg[k] for k in ("converged", "node_limited", "max_depth", "num_regions", "num_subregion_solves", "abserr", "node_error_signed_sum", "node_error_abs_sum", "abserr_node_signed")},
                  "06h": {"converged": ch["converged"], "max_depth": ch["max_depth"], "num_regions": ch["num_regions"], "abserr": ch["abserr"], "sum_abs_E": ch["cc_abs_E_06h"], "sum_E_pickle": e06h}}

    print("\n== V7 — H4's call, g-1e-8-3-1072 call 13 (Part A V7; Part B rule 7)")
    ug, uh, cg, ch = find("1e-8", "3-1072", 13)
    with open(run / ug["pickle"][0], "rb") as fh:
        fh.seek(ug["pickle"][1])
        blob = fh.read(ug["pickle"][2])
    assert hashlib.sha256(blob).hexdigest() == ug["pickle"][3]
    rec = pickle.loads(blob)
    det = next(x for x in rec["premise_06g"] if x["index"] == 13)["checks"]
    v7_holds = bool(cg["converged"])
    for name, c in (("g", cg), ("06h", ch)):
        req = max(float(c["atol"]), float(c["rtol"]) * abs(H(c["value"])))
        print(f"   {name}: {c['notify_label']}; converged {c['converged']}; abserr {H(c['abserr']):.6e}; request {req:.6e}; regions {c['num_regions']}; "
              f"depth {c['max_depth']}; premise checks {c['premise_checks']}, failed {c['premise_failed']}" +
              (f"; premise_06g {c['premise_06g']}" if name == "g" else ""))
    for x in det:
        print(f"      g premise check: result {x['result']}; split could meet budget {x['split_could_meet_budget']}; halving {x['halving']}; "
              f"floors {float.fromhex(x['phase_err']):.4e} | {float.fromhex(x['phase_err_L']):.4e} + {float.fromhex(x['phase_err_R']):.4e}")
    print(f"   V7: {verdict(not v7_holds).upper()}")
    desc["V7"] = {"holds": v7_holds, "g_premise_checks": det}

    # ------------------------------------------------------------------ descriptive
    for rtol in RTOLS:
        print(f"\n== descriptive, rtol {rtol}")
        pairs_all = []
        for (rt, rk), ug in G.items():
            if rt != rtol:
                continue
            uh = Hh[(rt, rk)]
            pr, _ = match(ug, uh)
            pairs_all += [(rk, ug, uh, cg, ch) for cg, ch in pr]
        out_ = [(rk, ug, uh, cg, ch) for rk, ug, uh, cg, ch in pairs_all if rk not in sp]
        d_ = {}
        # the values
        cls_vals = collections.defaultdict(list)
        n_over1 = collections.Counter()
        abs_moves = collections.defaultdict(list)
        for rk, ug, uh, cg, ch in out_:
            eg = H(cg["abserr"])
            if not (eg and math.isfinite(eg)):
                continue
            same = PG[ug["unit"]]["spans"].get((cg["kind"], cg["index"])) == PH[uh["unit"]]["spans"].get((ch["kind"], ch["index"]))
            nq = (cg.get("values_06g") or {}).get("n_quarters") or 0
            c_ = "different partition" if not same else ("same partition, quarters returned" if nq else "same partition, no quarters")
            r_ = abs(H(cg["value"]) - H(ch["value"])) / eg
            cls_vals[c_].append(r_)
            n_over1[c_] += r_ > 1.0
            if same:
                eh = H(ch["abserr"])
                if eh:
                    abs_moves["CC leaf" if cg["has_cc_leaf"] else "no CC leaf"].append(eg / eh)
        print("   the values, |v_g - v_06h| / abserr_g per call (outside the 720), by cause:")
        for c_ in ("different partition", "same partition, quarters returned", "same partition, no quarters"):
            v = cls_vals[c_]
            print(f"      {c_}: {len(v)} calls; ratio {q(v)}; > 1: {n_over1[c_]}; exactly 0: {sum(1 for x in v if x == 0)}")
        print("   abserr_g / abserr_06h on the same partition, by CC leaf (sum E in the error):")
        for k, v in abs_moves.items():
            print(f"      {k}: {len(v)} calls; {q(v)}; unchanged {sum(1 for x in v if x == 1.0)}")
        d_["values"] = {c_: {"n": len(v), "over1": n_over1[c_], "p50": sorted(v)[len(v)//2] if v else None, "max": max(v) if v else None} for c_, v in cls_vals.items()}
        # slow leaves
        all_c, ret_c, h_ret = collections.Counter(), collections.Counter(), collections.Counter()
        for (rt, rk), ug in G.items():
            if rt != rtol:
                continue
            for c in ug["calls"]:
                if c["checked"]:
                    all_c.update(c["checked"]["classes_all"])
                    ret_c.update(c["checked"]["classes_returned"])
            h_ret.update(PH[Hh[(rt, rk)]["unit"]]["classes_ret"])
        def table(cnt):
            return " | ".join(f"{b}: {cnt.get(b + '|d1>phi', 0)}/{cnt.get(b + '|d1<=phi', 0)}" for b in BANDS) + f" | total {sum(cnt.values())}"
        print("   slow leaves, test 3's classes as (d1 > phi)/(d1 <= phi), all rows:")
        print(f"      g, every checked leaf (returned or bisected): {table(all_c)}")
        print(f"      g, returned checked leaves: {table(ret_c)}")
        print(f"      06h, returned checked leaves (06h did not record bisected ones): {table(h_ret)}")
        b_calls = [(cg, ch) for rk, ug, uh, cg, ch in out_ if cg["checked"] and cg["checked"]["n_b_untrusted"] >= 1]
        dreg = sum(cg["num_regions"] - ch["num_regions"] for cg, ch in b_calls)
        dsol = sum(cg["num_subregion_solves"] - ch["num_subregion_solves"] for cg, ch in b_calls)
        print(f"      calls outside the 720 with a (b) leaf: {len(b_calls)} (group {sum(1 for c, _ in b_calls if c['kind']=='group')}, analytic {sum(1 for c, _ in b_calls if c['kind']=='analytic')}); "
              f"regions added {dreg}, solves added {dsol}; of them converged under g {sum(1 for c, _ in b_calls if c['converged'])}")
        d_["slow_leaves"] = {"g_all": dict(all_c), "g_returned": dict(ret_c), "06h_returned": dict(h_ret), "b_calls": len(b_calls), "regions_added": dreg, "solves_added": dsol}
        # the node sum
        ns = []
        n_cc = n_zero = 0
        for rk, ug, uh, cg, ch in out_:
            ea, es, eg = H(cg["node_error_abs_sum"]), H(cg["node_error_signed_sum"]), H(cg["abserr"])
            if cg["has_cc_leaf"]:
                n_cc += 1
            if ea is None or not eg or not math.isfinite(eg) or ea < 0.01 * eg:
                continue
            if es == 0:
                n_zero += 1
                continue
            ns.append(ea / abs(es))
        print(f"   the node sum: calls with a CC leaf {n_cc}; with sum|E| >= 1% of abserr_g: {len(ns) + n_zero} (sum E = 0 exactly: {n_zero}); "
              f"sum|E| / |sum E|: {q(ns)}")
        d_["node_sum"] = {"n_cc": n_cc, "n_material": len(ns) + n_zero, "n_zero": n_zero, "p50": sorted(ns)[len(ns)//2] if ns else None, "max": max(ns) if ns else None}
        # the exit
        fh_, fg_, chh, chg = 0, 0, 0, 0
        changed = collections.Counter()
        for rk, ug, uh, cg, ch in out_:
            fh = (ch["premise_checks"] or 0) - (ch["premise_failed"] or 0)
            fg = (cg["premise_06g"] or {}).get("n_fired") or 0
            fh_ += fh
            fg_ += fg
            chh += ch["premise_checks"] or 0
            chg += (cg["premise_06g"] or {}).get("n_checks") or 0
            if fh != fg:
                changed["more firings" if fg > fh else "fewer firings"] += 1
                changed[f"{cg['kind']}"] += 1
        print(f"   the exit (outside the 720): firings 06h {fh_} -> g {fg_}; premise checks 06h {chh} -> g {chg}; calls whose firing count changed: {dict(changed)}")
        d_["exit"] = {"firings_06h": fh_, "firings_g": fg_, "checks_06h": chh, "checks_g": chg, "changed": dict(changed)}
        # cost
        for lab, keep in (("all rows", lambda rk: True), ("outside the 720", lambda rk: rk not in sp)):
            for kind in ("group", "analytic"):
                rg = sum(cg["num_regions"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                rh = sum(ch["num_regions"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                sg = sum(cg["num_subregion_solves"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                sh = sum(ch["num_subregion_solves"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                print(f"   cost, {lab}, {kind}: regions 06h {rh} -> g {rg} (x{rg/rh:.3f}); solves 06h {sh} -> g {sg} (x{sg/sh:.3f})")
                d_[f"cost {lab} {kind}"] = {"regions_06h": rh, "regions_g": rg, "solves_06h": sh, "solves_g": sg}
        wg = sum(u["wall_s"] for (rt, rk), u in G.items() if rt == rtol)
        wh = sum(u["wall_s"] for (rt, rk), u in Hh.items() if rt == rtol)
        pg = sum(u["post_s"] for (rt, rk), u in G.items() if rt == rtol)
        ph = sum(u["post_s"] for (rt, rk), u in Hh.items() if rt == rtol)
        mg = max(((u["wall_s"], u["unit"]) for (rt, rk), u in G.items() if rt == rtol))
        mh = max(((u["wall_s"], u["unit"]) for (rt, rk), u in Hh.items() if rt == rtol))
        print(f"   cost, unit wall (wall_s summed): 06h {wh/3600:.3f} h -> g {wg/3600:.3f} h (x{wg/wh:.3f}); largest unit 06h {mh[0]:.1f} s ({mh[1]}) -> g {mg[0]:.1f} s ({mg[1]}); "
              f"post-timed seconds summed 06h {ph/3600:.3f} h -> g {pg/3600:.3f} h")
        d_["wall"] = {"unit_wall_h_06h": wh / 3600, "unit_wall_h_g": wg / 3600, "largest_06h": mh, "largest_g": mg, "post_h_06h": ph / 3600, "post_h_g": pg / 3600}
        for lab, keep in (("outside the 720", lambda rk: rk not in sp), ("the 720", lambda rk: rk in sp)):
            for kind in ("group", "analytic"):
                hg = collections.Counter(cg["max_depth"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                hh = collections.Counter(ch["max_depth"] for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                cgc = sum(1 for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind and cg["converged"])
                chc = sum(1 for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind and ch["converged"])
                n_ = sum(1 for rk, ug, uh, cg, ch in pairs_all if keep(rk) and cg["kind"] == kind)
                print(f"   depth and convergence, {lab}, {kind} calls ({n_}): converged 06h {chc} -> g {cgc}; depth histogram 06h {dict(sorted(hh.items()))}")
                print(f"      g {dict(sorted(hg.items()))}")
                d_[f"depth {lab} {kind}"] = {"converged_06h": chc, "converged_g": cgc, "n": n_, "hist_06h": dict(sorted(hh.items())), "hist_g": dict(sorted(hg.items()))}
            tg = sum(1 for (rt, rk), u in G.items() if rt == rtol and keep(rk) and u["total_converged"] is False)
            th = sum(1 for (rt, rk), u in Hh.items() if rt == rtol and keep(rk) and u["total_converged"] is False)
            print(f"   rows with total_converged False, {lab}: 06h {th} -> g {tg}")
            d_[f"rows unconverged {lab}"] = {"06h": th, "g": tg}
        # the 720 rows
        moved = sum(1 for (rt, rk), u in G.items() if rt == rtol and rk in sp and u["total"] != Hh[(rt, rk)]["total"])
        omoved = sum(1 for (rt, rk), u in G.items() if rt == rtol and rk in sp and u["oracle"] != Hh[(rt, rk)]["oracle"])
        lost = sum(1 for rk, ug, uh, cg, ch in pairs_all if rk in sp and ch["converged"] and not cg["converged"])
        gained = sum(1 for rk, ug, uh, cg, ch in pairs_all if rk in sp and not ch["converged"] and cg["converged"])
        print(f"   the 720 rows: totals moved {moved}; oracles moved {omoved}; calls lost {lost}, gained {gained}")
        d_["the 720"] = {"totals_moved": moved, "oracles_moved": omoved, "calls_lost": lost, "calls_gained": gained}
        desc[rtol] = d_

    # ------------------------------------------------------------------ the near-singular class (06h's rule)
    print("\n== the near-singular class (06h's rule, |K| / (k + q c_s + r c_s) <= 1e-3, the minimum over the three minus groups)")
    near = []
    for key, r in rows.items():
        b = PG[G[("1e-7", key)]["unit"]]["b"]
        if b is None:
            continue
        k, qq, rr = float(r["k"]), float(r["q"]), float(r["r"])
        cs = math.sqrt((1.0 - b) / (1.0 + b) / 3.0)
        den = k + qq * cs + rr * cs
        ks = {"(+,+,-)": k + qq * cs - rr * cs, "(+,-,+)": k - qq * cs + rr * cs, "(+,-,-)": k - qq * cs - rr * cs}
        gname, K = min(ks.items(), key=lambda t: abs(t[1]))
        if abs(K) / den <= 1e-3:
            near.append((key, abs(K) / den, gname))
    print(f"   rows: {len(near)} (06h's census: 435)")
    ns = {}
    for rtol in RTOLS:
        an_lost = an_gained = gr_lost = gr_gained = 0
        o_conv = [0, 0]
        ratios = []
        for key, _rel, _g in near:
            ug, uh = G[(rtol, key)], Hh[(rtol, key)]
            for cg, ch in match(ug, uh)[0]:
                lo = ch["converged"] and not cg["converged"]
                ga = cg["converged"] and not ch["converged"]
                if cg["kind"] == "analytic":
                    an_lost += lo
                    an_gained += ga
                else:
                    gr_lost += lo
                    gr_gained += ga
            o_conv[0] += bool(uh["oracle_converged"])
            o_conv[1] += bool(ug["oracle_converged"])
            if ug["oracle"] and uh["oracle"] and H(ug["oracle_abserr"]) is not None:
                bound = H(ug["oracle_abserr"]) + H(uh["oracle_abserr"])
                ratios.append(abs(H(ug["oracle"]) - H(uh["oracle"])) / bound if bound else 0.0)
        in_sp = sum(1 for key, _, _ in near if key in sp)
        print(f"   {rtol}: analytic calls lost {an_lost}, gained {an_gained}; group calls lost {gr_lost}, gained {gr_gained}; "
              f"oracle converged 06h {o_conv[0]} -> g {o_conv[1]} of {len(near)}; oracle |d|/(e_06h+e_g) {q(ratios)}; rows in the 720 {in_sp}")
        ns[rtol] = {"rows": len(near), "analytic_lost": an_lost, "analytic_gained": an_gained, "group_lost": gr_lost, "group_gained": gr_gained,
                    "oracle_converged_06h": o_conv[0], "oracle_converged_g": o_conv[1], "max_ratio": max(ratios) if ratios else None}
    desc["near_singular"] = ns
    (WIDER / "out" / "descriptive_06g.json").write_text(json.dumps(desc, indent=1, default=str))
    print(f"\n== verdicts: V2 {verdict(v2_fail)}; V5 {verdict(v5_fail)}; V6 {verdict(not v6_holds)}; V7 {verdict(not v7_holds)} (V8: out/v8.md)")


if __name__ == "__main__":
    main()
