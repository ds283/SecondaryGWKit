"""levin-phase-floor prompt 06j: draw Part S's and Part W's samples, by the rules of
state/predictions_and_rules_20261103T0905Z.md (written before this script ran; sections 2 and 4), and
write the capture selection. New in 06j. Reads only, except out/sample/.

Part S: 24 of V8's lost calls outside the 720 (12 per rtol; groups G+q+r, G+q-r, G-q+r, G-q-r; strata by
slow-leaf clusters per call c = 1, c in {2, 3}, c >= 4), seed 20261103, no two from one row; up to 10
scoped slow leaves per call, chosen across clusters. Each leaf is matched by its exact (start, end) to
the census's region pickle record ``checked_leaves_06g`` and its r, delta1, delta2 and phase_err are
taken from there (and checked against v8_slow_leaves.json's r, d1 and phi).

Part W: W-5 (v2_rows.csv.gz's violating ``total`` rows, call index 1, at each rtol) and W-12 (the call-
level set, recomputed by values_beyond06g.py's rule, 6 per rtol at evenly spaced quantiles of the ratio,
at most two per group per rtol).

Writes out/sample/s_sample.json, out/sample/w_sample.json, out/sample/selection.json (x1_capture.py's
SELECTION_JSON, with both censuses' records) and prints a summary.

Usage (from var/levin-phase-floor-06j/): python -B scripts/draw_06j.py
"""

import collections
import csv
import gzip
import hashlib
import json
import pickle
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402
from analyse06g import load, match, scan06a_sp, H  # noqa: E402

SEED = 20261103
GROUPS = ("G+q+r", "G+q-r", "G-q+r", "G-q-r")
STRATA = ("c=1", "c in {2,3}", "c>=4")
V06G = RC.MAIN / "var" / "levin-phase-floor-06g" / "census"
V8_CALLS = V06G / "out" / "v8_calls.json"
V8_CALLS_SHA = "1e9a62b0f8d95deade4f265d1fa1a8b5bbd63f587f7c9761d78882db5367c42e"
V8_LEAVES = V06G / "out" / "v8_slow_leaves.json"
V8_LEAVES_SHA = "b4643033a1249bb036524ec041fc9594471a56a932a1ded593101d1c486da9ba"
V2_ROWS = V06G / "out" / "v2_rows.csv.gz"
CK06G_SHA = "03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa"
GAP_WIDTHS = 64
MAX_LEAVES = 10


def stratum(c):
    return "c=1" if c == 1 else ("c in {2,3}" if c <= 3 else "c>=4")


def clusters(leaves, span):
    a0, b0 = span
    w = (b0 - a0) * 2.0**-20
    es = sorted((e for e in leaves if e["depth"] == 20), key=lambda e: e["start"])
    cls = [[es[0]]]
    for e in es[1:]:
        if e["start"] - cls[-1][-1]["end"] <= GAP_WIDTHS * w:
            cls[-1].append(e)
        else:
            cls.append([e])
    return cls


def census_pickle_record(run, data):
    with open(run / data["pickle"], "rb") as f:
        f.seek(data["pickle_offset"])
        blob = f.read(data["pickle_length"])
    if hashlib.sha256(blob).hexdigest() != data["pickle_record_sha256"]:
        raise SystemExit(f"{data['pickle']}@{data['pickle_offset']}: sha256 differs")
    return pickle.loads(blob)


def main():
    run06g = RC.census_run_06g()
    ck = {"g": run06g / "checkpoint.jsonl", "06h": RC.RUN06H / "checkpoint.jsonl"}
    outd = RC.ROOT / "out" / "sample"
    outd.mkdir(parents=True, exist_ok=True)
    for name in ("s_sample.json", "w_sample.json", "selection.json"):
        if (outd / name).exists():
            raise SystemExit(f"{outd / name} exists; not overwritten")
    hdr = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06j: Part S's and Part W's samples, and the capture selection",
        [V8_CALLS, V8_LEAVES, V2_ROWS, ck["g"], ck["06h"],
         RC.ROOT / "state" / "predictions_and_rules_20261103T0905Z.md"],
    )
    for l in hdr:
        print(l)
    for p, want in ((V8_CALLS, V8_CALLS_SHA), (V8_LEAVES, V8_LEAVES_SHA), (ck["g"], CK06G_SHA),
                    (ck["06h"], RC.CHECKPOINT06H_SHA256)):
        if RC.sha256(p) != want:
            raise SystemExit(f"{p}: sha256 is not {want}")

    # ---------------- Part S ----------------
    v8 = json.loads(V8_CALLS.read_text())
    lost = [x for x in v8["lost"] if not x["stationary_point_row"]]
    leaves = json.loads(V8_LEAVES.read_text())
    per_call = collections.defaultdict(list)
    for e in leaves:
        per_call[(e["unit"], e["index"])].append(e)
    cells = collections.defaultdict(list)
    info = {}
    for x in lost:
        g = x["g"]
        if g["kind"] != "group" or g["group"] not in GROUPS:
            continue
        key = (x["unit"], g["index"])
        L = per_call.get(key)
        if not L:
            continue
        cls = clusters(L, L[0]["span"])
        st = stratum(len(cls))
        info[key] = {"x": x, "leaves": L, "clusters": cls, "stratum": st}
        cells[(x["rtol"], g["group"], st)].append(key)
    rng = random.Random(SEED)
    ranked = {}
    for rtol in RC.RTOLS:
        for grp in GROUPS:
            for st in STRATA:
                lst = sorted(cells.get((rtol, grp, st), []), key=lambda k: (info[k]["x"]["row"], k[1]))
                rng.shuffle(lst)
                ranked[(rtol, grp, st)] = lst
    used_rows = set()
    s_calls = []
    for rtol in RC.RTOLS:
        for grp in GROUPS:
            for st in STRATA:
                pick, via = None, st
                for k in ranked[(rtol, grp, st)]:
                    if info[k]["x"]["row"] not in used_rows:
                        pick = k
                        break
                if pick is None:
                    for st2 in STRATA:
                        for k in ranked[(rtol, grp, st2)]:
                            if info[k]["x"]["row"] not in used_rows:
                                pick, via = k, st2
                                break
                        if pick is not None:
                            break
                if pick is None:
                    raise SystemExit(f"no eligible call for {(rtol, grp, st)}")
                used_rows.add(info[pick]["x"]["row"])
                s_calls.append({"cell": [rtol, grp, st], "taken_from_stratum": via, "key": pick})
    # leaves per call, across clusters
    s_out = []
    g_units = {k["key"][0] for k in s_calls}
    found = RC.scan_checkpoint(ck["g"], {(RC.rest_of(u), i) for u, i in (c["key"] for c in s_calls)})
    for kpos, c in enumerate(s_calls):
        key = c["key"]
        inf = info[key]
        cls = sorted(inf["clusters"], key=lambda cl: sum((e["start"] + e["end"]) / 2 for e in cl) / len(cl))
        r2 = random.Random(SEED + 1000 * kpos)
        orders = []
        for cl in cls:
            lst = sorted(cl, key=lambda e: e["start"])
            r2.shuffle(lst)
            orders.append(lst)
        chosen = []
        nleaf = sum(len(o) for o in orders)
        if nleaf <= MAX_LEAVES:
            for ci, o in enumerate(orders):
                for e in o:
                    chosen.append((ci, e))
        else:
            ptr = [0] * len(orders)
            while len(chosen) < MAX_LEAVES:
                for ci, o in enumerate(orders):
                    if len(chosen) >= MAX_LEAVES:
                        break
                    if ptr[ci] < len(o):
                        chosen.append((ci, o[ptr[ci]]))
                        ptr[ci] += 1
        chosen.sort(key=lambda t: t[1]["start"])
        # the census record and its region pickle: exact leaves
        recs = [x for x in found.get((RC.rest_of(key[0]), key[1]), []) if x[1] == "done"]
        if len(recs) != 1:
            raise SystemExit(f"{key}: {len(recs)} done census records")
        unit, status, kind, crec, data = recs[0]
        prec = census_pickle_record(run06g, data)
        chk = {e["index"]: e for e in prec["checked_leaves_06g"] if e is not None}[key[1]]
        n = len(chk["depth"])
        idx = {(float(chk["start"][i]), float(chk["end"][i])): i for i in range(n)}
        lv = []
        for ci, e in chosen:
            i = idx.get((e["start"], e["end"]))
            if i is None:
                raise SystemExit(f"{key}: leaf {e['start']!r}..{e['end']!r} not in checked_leaves_06g")
            d = {k: float(chk[k][i]) for k in ("start", "end", "delta1", "delta2", "convergence_ratio", "phase_err",
                                              "abserr", "value", "value_own", "budget_err")}
            d.update(depth=int(chk["depth"][i]), type=int(chk["type"][i]), trusted_final=int(chk["trusted_final"][i]),
                     returned=bool(chk["returned"][i]), cluster=ci,
                     hex={k: float(chk[k][i]).hex() for k in ("start", "end", "delta1", "delta2", "convergence_ratio", "phase_err")})
            if not (d["convergence_ratio"] == e["r"] and d["delta1"] == e["d1"] and d["phase_err"] == e["phi"]):
                raise SystemExit(f"{key}: leaf numbers differ from v8_slow_leaves.json")
            lv.append(d)
        x = inf["x"]
        s_out.append({
            "sid": f"s{kpos:02d}", "cell": c["cell"], "taken_from_stratum": c["taken_from_stratum"],
            "unit": key[0], "index": key[1], "rtol": x["rtol"], "row": x["row"], "group": x["g"]["group"],
            "notify_label": x["g"]["notify_label"], "x_span": x["g"]["x_span"],
            "n_scoped_leaves": nleaf, "n_clusters": len(cls),
            "cluster_centres_u": [((sum((e["start"] + e["end"]) / 2 for e in cl) / len(cl)) - inf["leaves"][0]["span"][0])
                                  / (inf["leaves"][0]["span"][1] - inf["leaves"][0]["span"][0]) for cl in cls],
            "leaves": lv,
        })
    replacement_lists = {"|".join(k): [list(x) for x in v] for k, v in ranked.items()}

    # ---------------- Part W ----------------
    rows_v2 = []
    with gzip.open(V2_ROWS, "rt") as f:
        for r in csv.DictReader(f):
            if r["reading"] == "total" and r["violates"] == "True":
                rows_v2.append(r)
    w5 = []
    for rtol in RC.RTOLS:
        rs = sorted((r for r in rows_v2 if r["rtol"] == rtol), key=lambda r: r["row"])
        if len(rs) != 5:
            raise SystemExit(f"{rtol}: {len(rs)} violating total rows, not 5")
        for r in rs:
            w5.append({"rtol": rtol, "row": r["row"], "index": 1, "v2_ratio": float(r["ratio"])})
    sp = scan06a_sp()
    G, Hh = load(run06g), load(RC.RUN06H)
    cl = []
    for (rt, rk), ug in G.items():
        if rk in sp:
            continue
        for cg, ch in match(ug, Hh[(rt, rk)])[0]:
            eg, eh = H(cg["abserr"]), H(ch["abserr"])
            if eg is None or eh is None or eg != eg or eh != eh or eg == float("inf") or eh == float("inf"):
                continue
            d = abs(Fraction(H(cg["value"])) - Fraction(H(ch["value"])))
            if d > Fraction(eg) + Fraction(eh):
                cl.append({"rtol": rt, "row": rk, "index": cg["index"], "kind": cg["kind"],
                           "group": cg["group"] or cg["notify_label"],
                           "ratio": float(d / (Fraction(eg) + Fraction(eh))), "dv": float(d)})
    n_cl = collections.Counter(x["rtol"] for x in cl)
    print(f"call-level set: {dict(n_cl)} (expected 62 at 1e-7, 80 at 1e-8)")
    if n_cl.get("1e-7") != 62 or n_cl.get("1e-8") != 80:
        raise SystemExit("the call-level set does not reproduce 62 and 80")
    w5_keys = {(x["rtol"], x["row"], x["index"]) for x in w5}
    for x in w5:
        m = [c for c in cl if (c["rtol"], c["row"], c["index"]) == (x["rtol"], x["row"], x["index"])]
        x["in_call_level_set"] = bool(m)
        x["call_ratio"] = m[0]["ratio"] if m else None
    w12 = []
    taken_1e7 = set()
    for rtol in RC.RTOLS:
        cand = [c for c in cl if c["rtol"] == rtol and (c["rtol"], c["row"], c["index"]) not in w5_keys]
        if rtol == "1e-8":
            cand = [c for c in cand if (c["row"], c["index"]) not in taken_1e7]
        cand.sort(key=lambda c: (c["ratio"], c["row"], c["index"]))
        n = len(cand)
        picked, gcount = [], collections.Counter()
        for j in range(6):
            p = (j + 0.5) / 6 * (n - 1)
            best = None
            for rank, c in enumerate(cand):
                if rank in [q[0] for q in picked] or gcount[c["group"]] >= 2:
                    continue
                dist = abs(rank - p)
                if best is None or dist < best[0] or (dist == best[0] and rank < best[1]):
                    best = (dist, rank)
            rank = best[1]
            picked.append((rank, cand[rank], p))
            gcount[cand[rank]["group"]] += 1
        for rank, c, p in picked:
            w12.append(dict(c, rank=rank, n_candidates=n, target_rank=p))
            if rtol == "1e-7":
                taken_1e7.add((c["row"], c["index"]))

    # ---------------- the selection ----------------
    entries = []
    for s in s_out:
        entries.append({"sid": s["sid"], "part": "S", "rest": RC.rest_of(s["unit"]), "index": s["index"],
                        "notify_label": s["notify_label"], "rtol": s["rtol"]})
    for i, x in enumerate(w5):
        x["sid"] = f"f{i:02d}"
        entries.append({"sid": x["sid"], "part": "W-5", "rest": f"{x['rtol']}-{x['row']}", "index": 1,
                        "notify_label": None, "rtol": x["rtol"]})
    for i, x in enumerate(w12):
        x["sid"] = f"c{i:02d}"
        entries.append({"sid": x["sid"], "part": "W-12", "rest": f"{x['rtol']}-{x['row']}", "index": x["index"],
                        "notify_label": None, "rtol": x["rtol"]})
    wanted = {(e["rest"], e["index"]) for e in entries}
    found2 = {name: RC.scan_checkpoint(ck[name], wanted) for name in ck}
    sel, problems = [], []
    for e in entries:
        k = (e["rest"], e["index"])
        r = dict(e)
        for name in ("g", "06h"):
            recs = [x for x in (found2[name].get(k) or []) if x[1] == "done"]
            if not recs or len({RC.call_key(x[3]) for x in recs}) != 1:
                problems.append(f"{e['sid']} {k}: {len(recs)} done {name} records (or they disagree)")
                continue
            unit, status, kind, c, data = recs[0]
            if r["notify_label"] is None:
                r["notify_label"] = c["notify_label"]
            if c["notify_label"] != r["notify_label"]:
                problems.append(f"{e['sid']} {k}: {name} label differs")
            if r.get("kind") is not None and r["kind"] != kind:
                problems.append(f"{e['sid']}: kinds differ")
            r["kind"] = kind
            xs = [RC.hexof(v) for v in c["x_span"]]
            if r.get("x_span") is not None and r["x_span"] != xs:
                problems.append(f"{e['sid']}: x_span differs between censuses")
            r["x_span"] = xs
            r[f"{name}_unit"] = unit
            r[f"{name}_record"] = c
            r[f"{name}_pickle"] = [data.get("pickle"), data.get("pickle_offset"), data.get("pickle_length"),
                                   data.get("pickle_record_sha256")]
            if name == "g":
                r["census_unit_budget_s"] = data.get("budget_s")
                r["census_unit_status"] = status
        if "g_record" in r:
            g = r["g_record"]
            r["unit"] = "g-" + e["rest"]
            r["census_value"] = g["value"]["hex"]
            r["census_abserr"] = g["abserr"]["hex"]
            r["census_num_regions"] = g["num_regions"]
            r["census_max_depth"] = g["max_depth"]
            r["census_converged"] = bool(g["converged"])
            r["census"] = f"06g {ck['g']}"
        sel.append(r)
    if problems:
        for p in problems:
            print("PROBLEM", p)
        raise SystemExit(f"{len(problems)} problems; nothing written")
    s_doc = {"header": hdr, "seed": SEED, "rule": "state/predictions_and_rules_20261103T0905Z.md section 2",
             "n_eligible_by_cell": {"|".join(k): len(v) for k, v in ranked.items()},
             "calls": s_out, "ranked_lists_for_replacement": replacement_lists}
    w_doc = {"header": hdr, "rule": "state/predictions_and_rules_20261103T0905Z.md section 4",
             "w5": w5, "w12": w12, "call_level_set": cl}
    (outd / "s_sample.json").write_text(json.dumps(s_doc, indent=1))
    (outd / "w_sample.json").write_text(json.dumps(w_doc, indent=1))
    (outd / "selection.json").write_text(json.dumps(sel, indent=1))
    print("\n== Part S")
    for s in s_out:
        print(f"{s['sid']} {s['unit']} #{s['index']} {s['group']} cell {s['cell']} (from {s['taken_from_stratum']}): "
              f"clusters {s['n_clusters']}, scoped leaves {s['n_scoped_leaves']}, sampled {len(s['leaves'])}")
    print(f"Part S: {len(s_out)} calls, {sum(len(s['leaves']) for s in s_out)} leaves")
    print("\n== W-5")
    for x in w5:
        print(f"{x['sid']} {x['rtol']} {x['row']} #1: V2 total ratio {x['v2_ratio']:.4g}; call ratio {x['call_ratio']}")
    print("\n== W-12")
    for x in w12:
        print(f"{x['sid']} {x['rtol']} {x['row']} #{x['index']} {x['group']}: ratio {x['ratio']:.4g} (rank {x['rank']} of "
              f"{x['n_candidates']}, target {x['target_rank']:.2f})")
    for r in sel:
        print(f"{r['sid']} {r['unit']} #{r['index']} {r['kind']}: g {r['census_value']} e {r['census_abserr']} conv "
              f"{r['census_converged']} regions {r['census_num_regions']} depth {r['census_max_depth']} | 06h "
              f"{r['06h_record']['value']['hex']} e {r['06h_record']['abserr']['hex']} conv {r['06h_record']['converged']} "
              f"regions {r['06h_record']['num_regions']} depth {r['06h_record']['max_depth']} | budget {r['census_unit_budget_s']}")
    for name in ("s_sample.json", "w_sample.json", "selection.json"):
        print(f"wrote {outd / name} sha256 {RC.sha256(outd / name)}")


if __name__ == "__main__":
    main()
