"""levin-phase-floor prompt 06e section 5: the census's scores Z2, Z4, Z5 and the descriptive items.
New in 06e (its loaders follow 06c's analyse06c.py; a record kept in state/scripts_06c_reference/).
Reads, read only: this census run's checkpoint (RUN_DIR), 06c's census run's checkpoint
(prov.RUN06C), 06a's run's 1e-7 checkpoint lines (the 720 stationary-point rows, as analyse06c.py's
scan06a reads them), state/rows.json, and 06c's y1-references results.json (in the tree g, for Y1's
12 rows). Every rule is state/predictions.md's (Part A the prompt's text, Part B its readings).
Writes out/census_rows_06e.csv.gz (one line per unit, both runs side by side) and
out/deep_calls_06e.json (every call past depth 12, its line compact with its tiling).
Usage: python -B analyse06e.py RUN_DIR"""
import collections
import csv
import gzip
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import G_TREE, RUN06A, RUN06C, WIDER, header, sha256  # noqa: E402

RTOLS = ("1e-7", "1e-8")
DEPTH_MAX = 20
Y1_JSON = G_TREE / "docs" / "levin-phase-floor" / "06c" / "y1-references" / "results.json"


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def F(x):
    return Fraction(H(x))


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def slim_call(kind, c):
    out = {k: c.get(k) for k in ("index", "notify_label", "group", "converged", "max_depth", "num_regions",
                                  "num_subregion_solves", "E_largest_share", "n_at_depth_max")}
    out["kind"] = kind
    out["value"] = c["value"]["hex"]
    out["abserr"] = c["abserr"]["hex"]
    out["cc_floor"] = c.get("cc_floor")
    return out


def load(run_dir, rtol, keep_deep=False):
    """{rowkey: slim unit} at one rtol."""
    out = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"rtol": "{rtol}"' not in line:
                continue
            r = json.loads(line)
            d = r["data"]
            if d["rtol"] != rtol:
                continue
            res = d.get("result")
            u = {
                "unit": r["unit"], "status": d["status"], "wall_s": d["wall_s"], "row": d["row"],
                "total": None if res is None else res["total"]["hex"],
                "total_abserr": None if res is None else res["total_abserr"]["hex"],
                "total_converged": None if res is None else res["total_converged"],
                "calls": [slim_call("group", c) for c in d["calls"]] + [slim_call("analytic", c) for c in d["other_calls"]],
            }
            if keep_deep:
                u["deep06e"] = (d.get("deep_lines_06e") or {}).get("lines") or []
            out[rowkey(r["unit"])] = u
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


def match(ug, uc):
    """[(cg, cc)] by index with equal labels; problems."""
    probs = []
    if len(ug["calls"]) != len(uc["calls"]):
        probs.append(f"{ug['unit']}: call counts differ {len(ug['calls'])} vs {len(uc['calls'])}")
    bi = {c["index"]: c for c in uc["calls"]}
    out = []
    for cg in ug["calls"]:
        cc = bi.get(cg["index"])
        if cc is None or cc["notify_label"] != cg["notify_label"]:
            probs.append(f"{ug['unit']} call {cg['index']}: no 06c call with its label")
            cc = None
        out.append((cg, cc))
    return out, probs


def verdict(fail, undecided):
    return "fails" if fail else ("undecided" if undecided else "holds")


def q(vals, ps=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = sorted(vals)
    if not v:
        return "none"
    return ", ".join(f"p{int(p*100)} {v[min(len(v)-1, int(p*(len(v)-1)))]:.3g}" for p in ps)


def main():
    run_dir = Path(sys.argv[1])
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06e: the census's scores and descriptive items",
                    [run_dir / "checkpoint.jsonl", RUN06C / "checkpoint.jsonl", WIDER / "state" / "rows.json",
                     WIDER / "state" / "predictions.md", Y1_JSON]):
        print(l)
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]}
    sp = scan06a_sp()
    print(f"\n06a's stationary-point rows: {len(sp)} (must be 720: {len(sp) == 720})")
    # Y1's 12 rows
    y1 = json.loads(Y1_JSON.read_text())
    y1rows = y1["rows"] if isinstance(y1, dict) and "rows" in y1 else y1
    y12 = {}
    for r in (y1rows.values() if isinstance(y1rows, dict) else y1rows):
        if r.get("rank") is not None and r["rank"] <= 12:
            m = [k for k, rr in rows.items() if float(rr["k"]) == r["k"] and float(rr["q"]) == r["q"] and float(rr["r"]) == r["r"]
                 and float(rr["z_response"]) == r["z_response"]]
            y12[r["id"]] = m
    print("Y1's 12 rows (y1-references rank 1-12) -> rows.json:", {k: v for k, v in sorted(y12.items())})

    csv_rows = []
    deep_out = []
    summary = {}
    for rtol in RTOLS:
        G = load(run_dir, rtol, keep_deep=True)
        C = load(RUN06C, rtol)
        print(f"\n================ rtol {rtol}: units g {len(G)}, 06c {len(C)}")
        st = collections.Counter(u["status"] for u in G.values())
        st_c = collections.Counter(u["status"] for u in C.values())
        print(f"status g {dict(st)}; 06c {dict(st_c)}")
        over = [k for k, u in G.items() if u["status"] == "over budget"]
        over_out = [k for k in over if k not in sp]
        over300 = [(u["unit"], round(u["wall_s"], 1)) for u in G.values() if u["wall_s"] and u["wall_s"] > 300]
        print(f"over budget: {len(over)} ({len(over_out)} outside the 720); done with wall_s > 300 s: {over300}")
        probs = []
        # Z2
        z2_viol, z2_sp = [], []
        # Z4
        z4_n = z4_nonfinite = 0
        z4_viol = []
        # Z5
        z5_n = 0
        z5_fail = []
        z5_und = 0
        for key, ug in G.items():
            uc = C.get(key)
            in_sp = key in sp
            if uc is None:
                probs.append(f"{key}: no 06c unit")
                continue
            pairs, pr = match(ug, uc)
            probs += pr
            for cg, cc in pairs:
                if cg["max_depth"] >= DEPTH_MAX:
                    (z2_sp if in_sp else z2_viol).append((ug["unit"], cg["index"], cg["kind"], cg["notify_label"], cg["max_depth"]))
                if not in_sp:
                    if cc is None:
                        z5_und += 1
                    elif not cc["converged"]:
                        z5_n += 1
                        if not cg["converged"]:
                            z5_fail.append((ug["unit"], cg["index"], cg["kind"], cg["notify_label"], cg["max_depth"], cg["abserr"]))
            if not in_sp and ug["total"] is not None and uc["total"] is not None:
                ag, ac = H(ug["total_abserr"]), H(uc["total_abserr"])
                if not (math.isfinite(ag) and math.isfinite(ac)):
                    z4_nonfinite += 1
                else:
                    z4_n += 1
                    lhs = abs(F(ug["total"]) - F(uc["total"]))
                    rhs = F(ug["total_abserr"]) + F(uc["total_abserr"])
                    if lhs > rhs:
                        z4_viol.append((ug["unit"], float(lhs / rhs)))
        print(f"matching problems: {len(probs)} {probs[:10]}")
        z2v = verdict(bool(z2_viol) or bool(over_out), False)
        print(f"\n== Z2 (rtol {rtol}): calls outside the 720 reaching depth {DEPTH_MAX}: {len(z2_viol)}; over-budget units outside: {len(over_out)} -> {z2v}")
        for v in z2_viol:
            print(f"   {v}")
        print(f"   (reported) calls of the 720 at depth {DEPTH_MAX}: {len(z2_sp)}")
        for v in z2_sp:
            print(f"      {v}")
        z4v = verdict(bool(z4_viol), bool(over_out))
        print(f"\n== Z4 (rtol {rtol}): rows scored {z4_n}; non-finite total_abserr in either {z4_nonfinite}; violating {len(z4_viol)} -> {z4v}")
        for v in sorted(z4_viol, key=lambda t: -t[1])[:30]:
            print(f"   {v[0]}: |dtotal|/(e_06c+e_g) = {v[1]:.4g}")
        z5v = verdict(bool(z5_fail), (z5_und > 0 or bool(over_out)))
        print(f"\n== Z5 (rtol {rtol}): Levin calls outside the 720 that 06c reported unconverged: {z5_n}; unconverged under g: {len(z5_fail)}; unmatched {z5_und} -> {z5v}")
        for v in z5_fail:
            print(f"   {v}")
        summary[rtol] = {"Z2": z2v, "Z4": z4v, "Z5": z5v, "z2_viol": len(z2_viol), "z2_sp": len(z2_sp), "z4_n": z4_n,
                         "z4_viol": len(z4_viol), "z4_nonfinite": z4_nonfinite, "z5_n": z5_n, "z5_fail": len(z5_fail),
                         "over": len(over)}

        # ---- descriptive
        def conv_counts(U):
            c = collections.Counter()
            for u in U.values():
                if u["total_converged"] is False:
                    c["rows total_converged false"] += 1
                for cl in u["calls"]:
                    c[f"{cl['kind']} calls"] += 1
                    if not cl["converged"]:
                        c[f"{cl['kind']} unconverged"] += 1
                    if cl["max_depth"] >= DEPTH_MAX:
                        c[f"{cl['kind']} at depth {DEPTH_MAX}"] += 1
            return c
        cg_, cc_ = conv_counts(G), conv_counts(C)
        print(f"\n-- descriptive (rtol {rtol}): convergence and depth-20 calls, g vs 06c (all rows)")
        for k in sorted(set(cg_) | set(cc_)):
            print(f"   {k}: g {cg_[k]}  06c {cc_[k]}")
        cgo = conv_counts({k: v for k, v in G.items() if k not in sp}); cco = conv_counts({k: v for k, v in C.items() if k not in sp})
        print(f"   outside the 720:")
        for k in sorted(set(cgo) | set(cco)):
            print(f"      {k}: g {cgo[k]}  06c {cco[k]}")

        def row_depth(U):
            h = collections.Counter()
            for u in U.values():
                if u["calls"]:
                    h[max(c["max_depth"] for c in u["calls"])] += 1
            return h
        hg, hc = row_depth(G), row_depth(C)
        print("   row depth (max over the row's Levin calls) histogram, depth: g / 06c")
        print("   " + "; ".join(f"{d}: {hg[d]}/{hc[d]}" for d in sorted(set(hg) | set(hc))))

        def call_depth(U):
            h = collections.Counter()
            for u in U.values():
                for c in u["calls"]:
                    h[c["max_depth"]] += 1
            return h
        dg, dc = call_depth(G), call_depth(C)
        print("   call depth histogram, depth: g / 06c")
        print("   " + "; ".join(f"{d}: {dg[d]}/{dc[d]}" for d in sorted(set(dg) | set(dc))))

        def cost(U):
            return (sum(c["num_regions"] or 0 for u in U.values() for c in u["calls"]),
                    sum(c["num_subregion_solves"] or 0 for u in U.values() for c in u["calls"]),
                    math.fsum(u["wall_s"] or 0.0 for u in U.values()),
                    max((u["wall_s"] or 0.0, u["unit"]) for u in U.values()))
        rg, sg, wg, mg = cost(G); rc, sc, wc, mc = cost(C)
        print(f"   cost: regions g {rg} / 06c {rc} (x{rg/rc:.3f}); solves g {sg} / 06c {sc} (x{sg/sc:.3f}); "
              f"unit walls summed g {wg/3600:.3f} h / 06c {wc/3600:.3f} h (x{wg/wc:.3f}); largest unit g {mg[1]} {mg[0]:.1f} s / 06c {mc[1]} {mc[0]:.1f} s")
        summary[rtol].update({"regions": (rg, rc), "solves": (sg, sc), "wall_h": (wg / 3600, wc / 3600), "conv": (dict(cg_), dict(cc_)),
                              "row_depth": ({int(k): v for k, v in hg.items()}, {int(k): v for k, v in hc.items()})})

        # where the budget went
        share = collections.Counter()
        terms = collections.defaultdict(list)
        n_cc_calls = 0
        for u in G.values():
            for c in u["calls"]:
                share[(c["kind"], c["E_largest_share"])] += 1
                f = c.get("cc_floor")
                if f and "sums_floor_attributed_cc_leaves" in f and f["n_floor_attributed"] > 0:
                    n_cc_calls += 1
                    for t, v in f["sums_floor_attributed_cc_leaves"].items():
                        terms[t].append(H(v))
        print(f"   where the budget went (per call, largest of residual/floor/truncation): {dict(sorted(share.items(), key=str))}")
        if terms:
            tot = {t: math.fsum(v) for t, v in terms.items()}
            print(f"   CC floor terms summed over floor-attributed CC leaves ({n_cc_calls} calls): "
                  + ", ".join(f"{t} {tot[t]:.4g}" for t in sorted(tot)))
            fl = tot.get("phase_err_resolve") or 0.0
            if fl:
                print("   as shares of the floor sum: " + ", ".join(f"{t} {tot[t]/fl:.4f}" for t in
                      ("t_value_rounding", "t_argument", "node_term", "abs_E", "R_g", "R_2", "R_c", "declared")))
            # per-call: which floor term is largest
            big = collections.Counter()
            for i in range(len(terms["node_term"])):
                cand = {t: terms[t][i] for t in ("t_value_rounding", "t_argument", "node_term", "declared")}
                big[max(cand, key=cand.get)] += 1
            print(f"   per call, largest floor term among (n+4)uS, argument, node, declared: {dict(big)}")

        # anything still deep: past depth 12, outside the 720
        print("   calls past depth 12 (outside the 720 / inside):", end=" ")
        n_in = n_out = 0
        for key, u in G.items():
            for l in u["deep06e"]:
                if "error" in l:
                    continue
                inside = key in sp
                n_in += inside; n_out += not inside
                lev = l["levels"]
                rof = []
                for L_ in lev:
                    r_, f_ = H(L_[1]), H(L_[2])
                    rof.append(None if f_ == 0 else r_ / f_)
                cl = next(c for c in u["calls"] if c["index"] == l["index"])
                deep_out.append({"rtol": rtol, "unit": u["unit"], "index": l["index"], "notify_label": l["notify_label"],
                                 "stationary_point_row": inside, "max_depth": l["max_depth"], "converged": cl["converged"],
                                 "every_level_tiles_at_N": l.get("every_level_tiles_at_N"),
                                 "tiling": l.get("tiling"), "types": [L_[0] for L_ in lev],
                                 "resid_over_floor": rof, "eLR_over_eP": [L_[5] for L_ in lev]})
        print(f"{n_out} / {n_in}")
        for d_ in deep_out:
            if d_["rtol"] == rtol and not d_["stationary_point_row"]:
                last5 = [None if x is None else float(f"{x:.3g}") for x in d_["resid_over_floor"][-5:]]
                print(f"      {d_['unit']} #{d_['index']} {d_['notify_label'][-40:]}: depth {d_['max_depth']} conv {d_['converged']} "
                      f"tiles {d_['every_level_tiles_at_N']}; resid/floor last five {last5}")

        # stationary-point rows
        moved = beyond = 0
        rel = []
        for key in sp:
            ug, uc = G.get(key), C.get(key)
            if not ug or not uc or ug["total"] is None or uc["total"] is None:
                continue
            if ug["total"] != uc["total"]:
                moved += 1
            ec = H(uc["total_abserr"])
            dt = abs(H(ug["total"]) - H(uc["total"]))
            if math.isfinite(ec) and ec > 0:
                rel.append(dt / ec)
                beyond += dt > ec
        print(f"   the 720: totals moved {moved}; by more than total_abserr_06c {beyond}; |dtotal|/e_06c {q(rel)}")
        for yid, keys in sorted(y12.items()):
            for key in keys:
                ug, uc = G[key], C[key]
                gpr = [c for c in ug["calls"] if c["notify_label"] and "G+q+r" in c["notify_label"]]
                print(f"      {yid} {key} (in 720: {key in sp}): g total {ug['total']} e {ug['total_abserr']} conv {ug['total_converged']} | "
                      f"06c {uc['total']} e {uc['total_abserr']} conv {uc['total_converged']}; |dT|/(e_g+e_06c) "
                      f"{abs(H(ug['total'])-H(uc['total']))/(H(ug['total_abserr'])+H(uc['total_abserr'])):.3g}; "
                      f"G+q+r g depth {[c['max_depth'] for c in gpr]} conv {[c['converged'] for c in gpr]} regions {[c['num_regions'] for c in gpr]}")
        for key in ("1-2174",):
            ug, uc = G[key], C[key]
            cg = next(c for c in ug["calls"] if c["index"] == 1); cc = next(c for c in uc["calls"] if c["index"] == 1)
            print(f"      {ug['unit']} call 1 {cg['notify_label'][-30:]} (in 720: {key in sp}): g value {cg['value']} abserr {cg['abserr']} depth {cg['max_depth']} "
                  f"regions {cg['num_regions']} conv {cg['converged']} | 06c {cc['value']} {cc['abserr']} depth {cc['max_depth']} regions {cc['num_regions']} "
                  f"conv {cc['converged']}; |dv|/abserr_06c {abs(H(cg['value'])-H(cc['value']))/H(cc['abserr']):.4g}, /abserr_g {abs(H(cg['value'])-H(cc['value']))/H(cg['abserr']):.4g}")

        for key, ug in G.items():
            uc = C.get(key) or {}
            csv_rows.append([ug["unit"], rows[key]["band"], rows[key]["k_key"], key in sp, ug["status"], ug["wall_s"], uc.get("wall_s"),
                             ug["total"], ug["total_abserr"], ug["total_converged"], uc.get("total"), uc.get("total_abserr"), uc.get("total_converged"),
                             max((c["max_depth"] for c in ug["calls"]), default=None), max((c["max_depth"] for c in uc.get("calls", [])), default=None)])
        del G, C

    with gzip.open(WIDER / "out" / "census_rows_06e.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "band", "k_key", "stationary_point_row", "status", "wall_g", "wall_06c", "total_g", "total_abserr_g",
                    "total_converged_g", "total_06c", "total_abserr_06c", "total_converged_06c", "row_depth_g", "row_depth_06c"])
        w.writerows(csv_rows)
    (WIDER / "out" / "deep_calls_06e.json").write_text(json.dumps(deep_out, indent=0))
    print("\n== summary")
    print(json.dumps(summary, indent=1, default=str))
    print(f"wrote out/census_rows_06e.csv.gz ({len(csv_rows)} units) and out/deep_calls_06e.json ({len(deep_out)} calls)")


if __name__ == "__main__":
    main()
