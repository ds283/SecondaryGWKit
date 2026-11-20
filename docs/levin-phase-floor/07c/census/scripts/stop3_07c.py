"""levin-phase-floor prompt 07c section 6, planned stop 3 (descriptive only; no verdict is scored here). New in 07c, after
07b's stop3_cost_07b.py (a record, not edited). Read only.

1. **Y1's counts** from compare_07c.py mode ``calls`` (CALLS_JSONL): calls untouched (num_chase_stops 0) and touched, by
   rtol and kind (analytic or group), with how many of each equal 07b's in Y1's five outputs.
2. **``converged`` against 07b's census**, by rtol and kind, in each direction. Every call newly unconverged (converged at
   07b, unconverged here) is listed with the stopped leaf carrying the most of its abserr, from the stop's instrument
   (``chase_07c``'s ``carrier``: its route, the region's type ``direct`` being the Clenshaw-Curtis pair, its trust before
   the stop, depth, r, declared resolution, e' as returned, and share = e' / the call's abserr), or "no stopped leaf"
   where the stop did not act (num_chase_stops 0) or stopped no leaf; with the call's abserr ratio against 07b's and
   whether |v - v_07b| exceeds the sum of the two abserrs (exact Fractions from the hex values).
3. **Cost:** the census's wall (manifest ``created`` to the last heartbeat) against the projection
   (out/census_projection.json: central, central x 07b's shortfall, highest margin) and against 07b's census; unit walls
   summed by rtol; regions by rtol and kind against 07b's.

Writes OUT_JSON and OUT_MD (the list as a table) and prints a summary.
Usage: python -B stop3_07c.py RUN_DIR CALLS_JSONL OUT_JSON OUT_MD
"""
import collections
import datetime
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN07B, WIDER, header, sha256  # noqa: E402


def F(h):
    return Fraction(float.fromhex(h))


def rp(v):
    return v.get("repr") if isinstance(v, dict) else v


def main():
    run_dir, calls_p, out_p, md_p = map(Path, sys.argv[1:5])
    proj_p = WIDER / "out" / "census_projection.json"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: planned stop 3, descriptive (Y1's counts, converged "
                 "against 07b, the newly unconverged calls, cost)", [calls_p, proj_p])
    out = {"header": hdr}
    y1 = collections.Counter()
    conv = collections.Counter()
    reg = collections.Counter()
    newly, newly_conv = [], []
    for line in open(calls_p):
        c = json.loads(line)
        k = (c["rtol"], c["kind"])
        y1[k + (c["class"], "equal" if c["u1_equal"] else "differing")] += 1
        a, b = c["converged"]
        conv[k + (f"07b {b} -> 07c {a}",)] += 1
        reg[k + ("07c",)] += c["num_regions"][0] or 0
        reg[k + ("07b",)] += c["num_regions"][1] or 0
        if b and not a:
            v, v7 = F(c["value"][0]), F(c["value"][1])
            e, e7 = F(c["abserr"][0]), F(c["abserr"][1])
            ch = c.get("chase_07c") or {}
            car = ch.get("carrier")
            newly.append({
                "unit": c["unit"], "rtol": c["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"],
                "class": c["class"], "counts": [c["num_chase_stops"], c["num_chase_stopped_leaves"],
                                                 c["num_chase_noise_bisections"]],
                "rate_rule_refusals": [c["num_edge_rule_refusals"], c["num_range_rule_refusals"]],
                "regions": c["num_regions"], "max_depth": c["max_depth"], "untrusted": c["untrusted"],
                "abserr_ratio_07b": float(e / e7) if e7 else None,
                "moved_past_sum": abs(v - v7) > e + e7, "move_over_sum": float(abs(v - v7) / (e + e7)) if e + e7 else None,
                "n_stopped": ch.get("n_stopped"), "by_route": ch.get("by_route"), "share_sum": rp(ch.get("share_sum")),
                "carrier": None if car is None else {k_: rp(v_) for k_, v_ in car.items()},
            })
        if a and not b:
            newly_conv.append({"unit": c["unit"], "rtol": c["rtol"], "kind": c["kind"], "index": c["index"],
                               "class": c["class"], "regions": c["num_regions"], "max_depth": c["max_depth"]})
    out["y1_counts"] = {"|".join(k): v for k, v in sorted(y1.items())}
    out["converged_changes"] = {"|".join(k): v for k, v in sorted(conv.items())}
    out["newly_unconverged"] = newly
    out["newly_unconverged_counts"] = {"|".join(k): v for k, v in sorted(collections.Counter(
        (x["rtol"], x["kind"]) for x in newly).items())}
    out["newly_unconverged_by_carrier_route"] = dict(collections.Counter(
        ("no stopped leaf" if x["carrier"] is None else
         f"{'Clenshaw-Curtis' if x['carrier']['type'] != 'Levin' else 'Levin'}"
         f"{'' if x['carrier']['trusted_before'] is None else (' trusted' if x['carrier']['trusted_before'] else ' untrusted')}")
        for x in newly))
    out["newly_converged"] = newly_conv
    out["newly_converged_counts"] = {"|".join(k): v for k, v in sorted(collections.Counter(
        (x["rtol"], x["kind"]) for x in newly_conv).items())}
    out["regions"] = {f"{a}|{b}": {"07c": reg[(a, b, "07c")], "07b": reg[(a, b, "07b")],
                                   "ratio": reg[(a, b, "07c")] / reg[(a, b, "07b")]}
                      for a, b in sorted({(k[0], k[1]) for k in reg})}
    tot = (sum(v for k, v in reg.items() if k[2] == "07c"), sum(v for k, v in reg.items() if k[2] == "07b"))
    out["regions_total"] = {"07c": tot[0], "07b": tot[1], "ratio": tot[0] / tot[1]}
    for name, rd in (("07c", run_dir), ("07b", RUN07B)):
        man = json.loads((rd / "manifest.json").read_text())
        st = json.loads((rd / "status.json").read_text())
        t0 = datetime.datetime.fromisoformat(man["created"]).timestamp()
        t1 = datetime.datetime.fromisoformat(st["heartbeat"]).timestamp()
        walls = collections.Counter()
        n_ob = n = 0
        mx = 0.0
        for line in open(rd / "checkpoint.jsonl"):
            d = json.loads(line)["data"]
            n += 1
            n_ob += d["status"] == "over budget"
            walls[d["rtol"]] += d.get("wall_s") or 0.0
            mx = max(mx, d.get("wall_s") or 0.0)
        out[name] = {"created": man["created"], "last_heartbeat": st["heartbeat"], "wall_s": t1 - t0, "n_records": n,
                     "n_over_budget": n_ob, "unit_wall_sum_s": dict(walls), "largest_unit_wall_s": mx}
    pj = json.loads(proj_p.read_text())
    w = out["07c"]["wall_s"]
    out["wall_against_projection"] = {
        "wall_s": w, "central_s": pj["projection_central_s"], "wall_over_central": w / pj["projection_central_s"],
        "central_x_shortfall_s": pj["projection_central_x_07b_shortfall_s"],
        "highest_s": pj["projection_highest_with_overbudget_s"], "wall_over_07b_census": w / out["07b"]["wall_s"]}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    L = ["# levin-phase-floor 07c, planned stop 3: calls newly unconverged against 07b's census", "",
         *[f"<!-- {h[2:]} -->" for h in hdr], "",
         f"{len(newly)} calls, by rtol and kind: {out['newly_unconverged_counts']}; by the carrying stopped leaf: "
         f"{out['newly_unconverged_by_carrier_route']}. Share = the stopped leaf's e' as returned over the call's abserr.",
         "",
         "| call (unit, kind, index) | label | stops / stopped leaves / counted | regions 07c / 07b | depth 07c / 07b | abserr / 07b's | past the sum | carrying stopped leaf: route, trust before, depth, r, share | its declared resolution, delta1, residual |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(newly, key=lambda x: (x["rtol"], x["kind"], x["unit"], x["index"])):
        cr = x["carrier"]
        if cr is None:
            cs, cd = "no stopped leaf", ""
        else:
            route = "Clenshaw-Curtis" if cr["type"] != "Levin" else "Levin"
            tb = {True: "trusted", False: "untrusted", None: "unchecked"}[cr["trusted_before"]]
            cs = f"{route}, {tb}, depth {cr['depth']}, r {float(cr['r']):.3g}, share {float(cr['share']):.3g}"
            cd = f"{float(cr['declared_resolution']):.3g}, {float(cr['delta1']):.3g}, {float(cr['residual']):.3g}" \
                if cr['declared_resolution'] is not None else f"(none), {cr['delta1']}, {cr['residual']}"
        L.append(f"| {x['unit']} {x['kind']} #{x['index']} | {(x['label'] or '').split(', group ')[-1][:20]} | "
                 f"{'/'.join(map(str, x['counts']))} | {x['regions'][0]} / {x['regions'][1]} | "
                 f"{x['max_depth'][0]} / {x['max_depth'][1]} | {x['abserr_ratio_07b']:.4g} | {x['moved_past_sum']} | {cs} | {cd} |")
    md_p.write_text("\n".join(L) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("header", "newly_unconverged", "newly_converged")},
                     indent=1))
    print(f"newly unconverged: {len(newly)}; newly converged: {len(newly_conv)}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    print(f"wrote {md_p} sha256 {sha256(md_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
