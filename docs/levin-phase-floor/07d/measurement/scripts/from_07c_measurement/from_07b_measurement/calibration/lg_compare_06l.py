"""levin-phase-floor prompt 06l, Part L: copied from 06k's lg_compare_06k.py (a record; in
state/scripts_06k_reference/calibration/), changed only where marked "prompt 06l": the variants are identity and routeB,
read from the one run file (06k's second file, route 2 literal's, is not read), a call's firings are route B's
``route_06l`` count where it has one, and each convergence change keeps its route B record. 06k's text follows.

levin-phase-floor prompt 06k, Part L ("The LiouvilleGreen draws"): each variant's draws against the tree's
(06g's code), from ../../out/lg_draws/lg_seeded_draws.jsonl (lg_seeded_draws.py run). New in 06k. Read only.

Per variant mode (identity, route1, route2, route3) against mode ``tree``, unit by unit (seed, test):
- **the test outcomes** (pass, fail, error) under both;
- **the Levin calls**, matched by their order in the unit (the call sequence must be equal: same count,
  spans, atol, rtol and labels), and every call whose ``converged`` changes, in each direction; with the
  calls whose value, abserr or regions (``regions_sha256``) differ, and the route's firings;
- **the printed cases** (the test's own true error and reported abserr, parsed by 06g's ``_printed_cases``),
  matched by order and label: honest means true <= reported. Every case that moves between honest and
  dishonest, in each direction, and the dishonest-and-converged counts, with test_YJJ_log_singularity's
  near-singular cases (eps <= 1e-4, 06g's ``_near_singular``) apart from the rest.
The identity mode must equal the tree on every call and case bit for bit.
Writes ../../out/lg_draws/compare.json and compare.txt. Usage: python -B lg_compare_06k.py
"""

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402
import lg_seeded_draws as L  # noqa: E402

RUN = prov.OUT / "lg_seeded_draws.jsonl"
VARIANTS = ("identity", "routeB")  # prompt 06l (06k: identity, route1, route2, route3, route2lit)
CALL_KEYS = ("x_span", "atol", "rtol", "notify_label")


def fx(v):
    return None if v is None else float(v).hex()


def main():
    lines = open(RUN).read().splitlines()
    run_header = json.loads(lines[0])["header"]
    recs = [json.loads(l) for l in lines[1:]]
    # prov.header names inputs relative to the tree; this input is in the records directory
    hdr = prov.header(__file__, "the LiouvilleGreen draws, each variant against the tree") + [f"# input {RUN} sha256 {prov.sha256(RUN)}"]  # prompt 06l: one run file
    out_lines = list(hdr) + ["# the run's header:"] + [f"#   {h}" for h in run_header]
    by = {(r["seed"], r["mode"], r["test"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    tests = [t for t in L.TESTS if any(r["test"] == t for r in recs)]
    doc = {"header": hdr, "n_records": len(recs), "variants": {}}
    out_lines.append(f"records {len(recs)}; seeds {seeds[0]}-{seeds[-1]}; tests {tests}; modes {sorted({r['mode'] for r in recs})}")
    # the tree's own honesty table, for reference
    for v in VARIANTS:
        res = {"outcomes": collections.Counter(), "outcome_changes": [], "sequence_mismatch": [],
               "calls": 0, "conv_changes": {"F->T": [], "T->F": []}, "calls_value_or_abserr_differ": 0,
               "calls_regions_differ": 0, "calls_fired": 0,
               "cases": {"ns": collections.Counter(), "rest": collections.Counter()},
               "moves": {"ns": {"honest->dishonest": [], "dishonest->honest": []},
                         "rest": {"honest->dishonest": [], "dishonest->honest": []}},
               "case_mismatch": []}
        for t in tests:
            for s in seeds:
                a, b = by.get((s, "tree", t)), by.get((s, v, t))
                if a is None or b is None:
                    res["sequence_mismatch"].append([s, t, "unit missing"])
                    continue
                res["outcomes"][f"{a['outcome']}->{b['outcome']}"] += 1
                if a["outcome"] != b["outcome"]:
                    res["outcome_changes"].append([s, t, a["outcome"], b["outcome"]])
                ca, cb = a["calls"], b["calls"]
                if len(ca) != len(cb) or any(any(x[k] != y[k] for k in CALL_KEYS) for x, y in zip(ca, cb)):
                    res["sequence_mismatch"].append([s, t, len(ca), len(cb)])
                    continue
                for x, y in zip(ca, cb):
                    res["calls"] += 1
                    fired = (y.get("route_06l") or y.get("route_06k") or {}).get("n_firings", 0) or 0  # prompt 06l: route_06l
                    res["calls_fired"] += fired > 0
                    if fx(x["value"]) != fx(y["value"]) or fx(x["abserr"]) != fx(y["abserr"]):
                        res["calls_value_or_abserr_differ"] += 1
                    if x["regions_sha256"] != y["regions_sha256"]:
                        res["calls_regions_differ"] += 1
                    if bool(x["converged"]) != bool(y["converged"]):
                        key = "F->T" if y["converged"] else "T->F"
                        res["conv_changes"][key].append({
                            "seed": s, "test": t, "index": x["index"], "label": x["notify_label"],
                            "tree": [x["value"], x["abserr"], x["num_regions"], x["max_depth"]],
                            "route": [y["value"], y["abserr"], y["num_regions"], y["max_depth"]],
                            "firings": fired, "route_06k": y.get("route_06k"), "route_06l": y.get("route_06l")})  # prompt 06l
                pa, pb = L._printed_cases(a["stdout"]), L._printed_cases(b["stdout"])
                if len(pa) != len(pb) or any(x["label"] != y["label"] for x, y in zip(pa, pb)):
                    res["case_mismatch"].append([s, t, len(pa), len(pb)])
                    continue
                for x, y in zip(pa, pb):
                    if x["true"] is None or y["true"] is None:
                        continue
                    grp = "ns" if (t == "test_YJJ_log_singularity" and L._near_singular(x["label"])) else "rest"
                    ha, hb = x["true"] <= x["reported"], y["true"] <= y["reported"]
                    c = res["cases"][grp]
                    c["n"] += 1
                    c["dishonest_tree"] += not ha
                    c["dishonest_route"] += not hb
                    c["dishonest_and_converged_tree"] += (not ha) and bool(x["converged"])
                    c["dishonest_and_converged_route"] += (not hb) and bool(y["converged"])
                    c["converged_tree"] += bool(x["converged"])
                    c["converged_route"] += bool(y["converged"])
                    c["true_error_changed"] += x["true"] != y["true"]
                    if ha != hb:
                        res["moves"][grp]["honest->dishonest" if ha else "dishonest->honest"].append({
                            "seed": s, "test": t, "label": x["label"],
                            "tree": [x["true"], x["reported"], x["converged"]],
                            "route": [y["true"], y["reported"], y["converged"]],
                            "ratio_route": y["true"] / y["reported"] if y["reported"] else None})
        res["outcomes"] = dict(res["outcomes"])
        res["cases"] = {g: dict(c) for g, c in res["cases"].items()}
        doc["variants"][v] = res
        out_lines.append(f"\n== {v} against tree")
        out_lines.append(f"outcomes (tree->{v}): {res['outcomes']}; changed: {res['outcome_changes']}")
        out_lines.append(f"call or case sequences that do not match: {res['sequence_mismatch']} {res['case_mismatch']}")
        out_lines.append(f"Levin calls {res['calls']}; with a firing {res['calls_fired']}; value or abserr differ "
                         f"{res['calls_value_or_abserr_differ']}; regions differ {res['calls_regions_differ']}")
        out_lines.append(f"convergence changes: unconverged->converged {len(res['conv_changes']['F->T'])}, "
                         f"converged->unconverged {len(res['conv_changes']['T->F'])}")
        for k in ("F->T", "T->F"):
            byt = collections.Counter(c["test"] for c in res["conv_changes"][k])
            out_lines.append(f"   {k} by test: {dict(byt)}")
            for c in res["conv_changes"][k][:40]:
                out_lines.append(f"   {k} seed {c['seed']} {c['test']} #{c['index']} {c['label']}: tree value {c['tree'][0]:.6g} "
                                 f"abserr {c['tree'][1]:.3g} regions {c['tree'][2]} depth {c['tree'][3]} | route value "
                                 f"{c['route'][0]:.6g} abserr {c['route'][1]:.3g} regions {c['route'][2]} depth {c['route'][3]}; "
                                 f"firings {c['firings']}")
        for g, name in (("ns", "near-singular test_YJJ_log_singularity cases (eps <= 1e-4)"), ("rest", "every other case")):
            c = res["cases"].get(g, {})
            out_lines.append(f"{name}: {c.get('n', 0)} cases; dishonest tree {c.get('dishonest_tree', 0)} -> {v} "
                             f"{c.get('dishonest_route', 0)}; dishonest and converged {c.get('dishonest_and_converged_tree', 0)} -> "
                             f"{c.get('dishonest_and_converged_route', 0)}; converged {c.get('converged_tree', 0)} -> "
                             f"{c.get('converged_route', 0)}; true error changed {c.get('true_error_changed', 0)}")
            for k in ("honest->dishonest", "dishonest->honest"):
                mv = res["moves"][g][k]
                out_lines.append(f"   {k}: {len(mv)}")
                for m in mv[:30]:
                    out_lines.append(f"      seed {m['seed']} {m['test']} {m['label']}: tree true {m['tree'][0]:.3e} reported "
                                     f"{m['tree'][1]:.3e} conv {m['tree'][2]} | {v} true {m['route'][0]:.3e} reported "
                                     f"{m['route'][1]:.3e} conv {m['route'][2]} (true/reported {m['ratio_route']:.3g})")
    oj, ot = prov.OUT / "compare.json", prov.OUT / "compare.txt"
    oj.write_text(json.dumps(doc, indent=1, default=str))
    ot.write_text("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))
    print(f"wrote {oj} sha256 {prov.sha256(oj)}; {ot} sha256 {prov.sha256(ot)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
