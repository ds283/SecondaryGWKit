"""levin-phase-floor 07 section 2.4: the LiouvilleGreen draws, the parent against section 1's code. New in 07.

Reads lg_seeded_draws.py's run file (out/lg_seeded_draws.jsonl; 07's copy of 06m's harness, modes ``parent`` and
``commit``) and scores section 2.4's three items:
1. **No test fails under the code that passes under the parent**: every (seed, test) unit's outcome under both.
2. **Every Levin call in which nothing is located and no leaf is guarded is bit-identical to the parent's**: the
   calls of one unit are paired by their index (the draw is seeded, and the calls are made in the same order);
   a commit call with no located point and num_guarded_regions 0 must equal the parent's call in value and abserr
   (float hex), in regions_full_sha256 (every returned region, field by field, 06m's digest) and in
   result_sha256_07 (every result key but elapsed, regions and the guard's six new keys).
3. **What location does on these calls is recorded**: every call that locates a point, with each point's kind,
   x, lambda for a minimum, its window, and the call's value, abserr and converged under both codes; and every
   call that guards a leaf.
The three-Bessel results each test asserts on (quad_JJJ / quad_YJJ: value, abserr, converged) are listed under
both codes for every unit that has a locating call.

Writes out/lg_compare_07.json and out/lg_compare_07.txt.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

RUN = prov.OUT / "lg_seeded_draws.jsonl"


def main():
    lines = RUN.read_text().splitlines()
    run_header = json.loads(lines[0])["header"]
    units = [json.loads(line) for line in lines[1:]]
    by = {(u["seed"], u["test"], u["mode"]): u for u in units}
    keys = sorted({(u["seed"], u["test"]) for u in units})
    out = {
        "header": prov.json_header(
            __file__, "section 2.4, the LiouvilleGreen draws compared", (RUN,)
        ),
        "run_header": run_header,
        "n_units": len(units),
    }
    outcomes = []
    moved_tests = []
    pairs = 0
    identical = 0
    not_identical = []
    locating = []
    guarding = []
    unpaired = []
    for seed, test in keys:
        p = by.get((seed, test, "parent"))
        c = by.get((seed, test, "commit"))
        outcomes.append(
            {
                "seed": seed,
                "test": test,
                "parent": None if p is None else p["outcome"],
                "commit": None if c is None else c["outcome"],
                "elapsed_parent": None if p is None else p["elapsed_s"],
                "elapsed_commit": None if c is None else c["elapsed_s"],
            }
        )
        if p is None or c is None:
            unpaired.append([seed, test])
            continue
        if c["outcome"] != "pass" and p["outcome"] == "pass":
            moved_tests.append(
                {
                    "seed": seed,
                    "test": test,
                    "commit": c["outcome"],
                    "detail": c["detail"][-2000:],
                }
            )
        if len(p["calls"]) != len(c["calls"]):
            unpaired.append([seed, test, len(p["calls"]), len(c["calls"])])
        unit_locates = False
        for pc, cc in zip(p["calls"], c["calls"]):
            pairs += 1
            pts = cc.get("stationary_points_07") or []
            guarded = cc.get("num_guarded_regions_07") or 0
            if pts or guarded:
                entry = {
                    "seed": seed,
                    "test": test,
                    "index": cc["index"],
                    "notify_label": cc["notify_label"],
                    "x_span": cc["x_span"],
                    "points": pts,
                    "window_pieces": cc.get("num_window_pieces_07"),
                    "guarded_regions": guarded,
                    "guard_at_depth_max": cc.get("num_guard_at_depth_max_07"),
                    "parent": {
                        "value": pc["value"],
                        "abserr": pc["abserr"],
                        "converged": pc["converged"],
                        "regions": pc["num_regions"],
                        "max_depth": pc["max_depth"],
                    },
                    "commit": {
                        "value": cc["value"],
                        "abserr": cc["abserr"],
                        "converged": cc["converged"],
                        "regions": cc["num_regions"],
                        "max_depth": cc["max_depth"],
                        "node_limited": cc.get("node_limited"),
                    },
                    "value_change": cc["value"] - pc["value"],
                    "value_change_over_parent_abserr": (
                        abs(cc["value"] - pc["value"]) / pc["abserr"]
                        if pc["abserr"] > 0
                        else None
                    ),
                    "value_change_over_commit_abserr": (
                        abs(cc["value"] - pc["value"]) / cc["abserr"]
                        if cc["abserr"] > 0
                        else None
                    ),
                    "section_2_5": cc.get("section_2_5_07"),
                }
                if pts:
                    locating.append(entry)
                    unit_locates = True
                if guarded:
                    guarding.append(entry)
                continue
            same = (
                pc["value_hex_07"] == cc["value_hex_07"]
                and pc["abserr_hex_07"] == cc["abserr_hex_07"]
                and pc["regions_full_sha256"] == cc["regions_full_sha256"]
                and pc["result_sha256_07"] == cc["result_sha256_07"]
            )
            if same:
                identical += 1
            else:
                not_identical.append(
                    {
                        "seed": seed,
                        "test": test,
                        "index": cc["index"],
                        "notify_label": cc["notify_label"],
                        "parent": [pc["value_hex_07"], pc["abserr_hex_07"]],
                        "commit": [cc["value_hex_07"], cc["abserr_hex_07"]],
                    }
                )
        if unit_locates:
            outcomes[-1]["results_parent"] = p["results"]
            outcomes[-1]["results_commit"] = c["results"]
    out["outcomes"] = outcomes
    out["item1_tests_failing_under_commit_that_pass_under_parent"] = moved_tests
    out["item2_calls_paired"] = pairs
    out["item2_calls_locating_nothing_guarding_nothing"] = identical + len(
        not_identical
    )
    out["item2_bit_identical"] = identical
    out["item2_not_identical"] = not_identical
    out["item3_calls_locating"] = locating
    out["item3_calls_guarding"] = guarding
    out["unpaired"] = unpaired
    kinds = {}
    for e in locating:
        for p in e["points"]:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    out["item3_point_kinds"] = kinds
    (prov.OUT / "lg_compare_07.json").write_text(json.dumps(out, indent=1, default=str))

    txt = (
        [f"# {x}" for x in out["header"]]
        + ["# the run's header:"]
        + [f"#   {x}" for x in run_header]
        + [""]
    )
    txt.append(
        f"units {len(units)}; (seed, test) pairs {len(keys)}; unpaired {unpaired}"
    )
    txt.append("")
    txt.append(
        "1. Tests failing under the commit that pass under the parent: "
        + (str(len(moved_tests)) if moved_tests else "none")
    )
    for m in moved_tests:
        txt.append(f"   seed {m['seed']} {m['test']}: {m['commit']}")
    fails = [o for o in outcomes if o["parent"] != "pass" or o["commit"] != "pass"]
    txt.append(
        f"   units not passing in either mode: {[(o['seed'], o['test'], o['parent'], o['commit']) for o in fails]}"
    )
    txt.append("")
    txt.append(
        f"2. Calls paired {pairs}; locating nothing and guarding nothing {identical + len(not_identical)}; "
        f"bit-identical to the parent {identical}; not identical {len(not_identical)}"
    )
    for n in not_identical:
        txt.append(
            f"   NOT IDENTICAL seed {n['seed']} {n['test']} #{n['index']}: {n['parent']} -> {n['commit']}"
        )
    txt.append("")
    txt.append(
        f"3. Calls locating a point: {len(locating)} (point kinds {kinds}); calls guarding a leaf: {len(guarding)}"
    )
    txt.append(
        "   seed | test | # | points (kind x [lambda]) | window pieces | parent value, abserr, conv | "
        "commit value, abserr, conv, node_limited | |dv|/abserr_parent, |dv|/abserr_commit | regions p->c | depth p->c"
    )
    for e in locating + [g for g in guarding if not g["points"]]:
        pts = "; ".join(
            f"{p['kind']} {p['x']:.8g}"
            + (f" [{p['lambda']:.3g}]" if p.get("lambda") is not None else "")
            for p in e["points"]
        )
        pp, cc = e["parent"], e["commit"]
        txt.append(
            f"   {e['seed']} | {e['test']} | {e['index']} | {pts} | {e['window_pieces']} | "
            f"{pp['value']:.10e}, {pp['abserr']:.2e}, {pp['converged']} | "
            f"{cc['value']:.10e}, {cc['abserr']:.2e}, {cc['converged']}, {cc['node_limited']} | "
            f"{e['value_change_over_parent_abserr'] if e['value_change_over_parent_abserr'] is None else format(e['value_change_over_parent_abserr'], '.3g')}, "
            f"{e['value_change_over_commit_abserr'] if e['value_change_over_commit_abserr'] is None else format(e['value_change_over_commit_abserr'], '.3g')} | "
            f"{pp['regions']}->{cc['regions']} | {pp['max_depth']}->{cc['max_depth']}"
            + (
                f" | guarded {e['guarded_regions']}, at depth_max {e['guard_at_depth_max']}"
                if e["guarded_regions"]
                else ""
            )
        )
    txt.append("")
    txt.append(
        "The three-Bessel results asserted on, in units with a locating call (value, abserr, converged):"
    )
    for o in outcomes:
        if "results_commit" not in o:
            continue
        txt.append(
            f"   seed {o['seed']} {o['test']}: parent {o['parent']}, commit {o['commit']}"
        )
        for rp, rc in zip(o["results_parent"], o["results_commit"]):
            txt.append(
                f"      {rp['kind']} {rp['orders']} kqs {['%.4g' % v for v in rp['kqs']]}: parent {rp['value']:.12e} "
                f"{rp['abserr']:.2e} {rp['converged']} | commit {rc['value']:.12e} {rc['abserr']:.2e} {rc['converged']}"
            )
    (prov.OUT / "lg_compare_07.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt[:60]))


if __name__ == "__main__":
    main()
