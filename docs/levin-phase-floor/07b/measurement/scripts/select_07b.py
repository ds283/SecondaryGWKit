"""levin-phase-floor prompt 07b section 2.4: the reference calls, selected before any reading, with the seeds of
state/seeds_07b.json (written before this script ran). New in 07b. Read only but for its output.

- **R1, class (c):** 30 of the 396 rows of 07's cross-check (07's state/xcheck_capture_selection.json), stratified by
  (band, k_key) of rows.json in proportion: each stratum gets floor(30 n_s / 396), and the remaining picks go to the
  largest remainders (ties: the stratum's sorted key); within a stratum the rows are drawn without replacement by
  ``rng.choice`` on the rows sorted by sid, one generator for all strata, strata in sorted order. Each row's call is its
  stationary-point call at 1e-8 (unit h-1e-8-<row>, the selection entry's index), with 07's capture sid noted.
- **R2, the whole census:** the population is every call of 07a's census at both rtols, group and analytic: every line of
  07a's stop-3 call comparison (``out/stop3/compare_census_calls.calls.jsonl``), one per call at 07a's code, sorted by
  (unit, kind, index). 30 drawn uniformly without replacement, kept in draw order; a cut to 20 (section 2.4) keeps the
  first 20 of that order. The strata each lands in are recorded.
- **R3, targeted:**
  - W7: ``h-1e-8-0-33950`` and ``h-1e-8-0-2006`` (index 0), and the nineteen-minimum call at k = 1e5 that moved most
    (07a's out/scores/w7_detail.json, ``violating``, largest ``move_over_abserr07``);
  - auxiliary routes: ``3-843`` analytic J2 (index 5) at 1e-7, ``3-13508`` G+q-r (index 2) at 1e-7; and the four of the
    47 shrinkers (moves_beyond.json ``aux_shrink_gt_2x``) whose abserr shrank most, other than row 3-678, counted as four
    distinct (row, kind, index): the smallest ``abserr_h_over_07`` first, each at its rtol of first appearance (1e-7
    before 1e-8 on a tie);
  - own route changes: three calls drawn uniformly from the own-class calls whose value differs between 07 and 07a
    (the comparison's ``class == "own"`` and ``value[0] != value[1]``), other than row 1-34599, sorted by (unit, kind,
    index) before the draw;
  - the two pairs: 3-14501 and 3-35006, group index 1, at 1e-7 (one reference per pair scores both rtols).

Each entry carries 07a's (``census_*``) and 07's (``census07_*``) records from the comparison, as x1_capture_07a.py and
check_captures_07a.py read them. Writes OUT_JSON (refuses to overwrite).
Usage: PYTHONPATH=<tree>:<scripts> python -B select_07b.py OUT_JSON
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07, REC07A, WIDER, header, sha256  # noqa: E402

SEEDS = WIDER / "state" / "seeds_07b.json"
SEEDS_SHA = "6d99b42efba57fc3b72b4cc6a872d63371220cfd874657046ac4c6aee6ed24fe"
XSEL = REC07 / "state" / "xcheck_capture_selection.json"
ROWS = REC07A / "state" / "rows.json"
CALLS = REC07A / "out" / "stop3" / "compare_census_calls.calls.jsonl"
MB = REC07A / "out" / "scores" / "moves_beyond.json"
W7 = REC07A / "out" / "scores" / "w7_detail.json"
N_R1, N_R2, N_OWN = 30, 30, 3


def entry(c, set_, sid, **extra):
    return {"sid": sid, "set": set_, "unit": c["unit"], "g_unit": "g-" + c["unit"].split("-", 1)[1], "index": c["index"],
            "kind": c["kind"], "notify_label": c["label"], "class": c["class"], "rtol": c["rtol"],
            "census_value": c["value"][0], "census_abserr": c["abserr"][0], "census_num_regions": c["num_regions"][0],
            "census_max_depth": c["max_depth"][0], "census_converged": c["converged"][0],
            "census07_value": c["value"][1], "census07_abserr": c["abserr"][1], "census07_num_regions": c["num_regions"][1],
            "census07_max_depth": c["max_depth"][1], "census07_converged": c["converged"][1], **extra}


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if sha256(SEEDS) != SEEDS_SHA:
        raise SystemExit("the seeds file has another SHA-256")
    seeds = json.loads(SEEDS.read_text())["seeds"]
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads(ROWS.read_text())["rows"]}
    calls = {}
    with open(CALLS) as f:
        for line in f:
            c = json.loads(line)
            calls[(c["unit"], c["kind"], c["index"])] = c
    keys_sorted = sorted(calls)

    def row_of(unit):
        return "-".join(unit.split("-")[3:])

    def stratum(c):
        R = rows[row_of(c["unit"])]
        return {"kind": c["kind"], "band": R["band"], "k_key": R["k_key"], "k": R["k"], "rtol": c["rtol"], "class": c["class"],
                "label": c["label"]}

    sel = []
    # R1
    x = json.loads(XSEL.read_text())
    by_s = defaultdict(list)
    for e in x:
        p = e["unit"].split("-")
        R = rows[f"{p[3]}-{p[4]}"]
        by_s[(R["band"], R["k_key"])].append(e)
    n_tot = len(x)
    quota = {s: (N_R1 * len(v)) // n_tot for s, v in by_s.items()}
    rem = sorted(by_s, key=lambda s: (-((N_R1 * len(by_s[s])) % n_tot), s))
    for s in rem[: N_R1 - sum(quota.values())]:
        quota[s] += 1
    rng1 = np.random.default_rng(seeds["R1_class_c_within_strata"])
    alloc = {}
    for s in sorted(by_s):
        pool = sorted(by_s[s], key=lambda e: e["sid"])
        pick = rng1.choice(len(pool), size=quota[s], replace=False) if quota[s] else []
        alloc["/".join(s)] = {"n_rows": len(pool), "quota": quota[s], "share": N_R1 * len(pool) / n_tot}
        for i in sorted(int(j) for j in pick):
            e = pool[i]
            p = e["unit"].split("-")
            unit = f"h-1e-8-{p[3]}-{p[4]}"
            c = calls[(unit, e["kind"], e["index"])]
            if c["label"] != e["notify_label"]:
                raise SystemExit(f"{e['sid']}: label differs")
            sel.append(entry(c, "R1", f"R1-{e['sid']}", stratum=stratum(c), xcheck_sid=e["sid"], xcheck_unit=e["unit"]))
    if sum(1 for e in sel if e["set"] == "R1") != N_R1:
        raise SystemExit("R1 is not 30")
    # R2
    rng2 = np.random.default_rng(seeds["R2_whole_census"])
    order = rng2.choice(len(keys_sorted), size=N_R2, replace=False)
    for rank, j in enumerate(int(j) for j in order):
        c = calls[keys_sorted[j]]
        sel.append(entry(c, "R2", f"R2-{rank:02d}", stratum=stratum(c), draw_rank=rank, population_index=j))
    # R3: W7
    w7 = json.loads(W7.read_text())["violating"]
    w7_ids = [("h-1e-8-0-33950", 0), ("h-1e-8-0-2006", 0)]
    v19 = sorted((v for v in w7 if v["points"].get("minimum") == 19 and float(v["k"]) == 1e5),
                 key=lambda v: -v["move_over_abserr07"])
    w7_ids.append((v19[0]["unit"], v19[0]["index"]))
    for u, i in w7_ids:
        c = calls[(u, "group", i)]
        mv = [v for v in w7 if v["unit"] == u and v["index"] == i][0]
        sel.append(entry(c, "R3-W7", f"R3w-{row_of(u)}-{i}", stratum=stratum(c), w7_move_over_abserr07=mv["move_over_abserr07"],
                         w7_points=mv["points"]))
    # R3: auxiliary routes
    for u, k, i in (("h-1e-7-3-843", "analytic", 5), ("h-1e-7-3-13508", "group", 2)):
        c = calls[(u, k, i)]
        sel.append(entry(c, "R3-aux", f"R3a-{row_of(u)}-{k[0]}{i}", stratum=stratum(c), aux_reason="moved beyond 07's abserr"))
    mb = json.loads(MB.read_text())
    seen = set()
    for s in sorted(mb["aux_shrink_gt_2x"], key=lambda s: (s["abserr_h_over_07"], s["unit"].split("-")[1:3] != ["1e", "7"], s["unit"], s["index"])):
        rk = (row_of(s["unit"]), s["kind"], s["index"])
        if rk[0] == "3-678" or rk in seen:
            continue
        seen.add(rk)
        c = calls[(s["unit"], s["kind"], s["index"])]
        sel.append(entry(c, "R3-aux", f"R3a-{rk[0]}-{rk[1][0]}{rk[2]}", stratum=stratum(c),
                         aux_reason="shrinker", abserr_h_over_07=s["abserr_h_over_07"]))
        if len(seen) == 4:
            break
    # R3: own route changes
    own = [k for k in keys_sorted if calls[k]["class"] == "own" and calls[k]["value"][0] != calls[k]["value"][1]
           and row_of(k[0]) != "1-34599"]
    rng3 = np.random.default_rng(seeds["R3_own_route_changes"])
    for j in sorted(int(j) for j in rng3.choice(len(own), size=N_OWN, replace=False)):
        c = calls[own[j]]
        sel.append(entry(c, "R3-own", f"R3o-{row_of(c['unit'])}-{c['kind'][0]}{c['index']}-{c['rtol'][-1]}",
                         stratum=stratum(c), own_population=len(own), own_index=j))
    # R3: the pairs
    for row in ("3-14501", "3-35006"):
        c = calls[(f"h-1e-7-{row}", "group", 1)]
        c8 = calls[(f"h-1e-8-{row}", "group", 1)]
        sel.append(entry(c, "R3-pair", f"R3p-{row}", stratum=stratum(c), pair_1e8={"value": c8["value"], "abserr": c8["abserr"],
                         "converged": c8["converged"], "num_regions": c8["num_regions"], "max_depth": c8["max_depth"]}))
    sids = [e["sid"] for e in sel]
    if len(set(sids)) != len(sids):
        raise SystemExit("duplicate sids")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: section 2.4's reference calls, selected",
                 [SEEDS, XSEL, ROWS, CALLS, MB, W7])
    doc = {"header": hdr, "seeds": seeds, "r1_allocation": alloc, "r2_population": len(keys_sorted),
           "r3_own_population": len(own), "calls": sel}
    out_p.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print("R1 allocation:", json.dumps(alloc))
    print("R2 population", len(keys_sorted), "; own population", len(own))
    for e in sel:
        s = e["stratum"]
        print(f"{e['sid']:28s} {e['unit']:18s} {e['kind']:8s} #{e['index']:<2d} {s['band']:5s} {s['k_key']:6s} k={float(s['k']):.3g} "
              f"class {e['class']:9s} conv {e['census_converged']} regions {e['census_num_regions']} | {e['notify_label'][-40:]}")
    print(Counter(e["set"] for e in sel))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
