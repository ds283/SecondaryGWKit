"""levin-phase-floor prompt 07d Part A' (section 9.2, "the projection comes first"): the projection of item 0's references,
item 3's new band, items 1, 2, 4 and 5's leaf work, item 6's variants and any new call references. New in Part A'.

Reads only counts and costs: Part A's leaf records' ids, kinds and n_knots (no value), probe_v4_cost.json's walls, and
Part A's registered runs' walls (variants_results.json, leaves-v3_results.json, refs_results.json).
Model (each term stated in the output):
- v4 per leaf: T_sigma + c_eval * E_piece * (n_knots + 1), c_eval and E_piece fitted on the probe (wall over evaluations,
  evaluations over pieces); a leaf whose knots are unknown counts as one piece;
- items 1, 2, 4, 5 per leaf: Part A's leafcc_07d.py wall per leaf (leaves-v3 run: its wall x 6 workers / leaves) times a
  stated factor for the extra solves (item 1: seven T and two fallbacks; item 4: splits by kind);
- the new band [20 pi, 32 pi): 4 strata (kind x rtol) at Part A's largest band draw (80 leaves), with band_16_20's knots;
- the instrumented run: Part A's tree replay of the 4,220 calls (Eg's wall / 1.31) x 1.2;
- the variants: 8, each at the wall of the Part A variant most like it (V93 ~ Ea; V93-T ~ ET12, ET20, T20; K, KT, G and
  one other ~ ET20), x 1.2;
- new call references: at most 30 readings at Part A's largest per-reading elapsed (refs run), two at a time;
- a margin of 1.5 on everything (07c's issues item 9 found the references' projection a third high; this Part's models are
  new, so a margin over, not under).
Usage: python -B projection_a2.py OUT_JSON OUT_MD
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, ROOT, header  # noqa: E402

MARGIN = 1.5
WORKERS = 6


def main():
    oj, om = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (oj, om):
        if p.exists():
            raise SystemExit(f"{p} exists")
    probe = json.loads((ROOT / "out" / "projection" / "probe_v4_cost.json").read_text())["rows"]
    c_eval = sum(r["wall_v4_s"] for r in probe) / sum(r["evals_v4"] for r in probe)
    e_piece = statistics.median(r["evals_v4"] / ((r["n_knots"] or 0) + 1) for r in probe)
    e_piece_max = max(r["evals_v4"] / ((r["n_knots"] or 0) + 1) for r in probe)
    t_sigma = max(r["wall_v3_sigma_s"] for r in probe)
    leaves = {}
    for rd in sorted((PARTA / "out" / "leaves").glob("round*")):
        for p in sorted(rd.glob("*.json")):
            doc = json.loads(p.read_text())
            for L in doc["leaves"]:
                if "error" in L:
                    continue
                leaves[L["id"]] = (doc["sid"], L["start"], L.get("n_knots"))
    distinct = {}
    for i, (sid, start, k) in leaves.items():
        distinct[(sid, start)] = k
    kn = list(distinct.values())
    v4_cpu = sum(t_sigma + c_eval * e_piece * ((k or 0) + 1) for k in kn)
    v4_cpu_hi = sum(t_sigma + c_eval * e_piece_max * ((k or 0) + 1) for k in kn)
    band1620 = [k for i, (sid, s, k) in leaves.items() if i.startswith("band_16_20|group")]
    new_band = 4 * 80
    nb_cpu = new_band * (t_sigma + c_eval * e_piece_max * (statistics.mean([(k or 0) for k in band1620]) + 1))
    lv3 = json.loads((PARTA / "out" / "leaves-v3_results.json").read_text())["results"][0]["wall_s"]
    n_v3 = sum(1 for _ in leaves)
    leafcc_cpu_per_leaf = lv3 * WORKERS / n_v3
    ITEMS_FACTOR = 8.0  # items 1, 2, 4, 5: seven T x two fallbacks of Levin checks, the splits by knot kind, the timings
    items_cpu = (len(kn) + new_band) * leafcc_cpu_per_leaf * ITEMS_FACTOR
    var = {r["name"]: r["wall_s"] for r in json.loads((PARTA / "out" / "variants_results.json").read_text())["results"]}
    tree_wall = var["replay_Eg"] / 1.31
    instr = tree_wall * 1.2
    variants = 1.2 * (var["replay_Ea"] + var["replay_ET12"] + var["replay_ET20"] + var["replay_T20"]
                      + 4 * var["replay_ET20"])
    refs = json.loads((PARTA / "out" / "refs_results.json").read_text())["results"]
    per = max(r["wall_s"] for r in refs)
    new_refs = 30 * per / 2
    rows = {
        "instrumented run at 6 pi (4,220 calls), with the new band's population": instr,
        "item 0: v4 on Part A's D-E and D-T leaves (%d distinct of %d drawn), central" % (len(kn), n_v3): v4_cpu / WORKERS,
        "item 0: the same at the probe's largest evaluations a piece": v4_cpu_hi / WORKERS,
        "item 3: the new band, 4 strata x 80 leaves, v4 at the largest": nb_cpu / WORKERS,
        "items 1, 2, 4, 5 on every leaf": items_cpu / WORKERS,
        "item 6: eight variants x 4,220 calls": variants,
        "new call references: at most 30 readings, two at a time": new_refs,
    }
    central = (rows[list(rows)[0]] + rows[list(rows)[1]] + rows[list(rows)[3]] + rows[list(rows)[4]] + rows[list(rows)[5]]
               + rows[list(rows)[6]])
    high = central - rows[list(rows)[1]] + rows[list(rows)[2]]
    doc = {"header": header("levin-phase-floor 07d Part A': the projection (section 9.2)", __file__,
                            [ROOT / "out" / "projection" / "probe_v4_cost.json", PARTA / "out" / "leaves-v3_results.json",
                             PARTA / "out" / "variants_results.json", PARTA / "out" / "refs_results.json"]),
           "model": {"c_eval_s": c_eval, "evals_per_piece_median": e_piece, "evals_per_piece_max": e_piece_max,
                     "t_sigma_s": t_sigma, "leafcc_cpu_s_per_leaf": leafcc_cpu_per_leaf, "items_factor": ITEMS_FACTOR,
                     "tree_replay_wall_s": tree_wall, "ref_reading_max_s": per, "margin": MARGIN, "workers": WORKERS},
           "knots_distinct_leaves": dict(sorted(Counter("unknown" if k is None else "0" if k == 0 else "1-4" if k <= 4
                                                        else "5-30" if k <= 30 else "31-100" if k <= 100 else ">100"
                                                        for k in kn).items())),
           "rows_s": rows, "central_s": central, "high_s": high,
           "central_with_margin_h": central * MARGIN / 3600, "high_with_margin_h": high * MARGIN / 3600, "limit_h": 36}
    oj.write_text(json.dumps(doc, indent=1))
    L = doc["header"] + ["", "| item | elapsed (s) |", "|---|---|"] + [f"| {k} | {v:,.0f} |" for k, v in rows.items()]
    L += ["", f"central {central:,.0f} s; with the v4 high case {high:,.0f} s; x{MARGIN} margin: "
          f"{central * MARGIN / 3600:.1f} h central, {high * MARGIN / 3600:.1f} h high, against the 36 h limit",
          f"model: {json.dumps(doc['model'])}", f"knots on the distinct Part A leaves: {doc['knots_distinct_leaves']}"]
    om.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
