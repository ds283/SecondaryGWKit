"""levin-phase-floor prompt 06k, Part L for route 2 literal (after planned stop 1, on the user's ruling): every
L-c withdrawal listed in full, and the leaves' corrections by the kinds of their seven solves. New in 06k. Read
only: out/partL/<sid>.json (probe_l.py, whose ``route2_literal`` decision is the literal rule).

A **withdrawal** is a returned leaf that 06g trusted (L-c) and the literal rule does not trust. For each: the call,
the leaf (depth, span), its parent's branch and its seven solves' branches, r, delta1, phi_P, delta1c, r_c, and both
declarations. 06g's is max(residual, delta1)/(1 - r). The literal one is what the variant declares for an untrusted
leaf: None if it is below depth_max (20; the leaf is then refined), else max(residual, delta1c)/(1 - r_c) if
r_c < 1, else max(residual, delta1c + delta2c).

**The corrections by solve kind** (the claim that a leaf with no E is never changed): over L-a, L-b and L-c, every
leaf is classed by its parent's branch and by whether any of its seven solves is a Clenshaw-Curtis solve (one that
can carry an E). Within each class it counts the leaves whose corrected numbers (delta1c, delta2c) differ from the
uncorrected ones, and the leaves whose literal decision differs from 06g's.

Writes out/partL_route2lit_withdrawals.csv and out/partL_route2lit_by_solve_kind.json.
Usage: python -B withdrawals_l.py
"""

import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

DEPTH_MAX = 20
SOLVES = ("P", "L", "R", "LL", "LR", "RL", "RR")


def fx(v):
    return None if v is None else float(v).hex()


def main():
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    files = [RC.ROOT / "out" / "partL" / f"{e['sid']}.json" for e in sel]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06k: route 2 literal at leaf level, withdrawals and solve kinds", files)
    rows = []
    classes = collections.defaultdict(collections.Counter)
    for e, f in zip(sel, files):
        d = json.loads(f.read_text())
        for L in d["leaves"]:
            kinds = L["types"]
            any_cc = any(kinds[k] == "direct" for k in SOLVES)
            cls = f"{L['population']}|{'CC' if kinds['P'] == 'direct' else 'Levin'} parent|{'some CC solve' if any_cc else 'all seven Levin'}"
            c = classes[cls]
            c["leaves"] += 1
            c["corrected_numbers_differ"] += fx(L["delta1c"]) != fx(L["delta1"]) or fx(L["delta2c"]) != fx(L["delta2"])
            c["any_E_nonzero"] += any((L["E"][k] or 0.0) != 0.0 for k in SOLVES)
            c["literal_decision_differs"] += L["decision"]["route2_literal"] != L["decision"]["g"]
            if L["population"] != "L-c" or L["decision"]["route2_literal"] == "trusted":
                continue
            r, rc = L["r"], L["r_c"]
            dec_g = max(L["residual"], L["delta1"]) / (1.0 - r)
            if L["depth"] < DEPTH_MAX:
                dec_lit = None
            elif rc < 1.0:
                dec_lit = max(L["residual"], L["delta1c"]) / (1.0 - rc)
            else:
                dec_lit = max(L["residual"], L["delta1c"] + L["delta2c"])
            rows.append({
                "sid": d["sid"], "unit": d["unit"], "index": d["index"], "rtol": d["rtol"],
                "group": e.get("g_record", {}).get("group"), "depth": L["depth"],
                "start": L["start_hex"], "end": L["end_hex"], "start_float": L["start"], "end_float": L["end"],
                "parent_branch": kinds["P"], "solve_branches": "/".join(kinds[k] for k in SOLVES),
                "r": r, "delta1": L["delta1"], "delta2": L["delta2"], "phi_P": L["phi_P"],
                "d1_over_phiP": L["d1_over_phiP"], "delta1c": L["delta1c"], "delta2c": L["delta2c"], "r_c": rc,
                "d1c_over_phiP": L["d1c_over_phiP"], "residual": L["residual"],
                "declared_g": dec_g, "declared_literal": dec_lit,
                "literal_note": "refined (below depth_max)" if dec_lit is None else "returned untrusted at depth_max",
                "r_lt_quarter": r < 0.25,
            })
    out_csv = RC.ROOT / "out" / "partL_route2lit_withdrawals.csv"
    with open(out_csv, "w", newline="") as fo:
        for l in hdr:
            fo.write(l + "\n")
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summ = {
        "header": hdr, "n_withdrawals": len(rows),
        "by_r": {"r<1/4": sum(1 for r in rows if r["r"] < 0.25), "r>=1/4": sum(1 for r in rows if r["r"] >= 0.25)},
        "by_r_c": dict(collections.Counter("r_c<1/4" if r["r_c"] < 0.25 else ("r_c<1" if r["r_c"] < 1 else "r_c>=1") for r in rows)),
        "by_parent_branch": dict(collections.Counter(r["parent_branch"] for r in rows)),
        "by_depth": dict(sorted(collections.Counter(r["depth"] for r in rows).items())),
        "by_call": dict(collections.Counter(r["sid"] for r in rows)),
        "at_depth_max": sum(1 for r in rows if r["declared_literal"] is not None),
        "classes_by_solve_kind": {k: dict(v) for k, v in sorted(classes.items())},
    }
    out_j = RC.ROOT / "out" / "partL_route2lit_by_solve_kind.json"
    out_j.write_text(json.dumps(summ, indent=1, default=str))
    for l in hdr:
        print(l)
    print(json.dumps({k: v for k, v in summ.items() if k != "header"}, indent=1, default=str))
    print(f"wrote {out_csv} sha256 {RC.sha256(out_csv)}; {out_j} sha256 {RC.sha256(out_j)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
