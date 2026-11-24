"""levin-phase-floor prompt 07d Part A' (section 9.2, the projection): the cost of v4 (state/refs_rule_leaf_v4.txt) on a
few Part A leaves of each knot class, timed for the projection. **No value is kept or printed:** only the wall, the
evaluation count, the panels a piece reached and whether the leaf settled (as 07d Part A's refs_costA_07d.py timed its
tool). Its output is read by the projection alone, never by a table. New in Part A'.
Usage: PYTHONPATH=<tree>:<scripts> python -B probe_v4_cost.py OUT
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, TREE_LQ_SHA, header, sha256  # noqa: E402
import leafref_v4 as R4  # noqa: E402


def leaf_index():
    """Every Part A leaf record (rounds 1-3), by id, with its spec."""
    idx = {}
    for rd in sorted((PARTA / "out" / "leaves").glob("round*")):
        for p in sorted(rd.glob("*.json")):
            doc = json.loads(p.read_text())
            spec = json.loads((PARTA / "out" / "leaves" / "specs" / rd.name / p.name).read_text())
            for L in doc["leaves"]:
                if "error" not in L:
                    idx[L["id"]] = (spec, L)
    return idx


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    assert sha256(LQ.__file__) == TREE_LQ_SHA
    idx = leaf_index()
    # pick, by a fixed rule on Part A's recorded n_knots and kind (no value read): per class, the first 3 ids in sorted order
    classes = {"analytic": [], "group k=0": [], "group k=1-4": [], "group k=5-30": [], "group k=31-100": [], "group k>100": []}
    for i in sorted(idx):
        spec, L = idx[i]
        kind = "analytic" if "|analytic|" in i else "group"
        k = L["n_knots"]
        if kind == "analytic":
            c = "analytic"
        elif k is None:
            continue
        elif k == 0:
            c = "group k=0"
        elif k <= 4:
            c = "group k=1-4"
        elif k <= 30:
            c = "group k=5-30"
        elif k <= 100:
            c = "group k=31-100"
        else:
            c = "group k>100"
        if len(classes[c]) < 3:
            classes[c].append(i)
    caps = {}
    rows = []
    for c, ids in classes.items():
        for i in ids:
            spec, L = idx[i]
            key = spec["capture"]["sha256"]
            if key not in caps:
                caps[key] = R4.load_capture(spec["capture"])
            rec = caps[key]
            xs0, xs1 = (float.fromhex(v) for v in spec["x_span"])
            edges, _, _ = R4.call_knots(rec, xs0, xs1)
            lo, hi = float.fromhex(L["start"]), float.fromhex(L["end"])
            kn = R4.interior(edges, lo, hi)
            G = R4.Integrand(LQ, rec)
            t0 = time.perf_counter()
            s3 = R4.v3_sigma(G, lo, hi, kn, L["V"])
            t1 = time.perf_counter()
            n1 = G.n_eval
            r4 = R4.v4_reference(G, lo, hi, kn, s3["sigma_hat"])
            t2 = time.perf_counter()
            rows.append({"class": c, "id": i, "n_knots": None if kn is None else len(kn), "V_over_pi": L["V"] / math.pi,
                         "wall_v3_sigma_s": t1 - t0, "wall_v4_s": t2 - t1, "evals_v3_sigma": n1,
                         "evals_v4": G.n_eval - n1, "max_panels": max(r4["panels"]), "settled": r4["settled"],
                         "sigma_equal_part_a": s3["sigma_hat"] == L["noise"]["sigma_hat"]})
            print(json.dumps(rows[-1]), flush=True)
    doc = {"header": header("levin-phase-floor 07d Part A': v4's cost, timed for the projection (no value kept)", __file__,
                            [R4.__file__]), "rows": rows}
    out.write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
