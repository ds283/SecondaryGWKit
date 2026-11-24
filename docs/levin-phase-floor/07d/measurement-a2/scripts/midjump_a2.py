"""levin-phase-floor prompt 07d Part A' (the orchestrator's second message, for issues.md): the size of f_0's and f_1's
value jump at the nearest-node switch midpoints ("mid" knots of production_edges' walk, leaf_a2.typed_knots), relative to
|f| near it, on a few leaves. New in Part A'.

**The leaves (fixed here, before any is read):** the group leaves of this Part's leaf records (out/leaves_a2/partA_round*
and nb1) whose item 4 counts at least one "mid" knot inside the leaf; sorted by leaf id; the first two of each stratum (the
id up to '#'), at most 16 leaves in all, in that order.
**The knots:** every mid knot inside the leaf, at most the 12 nearest the leaf's midpoint.
**The measurement**, for f_0 and f_1 as the integrand evaluates them (LQ._sample_vectorized, as item 4), at each knot xm
and at the leaf's control point (the midpoint of its largest knot-free gap, as item 4), for k = 1, 4 and 64 ulps
(np.spacing(xm)): the direct step d_k = |f(xm + k ulp) - f(xm - k ulp)|; the local scale |f|_loc = max(|f(xm - k ulp)|,
|f(xm + k ulp)|); and |f|_win, the largest |f| on 33 points across [xm - h, xm + h], h = 0.45 x the distance to the nearer
neighbouring knot of any kind. Reported: d_k / |f|_loc (None where |f|_loc = 0), d_k / |f|_win, and the control's same;
beside them item 4's fit-based value jump at the leaf's nearest mid knot over its value_abs.
Usage: PYTHONPATH=<tree>:<scripts> python -B midjump_a2.py OUT
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, TREE_LQ_SHA, header, sha256  # noqa: E402

fx = float.fromhex
MAX_LEAVES, PER_STRATUM, MAX_KNOTS = 16, 2, 12
ULPS = (1, 4, 64)


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    assert sha256(LQ.__file__) == TREE_LQ_SHA
    import checkrepro_a2 as CR
    import leafref_v4 as R4
    from leaf_a2 import typed_knots

    cand = []
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if d.is_dir() and (d.name.startswith("partA_") or d.name.startswith("nb")):
            for p in sorted(d.glob("*.json")):
                doc = json.loads(p.read_text())
                for L in doc["leaves"]:
                    i4 = L.get("item4")
                    if i4 and i4["counts"].get("mid", 0) > 0 and "|group|" in L["id"]:
                        cand.append((L["id"], str(p)))
    cand.sort()
    chosen, per = [], {}
    for i, p in cand:
        st = i.split("#")[0]
        if per.get(st, 0) < PER_STRATUM and len(chosen) < MAX_LEAVES:
            per[st] = per.get(st, 0) + 1
            chosen.append((i, p))
    rows = []
    for i, p in chosen:
        doc = json.loads(Path(p).read_text())
        L = next(x for x in doc["leaves"] if x["id"] == i)
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        rec = R4.load_capture(spec["capture"])
        kw = dict(rec["kw"])
        kw["rtol"] = float(spec["rtol"])
        C = CR.CallContext(LQ, rec, kw)
        a, b = fx(L["start"]), fx(L["end"])
        tk = typed_knots(rec)
        allk = sorted({x for v in tk.values() for x in v if a < x < b})
        mids = sorted((x for x in tk["mid"] if a < x < b), key=lambda x: abs(x - 0.5 * (a + b)))[:MAX_KNOTS]
        pts = [a] + allk + [b]
        gaps = sorted(((pts[j + 1] - pts[j], j) for j in range(len(pts) - 1)), reverse=True)
        ctrl = 0.5 * (pts[gaps[0][1]] + pts[gaps[0][1] + 1])
        f0, f1 = rec["f"]
        funcs = {"f0": f0, "f1": f1}

        def F(fn, xs):
            return np.asarray(LQ._sample_vectorized(fn, np.asarray(xs, dtype=float), C.cache, ("f", id(fn))), dtype=float)

        def measure(fn, x0, h):
            u = float(np.spacing(x0))
            r = {}
            win = F(fn, np.linspace(x0 - h, x0 + h, 33))
            fwin = float(np.max(np.abs(win)))
            for k in ULPS:
                lo, hi = F(fn, [x0 - k * u, x0 + k * u])
                d = abs(float(hi) - float(lo))
                loc = max(abs(float(lo)), abs(float(hi)))
                r[str(k)] = {"step": d, "f_loc": loc, "rel_loc": d / loc if loc > 0 else None,
                             "rel_win": d / fwin if fwin > 0 else None}
            r["f_win"] = fwin
            return r

        knots = []
        for xm in mids:
            nb = [x for x in pts if x != xm]
            h = 0.45 * min(abs(x - xm) for x in nb)
            knots.append({"x": xm.hex(), "h": h, **{nm: measure(fn, xm, h) for nm, fn in funcs.items()}})
        hc = 0.45 * 0.5 * gaps[0][0]
        control = {"x": ctrl.hex(), "h": hc, **{nm: measure(fn, ctrl, hc) for nm, fn in funcs.items()}}
        i4 = (L["item4"]["jumps"] or {}).get("mid") or {}
        fit = {nm: ({"jump0": i4[nm]["jump"][0], "value_abs": i4[nm]["value_abs"],
                     "rel": (i4[nm]["jump"][0] / max(i4[nm]["value_abs"]) if max(i4[nm]["value_abs"]) > 0 else None)}
                    if nm in i4 and "jump" in i4[nm] else None) for nm in ("f0", "f1")}
        rows.append({"id": i, "n_mid": L["item4"]["counts"]["mid"], "n_measured": len(knots), "knots": knots,
                     "control": control, "item4_fit": fit})
        r1 = [k[nm]["1"]["rel_loc"] for k in knots for nm in ("f0", "f1") if k[nm]["1"]["rel_loc"] is not None]
        print(i, len(knots), "max rel_loc(1 ulp)", max(r1) if r1 else None, flush=True)

    def summ(key):
        s = {}
        for nm in ("f0", "f1"):
            for k in map(str, ULPS):
                kn = [r_[nm][k][key] for row in rows for r_ in row["knots"] if r_[nm][k][key] is not None]
                ct = [row["control"][nm][k][key] for row in rows if row["control"][nm][k][key] is not None]
                s[f"{nm} {k} ulp"] = {"knots_n": len(kn), "knots_median": float(np.median(kn)) if kn else None,
                                      "knots_max": max(kn) if kn else None,
                                      "knots_zero_step": sum(1 for row in rows for r_ in row["knots"] if r_[nm][k]["step"] == 0),
                                      "control_n": len(ct), "control_median": float(np.median(ct)) if ct else None,
                                      "control_max": max(ct) if ct else None}
        return s

    doc = {"header": header("levin-phase-floor 07d Part A': the value jump at the nearest-node midpoints, relative to |f|",
                            __file__, sorted({Path(p) for _, p in chosen})),
           "selection": [i for i, _ in chosen], "summary_rel_loc": summ("rel_loc"), "summary_rel_win": summ("rel_win"),
           "rows": rows}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["summary_rel_loc"], indent=1))


if __name__ == "__main__":
    main()
