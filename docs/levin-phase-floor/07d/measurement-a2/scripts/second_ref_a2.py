"""levin-phase-floor prompt 07d Part A' (section 2.3, "the reference's own failure, checked by a second reference rule"):
on the leaves where a scored declaration understates against v4, an independent reference, and every value's error
against both. New in Part A'.

The second rule (fixed here, before it is read): composite Gauss-Legendre of order 48 (numpy leggauss(48)), on the leaf
split at the same knots as v4, at 256 and at 512 uniform panels a piece where the leaf has at most 10 pieces, at 64 and
128 where it has more (for cost: 48 points x 512 panels on 150 pieces is about 5 minutes a leaf) (math.fsum); ref2 = the
finer value, its stated error eps2 = |I_fine - I_coarse| + 2 sigma_hat sqrt(sum w^2 of the finer rule) (v4's form,
sigma_hat v4's). It differs from v4 in its order (48, not 24) and takes one panel count on every piece.

For each listed leaf: v4's ref and eps, ref2 and eps2, whether the two references agree within eps + eps2; and the error
against each of V_P, CC_47, CC_93, CC_185, the knot-split values (item 4's "all"), and every Q and declaration of an item-1
check that understates. The listed leaves are every leaf on which a declaration of tables 2, 3 or 4 understates (read from
out/tables/), the list written to the output.
Usage: PYTHONPATH=<tree>:<scripts> python -B second_ref_a2.py OUT
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, TREE_LQ_SHA, header, sha256  # noqa: E402

GX, GW = np.polynomial.legendre.leggauss(48)
fx = float.fromhex


def gl(G, brk, n):
    terms, w2 = [], 0.0
    for pa, pb in zip(brk[:-1], brk[1:]):
        h = (pb - pa) / n
        for j in range(n):
            x0 = pa + j * h
            x1 = pa + (j + 1) * h if j < n - 1 else pb
            ww = 0.5 * (x1 - x0) * GW
            terms.append(ww * G.g_on(0.5 * (x0 + x1) + 0.5 * (x1 - x0) * GX))
            w2 += float(np.sum(ww * ww))
    return math.fsum(np.concatenate(terms)), w2


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    assert sha256(LQ.__file__) == TREE_LQ_SHA
    import leafref_v4 as R4

    T = ROOT / "out" / "tables"
    ids = set()
    for u in json.loads((T / "t2_de_v4.json").read_text())["tree_understates"]:
        ids.add(u["id"])
    for u in json.loads((T / "t3_cc93.json").read_text())["understatements"]:
        ids.add(u["id"])
    for L in json.loads((T / "t4_dt.json").read_text())["per_leaf"]:
        for k, v in L["by_T"].items():
            if (isinstance(v, dict) and v.get("v") == "understates") or v == "understates":
                ids.add(L["id"])
    recs = {}
    for d in sorted((ROOT / "out" / "leaves_a2").iterdir()):
        if d.is_dir() and (d.name.startswith("partA_") or d.name.startswith("nb")):
            for p in sorted(d.glob("*.json")):
                doc = json.loads(p.read_text())
                for L in doc["leaves"]:
                    if L["id"] in ids:
                        recs[L["id"]] = (doc, L)
    rows = []
    caps = {}
    for i in sorted(ids):
        doc, L = recs[i]
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        key = spec["capture"]["sha256"]
        if key not in caps:
            caps = {key: R4.load_capture(spec["capture"])}
        rec = caps[key]
        xs0, xs1 = (float.fromhex(v) for v in spec["x_span"])
        edges, _, _ = R4.call_knots(rec, xs0, xs1)
        a, b = fx(L["start"]), fx(L["end"])
        kn = R4.interior(edges, a, b)
        brk = [a] + (kn or []) + [b]
        G = R4.Integrand(LQ, rec)
        n_hi = 512 if len(brk) - 1 <= 10 else 128
        i256, _ = gl(G, brk, n_hi // 2)
        i512, w2 = gl(G, brk, n_hi)
        sig = L["v3_sigma"]["sigma_hat"]
        ref2, eps2 = i512, abs(i512 - i256) + 2 * sig * math.sqrt(w2)
        ref, eps = fx(L["v4"]["ref"]), fx(L["v4"]["eps_ref"])
        vals = {"V_P": L["levin_one"].get("V_P"), "CC47": L["cc"]["cc47"], "CC93": L["cc"]["cc93"], "CC185": L["cc"]["cc185"]}
        if L.get("item4") and L["item4"]["split"].get("all"):
            sp = L["item4"]["split"]["all"]
            vals.update({"split V_P": sp.get("V_P"), "split CC47": sp["cc47"], "split CC93": sp["cc93"]})
        for c in L["item1"]:
            if c.get("checked") and c.get("trusted"):
                vals[f"Q T{c['T_pi']} {c['fallback']}"] = c["Q"]
        errs = {k: {"v4": abs(fx(v) - ref), "ref2": abs(fx(v) - ref2)} for k, v in vals.items() if v is not None}
        rows.append({"id": i, "n_knots": L["n_knots"], "panels_fine": n_hi, "V_over_pi": L["V_over_pi"], "ref_v4": ref, "eps_v4": eps,
                     "ref2": ref2, "eps2": eps2, "refs_agree": abs(ref - ref2) <= eps + eps2,
                     "refs_diff_over_eps_v4": abs(ref - ref2) / eps if eps else None, "errors": errs})
        print(i, "agree" if rows[-1]["refs_agree"] else "DISAGREE", f"|ref-ref2|/eps_v4 {rows[-1]['refs_diff_over_eps_v4']:.3g}",
              flush=True)
    out.write_text(json.dumps({"header": header("levin-phase-floor 07d Part A': the second reference on the leaves that "
                                                "understate", __file__, [T / "t2_de_v4.json", T / "t3_cc93.json",
                                                                         T / "t4_dt.json"]), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
