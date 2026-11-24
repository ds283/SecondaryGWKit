"""levin-phase-floor 07d Part A' addendum, item B: the call R:a:W2-0-12977-5 under the tree, V93T8, V93T10 and V93T12,
region by region. New in the addendum; the rule is fixed here before anything is read.

**Inputs (read only):** Part A's tree replay (measurement/out/replays/tree/), this Part's replays at V93T8, V93T10 and
V93T12 (measurement-a2/out/replays/<v>/), Part A's call reference for the call (measurement/out/tables/scores_07d.json
``calls``, 25 digits), and the call's capture (as replay_a2.py loads it).
**Differing regions:** a returned region (start, end) is "common" if it is returned with the same span, type and value
(hex) under all four codes; every other returned region of any code is "differing". For each differing region:
its V (the tree's chebyshev_matrices at the call's order and build_Levin_data, as leaf_a2.py), its route (``type``),
its value, its declared error (``total_err`` and ``budget_err``), its depth, and its true error against ref48
(state/rule_ref48_add.md section A.2: 48-point Gauss-Legendre at 512 and 1,024 panels a piece, 2,048 where the last
doubling still moves, noise 2 sigma_hat sqrt(sum w^2) with sigma_hat from leafref_v4.v3_sigma on the region; pieces at the knots leafref_v4.call_knots reads, the whole region where none). A region returned
under several codes with the same span is measured once.
**Per code:** the call's value, abserr, |value - ref_call|, the ratio; over its differing regions: the sum of (value -
ref48), its absolute sum, the sum of declared errors, and per region the ratio declared / |true error|. The regions are
classed: CC with V in [10, 12) pi ("CC_93 leaf, new under V93T12"), CC with V in [8, 10) pi, CC with V < 8 pi, Levin.
A Levin region with the same span under two codes but another value is listed as "same span, value differs" (a change
in its trust check's sub-solves, or its own solve's).
Writes addendum/out/dissect_b.json and .md.
Usage: PYTHONPATH=<tree>:<a2 scripts>:<addendum scripts> python -B dissect_b.py measure I N | combine
"""

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

mp.dps = 40
ADD = Path(__file__).resolve().parent.parent
A2 = ADD.parent
PARTA = A2.parent / "measurement"
SID = "R:a:W2-0-12977-5"
CODES = ["tree", "V93T8", "V93T10", "V93T12"]
fx = float.fromhex


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def main():
    import AdaptiveLevin.levin_quadrature as LQ
    import checkrepro_a2 as CR
    import leafref_v4 as R4
    from prov import TREE_LQ_SHA, header, sha256
    from ref48_add import gl
    from replay_07d import load_capture

    assert sha256(LQ.__file__) == TREE_LQ_SHA
    rp = {c: (PARTA if c == "tree" else A2) / "out" / "replays" / c / f"{fname(SID)}.json" for c in CODES}
    rep = {c: json.loads(p.read_text()) for c, p in rp.items()}
    sc = json.loads((PARTA / "out" / "tables" / "scores_07d.json").read_text())["calls"][SID]
    ref_call = mpf(sc["ref"])
    man = {}
    for p in sorted((PARTA / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    c = man[SID]
    rec = load_capture(c)
    kw = dict(rec["kw"])
    if c["override"]:
        census = {}
        for p in sorted((PARTA / "state").glob("census_refs_07d*.pkl")):
            census.update(pickle.loads(p.read_bytes())["calls"])
        cen = None if c["census"].startswith("r32:") else census[c["census"]]
        kw["rtol"] = c["override"]["rtol"]
        kw["atol"] = c["override"]["atol"] if c["override"].get("atol") is not None else cen["call"]["atol"]
    C = CR.CallContext(LQ, rec, kw)
    N = C.N
    G = R4.Integrand(LQ, rec)
    xs0, xs1 = min(rec["x_span"]), max(rec["x_span"])
    edges, _, knote = R4.call_knots(rec, xs0, xs1)

    def key(l):
        return (l["start"], l["end"])

    sig = {cd: {(key(l), l["type"], l["value"]) for l in rep[cd]["leaves"]} for cd in CODES}
    common = set.intersection(*sig.values())
    spans = {}
    for cd in CODES:
        for l in rep[cd]["leaves"]:
            if (key(l), l["type"], l["value"]) not in common:
                spans.setdefault(key(l), []).append(cd)
    # (run as a registered run) mode "measure I N": measure the regions I, I+N, ... of the sorted list into
    # out/dissect_b_meas/<index>.json; mode "combine": read them all and write the tables
    mode = sys.argv[1] if len(sys.argv) > 1 else "combine"
    order = sorted(spans, key=lambda s: fx(s[0]))
    mdir = ADD / "out" / "dissect_b_meas"
    if mode == "combine":
        meas = {}
        for k, sp in enumerate(order):
            m = json.loads((mdir / f"{k:03d}.json").read_text())
            assert (m["start"], m["end"]) == sp
            m["ref48"] = fx(m["ref48"])
            meas[sp] = m
    else:
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        mdir.mkdir(parents=True, exist_ok=True)
    for k, sp in enumerate(order):
        if mode == "combine" or k % nparts != part:
            continue
        if (mdir / f"{k:03d}.json").exists():
            raise SystemExit(f"{k:03d}.json exists")
        a, b = fx(sp[0]), fx(sp[1])
        grid, D = LQ.chebyshev_matrices((a, b), N)
        V = float(C.A.build_Levin_data(grid, D, need_AmatT=False, vectorize_cache=C.cache)[4])
        kn = R4.interior(edges, a, b)
        brk = [a] + (kn or []) + [b]
        sg = R4.v3_sigma(G, a, b, kn, V)["sigma_hat"]
        lv = {n: gl(G, brk, n) for n in (512, 1024)}
        ch = abs(lv[1024][0] - lv[512][0])
        noise = 2 * sg * math.sqrt(lv[1024][1])
        fin = 1024
        if ch > noise:
            lv[2048] = gl(G, brk, 2048)
            ch = abs(lv[2048][0] - lv[1024][0])
            noise = 2 * sg * math.sqrt(lv[2048][1])
            fin = 2048
        m = {"start": sp[0], "end": sp[1], "V_over_pi": V / math.pi, "n_pieces": len(brk) - 1, "ref48": lv[fin][0].hex(),
             "change": ch, "noise": noise, "eps48": ch + noise, "sigma_hat": sg, "finest": fin,
             "levels": {str(n): v[0].hex() for n, v in lv.items()}}
        (mdir / f"{k:03d}.json").write_text(json.dumps(m, indent=1))
        print(k, sp, f"V/pi {V / math.pi:.3g}", fin, flush=True)
    if mode != "combine":
        return
    per = {}
    for cd in CODES:
        r = rep[cd]["replay"]
        val, ab = mpf(fx(r["value"])), fx(r["abserr"])
        rows = []
        for l in rep[cd]["leaves"]:
            if (key(l), l["type"], l["value"]) in common:
                continue
            m = meas[key(l)]
            te = fx(l["value"]) - m["ref48"]
            V = m["V_over_pi"]
            cls = ("Levin" if "levin" in str(l["type"]).lower() and "direct" not in str(l["type"]).lower()
                   else "CC, V in [10, 12) pi" if 10 <= V < 12 else "CC, V in [8, 10) pi" if 8 <= V < 10
                   else "CC, V < 8 pi" if V < 8 else "CC, V >= 12 pi")
            others = [o for o in spans[key(l)] if o != cd]
            same_span_other_value = sorted({o for o in CODES if o != cd and any(
                key(x) == key(l) and x["value"] != l["value"] for x in rep[o]["leaves"])})
            rows.append({"start": l["start"], "end": l["end"], "depth": l.get("depth"), "type": l["type"], "class": cls,
                         "V_over_pi": V, "value": fx(l["value"]), "total_err": fx(l["total_err"]) if l.get("total_err") else None,
                         "budget_err": fx(l["budget_err"]) if l.get("budget_err") else None, "true_err": te,
                         "ref_change": m["change"], "ref_noise": m["noise"], "eps48": m["eps48"], "trusted": l.get("trusted"),
                         "decl_over_true": (fx(l["total_err"]) / abs(te)) if l.get("total_err") and te else None,
                         "also_under": others, "same_span_other_value": same_span_other_value})
        per[cd] = {"value": float(val), "abserr": ab, "err_call": float(abs(val - ref_call)),
                   "ratio": float(abs(val - ref_call) / ab), "n_regions": r["num_regions"], "n_differing": len(rows),
                   "sum_true_err_differing": math.fsum(x["true_err"] for x in rows),
                   "sum_abs_true_err_differing": math.fsum(abs(x["true_err"]) for x in rows),
                   "sum_total_err_differing": math.fsum(x["total_err"] or 0 for x in rows),
                   "sum_budget_err_differing": math.fsum(x["budget_err"] or 0 for x in rows),
                   "sum_total_err_all": math.fsum(fx(l["total_err"]) for l in rep[cd]["leaves"] if l.get("total_err")),
                   "sum_budget_err_all": math.fsum(fx(l["budget_err"]) for l in rep[cd]["leaves"] if l.get("budget_err")),
                   "by_class": {k: {"n": sum(1 for x in rows if x["class"] == k),
                                    "sum_true_err": math.fsum(x["true_err"] for x in rows if x["class"] == k),
                                    "sum_total_err": math.fsum(x["total_err"] or 0 for x in rows if x["class"] == k)}
                                for k in sorted({x["class"] for x in rows})},
                   "regions": rows}
    hdr = header(f"levin-phase-floor 07d Part A' addendum: item B, {SID} region by region", __file__, list(rp.values()))
    doc = {"header": hdr, "sid": SID, "ref_call": sc["ref"], "spread": sc["spread"], "knots_note": knote,
           "n_common": len(common), "per_code": per}
    (ADD / "out" / "dissect_b.json").write_text(json.dumps(doc, indent=1))
    L = hdr + ["", f"call {SID}; ref {sc['ref']}; common regions {len(common)}", ""]
    L += ["| code | value | abserr | |v - ref| | ratio | regions | differing | sum true err (diff) | sum total_err (diff) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cd, p in per.items():
        L.append(f"| {cd} | {p['value']:.6e} | {p['abserr']:.3g} | {p['err_call']:.3g} | {p['ratio']:.3g} | {p['n_regions']} | "
                 f"{p['n_differing']} | {p['sum_true_err_differing']:.3g} | {p['sum_total_err_differing']:.3g} |")
    for cd, p in per.items():
        L += ["", f"### {cd}: differing regions", "", "| start | end | depth | type | class | V/pi | total_err | true err | decl/true | same span, other value under |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in p["regions"]:
            L.append(f"| {x['start']} | {x['end']} | {x['depth']} | {x['type']} | {x['class']} | {x['V_over_pi']:.3g} | "
                     f"{(x['total_err'] or 0):.3g} | {x['true_err']:.3g} | {(x['decl_over_true'] or float('nan')):.3g} | "
                     f"{x['same_span_other_value']} |")
    (ADD / "out" / "dissect_b.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[len(hdr):len(hdr) + 9]))


if __name__ == "__main__":
    main()
