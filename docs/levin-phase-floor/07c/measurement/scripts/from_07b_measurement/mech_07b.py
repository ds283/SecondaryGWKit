"""levin-phase-floor prompt 07b Part A, section 2.2: the mechanism tables, per leaf, from the leaf probes
(out/probe/<variant>_<sid>.json) and the piece references (out/leafrefs/<name>.json). New in 07b.

For each probed leaf whose span and sub-spans are covered by the reading's pieces:
- each sub-span's reference is the sum of the reading's pieces inside it (resolution 2), its stated accuracy
  |ref_2 - ref_1| + sqrt(sum of resolution 2's fit-noise variances);
- the true errors, signed: of V_P, of S1 = V_L + V_R, of Q and (eighths probed) of S3, against the leaf's reference; of
  each half, quarter and eighth's own solve against its own reference;
- (M1) the common error: err(S1), err(Q), err(S3) and their spread |max - min| / |err(Q)|; where along the span: the
  halves' and quarters' own errors as shares of err(Q);
- (M2) the observed fall |err(S1)|/|err(V_P)|, |err(Q)|/|err(S1)| (and |err(S3)|/|err(Q)|) beside r;
- (M3) each perturbation's change of V_P, S1 and Q beside err(Q);
- the edge diagnostic: the rate's ratio across the grid's first spacing at a window edge (chebyshev_matrices at the
  solve's order) and l = |theta'/theta''| there (theta'' by a one-sided difference of theta_deriv over 1e-3 of the first spacing).
Also, for each named variant leaf of a reading (b4, b8), its returned value against its span's reference.

Writes OUT_JSON and OUT_MD. Usage: PYTHONPATH=<tree>:<scripts> python -B mech_07b.py OUT_JSON OUT_MD PAIR [PAIR ...],
each PAIR ``PROBE_JSON=LEAFREF_JSON``.
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 50
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return None if h is None else float.fromhex(h)


def main():
    out_j, out_m = Path(sys.argv[1]), Path(sys.argv[2])
    pairs = [a.split("=") for a in sys.argv[3:]]
    import AdaptiveLevin.levin_quadrature as LQ
    import ray.cloudpickle as cp

    rows, md = [], []
    inputs = {}
    for probe_p, ref_p in pairs:
        pr = json.loads(Path(probe_p).read_text())
        rf = json.loads(Path(ref_p).read_text())
        inputs[probe_p] = sha256(probe_p)
        inputs[ref_p] = sha256(ref_p)
        pieces = [(fx(p["lo"]), fx(p["hi"])) for p in rf["pieces"]]
        r1 = rf["readings"].get("1")
        r2 = rf["readings"]["2"]

        def ref(lo, hi):
            ks = [k for k, (a, b) in enumerate(pieces) if a >= lo and b <= hi]
            if not ks or pieces[ks[0]][0] != lo or pieces[ks[-1]][1] != hi or any(
                    pieces[k][1] != pieces[k2][0] for k, k2 in zip(ks[:-1], ks[1:])):
                return None, None
            v2 = mp.fsum(mpf(r2["pieces"][k]["value_30"]) for k in ks)
            var = sum(r2["pieces"][k]["fit_noise_var"] for k in ks)
            acc = float(var) ** 0.5
            if r1 is not None:
                v1 = mp.fsum(mpf(r1["pieces"][k]["value_30"]) for k in ks)
                acc += float(abs(v2 - v1))
            return v2, acc

        c = pr["header"]["capture"]
        with open(c["path"], "rb") as f:
            f.seek(c["offset"])
            rec = cp.loads(f.read(c["length"]))
        thd = rec["theta"]["theta_deriv"]
        order = max(rec["kw"].get("chebyshev_order", 16), 8)
        for e in pr["leaves"]:
            L, p = e["leaf"], e["probe"]
            lo, hi = fx(L["start"]), fx(L["end"])
            refP, accP = ref(lo, hi)
            if refP is None:
                continue
            ep = fx(L["budget_err"])
            row = {"probe": probe_p, "sid": pr["sid"], "variant": pr["variant"], "i": e["i"], "lo": lo, "hi": hi,
                   "depth": L["depth"], "type": L["type"], "trusted": L["trusted"], "clause": p["clause"],
                   "r": fx(p["r"]), "delta1": fx(p["delta1"]), "delta2": fx(p["delta2"]), "nu_B": fx(p["nu_B"]),
                   "residual": fx(L["abserr"]), "floor": fx(L["phase_err"]), "truncation": fx(L["abserr_truncation"]),
                   "e_prime": ep, "V": fx(p["pieces"]["P"]["phase_span"]), "lu": p["pieces"]["P"]["lu"],
                   "theta_prime_lo": fx(e["ends"]["theta_prime_a"]), "theta_prime_hi": fx(e["ends"]["theta_prime_b"]),
                   "touches": L["touches_window"], "edge_lo": L["edge_lo_is"], "edge_hi": L["edge_hi_is"],
                   "ref": mp.nstr(refP, 25), "ref_acc": accP, "ref_acc_over_eprime": accP / ep}
            # true errors of the level sums
            errs = {}
            for k in ("V_P", "S1", "Q", "S3"):
                if k in p:
                    errs[k] = float(mpf(fx(p[k])) - refP)
            row["err"] = errs
            row["err_over_eprime"] = {k: v / ep for k, v in errs.items()}
            if "S1" in errs and "Q" in errs:
                vals = [errs[k] for k in ("S1", "Q", "S3") if k in errs]
                row["M1_spread_over_errQ"] = (max(vals) - min(vals)) / abs(errs["Q"]) if errs["Q"] else None
                row["M1_same_sign"] = all(v > 0 for v in vals) or all(v < 0 for v in vals)
                row["M2_fall"] = {"S1/V_P": abs(errs["S1"] / errs["V_P"]) if errs["V_P"] else None,
                                  "Q/S1": abs(errs["Q"] / errs["S1"]) if errs["S1"] else None,
                                  "S3/Q": abs(errs["S3"] / errs["Q"]) if "S3" in errs and errs["Q"] else None}
            pieces_err = {}
            for k, sp in p["spans"].items():
                if k == "P":
                    continue
                rr, aa = ref(fx(sp[0]), fx(sp[1]))
                if rr is None:
                    continue
                v = fx(p["pieces"][k]["value"])
                pieces_err[k] = {"err": float(mpf(v) - rr), "acc": aa, "share_of_errQ": float(mpf(v) - rr) / errs["Q"] if errs.get("Q") else None}
            row["pieces_err"] = pieces_err
            if "perturb" in e:
                row["M3"] = {k: {q: fx(v[q]) - fx(p[q]) for q in ("V_P", "S1", "Q")} | {"over_errQ": (fx(v["Q"]) - fx(p["Q"])) / errs["Q"]}
                             for k, v in e["perturb"]["results"].items()}
            # the edge diagnostic
            grid, _ = LQ.chebyshev_matrices((lo, hi), order)
            edge = {}
            for side, x0, x1 in (("lo", lo, grid[-2]), ("hi", hi, grid[1])):
                t0, t1 = abs(float(thd(x0))), abs(float(thd(x1)))
                hh = abs(x1 - x0) * 1e-3
                xs = x0 + (hh if side == "lo" else -hh)
                tdd = abs(float(thd(xs)) - float(thd(x0))) / hh
                edge[side] = {"rate_ratio_first_spacing": max(t0, t1) / min(t0, t1) if min(t0, t1) > 0 else None,
                              "first_spacing": abs(x1 - x0), "l": t0 / tdd if tdd > 0 else None,
                              "spacing_over_l": abs(x1 - x0) * tdd / t0 if t0 > 0 else None,
                              "eps_end": tdd / t0 ** 2 if t0 > 0 else None}
            row["edge"] = edge
            rows.append(row)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "inputs": inputs}, "rows": rows}
    out_j.write_text(json.dumps(doc, indent=1, default=str))
    md.append("| call | var | leaf | depth | span | V | clause | r | e' | err(V_P)/e' | err(S1)/e' | err(Q)/e' | err(S3)/e' | "
              "spread/err(Q) | ref acc/e' | end | rate ratio | spacing/l | eps at end |")
    md.append("|" + "---|" * 19)
    for r in rows:
        eo = r["err_over_eprime"]
        side = "lo" if r["edge_lo"] else ("hi" if r["edge_hi"] else None)
        if side is None:  # no window edge: the end where the grid resolves the rate worst
            side = max(("lo", "hi"), key=lambda k: r["edge"][k]["spacing_over_l"] or 0.0)
        ed = r["edge"].get(side)
        md.append(f"| {r['sid']} | {r['variant']} | {r['i']} | {r['depth']} | [{r['lo']:.6g}, {r['hi']:.6g}] | {r['V']:.3g} | "
                  f"{r['clause']} | {r['r']:.3g} | {r['e_prime']:.3e} | {eo.get('V_P', float('nan')):+.3g} | "
                  f"{eo.get('S1', float('nan')):+.3g} | {eo.get('Q', float('nan')):+.3g} | {eo.get('S3', float('nan')):+.3g} | "
                  f"{(r.get('M1_spread_over_errQ') or float('nan')):.3g} | {r['ref_acc_over_eprime']:.2g} | "
                  f"{side}{'(w)' if (r['edge_lo'] if side == 'lo' else r['edge_hi']) else ''} | "
                  f"{(ed['rate_ratio_first_spacing'] if ed else float('nan')):.3g} | {(ed['spacing_over_l'] if ed else float('nan')):.3g} | "
                  f"{(ed['eps_end'] if ed else float('nan')):.3g} |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
