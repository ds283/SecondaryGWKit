"""levin-phase-floor prompt 07b extension, section 2.7.1 (X1): (e)'s remainder, leaf by leaf. New in the extension.

For each of (e)'s 11 calls converged and dishonest (state/ext/x1/plan.json), at its replay under (e)
(out/replays/e/<sid>.json; where (e) runs the tree's code it is the tree's replay, bit for bit):
- **every non-window leaf's true error,** its value against the sum of its reading's pieces (resolution 2;
  leafref_07b.py's output: the extension's out/ext/leafrefs/x1_<sid>.json, or Part A's out/leafrefs/ where Part A
  attributed the call), over the leaf's declared e' and over the call's abserr;
- **the window leaves together,** their summed value against the call's reference less the non-window leaves'
  references (subtraction: the window leaves hold the stationary point and are not read);
- **the closure:** where the call has no window, the non-window references' sum against the call's reference (section
  2.4's, or S_10 for W2), over abserr; where it has, closure holds by construction and is not a test;
- **the carrier:** the leaf, or the window set, carrying the largest share of the call's error, with its sign; whether
  it touches a window; and its section 2.2 records (from the mechanism rows: the extension's out/ext/x1/mech_x1.json,
  else Part A's out/mech/mech_all.json). Where Part A's reading covers only the carrier (v09, carrier_v09.json), the
  call is marked partial: its other leaves are not referenced, and the carrier is the one Part A found.
Writes OUT_JSON and OUT_MD. Usage: PYTHONPATH=<tree>:<scripts> python -B x1_table.py OUT_JSON OUT_MD
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 50
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score_07b as S  # noqa: E402

ROOT = HERE.parent.parent


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return None if h is None else float.fromhex(h)


def main():
    out_j, out_m = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    plan_p = ROOT / "state" / "ext" / "x1" / "plan.json"
    plan = json.loads(plan_p.read_text())["calls"]
    R = S.refs()
    mech_rows = json.loads((ROOT / "out" / "ext" / "x1" / "mech_x1.json").read_text())["rows"] + \
        json.loads((ROOT / "out" / "mech" / "mech_all.json").read_text())["rows"]
    inputs = {str(plan_p): sha256(plan_p)}
    out, md = [], []
    for sid, e in plan.items():
        rep_p = ROOT / "out" / "replays" / "e" / f"{sid}.json"
        rep = json.loads(rep_p.read_text())
        inputs[str(rep_p)] = sha256(rep_p)
        lr_p = Path(e["reference"]) if "reference" in e else ROOT / "out" / "ext" / "leafrefs" / f"x1_{sid}.json"
        lr = json.loads(lr_p.read_text())
        inputs[str(lr_p)] = sha256(lr_p)
        pieces = [(fx(p["lo"]), fx(p["hi"])) for p in lr["pieces"]]
        r1, r2 = lr["readings"].get("1"), lr["readings"]["2"]

        def ref(lo, hi):
            ks = [k for k, (a, b) in enumerate(pieces) if a >= lo and b <= hi]
            if not ks or pieces[ks[0]][0] != lo or pieces[ks[-1]][1] != hi or any(
                    pieces[k][1] != pieces[k2][0] for k, k2 in zip(ks[:-1], ks[1:])):
                return None, None
            v2 = mp.fsum(mpf(r2["pieces"][k]["value_30"]) for k in ks)
            acc = float(sum(r2["pieces"][k]["fit_noise_var"] for k in ks)) ** 0.5
            if r1 is not None:
                acc += float(abs(v2 - mp.fsum(mpf(r1["pieces"][k]["value_30"]) for k in ks)))
            return v2, acc

        rk = S.ref_key(sid, None, None)
        cref = R[rk]["ref"]
        v, abserr = fx(rep["replay"]["value"]), fx(rep["replay"]["abserr"])
        call_err = mpf(v) - cref
        leaves, nw_ref, nw_acc, w_val, missing = [], mpf(0), 0.0, mpf(0), []
        for i, L in enumerate(rep["leaves"]):
            if L["window"]:
                w_val += mpf(fx(L["value"]))
                continue
            lo, hi = fx(L["start"]), fx(L["end"])
            rr, aa = ref(min(lo, hi), max(lo, hi))
            if rr is None:
                missing.append(i)
                continue
            if lo > hi:
                rr = -rr
            err = float(mpf(fx(L["value"])) - rr)
            ep = fx(L["budget_err"])
            nw_ref += rr
            nw_acc += aa
            leaves.append({"i": i, "lo": lo, "hi": hi, "depth": L["depth"], "type": L["type"], "trusted": L["trusted"],
                           "value_kind": L["value_kind"], "touches_window": bool(L["touches_window"]), "e_prime": ep,
                           "err": err, "err_over_eprime": err / ep if ep else None, "err_over_abserr": err / abserr,
                           "ref_acc": aa, "ref_acc_over_eprime": aa / ep if ep else None})
        partial = bool(missing)  # Part A's reading of v09 covers its carrier leaf only (carrier_v09.json)
        n_win = sum(1 for L in rep["leaves"] if L["window"])
        win = None
        if partial:
            closure = None
        elif n_win:
            w_ref = cref - nw_ref
            win = {"n": n_win, "err": float(w_val - w_ref), "err_over_abserr": float((w_val - w_ref) / abserr),
                   "e_prime_sum": sum(fx(L["budget_err"]) for L in rep["leaves"] if L["window"])}
            closure = None
        else:
            closure = float((nw_ref - cref) / abserr)
        cands = [(x["err_over_abserr"], f"leaf {x['i']}", x) for x in leaves] + ([(win["err_over_abserr"], "window leaves", win)] if win else [])
        sign = 1 if call_err > 0 else -1
        carrier = max(cands, key=lambda c: sign * c[0])
        mrow = None
        if carrier[1].startswith("leaf"):
            ci = carrier[2]["i"]
            want = ("e", "tree") if e["e_equals_tree"] else ("e",)
            ms = [r for r in mech_rows if r["sid"] == sid and r["i"] == ci and r["variant"] in want]
            mrow = ms[0] if ms else None
        row = {"sid": sid, "partial": partial, "leaves_not_referenced": missing, "e_equals_tree": e["e_equals_tree"], "value": v, "abserr": abserr, "ref": mp.nstr(cref, 25),
               "call_err_over_abserr": float(call_err / abserr), "ref_spread_over_abserr": R[rk]["spread"] / abserr,
               "n_leaves": len(rep["leaves"]), "n_window_leaves": n_win, "closure_over_abserr": closure,
               "nonwindow_ref_acc_over_abserr": nw_acc / abserr, "window_set": win,
               "carrier": {"name": carrier[1], "err_over_abserr": carrier[0],
                           **({k: carrier[2][k] for k in ("i", "depth", "type", "trusted", "touches_window", "err_over_eprime",
                                                          "ref_acc_over_eprime", "lo", "hi")} if carrier[1].startswith("leaf") else {})},
               "carrier_mech": None if mrow is None else {
                   "variant": mrow["variant"], "clause": mrow["clause"], "r": mrow["r"], "delta1": mrow["delta1"],
                   "delta2": mrow["delta2"], "e_prime": mrow["e_prime"], "err_over_eprime": mrow["err_over_eprime"],
                   "M1_spread_over_errQ": mrow.get("M1_spread_over_errQ"), "M1_same_sign": mrow.get("M1_same_sign"),
                   "M2_fall": mrow.get("M2_fall"), "rate_end_to_end": max(abs(mrow["theta_prime_lo"]), abs(mrow["theta_prime_hi"])) /
                   min(abs(mrow["theta_prime_lo"]), abs(mrow["theta_prime_hi"])), "ref_acc_over_eprime": mrow["ref_acc_over_eprime"],
                   "edge": mrow["edge"]},
               "top": [{"name": c[1], "err_over_abserr": c[0]} for c in sorted(cands, key=lambda c: -sign * c[0])[:4]],
               "leaves": leaves}
        out.append(row)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "inputs": inputs}, "calls": out}
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(doc, indent=1, default=str))
    md = ["## (X1) (e)'s remaining converged-and-dishonest calls, leaf by leaf (at (e); errors over the call's abserr)", "",
          "| call | (e) = tree | ratio | leaves (window) | closure / ref acc | carrier | its err/abserr | touches a window | carrier err(Q)/e' | clause, r | M1: err(S1), err(Q), err(S3) over e' | M2 falls S1/V_P, Q/S1, S3/Q | rate range | ref acc/e' | next largest |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda v: "-" if v is None else f"{v:.3g}"
    for r in out:
        c, m = r["carrier"], r["carrier_mech"]
        clos = (f"{g(r['closure_over_abserr'])} / {g(r['nonwindow_ref_acc_over_abserr'])}" if r["closure_over_abserr"] is not None
                else f"partial: carrier only (Part A) / {g(r['nonwindow_ref_acc_over_abserr'])}" if r["partial"]
                else f"by subtraction / {g(r['nonwindow_ref_acc_over_abserr'])}")
        eo = m["err_over_eprime"] if m else {}
        fall = m["M2_fall"] if m else {}
        md.append(f"| {r['sid']} | {r['e_equals_tree']} | {r['call_err_over_abserr']:+.3g} | {r['n_leaves']} ({r['n_window_leaves']}) | {clos} | "
                  f"{c['name']} | {c['err_over_abserr']:+.3g} | {c.get('touches_window', '-')} | {g(c.get('err_over_eprime'))} | "
                  f"{(m['clause'] + ', ' + g(m['r'])) if m else '-'} | {g(eo.get('S1'))}, {g(eo.get('Q'))}, {g(eo.get('S3'))} | "
                  f"{g(fall.get('S1/V_P'))}, {g(fall.get('Q/S1'))}, {g(fall.get('S3/Q'))} | {g(m['rate_end_to_end']) if m else '-'} | "
                  f"{g(c.get('ref_acc_over_eprime'))} | " + "; ".join(f"{t['name']} {t['err_over_abserr']:+.2g}" for t in r["top"][1:3]) + " |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
