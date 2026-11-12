"""levin-phase-floor prompt 07a Part A, section 2.4 item 5: the attribution (H1-H3) of section 2.2, a table per call.
New in 07a. Read only.

For each call with leaf references (attr_07a.py's OUT_JSON, resolutions 1 and 2):
- **the leaf references:** ref_i, resolution 2's value; the stated accuracy a_i = max(|ref_i,2 - ref_i,1|,
  sqrt(fit-noise variance at 2)); a leaf meets section 2.2's accuracy rule iff a_i <= e'_i / 100 (e'_i its declared
  error, ``budget_err``), and every leaf that does not is listed;
- **the closure check:** |sum_i ref_i - ref| <= tol, with ref the call's own reference (07's FCC resolution-2
  reading, ``value_30``; z12 06g's ``ref_30``) and tol the reading's tolerance (07's rule, min(abserr_06m,
  abserr_g) / 100, v4_scores.json's ``tol``). A call that fails it is reported, not attributed;
- **the parts:** (H1) the stopped window's leaves falling at the stop (``fell_at_last_bisection`` True), (H2) its
  other leaves (H2's mechanism, a declaration widened from a trusted parent, is checked separately from the
  instrument), (H3) every leaf outside every window. For each part: its true error sum_i (v_i - ref_i), its declared
  error sum_i e'_i + |sum_i E_i| (E the signed node errors of its Clenshaw-Curtis leaves; the call's abserr is
  sum e' + |sum E|), their ratio, and its references' summed accuracy. Exact: doubles as Fractions of their hex and
  references of their 30-digit strings.

Writes OUT_JSON and OUT_MD. Usage: python -B attr_tables_07a.py OUT_JSON OUT_MD ATTR_JSON [ATTR_JSON ...]
"""

import datetime
import hashlib
import json
import socket
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
V4OUT = TREE / "docs" / "levin-phase-floor" / "07" / "census" / "out" / "v4"
SCORES06G = TREE / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "scores_stage3" / "scores.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def FH(h):
    return Fraction(float.fromhex(h))


def refs():
    sc = json.loads((V4OUT / "v4_scores.json").read_text())
    out = {}
    for c in sc["calls"]:
        rd = json.loads((V4OUT / "readings" / f"{c['sid']}-fcc-r2.json").read_text())
        out[c["sid"]] = (rd["value_30"], c["tol"])
    z = [e for e in sc["z"] if e["sid"] == "z12"][0]
    r06 = [r for r in json.loads(SCORES06G.read_text())["rows"] if r["sid"] == "z12"][0]
    out["cal_z12"] = (r06["ref_30"], z["tol"])
    return out


def main():
    out_json, out_md = Path(sys.argv[1]), Path(sys.argv[2])
    R = refs()
    res = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "host": socket.gethostname(),
                      "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "inputs": {p: sha256(p) for p in sys.argv[3:]}}, "calls": {}}
    L = ["# levin-phase-floor 07a Part A: the attribution (section 2.2; section 2.4 item 5)", "",
         f"written {res['header']['written']} on {res['header']['host']}; script attr_tables_07a.py sha256 {res['header']['script_sha256']}", ""]
    for p in sys.argv[3:]:
        d = json.loads(Path(p).read_text())
        sid = d["sid"]
        rep = json.loads(Path(d["header"]["replay_json"]).read_text())
        ref_s, tol = R[sid]
        ref = Fraction(ref_s)
        v = FH(rep["replay"]["value"])
        e = FH(rep["replay"]["abserr"])
        r1 = {x["i"]: x for x in d["readings"]["1"]["leaves"]} if "1" in d["readings"] else {}
        r2 = {x["i"]: x for x in d["readings"]["2"]["leaves"]}
        stopped = {w_i for w_i, w in enumerate(rep["windows"]) if w.get("stopped")}
        parts = {"H1 falling": [], "H2 other window": [], "H3 outside": []}
        leaf_rows = []
        sum_ref = Fraction(0)
        for lf in d["leaves"]:
            i = lf["i"]
            ref_i = Fraction(r2[i]["value_30"])
            sum_ref += ref_i
            acc = max(abs(ref_i - Fraction(r1[i]["value_30"])) if i in r1 else Fraction(0),
                      Fraction(float(r2[i]["fit_noise_var"]) ** 0.5))
            ep = FH(lf["budget_err"])
            E = FH(lf["node_error_signed"]) if lf["node_error_signed"] is not None else Fraction(0)
            true = FH(lf["value"]) - ref_i
            if lf["window"] and lf["window_index"] in stopped and lf["fell"] is True:
                part = "H1 falling"
            elif lf["window"]:
                part = "H2 other window"
            else:
                part = "H3 outside"
            parts[part].append((true, ep, E, acc))
            leaf_rows.append({"i": i, "start": lf["start"], "end": lf["end"], "depth": lf["depth"], "type": lf["type"],
                              "part": part, "fell": lf["fell"], "value": lf["value"], "ref_30": r2[i]["value_30"],
                              "true_error": float(true), "eprime": float(ep), "E": float(E), "accuracy": float(acc),
                              "true_over_eprime": float(abs(true) / ep) if ep > 0 else None,
                              "meets_100x": bool(acc * 100 <= ep), "direct_segments": r2[i]["direct_segments"],
                              "n_segments": r2[i]["n_segments"]})
        closure = abs(sum_ref - ref)
        closure_ok = closure <= Fraction(tol)
        ptab = {}
        for name, items in parts.items():
            true = sum((t for t, _, _, _ in items), Fraction(0))
            decl = sum((ep for _, ep, _, _ in items), Fraction(0)) + abs(sum((E for _, _, E, _ in items), Fraction(0)))
            acc = sum((a for _, _, _, a in items), Fraction(0))
            ptab[name] = {"n_leaves": len(items), "true_error": float(true), "declared": float(decl),
                          "abs_true_over_declared": float(abs(true) / decl) if decl > 0 else None,
                          "true_over_call_abserr": float(true / e), "reference_accuracy": float(acc)}
        nmiss = [r_["i"] for r_ in leaf_rows if not r_["meets_100x"]]
        rec = {"sid": sid, "value": rep["replay"]["value"], "abserr": rep["replay"]["abserr"], "ref_30": ref_s, "tol": tol,
               "call_error": float(v - ref), "call_error_over_abserr": float(abs(v - ref) / e),
               "sum_leaf_refs_minus_ref": float(sum_ref - ref), "closure_ok": bool(closure_ok),
               "closure_over_tol": float(closure / Fraction(tol)), "parts": ptab if closure_ok else None,
               "parts_if_not_closed": None if closure_ok else ptab,
               "leaves_not_meeting_100x": nmiss, "leaves": leaf_rows,
               "attr_json": p, "wall_s": {k: x["wall_s"] for k, x in d["readings"].items()}}
        res["calls"][sid] = rec
        L += [f"## {sid}", "",
              f"value {float(v):.16e}, abserr {float(e):.3e}, ref {ref_s}; (v - ref)/abserr {float((v - ref) / e):+.3f}; "
              f"closure |sum ref_i - ref| / tol = {rec['closure_over_tol']:.3g} ({'holds' if closure_ok else 'FAILS: reported, not attributed'}); "
              f"leaves not meeting the 100x accuracy rule: {len(nmiss)} of {len(leaf_rows)}", "",
              "| part | leaves | true error | declared | abs(true)/declared | true / call abserr | reference accuracy |",
              "|---|---|---|---|---|---|---|"]
        for name, t in ptab.items():
            L.append(f"| {name} | {t['n_leaves']} | {t['true_error']:+.3e} | {t['declared']:.3e} | "
                     f"{'-' if t['abs_true_over_declared'] is None else format(t['abs_true_over_declared'], '.3g')} | "
                     f"{t['true_over_call_abserr']:+.3f} | {t['reference_accuracy']:.2e} |")
        L.append("")
    out_json.write_text(json.dumps(res, indent=1))
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out_json, sha256(out_json), out_md, sha256(out_md))


if __name__ == "__main__":
    sys.exit(main())
