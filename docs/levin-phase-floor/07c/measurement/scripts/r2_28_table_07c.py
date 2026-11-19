"""levin-phase-floor prompt 07c Part A extension (R2-28), question 1: every leaf of R2-28's partitions set against its own
reference. New in the extension, after 07b's extension's x1_table.py (a record), whose reference rule it keeps.

**The reference of a span** [lo, hi] (both points of the reading out/r2-28/leafref.json): the sum, at resolution 2, of
the reading's pieces between them; its stated error sqrt(sum of resolution 2's fit-noise variances) + |ref_2 - ref_1|
(07b's x1_table.py). **The call's reference** is 07b Part A's (ref-R2-28-fccsplit, its resolution 2), and the sum of every
piece of this reading is set beside it.

For each partition (a:R2-28 and a:R2-28~o under the tree and the seven variants, out/replays/<v>/; 07a's one region,
07b Part A's out/replays/tree/R2-28.json): per leaf, its value v, its reference, the reference's stated error, the error
v - ref, its declared error e' (``total_err``; the leaves' e' sum to the call's abserr to 1.4e-42), and
(v - ref)/e'. The leaf's kind: trusted Levin, trusted direct (Clenshaw-Curtis by the C2 gate, trust-checked),
floor-limited Levin or direct (``phase_limited``, value its own), untrusted direct. Its ancestry from the instrumented
replay (out/replays/instr/a:R2-28.instrument.json, the tree's partition only, and S_i's for S where present): every refused
ancestor, its depth and rule (``rate_rule``), and the rate rule's q where the leaf was checked.
**Sums:** per partition, the sum of the errors by kind, against the call's error.

Writes out/r2-28/leaves.json and out/r2-28/leaves.md (refuses to overwrite). Usage: python -B r2_28_table_07c.py
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
ROOT = Path(__file__).resolve().parent.parent
R07B = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement")
VARIANTS = ["tree", "S", "S2", "N", "Eg", "Ea", "SEg", "SEa"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def kind(L):
    if L["type"] == "Levin":
        return "levin_trusted" if L["trusted"] else ("levin_floor_limited" if L["phase_limited"] else "levin_untrusted")
    return "direct_trusted" if L["trusted"] else ("direct_floor_limited" if L["phase_limited"] else "direct_untrusted")


def ancestry(instr_path):
    """Per final leaf (start, end): its refused ancestors [(depth, rule, q)] and its own check's rate rule."""
    if not instr_path.exists():
        return None
    R = json.loads(instr_path.read_text())["records"]
    checks = {}
    for r in R:
        if r["kind"] == "check":
            checks[(r["start"], r["end"])] = r
    refused = [(fx(r["start"]), fx(r["end"]), r["depth"], (r.get("rate_rule") or {}).get("rule"), (r.get("rate_rule") or {}).get("q"))
               for r in checks.values() if (r.get("rate_rule") or {}).get("refused")]
    out = {}
    for r in R:
        if r["kind"] != "final":
            continue
        a, b = fx(r["start"]), fx(r["end"])
        anc = sorted([(d, rule, q, (lo.hex(), hi.hex())) for lo, hi, d, rule, q in refused
                      if lo <= a and b <= hi and (lo, hi) != (a, b)])
        own = checks.get((r["start"], r["end"]))
        out[(r["start"], r["end"])] = {"refused_ancestors": anc,
                                       "own_rate_rule": (own or {}).get("rate_rule"),
                                       "own_clause_failed": (own or {}).get("clause_failed"),
                                       "own_checked": own is not None}
    return out


def main():
    oj, om = ROOT / "out" / "r2-28" / "leaves.json", ROOT / "out" / "r2-28" / "leaves.md"
    for p in (oj, om):
        if p.exists():
            raise SystemExit(f"{p} exists")
    lr_p = ROOT / "out" / "r2-28" / "leafref.json"
    lr = json.loads(lr_p.read_text())
    pcs = [(fx(p["lo"]), fx(p["hi"])) for p in lr["pieces"]]
    r1, r2 = lr["readings"]["1"], lr["readings"]["2"]
    idx = {a: k for k, (a, b) in enumerate(pcs)}
    ends = {b: k for k, (a, b) in enumerate(pcs)}

    def ref(lo, hi):
        k0, k1 = idx[lo], ends[hi]
        ks = range(k0, k1 + 1)
        v2 = mp.fsum(mpf(r2["pieces"][k]["value_30"]) for k in ks)
        v1 = mp.fsum(mpf(r1["pieces"][k]["value_30"]) for k in ks)
        acc = float(sum(r2["pieces"][k]["fit_noise_var"] for k in ks)) ** 0.5 + float(abs(v2 - v1))
        return v2, acc, v1

    call_ref_doc = json.loads((R07B / "out" / "refs" / "readings" / "ref-R2-28-fccsplit.json").read_text())
    call_ref = mpf(call_ref_doc["readings"]["2"]["pieces"][0]["value_30"])
    whole, whole_acc, whole1 = ref(pcs[0][0], pcs[-1][1])
    doc = {"inputs": {"leafref": str(lr_p), "leafref_sha256": sha256(lr_p)},
           "call_ref_07b": mp.nstr(call_ref, 25), "this_reading_whole": mp.nstr(whole, 25), "this_reading_whole_r1": mp.nstr(whole1, 25),
           "this_reading_whole_acc": whole_acc, "whole_minus_07b": float(whole - call_ref), "partitions": {}}
    md = ["# R2-28: every leaf against its own reference (extension, question 1)", "",
          f"- Call reference (07b Part A, fccsplit r2): {mp.nstr(call_ref, 20)}; this reading's whole span: r2 {mp.nstr(whole, 20)}, "
          f"r1 {mp.nstr(whole1, 20)}, stated error {whole_acc:.3g}; r2 minus 07b's {float(whole - call_ref):.3g}", ""]
    srcs = {}
    for v in VARIANTS:
        for sid in ("a:R2-28", "a:R2-28~o"):
            srcs[f"{v}|{sid}"] = (ROOT / "out" / "replays" / v / f"{sid}.json",
                                  ROOT / "out" / "replays" / ("instr" if v == "tree" else f"{v}_i") / f"{sid}.instrument.json")
    srcs["07a|a:R2-28"] = (R07B / "out" / "replays" / "tree" / "R2-28.json", Path("/nonexistent"))
    for name, (p, ip) in srcs.items():
        d = json.loads(p.read_text())
        doc["inputs"][str(p)] = sha256(p)
        anc = ancestry(ip)
        if anc is not None:
            doc["inputs"][str(ip)] = sha256(ip)
        v_call = mpf(fx(d["replay"]["value"]))
        rows, by_kind = [], defaultdict(lambda: [mpf(0), 0.0, 0, mpf(0)])
        for i, L in enumerate(d["leaves"]):
            lo, hi = fx(L["start"]), fx(L["end"])
            rv, acc, rv1 = ref(lo, hi)
            v = mpf(fx(L["value"]))
            e = fx(L["total_err"])
            err = v - rv
            k = kind(L)
            a = None if anc is None else anc.get((L["start"], L["end"]))
            rows.append({"i": i, "lo": L["start"], "hi": L["end"], "depth": L["depth"], "kind": k, "value": L["value"],
                         "value_kind": L["value_kind"], "ref": mp.nstr(rv, 25), "ref_acc": acc, "err": float(err), "eprime": e,
                         "err_over_eprime": float(err / e) if e else None, "resolved": abs(float(err)) > acc,
                         "ancestry": a})
            bk = by_kind[k]
            bk[0] += err
            bk[1] += acc
            bk[2] += 1
            bk[3] += mpf(e)
        call_err = v_call - call_ref
        sum_err = mp.fsum(r["err"] for r in rows)
        doc["partitions"][name] = {"call_value": d["replay"]["value"], "call_abserr": d["replay"]["abserr"],
                                   "regions": d["replay"]["num_regions"], "call_err_vs_07b_ref": float(call_err),
                                   "sum_leaf_err": float(sum_err), "by_kind": {k: {"err": float(x[0]), "ref_acc": x[1], "n": x[2], "eprime": float(x[3])}
                                                                                for k, x in sorted(by_kind.items())},
                                   "leaves": rows}
        md += [f"## {name}: {d['replay']['num_regions']} regions, value {fx(d['replay']['value']):.17g}, abserr {fx(d['replay']['abserr']):.4g}",
               f"- call error against 07b's reference {float(call_err):.4g}; sum of the leaves' errors {float(sum_err):.4g}", "",
               "| kind | n | sum of errors | sum of stated ref errors | sum of e' |", "|---|---|---|---|---|"]
        for k, x in sorted(by_kind.items()):
            md.append(f"| {k} | {x[2]} | {float(x[0]):.4g} | {x[1]:.3g} | {float(x[3]):.3g} |")
        top = sorted(rows, key=lambda r: -abs(r["err"]))[:12]
        md += ["", "Largest |error| first:", "",
               "| i | depth | kind | span | error | ref err | e' | error/e' | refused ancestors (depth, rule) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in top:
            an = "-" if r["ancestry"] is None else ", ".join(f"{dd} {rule}" for dd, rule, q, sp in r["ancestry"]["refused_ancestors"])
            md.append(f"| {r['i']} | {r['depth']} | {r['kind']} | [{fx(r['lo']):.9g}, {fx(r['hi']):.9g}] | {r['err']:.3g} | {r['ref_acc']:.2g} | "
                      f"{r['eprime']:.3g} | {r['err_over_eprime'] if r['err_over_eprime'] is None else round(r['err_over_eprime'], 3)} | {an} |")
        md.append("")
    oj.write_text(json.dumps(doc, indent=1))
    om.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
