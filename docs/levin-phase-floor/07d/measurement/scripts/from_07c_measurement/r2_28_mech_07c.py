"""levin-phase-floor prompt 07c Part A extension (R2-28), questions 2 and 3: the probes of out/r2-28/probe.json set against
the leaf references of out/r2-28/leafref.json. New in the extension.

For each probed span: the probe's driver solve must reproduce the replay's leaf (V_P = ``value_own``, Q = ``value`` where
``value_kind`` is quarters; by hex). Then for every piece (P, L, R, the quarters, the eighths) and every rule (the driver's
solve, the forced Clenshaw-Curtis pair, Levin at N + 8 and N - 8): the piece's value less its reference, the reference's
stated error (r2_28_table_07c.py's rule), and the piece's own floor and truncation where the rule reports them; and for
the sums V_P, S1 = L + R, Q and S3 (the eighths' sum) the same. The **signature**: whether the error is common to P, its
halves and its quarters (07b's M2, "the error common to the leaf, halves and quarters") and where it sits among the
pieces.

Writes out/r2-28/mech.json and out/r2-28/mech.md (refuses to overwrite). Usage: python -B r2_28_mech_07c.py
"""

import hashlib
import json
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
ROOT = Path(__file__).resolve().parent.parent
R07B = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement")
ORDER = ["P", "L", "R", "LL", "LR", "RL", "RR", "LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def main():
    oj, om = ROOT / "out" / "r2-28" / "mech.json", ROOT / "out" / "r2-28" / "mech.md"
    for p in (oj, om):
        if p.exists():
            raise SystemExit(f"{p} exists")
    lr_p, pr_p = ROOT / "out" / "r2-28" / "leafref.json", ROOT / "out" / "r2-28" / "probe.json"
    lr, pr = json.loads(lr_p.read_text()), json.loads(pr_p.read_text())
    pcs = [(fx(p["lo"]), fx(p["hi"])) for p in lr["pieces"]]
    r1, r2 = lr["readings"]["1"], lr["readings"]["2"]
    idx = {a: k for k, (a, b) in enumerate(pcs)}
    ends = {b: k for k, (a, b) in enumerate(pcs)}

    def ref(lo, hi):
        ks = range(idx[lo], ends[hi] + 1)
        v2 = mp.fsum(mpf(r2["pieces"][k]["value_30"]) for k in ks)
        v1 = mp.fsum(mpf(r1["pieces"][k]["value_30"]) for k in ks)
        return v2, float(sum(r2["pieces"][k]["fit_noise_var"] for k in ks)) ** 0.5 + float(abs(v2 - v1))

    tree = json.loads((ROOT / "out" / "replays" / "tree" / "a:R2-28.json").read_text())["leaves"]
    a07 = json.loads((R07B / "out" / "replays" / "tree" / "R2-28.json").read_text())["leaves"]
    leaf_of = {"tree:295": tree[295], "tree:294": tree[294], "07a:0": a07[0]}
    doc = {"inputs": {str(lr_p): sha256(lr_p), str(pr_p): sha256(pr_p)}, "spans": {}}
    md = ["# R2-28: the carrying leaf, its pieces and second rules, against the leaf references (extension, questions 2 and 3)", ""]
    for s in pr["spans"]:
        L = leaf_of[s["name"]]
        drv = s["rules"]["driver"]
        repro = {"V_P": drv["V_P"] == L["value_own"], "Q": drv["Q"] == L["value"] if L["value_kind"] == "quarters" else None}
        refs = {k: ref(fx(a), fx(b)) for k, (a, b) in s["spans"].items()}
        sums = {"V_P": ("P",), "S1": ("L", "R"), "Q": ("LL", "LR", "RL", "RR"),
                "S3": ("LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR")}
        ent = {"leaf": {k: L[k] for k in ("start", "end", "depth", "type", "value", "value_own", "value_kind", "total_err", "delta1", "delta2", "convergence_ratio", "trusted")},
               "reproduces": repro, "theta_prime_abs": s["theta_prime_abs"], "pieces": {}, "sums": {}}
        P_ref = refs["P"][0]
        for rn, R in s["rules"].items():
            ent["pieces"][rn] = {}
            for k in ORDER:
                pc = R["pieces"][k]
                rv, acc = refs[k]
                ent["pieces"][rn][k] = {"err": float(mpf(fx(pc["value"])) - rv), "ref_acc": acc, "is_direct": pc["is_direct"],
                                        "phase_span": fx(pc["phase_span"]), "floor": fx(pc["phase_err"]),
                                        "truncation": fx(pc["abserr_truncation"]), "order": pc["chebyshev_order"]}
            ent["sums"][rn] = {sn: float(mpf(fx(R[sn])) - P_ref) for sn in sums}
        doc["spans"][s["name"]] = ent
        tp = s["theta_prime_abs"]
        md += [f"## {s['name']}: [{fx(L['start']):.10g}, {fx(L['end']):.10g}], depth {L['depth']}, {L['type']}, trusted {L['trusted']}, "
               f"value_kind {L['value_kind']}",
               f"- reproduces the replay's leaf by hex: {repro}; |theta'| {fx(tp['min']):.4g} .. {fx(tp['max']):.4g} (ratio {fx(tp['ratio']):.4g})",
               f"- the leaf's e' {fx(L['total_err']):.4g}, delta1 {fx(L['delta1']):.3g}, delta2 {fx(L['delta2']):.3g}, r {fx(L['convergence_ratio']):.3g}; "
               f"its reference {mp.nstr(P_ref, 20)} (stated error {refs['P'][1]:.2g})", "",
               "**Sums against P's reference** (V_P, S1 = L + R, Q, S3 = eighths' sum):", "",
               "| rule | V_P | S1 | Q | S3 |", "|---|---|---|---|---|"]
        for rn in s["rules"]:
            x = ent["sums"][rn]
            md.append(f"| {rn} | {x['V_P']:.4g} | {x['S1']:.4g} | {x['Q']:.4g} | {x['S3']:.4g} |")
        md += ["", "**Each piece's error against its own reference** (driver: route D = Clenshaw-Curtis, L = Levin; floor; V):", "",
               "| piece | ref err | driver | route | floor | V | cc (forced) | Levin N+8 | Levin N-8 |", "|---|---|---|---|---|---|---|---|---|"]
        for k in ORDER:
            d = ent["pieces"]["driver"][k]
            md.append(f"| {k} | {d['ref_acc']:.2g} | {d['err']:.4g} | {'D' if d['is_direct'] else 'L'} | {d['floor']:.3g} | {d['phase_span']:.4g} | "
                      f"{ent['pieces']['cc'][k]['err']:.4g} | {ent['pieces']['levin_Np8'][k]['err']:.4g} | {ent['pieces']['levin_Nm8'][k]['err']:.4g} |")
        md.append("")
    oj.write_text(json.dumps(doc, indent=1))
    om.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
