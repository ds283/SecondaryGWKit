"""levin-phase-floor prompt 07a Part A: the O9 check's table for section 2.5, from o9_check_07a.py's outputs
(out/o9/<sid>.json) for the five failures. New in 07a. Read only.

One summary row per failure (the H3 leaf carrying most of the H3 true error, its parent and sibling, and counts over
every checked region), then each call's full table (o9_check_07a.py's OUT_MD). Usage: python -B o9_table_07a.py
OUT_MD SID [SID ...]
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    sids = sys.argv[2:]
    L = ["# levin-phase-floor 07a Part A: the O9 check on the H3 leaves of the five failures", "",
         f"script o9_table_07a.py sha256 {sha256(__file__)}. A measurement at 07's code (no variant), added at the user's "
         "word on 2026-11-11 (log deviations 21-22). 'route' is the gate's: Clenshaw-Curtis below SIX_PI; Levin, LU tried "
         "first, above _LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20 pi; Levin, least squares, between.", "",
         "| call | N | H3 leaves / carrying excess (share of H3 true error) | regions checked | route changes | theta' sign changes | true / estimate (min-max) | main leaf: depth, span, true/declared, share | its phase_span (07) / true variation | route (07 = true) | parent phase_span / route | sibling phase_span / route | input sha256 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sids:
        p = ROOT / "out" / "o9" / f"{s}.json"
        d = json.loads(p.read_text())
        regs = [r[w] for r in d["rows"] for w in ("leaf", "parent", "sibling") if r.get(w)]
        ch = sum(1 for m in regs if m["route_changes"])
        sc = sum(1 for m in regs if m["theta_prime_sign_change"])
        tr = [m["true_over_estimate"] for m in regs]
        r = max(d["rows"], key=lambda r: abs(r["true_error"]))
        m, pa, si = r["leaf"], r.get("parent"), r.get("sibling")
        fmt = lambda x: "none (the leaf is a seed region)" if x is None else f"{x['phase_span_07']:.4g} / {x['route_07']}"
        L.append(f"| {s} | {d['chebyshev_order']} | {d['h3_leaves']} / {d['carriers']} ({d['carriers_share_of_H3_true_error']:.3f}) | "
                 f"{len(regs)} | {ch} | {sc} | {min(tr):.3f}-{max(tr):.3f} | d{r['depth']}, [{m['span_float'][0]:.6g}, {m['span_float'][1]:.6g}], "
                 f"{r['true_over_declared']:.3g}, {r['share_of_H3_true_error']:.3f} | {m['phase_span_07']:.4g} / {m['true_variation']:.4g} | "
                 f"{m['route_07']}{'' if not m['route_changes'] else ' -> ' + m['route_true']} | {fmt(pa)} | {fmt(si)} | {sha256(p)[:12]} |")
    L.append("")
    for s in sids:
        L += [(ROOT / "out" / "o9" / f"{s}.md").read_text(), ""]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("wrote", out, sha256(out))


if __name__ == "__main__":
    main()
