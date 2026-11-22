"""levin-phase-floor prompt 07a Part A: the O9 check on the H3 leaves of a failure, added at the user's word
(2026-11-11, relayed by the orchestrator). New in 07a. A measurement only: 07's code as it is, no variant.

For one call (a capture, loaded as replay_07a.py loads it, with the 1e-8 override where there is one), its
attribution table entry (attr_tables_07a.py's OUT_JSON, ``calls[SID]["leaves"]``) and its replay at (a)
(out/replays/a/SID.json, for the window bounds and the leaves): every H3 leaf ("H3 outside") that **carries excess
error** -- its |true error| above its declared error e' + |E| -- and, labelled, the three H3 leaves of largest
|true error| whatever their ratio. For each such leaf, and for its parent region and its sibling:

- **the gate's estimate as 07 computes it**: ``phase_span`` from ``_Basis_SinCos.build_Levin_data`` on the region's
  own Chebyshev grid (``chebyshev_matrices(span, N)``, N the call's ``chebyshev_order``), i.e. mean |theta'| on the
  nodes times the width, theta' the caller's ``theta_deriv``;
- **the true variation**: |theta(b) - theta(a)| from the raw phase ``theta["theta"]`` at the two ends; whether theta'
  changes strict sign on the nodes (a zero counts as a change, as 07's guard counts it); and the node-partition sum
  sum_j |theta(x_{j+1}) - theta(x_j)| over the nodes in order (given always; it is the true variation used below
  only where theta' changes sign);
- **each against SIX_PI and _LEVIN_DIRECT_SOLVE_PHASE_SPAN**: the route 07's gate takes on its estimate
  (Clenshaw-Curtis below SIX_PI; Levin, LU tried first, above 20 pi; Levin, least squares, between) and the route
  it would take on the true variation, and whether they differ.

The parent and the sibling are found by descending from the leaf's seed region (the range between windows that
holds it, 07's initial partition) by ``break_point`` = (start + end) / 2.0, the driver's own split; the leaf must be
reached exactly at its own depth. Regions are oriented as x_span, as the driver orients them.

Usage: PYTHONPATH=<tree>:<this dir> python -B o9_check_07a.py SID ATTR_TABLES_JSON OUT_JSON OUT_MD
"""

import datetime
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sid, attr_json, out_json, out_md = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    import AdaptiveLevin.levin_quadrature as LQ
    import ray.cloudpickle as cp

    mod = Path(LQ.__file__).resolve()
    if mod != TREE / "AdaptiveLevin" / "levin_quadrature.py":
        raise SystemExit(f"module is {mod}, not the tree's")
    c = json.loads((ROOT / "state" / "calls_07a.json").read_text())["calls"][sid]
    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture sha256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c["override"]:
        kw["rtol"], kw["atol"] = c["override"]["rtol"], c["override"]["atol"]
    N = int(kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
    theta = rec["theta"]
    x_span = tuple(rec["x_span"])
    reversed_span = x_span[0] > x_span[1]
    replay_path = ROOT / "out" / "replays" / "a" / f"{sid}.json"
    rp = json.loads(replay_path.read_text())
    at = json.loads(attr_json.read_text())["calls"][sid]
    SIX_PI, LU = LQ.SIX_PI, LQ._LEVIN_DIRECT_SOLVE_PHASE_SPAN

    def route(v):
        return "Clenshaw-Curtis" if v < SIX_PI else ("Levin (LU tried first)" if v > LU else "Levin (least squares)")

    def oriented(lo, hi):
        return (hi, lo) if reversed_span else (lo, hi)

    def measure(a, b):
        grid, Dmat = LQ.chebyshev_matrices((a, b), N)
        B = LQ._Basis_SinCos(theta)
        _, tp, _, _, ps = B.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache={})
        tp = np.asarray(tp, dtype=float)
        th_nodes = np.array([float(theta["theta"](float(x))) for x in grid])
        order = np.argsort(np.asarray(grid, dtype=float))
        node_sum = float(np.sum(np.abs(np.diff(th_nodes[order]))))
        net = abs(float(theta["theta"](float(b))) - float(theta["theta"](float(a))))
        sign_change = not (bool(np.all(tp > 0)) or bool(np.all(tp < 0)))
        true_var = node_sum if sign_change else net
        return {"span": [float(a).hex(), float(b).hex()], "span_float": [float(a), float(b)], "width": abs(b - a),
                "n_nodes": int(len(grid)), "phase_span_07": ps, "net_variation": net, "theta_prime_sign_change": sign_change,
                "theta_prime_min": float(tp.min()), "theta_prime_max": float(tp.max()), "node_partition_sum": node_sum,
                "true_variation": true_var, "true_variation_basis": "node-partition sum" if sign_change else "|theta(b) - theta(a)|",
                "true_over_estimate": true_var / ps if ps > 0 else None,
                "route_07": route(ps), "route_true": route(true_var), "route_changes": route(ps) != route(true_var),
                "estimate_vs_SIX_PI": "below" if ps < SIX_PI else "at or above",
                "true_vs_SIX_PI": "below" if true_var < SIX_PI else "at or above",
                "estimate_vs_LU": "above" if ps > LU else "at or below", "true_vs_LU": "above" if true_var > LU else "at or below"}

    # seeds: 07's initial partition, the ranges of x_span between windows (H3 leaves lie in these only)
    lo_span, hi_span = min(x_span), max(x_span)
    gaps, cursor = [], lo_span
    for w in sorted(rp["windows"], key=lambda w: float.fromhex(w["lo"])):
        wlo, whi = float.fromhex(w["lo"]), float.fromhex(w["hi"])
        if wlo > cursor:
            gaps.append((cursor, wlo))
        cursor = max(cursor, whi)
    if hi_span > cursor:
        gaps.append((cursor, hi_span))

    def family(lo, hi, depth):
        seed = [g for g in gaps if g[0] <= lo and hi <= g[1]]
        if len(seed) != 1:
            return None, f"no unique seed ({len(seed)})"
        a, b, d, parent, sibling = seed[0][0], seed[0][1], 0, None, None
        while d < depth:
            m = (a + b) / 2.0
            if hi <= m:
                parent, sibling, (a, b) = (a, b), (m, b), (a, m)
            elif lo >= m:
                parent, sibling, (a, b) = (a, b), (a, m), (m, b)
            else:
                return None, f"leaf straddles break point at depth {d}"
            d += 1
        if (a, b) != (lo, hi):
            return None, f"descent reached [{a!r}, {b!r}] at depth {depth}, not the leaf"
        return {"parent": parent, "sibling": sibling}, None

    h3 = [l for l in at["leaves"] if l["part"] == "H3 outside"]
    h3_true = sum(l["true_error"] for l in h3)
    carriers = [l for l in h3 if abs(l["true_error"]) > l["eprime"] + abs(l["E"])]
    top = sorted(h3, key=lambda l: -abs(l["true_error"]))[:3]
    chosen = {l["i"]: l for l in carriers + top}
    rows = []
    for i in sorted(chosen):
        l = chosen[i]
        s, e = float.fromhex(l["start"]), float.fromhex(l["end"])
        lo, hi = min(s, e), max(s, e)
        fam, why = family(lo, hi, l["depth"])
        row = {"i": i, "depth": l["depth"], "type": l["type"],
               "why_chosen": ("carries excess error" if i in {x["i"] for x in carriers} else "") +
                             (" ; " if i in {x["i"] for x in carriers} and i in {x["i"] for x in top} else "") +
                             ("top-3 |true error| in H3" if i in {x["i"] for x in top} else ""),
               "true_error": l["true_error"], "declared": l["eprime"] + abs(l["E"]),
               "true_over_declared": abs(l["true_error"]) / (l["eprime"] + abs(l["E"])) if l["eprime"] + abs(l["E"]) > 0 else None,
               "share_of_H3_true_error": l["true_error"] / h3_true if h3_true else None,
               "leaf": measure(s, e)}
        if fam is None:
            row["family_error"] = why
        else:
            row["parent"] = measure(*oriented(*fam["parent"])) if fam["parent"] else None
            row["sibling"] = measure(*oriented(*fam["sibling"])) if fam["sibling"] else None
        rows.append(row)
    doc = {"header": {"title": "levin-phase-floor 07a Part A: the O9 check on the H3 leaves (added at the user's word, 2026-11-11)",
                      "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "module": str(mod), "module_sha256": sha256(mod), "capture": c["path"], "capture_sha256": c["sha256"],
                      "attr_tables_json": str(attr_json), "attr_tables_json_sha256": sha256(attr_json),
                      "replay_json": str(replay_path), "replay_json_sha256": sha256(replay_path)},
           "sid": sid, "chebyshev_order": N, "rtol": kw["rtol"], "atol": kw["atol"], "SIX_PI": SIX_PI,
           "LEVIN_DIRECT_SOLVE_PHASE_SPAN": LU, "x_span": [float(x).hex() for x in x_span], "seeds": [[a.hex(), b.hex()] for a, b in gaps],
           "h3_true_error": h3_true, "h3_leaves": len(h3), "carriers": len(carriers),
           "carriers_share_of_H3_true_error": sum(l["true_error"] for l in carriers) / h3_true if h3_true else None,
           "rows": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    L = [f"## {sid}: O9 check on its H3 leaves", "",
         f"N = {N} nodes; H3 true error {h3_true:+.3e} over {len(h3)} leaves; {len(carriers)} leaves carry excess error, "
         f"together {doc['carriers_share_of_H3_true_error']:.3f} of it. Script sha256 {doc['header']['script_sha256'][:12]}, "
         f"module sha256 {doc['header']['module_sha256'][:12]}.", "",
         "| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for which in ("leaf", "parent", "sibling"):
            m = r.get(which)
            if m is None:
                L.append(f"| {r['i']} | {r['why_chosen']} | {which} | {r.get('family_error', '-')} | | | | | | | | |")
                continue
            L.append(f"| {r['i']} (d{r['depth']}, true/decl {r['true_over_declared']:.3g}) | {r['why_chosen']} | {which} | "
                     f"[{m['span_float'][0]:.6g}, {m['span_float'][1]:.6g}] | {m['phase_span_07']:.4g} | {m['net_variation']:.4g} | "
                     f"{'yes' if m['theta_prime_sign_change'] else 'no'} | {m['node_partition_sum']:.4g} | "
                     f"{m['true_over_estimate']:.4g} | {m['route_07']} | {m['route_true']} | {'YES' if m['route_changes'] else 'no'} |")
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out_json, sha256(out_json), out_md, sha256(out_md))


if __name__ == "__main__":
    main()
