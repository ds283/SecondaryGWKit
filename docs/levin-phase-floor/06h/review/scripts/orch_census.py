"""
Orchestrator's own re-score of prompt 06h §5's H3 and H4, and a check of the descriptive counts,
from the two census checkpoints (06e's, the "before"; 06h's, the tree h). Read-only.

Usage: orch_census.py CHECKPOINT_06E CHECKPOINT_06H CHECKPOINT_06A

- Units are keyed by (rtol, row); the last record of a unit wins, and duplicates are counted.
- The stationary-point rows are 06a's: rows whose 1e-7 record has a scanned call with
  n_sign_changes > 0 (the rule of analyse06c.scan06a), asserted to be 720.
- H3 (as written): outside them, rows with finite total_abserr under both,
  |total_h - total_06e| <= abserr_06e + abserr_h. Second reading: the same on analytic_rad, with
  the oracle abserr where both records carry one (field names passed through ORACLE_* below).
- H4: outside them, every Levin call (group calls and analytic-block calls, matched by index with
  equal labels) that 06e reported converged is converged under h.
Floats are compared by their recorded hex.
"""

import collections
import json
import math
import sys


def fl(v):
    if isinstance(v, dict) and "hex" in v:
        return float.fromhex(v["hex"])
    return v


def load(path, rtol_filter=None):
    recs, dup = {}, 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            key = (d["rtol"], d["row"])
            if key in recs:
                dup += 1
            recs[key] = slim(d)
    return recs, dup


CALL_KEYS = ("index", "notify_label", "converged", "num_regions", "num_subregion_solves", "max_depth")


def slim(d):
    # keep only what is scored, so both checkpoints fit in memory alongside the phase job
    return dict(
        status=d["status"],
        wall_s=d.get("wall_s"),
        result=d.get("result") or {},
        calls=[{k: c.get(k) for k in CALL_KEYS} for c in d.get("calls") or []],
        other_calls=[{k: c.get(k) for k in CALL_KEYS} for c in d.get("other_calls") or []],
        analytic_06h=d.get("analytic_06h"),
        head=(d.get("tree_head"), d.get("tree_status_short")),
    )


def stationary(path06a):
    sp = set()
    with open(path06a) as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    assert len(sp) == 720, len(sp)
    return sp


def calls(d):
    return [("group", c) for c in d.get("calls") or []] + [
        ("analytic", c) for c in d.get("other_calls") or []
    ]


def main():
    e, dup_e = load(sys.argv[1])
    h, dup_h = load(sys.argv[2])
    sp = stationary(sys.argv[3])
    print(f"06e units {len(e)} (dup records {dup_e}); 06h units {len(h)} (dup records {dup_h})")
    print(f"keys equal: {set(e) == set(h)}; stationary-point rows: {len(sp)}")
    heads = collections.Counter(d["head"] for d in h.values())
    print(f"06h tree heads/status: {dict(heads)}")
    status_h = collections.Counter(d["status"] for d in h.values())
    status_e = collections.Counter(d["status"] for d in e.values())
    print(f"status 06e {dict(status_e)}; status 06h {dict(status_h)}")

    for rtol in ("1e-7", "1e-8"):
        keys = sorted(k for k in e if k[0] == rtol and k in h)
        # --- H3 -------------------------------------------------------------------------------
        n = viol = nonfin = ident = notdone = 0
        worst = (0.0, None)
        an_n = an_ident = 0
        for k in keys:
            if k[1] in sp:
                continue
            de, dh = e[k], h[k]
            if de["status"] != "done" or dh["status"] != "done":
                notdone += 1
                continue
            te, th = fl(de["result"]["total"]), fl(dh["result"]["total"])
            ee, eh = fl(de["result"]["total_abserr"]), fl(dh["result"]["total_abserr"])
            if not (math.isfinite(ee) and math.isfinite(eh)):
                nonfin += 1
                continue
            n += 1
            ident += te.hex() == th.hex()
            d = abs(th - te)
            r = d / (ee + eh) if ee + eh > 0 else (0.0 if d == 0 else math.inf)
            if r > worst[0]:
                worst = (r, k)
            if d > ee + eh:
                viol += 1
                print(f"  H3 VIOLATION {k}: {te.hex()} {th.hex()} abserr {ee.hex()} {eh.hex()}")
            ae = fl(de["result"].get("analytic_rad"))
            ah = fl(dh["result"].get("analytic_rad"))
            if ae is not None and ah is not None:
                an_n += 1
                an_ident += ae.hex() == ah.hex()
        print(
            f"[{rtol}] H3 total: scored {n}, bit-identical {ident}, violations {viol},"
            f" non-finite {nonfin}, not done {notdone}, worst ratio {worst[0]:.6e} at {worst[1]}"
        )
        print(f"[{rtol}] analytic_rad present in both {an_n}, bit-identical {an_ident}")

        # --- H3, second reading: the oracle, h against 06e's oracle recomputed in the h unit ----
        n2 = viol2 = cf_match = cf_mismatch = missing = nonfin2 = 0
        worst2 = (0.0, None)
        for k in keys:
            if k[1] in sp:
                continue
            de, dh = e[k], h[k]
            if de["status"] != "done" or dh["status"] != "done":
                continue
            ah = (dh.get("analytic_06h") or {})
            oh, oc = ah.get("h"), ah.get("parent_counterfactual")
            if not oh or not oc:
                missing += 1
                continue
            vh, eh = fl(oh["value"]), fl(oh["abserr"])
            vc, ec = fl(oc["value"]), fl(oc["abserr"])
            ae = de["result"].get("analytic_rad")
            if ae is not None and fl(ae).hex() == vc.hex():
                cf_match += 1
            else:
                cf_mismatch += 1
            if not (math.isfinite(eh) and math.isfinite(ec)):
                nonfin2 += 1
                continue
            n2 += 1
            d = abs(vh - vc)
            r = d / (eh + ec) if eh + ec > 0 else (0.0 if d == 0 else math.inf)
            if r > worst2[0]:
                worst2 = (r, k)
            if d > eh + ec:
                viol2 += 1
                print(f"  H3-oracle VIOLATION {k}: {vc.hex()} {vh.hex()} abserr {ec.hex()} {eh.hex()}")
        print(
            f"[{rtol}] H3 oracle: scored {n2}, violations {viol2}, non-finite {nonfin2},"
            f" no oracle record {missing}, worst ratio {worst2[0]:.6e} at {worst2[1]};"
            f" counterfactual == 06e analytic_rad in {cf_match}, differs/absent in {cf_mismatch}"
        )

        # --- H4 -------------------------------------------------------------------------------
        scored = collections.Counter()
        lost, gained, problems = [], [], []
        depth_e, depth_h = collections.Counter(), collections.Counter()
        conv_e = conv_h = 0
        reg_e = reg_h = sol_e = sol_h = 0
        wall_e = wall_h = 0.0
        for k in keys:
            de, dh = e[k], h[k]
            wall_e += de.get("wall_s") or 0.0
            wall_h += dh.get("wall_s") or 0.0
            ce, ch = calls(de), calls(dh)
            for _, c in ce:
                depth_e[c.get("max_depth")] += 1
                conv_e += bool(c.get("converged"))
                reg_e += c.get("num_regions") or 0
                sol_e += c.get("num_subregion_solves") or 0
            for _, c in ch:
                depth_h[c.get("max_depth")] += 1
                conv_h += bool(c.get("converged"))
                reg_h += c.get("num_regions") or 0
                sol_h += c.get("num_subregion_solves") or 0
            if k[1] in sp:
                continue
            bi = {c["index"]: (kind, c) for kind, c in ch}
            if len(ce) != len(ch):
                problems.append(f"{k}: call counts {len(ce)} vs {len(ch)}")
            for kind, c in ce:
                m = bi.get(c["index"])
                if m is None or m[1]["notify_label"] != c["notify_label"] or m[0] != kind:
                    problems.append(f"{k} call {c['index']}: unmatched")
                    continue
                scored[kind] += 1
                if c["converged"] and not m[1]["converged"]:
                    lost.append((k, kind, c["index"], c["notify_label"]))
                if not c["converged"] and m[1]["converged"]:
                    gained.append((k, kind, c["index"], c["notify_label"]))
        print(
            f"[{rtol}] H4: calls scored {dict(scored)}; lost convergence {len(lost)};"
            f" gained {len(gained)}; matching problems {len(problems)}"
        )
        for x in lost[:50]:
            print(f"  LOST {x}")
        for x in gained[:20]:
            print(f"  gained {x}")
        for x in problems[:20]:
            print(f"  problem {x}")
        print(f"[{rtol}] converged calls 06e {conv_e}, 06h {conv_h}")
        print(f"[{rtol}] regions 06e {reg_e}, 06h {reg_h}; solves 06e {sol_e}, 06h {sol_h}")
        print(f"[{rtol}] unit wall_s 06e {wall_e/3600:.3f} h, 06h {wall_h/3600:.3f} h")
        print(f"[{rtol}] max_depth 06e {sorted(depth_e.items(), key=str)}")
        print(f"[{rtol}] max_depth 06h {sorted(depth_h.items(), key=str)}")


if __name__ == "__main__":
    main()
