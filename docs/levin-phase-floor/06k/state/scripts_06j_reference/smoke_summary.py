"""levin-phase-floor 06g (census agent): the smoke ledger summarised, read only (06h's 22b did the same inline):
per unit its status, tree, foreign modules, and 06g's per-call records (errors, abserr and value reassembled,
the budget record parsed, node-limited calls, checked leaves, premise checks), and 06h's re-solving records.
New in 06g. Usage: python -B smoke_summary.py LEDGER"""
import json
import sys

for line in open(sys.argv[1]):
    r = json.loads(line)
    d = r["data"]
    cl = d["calls"] + d["other_calls"]
    ck = [c.get("checked_06g") or {} for c in cl]
    pm = [c.get("premise_06g") or {} for c in cl]
    print(r["unit"], d["status"], "wall %.2f" % d["wall_s"], d["tree_head"][:7], repr(d["tree_status_short"]), "foreign", d["modules"]["foreign"])
    print("   calls", len(cl), "error_06g", sum("error_06g" in c for c in cl),
          "abserr reassembles", sum(c.get("abserr_reassembles") is True for c in cl),
          "value reassembles", sum((c.get("values_06g") or {}).get("value_reassembles") is True for c in cl),
          "budget parsed", sum("node_limited" in (c.get("budget") or {}) for c in cl),
          "node_limited", sum(bool(c.get("node_limited")) for c in cl),
          "converged", sum(c["converged"] for c in cl), "max depth", max(c["max_depth"] for c in cl))
    print("   checked leaves", sum(x.get("n_checked", 0) for x in ck), "returned", sum(x.get("n_checked_returned", 0) for x in ck),
          "b-untrusted", sum(x.get("n_b_untrusted", 0) for x in ck),
          "trust_g != final", sum(x.get("n_trust_g_differs_from_trusted_final", 0) for x in ck),
          "checked errors", sum("error" in x for x in ck),
          "premise checks", sum(x.get("n_checks", 0) for x in pm), "fired", sum(x.get("n_fired", 0) for x in pm),
          "split could meet", sum(x.get("n_split_could_meet_budget") or 0 for x in pm),
          "quarters", sum((c.get("values_06g") or {}).get("n_quarters", 0) for c in cl))
    cc = [c.get("cc_floor") or {} for c in cl]
    dc = [c.get("declared_06h") or {} for c in cl]
    print("   cc_floor leaves", sum(x.get("n_cc_leaves", 0) for x in cc), "reproduced", sum(x.get("n_reproduced", 0) for x in cc),
          "declared leaves", sum(x.get("n_leaves", 0) for x in dc), "reproduced", sum(x.get("n_reproduced", 0) for x in dc))
    an = d.get("analytic_06h") or {}
    cf = an.get("parent_counterfactual") or {}
    print("   analytic_06h h", an.get("h"), "captured", an.get("n_captured"), "counterfactual", "error" if "error" in cf else "ok")
