"""levin-phase-floor prompt 07d Part A, section 2.3: the candidates of state/candidates_07d.json, and the cause of every
understatement by the tree's declaration, on 07c's 176 truncation-dominated chase leaves as re-measured
(out/trunc176/out/, trunc176_07d.py). New in 07d. Verdicts by state/refs_rule_leaf_07d.txt (v3).
Cause, read in this order: ``knot`` (interior knots, and the leaf split at them has less than half the true error),
``noise`` (true error <= 3 s_CC), ``noise 3-4 s_CC``, ``knots, split does not fix``, ``other``.
Writes out/tables/trunc176.json and .md. Usage: PYTHONPATH=<tree>:<scripts> python -B trunc176_table_07d.py
"""
import json, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402
from d_e_table_07d import declaration, verdict  # noqa: E402
fx = float.fromhex


def cause(l, te):
    s = l["noise"]["s_cc"]
    sp = l.get("split_at_knots")
    if l["n_knots"] and sp is not None and abs(fx(sp["err_split"])) < 0.5 * te:
        return "knot"
    if te <= 3 * s:
        return "noise"
    if te <= 4 * s:
        return "noise 3-4 s_CC"
    return "knots, split does not fix" if l["n_knots"] else "other"


def main():
    cdoc = json.loads((ROOT / "state" / "candidates_07d.json").read_text())
    cands, consts = cdoc["candidates"], cdoc["constants"]
    tab = {n: Counter() for n in cands}
    causes, rows = Counter(), []
    for f in sorted((ROOT / "out" / "trunc176" / "out").glob("*.json")):
        for l in json.loads(f.read_text())["leaves"]:
            if "error" in l:
                continue
            te, eps = abs(fx(l["err47"])), fx(l["eps_ref"])
            v = {}
            for n, c in cands.items():
                D, _ = declaration(l, c, consts)
                v[n] = "n/a" if D is None else verdict(D, te, eps)
                tab[n][v[n]] += 1
            cz = cause(l, te) if v["tree"] == "understates" else None
            if cz:
                causes[cz] += 1
            rows.append({"id": l["id"], "verdicts": v, "true_err": te, "eps_ref": eps, "cause": cz, "n_knots": l["n_knots"],
                         "ref_check": l["ref_check"], "te_over_s_cc": te / l["noise"]["s_cc"] if l["noise"]["s_cc"] else None})
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: 07c's 176 truncation-dominated chase leaves", [ROOT / "state" / "candidates_07d.json"])
    od = ROOT / "out" / "tables"
    (od / "trunc176.json").write_text(json.dumps({"header": hdr, "by_candidate": {k: dict(v) for k, v in tab.items()},
                                                   "tree_understates_causes": dict(causes), "leaves": rows}, indent=1))
    L = [l.replace("# ", "", 1) for l in hdr] + ["", "| candidate | bounded | understates | undecided |", "|---|---|---|---|"]
    for n, c in tab.items():
        L.append(f"| {n} | {c['bounded']} | {c['understates']} | {c['undecided']} |")
    L += ["", "Causes of the tree's understatements: " + ", ".join(f"{k} {v}" for k, v in causes.items())]
    (od / "trunc176.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
