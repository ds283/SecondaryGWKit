"""levin-phase-floor prompt 07d Part A, section 2.7 table 2: the cause found on every scored leaf where the tree's
declaration understates, one cause per leaf, tested in the order the prompt gives (knots, then noise). New in 07d.

The rule (as the agent applied it reading out/tables/d_e.json at about 10:31Z, written here unchanged):
1. **knots:** the leaf holds interior knots (n_knots > 0) and the Clenshaw-Curtis rule split at the knots
   (leafcc_07d.py ``split_at_knots``, the same nodes per knot interval) has an error below half the leaf's true error:
   the knot carries most of the error;
2. **noise:** otherwise, if the true error is at most 4 s_CC (s_CC = sigma sqrt(sum w^2) on CC_{2N-1}, the leaf's own
   measured scatter): the error is the integrand's noise passed through the rule;
3. **unexplained:** neither.
Reported beside, not used as causes: the reference's check, and the "common" flag (CC_N and CC_{2N-1} err with one sign
and |e47| >= |e24|/2), which a noise-limited leaf can pass by chance (issues.md item 8).
Also counts the window pieces among the noise-caused leaves. Writes out/tables/causes.json and .md.
Usage: PYTHONPATH=<tree>:<scripts> python -B causes_07d.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402


def cause(x):
    if x["n_knots"] and x["err_split_at_knots"] is not None and abs(x["err_split_at_knots"]) < x["true_err"] / 2:
        return "knots"
    if x["noise_ratio"] is not None and x["noise_ratio"] <= 4:
        return "noise"
    return "unexplained"


def main():
    src = ROOT / "out" / "tables" / "d_e.json"
    u = json.loads(src.read_text())["tree_understates"]
    rows = [dict(x, cause=cause(x), window="window" in x["stratum"]) for x in u]
    by = Counter(r["cause"] for r in rows)
    by_st = {}
    for r in rows:
        by_st.setdefault(r["stratum"], Counter())[r["cause"]] += 1
    win = Counter(r["cause"] for r in rows if r["window"])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: causes of the tree's understatements (table 2)", [src])
    od = ROOT / "out" / "tables"
    (od / "causes.json").write_text(json.dumps({"header": hdr, "by_cause": by, "by_stratum": {k: dict(v) for k, v in by_st.items()},
                                                "window_pieces_by_cause": win, "leaves": rows}, indent=1))
    g = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
    L = [l.replace("# ", "", 1) for l in hdr] + ["", f"{len(rows)} leaves: " + ", ".join(f"{k} {v}" for k, v in by.most_common())
         + f"; window pieces among them: " + ", ".join(f"{k} {v}" for k, v in win.most_common()), "",
         "| stratum | knots | noise | unexplained |", "|---|---|---|---|"]
    for st, c in sorted(by_st.items()):
        L.append(f"| {st} | {c['knots']} | {c['noise']} | {c['unexplained']} |")
    L += ["", "| stratum | rank | cause | knots | true err | D tree | err split at knots | true/s_CC | ref check | common |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['stratum']} | {r['rank']} | {r['cause']} | {r['n_knots']} | {r['true_err']:.3g} | {r['D_tree']:.3g} | "
                 f"{g(r['err_split_at_knots'])} | {g(r['noise_ratio'])} | {r['ref_check']} | {r['common']} |")
    (od / "causes.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[7:9]))
    print("wrote", od / "causes.json", sha256(od / "causes.json"))


if __name__ == "__main__":
    main()
