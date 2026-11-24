"""levin-phase-floor prompt 07d Part A' (section 9.2 item 6, last paragraph; the orchestrator's second message): the calls
K and KT change, stratified as Part A's (Rt) strata, and the referenced calls in each. New in Part A'.

**Changed by the bisection:** a call whose returned leaves' spans under the variant differ from those under its base
without knots (K against V93, KT against V93T<T>: the same declaration and gate, the tree's midpoints).
**Its stratum:** direction x kind x k band x rtol, Part A's (Rt) strata (draw_rt_07d.py) with the direction
``Levin->CC`` where the variant changed a route (``num_a2_route_changed`` > 0, i.e. against the tree's 6 pi gate) and
``layout changed, routes unchanged`` otherwise; kind by notify_label ("group " or not); k band from Part A's census index
(``k_band`` of the call's census row); rtol the call's. Only this Part's own drawn integrands (``d:`` sids) are
stratified, as Part A's (Rt) did; the regression set's changed calls (``R:``) are counted beside.
**Referenced:** a changed call is referenced if it is one of Part A's scored calls (Part A's scores_07d.json ``calls``),
by set (Ra, Rt, regression).
A stratum with changed calls and no referenced changed call is **empty**.
Usage: python -B strata_k_a2.py VARIANT:BASE [VARIANT:BASE ...]
"""

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, ROOT, header  # noqa: E402


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def spans(doc):
    return [(l["start"], l["end"]) for l in doc["leaves"]]


def main():
    pairs = [a.split(":") for a in sys.argv[1:]]
    sids = [s for s in (ROOT / "state" / "sids_all_a2.txt").read_text().split() if s]
    man = {}
    for p in sorted((PARTA / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    kb = {r["sid"]: r["k_band"] for r in pickle.loads((PARTA / "state" / "census_index_07d.pkl").read_bytes())["rows"]}
    sc = json.loads((PARTA / "out" / "tables" / "scores_07d.json").read_text())["calls"]
    out = {}
    L = []
    for v, base in pairs:
        strata = defaultdict(lambda: {"changed": 0, "referenced": Counter(), "referenced_sids": []})
        reg = Counter()
        for sid in sids:
            dv = json.loads((ROOT / "out" / "replays" / v / f"{fname(sid)}.json").read_text())
            db = json.loads((ROOT / "out" / "replays" / base / f"{fname(sid)}.json").read_text())
            if spans(dv) == spans(db):
                continue
            ref = sc.get(sid)
            if not sid.startswith("d:"):
                reg["changed"] += 1
                if ref:
                    reg["referenced " + ref["set"].split("~")[0]] += 1
                continue
            kind = "group" if "group " in dv["notify_label"] else "analytic"
            rt = "1e-7" if float.fromhex(dv["rtol"]) == 1e-7 else "1e-8"
            direction = "Levin->CC" if int((dv.get("extra_a2") or {}).get("num_a2_route_changed", 0)) > 0 \
                else "layout changed, routes unchanged"
            st = f"{direction}|{kind}|k{kb.get(man[sid]['census'])}|{rt}"
            S = strata[st]
            S["changed"] += 1
            if ref:
                S["referenced"][ref["set"].split("~")[0]] += 1
                S["referenced_sids"].append(sid)
        out[v] = {"base": base, "strata": {k: {"changed": s["changed"], "referenced": dict(s["referenced"]),
                                                "referenced_sids": s["referenced_sids"],
                                                "empty": s["changed"] > 0 and not s["referenced"]}
                                            for k, s in sorted(strata.items())},
                  "regression_set": dict(reg)}
        L += [f"### {v} against {base}", "", "| stratum | changed | referenced (by set) | empty |", "|---|---|---|---|"]
        L += [f"| {k} | {s['changed']} | {s['referenced']} | {'EMPTY' if s['empty'] else ''} |" for k, s in out[v]["strata"].items()]
        L += ["", f"regression set (R: sids): {dict(reg)}", ""]
    hdr = header("levin-phase-floor 07d Part A': K's and KT's changed calls by (Rt) stratum", __file__,
                 [PARTA / "out" / "tables" / "scores_07d.json", PARTA / "state" / "census_index_07d.pkl"])
    tag = "_".join(v for v, _ in pairs)
    (ROOT / "out" / "tables" / f"strata_{tag}.json").write_text(json.dumps({"header": hdr, **out}, indent=1))
    (ROOT / "out" / "tables" / f"strata_{tag}.md").write_text("\n".join(hdr + [""] + L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
