"""levin-phase-floor prompt 07c Part A, section 2.3 (D2): each measured Clenshaw-Curtis leaf classed by what dominates CC_{2N-1}'s
true error. New in 07c. The noise CC_{2N-1} carries from independent integrand noise of RMS sigma per sample (leafcc's
``noise_sigma``, the least-squares fit's residual) is s_CC = sigma sqrt(sum_j w_j^2), w_j the fine rule's weights on the
leaf (``_cc_weights``). A leaf is *noise-dominated* if its true error is at most 3 s_CC, else *truncation-dominated*.
Per regime and stratum, each candidate's ratio estimate / true error over the resolved leaves (min, median, max, below 1),
and the same for max(estimate, floor), the declaration the leaf would carry; and s_CC against the floor.
Usage: PYTHONPATH=<tree> python -B d2_regimes_07c.py OUT_JSON DIR[:LABEL] [DIR[:LABEL] ...]"""
import json, statistics, sys, math
from pathlib import Path
import numpy as np
import AdaptiveLevin.levin_quadrature as LQ
fx = float.fromhex
def st(xs):
    return None if not xs else {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs), "below_1": sum(x < 1 for x in xs)}
out = {}
for arg in sys.argv[2:]:
    d, lab = (arg.split(":") + [None])[:2]
    for p in sorted(Path(d).glob("*.json")):
        doc = json.loads(p.read_text())
        N = doc["N"]
        wb2 = float(np.sum(LQ._cc_weights_base(2 * N - 1) ** 2))
        for l in doc["leaves"]:
            if "error" in l:
                continue
            w = abs(fx(l["end"]) - fx(l["start"]))
            sCC = fx(l["noise_sigma"]) * math.sqrt(wb2) * w / 2
            te = fx(l["true_err"])
            regime = "noise" if te <= 3 * sCC else "truncation"
            stratum = (lab or "?") if lab else l["id"].split("|")[0] + "|" + l["id"].split("|")[1]
            key = f"{stratum}|{regime}"
            o = out.setdefault(key, {"n": 0, "resolved": 0, "sCC_over_floor": [], "c": {}, "cf": {}})
            o["n"] += 1
            o["sCC_over_floor"].append(sCC / fx(l["phase_err"]))
            if not l["resolved"] or te == 0:
                continue
            o["resolved"] += 1
            for c, v in l["candidates"].items():
                if v is None:
                    continue
                o["c"].setdefault(c, []).append(fx(v) / te)
                o["cf"].setdefault(c, []).append(max(fx(v), fx(l["phase_err"])) / te)
res = {k: {"n": v["n"], "resolved": v["resolved"], "sCC_over_floor": st(v["sCC_over_floor"]),
           "ratio": {c: st(x) for c, x in v["c"].items()}, "ratio_max_est_floor": {c: st(x) for c, x in v["cf"].items()}}
       for k, v in sorted(out.items())}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print("| stratum|regime | n | resolved | s_CC/floor median | candidate | est/true min | median | below 1 | max(est,floor)/true min | below 1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, v in res.items():
    for c in ("pair", "nest3", "tail_sum4", "tail_max4x2"):
        a, b = v["ratio"].get(c), v["ratio_max_est_floor"].get(c)
        print(f"| {k} | {v['n']} | {v['resolved']} | {v['sCC_over_floor']['median']:.3g} | {c} | "
              + (f"{a['min']:.3g} | {a['median']:.3g} | {a['below_1']} | {b['min']:.3g} | {b['below_1']} |" if a else "- | - | - | - | - |"))
