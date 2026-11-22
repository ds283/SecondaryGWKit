"""levin-phase-floor prompt 07c Part A, section 2.3 (D2): the Clenshaw-Curtis leaves the sample is drawn from, with the
stratum of each, and the draw. New in 07c.

**The population** is every Clenshaw-Curtis leaf (``type`` direct) that 07b's census returned in a call section 2.1 replays
at 07b's code and reproduces (out/replays/tree/<sid>.json ``reproduces_census``, its ``leaves``): the chase's 198 calls and
their counterparts, 07b Part A's 187 (group and analytic), R5's calls and counterparts, and R31's record. A replay that
reproduces its census record reproduces its leaves, so these are census leaves. Only calls with captures can be measured
(the integrand is needed), so the population is not the whole census's: its gate leaves come from the calls these sets
hold, 07b Part A's R2 (30 calls drawn at random from the whole census) among them.

**The strata** (prompt section 2.3):
- ``gate_q1``..``gate_q4``: a leaf the C2 gate sent to the fallback (not a window piece, not a guarded leaf), outside the
  chase, by V / 6 pi in [0, 1/4), [1/4, 1/2), [1/2, 3/4), [3/4, 1); V is the leaf's phase variation as the driver computes
  it (``_Basis_SinCos.build_Levin_data`` on the leaf's order-N nodes);
- ``chase``: a Clenshaw-Curtis leaf of a chase call (or R2-28, R31's record) that descends from a leaf the rate rules
  refused (out/d1/d1_descent.json's leaf lists, ``d1_leaves``);
- ``window``: a window piece or a descendant of one (``window`` true);
- ``guarded``: a guarded leaf re-solved at depth_max (none expected);
each split by the call's kind (``group``, ``analytic``). Both rtols are in every stratum and reported.

**The draw:** per stratum, ``numpy.random.default_rng(seed + i)`` with the D2 seed of state/seeds_07c.json and i the
stratum's index in sorted order, a permutation of its leaves sorted by (sid, start), the first N_PER taken. Written to
state/selection_d2.json before any reference is computed.

Usage: PYTHONPATH=<tree>:<scripts> python -B d2_population_07c.py D1_LEAVES_JSON
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_PER = 25
BINS = [0.25, 0.5, 0.75, 1.0]


def main():
    out_pop = ROOT / "out" / "d2" / "population.json"
    out_sel = ROOT / "state" / "selection_d2.json"
    for p in (out_pop, out_sel):
        if p.exists():
            raise SystemExit(f"{p} exists")
    out_pop.parent.mkdir(parents=True, exist_ok=True)
    d1_leaves = json.loads(Path(sys.argv[1]).read_text())  # {sid: [[start_hex, end_hex], ...]} the chase's CC leaves
    chase_spans = {s: {tuple(x) for x in v} for s, v in d1_leaves["chase_cc_leaves"].items()}
    import AdaptiveLevin.levin_quadrature as LQ
    from leafcc_07c import load_capture

    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
    if (ROOT / "state" / "calls_07c_r5.json").exists():
        man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    pop = []
    for sid in sorted(man):
        rp = ROOT / "out" / "replays" / "tree" / f"{sid}.json"
        if not rp.exists():
            continue
        rep = json.loads(rp.read_text())
        if not rep["reproduces_census"]:
            continue
        cc = [l for l in rep["leaves"] if l["type"] == "direct"]
        if not cc:
            continue
        c = man[sid]
        rec, kw = load_capture(c)
        kind = "group" if "group " in rec["notify_label"] else "analytic"
        A = LQ._Basis_SinCos(rec["theta"])
        N = int(kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
        cache = {}
        for l in cc:
            a, b = float.fromhex(l["start"]), float.fromhex(l["end"])
            grid, D = LQ.chebyshev_matrices((a, b), N)
            V = float(A.build_Levin_data(grid, D, need_AmatT=False, vectorize_cache=cache)[4])
            q = V / (6 * math.pi)
            if l["guarded"]:  # a returned guarded leaf is one re-solved at depth_max (below it, it is bisected)
                st = "guarded"
            elif l["window"]:
                st = "window"
            elif (l["start"], l["end"]) in chase_spans.get(sid, set()):
                st = "chase"
            else:
                st = "gate_q" + str(1 + next(i for i, e in enumerate(BINS) if q < e or i == 3))
            pop.append({"sid": sid, "kind": kind, "rtol": kw["rtol"], "start": l["start"], "end": l["end"],
                        "depth": l["depth"], "window": l["window"], "value_own": l["value_own"], "V_over_6pi": q,
                        "stratum": f"{st}|{kind}"})
    seed = json.loads((ROOT / "state" / "seeds_07c.json").read_text())["seeds"]["D2"]["seed"]
    strata = sorted({p["stratum"] for p in pop})
    sel = {}
    for i, st in enumerate(strata):
        members = sorted((p for p in pop if p["stratum"] == st), key=lambda p: (p["sid"], p["start"]))
        rng = np.random.default_rng(seed + i)
        order = rng.permutation(len(members))
        sel[st] = [members[j] for j in order[:N_PER]]
    counts = Counter(p["stratum"] for p in pop)
    out_pop.write_text(json.dumps({"n": len(pop), "counts": dict(sorted(counts.items())), "leaves": pop}))
    out_sel.write_text(json.dumps({"seed": seed, "n_per": N_PER, "strata": strata, "population_counts": dict(sorted(counts.items())),
                                   "population_sha256": hashlib.sha256(out_pop.read_bytes()).hexdigest(),
                                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "selection": sel}, indent=1))
    print("population by stratum:", dict(sorted(counts.items())))
    print("taken by stratum:", {k: len(v) for k, v in sel.items()})
    print("wrote", out_sel, hashlib.sha256(out_sel.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
