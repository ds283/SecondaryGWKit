"""levin-phase-floor prompt 07d Part A, section 2.6 (Rt): about 30 own-class calls whose route changes under any (T) or
(ET) variant, drawn with the RT seed. New in 07d.

**The population** is this Part's own drawn integrands (the replay manifest's ``d:`` entries: section 2.1's class draw,
its analytic top-up with ride-along; not the ``R:`` regression entries, which 07c's references score), each integrand
counted once (its two entries, one per rtol, share one capture), whose replay under ET20, ET12 or T20 changes the route of
at least one region at either rtol (the variant's own counter ``num_07d_route_changed``). The (Ra) integrands are left out:
their references are read in any case.
**The direction** is Levin to Clenshaw-Curtis for every member: each (T) and (ET) variant raises the threshold above 6 pi,
so no region can move the other way; the reverse stratum is empty by construction, and its emptiness is recorded.
**The strata** are kind (group or analytic), k band (the census index's ``k_band``) and rtol (the integrand's drawn
entry's, not its counterpart's). **The draw** is (Ra)'s: per stratum (sorted names, index i),
``numpy.random.default_rng(seed + i)`` permutes its integrands sorted by sid; the first BASE are taken, then one more from
each stratum in sorted order, round after round, until TOTAL are taken. Each is scored at both rtols.
Writes state/selection_rt_07d.json before any reading. Usage: PYTHONPATH=<tree>:<scripts> python -B draw_rt_07d.py
"""

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

BASE, TOTAL = 1, 30
VARS = ("ET20", "ET12", "T20")


def main():
    sel_p = ROOT / "state" / "selection_rt_07d.json"
    if sel_p.exists():
        raise SystemExit(f"{sel_p} exists")
    from replay_07d import manifest

    m = manifest()
    idx_p = ROOT / "state" / "census_index_07d.pkl"
    rows = {r["sid"]: r for r in pickle.loads(idx_p.read_bytes())["rows"]}
    seeds_p = ROOT / "state" / "seeds_07d.json"
    seed = json.loads(seeds_p.read_text())["seeds"]["RT"]["seed"]
    by = defaultdict(list)
    for k, c in m.items():
        if k.startswith("d:"):
            by[c["sha256"]].append(k)
    ra = {c["sha256"] for c in m.values() if c["set"] in ("ra", "ra~o")}
    srcs = [idx_p, seeds_p]
    rc = {}
    for v in VARS:
        for ks in by.values():
            for k in ks:
                p = ROOT / "out" / "replays" / v / (k.replace(":", "_").replace("|", "_") + ".json")
                rc.setdefault(k, {})[v] = json.loads(p.read_text())["extra_07d"]["num_07d_route_changed"]
    strata, n_ra_changed = {}, 0
    for h, ks in sorted(by.items()):
        if not any(rc[k][v] > 0 for k in ks for v in VARS):
            continue
        if h in ra:
            n_ra_changed += 1
            continue
        prim = [k for k in ks if not m[k]["set"].endswith("~o")]
        assert len(prim) == 1, ks
        cp = [k for k in ks if k != prim[0]][0]
        r = rows[m[prim[0]]["census"]]
        st = f"Levin->CC|{r['kind']}|k{r['k_band']}|{r['rtol']}"
        strata.setdefault(st, []).append({"sid": prim[0][2:], "counterpart_sid": cp[2:], "unit": r["unit"], "index": r["index"],
                                          "kind": r["kind"], "k_band": r["k_band"], "rtol": r["rtol"],
                                          "notify_label": r["notify_label"], "set": m[prim[0]]["set"],
                                          "route_changed": {k[2:]: rc[k] for k in (prim[0], cp)}})
    orders = {}
    for i, st in enumerate(sorted(strata)):
        mm = sorted(strata[st], key=lambda e: e["sid"])
        orders[st] = [mm[j] for j in np.random.default_rng(seed + i).permutation(len(mm))]
    taken = {st: min(BASE, len(o)) for st, o in orders.items()}
    while sum(taken.values()) < TOTAL and any(taken[st] < len(orders[st]) for st in orders):
        for st in sorted(orders):
            if sum(taken.values()) >= TOTAL:
                break
            if taken[st] < len(orders[st]):
                taken[st] += 1
    sel = [dict(e, rt_stratum=st) for st in sorted(orders) for e in orders[st][:taken[st]]]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: section 2.6's (Rt) draw", srcs)
    sel_p.write_text(json.dumps({"header": hdr, "seed": seed, "base": BASE, "total": TOTAL, "variants_read": list(VARS),
                                 "population": {st: len(o) for st, o in sorted(orders.items())},
                                 "reverse_direction_population": 0, "ra_integrands_route_changed_left_out": n_ra_changed,
                                 "taken": dict(sorted(taken.items())), "selection": sel}, indent=1))
    for l in hdr:
        print(l)
    print("population:", {st: len(o) for st, o in sorted(orders.items())}, "total", sum(len(o) for o in orders.values()))
    print("(Ra) integrands with a route change left out:", n_ra_changed)
    print("taken:", dict(sorted(taken.items())), "total", len(sel), "by kind", dict(Counter(e["kind"] for e in sel)))
    print(f"wrote {sel_p} {sha256(sel_p)}")


if __name__ == "__main__":
    main()
