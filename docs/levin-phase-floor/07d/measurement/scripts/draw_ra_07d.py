"""levin-phase-floor prompt 07d Part A, section 2.6 (Ra): about 30 analytic calls of 07c's census in which the chase stop
acts (``num_chase_stops`` positive), drawn with the RA seed. New in 07d.

**The strata** are three properties of the stopped leaf carrying most of the call's abserr (07c's census instrument,
``chase_07c.carrier``):
- its route: ``direct`` (Clenshaw-Curtis) or ``Levin``;
- its trust at the stop: ``trusted_before`` True, False, or None (unchecked) as ``T``, ``U``, ``N``;
- its share of the call's abserr: ``s1`` below 0.1, ``s2`` 0.1 to 0.5, ``s3`` 0.5 and above.
A call with no recorded carrier is in the stratum ``nocarrier``. **The draw:** per stratum (sorted names, index i),
``numpy.random.default_rng(seed + i)`` permutes its calls sorted by sid; the first BASE are taken, then one more from
each stratum in sorted order, round after round, while any is left, until TOTAL are taken. Each drawn call's integrand
is captured at its own rtol and scored at both rtols (its counterpart at the other rtol is the same row and index).
Writes state/selection_ra_07d.json and state/capture_selection_ra.json before any capture or reading.
Usage: PYTHONPATH=<tree>:<scripts> python -B draw_ra_07d.py
"""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

BASE, TOTAL = 2, 30


def stratum(r):
    c = r["carrier"]
    if not c or c.get("type") is None:
        return "nocarrier"
    tb = c.get("trusted_before")
    t = "T" if tb is True else ("U" if tb is False else "N")
    sh = c.get("share")
    s = "s?" if sh is None else ("s1" if sh < 0.1 else ("s2" if sh < 0.5 else "s3"))
    return f"{c['type']}|{t}|{s}"


def main():
    sel_p = WIDER / "state" / "selection_ra_07d.json"
    cap_p = WIDER / "state" / "capture_selection_ra.json"
    for p in (sel_p, cap_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    idx_p = WIDER / "state" / "census_index_07d.pkl"
    rows = [r for r in pickle.loads(idx_p.read_bytes())["rows"] if r["kind"] == "analytic" and r["num_chase_stops"]]
    seeds_p = WIDER / "state" / "seeds_07d.json"
    seed = json.loads(seeds_p.read_text())["seeds"]["RA"]["seed"]
    strata = {}
    for r in rows:
        strata.setdefault(stratum(r), []).append(r)
    orders = {}
    for i, st in enumerate(sorted(strata)):
        m = sorted(strata[st], key=lambda r: r["sid"])
        orders[st] = [m[j] for j in np.random.default_rng(seed + i).permutation(len(m))]
    taken = {st: min(BASE, len(o)) for st, o in orders.items()}
    while sum(taken.values()) < TOTAL and any(taken[st] < len(orders[st]) for st in orders):
        for st in sorted(orders):
            if sum(taken.values()) >= TOTAL:
                break
            if taken[st] < len(orders[st]):
                taken[st] += 1
    sel = [dict(r, ra_stratum=st) for st in sorted(orders) for r in orders[st][:taken[st]]]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: section 2.6's (Ra) draw", [idx_p, seeds_p])
    sel_p.write_text(json.dumps({"header": hdr, "seed": seed, "base": BASE, "total": TOTAL,
                                 "population": {st: len(o) for st, o in sorted(orders.items())},
                                 "taken": dict(sorted(taken.items())), "selection": sel}, indent=1))
    cap = [{"unit": r["unit"], "index": r["index"], "notify_label": r["notify_label"], "census_value": r["value"],
            "census_abserr": r["abserr"], "census_num_regions": r["num_regions"], "census_max_depth": r["max_depth"],
            "census_converged": r["converged"], "census_unit_budget_s": r["budget_s"], "sid": r["sid"]} for r in sel]
    cap_p.write_text(json.dumps(cap, indent=1))
    for l in hdr:
        print(l)
    print("population by stratum:", {st: len(o) for st, o in sorted(orders.items())})
    print("taken:", dict(sorted(taken.items())), "total", len(sel), "by rtol", dict(Counter(r["rtol"] for r in sel)))
    print(f"wrote {sel_p} {sha256(sel_p)}; {cap_p} {sha256(cap_p)}")


if __name__ == "__main__":
    main()
