"""levin-phase-floor prompt 07d Part A, section 2.1 item 1: the stratified call sample from 07c's census. New in 07d.

The leaf strata of sections 2.3 and 2.4 are by a leaf's phase variation V, which 07c's census does not record (it needs
the integrand), so the calls are drawn by what the census does record, in classes that hold those leaves, and the
leaf strata are formed from the captured calls' leaves afterwards (leaf_population_07d.py):
- ``G``: a Clenshaw-Curtis leaf and no window (gate leaves of every quarter of V/6pi);
- ``L``: a Levin leaf (Levin leaves of V in [6pi, 20pi) for section 2.4's upper bands);
- ``W``: a window (window pieces);
- ``Ccc``: the chase stop returned a Clenshaw-Curtis leaf (``chase_07c.by_route`` has a ``direct:`` key);
- guarded leaves at the cap: the census has none (census_index_07d_summary.json), so no class.
Cells are (kind, rtol, k band, class): 2 x 2 x 4 x 4 = 64. Per cell, ``numpy.random.default_rng(seed + i)`` (CALLS
seed of state/seeds_07d.json, i the cell's index in sorted order of its name) permutes the cell's calls sorted by sid,
and the first N_PER are taken. The sample is the union. Every call is captured at its own rtol and replayed at both
(the counterpart unit at the other rtol is the same row, same index; 07c Part A's deviation 7).

**Top-up rule, fixed now:** if a leaf stratum is exhausted with fewer than 40 scored leaves, round r (1, 2, ...) draws
N_PER more calls from each cell of the classes that hold that stratum's leaves, with seed + 1000 r, excluding calls
already drawn, by this script with ``--round r``.

Writes state/selection_calls_07d[_rN].json before any capture, and state/capture_selection_07d[_rN].json (x1_capture's
input). Usage: PYTHONPATH=<tree>:<scripts> python -B draw_calls_07d.py [--round R --kinds K[,K] --classes C[,C]]
"""

import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

N_PER = 5
CLASSES = ("G", "L", "W", "Ccc")


def classes(r):
    out = []
    if r["n_cc"] and not r["n_windows"]:
        out.append("G")
    if r["n_levin"]:
        out.append("L")
    if r["n_windows"]:
        out.append("W")
    if any(k.startswith("direct") for k in (r["chase_by_route"] or {})):
        out.append("Ccc")
    return out


def main():
    rnd = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 0
    suf = f"_r{rnd}" if rnd else ""
    sel_p = WIDER / "state" / f"selection_calls_07d{suf}.json"
    cap_p = WIDER / "state" / f"capture_selection_07d{suf}.json"
    for p in (sel_p, cap_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    idx_p = WIDER / "state" / "census_index_07d.pkl"
    rows = pickle.loads(idx_p.read_bytes())["rows"]
    seeds_p = WIDER / "state" / "seeds_07d.json"
    seed = json.loads(seeds_p.read_text())["seeds"]["CALLS"]["seed"] + 1000 * rnd
    taken = set()
    for r_ in range(rnd):
        prev = WIDER / "state" / (f"selection_calls_07d_r{r_}.json" if r_ else "selection_calls_07d.json")
        taken |= {e["sid"] for e in json.loads(prev.read_text())["selection"]}
    # the top-up's restriction (round > 0): only the cells of the classes that hold the short strata' leaves,
    # given as --kinds K[,K] --classes C[,C] (the rule above), recorded in the selection
    kinds = sys.argv[sys.argv.index("--kinds") + 1].split(",") if "--kinds" in sys.argv else None
    want_cls = sys.argv[sys.argv.index("--classes") + 1].split(",") if "--classes" in sys.argv else None
    cells = {}
    for r in rows:
        for c in classes(r):
            if (kinds and r["kind"] not in kinds) or (want_cls and c not in want_cls):
                continue
            cells.setdefault(f"{r['kind']}|{r['rtol']}|k{r['k_band']}|{c}", []).append(r)
    sel, by_cell = {}, {}
    for i, name in enumerate(sorted(cells)):
        members = sorted((r for r in cells[name] if r["sid"] not in taken), key=lambda r: r["sid"])
        order = np.random.default_rng(seed + i).permutation(len(members))
        got = [members[j] for j in order[:N_PER]]
        by_cell[name] = {"population": len(members), "taken": [r["sid"] for r in got]}
        for r in got:
            sel.setdefault(r["sid"], dict(r, cells=[]))["cells"].append(name)
    selection = sorted(sel.values(), key=lambda r: r["sid"])
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07d Part A: section 2.1 item 1's call sample, round {rnd}",
                 [idx_p, seeds_p])
    sel_p.write_text(json.dumps({"header": hdr, "round": rnd, "seed": seed, "n_per": N_PER, "kinds": kinds,
                                 "classes": want_cls, "cells": by_cell,
                                 "selection": selection}, indent=1))
    cap = [{"unit": r["unit"], "index": r["index"], "notify_label": r["notify_label"], "census_value": r["value"],
            "census_abserr": r["abserr"], "census_num_regions": r["num_regions"], "census_max_depth": r["max_depth"],
            "census_converged": r["converged"], "census_unit_budget_s": r["budget_s"], "sid": r["sid"]}
           for r in selection]
    cap_p.write_text(json.dumps(cap, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(selection)} calls in {len({r['unit'] for r in selection})} units; by kind/rtol:",
          dict(Counter((r["kind"], r["rtol"]) for r in selection)))
    print("cells with fewer than N_PER:", {k: v["population"] for k, v in by_cell.items() if len(v["taken"]) < N_PER})
    print(f"wrote {sel_p} {sha256(sel_p)}; {cap_p} {sha256(cap_p)}")


if __name__ == "__main__":
    main()
