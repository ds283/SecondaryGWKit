"""levin-phase-floor prompt 07d Part A, sections 2.2-2.4: the leaf population, its strata and their seeded order, written
before any leaf is solved. New in 07d.

**The population** is every leaf the tree returned in a section 2.1 item 1 call (``d:`` sids, both rtols), replayed at the
tree and reproducing 07c's census record (out/replays/tree/<sid>.json; a replay that does not reproduce is a stop, so
none is left out silently). Each leaf's V is computed as the driver computes it (``A.build_Levin_data`` on its order-N
nodes, from the call's capture).

**The strata** (prompt sections 2.3 and 2.4), each split by the call's kind (group, analytic) and rtol:
- Clenshaw-Curtis leaves (``type`` direct): ``window`` (a window piece or a descendant of one); else ``chase`` (returned
  by the chase stop: the replay's stop instrument recorded it); else ``guarded`` (re-solved at depth_max; none in 07c's
  census); else ``gate_q1``..``gate_q4`` by V/6pi in [0, 1/4), [1/4, 1/2), [1/2, 3/4), [3/4, 1);
- Levin leaves, section 2.4's upper bands by V: ``band_6_8`` [6pi, 8pi), ``band_8_10``, ``band_10_12``, ``band_12_16``,
  ``band_16_20`` [16pi, 20pi). Levin leaves outside them are not sampled. Section 2.4's quarters below 6pi are the gate
  strata (the same leaves, solved both ways).
**The order:** per stratum, ``numpy.random.default_rng(seed + i)``, i the stratum's index in sorted order among the
strata of its seed, permutes its leaves sorted by (sid, start); the LEAVES seed for the Clenshaw-Curtis strata, BANDS
for the Levin bands (state/seeds_07d.json). Leaves are filled in that order (fill_leaves_07d.py).
Writes out/leaves/population.json and state/selection_leaves_07d.json.
Usage: PYTHONPATH=<tree>:<scripts> python -B leaf_population_07d.py
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
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

BINS = [0.25, 0.5, 0.75, 1.0]
BANDS = [(6, 8), (8, 10), (10, 12), (12, 16), (16, 20)]


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def main():
    # --extend NAME (the top-up, log deviation 8): the leaves of the calls of sets NAME and NAME~o are appended to each
    # stratum's order after its existing leaves, in a seeded order of their own (seed + 100 r + i, r the extension's
    # number), so that every prefix already filled is unchanged. The selection before the extension is kept as
    # state/superseded/selection_leaves_07d_before_<NAME>.json.
    ext = sys.argv[sys.argv.index("--extend") + 1] if "--extend" in sys.argv else None
    out_pop = ROOT / "out" / "leaves" / ("population.json" if not ext else f"population_{ext}.json")
    out_sel = ROOT / "state" / "selection_leaves_07d.json"
    if ext:
        if out_pop.exists():
            raise SystemExit(f"{out_pop} exists")
        keep = ROOT / "state" / "superseded" / f"selection_leaves_07d_before_{ext}.json"
        if keep.exists():
            raise SystemExit(f"{keep} exists")
        old_sel = json.loads(out_sel.read_text())
        keep.write_text(out_sel.read_text())
    else:
        for p in (out_pop, out_sel):
            if p.exists():
                raise SystemExit(f"{p} exists")
    out_pop.parent.mkdir(parents=True, exist_ok=True)
    import AdaptiveLevin.levin_quadrature as LQ
    from replay_07d import load_capture, manifest

    man = manifest()
    pop, not_rep = [], []
    sids_ = sorted(s for s in man if s.startswith("d:") and (ext is None or man[s]["set"] in (ext, f"{ext}~o"))
                   and (ext is not None or man[s]["set"] in ("calls", "calls~o")))
    for sid in sids_:
        rp = ROOT / "out" / "replays" / "tree" / f"{fname(sid)}.json"
        rep = json.loads(rp.read_text())
        if not rep["reproduces_census"]:
            not_rep.append(sid)
            continue
        c = man[sid]
        rec = load_capture(c)
        kind = "group" if "group " in rec["notify_label"] else "analytic"
        A = LQ._Basis_SinCos(rec["theta"])
        N = int(rec["kw"].get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
        cache = {}
        rtl = "1e-7" if sid.split("|")[0].split("-")[1:3] == ["1e", "7"] else "1e-8"  # the unit's rtol, by name
        for l in rep["leaves"]:
            a, b = float.fromhex(l["start"]), float.fromhex(l["end"])
            grid, D = LQ.chebyshev_matrices((min(a, b), max(a, b)), N)
            V = float(A.build_Levin_data(grid, D, need_AmatT=False, vectorize_cache=cache)[4])
            q = V / (6 * math.pi)
            if l["type"] == "direct":
                if l["window"]:
                    st = "window"
                elif l["chase_stopped"]:
                    st = "chase"
                elif l["guard_at_depth_max"]:
                    st = "guarded"
                else:
                    st = "gate_q" + str(1 + next(i for i, e in enumerate(BINS) if q < e or i == 3))
                seedname = "LEAVES"
            else:
                bd = [f"band_{lo}_{hi}" for lo, hi in BANDS if lo * math.pi <= V < hi * math.pi]
                if not bd:
                    continue
                st = bd[0]
                seedname = "BANDS"
            pop.append({"sid": sid, "kind": kind, "rtol": rtl, "start": l["start"], "end": l["end"],
                        "type": l["type"], "depth": l["depth"], "window": l["window"], "chase_stopped": l["chase_stopped"],
                        "value_own": l["value_own"], "V": V, "V_over_6pi": q, "stratum": f"{st}|{kind}|{rtl}",
                        "seed_name": seedname})
    if not_rep:
        raise SystemExit(f"replays that do not reproduce 07c's census: {not_rep} (a stop, section 8)")
    seeds = json.loads((ROOT / "state" / "seeds_07d.json").read_text())["seeds"]
    sel = {}
    r_ext = 0 if not ext else 1 + len(old_sel.get("extensions", []))
    for seedname in ("LEAVES", "BANDS"):
        strata = sorted({p["stratum"] for p in pop if p["seed_name"] == seedname})
        for i, st in enumerate(strata):
            members = sorted((p for p in pop if p["stratum"] == st), key=lambda p: (p["sid"], p["start"]))
            sd = seeds[seedname]["seed"] + 100 * r_ext + i
            order = np.random.default_rng(sd).permutation(len(members))
            sel[st] = {"seed": sd, "order": [members[j] for j in order]}
    if ext:
        merged = old_sel["selection"]
        for st, v in sel.items():
            if st in merged:
                merged[st]["order"].extend(v["order"])
                merged[st].setdefault("extensions", []).append({"name": ext, "seed": v["seed"], "n": len(v["order"])})
            else:
                merged[st] = {"seed": v["seed"], "order": v["order"], "extensions": [{"name": ext, "seed": v["seed"], "n": len(v["order"])}]}
        sel = merged
    counts = Counter(p["stratum"] for p in pop)
    out_pop.write_text(json.dumps({"n": len(pop), "counts": dict(sorted(counts.items())), "leaves": pop}))
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: the leaf strata and their seeded order",
                 [out_pop, ROOT / "state" / "seeds_07d.json"])
    doc = {"header": hdr, "population_counts": dict(sorted(counts.items())), "population_sha256": sha256(out_pop), "selection": sel}
    if ext:
        doc = dict(old_sel, selection=sel, extensions=old_sel.get("extensions", []) + [
            {"name": ext, "header": hdr, "population_counts": dict(sorted(counts.items())), "population_sha256": sha256(out_pop)}])
    out_sel.write_text(json.dumps(doc))
    for l in hdr:
        print(l)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("wrote", out_sel, sha256(out_sel))


if __name__ == "__main__":
    main()
