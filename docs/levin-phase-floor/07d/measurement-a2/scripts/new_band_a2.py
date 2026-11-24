"""levin-phase-floor prompt 07d Part A' (section 9.2 item 3): the new band [20 pi, 32 pi), its population, its seeded order
and its specs, written before any of its leaves is solved. New in Part A'.

**The population** ("visited, not returned"; the orchestrator's go-ahead, state/goahead_orchestrator_20261123.md item 1):
every region I6's replay at 6 pi made by the Levin route (``is_direct`` False, not a window piece) with phase_span in
[20 pi, 32 pi), on Part A's section 2.1 item 1 calls (``d:`` sids of the sets ``calls``, ``calls~o``, ``r1x`` and
``r1x~o``, as Part A's population took them). Each region's **fate** in the tree, from I6's records: ``returned`` (never
removed), else, by the last trust check of its seq: ``untrusted`` (trusted False, no rate rule), ``rate_rule`` (refused by
a rate rule), ``trusted_bisected`` (trusted, bisected by the budget), or ``guarded`` / ``unchecked_bisected`` where no
check was taken; its depth; and, if checked, the tree's check (delta1, delta2, r, nu_B, trusted) and its own declaration
at making (own_err = max(residual, floor) + truncation).
**The strata** (v2): ``band_20_32|<group|analytic>|<1e-7|1e-8>``, the returned leaves (fate ``returned``), as section
2.4 draws its bands; ``band_20_32v|...``, every other fate, the set "visited, not returned", kept apart. v1 mixed the two
(state/superseded/). The strata were (kind by "group " in notify_label, rtol the replay's).
**The order:** per stratum, its regions sorted by (sid, start), permuted by numpy.random.default_rng(BANDS2 + i), i the
stratum's index in sorted order (state/seeds_a2.json).
Writes out/leaves_a2/new_band_population.json and state/selection_new_band_a2.json (refuses to overwrite).
With ``--specs ROUND K``: writes the specs of the next K leaves of each stratum not yet specified (out/leaves_a2/specs/
<round>/), recording the prefix in state/new_band_fill.json.
Usage: python -B new_band_a2.py [--specs ROUND K]
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import PARTA, ROOT, header, sha256  # noqa: E402


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def population():
    man = {}
    for p in sorted((PARTA / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    pop = []
    for sid in sorted(man):
        if not sid.startswith("d:") or man[sid]["set"] not in ("calls", "calls~o", "r1x", "r1x~o"):
            continue
        d = json.loads((ROOT / "out" / "replays" / "I6" / f"{fname(sid)}.json").read_text())
        if not d["equals_tree"]:
            raise SystemExit(f"{sid}: I6 does not equal the tree")
        R = d["a2_records"]
        removed = {r[0]: r for r in R["removed"]}
        checks = {}
        for c in R["checks"]:
            checks[c[0]] = c  # the last check of each seq
        kind = "group" if "group " in d["notify_label"] else "analytic"
        rtol = "1e-7" if float.fromhex(d["rtol"]) == 1e-7 else "1e-8"
        for m in R["made"]:
            seq, a, b, depth, is_direct, ps, window, guarded, gdm = m[:9]
            if is_direct or window or ps is None:
                continue
            V = float.fromhex(ps)
            if not 20 * math.pi <= V < 32 * math.pi:
                continue
            c = checks.get(seq)
            if seq not in removed:
                fate = "returned"
            elif c is not None:
                fate = "rate_rule" if c[14] else ("untrusted" if not c[13] else "trusted_bisected")
            else:
                fate = "guarded" if guarded else "unchecked_bisected"
            pop.append({"sid": sid, "kind": kind, "rtol": rtol, "start": a, "end": b, "depth": depth, "V": V,
                        "V_over_pi": V / math.pi, "seq": seq, "guarded": guarded, "fate": fate,
                        "own_err_at_making": m[13], "value_own": m[9],
                        "tree_check": None if c is None else {"delta1": c[9], "delta2": c[10], "r": c[11], "nu_B": c[12],
                                                              "trusted": c[13], "refused": c[14], "rule": c[15],
                                                              "sub_direct": c[18]},
                        # v2: returned leaves are the band, as section 2.4 draws its bands; the regions the tree
                        # visited and bisected are a separate set, "visited, not returned" (band_20_32v)
                        "stratum": f"band_20_32{'' if fate == 'returned' else 'v'}|{kind}|{rtol}"})
    return pop, man


def main():
    if "--specs" in sys.argv:
        return specs(sys.argv[sys.argv.index("--specs") + 1], int(sys.argv[sys.argv.index("--specs") + 2]))
    op = ROOT / "out" / "leaves_a2" / "new_band_population.json"
    osel = ROOT / "state" / "selection_new_band_a2.json"
    for p in (op, osel):
        if p.exists():
            raise SystemExit(f"{p} exists")
    seeds = json.loads((ROOT / "state" / "seeds_a2.json").read_text())["seeds"]
    pop, _ = population()
    strata = defaultdict(list)
    for x in pop:
        strata[x["stratum"]].append(x)
    sel = {}
    for fam, sname in (("band_20_32|", "BANDS2"), ("band_20_32v|", "BANDS2V")):  # v2: two families, a seed each
        for i, st in enumerate(sorted(s_ for s_ in strata if s_.startswith(fam))):
            L = sorted(strata[st], key=lambda x: (x["sid"], float.fromhex(x["start"])))
            perm = np.random.default_rng(seeds[sname]["seed"] + i).permutation(len(L))
            sel[st] = {"seed": seeds[sname]["seed"] + i, "order": [L[j] for j in perm]}
    op.parent.mkdir(parents=True, exist_ok=True)
    hdr = header("levin-phase-floor 07d Part A': the new band [20 pi, 32 pi), visited, not returned", __file__,
                 [ROOT / "state" / "seeds_a2.json"])
    op.write_text(json.dumps({"header": hdr, "n": len(pop), "leaves": pop}))
    osel.write_text(json.dumps({"header": hdr, "counts": {k: len(v) for k, v in sorted(strata.items())},
                                "fates": {k: dict(sorted(((f, sum(1 for x in v if x["fate"] == f)) for f in
                                                          {x["fate"] for x in v}))) for k, v in sorted(strata.items())},
                                "selection": sel}, indent=1))
    print(json.dumps({k: len(v) for k, v in sorted(strata.items())}))


def specs(rnd, k):
    sel = json.loads((ROOT / "state" / "selection_new_band_a2.json").read_text())["selection"]
    fp = ROOT / "state" / "new_band_fill.json"
    fill = json.loads(fp.read_text()) if fp.exists() else {"specified": {}, "rounds": []}
    _, man = population()
    by_sid = defaultdict(list)
    added = {}
    for st, s in sorted(sel.items()):
        n0 = fill["specified"].get(st, 0)
        take = s["order"][n0:n0 + k]
        for j, x in enumerate(take):
            by_sid[x["sid"]].append({"id": f"{st}#{n0 + j}", "start": x["start"], "end": x["end"], "type": "Levin",
                                     "value_own": x["value_own"], "fate": {f: x[f] for f in ("fate", "depth", "seq",
                                     "guarded", "own_err_at_making", "tree_check", "V")}})
        fill["specified"][st] = n0 + len(take)
        added[st] = len(take)
    sd = ROOT / "out" / "leaves_a2" / "specs" / rnd
    sd.mkdir(parents=True, exist_ok=False)
    for sid, leaves in sorted(by_sid.items()):
        d = json.loads((ROOT / "out" / "replays" / "I6" / f"{fname(sid)}.json").read_text())
        c = {k: v for k, v in man[sid].items() if k in ("path", "offset", "length", "sha256")}
        atol = d["atol"]
        spec = {"sid": sid, "capture": c, "rtol": float.fromhex(d["rtol"]),
                "atol": float.fromhex(atol) if isinstance(atol, str) else atol, "call_value": d["replay"]["value"],
                "x_span": d["x_span"], "leaves": leaves}
        (sd / f"{fname(sid)}.json").write_text(json.dumps(spec))
    fill["rounds"].append({"round": rnd, "k": k, "added": added, "n_specs": len(by_sid)})
    fp.write_text(json.dumps(fill, indent=1))
    print(json.dumps(added), len(by_sid), "specs")


if __name__ == "__main__":
    main()
