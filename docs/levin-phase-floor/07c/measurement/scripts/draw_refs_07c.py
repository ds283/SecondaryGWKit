"""levin-phase-floor prompt 07c Part A, section 2.5: the reference draws R4 and R5, by a rule fixed here and written to
state/ before the draw. New in 07c.

**The unit drawn is an integrand** (shard, serial, kind, index): one reading scores every call of it, at both rtols.

**R5** (``draw_refs_07c.py r5``): the population is the 306 unreferenced calls of state/populations_07c.json
``moved_beyond_sum`` (moved by more than the sum of the two codes' abserrs, converged under both, not referenced by 07b's
U4 or U6). Each integrand's strata are its group label, its k (the notify label's ``k=``) and which code declares the
smaller abserr on its first call in rtol order 1e-7, 1e-8 (``07a`` or ``07b``). Every call of the 306 belonging to a drawn
integrand is scored.

**R4** (``draw_refs_07c.py r4 D1_DESCENT_JSON``): the population is the chase's 103 integrands (state/populations_07c.json
``chase``) less R2-28's (h-...-0-13757 #5, already referenced). Each integrand's strata are its group label, its k and the
refused ancestor's depth: the smallest depth of a rate-rule-refused ancestor of any of its depth-20 leaves, over its chase
calls, from Part A's own descent analysis (D1_DESCENT_JSON, written by d1_descent_07c.py), ``none`` where no depth-20
leaf descends from a refused leaf.

**The rule (both):** ``rng = numpy.random.default_rng(seed)`` with the seed of state/seeds_07c.json; ``order =
rng.permutation(n)`` over the integrands sorted by key. Pass 1 (coverage) walks the order and takes an integrand if it
brings a value of any stratum variable not yet taken. Pass 2 (fill) walks the order again and takes integrands not yet
taken until 30 are taken. The order is kept, so a cut would take a prefix of each pass.

Writes state/<draw>_rule.txt (this rule, before the draw) and state/selection_<draw>.json. Refuses to overwrite.
Usage: python -B draw_refs_07c.py r5 | r4 D1_DESCENT_JSON
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

N_DRAW = 30


def ikey(unit, kind, index):
    p = unit.split("-")
    return f"{p[-2]}-{p[-1]}-{kind[0]}{index}"


def kval(label):
    return re.search(r"k=([^/]+)/Mpc", label).group(1)


def glabel(label):
    return label.split("group ")[-1] if "group " in label else label


def draw(strata, seed):
    keys = sorted(strata)
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(len(keys))]
    taken, seen, passes = [], {}, {}
    for k in order:
        new = [v for v in strata[k].items() if v[1] not in seen.get(v[0], set())]
        if new:
            taken.append(k)
            passes[k] = 1
            for var, val in strata[k].items():
                seen.setdefault(var, set()).add(val)
    for k in order:
        if len(taken) >= N_DRAW:
            break
        if k not in passes:
            taken.append(k)
            passes[k] = 2
    return order, taken, passes


def main():
    which = sys.argv[1]
    rule_p = WIDER / "state" / f"{which}_rule.txt"
    sel_p = WIDER / "state" / f"selection_{which}.json"
    for p in (rule_p, sel_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    rule_p.write_text(__doc__)
    seeds_p = WIDER / "state" / "seeds_07c.json"
    seed = json.loads(seeds_p.read_text())["seeds"]["R5" if which == "r5" else "R4"]["seed"]
    pops_p = WIDER / "state" / "populations_07c.json"
    pops = json.loads(pops_p.read_text())
    inputs = [seeds_p, pops_p]
    calls_by_int = {}
    strata = {}
    if which == "r5":
        calls = [c for c in pops["moved_beyond_sum"]["calls"] if not c["referenced"]]
        if len(calls) != 306:
            raise SystemExit(f"{len(calls)} unreferenced calls, not 306")
        for c in sorted(calls, key=lambda c: (c["unit"].split("-")[-2:], c["rtol"] != "1e-7")):
            k = ikey(c["unit"], c["kind"], c["index"])
            calls_by_int.setdefault(k, []).append(c)
        for k, cs in calls_by_int.items():
            c0 = sorted(cs, key=lambda c: c["rtol"] != "1e-7")[0]
            strata[k] = {"label": glabel(c0["label"]), "k": kval(c0["label"]), "smaller": c0["smaller_abserr"]}
    elif which == "r4":
        d1_p = Path(sys.argv[2])
        inputs.append(d1_p)
        desc = json.loads(d1_p.read_text())["calls"]
        for c in pops["chase"]["calls"]:
            k = ikey(c["unit"], c["kind"], c["index"])
            calls_by_int.setdefault(k, []).append(c)
        calls_by_int.pop("0-13757-g5", None)  # R2-28's integrand, referenced already
        for k, cs in calls_by_int.items():
            depths = []
            for c in cs:
                p = c["unit"].split("-")
                sid = f"c-{'-'.join(p[1:-2])}-{p[-2]}-{p[-1]}-{c['index']}"
                d = desc[sid]
                if d.get("min_refused_ancestor_depth_of_depth20") is not None:
                    depths.append(d["min_refused_ancestor_depth_of_depth20"])
            strata[k] = {"label": glabel(cs[0]["label"]), "k": kval(cs[0]["label"]),
                         "refused_depth": str(min(depths)) if depths else "none"}
    else:
        raise SystemExit("r5 or r4")
    order, taken, passes = draw(strata, seed)
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07c Part A: the {which.upper()} draw", inputs + [rule_p])
    doc = {"header": hdr, "seed": seed, "rule_file": str(rule_p), "rule_sha256": sha256(rule_p),
           "population_integrands": len(strata), "population_calls": sum(len(v) for v in calls_by_int.values()),
           "order": order, "taken": taken, "pass": passes, "strata": {k: strata[k] for k in taken},
           "calls": {k: [{"unit": c["unit"], "rtol": c["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"]}
                         for c in calls_by_int[k]] for k in taken},
           "population_strata_counts": {var: dict(sorted(__import__("collections").Counter(s[var] for s in strata.values()).items()))
                                        for var in next(iter(strata.values()))},
           "taken_strata_counts": {var: dict(sorted(__import__("collections").Counter(strata[k][var] for k in taken).items()))
                                   for var in next(iter(strata.values()))}}
    sel_p.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print(f"{which}: population {len(strata)} integrands; taken {len(taken)} ({sum(1 for k in taken if passes[k] == 1)} in pass 1), "
          f"{sum(len(doc['calls'][k]) for k in taken)} calls")
    print("population strata:", doc["population_strata_counts"])
    print("taken strata:", doc["taken_strata_counts"])
    print(f"wrote {sel_p} sha256 {sha256(sel_p)}")


if __name__ == "__main__":
    main()
