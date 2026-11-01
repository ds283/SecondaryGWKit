"""levin-phase-floor prompt 06f section 1: the population and the seeded samples M1 and M2. New in 06f.

Reads, read only: 06e's census checkpoint (prov.RUN06E, SHA-256 checked against
prov.CHECKPOINT06E_SHA256) and 06a's census checkpoint's 1e-7 lines (the 720 stationary-point rows,
as 06e's analyse06e.py's scan06a_sp reads them).

The population (prompt section 1): every call of every unit (group calls under "calls", analytic
blocks under "other_calls") whose ``cc_floor`` has ``n_floor_attributed > 0``. A call's largest floor
term is taken among t_value_rounding ((n+4)uS), t_argument, node_term and declared, from
``sums_floor_attributed_cc_leaves``, with max() keeping the first of equals in that order, exactly as
analyse06e.py counts it (its "per call, largest floor term" line); the counts are printed beside
06e's (491 / 1,323 node-dominated, 2,493 / 2,785 declared-dominated at 1e-7 / 1e-8) as a check.
A call is eligible for a sample only if it lies outside the 720, and its census re-solve reproduced
every Clenshaw-Curtis leaf (``n_reproduced == n_cc_leaves``, ``n_errors == 0``), so that F0's term
check has a census sum to meet.

The samples, drawn with random.Random(SEED), in this order:
- M1: Z2's call (g-1e-8-0-33995 call 5), taken by name; then at each rtol (1e-7, then 1e-8), from
  the node-dominated eligible calls other than Z2's: 5 drawn from those with n_cc_leaves >= 100 (all
  of them if fewer than 5), then the rest of the 10 drawn from every remaining one. The remaining
  calls of each stratum, in a seeded order, are the replacements.
- M2: at each rtol, from the declared-dominated eligible calls: the analytic blocks that have any
  ("analytic J1" ... "analytic Y4") in a seeded order, one call drawn from each in turn, round and
  round, until 10. The remaining calls of each block, in a seeded order, are the replacements.

Writes OUT_DIR/sample.json (the samples, the replacement orders, the counts) and OUT_DIR/selection.json
(x1_capture.py's input: one entry per sampled call). Usage: python -B draw_sample_06f.py OUT_DIR
"""

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06E_SHA256, RUN06A, RUN06E, header, sha256  # noqa: E402

SEED = 20261031
RTOLS = ("1e-7", "1e-8")
TERMS = ("t_value_rounding", "t_argument", "node_term", "declared")
Z2 = ("g-1e-8-0-33995", 5)
N_PER_RTOL = 10
N_BIG = 5
BIG_CC = 100


def H(x):
    return float.fromhex(x["hex"] if isinstance(x, dict) else x)


def scan06a_sp():
    """analyse06e.py's scan06a_sp, unchanged."""
    sp = set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    return sp


def main():
    out_dir = Path(sys.argv[1])
    ck = RUN06E / "checkpoint.jsonl"
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06f: the population and the seeded samples",
        [ck, RUN06A / "checkpoint.jsonl"],
    ):
        print(l)
    if sha256(ck) != CHECKPOINT06E_SHA256:
        raise SystemExit("06e's census checkpoint is not a0dc4d7f...")
    print(f"# seed {SEED}")
    sp = scan06a_sp()
    print(f"06a's stationary-point rows: {len(sp)} (must be 720: {len(sp) == 720})")
    if len(sp) != 720:
        raise SystemExit("not 720")
    calls = []
    counts_06e = {r: collections.Counter() for r in RTOLS}
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            if d["status"] != "done":
                continue
            row = f"{d['shard']}-{d['serial']}"
            for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                for c in d.get(key) or []:
                    fl = c.get("cc_floor")
                    if (
                        not fl
                        or "sums_floor_attributed_cc_leaves" not in fl
                        or fl["n_floor_attributed"] <= 0
                    ):
                        continue
                    sums = {
                        t: H(fl["sums_floor_attributed_cc_leaves"][t]) for t in TERMS
                    }
                    big = max(
                        TERMS, key=lambda t: sums[t]
                    )  # first of equals, as analyse06e.py
                    counts_06e[d["rtol"]][big] += 1
                    calls.append(
                        {
                            "unit": r["unit"],
                            "rtol": d["rtol"],
                            "row": row,
                            "kind": kind,
                            "index": c["index"],
                            "notify_label": c["notify_label"],
                            "largest": big,
                            "n_cc_leaves": fl["n_cc_leaves"],
                            "n_floor_attributed": fl["n_floor_attributed"],
                            "reproduced_all": fl["n_reproduced"] == fl["n_cc_leaves"]
                            and fl["n_errors"] == 0,
                            "stationary_point_row": row in sp,
                            "census_value": c["value"]["hex"],
                            "census_abserr": c["abserr"]["hex"],
                            "census_num_regions": c["num_regions"],
                            "census_max_depth": c["max_depth"],
                            "census_converged": c["converged"],
                            "sums_floor_attributed": {
                                t: fl["sums_floor_attributed_cc_leaves"][t]["hex"]
                                for t in TERMS
                            },
                        }
                    )
    print(
        "per call, largest floor term among (n+4)uS, argument, node, declared (06e's report: node 491 / 1323, "
        "declared 2493 / 2785):"
    )
    for rt in RTOLS:
        print(f"   {rt}: {dict(counts_06e[rt])}")
    rng = random.Random(SEED)
    sample = {"seed": SEED, "M1": [], "M2": [], "replacements": {}, "strata": {}}
    # Z2's call, by name
    z2 = [c for c in calls if (c["unit"], c["index"]) == Z2]
    if len(z2) != 1:
        raise SystemExit(f"Z2's call found {len(z2)} times in the population")
    z2 = z2[0]
    print(
        f"Z2's call: largest term {z2['largest']}, n_cc_leaves {z2['n_cc_leaves']}, reproduced_all {z2['reproduced_all']}, "
        f"in the 720 {z2['stationary_point_row']}"
    )
    sample["M1"].append(dict(z2, sample="M1", stratum="Z2"))

    def elig(c):
        return (
            c["reproduced_all"]
            and not c["stationary_point_row"]
            and (c["unit"], c["index"]) != Z2
        )

    for rt in RTOLS:
        node = [
            c
            for c in calls
            if c["rtol"] == rt and c["largest"] == "node_term" and elig(c)
        ]
        node.sort(key=lambda c: (c["unit"], c["index"]))
        big = [c for c in node if c["n_cc_leaves"] >= BIG_CC]
        print(
            f"M1 {rt}: node-dominated eligible {len(node)}, of them with n_cc_leaves >= {BIG_CC}: {len(big)}; "
            f"ineligible node-dominated (in the 720 or not reproduced): "
            f"{sum(1 for c in calls if c['rtol'] == rt and c['largest'] == 'node_term') - len(node) - (1 if (rt == '1e-8' and z2['largest'] == 'node_term') else 0)}"
        )
        sample["strata"][f"M1 {rt}"] = {"eligible": len(node), "n_cc_ge_100": len(big)}
        chosen = rng.sample(big, min(N_BIG, len(big)))
        rest = [c for c in node if c not in chosen]
        chosen += rng.sample(rest, N_PER_RTOL - len(chosen))
        for c in chosen:
            sample["M1"].append(
                dict(
                    c,
                    sample="M1",
                    stratum=f"M1 {rt} "
                    + ("n_cc>=100" if c["n_cc_leaves"] >= BIG_CC else "n_cc<100"),
                )
            )
        left = [c for c in node if c not in chosen]
        rng.shuffle(left)
        sample["replacements"][f"M1 {rt} n_cc>=100"] = [
            (c["unit"], c["index"]) for c in left if c["n_cc_leaves"] >= BIG_CC
        ]
        sample["replacements"][f"M1 {rt} any"] = [(c["unit"], c["index"]) for c in left]
    for rt in RTOLS:
        decl = [
            c
            for c in calls
            if c["rtol"] == rt and c["largest"] == "declared" and elig(c)
        ]
        decl.sort(key=lambda c: (c["unit"], c["index"]))
        kinds = collections.Counter(c["kind"] for c in decl)
        blocks = collections.defaultdict(list)
        for c in decl:
            blocks[c["notify_label"]].append(c)
        names = sorted(blocks)
        print(
            f"M2 {rt}: declared-dominated eligible {len(decl)} by kind {dict(kinds)}; by block "
            + ", ".join(f"{n}: {len(blocks[n])}" for n in names)
        )
        sample["strata"][f"M2 {rt}"] = {
            "eligible": len(decl),
            "by_kind": dict(kinds),
            "by_block": {n: len(blocks[n]) for n in names},
        }
        order = list(names)
        rng.shuffle(order)
        pools = {}
        for n in order:
            p = list(blocks[n])
            rng.shuffle(p)
            pools[n] = p
        chosen = []
        while len(chosen) < N_PER_RTOL and any(pools.values()):
            for n in order:
                if pools[n] and len(chosen) < N_PER_RTOL:
                    chosen.append(pools[n].pop(0))
        for c in chosen:
            sample["M2"].append(
                dict(c, sample="M2", stratum=f"M2 {rt} {c['notify_label']}")
            )
        for n in order:
            sample["replacements"][f"M2 {rt} {n}"] = [
                (c["unit"], c["index"]) for c in pools[n]
            ]
    print("\nM1:")
    for c in sample["M1"]:
        print(
            f"   {c['unit']:>18s} #{c['index']:<2d} {c['kind']:8s} n_cc {c['n_cc_leaves']:5d} depth {c['census_max_depth']:2d} "
            f"conv {c['census_converged']} | {c['stratum']} | {c['notify_label'][:70]}"
        )
    print("M2:")
    for c in sample["M2"]:
        print(
            f"   {c['unit']:>18s} #{c['index']:<2d} {c['kind']:8s} n_cc {c['n_cc_leaves']:5d} depth {c['census_max_depth']:2d} "
            f"conv {c['census_converged']} | {c['stratum']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.json").write_text(json.dumps(sample, indent=1))
    sel = [
        {
            k: c[k]
            for k in (
                "unit",
                "index",
                "notify_label",
                "census_value",
                "census_abserr",
                "census_num_regions",
                "census_max_depth",
                "census_converged",
                "sample",
                "stratum",
                "kind",
                "n_cc_leaves",
            )
        }
        for c in sample["M1"] + sample["M2"]
    ]
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"\nwrote {out_dir / 'sample.json'} sha256 {sha256(out_dir / 'sample.json')}")
    print(
        f"wrote {out_dir / 'selection.json'} sha256 {sha256(out_dir / 'selection.json')} ({len(sel)} calls in "
        f"{len({e['unit'] for e in sel})} units)"
    )


if __name__ == "__main__":
    main()
