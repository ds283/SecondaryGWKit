"""Orchestrator's own flag-cost table (prompt 06m §5, descriptive) from a census checkpoint.

Usage: python -I -B flag_cost_orchestrator.py CHECKPOINT [CHECKPOINT ...]

For every unconverged call (`converged` false) in each unit record, group calls (`calls`) and
analytic calls (`other_calls`) apart, at each rtol, outside and inside 06a's 720 stationary-point
rows: its abserr against its request max(atol, rtol*|value|) (compared in exact rational
arithmetic), whether it is node_limited, and its share abserr / the unit's result.total_abserr.
06a's 720 are found by 06a's rule as 06g's rehearsal tabulator states it: a row whose 1e-7 record
in 06a's census checkpoint has any scan entry with n_sign_changes > 0; rows are named
"<shard>-<serial>", the census unit's `row`.
"""

import hashlib
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

CK06A = Path(
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
)


def sp_rows():
    sp = set()
    for line in open(CK06A):
        if '"rtol": "1e-7"' not in line:
            continue
        d = json.loads(line)["data"]
        if d["rtol"] != "1e-7":
            continue
        if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
            sp.add(f"{d['shard']}-{d['serial']}")
    assert len(sp) == 720, len(sp)
    return sp


def fx(x):
    return float.fromhex(x["hex"]) if isinstance(x, dict) else float(x)


def main():
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"# flag_cost_orchestrator.py sha256 {me}; 06a checkpoint {CK06A}")
    sp = sp_rows()
    for ck in sys.argv[1:]:
        print(f"\n## {ck}")
        agg = defaultdict(lambda: dict(n=0, units=set(), within=0, node=0, worst=0.0, share=0.0, notdone=0))
        n_units = 0
        status = defaultdict(int)
        for line in open(ck):
            d = json.loads(line)["data"]
            n_units += 1
            status[d["status"]] += 1
            if d["status"] != "done":
                continue
            tot = fx(d["result"]["total_abserr"])
            where = "in720" if d["row"] in sp else "outside"
            for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                for c in d.get(key) or []:
                    if c["converged"]:
                        continue
                    a = agg[(d["rtol"], where, kind)]
                    a["n"] += 1
                    a["units"].add(d["unit"])
                    ab = fx(c["abserr"])
                    req = max(Fraction(c["atol"]), Fraction(c["rtol"]) * abs(Fraction(fx(c["value"]))))
                    if Fraction(ab) <= req:
                        a["within"] += 1
                    a["worst"] = max(a["worst"], float(Fraction(ab) / req) if req else float("inf"))
                    if c.get("node_limited"):
                        a["node"] += 1
                    if tot > 0:
                        a["share"] = max(a["share"], ab / tot)
        print(f"units {n_units}; status {dict(status)}")
        print("rtol  rows     kind      unconverged  units  within-request  node_limited  max abserr/request  max share")
        for k in sorted(agg):
            a = agg[k]
            print(
                f"{k[0]}  {k[1]:8s} {k[2]:9s} {a['n']:11d}  {len(a['units']):5d}  {a['within']:14d}  {a['node']:12d}"
                f"  {a['worst']:18.4g}  {a['share']:.4g}"
            )
        for where in ("outside", "in720"):
            for kind in ("group", "analytic"):
                ks = [k for k in agg if k[1] == where and k[2] == kind]
                n = sum(agg[k]["n"] for k in ks)
                u = set().union(*[agg[k]["units"] for k in ks]) if ks else set()
                w = sum(agg[k]["within"] for k in ks)
                nl = sum(agg[k]["node"] for k in ks)
                wo = max([agg[k]["worst"] for k in ks], default=0)
                sh = max([agg[k]["share"] for k in ks], default=0)
                print(
                    f"both  {where:8s} {kind:9s} {n:11d}  {len(u):5d}  {w:14d}  {nl:12d}  {wo:18.4g}  {sh:.4g}"
                )


if __name__ == "__main__":
    main()
