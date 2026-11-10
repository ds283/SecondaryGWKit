"""Orchestrator's own re-count of the census agent's cross-check (levin-phase-floor 07 Part B, beyond §6). Read only.

Usage: nice -n 19 python -I -B xcheck_recount.py OUT_JSON

Independent of the agent's xcheck_07.py: reads 07's census checkpoint whole and 06m's lines at the census agent's
index (each line checked by its SHA-256 and unit name), both files hashed before and after. Implements the rule as
fixed in census/state/xcheck_rule_20261109.md (e06a1058…): population = 07 calls with guard_07.locates or .guards at
either rtol; pairing by row, kind, index, notify_label and x_span hex; exact Fraction arithmetic; listed iff
|v_g - v_m| > max(e_g, e_m); classes (a)-(e). Then compares its list, classes and r_g > 1 pairs with the agent's
out/xcheck/xcheck.json, call by call.
"""
import hashlib, json, sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

ROOT = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
C07 = ROOT / "var/runs/levin-phase-floor-07-census-20261108T171137/checkpoint.jsonl"
C06 = ROOT / "var/runs/levin-phase-floor-06m-census-20261106T102153/checkpoint.jsonl"
IDX = ROOT / "var/levin-phase-floor-07/census/state/index_06m.json"
AG = ROOT / "var/levin-phase-floor-07/census/out/xcheck/xcheck.json"
EXP = {C07: "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10",
       C06: "ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fx(d):
    return F(float.fromhex(d["hex"]))


def ratio(n, d):
    if d == 0:
        return 0.0 if n == 0 else float("inf")
    return float(n / d)


def main():
    out = Path(sys.argv[1])
    before = {str(p): sha(p) for p in EXP}
    for p, e in EXP.items():
        assert before[str(p)] == e, (p, before[str(p)])
    idx = json.load(open(IDX))["units"]
    calls07 = {}   # (rtol,row,kind,index) -> (call, unit)
    pop_keys = set()
    for line in open(C07):
        d = json.loads(line)["data"]
        for kind in ("calls", "other_calls"):
            for c in d[kind]:
                k = (d["rtol"], d["row"], kind, c["index"])
                calls07[k] = (c, d["unit"])
                g = c.get("guard_07") or {}
                if g.get("locates") or g.get("guards"):
                    pop_keys.add((d["row"], kind, c["index"]))
    # 06m records for every row in the population, both rtols
    rows_needed = {r for r, _, _ in pop_keys}
    calls06 = {}
    with open(C06, "rb") as f:
        for rt in ("1e-7", "1e-8"):
            for r in rows_needed:
                u = f"g-{rt}-{r}"
                e = idx[u]
                f.seek(e["offset"]); raw = f.read(e["length"])
                assert hashlib.sha256(raw).hexdigest() == e["line_sha256"]
                d = json.loads(raw)["data"]
                assert d["unit"] == u
                for kind in ("calls", "other_calls"):
                    for c in d[kind]:
                        calls06[(rt, r, kind, c["index"])] = c

    def ident(c):
        return (c["notify_label"], tuple(x["hex"] for x in c["x_span"]))

    pairs, listed, unpaired = {}, [], []
    for (r, kind, i) in sorted(pop_keys):
        ks = {rt: (rt, r, kind, i) for rt in ("1e-7", "1e-8")}
        if not all(k in calls07 and k in calls06 for k in ks.values()):
            unpaired.append((r, kind, i)); continue
        g7, g8 = calls07[ks["1e-7"]][0], calls07[ks["1e-8"]][0]
        m7, m8 = calls06[ks["1e-7"]], calls06[ks["1e-8"]]
        # labels differ by rtol only if they embed rtol; compare label+span across codes at each rtol, and span across rtols
        ok = ident(g7) == ident(m7) and ident(g8) == ident(m8) and ident(g7)[1] == ident(g8)[1]
        if not ok:
            unpaired.append((r, kind, i, "ident")); continue
        vg7, vg8, eg7, eg8 = fx(g7["value"]), fx(g8["value"]), fx(g7["abserr"]), fx(g8["abserr"])
        vm7, vm8, em7, em8 = fx(m7["value"]), fx(m8["value"]), fx(m7["abserr"]), fx(m8["abserr"])
        rg = ratio(abs(vg7 - vg8), eg7); rm = ratio(abs(vm7 - vm8), em7)
        d8 = abs(vg8 - vm8)
        agree8 = d8 <= max(eg8, em8)
        sg, sm = rg <= 1, rm <= 1
        if sg and not sm and agree8: cls = "a"
        elif sm and not sg and agree8: cls = "b"
        elif sg and sm: cls = "c"
        elif not sg and not sm: cls = "d"
        else: cls = "e"
        pairs[(r, kind, i)] = dict(rg=rg, rm=rm, cls=cls, group=g7.get("group") or g7["notify_label"].split("group ")[-1])
        for rt, vg, vm, eg, em in (("1e-7", vg7, vm7, eg7, em7), ("1e-8", vg8, vm8, eg8, em8)):
            if abs(vg - vm) > max(eg, em):
                listed.append(dict(row=r, kind=kind, index=i, rtol=rt, cls=cls, unit=f"g-{rt}-{r}", group=pairs[(r, kind, i)]["group"]))
    after = {str(p): sha(p) for p in EXP}
    by_cls = Counter((x["cls"], x["rtol"]) for x in listed)
    by_group = Counter(x["group"] for x in listed)
    rg_gt1 = sorted((r, kind, i, p["rg"], p["rm"], p["cls"]) for (r, kind, i), p in pairs.items() if p["rg"] > 1)
    n_rg_le1 = sum(p["rg"] <= 1 for p in pairs.values()); n_rm_le1 = sum(p["rm"] <= 1 for p in pairs.values())
    # compare with the agent's
    ag = json.load(open(AG))
    agB = ag["B"]["calls"] if isinstance(ag.get("B"), dict) else ag["B"]
    def akey(x):
        return (x["unit"], int(x["index"]), x.get("rtol"))
    mine_set = {(x["unit"], x["index"], x["rtol"]): x["cls"] for x in listed}
    ag_set = {}
    for x in agB:
        cls = x.get("class") or x.get("cls")
        ag_set[(x["unit"], int(x["index"]), x["rtol"])] = str(cls).strip("()")
    only_mine = sorted(set(mine_set) - set(ag_set)); only_ag = sorted(set(ag_set) - set(mine_set))
    cls_diff = sorted(k for k in set(mine_set) & set(ag_set) if mine_set[k] != ag_set[k])
    res = dict(
        script_sha256=sha(Path(__file__)), inputs_before=before, inputs_after=after,
        population_calls=2 * len(pop_keys), pairs=len(pairs), unpaired=unpaired,
        n_rg_le1=n_rg_le1, n_rm_le1=n_rm_le1, listed=len(listed),
        by_class_rtol={f"{c} {r}": n for (c, r), n in sorted(by_cls.items())}, by_group=dict(by_group),
        rg_gt1=rg_gt1, agent_listed=len(ag_set), only_mine=only_mine[:50], only_agent=only_ag[:50],
        class_differs=cls_diff[:50], n_only_mine=len(only_mine), n_only_agent=len(only_ag), n_class_differs=len(cls_diff),
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("inputs_before", "inputs_after")}, indent=1, default=str))
    print("inputs unchanged:", before == after)


if __name__ == "__main__":
    main()
