import json, os, collections, numpy as np
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
seeds=json.load(open(R+"/state/seeds_07d.json"))["seeds"]
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
def strat(p):
    k=f'{p["kind"]}|{p["rtol"]}'; v=p["V_over_6pi"]
    if p["type"]=="direct":
        if p["window"]: return "window|"+k
        if p["chase_stopped"]: return "chase|"+k
        if p.get("guarded"): return "guarded|"+k
        for q,b in enumerate([0.25,0.5,0.75,1.0]):
            if v<b: return f"gate_q{q+1}|"+k
        return "gate_above|"+k
    V=v*6
    for lo,hi in [(6,8),(8,10),(10,12),(12,16),(16,20)]:
        if lo<=V<hi: return f"band_{lo}_{hi}|"+k
    return None
mine={}; bad_strat=0
for r_ext,fn in [(0,"population.json"),(1,"population_r1x.json")]:
    pop=json.load(open(R+"/out/leaves/"+fn))["leaves"]
    for p in pop:
        if strat(p)!=p["stratum"]: bad_strat+=1
    for sn in ("LEAVES","BANDS"):
        sts=sorted({p["stratum"] for p in pop if p["seed_name"]==sn})
        for i,st in enumerate(sts):
            m=sorted((p for p in pop if p["stratum"]==st),key=lambda p:(p["sid"],p["start"]))
            o=np.random.default_rng(seeds[sn]["seed"]+100*r_ext+i).permutation(len(m))
            mine.setdefault(st,[]).extend((m[j]["sid"],m[j]["start"]) for j in o)
print("population leaves whose stratum label I derive differently:",bad_strat)
same=sum(1 for st in sel if [(x["sid"],x["start"]) for x in sel[st]["order"]]==mine.get(st)); print("strata whose order I reproduce:",same,"of",len(sel), "mine",len(mine))
de=json.load(open(R+"/out/tables/d_e.json"))
byst=collections.defaultdict(list)
for L in de["leaves"]: byst[L["stratum"]].append(L)
cands=de["candidates"]; under=collections.Counter(); und=collections.Counter(); verdict_mismatch=0
for st,Ls in sorted(byst.items()):
    Ls.sort(key=lambda L:L["rank"]); ranks=[L["rank"] for L in Ls]
    prefix = ranks==list(range(len(Ls)))
    order=mine[st]; starts_ok=all(order[L["rank"]][1]==L["start"] for L in Ls)
    nsc=sum(L["scored"] for L in Ls)
    print(f"{st:28s} drawn {len(Ls):4d} scored {nsc:3d} pop {len(order):5d} prefix {prefix} order-match {starts_ok}")
    for L in Ls:
        if not L["scored"]: continue
        t,e=L["true_err"],L["eps_ref"]
        for c in cands:
            D=L["D"].get(c)
            if D is None: continue
            v="bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
            if v!=L["verdicts"].get(c): verdict_mismatch+=1
            if v=="understates": under[c]+=1
            if v=="undecided": und[c]+=1
print("verdicts differing from the agent's under my rule:",verdict_mismatch)
print("understates:",dict(under)); print("undecided:",dict(und))
