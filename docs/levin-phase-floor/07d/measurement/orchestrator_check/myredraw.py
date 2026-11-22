import json, os, pickle, numpy as np, collections
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
seeds=json.load(open(R+"/state/seeds_07d.json"))["seeds"]
def draw(strata,seed,BASE,TOTAL):
    orders={}
    for i,st in enumerate(sorted(strata)):
        m=sorted(strata[st]); orders[st]=[m[j] for j in np.random.default_rng(seed+i).permutation(len(m))]
    taken={st:min(BASE,len(o)) for st,o in orders.items()}
    while sum(taken.values())<TOTAL and any(taken[s]<len(orders[s]) for s in orders):
        for st in sorted(orders):
            if sum(taken.values())>=TOTAL: break
            if taken[st]<len(orders[st]): taken[st]+=1
    return {st:len(o) for st,o in orders.items()}, [s for st in sorted(orders) for s in orders[st][:taken[st]]]
# (Ra) from 07c's census checkpoint itself
st=collections.defaultdict(list)
for line in open(M+"/var/runs/levin-phase-floor-07c-census-20261119T201419/checkpoint.jsonl"):
    r=json.loads(line); d=r["data"]
    for c in d.get("other_calls") or []:
        if not c.get("num_chase_stops"): continue
        car=(c.get("chase_07c") or {}).get("carrier")
        if not car or car.get("type") is None: s="nocarrier"
        else:
            tb=car.get("trusted_before"); t="T" if tb is True else ("U" if tb is False else "N")
            sh=car.get("share"); sh=float.fromhex(sh["hex"]) if isinstance(sh,dict) else sh; s=f"{car['type']}|{t}|"+("s?" if sh is None else ("s1" if sh<0.1 else ("s2" if sh<0.5 else "s3")))
        st[s].append(f'{r["unit"]}|a{c["index"]}')
pop,sel=draw(st,seeds["RA"]["seed"],2,30)
A=json.load(open(R+"/state/selection_ra_07d.json"))
print("(Ra) population same:",pop==A["population"],"; selection same:",sel==[e["sid"] for e in A["selection"]], len(sel))
if pop!=A["population"]: print(" mine",pop,"\n agent",A["population"])
# (Rt) from the variants' replay outputs
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
rows={r["sid"]:r for r in pickle.load(open(R+"/state/census_index_07d.pkl","rb"))["rows"]}
by=collections.defaultdict(list)
for k,c in man.items():
    if k.startswith("d:"): by[c["sha256"]].append(k)
ra={c["sha256"] for c in man.values() if c["set"] in ("ra","ra~o")}
st=collections.defaultdict(list); nra=0
for h,ks in by.items():
    ch=any(json.load(open(f'{R}/out/replays/{v}/{k.replace(":","_").replace("|","_")}.json'))["extra_07d"]["num_07d_route_changed"]>0 for k in ks for v in ("ET20","ET12","T20"))
    if not ch: continue
    if h in ra: nra+=1; continue
    prim=[k for k in ks if not man[k]["set"].endswith("~o")][0]; r=rows[man[prim]["census"]]
    st[f"Levin->CC|{r['kind']}|k{r['k_band']}|{r['rtol']}"].append(prim[2:])
pop,sel=draw(st,seeds["RT"]["seed"],1,30)
T=json.load(open(R+"/state/selection_rt_07d.json"))
print("(Rt) population same:",pop==T["population"],"; selection same:",sel==[e["sid"] for e in T["selection"]],len(sel),"; (Ra) left out",nra,"agent",T["ra_integrands_route_changed_left_out"])
