import json, os, glob, re, collections
from fractions import Fraction as F
from decimal import Decimal
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
def fd(s): return F(Decimal(s))
def fh(h): return F(float.fromhex(h))
codes=["tree","V93","V93tail","V93T8","V93T10","V93T12","K","G","KT"]
RA2=M+"/var/levin-phase-floor-07d/measurement-a2"
refs={}
keys=collections.defaultdict(list)
for f in glob.glob(R+"/out/refs/readings/*.json"):
    b=os.path.basename(f); m=re.match(r"(r[at]-h-1e-[78]-\d-\d+-[ag]\d+)-(.*)\.json",b); keys[m.group(1)].append(f)
def oth(s):
    u,rest=s.split("|"); p=u.split("-"); return "-".join([p[0],"1e-8" if p[1]+"-"+p[2]=="1e-7" else "1e-7",p[3],p[4]])+"|"+rest
for k,fs in keys.items():
    vals={}
    for f in fs:
        d=json.load(open(f))
        if "readings" in d:
            for res,r in d["readings"].items(): vals[f"fccsplit-r{res}"]=sum((fd(x["value_30"]) for x in r["pieces"]),F(0))
        elif d.get("value_30") is not None: vals[f"{d['impl']}-r{d['res']}"]=fd(d["value_30"])
    rk="fccsplit-r2" if "fccsplit-r2" in vals else ("fcc-r2" if "fcc-r2" in vals else None)
    if rk is None: continue
    p=k[3:].split("-"); sid="-".join(p[:5])+"|"+p[5]
    v=list(vals.values()); r={"ref":vals[rk],"spread":max(v)-min(v),"set":k[:2],"basis":sorted(vals)}
    refs[sid]=r; refs[oth(sid)]=r
sc7=json.load(open(M+"/var/levin-phase-floor-07c/measurement/out/tables/scores_07c.json"))["scored"]
man={}
for p in glob.glob(R+"/state/calls_07d*.json"): man.update(json.load(open(p))["calls"])
agentA=json.load(open(R+"/out/tables/scores_07d.json"))["calls"]

import sys
t7=collections.defaultdict(collections.Counter); nscored=0; nref=0; newdis=collections.defaultdict(list); sA=0; diffA=0
for sid,c in sorted(man.items()):
    if sid.startswith("R:"):
        s7=c.get("census_07c_sid") or ("r31-3-20363-1" if sid=="R:r31" else None); r7=sc7.get(s7)
        if not r7: continue
        ref,spread=fd(r7["ref"]),fd(str(r7["spread"]))
    else:
        r=refs.get(sid[2:])
        if not r: continue
        ref,spread=r["ref"],r["spread"]
    rep={}
    fn=sid.replace(":","_").replace("|","_")
    for cd in codes:
        p=(f"{R}/out/replays/tree/{fn}.json" if cd=="tree" else f"{RA2}/out/replays/{cd}/{fn}.json")
        if os.path.exists(p): rep[cd]=json.load(open(p))["replay"]
    if len(rep)<len(codes): print("missing",sid,set(codes)-set(rep)); continue
    nref+=1
    ag=agentA.get(sid); scored=bool(ag and ag["scored"])   # Part A's scored set (its rule, its codes)
    if not scored: continue
    nscored+=1
    vt=None
    for cd in codes:
        v,a,cv=fh(rep[cd]["value"]),fh(rep[cd]["abserr"]),rep[cd]["converged"]
        ratio=abs(v-ref)/a
        vd="unconverged" if not cv else ("honest" if ratio<=1 else "dishonest")
        t7[cd][vd]+=1
        if cd=="tree": vt=vd
        elif vd=="dishonest" and vt!="dishonest": newdis[cd].append((sid,float(ratio)))
print("referenced",nref,"scored (Part A's set)",nscored)
for cd in codes: print(f"  {cd:8s}",dict(t7[cd]),"new converged-dishonest",newdis.get(cd,[]))
