import json, os, glob, re, collections
from fractions import Fraction as F
from decimal import Decimal
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
def fd(s): return F(Decimal(s))
def fh(h): return F(float.fromhex(h))
codes=["tree","Eg","Ea","ET20","ET12","Ep","T20"]
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
agent=json.load(open(R+"/out/tables/scores_07d.json"))["calls"]
t4={c:collections.Counter() for c in codes}; mism=0; nscored=0; nref=0; newdis=collections.defaultdict(list); shown=set()
for sid,c in sorted(man.items()):
    if sid.startswith("R:"):
        s7=c.get("census_07c_sid") or ("r31-3-20363-1" if sid=="R:r31" else None); r7=sc7.get(s7)
        if not r7: continue
        ref,spread,st=fd(r7["ref"]),fd(str(r7["spread"])),"regression"
    else:
        r=refs.get(sid[2:])
        if not r: continue
        ref,spread,st=r["ref"],r["spread"],r["set"]
    rep={}
    for cd in codes:
        p=f'{R}/out/replays/{cd}/{sid.replace(":","_").replace("|","_")}.json'
        if os.path.exists(p): rep[cd]=json.load(open(p))["replay"]
    if len(rep)<len(codes): continue
    nref+=1
    mina=min(fh(rep[cd]["abserr"]) for cd in codes)
    scored= spread<=mina/100
    ag=agent.get(sid)
    if ag is None or ag["scored"]!=scored: mism+=1
    if not scored: continue
    nscored+=1
    for cd in codes:
        v,a,cv=fh(rep[cd]["value"]),fh(rep[cd]["abserr"]),rep[cd]["converged"]
        ratio=abs(v-ref)/a
        vd="unconverged" if not cv else ("honest" if ratio<=1 else "dishonest")
        t4[cd][vd]+=1
        if ag and ag["by"][cd]["verdict"]!=vd: mism+=1
        if cd!="tree" and vd=="dishonest": newdis[cd].append(sid)
    if st not in shown and st in ("rt","ra","regression"):
        shown.add(st); cd="tree"; v,a=fh(rep[cd]["value"]),fh(rep[cd]["abserr"])
        print(f"[{st}] {sid}: ref {float(ref):.17g} spread {float(spread):.3g} min abserr/100 {float(mina/100):.3g}; tree |v-ref|/abserr = {float(abs(v-ref)/a):.6g} exactly {abs(v-ref)/a if len(str(abs(v-ref)/a))<80 else '(long fraction)'}; agent {ag['by'][cd]['ratio']:.6g}")
print("calls referenced",nref,"scored",nscored,"; disagreements with the agent (scored flag or verdict):",mism)
for cd in codes: print(f"  {cd:5s}", dict(t4[cd]))
treedis={s for s in man if False}
print("dishonest under a variant (sids):",{k:v for k,v in newdis.items()})
