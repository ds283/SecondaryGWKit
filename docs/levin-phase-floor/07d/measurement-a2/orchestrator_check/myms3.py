import json, os, sys, glob, math, hashlib
import numpy as np, mpmath as mp
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement-a2"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
import ray.cloudpickle as cp
from tools.filon_reference import production_edges
fx=float.fromhex
def capload(c):
    fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]; return cp.loads(b)
recs={}
for f in sorted(glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True)):
    if "/specs/" in f or "/logs/" in f or "population" in f: continue
    d=json.load(open(f))
    for L in d.get("leaves",[]):
        if "error" not in L: recs.setdefault(L["id"],(d["header"]["spec"],L))

print("== 5c: K's split points, from production_edges' interior edges, each window piece recursed from depth 0")
import checkrepro_a2 as CR
kr=json.load(open(R+"/out/replays/K/R_a_R1-x007-7.json")); rec=capload(kr["header"]["capture"])
e,_=production_edges(rec); kn=np.array(e[1:-1]); print("interior edges",len(kn),"supplied",kr["n_knots_supplied"])
kw=dict(rec["kw"]); kw["rtol"]=fx(kr["rtol"]); C=CR.CallContext(LQ,rec,kw)
lv=[(fx(l["start"]),fx(l["end"]),l["depth"]) for l in kr["leaves"]]
leafset={(s,e_):d for s,e_,d in lv}; bounds={s for s,_,_ in lv}|{e_ for _,e_,_ in lv}
A0,B0=fx(kr["x_span"][0]),fx(kr["x_span"][1])
init=sorted({A0,B0}|{w for w in C.window_edges if A0<w<B0}); print("initial pieces",len(init)-1)
def brk(a,b):
    lo,hi=min(a,b),max(a,b); q=0.25*(hi-lo)
    i0=int(np.searchsorted(kn,lo+q,side="left")); i1=int(np.searchsorted(kn,hi-q,side="right"))
    if i1>i0:
        ks=kn[i0:i1]; k=float(ks[int(np.argmin(np.abs(ks-0.5*(lo+hi))))])
        if lo<k<hi: return k,"knot"
    return (b+a)/2.0,"mid"
stats={"knot":0,"mid":0}; bad=[]
def rec_(a,b,d):
    if (a,b) in leafset:
        if leafset[(a,b)]!=d: bad.append(("depth",a,b,d,leafset[(a,b)]))
        return
    s,kind=brk(a,b)
    if s not in bounds: bad.append(("split not a leaf edge",a,b,s,kind)); return
    stats[kind]+=1; rec_(a,s,d+1); rec_(s,b,d+1)
for a,b in zip(init[:-1],init[1:]): rec_(a,b,0)
covered=sum(1 for _ in lv)
print("leaves",len(lv),"splits re-derived",stats,"problems",len(bad),bad[:4],"; replay's knot-break counter",kr["extra_a2"])
