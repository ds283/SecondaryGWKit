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
mp.mp.dps=50
print("== 4b again: exact DCT-I of the same 93 doubles (mpmath, 50 digits)")
for lid in ("band_10_12|group|1e-7#0","band_10_12|analytic|1e-7#0","band_12_16|analytic|1e-7#0"):
    specp,L=recs[lid]; rec=capload(json.load(open(specp))["capture"]); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    a,b=fx(L["start"]),fx(L["end"]); n=93
    grid,_=LQ.chebyshev_matrices((a,b),n)
    fs=np.vstack([LQ._sample_vectorized(fn,grid,cache,("f",id(fn))) for fn in rec["f"]])
    bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in grid]).T; g=np.sum(fs*bs,axis=0)
    gj=[mp.mpf(float(v)) for v in g[::-1]]
    ck=[]
    for k in range(n):
        s=mp.fsum((mp.mpf(0.5) if j in (0,n-1) else 1)*gj[j]*mp.cos(mp.pi*j*k/(n-1)) for j in range(n))*2/(n-1)
        if k in (0,n-1): s/=2
        ck.append(float(s))
    ck=np.array(ck); cut=L["cc"]["c93"]["chop"]; tail=ck[cut:] if cut<n else ck[-8:]
    w=LQ._cc_weights((a,b),n)
    s=float(np.sqrt(np.mean(tail*tail))*math.sqrt((n-1)/2.0))*math.sqrt(float(np.sum(w*w)))
    print(f"{lid}: cut {cut}: exact s_own93 {s:.6g}, record {L['cc']['c93']['s_own']:.6g} (ratio {s/L['cc']['c93']['s_own']:.3f}); |CC93| {abs(fx(L['cc']['cc93'])):.3g}; v4 eps {fx(L['v4']['eps_ref']):.3g}")
print("== 5c again: K's split points from production_edges' interior edges")
kr=json.load(open(R+"/out/replays/K/R_a_R1-x007-7.json")); rec=capload(kr["header"]["capture"])
e,_=production_edges(rec); kn=np.array(e[1:-1]); print("interior edges",len(kn),"supplied",kr["n_knots_supplied"])
lv=[(fx(l["start"]),fx(l["end"]),l["depth"]) for l in kr["leaves"]]
leafset={(s,e_):d for s,e_,d in lv}; bounds={s for s,_,_ in lv}|{e_ for _,e_,_ in lv}
A0,B0=fx(kr["x_span"][0]),fx(kr["x_span"][1]); print("span",A0,B0,"leaf range",min(s for s,_,_ in lv),max(e_ for _,e_,_ in lv))
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
rec_(A0,B0,0)
print("leaves",len(lv),"splits",stats,"problems",len(bad),bad[:4])
print("depth-0/1 leaves:",[(s,e_,d) for s,e_,d in lv if d<=1][:6])
