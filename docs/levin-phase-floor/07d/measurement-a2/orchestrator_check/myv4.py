# Orchestrator's check (orchestrator note A'.3 item 1): four leaves re-referenced by an independent rule.
# 32-point Gauss-Legendre (not v4's 24 nor ref2's 48), panels doubled 32 -> 1024 per piece (one step past v4's 512),
# integrand built here from the capture (not leafref_v4.Integrand); knots read by production_edges (group) or the
# agent's analytic reader (knot locations only). Read-only; prints only.
import json, os, sys, glob, math, hashlib, time
import numpy as np
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement-a2"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
import ray.cloudpickle as cp
from tools.filon_reference import production_edges
import leafref_v4 as R4  # call_knots only (analytic knot locations)
IDS=sys.argv[1:]
want=set(IDS); found={}
for f in glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True):
    if "/specs/" in f or "/logs/" in f or "population" in f: continue
    try: d=json.load(open(f))
    except Exception: continue
    for L in d.get("leaves",[]):
        if L.get("id") in want and "error" not in L: found[L["id"]]=(f,d["header"]["spec"],L)
ref2={r["id"]:r for r in json.load(open(R+"/out/repro/second_ref_a2.json"))["rows"]}
gx,gw=np.polynomial.legendre.leggauss(32)
for lid in IDS:
    f,specp,L=found[lid]; spec=json.load(open(specp)); c=spec["capture"]
    fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
    rec=cp.loads(b); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    def g(x):
        fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
        bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
        return np.sum(fs*bs,axis=0)
    a,bb=float.fromhex(L["start"]),float.fromhex(L["end"])
    xs0,xs1=(float.fromhex(v) for v in spec["x_span"])
    try:
        e,_=production_edges(rec); src="production_edges"
    except Exception:
        e,_,_=R4.call_knots(rec,xs0,xs1); src="analytic reader"
    kn=[x for x in (e or []) if a<x<bb]; brk=[a]+kn+[bb]
    sig=L["v3_sigma"]["sigma_hat"]
    def gl(n):
        terms=[]; w2=0.0
        for pa,pb in zip(brk[:-1],brk[1:]):
            h=(pb-pa)/n
            x0=pa+h*np.arange(n); x1=np.append(x0[1:],pb)
            mid=0.5*(x0+x1)[:,None]; hw=0.5*(x1-x0)[:,None]
            xx=(mid+hw*gx).ravel(); ww=(hw*gw).ravel()
            terms.append(ww*g(xx)); w2+=float(np.sum(ww*ww))
        return math.fsum(np.concatenate(terms)),w2
    top=1024 if len(brk)-1<=40 else 256
    t0=time.time(); hist=[]; n=32
    while n<=top: hist.append((n,)+gl(n)); n*=2
    (n1,I1,_),(n2,I2,w2)=hist[-2],hist[-1]
    eps=abs(I2-I1)+2*sig*math.sqrt(w2)
    v4=float.fromhex(L["v4"]["ref"]); e4=float.fromhex(L["v4"]["eps_ref"])
    r2=ref2.get(lid)
    print(f"{lid}: pieces {len(brk)-1} ({src}), V/pi {L['V_over_pi']:.2f}, panels to {n2}, {time.time()-t0:.0f}s")
    print("   ladder:", "  ".join(f"{n}:{I!r}" for n,I,_ in hist[-4:]))
    print(f"   mine {I2!r} eps_mine {eps:.3g} | v4 {v4!r} eps_v4 {e4:.3g} | |mine-v4| {abs(I2-v4):.3g} = {abs(I2-v4)/(eps+e4):.2f} x (eps_mine+eps_v4)")
    if r2: print(f"   ref2 {r2['ref2']!r} eps2 {r2['eps2']:.3g} | |mine-ref2| {abs(I2-r2['ref2']):.3g} = {abs(I2-r2['ref2'])/(eps+r2['eps2']):.2f} x (eps_mine+eps2)")
    vp=L["levin_one"].get("V_P"); 
    for nm,v in (("V_P",vp),("CC93",L["cc"]["cc93"]),("CC185",L["cc"]["cc185"])):
        if v: print(f"   {nm} error vs mine {abs(float.fromhex(v)-I2):.3g}")
