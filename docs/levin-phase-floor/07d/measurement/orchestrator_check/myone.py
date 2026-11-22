import json, os, sys, hashlib, math, glob, numpy as np
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
from tools.filon_reference import production_edges
import ray.cloudpickle as cp
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
gx,gw=np.polynomial.legendre.leggauss(24)
for st,rk in [("band_16_20|group|1e-8",32),("band_10_12|group|1e-7",3)]:
    o=sel[st]["order"][rk]
    L=next(L for f in glob.glob(R+"/out/leaves/round*/"+o["sid"].replace(":","_").replace("|","_")+".json") for L in json.load(open(f))["leaves"] if L["start"]==o["start"])
    c=man[o["sid"]]; fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"])
    rec=cp.loads(b); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    def g(x):
        fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
        bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
        return np.sum(fs*bs,axis=0)
    a,bb=float.fromhex(o["start"]),float.fromhex(o["end"])
    edges,_=production_edges(rec); kn=[e for e in edges if a<e<bb]
    print(st,rk,o["sid"],"leaf",a,bb,"width",bb-a,"knot at frac",[(k-a)/(bb-a) for k in kn])
    for k in ["ref","I_lo","GL","eps_ref","eps_LS","s_LS","s_GL","ref_check","deg","deg_pieces","n_pieces","cc47","cc93","cc185"]:
        v=L.get(k); print("   ",k,float.fromhex(v) if isinstance(v,str) and ("0x" in v) else v)
    print("    levin",{k:(float.fromhex(v) if isinstance(v,str) else v) for k,v in L["levin"].items() if k!="guarded"})
    def gl(brk,npan):
        t=[]
        for pa,pb in zip(brk[:-1],brk[1:]):
            h=(pb-pa)/npan
            for j in range(npan):
                x0=pa+j*h; x1=pa+(j+1)*h if j<npan-1 else pb
                t.append(0.5*(x1-x0)*gw*g(0.5*(x0+x1)+0.5*(x1-x0)*gx))
        return math.fsum(np.concatenate(t))
    for n in (16,64,256):
        print("    my GL split",n,repr(gl([a]+kn+[bb],n)),"  unsplit",repr(gl([a,bb],n)))
    # jump test at the knot: one-sided differences of g
    k=kn[0]; h=(bb-a)*1e-4
    xl=np.array([k-3*h,k-2*h,k-h]); xr=np.array([k+h,k+2*h,k+3*h])
    print("    g left",g(xl),"right",g(xr))
