import json, os, sys, hashlib, math, numpy as np
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
from tools.filon_reference import production_edges
import ray.cloudpickle as cp
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
de={(L["stratum"],L["rank"]):L for L in json.load(open(R+"/out/tables/d_e.json"))["leaves"]}
gx,gw=np.polynomial.legendre.leggauss(24)
def run(st,rank):
    o=sel[st]["order"][rank]; sid=o["sid"]; c=man[sid]
    fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
    rec=cp.loads(b); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    N=int(rec["kw"].get("chebyshev_order",24))
    def g(x):
        fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
        bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
        return np.sum(fs*bs,axis=0)
    a,bb=float.fromhex(o["start"]),float.fromhex(o["end"])
    try:
        edges,_=production_edges(rec); kn=[e for e in edges if a<e<bb]
    except Exception as e:
        kn=None
    brk=[a]+(kn or [])+[bb]
    def gl(npan):
        terms=[]
        for pa,pb in zip(brk[:-1],brk[1:]):
            h=(pb-pa)/npan
            for j in range(npan):
                x0=pa+j*h; x1=pa+(j+1)*h if j<npan-1 else pb
                terms.append(0.5*(x1-x0)*gw*g(0.5*(x0+x1)+0.5*(x1-x0)*gx))
        return math.fsum(np.concatenate(terms))
    r1,r2=gl(16),gl(32)
    def cc(lo,hi,Mn):
        grid,_=LQ.chebyshev_matrices((lo,hi),Mn); return math.fsum(LQ._cc_weights((lo,hi),Mn)*g(grid))
    cc47=cc(a,bb,2*N-1); cc24=cc(a,bb,N)
    split=math.fsum(cc(pa,pb,2*N-1) for pa,pb in zip(brk[:-1],brk[1:])) if kn else None
    L=de[(st,rank)]
    print(f"{st} #{rank} {sid} knots={None if kn is None else len(kn)}")
    print(f"  my ref (GL 16 vs 32 panels/piece): {r2!r}  |r16-r32| {abs(r1-r2):.3g}")
    print(f"  my |cc47-ref| {abs(cc47-r2):.4g}   agent true_err {L['true_err']:.4g}  agent eps_ref {L['eps_ref']:.3g}")
    print(f"  my |cc24-ref| {abs(cc24-r2):.4g}   pair {abs(cc47-cc24):.4g}")
    if split is not None: print(f"  my split-at-knots CC47 error {abs(split-r2):.4g}  agent err_split {L.get('err_split_at_knots')}")
    for cnd,D in L["D"].items():
        t=abs(cc47-r2); e=L["eps_ref"]+abs(r1-r2)
        v="bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
        print(f"    {cnd:10s} D {D:.3g}  mine {v:11s} agent {L['verdicts'][cnd]}")
for st,rank in [("chase|group|1e-7",13),("window|group|1e-8",39),("chase|analytic|1e-7",0)]: run(st,rank)
