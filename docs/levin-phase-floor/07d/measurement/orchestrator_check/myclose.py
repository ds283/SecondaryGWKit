import json, os, sys, hashlib, math, glob, numpy as np, time
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
from tools.filon_reference import production_edges
import ray.cloudpickle as cp
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
de=[L for L in json.load(open(R+"/out/tables/d_e.json"))["leaves"] if L["scored"]]
NSEL=int(sys.argv[1]) if len(sys.argv)>1 else 25
cand=sorted([L for L in de if L["true_err"]>0],key=lambda L:L["D"]["nest3N4"]/L["true_err"])[:NSEL]
gx,gw=np.polynomial.legendre.leggauss(24)
caps={}
def leafrec(o):
    for f in glob.glob(R+"/out/leaves/round*/"+o["sid"].replace(":","_").replace("|","_")+".json"):
        for L in json.load(open(f))["leaves"]:
            if L["start"]==o["start"]: return L
flips=0
for L in cand:
    t0=time.time()
    o=sel[L["stratum"]]["order"][L["rank"]]; sid=o["sid"]; c=man[sid]
    fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
    rec=cp.loads(b); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    def g(x):
        fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
        bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
        return np.sum(fs*bs,axis=0)
    a,bb=float.fromhex(o["start"]),float.fromhex(o["end"])
    try: edges,_=production_edges(rec); kn=[e for e in edges if a<e<bb]
    except Exception: kn=None
    brk=[a]+(kn or [])+[bb]
    def gl(npan):
        terms=[];w2=0.0
        for pa,pb in zip(brk[:-1],brk[1:]):
            h=(pb-pa)/npan
            for j in range(npan):
                x0=pa+j*h; x1=pa+(j+1)*h if j<npan-1 else pb
                ww=0.5*(x1-x0)*gw; terms.append(ww*g(0.5*(x0+x1)+0.5*(x1-x0)*gx)); w2+=float(np.sum(ww*ww))
        return math.fsum(np.concatenate(terms)),w2
    (rA,_),(rB,w2B)=gl(64),gl(256)
    lr=leafrec(o); cc47=float.fromhex(lr["cc47"]); sig=lr["noise"]["sigma_hat"]
    t=abs(cc47-rB); e=abs(rA-rB)+2*sig*math.sqrt(w2B)
    D=L["D"]["nest3N4"]; Dt=L["D"]["tree"]
    v=lambda D:"bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
    flips+= v(D)=="understates"
    print(f"{L['stratum']:22s} #{L['rank']:3d} knots {None if kn is None else len(kn)!s:4s} agent true {L['true_err']:.3g} eps {L['eps_ref']:.2g} | mine true {t:.3g} eps {e:.2g} | nest3N4 D/true agent {D/L['true_err']:.3f} mine {D/t:.3f} -> mine {v(D)} agent {L['verdicts']['nest3N4']} | tree mine {v(Dt)} agent {L['verdicts']['tree']}  {time.time()-t0:.0f}s",flush=True)
print("nest3N4 understates under my reference:",flips,"of",len(cand))
