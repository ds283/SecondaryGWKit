import json, os, sys, hashlib, math, glob, numpy as np, time, random
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
from tools.filon_reference import production_edges
import ray.cloudpickle as cp
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
rows=[r for r in json.load(open(R+"/out/tables/d_t.json"))["rows"] if r["band"].startswith("band_") and r["kind"]=="group" and r["levin"][0]=="understates"]
rows.sort(key=lambda r:r["levin"][1])
mid=[r for r in rows if 0.5<r["levin"][1]<1.0]
random.seed(7); pick=rows[:14]+random.sample(mid,6)
print("group band leaves the agent scores Levin-understating:",len(rows),"; with D/true in (0.5,1):",len(mid))
gx,gw=np.polynomial.legendre.leggauss(24)
def leafrec(o):
    for f in glob.glob(R+"/out/leaves/round*/"+o["sid"].replace(":","_").replace("|","_")+".json"):
        for L in json.load(open(f))["leaves"]:
            if L["start"]==o["start"]: return L
agree=0
for r in pick:
    t0=time.time()
    o=sel[r["stratum"]]["order"][r["rank"]]; c=man[o["sid"]]
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
    npA,npB=(32,128) if (kn and len(kn)>10) else (64,256)
    (rA,_),(rB,w2)=gl(npA),gl(npB)
    L=leafrec(o); lv=L["levin"]; vP=float.fromhex(lv["vP"]); sig=L["noise"]["sigma_hat"]
    t=abs(vP-rB); e=abs(rA-rB)+2*sig*math.sqrt(w2)
    D=r["levin"][1]*r["te_levin"]
    v="bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
    agree+= v=="understates"
    s1,s2=float.fromhex(lv["s1"]),float.fromhex(lv["s2"])
    print(f"{r['stratum']:22s}#{r['rank']:3d} V/pi {r['V_over_pi']:5.1f} knots {None if kn is None else len(kn)!s:3s} | agent te {r['te_levin']:.3g} eps {r['eps']:.2g} D/true {r['levin'][1]:.3g} | mine te {t:.3g} eps {e:.2g} D/true {D/t if t else float('inf'):.3g} -> {v:11s} | |s1-ref| {abs(s1-rB):.2g} |s2-ref| {abs(s2-rB):.2g} own_decl {float.fromhex(lv['own_decl']):.2g} r {lv['r']:.2g} | levin_tc agent {r['levin_tc'][0]} te {r['te_levin_tc']:.2g} | {time.time()-t0:.0f}s",flush=True)
print("still understating under my reference:",agree,"of",len(pick))
