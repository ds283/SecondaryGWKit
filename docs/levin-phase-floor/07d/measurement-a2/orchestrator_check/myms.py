# Orchestrator's checks at Part A''s stop (A'.3 items 2b, 4b, 5, 6), read-only, prints only.
import json, os, sys, glob, math, hashlib
import numpy as np
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement-a2"
sys.path.insert(1,R+"/scripts")  # the captures unpickle objects of module harness
import AdaptiveLevin.levin_quadrature as LQ
import ray.cloudpickle as cp
fx=float.fromhex
recs={}
for f in sorted(glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True)):
    if "/specs/" in f or "/logs/" in f or "population" in f: continue
    d=json.load(open(f))
    for L in d.get("leaves",[]):
        if "error" not in L: recs.setdefault(L["id"],(d["header"]["spec"],L))
caps={}
def load(specp):
    spec=json.load(open(specp)); c=spec["capture"]; k=(c["path"],c["offset"])
    if k not in caps:
        fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
        caps[k]=cp.loads(b)
    return spec,caps[k]
def V_of(rec,A,cache,a,b,N=24):
    g,D=LQ.chebyshev_matrices((a,b),N)
    return float(A.build_Levin_data(g,D,need_AmatT=False,vectorize_cache=cache)[4])
def g_on(rec,A,cache,x):
    fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
    bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
    return np.sum(fs*bs,axis=0)
ids=sorted(recs)
# ---- 2b: routes of the sub-solves at T = 6 pi and 12 pi, from their phase variations
def pick(pref,lo,hi):
    for i in ids:
        if i.startswith(pref):
            L=recs[i][1]
            if lo<=L["V_over_pi"]<hi and any(c["T_pi"]==6 and c.get("checked") for c in L["item1"]): return i
print("== 2b routes")
for lid in (pick("band_10_12|group",10.5,12), pick("band_12_16|group",12.5,16)):
    specp,L=recs[lid]; spec,rec=load(specp); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    a,b=fx(L["start"]),fx(L["end"]); c=(b+a)/2; cL=(c+a)/2; cR=(b+c)/2
    subs=((a,c),(c,b),(a,cL),(cL,c),(c,cR),(cR,b))
    Vs=[V_of(rec,A,cache,*s) for s in subs]; Vown=V_of(rec,A,cache,a,b)
    print(lid,"V/pi own",round(Vown/math.pi,3),"record",round(L["V_over_pi"],3),"subs V/pi",[round(v/math.pi,2) for v in Vs])
    for Tp in (6,12):
        T=Tp*math.pi; mine_own="cc" if Vown<T else "levin"; mine=["cc" if v<T else "levin" for v in Vs]
        rec_ch=[ch for ch in L["item1"] if ch["T_pi"]==Tp]
        have=[(ch["fallback"],ch.get("own_route"),ch.get("sub_routes")) for ch in rec_ch]
        ok=all(o==mine_own and (sr is None or sr==mine) for _,o,sr in have) if have else (mine_own=="cc")
        print(f"   T={Tp}pi mine own {mine_own} subs {mine} | record {have if have else 'no check (V<T: the leaf itself goes to CC)'} -> {'AGREE' if ok else 'DIFFER'}")
# ---- 4b: s_own,93 on three leaves, coefficients by numpy's chebfit (not the agent's DCT), cut from the record
print("== 4b s_own93")
for lid in (pick("band_10_12|group",10,12), pick("gate_q2|group",0,99) or ids[0], pick("band_12_16|analytic",12,16) or [i for i in ids if "|analytic|" in i][0]):
    specp,L=recs[lid]; spec,rec=load(specp); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    a,b=fx(L["start"]),fx(L["end"]); n=93
    grid,_=LQ.chebyshev_matrices((a,b),n); g=g_on(rec,A,cache,grid)
    t=(2*grid-(a+b))/(b-a)
    ck=np.polynomial.chebyshev.chebfit(t,g,n-1)
    cut=L["cc"]["c93"]["chop"]; tail=ck[cut:] if cut<n else ck[-8:]
    w=LQ._cc_weights((a,b),n)
    s=float(np.sqrt(np.mean(tail*tail))*math.sqrt((n-1)/2.0))*math.sqrt(float(np.sum(w*w)))
    cc93=float(np.dot(w,g))
    print(f"{lid}: cut {cut}, mine s_own93 {s:.6g} record {L['cc']['c93']['s_own']:.6g} ratio {s/L['cc']['c93']['s_own']:.4f}; CC93 mine==record {cc93.hex()==L['cc']['cc93']}")
# ---- 5: knot counts by kind on two leaves (my own walk of the integrand's closures), one jump, K's split points
print("== 5 knots")
def kinds(rec):
    ta=rec["f"][0].__closure__[2].cell_contents
    cl={n_:c_.cell_contents for n_,c_ in zip(ta.__wrapped__.__code__.co_freevars, ta.__wrapped__.__closure__)}
    active=rec["theta"]["theta"].__closure__[0].cell_contents
    K={k:set() for k in ("phase","G","node","mid","break")}; tabs={}
    for _,fac in active:
        K["phase"].update(map(float,fac.phase._spline.t)); tabs[id(fac.phase._leading._table)]=fac.phase._leading._table
    for nm in ("q_factor","r_factor"):
        f_=cl[nm]
        if hasattr(f_,"_f") and hasattr(f_._f,"_friction_F"): tabs[id(f_._f._friction_F._table)]=f_._f._friction_F._table
    G=cl["G_factor"]
    if hasattr(G,"_Gk"): K["G"].update(map(float,G._Gk.sin_amplitude._spline.t))
    for t_ in tabs.values():
        u=np.asarray(t_._u_asc,dtype=float); K["node"].update(u.tolist()); K["mid"].update((u[:-1]+(u[1:]-u[:-1])/2).tolist())
        if t_._breaks is not None: K["break"].update(map(float,np.asarray(t_._breaks).ravel()))
    return K
grp=[i for i in ids if "|group|" in i and recs[i][1].get("item4") and recs[i][1]["n_knots"] and recs[i][1]["n_knots"]>=3 and not i.startswith("band_20_32v")]
for lid in (grp[0], grp[len(grp)//2]):
    specp,L=recs[lid]; spec,rec=load(specp); a,b=fx(L["start"]),fx(L["end"])
    K=kinds(rec); mine={k:sum(1 for x in v if a<x<b) for k,v in K.items()}
    mine["all"]=len(set().union(*[{x for x in v if a<x<b} for v in K.values()]))
    print(lid,"mine",mine,"record",L["item4"]["counts"],"record n_knots",L["n_knots"])
    co=sorted({x for x in K["phase"] if a<x<b}), sorted({x for x in K["G"] if a<x<b}), sorted({x for x in K["node"] if a<x<b})
    print("   phase==G==node sets:", co[0]==co[1]==co[2])
    # one jump: f0's value step across the record's 'mid' knot, by a direct difference (not the agent's fits)
    jm=L["item4"]["jumps"].get("mid")
    if jm:
        xk=fx(jm["x"]); cache={}; f0=rec["f"][0]
        for u_ in (1,64):
            u=u_*np.spacing(xk); v=LQ._sample_vectorized(f0,np.array([xk-u,xk+u]),cache,("f",id(f0)))
            print(f"   mid knot {xk!r}: |f0(x+u)-f0(x-u)|/|f0| at {u_} ulp = {abs(v[1]-v[0])/abs(v[0]):.3g}; record fit jump/|f0| {jm['f0']['jump'][0]/max(jm['f0']['value_abs']):.3g}")
# K's split points on one call, re-derived from the knots: the leaves must be the recursive splits of the span
print("== 5c K split points")
kp=R+"/out/replays/K/R_a_R1-x007-7.json"; kr=json.load(open(kp)); c=kr["header"]["capture"]
fh=open(c["path"],"rb"); fh.seek(c["offset"]); bb=fh.read(c["length"]); assert hashlib.sha256(bb).hexdigest()==c["sha256"]
rec=cp.loads(bb); K=kinds(rec); kn=np.array(sorted(set().union(*K.values())))
print("knots mine",len(kn),"supplied",kr["n_knots_supplied"])
def brk(a,b):
    lo,hi=min(a,b),max(a,b); q=0.25*(hi-lo)
    ks=kn[(kn>=lo+q)&(kn<=hi-q)]
    if len(ks):
        k=float(ks[np.argmin(np.abs(ks-0.5*(lo+hi)))])
        if lo<k<hi: return k,"knot"
    return (b+a)/2.0,"mid"
lv=[(fx(l["start"]),fx(l["end"]),l["depth"]) for l in kr["leaves"]]
leafset={(s,e):d for s,e,d in lv}; bounds={s for s,_,_ in lv}|{e for _,e,_ in lv}
A0,B0=fx(kr["x_span"][0]),fx(kr["x_span"][1]); stats={"knot":0,"mid":0}; bad=[]
def rec_(a,b,d):
    if (a,b) in leafset:
        if leafset[(a,b)]!=d: bad.append(("depth",a,b,d,leafset[(a,b)]))
        return
    s,kind=brk(a,b)
    if s not in bounds: bad.append(("split not a leaf edge",a,b,s)); return
    stats[kind]+=1; rec_(a,s,d+1); rec_(s,b,d+1)
rec_(A0,B0,0)
print("leaves",len(lv),"splits re-derived",stats,"problems",bad[:5],len(bad))
# ---- 6: one leaf's evaluation counts, by counting the points the module samples
print("== 6 cost counters")
lid=pick("band_10_12|group",10,12); specp,L=recs[lid]; spec,rec=load(specp); a,b=fx(L["start"]),fx(L["end"])
orig=LQ._sample_vectorized; cnt={"n":0}
def counting(func,grid,cache,key):
    if key[0]=="f": cnt["n"]+=len(grid)
    return orig(func,grid,cache,key)
LQ._sample_vectorized=counting
nf=len(rec["f"])
try:
    A=LQ._Basis_SinCos(rec["theta"])
    for nm,fn in (("CC solve (force_direct)",lambda c_: LQ._adaptive_levin_subregion((a,b),rec["f"],A,id_label=LQ._LazyUUID(),chebyshev_order=24,rtol=1e-7,vectorize_cache=c_,force_direct=True)),
                  ("one Levin solve (gate off)",None)):
        cnt["n"]=0; cache={}
        if fn is None:
            six=LQ.SIX_PI; LQ.SIX_PI=-math.inf
            try: LQ._adaptive_levin_subregion((a,b),rec["f"],A,id_label=LQ._LazyUUID(),chebyshev_order=24,rtol=1e-7,vectorize_cache=cache)
            finally: LQ.SIX_PI=six
        else: fn(cache)
        print(f"{lid}: {nm}: f-sample points per component {cnt['n']/nf:g}")
finally:
    LQ._sample_vectorized=orig
print("record points:",L["cost"]["points"])
