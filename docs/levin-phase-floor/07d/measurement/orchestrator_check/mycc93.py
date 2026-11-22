# Orchestrator check, 07d Part A: would returning CC93 beat CC47; CC vs Levin by band.
import json, glob, os, statistics as S, collections
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
h=float.fromhex
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
recs={}
for f in glob.glob(R+"/out/leaves/round*/*.json"):
    d=json.load(open(f))
    for L in d["leaves"]: recs[(d["sid"],L["start"])]=L
def rec(st,rk):
    o=sel[st]["order"][rk]; return recs.get((o["sid"],o["start"])), o["sid"]
def q(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))] if xs else float("nan")
# ---- 1. CC93 as the value, on D-E's scored leaves and D-T's band leaves (deduplicated)
rows=[]; seen=set()
for x in json.load(open(R+"/out/tables/d_e.json"))["leaves"]:
    if not x["scored"]: continue
    L,sid=rec(x["stratum"],x["rank"])
    if L is None or (sid,L["start"]) in seen: continue
    seen.add((sid,L["start"])); rows.append((x["stratum"],L,x["D"]["nest3N4"]))
for x in json.load(open(R+"/out/tables/d_t.json"))["rows"]:
    L,sid=rec(x["stratum"],x["rank"])
    if L is None or (sid,L["start"]) in seen or x["cc"]["nest3N4"][1] is None: continue
    seen.add((sid,L["start"])); rows.append((x["stratum"],L,x["cc"]["nest3N4"][1]*x["te_cc"]))
print("leaves (distinct):",len(rows))
def kn(L): n=L.get("n_knots"); return "none" if not n else ("1-4" if n<5 else "5+")
C=collections.defaultdict(collections.Counter); rat=collections.defaultdict(list); gain=collections.defaultdict(list)
for st,L,D in rows:
    ref,eps=h(L["ref"]),h(L["eps_ref"]); t47,t93=abs(h(L["cc47"])-ref),abs(h(L["cc93"])-ref)
    t185=abs(h(L["cc185"])-ref) if L.get("cc185") else None
    k=(st.split("|")[1],kn(L)); c=C[k]; c["n"]+=1
    # which value is closer, only where the reference can tell them apart
    if abs(t47-t93)>2*eps: c["93 better" if t93<t47 else "47 better"]+=1
    else: c["cannot tell"]+=1
    if t47>3*eps: gain[k].append(t47/max(t93,eps))
    for nm,t in (("D vs CC47",t47),("D vs CC93",t93)):
        c[nm+": bounded" if D>=t+eps else (nm+": understates" if D<t-eps else nm+": undecided")]+=1
    if t93>eps: rat[k].append(D/t93)
for k in sorted(C):
    c=C[k]; print(k,dict(sorted(c.items())))
    if gain[k]: print("   CC47 err / CC93 err where CC47 err > 3 eps: n",len(gain[k]),"median %.3g p10 %.3g p90 %.3g"%(S.median(gain[k]),q(gain[k],.1),q(gain[k],.9)))
    if rat[k]: print("   nest3N4 D / CC93 true err where > eps: median %.3g p01 %.3g min %.3g"%(S.median(rat[k]),q(rat[k],.01),min(rat[k])))
# ---- 2. CC vs Levin by band, on D-T's rows (V_P is the leaf's own solve: its value does not depend on the sub-solves)
print("\nD-T: per band and kind -- distinct leaves; CC47, CC93, V_P true errors over the leaf's request share")
B=collections.defaultdict(list); seen=set()
for x in json.load(open(R+"/out/tables/d_t.json"))["rows"]:
    L,sid=rec(x["stratum"],x["rank"])
    if L is None or (sid,L["start"]) in seen: continue
    seen.add((sid,L["start"])); B[(x["band"],x["kind"])].append((x,L))
order=["gate_q1","gate_q2","gate_q3","gate_q4","band_6_8","band_8_10","band_10_12","band_12_16","band_16_20"]
for (band,kind) in sorted(B,key=lambda k:(order.index(k[0]) if k[0] in order else 99,k[1])):
    xs=B[(band,kind)]; ref=lambda L:h(L["ref"])
    r47=[abs(h(L["cc47"])-ref(L))/x["request_share"] for x,L in xs]
    r93=[abs(h(L["cc93"])-ref(L))/x["request_share"] for x,L in xs]
    rl=[x["te_levin"]/x["request_share"] for x,L in xs]
    better=sum(1 for x,L in xs if abs(abs(h(L["cc47"])-ref(L))-x["te_levin"])>2*h(L["eps_ref"]) and abs(h(L["cc47"])-ref(L))<x["te_levin"])
    worse=sum(1 for x,L in xs if abs(abs(h(L["cc47"])-ref(L))-x["te_levin"])>2*h(L["eps_ref"]) and abs(h(L["cc47"])-ref(L))>x["te_levin"])
    ok47=sum(1 for v in r47 if v<=1); ok93=sum(1 for v in r93 if v<=1); okl=sum(1 for v in rl if v<=1)
    print(f"{band:10s} {kind:8s} n {len(xs):3d} | CC47 more accurate than V_P {better:3d}, less {worse:3d} | within request share: CC47 {ok47:3d} CC93 {ok93:3d} V_P {okl:3d} | median err/share CC47 {S.median(r47):.2g} CC93 {S.median(r93):.2g} V_P {S.median(rl):.2g} | p90 CC47 {q(r47,.9):.2g} CC93 {q(r93,.9):.2g} V_P {q(rl,.9):.2g}")
