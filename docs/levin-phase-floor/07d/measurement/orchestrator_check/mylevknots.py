# Orchestrator check, 07d Part A: V_P against CC47/CC93 on group band leaves, by knot count; knot counts per band.
import json, glob, os, statistics as S, collections
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
h=float.fromhex
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
recs={}
for f in glob.glob(R+"/out/leaves/round*/*.json"):
    d=json.load(open(f))
    for L in d["leaves"]: recs[(d["sid"],L["start"])]=L
B=collections.defaultdict(list); seen=set(); nk=collections.defaultdict(list)
for x in json.load(open(R+"/out/tables/d_t.json"))["rows"]:
    o=sel[x["stratum"]]["order"][x["rank"]]; L=recs.get((o["sid"],o["start"]))
    if L is None or (o["sid"],o["start"]) in seen: continue
    seen.add((o["sid"],o["start"]))
    if x["kind"]!="group": continue
    n=L.get("n_knots"); nk[x["band"]].append(n if n is not None else -1)
    if not x["band"].startswith("band_"): continue
    c="none" if not n else ("1-4" if n<5 else "5+")
    B[c].append((x,L))
for c,xs in sorted(B.items()):
    sh=lambda x:x["request_share"]
    rl=[x["te_levin"]/sh(x) for x,L in xs]; r93=[abs(h(L["cc93"])-h(L["ref"]))/sh(x) for x,L in xs]
    lw=sum(1 for x,L in xs if x["te_levin"]>abs(h(L["cc93"])-h(L["ref"]))+2*h(L["eps_ref"]))
    cw=sum(1 for x,L in xs if abs(h(L["cc93"])-h(L["ref"]))>x["te_levin"]+2*h(L["eps_ref"]))
    print(f"group band leaves, knots {c:4s} n {len(xs):3d} | V_P worse than CC93 {lw:3d}, better {cw:3d} | median err/share V_P {S.median(rl):.2g} CC93 {S.median(r93):.2g} | V_P over its share {sum(v>1 for v in rl)}")
for b,v in sorted(nk.items()):
    v=sorted(v); print(f"{b:10s} knots per group leaf: n {len(v)} none {sum(1 for t in v if t==0)} unknown {sum(1 for t in v if t<0)} median {v[len(v)//2]} p90 {v[int(.9*len(v))]} max {v[-1]}")
