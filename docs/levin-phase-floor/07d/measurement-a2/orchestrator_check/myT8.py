# Orchestrator's check (A'.3 items 2a, 3, 4a): re-read the reproduction check; re-score table 4 at T = 8 pi and
# table 3's candidate counts, from the per-leaf records, by my own reader. Exact Fractions for every verdict.
import json, glob, math, sys
from fractions import Fraction as F
from collections import Counter, defaultdict
R=sys.argv[1]
fx=float.fromhex
vr=json.load(open(R+"/out/repro/verify_repro_a2.json"))
rows=vr["rows"]; eq=sum(1 for r in rows if r["equal"] and not r["differs"])
print("repro rows",len(rows),"equal",eq,"by kind",Counter((r["kind"],r["equal"]) for r in rows))
leaves={}
for f in glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True):
    if "population" in f: continue
    d=json.load(open(f))
    for L in d.get("leaves",[]):
        if "error" in L: continue
        leaves.setdefault(L["id"],L)
print("distinct leaf records",len(leaves))
def verdict(D,v,ref,eps):
    t=abs(F(v)-F(ref)); D=F(D); e=F(eps)
    return "bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
kappa=4
T=8*math.pi
tab=defaultdict(Counter)
for lid,L in leaves.items():
    st=lid.split("#")[0]; kind=st.split("|")[1]
    if not (st.startswith("band_") or st.startswith("gate_q")) or st.startswith("band_20_32v"): continue
    if not L["v4"]["settled"]: continue
    ref=fx(L["v4"]["ref"]); eps=fx(L["v4"]["eps_ref"])
    if L["V"]>=T:
        for ch in L["item1"]:
            if ch["T_pi"]!=8: continue
            fb=ch["fallback"]
            if not ch.get("checked"): tab[(kind,"levin",fb)]["not checked"]+=1; continue
            if not ch["trusted"]: tab[(kind,"levin",fb)]["untrusted"]+=1; continue
            v=verdict(fx(ch["total_err_trusted"]),fx(ch["Q"]),ref,eps)
            tab[(kind,"levin",fb)][v]+=1
            if v=="understates": print("  LEVIN UNDERSTATES",lid,fb)
    else:
        c=L["cc"]; cc47=fx(c["cc47"]); cc93=fx(c["cc93"]); cc185=fx(c["cc185"]); fl=fx(c["floor"])
        s93=c["c93"]["s_own"]; s47=c["c47"]["s_own"]
        tree=max(fx(c["pair"]),fl)
        nest3N4_47=abs(cc93-cc47)+4*s47     # Part A's nest3N4 on CC47: |CC93-CC47| + 4 s_own47 (value CC47)
        pair93=max(abs(cc185-cc93),fl)+kappa*s93
        for nm,D,v in (("tree CC47",tree,cc47),("pair93 CC93",pair93,cc93)):
            vv=verdict(D,v,ref,eps); tab[(kind,"cc",nm)][vv]+=1
for k in sorted(tab): 
    c=tab[k]; sc=c["bounded"]+c["understates"]
    print(k, dict(c), "scored",sc, "->", "vacuous" if sc==0 else ("fails" if c["understates"] else "bounded"))
