import json, glob, math, sys
from fractions import Fraction as F
from collections import Counter, defaultdict
R=sys.argv[1]; fx=float.fromhex
leaves={}
for f in glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True):
    if "population" in f: continue
    for L in json.load(open(f)).get("leaves",[]):
        if "error" not in L: leaves.setdefault(L["id"],L)
ref2={r["id"]:r for r in json.load(open(R+"/out/repro/second_ref_a2.json"))["rows"]}
def verdict(D,v,ref,eps):
    t=abs(F(v)-F(ref)); D=F(D); e=F(eps)
    return "bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
for which in ("v4","ref2 where taken"):
  print("== reference:",which)
  for Tp in (6,8,10,12):
    T=Tp*math.pi; tab=defaultdict(Counter); bad=[]
    for lid,L in leaves.items():
        st=lid.split("#")[0]; kind=st.split("|")[1]
        if not (st.startswith("band_") or st.startswith("gate_q")) or st.startswith("band_20_32v"): continue
        if not L["v4"]["settled"]: continue
        ref=fx(L["v4"]["ref"]); eps=fx(L["v4"]["eps_ref"])
        if which!="v4" and lid in ref2: ref,eps=ref2[lid]["ref2"],ref2[lid]["eps2"]
        if L["V"]>=T:
            for ch in L["item1"]:
                if ch["T_pi"]!=Tp or ch["fallback"]!="cc93" or not ch.get("checked") or not ch["trusted"]: continue
                v=verdict(fx(ch["total_err_trusted"]),fx(ch["Q"]),ref,eps); tab[(kind,"levin")][v]+=1
                if v=="understates": bad.append(lid)
        else:
            c=L["cc"]; cc93,cc185,fl=fx(c["cc93"]),fx(c["cc185"]),fx(c["floor"]); s93=c["c93"]["s_own"]
            for nm,D in (("pair93",max(abs(cc185-cc93),fl)+4*s93),("tail93",max(fx(c["c93"]["tail_sum4"]),fl)+4*s93)):
                v=verdict(D,cc93,ref,eps); tab[(kind,"cc "+nm)][v]+=1
                if v=="understates": bad.append((nm,lid))
    print(f" T={Tp}pi:", "; ".join(f"{k[0]} {k[1]}: {c['bounded']+c['understates']} scored, {c['understates']} under" for k,c in sorted(tab.items())), "|", bad)
