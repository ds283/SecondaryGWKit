import json, glob, sys
from fractions import Fraction as F
from collections import Counter
R=sys.argv[1]; fx=float.fromhex
leaves={}
for f in glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True):
    if "population" in f: continue
    for L in json.load(open(f)).get("leaves",[]):
        if "error" not in L: leaves.setdefault(L["id"],L)
def verdict(D,v,ref,eps):
    t=abs(F(v)-F(ref)); D=F(D); e=F(eps)
    return "bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
k=4; res={}
for fam in ("returned","visited"):
  for nm in ("nest3N4","pair93","nest93","tail93"): res[(fam,nm)]=Counter()
for lid,L in leaves.items():
    if not L["v4"]["settled"]: continue
    fam="visited" if lid.startswith("band_20_32v") else "returned"
    ref=fx(L["v4"]["ref"]); eps=fx(L["v4"]["eps_ref"]); c=L["cc"]
    cc47,cc93,cc185,fl=fx(c["cc47"]),fx(c["cc93"]),fx(c["cc185"]),fx(c["floor"])
    s93=c["c93"]["s_own"]; s47=c["c47"]["s_own"]
    D={"nest3N4":max(abs(cc93-cc47),fl)+4*s47,"pair93":max(abs(cc185-cc93),fl)+k*s93,
       "nest93":max(abs(cc93-cc47),fl)+k*s93,"tail93":max(fx(c["c93"]["tail_sum4"]),fl)+k*s93}
    for nm,d in D.items():
        v=verdict(d,cc93,ref,eps); res[(fam,nm)][v]+=1
        if v=="understates": print("UNDERSTATES",nm,lid)
for kk,c in sorted(res.items()): print(kk,dict(c))
