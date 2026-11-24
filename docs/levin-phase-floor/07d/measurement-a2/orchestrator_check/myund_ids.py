import json, glob, math, sys
R=sys.argv[1]; fx=float.fromhex
leaves={}
for f in glob.glob(R+"/out/leaves_a2/**/*.json",recursive=True):
    if "population" in f: continue
    for L in json.load(open(f)).get("leaves",[]):
        if "error" not in L: leaves.setdefault(L["id"],L)
ids=set()
for lid,L in leaves.items():
    st=lid.split("#")[0]
    if not (st.startswith("band_") or st.startswith("gate_q")) or st.startswith("band_20_32v") or not L["v4"]["settled"]: continue
    ref=fx(L["v4"]["ref"]); eps=fx(L["v4"]["eps_ref"])
    for ch in L["item1"]:
        if ch["T_pi"] in (6,8,10,12) and ch["fallback"]=="cc93" and ch.get("checked") and ch["trusted"]:
            D=fx(ch["total_err_trusted"]); t=abs(fx(ch["Q"])-ref)
            if not (D>=t+eps) and not (D<t-eps) and D<t: ids.add(lid)
print("\n".join(sorted(ids)))
