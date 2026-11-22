import json, os, sys, glob
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
def leafrec(o):
    for f in glob.glob(R+"/out/leaves/round*/"+o["sid"].replace(":","_").replace("|","_")+".json"):
        for L in json.load(open(f))["leaves"]:
            if L["start"]==o["start"]: return L,f
for st,rk in [("gate_q1|group|1e-8",73),("gate_q3|group|1e-7",3),("gate_q4|group|1e-8",64)]:
    o=sel[st]["order"][rk]; L,f=leafrec(o)
    a,b=float.fromhex(L["start"]),float.fromhex(L["end"])
    print(st,rk,os.path.basename(f),"width",b-a,"n_knots",L["n_knots"],"n_pieces",L["n_pieces"],"deg",L["deg"],"deg_pieces",L["deg_pieces"])
    for k in ["ref","I_lo","GL","eps_ref","eps_LS","s_LS","s_GL","cc47","cc93","cc185","ref_check"]:
        v=L.get(k); print("  ",k, float.fromhex(v) if isinstance(v,str) else v)
    print("   keys:",[k for k in L.keys() if k not in("noise","ingredients","levin")])
