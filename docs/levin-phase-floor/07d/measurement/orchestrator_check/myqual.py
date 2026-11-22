import json, os, math
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
de=json.load(open(R+"/out/tables/d_e.json")); dt=json.load(open(R+"/out/tables/d_t.json"))["rows"]
def verdict(D,t,e): return "bounded" if D>=t+e else ("understates" if D<t-e else "undecided")
under={}
for L in de["leaves"]:
    if not L["scored"]: continue
    for c,D in L["D"].items():
        if D is None: continue
        under[c]=under.get(c,0)+(verdict(D,L["true_err"],L["eps_ref"])=="understates")
est_ok={c:under.get(c,0)==0 for c in de["candidates"]}
print("table 2 understatements:",under,"\n estimate qualifies:",est_ok)
def thr(Tpi,est,route="levin"):
    cc_bad=lv_bad=n_above=0
    for r in dt:
        V=r["V_over_pi"]; t_cc=r["te_cc"]; e=r["eps"]
        if V<Tpi:
            vv,ratio,_=r["cc"][est]
            if ratio is not None and verdict(ratio*t_cc,t_cc,e)=="understates": cc_bad+=1
        else:
            n_above+=1
            vv,ratio=r[route][0],r[route][1]; te=r["te_levin"] if route=="levin" else r["te_levin_tc"]
            if ratio is not None and verdict(ratio*te,te,e)=="understates": lv_bad+=1
            elif ratio is None and vv=="understates": lv_bad+=1
    return cc_bad,lv_bad,n_above
EST={"Eg":"nest3N4","Ea":"nest3N4","ET20":"nest3N4","ET12":"nest3N4","Ep":"pairN4","T20":"tree"}
T={"Eg":6,"Ea":6,"ET20":20,"ET12":12,"Ep":6,"T20":20}
newdis={"ET12":["R:a:W2-0-12977-5"],"T20":["R:a:R2-08~o"]}
for v in EST:
    cc_bad,lv_bad,nab=thr(T[v],EST[v]); cb2,lv2,_=thr(T[v],EST[v],"levin_tc")
    both=cc_bad==0 and lv_bad==0
    q=both and est_ok[EST[v]] and not newdis.get(v)
    print(f"{v:5s} est {EST[v]:8s} ok {est_ok[EST[v]]!s:5s} | T={T[v]}pi: CC understates below {cc_bad}, Levin understates at/above {lv_bad} of {nab} (trust-checked {lv2}) -> bounded both sides {both} | new dishonest {newdis.get(v,[])} | QUALIFIES {q}")
