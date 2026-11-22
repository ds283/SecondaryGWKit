import json, os, glob, collections
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
sel=json.load(open(R+"/state/selection_leaves_07d.json"))["selection"]
idx={}
for f in glob.glob(R+"/out/leaves/round*/*.json"):
    for L in json.load(open(f))["leaves"]:
        if "start" in L: idx[(os.path.basename(f),L["start"])]=L
def leaf(st,rank):
    o=sel[st]["order"][rank]; return idx.get((o["sid"].replace(":","_").replace("|","_")+".json",o["start"]))
def kb(n): return "unknown" if n is None else ("0" if n==0 else "1" if n==1 else "2-4" if n<5 else "5+")
rows=json.load(open(R+"/out/tables/d_t.json"))["rows"]
for route in ("levin","levin_tc"):
  for zone in ("gate","band"):
    c=collections.Counter(); tot=0
    for r in rows:
        if not r["band"].startswith(zone) or r[route][0]!="understates": continue
        L=leaf(r["stratum"],r["rank"]); tot+=1
        c[(r["kind"],kb(L["n_knots"]),"check ok" if L["ref_check"] else "check FAILED")]+=1
    print(route,zone,"understates",tot); [print("   ",k,v) for k,v in sorted(c.items())]
de=json.load(open(R+"/out/tables/d_e.json"))
c=collections.Counter()
for L0 in de["leaves"]:
    if L0["scored"] and L0["verdicts"]["tree"]=="understates":
        L=leaf(L0["stratum"],L0["rank"]); c[(kb(L["n_knots"]),"check ok" if L["ref_check"] else "check FAILED")]+=1
print("table 2 tree understates by knots and reference check:"); [print("   ",k,v) for k,v in sorted(c.items())]
c=collections.Counter()
for L0 in de["leaves"]:
    if L0["scored"]:
        L=leaf(L0["stratum"],L0["rank"]); c[(kb(L["n_knots"]),"ok" if L["ref_check"] else "FAILED")]+=1
print("table 2 all scored leaves:"); [print("   ",k,v) for k,v in sorted(c.items())]
