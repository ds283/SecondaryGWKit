import json, os, glob
R=os.path.expanduser("~/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement")
for v,kind in [("Ea","|a"),("ET20","|g")]:
    best=None
    for f in sorted(glob.glob(f"{R}/out/replays/{v}/d_*.json")):
        d=json.load(open(f))
        if kind not in d["sid"]: continue
        if not d["wall_s"]<20: continue
        t=json.load(open(f.replace(f"/{v}/","/tree/")))["replay"]; r=d["replay"]
        if r["value"]!=t["value"] and (v=="Ea" or d["extra_07d"].get("num_07d_route_changed",0)>0) and d["wall_s"]<20:
            best=(d["sid"],r,t,d["extra_07d"]); break
    print(v,best[0],"variant",best[1],"tree",best[2],best[3])
