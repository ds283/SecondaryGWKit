import json,os
os.chdir(os.path.expanduser("~/Documents/Code/SecondaryGWKit"))
want={"h-1e-7-0-12636","h-1e-8-0-2275","h-1e-8-1-10922"}
for line in open("var/runs/levin-phase-floor-07c-census-20261119T201419/checkpoint.jsonl"):
    r=json.loads(line)
    if r["unit"] in want:
        d=r["data"]; print(r["unit"], sorted(d.keys()))
        for k,v in d.items():
            if isinstance(v,list) and v and isinstance(v[0],dict): print(" ",k,len(v),sorted(v[0].keys())[:60]); print("  ",json.dumps(v[0])[:800])
            elif isinstance(v,dict) and k not in("scripts_sha256",): print(" ",k,json.dumps(v)[:300])
        want.discard(r["unit"])
