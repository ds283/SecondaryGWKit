import json, os, sys, hashlib, re
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit")
R=M+"/var/levin-phase-floor-07d/measurement"
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]:
    man.update(json.load(open(f"{R}/state/{p}"))["calls"])
sids=sys.argv[1:]
want={}
for s in sids: 
    c=man[s]["census"]; u,ix=c.split("|"); want.setdefault(u,[]).append((s,ix))
cen={}
for line in open(M+"/var/runs/levin-phase-floor-07c-census-20261119T201419/checkpoint.jsonl"):
    r=json.loads(line)
    if r["unit"] in want:
        d=r["data"]
        for s,ix in want[r["unit"]]:
            lst=d["calls"] if ix[0]=="g" else d["other_calls"]
            cen[s]=next(x for x in lst if x["index"]==int(ix[1:]))
sys.path.insert(1, R+"/scripts")
import ray.cloudpickle as cp
import AdaptiveLevin.levin_quadrature as LQ
assert hashlib.sha256(open(LQ.__file__,"rb").read()).hexdigest().startswith("d141a80b"), LQ.__file__
for s in sids:
    c=man[s]; blob=open(c["path"],"rb"); blob.seek(c["offset"]); b=blob.read(c["length"])
    assert hashlib.sha256(b).hexdigest()==c["sha256"]
    rec=cp.loads(b); kw=dict(rec["kw"]); cr=cen[s]
    if c["override"]:
        kw["rtol"]=c["override"]["rtol"]; kw["atol"]=c["override"]["atol"] if c["override"].get("atol") is not None else cr["atol"]
    res=LQ.adaptive_levin_sincos(rec["x_span"],rec["f"],rec["theta"],**kw)
    got=(float(res["value"]).hex(),float(res["abserr"]).hex(),int(res["num_regions"]),max(r.depth for r in res["regions"]),bool(res["converged"]),int(res["num_chase_stops"]),int(res["num_chase_stopped_leaves"]))
    exp=(cr["value"]["hex"],cr["abserr"]["hex"],cr["num_regions"],cr["max_depth"],bool(cr["converged"]),cr["num_chase_stops"],cr["num_chase_stopped_leaves"])
    print(s, "label ok" if rec["notify_label"]==cr["notify_label"] else "LABEL DIFFERS", "SAME" if got==exp else "DIFFERS", got, exp if got!=exp else "")
