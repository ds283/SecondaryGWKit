import json, os, sys, hashlib, importlib.util
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
v,sid=sys.argv[1],sys.argv[2]
vj=json.load(open(R+"/out/variants/variants.json"))["variants"][v]
assert hashlib.sha256(open(vj["path"],"rb").read()).hexdigest()==vj["sha256"]
import AdaptiveLevin
spec=importlib.util.spec_from_file_location("AdaptiveLevin.levin_quadrature",vj["path"])
LQ=importlib.util.module_from_spec(spec); sys.modules["AdaptiveLevin.levin_quadrature"]=LQ; AdaptiveLevin.levin_quadrature=LQ; spec.loader.exec_module(LQ)
assert LQ._VARIANT_07D==v
sys.path.insert(1,R+"/scripts")
import ray.cloudpickle as cp
man={}
for p in ["calls_07d.json","calls_07d_r1x.json","calls_07d_ra.json"]: man.update(json.load(open(f"{R}/state/{p}"))["calls"])
c=man[sid]; fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
rec=cp.loads(b); kw=dict(rec["kw"])
if c["override"]:
    tr=json.load(open(f'{R}/out/replays/tree/{sid.replace(":","_").replace("|","_")}.json'))
    kw["rtol"]=c["override"]["rtol"]; kw["atol"]=c["override"]["atol"] if c["override"].get("atol") is not None else (float.fromhex(tr["atol"]) if isinstance(tr["atol"],str) else tr["atol"])
    print("override: rtol",kw["rtol"],"atol",kw["atol"])
res=LQ.adaptive_levin_sincos(rec["x_span"],rec["f"],rec["theta"],**kw)
mine={"value":float(res["value"]).hex(),"abserr":float(res["abserr"]).hex(),"num_regions":int(res["num_regions"]),"max_depth":max(r.depth for r in res["regions"]),"converged":bool(res["converged"])}
ag=json.load(open(f'{R}/out/replays/{v}/{sid.replace(":","_").replace("|","_")}.json'))["replay"]
print(v,sid,"SAME as the agent's replay" if all(mine[k]==ag[k] for k in mine) else f"DIFFERS mine {mine} agent {ag}", {k:res[k] for k in res if k.startswith("num_07d_")})
