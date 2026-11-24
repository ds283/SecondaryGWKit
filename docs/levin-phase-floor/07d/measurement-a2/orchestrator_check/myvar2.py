import json, os, sys, hashlib, importlib.util, glob
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement-a2"; PA=M+"/var/levin-phase-floor-07d/measurement"
v,sid=sys.argv[1],sys.argv[2]
vj=json.load(open(R+"/out/variants/variants.json"))["variants"][v]
assert hashlib.sha256(open(vj["path"],"rb").read()).hexdigest()==vj["sha256"]
import AdaptiveLevin
spec=importlib.util.spec_from_file_location("AdaptiveLevin.levin_quadrature",vj["path"])
LQ=importlib.util.module_from_spec(spec); sys.modules["AdaptiveLevin.levin_quadrature"]=LQ; AdaptiveLevin.levin_quadrature=LQ; spec.loader.exec_module(LQ)
assert LQ._VARIANT_A2==v
sys.path.insert(1,R+"/scripts")
import ray.cloudpickle as cp
fn=sid.replace(":","_").replace("|","_")
ag=json.load(open(f'{R}/out/replays/{v}/{fn}.json'))
c=ag["header"]["capture"]; fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
rec=cp.loads(b); kw=dict(rec["kw"])
kw["rtol"]=float.fromhex(ag["rtol"]); kw["atol"]=float.fromhex(ag["atol"]) if isinstance(ag["atol"],str) else ag["atol"]
res=LQ.adaptive_levin_sincos(rec["x_span"],rec["f"],rec["theta"],**kw)
mine={"value":float(res["value"]).hex(),"abserr":float(res["abserr"]).hex(),"num_regions":int(res["num_regions"]),"max_depth":max(r.depth for r in res["regions"]),"converged":bool(res["converged"])}
a=ag["replay"]
print(v,sid,"SAME as the agent's replay" if all(mine[k]==a[k] for k in mine) else f"DIFFERS mine {mine} agent {a}", "| tree", ag["tree_replay"]["value"], ag["tree_replay"]["abserr"], {k:res[k] for k in res if k.startswith("num_a2_")})
