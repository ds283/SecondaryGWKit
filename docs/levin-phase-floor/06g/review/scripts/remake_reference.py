"""Orchestrator's review: one new reference re-made another way.

FCC alone at a finer resolution than the tool's two, by adding a resolution 3 to the tool's own
tables in this process only (workers=1, so no spawned process re-imports the module):
Route B fit (M, K) = (28, 361) and FCC 5 t-pieces of N = 64 per segment; Route A FCC du = 0.085,
N = 64. The tool's file is not changed. Usage: remake_reference.py SID
"""
import json, sys, time, hashlib
import tools.filon_reference as F

sid = sys.argv[1]
R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/"
pkl = f"{R}state/inputs/{sid}.pkl"
F.ROUTE_B_FIT[3] = dict(M=28, K=361)
F.ROUTE_B_FCC[3] = dict(pieces=5, N=64)
F.ROUTE_A_FCC[3] = dict(du=0.085, N=64)
call = F._load_pickle(pkl, "ray.cloudpickle")
t0 = time.time()
r = F.reading(call, "fcc", 3, workers=1)
out = {k: r[k] for k in ("route", "value_hex", "value_30", "imag_30") if k in r}
out.update(sample=sid, pickle=pkl, pickle_sha256=hashlib.sha256(open(pkl, "rb").read()).hexdigest(),
           tool_sha256=hashlib.sha256(open(F.__file__, "rb").read()).hexdigest(), tool=F.__file__,
           config="res 3: Route B fit (28, 361), FCC pieces 5, N 64; Route A FCC du 0.085, N 64",
           wall_s=time.time() - t0, all_monotone=(r.get("tmap") or {}).get("all_monotone"))
print(json.dumps(out, indent=1))
