"""Orchestrator's reader: z12/z00 and V1's 12 from the rerun readings, independently of score_refs.py."""
import json, os, itertools

R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/"
cap = {c["sid"]: c for c in json.load(open(R + "out/capture.json"))}


def readings(sid):
    out = {}
    for impl, res in itertools.product(("qawo", "fcc"), (1, 2)):
        f = f"{R}out/readings/{sid}-{impl}-r{res}-rerun.json"
        if os.path.exists(f):
            d = json.load(open(f))
            out[f"{impl}{res}"] = (float.fromhex(d["value_hex"]), d["value_hex"], (d.get("tmap") or {}).get("direct_segments"))
    return out


for sid in ("z00", "z12"):
    c = cap[sid]
    vg, eg = float.fromhex(c["g_record"]["value"]["hex"]), float.fromhex(c["g_record"]["abserr"]["hex"])
    vh, eh = float.fromhex(c["06h_record"]["value"]["hex"]), float.fromhex(c["06h_record"]["abserr"]["hex"])
    tol = min(eg, eh) / 100
    rd = readings(sid)
    vals = [v[0] for v in rd.values()]
    sp = max(vals) - min(vals)
    print(f"{sid} {c['g_unit']} #{c['index']} rtol {c['rtol']}: g {vg.hex()} e_g {eg:.3e}; 06h {vh.hex()} e_06h {eh:.3e}; tol {tol:.3e}")
    for k, v in rd.items():
        print(f"   {k} {v[1]} direct_segments {v[2]}  (v_g-v)/e_g {(vg - v[0]) / eg:+.2f}  (v_06h-v)/e_06h {(vh - v[0]) / eh:+.2f}")
    print(f"   spread {sp:.3e} = {sp / tol:.3f} x tol; four readings agree within tol: {sp <= tol}")

# V1's 12: the readings and R23/R24 values from the rehearsal agent's detail file
import re
det = open("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/rehearsal/out/v1_violations_detail.txt").read()
groups = re.findall(r"v23 (\S+) v24 (\S+) e23 (\S+) e24 (\S+);", det)
print("\nV1 detail groups parsed:", len(groups))
plan = json.load(open(R + "out/rerun_plan.json"))["plan"]
wsids = sorted({p["sid"] for p in plan if p["sid"].startswith("w")})
for sid in wsids:
    rd = readings(sid)
    print(f"{sid}: " + "  ".join(f"{k} {v[1]}" for k, v in rd.items()))
