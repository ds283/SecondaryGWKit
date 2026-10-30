"""Orchestrator's diagnostic (06c Part B review): g-1e-7-0-14127 call 3 re-run with atol tightened
as well as rtol, to see whether a tighter request moves the driver towards S10. Read-only."""
import hashlib, json, pickle
import AdaptiveLevin.levin_quadrature as LQ
C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/census/"
S10 = float.fromhex("-0x1.9c9fd7aecc143p-99")
e = next(x for x in json.load(open(C + "state/x1_captured.json")) if x["unit"] == "g-1e-7-0-14127" and x["index"] == 3)
with open(C + e["dir"] + "/" + e["pickle"], "rb") as f:
    f.seek(e["offset"]); blob = f.read(e["length"])
assert hashlib.sha256(blob).hexdigest() == e["sha256"]
rec = pickle.loads(blob)
eg = float.fromhex(e["census_abserr"])
print("levin_quadrature from", LQ.__file__, "; call atol", rec["kw"]["atol"], "rtol", rec["kw"]["rtol"])
for atol, rtol in [(rec["kw"]["atol"], 1e-7), (1e-40, 1e-7), (1e-42, 1e-9), (1e-45, 1e-11)]:
    kw = dict(rec["kw"]); kw["atol"] = atol; kw["rtol"] = rtol
    r = LQ.adaptive_levin_sincos(tuple(rec["x_span"]), rec["f"], theta=rec["theta"], **kw)
    print(f"atol {atol:.3g} rtol {rtol:.0e}: value {r['value'].hex()} abserr {r['abserr']:.4g} regions {r.get('num_regions')} "
          f"max_depth {r.get('max_depth')} converged {r.get('converged')} untrusted {r.get('num_untrusted_regions')}; "
          f"|v-S10| = {abs(r['value']-S10):.4g} = {abs(r['value']-S10)/eg:.4g} abserr_g; |v-S10|/abserr = {abs(r['value']-S10)/r['abserr']:.4g}")
