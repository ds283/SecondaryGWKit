"""levin-phase-floor 06b, W2 (descriptive): each sampled call's error against the reference, as a
fraction of the call's own request max(atol, rtol |value_b|), for b and for 06a. New in 06b.
Usage: python -B w2_request.py REF_JSON"""
import json
import sys
from pathlib import Path

ref = json.loads(Path(sys.argv[1]).read_text())["results"]
n_out_b = n_out_a = 0
print("unit                call err_b/abserr_b err_b/request err_06a/request abserr_b/request abserr_06a/request")
for r in ref:
    S = float.fromhex(r["ref"])
    vb, eb = float.fromhex(r["value_b"]), float.fromhex(r["abserr_b"])
    va, ea = float.fromhex(r["value_06a"]), float.fromhex(r["abserr_06a"])
    req = max(r["kw"]["atol"], r["kw"]["rtol"] * abs(vb))
    n_out_b += abs(vb - S) > req
    n_out_a += abs(va - S) > req
    print(f"{r['unit']:<19} {r['index']:>3} {abs(vb - S) / eb:14.3g} {abs(vb - S) / req:13.3g} {abs(va - S) / req:15.3g} {eb / req:16.3g} {ea / req:18.3g}")
print(f"calls whose error exceeds the request: b {n_out_b} of {len(ref)}; 06a {n_out_a} of {len(ref)}")
