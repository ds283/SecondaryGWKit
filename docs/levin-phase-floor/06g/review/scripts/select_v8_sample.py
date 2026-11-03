"""Select the V8 reference sample (the user's ruling of 2026-11-01, relayed by the laptop).

Rule: from census/out/v8_calls.json, at each rtol, take the lost calls outside the 720
stationary-point rows whose class is "(b), refined to the depth cap" or "(b), below the depth cap"
(node_limited excluded). Sort by abserr_g / request_g ascending, ties broken by (unit, call index).
With n calls, take positions round(q * (n - 1)) for q = 0, 0.2, 0.4, 0.6, 0.8, 1.
Python's round (banker's rounding on .5) is used as written here; positions are printed.
"""

import hashlib, json, sys

src = sys.argv[1]
out_json = sys.argv[2]
raw = open(src, "rb").read()
d = json.loads(raw)
QS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sel = {"source": src, "source_sha256": hashlib.sha256(raw).hexdigest(), "rule": __doc__,
       "samples": []}
for rtol in ("1e-7", "1e-8"):
    calls = [x for x in d["lost"] if x["rtol"] == rtol and not x["stationary_point_row"]
             and x["class"].startswith("(b)")]
    def ratio(x):
        return float.fromhex(x["g"]["abserr"]) / x["request_g"]
    calls.sort(key=lambda x: (ratio(x), x["unit"], x["g"]["index"]))
    n = len(calls)
    print(f"rtol {rtol}: {n} eligible calls")
    for q in QS:
        pos = round(q * (n - 1))
        x = calls[pos]
        s = {"rtol": rtol, "q": q, "position": pos, "n": n, "unit": x["unit"], "row": x["row"],
             "call_index": x["g"]["index"], "group": x["g"]["group"], "kind": x["g"]["kind"],
             "class": x["class"], "ratio_abserr_over_request_g": ratio(x),
             "g": {k: x["g"][k] for k in ("value", "abserr", "converged", "num_regions", "max_depth", "has_cc_leaf")},
             "06h": {k: x["06h"][k] for k in ("value", "abserr", "converged", "num_regions", "max_depth", "has_cc_leaf")},
             "request_g": x["request_g"], "request_06h": x["request_06h"],
             "notify_label": x["g"]["notify_label"]}
        sel["samples"].append(s)
        print(f"  q={q:.1f} pos={pos:4d} {x['unit']} #{x['g']['index']} {x['g']['group']} "
              f"ratio={ratio(x):.3e} regions {x['06h']['num_regions']}->{x['g']['num_regions']}")
json.dump(sel, open(out_json, "w"), indent=1)
