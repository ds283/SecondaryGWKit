"""levin-phase-floor prompt 06j: a development check of probe_s.py's and seams_06j.py's functions on a call
outside 06j's sample, before they are run on it. New in 06j; descriptive only, scores nothing.

The call is 06g's references' capture v00 (g-1e-7-2-1380 #4, G+q-r, one of V8's lost calls; its pickle under
var/levin-phase-floor-06g/references/state/inputs/, read only, SHA-256 printed). On its first two scoped slow
leaves (v8_slow_leaves.json) it runs the leaf rule on the leaf's own span (which must reproduce 06g's census
delta1, delta2, r and phi bit for bit), one draw of each S-c set, the seam test, and one S-e stencil if a seam is
held; and the seams' merge against the tool's production_edges. Prints timings.

Usage: PYTHONPATH=<tree>:<this directory> python -B dev_probe.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_s as PS  # noqa: E402
import seams_06j as SM  # noqa: E402

PKL = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/state/inputs/v02.pkl")  # v02 (g-1e-7-3-34212 #4): v00 has no scoped leaf (out/cmd/14-16)
LEAVES = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/census/out/v8_slow_leaves.json")


def main():
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ

    print(f"# input {PKL} sha256 {hashlib.sha256(PKL.read_bytes()).hexdigest()}")
    print(f"# input {LEAVES} sha256 {hashlib.sha256(LEAVES.read_bytes()).hexdigest()}")
    rec = cp.loads(PKL.read_bytes())
    print(rec["unit"], rec["index"], rec["x_span"], rec["kw"])
    t0 = time.perf_counter()
    sm, nt = SM.seams(rec)
    print("seams", len(sm), "tables", nt, "check", SM.check_against_tool(rec, sm, nt), f"{time.perf_counter() - t0:.2f} s")
    sm_x = np.array(sorted({s["x"] for s in sm}))
    lv = [e for e in json.loads(LEAVES.read_text()) if e["unit"] == rec["unit"] and e["index"] == rec["index"]][:2]
    probe = PS.Probe(LQ, rec)
    for e in lv:
        a, b = e["start"], e["end"]
        t1 = time.perf_counter()
        r = probe.rule(a, b)
        print(f"leaf [{a!r}, {b!r}]: probe r {r['r']!r} d1 {r['delta1']!r} phi {r['phi']!r}; census r {e['r']!r} d1 {e['d1']!r} phi {e['phi']!r}; "
              f"equal {r['r'] == e['r'] and r['delta1'] == e['d1'] and r['phi'] == e['phi']}; is_direct {r['is_direct']}; {time.perf_counter() - t1:.2f} s")
        hs = SM.held(sm_x, a, b)
        print("  held seams", [(float(sm_x[j]), [s["kind"] for s in sm if s["x"] == sm_x[j]]) for j in hs],
              "distance" if not hs else "", SM.distance(sm_x, a, b) if not hs else "")
        t2 = time.perf_counter()
        pi, hi = PS.set_i(probe, a, b, 0)
        pii, hii = PS.set_ii(probe, a, b, 0)
        print(f"  S-c one draw: (i) dV_P {pi - r['V_P']!r} (ii) dV_P {pii - r['V_P']!r}; {time.perf_counter() - t2:.2f} s")
        if hs:
            xs = float(sm_x[hs[0]])
            st = PS.stencil(probe, xs, (b - a) * 2.0**-4)
            print("  S-e jump (value, d1, d2) F", st["F"]["jump"], "theta_mod_2pi", st["theta_mod_2pi"]["jump"])
    print("solves", probe.n_solves, f"total {time.perf_counter() - t0:.1f} s")
    print("null20", SM.null_fraction(sm_x, float(rec["x_span"][0]), float(rec["x_span"][1]), 20))


if __name__ == "__main__":
    main()
