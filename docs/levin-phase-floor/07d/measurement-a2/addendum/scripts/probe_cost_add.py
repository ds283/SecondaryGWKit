"""levin-phase-floor 07d Part A' addendum: the integrand's evaluation cost on three selected leaves (one group returned,
one analytic, one visited; the first of each in selection order), for the projection of A.2. Evaluates g on 48 x 64
points of each leaf's span and times it; reads no reference. Usage: PYTHONPATH=... python -B probe_cost_add.py"""
import json, sys, time
from pathlib import Path
import numpy as np
ADD = Path(__file__).resolve().parent.parent
sel = json.loads((ADD / "state" / "selection_add.json").read_text())
import AdaptiveLevin.levin_quadrature as LQ
import leafref_v4 as R4
picked = {}
for i, lv in sel["leaves"].items():
    k = f"{lv['set']}|{lv['kind']}"
    picked.setdefault(k, i)
out = {}
for k, i in picked.items():
    lv = sel["leaves"][i]
    doc = json.loads(Path(lv["file"]).read_text())
    spec = json.loads(Path(doc["header"]["spec"]).read_text())
    rec = R4.load_capture(spec["capture"])
    G = R4.Integrand(LQ, rec)
    a, b = float.fromhex(lv["start"]), float.fromhex(lv["end"])
    xs = np.linspace(a, b, 48 * 64)
    t0 = time.perf_counter(); G.g_on(xs); dt = time.perf_counter() - t0
    out[k] = {"leaf": i, "us_per_eval": 1e6 * dt / len(xs)}
n = sum(v["n_pieces"] for v in sel["leaves"].values())
us = max(v["us_per_eval"] for v in out.values())
ev_lo, ev_hi = n * 48 * (512 + 1024), n * 48 * (512 + 1024 + 2048)
proj = {"pieces": n, "evals_without_2048": ev_lo, "evals_all_2048": ev_hi, "us_per_eval_max": us,
        "cpu_s_lo": ev_lo * us * 1e-6, "cpu_s_hi": ev_hi * us * 1e-6,
        "wall_s_6_workers_lo": ev_lo * us * 1e-6 / 6, "wall_s_6_workers_hi": ev_hi * us * 1e-6 / 6, "probes": out}
(ADD / "out" / "projection_ref48.json").write_text(json.dumps(proj, indent=1))
print(json.dumps(proj, indent=1))
