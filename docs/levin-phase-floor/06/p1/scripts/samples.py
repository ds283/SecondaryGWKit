"""levin-phase-floor 06, finding P1, step 4 (sampling half): under one tree, after the validated
rebuild (p1lib), evaluate every ingredient of the all-oscillatory sub-interval's integrand at a
common, tree-independent set of points, for compare.py.

Points, per row, in x = log(1+z): NUNI uniform points over the sub-interval [x0, x1] (the
captured x_span), plus NDENSE uniform points in [x_s - HALF, x_s + HALF] clipped to [x0, x1],
with x_s the row's G+q+r stationary point as analyse.py found it under r13 (out/analyse_r13.json;
r14's agrees to every printed digit).

Recorded at each point (all float64, exactly as the callables return them):
  * each group: theta (raw), theta_mod_2pi, theta_deriv, f_sin, f_cos;
  * G: phase.raw_theta, phase.theta_mod_2pi, phase.phi, phase.theta_deriv (log), sin_amplitude;
  * Tq and Tr: phase.raw_theta, phase.theta_mod_2pi, phase.phi, M, dlnM_dz, omega;
  * Hubble(z) and wBackground(z), z = expm1(x);
  * the knots of each factor's phi spline (PrimitivePhase.z_samples, .phi_samples) and the raw
    phase at each knot.

Usage: PYTHONPATH=<tree> OMP_NUM_THREADS=1 python -B samples.py r13|r14
Writes out/samples_<tree>.npz.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from prov import OUT, header, sha256  # noqa: E402
import p1lib  # noqa: E402

NUNI = 20001
NDENSE = 8001
HALF = 0.05


def points(x0, x1, xs):
    uni = np.linspace(x0, x1, NUNI)
    dense = np.linspace(max(x0, xs - HALF), min(x1, xs + HALF), NDENSE)
    return np.unique(np.concatenate([uni, dense]))


def main():
    tree = sys.argv[1]
    log = lambda s: print(s, flush=True)  # noqa: E731
    ana = json.loads((OUT / "analyse_r13.json").read_text())
    data = {}
    with p1lib.rebuilt_rows(tree, log=log) as (ev, ctx, rows):
        for name, row in rows.items():
            if not row["validation"]["all_identical"]:
                raise SystemExit(f"p1: {tree} {name} does not reproduce its store; stopping")
            gc, groups, calls = p1lib.low_subinterval_groups(row)
            x0, x1 = calls["G+q+r"]["args"][0]
            sp = ana["rows"][name]["groups"]["G+q+r"]["phase"]["stationary_points"]
            xs = float.fromhex(sp[0]["x"]["hex"])
            X = points(x0, x1, xs)
            data[f"{name}|x"] = X
            data[f"{name}|x_s"] = np.array([xs])
            kw = gc["kwargs"]
            Gk, Tq, Tr, mf, wb = kw["Gk"], kw["Tq"], kw["Tr"], kw["model_functions"], kw["w_background"]
            cols = {}

            def put(key, fn):
                cols[key] = np.array([fn(float(x)) for x in X])

            for label, g in groups.items():
                put(f"{label}.theta", g.theta)
                put(f"{label}.theta_mod_2pi", g.theta_mod_2pi)
                put(f"{label}.theta_deriv", g.theta_deriv)
                put(f"{label}.f_sin", g.f_sin)
                put(f"{label}.f_cos", g.f_cos)
            put("G.raw_theta", lambda x: Gk.phase.raw_theta(x, x_is_log=True))
            put("G.theta_mod_2pi", lambda x: Gk.phase.theta_mod_2pi(x, x_is_log=True))
            put("G.phi", lambda x: Gk.phase.phi(x, x_is_log=True))
            put("G.theta_deriv", lambda x: Gk.phase.theta_deriv(x, x_is_log=True, log_derivative=True))
            put("G.sin_amplitude", lambda x: Gk.sin_amplitude(x, z_is_log=True))
            for nm, T in (("Tq", Tq), ("Tr", Tr)):
                put(f"{nm}.raw_theta", lambda x, T=T: T.phase.raw_theta(x, x_is_log=True))
                put(f"{nm}.theta_mod_2pi", lambda x, T=T: T.phase.theta_mod_2pi(x, x_is_log=True))
                put(f"{nm}.phi", lambda x, T=T: T.phase.phi(x, x_is_log=True))
                put(f"{nm}.M", lambda x, T=T: T.M(x, z_is_log=True))
                put(f"{nm}.dlnM_dz", lambda x, T=T: T.dlnM_dz(x, z_is_log=True))
                put(f"{nm}.omega", lambda x, T=T: T.omega(x, z_is_log=True))
            put("Hubble", lambda x: mf.Hubble(math.expm1(x)))
            put("wBackground", lambda x: wb(math.expm1(x)))
            for k, v in cols.items():
                data[f"{name}|{k}"] = v
            # the phi spline's knots, and the raw phase there, for compare.py's smoothness test
            for nm, P in (("G", Gk.phase), ("Tq", Tq.phase), ("Tr", Tr.phase)):
                zk = np.asarray(P.z_samples, dtype=float)
                data[f"{name}|knots.{nm}.z"] = zk
                data[f"{name}|knots.{nm}.phi"] = np.asarray(P.phi_samples, dtype=float)
                data[f"{name}|knots.{nm}.raw_theta"] = np.array([P.raw_theta(float(z)) for z in zk])
            log(f"   {tree} {name}: {len(X)} points, {len(cols)} columns")
    npz = OUT / f"samples_{tree}.npz"
    np.savez(npz, **data)
    hdr = OUT / f"samples_{tree}.header.txt"
    hdr.write_text("\n".join(header(Path(__file__), f"p1 samples, tree {tree}")) + "\n"
                   + f"# evidence {json.dumps(ev, default=str)}\n# npz {npz.name} sha256 {sha256(npz)}\n")
    log(f"wrote {npz} sha256 {sha256(npz)}")


if __name__ == "__main__":
    main()
