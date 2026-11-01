"""levin-phase-floor prompt 06h section 5, the phase re-measurement: the check of the reference at 50
digits. Copied from 06f's review script docs/levin-phase-floor/06f/review/scripts/orch_m2.py (a
record, not edited; its copy is in state/scripts_06f_reference/review_scripts/), whose method --
the Hankel formula, the exact e^u, 30 and 50 digits -- is used as it is. Changed, each change
marked "prompt 06h", only in:
- its paths: 06h's phase directory, and each tree's captures and replays (capture_<tree>.json,
  out/replay_<tree>/), the tree read from the tag's prefix (``h-...`` or ``p-...``);
- the points of ``nodes``: the replay's nodes **and** its dense sample (dense.json.gz), so that
  the worst point may be either; the stored error compared with is the one at the exact e^u
  (replay06f.py's ``delta_theta``), with 06f's at x_hat beside it;
- a mode ``worst``: over every call of a tree, the point with the largest |delta theta| /
  theta_abserr, and ``nodes`` run on its call with that point first.
06f's review text follows.

06f review, check 4 (the orchestrator's own): M2 re-checked on captured calls, independently of
replay06f.py's m2().

(a) delta theta at chosen nodes of one call. Production's group.theta_mod_2pi(u), recomputed here,
    against a reference built another way than the agent's: each constituent's Liouville-Green
    phase as arg(H1_nu(y)) + pi/2 (mpmath hankel1; arg(J + iY) = phi with J = M cos phi,
    Y = M sin phi, and theta_nu = phi + pi/2 so that sin theta = J/M, cos theta = -Y/M), composed with
    the group's signs, reduced against 2 pi. At 30 and at 50 digits, and with two arguments:
    (i) y_i = c_i * x_hat exactly, x_hat = np.exp(u) the double production evaluates at (the
    agent's convention); (ii) y_i = c_i * exp(u) exactly (the rounding of exp(u) counted as phase
    error too). Compared with the agent's nodes.json.gz delta_theta and theta_abserr at the node.

(b) delta I for a call by a method other than the agent's (Gauss-Legendre in u with doubling):
    the Levin driver itself, adaptive_levin_sincos on the integrand
    s * theta_abserr(u) * (f0 cos theta - f1 sin theta) = [-s ta f1] sin theta + [s ta f0] cos theta,
    with the call's phase dict minus its theta_abserr key (so no declared term), at rtol 1e-10 and
    1e-12, split at the theta_abserr kinks the agent records. Compared with result.json M2 dI_plus /
    dI_minus.

Usage: PYTHONPATH=<tree g>:<06f scripts> python -B orch_m2.py nodes TAG K   |   dI TAG [TAG ...]
(prompt 06h: PYTHONPATH=<tree T>:<scripts> python -B orch_m2.py nodes TAG K | worst T K | dI TAG ...)"""

import gzip
import hashlib
import json
import math
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
V = MAIN / "var/levin-phase-floor-06h/phase"  # prompt 06h (06f: var/levin-phase-floor-06f)


def rdir(tag):
    """prompt 06h: the replay directory of a tag, by its tree prefix."""
    return V / f"out/replay_{tag.split('-', 1)[0]}" / tag


def load(tag):
    import ray.cloudpickle as cp

    unit, index = tag.rsplit("_", 1)
    tree = unit.split("-", 1)[0]  # prompt 06h
    cap = [e for e in json.loads((V / f"out/capture_{tree}.json").read_text()) if e["unit"] == unit and e["index"] == int(index)][0]
    with open(Path(cap["dir"]) / cap["pickle"], "rb") as f:
        f.seek(cap["offset"])
        blob = f.read(cap["length"])
    assert hashlib.sha256(blob).hexdigest() == cap["sha256"]
    return cp.loads(blob), json.loads((rdir(tag) / "result.json").read_text())  # prompt 06h


def ref_phase(mp, group, y_of_i):
    tot = mp.mpf(0)
    for i, (phase, s) in enumerate(zip(group.phases, group.signs)):
        h = mp.hankel1(mp.mpf(phase.nu), y_of_i(i))
        tot += mp.mpf(s) * (mp.arg(h) + mp.pi / 2)
    return tot


def wrap(mp, d):
    return d - 2 * mp.pi * mp.nint(d / (2 * mp.pi))


def nodes(tag, k):
    import mpmath as mp

    rec, res = load(tag)
    group = rec["theta"]["theta_mod_2pi"].__self__
    # prompt 06h: the nodes and the dense sample, each row marked with its set
    rows = [dict(r, set="node") for r in json.load(gzip.open(rdir(tag) / "nodes.json.gz", "rt"))]
    rows += [dict(r, set="dense", leaf=None, k=r["j"]) for r in json.load(gzip.open(rdir(tag) / "dense.json.gz", "rt"))]
    viol = sorted(rows, key=lambda r: -(r["ratio"] or 0))[:3]
    rng = random.Random(20261031)
    rest = [r for r in rows if r not in viol]
    pick = viol + rng.sample(rest, k - 3)
    print(f"== {tag}: {len(rows)} points in the agent's tables (nodes and dense); 3 largest ratios and {k - 3} drawn (seed 20261031)")  # prompt 06h
    print("   group coefficients", group.coefficients, "signs", group.signs, "nu", [p.nu for p in group.phases])
    for r in pick:
        u = float.fromhex(r["u"])
        th = float(group.theta_mod_2pi(u))
        ta = float(group.theta_abserr(u))
        xh = float(np.exp(u))
        out = {"set": r["set"], "leaf": r["leaf"], "k": r["k"], "u": r["u"], "x_hat": xh.hex(), "K_x_hat": group.K * xh,  # prompt 06h: set
               "theta_mod_2pi_equal_agent": th.hex() == r["theta_mod_2pi"],
               "theta_abserr_mine": ta, "theta_abserr_agent": r["theta_abserr"], "agent_delta_theta": r["delta_theta"],
               "agent_delta_theta_float": r.get("delta_theta_float"), "agent_ref_exact_mod_2pi": r.get("ref_exact_mod_2pi"),
               "theta_abserr_parts": r.get("theta_abserr_parts")}  # prompt 06h
        for dps in (30, 50):
            mp.mp.dps = dps
            xe = mp.exp(mp.mpf(u))
            d_hat = wrap(mp, mp.mpf(th) - ref_phase(mp, group, lambda i: mp.mpf(group.coefficients[i]) * mp.mpf(xh)))
            d_exa = wrap(mp, mp.mpf(th) - ref_phase(mp, group, lambda i: mp.mpf(group.coefficients[i]) * xe))
            out[f"dtheta_xhat_{dps}"] = float(d_hat)
            out[f"dtheta_exact_exp_{dps}"] = float(d_exa)
        out["ratio_xhat_30"] = abs(out["dtheta_xhat_30"]) / ta
        out["ratio_exact_exp_30"] = abs(out["dtheta_exact_exp_30"]) / ta
        # prompt 06h: the agent's delta_theta is now at the exact e^u, and 06f's (at x_hat) beside it
        out["rel_diff_mine_vs_agent"] = abs(out["dtheta_exact_exp_30"] - r["delta_theta"]) / abs(r["delta_theta"]) if r["delta_theta"] else None
        out["rel_diff_mine_vs_agent_50"] = abs(out["dtheta_exact_exp_50"] - r["delta_theta"]) / abs(r["delta_theta"]) if r["delta_theta"] else None
        out["rel_diff_mine_vs_agent_xhat"] = (abs(out["dtheta_xhat_30"] - r["delta_theta_float"]) / abs(r["delta_theta_float"])
                                              if r.get("delta_theta_float") else None)
        print(json.dumps(out))


def worst(tree, k):
    """prompt 06h: the point of largest |delta theta| / theta_abserr over every call of ``tree``
    (nodes and dense), then ``nodes`` on its call."""
    best = None
    for d in sorted((V / f"out/replay_{tree}").iterdir()):
        for name, kind in (("nodes.json.gz", "node"), ("dense.json.gz", "dense")):
            for r in json.load(gzip.open(d / name, "rt")):
                if r["ratio"] is not None and (best is None or r["ratio"] > best[0]):
                    best = (r["ratio"], d.name, kind, r["u"])
    print(f"== worst over {tree}: ratio {best[0]!r} on {best[1]} ({best[2]}, u {best[3]})")
    nodes(best[1], k)


def dI(tag):
    import AdaptiveLevin.levin_quadrature as LQ

    rec, res = load(tag)
    m2 = res["M2"]
    theta = dict(rec["theta"])
    group = theta["theta_mod_2pi"].__self__
    theta.pop("theta_abserr", None)
    f0, f1 = rec["f"]

    def ta(u):
        if np.ndim(u) == 0:
            return float(group.theta_abserr(float(u)))
        return np.array([float(group.theta_abserr(float(x))) for x in np.ravel(u)]).reshape(np.shape(u))

    def F(fun):
        def g(u):
            if np.ndim(u) == 0:
                return float(fun(float(u)))
            return np.array([float(fun(float(x))) for x in np.ravel(u)]).reshape(np.shape(u))

        return g

    a0, b0 = (float(x) for x in rec["x_span"])
    cuts = [a0] + sorted(set(k for k in m2["theta_abserr_kinks_u"] if a0 < k < b0)) + [b0]  # (dI: not used by 06h)
    print(f"== {tag}: span ({a0!r}, {b0!r}), cuts {cuts[1:-1]}, agent dI_plus {m2['dI_plus']!r} dI_minus {m2['dI_minus']!r}"
          f" (converged rel {m2['dI_converged_rel']!r}); D {m2['D']!r}; sum|dI_leaf| {m2['sum_abs_dI_leaf']!r}")
    for rtol in (1e-10, 1e-12):
        tot = {}
        for s in (+1, -1):
            fs = [F(lambda u, s=s: -s * ta(u) * f1(u)), F(lambda u, s=s: s * ta(u) * f0(u))]
            parts = []
            for j in range(len(cuts) - 1):
                rr = LQ.adaptive_levin_sincos((cuts[j], cuts[j + 1]), fs, theta=theta, atol=1e-300, rtol=rtol)
                parts.append((float(rr["value"]), float(rr["abserr"]), bool(rr["converged"]), rr["num_regions"]))
            tot[s] = (math.fsum(p[0] for p in parts), math.fsum(p[1] for p in parts), all(p[2] for p in parts),
                      sum(p[3] for p in parts))
        mx = max(abs(tot[1][0]), abs(tot[-1][0]))
        print(f"  driver rtol {rtol:g}: dI(+) {tot[1][0]!r} +- {tot[1][1]:.2e} conv {tot[1][2]} regions {tot[1][3]};"
              f" dI(-) {tot[-1][0]!r} conv {tot[-1][2]}")
        print(f"     rel diff vs agent: + {abs(tot[1][0] - m2['dI_plus']) / abs(m2['dI_plus']):.2e},"
              f" - {abs(tot[-1][0] - m2['dI_minus']) / abs(m2['dI_minus']):.2e};"
              f" D/max|dI| mine {m2['D'] / mx!r} agent {m2['D_over_max_abs_dI']!r}")


def main():
    import AdaptiveLevin.levin_quadrature as LQ

    print("# host", socket.gethostname(), "utc", datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "tree", LQ.__file__.split("/AdaptiveLevin/")[0])
    print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if sys.argv[1] == "nodes":
        nodes(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "worst":  # prompt 06h
        worst(sys.argv[2], int(sys.argv[3]))
    else:
        for t in sys.argv[2:]:
            dI(t)


if __name__ == "__main__":
    main()
