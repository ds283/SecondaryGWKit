"""
Orchestrator's own recomputation of delta-theta at one point of a captured Bessel phase group,
prompt 06h §5 H1. Loads the capture blob named in the phase agent's capture_<tree>.json (checking
its SHA-256), finds the BesselPhaseGroup in it, evaluates the code's theta_mod_2pi and
theta_abserr at u, and compares with sum_i s_i (arg H1_nu_i(m_i e^u) + pi/2) in mpmath at the
exact e^u (u the double given), at 30 and 50 digits.

Usage: orch_dtheta.py CAPTURE_JSON UNIT INDEX U_HEX
"""

import hashlib
import json
import sys
from pathlib import Path


def find_groups(obj, seen, out, depth=0):
    from LiouvilleGreen.three_bessel_integrals import BesselPhaseGroup

    if id(obj) in seen or depth > 8:
        return
    seen.add(id(obj))
    if isinstance(obj, BesselPhaseGroup):
        out.append(obj)
        return
    if isinstance(getattr(obj, "__self__", None), BesselPhaseGroup):
        out.append(obj.__self__)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            find_groups(v, seen, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            find_groups(v, seen, out, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            find_groups(v, seen, out, depth + 1)
    elif callable(obj) and getattr(obj, "__closure__", None):
        for c in obj.__closure__:
            try:
                find_groups(c.cell_contents, seen, out, depth + 1)
            except ValueError:
                pass


def main():
    import mpmath as mp
    import ray.cloudpickle as cp

    cap_json, unit, index, u_hex = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    u = float.fromhex(u_hex)
    cap = [e for e in json.loads(Path(cap_json).read_text()) if e["unit"] == unit and e["index"] == index]
    assert len(cap) == 1, len(cap)
    cap = cap[0]
    with open(Path(cap["dir"]) / cap["pickle"], "rb") as f:
        f.seek(cap["offset"])
        blob = f.read(cap["length"])
    assert hashlib.sha256(blob).hexdigest() == cap["sha256"]
    obj = cp.loads(blob)
    groups = []
    find_groups(obj, set(), groups)
    print(f"capture {unit} #{index}: blob sha256 ok; top-level {type(obj).__name__}; groups found {len(groups)}")
    uniq = {(g.coefficients, g.signs, tuple(p.nu for p in g.phases)): g for g in groups}
    for key, g in uniq.items():
        print(f"group coefficients {key[0]} signs {key[1]} nu {key[2]}; K {g.K!r} K_lo {getattr(g, "K_lo", None)!r}")
        th = g.theta_mod_2pi(u)
        ae = g.theta_abserr(u)
        print(f"  u {u_hex} = {u!r}; code theta_mod_2pi {th!r}; theta_abserr {ae!r}")
        for dps in (30, 50):
            mp.mp.dps = dps
            x = mp.exp(mp.mpf(u))
            ref = mp.mpf(0)
            for phase, s, m in zip(g.phases, g.signs, g.coefficients):
                h = mp.hankel1(mp.mpf(phase.nu), mp.mpf(m) * x)
                ref += mp.mpf(s) * (mp.arg(h) + mp.pi / 2)
            d = mp.mpf(th) - ref
            d = d - 2 * mp.pi * mp.nint(d / (2 * mp.pi))
            print(f"  dps {dps}: delta_theta {mp.nstr(d, 17)}; |delta|/theta_abserr {mp.nstr(abs(d) / ae, 10)}")


if __name__ == "__main__":
    main()
