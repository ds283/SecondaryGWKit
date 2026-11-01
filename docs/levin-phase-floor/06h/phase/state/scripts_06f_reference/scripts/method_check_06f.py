"""levin-phase-floor prompt 06f section 3, item 1's method check. New in 06f.

The comparison item 1 makes (production's phase against atan2(J, -Y) from mpmath at 30 digits) is
made here on one constituent at one argument, where LiouvilleGreen/bessel_phase.py documents the
error: EVALUATION_FLOOR's comment, "at nu = 1/2 the residual is identically zero and the series is
exact at every x, yet the angle addition of BesselPhaseFunction.sin_cos_theta still performs four
multiplications and an addition, and the measured phase-pair error against the committed 40-digit
corners is 1.11e-16 -- half an ulp of unity".

From one captured M2 call's group, its nu = 1/2 constituent is taken, at the argument c_0 eta_hat of
the call's lower endpoint (eta_hat = np.exp(u), as production forms it) and at x = 1e3, and:
- the phase-pair error max(|sin_p - sin t|, |cos_p - cos t|), with (sin_p, cos_p) from
  sin_cos_theta(x) and t = atan2(J_nu(x), -Y_nu(x)) at 30 digits: documented as at most 1.11e-16;
- the error of theta_mod_2pi(x) itself, reduced into (-pi, pi], which is what item 1 compares;
- the same reference at 50 digits, to show 30 are enough at these arguments.
The check passes if the phase-pair error is at most 1.11e-16 at both arguments and the 30- and
50-digit references agree to 1e-25. The same is printed, descriptively, for the call's nu = 5/2
constituent (near region, declared 5e-13) and for the group at the call's lower endpoint.

Usage: PYTHONPATH=<tree g>:<scripts> python -B method_check_06f.py CAPTURE_JSON UNIT INDEX
"""

import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import TREES, sha256, tree_state  # noqa: E402
from replay06f import load_capture, mp_group_phase, wrap  # noqa: E402

DOCUMENTED = 1.11e-16


def ref_theta(mp, nu, x, dps):
    with mp.workdps(dps):
        y = mp.mpf(float(x))
        return mp.atan2(mp.besselj(mp.mpf(nu), y), -mp.bessely(mp.mpf(nu), y))


def main():
    import mpmath as mp

    import AdaptiveLevin.levin_quadrature as LQ

    cap_json, uid, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    head, status = tree_state(tree)
    print("# levin-phase-floor 06f: item 1's method check")
    print(
        f"# written {datetime.now(timezone.utc).isoformat(timespec='seconds')} on {socket.gethostname()}"
    )
    print(
        f"# script method_check_06f.py sha256 {sha256(Path(__file__).resolve())}; replay06f.py sha256 {sha256(HERE / 'replay06f.py')}"
    )
    print(
        f"# tree {tree} HEAD {head} (status --short: {'empty' if not status.strip() else repr(status)}); expected {TREES['g']}"
    )
    print(f"# input {cap_json} sha256 {sha256(cap_json)}; call {uid} #{index}")
    rec, cap = load_capture(cap_json, uid, index)
    print(f"# capture blob sha256 {cap['sha256']}")
    group = rec["theta"]["theta_mod_2pi"].__self__
    mp.mp.dps = 30
    u0 = float(rec["x_span"][0])
    eta_hat = np.exp(u0)
    ok = True
    for i, (phase, c) in enumerate(zip(group.phases, group.coefficients)):
        print(
            f"\nconstituent {i}: nu = {phase.nu}, x_star = {phase.x_star}, near region {phase._corrections.near is not None}, "
            f"declared scalar theta_abserr {phase.theta_abserr:.3e}"
        )
        for x in (float(c * eta_hat), 1e3):
            s_p, c_p = phase.sin_cos_theta(x)
            t30 = ref_theta(mp, phase.nu, x, 30)
            t50 = ref_theta(mp, phase.nu, x, 50)
            with mp.workdps(30):
                pair = max(
                    abs(mp.mpf(float(s_p)) - mp.sin(t30)),
                    abs(mp.mpf(float(c_p)) - mp.cos(t30)),
                )
                dth = wrap(mp, mp.mpf(float(phase.theta_mod_2pi(x))) - t30)
            with mp.workdps(50):
                ref_diff = abs(wrap(mp, t30 - t50))
            ta = float(phase.theta_abserr_at(x))
            print(
                f"   x = {x!r}: phase-pair error {float(pair):.4e}; theta_mod_2pi error {float(dth):+.4e}; "
                f"declared theta_abserr_at(x) {ta:.4e}; |ref(30) - ref(50)| {float(ref_diff):.2e}"
            )
            if phase.nu == 0.5:
                passed = float(pair) <= DOCUMENTED and float(ref_diff) < 1e-25
                ok &= passed
                print(
                    f"      nu = 1/2, documented phase-pair error 1.11e-16: measured {float(pair):.4e} -> {'reproduces' if passed else 'DOES NOT reproduce'}"
                )
    with mp.workdps(30):
        g30 = mp_group_phase(group, eta_hat, mp)
        d30 = wrap(mp, mp.mpf(float(group.theta_mod_2pi(u0))) - g30)
    with mp.workdps(50):
        g50 = mp_group_phase(group, eta_hat, mp)
        d50 = wrap(mp, mp.mpf(float(group.theta_mod_2pi(u0))) - g50)
    print(
        f"\ngroup at the lower endpoint u = {u0!r}: delta theta at 30 digits {float(d30):+.6e}, at 50 digits {float(d50):+.6e}; "
        f"theta_abserr {float(group.theta_abserr(u0)):.4e}"
    )
    ok &= abs(float(d30) - float(d50)) < 1e-25
    print(f"\nmethod check: {'passes' if ok else 'FAILS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
