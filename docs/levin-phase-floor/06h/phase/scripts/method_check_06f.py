"""levin-phase-floor prompt 06h section 5, the phase re-measurement's method check: copied from prompt
06f's scripts/method_check_06f.py (a record, not edited; its copy is in
state/scripts_06f_reference/scripts/), adapted to the reference at the exact e^u (prompt 06h
section 2), each change marked "prompt 06h":
- the tree is the one on PYTHONPATH, one of prov.TREES, and the capture JSON is that tree's;
- the group's check at the call's lower endpoint is made at the **exact** e^u (mp.exp of the
  double u at the working precision), at 30 and at 50 digits, which must agree to 1e-25 (06f: at
  the double x_hat); the error at x_hat is printed beside it;
- the second reference formula, arg H1_nu(y) + pi/2 (06f's review, orch_m2.py; prompt 06h section 2
  test 7), is compared with atan2(J, -Y) at 30 digits: at each constituent's two arguments, and for
  the group at the exact e^u. They must agree to 1e-25;
- at h the declaration's three parts at the endpoint are printed.
The constituents' checks are at their own double arguments, as 06f's are: that is where
bessel_phase.py documents each one's error. 06f's text follows.

levin-phase-floor prompt 06f section 3, item 1's method check. New in 06f.

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
(prompt 06h: PYTHONPATH=<tree T>:<scripts>, CAPTURE_JSON that tree's)
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


def ref_theta_hankel(mp, nu, y):
    """prompt 06h: the other formula, arg H1_nu(y) + pi/2 (06f's review, orch_m2.py's ref_phase),
    at the working precision, y an mpmath number taken as it is."""
    return mp.arg(mp.hankel1(mp.mpf(nu), y)) + mp.pi / 2


def group_ref_hankel(mp, group, eta):
    """prompt 06h: the group phase by the Hankel formula, at the abscissa eta (an mpmath number)."""
    tot = mp.mpf(0)
    for phase, c, s in zip(group.phases, group.coefficients, group.signs):
        tot += mp.mpf(s) * ref_theta_hankel(mp, phase.nu, mp.mpf(c) * eta)
    return tot


def main():
    import mpmath as mp

    import AdaptiveLevin.levin_quadrature as LQ

    cap_json, uid, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    head, status = tree_state(tree)
    print(
        "# levin-phase-floor 06h: the phase re-measurement's method check (06f's, at the exact e^u)"
    )  # prompt 06h
    print(
        f"# written {datetime.now(timezone.utc).isoformat(timespec='seconds')} on {socket.gethostname()}"
    )
    print(
        f"# script method_check_06f.py sha256 {sha256(Path(__file__).resolve())}; replay06f.py sha256 {sha256(HERE / 'replay06f.py')}"
    )
    names = [n for n, (t, _s) in TREES.items() if str(t) == tree]  # prompt 06h
    print(
        f"# tree {tree} HEAD {head} (status --short: {'empty' if not status.strip() else repr(status)}); expected "
        f"{TREES[names[0]] if len(names) == 1 else 'NOT ONE OF prov.TREES'}"
    )
    if len(names) != 1 or head != TREES[names[0]][1] or status.strip():
        print("tree evidence does not hold")
        return 1
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
            with mp.workdps(30):  # prompt 06h: the other formula
                h30 = ref_theta_hankel(mp, phase.nu, mp.mpf(float(x)))
                formulas = abs(wrap(mp, t30 - h30))
            ok &= float(formulas) < 1e-25  # prompt 06h
            ta = float(phase.theta_abserr_at(x))
            print(
                f"   x = {x!r}: phase-pair error {float(pair):.4e}; theta_mod_2pi error {float(dth):+.4e}; "
                f"declared theta_abserr_at(x) {ta:.4e}; |ref(30) - ref(50)| {float(ref_diff):.2e}; "
                f"|atan2(J,-Y) - (arg H1 + pi/2)| {float(formulas):.2e}"  # prompt 06h
            )
            if phase.nu == 0.5:
                passed = float(pair) <= DOCUMENTED and float(ref_diff) < 1e-25
                ok &= passed
                print(
                    f"      nu = 1/2, documented phase-pair error 1.11e-16: measured {float(pair):.4e} -> {'reproduces' if passed else 'DOES NOT reproduce'}"
                )
    # prompt 06h: the group at the exact e^u (06f: at eta_hat), at 30 and 50 digits, by both formulas
    with mp.workdps(30):
        g30 = mp_group_phase(group, mp.exp(mp.mpf(u0)), mp)
        d30 = wrap(mp, mp.mpf(float(group.theta_mod_2pi(u0))) - g30)
        h30 = group_ref_hankel(mp, group, mp.exp(mp.mpf(u0)))
        group_formulas = abs(wrap(mp, g30 - h30))
        d30_hat = wrap(
            mp,
            mp.mpf(float(group.theta_mod_2pi(u0))) - mp_group_phase(group, eta_hat, mp),
        )
    with mp.workdps(50):
        g50 = mp_group_phase(group, mp.exp(mp.mpf(u0)), mp)
        d50 = wrap(mp, mp.mpf(float(group.theta_mod_2pi(u0))) - g50)
        ref_30_50 = abs(wrap(mp, mp.mpf(g30) - g50))
    print(
        f"\ngroup at the lower endpoint u = {u0!r}, at the exact e^u: delta theta at 30 digits {float(d30):+.6e}, "
        f"at 50 digits {float(d50):+.6e}; |ref(30) - ref(50)| {float(ref_30_50):.2e}; "
        f"|atan2(J,-Y) - (arg H1 + pi/2)| summed over the group {float(group_formulas):.2e}; "
        f"theta_abserr {float(group.theta_abserr(u0)):.4e}"
    )
    print(
        f"   (06f's convention, at x_hat = np.exp(u): delta theta {float(d30_hat):+.6e})"
    )
    if hasattr(group, "_declared_error_parts"):
        c_, a_, f_ = (float(v) for v in group._declared_error_parts(u0))
        print(
            f"   theta_abserr's parts: constituents {c_:.4e}, abscissa {a_:.4e}, phi_eval {f_:.4e}"
        )
    ok &= abs(float(d30) - float(d50)) < 1e-25
    ok &= float(ref_30_50) < 1e-25 and float(group_formulas) < 1e-25
    print(f"\nmethod check: {'passes' if ok else 'FAILS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
