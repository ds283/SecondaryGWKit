"""
How sharp is the equation of state's step, as the consumer's spline meets it?

Written for prompt 09 of ``prompts/qcd-background-audit/`` to establish the *cause* of the only
numbers that moved the wrong way in the campaign's close-out: the QCD ``k = 1e5`` rows of
``docs/gktk-remedial-verification.md`` §3.5 (4.25x and 4.39x worse) and §3.6 (2.00x and 2.79x),
all four of them at ``QCD_EOS``'s ``T_LO`` crossing.

The claim it measures is that the *corrected* background presents a sharper feature at that
crossing than the defective one did, and that a cubic spline of ``phi`` with default knots
therefore does worse on it -- which is ``[13-consumer-spline-crosses-eos-break-points]`` seen
without the old representation's smoothing on top, not a regression of the background.

``T(z)`` jumps at the crossing on *both* trees, because the jump is the equation of state's and
the equation of state was never touched (README §0.5). What changed is how the jump is delivered
to the production source grid:

  * the pre-campaign representation was a 500-node, order-3 spline of ``T`` against
    ``u = log(1+z)``, whose **knot spacing was 4.04x the production grid spacing**, so it smeared
    the step over about four grid intervals and carried its own ~1e-3 scatter in
    ``dlnH/du`` for several intervals either side;
  * the shipped representation is segmented *at* the crossing, so the step is the genuine one and
    falls entirely inside the single grid interval that contains it.

The measure is ``dlnH/du`` differenced on production-grid spacing over 25 intervals centred on the
crossing, reported as a deviation from the local median. Both the peak and **how concentrated it
is** matter: a cubic spline's error responds to the concentration.

**Read the concentration statistics only at ``T_LO``.** There the smooth background is locally
flat -- ``dlnH/du`` is 1.99996 with a drift of ~1e-5 across the whole profile -- so the deviation
from the local median is the step and nothing else. At ``EOS_T_LO`` and ``T_120_MEV`` the profile
sits inside the QCD transition itself, where ``g_*(T)`` is changing fast and the smooth variation
of ``dlnH/du`` across 25 grid intervals swamps the step: the "share inside the crossing's own
interval" and "intervals carrying > 10 % of the peak" lines are then measuring the transition, not
the discontinuity, and mean nothing. The per-interval table above them is still correct at all
three. ``T_LO`` is the default and is the crossing §3.5 and §3.6 rise at.

No Ray, no datastore; ~1 s.

Usage, from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_break_point_profile.py

To take the same profile on another tree -- e.g. a ``git worktree`` at the campaign base
``e8f746d`` (or ``2a5e0fa``, which differs from it only under ``docs/`` and ``prompts/``) -- pass
that checkout's root; the script inserts it at the front of ``sys.path`` and is otherwise
identical, so the two runs differ only in
``CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py``::

    PYTHONPATH=. /path/to/worktree/venv/bin/python \
        docs/qcd-background-audit/consumer_break_point_profile.py --root /path/to/worktree

(A worktree has no ``./venv``; symlink the main checkout's.)
"""

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root",
    default=None,
    help="checkout to import the cosmology from (default: the current PYTHONPATH)",
)
parser.add_argument(
    "--crossing",
    default="T_LO",
    choices=("T_LO", "EOS_T_LO", "T_120_MEV"),
    help="which equation-of-state crossing to profile (default: T_LO, the one §3.5 rises at)",
)
args = parser.parse_args()

if args.root is not None:
    sys.path.insert(0, args.root)

import numpy as np  # noqa: E402

from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology  # noqa: E402
from CosmologyModels.LambdaCDM import Planck2018  # noqa: E402
from Units import Mpc_units  # noqa: E402

# The three crossings, in u = log(1+z), to 17 digits. These are prompt 01's independently
# bisected `T_z_reference.jump_locations` values and prompt 06's segment edges, which agree to
# 0, 1 and 0 ulp; they are transcribed rather than re-derived so that this script measures the
# same abscissae on a tree that predates either.
CROSSINGS = {
    "T_LO": (17.565806941870026, "1e-5 GeV"),
    "EOS_T_LO": (23.197460552819653, "0.002 GeV"),
    "T_120_MEV": (27.485391822044257, "0.12 GeV"),
}

# populate_z_sample's production geometry: 100 samples per decade of (1+z), i.e. a fixed spacing
# in u. Pinned rather than rebuilt, for the same reason as above.
GRID_DU = 2.3031952967555162e-02

# Half-width of the profile, in grid intervals.
HALF_WIDTH = 12

# Offset of the sample lattice within a grid interval, so that no sample lands exactly on the
# crossing (which would make the answer depend on which side the last bit falls).
PHASE = 0.37


def main() -> None:
    u_c, T_break = CROSSINGS[args.crossing]

    cosmology = QCD_Cosmology(
        store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
    )

    offsets = np.arange(-HALF_WIDTH, HALF_WIDTH + 1) + PHASE
    us = u_c + GRID_DU * offsets
    zs = np.expm1(us)

    lnH = np.log(np.array([cosmology.Hubble(float(z)) for z in zs]))
    d = np.diff(lnH) / GRID_DU
    median = float(np.median(d))
    dev = d - median

    print()
    print("=" * 94)
    print(
        f"dlnH/du across the {args.crossing} crossing ({T_break}), on production-grid spacing"
    )
    print("=" * 94)
    print(f"   u_c = {u_c!r}   z_c = {np.expm1(u_c):.9e}")
    print(
        f"   grid spacing du = {GRID_DU:.10e}   ({2 * HALF_WIDTH} intervals profiled)"
    )
    print()
    print(f"   {'u_mid':>12} {'z':>14} {'dlnH/du':>20} {'dev from median':>17}")
    for i in range(len(d)):
        u_mid = 0.5 * (us[i] + us[i + 1])
        inside = us[i] < u_c < us[i + 1]
        print(
            f"   {u_mid:12.6f} {np.expm1(u_mid):14.6e} {d[i]:20.15f} {dev[i]:+17.6e}"
            + ("   <-- the crossing is in this interval" if inside else "")
        )

    peak = float(np.max(np.abs(dev)))
    total = float(np.sum(np.abs(dev)))
    inside_idx = int(np.argmax([us[i] < u_c < us[i + 1] for i in range(len(d))]))
    share = abs(float(dev[inside_idx])) / total if total > 0.0 else float("nan")
    spread = int(np.sum(np.abs(dev) > 0.1 * peak))

    print()
    print(f"   median dlnH/du                                     {median:.15f}")
    print(f"   peak |deviation|                                   {peak:.6e}")
    print(
        f"   deviation in the crossing's own interval           {dev[inside_idx]:+.6e}"
    )
    print(f"   total sum |deviation| over the profile             {total:.6e}")
    print(
        f"   fraction of that total in the crossing's interval  {100.0 * share:.2f} %"
    )
    print(f"   intervals carrying > 10 % of the peak              {spread}")
    print()
    print(
        "   The last two lines are the ones that matter to a consumer: a cubic spline of phi\n"
        "   meets a step that is concentrated in one interval far less gracefully than the same\n"
        "   step smeared across several."
    )
    if args.crossing != "T_LO":
        print()
        print(
            "   CAUTION: this profile sits inside the QCD transition, where g_*(T) is changing\n"
            "   fast, so the smooth variation of dlnH/du across 25 grid intervals swamps the\n"
            "   step and the two summary lines above measure the transition rather than the\n"
            "   discontinuity. Only T_LO, where the background is locally flat to ~1e-5, gives\n"
            "   a meaningful concentration figure. The per-interval table is correct either way."
        )
    print()


if __name__ == "__main__":
    main()
