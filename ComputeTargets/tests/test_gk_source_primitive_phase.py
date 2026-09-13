"""
The `GkSource` rectifier and `PrimitivePhase` together (GkTk-remedial prompt 09, Sec 4).

`GkSourcePolicyData` builds one phase object from the stored samples of ~1300 *independent*
`GkWKBIntegration` objects, one per source redshift. Prompt 06 removed the cross-sample rebase that
used to make neighbouring objects disagree by a whole cycle, but it did not remove -- and campaign
decision D5 says it must not -- the *second* mechanism review Sec 8.3 measured: in the
numeric-initialised band the initial-data offset `delta = atan2(raw_cos, raw_sin)` is computed per
object, and where the numeric stop point `z_init(z_s)` moves to the next extremum between
neighbouring source redshifts the stored cycle count steps by one while the physical phase stays
smooth (`RECONCILIATION.md` Sec 2 item 6). Prompt 06's log measures 90 such steps in 990 objects
and names the concrete case reproduced below: `k = 1e7`, `x_r = 1e3`, 22 objects, 2 steps, between
`z_s = 316700 -> 324331` and `401839 -> 411522`.

`GkSource.assemble_GkSource_values` (:166-233) repairs exactly those steps. This module checks that
the repair is what `phi` needs:

1. `phi` built from the **rectified** cycle counts has no 2pi jumps; built from the raw counts it
   has one at every stop-point transition and nowhere else.
2. The resulting `PrimitivePhase` reproduces the exact radiation phase between the samples.
3. On pure-WKB objects -- source redshifts below `sqrt(z_e3 z_e4)`, which carry their own analytic
   initial data rather than a numeric hand-over -- the rectifier makes no correction at all, so it
   is inert exactly where it has nothing to repair.

The rectifier is **copied** here, not imported: `assemble_GkSource_values` is a Ray remote over
datastore objects (`GkSourceValue`, `redshift_array`, ...) and cannot be called offline. The copy
below is the phase-bookkeeping half of `ComputeTargets/GkSource.py:166-233`, transcribed, and is
the same device the review's own `docs/gk-wkb-review-fable-2026-09-09/t6_sweep.py` used. Anything
that changes the rectifier must change this copy too -- which is the point: D5 makes its logic a
stop condition, and this module is what would notice.

Offline: exact radiation, prompt 06's sweep geometry and its `store_algebra` fixture, a real
`CumulativeTable`. No Ray, no datastore.
"""

import unittest
from math import ceil, cos, exp, fabs, sin, sqrt, pi

import numpy as np

from ComputeTargets.BackgroundModel import TablePrimitive
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples
from ComputeTargets.tests.test_gk_wkb_phase import store_algebra
from LiouvilleGreen.WKBtools import WKB_mod_2pi, apply_phase_offset
from LiouvilleGreen.constants import TWO_PI

# prompt 06's sweep geometry (test_gk_wkb_phase._sweep), reproduced with the response point and
# the source samples taken from the background grid, as production does (main.py:476)
RADIATION_H0 = 1.0
SAMPLES_PER_DECADE = 100
RESPONSE_SPARSENESS = 12

# the concrete case prompt 06's log names
CASE_K = 1.0e7
CASE_X_R = 1.0e3

# prompt 09 Sec 4 test 1: after rectification, neighbouring phi differ by less than this
PHI_JUMP_TOL = 0.1
# prompt 09 Sec 4 test 2: raw_theta against the exact radiation phase
RAW_THETA_TOL = 1.0e-8

TAU_GAUSS_ORDER = 4


# ------------------------------------------------------------------------------------------------
# exact radiation background
# ------------------------------------------------------------------------------------------------


def _Hubble(z: float) -> float:
    return RADIATION_H0 * (1.0 + z) * (1.0 + z)


class _Functions:
    Hubble = staticmethod(_Hubble)


def _theta(s_a: float, s_b: float) -> float:
    """`theta(z_b; z_a) = -k (1/s_b - 1/s_a)` without the `k`, factored so that a short baseline
    keeps its digits: returns `-(s_a - s_b)/(s_a s_b)`, negative for `s_b < s_a`."""
    return -(s_a - s_b) / (s_a * s_b)


# ------------------------------------------------------------------------------------------------
# a faithful copy of the GkSource rectifier
# ------------------------------------------------------------------------------------------------


def rectify(div_2pi, mod_2pi):
    """
    The phase-bookkeeping half of `ComputeTargets/GkSource.py:166-233`, transcribed verbatim in
    behaviour. The samples must be supplied **from low z to high z**, which is the order
    `assemble_GkSource_values` iterates in (`for z_source in reversed(list(z_sample))`).

    Returns `(rectified_div_2pi, corrections)` where `corrections` is the number of samples at
    which the non-monotonicity branch fired, i.e. the number of repairs the rectifier actually
    made.
    """
    current_2pi_block_subtraction = None
    current_2pi_block = None
    last_theta_mod_2pi = None
    last_theta_div_2pi = None

    rectified = []
    corrections = 0

    for theta_div_2pi, theta_mod_2pi in zip(div_2pi, mod_2pi):
        if last_theta_mod_2pi is None:
            current_2pi_block_subtraction = theta_div_2pi
            current_2pi_block = 0
            rectified_theta_div_2pi = 0
        else:
            last_theta = TWO_PI * last_theta_div_2pi + last_theta_mod_2pi
            theta = TWO_PI * theta_div_2pi + theta_mod_2pi
            if theta > last_theta:
                corrections += 1
                delta_theta_mod_2pi = theta_mod_2pi - last_theta_mod_2pi

                delta_theta_mod_2pi_0 = fabs(delta_theta_mod_2pi)
                delta_theta_mod_2pi_p1 = fabs(delta_theta_mod_2pi + TWO_PI)
                delta_theta_mod_2pi_m1 = fabs(delta_theta_mod_2pi - TWO_PI)

                jump_options = [
                    (delta_theta_mod_2pi_0, 0),
                    (delta_theta_mod_2pi_p1, +1),
                    (delta_theta_mod_2pi_m1, -1),
                ]
                best_jump = min(jump_options, key=lambda x: x[0])[1]

                rectified_theta_div_2pi = current_2pi_block + best_jump
                current_2pi_block_subtraction = theta_div_2pi - rectified_theta_div_2pi
            else:
                rectified_theta_div_2pi = theta_div_2pi - current_2pi_block_subtraction

            current_2pi_block = rectified_theta_div_2pi

        last_theta_mod_2pi = theta_mod_2pi
        last_theta_div_2pi = theta_div_2pi
        rectified.append(rectified_theta_div_2pi)

    return rectified, corrections


# ------------------------------------------------------------------------------------------------
# the sweep: one GkSource's worth of stand-in GkSourceValues
# ------------------------------------------------------------------------------------------------


class Sweep:
    """
    One `GkSource` at fixed response redshift, in exact radiation, built from independent
    `GkWKBIntegration` stand-ins exactly as prompt 06's `test_gk_wkb_phase._sweep` builds them:
    the numeric stop point is the first **maximum** of `G` past the 4-e-fold point (the production
    geometry, README Sec 2 (h)), the exact initial data there go through the real `store()`
    algebra, and `apply_phase_offset` attaches `delta` per sample with no rebase.

    The source samples and the response point are taken from the background grid, so every
    `tau.delta` call has both endpoints on a node -- as in production, where the background model
    is built on the source grid.
    """

    def __init__(self, k: float, x_r: float):
        self.k = float(k)
        e3, e4 = exp(3.0), exp(4.0)
        self.s_e3 = k / e3
        self.s_e4 = k / e4
        self.s_lim = sqrt(self.s_e3 * self.s_e4)

        # the background/source grid, 100 per decade of s = 1+z
        n = int(np.log10(self.s_e3) * SAMPLES_PER_DECADE) + 1
        self.full = np.geomspace(1.0, self.s_e3, n)
        response = self.full[::-1][::RESPONSE_SPARSENESS][::-1]
        self.s_r = float(response[np.argmin(np.abs(response - k / x_r))])

        z_nodes = (self.full - 1.0)[::-1]  # descending
        table = CumulativeTable(
            z_nodes, lambda z: 1.0 / _Hubble(z), TAU_GAUSS_ORDER, label="tau"
        )
        self.leading = TablePrimitive(table, "tau")
        self.z_response = self.s_r - 1.0

        # the numeric-initialised band, ascending in z
        band = self.full[(self.full >= self.s_lim) & (self.full <= self.s_e3)]
        self.numeric_sources = band

        # the pure-WKB band: source redshifts at or below sqrt(z_e3 z_e4), from the response
        # point (where theta vanishes) upwards
        self.wkb_sources = self.full[
            (self.full >= self.s_r) & (self.full <= self.s_lim)
        ]

    # -- the two producers -------------------------------------------------------------------

    def numeric_initialised(self):
        """
        One record per source in the numeric-initialised band: the stop index `n`, the offset
        `delta` from the store algebra, the stored `(div, mod)` at the response point, and the
        exact phase there.
        """
        k = self.k
        e4 = exp(4.0)
        out = []
        for s_s in self.numeric_sources:
            x_s = k / s_s
            p4 = x_s - e4
            # theta_i = -(3pi/2 + 2 pi n): the maxima of sin(theta) on the negative axis
            n = max(0, ceil((-p4 - 1.5 * pi) / TWO_PI))
            theta_i = -(1.5 * pi + TWO_PI * n)
            s_i = 1.0 / (1.0 / s_s - theta_i / k)

            G_i = (s_s * s_s / k) * sin(theta_i)
            Gprime_i = cos(theta_i) * s_s * s_s / (s_i * s_i)
            omega_sq_i = (k / (s_i * s_i)) ** 2
            B, delta, _, _ = store_algebra(
                omega_sq_i, -2.0 / s_i, 2.0, s_i - 1.0, G_i, Gprime_i
            )
            assert B > 0.0

            div, mod = WKB_mod_2pi(k * _theta(s_i, self.s_r))
            div2, mod2 = apply_phase_offset([div], [mod], delta)

            out.append(
                {
                    "s_s": float(s_s),
                    "z_s": float(s_s - 1.0),
                    "n": int(n),
                    "delta": float(delta),
                    "div": int(div2[0]),
                    "mod": float(mod2[0]),
                    "raw_div": int(div2[0]),
                    "exact": k * _theta(s_s, self.s_r),
                }
            )
        return out

    def pure_WKB(self):
        """
        One record per source below `sqrt(z_e3 z_e4)`. These objects take their initial data from
        the analytic Green's function at `z_init = z_source` itself, where the unit-jump condition
        gives `G = 0` and `G' > 0`; the store algebra then returns `delta = atan2(0, positive) = 0`
        exactly, the same for every object, so the stored phase is the physical one and there is
        nothing for the rectifier to repair.
        """
        k = self.k
        out = []
        for s_s in self.wkb_sources:
            s_i = float(s_s)
            omega_sq_i = (k / (s_i * s_i)) ** 2
            B, delta, _, _ = store_algebra(
                omega_sq_i, -2.0 / s_i, 2.0, s_i - 1.0, 0.0, s_s * s_s / (s_i * s_i)
            )
            assert B > 0.0
            assert delta == 0.0

            div, mod = WKB_mod_2pi(k * _theta(s_i, self.s_r))
            div2, mod2 = apply_phase_offset([div], [mod], delta)
            out.append(
                {
                    "s_s": float(s_s),
                    "z_s": float(s_s - 1.0),
                    "delta": float(delta),
                    "div": int(div2[0]),
                    "mod": float(mod2[0]),
                    "exact": k * _theta(s_s, self.s_r),
                }
            )
        return out

    # -- the consumer ------------------------------------------------------------------------

    def phi(self, records, divs):
        """`phi` at the source samples, from the supplied cycle counts (raw or rectified)."""
        z = [r["z_s"] for r in records]
        theta = [d * TWO_PI + r["mod"] for d, r in zip(divs, records)]
        return build_phi_samples(
            self.k, self.leading, self.z_response, z, theta, sign=-1
        )

    def phase(self, records, divs) -> PrimitivePhase:
        return PrimitivePhase(
            self.k,
            self.leading,
            self.z_response,
            [r["z_s"] for r in records],
            self.phi(records, divs),
            sign=-1,
            model_functions=_Functions,
            label="sweep",
        )


# ------------------------------------------------------------------------------------------------
# 1. phi has no 2pi jumps after rectification, and has them before it
# ------------------------------------------------------------------------------------------------


class TestRectifierMakesPhiSmooth(unittest.TestCase):
    """Prompt 09 Sec 4 (`test_gk_source_primitive_phase`) test 1. This documents D5."""

    def test_the_named_case_from_prompt_06(self):
        sweep = Sweep(CASE_K, CASE_X_R)
        recs = sweep.numeric_initialised()
        raw_div = [r["div"] for r in recs]
        rect_div, corrections = rectify(raw_div, [r["mod"] for r in recs])

        phi_raw = sweep.phi(recs, raw_div)
        phi_rect = sweep.phi(recs, rect_div)

        d_raw = np.diff(phi_raw)
        d_rect = np.diff(phi_rect)

        transitions = [
            i for i in range(len(recs) - 1) if recs[i + 1]["n"] != recs[i]["n"]
        ]
        jumps = [i for i in range(len(d_raw)) if fabs(d_raw[i]) > PHI_JUMP_TOL]

        print(
            f"\n[prompt 09 test 1] k = {CASE_K:.3g}/Mpc, x_r = {CASE_X_R:.0e} "
            f"(z_response = {sweep.z_response:.6g}): {len(recs)} objects in the "
            f"numeric-initialised band z_s in "
            f"[{sweep.s_lim - 1.0:.6g}, {sweep.s_e3 - 1.0:.6g}]"
        )
        print(
            f"  before rectification: {len(jumps)} jumps in phi, at z_s "
            + ", ".join(
                f"({recs[i]['z_s']:.6g} -> {recs[i + 1]['z_s']:.6g}: "
                f"{d_raw[i] / TWO_PI:+.4f} cycles)"
                for i in jumps
            )
        )
        print(
            f"  stop-point transitions at z_s "
            + ", ".join(
                f"({recs[i]['z_s']:.6g} -> {recs[i + 1]['z_s']:.6g})"
                for i in transitions
            )
        )
        print(
            f"  after rectification ({corrections} corrections applied): "
            f"max |d phi| = {np.max(np.abs(d_rect)) if len(d_rect) else 0.0:.3g} rad, "
            f"phi = {phi_rect[0]:.6g} rad = {phi_rect[0] / TWO_PI:.6g} cycles"
        )

        # the jumps before rectification are exactly the stop-point transitions ...
        self.assertEqual(jumps, transitions)
        self.assertGreater(len(jumps), 0)
        # ... each of exactly one cycle, in the direction the rectifier is built to catch
        for i in jumps:
            self.assertAlmostEqual(d_raw[i] / TWO_PI, 1.0, places=9)
        # ... and the rectifier repairs every one of them
        self.assertEqual(corrections, len(transitions))
        self.assertLessEqual(float(np.max(np.abs(d_rect))), PHI_JUMP_TOL)

        # Prompt 06's log for this case: 22 objects, 2 steps, between z_s = 316700 -> 324331 and
        # 401839 -> 411522. The source samples here are taken from the background grid rather
        # than from an independent geomspace over the band (production builds the background
        # model on the source grid, main.py:476), so the sample redshifts differ from prompt 06's
        # by up to one grid spacing; the transitions are required to land in the same place to
        # that accuracy.
        spacing = 10.0 ** (1.0 / SAMPLES_PER_DECADE)
        self.assertEqual(len(recs), 22)
        self.assertEqual(len(jumps), 2)
        for i, (lo, hi) in zip(jumps, ((316700.0, 324331.0), (401839.0, 411522.0))):
            self.assertLessEqual(fabs(np.log(recs[i]["z_s"] / lo)), np.log(spacing))
            self.assertLessEqual(fabs(np.log(recs[i + 1]["z_s"] / hi)), np.log(spacing))

    def test_across_the_sweep(self):
        """The same statement over prompt 06's 15 x 3 grid of wavenumbers and response points."""
        totals = {"objects": 0, "jumps": 0, "transitions": 0, "corrections": 0}
        worst_rect = 0.0
        for k in np.geomspace(1.0e6, 1.0e8, 15):
            for x_r in (1.0e2, 1.0e3, 1.0e4):
                sweep = Sweep(float(k), x_r)
                recs = sweep.numeric_initialised()
                raw_div = [r["div"] for r in recs]
                rect_div, corrections = rectify(raw_div, [r["mod"] for r in recs])

                d_raw = np.diff(sweep.phi(recs, raw_div))
                d_rect = np.diff(sweep.phi(recs, rect_div))

                transitions = [
                    i for i in range(len(recs) - 1) if recs[i + 1]["n"] != recs[i]["n"]
                ]
                jumps = [i for i in range(len(d_raw)) if fabs(d_raw[i]) > PHI_JUMP_TOL]

                self.assertEqual(jumps, transitions, msg=f"k={k:.4g}, x_r={x_r:g}")
                totals["objects"] += len(recs)
                totals["jumps"] += len(jumps)
                totals["transitions"] += len(transitions)
                totals["corrections"] += corrections
                if len(d_rect):
                    worst_rect = max(worst_rect, float(np.max(np.abs(d_rect))))

        print(
            f"[prompt 09 test 1] sweep: {totals['objects']} objects, {totals['jumps']} phi "
            f"jumps before rectification = {totals['transitions']} stop-point transitions, "
            f"{totals['corrections']} rectifier corrections; after rectification "
            f"max |d phi| = {worst_rect:.3g} rad"
        )
        self.assertGreater(totals["jumps"], 0)
        self.assertEqual(totals["jumps"], totals["transitions"])
        self.assertEqual(totals["corrections"], totals["transitions"])
        self.assertLessEqual(worst_rect, PHI_JUMP_TOL)


# ------------------------------------------------------------------------------------------------
# 2. the resulting PrimitivePhase reproduces the exact phase
# ------------------------------------------------------------------------------------------------


class TestPhaseFromRectifiedSamples(unittest.TestCase):
    """Prompt 09 Sec 4 (`test_gk_source_primitive_phase`) test 2."""

    def test_raw_theta_against_the_exact_radiation_phase(self):
        sweep = Sweep(CASE_K, CASE_X_R)
        recs = sweep.numeric_initialised()
        rect_div, _ = rectify([r["div"] for r in recs], [r["mod"] for r in recs])
        phase = sweep.phase(recs, rect_div)

        # 10 points per interval over the band
        u = np.log1p(np.array([r["z_s"] for r in recs]))
        fine = np.concatenate(
            [np.linspace(u[i], u[i + 1], 12)[1:-1] for i in range(len(u) - 1)]
        )

        # the stored phase differs from the exact one by a whole number of cycles, fixed by the
        # rectifier's rebasing of the first sample into the fundamental block; sin(theta) is
        # unchanged by it, and so is every consumer
        offsets = []
        for uu in fine:
            z = float(np.expm1(uu))
            s = 1.0 + z
            exact = sweep.k * _theta(s, sweep.s_r)
            offsets.append(phase.raw_theta(float(uu), x_is_log=True) - exact)
        offsets = np.array(offsets)
        cycles = offsets / TWO_PI
        N = int(round(float(np.mean(cycles))))
        residual = np.max(np.abs(offsets - N * TWO_PI))

        print(
            f"\n[prompt 09 test 2] k = {CASE_K:.3g}/Mpc, {len(fine)} interior points: "
            f"raw_theta - exact = {N} cycles + {residual:.3g} rad "
            f"(|theta| up to {max(abs(r['exact']) for r in recs):.6g} rad)"
        )
        self.assertLessEqual(residual, RAW_THETA_TOL)
        # the offset really is whole cycles, not a fitted constant
        self.assertLessEqual(float(np.max(np.abs(cycles - N))), 1.0e-9)

    def test_theta_mod_2pi_reproduces_the_stored_remainder(self):
        """At a sample, the reduced phase is the stored `theta_mod_2pi` -- this is what
        `GkWKBSplineWrapper` feeds to `sin()`."""
        sweep = Sweep(CASE_K, CASE_X_R)
        recs = sweep.numeric_initialised()
        rect_div, _ = rectify([r["div"] for r in recs], [r["mod"] for r in recs])
        phase = sweep.phase(recs, rect_div)

        worst = max(
            fabs(phase.theta_mod_2pi(r["z_s"]) - r["mod"])
            for r in recs
            # the remainder wraps discontinuously; compare only where it is not within a
            # rounding of the (-2pi, 0] boundary
            if -TWO_PI + 1.0e-6 < r["mod"] < -1.0e-6
        )
        print(
            f"[prompt 09 test 2] theta_mod_2pi at the samples reproduces the stored "
            f"remainder to {worst:.3g} rad"
        )
        self.assertLessEqual(worst, 1.0e-8)


# ------------------------------------------------------------------------------------------------
# 3. the rectifier is inert on pure-WKB objects
# ------------------------------------------------------------------------------------------------


class TestRectifierIsInertOnPureWKB(unittest.TestCase):
    """Prompt 09 Sec 4 (`test_gk_source_primitive_phase`) test 3."""

    def test_no_corrections_and_the_cycle_counts_are_untouched(self):
        for k, x_r in ((CASE_K, CASE_X_R), (1.0e6, 1.0e2), (1.0e8, 1.0e4)):
            with self.subTest(k=k, x_r=x_r):
                sweep = Sweep(k, x_r)
                recs = sweep.pure_WKB()
                raw_div = [r["div"] for r in recs]
                rect_div, corrections = rectify(raw_div, [r["mod"] for r in recs])

                phi = sweep.phi(recs, rect_div)
                print(
                    f"\n[prompt 09 test 3] k = {k:.3g}/Mpc, x_r = {x_r:.0e}: "
                    f"{len(recs)} pure-WKB objects (z_s from {recs[0]['z_s']:.6g} to "
                    f"{recs[-1]['z_s']:.6g}, all delta = 0), {corrections} rectifier "
                    f"corrections, max |phi| = {np.max(np.abs(phi)):.3g} rad"
                )
                self.assertEqual(corrections, 0)
                # the rectifier's only action is the rebase of the first sample, which is the
                # identity here because theta vanishes at the response point
                self.assertEqual(raw_div[0], 0)
                self.assertEqual(list(rect_div), list(raw_div))
                # and phi is the residual, which is identically zero in radiation
                self.assertLessEqual(float(np.max(np.abs(phi))), 1.0e-6)

    def test_the_phase_is_the_exact_one(self):
        sweep = Sweep(CASE_K, CASE_X_R)
        recs = sweep.pure_WKB()
        rect_div, _ = rectify([r["div"] for r in recs], [r["mod"] for r in recs])
        phase = sweep.phase(recs, rect_div)

        u = np.log1p(np.array([r["z_s"] for r in recs]))
        fine = np.concatenate(
            [np.linspace(u[i], u[i + 1], 12)[1:-1] for i in range(len(u) - 1)]
        )
        worst = 0.0
        for uu in fine:
            z = float(np.expm1(uu))
            exact = sweep.k * _theta(1.0 + z, sweep.s_r)
            worst = max(worst, fabs(phase.raw_theta(float(uu), x_is_log=True) - exact))
        print(
            f"[prompt 09 test 3] pure-WKB band, {len(fine)} interior points: "
            f"max |raw_theta - exact| = {worst:.3g} rad "
            f"(|theta| up to {max(abs(r['exact']) for r in recs):.6g} rad)"
        )
        self.assertLessEqual(worst, RAW_THETA_TOL)


if __name__ == "__main__":
    unittest.main()
