"""
The background-against-background guard for the QCD temperature
(``prompts/qcd-background-audit/``, prompt 01; audit §8 recommendation 4).

Every measurement in ``docs/gktk-remedial-verification.md`` §3.5-§3.6 scores a **consumer against
a producer**, and both are built from the same ``BackgroundModel``, hence the same ``H``, hence the
same ``tau``. An error in the background cancels *exactly* in that comparison. That is how §3.5 can
read 1.00 ulp of the span while the background underneath both sides carries, at
``k = 3e8 /Mpc``, of order 1.4e5 radians (audit §5, §6). The repository therefore contained no test
that could fail because the background was wrong.

This module is that test. It scores the shipped ``T(z)`` against an **independent** reference --
the defining equation root-solved to ``rtol = 1e-14``, in ``CosmologyModels/tests/T_z_reference.py``
-- and it scores the conformal time the shipped background produces against the conformal time the
exact background produces.

**Written on 2026-09-13 as a characterisation harness; since prompt 06 it is a guard.** Each
threshold was set at the value the tree achieved when the case was written and carries a comment
naming the campaign prompt that tightened it (README §6.1, §6.2); prompts 04, 05 and 06 have taken
them to README §6.1's final column, and the conformal-time case now reads at zero -- the shipped
background and the exact background give the same ``int dz/H`` to the last bit.

Three of the cases are not accuracy measurements at all:

* ``test_the_branch_joins_are_where_the_fixture_puts_them`` pins an upstream data fixture this
  campaign deliberately does not repair (README §0.5, §7 D6);
* ``test_a_segment_edge_bisected_and_one_root_found_disagree`` pins the implementation trap
  prompt 06 had to avoid (README §2 (b)), and
  ``test_an_edge_misplaced_by_one_node_restores_the_error`` is the same trap the other way round:
  it builds the representation prompt 06 would have built had it fallen for it, and requires the
  error to come back. The failure mode is silent -- every other statistic still improves -- so it
  is guarded rather than argued.

No Ray and no datastore is needed.
"""

import unittest
from math import expm1, log, log1p, sin

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import root_scalar

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import (
    DEFAULT_T_Z_SPLINE_ORDER,
    DEFAULT_T_Z_SPLINE_SAMPLES,
    SEGMENT_EDGE_PAD_LOG1PZ,
    LambdaCDM_GenericEOS,
    SegmentedEntropyFactor,
    TemperatureRepresentation,
    build_segmented_entropy_spline,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.tests.T_z_reference import (
    PRODUCTION_FLOOR_RAD,
    PRODUCTION_SPANS_RAD,
    Hubble_with,
    Stats,
    accurate_T,
    entropy_factor,
    inverse_Hubble_integral,
    jump_locations,
    probe_set,
    production_source_z_values,
    reference_temperatures,
    relative,
    tabulated_u_range,
)
from CosmologyModels.tests.test_wPerturbations import PureRadiationEOS, lambdaCDM_gstar
from Units import Mpc_units

# the production max_z of config/model_list.py
PRODUCTION_MAX_Z = 1.0e20

# ------------------------------------------------------------------------------------------
# thresholds. Each is the value the tree achieves on 2026-09-13, rounded up in the last place
# it is quoted to, together with the prompt that tightens it.
# ------------------------------------------------------------------------------------------

# case 1 -- T_photon against the defining equation, audit §3 / README §6.1.
# Tightened by prompt 06 to README §6.1's "After 06" column, which is the end of the road for
# this table: the entropy factor is now splined one segment per branch of the equation of state,
# with the segment edges bisected onto the redshifts at which T(z) genuinely jumps, at 3,000
# nodes of order 5. The max falls seven orders, 7.236e-04 -> 6.807e-11, because it was never an
# accuracy at all -- it was the height of the jump, carried by a single spline that ran straight
# across a step and could not do otherwise at any node count. With the step removed the node
# count buys the p90 and the median in the ordinary way: 8.912e-08 -> 3.237e-15 and
# 2.599e-10 -> 1.765e-16, both at the audit's §4 figures (3.123e-15, 1.773e-16).
T_PHOTON_MAX = 1.0e-10
T_PHOTON_P90 = 1.0e-14
T_PHOTON_MEDIAN = 1.0e-15

# case 1b -- H(z) on the production source grid, audit §5 / README §6.2. rho_r = a g(T) T^4 and
# H ~ sqrt(rho), so a relative error eps in T reaches H at roughly 2 eps in the radiation era,
# and that is what is measured: 1.690e-10 / 6.276e-15 / 2.804e-16 against the audit's
# 1.690e-10 / 6.314e-15 / 2.803e-16, from 1.278e-03 / 2.895e-05 / 3.424e-07 before the campaign.
# Introduced by prompt 06; the harness (Hubble_with) has been in T_z_reference since prompt 01.
HUBBLE_MAX = 2.0e-10
HUBBLE_P90 = 1.0e-14
HUBBLE_MEDIAN = 1.0e-15

# case 2 -- the node solve's own rtol = 1e-4, audit §3 (T2). Tightened by prompt 04 to 1e-14
# (achieved: bit-identical to the rtol=1e-14 reference on the probe set, so exactly 0.0).
NODE_SOLVE_MAX = 1.0e-14

# case 3 -- the T1 guard, audit §5 / README §6.2, and the campaign's headline measurement.
#
# 3.4605e-08 (prompt 01) -> 3.4509e-08 (prompt 04) -> 5.4264e-10 (prompt 05) -> **0.0** here:
# on this tree the segmented representation gives an int dz/H that is *bit-identical* to the one
# the exact background gives, at all 17 digits, which is what audit §5's "improved" row reports.
# The threshold is README §6.2's 1e-15 rather than an equality, because the identity is the last
# bit of a sum of a few hundred quadrature panels and is not something to assert on; the test
# prints whether it holds, and the phase floors below are asserted instead.
#
# Prompt 05 took the factor of 64 by removing the interpolation error the T-against-u spline
# carried across the whole range; what this prompt removes is the rest. Note that the jump itself
# is a set of measure zero in an integral, so it is not the jump that is being paid for here but
# the interpolation error a spline makes *near* a step it cannot represent, which is spread over
# the neighbouring node intervals and is not small.
CONFORMAL_TIME_REL = 1.0e-15
CONFORMAL_TIME_Z_LO = 1.0e2
CONFORMAL_TIME_Z_HI = 1.0e12

# case 4 -- the equation-of-state branch joins, audit §1. These are a *characterisation* of an
# upstream fixture, so they have no "tightened by" prompt: if one of them moves, the fixture
# changed. Keyed by break temperature in GeV, values (dg/g, dgs/gs).
BRANCH_JOINS = {
    1.0e16: (+1.454e-02, +1.395e-02),
    0.12: (-2.075e-04, -3.744e-04),
    1.0e-5: (+8.876e-04, -2.284e-03),
}
# the join that does match, and the bound that says so
CONTINUOUS_JOIN_GeV = 0.002
CONTINUOUS_JOIN_BOUND = 1.0e-10
# the audit quotes the three discontinuous joins to four significant figures
BRANCH_JOIN_RTOL = 1.0e-03
BRANCH_JOIN_SEPARATION = 1.0e-09

# case 5 -- the step in T(z) at the lowest crossing, audit §2.
LOWEST_CROSSING_Z = 4.25337e07
# the audit's §2 table, which prints F to twelve decimals as 0.000762292290
ENTROPY_FACTOR_STEP = 7.622923e-04
ENTROPY_FACTOR_STEP_TOL = 1.0e-09
# §1's forced dT/T = -(1/3) dgs/gs, which is the *linearised* figure; the step in F above is
# -(1/3) log(1 + dgs/gs) and the relative jump in T is expm1 of it. All three agree to 0.2 %.
LINEARISED_JUMP = 7.614e-04
LINEARISED_JUMP_TOL = 2.0e-06
# "flat to twelve decimals on each side" (audit §2)
BRANCH_FLATNESS = 1.0e-12

# case 6 -- a constant-gs equation of state is an exact ramp, README §2 (g).
# Tightened by prompt 05 from 2.0e-07 to 1.0e-15: with g_s constant, F(u) is identically zero,
# the interpolating spline of a constant is that constant, and the representation returns
# T_CMB (1+z) * exp(0) -- the closed-form answer, not an approximation to it. Achieved 2.928e-16,
# about 1.3 ulp, against the 1.9e-07 the T-against-u spline managed on the same model. This is
# the cleanest demonstration available that the new shape is doing what it claims: the two other
# representation defects (node accuracy, segmentation) are both switched off on this equation of
# state, so what is left is T3 alone.
EXACT_RAMP_MAX = 1.0e-15
EXACT_RAMP_NODE_MAX = 1.0e-15

# case 7 -- the segment-edge trap, README §2 (b). A relative residual this far above machine
# epsilon means the bracketing solver did not find a root; it reported one anyway.
NON_ROOT_RESIDUAL = 1.0e-08
# the padding the audit's segmented build uses to keep each branch's nodes inside its own segment
# (docs/qcd-background-audit/measure_T_z_representation.py, build_segmented). Since prompt 06 the
# production representation uses the same number, declared as SEGMENT_EDGE_PAD_LOG1PZ, and the
# two are asserted equal below.
SEGMENT_PAD = 1.0e-12

# ------------------------------------------------------------------------------------------
# prompt 06's own constants -- the segmentation
# ------------------------------------------------------------------------------------------

# case 8 -- where the production segment edges are. They are scored against
# T_z_reference.jump_locations, which bisects independently; the two agree to 0, 1 and 0 ulp of u
# at the three production crossings. The tolerance is in ulp of u rather than relative, because
# what matters about an edge is which side of a step it falls on, and that question is asked one
# bit at a time.
EDGE_AGREEMENT_ULP = 2.0

# case 9 -- an edge misplaced by one node of its own segment. The audit measured 5.7e-04 on a
# first attempt that root-found instead of bisecting; the guard asserts that the probe-set
# maximum comes back to the 1e-4 regime, which is four orders above what prompt 06 achieves and
# is the whole of what segmentation buys.
MISPLACED_EDGE_MIN_MAX = 1.0e-05

# case 10 -- the step, reproduced rather than smoothed. Either side of an edge the representation
# has to agree with the defining equation to the same floor it reaches anywhere else, with the
# full relative step in T between the two. The step at the lowest crossing is
# expm1(7.6229229003969e-04) = 7.625829e-04 (log 01 deviation 2); the third crossing's is
# measured here and is the +1.0108e-04 that `T_120_MEV` forces.
STEP_FLOOR = 1.0e-10


def _qcd_cosmology():
    """The production QCD cosmology: no Ray, no datastore, ~30 ms to build."""
    return QCD_Cosmology(
        store_id=0, units=Mpc_units(), params=Planck2018(), max_z=PRODUCTION_MAX_Z
    )


class TestQCDTemperatureRepresentation(unittest.TestCase):
    """
    Constructing a ``QCD_Cosmology`` costs 500 root solves plus the two equality solves, and the
    640-point reference costs 8 us per point. Neither is large in absolute terms (~30 ms and ~5 ms
    measured), but both are built once for the whole class rather than once per test.
    """

    @classmethod
    def setUpClass(cls):
        cls.cosmology = _qcd_cosmology()
        cls.probe_z = probe_set()
        cls.T_ref = reference_temperatures(cls.cosmology, cls.probe_z)
        cls.jumps = jump_locations(cls.cosmology)

        grid_u = np.log1p(np.sort(production_source_z_values()))
        cls.grid_spacing_u = float(np.median(np.diff(grid_u)))

    def test_T_z_matches_the_defining_equation(self):
        """
        The shipped ``T_photon`` against ``accurate_T`` on the audit's 640-point probe set.

        This is the whole of the audit's §3 table in one assertion. The three statistics separate
        the three representation defects: prompt 04 (accurate nodes) fixes the p90, prompt 05 (the
        entropy factor) fixes the median, prompt 06 (segmentation) fixes the max.
        """
        stats = Stats.of(
            relative(
                np.array([self.cosmology.T_photon(float(z)) for z in self.probe_z]),
                self.T_ref,
            )
        )
        print(
            f"\n[T(z) vs the defining equation] {len(self.probe_z)} probes, "
            f"z in [{self.probe_z.min():.4g}, {self.probe_z.max():.4g}]"
        )
        print("  " + stats.format("shipped T_photon"))

        self.assertLessEqual(stats.max, T_PHOTON_MAX)
        self.assertLessEqual(stats.p90, T_PHOTON_P90)
        self.assertLessEqual(stats.median, T_PHOTON_MEDIAN)

    def test_the_node_solve_converges(self):
        """
        ``_solve_T_z`` against ``accurate_T`` -- finding T2, and the root of
        ``[02-qcd-T-z-spline-node-tolerance]``.

        The shipped solve passes ``xtol=1e-6, rtol=1e-4`` to ``root_scalar``, and the 500 spline
        nodes are converged independently, so neighbouring nodes carry uncorrelated errors. A
        cubic through scattered nodes scatters, and its derivative scatters worse.

        This is the one test in the module that may call ``_solve_T_z``, because ``_solve_T_z`` is
        its subject rather than its reference.
        """
        stats = Stats.of(
            relative(
                np.array([self.cosmology._solve_T_z(float(z)) for z in self.probe_z]),
                self.T_ref,
            )
        )
        print("\n[node solve] " + stats.format("shipped _solve_T_z"))

        self.assertLessEqual(stats.max, NODE_SOLVE_MAX)

    def test_conformal_time_matches_the_exact_background(self):
        """
        **The T1 guard, and the reason this module exists.**

        ``int dz/H`` over ``z in [1e2, 1e12]``, computed twice from the same cosmology: once with
        the temperature as shipped, once with it replaced by ``accurate_T``. Nothing cancels
        between the two, so this is the only measurement in the repository that can fail because
        the background is wrong.

        Since ``theta = -[k dtau + drho]``, a relative error in ``tau`` is a phase error
        proportional to ``k tau``; the printed columns turn the measured relative error into
        radians at the three production wavenumbers, against one ulp of each span.
        """
        u_a, u_b = log1p(CONFORMAL_TIME_Z_LO), log1p(CONFORMAL_TIME_Z_HI)
        interior = [u for u in self.jumps if u_a < u < u_b]

        shipped = inverse_Hubble_integral(
            self.cosmology, self.cosmology._T_z_spline, u_a, u_b, interior
        )
        exact = inverse_Hubble_integral(
            self.cosmology,
            lambda z: accurate_T(self.cosmology, z),
            u_a,
            u_b,
            interior,
        )
        rel = abs(shipped - exact) / abs(exact)

        print(
            f"\n[conformal time] int dz/H over z in "
            f"[{CONFORMAL_TIME_Z_LO:.0e}, {CONFORMAL_TIME_Z_HI:.0e}], "
            f"{len(interior)} interior jumps given to the integrator"
        )
        print(f"  shipped background = {shipped:.16e}")
        print(f"  exact   background = {exact:.16e}")
        print(
            f"  relative error in tau = {rel:.4e}"
            f"   (bit-identical: {shipped == exact})"
        )
        for k, span in PRODUCTION_SPANS_RAD.items():
            print(
                f"    k = {k:9.3g} /Mpc:  {rel * span:10.3e} rad   "
                f"against a 1-ulp floor of {PRODUCTION_FLOOR_RAD[k]:.3e} rad"
            )

        self.assertLessEqual(rel, CONFORMAL_TIME_REL)

        # audit §5's phase columns: theta = -[k dtau + drho], so a relative error in tau is a
        # phase error proportional to k tau. Before this campaign these read 47.5 / 4.75e3 /
        # 1.43e5 radians against floors of 3.05e-07 / 3.05e-05 / 9.15e-04.
        for k, span in PRODUCTION_SPANS_RAD.items():
            with self.subTest(k=k):
                self.assertLessEqual(rel * span, PRODUCTION_FLOOR_RAD[k])

    def test_the_branch_joins_are_where_the_fixture_puts_them(self):
        """
        **A characterisation test of an upstream data fixture, not an accuracy test.**

        ``QCD_EOS`` is a transcription of the Saikawa & Shirai parametrisation, and three of its
        four declared branch joins do not match: ``g`` and ``g_s`` jump across them (audit §1).
        This campaign **deliberately does not repair that** (README §0.5); the open question for
        the fixture's authors is README §7 D6, and the board entry is
        ``[00-eos-branch-joins-do-not-match]``.

        A failure here therefore means **the fixture changed**, not that the code regressed -- and
        it means the campaign's segment edges, which are placed at exactly these crossings, must be
        re-derived.
        """
        eos = self.cosmology._eos
        GeV = self.cosmology.units.GeV

        print("\n[EOS branch joins] evaluated at T*(1 +/- 1e-9)")
        measured = {}
        for T_GeV in sorted(set(eos.break_temperatures_GeV), reverse=True):
            T = T_GeV * GeV
            g_lo = eos.G(T * (1.0 - BRANCH_JOIN_SEPARATION))
            g_hi = eos.G(T * (1.0 + BRANCH_JOIN_SEPARATION))
            s_lo = eos.Gs(T * (1.0 - BRANCH_JOIN_SEPARATION))
            s_hi = eos.Gs(T * (1.0 + BRANCH_JOIN_SEPARATION))
            dg = (g_hi - g_lo) / g_lo
            ds = (s_hi - s_lo) / s_lo
            measured[T_GeV] = (dg, ds)
            print(
                f"  T = {T_GeV:>10g} GeV:  dg/g = {dg:+.6e}   dgs/gs = {ds:+.6e}"
                f"   -> forced dT/T = {-ds / 3.0:+.6e}"
            )

        self.assertEqual(
            set(eos.break_temperatures_GeV),
            set(BRANCH_JOINS) | {CONTINUOUS_JOIN_GeV},
            "QCD_EOS.break_temperatures_GeV changed; the audit's §1 table no longer applies",
        )

        for T_GeV, (dg_expected, ds_expected) in BRANCH_JOINS.items():
            dg, ds = measured[T_GeV]
            with self.subTest(T_GeV=T_GeV):
                self.assertAlmostEqual(
                    dg / dg_expected,
                    1.0,
                    delta=BRANCH_JOIN_RTOL,
                    msg=f"g jumps by {dg:.6e} at {T_GeV} GeV, audit §1 says {dg_expected:.4e}",
                )
                self.assertAlmostEqual(
                    ds / ds_expected,
                    1.0,
                    delta=BRANCH_JOIN_RTOL,
                    msg=f"g_s jumps by {ds:.6e} at {T_GeV} GeV, audit §1 says {ds_expected:.4e}",
                )

        dg, ds = measured[CONTINUOUS_JOIN_GeV]
        self.assertLessEqual(abs(dg), CONTINUOUS_JOIN_BOUND)
        self.assertLessEqual(abs(ds), CONTINUOUS_JOIN_BOUND)
        self.assertEqual(
            QCD_EOS.EOS_T_LO,
            CONTINUOUS_JOIN_GeV,
            "the one join that matches is EOS_T_LO, where only w kinks",
        )

    def test_T_z_is_a_step_at_the_lowest_crossing(self):
        """
        ``T(z)`` is a **step**, not a kink (audit §2). This is what justifies segmenting the
        representation in prompt 06, and it is the fact that makes segmentation *sufficient*: the
        entropy factor is exactly constant on each side, so a segmented representation reproduces
        the fixture exactly rather than merely accurately.

        ``g_s`` is piecewise constant across the lowest crossing (3.940 -> 3.931), so on each
        branch ``T ~ (1+z)`` exactly and ``F(u) = log(T / (T_CMB (1+z)))`` is flat to twelve
        decimals on either side with a step between.
        """
        z_c = expm1(self.jumps[0])
        self.assertAlmostEqual(
            z_c / LOWEST_CROSSING_Z,
            1.0,
            delta=1.0e-05,
            msg=f"the lowest crossing moved: {z_c:.6e} against the audit's {LOWEST_CROSSING_Z:.6e}",
        )

        below = [
            entropy_factor(self.cosmology, log1p(z_c * f)) for f in (0.95, 0.99, 0.999)
        ]
        above = [
            entropy_factor(self.cosmology, log1p(z_c * f)) for f in (1.001, 1.01, 1.05)
        ]

        print(f"\n[T(z) is a step] lowest crossing at z_c = {z_c:.6e}")
        for f, F in zip((0.95, 0.99, 0.999), below):
            print(f"  z = {z_c * f:16.6e}   F = {F:.12f}")
        for f, F in zip((1.001, 1.01, 1.05), above):
            print(f"  z = {z_c * f:16.6e}   F = {F:.12f}")

        step = above[0] - below[0]
        print(
            f"  step in F = {step:.12e}   -> relative jump in T = {expm1(step):.6e}\n"
            f"  (audit §1's linearised -(1/3) dgs/gs is {LINEARISED_JUMP:.4e}; the difference is "
            f"the linearisation)"
        )

        self.assertLessEqual(max(below) - min(below), BRANCH_FLATNESS)
        self.assertLessEqual(max(above) - min(above), BRANCH_FLATNESS)
        self.assertAlmostEqual(step, ENTROPY_FACTOR_STEP, delta=ENTROPY_FACTOR_STEP_TOL)
        self.assertAlmostEqual(expm1(step), LINEARISED_JUMP, delta=LINEARISED_JUMP_TOL)

    def test_a_segment_edge_bisected_and_one_root_found_disagree(self):
        """
        **The trap prompt 06 must not fall into** (README §2 (b), audit §2).

        ``T(z) - T_break`` has **no root** at a jump. Across a single ulp of ``u`` at the lowest
        crossing it steps from ``-7.53e-04`` to ``+8.84e-06`` relative, never passing through
        zero. A bracketing solver applied to it nevertheless reports ``converged`` and returns a
        point whose residual is 1.2 % of the jump height -- a non-root, announced as a root.

        The practical damage is that the point it returns then depends on the solver's tolerance,
        and lands on the *upper* side of the jump. A naive implementation bracketing over the whole
        tabulated range with ``root_scalar``'s default tolerances lands ~1.1e-12 above the jump,
        which is more than the 1e-12 padding the audit's segmented build uses to keep each
        branch's nodes strictly inside its own segment: the segment *below* the jump would then be
        fitted through nodes taken from the branch *above* it, and the discontinuity would be
        interpolated across after all. The audit records a first attempt that did exactly this and
        measured 5.7e-04 -- the full error, still in place.

        ``jump_locations`` therefore bisects the *monotone* ``T(z)``, which has no such freedom:
        it converges onto the last representable ``u`` on one side of the step.

        A run in which the bracketing solver returned a genuine root would be a **stop**: it would
        mean audit §2's central claim does not hold on this tree, and prompt 06's design rests on
        it.
        """
        u_edge = self.jumps[0]
        T_break = QCD_EOS.T_LO * self.cosmology.units.GeV
        u_lo, u_hi = tabulated_u_range(self.cosmology)

        def difference(u: float) -> float:
            return accurate_T(self.cosmology, expm1(u)) - T_break

        ulp = float(np.spacing(u_edge))
        residual_above = difference(u_edge) / T_break
        residual_below = difference(u_edge - ulp) / T_break

        print(
            f"\n[segment edge] bisected edge u = {u_edge!r}  (z = {expm1(u_edge):.6e})"
        )
        print(
            f"  (T - T_break)/T_break at the edge     = {residual_above:+.6e}\n"
            f"  (T - T_break)/T_break one ulp below   = {residual_below:+.6e}\n"
            "  -- the sign changes inside one ulp without passing through zero: there is no root"
        )

        # neither side of the step is a root: the step is crossed inside one ulp of u
        self.assertGreater(residual_above, NON_ROOT_RESIDUAL)
        self.assertLess(residual_below, -NON_ROOT_RESIDUAL)

        for label, bracket, kwargs in (
            ("naive, full range, defaults", (u_lo, u_hi), {}),
            (
                "naive, full range, tight",
                (u_lo, u_hi),
                dict(xtol=1.0e-15, rtol=1.0e-15),
            ),
            ("naive, local bracket, defaults", (u_edge - 0.5, u_edge + 0.5), {}),
        ):
            with self.subTest(solver=label):
                solved = root_scalar(difference, bracket=bracket, **kwargs)
                residual = difference(solved.root) / T_break
                offset = solved.root - u_edge
                print(
                    f"  {label:<32s} converged={solved.converged}  "
                    f"u - u_bisect = {offset:+.3e} ({offset / ulp:+.1f} ulp)  "
                    f"residual = {residual:+.6e}"
                )
                # it reports success and returns a non-root: that is the trap
                self.assertTrue(solved.converged)
                self.assertGreater(abs(residual), NON_ROOT_RESIDUAL)

        print(
            f"  the audit's segmented build pads by {SEGMENT_PAD:.0e} in u; the production grid "
            f"spacing is {self.grid_spacing_u:.4e},\n"
            "  so a misplaced edge is invisible to any grid-scale comparison"
        )

    # ---------------------------------------------------------------------------------------
    # prompt 06 -- the segmentation
    # ---------------------------------------------------------------------------------------

    def test_H_z_matches_the_exact_background(self):
        """
        ``H(z)`` on the production source grid, against the same cosmology with its temperature
        replaced by ``accurate_T`` (audit §5, README §6.2).

        ``rho_r = a g(T) T^4`` and ``H ~ sqrt(rho)``, so this is the error in ``T`` arriving at
        roughly twice its size in the radiation era -- and it is the quantity the conformal-time
        guard below integrates. Measured on the production grid rather than on the probe set
        because that is where the audit measured it and where every cumulative table is built.
        """
        z_dense = np.sort(production_source_z_values())
        z_dense = z_dense[(z_dense > 1.0) & (z_dense < 1.0e16)]

        shipped = np.array(
            [self.cosmology.Hubble(float(z)) for z in z_dense], dtype=float
        )
        exact = Hubble_with(
            self.cosmology, lambda z: accurate_T(self.cosmology, z), z_dense
        )
        stats = Stats.of(relative(shipped, exact))

        print(f"\n[H(z)] {len(z_dense)} production grid nodes")
        print("  " + stats.format("shipped Hubble against the exact background"))

        self.assertLessEqual(stats.max, HUBBLE_MAX)
        self.assertLessEqual(stats.p90, HUBBLE_P90)
        self.assertLessEqual(stats.median, HUBBLE_MEDIAN)

    def test_the_segment_edges_are_bisected_onto_the_jumps(self):
        """
        The production segment edges are the jumps, to the last bit.

        Two independent statements, and the second is the one that matters:

        * the edges agree with ``T_z_reference.jump_locations`` -- a separate implementation,
          written by prompt 01 against the defining equation rather than against ``_solve_T_z`` --
          to within :data:`EDGE_AGREEMENT_ULP` ulp of ``u``;
        * each edge is *the crossing itself*: ``T(edge) >= T_break`` and ``T(edge - 1 ulp) <
          T_break``, so the edge is the first representable ``u`` at or above the step and the
          point one ulp below it is on the other branch. That is a first-principles check on a
          monotone function, and it is what licenses
          :class:`SegmentedEntropyFactor`'s dispatch, which places an ``u`` exactly equal to an
          edge in the segment *above* it.

        A bracketing solver on ``T(z) - T_break`` passes the first of these (it lands 4 to 317 ulp
        away, which is well inside any relative tolerance one would think to write) and fails the
        second at the tolerances it would plausibly be given -- see
        ``test_a_segment_edge_bisected_and_one_root_found_disagree``.
        """
        edges = self.cosmology._T_z_spline.segment_edges
        GeV = self.cosmology.units.GeV

        # T(z) is increasing, so the k-th edge is the crossing of the k-th coldest break
        # temperature that is reached at all inside the tabulated range. T_HI = 1e16 GeV is not:
        # it is reached at z ~ 1e28, far above max_z.
        u_top = tabulated_u_range(self.cosmology)[1]
        T_top = accurate_T(self.cosmology, expm1(u_top))
        crossings = [
            T_GeV
            for T_GeV in sorted(set(self.cosmology._eos.break_temperatures_GeV))
            if T_GeV * GeV < T_top
        ]

        self.assertEqual(len(edges), len(self.jumps))
        self.assertEqual(len(edges), len(crossings))
        self.assertEqual(
            len(edges), 3, "the production QCD model crosses three of the four"
        )

        print("\n[segment edges] production against T_z_reference.jump_locations")
        for edge, reference in zip(edges, self.jumps):
            ulp = float(np.spacing(edge))
            with self.subTest(edge=edge):
                print(
                    f"  u = {edge!r} ({edge.hex()})   z = {expm1(edge):.9e}   "
                    f"reference - production = {(reference - edge) / ulp:+.1f} ulp"
                )
                self.assertLessEqual(abs(edge - reference), EDGE_AGREEMENT_ULP * ulp)

        # and each edge is the crossing itself, to the last bit
        for edge, T_GeV in zip(edges, crossings):
            T_break = T_GeV * GeV
            ulp = float(np.spacing(edge))
            at = accurate_T(self.cosmology, expm1(edge)) / T_break - 1.0
            below = accurate_T(self.cosmology, expm1(edge - ulp)) / T_break - 1.0
            with self.subTest(T_GeV=T_GeV):
                print(
                    f"  T_break = {T_GeV:>10g} GeV:  (T - T_break)/T_break = {at:+.6e} at the "
                    f"edge, {below:+.6e} one ulp below"
                )
                self.assertGreaterEqual(at, 0.0)
                self.assertLess(below, 0.0)

        self.assertEqual(SEGMENT_EDGE_PAD_LOG1PZ, SEGMENT_PAD)

    def test_an_edge_misplaced_by_one_node_restores_the_error(self):
        """
        **The audit's 5.7e-04, turned into a guard** (README §2 (b), prompt 06 §3 test 3).

        The failure mode this whole design exists to prevent is silent: an edge that misses its
        jump by one node leaves the full jump-height error in place while every other statistic in
        the representation still improves. Here the lowest edge is moved deliberately up by one
        node of its own segment -- so that segment's topmost nodes are taken from the branch above
        the step, and the spline through them interpolates across it -- and the probe-set maximum
        is required to come back to the 1e-4 regime it was at before this prompt.

        The move is one node, not something visible: the production source grid's spacing in ``u``
        is 2.3e-02, and one node of the lowest segment is 1.5e-02, so the misplaced edge is at a
        redshift a *couple of per cent* away from the right one. Nothing that compares redshifts
        at grid resolution would see it.
        """
        edges = list(self.cosmology._T_z_spline.segment_edges)
        u_lo, u_hi = tabulated_u_range(self.cosmology)

        bounds = np.array([u_lo] + edges + [u_hi], dtype=float)
        widths = np.diff(bounds)
        nodes_0 = max(
            DEFAULT_T_Z_SPLINE_ORDER + 1,
            int(round(DEFAULT_T_Z_SPLINE_SAMPLES * widths[0] / widths.sum())),
        )
        spacing = (edges[0] - SEGMENT_EDGE_PAD_LOG1PZ - u_lo) / (nodes_0 - 1)

        moved = [edges[0] + spacing] + edges[1:]
        misplaced = TemperatureRepresentation(
            build_segmented_entropy_spline(
                self.cosmology._entropy_factor_log1pz,
                moved,
                u_lo,
                u_hi,
                samples=DEFAULT_T_Z_SPLINE_SAMPLES,
                order=DEFAULT_T_Z_SPLINE_ORDER,
            ),
            T_CMB=self.cosmology._T_CMB,
            label="T(z) [edge moved by one node]",
            min_z=self.cosmology._T_z_spline._min_z,
            max_z=self.cosmology._T_z_spline._max_z,
        )

        stats = Stats.of(
            relative(np.array([misplaced(float(z)) for z in self.probe_z]), self.T_ref)
        )
        print(
            f"\n[misplaced edge] lowest edge moved up by one node of its own segment "
            f"({spacing:.4e} in u, {nodes_0} nodes below the jump)"
        )
        print(f"  z_edge {expm1(edges[0]):.6e} -> {expm1(moved[0]):.6e}")
        print("  " + stats.format("segmented, one edge one node too high"))

        self.assertGreaterEqual(stats.max, MISPLACED_EDGE_MIN_MAX)

        # and, in the window between the true jump and the edge it was moved to, the full
        # jump-height error is back: those redshifts are on the hot branch, but they dispatch to
        # the segment below, whose spline was fitted through nodes that are mostly on the cold
        # one. The audit's 5.7e-04 is this. Note that the probe set above does not sample this
        # window -- its spacing in u is 5.8e-02 against the window's 1.5e-02 -- which is exactly
        # why a statistic over a grid is not enough to catch a misplaced edge.
        window = [edges[0] + f * spacing for f in (0.25, 0.5, 0.75)]
        worst = 0.0
        for u in window:
            error = abs(
                float(misplaced(u, z_is_log=True))
                - accurate_T(self.cosmology, expm1(u))
            ) / accurate_T(self.cosmology, expm1(u))
            worst = max(worst, error)
            print(f"  inside the displaced window, z = {expm1(u):.6e}: {error:.3e}")
        self.assertGreaterEqual(worst, 1.0e-04)

    def test_the_step_is_reproduced_on_both_sides_of_each_edge(self):
        """
        The representation reproduces each step rather than smoothing it (prompt 06 §3 test 4).

        Immediately either side of every edge -- one ulp of ``u`` below, and at the edge itself --
        the representation is required to agree with ``accurate_T`` to the same floor it reaches
        anywhere else, and the relative step between the two values is quoted. A representation
        that smoothed the step by even one node interval would fail on the lower side, because
        the value there would carry a fraction of the jump.
        """
        edges = self.cosmology._T_z_spline.segment_edges

        print("\n[the step] evaluated one ulp of u either side of each edge")
        for edge in edges:
            ulp = float(np.spacing(edge))
            below_u, above_u = edge - ulp, edge

            with self.subTest(edge=edge):
                # evaluated on u directly rather than through T_photon(expm1(u)): which side of
                # the step a point is on is a last-bit question, and a round trip through z would
                # be asking it of a recovered redshift (CLAUDE.md, README §2 (i))
                values = {}
                for label, u in (("below", below_u), ("at the edge", above_u)):
                    shipped = float(self.cosmology._T_z_spline(u, z_is_log=True))
                    exact = accurate_T(self.cosmology, expm1(u))
                    values[label] = (shipped, exact)
                    error = abs(shipped - exact) / abs(exact)
                    print(
                        f"  z = {expm1(u):.9e} ({label:<11s}): T = {shipped:.12e}, "
                        f"relative error against the defining equation {error:.3e}"
                    )
                    self.assertLessEqual(error, STEP_FLOOR)

                step = values["at the edge"][0] / values["below"][0] - 1.0
                exact_step = values["at the edge"][1] / values["below"][1] - 1.0
                print(
                    f"    step in T across the edge: {step:+.6e} "
                    f"(the defining equation gives {exact_step:+.6e})"
                )
                if abs(exact_step) > STEP_FLOOR:
                    self.assertAlmostEqual(step / exact_step, 1.0, delta=1.0e-06)
                else:
                    # EOS_T_LO = 0.002 GeV, where g_s is continuous to 1.751e-11 and only w
                    # kinks: there is no step to reproduce, and the two segments meeting there
                    # have to agree to the floor instead (README §7 D4)
                    self.assertLessEqual(abs(step), STEP_FLOOR)


class TestConstantEntropyEquationOfState(unittest.TestCase):
    """
    README §2 (g): a ``LambdaCDM_GenericEOS`` built on a constant-``g_s`` equation of state has
    ``F(u) == 0`` identically and ``T = T_CMB (1+z)`` in closed form, so the improved
    representation of prompts 05 and 06 is **exact** on it rather than merely accurate. What the
    *shipped* representation does to that exact ramp is measured here, so that prompt 05 has
    something to beat.
    """

    @classmethod
    def setUpClass(cls):
        units = Mpc_units()
        params = Planck2018()
        cls.cosmology = LambdaCDM_GenericEOS(
            store_id=1,
            eos=PureRadiationEOS(units, lambdaCDM_gstar(params.Neff)),
            units=units,
            params=params,
            max_z=PRODUCTION_MAX_Z,
        )
        cls.probe_z = probe_set()

    def test_a_constant_gs_equation_of_state_is_an_exact_ramp(self):
        """
        With ``g_s`` constant the defining equation is linear in ``T``, so ``_solve_T_z`` is exact
        to round-off even at ``rtol = 1e-4``. The *spline* over those exact nodes was nevertheless
        wrong by 1.9e-07, because it spent 500 points in ``u`` re-deriving a ``(1+z)`` ramp that is
        known in closed form: that was finding T3 with the other two defects switched off.

        Since prompt 05 the splined quantity is ``F(u) = log(T / [T_CMB (1+z)])``, which on this
        equation of state is identically zero. The interpolating spline of a constant is that
        constant, so the representation returns ``T_CMB (1+z)`` in closed form and the measured
        error is 2.9e-16 -- round-off, about 1.3 ulp, rather than interpolation. The
        representation is **exact** on a constant-``g_s`` model, not merely accurate
        (README §2 (g)).

        The equation of state declares no break temperatures, so ``jump_locations`` is empty and
        the model takes the unchanged code path throughout the campaign.
        """
        exact = self.cosmology._T_CMB * (1.0 + self.probe_z)

        reference = reference_temperatures(self.cosmology, self.probe_z)
        nodes = np.array(
            [self.cosmology._solve_T_z(float(z)) for z in self.probe_z], dtype=float
        )
        shipped = np.array(
            [self.cosmology.T_photon(float(z)) for z in self.probe_z], dtype=float
        )

        reference_stats = Stats.of(relative(reference, exact))
        node_stats = Stats.of(relative(nodes, exact))
        shipped_stats = Stats.of(relative(shipped, exact))

        print("\n[constant g_s: T = T_CMB (1+z) exactly]")
        print("  " + reference_stats.format("accurate_T (the reference itself)"))
        print("  " + node_stats.format("shipped _solve_T_z"))
        print("  " + shipped_stats.format("shipped T_photon (the spline)"))

        self.assertEqual(jump_locations(self.cosmology), [])
        # and the representation knows it: no break temperatures, one segment, and the object
        # inside TemperatureRepresentation is a plain BSpline rather than a SegmentedEntropyFactor
        self.assertEqual(self.cosmology._T_z_spline.segment_edges, ())
        self.assertIsInstance(self.cosmology._T_z_spline._spline, BSpline)
        # the reference is exact on an exactly-linear equation, which is a check on the reference
        self.assertLessEqual(reference_stats.max, EXACT_RAMP_NODE_MAX)
        # so is the shipped node solve: brentq lands on a linear root regardless of rtol
        self.assertLessEqual(node_stats.max, EXACT_RAMP_NODE_MAX)
        # the interpolation on top of it is not
        self.assertLessEqual(shipped_stats.max, EXACT_RAMP_MAX)

    def test_the_single_segment_path_is_prompt_05s_path_bit_for_bit(self):
        """
        **A cosmology that declares no break temperatures takes the unchanged code path**
        (README §2 (g), prompt 06 §3 test 6).

        Not "close to prompt 05's", but the same arithmetic: ``build_segmented_entropy_spline``
        with no edges lays down ``linspace(u_lo, u_hi, samples)`` and calls ``make_interp_spline``
        on it, which is what prompt 05's ``_build_T_z_spline`` did in line. The comparison here
        rebuilds prompt 05's construction explicitly, from the same ``_solve_T_z``, and requires
        the two to agree **bit for bit** over 2,000 probes -- not to a tolerance, which would hide
        exactly the kind of drift this is here to exclude.

        The node count and the order have to be passed explicitly, because the shipped defaults
        moved with this prompt (500 / k = 3 to 3,000 / k = 5) and a model with no jumps would
        otherwise be compared against a different tabulation rather than against a different code
        path.
        """
        representation = self.cosmology._T_z_spline
        min_z, max_z = representation._min_z, representation._max_z
        u_lo, u_hi = log(1.0 + min_z), log(1.0 + max_z)

        prompt_05_nodes = np.linspace(u_lo, u_hi, 500)
        prompt_05_spline = make_interp_spline(
            prompt_05_nodes,
            [self.cosmology._entropy_factor_log1pz(float(u)) for u in prompt_05_nodes],
            k=3,
        )
        prompt_05 = TemperatureRepresentation(
            prompt_05_spline,
            T_CMB=self.cosmology._T_CMB,
            label="T(z)",
            min_z=min_z,
            max_z=max_z,
        )
        prompt_06 = TemperatureRepresentation(
            build_segmented_entropy_spline(
                self.cosmology._entropy_factor_log1pz,
                [],
                u_lo,
                u_hi,
                samples=500,
                order=3,
            ),
            T_CMB=self.cosmology._T_CMB,
            label="T(z)",
            min_z=min_z,
            max_z=max_z,
        )

        probes = np.concatenate([[0.0], np.logspace(-3, 16, 2000)])
        identical = sum(
            float(prompt_05(float(z))) == float(prompt_06(float(z))) for z in probes
        )
        print(
            f"\n[single segment] {identical} of {len(probes)} probes bit-identical between "
            f"prompt 05's construction and this one at the same 500 nodes, k = 3"
        )
        self.assertEqual(identical, len(probes))
        self.assertEqual(
            list(np.asarray(prompt_05_spline.t)), list(np.asarray(prompt_06._spline.t))
        )


class TestSegmentGeometry(unittest.TestCase):
    """
    Degenerate segment geometry (prompt 06 §3 test 5).

    Every case here must either work or raise something that names the problem. **None may
    produce a silently wrong representation** -- a segment fitted through nodes from the wrong
    branch is the failure mode the whole design exists to prevent, and it is invisible: the
    representation keeps working, keeps being smooth, and carries the full jump-height error.

    The quantity tabulated is a smooth analytic function rather than a cosmology's entropy
    factor, because what is under test is the geometry rather than the physics: these are
    assertions about ``build_segmented_entropy_spline``, which is the production function the
    cosmology calls.
    """

    U_LO = 0.0
    U_HI = 10.0
    ORDER = 5
    SAMPLES = 200

    @staticmethod
    def F(u: float) -> float:
        return 0.1 * sin(u)

    def build(self, edges, samples=None, order=None, **kwargs):
        return build_segmented_entropy_spline(
            self.F,
            edges,
            self.U_LO,
            self.U_HI,
            samples=self.SAMPLES if samples is None else samples,
            order=self.ORDER if order is None else order,
            **kwargs,
        )

    def test_an_edge_outside_the_range_raises(self):
        for edges in ([self.U_HI + 1.0], [-1.0], [2.0, self.U_HI + 1.0]):
            with self.subTest(edges=edges):
                with self.assertRaises(RuntimeError) as caught:
                    self.build(edges)
                self.assertIn("ascending", str(caught.exception))

    def test_an_edge_exactly_at_a_tabulation_bound_raises(self):
        for edges in ([self.U_LO], [self.U_HI], [self.U_LO, 5.0], [5.0, self.U_HI]):
            with self.subTest(edges=edges):
                with self.assertRaises(RuntimeError):
                    self.build(edges)

    def test_edges_out_of_order_or_repeated_raise(self):
        for edges in ([5.0, 3.0], [4.0, 4.0]):
            with self.subTest(edges=edges):
                with self.assertRaises(RuntimeError):
                    self.build(edges)

    def test_two_edges_closer_than_a_node_spacing_still_build(self):
        """
        A segment narrower than the node spacing the proportional share would give it gets
        ``order + 1`` nodes -- the fewest an interpolating spline of that order can be built
        through -- rather than a lower-order fit, and the representation is still accurate.
        """
        edges = [5.0, 5.0 + 1.0e-06]
        representation = self.build(edges)
        self.assertIsInstance(representation, SegmentedEntropyFactor)
        self.assertEqual(representation.segment_edges, tuple(edges))

        middle = representation._splines[1]
        self.assertEqual(len(middle.c), self.ORDER + 1)

        worst = max(
            abs(float(representation(u)) - self.F(u))
            for u in np.linspace(self.U_LO + 1.0e-09, self.U_HI - 1.0e-09, 5000)
        )
        print(f"\n[segment geometry] two edges 1e-06 apart: worst |error| {worst:.3e}")
        self.assertLessEqual(worst, 1.0e-10)

    def test_a_segment_that_cannot_hold_its_nodes_raises(self):
        """
        Two failures are possible and both are caught: a segment narrower than twice the padding
        that holds its nodes inside their own branch, and a segment wide enough to pad but too
        narrow for ``order + 1`` distinct floats.
        """
        with self.assertRaises(RuntimeError) as caught:
            self.build([5.0, 5.0 + 1.0e-13])
        self.assertIn("padding", str(caught.exception))

        with self.assertRaises(RuntimeError) as caught:
            self.build([5.0, 5.0 + 1.0e-15], pad=0.0)
        self.assertIn("distinct nodes", str(caught.exception))

    def test_no_edges_returns_a_plain_spline(self):
        representation = self.build([])
        self.assertIsInstance(representation, BSpline)

    def test_a_range_with_no_crossings_inside_it_segments_at_nothing(self):
        """
        The production edge finder returns only the crossings strictly inside the tabulated range,
        so a cosmology (or a range) that contains none builds one segment and is unchanged. This
        is the same filter that drops ``T_HI = 1e16`` GeV, whose crossing is at ``z ~ 1e28``.
        """
        cosmology = _qcd_cosmology()
        self.assertEqual(
            cosmology._entropy_segment_edges_log1pz(log1p(1.0), log1p(1.0e3)), []
        )
        self.assertEqual(
            len(cosmology._entropy_segment_edges_log1pz(*tabulated_u_range(cosmology))),
            3,
        )


if __name__ == "__main__":
    unittest.main()
