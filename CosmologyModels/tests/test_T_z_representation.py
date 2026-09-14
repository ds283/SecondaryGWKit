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

**On 2026-09-13 this is a characterisation harness, not a pass/fail accuracy guard.** Every
threshold below is set at the value the tree *currently* achieves, and each carries a comment
naming the campaign prompt that tightens it (README §6.1, §6.2). The transition from
characterisation to guard is meant to be a one-line threshold edit in each case and nothing else.

Two of the seven cases are not accuracy measurements at all:

* ``test_the_branch_joins_are_where_the_fixture_puts_them`` pins an upstream data fixture this
  campaign deliberately does not repair (README §0.5, §7 D6);
* ``test_a_segment_edge_bisected_and_one_root_found_disagree`` pins the one implementation trap
  prompt 06 has to avoid (README §2 (b)).

No Ray and no datastore is needed.
"""

import unittest
from math import expm1, log1p

import numpy as np
from scipy.optimize import root_scalar

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.tests.T_z_reference import (
    PRODUCTION_FLOOR_RAD,
    PRODUCTION_SPANS_RAD,
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
# Tightened by prompt 04 (p90 -> 2.0e-07, achieved 1.936e-07; median -> 1.1e-07, achieved
# 1.071e-07). The max is *not* tightened here: accurate nodes fix the p90, not the max, which
# stays pinned near the jump height until prompt 06 segments the representation -- it in fact
# moves slightly worse, 7.177e-04 -> 7.26e-04 (achieved 7.2615e-04), because a different set of
# nodes now bracket the jump. This is README §6.1's "After 04" column, measured, not a regression:
# prompt 05 (median -> 3.0e-10) and prompt 06 (max -> 1e-10, p90 -> 1e-14, median -> 1e-15)
# tighten the rest.
T_PHOTON_MAX = 7.27e-04
T_PHOTON_P90 = 2.0e-07
T_PHOTON_MEDIAN = 1.1e-07

# case 2 -- the node solve's own rtol = 1e-4, audit §3 (T2). Tightened by prompt 04 to 1e-14
# (achieved: bit-identical to the rtol=1e-14 reference on the probe set, so exactly 0.0).
NODE_SOLVE_MAX = 1.0e-14

# case 3 -- the T1 guard, audit §5 / README §6.2. Tightened by prompt 06 to 1e-15.
CONFORMAL_TIME_REL = 4.0e-08
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
# Tightened by prompt 05, where the new representation should be exact to a few ulp.
EXACT_RAMP_MAX = 2.0e-07
EXACT_RAMP_NODE_MAX = 1.0e-15

# case 7 -- the segment-edge trap, README §2 (b). A relative residual this far above machine
# epsilon means the bracketing solver did not find a root; it reported one anyway.
NON_ROOT_RESIDUAL = 1.0e-08
# the padding the audit's segmented build uses to keep each branch's nodes inside its own segment
# (docs/qcd-background-audit/measure_T_z_representation.py, build_segmented)
SEGMENT_PAD = 1.0e-12


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
        print(f"  relative error in tau = {rel:.4e}")
        for k, span in PRODUCTION_SPANS_RAD.items():
            print(
                f"    k = {k:9.3g} /Mpc:  {rel * span:10.3e} rad   "
                f"against a 1-ulp floor of {PRODUCTION_FLOOR_RAD[k]:.3e} rad"
            )

        self.assertLessEqual(rel, CONFORMAL_TIME_REL)

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
        to round-off even at ``rtol = 1e-4`` -- and the shipped *spline* over those exact nodes is
        still wrong by 1.9e-07, because it spends 500 points in ``u`` re-deriving a ``(1+z)`` ramp
        that is known in closed form. That is finding T3 with the other two defects switched off.

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
        # the reference is exact on an exactly-linear equation, which is a check on the reference
        self.assertLessEqual(reference_stats.max, EXACT_RAMP_NODE_MAX)
        # so is the shipped node solve: brentq lands on a linear root regardless of rtol
        self.assertLessEqual(node_stats.max, EXACT_RAMP_NODE_MAX)
        # the interpolation on top of it is not
        self.assertLessEqual(shipped_stats.max, EXACT_RAMP_MAX)


if __name__ == "__main__":
    unittest.main()
