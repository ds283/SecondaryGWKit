"""
Tests for `ComputeTargets/primitive_phase.py` (GkTk-remedial prompt 09).

`PrimitivePhase` replaces the consumers' cubic spline of the *growing* WKB phase with a closed-form
leading term evaluated from the background model's double-double conformal-time table plus a cubic
spline of the small residual `phi`. The review (`docs/gk-wkb-review-fable-2026-09-09.md` Sec 5)
measures the representation it replaces at `h^4 x_source/384` -- 8.26e-3 rad at `x_source = 1e8`,
100 samples per decade -- so the test that matters is a head-to-head against a `phase_spline` of
exactly the same samples, which is what `TestInterpolationLaw` below does.

Offline: exact radiation (`H = H0 (1+z)^2`, `tau = 1/(H0(1+z))`, `rho == 0` identically), a real
`CumulativeTable` built from `f = 1/H` on a production-shaped 100-per-decade grid, and mpmath at
50 digits for the reference. No Ray, no datastore, no cosmology object.

**On the accuracy threshold.** Prompt 09 Sec 4 test 1 asks for <= 1e-8 rad on this geometry. The
geometry it specifies (`k = 1e8`, `z_r = 0.1`, `z_s` up to 1e4) has `|theta|` up to 9.09e7 rad, one
ulp of which is 1.49e-8 rad, so 1e-8 rad is below the representation floor and cannot be asserted:
2.0e-8 rad is the campaign's own `eps k tau` floor at this `k` (README Sec 2 (d), Sec 5 standing
note 2, decision D6). The measured error is 4.19e-8 rad = 2.81 ulp of the span, and what is
asserted here is README Sec 6's consumer row (<= 1e-6 rad) together with a floor-aware 6-ulp bound
that pins the result to the floor rather than to a round number. See the prompt 09 log,
deviation 1.
"""

import unittest
from math import log10, log1p, expm1

import mpmath as mp
import numpy as np

from ComputeTargets.BackgroundModel import TablePrimitive
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.phase_spline import phase_spline

mp.mp.dps = 50

# prompt 09 Sec 4: the consumer geometry, scaled to k = 1e8
K = 1.0e8
Z_RESPONSE = 0.1
Z_SOURCE_LO = 10.0
Z_SOURCE_HI = 1.0e4
PER_DECADE = 100
TAU_GAUSS_ORDER = 4

# README Sec 6's consumer row
CONSUMER_TARGET_RAD = 1.0e-6
# ... and the floor-aware bound, in ulp of the largest |theta| on the geometry (deviation 1)
CONSUMER_FLOOR_ULP = 6.0
# prompt 09 Sec 4 test 1: PrimitivePhase must beat a phase_spline of the same samples by this
PHASE_SPLINE_RATIO = 1.0e5

# test 2: the closed-form derivative, relative
DERIV_REL_TOL = 1.0e-12

# test 4: phi = PHI_AMPLITUDE * sin(log(1+z)), recovered to the cubic-spline law of phi itself
PHI_AMPLITUDE = 0.3
PHI_SPLINE_TOL = 1.0e-8

RADIATION_H0 = 1.0


# ------------------------------------------------------------------------------------------------
# exact radiation background
# ------------------------------------------------------------------------------------------------


def _Hubble(z: float) -> float:
    return RADIATION_H0 * (1.0 + z) * (1.0 + z)


class _Functions:
    """The one field `PrimitivePhase` reads off a `ModelFunctions`."""

    Hubble = staticmethod(_Hubble)


def _exact_theta(z_s, z_r) -> float:
    """
    `theta(z_r; z_s) = -k tau.delta(z_s, z_r) = -k [tau(z_r) - tau(z_s)] = -k (1/s_r - 1/s_s)`,
    negative for `z_s > z_r` (campaign README Sec 2 (c)). Evaluated at 50 digits from the supplied
    doubles, so that the reference is not itself the limiting error: at `k = 1e8` a double
    reference on this geometry carries 2.8e-8 rad of its own rounding, comparable with what is
    being measured.
    """
    s_s = mp.mpf(float(z_s)) + 1
    s_r = mp.mpf(float(z_r)) + 1
    return -mp.mpf(K) * (1 / s_r - 1 / s_s)


def _grid():
    """
    A production-shaped grid, 100 nodes per decade of `1+z`, whose bottom node is exactly the
    response redshift and which carries the source samples as nodes -- the production case, where
    the background model is built on the source grid (`main.py:476`) and the response grid is a
    subset of it.
    """
    s_lo = 1.0 + Z_RESPONSE
    s_hi = 1.0 + Z_SOURCE_HI
    n = int(round(log10(s_hi / s_lo) * PER_DECADE)) + 1
    s_nodes = np.geomspace(s_lo, s_hi, n)
    z_nodes = (s_nodes - 1.0)[::-1]  # descending, as CumulativeTable requires
    return np.ascontiguousarray(z_nodes)


class _Geometry:
    """Built once: the table, the samples and the reference."""

    z_nodes = None
    leading = None
    z_samples = None
    z_response = None

    @classmethod
    def build(cls):
        if cls.leading is not None:
            return
        cls.z_nodes = _grid()
        cls.z_response = float(cls.z_nodes[-1])
        table = CumulativeTable(
            cls.z_nodes, lambda z: 1.0 / _Hubble(z), TAU_GAUSS_ORDER, label="tau"
        )
        cls.leading = TablePrimitive(table, "tau")
        z = cls.z_nodes[::-1]  # ascending
        cls.z_samples = z[(z >= Z_SOURCE_LO) & (z <= Z_SOURCE_HI)]

    @classmethod
    def fine_abscissae(cls, skip_intervals: int = 3, per_interval: int = 10):
        """`per_interval` interior points in every sample interval, in `u = log(1+z)`, excluding
        `skip_intervals` intervals at each end (the review's Sec 5 protocol)."""
        u = np.log1p(cls.z_samples)
        out = []
        for i in range(skip_intervals, len(u) - 1 - skip_intervals):
            for j in range(1, per_interval + 1):
                out.append(u[i] + (u[i + 1] - u[i]) * j / (per_interval + 1.0))
        return np.array(out, dtype=float)


def _phase(phi_samples, spline_order: int = 3) -> PrimitivePhase:
    _Geometry.build()
    return PrimitivePhase(
        K,
        _Geometry.leading,
        _Geometry.z_response,
        _Geometry.z_samples,
        phi_samples,
        sign=-1,
        model_functions=_Functions,
        label="test",
        spline_order=spline_order,
    )


# ------------------------------------------------------------------------------------------------
# 1. the interpolation law, head to head against phase_spline
# ------------------------------------------------------------------------------------------------


class TestInterpolationLaw(unittest.TestCase):
    """Prompt 09 Sec 4 test 1 / README Sec 6's consumer row."""

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_raw_theta_beats_a_phase_spline_of_the_same_samples(self):
        z_s = _Geometry.z_samples
        pp = _phase(np.zeros_like(z_s))

        # a phase_spline of exactly the same samples, in the GkSourcePolicyData configuration
        pairs = [WKB_mod_2pi(float(_exact_theta(z, _Geometry.z_response))) for z in z_s]
        div, mod = map(list, zip(*pairs))
        spl = phase_spline(
            [log1p(float(z)) for z in z_s],
            div,
            mod,
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=125,
            increasing=False,
        )

        fine = _Geometry.fine_abscissae()
        err_pp = []
        err_spl = []
        worst = None
        for u in fine:
            z = expm1(float(u))
            exact = _exact_theta(z, _Geometry.z_response)
            e_pp = abs(mp.mpf(pp.raw_theta(float(u), x_is_log=True)) - exact)
            err_pp.append(float(e_pp))
            err_spl.append(
                float(abs(mp.mpf(spl.raw_theta(float(u), x_is_log=True)) - exact))
            )
            if worst is None or err_pp[-1] > worst[1]:
                worst = (z, err_pp[-1])

        max_pp = max(err_pp)
        max_spl = max(err_spl)
        ratio = max_spl / max_pp

        theta_max = max(abs(float(_exact_theta(z, _Geometry.z_response))) for z in z_s)
        ulp = float(np.spacing(theta_max))
        eps_k_tau = float(np.finfo(float).eps * theta_max)

        print(
            f"\n[prompt 09 test 1] k = {K:.3g}/Mpc, z_r = {_Geometry.z_response:.6g}, "
            f"z_s in [{Z_SOURCE_LO:g}, {Z_SOURCE_HI:g}] at {PER_DECADE}/decade "
            f"({len(z_s)} samples, {len(fine)} interior evaluation points):"
        )
        print(
            f"  PrimitivePhase max |dtheta| = {max_pp:.4g} rad "
            f"(at z_s = {worst[0]:.6g}) = {max_pp / ulp:.2f} ulp of the "
            f"{theta_max:.4g} rad span"
        )
        print(
            f"  phase_spline   max |dtheta| = {max_spl:.4g} rad  ->  ratio "
            f"{ratio:.4g} (required > {PHASE_SPLINE_RATIO:.0e})"
        )
        print(
            f"  floors: eps*k*tau = {eps_k_tau:.3g} rad, 1 ulp = {ulp:.3g} rad; "
            f"README Sec 6 consumer target {CONSUMER_TARGET_RAD:.0e} rad"
        )

        # README Sec 6's consumer row
        self.assertLessEqual(max_pp, CONSUMER_TARGET_RAD)
        # ... and the floor-aware bound (prompt 09 asks for 1e-8 rad, which is below the
        # representation floor on its own geometry -- see the module docstring)
        self.assertLessEqual(max_pp, CONSUMER_FLOOR_ULP * ulp)
        # prompt 09 Sec 4 test 1
        self.assertGreater(ratio, PHASE_SPLINE_RATIO)
        # the phase_spline is the review's measurement, scaled: 8.26e-3 rad at x = 1e8
        self.assertGreaterEqual(max_spl, 1.0e-3)

    def test_at_the_samples_the_leading_term_is_all_there_is(self):
        """With `phi == 0` the value at a sample is the interval accessor alone, so the error
        there is the table's, not the spline's."""
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        errs = [
            float(
                abs(
                    mp.mpf(pp.raw_theta(float(z)))
                    - _exact_theta(z, _Geometry.z_response)
                )
            )
            for z in _Geometry.z_samples
        ]
        print(f"[prompt 09 test 1] at the samples: max |dtheta| = {max(errs):.4g} rad")
        self.assertLessEqual(max(errs), CONSUMER_TARGET_RAD)


# ------------------------------------------------------------------------------------------------
# 2. the closed-form derivative
# ------------------------------------------------------------------------------------------------


class TestClosedFormDerivative(unittest.TestCase):
    """Prompt 09 Sec 4 test 2."""

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_theta_deriv_matches_the_closed_form(self):
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        fine = _Geometry.fine_abscissae(per_interval=3)

        worst_dz = 0.0
        worst_dlog = 0.0
        for u in fine:
            z = expm1(float(u))
            # theta = -k(1/s_r - 1/s_s) so d theta/d z_s = -k/s_s^2 = -k/H (H0 = 1)
            exact_dz = -K / _Hubble(z)
            got_dz = pp.theta_deriv(float(u), x_is_log=True)
            worst_dz = max(worst_dz, abs(got_dz - exact_dz) / abs(exact_dz))

            exact_dlog = exact_dz * (1.0 + z)
            got_dlog = pp.theta_deriv(float(u), x_is_log=True, log_derivative=True)
            worst_dlog = max(worst_dlog, abs(got_dlog - exact_dlog) / abs(exact_dlog))

        print(
            f"\n[prompt 09 test 2] d theta/dz relative error {worst_dz:.3g}, "
            f"d theta/d log(1+z) relative error {worst_dlog:.3g} "
            f"(required <= {DERIV_REL_TOL:.0e})"
        )
        self.assertLessEqual(worst_dz, DERIV_REL_TOL)
        self.assertLessEqual(worst_dlog, DERIV_REL_TOL)

    def test_redshift_and_log_abscissae_agree(self):
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        for u in _Geometry.fine_abscissae(per_interval=2)[:50]:
            z = expm1(float(u))
            self.assertAlmostEqual(
                pp.theta_deriv(z), pp.theta_deriv(float(u), x_is_log=True), delta=0.0
            )
            self.assertAlmostEqual(
                pp.raw_theta(z), pp.raw_theta(float(u), x_is_log=True), delta=0.0
            )


# ------------------------------------------------------------------------------------------------
# 3. the mod-2pi convention
# ------------------------------------------------------------------------------------------------


class TestModTwoPiConvention(unittest.TestCase):
    """Prompt 09 Sec 4 test 3."""

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_negative_remainder_and_agreement_with_WKB_mod_2pi(self):
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        for u in _Geometry.fine_abscissae(per_interval=2):
            got = pp.theta_mod_2pi(float(u), x_is_log=True)
            self.assertGreater(got, -TWO_PI)
            self.assertLessEqual(got, 0.0)
            self.assertEqual(got, WKB_mod_2pi(pp.raw_theta(float(u), x_is_log=True))[1])


# ------------------------------------------------------------------------------------------------
# 4. a non-zero smooth residual
# ------------------------------------------------------------------------------------------------


def _phi_exact(z) -> float:
    return PHI_AMPLITUDE * np.sin(np.log1p(z))


class TestSmoothResidual(unittest.TestCase):
    """Prompt 09 Sec 4 test 4."""

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_phi_is_recovered_to_its_own_spline_law(self):
        z_s = _Geometry.z_samples
        pp = _phase(_phi_exact(z_s))

        fine = _Geometry.fine_abscissae()
        err = [
            abs(pp.phi(float(u), x_is_log=True) - _phi_exact(expm1(float(u))))
            for u in fine
        ]

        u = np.log1p(z_s)
        h = float(np.max(np.diff(u)))
        predicted = h**4 * PHI_AMPLITUDE / 384.0
        print(
            f"\n[prompt 09 test 4] phi = {PHI_AMPLITUDE} sin(log(1+z)): max |dphi| = "
            f"{max(err):.4g} rad against the cubic law h^4 max|phi''''|/384 = {predicted:.3g} "
            f"(h = {h:.4g})"
        )
        self.assertLessEqual(max(err), PHI_SPLINE_TOL)
        # the cubic law, to within an order of magnitude either way
        self.assertLessEqual(max(err), 10.0 * predicted)

    def test_theta_deriv_includes_phi_prime(self):
        z_s = _Geometry.z_samples
        pp = _phase(_phi_exact(z_s))

        worst = 0.0
        for u in _Geometry.fine_abscissae(per_interval=3):
            z = expm1(float(u))
            got = pp.theta_deriv(float(u), x_is_log=True, log_derivative=True)
            leading = -K * (1.0 + z) / _Hubble(z)
            # d phi/d log(1+z) = PHI_AMPLITUDE cos(log(1+z))
            worst = max(worst, abs(got - leading - PHI_AMPLITUDE * np.cos(u)))
        print(
            f"[prompt 09 test 4] d theta/d log(1+z) minus the closed-form leading term "
            f"reproduces d phi/d log(1+z) to {worst:.3g}"
        )
        self.assertLessEqual(worst, 1.0e-6)

    def test_phi_is_carried_into_raw_theta(self):
        """
        `raw_theta` is the leading term *plus* phi, not the leading term alone. The two values
        differenced here are both ~9e7 rad, so the comparison is only good to a few ulp of that
        (the eps k tau floor again), not to the accuracy of phi itself.
        """
        z_s = _Geometry.z_samples
        with_phi = _phase(_phi_exact(z_s))
        without = _phase(np.zeros_like(z_s))
        z = float(z_s[len(z_s) // 2])
        self.assertAlmostEqual(
            with_phi.raw_theta(z) - without.raw_theta(z),
            float(_phi_exact(z)),
            delta=1.0e-7,
        )


# ------------------------------------------------------------------------------------------------
# 5. the phase_spline protocol
# ------------------------------------------------------------------------------------------------


class TestProtocol(unittest.TestCase):
    """Prompt 09 Sec 4 test 5. The three methods every consumer calls, plus `num_chunks`."""

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_num_chunks_is_one(self):
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        self.assertEqual(getattr(pp, "num_chunks", None), 1)

    def test_clamped_phase_style_calls(self):
        """`QuadSourceIntegral._ClampedPhase` calls all three with `x_is_log=True` after clamping
        the abscissa into the phase's own range; `phase_groups._composed_*` does the same.
        """
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        lo, hi = pp.min_log_x, pp.max_log_x
        for u in (lo, 0.5 * (lo + hi), hi):
            self.assertTrue(np.isfinite(pp.raw_theta(u, x_is_log=True)))
            self.assertTrue(np.isfinite(pp.theta_mod_2pi(u, x_is_log=True)))
            self.assertTrue(
                np.isfinite(pp.theta_deriv(u, x_is_log=True, log_derivative=True))
            )

    def test_out_of_range_is_clamped_then_refused(self):
        pp = _phase(np.zeros_like(_Geometry.z_samples))
        lo, hi = pp.min_log_x, pp.max_log_x

        # inside the cushion: clamped, not refused, and continuous with the boundary value
        just_below = lo * (1.0 - 0.5e-3)
        just_above = hi * (1.0 + 0.5e-3)
        self.assertTrue(np.isfinite(pp.raw_theta(just_below, x_is_log=True)))
        self.assertTrue(np.isfinite(pp.raw_theta(just_above, x_is_log=True)))

        # beyond it: refused
        with self.assertRaises(RuntimeError):
            pp.raw_theta(lo * (1.0 - 1.0e-2), x_is_log=True)
        with self.assertRaises(RuntimeError):
            pp.raw_theta(hi * (1.0 + 1.0e-2), x_is_log=True)
        with self.assertRaises(RuntimeError):
            pp.theta_deriv(hi * (1.0 + 1.0e-2), x_is_log=True)

    def test_only_phi_is_clamped(self):
        """Inside the cushion the leading term is still evaluated at the requested redshift, so
        `raw_theta` keeps moving; only `phi` is frozen at the boundary."""
        pp = _phase(_phi_exact(_Geometry.z_samples))
        hi = pp.max_log_x
        at_edge = pp.raw_theta(hi, x_is_log=True)
        inside_cushion = pp.raw_theta(hi * (1.0 + 0.5e-3), x_is_log=True)
        self.assertNotEqual(at_edge, inside_cushion)
        self.assertEqual(
            pp.phi(hi, x_is_log=True), pp.phi(hi * (1.0 + 0.5e-3), x_is_log=True)
        )


# ------------------------------------------------------------------------------------------------
# 6. constructor contract
# ------------------------------------------------------------------------------------------------


class TestConstructor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _Geometry.build()

    def test_sign_must_be_plus_or_minus_one(self):
        for bad in (0, 2, -2):
            with self.assertRaises(ValueError):
                _ = PrimitivePhase(
                    K,
                    _Geometry.leading,
                    _Geometry.z_response,
                    _Geometry.z_samples,
                    np.zeros_like(_Geometry.z_samples),
                    sign=bad,
                    model_functions=_Functions,
                )

    def test_both_signs_describe_the_same_phase(self):
        """`sign = +1` is prompt 10's transfer-function convention: with the anchor and the
        abscissa exchanged the two give the same number, because `delta` is antisymmetric.
        """
        z_s = _Geometry.z_samples
        minus = _phase(np.zeros_like(z_s))
        z_probe = float(z_s[len(z_s) // 3])
        plus_leading = +1 * K * _Geometry.leading.delta(_Geometry.z_response, z_probe)
        self.assertAlmostEqual(minus.raw_theta(z_probe), plus_leading, delta=0.0)

    def test_too_few_samples_for_the_spline_order(self):
        with self.assertRaises(RuntimeError):
            _ = PrimitivePhase(
                K,
                _Geometry.leading,
                _Geometry.z_response,
                _Geometry.z_samples[:3],
                np.zeros(3),
                sign=-1,
                model_functions=_Functions,
            )

    def test_quintic_spline_order_is_allowed(self):
        z_s = _Geometry.z_samples
        pp = _phase(_phi_exact(z_s), spline_order=5)
        self.assertEqual(pp.spline_order, 5)
        err = max(
            abs(pp.phi(float(u), x_is_log=True) - _phi_exact(expm1(float(u))))
            for u in _Geometry.fine_abscissae(per_interval=3)
        )
        print(f"\n[prompt 09] quintic phi spline: max |dphi| = {err:.3g} rad")
        self.assertLessEqual(err, PHI_SPLINE_TOL)

    def test_build_phi_samples_inverts_raw_theta(self):
        """`build_phi_samples` is the inverse of the decomposition: feed it a stored phase and the
        resulting `PrimitivePhase` reproduces that phase at the samples."""
        z_s = _Geometry.z_samples
        stored = [
            float(_exact_theta(z, _Geometry.z_response)) + 3.0 * TWO_PI for z in z_s
        ]
        phi = build_phi_samples(
            K, _Geometry.leading, _Geometry.z_response, z_s, stored, sign=-1
        )
        pp = PrimitivePhase(
            K,
            _Geometry.leading,
            _Geometry.z_response,
            z_s,
            phi,
            sign=-1,
            model_functions=_Functions,
        )
        worst = max(abs(pp.raw_theta(float(z)) - s) for z, s in zip(z_s, stored))
        print(
            f"[prompt 09] build_phi_samples round trip at the samples: {worst:.3g} rad; "
            f"max |phi| = {np.max(np.abs(phi)):.4g} rad (3 cycles = {3 * TWO_PI:.4g})"
        )
        self.assertLessEqual(worst, 1.0e-7)
        # phi is the constant offset, recovered
        self.assertAlmostEqual(float(np.mean(phi)), 3.0 * TWO_PI, places=6)


if __name__ == "__main__":
    unittest.main()
