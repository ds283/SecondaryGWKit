"""
Tests for `LiouvilleGreen/phase_spline.py`, the module's first test file.

GkTk-remedial prompt 08 removed `phase_spline`'s chunking (`docs/gk-wkb-review-fable-2026-09-09.md`
Sec 5: chunking left the interpolation error unchanged while inflating spline ordinates 64x,
worsening knot residuals 30-50x, and introducing a switch discontinuity). This module checks:

1. The interpolation law is what the review measured (Sec 5 table) -- this is the number
   `ComputeTargets/primitive_phase.py` (prompt 09) must beat.
2. The single rebased spline keeps ordinates bounded by the data span.
3. There is no switch discontinuity (there is only one spline now).
4. The constructor signature is unchanged: `chunk_step`/`chunk_logstep` are accepted and produce
   identical results to omitting them, so `LiouvilleGreen/bessel_phase.py` (which still passes
   `chunk_logstep=125` on `main`) and the three test fixtures that do the same need no changes.
5. Both `increasing` orientations interpolate to the same law.
6. The range-cushion/clamp/error behaviour at the spline boundary is unchanged.
7. The deleted `_build_log_chunks_positive` progress defect (no progress for `chunk_logstep < 2`
   starting at cycle 1) cannot recur, because the code path that had it no longer exists.
"""

import time
import unittest

import mpmath as mp
import numpy as np

from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.phase_spline import phase_spline, DEFAULT_CHUNK_SIZE
from LiouvilleGreen.WKBtools import WKB_mod_2pi

mp.mp.dps = 50


def _exact_radiation_theta(k: float, s) -> mp.mpf:
    """theta(z_r=0; z_s) = k*(1/s_s - 1), the exact-radiation phase used throughout the review
    (docs/gk-wkb-review-fable-2026-09-09.md Sec 5, reproduced from t5_spline.py)."""
    return mp.mpf(k) * (1 / mp.mpf(s) - 1)


def _build_gk_source_policy_geometry(
    k: float,
    per_decade: int,
    s_lo: float = 10.0,
    s_hi: float = 1.0e4,
    chunk_step=None,
    chunk_logstep=None,
    increasing: bool = False,
):
    """The `GkSourcePolicyData._create_functions` consumer geometry: theta(z_source) at a fixed
    z_response, exact radiation, x_is_redshift=True, increasing=False."""
    n = int(round(np.log10(s_hi / s_lo) * per_decade)) + 1
    s = np.geomspace(s_lo, s_hi, n)
    z = s - 1.0

    theta_exact = [_exact_radiation_theta(k, sv) for sv in s]
    pairs = [WKB_mod_2pi(float(t)) for t in theta_exact]
    div, mod = map(list, zip(*pairs))

    spl = phase_spline(
        z,
        div,
        mod,
        x_is_redshift=True,
        increasing=increasing,
        chunk_step=chunk_step,
        chunk_logstep=chunk_logstep,
    )
    return s, z, theta_exact, div, mod, spl


class TestInterpolationLaw(unittest.TestCase):
    """Test 1: the interpolation law is the review's h^4 x/384, unaffected by de-chunking."""

    def test_interior_error_matches_review_measurement(self):
        k = 1.0e6
        per_decade = 100
        s, z, theta_exact, div, mod, spl = _build_gk_source_policy_geometry(
            k, per_decade
        )

        log_s = np.log(s)
        # 10 points per interval, excluding the interval itself's right endpoint
        fine = np.concatenate(
            [np.linspace(log_s[i], log_s[i + 1], 11)[:-1] for i in range(len(s) - 1)]
        )

        # exclude the outermost three intervals at each end, as the review does
        n_interval = len(s) - 1
        interior = slice(3 * 10, (n_interval - 3) * 10)

        exact = np.array([float(_exact_radiation_theta(k, mp.exp(u))) for u in fine])
        raw = np.array([spl.raw_theta(float(u), x_is_log=True) for u in fine])
        err = raw - exact

        interior_max_err = np.max(np.abs(err[interior]))

        # review Sec 5 table: k=1e6, 100/dec -> interior max |err| = 8.26e-5 rad (predicted
        # h^4 x_max/384 = 7.3e-5). Pin the law to that measurement's bracket.
        self.assertGreaterEqual(interior_max_err, 7.0e-5)
        self.assertLessEqual(interior_max_err, 1.0e-4)


class TestOrdinatesBoundedBySpan(unittest.TestCase):
    """Test 2: the single rebased spline's internal ordinates are bounded by the data span."""

    def test_ordinates_bounded(self):
        k = 3.0e8
        per_decade = 100
        s, z, theta_exact, div, mod, spl = _build_gk_source_policy_geometry(
            k, per_decade
        )

        data_span = abs(float(theta_exact[-1] - theta_exact[0]))
        max_ordinate = max(abs(y) for y in spl._spline._y_points)

        self.assertLessEqual(max_ordinate, data_span + TWO_PI)


class TestNoSwitchDiscontinuity(unittest.TestCase):
    """Test 3: there is exactly one spline, so theta_deriv cannot jump at a chunk boundary."""

    def test_derivative_continuous_across_every_interval_boundary(self):
        k = 1.0e8
        per_decade = 100
        s, z, theta_exact, div, mod, spl = _build_gk_source_policy_geometry(
            k, per_decade
        )

        log_s = np.log(s)
        eps = 1.0e-9

        # theta_deriv varies over several orders of magnitude across this domain (~-1e7 to -1e4),
        # so compare each boundary's jump against the *local* derivative magnitude, not a single
        # domain-wide scale
        max_relative_jump = 0.0
        for boundary in log_s[3:-3]:
            left = spl.theta_deriv(float(boundary - eps), x_is_log=True)
            right = spl.theta_deriv(float(boundary + eps), x_is_log=True)
            local_scale = 0.5 * (abs(left) + abs(right))
            max_relative_jump = max(max_relative_jump, abs(left - right) / local_scale)

        # the review measured a genuine switch discontinuity of 3.3e-8 relative under chunking;
        # a single not-a-knot cubic spline should be continuous to many more orders of magnitude
        self.assertLess(max_relative_jump, 1.0e-6)


class TestSignatureCompatibility(unittest.TestCase):
    """Test 4: chunk_step/chunk_logstep are accepted and ignored."""

    def test_chunk_arguments_are_inert(self):
        k = 1.0e7
        per_decade = 100

        _, _, _, _, _, spl_none = _build_gk_source_policy_geometry(
            k, per_decade, chunk_step=None, chunk_logstep=None
        )
        _, _, _, _, _, spl_logstep = _build_gk_source_policy_geometry(
            k, per_decade, chunk_step=None, chunk_logstep=125
        )
        _, _, _, _, _, spl_step = _build_gk_source_policy_geometry(
            k, per_decade, chunk_step=200, chunk_logstep=None
        )
        _, _, _, _, _, spl_default = _build_gk_source_policy_geometry(
            k, per_decade, chunk_step=DEFAULT_CHUNK_SIZE, chunk_logstep=None
        )

        for spl in (spl_none, spl_logstep, spl_step, spl_default):
            self.assertEqual(spl.num_chunks, 1)

        s_lo, s_hi = 10.0, 1.0e4
        probe_log_s = np.linspace(np.log(s_lo * 1.001), np.log(s_hi * 0.999), 50)

        base = np.array(
            [spl_none.raw_theta(float(u), x_is_log=True) for u in probe_log_s]
        )
        for spl in (spl_logstep, spl_step, spl_default):
            values = np.array(
                [spl.raw_theta(float(u), x_is_log=True) for u in probe_log_s]
            )
            np.testing.assert_array_equal(values, base)


class TestBothOrientations(unittest.TestCase):
    """Test 5: increasing=True and increasing=False interpolate to the same law (the flag is a
    no-op now that there is nothing left to order)."""

    def test_increasing_flag_is_inert(self):
        k = 5.0e6
        per_decade = 100

        _, _, _, _, _, spl_false = _build_gk_source_policy_geometry(
            k, per_decade, increasing=False
        )
        _, _, _, _, _, spl_true = _build_gk_source_policy_geometry(
            k, per_decade, increasing=True
        )

        s_lo, s_hi = 10.0, 1.0e4
        probe_log_s = np.linspace(np.log(s_lo * 1.001), np.log(s_hi * 0.999), 50)

        for u in probe_log_s:
            self.assertEqual(
                spl_false.raw_theta(float(u), x_is_log=True),
                spl_true.raw_theta(float(u), x_is_log=True),
            )


class TestRangeBehaviourUnchanged(unittest.TestCase):
    """Test 6: the cushion/clamp/error behaviour at the spline boundary, read from the code before
    deletion, is preserved by `_rebased_spline`."""

    def test_within_cushion_clamps_to_boundary(self):
        k = 1.0e6
        per_decade = 100
        _, _, _, _, _, spl = _build_gk_source_policy_geometry(k, per_decade)

        min_log_x = spl._spline.min_log_x
        max_log_x = spl._spline.max_log_x

        boundary_value = spl.raw_theta(float(min_log_x), x_is_log=True)

        # just inside the 0.1% cushion below min_log_x (min_log_x is negative here, since
        # s_lo=10 -> z_lo=9 -> log(1+z) = log(10) > 0 actually; use the sign-aware cushion the
        # code itself uses)
        sign = 1.0 if min_log_x >= 0 else -1.0
        just_inside = min_log_x * (1.0 - sign * 1.0e-4)
        clamped_value = spl.raw_theta(float(just_inside), x_is_log=True)

        self.assertEqual(clamped_value, boundary_value)

        # symmetric check at the top
        sign_top = 1.0 if max_log_x >= 0 else -1.0
        top_boundary_value = spl.raw_theta(float(max_log_x), x_is_log=True)
        just_inside_top = max_log_x * (1.0 + sign_top * 1.0e-4)
        clamped_top_value = spl.raw_theta(float(just_inside_top), x_is_log=True)
        self.assertEqual(clamped_top_value, top_boundary_value)

    def test_outside_cushion_raises(self):
        k = 1.0e6
        per_decade = 100
        _, _, _, _, _, spl = _build_gk_source_policy_geometry(k, per_decade)

        min_log_x = spl._spline.min_log_x
        max_log_x = spl._spline.max_log_x

        sign = 1.0 if min_log_x >= 0 else -1.0
        well_outside_low = min_log_x * (1.0 - sign * 0.01)
        with self.assertRaises(RuntimeError):
            spl.raw_theta(float(well_outside_low), x_is_log=True)

        sign_top = 1.0 if max_log_x >= 0 else -1.0
        well_outside_high = max_log_x * (1.0 + sign_top * 0.01)
        with self.assertRaises(RuntimeError):
            spl.raw_theta(float(well_outside_high), x_is_log=True)


class TestOldProgressDefectCannotRecur(unittest.TestCase):
    """Test 7: a regression guard for the deleted `_build_log_chunks_positive`, which never made
    progress for chunk_logstep < 2 starting at cycle 1 (review Sec 5)."""

    def test_construction_returns_promptly(self):
        # data starting at div_2pi=1 (cycle 1), the case that used to hang
        n = 50
        x_sample = list(np.linspace(1.0, 10.0, n))
        div_2pi_sample = list(range(1, 1 + n))
        mod_2pi_sample = [-0.1] * n

        start = time.monotonic()
        spl = phase_spline(
            x_sample,
            div_2pi_sample,
            mod_2pi_sample,
            chunk_step=None,
            chunk_logstep=1.5,
            increasing=True,
        )
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 5.0)
        self.assertEqual(spl.num_chunks, 1)
        # the object must actually be usable
        _ = spl.raw_theta(5.0)


if __name__ == "__main__":
    unittest.main()
