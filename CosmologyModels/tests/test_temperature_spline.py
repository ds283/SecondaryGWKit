"""
Regression tests for the T(z) tabulation inside LambdaCDM_GenericEOS.

`_build_T_z_spline` adds a 5 % buffer to the requested [min_z, max_z] so that a caller asking for
exactly the requested bounds is inside the tabulated range rather than on its edge. The buffer was
applied to z rather than to 1+z, which is only outward-going while z is positive: with
min_z = DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2, `0.95 * min_z` is -0.19, so the range was
*narrowed* by 5 % at the end where the padding was wanted, and the model could not be evaluated at
its own declared floor. 1+z is positive throughout (z > -1), so the buffer is applied to that.

The negative end exists because the model needs z = 0 to match the CMB temperature and a margin
below it so that numerical derivatives at z = 0 are accurate (the comment in the constructor).
Nothing in the pipeline asks for results below z = 0; this is padding, not a physics range.

Board issue `[01-genericeos-tz-spline-floor]` stays open: whether a fixed 500-point grid over
[min_z, max_z] is adequately defined is a separate question from the sign of this buffer, and the
grid was a hot-fix.

No Ray and no datastore is needed.
"""

import unittest
from math import log, exp

import numpy as np

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import (
    LambdaCDM_GenericEOS,
    DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT,
)
from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.tests.test_wPerturbations import PureRadiationEOS, lambdaCDM_gstar
from Units import Mpc_units

MAX_Z = 1.0e4

# The spline's own interpolation error at MAX_Z, from `[01-genericeos-tz-spline-floor]`. Any
# consequence of moving the buffer has to be below this to count as numerically invisible.
INTERPOLATION_FLOOR = 1.3e-9


class TestTemperatureSplineRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = Mpc_units()
        cls.model = LambdaCDM_GenericEOS(
            store_id=1,
            eos=PureRadiationEOS(cls.units, lambdaCDM_gstar(Planck2018().Neff)),
            units=cls.units,
            params=Planck2018(),
            max_z=MAX_Z,
        )
        cls.spline = cls.model._T_z_spline

    def test_the_buffer_widens_the_range_at_both_ends(self):
        lo, hi = self.spline._min_z, self.spline._max_z
        print(
            f"\n[T(z) buffer] requested [{DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT}, {MAX_Z:g}] "
            f"-> tabulated [{lo:.6g}, {hi:.6g}]  "
            f"(the retired buffer gave [{0.95 * DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT:.6g}, "
            f"{1.05 * MAX_Z:.6g}], narrower at the low end)"
        )
        self.assertLess(lo, DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT)
        self.assertGreater(hi, MAX_Z)
        # 5 % of 1+z at each end
        self.assertAlmostEqual(
            1.0 + lo, 0.95 * (1.0 + DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT)
        )
        self.assertAlmostEqual((1.0 + hi) / (1.0 + MAX_Z), 1.05)

    def test_the_declared_floor_is_evaluable(self):
        """
        The regression: `T(DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT)` raised
        `RuntimeError: ... out of bounds ... (min allowed z=-0.19 ...)` before the fix.
        """
        for z in (DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, -0.19, -0.1, 0.0):
            with self.subTest(z=z):
                T = self.spline(z)
                self.assertTrue(np.isfinite(T))
                self.assertGreater(T, 0.0)

    def test_the_requested_ceiling_is_evaluable(self):
        for z in (MAX_Z, 0.5 * MAX_Z, 1.0):
            with self.subTest(z=z):
                self.assertTrue(np.isfinite(self.spline(z)))

    def test_genuinely_out_of_range_still_raises(self):
        for z in (-0.5, 1.0e6):
            with self.subTest(z=z):
                with self.assertRaises(RuntimeError):
                    self.spline(z)

    def test_T_is_unchanged_where_the_pipeline_actually_looks(self):
        """
        Moving the low bound moves every spline node, so values in the physics region change at
        the interpolation-error level. Confirm that is all it is, by rebuilding the retired range
        through the same code path and comparing over z in [0, MAX_Z].
        """
        retired = self.model._build_T_z_spline(
            # inputs chosen so the corrected formula reproduces the retired bounds
            min_z=(1.0 + 0.95 * DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT) / 0.95 - 1.0,
            max_z=(1.0 + 1.05 * MAX_Z) / 1.05 - 1.0,
        )
        probes = np.concatenate([[0.0], np.logspace(-3, log(MAX_Z) / log(10.0), 2000)])
        rel = np.array(
            [abs(self.spline(z) - retired(z)) / abs(retired(z)) for z in probes]
        )
        print(
            f"[T(z) buffer] relative change over z in [0, {MAX_Z:g}]: "
            f"median {np.median(rel):.3e}, max {rel.max():.3e}; "
            f"the spline's own interpolation floor is ~{INTERPOLATION_FLOOR:.1e}"
        )
        self.assertLess(rel.max(), INTERPOLATION_FLOOR)

    def test_the_pure_radiation_stub_still_reproduces_T_CMB_times_one_plus_z(self):
        """
        With a constant-g_* equation of state the exact answer is T = T_CMB (1+z), which is the
        independent check that the tabulation itself is still right after the range moved.
        """
        T0 = self.spline(0.0)
        worst = 0.0
        for z in (0.0, 1.0, 10.0, 1.0e3, MAX_Z):
            worst = max(worst, abs(self.spline(z) / (T0 * (1.0 + z)) - 1.0))
        print(f"[T(z) buffer] worst departure from T_CMB (1+z): {worst:.3e}")
        self.assertLess(worst, INTERPOLATION_FLOOR)


if __name__ == "__main__":
    unittest.main()
