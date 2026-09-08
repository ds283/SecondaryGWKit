"""
Tests for ComputeTargets/QuadSource.py.

These cover the change made by prompt 06 of the source-remediation campaign: `QuadSource` is
now the *smooth part* of the source term, defined on exactly the region where both Tq and Tr
are still described by their numeric representation (plus the exactly-known super-horizon
region above it), rather than trying to be f(z' | q, r) over the whole source grid.

Everything here runs offline: no Ray, no datastore. The transfer functions are the exact
constant-w analytic solutions (`ComputeTargets/analytic_Tk.py`) on an exact constant-w
background, following the pattern of `docs/spec-code-audit/scripts/QS_02_deriv_and_f.py` and
`QS_03_spline_error.py`.
"""

import unittest
from math import sqrt, exp
from typing import List, Optional
from unittest import mock

import numpy as np
from scipy.interpolate import make_interp_spline

from ComputeTargets.QuadSource import (
    QuadSource,
    compute_quad_source,
    numeric_crossover_z,
    source_function,
)
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from CosmologyConcepts import redshift, redshift_array

# undo the @ray.remote wrapper so that the task body can be driven directly
compute_quad_source_fn = compute_quad_source._function

H0 = 1.0


# ---------------------------------------------------------------------------------------
# exact constant-w background, a_0 absorbed (H = H0 (1+z)^{3(1+w)/2}, tau = a_0 eta)


def Hubble(z: float, w: float) -> float:
    return H0 * (1.0 + z) ** (1.5 * (1.0 + w))


def tau(z: float, w: float) -> float:
    # solves d(a0 eta)/dz = -1/H with a0 eta -> 0 as z -> infinity, for w > -1/3
    p = 0.5 * (1.0 + 3.0 * w)
    return 2.0 / ((1.0 + 3.0 * w) * H0) * (1.0 + z) ** (-p)


def f_spec(Tq, Tr, dTq, dTr, z, w):
    """spec 03 R22 / section 0.3, transcribed literally and independently of the code.
    Copied from docs/spec-code-audit/scripts/QS_02_deriv_and_f.py."""
    return Tq * Tr + 2.0 / (3.0 * (1.0 + w)) * (Tq - (1.0 + z) * dTq) * (
        Tr - (1.0 + z) * dTr
    )


def f_exact(z: float, q: float, r: float, w: float) -> float:
    t, h = tau(z, w), Hubble(z, w)
    return source_function(
        compute_analytic_T(q, w, t),
        compute_analytic_T(r, w, t),
        compute_analytic_Tprime(q, w, t, h),
        compute_analytic_Tprime(r, w, t, h),
        z,
        w,
    )["source"]


# ---------------------------------------------------------------------------------------
# stand-ins, shaped like the real objects the compute task touches


class FakeModelFunctions:
    def __init__(self, w: float):
        self._w = w

    def wBackground(self, z: float) -> float:
        return self._w


class FakeModel:
    def __init__(self, w: float):
        self.functions = FakeModelFunctions(w)


class FakeModelProxy:
    def __init__(self, w: float):
        self._model = FakeModel(w)
        self.store_id = 1

    def get(self) -> FakeModel:
        return self._model


class FakeTkValue:
    """duck-types TkNumericValue"""

    def __init__(self, z: redshift, k: float, w: float):
        self.z = z
        t, h = tau(z.z, w), Hubble(z.z, w)
        self.T = compute_analytic_T(k, w, t)
        self.Tprime = compute_analytic_Tprime(k, w, t, h)
        self.analytic_T_rad = compute_analytic_T(k, 1.0 / 3.0, t)
        self.analytic_Tprime_rad = compute_analytic_Tprime(k, 1.0 / 3.0, t, h)
        self.analytic_T_w = self.T
        self.analytic_Tprime_w = self.Tprime


class FakeTk:
    """
    Duck-types TkNumericIntegration as `compute_quad_source` uses it. The z_sample /
    __getitem__ pair is deliberately retained, even though the new code reads only `values`,
    so that the *pre-commit* implementation can be driven with the same fixture.
    """

    def __init__(
        self,
        z_grid: List[redshift],
        k: float,
        w: float,
        z_exit: float,
        stop_deltaz_subh: Optional[float],
    ):
        self.values = [FakeTkValue(z, k, w) for z in z_grid]
        self.z_sample = redshift_array(z_grid)
        self.z_exit = z_exit
        self.stop_deltaz_subh = stop_deltaz_subh
        self.available = True
        self.store_id = 17

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


class FakeWavenumberExit:
    def __init__(self, k: float):
        self.k = k
        self.store_id = 3


def source_grid(
    z_init: float, z_end: float, samples_per_log10z: int = 100
) -> List[redshift]:
    """
    Mirror CosmologyConcepts.wavenumber.populate_z_sample: logspace in z, with
    `samples_per_log10z` samples per decade (DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100 in
    main.py). Returned in descending order of z, as redshift_array holds them.
    """
    n = int(
        round(
            samples_per_log10z * (np.log10(z_init) - np.log10(z_end)) + 0.5,
            0,
        )
    )
    zs = np.logspace(np.log10(z_init), np.log10(z_end), num=n)
    return [redshift(store_id=1000 + i, z=float(z)) for i, z in enumerate(zs)]


class TestQuadSourceKernel(unittest.TestCase):
    """The source kernel itself is untouched by prompt 06 (audit: 'kernel exact')."""

    def test_source_function_reproduces_spec_03_R22(self):
        worst = 0.0
        for w in (1.0 / 3.0, 0.1, 0.2, 0.5):
            for q, r in ((1.0, 1.0), (1.0, 30.0), (30.0, 30.0), (300.0, 700.0)):
                for z in (1.0e5, 1.0e4, 1.0e3, 1.0e2, 10.0, 1.0):
                    t, h = tau(z, w), Hubble(z, w)
                    Tq = compute_analytic_T(q, w, t)
                    Tr = compute_analytic_T(r, w, t)
                    dTq = compute_analytic_Tprime(q, w, t, h)
                    dTr = compute_analytic_Tprime(r, w, t, h)

                    spec = f_spec(Tq, Tr, dTq, dTr, z, w)
                    out = source_function(Tq, Tr, dTq, dTr, z, w)

                    self.assertAlmostEqual(
                        out["source"],
                        out["undiff"] + out["diff"],
                        delta=1e-14 * abs(spec),
                    )
                    worst = max(
                        worst, abs(spec - out["source"]) / max(abs(spec), 1e-300)
                    )

        print(
            f"\nsource_function vs independent spec 03 R22: worst relative = {worst:.3e}"
        )
        self.assertLess(worst, 1.0e-14)

    def test_source_function_is_symmetric(self):
        w, z = 1.0 / 3.0, 1.0e3
        t, h = tau(z, w), Hubble(z, w)
        Tq, Tr = compute_analytic_T(11.0, w, t), compute_analytic_T(97.0, w, t)
        dTq = compute_analytic_Tprime(11.0, w, t, h)
        dTr = compute_analytic_Tprime(97.0, w, t, h)

        self.assertEqual(
            source_function(Tq, Tr, dTq, dTr, z, w)["source"],
            source_function(Tr, Tq, dTr, dTq, z, w)["source"],
        )


class TestQuadSourceRegion(unittest.TestCase):
    """
    A3 regression: the source grid is walked only over the both-numeric region, and the Tk
    grids -- truncated at BOTH ends, exactly as main.py:505-507 with mode='stop' builds them
    -- no longer run out underneath it.
    """

    W = 1.0 / 3.0

    def setUp(self):
        # 3 decades at the production grid density: 301 points from z = 1e5 down to z = 1e2
        self.grid = source_grid(1.0e5, 1.0e2, samples_per_log10z=100)
        self.n = len(self.grid)

        # Tr carries the larger wavenumber (main.py schedules q.k <= r.k), so it both starts
        # its numeric integration earlier (higher z) and hands over to WKB earlier
        self.i_start_r, self.i_stop_r = 20, 180
        self.i_start_q, self.i_stop_q = 60, 240

        self.q_k, self.r_k = 3.0, 11.0

        self.Tq = self.build_Tk(self.i_start_q, self.i_stop_q, self.q_k)
        self.Tr = self.build_Tk(self.i_start_r, self.i_stop_r, self.r_k)

        self.model_proxy = FakeModelProxy(self.W)
        self.z_sample = redshift_array(self.grid)

    def build_Tk(self, i_start: int, i_stop: int, k: float) -> FakeTk:
        # choose z_exit / stop_deltaz_subh so that the hand-over redshift
        # z_exit - stop_deltaz_subh lands exactly on grid[i_stop]
        z_stop = self.grid[i_stop].z
        return FakeTk(
            self.grid[i_start : i_stop + 1],
            k,
            self.W,
            z_exit=3.0 * z_stop,
            stop_deltaz_subh=2.0 * z_stop,
        )

    def test_crossover_helper(self):
        self.assertAlmostEqual(
            numeric_crossover_z(self.Tq), self.grid[self.i_stop_q].z, places=10
        )
        self.assertAlmostEqual(
            numeric_crossover_z(self.Tr), self.grid[self.i_stop_r].z, places=10
        )
        # a Tk with no "stop" hand-over has no crossover redshift
        no_stop = FakeTk(
            self.grid[10:50], 1.0, self.W, z_exit=1.0, stop_deltaz_subh=None
        )
        self.assertIsNone(numeric_crossover_z(no_stop))

    def test_A3_regression_region_and_defaults(self):
        out = compute_quad_source_fn(self.model_proxy, self.z_sample, self.Tq, self.Tr)

        # the value list ends at max(z^X_q, z^X_r) = z^X_r
        self.assertEqual(len(out["source"]), self.i_stop_r + 1)
        self.assertEqual(len(out["z_store_ids"]), self.i_stop_r + 1)
        self.assertEqual(out["z_store_ids"][0], self.grid[0].store_id)
        self.assertEqual(out["z_store_ids"][-1], self.grid[self.i_stop_r].store_id)
        self.assertAlmostEqual(
            out["crossover_z_r"], self.grid[self.i_stop_r].z, places=10
        )
        self.assertAlmostEqual(
            out["crossover_z_q"], self.grid[self.i_stop_q].z, places=10
        )

        # the leading samples, above the start of *both* numeric grids, use the exact
        # super-horizon default T = 1, T' = 0
        for i in range(self.i_start_r):
            z = self.grid[i].z
            expected = source_function(1.0, 1.0, 0.0, 0.0, z, self.W)
            self.assertAlmostEqual(
                out["source"][i],
                expected["source"],
                delta=1e-14 * abs(expected["source"]),
            )

        # between the two grid starts only q is still on its default
        i = (self.i_start_r + self.i_start_q) // 2
        z = self.grid[i].z
        t, h = tau(z, self.W), Hubble(z, self.W)
        expected = source_function(
            1.0,
            compute_analytic_T(self.r_k, self.W, t),
            0.0,
            compute_analytic_Tprime(self.r_k, self.W, t, h),
            z,
            self.W,
        )["source"]
        self.assertAlmostEqual(out["source"][i], expected, delta=1e-13 * abs(expected))

        # and inside both grids the stored samples are used
        i = self.i_stop_r
        z = self.grid[i].z
        expected = f_exact(z, self.q_k, self.r_k, self.W)
        self.assertAlmostEqual(out["source"][i], expected, delta=1e-13 * abs(expected))

    def test_A3_regression_pre_commit_code_would_have_raised(self):
        """
        The pre-commit implementation walked the full source grid against Tk.z_sample and
        indexed past its end. Reproduce that failure mode on this fixture, so that the test
        records *why* the loop had to change (audit A3 / QS-6).
        """
        Tq_zsample, Tr_zsample = self.Tq.z_sample, self.Tr.z_sample
        q_idx = r_idx = 0
        with self.assertRaises(IndexError):
            for i in range(len(self.z_sample)):
                z = self.z_sample[i]
                Tq_z = Tq_zsample[q_idx]  # QuadSource.py:87 before this commit
                Tr_z = Tr_zsample[r_idx]
                if Tq_z.store_id == z.store_id:
                    q_idx += 1
                if Tr_z.store_id == z.store_id:
                    r_idx += 1

    def test_interior_gap_raises_naming_the_factor(self):
        # punch a hole in the middle of Tq's coverage, inside the both-numeric region
        hole = (self.i_start_q + self.i_stop_r) // 2
        Tq = self.build_Tk(self.i_start_q, self.i_stop_q, self.q_k)
        Tq.values = [v for v in Tq.values if v.z.store_id != self.grid[hole].store_id]

        with self.assertRaises(RuntimeError) as ctx:
            compute_quad_source_fn(self.model_proxy, self.z_sample, Tq, self.Tr)

        message = str(ctx.exception)
        self.assertIn("Tq", message)
        self.assertIn(str(self.grid[hole].store_id), message)

        # ... and the same hole in Tr names Tr
        Tr = self.build_Tk(self.i_start_r, self.i_stop_r, self.r_k)
        Tr.values = [v for v in Tr.values if v.z.store_id != self.grid[hole].store_id]
        with self.assertRaises(RuntimeError) as ctx:
            compute_quad_source_fn(self.model_proxy, self.z_sample, self.Tq, Tr)
        self.assertIn("Tr", str(ctx.exception))

    def test_region_is_clamped_to_the_sampled_coverage(self):
        """
        The hand-over redshift is found by a root search whose window bottom is where the
        integration terminates, so it can fall up to one grid step *below* the last stored
        sample. The region must then stop at the last stored sample, not at the hand-over.
        """
        # move Tr's nominal hand-over two grid steps below its last stored sample
        z_below = self.grid[self.i_stop_r + 2].z
        Tr = FakeTk(
            self.grid[self.i_start_r : self.i_stop_r + 1],
            self.r_k,
            self.W,
            z_exit=3.0 * z_below,
            stop_deltaz_subh=2.0 * z_below,
        )
        out = compute_quad_source_fn(self.model_proxy, self.z_sample, self.Tq, Tr)
        self.assertEqual(len(out["source"]), self.i_stop_r + 1)

    def test_store_truncates_z_sample_and_the_spline_range(self):
        """
        Drive QuadSource.store() with a stubbed Ray, and check that z_sample, values,
        numeric_region and the dense-output spline all describe the same range.
        """
        payload = compute_quad_source_fn(
            self.model_proxy, self.z_sample, self.Tq, self.Tr
        )

        obj = QuadSource(
            payload=None,
            model=self.model_proxy,
            z_sample=self.z_sample,
            q=FakeWavenumberExit(self.q_k),
            r=FakeWavenumberExit(self.r_k),
        )
        obj._compute_ref = object()
        obj._Tq_serial, obj._Tr_serial = 1, 2

        fake_ray = mock.MagicMock()
        fake_ray.wait.return_value = ([obj._compute_ref], [])
        fake_ray.get.return_value = payload
        with mock.patch("ComputeTargets.QuadSource.ray", fake_ray):
            self.assertTrue(obj.store())

        z_min = self.grid[self.i_stop_r].z
        z_max = self.grid[0].z

        self.assertEqual(len(obj.values), self.i_stop_r + 1)
        self.assertEqual(len(obj.z_sample), self.i_stop_r + 1)
        self.assertAlmostEqual(obj.z_sample.min.z, z_min, places=10)
        self.assertAlmostEqual(obj.z_sample.max.z, z_max, places=10)
        self.assertAlmostEqual(obj.numeric_region[0], z_max, places=10)
        self.assertAlmostEqual(obj.numeric_region[1], z_min, places=10)
        self.assertAlmostEqual(obj.crossover_z_r, z_min, places=10)
        self.assertAlmostEqual(obj.crossover_z_q, self.grid[self.i_stop_q].z, places=10)

        # values are attached to the right redshifts, in descending order
        self.assertEqual(obj.values[0].z.store_id, self.grid[0].store_id)
        self.assertEqual(obj.values[-1].z.store_id, self.grid[self.i_stop_r].store_id)

        functions = obj.functions
        self.assertAlmostEqual(functions.numeric_region[0], z_max, places=10)
        self.assertAlmostEqual(functions.numeric_region[1], z_min, places=10)

        # the spline reproduces its nodes, and refuses to be evaluated well below the region
        mid = self.i_stop_r // 2
        self.assertAlmostEqual(
            functions.source(self.grid[mid].z),
            obj.values[mid].source,
            delta=1e-12 * abs(obj.values[mid].source),
        )
        with self.assertRaises(RuntimeError):
            functions.source(0.5 * z_min)

    def test_datastore_round_trip_shape(self):
        """
        Mimic what Datastore/SQL/ObjectFactories/QuadSource.py does on read: it rebuilds
        z_sample from the stored value rows themselves (`imported_z_sample`, factory line
        261) and compares its length against the `z_samples` column, which store() writes as
        `len(obj.values)` (factory line 313). A truncated value list therefore reads back
        with a consistent z_sample and `available == True`; the hand-over redshifts are not
        persisted and come back as None.
        """
        payload = compute_quad_source_fn(
            self.model_proxy, self.z_sample, self.Tq, self.Tr
        )
        region_z = [z for z in self.grid if z.store_id in set(payload["z_store_ids"])]
        values = [
            _StoredQuadSourceValue(z, s) for z, s in zip(region_z, payload["source"])
        ]

        obj = QuadSource(
            payload={
                "store_id": 99,
                "compute_time": 1.0,
                "values": values,
                "Tq_serial": 1,
                "Tr_serial": 2,
            },
            model=self.model_proxy,
            z_sample=redshift_array(region_z),
            q=FakeWavenumberExit(self.q_k),
            r=FakeWavenumberExit(self.r_k),
        )

        self.assertTrue(obj.available)
        self.assertEqual(len(obj.z_sample), len(obj.values))
        self.assertEqual(len(obj.values), len(payload["z_store_ids"]))
        self.assertAlmostEqual(
            obj.numeric_region[1], self.grid[self.i_stop_r].z, places=10
        )
        self.assertIsNone(obj.crossover_z_q)
        self.assertIsNone(obj.crossover_z_r)
        self.assertAlmostEqual(
            obj.functions.numeric_region[1], self.grid[self.i_stop_r].z, places=10
        )


class _StoredQuadSourceValue:
    """the minimal shape _create_functions needs from a value row"""

    def __init__(self, z: redshift, source: float):
        self.z = z
        self.source = source


class TestQuadSourceSplineResidual(unittest.TestCase):
    """
    Section 3.4 of the prompt: report (do not assert) the spline residual of
    `_create_functions` inside the both-numeric region, in the worst case allowed by
    main.py's hand-over window. This bounds the accuracy of the all-smooth region of the
    source integral.

    Radiation, a_0 absorbed: H = H0(1+z)^2, so k/(aH) = k/(1+z) and x = k c_s a_0 eta
    = k/(sqrt(3)(1+z)). Hence at N e-folds inside the horizon x = e^N/sqrt(3), independently
    of k: x = 11.6 at z_exit_subh_e3 (the top of main.py's search window, where the phase
    minimum is normally found) and x = 232.9 at z_exit_subh_e6 (its bottom).
    """

    W = 1.0 / 3.0
    CS = sqrt(1.0 / 3.0)

    def residual(self, efolds_subh: float, window_factor: float = 1.0):
        # q = r = k, the deep sub-horizon diagonal pair; k fixed so that the region ends at
        # window_factor * z_exit_subh_{efolds}
        k = 1.0e4
        one_plus_z_exit = k  # k/(aH) = k/(1+z) = 1 at horizon crossing
        z_end = window_factor * (one_plus_z_exit / exp(efolds_subh) - 1.0)
        z_init = one_plus_z_exit * exp(5.0) - 1.0  # 5 e-folds outside the horizon

        zs = np.array([z.z for z in source_grid(z_init, z_end, 100)])
        y = np.array([f_exact(z, k, k, self.W) for z in zs])

        order = np.argsort(np.log(1.0 + zs))
        xs = np.log(1.0 + zs[order])
        ys = y[order]
        spline = make_interp_spline(xs, ys)  # exactly as QuadSource._create_functions

        worst_env, worst_z = 0.0, None
        for i in range(len(xs) - 1):
            xd = xs[i] + (xs[i + 1] - xs[i]) * np.linspace(0.0, 1.0, 41)[1:-1]
            fe = np.array([f_exact(exp(x) - 1.0, k, k, self.W) for x in xd])
            fs = spline(xd)
            lo, hi = max(0, i - 2), min(len(xs), i + 3)
            env = max(np.max(np.abs(ys[lo:hi])), np.max(np.abs(fe)))
            err = np.max(np.abs(fs - fe)) / max(env, 1e-300)
            if err > worst_env:
                worst_env = err
                worst_z = float(np.exp(xd[int(np.argmax(np.abs(fs - fe)))]) - 1.0)

        x_end = k * self.CS / (1.0 + z_end)
        return worst_env, worst_z, x_end

    def test_report_spline_residual_in_the_both_numeric_region(self):
        print(
            "\nspline residual of _create_functions inside the both-numeric region"
            "\n(q = r = k, exact radiation source, 100 samples per log10 z,"
            "\n region from 5 e-folds super-horizon down to the stated hand-over)"
        )
        print(
            f"{'hand-over':>34} {'x at hand-over':>15} {'cycles of f':>12} "
            f"{'max err / envelope':>19} {'at z':>12}"
        )
        for label, efolds, factor in (
            ("z_exit_subh_e3 (typical)", 3.0, 1.0),
            ("z_exit_subh_e6", 6.0, 1.0),
            ("0.85 * z_exit_subh_e6 (latest)", 6.0, 0.85),
        ):
            err, z_at, x_end = self.residual(efolds, factor)
            print(
                f"{label:>34} {x_end:15.4g} {x_end/np.pi:12.4g} "
                f"{err:19.3e} {z_at:12.4g}"
            )
            # reported, not asserted; only sanity-check that nothing is NaN
            self.assertTrue(np.isfinite(err))


if __name__ == "__main__":
    unittest.main()
