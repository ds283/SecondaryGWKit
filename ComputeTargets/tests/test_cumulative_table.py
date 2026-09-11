"""
Tests for ``ComputeTargets/cumulative_table.py`` against closed forms (prompts/GkTk-remedial,
prompt 03 §6). No cosmology, no Ray, no datastore.

The integrand is ``f(z) = (1+z)^-2``, whose primitive is ``T(z) = 1/(1+z) - 1/(1+z_top)``. Every
reference below is formed in the factored form ``(z_a - z_b)/((1+z_a)(1+z_b))`` rather than as a
difference of two reciprocals, so that the reference itself carries a few ulps and not the
cancellation error of the naive difference (the same reason the table exists).

Measured when this module was written, on a 100/decade grid from 1e12 to 0.1 (1,301 nodes):
nodes 4.9e-16 relative, adjacent-node ``delta`` 6.2e-16, 37 %-fraction ``delta`` 5.1e-16,
off-grid ``value`` 4.5e-16; single-double ``delta`` on the adjacent pair near z = 1 is 5x worse
than double-double (1.5e-15 against 3.0e-16), and 1.0e-6 once a 1e8 offset is attached
(the review §13.3 regime, in which the primitive dwarfs the increment).
"""

import unittest
from math import log1p, expm1

import numpy as np

from ComputeTargets.cumulative_table import CumulativeTable, two_sum, quick_two_sum

Z_TOP = 1.0e12
Z_END = 0.1
SAMPLES_PER_LOG10Z = 100
ORDER = 4

# prompt 03 §6 test 1 thresholds
NODE_REL_TOL = 2.0e-15
ADJACENT_DELTA_REL_TOL = 1.0e-14
FRACTION_DELTA_REL_TOL = 1.0e-14
OFFGRID_VALUE_REL_TOL = 1.0e-15
FRACTION = 0.37


def _grid() -> np.ndarray:
    """100 samples per decade of z, descending, like the production grid."""
    num = int(round(SAMPLES_PER_LOG10Z * (np.log10(Z_TOP) - np.log10(Z_END)))) + 1
    return np.logspace(np.log10(Z_TOP), np.log10(Z_END), num=num)


def _f(z: float) -> float:
    return 1.0 / ((1.0 + z) * (1.0 + z))


def _increment(z_hi: float, z_lo: float) -> float:
    """``int_{z_lo}^{z_hi} (1+z)^-2 dz``, factored so the reference carries a few ulps only."""
    return (z_hi - z_lo) / ((1.0 + z_hi) * (1.0 + z_lo))


def _fraction_point(z_hi: float, z_lo: float, fraction: float = FRACTION) -> float:
    """The point ``fraction`` of the way from ``z_hi`` to ``z_lo`` in ``u = log(1+z)``."""
    u_hi = log1p(z_hi)
    u_lo = log1p(z_lo)
    return expm1(u_hi + fraction * (u_lo - u_hi))


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want)


class _Counting:
    def __init__(self, f):
        self._f = f
        self.calls = 0

    def __call__(self, z):
        self.calls += 1
        return self._f(z)


class TestCumulativeTableClosedForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = _grid()
        cls.f = _Counting(_f)
        cls.table = CumulativeTable(cls.z, cls.f, ORDER, label="test")
        cls.build_calls = cls.f.calls

    def test_grid_is_as_described(self):
        self.assertEqual(len(self.table), len(self.z))
        self.assertEqual(self.z[0], Z_TOP)
        self.assertAlmostEqual(self.z[-1], Z_END, places=12)
        self.assertEqual(self.table.order, ORDER)
        # order integrand calls per interval, nothing else at build time
        self.assertEqual(self.build_calls, ORDER * (len(self.z) - 1))
        self.assertEqual(self.table.evaluations, self.build_calls)

    def test_nodes_against_closed_form(self):
        """Prompt 03 §6 test 1: nodes to <= 2e-15 relative."""
        z = self.z
        worst, worst_z = 0.0, None
        for zj in z[1:]:
            err = _rel(self.table.value(zj), _increment(z[0], zj))
            if err > worst:
                worst, worst_z = err, zj
        print(
            f"\n[cumulative_table] nodes: max rel err {worst:.3e} at z = {worst_z:.6g}"
        )
        self.assertLessEqual(worst, NODE_REL_TOL)
        self.assertEqual(self.table.value(z[0]), 0.0)

    def test_delta_between_adjacent_nodes(self):
        """Prompt 03 §6 test 1: adjacent-node delta to <= 1e-14 relative to the increment."""
        z = self.z
        worst, worst_z = 0.0, None
        for i in range(len(z) - 1):
            want = _increment(z[i], z[i + 1])
            err = _rel(self.table.delta(z[i], z[i + 1]), want)
            if err > worst:
                worst, worst_z = err, z[i]
        print(
            f"[cumulative_table] adjacent-node delta: max rel err {worst:.3e} at z = {worst_z:.6g}"
        )
        self.assertLessEqual(worst, ADJACENT_DELTA_REL_TOL)

    def test_delta_over_a_fraction_of_an_interval(self):
        """Prompt 03 §6 test 1: delta over 37 % of an interval (one off-grid endpoint)."""
        z = self.z
        worst, worst_z = 0.0, None
        for i in range(0, len(z) - 1, 3):
            z_frac = _fraction_point(z[i], z[i + 1])
            want = _increment(z[i], z_frac)
            err = _rel(self.table.delta(z[i], z_frac), want)
            if err > worst:
                worst, worst_z = err, z[i]
        print(
            f"[cumulative_table] 37%-fraction delta: max rel err {worst:.3e} at z = {worst_z:.6g}"
        )
        self.assertLessEqual(worst, FRACTION_DELTA_REL_TOL)

    def test_value_off_grid(self):
        """Prompt 03 §6 test 1: pointwise value off-grid to <= 1e-15 relative."""
        z = self.z
        worst, worst_z = 0.0, None
        for i in range(1, len(z) - 1, 3):
            for fraction in (0.11, FRACTION, 0.5, 0.83):
                z_off = _fraction_point(z[i], z[i + 1], fraction)
                err = _rel(self.table.value(z_off), _increment(z[0], z_off))
                if err > worst:
                    worst, worst_z = err, z_off
        print(
            f"[cumulative_table] off-grid value: max rel err {worst:.3e} at z = {worst_z:.6g}"
        )
        self.assertLessEqual(worst, OFFGRID_VALUE_REL_TOL)

    def test_single_double_floor_is_demonstrated(self):
        """
        Prompt 03 §6 test 2: the same delta from a single-double table (lo zeroed) is worse, and
        the test shows by how much rather than asserting the floor away.

        With f = (1+z)^-2 the primitive near z = 1 is T ~ 0.5 and an adjacent-node increment is
        ~0.0115, so the single-double floor is only ~ulp(T)/increment ~ 1e-14: the prompt's
        ">~ 1e-6" is not reachable on this integrand with the table anchored at zero (measured
        1.5e-15 single-double against 3.0e-16 double-double). The 1e-6 regime is the review
        §13.3 one -- a primitive that dwarfs the increment -- and is demonstrated here by
        attaching a 1e8 offset with ``shifted``: the double-double delta is unchanged, the
        single-double one is ~1e-6.
        """
        z = self.z
        j = int(np.argmin(np.abs(np.log10(z))))  # nearest node to z = 1
        want = _increment(z[j], z[j + 1])

        single = CumulativeTable(
            z, _f, ORDER, hi=self.table.hi, lo=np.zeros_like(self.table.lo)
        )
        err_dd = _rel(self.table.delta(z[j], z[j + 1]), want)
        err_sd = _rel(single.delta(z[j], z[j + 1]), want)

        shifted = self.table.shifted(1.0e8)
        shifted_single = CumulativeTable(
            z, _f, ORDER, hi=shifted.hi, lo=np.zeros_like(shifted.lo)
        )
        err_dd_shift = _rel(shifted.delta(z[j], z[j + 1]), want)
        err_sd_shift = _rel(shifted_single.delta(z[j], z[j + 1]), want)

        print(
            f"[cumulative_table] adjacent-node delta near z = {z[j]:.4g} (T = {self.table.value(z[j]):.6g}):"
            f" double-double {err_dd:.3e}, single-double {err_sd:.3e};"
            f" with a 1e8 offset: double-double {err_dd_shift:.3e}, single-double {err_sd_shift:.3e}"
        )
        self.assertLessEqual(err_dd, ADJACENT_DELTA_REL_TOL)
        self.assertGreater(err_sd, 3.0 * max(err_dd, 1.0e-16))
        self.assertLessEqual(err_dd_shift, ADJACENT_DELTA_REL_TOL)
        self.assertGreater(err_sd_shift, 1.0e-7)

    def test_exact_node_lookup(self):
        """Prompt 03 §6 test 3."""
        z = self.z
        for k, zk in enumerate(z):
            self.assertEqual(self.table.node_index(zk), k)
        self.assertIsNone(self.table.node_index(_fraction_point(z[10], z[11])))
        self.assertIsNone(self.table.node_index(z[10] * (1.0 + 1e-15)))

    def test_delta_of_a_point_with_itself_is_exactly_zero(self):
        z = self.z
        for zz in (z[0], z[7], z[-1], _fraction_point(z[40], z[41])):
            self.assertEqual(self.table.delta(zz, zz), 0.0)

    def test_delta_is_antisymmetric_to_the_bit(self):
        z = self.z
        z_off = _fraction_point(z[300], z[301], 0.6)
        pairs = [
            (z[0], z[-1]),
            (z[3], z[900]),
            (z[700], z[701]),
            (z_off, z[900]),
            (z_off, _fraction_point(z[1200], z[1201], 0.2)),
        ]
        for a, b in pairs:
            self.assertEqual(self.table.delta(a, b), -self.table.delta(b, a))

    def test_on_grid_delta_makes_no_integrand_calls(self):
        z = self.z
        before = self.f.calls
        self.table.delta(z[3], z[900])
        self.table.delta(z[-1], z[0])
        self.table.value(z[500])
        self.assertEqual(self.f.calls, before)

        # and an off-grid endpoint costs exactly `order` calls
        self.table.delta(z[3], _fraction_point(z[100], z[101]))
        self.assertEqual(self.f.calls, before + ORDER)

    def test_reconstruction_is_bit_for_bit(self):
        """Prompt 03 §6 test 4."""
        z = self.z
        rebuilt = CumulativeTable(z, _f, ORDER, hi=self.table.hi, lo=self.table.lo)
        self.assertEqual(rebuilt.evaluations, 0)
        points = [z[0], z[1], z[500], z[-1], _fraction_point(z[20], z[21], 0.3)]
        for a in points:
            self.assertEqual(rebuilt.value(a), self.table.value(a))
            for b in points:
                self.assertEqual(rebuilt.delta(a, b), self.table.delta(a, b))

    def test_shifted_adds_the_offset_in_double_double(self):
        z = self.z
        offset = 2.5e-11
        shifted = self.table.shifted(offset)
        for zz in (z[0], z[1], z[800], z[-1]):
            self.assertLessEqual(
                abs(shifted.value(zz) - (self.table.value(zz) + offset)),
                4.0 * np.finfo(float).eps * (self.table.value(zz) + offset),
            )
        # the low limb is a genuine second limb: below an ulp of the high one
        self.assertTrue(np.all(np.abs(shifted.lo) <= np.spacing(np.abs(shifted.hi))))
        # and deltas are unchanged to the last few bits
        for i in (0, 400, 1200):
            want = self.table.delta(z[i], z[i + 1])
            self.assertLessEqual(
                abs(shifted.delta(z[i], z[i + 1]) - want),
                4.0 * np.finfo(float).eps * want,
            )

    def test_error_free_transformations(self):
        a, b = 1.0, 1e-17
        s, e = two_sum(a, b)
        self.assertEqual(s, 1.0)
        self.assertEqual(e, 1e-17)
        s, e = quick_two_sum(a, b)
        self.assertEqual(s, 1.0)
        self.assertEqual(e, 1e-17)

    def test_out_of_range_is_refused_beyond_one_interval(self):
        z = self.z
        # within one interval beyond either end: a partial from the end node, allowed
        self.table.value(z[-1] * 0.995)
        self.table.value(z[0] * 1.01)
        with self.assertRaises(RuntimeError):
            self.table.value(z[0] * 1.2)
        with self.assertRaises(RuntimeError):
            self.table.value(-0.5)

    def test_constructor_validation(self):
        z = self.z
        with self.assertRaises(ValueError):
            CumulativeTable(z[::-1], _f, ORDER)
        with self.assertRaises(ValueError):
            CumulativeTable(z, _f, ORDER, hi=self.table.hi)
        with self.assertRaises(ValueError):
            CumulativeTable(z, None, ORDER)
        rebuilt = CumulativeTable(z, None, ORDER, hi=self.table.hi, lo=self.table.lo)
        self.assertEqual(rebuilt.value(z[10]), self.table.value(z[10]))
        with self.assertRaises(RuntimeError):
            rebuilt.value(_fraction_point(z[10], z[11]))


class TestCumulativeTableBreakPoints(unittest.TestCase):
    """
    The build scheme prompt 02 mandates: every panel is split at the declared break points of
    the integrand. A piecewise integrand with a jump at a known ``u`` shows the fixed-order rule
    failing without the split and at the floor with it.
    """

    @classmethod
    def setUpClass(cls):
        cls.z = _grid()
        z = cls.z
        cls.i = 650
        u_hi = log1p(z[cls.i])
        u_lo = log1p(z[cls.i + 1])
        cls.u_break = u_hi + 0.4 * (u_lo - u_hi)
        cls.z_break = expm1(cls.u_break)

        def f_jump(zz: float) -> float:
            return _f(zz) * (2.0 if log1p(zz) >= cls.u_break else 1.0)

        # staticmethod so that the closure is not re-bound as a method on attribute access
        cls.f_jump = staticmethod(f_jump)

    def _exact(self, z_hi: float, z_lo: float) -> float:
        zb = self.z_break
        if z_lo >= zb:
            return 2.0 * _increment(z_hi, z_lo)
        if z_hi <= zb:
            return _increment(z_hi, z_lo)
        return 2.0 * _increment(z_hi, zb) + _increment(zb, z_lo)

    def test_split_panel_is_at_the_floor_and_unsplit_is_not(self):
        z = self.z
        i = self.i
        plain = CumulativeTable(z, self.f_jump, ORDER)
        split = CumulativeTable(z, self.f_jump, ORDER, break_points=[self.u_break])
        want = self._exact(z[i], z[i + 1])
        err_plain = _rel(plain.delta(z[i], z[i + 1]), want)
        err_split = _rel(split.delta(z[i], z[i + 1]), want)
        print(
            f"\n[cumulative_table] jump inside interval {i}: unsplit rel err {err_plain:.3e}, "
            f"split rel err {err_split:.3e}; build calls {plain.evaluations} -> {split.evaluations}"
        )
        self.assertGreater(err_plain, 1.0e-3)
        self.assertLessEqual(err_split, 1.0e-13)
        # one extra panel of `order` calls, nothing else
        self.assertEqual(split.evaluations, plain.evaluations + ORDER)

        # every other node is unaffected by the declaration
        for zz in (z[i - 1], z[i + 2], z[-1]):
            self.assertLessEqual(_rel(split.value(zz), self._exact(z[0], zz)), 1.0e-13)

    def test_partial_through_a_break_is_split_too(self):
        z = self.z
        i = self.i
        split = CumulativeTable(z, self.f_jump, ORDER, break_points=[self.u_break])
        # an off-grid endpoint beyond the break, seen from the node on the other side
        z_off = _fraction_point(z[i], z[i + 1], 0.7)
        want = self._exact(z[i], z_off)
        self.assertLessEqual(_rel(split.delta(z[i], z_off), want), 1.0e-13)
        # and the pointwise value there
        self.assertLessEqual(
            _rel(split.value(z_off), self._exact(z[0], z_off)), 1.0e-13
        )

    def test_break_points_outside_every_panel_are_inert(self):
        z = self.z
        plain = CumulativeTable(z, _f, ORDER)
        inert = CumulativeTable(z, _f, ORDER, break_points=[log1p(z[0]) + 1.0, -5.0])
        self.assertEqual(inert.evaluations, plain.evaluations)
        self.assertTrue(np.array_equal(inert.hi, plain.hi))
        self.assertTrue(np.array_equal(inert.lo, plain.lo))


if __name__ == "__main__":
    unittest.main()
