"""
``extract_common.sensible_log_points``, the filter every plotted ``analytic_G_w`` series goes through
(prompts/handover prompt 09 §2).

**The defect guarded.** ``extract_GkWKB_data.py`` plotted |``analytic_G_w``| beside |G| on log-log
axes. On A3 v2 that column reaches 7.57e+243 where w(z) crosses -1/3, the y-axis spans ~230
decades, matplotlib proposes a tick past the largest double, and the reader died with
``OverflowError`` (``[08-the-gkwkb-reader-fails-plotting-…]``). The helper keeps only points that
are not None, finite, non-zero, and no more than ``10**decades`` times the largest |G| on the same
plot; its signed form keeps the same points with their sign, for ``extract_Gk_data.py``'s
linear-y plot (the addendum to prompt 09 §2).

``extract_common`` imports ``ray`` but does not start it. No Ray, no datastore.
"""

import unittest

import ray

from extract_common import ANALYTIC_PLOT_DECADES, sensible_log_points

INF = float("inf")
NAN = float("nan")


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_sensible_log_points")


class TestDropped(unittest.TestCase):
    """None, nan, +inf, -inf and 0 are dropped, with and without a reference."""

    POINTS = [
        (1.0, None),
        (2.0, NAN),
        (3.0, INF),
        (4.0, -INF),
        (5.0, 0.0),
        (6.0, -0.0),
        (7.0, 2.5),
    ]

    def test_dropped_with_no_reference(self):
        # no reference, so no ceiling: only the finiteness and non-zero rule can drop inf here
        self.assertEqual(sensible_log_points(self.POINTS, []), [(7.0, 2.5)])

    def test_dropped_with_a_reference(self):
        self.assertEqual(sensible_log_points(self.POINTS, [1.0, 3.0]), [(7.0, 2.5)])

    def test_dropped_in_the_signed_form(self):
        self.assertEqual(
            sensible_log_points(self.POINTS, [], signed=True), [(7.0, 2.5)]
        )
        self.assertEqual(
            sensible_log_points(self.POINTS, [1.0, 3.0], signed=True), [(7.0, 2.5)]
        )


class TestTheCeiling(unittest.TestCase):
    """The boundary is 10**decades times the largest finite non-zero |reference|, applied to |y|."""

    def test_the_constant_is_three_decades(self):
        self.assertEqual(ANALYTIC_PLOT_DECADES, 3)

    def test_just_above_is_dropped_and_just_below_is_kept(self):
        reference = [0.5, -2.0, 1.0]  # largest modulus 2.0
        above = 2.0 * 10.0**3.0001
        below = 2.0 * 10.0**2.9999
        self.assertEqual(
            sensible_log_points([(1.0, above), (2.0, below)], reference),
            [(2.0, below)],
        )

    def test_the_reference_ignores_none_nan_inf_and_zero(self):
        reference = [None, NAN, INF, -INF, 0.0, 2.0]
        above = 2.0 * 10.0**3.0001
        below = 2.0 * 10.0**2.9999
        self.assertEqual(
            sensible_log_points([(1.0, above), (2.0, below)], reference),
            [(2.0, below)],
        )

    def test_decades_is_honoured(self):
        self.assertEqual(
            sensible_log_points([(1.0, 50.0), (2.0, 5.0)], [1.0], decades=1),
            [(2.0, 5.0)],
        )

    def test_an_empty_reference_keeps_every_finite_non_zero_point(self):
        points = [(1.0, 1.0e-300), (2.0, -3.0), (3.0, 1.0e300)]
        self.assertEqual(
            sensible_log_points(points, []),
            [(1.0, 1.0e-300), (2.0, 3.0), (3.0, 1.0e300)],
        )

    def test_an_all_none_reference_keeps_every_finite_non_zero_point(self):
        points = [(1.0, 1.0e-300), (2.0, -3.0), (3.0, 1.0e300)]
        self.assertEqual(
            sensible_log_points(points, [None, None]),
            [(1.0, 1.0e-300), (2.0, 3.0), (3.0, 1.0e300)],
        )

    def test_the_a3_v2_values(self):
        # the failing pair's |G| lay in 9.6e18-4.0e21; analytic_G_w reached 1.4e34 and 7.57e243
        reference = [9.6e18, -4.0e21]
        points = [(0.1, 1.4e34), (0.398, 2.1e57), (0.5, 7.57e243), (100.0, -3.0e21)]
        self.assertEqual(sensible_log_points(points, reference), [(100.0, 3.0e21)])


class TestModulusAndOrder(unittest.TestCase):
    def test_negative_values_are_returned_as_their_modulus(self):
        self.assertEqual(
            sensible_log_points([(1.0, -2.0), (2.0, 3.0)], [1.0]),
            [(1.0, 2.0), (2.0, 3.0)],
        )

    def test_the_order_of_the_input_is_kept(self):
        points = [(5.0, 1.0), (1.0, 2.0), (3.0, None), (4.0, -3.0), (2.0, 4.0)]
        self.assertEqual(
            sensible_log_points(points, [10.0]),
            [(5.0, 1.0), (1.0, 2.0), (4.0, 3.0), (2.0, 4.0)],
        )

    def test_x_is_passed_through_untouched(self):
        marker = object()
        self.assertIs(sensible_log_points([(marker, 1.0)], [1.0])[0][0], marker)


class TestTheSignedForm(unittest.TestCase):
    """The signed form keeps the same points, with their sign; the boundary is on the modulus."""

    def test_a_negative_value_keeps_its_sign(self):
        self.assertEqual(
            sensible_log_points([(1.0, -2.0), (2.0, 3.0)], [1.0], signed=True),
            [(1.0, -2.0), (2.0, 3.0)],
        )

    def test_the_boundary_is_applied_to_the_modulus(self):
        reference = [-2.0]
        above = 2.0 * 10.0**3.0001
        below = 2.0 * 10.0**2.9999
        points = [(1.0, -above), (2.0, -below), (3.0, above), (4.0, below)]
        self.assertEqual(
            sensible_log_points(points, reference, signed=True),
            [(2.0, -below), (4.0, below)],
        )

    def test_the_signed_form_keeps_the_points_the_log_form_keeps(self):
        points = [
            (1.0, -5.0e6),
            (2.0, None),
            (3.0, -4.0),
            (4.0, INF),
            (5.0, 7.0),
            (6.0, 0.0),
        ]
        reference = [1.0, -3.0]
        log_form = sensible_log_points(points, reference)
        signed_form = sensible_log_points(points, reference, signed=True)
        self.assertEqual([x for x, _ in log_form], [x for x, _ in signed_form])
        self.assertEqual([abs(y) for _, y in signed_form], [y for _, y in log_form])
        self.assertEqual(signed_form, [(3.0, -4.0), (5.0, 7.0)])


if __name__ == "__main__":
    unittest.main()
