"""
``compute_analytic_G`` and ``compute_analytic_Gprime`` outside their domain (prompts/handover prompt 09
§1).

**The defect guarded.** Both functions evaluate the constant-w Bessel form of spec 02 R2-R9 at each
response point with that point's background ``wBackground``. The form's derivation needs
1 + 3w > 0; as w -> -1/3 from above its Bessel order 1/2 + b, b = (1 - 3w)/(1 + 3w), diverges, and
below -1/3 the form does not describe the background. A3 v2 stores ``analytic_G_w`` up to 7.57e+243
where LambdaCDM's w(z) crosses -1/3, and ``extract_GkWKB_data.py`` died plotting it
(``[08-the-gkwkb-reader-fails-plotting-…]``). The functions now return None outside 1 + 3w > 0 and
on a value that is not finite, and are bit-identical to the old expression everywhere else.

The old expressions are written out below from ``scipy.special`` exactly as they stood before
prompt 09, in the same order of operations, so that "bit-identical" is checked against the code as it
was and not against the code as it is. No Ray, no datastore.
"""

import struct
import sys
import unittest
import warnings
from math import cos, isfinite, pi, sin, sqrt

from scipy.special import jv, yv

from ComputeTargets.analytic_Gk import (
    compute_analytic_G,
    compute_analytic_Gprime,
    constant_w_form_applies,
)

EPS = sys.float_info.epsilon

# -1/3 as the float a caller holds. 3.0 * (-1.0 / 3.0) rounds to exactly -1.0 in binary64, so
# 1 + 3w evaluates to exactly 0.0 for it (asserted below): the boundary point itself.
W_THIRD = -1.0 / 3.0

OUTSIDE_W = (
    ("w = -1/3", W_THIRD),
    ("w = -1/3 - 1e-12", W_THIRD - 1.0e-12),
    ("w = -1/2", -0.5),
    ("w = -1", -1.0),
)

# (k, tau_source, tau, H_source, H): responses after the source (tau > tau_source), over small,
# order-one and large k tau.
ARGUMENTS = (
    (1.0e-3, 1.0e-2, 3.0e-2, 4.0, 1.5),
    (1.0, 0.5, 2.0, 2.0, 0.5),
    (1.0, 3.0, 17.0, 0.7, 0.2),
    (10.0, 100.0, 1.0e3, 3.7, 0.9),
    (1.0e5, 10.0, 10.1, 3.7, 0.9),
)

# w = -1/3 + 1e-3 has b = 666.67 and Bessel order 667.17. The form is finite there only once both
# k tau and k tau_source are well above the order; these arguments are, and the test asserts it.
NEAR_BOUNDARY_ARGUMENTS = (
    (1.0, 1.0e3, 2.0e3, 2.0, 0.5),
    (1.0, 2.0e3, 5.0e3, 2.0, 0.5),
    (1.0, 1.0e3, 1.5e3, 2.0, 0.5),
)

INSIDE_W = (
    ("w = 1/3", 1.0 / 3.0, ARGUMENTS),
    ("w = 0", 0.0, ARGUMENTS),
    ("w = 0.9", 0.9, ARGUMENTS),
    ("w = -1/3 + 1e-3", W_THIRD + 1.0e-3, NEAR_BOUNDARY_ARGUMENTS),
)


def old_G(k, w, tau_source, tau, H_source):
    """``compute_analytic_G`` as it stood before prompt 09, from ``scipy.special`` alone."""
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source
    n0 = 0.5 + b
    A = -H_source * pi / 2.0
    B = sqrt(tau * tau_source)
    C = jv(n0, k_tau_source) * yv(n0, k_tau)
    D = jv(n0, k_tau) * yv(n0, k_tau_source)
    return A * B * (C - D)


def old_Gprime(k, w, tau_source, tau, H_source, H):
    """``compute_analytic_Gprime`` as it stood before prompt 09, from ``scipy.special`` alone."""
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    k_tau = k * tau
    k_tau_source = k * tau_source
    n0 = 0.5 + b
    n1 = 1.5 + b
    A = pi / 2.0 * (H_source / H)
    B = sqrt(tau_source / tau)
    C1 = k_tau * jv(n1, k_tau) - (b + 1.0) * jv(n0, k_tau)
    C2 = yv(n0, k_tau_source)
    D1 = (b + 1.0) * yv(n0, k_tau) - k_tau * yv(n1, k_tau)
    D2 = jv(n0, k_tau_source)
    return A * B * (C1 * C2 + D1 * D2)


def bits(x) -> bytes:
    return struct.pack("<d", float(x))


class TestTheDomain(unittest.TestCase):
    """Outside 1 + 3w > 0 both functions return None, before any Bessel function is evaluated."""

    def test_one_plus_three_w_is_exactly_zero_at_the_float_minus_a_third(self):
        self.assertEqual(1.0 + 3.0 * W_THIRD, 0.0)
        self.assertEqual(3.0 * W_THIRD, -1.0)

    def test_the_predicate_is_the_strict_inequality(self):
        self.assertFalse(constant_w_form_applies(W_THIRD))
        self.assertFalse(constant_w_form_applies(W_THIRD - 1.0e-12))
        self.assertFalse(constant_w_form_applies(-0.5))
        self.assertFalse(constant_w_form_applies(-1.0))
        self.assertTrue(constant_w_form_applies(W_THIRD + 1.0e-3))
        self.assertTrue(constant_w_form_applies(0.0))
        self.assertTrue(constant_w_form_applies(1.0 / 3.0))

    def test_both_functions_return_none_outside_the_domain(self):
        for label, w in OUTSIDE_W:
            for k, tau_source, tau, H_source, H in ARGUMENTS:
                with self.subTest(w=label, k=k, tau_source=tau_source, tau=tau):
                    self.assertIsNone(
                        compute_analytic_G(k, w, tau_source, tau, H_source)
                    )
                    self.assertIsNone(
                        compute_analytic_Gprime(k, w, tau_source, tau, H_source, H)
                    )


class TestInsideTheDomain(unittest.TestCase):
    """Inside the domain with a finite result, the value is the old expression's, to the bit."""

    def test_values_are_bit_identical_to_the_old_expression(self):
        checked = 0
        for label, w, arguments in INSIDE_W:
            for k, tau_source, tau, H_source, H in arguments:
                with self.subTest(w=label, k=k, tau_source=tau_source, tau=tau):
                    G_old = old_G(k, w, tau_source, tau, H_source)
                    Gprime_old = old_Gprime(k, w, tau_source, tau, H_source, H)
                    # a precondition of the case, not of the code: the old values are finite here
                    self.assertTrue(isfinite(G_old) and isfinite(Gprime_old))

                    G_new = compute_analytic_G(k, w, tau_source, tau, H_source)
                    Gprime_new = compute_analytic_Gprime(
                        k, w, tau_source, tau, H_source, H
                    )
                    self.assertIsNotNone(G_new)
                    self.assertIsNotNone(Gprime_new)
                    self.assertEqual(bits(G_new), bits(G_old))
                    self.assertEqual(bits(Gprime_new), bits(Gprime_old))
                    checked += 1
        self.assertEqual(checked, 3 * len(ARGUMENTS) + len(NEAR_BOUNDARY_ARGUMENTS))

    def test_zero_is_returned_not_none(self):
        # tau = tau_source: C - D is exactly zero in the old expression, which is finite
        k, w, tau_source, H_source = 1.0, 0.0, 2.0, 1.0
        self.assertEqual(old_G(k, w, tau_source, tau_source, H_source), 0.0)
        self.assertEqual(
            compute_analytic_G(k, w, tau_source, tau_source, H_source), 0.0
        )


class TestOverflow(unittest.TestCase):
    """
    Inside the domain, a value that is not finite is None.

    At w = 0 (b = 1, order 3/2), Y_{3/2}(x) ~ -Gamma(3/2)/pi (2/x)^{3/2} as x -> 0, about
    -8e374 at x = 1e-250, which is beyond the largest double (1.8e308): ``yv(1.5, 1e-250)`` returns
    -inf. With k = 1, tau_source = 1e-250 and tau = 1, the term J_{3/2}(k tau) Y_{3/2}(k tau_source)
    is then -inf, the other term is J_{3/2}(1e-250) = 0 (underflow) times a finite Y, and the old
    expressions give -inf (G) and +inf (G').
    """

    CASE = dict(k=1.0, w=0.0, tau_source=1.0e-250, tau=1.0, H_source=1.0, H=1.0)

    def test_yv_returns_inf_at_the_case(self):
        self.assertEqual(yv(1.5, 1.0e-250), float("-inf"))
        self.assertEqual(jv(1.5, 1.0e-250), 0.0)

    def test_the_old_expression_is_not_finite_and_the_new_one_is_none(self):
        c = self.CASE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            G_old = old_G(c["k"], c["w"], c["tau_source"], c["tau"], c["H_source"])
            Gprime_old = old_Gprime(
                c["k"], c["w"], c["tau_source"], c["tau"], c["H_source"], c["H"]
            )
            G_new = compute_analytic_G(
                c["k"], c["w"], c["tau_source"], c["tau"], c["H_source"]
            )
            Gprime_new = compute_analytic_Gprime(
                c["k"], c["w"], c["tau_source"], c["tau"], c["H_source"], c["H"]
            )
        self.assertFalse(isfinite(G_old))
        self.assertFalse(isfinite(Gprime_old))
        self.assertIsNone(G_new)
        self.assertIsNone(Gprime_new)


class TestTheRadiationForm(unittest.TestCase):
    """
    At w = 1/3 (b = 0) the form is unchanged and equals spec 02's closed form.

    With b = 0 the order is 1/2, and J_{1/2}(x) = sqrt(2/(pi x)) sin x, Y_{1/2}(x) = -sqrt(2/(pi x)) cos x
    (DLMF 10.16.1). Spec 02 R9 then gives Gr_k = sin(k(eta - eta'))/k, and the code's normalisation
    (spec 02 §0.1 item (2), G_code = -H(z') Gr_k) gives

        G = -H_source sin(k (tau - tau_source)) / k,   dG/dz = (H_source / H) cos(k (tau - tau_source)),

    the second from d tau/dz = -1/H.

    **The tolerance and its source.** The Bessel form takes the phase from k tau and k tau_source
    separately, each rounded to half an ulp, so the phase k(tau - tau_source) it reproduces carries
    an absolute error of order EPS * max(k tau, k tau_source); scipy's half-integer Bessel functions
    add a few ulp of their own. Measured over this module's grid, the worst errors relative to the
    envelopes H_source/k and H_source/H were 4.75 EPS max(1, k tau) for G and 12.6 for G'. The
    tolerance is 64 EPS max(1, k tau, k tau_source) of the envelope: five times the worst measured.
    """

    TOLERANCE_IN_EPS = 64.0

    def test_equals_the_closed_form(self):
        w = 1.0 / 3.0
        cases = 0
        for k in (1.0e-3, 1.0, 1.0e3, 1.0e5):
            for tau_source in (1.0e-4, 1.0e-2, 1.0, 10.0):
                for ratio in (1.0 + 1.0e-6, 1.01, 2.0, 10.0, 1.0e3):
                    tau = tau_source * ratio
                    H_source, H = 3.7, 0.9
                    phase = k * (tau - tau_source)
                    scale = EPS * max(1.0, k * tau, k * tau_source)

                    G = compute_analytic_G(k, w, tau_source, tau, H_source)
                    Gprime = compute_analytic_Gprime(k, w, tau_source, tau, H_source, H)
                    with self.subTest(k=k, tau_source=tau_source, tau=tau):
                        self.assertIsNotNone(G)
                        self.assertIsNotNone(Gprime)
                        G_closed = -H_source * sin(phase) / k
                        Gprime_closed = (H_source / H) * cos(phase)
                        self.assertLessEqual(
                            abs(G - G_closed) / (H_source / k),
                            self.TOLERANCE_IN_EPS * scale,
                        )
                        self.assertLessEqual(
                            abs(Gprime - Gprime_closed) / (H_source / H),
                            self.TOLERANCE_IN_EPS * scale,
                        )
                    cases += 1
        self.assertEqual(cases, 80)


if __name__ == "__main__":
    unittest.main()
