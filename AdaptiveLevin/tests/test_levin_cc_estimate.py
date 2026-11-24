"""
prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md section 3, as amended at Part A''s stop (section 9.4):
the Clenshaw-Curtis rule returns CC_{4N-3}, declares |CC_{8N-7} - CC_{4N-3}| plus the noise term kappa s_own on
its estimate and on both floors, and the C2 gate sends a region to it below CC_GATE_PHASE_VARIATION = 10 pi
(variant V93T10 of 07d Part A').

The parent is the dispatch commit's module (PARENT_SHA_07D, 07c's code, read with `git show` and executed under
another name, as the other modules of this directory read their parents). cc_rule_07c() runs this module on 07c's
code path: the parent's _adaptive_levin_subregion_cc() in place of this module's, and the gate at 6 pi. Every other
line of the driver is 07c's, so on that path the module computes what the parent computes, bit for bit
(docs/levin-phase-floor/07d/code/compare_v93t10_07d.py checks it on every call it runs). The other modules of this
directory use it where a test certifies a mechanism other than the Clenshaw-Curtis rule on a construction made of
07c's rule (07d's log, "Existing assertions changed").

No Ray and no datastore.
"""

import contextlib
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import AdaptiveLevin.levin_quadrature as lq

PARENT_SHA_07D = "ca1d8cfa86896565017e50b9e2acc61a7cb8e92e"
_PARENT_NAME_07D = "levin_quadrature_parent_07d_ca1d8cf"
_REPO = Path(__file__).resolve().parents[2]


def load_parent_07d():
    """
    The dispatch commit's module: AdaptiveLevin/levin_quadrature.py at PARENT_SHA_07D (07c's code), read with
    `git show` and executed as a module of another name, once per process.
    """
    module = sys.modules.get(_PARENT_NAME_07D)
    if module is None:
        source = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO),
                "show",
                f"{PARENT_SHA_07D}:AdaptiveLevin/levin_quadrature.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        module = types.ModuleType(_PARENT_NAME_07D)
        module.__file__ = (
            f"<git show {PARENT_SHA_07D[:7]}:AdaptiveLevin/levin_quadrature.py>"
        )
        sys.modules[_PARENT_NAME_07D] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def cc_rule_07c(module=lq):
    """
    07c's code path in this module: the parent's _adaptive_levin_subregion_cc() (CC_{2N-1} with
    |CC_{2N-1} - CC_N| and no noise term) in place of this module's, and the C2 gate at the parent's 6 pi. Every
    caller reaches the rule through the module attribute (_adaptive_levin_subregion_impl() and _oriented_cc()),
    so the patch reaches every Clenshaw-Curtis solve; the counts of section 1.4 still count the solves, and no
    route change or noise term is counted. Nothing else differs from the parent.
    """
    parent = load_parent_07d()
    with mock.patch.object(
        module, "_adaptive_levin_subregion_cc", parent._adaptive_levin_subregion_cc
    ), mock.patch.object(module, "CC_GATE_PHASE_VARIATION", parent.SIX_PI):
        yield


# --- the problems -----------------------------------------------------------------------------------------------------
import math
import unittest

import mpmath
import numpy as np

from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos
from AdaptiveLevin.tests.test_levin_chase_stop import noise

PRODUCTION_ORDER = 24  # ComputeTargets/QuadSourceIntegral.py's CHEBYSHEV_ORDER: CC_47, CC_93 and CC_185
OTHER_ORDER = lq.DEFAULT_LEVIN_CHEBSHEV_ORDER  # 16: CC_31, CC_61 and CC_121
ORDERS = (PRODUCTION_ORDER, OTHER_ORDER)
KAPPA = 4.0  # section 1.2's _CC_OWN_NOISE_KAPPA, as a literal
SPAN = (0.0, 1.0)
ALPHA = 3.0  # (a): theta = 3x, V = 3 rad, a Clenshaw-Curtis region at any gate
THETA_A = {"theta": lambda x: ALPHA * x, "theta_deriv": lambda x: ALPHA + 0.0 * x}
# (b): theta = 0, so the integrand is f_cos itself and a scatter added to f_cos reaches the samples unchanged
THETA_ZERO = {"theta": lambda x: 0.0 * x, "theta_deriv": lambda x: 0.0 * x}
KINK_AT = 0.37
ETA_A = 1e-9  # (a)'s pseudo-noise: eta n(x) added to f_cos, n uniform on [-1, 1), so sigma = eta / sqrt(3)
ETA_B = (1e-9, 1e-12)
# (b)'s band for s_own / (sigma sqrt(sum_j w_j^2)), with its basis: 07d Part A' measured sigma_own,93 / sigma_hat on
# 1,089 group leaves with 5th percentile 0.838 and median 1.01 (out/tables/kappa93.md); for one draw of 4N - 3 samples
# the rms of the plateau's few dozen coefficients scatters by about 1 / sqrt(2 * 40) = 11% about 1, so [0.5, 2] is
# about four times that scatter either side. Measured at 07d: 0.773 and 0.935 at N = 24, 0.814 and 1.004 at N = 16.
NOISE_BAND = (0.5, 2.0)


def poly(x):
    """(b)'s polynomial, degree 10, which CC_{4N-3} integrates exactly at both orders (4N - 3 > 10)."""
    return 1.0 + x - 2.0 * x**3 + 0.5 * x**7 - 0.3 * x**10


def cc_solve(span, f, theta, order, kappa=None):
    """
    _adaptive_levin_subregion_cc() on span, as _adaptive_levin_subregion_impl() calls it, with theta' on the
    order-N grid from build_Levin_data(); kappa, if given, patches _CC_OWN_NOISE_KAPPA for the call.
    """
    basis = lq._Basis_SinCos(theta)
    grid, D = lq.chebyshev_matrices(span, order)
    _, theta_prime, _, _, V = basis.build_Levin_data(
        grid, D, need_AmatT=False, vectorize_cache={}
    )
    args = (span, f, basis, order, V, theta_prime, None, None)
    if kappa is None:
        return lq._adaptive_levin_subregion_cc(*args, vectorize_cache={})
    with mock.patch.object(lq, "_CC_OWN_NOISE_KAPPA", kappa):
        return lq._adaptive_levin_subregion_cc(*args, vectorize_cache={})


def clenshaw_curtis_weights(n):
    """
    The n-point Clenshaw-Curtis weights on [-1, 1] for the nodes cos(pi j / (n - 1)), j = 0 ... n - 1, by
    Trefethen's clencurt (Spectral Methods in MATLAB, ch. 12), written out here so that the test's rule is
    independent of the module's _cc_weights_base().
    """
    m = n - 1
    t = np.pi * np.arange(n) / m
    w = np.zeros(n)
    inner = np.arange(1, m)
    v = np.ones(m - 1)
    if m % 2 == 0:
        w[0] = w[-1] = 1.0 / (m * m - 1)
        for k in range(1, m // 2):
            v = v - 2.0 * np.cos(2 * k * t[inner]) / (4 * k * k - 1)
        v = v - np.cos(m * t[inner]) / (m * m - 1)
    else:
        w[0] = w[-1] = 1.0 / (m * m)
        for k in range(1, (m - 1) // 2 + 1):
            v = v - 2.0 * np.cos(2 * k * t[inner]) / (4 * k * k - 1)
    w[inner] = 2.0 * v / m
    return w


def cc_independent(span, integrand, n):
    """
    CC_n on span computed in the test: the integrand sampled at the module's n nodes (chebyshev_matrices(): the
    same doubles, so a pseudo-noise of the bits of x is the same noise), the test's own weights, the test's own sum.
    """
    nodes = lq.chebyshev_matrices(span, n)[0]
    w = clenshaw_curtis_weights(n) * (0.5 * (span[1] - span[0]))
    return float(np.sum(w * np.array([integrand(x) for x in nodes])))


def mp_reference(span, integrand_mp, breaks=()):
    """The integral at 40 digits, split at breaks; as a float, with its 30-digit difference as its error."""
    points = [span[0], *breaks, span[1]]
    with mpmath.workdps(30):
        v30 = mpmath.quad(integrand_mp, points)
    with mpmath.workdps(40):
        v40 = mpmath.quad(integrand_mp, points)
        return float(v40), float(abs(v40 - v30))


def tent_on_odd_node(order):
    """
    (a)'s second jump case: the centre c of a tent at an odd node of the (8N - 7)-point grid of SPAN, near the
    middle, and its half-width, half the distance to the nearer neighbouring node of that grid, which are nodes of
    the (4N - 3)-point grid. So no node of the (2N - 1)- or (4N - 3)-point grid sees the tent, and the (8N - 7)-point
    grid sees it at one node: the shape of an error CC_{2N-1} and CC_{4N-3} share (07d Part A': 19 of the 74 leaves
    on which |CC_{2N-1} - CC_N| understated), with derivative jumps at c - delta, c and c + delta.
    """
    nodes = lq.chebyshev_matrices(SPAN, 8 * order - 7)[0]
    j = (len(nodes) - 1) // 2
    if j % 2 == 0:
        j -= 1
    c = float(nodes[j])
    delta = 0.5 * min(float(nodes[j - 1]) - c, c - float(nodes[j + 1]))
    return c, delta


def a_cases(order):
    """(a)'s four integrands on SPAN with theta = 3x: name -> (f, integrand(x), reference, its error)."""
    sin, cos, exp = np.sin, np.cos, np.exp
    cases = {}
    f = [lambda x: exp(-x), lambda x: cos(x)]
    g = lambda x: exp(-x) * sin(ALPHA * x) + cos(x) * cos(ALPHA * x)
    gm = lambda x: mpmath.exp(-x) * mpmath.sin(ALPHA * x) + mpmath.cos(x) * mpmath.cos(
        ALPHA * x
    )
    cases["smooth"] = (f, g, *mp_reference(SPAN, gm))
    f = [lambda x: np.abs(x - KINK_AT), lambda x: 0.0 * x]
    g = lambda x: abs(x - KINK_AT) * sin(ALPHA * x)
    gm = lambda x: abs(x - KINK_AT) * mpmath.sin(ALPHA * x)
    cases["kink"] = (f, g, *mp_reference(SPAN, gm, (KINK_AT,)))
    c, d = tent_on_odd_node(order)
    tent = lambda x: np.maximum(0.0, 1.0 - np.abs(x - c) / d)
    f = [lambda x: exp(-x) + tent(x), lambda x: cos(x)]
    g = lambda x: (exp(-x) + tent(x)) * sin(ALPHA * x) + cos(x) * cos(ALPHA * x)
    gm = lambda x: (mpmath.exp(-x) + max(0, 1 - abs(x - c) / d)) * mpmath.sin(
        ALPHA * x
    ) + mpmath.cos(x) * mpmath.cos(ALPHA * x)
    cases["tent"] = (f, g, *mp_reference(SPAN, gm, (c - d, c, c + d)))
    f = [lambda x: exp(-x), lambda x: cos(x) + ETA_A * noise(x)]
    g = lambda x: exp(-x) * sin(ALPHA * x) + (cos(x) + ETA_A * noise(x)) * cos(
        ALPHA * x
    )
    gm = lambda x: mpmath.exp(-x) * mpmath.sin(ALPHA * x) + mpmath.cos(x) * mpmath.cos(
        ALPHA * x
    )
    # The true error is against the noise-free integral, the noise's sum in the rule included: the noise moves any
    # rule with positive weights summing to the width by at most eta * width = ETA_A (test (a) checks it on
    # CC_{4N-3}), and the declaration is tested against what it did move the returned value by.
    cases["noise"] = (f, g, *mp_reference(SPAN, gm))
    return cases


class _Recorder:
    """Records each _cc_own_noise() result (s_own) the module computes, through its module attribute."""

    def __init__(self):
        self.values = []
        self._original = lq._cc_own_noise

    def __call__(self, g, w):
        s = self._original(g, w)
        self.values.append(s)
        return s


class TestEstimate(unittest.TestCase):
    def test_a_estimate_directly(self):
        """
        (a) The estimate, directly (07d section 3 item 1, as amended), on _adaptive_levin_subregion_cc() at N = 24
        (production's) and N = 16, theta = 3x on [0, 1], f = (f_sin, f_cos):
          * smooth: (e^{-x}, cos x);
          * a derivative jump: (|x - 0.37|, 0), a kink;
          * derivative jumps CC_{2N-1} and CC_{4N-3} cannot see: (e^{-x} + a tent, cos x), the tent at an odd node
            of the (8N - 7)-point grid (tent_on_odd_node()); CC_{2N-1} and CC_{4N-3} agree there to rounding, so
            their difference cannot see the tent's integral;
          * a pseudo-noise: (e^{-x}, cos x + eta n(x)), eta = 1e-9, n(x) in [-1, 1) a hash of x's bits (sigma =
            eta / sqrt(3)); the noise moves any rule's answer by at most eta (the width times eta), checked on
            CC_{4N-3} (measured: 5.1e-11 at N = 24), and the true error is against the noise-free integral, with
            that shift in it.
        On each, the value equals CC_{4N-3} computed in the test (clenshaw_curtis_weights(), at the module's nodes)
        within 1e-14 of the integrand's scale, and is not CC_{2N-1} (on the kink they differ by 2.6e-5 to 2.7e-4);
        and the declaration max(abserr_direct, round_off_err) is at or above the true error |value - reference|
        plus the reference's error. References: mpmath at 40 digits, split at the breaks, its 30-digit difference
        as its error.

        Measured at 07d, N = 24: kink true 1.06e-5 against declaration 1.41e-4, |CC_{8N-7} - CC_{4N-3}| alone
        6.07e-6; tent true 4.26e-3 against 8.53e-3; noise true 1.0e-12 against 2.2e-10. On the kink the pair alone
        is below the true error (its error falls as h^2, so the pair is about 3/4 of it), and the noise term,
        reading the slowly decaying coefficients, carries the declaration (07d's log, Observations).
        """
        for order in ORDERS:
            for name, (f, g, ref, ref_err) in a_cases(order).items():
                with self.subTest(order=order, case=name):
                    data = cc_solve(SPAN, f, THETA_A, order)
                    cc_value = cc_independent(SPAN, g, 4 * order - 3)
                    self.assertLessEqual(abs(data["value"] - cc_value), 1e-14, name)
                    if name == "kink":
                        cc_fine = cc_independent(SPAN, g, 2 * order - 1)
                        self.assertGreater(abs(data["value"] - cc_fine), 1e-6, name)
                    if name == "noise":
                        clean = a_cases(order)["smooth"][1]
                        shift = cc_value - cc_independent(SPAN, clean, 4 * order - 3)
                        self.assertGreater(abs(shift), 0.0)
                        self.assertLessEqual(abs(shift), ETA_A * (SPAN[1] - SPAN[0]))
                    declaration = max(data["abserr_direct"], data["phase_err_roundoff"])
                    true_error = abs(data["value"] - ref) + ref_err
                    self.assertGreaterEqual(
                        declaration,
                        true_error,
                        f"{name} N={order}: declaration {declaration:.3e} < error {true_error:.3e}",
                    )


class TestNoiseTerm(unittest.TestCase):
    def test_b1_polynomial_at_rounding(self):
        """
        (b) item 1. theta = 0 and f = (0, p), p of degree 10, which CC_{4N-3} integrates exactly: the noise term
        kappa s_own is at the rounding level, at most the rule's own round-off floor (the floor with kappa = 0),
        and so moves no declaration by more than that floor. Measured at 07d: 3.2e-15 against a floor of 1.1e-14
        at N = 24, 2.2e-15 against 7.5e-15 at N = 16.
        """
        f = [lambda x: 0.0 * x, poly]
        for order in ORDERS:
            with self.subTest(order=order):
                data = cc_solve(SPAN, f, THETA_ZERO, order)
                bare = cc_solve(SPAN, f, THETA_ZERO, order, kappa=0.0)
                noise_term = data["abserr_direct"] - bare["abserr_direct"]
                self.assertGreater(noise_term, 0.0)
                self.assertLessEqual(noise_term, bare["phase_err_roundoff"])
                self.assertFalse(data["cc_noise_set"])

    def test_b2_scatter_in_band(self):
        """
        (b) item 2. theta = 0 and f = (0, e^{-x} + eta n(x)), eta = 1e-9 and 1e-12, so the samples carry a scatter
        of sigma = eta / sqrt(3): s_own / (sigma sqrt(sum_j w_j^2)), w the CC_{4N-3} weights, lies in NOISE_BAND
        (see its comment for the basis), and the noise term sets the declaration at eta = 1e-9.
        """
        for order in ORDERS:
            w = lq._cc_weights(SPAN, 4 * order - 3)
            for eta in ETA_B:
                with self.subTest(order=order, eta=eta):
                    f = [lambda x: 0.0 * x, lambda x, e=eta: np.exp(-x) + e * noise(x)]
                    recorder = _Recorder()
                    with mock.patch.object(lq, "_cc_own_noise", recorder):
                        data = cc_solve(SPAN, f, THETA_ZERO, order)
                    (s_own,) = recorder.values
                    ratio = s_own / (eta / math.sqrt(3.0) * math.sqrt(np.sum(w * w)))
                    self.assertGreaterEqual(ratio, NOISE_BAND[0])
                    self.assertLessEqual(ratio, NOISE_BAND[1])
                    if eta == ETA_B[0]:
                        self.assertTrue(data["cc_noise_set"])

    def test_b3_noise_in_estimate_and_both_floors(self):
        """
        (b) item 3. The noise term is in abserr_direct, round_off_err ("phase_err_roundoff") and
        round_off_err_less_node ("phase_err_less_node"; no theta_abserr here, so its declared term is 0) each by
        exactly KAPPA s_own, KAPPA = 4 (section 1.2) and s_own the value _cc_own_noise() returned in the call: each
        equals the same solve's quantity at kappa = 0 plus 4 s_own, bit for bit, on (a)'s four integrands and on
        (b)'s polynomial and scatter, at both orders. And cc_noise_set is True iff 4 s_own exceeds max(estimate,
        floor), each read at kappa = 0.
        """
        problems = []
        for order in ORDERS:
            for name, (f, _, _, _) in a_cases(order).items():
                problems.append((order, name, f, THETA_A))
            problems.append((order, "poly", [lambda x: 0.0 * x, poly], THETA_ZERO))
            problems.append(
                (
                    order,
                    "scatter",
                    [lambda x: 0.0 * x, lambda x: np.exp(-x) + 1e-9 * noise(x)],
                    THETA_ZERO,
                )
            )
        for order, name, f, theta in problems:
            with self.subTest(order=order, case=name):
                recorder = _Recorder()
                with mock.patch.object(lq, "_cc_own_noise", recorder):
                    data = cc_solve(SPAN, f, theta, order)
                bare = cc_solve(SPAN, f, theta, order, kappa=0.0)
                (s_own,) = recorder.values
                term = KAPPA * s_own
                self.assertEqual(data["phase_err_declared"], 0.0)
                self.assertEqual(data["value"], bare["value"])
                for key in (
                    "abserr_direct",
                    "phase_err_roundoff",
                    "phase_err_less_node",
                ):
                    self.assertEqual(
                        data[key].hex(), (bare[key] + term).hex(), f"{name} {key}"
                    )
                self.assertEqual(
                    data["cc_noise_set"],
                    term > max(bare["abserr_direct"], bare["phase_err_roundoff"]),
                )


# (c): theta = a x on [0, 1], V = a; f = (e^{-x}, cos x)
F_C = [lambda x: np.exp(-x), lambda x: np.cos(x)]


def linear_theta(a):
    return {"theta": lambda x: a * x, "theta_deriv": lambda x: a + 0.0 * x}


class TestThreshold(unittest.TestCase):
    def test_c1_route_either_side_of_the_gate(self):
        """
        (c) item 1. CC_GATE_PHASE_VARIATION is 10 pi. A region of V = 0.999 x 10 pi goes to the Clenshaw-Curtis
        rule, one of V = 1.001 x 10 pi to the Levin solve (_adaptive_levin_subregion()).
        """
        self.assertEqual(lq.CC_GATE_PHASE_VARIATION, 10.0 * np.pi)
        for factor, direct in ((0.999, True), (1.001, False)):
            with self.subTest(factor=factor):
                a = factor * 10.0 * np.pi
                data = lq._adaptive_levin_subregion(
                    SPAN,
                    F_C,
                    lq._Basis_SinCos(linear_theta(a)),
                    id_label=None,
                    rtol=1e-10,
                )
                self.assertLess(abs(data["phase_span"] - a), 1e-12 * a)
                self.assertEqual(bool(data["is_direct"]), direct)

    def test_c2_route_changes_counted(self):
        """
        (c) item 2. A region with 6 pi <= V < 10 pi (V = 8 pi, and 9.99 pi) is sent to the Clenshaw-Curtis rule by
        the gate and counted in num_cc_gate_route_changes: the call's root region, its halves and quarters (the trust
        check) below 6 pi, so the count is 1. One below 6 pi (V = 5 pi) is not counted: 0. Each call converges in
        one region with 7 Clenshaw-Curtis solves (the root, two halves, four quarters).
        """
        for factor, changes in ((8.0, 1), (9.99, 1), (5.0, 0)):
            with self.subTest(V_over_pi=factor):
                data = adaptive_levin_sincos(
                    SPAN, F_C, linear_theta(factor * np.pi), atol=1e-16, rtol=1e-10
                )
                self.assertTrue(data["converged"])
                self.assertEqual(data["num_regions"], 1)
                self.assertEqual(data["num_cc_solves"], 7)
                self.assertEqual(data["num_cc_gate_route_changes"], changes)

    def test_c3_rule_a_reads_no_threshold(self):
        """
        (c) item 3. Rule A reads no threshold (since 07a; 07d section 9.1 item 7): on theta = A (x - 1/2)^2 over
        [0, 1], V = A / 2 = 8 pi with a root at 1/2, the stationary points and windows the call locates are the same
        with the gate at 10 pi, at 6 pi and at 100 pi.
        """
        A = 16.0 * np.pi
        theta = {
            "theta": lambda x: A * (x - 0.5) ** 2,
            "theta_deriv": lambda x: 2.0 * A * (x - 0.5),
        }

        def located():
            data = adaptive_levin_sincos(SPAN, F_C, theta, atol=1e-16, rtol=1e-10)
            return repr(
                (
                    data["stationary_points"],
                    [(w["lo"], w["hi"]) for w in data["stationary_windows"]],
                    data["location_skipped_phases"],
                )
            )

        here = located()
        self.assertIn("root", here)
        for gate in (6.0 * np.pi, 100.0 * np.pi):
            with mock.patch.object(lq, "CC_GATE_PHASE_VARIATION", gate):
                self.assertEqual(located(), here)


# (d): 07c's (f1): theta = 1e3 x on [0, 1], f = (e^{-x}, cos 30x), atol 1e-14, rtol 1e-7
F_D = [lambda x: np.exp(-x), lambda x: np.cos(30.0 * x)]
THETA_D = {"theta": lambda x: 1.0e3 * x, "theta_deriv": lambda x: 1.0e3 + 0.0 * x}
REGION_FIELDS = (
    "start",
    "end",
    "depth",
    "type",
    "abserr",
    "relerr",
    "phase_err",
    "phase_limited",
    "value",
    "value_kind",
    "value_own",
    "trusted",
    "delta1",
    "delta2",
    "convergence_ratio",
    "total_err",
    "budget_err",
    "node_error_signed",
    "phase_err_less_node",
    "window",
    "guarded",
    "guard_at_depth_max",
)


class TestNotTouched(unittest.TestCase):
    def test_d_untouched_call_bit_identical(self):
        """
        (d) A call the fix must not touch (07d section 3 item 4): 07c's (f1), theta = 1e3 x on [0, 1]. No
        Clenshaw-Curtis solve anywhere: four Levin leaves of V = 250 rad = 79.6 pi each, whose trust checks' quarters
        have V = 19.9 pi, above the gate; no stationary point, so no window and no guard. num_cc_solves and the other
        three counts are 0, and the call is bit-identical to the parent's (ca1d8cfa, 07c's code): value, abserr,
        converged, num_regions, max_depth and every region field (REGION_FIELDS).
        """
        data = adaptive_levin_sincos((0.0, 1.0), F_D, THETA_D, atol=1e-14, rtol=1e-7)
        parent = load_parent_07d().adaptive_levin_sincos(
            (0.0, 1.0), F_D, THETA_D, atol=1e-14, rtol=1e-7
        )
        for key in (
            "num_cc_solves",
            "num_cc_gate_route_changes",
            "num_cc_noise_set",
            "num_cc_resampled",
            "num_guarded_regions",
            "num_window_pieces",
        ):
            self.assertEqual(data[key], 0, key)
        self.assertEqual(data["stationary_points"], [])
        for region in data["regions"]:
            self.assertGreaterEqual(1.0e3 * (region.end - region.start), 40.0 * np.pi)
        self.assertEqual(data["value"].hex(), parent["value"].hex())
        self.assertEqual(data["abserr"].hex(), parent["abserr"].hex())
        for key in ("converged", "num_regions", "max_depth"):
            self.assertEqual(data[key], parent[key], key)
        self.assertEqual(len(data["regions"]), len(parent["regions"]))
        for new, old in zip(data["regions"], parent["regions"]):
            for name in REGION_FIELDS:
                self.assertEqual(
                    repr(getattr(new, name)), repr(getattr(old, name)), name
                )


# (e), (f): theta = 1e3 (x - 1/2)^2 on [0, 1], a root at 1/2 and its window; f = (e^{-x}, cos x) times
# (1 + eta n(x)); atol 1e-16, rtol 1e-8, depth_max 12. eta = 0: 8 window pieces, 70 Clenshaw-Curtis solves;
# eta = 1e-13: 5 window pieces, 142 solves, 16 route changes, 61 with the noise term setting the declaration.
A_E = 1.0e3
THETA_E = {
    "theta": lambda x: A_E * (x - 0.5) ** 2,
    "theta_deriv": lambda x: 2.0 * A_E * (x - 0.5),
}
ETA_E = (0.0, 1e-13)
KWARGS_E = dict(atol=1e-16, rtol=1e-8, depth_max=12)


def f_e(eta):
    return [
        lambda x: np.exp(-x) * (1.0 + eta * noise(x)),
        lambda x: np.cos(x) * (1.0 + eta * noise(x)),
    ]


def resasc_independent(span, f, theta, order):
    """
    dqk21's resasc on the (4N - 3)-point rule of span, computed in the test: sum_j |w_j| |g_j - gbar|,
    gbar = CC_{4N-3} / sum_j w_j, the integrand sampled at the module's nodes, the test's own weights.
    """
    n = 4 * order - 3
    nodes = lq.chebyshev_matrices(span, n)[0]
    w = clenshaw_curtis_weights(n) * (0.5 * abs(span[1] - span[0]))
    basis = lq._Basis_SinCos(theta)
    g = np.array(
        [sum(fi(x) * bi for fi, bi in zip(f, basis.eval_basis(x))) for x in nodes]
    )
    mean = float(np.sum(w * g)) / float(np.sum(w))
    return float(np.sum(np.abs(w) * np.abs(g - mean)))


class TestWindowCount(unittest.TestCase):
    def test_e_count_reads_the_returned_estimate(self):
        """
        (e) The window's count (07d section 3, as amended). On (e)'s calls, both noise levels, every
        _cc_leaf_unresolved(abserr, resasc) the window's count makes is fed the abserr_direct and the resasc of a
        Clenshaw-Curtis solve the call returned (recorded through the module attribute), and that resasc is the
        (4N - 3)-point rule's, equal within 1e-10 relatively to resasc_independent() on the solve's span and not to
        the (2N - 1)-point rule's resasc there. The count is reached: each call has a window, and at least one
        bisection of a window piece is tested.
        """
        order = lq.DEFAULT_LEVIN_CHEBSHEV_ORDER
        for eta in ETA_E:
            with self.subTest(eta=eta):
                f = f_e(eta)
                solves = {}
                calls = []
                original_cc = lq._adaptive_levin_subregion_cc
                original_unresolved = lq._cc_leaf_unresolved

                def cc(x_span, *args, **kwargs):
                    data = original_cc(x_span, *args, **kwargs)
                    solves[(data["abserr_direct"], data["resasc"])] = tuple(x_span)
                    return data

                def unresolved(abserr, resasc):
                    calls.append((abserr, resasc))
                    return original_unresolved(abserr, resasc)

                with mock.patch.object(
                    lq, "_adaptive_levin_subregion_cc", cc
                ), mock.patch.object(lq, "_cc_leaf_unresolved", unresolved):
                    data = adaptive_levin_sincos(SPAN, f, THETA_E, **KWARGS_E)
                self.assertGreater(len(data["stationary_windows"]), 0)
                self.assertGreater(len(calls), 0)
                for abserr, resasc in calls:
                    self.assertIn((abserr, resasc), solves)
                    span = solves[(abserr, resasc)]
                    want = resasc_independent(span, f, THETA_E, order)
                    self.assertLessEqual(abs(resasc - want), 1e-10 * want)
                    n = 2 * order - 1
                    nodes = lq.chebyshev_matrices(span, n)[0]
                    w = clenshaw_curtis_weights(n) * (0.5 * abs(span[1] - span[0]))
                    basis = lq._Basis_SinCos(THETA_E)
                    g = np.array(
                        [
                            sum(fi(x) * bi for fi, bi in zip(f, basis.eval_basis(x)))
                            for x in nodes
                        ]
                    )
                    mean = float(np.sum(w * g)) / float(np.sum(w))
                    coarse = float(np.sum(np.abs(w) * np.abs(g - mean)))
                    self.assertGreater(abs(resasc - coarse), 1e-8 * want)


class TestCounts(unittest.TestCase):
    def test_f_counts(self):
        """
        (f) The counts (07d section 1.4). On (e)'s calls: num_cc_solves equals the number of calls to
        _adaptive_levin_subregion_cc(), counted through a wrapper on the module attribute;
        num_cc_gate_route_changes the number of those reached through the gate (not through _oriented_cc(), the
        forced path) with V >= 6 pi; num_cc_noise_set the number whose 4 s_own (s_own recorded from
        _cc_own_noise()) exceeds max(|CC_{8N-7} - CC_{4N-3}|, floor), each read by re-solving the span with
        kappa = 0; and num_cc_resampled 0. At eta = 1e-13 all three are positive and the noise term sets some
        solves and not others (measured: 142 solves, 16 route changes, 61 set).
        """
        for eta in ETA_E:
            with self.subTest(eta=eta):
                f = f_e(eta)
                records = []
                forced = {"on": False}
                original_cc = lq._adaptive_levin_subregion_cc
                original_oriented = lq._oriented_cc
                recorder = _Recorder()

                def oriented(*args, **kwargs):
                    previous = forced["on"]
                    forced["on"] = True
                    try:
                        return original_oriented(*args, **kwargs)
                    finally:
                        forced["on"] = previous

                def cc(*args, **kwargs):
                    before = len(recorder.values)
                    data = original_cc(*args, **kwargs)
                    (s_own,) = recorder.values[before:]
                    with mock.patch.object(lq, "_CC_OWN_NOISE_KAPPA", 0.0):
                        bare = original_cc(*args, **kwargs)
                    del recorder.values[before + 1 :]
                    records.append((args[4], forced["on"], s_own, bare))
                    return data

                with mock.patch.object(
                    lq, "_adaptive_levin_subregion_cc", cc
                ), mock.patch.object(lq, "_oriented_cc", oriented), mock.patch.object(
                    lq, "_cc_own_noise", recorder
                ):
                    data = adaptive_levin_sincos(SPAN, f, THETA_E, **KWARGS_E)
                self.assertEqual(data["num_cc_solves"], len(records))
                self.assertEqual(
                    data["num_cc_gate_route_changes"],
                    sum(1 for V, on, _, _ in records if not on and V >= 6.0 * np.pi),
                )
                self.assertEqual(
                    data["num_cc_noise_set"],
                    sum(
                        1
                        for _, _, s, bare in records
                        if KAPPA * s
                        > max(bare["abserr_direct"], bare["phase_err_roundoff"])
                    ),
                )
                self.assertEqual(data["num_cc_resampled"], 0)
                if eta > 0.0:
                    self.assertGreater(data["num_cc_gate_route_changes"], 0)
                    self.assertGreater(data["num_cc_noise_set"], 0)
                    self.assertLess(data["num_cc_noise_set"], data["num_cc_solves"])


if __name__ == "__main__":
    unittest.main()
