"""
Compatibility and consumer-integration tests for the two-region Bessel phase object.

Where ``test_bessel_two_region.py`` scores the construction's *numbers* against the independent
references of prompt 01, this module tests the *interface*: that every accessor a consumer calls
exists with the signature it is called with, that the declared phase error is honest and reaches
``AdaptiveLevin``, that the object survives the ``ray.put`` round trip ``BesselPhaseProxy``
performs, and that the members whose meaning the two-region construction destroyed fail loudly
rather than quietly.

The consumer inventory these are drawn from is ``RECONCILIATION.md`` §3.3, re-checked by grep for
this commit; the accessor table is prompt 06 §2.
"""

import inspect
import math
import pickle
import unittest
import warnings

import mpmath as mp
import numpy as np

from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos
from LiouvilleGreen.bessel_phase import (
    EVALUATION_FLOOR,
    XSplineWrapper,
    bessel_phase,
)
from LiouvilleGreen.tests import bessel_reference as br

#: Orders built here. Deliberately the cheap ones: this module is about the interface, and
#: ``test_bessel_two_region.py`` owns the expensive high-order accuracy runs.
ORDERS = (0.5, 1.5, 2.5, 20.5)

_BUILD_CACHE = {}


def build(nu: float, max_x: float, **kwargs):
    key = (nu, max_x, tuple(sorted(kwargs.items())))
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = bessel_phase(nu, max_x, **kwargs)
    return _BUILD_CACHE[key]


def both_region_grid(data, count_per_region: int = 60):
    """A logarithmic sweep of the near region and of the tail, plus the crossover itself."""
    min_x = data["min_x"]
    max_x = data["max_x"]
    x_star = data["x_star"]

    parts = []
    if x_star > min_x:
        parts.append(np.geomspace(min_x, x_star, count_per_region))
    if max_x > x_star:
        parts.append(np.geomspace(max(x_star, min_x), max_x, count_per_region))
    return np.unique(np.concatenate(parts))


class TestAdapterSurface(unittest.TestCase):
    """
    Prompt 06 §2's table, verified against the actual call sites rather than against the plan.

    The keyword names are part of the contract and differ between the two objects on purpose:
    ``phase.*`` takes ``x_is_log`` (what ``phase_spline`` took) and ``mod.*`` takes ``is_log``
    (what ``XSplineWrapper`` took). ``three_bessel_integrals.py``, ``QuadSourceIntegral.py`` and
    ``test_bessel_phase.py`` all call them positionally *or* by those names, so a "tidy-up" that
    unified them would break call sites this campaign may not edit.
    """

    def test_the_required_accessors_exist_with_the_documented_signatures(self):
        data = build(2.5, 1.0e7)
        phase = data["phase"]
        mod = data["mod"]

        expected = {
            "raw_theta": ["x", "x_is_log"],
            "theta_mod_2pi": ["x", "x_is_log"],
            "theta_deriv": ["x", "x_is_log", "log_derivative"],
            "residual": ["x", "x_is_log"],
            "residual_log_deriv": ["x", "x_is_log"],
            "log_amplitude": ["x", "x_is_log"],
            "sin_cos_theta": ["x", "x_is_log"],
            "theta_deriv_from_residual": ["x", "x_is_log"],
            "theta_abserr_at": ["x", "x_is_log"],
            "bessel_j": ["x", "is_log"],
            "bessel_y": ["x", "is_log"],
        }
        for name, parameters in expected.items():
            with self.subTest(accessor=name):
                self.assertTrue(hasattr(phase, name))
                signature = inspect.signature(getattr(phase, name))
                self.assertEqual(list(signature.parameters), parameters)

        for name, parameters in {
            "__call__": ["x", "is_log"],
            "a": ["x", "is_log"],
            "log_deriv": ["x", "is_log"],
        }.items():
            with self.subTest(accessor=f"mod.{name}"):
                signature = inspect.signature(getattr(mod, name))
                self.assertEqual(list(signature.parameters), parameters)

        for attribute in ("nu", "min_x", "max_x", "x_star", "c_nu", "c_nu_reduced"):
            with self.subTest(attribute=attribute):
                self.assertTrue(hasattr(phase, attribute))

    def test_raw_and_log_input_modes_agree_at_matched_arguments(self):
        """
        ``DRAFT-PLAN.md`` §7.5: "matched" means the log mode is scored at ``exp(u)``, the same
        double, not at the ``x`` that ``u`` was formed from. The accessors are compared to each
        other here, so the tolerance is the size of the disagreement the ``exp(log(x))`` round trip
        introduces, not an accuracy claim.
        """
        for nu in ORDERS:
            data = build(nu, 1.0e7)
            phase = data["phase"]
            mod = data["mod"]

            for x in both_region_grid(data, count_per_region=25):
                u = math.log(x)
                x_round = math.exp(u)
                if not (data["min_x"] <= x_round <= data["max_x"]):
                    continue
                with self.subTest(nu=nu, x=x):
                    # raw_theta is eps*theta-limited, so the agreement it can offer is a few ulps
                    # of the leading term, not an absolute constant
                    self.assertAlmostEqual(
                        phase.raw_theta(u, x_is_log=True),
                        phase.raw_theta(x_round),
                        delta=8.0 * float(np.spacing(x_round)) + 1.0e-12,
                    )
                    self.assertAlmostEqual(
                        phase.residual(u, x_is_log=True),
                        phase.residual(x_round),
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        phase.theta_mod_2pi(u, x_is_log=True),
                        phase.theta_mod_2pi(x_round),
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        phase.theta_deriv(u, x_is_log=True),
                        phase.theta_deriv(x_round),
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        phase.log_amplitude(u, x_is_log=True),
                        phase.log_amplitude(x_round),
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        mod(u, is_log=True), mod(x_round), delta=1.0e-13 * mod(x_round)
                    )

    def test_arrays_and_scalars_both_work(self):
        """Scalars in, scalars out; arrays in, arrays out, with the same values."""
        data = build(2.5, 1.0e7)
        phase = data["phase"]
        mod = data["mod"]
        xs = np.array([5.0, 30.0, data["x_star"], 1.0e4, 1.0e7])

        for accessor in (
            phase.raw_theta,
            phase.theta_mod_2pi,
            phase.theta_deriv,
            phase.residual,
            phase.log_amplitude,
            phase.theta_abserr_at,
        ):
            with self.subTest(accessor=accessor.__name__):
                vector = np.asarray(accessor(xs), dtype=float)
                scalars = np.array([accessor(float(x)) for x in xs])
                np.testing.assert_array_equal(vector, scalars)

        np.testing.assert_array_equal(
            np.asarray(mod(xs), dtype=float),
            np.array([mod(float(x)) for x in xs]),
        )

    def test_XSplineWrapper_is_still_importable_and_mod_is_one(self):
        """
        ``LiouvilleGreen/tests/test_three_bessel.py:10`` imports ``XSplineWrapper`` by name from
        this module and annotates ``mod`` with it at ``:67``; that file belongs to prompt 08, so
        both have to keep working here.
        """
        self.assertTrue(inspect.isclass(XSplineWrapper))
        self.assertIsInstance(build(2.5, 1.0e7)["mod"], XSplineWrapper)


class TestDeclaredPhaseError(unittest.TestCase):
    def test_theta_abserr_is_present_finite_and_positive(self):
        for nu in ORDERS:
            data = build(nu, 1.0e7)
            phase = data["phase"]
            with self.subTest(nu=nu):
                self.assertIn("theta_abserr", data)
                self.assertGreater(data["theta_abserr"], 0.0)
                self.assertTrue(math.isfinite(data["theta_abserr"]))
                self.assertEqual(phase.theta_abserr, data["theta_abserr"])
                for x in both_region_grid(data, count_per_region=20):
                    value = phase.theta_abserr_at(x)
                    self.assertGreater(value, 0.0)
                    self.assertTrue(math.isfinite(value))
                    self.assertLessEqual(value, data["theta_abserr"])
                    self.assertGreaterEqual(value, EVALUATION_FLOOR)

    def test_theta_abserr_at_never_under_reports_at_the_cached_corners(self):
        """
        The declared error must be at least the measured phase-pair error **at the same x**, in
        both regions, at every order the campaign covers. Scored against the committed 40-digit
        ``mpmath`` corners, which are independent of SciPy at every order (standing note 15: above
        ``nu = 20.5`` SciPy itself is not adequate below ~1e-12, so it could not settle this).

        An estimator that under-reports is worse than declaring nothing at all, so the inequality
        is asserted in the safe direction and the margin is printed.
        """
        worst_ratio = math.inf
        worst_at = None
        for nu in br.cached_orders():
            data = build(nu, 1.0e16)
            phase = data["phase"]
            for corner in br.cached_corners(nu):
                if not (data["min_x"] <= corner.x <= data["max_x"]):
                    continue
                sin_theta, cos_theta = phase.sin_cos_theta(corner.x)
                measured = max(
                    abs(sin_theta - corner.J / corner.amplitude),
                    abs(-cos_theta - corner.Y / corner.amplitude),
                )
                declared = phase.theta_abserr_at(corner.x)
                with self.subTest(nu=nu, x=corner.x):
                    self.assertGreaterEqual(declared, measured)
                if measured > 0.0 and declared / measured < worst_ratio:
                    worst_ratio = declared / measured
                    worst_at = (nu, corner.x, measured, declared)

        print(
            f"corner declared/measured: worst ratio {worst_ratio:.2f} at "
            f"nu={worst_at[0]}, x={worst_at[1]:.6g} (measured {worst_at[2]:.3e}, declared "
            f"{worst_at[3]:.3e})"
        )
        self.assertGreater(worst_ratio, 1.0)

    def test_theta_abserr_at_never_under_reports_on_a_dense_sweep(self):
        """
        The corners are 9-11 points per order. This is the same inequality on a dense sweep of both
        regions.

        The reference tier is ``mpmath``, not SciPy, and that is **required** rather than
        belt-and-braces: the tail declares as little as 8.9e-16 rad, and SciPy's own error is
        larger than that at low order well inside its validity boundary -- measured 1.07e-14 at
        ``nu = 1/2, x = 16.2576``, where the representation is exact and the closed form agrees to
        an ulp, so the whole 1.07e-14 is Amos's (log 06). Scoring the declaration against SciPy
        would therefore report a spurious under-declaration of an error we do not have. This is
        standing note 15's precision floor showing up two orders lower in ``nu`` than it was
        measured at, and it costs ~2.7 ms per point to avoid.
        """
        worst_ratio = math.inf
        worst_at = None
        for nu in (0.5, 1.5, 1.75, 2.5, 20.5):
            data = build(nu, 1.0e7)
            phase = data["phase"]
            xs = both_region_grid(data, count_per_region=100)
            J, Y = br.reference_JY(nu, xs, tier=br.TIER_MPMATH)
            amplitude = np.hypot(J, Y)
            sin_theta, cos_theta = phase.sin_cos_theta(xs)
            measured = np.maximum(
                np.abs(sin_theta - J / amplitude), np.abs(-cos_theta - Y / amplitude)
            )
            declared = np.asarray(phase.theta_abserr_at(xs), dtype=float)
            with self.subTest(nu=nu):
                self.assertTrue(np.all(declared >= measured))
            ratio = declared / np.maximum(measured, 1.0e-300)
            index = int(np.argmin(ratio))
            if ratio[index] < worst_ratio:
                worst_ratio = float(ratio[index])
                worst_at = (nu, float(xs[index]), float(measured[index]))

        print(
            f"dense declared/measured: worst ratio {worst_ratio:.2f} at nu={worst_at[0]}, "
            f"x={worst_at[1]:.6g} (measured {worst_at[2]:.3e})"
        )
        self.assertGreater(worst_ratio, 1.0)

    def test_the_tail_declaration_is_far_smaller_than_the_near_one(self):
        """
        The reason ``theta_abserr_at`` is a callable rather than the scalar: above ``x_star``
        nothing is interpolated and nothing samples ``hankel1e``, so the only representation error
        is the series remainder, which decays like ``x^-(2n+1)``.
        """
        data = build(2.5, 1.0e7)
        phase = data["phase"]
        near = phase.theta_abserr_at(0.5 * (data["min_x"] + data["x_star"]))
        tail = phase.theta_abserr_at(1.0e7)
        print(
            f"nu=2.5: near declares {near:.3e} rad, tail at 1e7 declares {tail:.3e} rad"
        )
        self.assertGreater(near / tail, 100.0)


class TestLevinConsumption(unittest.TestCase):
    """
    A real ``adaptive_levin_sincos`` call on ``integral A_nu sin theta_nu dx = integral J_nu dx``,
    driven by the four-key phase dictionary prompt 06 settles.

    This is the campaign's concrete demonstration that the accuracy claim reaches the consumer: the
    declared phase error appears in the reported ``abserr``, and the raw phase is never evaluated.
    """

    NU = 2.5
    NEAR_SPAN = (10.0, 60.0)
    TAIL_SPAN = (1000.0, 1100.0)

    @classmethod
    def setUpClass(cls):
        cls.data = build(cls.NU, 1.0e5)
        cls.phase = cls.data["phase"]
        cls.mod = cls.data["mod"]

    def _theta_dict(self, theta_abserr=None, theta=None):
        entries = {
            "theta": self.phase.raw_theta if theta is None else theta,
            "theta_mod_2pi": self.phase.theta_mod_2pi,
            "theta_deriv": self.phase.theta_deriv,
        }
        if theta_abserr is not None:
            entries["theta_abserr"] = theta_abserr
        return entries

    def _run(self, span, theta_abserr=None, theta=None):
        return adaptive_levin_sincos(
            span,
            [lambda x: self.mod(x), lambda x: 0.0],
            self._theta_dict(theta_abserr=theta_abserr, theta=theta),
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    @staticmethod
    def _reference(nu, a, b):
        """``integral_a^b J_nu`` from 40-digit ``mpmath``, split at 2 pi intervals."""
        with mp.workdps(40):
            count = max(2, int((b - a) / (2.0 * math.pi)) + 1)
            points = [
                mp.mpf(a) + (mp.mpf(b) - mp.mpf(a)) * index / count
                for index in range(count + 1)
            ]
            return float(mp.quad(lambda t: mp.besselj(nu, t), points))

    def test_it_converges_and_reproduces_an_independent_value(self):
        for span in (self.NEAR_SPAN, self.TAIL_SPAN):
            result = self._run(span, theta_abserr=self.phase.theta_abserr_at)
            reference = self._reference(self.NU, *span)
            with self.subTest(span=span):
                self.assertTrue(result["converged"])
                self.assertLess(abs(result["value"] - reference), 1.0e-13)
            print(
                f"span={span}: value={result['value']:.16e}, mpmath={reference:.16e}, "
                f"|difference|={abs(result['value'] - reference):.3e}, "
                f"abserr={result['abserr']:.6e}"
            )

    def test_the_declared_phase_error_dominates_the_reported_abserr(self):
        """
        Without a declaration the reported error is the eq. (151) round-off floor and the
        resolution residual, and says nothing about the phase construction. With one it is
        dominated by the declared value -- the point of ``levin_quadrature.py:2360``'s "so the
        caller sees an honest number instead of an artificially small one".
        """
        undeclared = self._run(self.NEAR_SPAN)
        declared = self._run(self.NEAR_SPAN, theta_abserr=self.phase.theta_abserr_at)

        print(
            f"near span {self.NEAR_SPAN}: abserr {undeclared['abserr']:.6e} -> "
            f"{declared['abserr']:.6e} (roundoff {undeclared['abserr_roundoff']:.6e} -> "
            f"{declared['abserr_roundoff']:.6e}, resolution "
            f"{declared['abserr_resolution']:.6e})"
        )
        self.assertGreater(declared["abserr"], 10.0 * undeclared["abserr"])
        self.assertGreater(
            declared["abserr_roundoff"], 100.0 * declared["abserr_resolution"]
        )
        # ... and the value itself is unchanged: declaring an error is bookkeeping, not a change
        # of quadrature.
        self.assertEqual(declared["value"], undeclared["value"])

    def test_the_callable_declaration_beats_the_scalar_in_the_tail(self):
        """
        Levin applies the declared error as an endpoint term on *every* region's floor, so a
        domain-wide scalar inflates the reported error of a far-tail region by the ratio of the two
        regions' errors. This is the measurement that justifies the callable.
        """
        callable_run = self._run(
            self.TAIL_SPAN, theta_abserr=self.phase.theta_abserr_at
        )
        scalar_run = self._run(self.TAIL_SPAN, theta_abserr=self.phase.theta_abserr)
        print(
            f"tail span {self.TAIL_SPAN}: abserr callable {callable_run['abserr']:.6e} vs "
            f"scalar {scalar_run['abserr']:.6e} "
            f"(factor {scalar_run['abserr'] / callable_run['abserr']:.1f})"
        )
        self.assertLess(callable_run["abserr"], 0.1 * scalar_run["abserr"])

    def test_the_raw_phase_is_never_evaluated(self):
        """
        ``RECONCILIATION.md`` §1's reading of ``levin_quadrature.py:1038``: ``need_theta_Cheb`` is
        ``False`` whenever both ``theta_mod_2pi`` and ``theta_deriv`` are supplied, so ``theta`` is
        a *required key* that is never called. Evidence rather than assertion: the same run with a
        ``theta`` that raises must produce bit-identical output.

        This matters because ``raw_theta`` is ``eps * theta``-limited by construction -- 2.2e-1 rad
        at ``x = 1e15`` -- so a route that evaluated it would silently discard the campaign's gain.
        """

        def exploding_theta(x):
            raise AssertionError(
                "levin_quadrature evaluated theta['theta'], which RECONCILIATION.md 1 says it "
                "must not when theta_mod_2pi and theta_deriv are both supplied"
            )

        for span in (self.NEAR_SPAN, self.TAIL_SPAN):
            with self.subTest(span=span):
                supplied = self._run(span, theta_abserr=self.phase.theta_abserr_at)
                exploding = self._run(
                    span,
                    theta_abserr=self.phase.theta_abserr_at,
                    theta=exploding_theta,
                )
                self.assertEqual(exploding["value"], supplied["value"])
                self.assertEqual(exploding["abserr"], supplied["abserr"])

    def test_theta_is_still_a_required_key(self):
        """``_Basis_SinCos.__init__`` raises without it (``levin_quadrature.py:948-952``)."""
        entries = self._theta_dict(theta_abserr=self.phase.theta_abserr_at)
        del entries["theta"]
        with self.assertRaises(RuntimeError):
            adaptive_levin_sincos(
                self.NEAR_SPAN,
                [lambda x: self.mod(x), lambda x: 0.0],
                entries,
                atol=1.0e-10,
                rtol=1.0e-10,
            )


class TestRemovedAndRetainedMembers(unittest.TestCase):
    def test_Q_is_gone_and_says_what_to_use_instead(self):
        """
        ``DRAFT-PLAN.md`` §8.1 forbids serving a different quantity under ``Q``'s name. It is
        therefore removed rather than redefined, and the ``KeyError`` names the replacement instead
        of being a bare ``KeyError: 'Q'``.
        """
        data = build(2.5, 1.0e7)
        self.assertNotIn("Q", data)
        self.assertIsNone(data.get("Q"))

        with self.assertRaises(KeyError) as caught:
            data["Q"]
        message = caught.exception.args[0]
        self.assertIn("removed", message)
        self.assertIn("phase.residual", message)
        self.assertIn("raw_theta", message)

        # an ordinary missing key is still an ordinary KeyError
        with self.assertRaises(KeyError) as caught:
            data["not_a_member"]
        self.assertEqual(caught.exception.args[0], "not_a_member")

    def test_phi_is_retained_and_zero(self):
        """
        ``DRAFT-PLAN.md`` §4.3 requires ``phi`` to be *reported* as identically zero rather than
        quietly dropped: a consumer that reads it should get the right answer. There is no root
        solve, so it is exactly 0.0 at every order, ``nu = 1/2`` included.
        ``test_bessel_two_region.TestStructure.test_phi_is_identically_zero`` covers the high
        orders as well.
        """
        for nu in ORDERS:
            with self.subTest(nu=nu):
                self.assertEqual(build(nu, 1.0e7)["phi"], 0.0)

    def test_the_other_pre_existing_members_are_unchanged(self):
        data = build(2.5, 1.0e7)
        for key in ("phase", "mod", "phi", "bessel_j", "bessel_y", "min_x", "max_x"):
            self.assertIn(key, data)
        self.assertEqual(data["bessel_j"](30.0), data["phase"].bessel_j(30.0))
        self.assertEqual(data["bessel_y"](30.0), data["phase"].bessel_y(30.0))

    def test_the_deprecation_shim(self):
        """
        ``atol``/``rtol`` named tolerances of an ODE solve that no longer exists, so they map to
        nothing. Production does not rely on this: ``main.py`` passes ``phase_atol``/
        ``amplitude_rtol`` as of this commit, and the shim is for script and third-party callers.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bessel_phase(2.5, 1000.0, phase_atol=1.0e-11, amplitude_rtol=1.0e-11)
        self.assertEqual(
            [w for w in caught if issubclass(w.category, DeprecationWarning)], []
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old = bessel_phase(2.5, 1000.0, atol=1e-25, rtol=5e-14)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        message = str(deprecations[0].message)
        self.assertIn("phase_atol", message)
        self.assertIn("amplitude_rtol", message)
        self.assertIn("no effect", message)
        # ignored, so the result is the default build
        default = bessel_phase(2.5, 1000.0)
        self.assertEqual(old["x_star"], default["x_star"])
        self.assertEqual(old["theta_abserr"], default["theta_abserr"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            both = bessel_phase(2.5, 1000.0, atol=1e-25, phase_atol=1.0e-9)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("take precedence", str(deprecations[0].message))
        self.assertEqual(both["accuracy"]["phase_atol"], 1.0e-9)


class TestSerialization(unittest.TestCase):
    """
    ``BesselPhaseProxy`` (``ComputeTargets/QuadSourceIntegral.py:121``) puts the whole returned
    mapping through ``ray.put`` and ``ray.get`` in the worker, so every accessor has to survive a
    pickle round trip and return *bit-identical* values afterwards.

    Plain ``pickle`` is tested unconditionally, because it is the stronger claim and needs no
    optional dependency (``README.md`` §5 item 7: tests must not need Ray). ``ray.cloudpickle`` --
    the codec ``ray.put`` actually uses -- is tested too when Ray imports.
    """

    GRID = (5.0, 30.0, 64.0, 100.0, 1.0e4, 1.0e7)

    def _assert_identical(self, original, revived):
        for x in self.GRID:
            if not (original["min_x"] <= x <= original["max_x"]):
                continue
            u = math.log(x)
            a, b = original["phase"], revived["phase"]
            with self.subTest(x=x):
                self.assertEqual(a.raw_theta(x), b.raw_theta(x))
                self.assertEqual(a.theta_mod_2pi(x), b.theta_mod_2pi(x))
                self.assertEqual(a.theta_deriv(x), b.theta_deriv(x))
                self.assertEqual(
                    a.theta_deriv(x, log_derivative=True),
                    b.theta_deriv(x, log_derivative=True),
                )
                self.assertEqual(a.residual(x), b.residual(x))
                self.assertEqual(a.residual_log_deriv(x), b.residual_log_deriv(x))
                self.assertEqual(a.log_amplitude(x), b.log_amplitude(x))
                self.assertEqual(a.sin_cos_theta(x), b.sin_cos_theta(x))
                self.assertEqual(a.theta_abserr_at(x), b.theta_abserr_at(x))
                self.assertEqual(a.bessel_j(x), b.bessel_j(x))
                self.assertEqual(a.bessel_y(x), b.bessel_y(x))
                # the logarithmic input mode too
                self.assertEqual(
                    a.raw_theta(u, x_is_log=True), b.raw_theta(u, x_is_log=True)
                )
                self.assertEqual(
                    a.theta_deriv(u, x_is_log=True), b.theta_deriv(u, x_is_log=True)
                )
                self.assertEqual(original["mod"](x), revived["mod"](x))
                self.assertEqual(
                    original["mod"](u, is_log=True), revived["mod"](u, is_log=True)
                )
                self.assertEqual(original["mod"].a(x), revived["mod"].a(x))
                self.assertEqual(
                    original["mod"].log_deriv(x), revived["mod"].log_deriv(x)
                )
                self.assertEqual(original["bessel_j"](x), revived["bessel_j"](x))
                self.assertEqual(original["bessel_y"](x), revived["bessel_y"](x))

        for key in ("phi", "min_x", "max_x", "nu", "x_star", "theta_abserr"):
            self.assertEqual(original[key], revived[key])
        self.assertEqual(original["accuracy"], revived["accuracy"])
        self.assertEqual(original["crossover"], revived["crossover"])

    def test_plain_pickle_round_trip_is_bit_identical(self):
        for nu in (0.5, 2.5, 20.5):
            with self.subTest(nu=nu):
                original = build(nu, 1.0e7)
                revived = pickle.loads(pickle.dumps(original))
                self.assertIsInstance(revived, dict)
                self._assert_identical(original, revived)

    def test_the_removed_key_still_explains_itself_after_a_round_trip(self):
        revived = pickle.loads(pickle.dumps(build(2.5, 1.0e7)))
        with self.assertRaises(KeyError) as caught:
            revived["Q"]
        self.assertIn("phase.residual", caught.exception.args[0])

    def test_ray_cloudpickle_round_trip_is_bit_identical(self):
        try:
            import ray.cloudpickle as cloudpickle
        except ImportError:  # pragma: no cover - Ray is optional at test time
            self.skipTest("ray is not importable; plain pickle covers the codec")

        original = build(2.5, 1.0e7)
        revived = cloudpickle.loads(cloudpickle.dumps(original))
        self._assert_identical(original, revived)


if __name__ == "__main__":
    unittest.main()
