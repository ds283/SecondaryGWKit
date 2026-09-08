"""
Regression tests for the derivative-building code path of ComputeTargets/BackgroundModel.py.

`compute_background` uses a cosmology's analytic derivative methods when it has them, and otherwise
estimates them by differentiating an interpolating spline. Only the second branch is exercised in
production, by `LambdaCDM_GenericEOS` and `QCD_Cosmology`, neither of which supplies analytic
derivatives -- but neither has an independent closed form to test against. We therefore drive the
spline branch with a `LambdaCDM` background, whose analytic derivatives supply the reference, by
wrapping it in a proxy that hides the five analytic methods (audit item A7 / TK-5 was measured
exactly this way, in docs/spec-code-audit/scripts/TK_06_spline_end_bias.py).

No Ray cluster and no datastore is needed: `compute_background` is called through the undecorated
function, and `redshift`/`redshift_array` are plain objects.
"""

import unittest
from math import log

import numpy as np

from ComputeTargets.BackgroundModel import compute_background
from CosmologyConcepts import redshift, redshift_array
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

# the grid main.py actually builds for the background model: z in [0.1, 1e12], 100 samples per decade
Z_MIN = 0.1
Z_MAX = 1.0e12
SAMPLES_PER_LOG10Z = 100

# the derivative methods LambdaCDM supplies analytically and LambdaCDM_GenericEOS does not
SPLINED_ATTRS = (
    "d_lnH_dz",
    "d2_lnH_dz2",
    "d3_lnH_dz3",
    "d_wPerturbations_dz",
    "d2_wPerturbations_dz2",
)

# acceptance threshold: the relative error at either end of the grid, measured against the exact
# derivatives, must be no worse than this multiple of the interior median relative error. Before
# the padded/refined fit grid was introduced the end/interior ratio reached 5e3 for eps'' and 2e7
# for w''.
END_TO_INTERIOR_TOLERANCE = 10.0


class _HideAnalyticDerivatives:
    """
    Proxy that forwards everything to the wrapped cosmology except the analytic derivative methods,
    which it hides so that `hasattr(cosmology, attr)` is False and compute_background is forced down
    its spline branch.
    """

    def __init__(self, cosmology):
        self._cosmology = cosmology

    def __getattr__(self, item):
        if item in SPLINED_ATTRS:
            raise AttributeError(item)
        return getattr(self._cosmology, item)


# compute_background is a ray remote function; the undecorated body is what we drive here
_compute_background = compute_background._function


def _z_sample(z_min: float = Z_MIN, z_max: float = Z_MAX) -> redshift_array:
    n = int(round(SAMPLES_PER_LOG10Z * (np.log10(z_max) - np.log10(z_min))))
    values = np.expm1(np.linspace(log(1.0 + z_min), log(1.0 + z_max), n))
    return redshift_array(
        [redshift(store_id=i, z=float(z)) for i, z in enumerate(values)]
    )


class TestBackgroundDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cosmology = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        cls.z_sample = _z_sample()
        cls.z = np.array([z.z for z in cls.z_sample], dtype=float)

    def _payload(self, cosmology):
        return _compute_background(cosmology, self.z_sample)

    def test_analytic_branch_is_untouched(self):
        """
        A cosmology that supplies analytic derivatives must be bit-identical to a direct evaluation
        of those methods -- the padded fit grid must not perturb it.
        """
        payload = self._payload(self.cosmology)

        for attr in SPLINED_ATTRS:
            method = getattr(self.cosmology, attr)
            expected = [method(z.z) for z in self.z_sample]
            got = payload[f"{attr}_sample"]
            self.assertEqual(len(got), len(expected))
            for i, (a, b) in enumerate(zip(got, expected)):
                self.assertEqual(
                    a, b, msg=f"{attr} differs at index {i}: {a!r} != {b!r}"
                )

    def test_sample_ordering_matches_z_sample(self):
        """
        z_sample is held in descending order of z; the returned samples must follow it, not the
        ascending order the splines are fitted in.
        """
        payload = self._payload(self.cosmology)
        d_lnH_dz = np.array(payload["d_lnH_dz_sample"])
        exact = np.array([self.cosmology.d_lnH_dz(z) for z in self.z])
        np.testing.assert_array_equal(d_lnH_dz, exact)

        spline_payload = self._payload(_HideAnalyticDerivatives(self.cosmology))
        splined = np.array(spline_payload["d_lnH_dz_sample"])
        # correlated with the exact values in the same order to well within the spline error
        self.assertLess(np.max(np.abs(splined / exact - 1.0)), 1.0e-6)

    def _spline_branch_errors(self):
        """
        Relative errors of the five splined quantities against LambdaCDM's exact derivatives,
        in the order of z_sample (descending, so index 0 is the z=1e12 end and index -1 the
        z=0.1 end).
        """
        payload = self._payload(_HideAnalyticDerivatives(self.cosmology))

        opz = 1.0 + self.z
        d1 = np.array(payload["d_lnH_dz_sample"])
        d2 = np.array(payload["d2_lnH_dz2_sample"])
        d3 = np.array(payload["d3_lnH_dz3_sample"])
        w1 = np.array(payload["d_wPerturbations_dz_sample"])
        w2 = np.array(payload["d2_wPerturbations_dz2_sample"])

        e1 = np.array([self.cosmology.d_lnH_dz(z) for z in self.z])
        e2 = np.array([self.cosmology.d2_lnH_dz2(z) for z in self.z])
        e3 = np.array([self.cosmology.d3_lnH_dz3(z) for z in self.z])
        ew1 = np.array([self.cosmology.d_wPerturbations_dz(z) for z in self.z])
        ew2 = np.array([self.cosmology.d2_wPerturbations_dz2(z) for z in self.z])

        quantities = {
            # epsilon = (1+z) d ln H/dz and its first two z-derivatives (spec 01 R15); these are
            # what WKB_Tk.Tk_omegaEff_sq / Tk_d_ln_omegaEff_dz consume
            "epsilon": (opz * d1, opz * e1),
            "d_epsilon_dz": (d1 + opz * d2, e1 + opz * e2),
            "d2_epsilon_dz2": (2.0 * d2 + opz * d3, 2.0 * e2 + opz * e3),
            "d_wPerturbations_dz": (w1, ew1),
            "d2_wPerturbations_dz2": (w2, ew2),
        }
        return {
            name: (np.abs((got - want) / want), np.abs(got - want))
            for name, (got, want) in quantities.items()
        }

    def test_low_z_end_has_no_bias(self):
        """
        The audit's finding A7 is a bias at the *low-z* end of the grid, where all five quantities
        are O(1) and the relative error is meaningful. Require the end-point relative error to be
        no worse than END_TO_INTERIOR_TOLERANCE times the interior median.

        Measured before the fix (relative error at z=0.1 / interior median):
            epsilon 1.3e-05/6.8e-13, eps' 9.4e-04/1.4e-06, eps'' 3.0e-01/6.3e-05,
            w' 1.7e-06/1.0e-09, w'' 3.8e-01/1.8e-08  -- ratios up to 2e+07.
        """
        errors = self._spline_branch_errors()

        for name, (rel, _) in errors.items():
            interior = np.median(rel[5:-5])
            with self.subTest(quantity=name):
                self.assertLessEqual(
                    rel[-1],
                    END_TO_INTERIOR_TOLERANCE * interior,
                    msg=f"{name}: z=0.1 end relative error {rel[-1]:.3e} exceeds "
                    f"{END_TO_INTERIOR_TOLERANCE}x the interior median {interior:.3e}",
                )

    def test_low_z_end_absolute_accuracy(self):
        """
        Absolute guard rails on the low-z end, so a regression is caught even if the ratio test
        above drifts (at this level the residual is float64 round-off in the differentiated
        spline, and the ratio statistic moves by a factor of a few with the exact grid). The
        thresholds are ~30x the values measured when this test was written, and 3-7 orders of
        magnitude below the pre-fix values quoted in test_low_z_end_has_no_bias.
        """
        errors = self._spline_branch_errors()

        thresholds = {
            "epsilon": 2.0e-11,
            "d_epsilon_dz": 5.0e-09,
            "d2_epsilon_dz2": 2.0e-06,
            "d_wPerturbations_dz": 5.0e-13,
            "d2_wPerturbations_dz2": 5.0e-07,
        }
        for name, limit in thresholds.items():
            rel, _ = errors[name]
            with self.subTest(quantity=name):
                self.assertLess(
                    rel[-1],
                    limit,
                    msg=f"{name}: z=0.1 end relative error {rel[-1]:.3e} exceeds {limit:.1e}",
                )

    def test_high_z_end_absolute_accuracy(self):
        """
        At the z=1e12 end the background is radiation dominated: epsilon -> 2 exactly and every
        derivative of it tends to zero, so the *relative* error there is meaningless (audit TK-5
        footnote) and only the absolute error matters. The thresholds are ~30x the values measured
        when this test was written; the pre-fix code gave 3.5e-18 for eps' and 4.5e-28 for eps''.
        """
        errors = self._spline_branch_errors()

        thresholds = {
            "epsilon": 5.0e-11,
            "d_epsilon_dz": 5.0e-21,
            "d2_epsilon_dz2": 5.0e-30,
            "d_wPerturbations_dz": 5.0e-25,
            "d2_wPerturbations_dz2": 5.0e-35,
        }
        for name, limit in thresholds.items():
            _, absolute = errors[name]
            with self.subTest(quantity=name):
                self.assertLess(
                    absolute[0],
                    limit,
                    msg=f"{name}: z=1e12 end absolute error {absolute[0]:.3e} exceeds {limit:.1e}",
                )


if __name__ == "__main__":
    unittest.main()
