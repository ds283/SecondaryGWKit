"""
Tests for ComputeTargets/GkSourcePolicyData.py.

Offline: no Ray runtime, no datastore. `_classify_Levin` is a pure function of a GkSource-like
object, a policy and the crossover classification, so it is driven directly through duck-typed
stand-ins shaped like the attributes it reads (`source.values[i].z_source`/`.has_WKB`/`.WKB`,
`source.z_sample`, `source.primary_WKB_largest_z`, `policy.Levin_threshold`, and -- since
GkTk-remedial prompt 09 -- `source.k`, `source.z_response` and `source.model_proxy`).

Since prompt 09 the threshold test is evaluated on a `ComputeTargets.primitive_phase.PrimitivePhase`
rather than on a `phase_spline` of the stored phase, so the stand-in has to carry a background
model: a wavenumber, a response redshift and a `functions.tau` interval accessor. The phase these
fixtures describe is still exactly linear in `log(1+z)`, `theta = rate * log(1+z)`, so
`|d theta/d log(1+z)| = rate` everywhere and every threshold answer below is unchanged. It stays
exact because the decomposition is exact: `PrimitivePhase` evaluates
`-k tau.delta(z_s, z_r) + phi` with `phi = theta_stored + k tau.delta(z_s, z_r)`, and with the
small `k = 1/Mpc` used here the leading term and the part of `phi` that cancels it are both O(1e-4)
and smooth, so their splined difference recovers `rate` to ~1e-7 -- ten orders below the
thresholds these tests turn on.

Board issue `[10-classify-levin-keyerror]`: `Levin_z` was only assigned inside the
`for z_source` loop, so a Green's function whose |d theta_G / d log(1+z)| never crosses
`policy.Levin_threshold` anywhere in its WKB range left the key absent and made
`apply_GkSource_policy` raise `KeyError: 'Levin_z'`. It is diagnostic-only since prompt 10 of
prompts/source-remediation, so `None` -- "no Levin quadrature indicated" -- is the right answer,
and it is what the function's own early-return path already reported for the cases it covers.
"""

import unittest
from math import log, exp, pi

import numpy as np

from ComputeTargets.BackgroundModel import TablePrimitive
from ComputeTargets.GkSourcePolicyData import GK_PHASE_SIGN, _classify_Levin
from ComputeTargets.cumulative_table import CumulativeTable

# a small wavenumber, so that the leading term of the decomposition is O(1e-4) on this geometry
# and the fixtures' phase stays what it says it is (see the module docstring)
FAKE_K = 1.0
TAU_GAUSS_ORDER = 4


def _Hubble(z: float) -> float:
    """Exact radiation, `H = (1+z)^2`, so `tau = 1/(1+z)`."""
    return (1.0 + z) * (1.0 + z)


class FakeModelFunctions:
    """The two fields the phase construction reads off a `ModelFunctions`."""

    def __init__(self, z_nodes):
        self.Hubble = _Hubble
        self.tau = TablePrimitive(
            CumulativeTable(
                z_nodes, lambda z: 1.0 / _Hubble(z), TAU_GAUSS_ORDER, label="tau"
            ),
            "tau",
        )


class FakeModel:
    def __init__(self, z_nodes):
        self.functions = FakeModelFunctions(z_nodes)


class FakeModelProxy:
    def __init__(self, model):
        self._model = model

    def get(self):
        return self._model


class FakeWavenumber:
    def __init__(self, k: float):
        self.k = float(k)
        self.k_inv_Mpc = float(k)
        self.store_id = 1


class FakeZ:
    def __init__(self, z: float, store_id: int):
        self.z = z
        self.store_id = store_id

    def __float__(self):
        return float(self.z)


class FakeWKB:
    """The two phase fields `_classify_Levin` reads, as TkWKBValue/GkWKBValue store them."""

    def __init__(self, theta: float):
        div, mod = divmod(theta, 2.0 * pi)
        self.theta_div_2pi = int(div)
        self.theta_mod_2pi = mod


class FakeValue:
    def __init__(self, z: float, store_id: int, theta: float):
        self.z_source = FakeZ(z, store_id)
        self.has_WKB = True
        self.WKB = FakeWKB(theta)


class FakeZSample:
    def __init__(self, z_values):
        self._z = sorted(z_values, reverse=True)
        self.max = FakeZ(self._z[0], 0)
        self.min = FakeZ(self._z[-1], len(self._z) - 1)

    def __len__(self):
        return len(self._z)

    def __iter__(self):
        return (FakeZ(z, i) for i, z in enumerate(self._z))


class FakeGkSource:
    """
    A "mixed"/"WKB" GkSource carrying a linear phase in log(1+z): theta = rate * log(1+z), so
    |d theta / d log(1+z)| = rate everywhere and the threshold test has an exactly known answer.
    """

    def __init__(self, z_top: float, z_bottom: float, rate: float, n: int = 40):
        log_top, log_bottom = log(1.0 + z_top), log(1.0 + z_bottom)
        logs = [log_bottom + (log_top - log_bottom) * i / (n - 1) for i in range(n)]
        zs = [exp(u) - 1.0 for u in logs]

        # prompts/levin-phase-floor prompt 06: _build_phase unwraps phi by whole cycles between
        # neighbouring samples, which presumes that the samples resolve phi -- that it moves by
        # less than pi from one to the next, as a producer's stored residual does. This phase
        # lives entirely in phi, so the WKB values are sampled densely enough for that, at most
        # pi/2 per step, on a grid that contains the n nodes above; the background table stays
        # on those n nodes
        dense = max(n, int(np.ceil(rate * (log_top - log_bottom) / (0.5 * pi))) + 1)
        per_node = int(np.ceil((dense - 1) / (n - 1)))
        sample_zs = [zs[0]]
        for i in range(n - 1):
            for j in range(1, per_node):
                u = logs[i] + (logs[i + 1] - logs[i]) * j / per_node
                sample_zs.append(exp(u) - 1.0)
            sample_zs.append(zs[i + 1])

        self.rate = rate
        self.values = [
            FakeValue(z, i, rate * log(1.0 + z)) for i, z in enumerate(sample_zs)
        ]
        self.z_sample = FakeZSample(sample_zs)
        self.primary_WKB_largest_z = FakeZ(z_top, 0)

        # prompt 09: the phase is now built from the background model's conformal-time table,
        # anchored at the response redshift. The n grid redshifts are the table's nodes, as in
        # production (the background model is built on the source grid, main.py:476), and the
        # response point is the bottom of the range. (Since prompt 06 of levin-phase-floor the
        # denser WKB samples between them, where there are any, are off the table's grid.)
        self.k = FakeWavenumber(FAKE_K)
        self.z_response = FakeZ(zs[0], 0)
        self.model_proxy = FakeModelProxy(
            FakeModel(np.array(sorted(zs, reverse=True), dtype=float))
        )

        # prompts/levin-phase-floor prompt 06: _build_phase builds phi from the producer's stored
        # residual. This closed-form phase has no separate small terms, so the residual is formed
        # in double, theta - GK_PHASE_SIGN k tau.delta(z, z_response) (prompt 06 §2). The values
        # are pure WKB (z_init null) and carry no wrap for the rectifier to repair
        tau = self.model_proxy.get().functions.tau
        for value in self.values:
            theta = rate * log(1.0 + value.z_source.z)
            value.WKB.theta_residual = theta - GK_PHASE_SIGN * FAKE_K * tau.delta(
                value.z_source.z, self.z_response.z
            )
            value.WKB.raw_theta_div_2pi = value.WKB.theta_div_2pi
            value.WKB.z_init = None


class FakePolicy:
    def __init__(self, Levin_threshold: float):
        self.Levin_threshold = Levin_threshold


MIXED = {"type": "mixed", "quality": "complete", "crossover_z": None}


class TestClassifyLevin(unittest.TestCase):
    def test_no_threshold_crossing_reports_None_rather_than_raising(self):
        """
        `[10-classify-levin-keyerror]`. The phase turns over at 50 rad per unit log(1+z) and the
        policy threshold is 1e4, so the loop runs to exhaustion without assigning Levin_z. Before
        the fix this returned a payload with no "Levin_z" key and apply_GkSource_policy raised
        KeyError on it.
        """
        source = FakeGkSource(1.0e8, 1.0e4, rate=50.0)
        payload = _classify_Levin(source, FakePolicy(1.0e4), MIXED)
        print(
            f"\n[10-classify-levin-keyerror] d theta/d log(1+z) = {source.rate:g} everywhere, "
            f"Levin_threshold = 1e4: payload keys {sorted(payload.keys())}, "
            f"Levin_z = {payload['Levin_z']}"
        )
        self.assertIn("Levin_z", payload)
        self.assertIsNone(payload["Levin_z"])
        self.assertIn("metadata", payload)
        self.assertNotIn("Levin_z_dtheta_dlogz", payload["metadata"])

    def test_a_threshold_crossing_still_reports_the_redshift(self):
        """The positive path is unchanged: the first sampled redshift above the threshold wins."""
        source = FakeGkSource(1.0e8, 1.0e4, rate=5.0e3)
        payload = _classify_Levin(source, FakePolicy(1.0e2), MIXED)
        print(
            f"[10-classify-levin-keyerror] d theta/d log(1+z) = {source.rate:g}, "
            f"Levin_threshold = 1e2: Levin_z = {payload['Levin_z'].z:.5g}, "
            f"recorded dtheta/dlogz = {payload['metadata']['Levin_z_dtheta_dlogz']:.5g}"
        )
        self.assertIsNotNone(payload["Levin_z"])
        # descending z_sample, so the crossing is reported at the top of the WKB range
        self.assertAlmostEqual(payload["Levin_z"].z, 1.0e8, delta=1.0)
        self.assertGreater(abs(payload["metadata"]["Levin_z_dtheta_dlogz"]), 1.0e2)
        # prompt 09: the closed-form leading term plus the splined residual recovers the
        # fixture's exactly linear phase, so the recorded derivative is still `rate`
        self.assertAlmostEqual(
            abs(payload["metadata"]["Levin_z_dtheta_dlogz"]) / source.rate,
            1.0,
            places=9,
        )

    def test_the_early_return_path_still_reports_None(self):
        """
        A "numeric" or "fail" source, or an "incomplete" one, has no WKB phase to test and
        returns before the spline is built. That path always supplied None; it is asserted here so
        that the two ways of reporting "no Levin_z" cannot drift apart.
        """
        source = FakeGkSource(1.0e8, 1.0e4, rate=5.0e3)
        for data in (
            {"type": "numeric", "quality": "complete", "crossover_z": None},
            {"type": "fail", "quality": "incomplete", "crossover_z": None},
            {"type": "mixed", "quality": "incomplete", "crossover_z": None},
        ):
            with self.subTest(type=data["type"], quality=data["quality"]):
                payload = _classify_Levin(source, FakePolicy(1.0e2), data)
                self.assertIn("Levin_z", payload)
                self.assertIsNone(payload["Levin_z"])


if __name__ == "__main__":
    unittest.main()
