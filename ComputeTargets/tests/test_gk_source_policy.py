"""
Tests for ComputeTargets/GkSourcePolicyData.py.

Offline: no Ray runtime, no datastore. `_classify_Levin` is a pure function of a GkSource-like
object, a policy and the crossover classification, so it is driven directly through duck-typed
stand-ins shaped like the attributes it reads (`source.values[i].z_source`/`.has_WKB`/`.WKB`,
`source.z_sample`, `source.primary_WKB_largest_z`, `policy.Levin_threshold`).

Board issue `[10-classify-levin-keyerror]`: `Levin_z` was only assigned inside the
`for z_source` loop, so a Green's function whose |d theta_G / d log(1+z)| never crosses
`policy.Levin_threshold` anywhere in its WKB range left the key absent and made
`apply_GkSource_policy` raise `KeyError: 'Levin_z'`. It is diagnostic-only since prompt 10 of
prompts/source-remediation, so `None` -- "no Levin quadrature indicated" -- is the right answer,
and it is what the function's own early-return path already reported for the cases it covers.
"""

import unittest
from math import log, exp, pi

from ComputeTargets.GkSourcePolicyData import _classify_Levin


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
        self.rate = rate
        self.values = [FakeValue(z, i, rate * log(1.0 + z)) for i, z in enumerate(zs)]
        self.z_sample = FakeZSample(zs)
        self.primary_WKB_largest_z = FakeZ(z_top, 0)


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
