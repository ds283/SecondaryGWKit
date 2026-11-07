"""
The stationary-point guard's counts in QuadSourceIntegral's records of its Levin calls
(prompts/levin-phase-floor/07-stationary-point-guard.md, section 1.6 and section 3 test 11).

Each record QuadSourceIntegral keeps of a Levin call gains four counts from the integrator: stationary_points
(the number located), window_pieces, guarded_regions and guard_at_depth_max. This test runs one group call
through phase_group_Levin_integral on test_quadsource_integral.py's exact-ingredient stand-in (its Case, as
that module's seam test runs it) and pins the group record's key set; and it checks that the four counts equal
the driver's own for each group.

No Ray and no datastore.
"""

import unittest
from math import sqrt

from ComputeTargets.QuadSourceIntegral import phase_group_Levin_integral
from ComputeTargets.tests.test_quadsource_integral import SHAPES, Case, FakeZ, z_at_tau

GROUP_RECORD_KEYS = {
    "label",
    "value",
    "abserr",
    "converged",
    "phase_limited",
    "regions",
    "simple_regions",
    "evaluations",
    "elapsed",
    # prompt 07 section 1.6
    "stationary_points",
    "window_pieces",
    "guarded_regions",
    "guard_at_depth_max",
}


class TestGroupRecordCounts(unittest.TestCase):
    def test_11_group_record_carries_the_four_counts(self):
        case = Case(0.0, SHAPES[0], 980.0, exact=True)
        Tq_f = case.Tk_builder(case.model, case.q.k, *case.Tq_inputs)
        Tr_f = case.Tk_builder(case.model, case.r.k, *case.Tr_inputs)
        cs = sqrt(case.w)
        z_hi = z_at_tau(case.model, 40.0 / (case.shape.r * cs))
        z_lo = case.z_resp
        out = phase_group_Levin_integral(
            case.model,
            case.k.k,
            case.q.k,
            case.r.k,
            (True, True, True),
            case.GkPolicy.functions,
            Tq_f,
            Tr_f,
            FakeZ(case.z_resp),
            max_z=z_hi,
            min_z=z_lo,
            atol=1e-25,
            rtol=1e-10,
        )
        self.assertGreater(len(out["groups"]), 0)
        for record, data in zip(out["groups"], out["Levin_results"]):
            with self.subTest(group=record["label"]):
                self.assertEqual(set(record), GROUP_RECORD_KEYS)
                self.assertEqual(
                    record["stationary_points"], len(data["stationary_points"])
                )
                self.assertEqual(record["window_pieces"], data["num_window_pieces"])
                self.assertEqual(record["guarded_regions"], data["num_guarded_regions"])
                self.assertEqual(
                    record["guard_at_depth_max"], data["num_guard_at_depth_max"]
                )


if __name__ == "__main__":
    unittest.main()
