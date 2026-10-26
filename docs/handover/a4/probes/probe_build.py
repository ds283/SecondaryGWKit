import sys, time, json
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.tests.test_quadsource_integral import SHAPES
base = [s for s in SHAPES if s.name == "together"][0]
for arm in C.ARMS:
    t = time.perf_counter()
    case, built = C.build_case(base, 980.0, arm)
    obj = C.phase_objects(case, built)
    print(arm, f"{time.perf_counter()-t:.2f}s", C.construction_problems(arm, obj))
    print(json.dumps(obj, default=repr)[:900])
