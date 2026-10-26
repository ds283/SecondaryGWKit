"""Is z_init a node of the union of a dropped _TableCsTauFixture's numeric and WKB nodes?"""
import sys
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.tests.test_quadsource_integral import SHAPES, w_of_b
from ComputeTargets.tests.test_g_phase_reanchor import _TableCsTauFixture
w = w_of_b(0.0)
for shape_name, rungs in C.LADDER.items():
    base = [s for s in SHAPES if s.name == shape_name][0]
    for x in rungs:
        shape = C.scaled_shape(base, x)
        for label, kk in (("q", shape.q), ("r", shape.r)):
            f = _TableCsTauFixture(w, k=kk, x_max=C.fixture_x_max(x, kk, shape), drop_first_WKB_sample=True)
            print(f"{shape_name:9s} {x:8.0e} T{label}: z_init={f.crossover_z!r} z_numeric[-1]={f.z_numeric[-1]!r} "
                  f"equal={f.z_numeric[-1] == f.crossover_z} z_numeric_below[0]={f.z_numeric_below[0]!r} "
                  f"z_WKB[0]={f.z_WKB[0]!r} below==WKB0={f.z_numeric_below[0] == f.z_WKB[0]} "
                  f"init in union={f.crossover_z in set(f.z_numeric) | set(f.z_numeric_below) | set(f.z_WKB)}")
