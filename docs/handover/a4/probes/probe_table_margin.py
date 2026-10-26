"""How far above the dropped table's top node z_init sits, against the table's own top interval."""
import sys
from math import log1p, ulp
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
            u_init = log1p(f.crossover_z); u0 = log1p(f.z_WKB[0]); u1 = log1p(f.z_WKB[1])
            excess = (u_init - u0) - (u0 - u1)
            print(f"{shape_name:9s} {x:8.0e} T{label}: z_init={f.crossover_z:.6g} top node={f.z_WKB[0]:.6g}; "
                  f"(u_init-u_top) - (u_top-u_next) = {excess:+.3e} = {excess/ulp(u_init):+.1f} ulp(u_init) -> "
                  f"{'REFUSED' if u_init > u0 + (u0 - u1) else 'accepted'}")
