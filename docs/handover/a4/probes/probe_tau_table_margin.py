"""How far below its bottom node _ProductionPhaseGk's tau table accepts an abscissa, in ulps of
u = log(1+z), at together 1e4 (raised in the run) and together 1e8 (no near-duplicate node)."""
import sys
import numpy as np
from math import log1p, expm1
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.tests.test_quadsource_integral import SHAPES
for shape_name, x in (("together", 1e4), ("together", 1e8), ("q-smooth", 980.0)):
    base = [s for s in SHAPES if s.name == shape_name][0]
    case, built = C.build_case(base, x, "P")
    table = case.GkPolicy.functions.phase._leading._table
    asc = table._u_asc
    print(f"{shape_name} {x:g}: asc[0]={asc[0]!r} asc[1]={asc[1]!r} first interval in u={asc[1]-asc[0]:.3e} "
          f"(ulp(u)={np.spacing(asc[0]):.3e})")
    u = asc[0]
    for k in range(1, 6):
        u = np.nextafter(u, -np.inf)
        try:
            table.end_partial(expm1(u)); v = "accepted"
        except RuntimeError as exc:
            v = "REFUSED"
        print(f"   u = asc[0] - {k} ulp: z={expm1(u)!r}: {v}")
