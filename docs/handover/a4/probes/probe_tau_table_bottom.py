"""Why _ProductionPhaseGk's tau table refuses an abscissa at z_response in the raised P/P0 cells.
Builds objects only; runs no integral."""
import sys
from math import log, log1p, expm1, ulp
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.tests.test_quadsource_integral import SHAPES

cells = [("together", 980.0), ("T-first", 980.0), ("q-smooth", 980.0),
         ("together", 1e4), ("T-first", 1e4), ("q-smooth", 1e4),
         ("together", 1e5), ("together", 1e6), ("together", 1e7), ("together", 1e8),
         ("T-first", 1e5), ("T-first", 1e6), ("T-first", 1e7), ("q-smooth", 1e5), ("q-smooth", 1e6)]
for shape_name, x in cells:
    base = [s for s in SHAPES if s.name == shape_name][0]
    case, built = C.build_case(base, x, "P")
    phase = case.GkPolicy.functions.phase
    table = phase._leading._table
    zs = sorted(table._z, reverse=True)
    z_resp = case.z_resp
    grid_min = built["G_grid"]["z_min"]
    # the two lowest table nodes, and the abscissa the Levin driver forms at the lower end
    lo1, lo2 = zs[-1], zs[-2]
    u_end = log(1.0 + z_resp)          # the x_span end QuadSourceIntegral forms
    z_back = expm1(u_end)              # what PrimitivePhase evaluates at
    try:
        table.end_partial(z_back)
        verdict = "accepted"
    except RuntimeError as exc:
        verdict = f"REFUSED ({exc})"
    print(f"{shape_name:9s} {x:8.0e}: z_resp={z_resp!r} grid_min={grid_min!r} "
          f"grid_min-z_resp={grid_min - z_resp:+.3e} ({(grid_min - z_resp)/ulp(z_resp):+.0f} ulp); "
          f"table nodes={len(zs)}, two lowest {lo2!r}, {lo1!r} (gap {lo2-lo1:.3e}); "
          f"anchor {phase.z_anchor!r}; expm1(log(1+z_resp))={z_back!r} "
          f"({(z_back - z_resp)/ulp(z_resp):+.0f} ulp of z_resp): {verdict}", flush=True)
