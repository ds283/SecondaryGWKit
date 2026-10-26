"""Does a4_cells.below_anchor_check catch run 1's construction? Builds the run-2 P case, then swaps
in G's phase built as run 1 built it (on Case's grid with its last node one ulp below z_resp)."""
import sys
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.tests.test_quadsource_integral import SHAPES, w_of_b
from ComputeTargets.tests.test_tk_source_functions import log_grid, PRODUCTION_SAMPLES_PER_LOG10Z
for shape_name, x in (("together", 1e4), ("q-smooth", 980.0), ("together", 980.0), ("together", 1e8)):
    base = [s for s in SHAPES if s.name == shape_name][0]
    for arm in ("P", "P0"):
        case, built = C.build_case(base, x, arm)
        new = C.below_anchor_check(case)
        grid = log_grid(1.0 + case.G_crossover_z, 1.0 + case.z_resp, PRODUCTION_SAMPLES_PER_LOG10Z)
        old_gk, _ = C._production_phase_gk(case.shape.k, w_of_b(0.0), case.z_resp, case.model, grid)
        case.GkPolicy.functions = case.GkPolicy.functions._replace(
            phase=old_gk.phase if arm == "P" else old_gk.old_phase)
        old = C.below_anchor_check(case)
        lows = [f"{s['regime']}:-{s['lowest']['below_a_ulp']:.0f}ulp@{s['lowest']['where']}" for s in new["subintervals"]]
        oldf = [f for s in old["subintervals"] for f in s["failures"]["G"]]
        print(f"{shape_name:9s} {x:6g} {arm:2s}: run-2 grid G_ok={new['G_ok']} T_ok={new['T_ok']} lowest {lows}; "
              f"run-1 grid G_ok={old['G_ok']} ({len(oldf)} failing points; first: {oldf[:1]})", flush=True)
