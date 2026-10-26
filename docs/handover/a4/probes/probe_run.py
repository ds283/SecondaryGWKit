import sys, time, json
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
arm = sys.argv[1]
rec = C.run_cell("together", 980.0, arm)
g = rec.pop("groups")
print(json.dumps({k: v for k, v in rec.items() if k not in ("handover",)}, default=repr, indent=1))
for x in g:
    print(json.dumps({k: x[k] for k in ("subinterval","regime","label","num_regions","max_depth","max_region_depth","num_phase_limited_regions","converged","deepest_count","deepest_z","depth_histogram")}))
