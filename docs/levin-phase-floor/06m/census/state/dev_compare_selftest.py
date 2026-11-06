"""levin-phase-floor 06m census agent, development only (not scored): exercise compare_06m.one_unit on three of route
B's census units compared with themselves (route B's line and pickle on both sides), read only. Expected: every unit
EQUAL on the compared keys, the matching holds, and the module evidence false (route B ran a variant)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06m/census/scripts")
import compare_06m as C

idx = json.loads(C.INDEX.read_text())["units"]
for uid in ("g-1e-7-0-37232", "g-1e-8-3-1264", "g-1e-7-2-21980"):
    ent = idx[uid]
    with open(C.RUN06L_B / "checkpoint.jsonl", "rb") as f:
        f.seek(ent["offset"])
        line = f.read(ent["length"])
    row = C.one_unit((uid, line, str(C.RUN06L_B), ent))
    print(uid, "ok", row["ok"], "differing", row["differing"][:5], "calls", row["n_calls"], "group", row["n_group_calls"],
          "regions", row["n_regions"], "returned leaves", row["n_regions_returned"], "B fired calls", row["n_calls_B_fired"],
          "firings", row["n_firings_B"])
    print("  evidence", row["evidence"])
    print("  excl differing", row["excl_differing"])
    print("  excl present", row["excl_present"])
