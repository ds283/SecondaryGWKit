"""levin-phase-floor 06g, descriptive: which lost calls (out/v8_calls.json, outside the 720) return no leaf in
v8_slow_leaves.py's scope, and what they return instead (counts from the calls' recorded fields). Read only.
Usage: python -B v8_slow_leaves_coverage.py"""
import json
from pathlib import Path
W = Path(__file__).resolve().parent.parent
d = json.loads((W / "out" / "v8_calls.json").read_text())
L = json.loads((W / "out" / "v8_slow_leaves.json").read_text())
have = {(e["rtol"], e["unit"], e["index"]) for e in L}
for rt in ("1e-7", "1e-8"):
    lost = [x for x in d["lost"] if x["rtol"] == rt and not x["stationary_point_row"]]
    no = [x for x in lost if (rt, x["unit"], x["g"]["index"]) not in have]
    print(rt, "lost calls", len(lost), "with a scoped leaf", len(lost) - len(no), "without", len(no),
          "; of those without: node_limited", sum(1 for x in no if x["g"]["node_limited"]),
          ", untrusted returned >= 1", sum(1 for x in no if (x["g"]["num_untrusted_regions"] or 0) >= 1),
          ", n_b_untrusted_returned == 0 (their (b) leaves were all bisected or exempted floor-limited)",
          sum(1 for x in no if x["g"]["checked_06g"]["n_b_untrusted_returned"] == 0))
