"""levin-phase-floor 06e: the run's records after it ended (new in 06e; 06c did the same inline in its
report). Per record: the registry's script stamp, the tree's HEAD and status, the scripts' SHA-256
against scripts/ now, status, wall_s > 300 s, foreign modules, deep-line and CC-term errors.
Usage: python -B checks_after_06e.py RUN_DIR"""
import collections, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import G_SHA, sha256  # noqa: E402
run = Path(sys.argv[1])
now = {p.name: sha256(p) for p in HERE.glob("*.py")}
c = collections.Counter(); over300 = []; bad = []
for line in open(run / "checkpoint.jsonl"):
    r = json.loads(line); d = r["data"]
    c["records"] += 1; c["status " + d["status"]] += 1
    c["script_sha " + str(r.get("script_sha256", r.get("script_hash", "?")))[:8]] += 1
    if d["tree_head"] != G_SHA or d["tree_status_short"].strip(): bad.append((r["unit"], "tree"))
    if any(now.get(k) != v for k, v in d["scripts_sha256"].items()): bad.append((r["unit"], "scripts"))
    if d["modules"]["foreign"]: bad.append((r["unit"], "foreign modules"))
    if d["wall_s"] is not None and d["wall_s"] > 300: over300.append((r["unit"], round(d["wall_s"], 1)))
    dl = d.get("deep_lines_06e") or {}
    c["deep06e lines"] += dl.get("n_calls", 0); c["deep06e errors"] += dl.get("n_errors", 0); c["deep06e not reproducing"] += dl.get("n_not_reproducing", 0)
    dc = d.get("deep_lines") or {}
    c["deep(cap) lines"] += dc.get("n_calls", 0)
    for cl in d["calls"] + d["other_calls"]:
        f = cl.get("cc_floor")
        if f is None: c["cc_floor None"] += 1; continue
        if "error" in f: c["cc_floor error"] += 1; continue
        c["cc leaves"] += f["n_cc_leaves"]; c["cc reproduced"] += f["n_reproduced"]; c["cc errors"] += f["n_errors"]
print("record keys seen:", sorted(k for k in json.loads(open(run / "checkpoint.jsonl").readline()).keys()))
for k, v in sorted(c.items()): print(f"{k}: {v}")
print("units with wall_s > 300 s:", len(over300), over300)
print("records failing tree/scripts/modules checks:", len(bad), bad[:20])
