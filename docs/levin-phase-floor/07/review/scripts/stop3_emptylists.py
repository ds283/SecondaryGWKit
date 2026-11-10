"""Stop 3, last attribution (read only): per-call lists that went empty <-> non-empty between 06m and the census
(so stop3_compare could not re-key them): data deep_lines.lines, data deep_lines_06e.lines, pickle w3, pickle
deep_lines_06e, pickle deep_lines. In every unit with an ACTIVE call, every entry's index on either side must be an
ACTIVE call, or, if quiet, must be present and equal on both sides.
Usage: python -B stop3_emptylists.py RUN_DIR STOP3_JSON"""
import hashlib, json, pickle, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stop3_compare as S
run_dir, s3 = sys.argv[1:3]
rows = {r["unit"]: r for r in json.load(open(s3))["rows"]}
idx = json.load(open(S.INDEX))["units"]
tot, bad = Counter(), []
for line in open(Path(run_dir) / "checkpoint.jsonl"):
    d = json.loads(line)["data"]; r = rows[d["unit"]]
    if not r["n_active"]:
        continue
    act = set(r["active"])
    e = idx[d["unit"]]
    g = json.loads(S.load_bytes(S.RUN06M / "checkpoint.jsonl", e["offset"], e["length"], e["line_sha256"]))["data"]
    p6 = pickle.loads(S.load_bytes(S.RUN06M / g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]))
    pm = pickle.loads(S.load_bytes(Path(run_dir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]))
    lists = {
        "data.deep_lines.lines": (d["deep_lines"]["lines"], g["deep_lines"]["lines"]),
        "data.deep_lines_06e.lines": (d["deep_lines_06e"]["lines"], g["deep_lines_06e"]["lines"]),
        "pickle.w3": (pm.get("w3") or [], p6.get("w3") or []),
        "pickle.deep_lines_06e": (pm.get("deep_lines_06e") or [], p6.get("deep_lines_06e") or []),
        "pickle.deep_lines": (pm.get("deep_lines") or [], p6.get("deep_lines") or []),
    }
    for name, (a, b) in lists.items():
        if bool(a) == bool(b):
            continue
        tot[name + " empty<->nonempty units"] += 1
        A, B = S.rekey_list(a, lambda x: x["index"]), S.rekey_list(b, lambda x: x["index"])
        for i in set(A) | set(B):
            if i in act:
                tot[name + " entries of active calls"] += 1
            elif S.diff(A.get(i), B.get(i)):
                bad.append((d["unit"], name, i))
            else:
                tot[name + " quiet entries equal"] += 1
print("# stop3_emptylists.py sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print(json.dumps(dict(tot), indent=1)); print("bad", len(bad), bad[:20])
sys.exit(1 if bad else 0)
