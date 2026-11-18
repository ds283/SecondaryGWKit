"""Orchestrator's own index of 07's census checkpoint, 07a Part C stop 3. Read only but for OUT_JSON.
usage: python -I -B index07.py CHECKPOINT OUT_JSON
Maps row@rtol -> [offset, length, line sha256, unit]; prints the file's SHA-256 and refuses duplicates."""
import hashlib, json, sys
ck, out = sys.argv[1:3]
h = hashlib.sha256(); idx = {}; off = 0
with open(ck, "rb") as f:
    for raw in f:
        h.update(raw)
        d = json.loads(raw)["data"]
        k = d["row"] + "@" + d["rtol"]
        assert k not in idx, k
        idx[k] = [off, len(raw), hashlib.sha256(raw).hexdigest(), d["unit"]]
        off += len(raw)
print("script", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
print("checkpoint", ck, "sha256", h.hexdigest(), "lines", len(idx))
json.dump({"checkpoint": ck, "sha256": h.hexdigest(), "units": idx}, open(out, "w"))
