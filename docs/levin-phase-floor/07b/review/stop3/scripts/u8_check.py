"""Orchestrator's own U8 check, levin-phase-floor 07b Part C stop 3. Read only; prints its result.
usage: python -I -B u8_check.py RUN07B INDEX07B WANTED_JSON REPLAYS_H_DIR
Each id of WANTED's `wanted` with which == "h" (every id but the 07-code one) names (unit, kind, index); the census
call is read from the checkpoint at my index's offset (checked by SHA-256) and compared with REPLAYS_H_DIR/<id>.json's
`replay` in value and abserr (float hex), converged, num_regions and max_depth."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

run, idxf, wantedf, repdir = sys.argv[1:5]
idx = json.load(open(idxf))
byunit = {}
for k, e in idx["units"].items():
    assert e[3] not in byunit, e[3]
    byunit[e[3]] = e
W = json.load(open(wantedf))["wanted"]
print("script", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print("wanted", wantedf, hashlib.sha256(Path(wantedf).read_bytes()).hexdigest(), "ids", len(W))
print("excluded (which != h):", sorted((k, v["which"], v["unit"]) for k, v in W.items() if v["which"] != "h"))
res = Counter(); fails = []
f = open(Path(run) / "checkpoint.jsonl", "rb")
for sid, w in sorted(W.items()):
    if w["which"] != "h":
        continue
    e = byunit[w["unit"]]
    f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    d = json.loads(raw)["data"]
    assert d["unit"] == w["unit"] and d["status"] == "done", (sid, d["status"])
    lst = d["calls"] if w["kind"] == "group" else d["other_calls"]
    m = [c for c in lst if c["index"] == w["index"]]
    assert len(m) == 1, (sid, len(m))
    c = m[0]
    rp = json.load(open(Path(repdir) / f"{sid}.json"))["replay"]
    got = {"value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "converged": c["converged"],
           "num_regions": c["num_regions"], "max_depth": c["max_depth"]}
    want = {k: rp[k] for k in got}
    ok = got == want
    res[(w["unit"].split("-")[1], w["kind"], ok)] += 1
    res["edge>0"] += c["num_edge_rule_refusals"] > 0
    res["range>0"] += c["num_range_rule_refusals"] > 0
    if not ok:
        fails.append((sid, w, {k: (got[k], want[k]) for k in got if got[k] != want[k]}))
print("results:", {str(k): v for k, v in sorted(res.items(), key=str)})
print("compared", sum(v for k, v in res.items() if isinstance(k, tuple)), "fails", len(fails))
for x in fails:
    print("FAIL", x)
