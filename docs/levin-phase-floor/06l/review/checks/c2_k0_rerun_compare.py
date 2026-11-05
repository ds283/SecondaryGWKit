# Orchestrator's review §B.3 check 2 (06l; 06k's review c2 script adapted): two K0 units re-run by the orchestrator
# under the identity variant and route B, each compared with 06g's census record. Per call (calls +
# other_calls) every key but "elapsed", "route_06k" and "route_06l" equal as JSON; the unit's region pickle leaf
# by leaf (arrays by bytes), ignoring only timing keys (seconds, elapsed, compute_time) and
# route_06k and route_06l. Route B fired nowhere on these units in K0, so it must equal 06g everywhere here too.
import json, os, pickle, numpy as np
H = os.path.dirname(os.path.abspath(__file__)) + "/k0rerun/"
C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06g-census-20261101T114315/"
uids = [u["uid"] for u in json.load(open(H + "units.json"))["units"]]
ref = {}
for line in open(C + "checkpoint.jsonl"):
    r = json.loads(line)
    if r["unit"] in set(uids):
        ref[r["unit"]] = r["data"]
SKIP = {"elapsed", "route_06k", "route_06l"}
IGN = {"seconds", "elapsed", "compute_time", "route_06k", "route_06l"}
def load(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return pickle.loads(f.read(n))
def walk(x, y, p, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k not in IGN:
                walk(x.get(k), y.get(k), p + "." + str(k), out)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append((p, "len")); return
        for i, (s, t) in enumerate(zip(x, y)):
            walk(s, t, p + f"[{i}]", out)
    elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype
                and x.shape == y.shape and x.tobytes() == y.tobytes()):
            out.append((p, "array"))
    elif isinstance(x, float) and isinstance(y, float):
        if not (x == y or (x != x and y != y)) or (x == y and np.signbit(x) != np.signbit(y)):
            out.append((p, x, y))
    elif type(x) is not type(y) or x != y:
        out.append((p, str(x)[:60], str(y)[:60]))
print("units", uids, "found in 06g checkpoint", len(ref))
for mode in ["identity", "routeB"]:
    recs = {}
    for line in open(H + f"{mode}/units.jsonl"):
        r = json.loads(line); recs[r["unit"]] = r["data"]
    for u in uids:
        d, g = recs[u], ref[u]
        n = bad = fired = 0
        for kind in ("calls", "other_calls"):
            assert len(d[kind]) == len(g[kind]), (mode, u, kind)
            for a, b in zip(d[kind], g[kind]):
                n += 1
                fired += (a.get("route_06l") or {}).get("n_firings", 0)
                ka = {k: v for k, v in a.items() if k not in SKIP}
                kb = {k: v for k, v in b.items() if k not in SKIP}
                bad += json.dumps(ka, sort_keys=True) != json.dumps(kb, sort_keys=True)
        sub = [s for s in os.listdir(H + mode) if os.path.exists(H + f"{mode}/{s}/{d['pickle']}")]
        assert len(sub) == 1, (mode, u, sub)
        out = []
        walk(load(H + f"{mode}/{sub[0]}/{d['pickle']}", d["pickle_offset"], d["pickle_length"]),
             load(C + g["pickle"], g["pickle_offset"], g["pickle_length"]), "", out)
        print(f"{mode} {u}: status {d['status']}; calls {n}, firings {fired}, calls differing {bad}; "
              f"region pickle differences (timing, route_06k aside) {len(out)}")
        for o in out[:5]:
            print("   DIFF", o)
