# Orchestrator's own reader: each K0 unit's region pickle against 06g's census region pickle,
# every leaf compared (arrays by bytes), ignoring only timing keys (seconds, elapsed, compute_time)
# and the new key route_06k. Identity: all 100 units; routes: units with no firing.
import json, pickle, numpy as np
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/"
C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06g-census-20261101T114315/"
uids = [u["uid"] for u in json.load(open(V + "out/k0/units.json"))["units"]]
ref = {}
for line in open(C + "checkpoint.jsonl"):
    r = json.loads(line)
    if r["unit"] in set(uids):
        ref[r["unit"]] = r["data"]
IGN = {"seconds", "elapsed", "compute_time", "route_06k"}
def load(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return pickle.loads(f.read(n))
def walk(x, y, p, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k in IGN:
                continue
            walk(x.get(k), y.get(k), p + "." + str(k), out)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append((p, "len"))
            return
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
for mode in ["identity", "route1", "route2", "route3"]:
    recs = {}
    for line in open(V + f"state/k0/{mode}/units.jsonl"):
        r = json.loads(line)
        recs[r["unit"]] = r["data"]
    n = n_eq = 0
    for u in uids:
        d, g = recs[u], ref[u]
        fired = any((c.get("route_06k") or {}).get("n_firings", 0) for k in ("calls", "other_calls") for c in d[k])
        if fired:
            continue
        sub = [s for s in __import__("os").listdir(V + f"state/k0/{mode}") if (V + f"state/k0/{mode}/{s}/{d['pickle']}") and __import__("os").path.exists(V + f"state/k0/{mode}/{s}/{d['pickle']}")]
        assert len(sub) == 1, (mode, u, sub)
        a = load(V + f"state/k0/{mode}/{sub[0]}/{d['pickle']}", d["pickle_offset"], d["pickle_length"])
        b = load(C + g["pickle"], g["pickle_offset"], g["pickle_length"])
        out = []
        walk(a, b, "", out)
        n += 1
        n_eq += not out
        if out:
            print("  DIFF", mode, u, out[:5])
    print(f"{mode}: units compared {n}, region pickles equal (timing and route_06k aside) {n_eq}")
