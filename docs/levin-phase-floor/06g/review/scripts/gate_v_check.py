"""Orchestrator's own check of Gate V parts 1 and 3 from the census agent's raw Gate V records.

Part 1: P1, P2, P3 (parent tree) against 06h's checkpoint records of the same units: result
total, total_abserr, analytic_rad, analytic_06h.h value/abserr/converged, and every call's value,
abserr, num_regions, max_depth, converged, phase_limited, by hex. Every key 06h recorded per call
is compared too, except timing.
Part 3: G1 against G2, every key of result and of every call except timing.
"""
import json

V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/census/state/V/"
H = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06h-census-20261101T005037/checkpoint.jsonl"
TIMING = {"elapsed", "compute_time", "seconds", "wall_s", "rebuild_s", "scan_s", "cc_floor_s", "declared_06h_s",
          "checked_06g_s", "recorded", "worker", "worker_pid", "unit", "tree", "tree_head", "tree_status_short",
          "modules", "worker_tree", "pickle", "pickle_offset", "pickle_length", "pickle_record_sha256",
          "scripts_sha256", "cache_built", "x1_inputs", "w3", "deep_lines", "deep_lines_06e"}


def unit_rec(name):
    for l in open(V + name + "/emit.jsonl"):
        d = json.loads(l)
        if d["kind"] == "unit":
            return d["data"]
    raise SystemExit(f"no unit in {name}")


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'b' if k not in a else 'a'}")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if a.hex() != b.hex():
            out.append(f"{path}: {a.hex()} vs {b.hex()}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]} vs {str(b)[:80]}")
    return out


want = {}
for name in ("P1", "P2", "P3"):
    u = unit_rec(name)
    want[(u["rtol"], int(u["shard"]), int(u["serial"]))] = (name, u)
found = {}
with open(H) as fh:
    for l in fh:
        d = json.loads(l)["data"]
        k = (d["rtol"], int(d["shard"]), int(d["serial"]))
        if k in want:
            found[k] = d
for k, (name, p) in want.items():
    h = found[k]
    print(f"== Gate V part 1: {name} {p['unit']} at {p['tree_head'][:7]} (status '{p['tree_status_short']}') vs 06h {h['unit']} at {h['tree_head'][:7]}")
    sections = {"result": (p["result"], h["result"]), "analytic_06h.h": (p["analytic_06h"]["h"], h["analytic_06h"]["h"])}
    hk = set().union(*(c.keys() for c in h["calls"] + h["other_calls"])) if (h["calls"] or h["other_calls"]) else set()
    calls_p = [{kk: v for kk, v in c.items() if kk in hk} for c in p["calls"]]
    others_p = [{kk: v for kk, v in c.items() if kk in hk} for c in p["other_calls"]]
    sections["calls"] = (calls_p, h["calls"])
    sections["other_calls"] = (others_p, h["other_calls"])
    for s, (a, b) in sections.items():
        dd = diff(strip(a), strip(b))
        n = len(a) if isinstance(a, list) else 1
        print(f"   {s} ({n}): {'IDENTICAL' if not dd else str(len(dd)) + ' differences'}")
        for x in dd[:10]:
            print("      ", x)
g1, g2 = unit_rec("G1"), unit_rec("G2")
print(f"== Gate V part 3: G1 vs G2, {g1['unit']}; workers {g1['worker_pid']} / {g2['worker_pid']}")
for s in ("result", "calls", "other_calls", "analytic_06h", "stored"):
    dd = diff(strip(g1[s]), strip(g2[s]))
    print(f"   {s}: {'IDENTICAL' if not dd else str(len(dd)) + ' differences'}")
    for x in dd[:10]:
        print("      ", x)
cc = [c for c in g1["calls"] + g1["other_calls"] if c.get("has_cc_leaf") and float.fromhex(c["node_error_signed_sum"]["hex"]) != 0.0]
print(f"   G1 calls with a CC leaf and sum E != 0: {len(cc)}")
q = sum((c.get("leaves") or {}).get("trusted", 0) for c in g1["calls"] + g1["other_calls"])
print(f"   G1 trusted leaves (leaves.trusted summed): {q}")
