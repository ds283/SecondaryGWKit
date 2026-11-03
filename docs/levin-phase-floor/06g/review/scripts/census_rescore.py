"""Orchestrator's own reader: V2, V5, V6, V7, V8 re-scored from the two census checkpoints.

Written independently of the census agent's scripts, from prompt 06g §5's rules:
- the 720 stationary-point rows: (shard, serial) of 06a 1e-7 units whose scan has a call with
  n_sign_changes > 0 (06a run levin-phase-floor-06a-census-20261027T023143);
- units matched by (rtol, shard, serial); calls by (kind, index), with equal notify_label and
  x_span by hex (kind "group" = data.calls, "analytic" = data.other_calls);
- V2: exact Fractions on total/total_abserr; oracle from data.analytic_06h.h (value, abserr),
  cross-checked against result.analytic_rad;
- V5: a g call with max_depth >= depth_max (also n_at_depth_max > 0 reported), or a unit over
  budget (status != done or wall_s > budget_s);
- V8: 06h converged and g not.
"""
import json, sys, math, collections
from fractions import Fraction

RUNS = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
G = RUNS + "levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"
H = RUNS + "levin-phase-floor-06h-census-20261101T005037/checkpoint.jsonl"
A = RUNS + "levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"


def hx(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return float.fromhex(v["hex"])
    return float(v)


def span(c):
    return tuple(x["hex"] if isinstance(x, dict) else float(x).hex() for x in (c.get("x_span") or []))


sp = set()
n06a = 0
with open(A) as fh:
    for line in fh:
        d = json.loads(line)
        D = d.get("data", d)
        if not D["unit"].split("-")[1:2] == ["1e"]:
            pass
        if D.get("rtol") not in ("1e-7", None) and "-1e-7-" not in D["unit"]:
            continue
        if "-1e-7-" not in D["unit"]:
            continue
        n06a += 1
        if any((s.get("n_sign_changes") or 0) > 0 for s in (D.get("scan") or [])):
            sp.add((int(D["shard"]), int(D["serial"])))
print("06a 1e-7 units", n06a, "stationary-point rows", len(sp))
assert len(sp) == 720


def load(path):
    units = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            D = d["data"]
            key = (D["rtol"], int(D["shard"]), int(D["serial"]))
            calls = {}
            for kind, lst in (("group", D.get("calls") or []), ("analytic", D.get("other_calls") or [])):
                for c in lst:
                    calls[(kind, c["index"])] = dict(
                        label=c.get("notify_label"), span=span(c), conv=c.get("converged"),
                        max_depth=c.get("max_depth"), depth_max=c.get("depth_max"),
                        n_at=c.get("n_at_depth_max"), node_limited=c.get("node_limited"),
                        value=hx(c.get("value")), abserr=hx(c.get("abserr")), regions=c.get("num_regions"),
                        group=c.get("group"), nE=c.get("node_error_signed_sum"), aE=c.get("node_error_abs_sum"),
                        atol=c.get("atol"), rtol=c.get("rtol"))
            R = D.get("result") or {}
            an = (D.get("analytic_06h") or {}).get("h") or {}
            units[key] = dict(unit=D["unit"], status=D.get("status"), wall=D.get("wall_s"), budget=D.get("budget_s"),
                              total=R.get("total"), err=R.get("total_abserr"), an_rad=R.get("analytic_rad"),
                              an_v=an.get("value"), an_e=an.get("abserr"), calls=calls,
                              head=D.get("tree_head"), dirty=D.get("tree_status_short"), script=d.get("script_sha256"),
                              error=D.get("error"))
    return units


g = load(G)
h = load(H)
print("units g", len(g), "06h", len(h), "keys equal", set(g) == set(h))
print("g heads", collections.Counter(u["head"] for u in g.values()), "dirty", collections.Counter(u["dirty"] for u in g.values()))
print("g script", collections.Counter(u["script"] for u in g.values()))
print("g status", collections.Counter(u["status"] for u in g.values()), "errors", sum(1 for u in g.values() if u["error"]))
print("g over 300 s", sum(1 for u in g.values() if (u["wall"] or 0) > (u["budget"] or 300)))

for rtol in ("1e-7", "1e-8"):
    print(f"\n===== rtol {rtol}")
    keys = sorted(k for k in g if k[0] == rtol)
    # V2
    out = {"total": [], "oracle": []}
    worst = {"total": (0, None), "oracle": (0, None)}
    nsc = collections.Counter()
    oracle_check = collections.Counter()
    for k in keys:
        if (k[1], k[2]) in sp:
            continue
        a, b = h[k], g[k]
        if a["status"] != "done" or b["status"] != "done":
            nsc["not done"] += 1
            continue
        for name, v, e in (("total", "total", "err"), ("oracle", "an_v", "an_e")):
            va, vb, ea, eb = hx(a[v]), hx(b[v]), hx(a[e]), hx(b[e])
            if None in (va, vb, ea, eb) or not all(map(math.isfinite, (va, vb, ea, eb))):
                nsc[(name, "unscored")] += 1
                continue
            nsc[(name, "scored")] += 1
            d = abs(Fraction(vb) - Fraction(va))
            s = Fraction(ea) + Fraction(eb)
            r = float(d / s) if s else (0.0 if d == 0 else math.inf)
            if d > s:
                out[name].append((r, b["unit"], a[v]["hex"] if isinstance(a[v], dict) else va.hex(),
                                  b[v]["hex"] if isinstance(b[v], dict) else vb.hex(), ea.hex(), eb.hex()))
            if r > worst[name][0]:
                worst[name] = (r, b["unit"])
        if b["an_rad"] is not None and b["an_v"] is not None:
            oracle_check[hx(b["an_rad"]).hex() == hx(b["an_v"]).hex()] += 1
    print("V2 counts", dict(nsc), "oracle value == result.analytic_rad (g):", dict(oracle_check))
    for name in ("total", "oracle"):
        print(f"V2 {name}: {len(out[name])} violations; worst ratio {worst[name][0]:.6g} at {worst[name][1]}")
        for t in sorted(out[name], reverse=True):
            print("   ", "%.4g" % t[0], t[1], "06h", t[2], "g", t[3], "e06h", t[4], "eg", t[5])
    # matching, V5, V8
    unmatched = []
    v5 = []
    v5_nat = 0
    lost = []
    gained = []
    scored = collections.Counter()
    over = []
    for k in keys:
        a, b = h[k], g[k]
        in_sp = (k[1], k[2]) in sp
        if b["status"] != "done" or (b["wall"] or 0) > (b["budget"] or 300):
            over.append((b["unit"], b["status"], b["wall"]))
        for ck in sorted(set(a["calls"]) | set(b["calls"])):
            ca, cb = a["calls"].get(ck), b["calls"].get(ck)
            if ca is None or cb is None or ca["label"] != cb["label"] or ca["span"] != cb["span"]:
                unmatched.append((b["unit"], ck))
                continue
            if in_sp:
                continue
            if cb["max_depth"] is not None and cb["depth_max"] is not None and cb["max_depth"] >= cb["depth_max"]:
                v5.append((b["unit"], ck, cb["group"], cb["conv"]))
            if (cb["n_at"] or 0) > 0:
                v5_nat += 1
            if ca["conv"]:
                scored[ck[0]] += 1
                if not cb["conv"]:
                    lost.append((b["unit"], k[1], k[2], ck, cb["group"], cb["node_limited"]))
            elif cb["conv"]:
                gained.append((b["unit"], ck))
    print("unmatched calls", len(unmatched), unmatched[:5])
    print("over budget / not done", len(over), over[:5])
    print("V5: g calls outside the 720 at max_depth >= depth_max:", len(v5), "in rows", len({u for u, *_ in v5}),
          "| with n_at_depth_max > 0:", v5_nat, "| of which converged:", sum(1 for x in v5 if x[3]),
          "| kinds", dict(collections.Counter(x[1][0] for x in v5)))
    h5 = sum(1 for k in keys if (k[1], k[2]) not in sp for c in h[k]["calls"].values()
             if c["max_depth"] is not None and c["depth_max"] is not None and c["max_depth"] >= c["depth_max"])
    print("    06h calls outside the 720 at the cap:", h5)
    print("V5:", "holds" if not v5 and not over else "fails")
    print("V8 scored calls by kind", dict(scored))
    nl = [x for x in lost if x[5]]
    print("V8: lost", len(lost), "calls in", len({x[0] for x in lost}), "rows; node_limited", len(nl), "in rows", len({x[0] for x in nl}),
          "| kinds", dict(collections.Counter(x[3][0] for x in lost)), "| groups", dict(collections.Counter(x[4] for x in lost)))
    print("    gained", len(gained), gained)
    print("V8:", "holds" if not lost else "fails")

for unit, idx, name in (("g-1e-8-0-33995", 5, "V6"), ("g-1e-8-3-1072", 13, "V7")):
    _, rt, s, se = unit.split("-")[0], "1e-8", int(unit.split("-")[3]), int(unit.split("-")[4])
    k = ("1e-8", s, se)
    for kind in ("group", "analytic"):
        if (kind, idx) in g[k]["calls"]:
            cg, ch = g[k]["calls"][(kind, idx)], h[k]["calls"][(kind, idx)]
            print(f"\n{name} {unit} call {idx} ({kind}, {cg['label'][-40:]})")
            for tag, c in (("06h", ch), ("g", cg)):
                req = max(c["atol"], c["rtol"] * abs(c["value"])) if c["value"] is not None else None
                print(f"   {tag}: converged {c['conv']} node_limited {c['node_limited']} max_depth {c['max_depth']}/{c['depth_max']}"
                      f" regions {c['regions']} abserr {c['abserr']!r} request {req!r} sumE {c['nE']} sum|E| {c['aE']}")
            if name == "V6":
                ok = cg["max_depth"] < cg["depth_max"] and (cg["conv"] or cg["node_limited"])
            else:
                ok = bool(cg["conv"])
            print(f"   {name}:", "holds" if ok else "fails")
