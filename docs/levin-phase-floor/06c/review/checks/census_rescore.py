"""Orchestrator's own re-score of 06c's census (06c Part B review, checks 7 and 9).

Reads 06c's census checkpoint and 06b's (both read only, streamed), and 06a's 1e-7 scan for the
stationary-point rows. Scores X2, X3, X4 and X5 by the rules of prompt 06c section 5 as written,
and checks every 06c record's provenance (tree, status, script hash, module roots). Floats are
compared by bit pattern; X4's inequality is evaluated exactly in rationals.
"""

import collections
import json
import math
from fractions import Fraction

R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
G = R + "levin-phase-floor-06c-census-20261029T130214/checkpoint.jsonl"
B = R + "levin-phase-floor-06b-census-20261028T161212/checkpoint.jsonl"
A = R + "levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
DEPTH_MAX = 20
G_HEAD = "e5764a6708757cb11704b95bcd66825baab4ba29"
G_ROOT = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6/"


def hx(v):
    return v["hex"] if isinstance(v, dict) else (v.hex() if isinstance(v, float) else str(v))


def fl(v):
    return float.fromhex(hx(v)) if hx(v) not in ("inf", "-inf", "nan") else float(hx(v))


def compact(c):
    return dict(idx=c["index"], label=c["notify_label"], v=hx(c["value"]), e=hx(c["abserr"]),
                conv=c["converged"], nreg=c["num_regions"], depth=c["max_depth"],
                ncap=c.get("n_at_depth_max"), npl=c.get("n_phase_limited_regions"),
                nun=c.get("num_untrusted_regions"))


def load(path, prov=None):
    units = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            dd = d.get("data")
            if not isinstance(dd, dict) or "calls" not in dd:
                continue
            if prov is not None:
                prov["records"] += 1
                prov["script"][d.get("script_sha256")] += 1
                prov["tree_head"][dd.get("tree_head")] += 1
                prov["status_short"][repr(dd.get("tree_status_short"))] += 1
                prov["status"][dd.get("status")] += 1
                m = dd.get("modules") or {}
                prov["foreign_nonempty"] += 1 if m.get("foreign") else 0
                files = m.get("files") or {}
                lq = files.get("AdaptiveLevin.levin_quadrature")
                prov["levin_quadrature"][lq] += 1
                prov["own_files_outside_g"] += sum(
                    1 for p in files.values() if p and "SecondaryGWKit" in p and not p.startswith(G_ROOT)
                    and "/venv/" not in p and "/var/levin-phase-floor-06c/census/scripts/" not in p)
                if dd.get("wall_s", 0) > dd.get("budget_s", 300):
                    prov["wall_over_budget"].append((dd["unit"], round(dd["wall_s"], 1), dd["status"]))
            units[(dd["rtol"], dd["row"])] = dict(
                unit=dd["unit"], status=dd["status"], wall=dd.get("wall_s"),
                calls=[("group", compact(c)) for c in dd["calls"]] + [("analytic", compact(c)) for c in dd["other_calls"]],
                T=hx(dd["result"]["total"]) if dd.get("result") else None,
                E=hx(dd["result"]["total_abserr"]) if dd.get("result") else None,
                conv=dd["result"].get("total_converged") if dd.get("result") else None)
    return units


sp = set()
with open(A) as f:
    for line in f:
        if '"rtol": "1e-7"' not in line:
            continue
        dd = json.loads(line).get("data") or {}
        if dd.get("rtol") != "1e-7" or "scan" not in dd:
            continue
        if any(s and s["n_sign_changes"] > 0 for s in dd["scan"] or []):
            sp.add(dd["row"] if "row" in dd else f"{dd['shard']}-{dd['serial']}")
print("06a stationary-point rows:", len(sp))

prov = dict(records=0, script=collections.Counter(), tree_head=collections.Counter(), status_short=collections.Counter(),
            status=collections.Counter(), foreign_nonempty=0, levin_quadrature=collections.Counter(), own_files_outside_g=0,
            wall_over_budget=[])
g = load(G, prov)
b = load(B)
print("units g", len(g), "06b", len(b), "same keys", set(g) == set(b))
print("== check 7: provenance of every 06c record")
for k, v in prov.items():
    if isinstance(v, collections.Counter):
        v = dict(v)
    print("  ", k, ":", v)

for rtol in ("1e-7", "1e-8"):
    keys = sorted(k for k in g if k[0] == rtol)
    x2n = x2fail = 0
    x2n_grp = x2fail_grp = 0
    x2kinds = collections.Counter()
    x3inf = x3bad = x3nan = 0
    x5moved = 0
    x5cap = []
    x4n = x4v = x4nonfin = 0
    x4worst = 0.0
    label_mismatch = 0
    rows_unconv = [0, 0]
    over_budget_units = sum(1 for k in keys if g[k]["status"] != "done")
    for k in keys:
        ug, ub = g[k], b[k]
        rows_unconv[0] += 0 if ub["conv"] else 1
        rows_unconv[1] += 0 if ug["conv"] else 1
        bi = {c["idx"]: c for _, c in ub["calls"]}
        for kind, cg in ug["calls"]:
            # X3 over every g call
            eg = fl(cg["e"]) if cg["e"] not in ("nan",) else math.nan
            if math.isnan(eg):
                x3nan += 1
            if math.isinf(eg) or math.isnan(eg):
                x3inf += 1
                if not (cg["ncap"] or 0) > 0:
                    x3bad += 1
            cb = bi.get(cg["idx"])
            if cb is None or cb["label"] != cg["label"]:
                label_mismatch += 1
                continue
            mv = cg["v"] != cb["v"] or cg["e"] != cb["e"] or cg["nreg"] != cb["nreg"] or cg["depth"] != cb["depth"]
            if mv:
                x5moved += 1
                if cg["depth"] >= DEPTH_MAX:
                    x5cap.append((ug["unit"], cg["idx"], kind, cg["nreg"], cg["conv"]))
            if k[1] not in sp:
                eb = fl(cb["e"])
                if (not cb["conv"]) and cb["depth"] < DEPTH_MAX and (cb["npl"] or 0) == 0 and math.isfinite(eb):
                    x2n += 1
                    if kind == "group":
                        x2n_grp += 1
                    if not cg["conv"]:
                        x2fail += 1
                        if kind == "group":
                            x2fail_grp += 1
                        x2kinds["untrusted leaf at cap" if (cg["nun"] or 0) > 0 else (
                            "depth_max, all trusted" if cg["depth"] >= DEPTH_MAX else "below depth_max (floor/other)")] += 1
        if k[1] not in sp:
            Tg, Eg, Tb, Eb = ug["T"], ug["E"], ub["T"], ub["E"]
            if Tg is None or Tb is None:
                continue
            if not (math.isfinite(fl(Eg)) and math.isfinite(fl(Eb))):
                x4nonfin += 1
                continue
            x4n += 1
            d = abs(Fraction(fl(Tg)) - Fraction(fl(Tb)))
            e = Fraction(fl(Eg)) + Fraction(fl(Eb))
            if d > e:
                x4v += 1
                x4worst = max(x4worst, float(d / e) if e else math.inf)
    print(f"== {rtol}")
    print(f"   units {len(keys)}, not done {over_budget_units}, call label mismatches {label_mismatch}")
    print(f"   rows total_converged false: 06b {rows_unconv[0]}, g {rows_unconv[1]}")
    print(f"   X2: 06b X2-type calls outside SP rows {x2n} (group {x2n_grp}); unconverged under g {x2fail} (group {x2fail_grp}); kinds {dict(x2kinds)}"
          f" -> {'holds' if x2fail == 0 else 'FAILS'}")
    print(f"   X3: g calls with non-finite abserr {x3inf} (NaN {x3nan}); of them with no leaf at depth_max {x3bad} -> {'holds' if x3bad == 0 else 'FAILS'}")
    print(f"   X4: rows scored {x4n}, non-finite apart {x4nonfin}, violations {x4v}, worst ratio {x4worst:.4g} -> {'holds' if x4v == 0 else 'FAILS'}")
    kinds = collections.Counter(t[2] for t in x5cap)
    big = max(x5cap, key=lambda t: t[3]) if x5cap else None
    print(f"   X5: moved calls {x5moved}; moved calls at depth_max {len(x5cap)} by kind {dict(kinds)}; converged {sum(1 for t in x5cap if t[4])};"
          f" largest {big} -> {'holds' if not x5cap and over_budget_units == 0 else 'FAILS'}")
print("units with wall_s over budget_s (all rtols):", len(prov["wall_over_budget"]))
for t in sorted(prov["wall_over_budget"], key=lambda t: -t[1]):
    print("   ", t)
