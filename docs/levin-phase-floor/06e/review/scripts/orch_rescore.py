"""Orchestrator's own re-scoring of 06e's Z2, Z4 and Z5, and the descriptive counts, from the two
census checkpoints. Reads files only. Floats compared exactly (Fraction of float.fromhex)."""

import collections
import hashlib
import json
import math
import sys
from fractions import Fraction

C06 = "var/runs/levin-phase-floor-06c-census-20261029T130214/checkpoint.jsonl"
C06E = "var/runs/levin-phase-floor-06e-census-20261030T081318/checkpoint.jsonl"
C06A = "var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["unit"]] = r["data"]
    return out


def fx(v):
    return float.fromhex(v["hex"])


def calls(d):
    for c in d.get("calls") or []:
        yield ("calls", c["index"], c["notify_label"]), c
    for c in d.get("other_calls") or []:
        yield ("other_calls", c["index"], c["notify_label"]), c


for p in (C06, C06E, C06A):
    print("input", p, "sha256", sha(p))

a = load(C06A)
sp = set()
for u, d in a.items():
    if d["rtol"] != "1e-7":
        continue
    if any((s or {}).get("n_sign_changes", 0) > 0 for s in (d.get("scan") or [])):
        sp.add((d["shard"], d["serial"]))
print("stationary-point rows from 06a's 1e-7 scans (n_sign_changes > 0):", len(sp))

c6 = load(C06)
ce = load(C06E)
print("units: 06c", len(c6), " 06e", len(ce), " same unit names:", set(c6) == set(ce))

for rtol in ("1e-7", "1e-8"):
    print(f"\n======== rtol {rtol}")
    units = sorted(u for u in ce if ce[u]["rtol"] == rtol)
    status = collections.Counter(ce[u]["status"] for u in units)
    print("06e unit status:", dict(status))
    over300 = [u for u in units if (ce[u].get("wall_s") or 0) > 300]
    print("06e units with wall_s > 300:", over300)
    wall_e = sum(ce[u].get("wall_s") or 0 for u in units)
    wall_c = sum(c6[u].get("wall_s") or 0 for u in units)
    print(f"unit wall summed: 06e {wall_e/3600:.3f} h, 06c {wall_c/3600:.3f} h, ratio {wall_e/wall_c:.3f}")
    print(
        "max unit wall: 06e %.1f s, 06c %.1f s"
        % (max(ce[u]["wall_s"] for u in units), max(c6[u]["wall_s"] for u in units))
    )

    # Z2
    z2_out, z2_in, z2_06c_out = [], [], 0
    deep_out = 0
    for u in units:
        d = ce[u]
        insp = (d["shard"], d["serial"]) in sp
        for k, c in calls(d):
            dm = c.get("depth_max", 20)
            if c["max_depth"] >= dm:
                (z2_in if insp else z2_out).append((u, k, c["max_depth"], c["converged"]))
            if not insp and c["max_depth"] > 12:
                deep_out += 1
        if not insp:
            for k, c in calls(c6[u]):
                if c["max_depth"] >= c.get("depth_max", 20):
                    z2_06c_out += 1
    over_out = [u for u in units if ce[u]["status"] != "done" and (ce[u]["shard"], ce[u]["serial"]) not in sp]
    print(f"Z2: calls outside the 720 reaching depth_max under g: {len(z2_out)} {z2_out}")
    print(f"    inside the 720 (reported, not scored): {len(z2_in)}; over-budget units outside: {len(over_out)}")
    print(f"    06c, outside the 720, calls reaching depth_max: {z2_06c_out}; g calls outside past depth 12: {deep_out}")
    print("    Z2 at this rtol:", "HOLDS" if not z2_out and not over_out else "FAILS")

    # Z4
    n = viol = nonfin = 0
    worst = (Fraction(0), None)
    for u in units:
        d = ce[u]
        if (d["shard"], d["serial"]) in sp:
            continue
        rg, rc = d["result"], c6[u]["result"]
        ag, ac = fx(rg["total_abserr"]), fx(rc["total_abserr"])
        if not (math.isfinite(ag) and math.isfinite(ac)):
            nonfin += 1
            continue
        n += 1
        diff = abs(Fraction(fx(rg["total"])) - Fraction(fx(rc["total"])))
        bound = Fraction(ag) + Fraction(ac)
        if diff > bound:
            viol += 1
        r = diff / bound if bound else (Fraction(0) if diff == 0 else Fraction(10**9))
        if r > worst[0]:
            worst = (r, u, rg["total"]["hex"], rc["total"]["hex"], rg["total_abserr"]["hex"], rc["total_abserr"]["hex"])
    print(f"Z4: rows scored {n}, non-finite total_abserr {nonfin}, violations {viol}")
    print(f"    largest |dT|/(a06c+ag) = {float(worst[0]):.6g} at {worst[1:]}")
    print("    Z4 at this rtol:", "HOLDS" if viol == 0 else "FAILS")

    # Z5
    unc, fail, unmatched = 0, [], []
    for u in units:
        d6 = c6[u]
        if (d6["shard"], d6["serial"]) in sp:
            continue
        ge = dict(calls(ce[u]))
        for k, c in calls(d6):
            if c["converged"]:
                continue
            unc += 1
            g = ge.get(k)
            if g is None:
                unmatched.append((u, k))
            elif not g["converged"]:
                fail.append((u, k, g["max_depth"], g["num_regions"]))
    print(f"Z5: 06c unconverged calls outside the 720: {unc}; not converged under g: {len(fail)} {fail}; unmatched {unmatched}")
    print("    Z5 at this rtol:", "HOLDS" if not fail and not unmatched else ("FAILS" if fail else "UNDECIDED"))

    # descriptive
    for name, src in (("06c", c6), ("06e", ce)):
        tot = collections.Counter()
        hist = collections.Counter()
        for u in units:
            d = src[u]
            tot["rows_total_converged_false"] += not d["result"]["total_converged"]
            if not d["result"]["total_converged"] and (d["shard"], d["serial"]) not in sp:
                tot["...outside_720"] += 1
            rowmax = 0
            for k, c in calls(d):
                kind = "group" if k[0] == "calls" else "analytic"
                tot[f"{kind}_calls"] += 1
                tot[f"{kind}_unconverged"] += not c["converged"]
                tot["depth20_calls"] += c["max_depth"] >= c.get("depth_max", 20)
                tot["regions"] += c["num_regions"]
                tot["solves"] += c.get("num_subregion_solves") or 0
                rowmax = max(rowmax, c["max_depth"])
            hist[rowmax] += 1
        print(f"  {name}: {dict(tot)}")
        print(f"  {name} row max-depth histogram: {dict(sorted(hist.items()))}")
