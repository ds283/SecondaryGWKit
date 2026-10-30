"""Orchestrator's descriptive look at the census's deep lines (06c Part B review): on each depth-20
call's recorded deepest line, over its last five levels, which term of e dominates, the residual's
ratio to the floor, and how many low mantissa bits of the residual are zero (a residual that is a
difference of rounded values carries few significant bits). Read-only; the levels' fields are
[branch, residual, floor, truncation, e, (e_L+e_R)/e_P, its hex], as analyse06c.py writes them."""
import collections, json, statistics
L = json.load(open("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/census/out/deep_lines_06c.json"))
def tz(h):
    if h in ("0x0.0p+0", "inf", "nan"): return None
    m = h.split("p")[0].split(".")[1] if "." in h else ""
    m = m.ljust(13, "0"); b = bin(int(m, 16))[2:].zfill(52)
    return len(b) - len(b.rstrip("0"))
branch = collections.Counter(); dom = collections.Counter(); ratios = []; zeros = []; exact2 = 0; nlev = 0
for x in L:
    lv = x["levels"][-5:]
    for b, r, f, t, e, q, qh in lv:
        branch[b] += 1; nlev += 1
        r_, f_, t_ = [float.fromhex(v) if isinstance(v, str) else float(v or 0.0) for v in (r, f, t)]
        dom["residual" if r_ >= max(f_, t_) else ("floor" if f_ >= t_ else "truncation")] += 1
        if f_ > 0: ratios.append(r_ / f_)
        z = tz(r) if isinstance(r, str) else None
        if z is not None: zeros.append(z)
        exact2 += qh == "0x1.0000000000000p+1"
print("lines", len(L), "levels examined (last five per line)", nlev)
print("branch", dict(branch)); print("dominant term of e", dict(dom))
print("residual/floor: median %.3g, p10 %.3g, p90 %.3g, min %.3g" % (statistics.median(ratios), sorted(ratios)[len(ratios)//10], sorted(ratios)[9*len(ratios)//10], min(ratios)))
print("residual mantissa trailing zero bits (of 52): median", statistics.median(zeros), "p10", sorted(zeros)[len(zeros)//10], "p90", sorted(zeros)[9*len(zeros)//10])
print("levels with (e_L+e_R)/e_P exactly 2.0:", exact2)
