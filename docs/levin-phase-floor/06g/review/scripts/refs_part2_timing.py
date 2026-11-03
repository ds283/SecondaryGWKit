"""Orchestrator's reader: references part 2 timing against projection (read-only, while running)."""
import json, os, statistics, collections

R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/out/"
plan = json.load(open(R + "part2_plan.json"))["plan"]
rows = []
for p in plan:
    f = f"{R}readings/{p['sid']}-{p['impl']}-r{p['res']}.json"
    d = json.load(open(f)) if os.path.exists(f) else None
    rows.append((p, d))
done = [(p, d) for p, d in rows if d is not None]
pend = [p for p, d in rows if d is None]
print(f"plan {len(plan)}; reading files present {len(done)}; not present {len(pend)}")
st = collections.Counter(d.get("status", "value" if "value_hex" in d else "?") for _, d in done)
print("status of present readings:", dict(st))
rat = []
for p, d in done:
    w = d.get("wall_s") or d.get("elapsed_s") or d.get("process_wall_s")
    if w is None:
        continue
    rat.append((w / p["projection_s"], w, p, d))
rs = sorted(r[0] for r in rat)
print(f"wall/projection over {len(rs)}: median {statistics.median(rs):.3g}, p90 {rs[int(0.9*len(rs))]:.3g}, max {rs[-1]:.3g}")
for stage in ("v1", "v8", "ref", "ref_long"):
    s = sorted(r[0] for r in rat if r[2]["stage"] == stage)
    if s:
        print(f"  {stage}: n {len(s)} median {statistics.median(s):.3g} max {s[-1]:.3g}")
print("\nreadings more than 5x their projection (ratio, wall s, projection s, sample, unit, route, impl, res, status, segments, panels, ier_nonzero):")
for r, w, p, d in sorted(rat, key=lambda t: -t[0]):
    if r <= 5:
        continue
    s = d.get("stats") or {}
    print(f"  {r:8.1f}  {w:8.0f}  {p['projection_s']:7.1f}  {p['sid']} {p['unit']} #{p['index']} {p['route']} {p['impl']} r{p['res']}"
          f" status={d.get('status', 'value')} seg={d.get('n_segments')} panels={s.get('panels')} ier!=0={s.get('ier_nonzero')}")
print("\nRoute B QAWO r1 readings done: panels per segment and ier_nonzero share")
for p, d in done:
    if p["route"] == "B" and p["impl"] == "qawo" and "stats" in d and d.get("n_segments"):
        s = d["stats"]
        print(f"  {p['sid']} r{p['res']} seg {d['n_segments']} panels/seg {s['panels']/d['n_segments']:.0f} ier!=0 {s['ier_nonzero']/max(s['panels'],1):.2f} wall {d['wall_s']:.0f}")
print("\nremaining (not yet written):")
tot = 0
for p in pend:
    tot += p["projection_s"]
    print(f"  n={p['n']} {p['stage']} {p['sid']} {p['unit']} #{p['index']} {p['route']} {p['impl']} r{p['res']} projection {p['projection_s']:.0f} s ceiling {p['ceiling_s']:.0f} s")
print(f"sum of remaining projections {tot:.0f} s")
