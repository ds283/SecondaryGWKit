"""levin-phase-floor 06g: the stage-3 backstop's projections, recomputed (a finding, nothing rerun).
run_rerun.py projected el * total / done, which at done = 1 ignores the five other chunks in flight
and counts setup. Here, per stopped reading, from its progress file (the time the first chunk came
back, t1) and its call's segments (the FCC res-1 reading's t_span, chunked as the tool chunks Route
B: 16 segments per chunk, in order):
  (A) six in flight, uniform chunks: t1 * total / 6;
  (B) six in flight, chunk cost proportional to its phase span (QAWO's panels are uniform in phase):
      the six workers take chunks 0..5 at t = 0 and then the next chunk as each frees (the tool's
      imap order); the first completion is the cheapest of chunks 0..5, which fixes the cost per
      radian k = t1 / span(argmin); the projected total is the makespan.
(B) is checked on the two stage-3 readings that finished (r24 and r53 qawo r1): k from their actual
wall time, then the model's chunks done at each monitor time against the monitor's log.
Usage: python -B stage3_projection.py"""
import json, heapq, math, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CH = 16
def chunk_spans(sid):
    d = json.loads((ROOT / "out" / "readings" / f"{sid}-fcc-r1.json").read_text())
    s = [abs(x["t_span"]) for x in d["segments"]]
    return [sum(s[i:i + CH]) for i in range(0, len(s), CH)]
def schedule(costs, w=6):
    free = [(0.0, j) for j in range(w)]; heapq.heapify(free); done = []
    for c in costs:
        t, j = heapq.heappop(free); heapq.heappush(free, (t + c, j)); done.append(t + c)
    return sorted(done)
out = {}
print("== validation on the finished stage-3 readings (k from the actual wall time)")
mon = (ROOT / "out" / "rerun_monitor.log").read_text().splitlines()
for sid, impl, wall in (("r24", "qawo-r1", None), ("r53", "qawo-r1", None)):
    prog = json.loads((ROOT / "out" / "progress_rerun" / f"stopped-{sid}-{impl}.json").read_text())
    d = json.loads((ROOT / "out" / "readings" / f"{sid}-{impl}-rerun.json").read_text())
    wall = d["wall_s"]
    sp = chunk_spans(sid)
    unit = schedule(sp)[-1]
    k = wall / unit
    fin = schedule([k * x for x in sp])
    print(f"{sid} {impl}: {len(sp)} chunks, actual wall {wall:.0f} s, fitted k {k:.4g} s/rad ({k*1e6:.4f} us/rad); first chunk at {fin[0]:.0f} s in the model")
    for line in mon:
        m = re.search(rf"stopped {sid}-{impl}: elapsed (\d+) s.*progress (\d+)/(\d+)", line)
        if m:
            t, n = int(m.group(1)), int(m.group(2))
            print(f"   at {t} s: monitor {n} done, model {sum(1 for x in fin if x <= t)} done")
    out[sid + "-" + impl] = {"wall": wall, "k": k}
print("\n== the three stopped readings")
for sid, impl, est in (("r47", "qawo-r2", (11004, 18340)), ("r53", "qawo-r2", (32544, 54240)), ("r26", "qawo-r1", (51728, 86213))):
    prog = json.loads((ROOT / "out" / "progress_rerun" / f"stopped-{sid}-{impl}.json").read_text())
    t1, done, total = prog["elapsed_s"], prog["done"], prog["total"]
    sp = chunk_spans(sid)
    assert len(sp) == total
    A = t1 * total / 6
    j = min(range(6), key=lambda i: sp[i])
    k = t1 / sp[j]
    B = schedule([k * x for x in sp])[-1]
    share6 = sum(sp[:6]) / sum(sp)
    print(f"{sid} {impl}: total {total} chunks, first chunk back at t1 = {t1:.0f} s (done {done}); stop's projection {t1*total/done:.0f} s")
    print(f"   (A) uniform, six in flight: {t1:.0f} x {total} / 6 = {A:.0f} s ({A/3600:.1f} h): {'a day or more' if A >= 86400 else 'under a day'}")
    print(f"   (B) by phase span: chunks 0-5 hold {share6:.1%} of the span; cheapest of them is chunk {j} ({sp[j]:.4g} rad), k = {t1:.0f}/{sp[j]:.4g} = {k:.4g} s/rad ({k*1e6:.4f} us/rad); makespan {B:.0f} s ({B/3600:.1f} h): {'a day or more' if B >= 86400 else 'under a day'}")
    print(f"   estimate {est[0]}-{est[1]} s")
    out[sid + "-" + impl] = {"t1": t1, "total": total, "stop_projection": t1 * total / done, "A": A, "B": B, "k": k, "share_first6": share6}
(ROOT / "out" / "stage3_projection.json").write_text(json.dumps(out, indent=1))
