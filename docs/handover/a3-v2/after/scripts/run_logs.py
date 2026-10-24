"""Prompt 08 §4's two log measurements: the QSI stage's tail, and the IntegrationWarnings.

Read-only. From the repository root:  ./venv/bin/python var/handover-08/scripts/run_logs.py
"""

import datetime as dt
import os
import re

D = "var/runs/handover-07-a3-v2-20261022T211155"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def lines(p):
    return [ANSI.sub("", l) for l in open(p, errors="replace").read().splitlines()]


out = lines(f"{D}/stdout.log")
i0 = next(
    i
    for i, l in enumerate(out)
    if l.startswith("** CALCULATE QUADRATIC SOURCE INTEGRALS")
)
stage = out[i0:]
print(
    "== 1. The tail of CALCULATE QUADRATIC SOURCE INTEGRALS (stdout.log, from line %d)"
    % (i0 + 1)
)
prog = []
for j, l in enumerate(stage):
    m = re.match(
        r"\s+-- (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \((.*?) running\): (\d+)/(\d+) work items remaining",
        l,
    )
    if m:
        nxt = stage[j + 1] if j + 1 < len(stage) else ""
        f = re.search(
            r"inflight: (\d+) lookup, (\d+) compute, (\d+) store \| completed: (\d+) lookup.*?, (\d+) compute.*?, (\d+) store",
            nxt,
        )
        prog.append(
            (
                dt.datetime.fromisoformat(m.group(1)),
                m.group(2),
                int(m.group(3)),
                int(m.group(4)),
                tuple(int(x) for x in f.groups()) if f else None,
            )
        )
end_line = next(l for l in stage if "ALL WORK ITEMS COMPLETE" in l)
print("stage end line:", end_line.strip())
start = prog[0][0] - dt.timedelta(
    seconds=5 * 60 + 2.7
)  # the first line's own "(5m 2.7s running)"
print(
    f"stage start (first progress line minus its running time): {start:%Y-%m-%d %H:%M:%S}"
)
st = os.stat(f"{D}/status.json")
print(
    f"{len(prog)} progress lines. Batches remaining / inflight (lookup, compute, store) / completed (lookup, compute, store):"
)
for t, run, rem, tot, f in prog:
    print(
        f"  {t:%Y-%m-%d %H:%M:%S} ({run}) {rem}/{tot} batches; inflight {f[:3] if f else '?'}; completed {f[3:] if f else '?'}"
    )
last = [x for x in prog if x[4] and x[4][1] >= 6][
    -1
]  # the last line with six or more compute items in flight
final = prog[-1]
stage_end = dt.datetime(
    2026, 10, 24, 21, 56, 16
)  # the store's last flush and the fingerprint's taken (status.json)
total = stage_end - start
print(
    f"\nstage wall time: {start:%m-%d %H:%M:%S} -> {stage_end:%m-%d %H:%M:%S} = {total} ({total.total_seconds() / 3600:.2f} h); the log's own figure: '2d 33m 16.8s'"
)
print(
    f"last progress line with >= 6 compute items in flight: {last[0]:%m-%d %H:%M:%S}: compute {last[4][1]}, store {last[4][2]}"
)
print(
    f"the next and final progress line: {final[0]:%m-%d %H:%M:%S}: compute {final[4][1]}, store {final[4][2]} (every item computed, the last store batch pending); the stage ended at 21:56:16."
)
print(
    "No progress line shows between 1 and 5 compute items in flight: none was printed between those two."
)
print(
    "Bounds on the tail (fewer than six items in flight, i.e. some of the six workers idle):"
)
print(
    f"  - it began after {last[0]:%H:%M:%S} (52 in flight) and before 21:48:xx, when the session's one ps snapshot (2026-10-24,"
)
print(
    "    'Can we do another check?') found five workers ray::IDLE and one (pid 84484) at 99 % CPU;"
)
ub = stage_end - last[0]
print(
    f"  - so it lasted at most {ub} ({ub.total_seconds() / 3600:.2f} h) = {100 * ub / total:.1f} % of the stage, and at least the ~8 min from 21:48 to 21:56."
)
print(
    "  - the longest-running of the last 52 items (stored together at 21:56:16) took 6580.6 s (1 h 49.7 m): (k, q, r) ="
)
print(
    "    (3.0455e7, 3.0917e6, 3.0455e7) /Mpc at z_response 0.1738, shard 0 serial 39197; from compute_time alone it began"
)
print(
    "    no later than ~20:06:35, so it overlapped the others; which item ran alone last is not recorded."
)
print(
    "  => the serial tail is under 10 % of the stage's wall time by the upper bound; not opened."
)
print(
    "  After 2026-10-23 03:57:52 (6 h 34 m 53 s in) every batch had been handed out (0/13): the remaining ~42 h were"
)
print(
    "  the six workers draining the in-flight items, mostly at full occupancy (the session's ps snapshots at 2026-10-23 09:34 and 2026-10-24 01:40: six workers at 84-100 %)."
)

print("\n== 2. IntegrationWarnings in stderr.log")
err = lines(f"{D}/stderr.log")
print(
    f"stderr.log: {len(err)} lines; mtime {dt.datetime.fromtimestamp(os.stat(f'{D}/stderr.log').st_mtime):%Y-%m-%d %H:%M:%S}"
    f" (the QSI stage began {start:%H:%M:%S}, so every warning came in its first ~{(dt.datetime.fromtimestamp(os.stat(f'{D}/stderr.log').st_mtime) - start).total_seconds() / 60:.1f} min)"
)
kinds = {
    "maximum number of subdivisions (100)": 0,
    "probably divergent, or slowly convergent": 0,
    "roundoff error": 0,
}
events = []
for n, l in enumerate(err, 1):
    m = re.match(
        r"\((\S+) pid=(\d+)\) (\S+): IntegrationWarning: (.*?)( \[repeated (\d+)x across cluster\].*)?$",
        l,
    )
    if m:
        task, pid, where, msg, _, rep = m.groups()
        rep = int(rep) if rep else 1
        k = next(k for k in kinds if k in msg)
        kinds[k] += rep
        events.append((n, task, pid, where, k, rep))
for n, task, pid, where, k, rep in events:
    print(
        f"  stderr.log:{n}: {task}, worker pid {pid}, {where}: '{k}'"
        + (
            f" x{rep} (Ray's dedup: [repeated {rep}x across cluster], pids not given)"
            if rep > 1
            else ""
        )
    )
print("counts:", kinds, "; total", sum(kinds.values()))
print(
    "Every warning is SciPy quad's, raised at Quadrature/simple_quadrature.py:92 (quad(RHS, a=a, b=b, epsabs=atol, epsrel=rtol,"
)
print(
    "limit=100)) inside a compute_QuadSource_integral Ray task, in the QSI stage. The log names a worker pid, never a row,"
)
print(
    "a (k, q, r) or a z: no warning can be tied to a stored QuadSourceIntegral or QuadSource row from the log. Nothing recomputed."
)
