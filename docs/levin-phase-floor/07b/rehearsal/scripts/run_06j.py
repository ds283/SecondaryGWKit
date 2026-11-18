"""levin-phase-floor prompt 06j: the registered run's driver. New in 06j (after 06g's run_refs.py, kept in
state/scripts_06g_reference/). It runs a plan of stages, each a list of commands; a stage's commands run at most
``max_parallel`` at once (never more than six computing processes). Each command's combined stdout and stderr go
to out/cmd/<run label>/<name>.txt with rec.sh's header and footer (command line, cwd, host, start, end, exit), and
are appended to out/commands.log. A command whose ``requires`` path (relative to the records directory) is absent is
skipped and recorded as skipped. A stage runs whatever earlier stages returned; the driver records every exit code
and goes on (the agent judges the results afterwards), except that a stage marked ``stop_on_fail`` stops the run.

Registry: the launcher (launch_06j.py) has called ``runregistrykit.begin`` and recorded the detached sh's pid; the
driver records its own pid at start, a heartbeat at each command's start and end and every 60 s from a thread, a
checkpoint record per command (Run.record) (name, exit code, wall time), and finishes ``done`` (every command exit 0), ``failed``
(otherwise) or ``killed`` (SIGTERM/SIGINT, children stopped). Ray is never started.

**The readings' monitor** (added for the ``readings`` run, marked "06j readings"; run 1's text is
state/run_06j_as_run1.py; state/readings_plan_20261103T0935Z.md section 5). A command with a ``monitor`` entry is a
reading whose progress file (one_reading.py --progress-order unordered) lists the tool's chunks and the indices
finished. Every 30 s the driver weights each chunk by w_c = a * (its segments) + b * (its phase span)
(out/w_segments.json; a, b the reading's fit), counts done (finished chunks) and in flight (the unfinished chunks
among the first finished + 6 in index order), and forms P_c = elapsed W / (done + in_flight / 2) and
P_o = elapsed W / (done + in_flight), with the plain chunk-count forms. Past 30 minutes or past the reading's
envelope projection it writes a report line to out/reading_monitor.log and the heartbeat, then every 10 minutes.
It stops the reading (SIGTERM to its process group, then SIGKILL after 30 s) only if P_o >= 86400 s, records it
as stopped, and goes on. Nothing else stops a reading.

**The run backstop** (prompt 07 section 6, V4's references, the ruling at stop 4; marked "prompt 07"): a plan with
``backstop_s`` gives each command its ``projected_s``. Every 60 s the heartbeat thread forms the projected finish
P = elapsed + (the projections of the commands not started) + max(0, projection - elapsed) of each command running.
Once a command has run past the plan's ``check_after_s`` (1,800 s), or P exceeds 1.1 times the plan's summed
projection, a line
with the command, its elapsed time, its projection and P goes to out/references_monitor.log and the heartbeat every 10
minutes; nothing is stopped for that. If P >= ``backstop_s``, the line says so and the run is stopped as on SIGTERM.
A plan without ``backstop_s`` runs exactly as before.

Usage: python -B run_06j.py --run-dir DIR --plan PLAN_JSON
"""

import argparse
import datetime
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_at(t):
    try:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, ValueError, OSError):
        return "never"


DAY_S = 86400.0  # 06j readings
REPORT_AFTER_S = 1800.0
REPORT_EVERY_S = 600.0
CHECK_EVERY_S = 30.0
WORKERS = 6


def projections(mon, prog, elapsed, segs):
    """06j readings: (P_c, P_o, count-based P_c, P_o, done, in flight, total) from a progress file."""
    jobs = prog.get("jobs")
    total = prog.get("total") or 0
    done_idx = set(prog.get("done_indices") or [])
    if not jobs or not total:
        return None
    chunks = segs["chunks"]
    w = []
    for i, (kind, args) in enumerate(jobs):
        i0, i1 = args[0], args[1]
        span = chunks[i][2] if i < len(chunks) and chunks[i][0] == i0 and chunks[i][1] == i1 else None
        if span is None:
            return {"error": f"chunk {i} ({i0}, {i1}) does not match out/w_segments.json"}
        w.append(mon["a"] * (i1 - i0) + mon["b"] * span)
    W = sum(w)
    first = sorted(set(range(total)) - done_idx)[:WORKERS]
    d = sum(w[i] for i in done_idx)
    f = sum(w[i] for i in first)
    nd, nf = len(done_idx), len(first)
    inf = float("inf")
    return {"P_c": elapsed * W / (d + f / 2) if d + f > 0 else inf, "P_o": elapsed * W / (d + f) if d + f > 0 else inf,
            "P_c_count": elapsed * total / (nd + nf / 2) if nd + nf > 0 else inf,
            "P_o_count": elapsed * total / (nd + nf) if nd + nf > 0 else inf,
            "done": nd, "in_flight": nf, "total": total, "weight_done_fraction": d / W if W else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    import runregistrykit

    run = runregistrykit.Run(a.run_dir)
    plan = json.loads(Path(a.plan).read_text())
    label = plan["label"]
    cmd_dir = ROOT / "out" / "cmd" / label
    cmd_dir.mkdir(parents=True, exist_ok=True)
    running = {}
    lock = threading.Lock()
    stop = threading.Event()

    def terminal(signum, _frame):
        stop.set()
        with lock:
            for p, _ in list(running.values()):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception:
                    pass
        run.finish("killed", exit_code=128 + signum)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, terminal)
    signal.signal(signal.SIGINT, terminal)
    run.heartbeat(pid=os.getpid(), stage=f"started: {label}, {len(plan['stages'])} stages")

    t_run0 = time.time()  # prompt 07: the run backstop
    last_line = [0.0]
    all_jobs = [j for st in plan["stages"] for j in st["jobs"]]
    total_proj = sum(float(j.get("projected_s") or 0.0) for j in all_jobs)

    def backstop():  # prompt 07
        if not plan.get("backstop_s"):
            return
        now = time.time()
        with lock:
            run_now = {n: (rest[0], rest[3]) for n, (p, rest) in running.items()}
        done = {r["name"] for r in results}
        rem = 0.0
        longest = None
        for j in all_jobs:
            if j["name"] in done:
                continue
            pj = float(j.get("projected_s") or 0.0)
            if j["name"] in run_now:
                el = now - run_now[j["name"]][1]
                rem += max(0.0, pj - el)
                if longest is None or el > longest[1]:
                    longest = (j["name"], el, pj)
            else:
                rem += pj
        P = (now - t_run0) + rem
        flag = (longest is not None and longest[1] > float(plan.get("check_after_s") or 1800.0)) or P > 1.1 * total_proj
        stop_now = P >= float(plan["backstop_s"])
        if (flag and now - last_line[0] >= 600.0) or stop_now:
            last_line[0] = now
            line = (f"{utc()} elapsed {now - t_run0:.0f} s; running {longest[0] if longest else None} for "
                    f"{longest[1] if longest else 0:.0f} s (projected {longest[2] if longest else 0:.0f} s); commands done "
                    f"{len(done)}/{len(all_jobs)}; projected finish P {P:.0f} s (summed projection {total_proj:.0f} s)")
            if stop_now:
                line += f"; P >= {float(plan['backstop_s']):.0f} s: STOPPED by the backstop"
            with open(ROOT / "out" / "references_monitor.log", "a") as ml:
                ml.write(line + "\n")
            run.heartbeat(stage=line[21:200])
            if stop_now:
                os.kill(os.getpid(), signal.SIGTERM)

    def beat():
        while not stop.wait(60):
            try:
                with lock:
                    names = sorted(running)
                run.heartbeat(stage=f"running: {', '.join(names)[:180]}" if names else None)
            except Exception:
                pass
            try:  # prompt 07
                backstop()
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()
    results = []
    mon_state, stopped = {}, set()  # 06j readings
    segs_all = json.loads((ROOT / "out" / "w_segments.json").read_text())["calls"] if (ROOT / "out" / "w_segments.json").exists() else {}
    n_total = sum(len(st["jobs"]) for st in plan["stages"])

    def start(job):
        out = cmd_dir / f"{job['name']}.txt"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if job.get("pythonpath"):
            env["PYTHONPATH"] = job["pythonpath"]
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        f = open(out, "w")
        f.write("$" + "".join(" " + shlex.quote(x) for x in job["argv"]) + "\n")
        f.write(f"# PYTHONPATH {env.get('PYTHONPATH')}\n# cwd {ROOT}\n# host {socket.gethostname()}\n# start {utc()}\n# ----\n")
        f.flush()
        p = subprocess.Popen(job["argv"], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        return p, f, out, time.time()

    def finish_job(job, p, f, out, t0):
        rc = p.returncode
        f.write(f"# ----\n# end {utc()}\n# exit {rc}\n")
        f.close()
        with open(ROOT / "out" / "commands.log", "a") as log:
            log.write(f"=================== {label}/{job['name']} (out/cmd/{label}/{job['name']}.txt)\n")
            log.write(out.read_text())
        wall = time.time() - t0
        results.append({"name": job["name"], "exit": rc, "wall_s": wall})
        run.record(job["name"], {"exit": rc, "wall_s": wall, "at": utc()})
        run.heartbeat(units_done=len(results), units_total=n_total, stage=f"{job['name']} exit {rc} ({wall:.0f} s)")
        return rc

    ok = True
    for st in plan["stages"]:
        jobs = list(st["jobs"])
        par = min(int(st.get("max_parallel", 1)), 6)
        run.heartbeat(stage=f"stage {st['name']}: {len(jobs)} commands, {par} at once")
        stage_ok = True
        pending = list(jobs)
        while pending or running:
            while pending and len(running) < par:
                job = pending.pop(0)
                if job.get("requires") and not (ROOT / job["requires"]).exists():
                    # a command whose input was not made (e.g. a capture that did not reproduce) is skipped
                    results.append({"name": job["name"], "exit": None, "skipped": f"{job['requires']} absent"})
                    run.record(job["name"], {"exit": None, "skipped": f"{job['requires']} absent", "at": utc()})
                    continue
                p, f, out, t0 = start(job)
                with lock:
                    running[job["name"]] = (p, (job, f, out, t0))
                run.heartbeat(stage=f"{job['name']} started")
            for name, (p, rest) in list(running.items()):
                job_ = rest[0]
                mon = job_.get("monitor")  # 06j readings
                if mon is not None and p.poll() is None and time.time() - mon_state.setdefault(name, {"check": 0.0, "report": 0.0})["check"] >= CHECK_EVERY_S:
                    ms = mon_state[name]
                    ms["check"] = time.time()
                    el = time.time() - rest[3]
                    try:
                        prog = json.loads((ROOT / mon["progress"]).read_text())
                        pr = projections(mon, prog, el, segs_all[mon["sid"]])
                    except Exception as exc_:
                        prog, pr = None, {"error": f"progress unreadable: {exc_}"}
                    if pr and "error" not in pr:
                        if (el > REPORT_AFTER_S or el > mon["envelope_s"]) and time.time() - ms["report"] >= REPORT_EVERY_S:
                            ms["report"] = time.time()
                            line = (f"{utc()} {name}: elapsed {el:.0f} s (envelope {mon['envelope_s']:.0f} s); chunks done "
                                    f"{pr['done']}/{pr['total']}, in flight {pr['in_flight']}; weighted done {pr['weight_done_fraction']:.4f}; "
                                    f"P_c {pr['P_c']:.0f} s (finish {utc_at(rest[3] + pr['P_c'])}), P_o {pr['P_o']:.0f} s; "
                                    f"by count P_c {pr['P_c_count']:.0f} s, P_o {pr['P_o_count']:.0f} s")
                            with open(ROOT / "out" / "reading_monitor.log", "a") as ml:
                                ml.write(line + "\n")
                            run.heartbeat(stage=line[21:200])
                        if pr["P_o"] >= DAY_S:
                            line = f"{utc()} {name}: STOPPED by the backstop: P_o {pr['P_o']:.0f} s >= {DAY_S:.0f} s ({pr})"
                            with open(ROOT / "out" / "reading_monitor.log", "a") as ml:
                                ml.write(line + "\n")
                            try:
                                os.killpg(p.pid, signal.SIGTERM)
                                for _ in range(60):
                                    if p.poll() is not None:
                                        break
                                    time.sleep(0.5)
                                if p.poll() is None:
                                    os.killpg(p.pid, signal.SIGKILL)
                                    p.wait()
                            except Exception:
                                pass
                            stopped.add(name)
                    elif pr and "error" in pr and not ms.get("error_logged"):
                        ms["error_logged"] = True
                        with open(ROOT / "out" / "reading_monitor.log", "a") as ml:
                            ml.write(f"{utc()} {name}: monitor: {pr['error']}\n")
                if p.poll() is not None:
                    with lock:
                        del running[name]
                    rc = finish_job(rest[0], p, *rest[1:])
                    if name in stopped:  # 06j readings
                        results[-1]["stopped_by_backstop"] = True
                    if rc != 0:
                        stage_ok = False
            time.sleep(0.5)
        ok &= stage_ok
        if not stage_ok and st.get("stop_on_fail"):
            run.heartbeat(stage=f"STOP: stage {st['name']} failed")
            break
    (ROOT / "out" / f"{label}_results.json").write_text(json.dumps({"label": label, "results": results, "ok": ok,
                                                                     "ended": utc()}, indent=1))
    stop.set()
    run.heartbeat(stage=f"{label} finished: {'all exit 0' if ok else 'some command failed'}")
    run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
