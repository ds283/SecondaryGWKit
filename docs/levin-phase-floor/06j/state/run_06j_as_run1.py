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

    def beat():
        while not stop.wait(60):
            try:
                with lock:
                    names = sorted(running)
                run.heartbeat(stage=f"running: {', '.join(names)[:180]}" if names else None)
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()
    results = []
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
                if p.poll() is not None:
                    with lock:
                        del running[name]
                    rc = finish_job(rest[0], p, *rest[1:])
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
