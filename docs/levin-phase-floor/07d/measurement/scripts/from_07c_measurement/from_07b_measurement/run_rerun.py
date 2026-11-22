"""levin-phase-floor prompt 06g section 5, the references: the registered rerun batch (slug
``references-rerun``), per addenda 4 and 5 (the user's rulings of 2026-11-02). New in 06g, after
run_refs.py's part 2.

Runs out/rerun_plan.json's readings (make_rerun_plan.py) in order, one at a time, each by
one_reading.py in its own process group with six workers, the tool at ff3e8394 first on
PYTHONPATH and the sample's harness after it (the plan's ``pythonpath``), progress counted out of
order (``--progress-order unordered``), marked ``--rerun``:
- ``reread``: into out/rereads/<tag>.json; each must equal its old reading's ``value_hex``,
  ``value_30`` and ``imag_30`` bit for bit; **one that does not, or that fails, is a stop**: the
  run is recorded ``failed`` and nothing further runs;
- ``stationary`` and ``stopped``: into out/readings/<tag>-rerun.json (the old readings stay where
  they are, unscored for the 14 stationary-point samples).
**The backstop** (addenda 4 and 5): no reading is stopped for passing 30 minutes or for a multiple
of its estimate. From 30 minutes, every 5 minutes, its progress and projected finish (start +
elapsed x total / done, chunks counted out of order; with no chunk back yet, its estimate's high
end) go to the heartbeat and out/rerun_monitor.log. It is stopped (SIGTERM to its group, SIGKILL
after 30 s), recorded ``stopped`` with the reason, only when its projected total is a day or more
or when it has run 24 h. Each reading is one registry unit; at most six computing processes.

Usage: python -B run_rerun.py --run-dir RUN_DIR   (launch_rerun.py starts it)
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402

PYTHON = str(RC.PYTHON)
DAY_S = 86400.0
CHECK_FROM_S = 1800.0
CHECK_EVERY_S = 300.0
KILL_GRACE_S = 30.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    import runregistrykit

    run = runregistrykit.Run(a.run_dir)
    current = {"child": None}

    def terminal(signum, _frame):
        ch = current["child"]
        if ch is not None and ch.poll() is None:
            try:
                os.killpg(ch.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        run.finish("killed", exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, terminal)
    run.heartbeat(pid=os.getpid(), stage="started: the re-reads are next")
    stop = threading.Event()

    def beat():
        while not stop.wait(60.0):
            try:
                run.heartbeat()
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()
    doc = json.loads((ROOT / "out" / "rerun_plan.json").read_text())
    mon = ROOT / "out" / "rerun_monitor.log"
    counts = {"done": 0, "error": 0, "stopped": 0}
    try:
        for x in doc["plan"]:
            tag = x["tag"]
            dst = (
                (ROOT / "out" / "rereads" / f"{tag}.json")
                if x["stage"] == "reread"
                else (ROOT / "out" / "readings" / f"{tag}-rerun.json")
            )
            log = ROOT / "out" / "reading_logs" / "rerun" / f"{x['stage']}-{tag}.txt"
            prog = ROOT / "out" / "progress_rerun" / f"{x['stage']}-{tag}.json"
            for p_ in (dst, log, prog):
                if p_.exists():
                    raise SystemExit(f"{p_} exists; not overwritten")
                p_.parent.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                PYTHONPATH=x["pythonpath"],
                PYTHONDONTWRITEBYTECODE="1",
                GIT_OPTIONAL_LOCKS="0",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            argv = [
                PYTHON,
                "-B",
                str(HERE / "one_reading.py"),
                x["pickle"],
                x["impl"],
                str(x["res"]),
                str(dst),
                "--sid",
                x["sid"],
                "--unit",
                x["unit"],
                "--index",
                str(x["index"]),
                "--rtol",
                x["rtol"],
                "--workers",
                "6",
                "--progress",
                str(prog),
                "--progress-order",
                "unordered",
                "--rerun",
            ]
            name = f"{x['stage']} {tag}"
            hi = x["estimate_hi_s"]
            run.heartbeat(
                stage=f"{name} running (estimate {x['estimate_lo_s']:.0f}-{hi:.0f} s)"
            )
            notes, reason, last, status = [], None, None, None
            with open(log, "wb") as fo:
                fo.write(
                    (
                        " ".join(argv)
                        + f"\n# PYTHONPATH {x['pythonpath']}\n# cwd {ROOT}\n# start {utc()}\n"
                        f"# estimate_s {x['estimate_lo_s']:.1f}-{hi:.1f}\n# ----\n"
                    ).encode()
                )
                fo.flush()
                t0 = time.monotonic()
                ch = subprocess.Popen(
                    argv,
                    cwd=str(ROOT),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=fo,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                current["child"] = ch
                next_check = CHECK_FROM_S
                while True:
                    try:
                        rc = ch.wait(timeout=5.0)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    el = time.monotonic() - t0
                    if el < next_check:
                        continue
                    next_check = el + CHECK_EVERY_S
                    try:
                        last = json.loads(prog.read_text())
                    except Exception:
                        last = None
                    done, total = (last or {}).get("done"), (last or {}).get("total")
                    if done:
                        ptot, basis = (
                            el * total / done,
                            f"progress {done}/{total} chunks (out of order)",
                        )
                    else:
                        ptot, basis = (
                            max(hi, el),
                            f"no chunk back yet ({done}/{total}); the estimate's high end",
                        )
                    line = (
                        f"{utc()} {name}: elapsed {el:.0f} s, estimate {x['estimate_lo_s']:.0f}-{hi:.0f} s, {basis}, "
                        f"projected total {ptot:.0f} s, finish in {max(0.0, ptot - el):.0f} s"
                    )
                    notes.append(line)
                    with open(mon, "a") as fm:
                        fm.write(line + "\n")
                    run.heartbeat(stage=line[21:220])
                    if ptot >= DAY_S:
                        reason = (
                            f"projected total {ptot:.0f} s >= {DAY_S:.0f} s ({basis})"
                        )
                    elif el >= DAY_S:
                        reason = f"elapsed {el:.0f} s >= {DAY_S:.0f} s"
                    if reason:
                        status = "stopped"
                        for sig, wait in (
                            (signal.SIGTERM, KILL_GRACE_S),
                            (signal.SIGKILL, None),
                        ):
                            try:
                                os.killpg(ch.pid, sig)
                            except ProcessLookupError:
                                pass
                            try:
                                rc = ch.wait(timeout=wait)
                                break
                            except subprocess.TimeoutExpired:
                                continue
                        with open(mon, "a") as fm:
                            fm.write(f"{utc()} {name}: STOPPED, {reason}\n")
                        break
                wall = time.monotonic() - t0
                current["child"] = None
                if status is None:
                    status = "done" if rc == 0 else "error"
                fo.write(
                    f"# ----\n# end {utc()}\n# exit {rc}\n# status {status}\n# wall_s {wall:.3f}\n".encode()
                )
                if reason:
                    fo.write(f"# {reason}\n".encode())
            rec = {
                "status": status,
                "exit_code": rc,
                "driver_wall_s": wall,
                "estimate_s": [x["estimate_lo_s"], hi],
                "reason": reason,
                "last_progress": last,
                "monitor": notes,
                "log": str(log),
                "log_sha256": sha(log),
                "reading": str(dst) if dst.exists() else None,
                "pythonpath": x["pythonpath"],
            }
            if dst.exists():
                d = json.loads(dst.read_text())
                rec.update(
                    value_hex=d["value_hex"],
                    value_30=d["value_30"],
                    imag_30=d["imag_30"],
                    wall_s=d["wall_s"],
                    route=d["route"],
                    tool=d["tool"],
                    tool_sha256=d["tool_sha256"],
                    reading_sha256=sha(dst),
                    tmap={
                        k: (d.get("tmap") or {}).get(k)
                        for k in ("all_monotone", "direct_segments")
                    },
                )
            if x["stage"] == "reread":
                old = json.loads(Path(x["compare_with"]).read_text())
                same = dst.exists() and all(
                    d[k] == old[k] for k in ("value_hex", "value_30", "imag_30")
                )
                rec.update(compare_with=x["compare_with"], bit_identical=same)
                if status != "done" or not same:
                    run.record(name, rec)
                    run.heartbeat(
                        stage=f"STOP: the re-read {tag} does not reproduce its old reading ({status})"
                    )
                    raise RuntimeError(f"stop: re-read {tag}")
            run.record(name, rec)
            run.heartbeat(stage=f"{name} {status} ({wall:.0f} s)")
            counts[status] += 1
    except BaseException:
        stop.set()
        run.finish("failed", exit_code=1)
        raise
    stop.set()
    run.heartbeat(stage=f"rerun done: {counts}")
    run.finish("done", exit_code=0)


if __name__ == "__main__":
    main()
