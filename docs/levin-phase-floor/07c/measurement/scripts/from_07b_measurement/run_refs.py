"""levin-phase-floor prompt 06g section 5, the references: the registered run (slug ``references``).
New in 06g, adapted from 06h's phase agent's run_06h.py (kept in state/scripts_06h_phase_reference/):
its children, heartbeat and signal handling.

**Two parts** (the coordinator, 2026-11-01: the total wall time is projected from 06i's 13 before
the rest is launched): ``--part 1`` runs stages 1-6 below over the selection, which also holds
addendum 2's V8 calls (captured and checked here, read in part 2), and ends ``done``; ``--part 2``
(stage 7, then addendum 1's V1 calls and addendum 2's V8 calls) is built after the projection.

Stages, strictly in order, from var/levin-phase-floor-06g/references/; **at most six computing
processes at any time** (one stage, one child after another; a capture or a reading runs six worker
processes under a parent that waits on them):
1. ``capture``: x1_capture.py (06c's, 06g's copy) on out/sample/selection.json (select_refs.py: Z3's
   30 and the 60 drawn), six census workers at g, into state/capture/ and out/capture.json; output
   out/cmd/30_capture.txt. Every capture must reproduce its g census record (value, abserr,
   num_regions, max_depth, converged, label).
2. ``check``: check_captures.py, the region arrays and the call records against the g census;
   out/check_captures.json, output out/cmd/31_check_captures.txt.
   **A capture that does not reproduce its census record is a stop** (the brief): nothing further
   runs and the run is recorded ``failed``.
3. ``slice``: slice_refs.py, each capture into state/inputs/<sid>.pkl; 06i's inputs checked against
   06i's SHA-256 list; out/inputs.json, output out/cmd/32_slice.txt.
4. ``compare13``: compare13.py, 06i's 13 groups at g against 06i's inputs; out/compare13.json,
   output out/cmd/33_compare13.txt. A group that differs is a stop.
5. ``routes``: routes_refs.py, every capture's route and Route B's default edges; out/routes.json,
   output out/cmd/34_routes.txt. A capture whose edges cannot be found is a stop.
6. **the tool on 06i's 13**: one_reading.py on each of the 13 captures, four readings each (QAWO and
   FCC, resolutions 1 and 2, the tool's order), into out/readings_06i13/; then the tool's own command
   line on two of them (compare06i.CLI_SAMPLES) into out/cli/; then compare06i.py, out/tool_vs_06i.json,
   output out/cmd/35_tool_vs_06i.txt. **A reading that disagrees with 06i's is a stop** (prompt 06g
   section 7), as is a reading that fails or exceeds 30 minutes here.
7. **the references**: one_reading.py on each of the 77 new samples (Z3's 17 not in 06i, then the 60
   drawn), four readings each, into out/readings/<sid>-<impl>-r<res>.json. Each reading is one
   registry unit, with its status (``done``, ``timeout``, ``error``) and wall time. **A reading over
   30 minutes is stopped** (its process group sent SIGTERM, then SIGKILL after 30 s), recorded
   ``timeout``, and the sample's other readings still run; the sample's reference is dropped at
   scoring. No reading's failure stops the run.
Every reading's own output goes to out/reading_logs/<set>/<sid>-<impl>-r<res>.txt.

Every child runs with no bytecode, git's optional locks off and one BLAS/OpenMP thread; x1_capture.py
with PYTHONPATH the tree g alone, as it requires; the others with the tree g, then scripts/ (the
pickles name the census harness's functions). Heartbeat every 60 s and at each child's start and end;
on SIGTERM or SIGINT the running child's process group is sent SIGTERM, the run is recorded
``killed``, and the driver dies by the signal. No Ray, no lock, no store opened except the scratch
copy, read-only, by the capture's census workers; nothing is written under var/datastores/.

Usage: python -B run_refs.py --run-dir RUN_DIR --part 1|2   (launch_refs.py starts it)
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
from compare06i import CLI_SAMPLES  # noqa: E402

PYTHON = str(RC.PYTHON)
MAX_PROCS = 6
KILL_GRACE_S = 30.0
DAY_S = 86400.0  # addendum 3: "If it's not projected to finish for a day, then we would kill it"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def plan(selection):
    """(the 13 of 06i, the 77 new, addendum 2's V8 calls not among the 90), each a list of
    selection entries in run order."""
    i06 = [e for e in selection if e["in_06i"]]
    z_new = [e for e in selection if e["source"] == "z3" and not e["in_06i"]]
    drawn = [e for e in selection if e["source"] == "draw"]
    v8 = [e for e in selection if e["source"] == "v8"]
    assert len(i06) == 13 and len(z_new) == 17 and len(drawn) == 60
    assert len(selection) == 90 + len(v8)
    return i06, z_new + drawn, v8


def expected_units(part, n13=13, n77=77, n_v8=12):
    if part == "1":
        return 5 + 4 * n13 + len(CLI_SAMPLES) + 1
    return json.loads((ROOT / "out" / "part2_plan.json").read_text())[
        "n_readings"
    ]  # addendum 3


def part2(run, current, env):
    """Addendum 3 (the user's ruling, 2026-11-01): out/part2_plan.json's readings in order
    (make_part2_plan.py), each in its own process group, never stopped for passing 30 minutes. From
    max(30 min, its projection) on, every 5 minutes, its progress (one_reading.py --progress: chunks
    done of total) and projected finish (start + elapsed x total / done) go to the heartbeat and
    out/reading_monitor.log. **The backstop:** it is stopped (SIGTERM to its group, SIGKILL after
    30 s), recorded ``stopped`` with the reason, only if (a) after 30 minutes its progress projects
    a total of DAY_S or more, (b) its elapsed time reaches its ceiling, max(6 x projection,
    projection + 1 h), or (c) DAY_S. Each reading is one registry unit."""
    doc = json.loads((ROOT / "out" / "part2_plan.json").read_text())
    mon = ROOT / "out" / "reading_monitor.log"
    counts = {"done": 0, "error": 0, "stopped": 0}
    for x in doc["plan"]:
        tag = f"{x['sid']}-{x['impl']}-r{x['res']}"
        dst = ROOT / "out" / "readings" / f"{tag}.json"
        log = ROOT / "out" / "reading_logs" / x["stage"] / f"{tag}.txt"
        prog = ROOT / "out" / "progress" / f"{tag}.json"
        for p_ in (dst, log, prog):
            if p_.exists():
                raise SystemExit(f"{p_} exists; not overwritten")
            p_.parent.mkdir(parents=True, exist_ok=True)
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
            str(MAX_PROCS),
            "--progress",
            str(prog),
        ]
        name = f"{x['stage']} {x['sid']} {x['impl']} r{x['res']}"
        proj, ceil_ = x["projection_s"], x["ceiling_s"]
        run.heartbeat(stage=f"{name} running (projection {proj:.0f} s)")
        notes = []
        with open(log, "wb") as fo:
            fo.write(
                (
                    " ".join(argv)
                    + f"\n# cwd {ROOT}\n# start {utc()}\n# projection_s {proj:.1f} ceiling_s {ceil_:.1f}\n# ----\n"
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
            current.update(child=ch, group=True)
            status, reason, last = None, None, None
            next_log = max(RC.READING_LIMIT_S, proj)
            while True:
                try:
                    rc = ch.wait(timeout=5.0)
                    break
                except subprocess.TimeoutExpired:
                    pass
                el = time.monotonic() - t0
                if el < next_log:
                    continue
                next_log = el + 300.0
                try:
                    last = json.loads(prog.read_text())
                except Exception:
                    last = None
                done, total = (last or {}).get("done"), (last or {}).get("total")
                ptot = el * total / done if done else None
                line = (
                    f"{utc()} {name}: elapsed {el:.0f} s, projection {proj:.0f} s, ceiling {ceil_:.0f} s, "
                    f"progress {done}/{total} chunks, projected total "
                    + (
                        f"{ptot:.0f} s (finish in {ptot - el:.0f} s)"
                        if ptot
                        else "unknown (no chunk back yet)"
                    )
                )
                notes.append(line)
                with open(mon, "a") as fm:
                    fm.write(line + "\n")
                run.heartbeat(stage=line[21:200])
                if el >= RC.READING_LIMIT_S and ptot is not None and ptot >= DAY_S:
                    reason = (
                        f"backstop (a): projected total {ptot:.0f} s >= {DAY_S:.0f} s"
                    )
                elif el >= ceil_:
                    reason = f"backstop (b): elapsed {el:.0f} s reached the ceiling {ceil_:.0f} s"
                elif el >= DAY_S:
                    reason = f"backstop (c): elapsed {el:.0f} s"
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
            current.update(child=None, group=False)
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
            "projection_s": proj,
            "ceiling_s": ceil_,
            "reason": reason,
            "last_progress": last,
            "monitor": notes,
            "log": str(log),
            "log_sha256": sha(log),
            "reading": str(dst) if dst.exists() else None,
        }
        if dst.exists():
            d = json.loads(dst.read_text())
            rec.update(
                value_hex=d["value_hex"],
                value_30=d["value_30"],
                wall_s=d["wall_s"],
                route=d["route"],
                reading_sha256=sha(dst),
            )
        run.record(name, rec)
        run.heartbeat(stage=f"{name} {status} ({wall:.0f} s)")
        counts[status] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--part", choices=("1", "2"), required=True)
    a = ap.parse_args()
    import runregistrykit

    run = runregistrykit.Run(a.run_dir)
    current = {"child": None, "group": False}

    def kill_child(sig):
        ch = current["child"]
        if ch is None or ch.poll() is not None:
            return
        try:
            if current["group"]:
                os.killpg(ch.pid, sig)
            else:
                ch.send_signal(sig)
        except ProcessLookupError:
            pass

    def terminal(signum, _frame):
        kill_child(signal.SIGTERM)
        run.finish("killed", exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, terminal)
    run.heartbeat(pid=os.getpid(), stage="started: capture is next")
    stop = threading.Event()

    def beat():
        while not stop.wait(60.0):
            try:
                run.heartbeat()
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()
    base = dict(os.environ)
    base.update(
        PYTHONDONTWRITEBYTECODE="1",
        GIT_OPTIONAL_LOCKS="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env_tree = dict(base, PYTHONPATH=str(RC.G_TREE))
    env_rd = dict(base, PYTHONPATH=str(RC.G_TREE) + ":" + str(HERE))

    def one(argv, env, out, limit=None):
        """Run one child to completion in its own process group, its output in ``out``; stop it at
        ``limit`` seconds. Returns (status, exit code, wall seconds)."""
        if out.exists():
            raise SystemExit(f"{out} exists; not overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fo:
            fo.write(
                (" ".join(argv) + f"\n# cwd {ROOT}\n# start {utc()}\n# ----\n").encode()
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
            current.update(child=ch, group=True)
            status = None
            try:
                rc = ch.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                status = "timeout"
                try:
                    os.killpg(ch.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    rc = ch.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(ch.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = ch.wait()
            wall = time.monotonic() - t0
            current.update(child=None, group=False)
            if status is None:
                status = "done" if rc == 0 else "error"
            fo.write(
                f"# ----\n# end {utc()}\n# exit {rc}\n# status {status}\n# wall_s {wall:.3f}\n".encode()
            )
        return status, rc, wall

    def stage(name, argv, env, out_name):
        out = ROOT / "out" / "cmd" / out_name
        run.heartbeat(stage=f"{name} running")
        status, rc, wall = one(argv, env, out)
        run.record(
            name,
            {
                "status": status,
                "exit_code": rc,
                "wall_s": wall,
                "output": str(out),
                "output_sha256": sha(out),
            },
        )
        run.heartbeat(stage=f"{name} exit {rc}")
        return rc

    def reading(set_name, e, impl, res, rd_dir):
        sid = e["sid"]
        pkl = ROOT / "state" / "inputs" / f"{sid}.pkl"
        dst = rd_dir / f"{sid}-{impl}-r{res}.json"
        log = ROOT / "out" / "reading_logs" / set_name / f"{sid}-{impl}-r{res}.txt"
        argv = [
            PYTHON,
            "-B",
            str(HERE / "one_reading.py"),
            str(pkl),
            impl,
            str(res),
            str(dst),
            "--sid",
            sid,
            "--unit",
            e["unit"],
            "--index",
            str(e["index"]),
            "--rtol",
            e["rtol"],
            "--workers",
            str(MAX_PROCS),
        ]
        name = f"{set_name} {sid} {impl} r{res}"
        run.heartbeat(stage=f"{name} running")
        status, rc, wall = one(argv, env_rd, log, limit=RC.READING_LIMIT_S)
        rec = {
            "status": status,
            "exit_code": rc,
            "driver_wall_s": wall,
            "limit_s": RC.READING_LIMIT_S,
            "log": str(log),
            "log_sha256": sha(log),
            "reading": str(dst) if dst.exists() else None,
        }
        if dst.exists():
            d = json.loads(dst.read_text())
            rec.update(
                value_hex=d["value_hex"],
                value_30=d["value_30"],
                wall_s=d["wall_s"],
                route=d["route"],
                reading_sha256=sha(dst),
            )
        run.record(name, rec)
        run.heartbeat(stage=f"{name} {status} ({wall:.0f} s)")
        return status

    ok = True
    try:
        sel_p = ROOT / "out" / "sample" / "selection.json"
        selection = json.loads(sel_p.read_text())
        i06, new77, v8 = plan(selection)
        if a.part == "2":
            counts = part2(run, current, env_rd)
            run.heartbeat(stage=f"part 2 done: {counts}")
            stop.set()
            run.finish("done", exit_code=0)
            return  # not sys.exit: SystemExit would reach the except below and record "failed"
        # 1-2. the captures, and their check against the census
        for p_ in (ROOT / "state" / "capture", ROOT / "out" / "capture.json"):
            if p_.exists():
                raise SystemExit(f"{p_} exists; not overwritten")
        rc = stage(
            "capture",
            [
                PYTHON,
                "-B",
                str(HERE / "x1_capture.py"),
                str(sel_p),
                str(ROOT / "state" / "capture"),
                str(ROOT / "out" / "capture.json"),
                "--workers",
                str(MAX_PROCS),
            ],
            env_tree,
            "30_capture.txt",
        )
        if rc != 0:
            run.heartbeat(
                stage="STOP: a capture does not reproduce its census record (x1_capture)"
            )
            raise RuntimeError("stop: capture")
        rc = stage(
            "check",
            [
                PYTHON,
                "-B",
                str(HERE / "check_captures.py"),
                str(ROOT / "out" / "capture.json"),
                str(sel_p),
                str(ROOT / "out" / "check_captures.json"),
            ],
            env_rd,
            "31_check_captures.txt",
        )
        if rc != 0:
            run.heartbeat(
                stage="STOP: a capture does not reproduce its census record (regions)"
            )
            raise RuntimeError("stop: check")
        # 3-5. slice, 06i's 13 groups, the routes
        rc = stage(
            "slice",
            [
                PYTHON,
                "-B",
                str(HERE / "slice_refs.py"),
                str(ROOT / "out" / "capture.json"),
                str(ROOT / "out" / "check_captures.json"),
                str(sel_p),
            ],
            base,
            "32_slice.txt",
        )
        if rc != 0:
            run.heartbeat(stage="STOP: slicing or 06i's inputs' SHA-256")
            raise RuntimeError("stop: slice")
        rc = stage(
            "compare13",
            [
                PYTHON,
                "-B",
                str(HERE / "compare13.py"),
                str(ROOT / "out" / "inputs.json"),
            ],
            env_rd,
            "33_compare13.txt",
        )
        if rc != 0:
            run.heartbeat(stage="STOP: one of 06i's 13 groups differs at g")
            raise RuntimeError("stop: compare13")
        rc = stage(
            "routes",
            [
                PYTHON,
                "-B",
                str(HERE / "routes_refs.py"),
                str(ROOT / "out" / "inputs.json"),
            ],
            env_rd,
            "34_routes.txt",
        )
        if rc != 0:
            run.heartbeat(stage="STOP: a capture's Route B edges cannot be found")
            raise RuntimeError("stop: routes")
        # 6. the tool on 06i's 13, its command line on two, and the comparison with 06i
        rd13 = ROOT / "out" / "readings_06i13"
        bad13 = []
        for e in i06:
            for impl in RC.IMPLS:
                for res in RC.RESS:
                    st = reading("06i13", e, impl, res, rd13)
                    if st != "done":
                        bad13.append(f"{e['sid']} {impl} r{res} {st}")
        if bad13:
            run.heartbeat(stage=f"STOP: readings on 06i's 13 failed: {bad13}")
            raise RuntimeError(f"stop: readings on 06i's 13: {bad13}")
        by_no = {e["sample_no"]: e for e in i06}
        for n in CLI_SAMPLES:
            e = by_no[n]
            g = e["g_record"]
            h = e["06h_record"]
            tol = min(RC.H(g["abserr"]), RC.H(h["abserr"])) / 100
            out_json = ROOT / "out" / "cli" / f"{e['sid']}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            argv = [
                PYTHON,
                "-B",
                "-m",
                "tools.filon_reference",
                str(ROOT / "state" / "inputs" / f"{e['sid']}.pkl"),
                "--tolerance",
                repr(tol),
                "--unpickler",
                "ray.cloudpickle",
                "--workers",
                str(MAX_PROCS),
                "--out",
                str(out_json),
            ]
            name = f"cli {e['sid']}"
            run.heartbeat(stage=f"{name} running")
            status, rc, wall = one(
                argv,
                env_rd,
                ROOT / "out" / "reading_logs" / "cli" / f"{e['sid']}.txt",
                limit=4 * RC.READING_LIMIT_S,
            )
            run.record(
                name,
                {
                    "status": status,
                    "exit_code": rc,
                    "driver_wall_s": wall,
                    "out": str(out_json) if out_json.exists() else None,
                },
            )
            if status != "done":
                run.heartbeat(stage=f"STOP: the command line on {e['sid']} {status}")
                raise RuntimeError(f"stop: cli {e['sid']}")
        rc = stage(
            "tool_vs_06i",
            [
                PYTHON,
                "-B",
                str(HERE / "compare06i.py"),
                str(rd13),
                str(ROOT / "out" / "cli"),
            ],
            base,
            "35_tool_vs_06i.txt",
        )
        if rc != 0:
            run.heartbeat(
                stage="STOP: a reading disagrees with 06i's (or the command line differs)"
            )
            raise RuntimeError("stop: tool_vs_06i")
        run.heartbeat(
            stage="part 1 done: 06i's 13 agree; the cost projection comes next"
        )
    except BaseException:
        stop.set()
        run.finish("failed", exit_code=1)
        raise
    stop.set()
    run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
