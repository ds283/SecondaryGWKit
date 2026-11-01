"""levin-phase-floor prompt 06f: the registered run (slug ``measure``). New in 06f.

Stages, strictly in order, from var/levin-phase-floor-06f/:
1. capture: x1_capture.py (06e's, changed only in ROWS_JSON) on out/sample/selection.json, six
   workers, into state/capture/ and out/capture.json; its output in out/cmd/30_capture.txt. One
   checkpoint unit ``capture``.
2. replay: replay06f.py on every captured call whose capture matches the census bit for bit, one
   process per call, at most six at once (Z2's call first, the M2 calls with --m2), into
   out/replay/; then Z2's call once more in a fresh process (tag ``_det2``), for F0's
   determinism clause. Each process's output in out/replay_logs/. One unit per process.
3. method: method_check_06f.py on the first M2 call, output in out/cmd/31_method_check.txt. One unit.
Every child runs with the tree g first on PYTHONPATH (x1_capture.py: the tree alone, as it
requires), no bytecode, git's optional locks off, one BLAS/OpenMP thread. Heartbeat every 60 s; on
SIGTERM or SIGINT the children are sent SIGTERM, the run is recorded ``killed``, and the driver dies
by the signal. No Ray, no lock, no store opened except the scratch copy, read-only, by the capture's
census workers.

Usage: python -B run_06f.py --run-dir RUN_DIR   (launch_06f.py starts it)"""

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
G_TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1"
PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
Z2 = ("g-1e-8-0-33995", 5)
MAX_PROCS = 6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    import runregistrykit

    run = runregistrykit.Run(a.run_dir)
    children = {}

    def terminal(signum, _frame):
        for ch in list(children.values()):
            if ch.poll() is None:
                try:
                    ch.send_signal(signal.SIGTERM)
                except Exception:
                    pass
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
    env_cap = dict(base, PYTHONPATH=G_TREE)
    env_rep = dict(base, PYTHONPATH=G_TREE + ":" + str(HERE))
    ok = True
    try:
        # 1. capture
        sel = ROOT / "out" / "sample" / "selection.json"
        cap_dir = ROOT / "state" / "capture"
        cap_json = ROOT / "out" / "capture.json"
        out1 = ROOT / "out" / "cmd" / "30_capture.txt"
        for p in (cap_dir, cap_json, out1):
            if p.exists():
                raise SystemExit(f"{p} exists; not overwritten")
        run.heartbeat(stage="capture running")
        argv = [
            PYTHON,
            "-B",
            str(HERE / "x1_capture.py"),
            str(sel),
            str(cap_dir),
            str(cap_json),
            "--workers",
            str(MAX_PROCS),
        ]
        with open(out1, "wb") as fo:
            fo.write(
                (
                    " ".join(argv)
                    + f"\n# cwd {ROOT}\n# start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n# ----\n"
                ).encode()
            )
            fo.flush()
            ch = subprocess.Popen(
                argv,
                cwd=str(ROOT),
                env=env_cap,
                stdin=subprocess.DEVNULL,
                stdout=fo,
                stderr=subprocess.STDOUT,
            )
            children["capture"] = ch
            rc = ch.wait()
            children.pop("capture")
            fo.write(
                f"# ----\n# end {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n# exit {rc}\n".encode()
            )
        caps = json.loads(cap_json.read_text()) if cap_json.exists() else []
        n_match = sum(1 for e in caps if e.get("capture_matches_census"))
        run.record(
            "capture",
            {
                "status": "done" if rc == 0 else "error",
                "exit_code": rc,
                "output": str(out1),
                "output_sha256": sha(out1),
                "capture_json_sha256": sha(cap_json) if cap_json.exists() else None,
                "n_selected": len(json.loads(sel.read_text())),
                "n_matching_census": n_match,
            },
        )
        run.heartbeat(stage=f"capture exit {rc}; {n_match} matching")
        # 2. replays
        todo = []
        for e in caps:
            if not e.get("capture_matches_census"):
                continue
            todo.append((e["unit"], e["index"], e.get("sample") == "M2", ""))
        todo.sort(key=lambda t: (0 if (t[0], t[1]) == Z2 else 1, t[0], t[1]))
        if any((t[0], t[1]) == Z2 for t in todo):
            todo.append((Z2[0], Z2[1], False, "_det2"))
        rep_dir = ROOT / "out" / "replay"
        log_dir = ROOT / "out" / "replay_logs"
        log_dir.mkdir(parents=True, exist_ok=False)
        running = {}
        pending = list(todo)
        while pending or running:
            while pending and len(running) < MAX_PROCS:
                uid, idx, is_m2, tag = pending.pop(0)
                if tag and any((u, i) == (uid, idx) for (u, i, _t) in running.values()):
                    pending.insert(
                        0, (uid, idx, is_m2, tag)
                    )  # the determinism replay starts after the first ends
                    break
                name = f"{uid}_{idx}{tag}"
                argv = (
                    [
                        PYTHON,
                        "-B",
                        str(HERE / "replay06f.py"),
                        str(cap_json),
                        uid,
                        str(idx),
                        str(rep_dir),
                    ]
                    + (["--m2"] if is_m2 else [])
                    + (["--tag", tag] if tag else [])
                )
                lf = open(log_dir / f"{name}.txt", "wb")
                lf.write(
                    (
                        " ".join(argv)
                        + f"\n# start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n# ----\n"
                    ).encode()
                )
                lf.flush()
                ch = subprocess.Popen(
                    argv,
                    cwd=str(ROOT),
                    env=env_rep,
                    stdin=subprocess.DEVNULL,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                lf.close()
                children[name] = ch
                running[name] = (uid, idx, tag)
                run.heartbeat(stage=f"replay {name} started")
            for name in list(running):
                ch = children[name]
                rc = ch.poll()
                if rc is None:
                    continue
                uid, idx, tag = running.pop(name)
                children.pop(name)
                lp = log_dir / f"{name}.txt"
                with open(lp, "ab") as lf:
                    lf.write(
                        f"# ----\n# end {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n# exit {rc}\n".encode()
                    )
                line = [
                    l
                    for l in lp.read_text(errors="replace").splitlines()
                    if l.startswith("{")
                ]
                run.record(
                    f"replay {name}",
                    {
                        "status": "done" if rc == 0 else "error",
                        "exit_code": rc,
                        "log": str(lp),
                        "log_sha256": sha(lp),
                        "summary": line[-1] if line else None,
                    },
                )
                run.heartbeat(stage=f"replay {name} exit {rc}")
                ok &= rc == 0
            time.sleep(0.5)
        # 3. method check
        m2 = [
            e
            for e in caps
            if e.get("capture_matches_census") and e.get("sample") == "M2"
        ]
        out3 = ROOT / "out" / "cmd" / "31_method_check.txt"
        if m2:
            e = m2[0]
            argv = [
                PYTHON,
                "-B",
                str(HERE / "method_check_06f.py"),
                str(cap_json),
                e["unit"],
                str(e["index"]),
            ]
            with open(out3, "wb") as fo:
                fo.write((" ".join(argv) + f"\n# cwd {ROOT}\n# ----\n").encode())
                fo.flush()
                ch = subprocess.Popen(
                    argv,
                    cwd=str(ROOT),
                    env=env_rep,
                    stdin=subprocess.DEVNULL,
                    stdout=fo,
                    stderr=subprocess.STDOUT,
                )
                children["method"] = ch
                rc = ch.wait()
                children.pop("method")
                fo.write(f"# ----\n# exit {rc}\n".encode())
            run.record(
                "method",
                {
                    "status": "done" if rc == 0 else "error",
                    "exit_code": rc,
                    "output": str(out3),
                    "output_sha256": sha(out3),
                },
            )
            ok &= rc == 0
    except BaseException:
        stop.set()
        run.finish("failed", exit_code=1)
        raise
    stop.set()
    run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
