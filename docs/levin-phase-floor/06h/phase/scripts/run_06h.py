"""levin-phase-floor prompt 06h section 5, the phase re-measurement: the registered run (slug ``phase``).
New in 06h, adapted from 06f's run_06f.py (its stages, children, heartbeat and signal handling).

Stages, strictly in order, from var/levin-phase-floor-06h/phase/; at most six child processes at
any time, one stage after another:
1. capture at p: x1_capture.py --tree p on out/sample/selection_p.json (select_06h.py; 06e's census
   record), six workers, into state/capture_p/ and out/capture_p.json; output
   out/cmd/30_capture_p.txt. Unit ``capture p``.
2. capture at h: the same with --tree h on selection_h.json (06h's census record), into
   state/capture_h/ and out/capture_h.json; output out/cmd/31_capture_h.txt. Unit ``capture h``.
   **A capture that does not reproduce its census record is a stop** (the dispatch brief): if any
   of the 40 does not, nothing further runs and the run is recorded ``failed``.
3. the method check: method_check_06f.py (06h's copy, at the exact e^u) on h's capture of
   ``h-1e-7-0-12689`` #19 (06f's worst call), output out/cmd/32_method_check.txt, **before any
   scoring**: if it fails nothing further runs. Unit ``method``.
4. the replays: replay06f.py --m2 (06h's copy) on every call in each tree, one process per call,
   into out/replay_<tree>/, each process's output in out/replay_logs/<tree>/. One unit per process.
5. the cross scoring: cross06h.py, each tree's code at the other tree's points (p at h's: H1's
   parent column on H1's points; h at p's: descriptive), into out/cross/<tree>_at_<other>/, output
   in out/cross_logs/. One unit per process.
6. the reference at 50 digits: orch_m2.py worst h 5 (06f's review method, 06h's copy) at h's worst
   point, output out/cmd/33_refcheck_50_digits.txt. Unit ``refcheck``.
Every child runs with its tree first on PYTHONPATH (x1_capture.py: the tree alone, as it requires;
the others the tree then scripts/, for the pickles' harness functions), no bytecode, git's optional
locks off, one BLAS/OpenMP thread. Heartbeat every 60 s; on SIGTERM or SIGINT the children are sent
SIGTERM, the run is recorded ``killed``, and the driver dies by the signal. No Ray, no lock, no store
opened except the scratch copy, read-only, by the capture's census workers.

Usage: python -B run_06h.py --run-dir RUN_DIR   (launch_06h.py starts it)"""

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
from prov import TREES  # noqa: E402

PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
METHOD_CALL = ("h-1e-7-0-12689", 19)
MAX_PROCS = 6


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
    run.heartbeat(pid=os.getpid(), stage="started: capture at p is next")
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
    env_cap = {t: dict(base, PYTHONPATH=str(TREES[t][0])) for t in TREES}
    env_rep = {
        t: dict(base, PYTHONPATH=str(TREES[t][0]) + ":" + str(HERE)) for t in TREES
    }

    def one(name, argv, env, out, unit):
        """Run one child to completion, its output in ``out``; record the unit."""
        if out.exists():
            raise SystemExit(f"{out} exists; not overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fo:
            fo.write(
                (" ".join(argv) + f"\n# cwd {ROOT}\n# start {utc()}\n# ----\n").encode()
            )
            fo.flush()
            ch = subprocess.Popen(
                argv,
                cwd=str(ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=fo,
                stderr=subprocess.STDOUT,
            )
            children[name] = ch
            rc = ch.wait()
            children.pop(name)
            fo.write(f"# ----\n# end {utc()}\n# exit {rc}\n".encode())
        return rc

    def pool(jobs, stage):
        """jobs: [(name, argv, env, log path)]; at most MAX_PROCS at once. Returns {name: rc}."""
        running = {}
        rcs = {}
        pending = list(jobs)
        while pending or running:
            while pending and len(running) < MAX_PROCS:
                name, argv, env, lp = pending.pop(0)
                if lp.exists():
                    raise SystemExit(f"{lp} exists; not overwritten")
                lp.parent.mkdir(parents=True, exist_ok=True)
                lf = open(lp, "wb")
                lf.write((" ".join(argv) + f"\n# start {utc()}\n# ----\n").encode())
                lf.flush()
                ch = subprocess.Popen(
                    argv,
                    cwd=str(ROOT),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                lf.close()
                children[name] = ch
                running[name] = lp
                run.heartbeat(stage=f"{stage} {name} started")
            for name in list(running):
                rc = children[name].poll()
                if rc is None:
                    continue
                lp = running.pop(name)
                children.pop(name)
                with open(lp, "ab") as lf:
                    lf.write(f"# ----\n# end {utc()}\n# exit {rc}\n".encode())
                line = [
                    l
                    for l in lp.read_text(errors="replace").splitlines()
                    if l.startswith("{")
                ]
                run.record(
                    f"{stage} {name}",
                    {
                        "status": "done" if rc == 0 else "error",
                        "exit_code": rc,
                        "log": str(lp),
                        "log_sha256": sha(lp),
                        "summary": line[-1] if line else None,
                    },
                )
                run.heartbeat(stage=f"{stage} {name} exit {rc}")
                rcs[name] = rc
            time.sleep(0.5)
        return rcs

    ok = True
    try:
        # 1-2. the captures, p then h
        caps = {}
        for i, tree in enumerate(("p", "h")):
            sel = ROOT / "out" / "sample" / f"selection_{tree}.json"
            cap_dir = ROOT / "state" / f"capture_{tree}"
            cap_json = ROOT / "out" / f"capture_{tree}.json"
            out1 = ROOT / "out" / "cmd" / f"3{i}_capture_{tree}.txt"
            for p_ in (cap_dir, cap_json):
                if p_.exists():
                    raise SystemExit(f"{p_} exists; not overwritten")
            run.heartbeat(stage=f"capture at {tree} running")
            argv = [
                PYTHON,
                "-B",
                str(HERE / "x1_capture.py"),
                str(sel),
                str(cap_dir),
                str(cap_json),
                "--tree",
                tree,
                "--workers",
                str(MAX_PROCS),
            ]
            rc = one(f"capture {tree}", argv, env_cap[tree], out1, f"capture {tree}")
            caps[tree] = json.loads(cap_json.read_text()) if cap_json.exists() else []
            n_sel = len(json.loads(sel.read_text()))
            n_match = sum(1 for e in caps[tree] if e.get("capture_matches_census"))
            run.record(
                f"capture {tree}",
                {
                    "status": "done" if rc == 0 else "error",
                    "exit_code": rc,
                    "output": str(out1),
                    "output_sha256": sha(out1),
                    "capture_json_sha256": sha(cap_json) if cap_json.exists() else None,
                    "n_selected": n_sel,
                    "n_matching_census": n_match,
                },
            )
            run.heartbeat(
                stage=f"capture at {tree} exit {rc}; {n_match} of {n_sel} matching"
            )
            ok &= rc == 0 and n_match == n_sel and len(caps[tree]) == n_sel
        if not ok:
            run.heartbeat(stage="STOP: a capture does not reproduce its census record")
            raise RuntimeError("a capture does not reproduce its census record: stop")
        # 3. the method check, before any scoring
        out3 = ROOT / "out" / "cmd" / "32_method_check.txt"
        argv = [
            PYTHON,
            "-B",
            str(HERE / "method_check_06f.py"),
            str(ROOT / "out" / "capture_h.json"),
            METHOD_CALL[0],
            str(METHOD_CALL[1]),
        ]
        run.heartbeat(stage="method check running")
        rc = one("method", argv, env_rep["h"], out3, "method")
        run.record(
            "method",
            {
                "status": "done" if rc == 0 else "error",
                "exit_code": rc,
                "output": str(out3),
                "output_sha256": sha(out3),
            },
        )
        if rc != 0:
            run.heartbeat(stage="STOP: the method check fails")
            raise RuntimeError("the method check fails: stop")
        # 4. the replays
        jobs = []
        for tree in ("h", "p"):
            for e in caps[tree]:
                name = f"{e['unit']}_{e['index']}"
                argv = [
                    PYTHON,
                    "-B",
                    str(HERE / "replay06f.py"),
                    str(ROOT / "out" / f"capture_{tree}.json"),
                    e["unit"],
                    str(e["index"]),
                    str(ROOT / "out" / f"replay_{tree}"),
                    "--m2",
                ]
                jobs.append(
                    (
                        name,
                        argv,
                        env_rep[tree],
                        ROOT / "out" / "replay_logs" / tree / f"{name}.txt",
                    )
                )
        rcs = pool(jobs, "replay")
        ok &= all(rc == 0 for rc in rcs.values()) and len(rcs) == len(jobs)
        # 5. the cross scoring (needs both trees' replays of a call)
        jobs = []
        by_rest = {
            t: {(e["unit"].split("-", 1)[1], e["index"]): e for e in caps[t]}
            for t in caps
        }
        for tree, other in (("p", "h"), ("h", "p")):
            for k, e in sorted(by_rest[tree].items()):
                eo = by_rest[other].get(k)
                if eo is None:
                    continue
                name = f"{e['unit']}_{e['index']}"
                own_dir = ROOT / "out" / f"replay_{tree}" / name
                other_dir = (
                    ROOT / "out" / f"replay_{other}" / f"{eo['unit']}_{eo['index']}"
                )
                if (
                    not (own_dir / "result.json").exists()
                    or not (other_dir / "result.json").exists()
                ):
                    continue
                argv = [
                    PYTHON,
                    "-B",
                    str(HERE / "cross06h.py"),
                    str(ROOT / "out" / f"capture_{tree}.json"),
                    e["unit"],
                    str(e["index"]),
                    str(own_dir),
                    str(other_dir),
                    str(ROOT / "out" / "cross" / f"{tree}_at_{other}"),
                ]
                jobs.append(
                    (
                        f"{tree}_at_{other} {name}",
                        argv,
                        env_rep[tree],
                        ROOT
                        / "out"
                        / "cross_logs"
                        / f"{tree}_at_{other}"
                        / f"{name}.txt",
                    )
                )
        rcs = pool(jobs, "cross")
        ok &= all(rc == 0 for rc in rcs.values()) and len(rcs) == 2 * len(caps["h"])
        # 6. the reference at 50 digits at h's worst point
        out6 = ROOT / "out" / "cmd" / "33_refcheck_50_digits.txt"
        argv = [PYTHON, "-B", str(HERE / "orch_m2.py"), "worst", "h", "5"]
        run.heartbeat(stage="refcheck running")
        rc = one("refcheck", argv, env_rep["h"], out6, "refcheck")
        run.record(
            "refcheck",
            {
                "status": "done" if rc == 0 else "error",
                "exit_code": rc,
                "output": str(out6),
                "output_sha256": sha(out6),
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
