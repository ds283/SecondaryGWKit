"""levin-phase-floor prompt 06e section 5, Z3: samples 3 and 15's references, the registered run (slug
``references``). New in 06e. Runs orch_ref.py (06c's orchestrator's, copied unchanged) on each
sample at P = 128 with 64/96 nodes, then at P = 64 with 96/128 nodes, six workers, strictly one
invocation at a time, in the environment of run_short.sh (the tree g on PYTHONPATH, no bytecode,
git's optional locks off, one BLAS/OpenMP thread per process). Each invocation's output goes to
references/out/ref-sNN-PXX.txt and is recorded as one checkpoint unit ``sNN-PXX`` (exit code,
output SHA-256, its S lines). Heartbeat every 60 s; on SIGTERM or SIGINT the child is sent SIGTERM,
the run is recorded ``killed``, and the driver dies by the signal.

Usage: python -B run_refs.py --run-dir RUN_DIR   (launch_refs.py starts it)"""
import argparse
import hashlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
G_TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad"
PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SAMPLES = ("03", "15")
RESOLUTIONS = ((128, 64, 96), (64, 96, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    import runregistrykit

    run = runregistrykit.Run(a.run_dir)
    current = {}

    def terminal(signum, _frame):
        ch = current.get("child")
        if ch is not None and ch.poll() is None:
            try:
                ch.send_signal(signal.SIGTERM)
            except Exception:
                pass
        run.finish("killed", exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, terminal)
    run.heartbeat(pid=os.getpid(), stage="started: s03 at P=128 is next")
    stop = threading.Event()

    def beat():
        while not stop.wait(60.0):
            try:
                run.heartbeat()
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()
    env = dict(os.environ)
    # the census scripts after the tree: the captured pickles name harness._rebuild_lru_cache
    env.update(PYTHONPATH=G_TREE + ":/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/census/scripts", PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")
    ok = True
    try:
        for n in SAMPLES:
            for P, n1, n2 in RESOLUTIONS:
                unit = f"s{n}-P{P}"
                out = HERE / "out" / f"ref-s{n}-P{P}.txt"
                if out.exists():
                    raise SystemExit(f"{out} exists; not overwritten")
                run.heartbeat(stage=f"{unit} running")
                with open(out, "wb") as fo:
                    ch = subprocess.Popen([PYTHON, "-B", "orch_ref.py", f"inputs/s{n}.pkl", "--P", str(P), "--n1", str(n1),
                                           "--n2", str(n2), "--workers", "6"], cwd=str(HERE), env=env,
                                          stdin=subprocess.DEVNULL, stdout=fo, stderr=subprocess.STDOUT)
                    current["child"] = ch
                    rc = ch.wait()
                    current["child"] = None
                text = out.read_text(errors="replace")
                s_lines = [l for l in text.splitlines() if l.startswith("S") or l.startswith("driver") or l.startswith("wall")]
                run.record(unit, {"status": "done" if rc == 0 else "error", "exit_code": rc, "output": str(out),
                                  "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "lines": s_lines})
                run.heartbeat(stage=f"{unit} exit {rc}")
                ok &= rc == 0
    except BaseException:
        run.finish("failed", exit_code=1)
        raise
    run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
