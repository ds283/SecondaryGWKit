"""
a3-v2-readiness prompt 04: a SIGTERM to a pipeline driver blocked in ``ray.wait`` is recorded
``killed``/143 at once, then gains its fingerprint, and the process exits 143 leaving nothing behind.
Measured on real local Ray instances.

Each run is a child process (``--child VARIANT DRIVER ROOT STORE``), built as ``sigterm_probe.py``
builds its children. The child:

- builds a small store in a temporary directory (``Datastore/tests/real_store_fixtures.py``
  ``build_real_store``) and registers a ``Run`` with ``RunRegistry.begin(..., results=<that store>,
  root=<a temporary directory>)``. Nothing is written under ``var/``;
- replaces ``ray.init`` with the **real** ``_local_ray_init`` of one driver, read out of its source
  with ``ast`` (``scoped_pipeline_run.main``'s or ``quadsource_atol_sweep.run_build``'s, alternating
  between runs) and compiled against the real ``ray.init``, so that it starts Ray with the driver's
  own arguments (``address="local"`` included) and calls the real
  ``scoped_pipeline_run.install_killed_handler``;
- stands in for ``main.py``: calls ``ray.init()``, submits one task that sleeps, and blocks in
  ``ray.wait`` with no timeout, as ``RayWorkPool`` does. A timer thread sends the child ``SIGTERM``
  2 s later;
- ends as both drivers end: ``except BaseException`` → the real
  ``scoped_pipeline_run.finish_after_exec(run, exc, latch, ray.shutdown)`` → ``raise``.

Variants (prompt §3), each run three times:

- ``plain``: the fingerprint as it is;
- ``slow-fingerprint``: the fingerprint sleeps 8 s before it starts (no abort may come);
- ``second-sigterm``: the fingerprint sleeps 4 s first, and a second ``SIGTERM`` is sent 1.5 s into
  that sleep.

The *runner* polls the run's ``status.json`` every 5 ms while the child runs, recording each
distinct ``(state, exit_code, fingerprint present)`` it sees with the file's mtime, so that the
time ``killed`` was first written is the mtime of the first ``killed`` it read, less the time the
child sent the signal. Around each run it lists every ``gcs_server|raylet|ray::`` process before and
after (``pgrep -lf``: a listing only; another project's cluster may appear in it), and records the
child's own descendants while it runs (``ps``, any command, new since before the child started).
After the child exits it checks which of its own processes still exist at +0.5, +2, +5 and +10 s,
then kills each survivor of its own with ``SIGTERM`` and, 3 s later, ``SIGKILL``, re-checking the
command line before each signal and printing every pid. **It never signals a process that is not a
descendant of its child.** The output ends with a line per run, the acceptance verdict, the Ray
processes before and after the whole probe, and every pid killed.

The process bookkeeping (``ray_procs``, ``all_procs``, ``descendants``, ``exists``, ``fmt``,
``short``, ``c_handler``) is ``sigterm_probe.py``'s, imported by path, not copied.

Usage, from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/sigterm_handler_probe.py \\
        > docs/a3-v2-readiness/sigterm_handler_probe.txt 2>&1

Then one more child, the ``phase`` section (added by prompt 04 when two of the first nine runs
wrote ``killed`` 1.005 s and 1.006 s after the signal): one Ray instance through the ``scoped``
wrapper, and one ``SIGTERM`` per offset in ``PHASE_OFFSETS``, each sent that long after the main
thread enters ``ray.wait``, with the real handler installed afresh before each. It reports how
long after the signal the handler's ``SystemExit`` reached the frame. It registers no run, and
is not counted in the acceptance line.

``--variants plain,slow-fingerprint``, ``--repeat N`` and ``--no-phase`` run a subset (the
deliberate-breakage runs of prompt 04's log); ``--variants ''`` runs the phase section alone.
"""

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCOPED_DRIVER = REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py"
SWEEP_DRIVER = REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py"
DRIVERS = {
    "scoped": (SCOPED_DRIVER, "main"),
    "run_build": (SWEEP_DRIVER, "run_build"),
}
VARIANTS = {
    "plain": "the fingerprint as it is",
    "slow-fingerprint": "the fingerprint sleeps 8 s before it starts",
    "second-sigterm": "the fingerprint sleeps 4 s first; a second SIGTERM 1.5 s into it",
}
PHASE = (
    "how long after a SIGTERM the handler's SystemExit reaches the frame around ray.wait, "
    "against when in the wait the signal lands; no run is registered"
)
PHASE_OFFSETS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0, 2.05, 2.5)
SIGNAL_AT = 2.0
BANNER = "*** SIGTERM received"
RAY_ADDRESS_FILE = Path("/tmp/ray/ray_current_cluster")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the audit's probe, for its process bookkeeping only; its own runner is not used
_SP = _load("_sigterm_probe", HERE / "sigterm_probe.py")
ray_procs, all_procs, descendants = _SP.ray_procs, _SP.all_procs, _SP.descendants
exists, fmt, short, c_handler = _SP.exists, _SP.fmt, _SP.short, _SP.c_handler
PATTERN = _SP.PATTERN


# ------------------------------------------------------------------------------------------------
# the child
# ------------------------------------------------------------------------------------------------

T_SIGNAL = [None]


def say(*parts):
    t = "t=-" if T_SIGNAL[0] is None else f"t={time.time() - T_SIGNAL[0]:+.3f}s"
    print(t, *parts, flush=True)


def driver_wrapper(which, namespace):
    """The driver's own `_local_ray_init`, read out of its source and compiled in `namespace`.

    Its free names (`args`, `_real_ray_init`, `run`, `latch`, and `install_killed_handler` or
    `scoped`) are looked up there; nothing in its body is changed."""
    path, outer = DRIVERS[which]
    tree = ast.parse(path.read_text())
    outer_def = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == outer
    )
    wrapper = next(
        n
        for n in ast.walk(outer_def)
        if isinstance(n, ast.FunctionDef) and n.name == "_local_ray_init"
    )
    code = compile(ast.Module(body=[wrapper], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace["_local_ray_init"], ast.unparse(wrapper)


def child(variant, which, root, store_dir):
    import argparse as _argparse

    import ray

    import RunRegistry
    import RunRegistry.stores as registry_stores
    from StoreFixtures.real_store_fixtures import build_real_store

    scoped = _load("_scoped_pipeline_run", SCOPED_DRIVER)
    say(
        f"child pid={os.getpid()} variant={variant} wrapper={which} ray={ray.__version__}"
    )
    store = build_real_store(Path(store_dir))
    run = RunRegistry.begin(
        campaign="a3-v2-readiness",
        prompt="04",
        slug=f"probe-{variant}",
        purpose="sigterm_handler_probe.py: a SIGTERM in ray.wait, in a temporary run root",
        argv=sys.argv,
        script=__file__,
        results=str(store.primary),
        root=root,
    )
    say(f"RUN {run.path}")
    latch = scoped.SignalLatch()

    namespace = {
        "args": _argparse.Namespace(cpus=1),
        "_real_ray_init": ray.init,
        "run": run,
        "latch": latch,
        "install_killed_handler": scoped.install_killed_handler,
        "scoped": scoped,
    }
    wrapper, source = driver_wrapper(which, namespace)
    say(f"the {which} driver's _local_ray_init, as compiled here:")
    for line in source.splitlines():
        say(f"    | {line}")
    ray.init = wrapper

    if variant in ("slow-fingerprint", "second-sigterm"):
        real_fingerprint = registry_stores.fingerprint_store
        pause = 8.0 if variant == "slow-fingerprint" else 4.0

        def slow_fingerprint(*a, **kw):
            say(f"fingerprint: sleeping {pause:g} s before it starts")
            if variant == "second-sigterm":

                def second():
                    say("SENDING the second SIGTERM")
                    os.kill(os.getpid(), signal.SIGTERM)

                threading.Timer(1.5, second).start()
            t0 = time.monotonic()
            while time.monotonic() - t0 < pause:
                time.sleep(0.5)
                say(
                    f"alive, {time.monotonic() - t0:.2f} s into the fingerprint's sleep"
                )
            say("fingerprint: starting")
            out = real_fingerprint(*a, **kw)
            say("fingerprint: done")
            return out

        registry_stores.fingerprint_store = slow_fingerprint

    def pipeline():
        """Stands for main.py: its one `ray.init`, then a stage blocked in `ray.wait`."""
        say(f"before ray.init: C SIGTERM {c_handler(signal.SIGTERM)}")
        ray.init()
        node = ray._private.worker._global_node
        gcs = node.gcs_address
        try:
            foreign = RAY_ADDRESS_FILE.read_text().strip()
        except OSError:
            foreign = None
        say(
            f"READY session={os.path.basename(node.get_session_dir_path())} gcs={gcs} "
            f"ray_current_cluster={foreign}"
        )
        if foreign is not None and gcs == foreign:
            say("CONNECTED TO A CLUSTER THIS CHILD DID NOT START: stopping at once")
            os._exit(3)
        say(
            f"after ray.init: C SIGTERM {c_handler(signal.SIGTERM)} | C SIGALRM "
            f"{c_handler(signal.SIGALRM)} | Python SIGTERM "
            f"{getattr(signal.getsignal(signal.SIGTERM), '__qualname__', None)} | Python "
            f"SIGINT {getattr(signal.getsignal(signal.SIGINT), '__qualname__', None)}"
        )

        @ray.remote
        def slow():
            time.sleep(120)
            return 1

        ref = slow.remote()
        time.sleep(0.5)

        def send():
            T_SIGNAL[0] = time.time()
            say(f"SENDING at={T_SIGNAL[0]:.6f}")
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Timer(SIGNAL_AT, send).start()
        ray.wait([ref])  # no timeout, as RayWorkPool's main loop
        say("ray.wait returned")

    try:
        pipeline()
    except BaseException as exc:
        say(
            f"reached the driver: {type(exc).__module__}.{type(exc).__name__} "
            f"code={getattr(exc, 'code', None)!r}; latch.signum={latch.signum}"
        )
        scoped.finish_after_exec(run, exc, latch, ray.shutdown)
        raise
    scoped.finish_after_exec(run, None, latch, ray.shutdown)


def phase_child(which):
    """The ``phase`` section: one local Ray instance through the driver's real wrapper, then one
    ``SIGTERM`` per offset, each sent that long after the main thread enters ``ray.wait``, with
    the real handler installed afresh (a new latch, no run) before each. Reports when the
    handler's ``SystemExit`` reached the frame, relative to the signal."""
    import argparse as _argparse

    import ray

    scoped = _load("_scoped_pipeline_run", SCOPED_DRIVER)
    say(f"child pid={os.getpid()} variant=phase wrapper={which} ray={ray.__version__}")
    namespace = {
        "args": _argparse.Namespace(cpus=1),
        "_real_ray_init": ray.init,
        "run": None,
        "latch": scoped.SignalLatch(),
        "install_killed_handler": scoped.install_killed_handler,
        "scoped": scoped,
    }
    wrapper, _ = driver_wrapper(which, namespace)
    ray.init = wrapper
    ray.init()
    node = ray._private.worker._global_node
    say(
        f"READY session={os.path.basename(node.get_session_dir_path())} gcs={node.gcs_address}"
    )

    @ray.remote
    def slow():
        time.sleep(600)
        return 1

    ref = slow.remote()
    time.sleep(1.0)
    for offset in PHASE_OFFSETS:
        scoped.install_killed_handler(None, scoped.SignalLatch())
        sent = [None]

        def send():
            sent[0] = time.time()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Timer(offset, send).start()
        try:
            ray.wait([ref])
            say(f"offset {offset:4.2f} s: ray.wait returned")
        except BaseException as e:
            caught = time.time()
            say(
                f"offset {offset:4.2f} s after entering ray.wait: "
                f"{type(e).__name__}(code={getattr(e, 'code', None)!r}) reached the frame "
                f"{caught - sent[0]:.3f} s after the signal"
            )
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    t = time.time()
    ray.shutdown()
    say(f"ray.shutdown() returned after {time.time() - t:.2f} s")


# ------------------------------------------------------------------------------------------------
# the runner
# ------------------------------------------------------------------------------------------------

KILLED = []


def run_once(variant, which, index):
    print(
        f"\n{'=' * 100}\n## {variant} #{index} ({which} wrapper): "
        f"{VARIANTS.get(variant, PHASE)}\n"
    )
    tmp = Path(tempfile.mkdtemp(prefix="sigterm-handler-probe-"))
    root, store_dir = tmp / "runs", tmp / "store"
    root.mkdir()
    store_dir.mkdir()
    print(f"temporary directory: {tmp}")
    before_ray = ray_procs()
    before_all = all_procs()
    print(f"Ray processes before ({PATTERN}):\n{fmt(before_ray)}")
    load_before = os.getloadavg()
    print(f"load average before: {load_before}")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    t_start = time.time()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            __file__,
            "--child",
            variant,
            which,
            str(root),
            str(store_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    out_lines, err_lines = [], []
    own, pgids = {}, {}
    t_signal = [None]
    status_path = [None]
    timeline = []
    done = threading.Event()
    gcs = [None]

    def snapshot():
        procs = all_procs()
        if proc.pid in procs:
            pgids[proc.pid] = procs[proc.pid][2]
        for pid in descendants(proc.pid, procs):
            if pid not in before_all and pid not in own:
                own[pid] = procs[pid][1]
                pgids[pid] = procs[pid][2]

    def read_out():
        for line in proc.stdout:
            line = line.rstrip("\n")
            out_lines.append(line)
            if " RUN " in line:
                status_path[0] = os.path.join(line.split(" RUN ", 1)[1], "status.json")
            if " READY " in line:
                gcs[0] = re.search(r"gcs=(\S+)", line).group(1)
                snapshot()
            m = re.search(r"SENDING at=([\d.]+)", line)
            if m:
                t_signal[0] = float(m.group(1))
                snapshot()

    def read_err():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\n"))

    def poll():
        last, last_snap = None, 0.0
        while not done.is_set():
            path = status_path[0]
            if path is not None:
                try:
                    m0 = os.stat(path).st_mtime
                    with open(path) as f:
                        status = json.load(f)
                    m1 = os.stat(path).st_mtime
                except (OSError, ValueError):
                    status = None
                if status is not None and m0 == m1:
                    key = (
                        status.get("state"),
                        status.get("exit_code"),
                        "fingerprint" in status,
                    )
                    if key != last:
                        timeline.append((key, m1, time.time()))
                        last = key
            if time.time() - last_snap > 0.5:
                snapshot()
                last_snap = time.time()
            time.sleep(0.005)

    threads = [threading.Thread(target=f) for f in (read_out, read_err, poll)]
    for t in threads:
        t.start()
    try:
        rc = proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        print("[runner] child did not exit in 240 s: SIGKILL to the child")
        proc.kill()
        KILLED.append((proc.pid, "child", "SIGKILL"))
        rc = proc.wait()
    t_exit = time.time()
    done.set()
    for t in threads:
        t.join()
    load_after = os.getloadavg()

    print("\n--- child stdout")
    print("\n".join(out_lines))
    print("--- child stderr")
    print("\n".join(err_lines))
    print("--- end of child output\n")

    signame = ""
    if rc < 0:
        signame = f" (killed by {signal.Signals(-rc).name}; shell status {128 - rc})"
    print(f"exit status: {rc}{signame}")
    if t_signal[0] is not None:
        print(
            f"exit seen by the runner {t_exit - t_signal[0]:.3f} s after the signal "
            f"(child wall {t_exit - t_start:.1f} s)"
        )
    print(
        "status.json as the runner saw it change (state, exit_code, fingerprint present):"
    )
    for key, mtime, seen in timeline:
        rel = f"{mtime - t_signal[0]:+.3f} s" if t_signal[0] else "-"
        print(
            f"    {key}  written {rel} after the signal (mtime), read {seen - mtime:.3f} s later"
        )
    first_killed = next((m for key, m, _ in timeline if key[0] == "killed"), None)
    to_killed = (
        first_killed - t_signal[0]
        if first_killed is not None and t_signal[0] is not None
        else None
    )
    final = None
    if status_path[0] is not None:
        try:
            with open(status_path[0]) as f:
                final = json.load(f)
        except (OSError, ValueError) as e:
            print(f"status.json could not be read back: {e!r}")
    if final is not None:
        shown = {
            k: final.get(k)
            for k in (
                "state",
                "exit_code",
                "fingerprint_error",
                "fingerprint_sidecar",
            )
        }
        shown["fingerprint"] = (
            None
            if "fingerprint" not in final
            else f"present ({len(json.dumps(final['fingerprint']))} bytes of JSON)"
        )
        print(f"status.json read back: {shown}")
    banner = [l for l in err_lines + out_lines if BANNER in l]
    print(f"Ray's '{BANNER}' banner printed: {len(banner)} time(s)")
    print(f"the child's GCS: {gcs[0]}")
    print(
        f"own processes (descendants of the child, new since before it started); the child "
        f"{proc.pid} was in process group {pgids.get(proc.pid)}:"
    )
    for pid, cmd in sorted(own.items()):
        print(f"    {pid} pgid={pgids.get(pid)} {short(cmd)[:150]}")
    own_gcs = [pid for pid, cmd in own.items() if "gcs_server" in cmd]

    alive_at = {}
    for wait in (0.5, 2.0, 5.0, 10.0):
        time.sleep(max(0.0, t_exit + wait - time.time()))
        alive = sorted(pid for pid in own if exists(pid))
        alive_at[wait] = alive
        print(f"own processes still existing at exit +{wait:g} s: {alive}")
    survivors = {pid: own[pid] for pid in alive_at[10.0] if exists(pid)}
    if survivors:
        print("survivors, with their command lines:")
        for pid, cmd in sorted(survivors.items()):
            print(f"    {pid} pgid={pgids.get(pid)} {short(cmd)}")
    for sig, pause in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
        targets = [pid for pid in survivors if exists(pid)]
        if not targets:
            break
        for pid in targets:
            cmd = all_procs().get(pid, (None, ""))[1]
            if cmd != survivors[pid]:
                print(f"[runner] pid {pid} now runs {cmd!r}, not ours: left alone")
                continue
            os.kill(pid, sig)
            KILLED.append((pid, short(survivors[pid])[:100], sig.name))
            print(
                f"[runner] sent {sig.name} to own pid {pid} ({short(survivors[pid])[:100]})"
            )
        time.sleep(pause)
    remaining = [pid for pid in survivors if exists(pid)]
    print(f"own processes remaining after the kills: {remaining}")
    after_ray = ray_procs()
    print(f"Ray processes after ({PATTERN}):\n{fmt(after_ray)}")
    print(f"load average after: {load_after}")
    shutil.rmtree(tmp, ignore_errors=True)

    result = {
        "variant": variant,
        "index": index,
        "which": which,
        "state": None if final is None else final.get("state"),
        "exit_code": None if final is None else final.get("exit_code"),
        "fingerprint": final is not None and "fingerprint" in final,
        "to_killed": to_killed,
        "rc": rc,
        "banner": len(banner),
        "survivors": sorted(survivors),
        "gcs_survivors": sorted(p for p in survivors if p in own_gcs),
        "own_gcs": own_gcs,
        "load": (load_before[0], load_after[0]),
        "foreign": any("CONNECTED TO A CLUSTER" in l for l in out_lines),
        "phase": [
            l.split(" ", 1)[1]
            for l in out_lines
            if " offset " in l or "ray.shutdown() returned" in l
        ],
    }
    result["ok"] = (
        result["state"] == "killed"
        and result["exit_code"] == 143
        and to_killed is not None
        and to_killed < 1.0
        and result["fingerprint"]
        and rc == 143
        and not result["gcs_survivors"]
        and bool(own_gcs)
        and not result["foreign"]
    )
    return result


def line(r):
    to = "never" if r["to_killed"] is None else f"{r['to_killed']:.3f} s"
    return (
        f"{r['variant']:<16} #{r['index']} {r['which']:<9} state={r['state']} "
        f"exit_code={r['exit_code']} killed after {to}; fingerprint "
        f"{'present' if r['fingerprint'] else 'ABSENT'}; exit status {r['rc']}; banner "
        f"{r['banner']}; survivors {r['survivors'] or 'none'} (own gcs_server "
        f"{r['own_gcs'] or 'NONE SEEN'}); load {r['load'][0]:.0f}->{r['load'][1]:.0f}; "
        f"{'PASS' if r['ok'] else 'FAIL'}"
    )


def provenance():
    """The git HEAD, whether the probe and the two drivers differ from it, and their SHA-256s."""
    import hashlib

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    ).stdout.strip()
    print(f"git HEAD {head}")
    for path in (Path(__file__).resolve(), SCOPED_DRIVER, SWEEP_DRIVER):
        rel = path.relative_to(REPO_ROOT)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(rel)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        ).stdout.strip()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(
            f"    {rel}: sha256 {digest}…, "
            + (f"differs from HEAD ({dirty.split()[0]})" if dirty else "as at HEAD")
        )


def runner(variants, repeat, phase):
    import ray

    print("# sigterm_handler_probe.py (a3-v2-readiness prompt 04)")
    provenance()
    print(
        f"ray {ray.__version__}, python {sys.version.split()[0]}, {os.uname().sysname} "
        f"{os.uname().release} {os.uname().machine}, {os.cpu_count()} CPUs"
    )
    print(
        f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; load average {os.getloadavg()}"
    )
    try:
        print(f"{RAY_ADDRESS_FILE}: {RAY_ADDRESS_FILE.read_text().strip()}")
    except OSError:
        print(f"{RAY_ADDRESS_FILE}: absent")
    t0 = time.monotonic()
    first = ray_procs()
    print(f"Ray processes before the probe ({PATTERN}):\n{fmt(first)}")
    results = []
    order = ["scoped", "run_build"]
    for variant in variants:
        for index in range(1, repeat + 1):
            which = order[(index - 1) % 2]
            t = time.monotonic()
            results.append(run_once(variant, which, index))
            print(f"run wall time: {time.monotonic() - t:.1f} s")
    phase_result = None
    if phase:
        t = time.monotonic()
        phase_result = run_once("phase", "scoped", 1)
        print(f"run wall time: {time.monotonic() - t:.1f} s")
    print(f"\n{'=' * 100}\n## summary: wall time {time.monotonic() - t0:.1f} s\n")
    for r in results:
        print(line(r))
    passed = sum(r["ok"] for r in results)
    print(
        f"\nacceptance (killed/143 written within 1 s of the signal; fingerprint present; exit "
        f"status 143; no abort; the child's own gcs_server seen, and not alive at +10 s): "
        f"{passed} of {len(results)} runs pass"
    )
    if phase_result is not None:
        print(f"\nphase ({PHASE}):")
        for text in phase_result["phase"]:
            print(f"    {text}")
        print(
            f"    exit status {phase_result['rc']}; own gcs_server "
            f"{phase_result['own_gcs'] or 'NONE SEEN'}; survivors "
            f"{phase_result['survivors'] or 'none'}"
        )
    print(f"\nRay processes before the probe ({PATTERN}):\n{fmt(first)}")
    print(f"Ray processes after the probe ({PATTERN}):\n{fmt(ray_procs())}")
    print("pids killed by this probe:")
    if KILLED:
        for pid, cmd, sig in KILLED:
            print(f"    {pid} {sig} {cmd}")
    else:
        print("    (none)")
    print(
        f"finished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; load average {os.getloadavg()}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--child" and sys.argv[2] == "phase":
        phase_child(sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == "--child":
        child(*sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
        parser.add_argument("--variants", default=",".join(VARIANTS))
        parser.add_argument("--repeat", type=int, default=3)
        parser.add_argument("--no-phase", action="store_true")
        a = parser.parse_args()
        chosen = [v for v in a.variants.split(",") if v]
        unknown = [v for v in chosen if v not in VARIANTS]
        if unknown:
            parser.error(f"unknown variant(s) {unknown}; known: {list(VARIANTS)}")
        runner(chosen, a.repeat, not a.no_phase)
