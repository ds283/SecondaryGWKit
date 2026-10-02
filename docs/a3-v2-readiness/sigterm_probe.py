"""
S1, S2, S3 (a3-v2-readiness audit 00): what a SIGTERM does to a driver blocked in ``ray.wait``,
measured with real local Ray instances.

Each *variant* is a child process (``--child VARIANT``) that starts its own local Ray instance
(``ray.init(num_cpus=1, include_dashboard=False)``), submits one task that sleeps, sends itself
``SIGTERM`` 2 s later from a timer thread while the main thread is in ``ray.wait``, and reports, with
times relative to the moment the signal was sent:

- the C-level handlers of ``SIGTERM`` and ``SIGALRM`` (``sigaction``), before ``ray.init``, after
  it, after any re-installation, and after the signal: each resolved with ``dladdr`` to its library
  and offset, and, for ``_raylet.so``, to the local symbol ``nm`` lists at that offset;
- the Python-level handler (``signal.getsignal``);
- which Python handler ran, and when, and on which thread;
- the exception that reached the frame around ``ray.wait``: type, ``code``, ``args``;
- in the cleanup: ``signal.alarm(0)``'s return value, which is the number of seconds left on any
  alarm that was pending (0 if none), where the variant cancels it; and a tick every 0.25 s for up
  to 8 s, so that the instant of death lies between the last tick and the exit the runner sees.

The *runner* (``S1``, ``S2`` or ``S3``) runs a set of variants one at a time. Around each it
records every process whose command line matches ``gcs_server|raylet|ray::`` before the child
starts and after it ends (``pgrep -lf``: a listing only), and the child's own descendants while it
runs (``ps``, any command: what the child started is its own). After the child exits it checks
which of its own processes still exist at +0.5, +2, +5 and +10 s, then kills each survivor of its
own with ``SIGTERM`` and, 3 s later, ``SIGKILL``, printing every pid. It never signals a process that
existed before the child started, or one that is not a descendant of the child. The child's stdout
and stderr are printed unfiltered. The output ends with the processes before and after the whole
probe and every pid killed.

Usage, from the repository root (one item per run, each committed beside this file):

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/sigterm_probe.py S1 \
        > docs/a3-v2-readiness/sigterm_probe_S1.txt 2>&1
"""

import ctypes
import ctypes.util
import os
import re
import signal
import subprocess
import sys
import threading
import time

PATTERN = "gcs_server|raylet|ray::"
SIGNAL_AT = 2.0

# ------------------------------------------------------------------------------------------------
# the child
# ------------------------------------------------------------------------------------------------


class _Sigaction(ctypes.Structure):
    # macOS user-space struct sigaction: handler, sa_mask (uint32), sa_flags (int)
    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("mask", ctypes.c_uint32),
        ("flags", ctypes.c_int),
    ]


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_nm_cache = {}


def _nm_symbols(path):
    if path not in _nm_cache:
        table = {}
        out = subprocess.run(["nm", path], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[1] in "tT":
                table[int(parts[0], 16)] = parts[2]
        _nm_cache[path] = table
    return _nm_cache[path]


def c_handler(signum) -> str:
    act = _Sigaction()
    if _libc.sigaction(signum, None, ctypes.byref(act)) != 0:
        return f"sigaction failed errno={ctypes.get_errno()}"
    addr = act.handler or 0
    if addr == 0:
        return "SIG_DFL"
    if addr == 1:
        return "SIG_IGN"
    info = _DlInfo()
    if _libc.dladdr(ctypes.c_void_p(addr), ctypes.byref(info)) == 0:
        return f"0x{addr:x} (dladdr found no image)"
    lib = info.dli_fname.decode()
    offset = addr - info.dli_fbase
    text = f"{os.path.basename(lib)}+0x{offset:x}"
    if os.path.basename(lib) == "_raylet.so":
        sym = _nm_symbols(lib).get(offset)
        text += f" = nm {sym}" if sym else " (no nm symbol at this offset)"
    elif info.dli_sname:
        text += f" (dladdr nearest {info.dli_sname.decode()})"
    return text


T_SIGNAL = [None]


def rel():
    if T_SIGNAL[0] is None:
        return "t=-"
    return f"t={time.time() - T_SIGNAL[0]:+.3f}s"


def say(*parts):
    print(rel(), *parts, flush=True)


def handlers(when):
    say(
        f"[{when}] C SIGTERM: {c_handler(signal.SIGTERM)} | C SIGALRM: "
        f"{c_handler(signal.SIGALRM)} | Python SIGTERM: {signal.getsignal(signal.SIGTERM)!r}"
    )


class SigtermReceived(BaseException):
    pass


def child(variant):
    import ray

    say(f"child pid={os.getpid()} variant={variant} ray={ray.__version__}")
    _nm_symbols(ray._raylet.__file__)  # read once now, so that nm is not running later
    handlers("before ray.init")
    ray.init(num_cpus=1, include_dashboard=False)
    node = ray._private.worker._global_node
    say(f"READY session={os.path.basename(node.get_session_dir_path())}")
    handlers("after ray.init")
    say(f"alarm pending after ray.init: signal.alarm(0) returned {signal.alarm(0)}")

    if variant.startswith("after-"):

        def handler(signum, frame):
            say(
                f"Python handler ran: signum={signum} thread={threading.current_thread().name} "
                f"frame={frame.f_code.co_name if frame else None}"
            )
            if variant in ("after-exit", "after-exit-get"):
                raise SystemExit(128 + signum)
            if variant == "after-kbint":
                raise KeyboardInterrupt(signum)
            if variant == "after-custom":
                raise SigtermReceived(signum)
            # after-noraise: record and return

        signal.signal(signal.SIGTERM, handler)
        handlers("after re-installing a Python handler")

    @ray.remote
    def slow():
        time.sleep(60)
        return 1

    ref = slow.remote()
    time.sleep(0.5)

    def send():
        say("SENDING")
        T_SIGNAL[0] = time.time()
        os.kill(os.getpid(), signal.SIGTERM)

    timeout = 6 if variant == "after-noraise" else 30
    threading.Timer(SIGNAL_AT, send).start()
    try:
        if variant.endswith("-get"):
            value = ray.get(ref, timeout=timeout)
            say(f"ray.get returned normally: {value!r}")
        else:
            ready, _ = ray.wait([ref], timeout=timeout)
            say(f"ray.wait returned normally: {len(ready)} ready")
    except BaseException as e:
        say(
            f"caught {type(e).__module__}.{type(e).__name__}: code={getattr(e, 'code', None)!r} "
            f"args={e.args!r}"
        )
    handlers("after the signal")

    if "alarm0" in variant:
        say(
            f"cleanup: signal.alarm(0) returned {signal.alarm(0)} (the alarm is cancelled)"
        )
        handlers("after alarm(0)")
    elif variant.startswith("after-"):
        remaining = signal.alarm(0)
        if remaining:
            signal.alarm(remaining)
        say(
            f"cleanup: signal.alarm(0) returned {remaining}; "
            + (
                f"re-armed for {remaining} s, so not cancelled"
                if remaining
                else "nothing to re-arm"
            )
        )
    if variant.endswith("wait3-osexit"):
        time.sleep(3.0)
    if variant.endswith("osexit"):
        say("cleanup: os._exit(1), no ray.shutdown, no atexit")
        os._exit(1)
    if variant.endswith("shutdown"):
        t = time.time()
        ray.shutdown()
        say(f"cleanup: ray.shutdown() returned after {time.time() - t:.2f} s")
    if not variant.endswith("shutdown"):
        say("cleanup: sleeping 8 s")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            time.sleep(0.25)
            say(f"alive, {time.monotonic() - t0:.2f} s into cleanup")
        say("survived 8 s of cleanup")
    say("leaving by sys.exit(1): atexit (and Ray's shutdown hook) runs")
    sys.exit(1)


# ------------------------------------------------------------------------------------------------
# the runner
# ------------------------------------------------------------------------------------------------


def ray_procs():
    out = subprocess.run(["pgrep", "-lf", PATTERN], capture_output=True, text=True)
    procs = {}
    for line in out.stdout.splitlines():
        pid, _, cmd = line.partition(" ")
        procs[int(pid)] = cmd
    return procs


def all_procs():
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,command="], capture_output=True, text=True
    )
    procs = {}
    for line in out.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if m:
            procs[int(m.group(1))] = (int(m.group(2)), m.group(4), int(m.group(3)))
    return procs


PYTHON_PREFIX = re.compile(r"^\S*/Python(\.app/Contents/MacOS/Python)?(?=\s|$)")


def short(cmd):
    """The command line with the interpreter's path written as `python` and the ray package's
    directory as `<ray>/`, so that what each process is can be read; nothing else is changed.
    """
    import ray as _ray

    cmd = PYTHON_PREFIX.sub("python", cmd)
    return cmd.replace(os.path.dirname(_ray.__file__) + "/", "<ray>/")


def descendants(root_pid, procs):
    found = set()
    frontier = {root_pid}
    while frontier:
        nxt = {pid for pid, (ppid, _, _) in procs.items() if ppid in frontier} - found
        found |= nxt
        frontier = nxt
    return found


def exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def fmt(procs):
    if not procs:
        return "    (none)"
    return "\n".join(
        f"    {pid} {short(cmd)[:160]}" for pid, cmd in sorted(procs.items())
    )


KILLED = []


def run_variant(variant, note):
    print(f"\n{'=' * 100}\n## variant {variant}: {note}\n")
    before_ray = ray_procs()
    before_all = all_procs()
    print(f"Ray processes before ({PATTERN}):\n{fmt(before_ray)}")
    print(f"load average before: {os.getloadavg()}")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    t_start = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", __file__, "--child", variant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out_lines, err_lines = [], []
    own = {}
    pgids = {}
    t_signal = [None]
    session = [None]

    def snapshot(tag):
        procs = all_procs()
        if proc.pid in procs:
            pgids[proc.pid] = procs[proc.pid][2]
        for pid in descendants(proc.pid, procs):
            if pid not in before_all and pid not in own:
                own[pid] = procs[pid][1]
                pgids[pid] = procs[pid][2]
        print(f"[runner] snapshot at {tag}: {len(own)} own descendant(s) so far")

    def read_out():
        for line in proc.stdout:
            out_lines.append(line.rstrip("\n"))
            if " READY " in line:
                session[0] = line.split("session=")[1].strip()
                snapshot("READY")
            if line.rstrip().endswith("SENDING"):
                t_signal[0] = time.time()
                snapshot("SENDING")

    def read_err():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\n"))

    to = threading.Thread(target=read_out)
    te = threading.Thread(target=read_err)
    to.start()
    te.start()
    try:
        rc = proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print("[runner] child did not exit in 90 s: SIGKILL to the child")
        proc.kill()
        KILLED.append((proc.pid, "child", "SIGKILL"))
        rc = proc.wait()
    t_exit = time.time()
    to.join()
    te.join()

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
            f"exit seen by the runner {t_exit - t_signal[0]:.3f} s after SENDING "
            f"(child wall {t_exit - t_start:.1f} s)"
        )
    banner = [l for l in err_lines + out_lines if "SIGTERM received" in l]
    print(f"Ray's '*** SIGTERM received' banner printed: {len(banner)} time(s)")
    print(f"session: {session[0]}")
    print(
        f"own processes (descendants of the child, new since before it started); the child "
        f"{proc.pid} was in process group {pgids.get(proc.pid)}:"
    )
    for pid, cmd in sorted(own.items()):
        print(f"    {pid} pgid={pgids.get(pid)} {short(cmd)[:150]}")

    for wait in (0.5, 2.0, 5.0, 10.0):
        time.sleep(max(0.0, t_exit + wait - time.time()))
        alive = {pid: cmd for pid, cmd in own.items() if exists(pid)}
        print(f"own processes still existing at exit +{wait:g} s: {sorted(alive)}")
    survivors = {pid: cmd for pid, cmd in own.items() if exists(pid)}
    if survivors:
        print("survivors, with their command lines:")
        for pid, cmd in sorted(survivors.items()):
            print(f"    {pid} pgid={pgids.get(pid)} {short(cmd)}")
        for pid, cmd in sorted(survivors.items()):
            m = re.search(r"--log_dir=(\S+)", cmd)
            if "gcs_server" in cmd and m:
                log = os.path.join(m.group(1), "gcs_server.out")
                try:
                    with open(log) as f:
                        tail = f.read().splitlines()[-8:]
                except OSError as e:
                    tail = [f"(could not read {log}: {e})"]
                print(f"    last lines of {log}:")
                for line in tail:
                    print(f"      {line[:300]}")
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
    new = {pid: cmd for pid, cmd in after_ray.items() if pid not in before_ray}
    print(f"new Ray processes left by this variant: {sorted(new)}")


ITEMS = {
    "S1": [
        (
            "keep-alarm0-sleep",
            "Ray's handlers kept (prompt 10's sigterm_in_ray_wait_probe.py); alarm(0) in cleanup",
        ),
        (
            "after-exit",
            "a Python handler installed after ray.init, raising SystemExit(143)",
        ),
        (
            "after-custom",
            "a Python handler installed after ray.init, raising a BaseException subclass",
        ),
        (
            "after-kbint",
            "a Python handler installed after ray.init, raising KeyboardInterrupt",
        ),
        (
            "after-noraise",
            "a Python handler installed after ray.init that records and returns",
        ),
        (
            "after-exit-get",
            "as after-exit, but the main thread is blocked in ray.get, not ray.wait",
        ),
    ],
    "S2": [
        ("keep-sleep", "Ray's handlers kept; cleanup sleeps 8 s with no alarm(0)"),
        (
            "keep-alarm0-sleep",
            "Ray's handlers kept; cleanup calls signal.alarm(0), then sleeps 8 s",
        ),
        ("keep-sleep", "repeat: no alarm(0)"),
        ("keep-alarm0-sleep", "repeat: alarm(0), then 8 s"),
        ("keep-sleep", "repeat: no alarm(0)"),
    ],
    "S3": [
        ("keep-sleep", "the abort: Ray's handlers kept, no alarm(0), 8 s cleanup"),
        (
            "keep-alarm0-osexit",
            "alarm(0), then os._exit(1): no ray.shutdown and no atexit",
        ),
        (
            "keep-alarm0-wait3-osexit",
            "alarm(0), 3 s of cleanup, then os._exit(1): the abort's timing without the abort",
        ),
        (
            "keep-alarm0-sleep",
            "alarm(0), 8 s cleanup, then sys.exit: no explicit ray.shutdown, atexit runs",
        ),
        ("keep-alarm0-shutdown", "alarm(0), explicit ray.shutdown(), then sys.exit"),
        ("keep-sleep", "repeat: the abort"),
        ("keep-alarm0-osexit", "repeat: alarm(0), then os._exit(1)"),
        ("keep-alarm0-wait3-osexit", "repeat: alarm(0), 3 s, then os._exit(1)"),
        ("keep-sleep", "repeat: the abort"),
        ("keep-alarm0-osexit", "repeat: alarm(0), then os._exit(1)"),
        ("keep-alarm0-wait3-osexit", "repeat: alarm(0), 3 s, then os._exit(1)"),
    ],
}


def runner(item):
    import ray

    print(f"# sigterm_probe.py {item} (a3-v2-readiness audit 00)")
    print(
        f"ray {ray.__version__}, python {sys.version.split()[0]}, {os.uname().sysname} "
        f"{os.uname().release} {os.uname().machine}, {os.cpu_count()} CPUs"
    )
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    t0 = time.monotonic()
    first = ray_procs()
    print(f"Ray processes before the probe ({PATTERN}):\n{fmt(first)}")
    for variant, note in ITEMS[item]:
        t = time.monotonic()
        run_variant(variant, note)
        print(f"variant wall time: {time.monotonic() - t:.1f} s")
    print(f"\n{'=' * 100}\n## end of {item}: wall time {time.monotonic() - t0:.1f} s")
    print(f"Ray processes before the probe ({PATTERN}):\n{fmt(first)}")
    print(f"Ray processes after the probe ({PATTERN}):\n{fmt(ray_procs())}")
    print("pids killed by this probe:")
    if KILLED:
        for pid, cmd, sig in KILLED:
            print(f"    {pid} {sig} {cmd}")
    else:
        print("    (none)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] in ITEMS:
        runner(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(2)
