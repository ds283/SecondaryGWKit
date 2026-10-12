"""
SecondaryGWKit's side of the shared rehearsal lock.

Three projects on this machine (SecondaryGWKit, ChamPBH and StochasticInstantons) run Ray builds.
They share one lock, the directory ``~/ScienceData/.rehearsal-lock``, so that only one rehearsal or
long build runs at a time. The contract is StochasticInstantons' brief,
``.prompts/dk-and-rpk-migration/rehearsal-lock-brief.md`` at its commit ``d6d550b``, sections 2 and
5; this campaign's README section 4.1 (``prompts/runregistry-extraction``) is SGK's protocol.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>

Commands (every one prints what it did):

``acquire --purpose TEXT``
    ``mkdir`` the lock (atomic, never ``exist_ok``) and write ``owner`` with ``pid: -``. Exit 0 held
    by SGK now; 3 already held (the owner is printed, nothing is touched); 4 a Ray process of this
    repository is alive (the lock is released again).
``set-pid PID``
    Record ``PID`` in SGK's own lock, changing no other line. Exit 0; 1 refused.
``release``
    Remove SGK's lock, once its pid is ``-`` or dead and no Ray process of this repository is
    alive. Exit 0 released, or no lock; 1 refused, with the reason.
``status``
    The lock, its owner file verbatim, whether the recorded pid is alive, and this repository's
    Ray processes. Exit 0.
``ray-processes``
    This repository's Ray processes, one per line. Exit 0 none; 1 some.

Usage errors exit 2.

There is no option to break a lock, to wait for one, or to retry. A held lock means stop and ask the
user; a stale lock is the user's to remove by hand. The helper never starts or stops Ray and
signals no process. It is not part of the run registry.

Standard library only.
"""

import argparse
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = "SecondaryGWKit"
IN_PROCESS = "in-process"
OWNER_KEYS = (
    "project",
    "purpose",
    "acquired",
    "host",
    "ray_temp_dir",
    "ray_address",
    "pid",
)
PS_COMMAND = ["ps", "-axww", "-o", "pid=", "-o", "ppid=", "-o", "command="]
LINE_WIDTH = 150

_PS_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(.*)$")


def default_lock_dir():
    """The shared lock, computed when called (it reads ``HOME`` at that moment, never at import)."""
    return Path.home() / "ScienceData" / ".rehearsal-lock"


def _resolve(lock_dir):
    return default_lock_dir() if lock_dir is None else Path(lock_dir)


def _short_hostname():
    """``hostname -s``; if that cannot be run, ``socket.gethostname()`` cut at its first dot."""
    try:
        result = subprocess.run(
            ["hostname", "-s"], capture_output=True, text=True, check=True
        )
        name = result.stdout.strip()
        if name:
            return name
    except (OSError, subprocess.CalledProcessError):
        pass
    return socket.gethostname().split(".")[0]


# ---- the owner file -------------------------------------------------------------------------


def owner_text(purpose, pid, acquired=None, host=None):
    """The seven lines of the owner file, in the brief's order."""
    if acquired is None:
        acquired = datetime.now().astimezone().isoformat(timespec="seconds")
    if host is None:
        host = _short_hostname()
    lines = [
        f"project: {PROJECT}",
        f"purpose: {purpose}",
        f"acquired: {acquired}",
        f"host: {host}",
        f"ray_temp_dir: {IN_PROCESS}",
        f"ray_address: {IN_PROCESS}",
        f"pid: {pid}",
    ]
    return "\n".join(lines) + "\n"


def _replace_owner(lock_dir, text):
    """Write ``owner.tmp`` in the lock directory, then rename it over ``owner``."""
    tmp = lock_dir / "owner.tmp"
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, lock_dir / "owner")


def write_owner(lock_dir, purpose, pid="-"):
    _replace_owner(lock_dir, owner_text(purpose, pid))


def parse_owner(text):
    """Any project's owner file: one ``key: value`` per line, the value after the first ``": "``."""
    owner = {}
    for line in text.splitlines():
        key, sep, value = line.partition(": ")
        if not sep:
            key = line.rstrip(":")
            value = ""
        owner[key] = value
    return owner


def read_owner(lock_dir):
    """The owner file's text, or ``None`` when there is none."""
    try:
        return (lock_dir / "owner").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _indented(text):
    if text is None:
        return "    (no owner file)"
    return "\n".join("    " + line for line in text.splitlines())


def pid_alive(pid):
    """Whether ``pid`` names a live process (``os.kill(pid, 0)``; a process we may not signal is alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _not_sgk_reason(owner):
    """``None`` when the owner is SGK's (``project`` and ``ray_temp_dir``), otherwise why not."""
    if owner.get("project") != PROJECT:
        return f"project is {owner.get('project')!r}, not {PROJECT!r}"
    if owner.get("ray_temp_dir") != IN_PROCESS:
        return f"ray_temp_dir is {owner.get('ray_temp_dir')!r}, not {IN_PROCESS!r}"
    return None


def _recorded_pid(owner):
    """The owner's pid as an integer, or ``None`` for ``-`` or anything that is not a number."""
    value = owner.get("pid", "-").strip()
    return int(value) if value.isdigit() else None


# ---- SGK's Ray processes --------------------------------------------------------------------


def _ps_output():
    """The process table. The one place ``ps`` is read; tests replace it with fixed text."""
    result = subprocess.run(PS_COMMAND, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ps failed with status {result.returncode}")
    return result.stdout


def venv_roots():
    """This checkout's ``venv/`` as written and as ``realpath`` gives it (one entry when they agree)."""
    root = Path(__file__).resolve().parent.parent
    written = Path(os.path.abspath(Path(__file__).parent.parent)) / "venv"
    roots = [str(written), os.path.realpath(root / "venv")]
    return list(dict.fromkeys(roots))


def ray_processes(ps_text=None, roots=None, self_pid=None):
    """
    Every process of this repository's Ray, as ``(pid, ppid, command)`` sorted by pid.

    A process is a seed if its command line contains this checkout's ``venv/`` path followed,
    anywhere later, by ``/site-packages/ray/`` (Ray's ``gcs_server``, ``raylet`` and agents name
    that path; a dashboard agent has a bare ``python`` as ``argv[0]``, so the whole line is
    searched). Every descendant of a seed is listed too: workers show only ``ray::<name>``. This
    process and its ancestors are never listed.
    """
    if ps_text is None:
        ps_text = _ps_output()
    if roots is None:
        roots = venv_roots()
    if self_pid is None:
        self_pid = os.getpid()

    table = {}
    for line in ps_text.splitlines():
        m = _PS_LINE.match(line)
        if m:
            table[int(m.group(1))] = (int(m.group(2)), m.group(3))

    safe = set()
    pid = self_pid
    while pid in table and pid not in safe and pid > 1:
        safe.add(pid)
        pid = table[pid][0]
    safe.add(self_pid)
    own = {self_pid}
    changed = True
    while changed:  # this process's own children (its ``ps``) are never listed either
        changed = False
        for p, (ppid, _) in table.items():
            if ppid in own and p not in own:
                own.add(p)
                changed = True
    safe |= own

    patterns = [
        re.compile(re.escape(root) + r"/.*site-packages/ray/") for root in roots
    ]
    marked = {
        p for p, (_, cmd) in table.items() if any(rx.search(cmd) for rx in patterns)
    }
    changed = True
    while changed:
        changed = False
        for p, (ppid, _) in table.items():
            if p not in marked and ppid in marked:
                marked.add(p)
                changed = True
    # this process and its ancestors seed the walk (a sibling below an ancestor Ray process is
    # Ray's), but are never themselves listed
    return [(p, table[p][0], table[p][1]) for p in sorted(marked - safe)]


def _process_lines(processes):
    return [f"{p} {pp} {cmd}"[:LINE_WIDTH] for p, pp, cmd in processes]


def _remove_lock(lock_dir):
    for name in ("owner", "owner.tmp"):
        try:
            os.remove(lock_dir / name)
        except FileNotFoundError:
            pass
    os.rmdir(lock_dir)


# ---- the commands ---------------------------------------------------------------------------


def acquire(purpose, lock_dir=None):
    d = _resolve(lock_dir)
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)  # never exist_ok: this is the atomic step
    except FileExistsError:
        print(f"the rehearsal lock is held ({d}); its owner file:", file=sys.stderr)
        print(_indented(read_owner(d)), file=sys.stderr)
        print(
            "stop and ask the user; this helper never waits on, retries or breaks the lock",
            file=sys.stderr,
        )
        return 3
    try:
        write_owner(d, purpose, "-")
    except OSError as exc:
        print(f"could not write the owner file ({exc}); releasing", file=sys.stderr)
        try:
            _remove_lock(d)
        except OSError:
            pass
        return 1
    try:
        procs = ray_processes()
    except (RuntimeError, OSError) as exc:
        print(f"cannot read the process table ({exc}); releasing", file=sys.stderr)
        _remove_lock(d)
        return 4
    if procs:
        print("Ray processes of this repository are alive:", file=sys.stderr)
        for line in _process_lines(procs):
            print("    " + line, file=sys.stderr)
        _remove_lock(d)
        print("released the lock again; not acquired", file=sys.stderr)
        return 4
    print(f"acquired the rehearsal lock ({d}) for {PROJECT}: {purpose}")
    return 0


def _own_lock(d, verb):
    """``(owner_text, owner)`` if ``d`` is SGK's lock; otherwise prints why and returns ``None``."""
    if not d.is_dir():
        print(f"{verb}: there is no lock at {d}", file=sys.stderr)
        return None
    text = read_owner(d)
    if text is None:
        print(f"{verb}: the lock has no owner file; not touching it", file=sys.stderr)
        print(f"    lock: {d}\n    expected: project {PROJECT!r}", file=sys.stderr)
        return None
    owner = parse_owner(text)
    reason = _not_sgk_reason(owner)
    if reason is not None:
        print(
            f"{verb}: the lock is not SGK's ({reason}); not touching it",
            file=sys.stderr,
        )
        print("    owner:\n" + _indented(text), file=sys.stderr)
        print(
            f"    SGK is project: {PROJECT}, ray_temp_dir: {IN_PROCESS}",
            file=sys.stderr,
        )
        return None
    return text, owner


def set_pid(pid, lock_dir=None):
    d = _resolve(lock_dir)
    if pid <= 0 or not pid_alive(pid):
        print(f"set-pid: {pid} is not a live process; nothing changed", file=sys.stderr)
        return 1
    held = _own_lock(d, "set-pid")
    if held is None:
        return 1
    text, _ = held
    lines = text.splitlines()
    index = [i for i, line in enumerate(lines) if line.startswith("pid:")]
    if len(index) != 1:
        print(
            "set-pid: the owner file has no single pid line; nothing changed",
            file=sys.stderr,
        )
        return 1
    lines[index[0]] = f"pid: {pid}"
    _replace_owner(d, "\n".join(lines) + "\n")
    print(f"recorded pid {pid} in {d / 'owner'}")
    return 0


def release(lock_dir=None):
    d = _resolve(lock_dir)
    if not d.is_dir():
        print(f"no lock to release ({d})")
        return 0
    held = _own_lock(d, "release")
    if held is None:
        return 1
    _, owner = held
    pid = _recorded_pid(owner)
    if pid is not None and pid_alive(pid):
        print(
            f"release: the recorded pid {pid} is alive; not releasing", file=sys.stderr
        )
        return 1
    try:
        procs = ray_processes()
    except (RuntimeError, OSError) as exc:
        print(
            f"release: cannot read the process table ({exc}); not releasing",
            file=sys.stderr,
        )
        return 1
    if procs:
        print("release: Ray processes of this repository are alive:", file=sys.stderr)
        for line in _process_lines(procs):
            print("    " + line, file=sys.stderr)
        return 1
    try:
        _remove_lock(d)
    except OSError as exc:
        print(f"release: could not remove {d} ({exc})", file=sys.stderr)
        return 1
    print(f"released the rehearsal lock ({d})")
    return 0


def status(lock_dir=None):
    d = _resolve(lock_dir)
    if d.is_dir():
        print(f"lock: HELD ({d})")
        text = read_owner(d)
        print(_indented(text))
        if text is not None:
            pid = _recorded_pid(parse_owner(text))
            if pid is None:
                print("owner pid: - (no pid recorded yet)")
            elif pid_alive(pid):
                print(f"owner pid {pid}: alive")
            else:
                print(f"owner pid {pid}: NOT alive")
    else:
        print(f"lock: free ({d} does not exist)")
    try:
        procs = ray_processes()
    except (RuntimeError, OSError) as exc:
        print(
            f"Ray processes of this repository: cannot read the process table ({exc})"
        )
        return 0
    if not procs:
        print("Ray processes of this repository: none")
    else:
        print(f"Ray processes of this repository: {len(procs)}")
        for line in _process_lines(procs):
            print("    " + line)
    return 0


def ray_processes_command():
    try:
        procs = ray_processes()
    except (RuntimeError, OSError) as exc:
        print(f"cannot read the process table ({exc})", file=sys.stderr)
        return 1
    for line in _process_lines(procs):
        print(line)
    return 1 if procs else 0


def _purpose(text):
    if not text.strip() or "\n" in text or "\r" in text:
        raise argparse.ArgumentTypeError("must be one non-empty line")
    return text


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tools.rehearsal_lock",
        description="SecondaryGWKit's side of the shared rehearsal lock. Never breaks, waits for or retries a lock.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "acquire", help="take the lock (exit 3 if held, 4 if Ray is alive)"
    )
    p.add_argument("--purpose", required=True, type=_purpose)
    p = sub.add_parser("set-pid", help="record a live pid in SGK's lock")
    p.add_argument("pid", type=int)
    sub.add_parser("release", help="release SGK's lock (exit 1 if refused)")
    sub.add_parser(
        "status", help="the lock, its owner, and this repository's Ray processes"
    )
    sub.add_parser(
        "ray-processes", help="this repository's Ray processes (exit 1 if any)"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        return acquire(args.purpose)
    if args.command == "set-pid":
        return set_pid(args.pid)
    if args.command == "release":
        return release()
    if args.command == "status":
        return status()
    return ray_processes_command()


if __name__ == "__main__":
    sys.exit(main())
