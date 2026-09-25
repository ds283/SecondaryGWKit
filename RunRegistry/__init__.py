"""A record on disk that a long-running job exists, what it is for, and how far it has got.

The rules this package exists to support are in `CLAUDE.md`, "Long-running jobs — the run
registry"; the failures it exists because of are `prompts/run-registry/README.md` §0. It is a
convention with a little code behind it, not a framework: it schedules nothing, supervises
nothing, locks nothing and deletes nothing. It also creates, adopts, copies and moves the stores it
manages, with their `<stem>.manifest.json` sidecars (`RunRegistry.stores`), because the user
decided that managing stores is part of managing the registry (`prompts/datastore-portability`
README §6.5); it still deletes nothing.

The layout, under `var/runs/` (gitignored, in the repository, never a session scratchpad and never
`/tmp` — README §0 item 4 is a datastore that was written into a scratchpad and is gone):

    var/runs/<campaign>-<prompt>-<slug>-<YYYYMMDDTHHMMSS>/
        manifest.json      written once at launch, never mutated
        status.json        the only mutable file; state, progress, heartbeat, pid, exit code
        checkpoint.jsonl   optional, append-only, one JSON object per completed unit
        stdout.log  stderr.log

The manifest keeps two different things in two different fields, and the difference is README §0.2:
`checkpoint` is the unit ledger above, which `record()` writes and `known()` reads; `results` names
the durable thing the job's results live in — a pipeline run's datastore — which this package names
and never writes; it reads it only to take its fingerprint at `finish(..., fingerprint=True)`. A job
may have either, both or neither.

Usage is three calls. The launching process writes the manifest before any work starts, so that a
crash in the first second still leaves a record:

    run = RunRegistry.begin(
        campaign="run-registry", prompt="01", slug="smoke",
        purpose="one line a stranger can read",
        script=__file__, unit="cell", expected_units=60, checkpoint=True,
    )
    known = run.known()                       # what a previous run already did
    for item in work:
        if key(item) in known:
            continue
        run.record(key(item), compute(item))   # appends, flushes, fsyncs, beats
    run.finish("done", exit_code=0)

`python -m RunRegistry list` prints every run, newest first, and says loudly which of them claim
to be running but are not.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "var", "runs")

TERMINAL_STATES = ("done", "failed", "killed")

# The lister's default staleness window. `liveness()` itself takes the window from its caller;
# fifteen minutes is a convention for the command line, overridable with --stale-after, and is
# chosen to be long against a heartbeat written once per completed unit.
DEFAULT_STALE_AFTER = 900.0

_ID_STAMP = "%Y%m%dT%H%M%S"


# =================================================================================================
# provenance, as `docs/handover/realistic_large_x.py` stamps it


def now_iso() -> str:
    """Local time with an offset, to the second — the format the hand-written A3 manifest used."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _slug(text) -> str:
    return re.sub(r"[^A-Za-z0-9._]+", "-", str(text)).strip("-")


def run_id(campaign, prompt, slug, when=None) -> str:
    """`<campaign>-<prompt>-<slug>-<YYYYMMDDTHHMMSS>`: it sorts, and a stranger reading
    `ls var/runs/` learns who owns each directory without asking anyone."""
    stamp = time.strftime(_ID_STAMP, time.localtime(when))
    return "-".join((_slug(campaign), _slug(prompt), _slug(slug), stamp))


def script_sha256(path) -> str:
    """SHA-256 of a script's source. Taken once, at `begin()`: that is the source the running
    interpreter actually holds, and re-taking it later would read a file the process is not
    running."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git_provenance(cwd=REPO_ROOT) -> dict:
    """`git rev-parse HEAD` and whether the tree was clean, as they stood at launch."""

    def run(args):
        try:
            return subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "git_head": run(["git", "rev-parse", "HEAD"]) or "unknown",
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
    }


# =================================================================================================
# files


def _repo_path(path) -> str:
    """A repository-relative path where that is meaningful, an absolute one otherwise."""
    absolute = os.path.abspath(path)
    if absolute.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(absolute, REPO_ROOT)
    return absolute


def _resolve(path) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def read_json(path):
    """The file's contents, or `None` if it is absent, unreadable or not JSON. A lister that
    raises on one bad directory tells you about none of the others."""
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_json_atomic(path, payload) -> None:
    """Write via a temporary file **in the same directory** and `os.replace`, so that a reader
    either sees the previous contents or the new ones and never a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# =================================================================================================
# liveness — README §0 item 3, which is why this package exists at all


def pid_alive(pid) -> bool:
    """Whether a process exists, by `os.kill(pid, 0)`.

    **Never identify a process by matching a pattern against command lines.** `pgrep -f <name>`
    also matches the *polling shell's own* command line, so "wait until no process matches" can
    never be satisfied; that mistake left nineteen immortal poll loops, fifteen of them waiting on
    a job that was already dead (`prompts/run-registry/README.md` §0 item 3).
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by somebody else
        return True
    except OSError:
        return False
    return True


def heartbeat_age(status, now=None):
    """Seconds since the heartbeat, or `None` if there is no readable one."""
    try:
        beat = datetime.fromisoformat(status.get("heartbeat"))
    except (AttributeError, TypeError, ValueError):
        return None
    if beat.tzinfo is None:  # a naive stamp is local time
        beat = beat.astimezone()
    return ((now or datetime.now(timezone.utc)) - beat).total_seconds()


def liveness(status, stale_after, now=None) -> str:
    """`alive`, `stale`, `finished` or `unknown` for one `status.json` payload.

    A run is **alive** iff its pid answers `kill -0` *and* its heartbeat is inside the window the
    caller supplies. A run whose state is still `running` but which fails either test is **stale**:
    a crashed job pretending to be alive, which is the thing worth shouting about.
    """
    if not status:
        return "unknown"
    if status.get("state") in TERMINAL_STATES:
        return "finished"
    if status.get("state") != "running":
        return "unknown"
    if not pid_alive(status.get("pid")):
        return "stale"
    age = heartbeat_age(status, now)
    if age is None or age > stale_after:
        return "stale"
    return "alive"


# =================================================================================================
# the run


def _not_a_ledger(path, number, line) -> ValueError:
    """The error `known()` raises rather than reporting "nothing done" for a file it cannot read.

    It names the other field, because the caller who gets here has almost always put the durable
    thing their results live in where the unit ledger goes.
    """
    return ValueError(
        f"checkpoint {path}: line {number} is not a JSON record, so this file is not a unit "
        f"ledger: {line[:60]!r}. known() will not report it as 'nothing done yet' — a resume that "
        f"believed that would recompute work that is already on disk. If this path is where the "
        f"job's results live, name it in the manifest's results field instead of checkpoint; "
        f"record() and known() read only the ledger."
    )


class Run:
    """One run directory. Construct it with `begin()`."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.id = os.path.basename(self.path)
        self.manifest = read_json(self.manifest_path) or {}

    # --- paths ---------------------------------------------------------------------------------

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.path, "manifest.json")

    @property
    def status_path(self) -> str:
        return os.path.join(self.path, "status.json")

    @property
    def stdout_path(self) -> str:
        return os.path.join(self.path, "stdout.log")

    @property
    def stderr_path(self) -> str:
        return os.path.join(self.path, "stderr.log")

    @property
    def checkpoint_path(self):
        """The unit ledger, and nothing else. `record()` appends to this and `known()` reads it;
        neither ever looks at `results_path`."""
        recorded = self.manifest.get("checkpoint")
        return _resolve(recorded) if recorded else None

    @property
    def results_path(self):
        """The durable thing this job's results live in — a pipeline run's datastore — which the
        registry names. README §0 item 4 is a datastore written into a session scratchpad with
        nothing on disk saying where the results went; §0 item 5 is a resume, which must know what
        it is resuming into. The registry never writes it. It reads it, read-only, in one place:
        `finish(..., fingerprint=True)` takes its content fingerprint (store-fingerprint prompt
        04)."""
        recorded = self.manifest.get("results")
        return _resolve(recorded) if recorded else None

    # --- status --------------------------------------------------------------------------------

    def status(self) -> dict:
        return read_json(self.status_path) or {}

    def _update(self, **fields) -> dict:
        status = self.status()
        status.update(fields)
        status["heartbeat"] = now_iso()
        write_json_atomic(self.status_path, status)
        return status

    def heartbeat(
        self, units_done=None, units_total=None, pid=None, stage=None
    ) -> dict:
        """Refresh the heartbeat, and with it whatever is now known. A launcher that spawns a
        detached child passes the child's `pid` here once it has one.

        `stage` is for a job that has no total to count against — a `main.py` pipeline reports
        progress per stage and per work queue, never as one number, and inventing a denominator
        would be worse than saying where it has got to. It is the field the hand-written
        `handover-A3-baseline-lambdacdm.manifest.json` carried as `stage_reached`, reconstructed
        by hand from a stdout file after the fact.
        """
        fields = {
            "units_done": units_done,
            "units_total": units_total,
            "pid": pid,
            "stage": stage,
        }
        return self._update(**{k: v for k, v in fields.items() if v is not None})

    def finish(self, state, exit_code=None, *, fingerprint=False) -> dict:
        """Set the terminal state and the exit code. The registry records; it does not kill,
        restart or reap, so `killed` is something a caller reports, not something observed.

        With ``fingerprint=True``, and a manifest that names `results`, it first fingerprints that
        store (`RunRegistry.stores.fingerprint_store`, read-only, with this run as `taken_by` so
        that it does not refuse its own caller) and puts the fingerprint in `status.json` under
        `fingerprint`. Where the store's sidecar is a problem-free registry sidecar the fingerprint
        is also written there, replacing only its `fingerprint` field, so that a run that changed
        its store never leaves a stale one in the place a reader looks; otherwise
        `fingerprint_sidecar` says why it was not. Pass it only once the store is closed: a
        driver's signal handler, which runs with the pool still open, does not.

        **A fingerprint never changes how a run ended.** Any refusal or error is recorded as
        `fingerprint_error`, a string, and the state and exit code given are written regardless.
        It never raises for the fingerprint's sake. The manifest is never touched.
        """
        if state not in TERMINAL_STATES:
            raise ValueError(
                f"terminal state must be one of {TERMINAL_STATES}: {state!r}"
            )
        taken = {}
        if fingerprint:
            try:
                taken = self._fingerprint_results()
            except Exception as e:
                taken = {"fingerprint_error": f"{type(e).__name__}: {e}"}
            except BaseException as e:
                # an interrupt while fingerprinting: the state is still written, then it goes on
                self._update(
                    state=state,
                    exit_code=exit_code,
                    fingerprint_error=f"interrupted: {type(e).__name__}",
                )
                raise
        return self._update(state=state, exit_code=exit_code, **taken)

    def _fingerprint_results(self) -> dict:
        """The `status.json` fields `finish(..., fingerprint=True)` records. May raise."""
        results = self.results_path
        if results is None:
            return {
                "fingerprint_error": "the manifest names no results store to fingerprint"
            }
        from .stores import _sidecar_refusal, fingerprint_store, read_sidecar

        reading = read_sidecar(results)
        taken = fingerprint_store(
            results,
            write=reading.ok,
            runs_root=os.path.dirname(self.path),
            taken_by=self,
        )
        return {
            "fingerprint": taken["fingerprint"],
            "fingerprint_sidecar": (
                "written"
                if taken["wrote"]
                else f"not written: {reading.path} is not a problem-free registry "
                f"sidecar ({_sidecar_refusal(reading)})"
            ),
        }

    # --- the checkpoint ------------------------------------------------------------------------

    def record(self, unit, data=None) -> dict:
        """Append one completed unit to `checkpoint.jsonl`, `flush`, `fsync`, and beat.

        The record carries the provenance triple — script hash, git head, dirty flag — taken at
        launch, exactly as `docs/handover/realistic_large_x.py` stamps each cell.

        It appends to the **ledger** and to nothing else. A run that declares no ledger is refused,
        by name: the caller who reaches here is usually one who copied the pipeline pattern — whose
        results are a datastore — and then gave their job units, and the fix they need is a ledger
        of their own, not a method that writes JSON-Lines into whatever the manifest happens to
        name.
        """
        path = self.checkpoint_path
        if path is None:
            store = self.manifest.get("results")
            names = (
                f"It names a results store ({store}), which this method will never write to: "
                f"the registry names where a job's results live, it does not write them. "
                if store
                else ""
            )
            raise ValueError(
                f"run {self.id} declares no checkpoint ledger in its manifest, so record() has "
                f"nothing to append to. {names}"
                f"A job with units to record needs a ledger of its own: pass checkpoint=True to "
                f"begin() (or a path to a .jsonl file), which is the field record() and known() "
                f"read."
            )
        record = {
            "unit": unit,
            "recorded": now_iso(),
            "script_sha256": self.manifest.get("script_sha256"),
            "git_head": self.manifest.get("git_head"),
            "git_dirty": self.manifest.get("git_dirty"),
            "data": data,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done = self.status().get("units_done") or 0
        self.heartbeat(units_done=done + 1)
        return record

    def known(self, notice=sys.stderr) -> dict:
        """Every completed unit already on disk, keyed by unit, so a resume recomputes only what
        it lacks.

        **Policy on a script-hash mismatch: discard, with a notice.** A record written by a
        different version of the script is dropped and its unit recomputed, rather than the run
        refusing to start — an edited script is then self-healing instead of stranding the operator
        on a file they must know to delete, and the notice is what makes the discard visible. This
        is `realistic_large_x.load_checkpoint`'s choice, generalised. Records are never blended
        across script versions, and an existing record is **never re-stamped** with the running
        hash: that would falsify the provenance the stamp exists to provide. Where the manifest
        records no script hash there is nothing to gate on and every record is reused.

        A malformed trailing line is ignored: the file is appended to as each unit lands, so a kill
        can truncate the last record. **A file that is not a ledger at all is a different thing and
        raises.** Answering `{}` — "nothing has been done yet" — for a file this method cannot read
        is the failure a resume then acts on by recomputing everything, and on the A3 baseline that
        is 10 h 33 m. The two are told apart by where the unreadable line is and what it looks
        like: a torn record is the **last** line and is a prefix of a JSON object, so it begins
        `{`. Anything else — a line that does not, or a readable line after one that did not, or a
        file that is not even text — means the path names something that is not a unit ledger.

        A ledger that does not exist yet is not an error, and returns nothing quietly: that is
        every first run.
        """
        path = self.checkpoint_path
        units, stale = {}, {}
        if path is None or not os.path.exists(path):
            return units
        wanted = self.manifest.get("script_sha256")
        try:
            with open(path, "r") as handle:
                torn = None
                for number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    if torn is not None:  # the torn line was not the final one
                        raise _not_a_ledger(path, torn[0], torn[1])
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.startswith("{"):
                            raise _not_a_ledger(path, number, line) from None
                        torn = (number, line)
                        continue
                    unit = record.get("unit")
                    if wanted and record.get("script_sha256") != wanted:
                        stale[unit] = record.get("script_sha256") or "unstamped"
                        units.pop(unit, None)
                        continue
                    units[unit] = record
        except UnicodeDecodeError:
            raise _not_a_ledger(path, 1, "<not text>") from None
        if stale and notice is not None:
            hashes = ", ".join(sorted({str(h)[:12] for h in stale.values()}))
            print(
                f"checkpoint {path}: discarding {len(stale)} record(s) written by a different "
                f"version of the script ({hashes}); this script is {str(wanted)[:12]}. Those "
                f"units will be recomputed.",
                file=notice,
                flush=True,
            )
        return units


def begin(
    campaign,
    prompt,
    slug,
    purpose,
    argv=None,
    cwd=None,
    script=None,
    unit=None,
    expected_units=None,
    checkpoint=None,
    results=None,
    scope=None,
    heartbeat_means=None,
    pid=None,
    root=None,
    when=None,
):
    """Create the run directory, write the immutable manifest, and write `status.json` as
    `running`. Call this **before** the work starts.

    `checkpoint` is the **unit ledger** and only that: the append-only JSON-Lines file `record()`
    writes and `known()` reads. `checkpoint=True` puts one inside the run directory; a path puts it
    wherever the job's own convention says, which is how a resume in a new run directory reads the
    previous run's units.

    `results` is the **durable thing the job's results live in** — for a registered `main.py`
    pipeline run, its SQLite datastore — which the registry names and never writes. It is a
    different field from `checkpoint` because it is a different thing, and conflating them let
    `record()` append JSON-Lines to a datastore (README §0.2: the registry records *existence*;
    checkpointing is per-job and is sometimes already solved). It traces to §0 item 4, a datastore
    written into a session scratchpad with nothing on disk saying where the results went, and to
    §0 item 5, since a resume must know what it is resuming into.

    `pid` defaults to the calling process; a launcher that spawns a detached child passes the
    child's pid to `heartbeat()` once it has one. Every manifest field but one is one README §0
    would have caught something with; there are no others.

    The one is `results_store_id`, which is `prompts/datastore-portability` README §6.5 point 5.
    When `results` has a problem-free registry sidecar beside it (`RunRegistry.stores`), it holds
    that sidecar's `store_id`, and otherwise `null`: no store, no sidecar, a legacy sidecar, or a
    sidecar with a problem. It matches a run to its store after the store has moved, which a path
    cannot, and it is how `store copy` / `store move` recognise a store a running run is using.
    The sidecar is read here and never written. Manifests written before the field existed lack
    it, and are read exactly as before.

    `scope` is one line saying what the run covers, and is what tells a stranger whether the run
    is still relevant — the hand-written `handover-A3-baseline-lambdacdm.manifest.json` carried it
    by hand, because `argv` alone is exact and unreadable. Derive it from the job's own arguments
    rather than typing it: a field somebody must remember to update is a field that goes stale.

    `heartbeat_means` is one line saying what a fresh heartbeat on *this* job actually asserts.
    It exists because the honest answer differs per job — "a unit was recorded" for a script with
    units, something weaker for a Ray-parallel pipeline whose parent may be blocked inside
    `ray.get` — and a heartbeat that implies a guarantee it does not give is worse than none.
    `None` means this package's default: the heartbeat is refreshed by `record()` and by an
    explicit `heartbeat()`, so it means "a unit completed", and nothing else.
    """
    identifier = run_id(campaign, prompt, slug, when)
    path = os.path.join(root or DEFAULT_ROOT, identifier)
    os.makedirs(path, exist_ok=True)
    if os.path.exists(os.path.join(path, "manifest.json")):
        raise FileExistsError(
            f"{path} already holds a manifest; a manifest is written once"
        )
    if checkpoint is True:
        checkpoint = os.path.join(path, "checkpoint.jsonl")
    results_store_id = None
    if results:
        from .stores import read_sidecar  # here, because `stores` imports this module

        results_store_id = read_sidecar(results).store_id
    manifest = {
        "run_id": identifier,
        "created": now_iso(),
        "purpose": purpose,
        "campaign": campaign,
        "prompt": str(prompt),
        "argv": list(sys.argv if argv is None else argv),
        "cwd": cwd or os.getcwd(),
        "script": _repo_path(script) if script else None,
        "script_sha256": script_sha256(script) if script else None,
        "unit": unit,
        "expected_units": expected_units,
        "checkpoint": _repo_path(checkpoint) if checkpoint else None,
        "results": _repo_path(results) if results else None,
        "results_store_id": results_store_id,
        "scope": scope,
        "heartbeat_means": heartbeat_means,
    }
    manifest.update(git_provenance())
    write_json_atomic(os.path.join(path, "manifest.json"), manifest)
    run = Run(path)
    run._update(
        state="running",
        units_done=0,
        units_total=expected_units,
        pid=os.getpid() if pid is None else pid,
        exit_code=None,
    )
    return run


# =================================================================================================
# the lister


def _created_epoch(entry_path, manifest):
    try:
        return datetime.fromisoformat(manifest["created"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return os.path.getmtime(entry_path)


def list_runs(root=None, stale_after=DEFAULT_STALE_AFTER, now=None) -> list:
    """Every run under `root`, newest first.

    A directory with no manifest is reported, not skipped and not repaired: `var/runs/` holds run
    directories that predate this package, and they are evidence. Loose files beside the run
    directories are ignored. Nothing here writes or deletes anything.
    """
    root = root or DEFAULT_ROOT
    entries = []
    try:
        names = os.listdir(root)
    except OSError:
        return entries
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        manifest = read_json(os.path.join(path, "manifest.json"))
        status = read_json(os.path.join(path, "status.json")) or {}
        entries.append(
            {
                "id": name,
                "path": path,
                "has_manifest": manifest is not None,
                "purpose": (manifest or {}).get("purpose"),
                "state": status.get("state"),
                "pid": status.get("pid"),
                "units_done": status.get("units_done"),
                "units_total": status.get("units_total")
                or (manifest or {}).get("expected_units"),
                "stage": status.get("stage"),
                "liveness": liveness(status, stale_after, now=now),
                "created_epoch": _created_epoch(path, manifest or {}),
            }
        )
    entries.sort(key=lambda entry: entry["created_epoch"], reverse=True)
    return entries
