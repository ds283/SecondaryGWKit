# Working notes for Claude

## Open issues — keep the project-wide index in step

[`docs/OPEN_ISSUES.md`](docs/OPEN_ISSUES.md) is the project-wide index of open issues. It exists
because issues opened by one campaign were being lost when that campaign closed.

**Whenever you add, narrow or close an entry in a campaign board's §3 (Active and unresolved
issues) or §4 (Resolved issues), update `docs/OPEN_ISSUES.md` in the same commit.** Specifically:

- **Opening an issue** — add a one-line row under the right heading, with the campaign board name
  and a hook short enough to read at a glance.
- **Closing one** — delete its row. Do not keep a "resolved" section here; the board's §4 is the
  record.
- **Assigning one to a future campaign** — move its row into the matching §1 subsection, and add
  an `**Assigned (date):**` line to the board entry saying which campaign owns it and why.
- Either way, correct the **count** and the **Last updated** date in the index header.

The index is an *index*. One line per issue, pointing at the board that holds the measurements,
the impact statement and the next step. Never copy issue content into it; if the two disagree the
board is right.

## Campaign conventions

Remediation work is organised as campaigns under `prompts/<campaign>/`: a `README.md` holding the
plan, numbered prompt files, a `logs/` directory, and `IMPLEMENTATION_STATE.md` as the status
board. `README.md` §5 of each campaign states the rules that campaign runs under. The invariants
that hold across all of them:

1. **One commit per prompt.** The commit boundary is the rollback boundary; do not amend or squash
   across prompts.
2. **Every prompt writes a log** to `logs/NN-<name>.md` using the template in that campaign's
   README §5.1, and classifies every deviation from its prompt as `STRUCTURALLY REQUIRED`,
   `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit** — its own row, the
   item-level table, and §3/§4 — plus `docs/OPEN_ISSUES.md` per the rule above.
4. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. Scope creep destroys the revert-per-prompt property. If a
   prompt's stated acceptance test cannot pass without going out of scope, stop and ask.
5. **Commit messages**: imperative, capitalised subject under ~72 characters with no prefix tag; a
   blank line; a prose body saying what was wrong, what changed and how it was verified, wrapped
   at ~80 columns; then `Co-Authored-By: Claude <model name> <noreply@anthropic.com>`.
6. **Verification documents are additive.** When a re-run supersedes a measurement, add a new
   subsection recording it; do not rewrite the original, which was correct for the tree it was
   taken on.

## The datastore layer is `datastorekit`; the work queue is `raypoolkit`; the registry is `runregistrykit`

Since 2026-10-11 the generic datastore layer is the package `datastorekit`, from DatastoreKit
(`/Users/ds283/Documents/Code/DatastoreKit`, https://github.com/ds283/DatastoreKit), pinned in
`requirements.txt` at `v0.5.0` (moved to it on 2026-10-22 in the `dk-v050-adoption` campaign), and
the work queue `RayWorkPool` is the package `raypoolkit`, from RayPoolKit
(`/Users/ds283/Documents/Code/RayPoolKit`, https://github.com/ds283/RayPoolKit), pinned at `v0.3.0`
(moved to it on 2026-10-25 in the `rpk-v030-adoption` campaign, which made its
`raypoolkit.rehearsal_lock` this repository's rehearsal lock). ChamPBH and StochasticInstantons use
DatastoreKit too; this repository adopted both in the `dk-and-rpk-migration` campaign. Since
2026-10-15 the run registry is the package `runregistrykit`, from RunRegistryKit
(`/Users/ds283/Documents/Code/RunRegistryKit`, https://github.com/ds283/RunRegistryKit), pinned at
`v0.2.0` with the extra `[stores]`, adopted in the `runregistry-extraction` campaign and moved to
`v0.2.0` on 2026-10-18 in the `rrk-v020-adoption` campaign.

**None of the three is this repository's code.** A defect in one is recorded on the board of the
campaign that found it, and in `docs/OPEN_ISSUES.md`. It is reported to that package's own
repository, fixed there in a release, and reaches this repository by moving the pin in a commit of
its own: it is never patched here, and never worked around. A prompt whose acceptance cannot pass
without a change to a package stops and asks. **A pin is a tag**; an editable install is never used
for a production run. This repository keeps its own `ray` and `SQLAlchemy` pins.

**Imports name the package:** `from datastorekit import DatastoreObject`,
`datastorekit.SQL.ShardedPool`, `datastorekit.SQL.factory_base`, and
`from raypoolkit import RayWorkPool`. The name `RayWorkPool` stays bare at every call, because five
`ComputeTargets/tests` modules find pool constructions by `ast.Name`. The actor's module is
`importlib.import_module("datastorekit.SQL.Datastore")`. The tools are
`python -m datastorekit.tools.sharded_store` and `python -m datastorekit.tools.shard_key_audit`.
The registry is `import runregistrykit` (`runregistrykit.begin`, `runregistrykit.list_runs`, …)
and `runregistrykit.stores` for the store layer; its command line is `python -m runregistrykit`
(`list`, `end`, and `store show | create | fingerprint | copy | move | retire | amend`).

**This repository's side of the layer changes as usual:** the factories under `ObjectFactories/`,
`config/datastore.py` and `config/sharding.py`, `tools/inventory_report.py`, the tests in
`ObjectFactories/tests/` and `tools/tests/`, the store fixtures in `StoreFixtures/`, and every
caller. Until 2026-10-12 the factories lived at `Datastore/SQL/ObjectFactories/`, and their tests,
the inventory-report tests and the store fixtures at `Datastore/tests/`. Campaign records and dated
`docs/` prose from before then still use those paths, and are left as they were written. Until
2026-10-15 the registry was this repository's own package `RunRegistry/`; campaign records from
before then use that name, and are left as written.

## Long-running jobs — the run registry

A job that takes hours must not be something a conversation owns. `runregistrykit`
(RunRegistryKit, pinned at `v0.2.0`) records that one exists, in
`var/runs/<campaign>-<prompt>-<slug>-<timestamp>/`. It is configured by `runregistrykit.toml` at
the repository root, which names the runs root, the stores root, the distributions every manifest
records and the store fingerprinter, `config/fingerprint.py`'s `fingerprinter`. A run launched from
a git worktree is recorded in the main checkout's `var/runs/`. `prompts/run-registry/README.md` §0
lists the five failures in one working session that are the reason for each rule below, and
RunRegistryKit's `README.md`, "The rules", tells each rule with the failure it answers, and its
section "Ending a run whose end was never recorded" tells rule 5 below. It records;
it does not schedule, supervise, restart or lock. It deletes nothing but a store's own files, and
those only through `store retire`, which a person runs and which leaves the store's sidecar behind
as its record. It never deletes a run directory, a sidecar or any other record.

This repository's launchers of registered runs live in `RunDrivers/`, next to the tests in
`RunDrivers/tests/`: `scoped_pipeline_run.py` (a scoped `main.py` pipeline) and
`quadsource_atol_sweep.py` (the tolerance sweeps and the `--build` A3 v2 runs under). Until
2026-10-19 they were `docs/gktk-remedial/scoped_pipeline_run.py` and
`docs/handover/quadsource_atol_sweep.py`; campaign records and dated `docs/` prose from before then
use those paths, and are left as written.

1. **Discovery.** Run `./venv/bin/python -m runregistrykit list` at the start of a session and
   before launching anything long, and report anything `running` or stale. A job's existence must
   not depend on a conversation remembering it.
2. **Liveness.** A run is alive iff its pid answers `kill -0` **and** its heartbeat is inside the
   staleness window. **Never a command-line pattern match**: `pgrep -f <name>` matches other
   pollers waiting on the same job, so the condition can never be satisfied — nineteen poll loops
   that could not exit, fifteen of them waiting on a job already dead. The test is made **on the
   host and boot the run recorded**: a pid names a process only there. A `running` run recorded on
   another host is `foreign`, and is not tested here. A `running` run from an earlier boot of this
   host is `stale` whatever its pid answers. A run with no host recorded, which is every run
   written before `v0.2.0`, is judged as before.
3. **Launch.** Detached (`nohup`, `disown`; `setsid` is not on macOS), manifest written *first*,
   stdout and stderr into the run directory, pid recorded — the launcher passes a detached child's
   pid to `Run.heartbeat(pid=…)`.
4. **Do not babysit.** Launch, verify in **one** check that it started, and **end the turn**.
   Polling a long job re-sends an entire context per poll and learns nothing; that, not wall-clock,
   is what makes waiting expensive. Resume once, when it is done.
5. **The route after a hard kill.** A run killed by `SIGKILL`, a power loss, a reboot or the OOM
   killer never records its end, and its status says `running` for ever. `list` shows it `stale`,
   `!!`, and a resume's `begin` on its store is refused, with nothing created: the registry does not
   decide that a run is dead. A person checks that it is over, then runs
   `./venv/bin/python -m runregistrykit end RUN_ID --reason TEXT` **on the host it ran on**. `end`
   refuses a pid that answers `kill -0`, with no override; `--unverified` is for a run on another
   host, or one whose status does not say, once checked there. The run is then `lost`, and the
   resume is begun. **`Run.finish` is never called by hand on a run that is not the caller's**: it
   checks nothing. The one case `end` refuses that is not a live run is a killed process its parent
   has not yet reaped, which still answers `kill -0` (RunRegistryKit's
   `[04-a-killed-process-not-yet-reaped-answers-kill-0]`); a detached launch through `nohup sh -c`
   is reaped by its `sh`. After `end`, check
   `./venv/bin/python -m raypoolkit.rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk`: a
   `SIGKILL`ed driver's GCS server may be left alive, reparented to pid 1, stuck in its own shutdown
   where a `SIGTERM` cannot end it (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`).
   One still listed ten minutes after the kill is ended **by a person** with `kill -KILL <pid>`, by
   its pid, never by pattern and never with `ray stop`; until then the rehearsal lock cannot be
   released.
6. **Durability.** `var/runs/`, which is gitignored and in the repository. Never a session
   scratchpad, never `/tmp` — a 462-row census died with a scratchpad and cannot be re-examined.
7. **Provenance.** Script hash and git SHA on every checkpoint record, taken at launch. A record
   whose script hash does not match is discarded and recomputed, never blended and **never
   re-stamped**: re-stamping falsifies the thing the stamp exists to prove.

## The shared rehearsal lock

Three projects on this machine (this one, ChamPBH and StochasticInstantons) run Ray builds. One
lock, the directory `~/.raypoolkit/rehearsal-lock`, lets only one rehearsal or long build run at a
time. It is the module `raypoolkit.rehearsal_lock` of the pinned `raypoolkit`, whose contract is
RayPoolKit's `docs/rehearsal-lock.md` at `v0.3.0`. This repository's values are
`--project SecondaryGWKit`, `--ray-temp-dir /tmp/ray-sgk` (`config/ray_local.py`) and
`--ray-address in-process`.

- **The protocol.** `acquire` before the first launch; `set-pid` with the driver's pid at each
  launch; held through the readers; `release` after them, once no Ray process is left. `status` and
  `ray-processes` only look.
  ```bash
  ./venv/bin/python -m raypoolkit.rehearsal_lock acquire --project SecondaryGWKit --purpose "<text>" --ray-temp-dir /tmp/ray-sgk --ray-address in-process
  ./venv/bin/python -m raypoolkit.rehearsal_lock set-pid --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk <pid>
  ./venv/bin/python -m raypoolkit.rehearsal_lock release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk
  ```
- **A held lock (exit 3) means stop and ask the user.** Never wait, retry or remove it; the module
  has no option to. Exit 3 also comes while the old location `~/ScienceData/.rehearsal-lock`
  exists: a project not yet moved to `v0.3.0` holds it, or it is stale.
- **The lock finds an instance by its temp dir.** Every Ray start here gives Ray `_temp_dir`
  (`RunDrivers/tests/test_ray_temp_dir.py`); one that does not is invisible to the lock.
- **Every Ray start runs with token authentication.** `config/ray_local.py` sets
  `RAY_AUTH_MODE=token` at import unless the environment already holds a value
  (`RunDrivers/tests/test_ray_auth_mode.py`). The token is `~/.ray/auth_token`, which Ray writes if
  it is missing. Keep it mode 600 on each host (`chmod 600`): Ray writes it readable by all.
- **Each lock command is a shell command of its own**, never the one that launches a build, and the
  only process that command starts: output by `>>` redirection, never a pipe such as `| tee`. A
  shell whose command line names `/tmp/ray-sgk` makes the lock list everything else it started.
- Every Ray driver names its address (`local` for this repository's), never `auto`, and `ray stop`
  is never run: it stops every project's Ray processes.

## Repository mechanics

- **The venv** is Python 3.13, MacPorts' 3.13.16, on the laptop and on the Mac Studio. It is
  built with `/opt/local/bin/python3.13 -m venv venv` and then
  `./venv/bin/pip install -r requirements.txt`, never by hand. It moved from 3.12.15 on
  2026-10-25 in the `py313-venv` campaign (the laptop in its prompt 01, the Mac Studio in 02).
- **Tests** live in `<package>/tests/` as `unittest` modules and run from the repository root. There
  are seven roots, each run by its own command:
  ```bash
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s ObjectFactories/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s RunDrivers/tests -t .
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s tools/tests -t .
  ```
  They must not need Ray or a datastore — use the stand-in model pattern in
  `ComputeTargets/tests/test_tk_source_functions.py` and `docs/spec-code-audit/scripts/`. A test
  that needs a store on disk builds one with `StoreFixtures/`, which any root may import:
  `shard_store_fixtures` (a primary with placeholder shards), `real_store_fixtures` (a small real
  store, no Ray) and `standin_pool` (a real `ShardedPool` on stand-in actors).
- **Formatting** is `black`, with no configuration. Run `./venv/bin/python -m black <files>` before
  committing; the tree outside `prompts/` is clean under `--check` (49 orchestrator measure
  scripts under `prompts/datastore-generic/` are records, left as written). The one exception is
  three scripts under `docs/handover/a3-v2/seam/scripts/`. They are records too, byte-identical to
  the run whose outputs record their SHA-256, so they are left unformatted.
- **`main.py` cannot be imported** — it parses `sys.argv`, opens a Ray connection and a
  `ShardedPool` at module scope, then runs the pipeline. To exercise it, use
  `RunDrivers/scoped_pipeline_run.py`; to extract one of its functions
  for a test, use `ComputeTargets/tests/test_main_plumbing.load_main_py_functions`, which reads it
  with `ast` rather than executing it.
- **Author conventions that are conventions, not defects** — do not "correct" them: $a_0$ is
  absorbed, never "set to 1" (it lives in $k/a_0$ and $a_0\eta$); the Green's function is the
  unit-jump $\bar G_k$ in $z$; $c_s^2$ is `wPerturbations` in the transfer-function sector while
  $w_0$ is `wBackground` inside $f$; Jacobian and phase sign choices are conventions. The spec §0
  blocks under `docs/spec/` record these.
- **Redshift arithmetic.** Integration is in $\log(1+z)$. Converting $z\to\log(1+z)$ costs ~1 ulp
  and is safe; $\log(1+z)\to z$ is irreducibly lossy at large $z$ (at $z\sim5\times10^{14}$ the
  recoverable $1+z$ has a granularity of ~0.02), so it must never appear in an equality-like
  comparison such as a region-coverage guard. It is harmless where the recovered $z$ is only a
  quadrature limit.
