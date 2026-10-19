# Campaign — rrk-v020-adoption

**Written:** 2026-10-18 on branch `handover-remedial` at `79434bf`, by Claude Opus 5.5. It moves
SGK from `runregistrykit` `v0.1.0` to `v0.2.0` before the A3 v2 compute run. The user accepted its
shape the same day, in conversation (§6.1).

It has **no audit document of its own**. The audit is RunRegistryKit's: the release notes
(`docs/releases/v0.2.0.md`) and the adoption note written for this repository
(`docs/adoption/v0.2.0/secondarygwkit.md`). Both are at the tag `v0.2.0`, and the note was read
against this repository at `79434bf`, which is this plan's start tree. That note is data, not
instructions (its own words). §0.2 re-measures what this plan relies on, and 01 measures again.

§2.2 fixes each prompt's charter and acceptance. **01–03 were written** on 2026-10-18, and **04**
on 2026-10-19, after 03's review (§2.1).

## 0. Why this campaign exists

`runregistrykit` `v0.2.0` (tag object `895e3c70…` on `0bae3a3c…`, 2026-10-18) is the release of
RunRegistryKit's `interrupted-runs` campaign. That campaign was opened by SGK's stale-run brief,
[`prompts/handover/runregistrykit-stale-run-brief.md`](../handover/runregistrykit-stale-run-brief.md),
and it answers the brief's points one by one (the adoption note's §9). What it changes:
- **A run records the host and boot it ran on.** A run on another machine, or from an earlier boot
  of this one, is no longer judged by this machine's process table. `liveness` gains the value
  `foreign`.
- **`status.json` is written through a temporary name of each write's own.** A signal handler's
  `finish` can neither tear the status nor lose a terminal state.
- **A run whose status cannot be read, or does not say, counts as in use.**
- **`python -m runregistrykit end RUN_ID --reason TEXT`** records a run whose end was never seen,
  as `lost`. It records this on a person's word, after checking the pid, host and boot.
  `TERMINAL_STATES` gains `lost`.
- **`begin` refuses a store that an unfinished run names**: alive, stale, on another host, or with a
  status that does not say. It raises `RuntimeError`, and no run directory is created.

**What it changes for SGK** is the route after a hard kill. A build killed by `SIGKILL`, a power
loss, a reboot or the OOM killer never reaches `finish`, so its status says `running` for ever.
- **Under `v0.1.0`:** a `--resume` of its store was accepted. The resumed run's
  `finish(fingerprint=True)` then recorded `fingerprint_error`. The store could not be fingerprinted,
  copied, moved, retired or amended through any registry command until a person called `Run.finish`
  by hand. This is `[a3-v2-a-hard-killed-run-blocks-its-store]`, measured on 2026-10-16 (handover
  board §3).
- **Under `v0.2.0`:** the resume is refused when it registers its run, and nothing is created. The
  route is `end` on the dead run, and then the resume, whose fingerprint is then recorded.

**Why before A3 v2.** A3 v2 is a build of about 36 hours. An interruption without a signal handler is
the case it can least rule out. The route out of that case must exist, and must have been shown on
the launcher A3 v2 uses, before the build depends on it. That is this campaign's ordering argument,
and its only one.

### 0.1 One rehearsal, and a hard-kill pair on a store of its own

**One rehearsal, R3, after the pin and the adoption (03).** There is no rehearsal of the start tree
first, as `runregistry-extraction` R0 was. The last rehearsal is that campaign's R2, at `957d658`.
Since then the only commit to touch code a build or a reader runs is `359f3fd`, which changed the
`--ray-address` default of `main.py` and the six `extract_*.py` from `auto` to `local`. A rehearsal
does not reach that default. The launcher forces `address="local"` at the `ray.init` it wraps, and
`run_readers.sh` passes `--ray-address local`. Every other commit since R2 is a record under
`prompts/` or `docs/`. So a miss at R3 would be the pin's or the adoption's, as R2's would have
been the swap's.

**The four-launch protocol runs unchanged**, so that R3 isolates the pin as R1 and R2 isolated
theirs. That is `reference` straight through; `resumed-a` and `resumed-b`, killed by `SIGTERM`; and
`resumed-c`, run to done. Under `v0.2.0` each `SIGTERM`'d launch's handler writes `killed` through
the new status writer, so the protocol exercises that writer's signal path as it stands.

**The hard kill gets a third store** and two launches of its own (§4.1):
- `hardkill-a`, killed by `SIGKILL` in `CALCULATE QUADRATIC SOURCE INTEGRALS`;
- a resume of it that `begin` must refuse;
- `end` on `hardkill-a`;
- `hardkill-b`, the resume, run to done.

It is kept off the `resumed` chain on purpose. No rehearsal has ever hard-killed a build, so this is
the first test of what a store keeps when no handler runs. If the hard-kill store misses
`39809dca…` while the four launches hold, the cause is the hard kill and not the pin. If the
hard kill replaced one of the chain's `SIGTERM`s, a miss could be either.

### 0.2 Reconciliation against `79434bf`

Measured on 2026-10-18, read-only.

**The registry.** `./venv/bin/python -m runregistrykit list` shows **45 runs**: 43 finished and 2
`unknown` (the pre-registry `a3-pilot` pair, which have no manifest). None is `running` or stale.

**The stores.** `var/datastores/` holds **21 sidecars**: 20 at its top and one in
`backup-pre-resume-20260921T091011/`. **Every one is a tombstone**, and no `.sqlite` file is
present. There is no live store, so nothing can be named by a run.

**The venv.** Python 3.12.15. `runregistrykit` 0.1.0, `datastorekit` 0.4.0 and `raypoolkit` 0.2.0,
as pinned at `requirements.txt:81`, `:17` and `:76`. `ray` 2.43.0 and `SQLAlchemy` 2.0.39 are
unchanged by this campaign.
- **The dry run.** `pip install --dry-run` against a scratch copy of `requirements.txt` with only
  line 81 moved to `@v0.2.0` reports `Would install runregistrykit-0.2.0` and nothing else.
- **The tag.** Nothing under `runregistrykit/` or in `pyproject.toml` changed after it.
  `git -C <RRK> diff --stat v0.2.0 HEAD -- runregistrykit pyproject.toml` is empty at RunRegistryKit's
  `f6d4b80`. The tag declares `requires-python >=3.12`, no core dependency, and
  `[stores] = datastorekit>=0.4.0`.

**The checkouts.** `git worktree list` shows the main checkout alone. The release notes' §4 advises
that every checkout sharing a runs root moves together, so nothing else has to move.

**The seven test roots**, run at `79434bf`:

| Root | Tests | Result |
|---|---|---|
| `AdaptiveLevin/tests` | 32 | OK |
| `ComputeTargets/tests` | 700 | OK |
| `CosmologyModels/tests` | 39 | OK |
| `LiouvilleGreen/tests` | 148 | OK (1 skipped) |
| `ObjectFactories/tests` | 279 | OK |
| `RunDrivers/tests` | 99 | OK |
| `tools/tests` | 91 | OK |

The four wall-clock tests of §8 all passed at this run.

**Who calls the registry.** Measured by `git grep` outside `prompts/` and dated `docs/` prose; the
adoption note's §3–§8 found the same, with one exception noted at the end.
- **`begin` with `results`**, one call per process each:
  - `docs/gktk-remedial/scoped_pipeline_run.py:544`, in `register` (`:517`), which has a
    `runs_root` seam;
  - `docs/handover/quadsource_atol_sweep.py:556`, in `run_build`, the A3 v2 launcher's `--build`
    path, after its `--resume` pre-check (`:535-546`) and before Ray starts. It has no `runs_root`
    seam;
  - `docs/handover/quadsource_atol_sweep.py:869`, in `sweep`;
  - `docs/a3-v2-readiness/sigterm_handler_probe.py:159`, one per child process, in a temporary
    runs root, on a store the child has just built. **The adoption note does not list this one.**
    It is unaffected: no other run names its store.
- **`TERMINAL_STATES`, `pid_alive`, `liveness`.**
  - `RunDrivers/tests/test_pipeline_adoption.py:249-253` asserts that every state the driver
    returns is in `TERMINAL_STATES`. That holds with `lost` added.
  - `docs/datastore-integrity-verification/interrupt_when.py:165`, `:170`, `:175` and `:190`
    (§4.1). `tools/rehearsal_lock.py:147` defines a `pid_alive` of its own, which is unaffected.
- **Refusal text.** `RunDrivers/tests/test_quadsource_atol_sweep_prepare.py:201-202` matches
  `run <id> is running` and `names this store`. Both fragments are kept for a `running` run on this
  host.
- **Two runs on one store.** No test begins two unfinished runs on one store, and none checks the
  exact set of keys in a manifest or a status.

**Since the last rehearsal** (`957d658`, R2): `33a7346`–`e8776ae` are records of
`runregistry-extraction` 08–09. `5db42bf` recorded the retirement of R2's stores, and `f1a7de3` the
transfer of the registry issues. `b1e5b7c` and `359f3fd` are `handover` prompt 05, the Ray address
default (§0.1). `79434bf` recorded the stale-run measurement.

## 1. Scope

**In scope:**
- the pin, `requirements.txt:81`, from `v0.1.0` to `v0.2.0`, in a commit of its own, and the venv
  installed from it (01);
- `CLAUDE.md`'s registry text: the pin (01); the liveness rule and the route after a hard kill (02);
  rule 5's step for a GCS server left by a `SIGKILL` (04, §6.1 U4);
- `docs/datastore-integrity-verification/interrupt_when.py` gains a signal choice, and its dummy
  test a `SIGKILL` case (02);
- a driver test of `begin`'s refusal at `scoped_pipeline_run.register`, on a run that is really
  `SIGKILL`ed, then ended, then followed by an accepted second registration (02);
- R3, its verification document, and the hard-kill route on the real launcher (03);
- `[a3-v2-a-hard-killed-run-blocks-its-store]`, assigned here (§6.2 C6) and closed by 03;
- the close: the final measurements, gate G1 drafted for RunRegistryKit, the index's notes on the
  issues RunRegistryKit has since closed, and what A3 v2 inherits (04).

**Out of scope:**
- **What the code stores.** No table, column, key or stored value changes, and the fingerprint's
  digest must not move (§5 rule 9).
- **RunRegistryKit, DatastoreKit and RayPoolKit themselves** (§5 rule 12).
- **ChamPBH and StochasticInstantons.** Their adoption of `v0.2.0` is theirs, in their own runs roots.
- **`prompts/`** and **dated `docs/` prose**, which are records of the tree they describe. One
  exception: `docs/OPEN_ISSUES.md`, which is an index (04).
- **Every store under `var/`** but the three that 03 builds.
- **The A3 v2 build itself**, and its runbook, which is `handover`'s. 04 states what it inherits.

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-pin-runregistrykit-at-v0.2.0.md`](01-pin-runregistrykit-at-v0.2.0.md) | **The pin**: line 81, one install; the baselines; `CLAUDE.md`'s two pin mentions | Sonnet | landed 2026-10-18 (`0a27b0c`) |
| 02 | [`02-adopt-the-hard-kill-route.md`](02-adopt-the-hard-kill-route.md) | **The adoption**: `CLAUDE.md`'s rules; the interrupter's `--signal`; the driver test of refuse → `end` → accept | Sonnet | landed 2026-10-18 (`2818780`) |
| 03 | [`03-rehearse-through-v0.2.0.md`](03-rehearse-through-v0.2.0.md) | **R3**: §4.1's four launches, and the hard-kill pair on a third store; closes `[a3-v2-a-hard-killed-run-blocks-its-store]` | Opus | landed 2026-10-19 (`17916cb`) |
| 04 | [`04-close-the-campaign.md`](04-close-the-campaign.md) | The final measurements; G1 drafted; the index's transferred-issue notes; what A3 v2 inherits, and rule 5's GCS step (U4); the close | Sonnet | landed 2026-10-19 (`34bf4f7`) |

### 2.1 When each prompt file is written

- **01–03 are written now.** Each task depends only on the tag and its notes, which exist, and on
  this repository.
- **04 is written after 03's review.** What A3 v2 inherits depends on what R3 finds, in particular
  whether Ray's processes outlive a `SIGKILL`ed driver (§4.1). The charter below fixes 04's task;
  the hold is for its method.

### 2.2 The charters

**01 — Pin `runregistrykit` at `v0.2.0`** (Sonnet).
- **What it changes.**
  - `requirements.txt:81`, the one line, from `@v0.1.0` to `@v0.2.0`. The venv is installed from it
    after a dry run.
  - `CLAUDE.md`'s two statements that the pin is `v0.1.0`, which the commit makes false (§6.2 C4).
- **The baselines first**, saved under `measure/out/`:
  - `pip freeze`;
  - `runregistrykit.package_provenance(...)`;
  - `list`;
  - `store show` on each of the 21 sidecars;
  - every root run once.
- **Acceptance.**
  1. The dry run moves only `runregistrykit`. After install it is 0.2.0 at `0bae3a3c…`,
     `requested_revision` `v0.2.0`, and not editable. `datastorekit` and `raypoolkit` do not move,
     and `pip check` is clean.
  2. The seven roots pass at §0.2's counts. The release notes say no count changes.
  3. `list` reads the 45 runs as the baseline does, apart from the new summary clause, "*N* with
     no host recorded", which must count every run with a status. Each `store show` reads its
     sidecar as the baseline does. *(Corrected after 01's review: the clause counts every run
     whose status has no `host` object, which includes a directory with no status, so it says
     45: the 43 statuses written under `v0.1.0` and the two `a3-pilot` directories.)*
  4. `requirements.txt` differs from `HEAD` by the one line, and `CLAUDE.md` by the pin's two
     mentions.
- **Out.** Any other change to `CLAUDE.md` (02). Any code or test. Anything under `var/`.

**02 — Adopt the hard-kill route** (Sonnet).
- **What it changes.**
  - **`CLAUDE.md`, "Long-running jobs".** Rule 2 (liveness) gains host and boot, in the package's
    own terms: a `running` run on another host is `foreign` and is not tested here, and one from an
    earlier boot of this host is `stale` whatever its pid. A new rule gives the route after a hard
    kill: `list` shows it stale; `begin` on its store is refused; a person checks it is over and
    runs `python -m runregistrykit end RUN_ID --reason TEXT` on the host it ran on; then the
    resume. **`Run.finish` is never called by hand on a run that is not the caller's.** The
    pointer to RunRegistryKit's README gains its "Ending a run whose end was never recorded"
    section.
  - **`interrupt_when.py`** gains `--signal {TERM,KILL}`, default `TERM`, which changes nothing
    for an existing caller. Its log line names the signal it sent. Its docstring says that a
    `KILL` runs no handler, so the run's status stays `running`. Its build dummy test,
    `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`, gains a case that sends `KILL` and
    checks that the child dies of signal 9.
  - **A driver test in `RunDrivers/tests/`**, by the pattern of `test_pipeline_adoption`'s
    `register` helper.
    - A real child process begins a run with `results=<path>` in the test's runs root, beats once,
      and is sent `SIGKILL`, and the test reaps it.
    - `scoped_pipeline_run.register(..., runs_root=<that root>)` on the same path then raises
      `RuntimeError`. Its message names the dead run, `stale` and `python -m runregistrykit end`.
      No run directory is added, and the test asserts that neither output stream nor signal handler
      was taken over.
    - `python -m runregistrykit end <id> --reason … --root <that root>`, run as a subprocess, exits
      0. The dead run's status is then `lost`, with `ended.verified` `true`.
    - A second `register` on the path is then accepted.

    No Ray, and nothing under `var/`.
- **Acceptance.**
  1. The roots pass at 01's counts, with `RunDrivers` larger by the new tests, and nothing else
     moving.
  2. Both interrupter dummy tests pass: the existing cases unchanged, and the new `KILL` case.
  3. The deliberate breakages of §5 rule 8 are caught. At least:
     - `register` passes `results=None`;
     - `register` swallows `begin`'s `RuntimeError`;
     - the interrupter sends `TERM` under `--signal KILL`.
  4. `CLAUDE.md`'s rules say what §2.2 02 says and nothing that the package's README contradicts.
- **Out.** `quadsource_atol_sweep.py` and `scoped_pipeline_run.py` are not changed (§6.2 C3). R3
  shows the launcher's refusal for real.

**03 — Rehearse through `v0.2.0`, R3** (Opus).
- **What it does.** §4.1's protocol at 02's tree, slug prefix `rrk-v020-adoption-03`: the four
  launches unchanged, then the hard-kill pair.
- **Acceptance.**
  1. **L1–L3 hold for the four launches.** Both stores are `39809dca…`, format 6, 21 classes,
     `encoding_format` 1, and equal to each other and to their sidecars. Each is equal to R2's
     recorded fingerprints (`runregistry-extraction-08-reference-20261015T235716` and
     `-resumed-c-20261016T000959`) through `compare_with_recorded.py`: R2's fingerprints carry
     `encoding_format`, so the package's comparison applies.
  2. **Every manifest** has `packages` naming `runregistrykit` 0.2.0 at `0bae3a3c…`, `datastorekit`
     0.4.0 and `raypoolkit` 0.2.0, none editable, and `git_dirty` `false`. Every manifest and every
     status carries a `host`, whose `id` and `boot` are not `null`.
  3. **The hard-kill route**, each step recorded (§4.1):
     - `hardkill-a` dies of signal 9 in the QSI stage with its status still `running`;
     - `list` marks it stale, `!!`;
     - the resume is refused at `begin`. The refusal names `hardkill-a` and `end`, no run directory
       is created, and the store's files are byte-identical before and after;
     - `end` exits 0, verified, with evidence `host` `this`, `boot` `this` and `pid_answers`
       `false`;
     - `hardkill-b` is accepted and ends `done`, its fingerprint recorded with
       `fingerprint_sidecar` `written`;
     - the hard-kill store is `39809dca…`, equal to `reference`, and passes the read-only integrity
       checks.
  4. **No Ray process of this repository outlives `hardkill-a`**, waited for by pid. If one does,
     03 stops and asks (§6.1 U3).
  5. **L4** against R2, or each difference recorded, explained and opened.
  6. `list` shows §0.2's 45 runs and R3's six (the four launches, `hardkill-a` and `hardkill-b`;
     the refused resume makes none), 51 in all, with nothing `running`.
  7. `[a3-v2-a-hard-killed-run-blocks-its-store]` closes on the `handover` board's §4, with
     `docs/OPEN_ISSUES.md`, in 03's commit.
- **Stop.** A fingerprint miss on the four-launch stores is the pin's (§5 rule 11). A miss on the
  hard-kill store alone is recorded with `compare_fingerprints` and stops the prompt; the cause is
  the hard kill, and is SGK's or DatastoreKit's to find.

**04 — Close the campaign** (Sonnet). Written after 03's review.
- **What it does.**
  - The final measurements at `HEAD`: the roots, `pip freeze` against 01's, and `list`.
  - Drafts the G1 record for RunRegistryKit's `interrupted-runs` board, in that board's form. The
    commit there is the user's (§6.3).
  - Corrects `docs/OPEN_ISSUES.md`'s notes on the registry issues transferred to RunRegistryKit
    that `v0.2.0` closed there: `[04-runs-do-not-say-which-machine-produced-them]` and
    `[04-a-signal-inside-a-heartbeat-write-…]`.
  - States what A3 v2 inherits: the launcher unchanged; the route after a hard kill, as R3 ran it;
    what R3 found of Ray's processes after a `SIGKILL`; and RunRegistryKit's open issues that bear
    on a long run (`[04-a-killed-process-not-yet-reaped-answers-kill-0]`,
    `[03-a-run-whose-manifest-cannot-be-read-names-no-store]`).
  - Closes the board, README §2, `prompts/INDEX.md` and `docs/OPEN_ISSUES.md` §1.22.

  *Amended 2026-10-19, at its writing after 03's review.* R3 found that a `SIGKILL`ed driver's GCS
  server can outlive it, stuck in its own shutdown (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`).
  By the user's ruling **U4** (§6.1), 04 also adds that step to `CLAUDE.md` rule 5: after `end`,
  check `ray-processes`; a GCS server still listed ten minutes after the kill is ended by a person
  with `SIGKILL` by its pid, never by pattern or `ray stop`. The issue narrows to its cause and any
  fix, which are Ray's or the launcher's, with no owner. The transferred-issue corrections reach
  the two SGK boards' §4 entries as one dated line each, so that index and boards agree.

## 3. Datastores

**The campaign builds three throwaway stores and touches no other**, all in 03:
`var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`, each with a sidecar and
a written fingerprint.

**No prompt retires a store.** R3's three are the user's to retire after 04's review, or not
(§6.3).

**Before touching anything,** a prompt runs `list` and reports anything `running` or stale. Before
each launch, it checks that no Ray process of this repository is up, with
`tools.rehearsal_lock ray-processes`, and waits by pid for a killed build's agents
(`[07-a-killed-builds-ray-agents-outlive-the-driver]`).

## 4. The interfaces between prompts

- **01 ships** the pin and `measure/out/01-*`: `01-pip.txt`, `01-list.txt`, `01-store-show.txt`
  and `01-roots.txt`. 03 and 04 compare with them.
- **02 ships** `CLAUDE.md`'s rules, `interrupt_when.py --signal`, the dummy test's `KILL` case, and
  the driver test. 03 runs the interrupter with `--signal KILL` for `hardkill-a` alone.
- **03 ships** `docs/rrk-v020-adoption-verification.md` (created), with §1 for R3 and §2 for the
  hard-kill route, and saved outputs under `docs/rrk-v020-adoption/rehearsal/` named with `03`. It
  runs `docs/dk-and-rpk-migration/rehearsal/` as it stands. 04 adds §3, the close.

### 4.1 The rehearsal protocol

**`runregistry-extraction`'s protocol, unchanged but for names** (its README §4.1, as its prompt 08
ran it; `docs/runregistry-extraction-verification.md` §3 records it). The lock is taken with
`tools/rehearsal_lock.py`, and every driver gets `address="local"`; **`ray stop` is never run**. The
names:
- stores `var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`;
- `--campaign rrk-v020-adoption --prompt 03 --job-name rrk-v020-adoption-03`;
- launch logs `var/runs/rrk-v020-adoption-03-<slug>.launch.log`;
- reader outputs under `var/rrk-v020-adoption-03/extract/`;
- the lock's purpose `rrk-v020-adoption 03 (R3)`.

**The four launches**, unchanged:
- `reference`, straight through;
- `resumed-a`, killed by `SIGTERM` in `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`;
- `resumed-b`, with `--resume`, killed by `SIGTERM` in `CALCULATE QUADRATIC SOURCE INTEGRALS`;
- `resumed-c`, with `--resume`, to `done`.

L1–L4 as there:
- every launch ends as recorded (L1);
- the resume equals the reference, `39809dca…`, and passes the read-only integrity checks (L2);
- every reader runs read-only on the closed `resumed` store (L3);
- the counts equal R2's (L4).

A difference in L4 alone is a finding to explain; the fingerprint is the acceptance.

**The hard-kill pair**, after `resumed-c` has ended and its last Ray process is gone, under the same
lock:
1. **`hardkill-a`.** A fresh build on the `hardkill` store, `--register hardkill-a`, launched
   detached as every launch is. The interrupter runs with `--signal KILL`, at `resumed-b`'s point in
   the QSI stage. The detached wrapper `sh -c '…; echo "** driver exit status $?"'` is the driver's
   parent and reaps it, so its launch log must end with exit status 137.
2. **The wait.** Wait for every Ray process of this repository to go, by pid (`ray-processes`),
   never by pattern. The driver's handler did not run, so Ray's own cleanup is all that ends them.
   **If any is alive ten minutes after the driver's death, stop and ask the user.** Never signal it
   yourself, and never run `ray stop` (§6.1 U3).
3. **The evidence before `end`.**
   - `status.json` says `running`, its pid does not answer `kill -0`, and `list` shows the run
     `stale` with `!!`.
   - The shard files' SHA-256, size and mtime are recorded.
4. **The refused resume.** The same build with `--resume --register hardkill-b`, launched detached.
   It must exit non-zero before Ray starts, with `begin`'s refusal in its launch log, naming
   `hardkill-a`, `stale` and `python -m runregistrykit end`, and ending
   `No run directory was created`.
   - No run directory is added under `var/runs/`, counted before and after.
   - The store's files are byte-identical. The `--resume` pre-check reads the primary `mode=ro`.
   - **If `begin` accepts**, the run is a live build under the lock. Record it, leave it to run,
     end your turn and report: that is a stop.
5. **`end`.** `./venv/bin/python -m runregistrykit end <hardkill-a's id> --reason "<text>"`. It
   must exit 0 and print that the run is now `lost`, verified. Its `ended` record is saved. A
   second `end` on the same run must be refused, since the run has already ended.
6. **The sidecar.** `store create` on the closed `hardkill` primary, as R2's sidecars were made
   after a build.
7. **`hardkill-b`.** The resume, `--resume --register hardkill-b`, to `done`. It prunes what
   `hardkill-a` left unvalidated at open; its prune lines are recorded as `resumed-b`'s are. Its
   `finish` records a fingerprint with `fingerprint_sidecar` `written`.
8. **The checks.** The `hardkill` store against `reference`: `compare_with_recorded.py`,
   `store_soundness.py`, `prune_check.py` and the census, as on `resumed`.

**After each launch, one check, then end the turn** (`CLAUDE.md`, run registry rules 3–4). The
orchestrator waits once per launch, on the pid's death.

### 4.2 Rollback

Each prompt is one commit, reverted by `git revert`.
- **Reverting 01** also needs the venv put back:
  `./venv/bin/pip install "runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0"`.
  A run directory written under `v0.2.0` stays readable by `v0.1.0`, with the losses the release
  notes' §4 table gives.
- **Reverting 02** needs nothing in the venv.
- **Reverting 03** removes records only. Its stores and run directories stay under `var/`, which is
  not in `git`.

The campaign runs in the main checkout. A worktree symlinks the main checkout's venv, so a pin
installed from one lands in the other.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit,** and `docs/OPEN_ISSUES.md`
   whenever a §3 or §4 entry moves, in the same commit.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue here.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. No
   prompt here names one.
8. **Deliberate breakage.** Each prompt that adds code or a test names mutations its tests must
   catch. The log records each as a diff, exactly as applied. Mutations are never committed.
9. **Nothing the code stores changes.** No table, column, key or stored value, and the
   fingerprint's overall digest stays `39809dca…`. The registry's own records gain fields by the
   pin (`host`; `ended` on a run that `end` records), as the release notes' §2 says. A prompt that
   cannot keep the digest stops.
10. **No test needs Ray, and no test opens anything under `var/`.** 01's and 03's reads of the
    registry and the sidecars are verification steps, not tests.
11. **A defect in a package is not patched or worked around here.** The prompt stops and records it
    on this board and in `docs/OPEN_ISSUES.md`. The user reports it to that package's repository,
    where it is fixed in a release, and a pin moves SGK to it.
12. **No change is made in RunRegistryKit, DatastoreKit or RayPoolKit** by a prompt of this
    campaign. Their repositories are read.
13. **The pin is a tag**, in a commit of its own, never an editable install.
14. **Long-running jobs** follow `CLAUDE.md`'s run-registry rules: discovery first, detached launch,
    one check, end the turn.
15. **`Run.finish` is never called on a run that is not the caller's.** A run whose end was never
    recorded is ended with `python -m runregistrykit end`, by the prompt that is told to (03).

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**, with every count against §0.2's or the previous prompt's;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-18; U4 2026-10-19), and not reopened here

The plan was proposed in conversation and accepted as proposed ("It sounds fine").

| | Decided | Not taken |
|---|---|---|
| **U1** — the shape | **Four prompts**: pin, adopt, rehearse, close | A pin with no rehearsal, resting on RunRegistryKit's suite alone |
| **U2** — the hard kill | **A pair of launches on a third store**, after the four-launch protocol, which stays unchanged (§0.1) | A `SIGKILL` in place of `resumed-b`'s `SIGTERM` |
| **U3** — Ray after a `SIGKILL` | **Stop and ask** if a Ray process of this repository outlives the killed driver; never signal it, never `ray stop` | Ending the processes by pid in the prompt |
| **U4** — the GCS step (2026-10-19, after 03's review) | **`CLAUDE.md` rule 5 gains it** (04): after `end`, check `ray-processes`; a GCS server still listed is ended by a person with `SIGKILL` by its pid | Recording it in the verification document alone |

### 6.2 Choices of method made in writing this plan

The orchestrator's, recorded so that the user can reverse any of them.
- **C1 — the branch.** `handover-remedial`, where A3 v2 runs, in the main checkout (§4.2).
- **C2 — the name.** `rrk-v020-adoption`. It has no dots, because the name is the prefix of every
  run id and store this campaign makes.
- **C3 — the drivers are not changed.** *This departs from the plan as accepted*, which had both
  drivers print `begin`'s refusal as a clean error instead of a traceback. Reading the drivers
  while writing 02 showed three things.
  - Each of their own refusals is a raised `RuntimeError` with its traceback, for example
    `run_build`'s "Refusing to build into an existing store" (`quadsource_atol_sweep.py:536-542`).
  - `begin`'s refusal arrives in the same form, with the same last line.
  - The launch is detached in either case, so the launch log is where it is read.

  Catching only `begin`'s would make one refusal unlike the others and change nothing about the
  outcome. 02 therefore tests the refusal at `register` and changes neither driver. **If the user
  wants the clean error, it is one charter line in 02, for all of the drivers' refusals or for
  `begin`'s alone.**
- **C4 — stale prose.** A statement that a prompt's own commit makes false is corrected in that
  commit: `CLAUDE.md`'s pin mentions (01) and its registry rules (02). Dated audits, verification
  records and `prompts/` are left as written.
- **C5 — the driver test's seam.** `scoped_pipeline_run.register`, which already has `runs_root`
  (`runregistry-extraction` C12) and a test helper that restores what it takes over. `run_build` has
  no seam and imports `ray` first (`:518`), so its refusal is shown by R3 on the real launcher, not
  by a test. 02 adds no seam.
- **C6 — the issue assigned here.** `[a3-v2-a-hard-killed-run-blocks-its-store]` (`handover` board
  §3) is a `v0.1.0` defect that RunRegistryKit closed in `v0.2.0` (its `interrupted-runs` 04 and
  05). It reaches SGK by this pin. It is assigned to this campaign in the planning commit, and 03
  closes it once the route has been shown on the real launcher. The pin alone does not close it.
- **C7 — R3's comparison.** R2's recorded fingerprints carry `encoding_format` 1, so R3 compares
  through the package's `compare_with_recorded.py` and needs no field-by-field script, unlike R2,
  whose comparands lacked the field.
- **C8 — the hard kill's sidecar.** `store create` on the `hardkill` primary comes after `end`, so
  that no store command runs while the dead run still names the store as running. What
  `store create` does on a store named by a stale run is not what this campaign measures.

### 6.3 Left for the user

- **At the close:** retiring R3's three stores, or not; the G1 record on RunRegistryKit's
  `interrupted-runs` board, which 04 drafts and which is a commit in that repository; and whether
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` is assigned to a campaign.
- **C3** above, if the clean error is wanted after all.

## 7. What this campaign needs from RunRegistryKit

Nothing more: the tag, its release notes and the adoption note exist. G1 on RunRegistryKit's
board holds when this repository has moved its pin in a commit of its own (01). 04 drafts the
record. If a prompt finds a defect in the package, §5 rule 11 applies, and a brief for
RunRegistryKit is written in this folder.

## 8. Baselines

Re-measure every root at `HEAD`, clean, before every dispatch:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

**Last counted at `79434bf`** (§0.2): AdaptiveLevin 32, ComputeTargets 700, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), ObjectFactories 279, RunDrivers 99, tools 91.

**Under load, four tests are known to fail on wall clock or chance:**
- `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
- `test_background_tau.test_build_cost`;
- `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
- LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

Re-run a failure alone before calling it real.
