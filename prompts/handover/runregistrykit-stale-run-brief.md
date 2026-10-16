# Brief — a hard-killed run blocks its store, and nothing in the registry ends it

**From:** SecondaryGWKit (SGK), campaign `handover` (2026-10-16, written at SGK `f1a7de3`; its
commit is named on SGK's board) · **For:** a session in RunRegistryKit
(`/Users/ds283/Documents/Code/RunRegistryKit`, `https://github.com/ds283/RunRegistryKit`), handed
over by the user · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) §3,
`[a3-v2-a-hard-killed-run-blocks-its-store]`

This is evidence for RunRegistryKit to check. It is not a specification of what RunRegistryKit
should do. Where RunRegistryKit's own records disagree with a line here, they are right, and the
disagreement comes back to SGK.

## 0. What this is, and what it is not

SGK is about to launch A3 v2. That is a ~36-hour Ray build of one LambdaCDM store, registered
through `runregistrykit` 0.1.0, and it may be interrupted and resumed. SGK has rehearsed the
interruption by `SIGTERM` end to end. The driver's handler ends the run `killed`/143. A resume
recomputes only what is missing, and the final fingerprint equals the reference. SGK has **not**
rehearsed a hard kill. Reading the package suggested a gap, and SGK measured it on a fixture store.
This brief reports it.

- **One entry follows (§1), complete.** It was read against RunRegistryKit at the tag `v0.1.0`
  (commit `9f8ebe4`). The installed `runregistrykit/__init__.py`, `stores.py` and `__main__.py` in
  SGK's venv are byte-identical to `git show v0.1.0:runregistrykit/<file>`, and `git diff --stat
  v0.1.0 79015a7 -- runregistrykit` is empty (read 2026-10-16). So the line numbers are the tag's,
  and also `HEAD`'s as of `79015a7`.
- **It is not in RunRegistryKit's records.** A search of `docs/OPEN_ISSUES.md` and both boards for
  "stale" finds only the inherited `[04-a-signal-inside-a-heartbeat-write-…]` (§1, "Related"),
  which reaches the same state by another route.
- **Much of the refusal is deliberate, and that design is not in question.** "The registry does not
  decide that a run is dead" (`stores.py:1031-1032`) is the right rule, and SGK does not ask to
  weaken it. The gap is narrower. The remedy the refusal names, ending the run with `Run.finish`,
  has no command line and checks nothing.
- **Nothing is transferred yet.** SGK's entry stays open on SGK's board until RunRegistryKit's
  index lists it. Then SGK moves it to its board's §4 as transferred. **Please tell SGK when that
  happens.**
- **Where it goes in RunRegistryKit is RunRegistryKit's decision.** The `inherited-issues` board is
  one candidate, though this issue is not inherited: SGK found it in the package. **It is assigned
  to nothing**, because SGK assigns nothing to a RunRegistryKit campaign.

---

## 1. `[a3-v2-a-hard-killed-run-blocks-its-store]`

*SGK `handover` board, §3. Opened 2026-10-16, ahead of the A3 v2 launch, by a measurement and not
by a prompt.*

- **The defect.** A run killed by `SIGKILL`, power loss, a reboot or the OOM killer never reaches
  `Run.finish`, so its `status.json` says `state: running` for ever. `list` reports it as stale,
  which is correct. Every store operation that checks for a running run counts it "alive **or
  stale**", so the store's fingerprint, copy, move, retire and amend are refused. The refusal is
  permanent unless someone ends the run, and **no command ends a run**.
- **The package's lines at `v0.1.0`.**
  - *The in-use check.* `_running_runs_naming` (`runregistrykit/stores.py:998`) keeps every entry
    with `entry["state"] == "running"` other than the excluded caller (`:1017`), whatever its
    `liveness`. Its callers are copy and move (`_refuse_if_in_use`, `:1029`, called at `:1102`),
    `fingerprint_store` (`:1338`), `retire_store` (`:1664`) and `amend_sidecar` (`:2017`). The
    module docstring says so (`:102-107`).
  - *The remedy the refusals name.* "a stale run is ended by a person, with Run.finish" appears at
    `:1038` (copy, move), `:1344` (fingerprint), `:1667` (retire) and `:2020` (amend).
    `tests/test_store_retire.py:436` (`test_a_stale_running_run_and_its_remedy`) asserts that
    message, using `a_stale_run` (`tests/store_fixtures.py:148`). That fixture is a run given a
    stand-in dead pid; it is not a killed process.
  - *No command to apply the remedy.* `__main__.py:408` `main` has `list` (`:418`) and `store
    show|create|adopt|copy|move|fingerprint|retire|amend` (`:433-522`). None of them writes a
    `status.json`.
  - *The remedy checks nothing.* `Run.finish` (`__init__.py:424-459`) checks only that `state` is
    in `TERMINAL_STATES` (`:99`, `:441`). It does not consult `pid_alive` (`:258`), the heartbeat
    or `liveness` (`:294`). A person who ends the wrong directory ends a live run.
  - *The resumed run's fingerprint.* `finish(..., fingerprint=True)` passes `taken_by=self`
    (`:476`), and that is the only exclusion (`stores.py:1338-1339`). A later run on the same store
    is therefore refused by its stale predecessor. It records `fingerprint_error` and writes no
    sidecar fingerprint.
  - *`begin` does not look.* `begin` (`__init__.py:604`) reads the results store's sidecar only to
    refuse a tombstone (`:686-698`). It does not ask whether a `running` run, alive or stale,
    already names that store. A second writer on a live run's store is not refused. The module
    docstring's "it is a check and not a lock" (`stores.py:106`) describes the store operations;
    `begin` does not make the check at all.
- **Measurement.** Taken 2026-10-16 at SGK `f1a7de3` (clean) with `runregistrykit` 0.1.0, no Ray.
  The store was a throwaway two-shard store built by SGK's `StoreFixtures.real_store_fixtures`, at
  `var/datastores/handover-stale-run-probe.sqlite` (`store_id` `dab353cdb5364ad3928f4f684a7ce4e6`).
  The runs went in SGK's `var/runs/`.
  1. `store create`, then `store fingerprint --write`: digest `d932853819d0…`, "wrote the
     fingerprint into the sidecar".
  2. A child process called `runregistrykit.begin(…, results=<store>)` and `heartbeat(stage=…)`,
     then slept. The parent sent it `SIGKILL`; returncode −9. `status.json` was identical before
     and after the kill: `state: running`, `pid 88637`, `exit_code: null`. `liveness(…, 900)` gave
     `stale`.
  3. `list`:
     ```
     !! handover-stale-run-probe-killed-20261016T110844   running   -   1s  Stale-run probe: …
     44 run(s) under …/var/runs: 41 finished, 1 stale, 2 unknown.
     !! 1 run(s) say they are running and are not: the pid does not answer kill -0, or the heartbeat is older than 900 s. Nothing here has killed or cleaned anything — go and look.
     ```
  4. A second `begin` on the same store returned normally, with `results_store_id`
     `dab353cd…`. Its `finish("done", exit_code=0, fingerprint=True)` wrote `state: done`,
     `exit_code: 0`, no `fingerprint` and no `fingerprint_sidecar`, and:
     ```
     fingerprint_error: RuntimeError: Cannot fingerprint ".../handover-stale-run-probe.sqlite": run handover-stale-run-probe-killed-20261016T110844 is running (stale, pid 88637) and names this store by its results and results_store_id. A fingerprint of a store that is being written describes no instant; a stale run is ended by a person, with Run.finish, first. Nothing was read or written
     ```
  5. `store fingerprint <store> --write`, `store retire <store> --reason … --dry-run`, the same
     with `--without-fingerprint`, and `store copy <store> <new> --purpose …` all exited 1. Each
     gave `!! Cannot <operation> …: run handover-stale-run-probe-killed-20261016T110844 is running
     (stale, pid 88637) and names this store by its results and results_store_id. A stale run is
     ended by a person, with Run.finish, …`. The sidecar's SHA-256 was unchanged by all four.
  6. Then the remedy. `kill -0 88637` gave "no such process". A person ran
     `runregistrykit.Run("var/runs/handover-stale-run-probe-killed-20261016T110844").finish("killed",
     exit_code=137)`. After that, `list` showed it `killed`, with no `!!` ("45 run(s) … 43
     finished, 2 unknown"). `store fingerprint` matched the recorded `d932853819d0…`, `store retire
     --dry-run` passed, and `store retire` deleted the primary and both shards. The sidecar is the
     tombstone (`completed 2026-10-16T11:11:25+01:00`).
- **Impact.**
  - *For SGK's A3 v2.* After a hard kill the store **can be resumed**. Neither `begin` nor SGK's
    `--resume` pre-check refuses, and the resumed run ends `done`. But the store **cannot be
    fingerprinted, copied, moved, retired or amended through any registry command**. The resumed
    run's own fingerprint is recorded only as `fingerprint_error`, and a new store's sidecar keeps
    no fingerprint. The one route out is a person writing Python against the package's API by
    hand, on a directory they chose by reading `list`, with no check that the process is gone. No
    rows are lost. The cost is that the documented way to end a run is unguarded and has no
    command line, so the person must run it by hand on the build that matters most.
  - *For any client.* Discovery rule 1 says to "report anything `running` or stale". The package
    reports it and then offers no recorded way to resolve it. A stale entry stays in `list` with
    `!!` until someone edits it through the API. The refusals then block every store operation on
    that store.
  - *Related.* `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` (the
    `inherited-issues` board) can leave a `SIGTERM`'d run reading `running` by a race (its
    "terminal state is lost silently" case). That run meets the same block, and the remedy below
    covers it too.
- **Next step: options. Choosing among them is RunRegistryKit's decision.** The first is SGK's
  preference, because it keeps "the registry does not decide that a run is dead": a person decides,
  and the package checks the evidence and records the decision.
  1. **A person-run command that records a stale run as terminal.** For example, `python -m
     runregistrykit end RUN_ID --reason TEXT`. It would refuse unless the run's state is `running`
     and its pid fails `kill -0`, which is `pid_alive` and is this machine's process table. A run
     that is stale only by an old heartbeat, with a live pid, is refused. It would write a terminal
     state, an exit code of `null` (nothing observed one), and a record of who ended the run, when,
     why and at what `git_head`. It would never write the manifest and never delete anything.
     Which state to write is the open design point. `killed` is in `TERMINAL_STATES` already, but
     `finish`'s docstring says `killed` is "something a caller reports, not something observed"
     (`__init__.py:425-426`). A new `lost` state is the honest word for "the registry never saw
     how it ended", at the cost of a fourth terminal state that `liveness` and `list` must learn.
     The same check could guard `Run.finish` when it is called on a run that is not the caller's
     own.
  2. **An explicit exclusion on the store operations.** For example, `--stale-run RUN_ID`, repeated
     for each run to ignore. It would be accepted only for a named run whose pid fails `kill -0`,
     and recorded in the sidecar's `history` or the tombstone, so a later reader sees that a stale
     run was overridden. This unblocks the store without ending the run, so `list` keeps its `!!`.
     It is weaker than option 1, because the dead run stays `running` for ever, but it changes
     nothing in `status.json`.
  3. **Teach `finish(fingerprint=True)` and `begin` about predecessors.** `begin` would name any
     `running` run on the same results store. It would refuse if that run is alive, and if it is
     stale, it would say so and record it (for example `supersedes: [RUN_ID]`). `finish` would then
     exclude runs this run superseded. That covers the resume case automatically, but it decides
     at `begin` that a stale run is dead, which is the line the package has chosen not to cross.
     Listed for completeness. The part that refuses a new run on a store an **alive** run names
     stands on its own as a second writer guard, and RunRegistryKit may want it regardless.

  A test for any of these: begin a run in a child process and `SIGKILL` it, then assert what the
  new route does. The existing `a_stale_run` fixture fakes the pid, so it never exercises a real
  kill.

**Proposed `docs/OPEN_ISSUES.md` row.**

| `[a3-v2-a-hard-killed-run-blocks-its-store]` | none | A `SIGKILL`'d run stays `running`, and every store operation counts a stale `running` run as in use (`stores.py:998`, `:1017`), so fingerprint, copy, move, retire, amend and a resumed run's `finish(fingerprint=True)` are refused. The named remedy, `Run.finish`, has no command and checks no pid. `begin` does not check for a run on its store |

---

## 2. What SGK asks for, and what it does not

- **Asked:** that RunRegistryKit read the entry against its own code, list it in its
  `docs/OPEN_ISSUES.md` if it accepts it, and tell SGK, so that SGK moves it to its board's §4 as
  transferred. If RunRegistryKit declines it, it should say so, and SGK records the decision on its
  board.
- **Not asked:** a release before A3 v2. A hard kill is not expected, and the launch does not wait
  for this. If one happens, SGK's user decides whether to use the package's documented route (§1,
  measurement step 6) or wait for a release.
- **No change to SGK follows from handing it over.** SGK patches nothing and works around nothing
  in its own code. A fix reaches SGK by a pin move, in a commit of its own, as every package change
  does.
- **SGK's conventions bind this hand-over.** No SGK session changes anything in RunRegistryKit,
  DatastoreKit or RayPoolKit. RunRegistryKit's records are RunRegistryKit's to write.
