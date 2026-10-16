# Brief — five issues SecondaryGWKit hands to RunRegistryKit

**From:** SecondaryGWKit (SGK), campaign `runregistry-extraction`, prompt 09 (2026-10-16, written at
SGK `ef34ed0`; its commit is named on SGK's board, §5.15) · **For:** a session in RunRegistryKit
(`/Users/ds283/Documents/Code/RunRegistryKit`, `https://github.com/ds283/RunRegistryKit`), handed
over by the user · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) §3 and §5.15

This is evidence for RunRegistryKit to check, not a specification of what it does. Where RunRegistryKit's
own records disagree with a line here, they are right, and the disagreement comes back to SGK.

## 0. What this is, and what it is not

SGK adopted `runregistrykit` at `v0.1.0` in its `runregistry-extraction` campaign (05–08). Before
that the registry was SGK's own package, `RunRegistry/`, and it carried open issues on SGK's boards.
06 deleted `RunRegistry/`; the code is RunRegistryKit's from `v0.1.0`. By SGK's `CLAUDE.md` a defect
in a package is **fixed in that package, in a release**, and reaches SGK by moving the pin; it is
never patched or worked around in SGK. So SGK hands the issues that are in the package to the
package.

- **Five entries follow (§2), each complete.** Four are issues SGK recorded against `RunRegistry/`.
  The fifth, the closed pipe, SGK found while adopting the package (06's review).
- **Each was read against RunRegistryKit at the tag `v0.1.0`** with `git show v0.1.0:<path>` and
  `git grep … v0.1.0`; the lines below are the tag's. `git diff --stat v0.1.0 caad864 --
  runregistrykit pyproject.toml` is empty (read 2026-10-16), so they are also `HEAD`'s as of
  `caad864`.
- **Nothing is transferred yet.** RunRegistryKit's `docs/OPEN_ISSUES.md` holds two issues of its own
  (`[00-the-shard-fixtures-reach-into-shardedpool]` and `[00-a-refusal-matches-datastorekit-message-text]`)
  and none of these five. SGK's entries stay open on SGK's boards, each with one line saying it was
  drafted for RunRegistryKit, until RunRegistryKit's index lists it. Then SGK moves each to its
  board's resolved section as transferred (SGK's board, §5.15). **Please tell SGK when it does.**
- **Where they go in RunRegistryKit is RunRegistryKit's to decide**: a board of their own, a
  campaign, or the `extraction` board's §3. DatastoreKit's `inherited-issues` board
  (`prompts/inherited-issues/README.md`, and `docs/OPEN_ISSUES.md` §1.2 there) is the precedent for a
  board with no prompts that holds handed-over issues in full. The form below is the
  `extraction` board's §3 entry form, shortened: SGK's name and board, when opened, the defect with the
  package's lines, the measurement, the impact and the next step.
- **None is assigned.** SGK assigns nothing to a RunRegistryKit campaign.

## 1. The two that are SGK's, not asked for

SGK also read two more issues that name registry behaviour, and found both to be its own. They are
listed so that nobody asks for them.

- **`[00-a-sigterm-pipeline-run-is-recorded-as-failed]`** (SGK `store-retirement` board, opened
  2026-09-25). A pipeline run stopped with `SIGTERM` while blocked in `ray.wait` is recorded
  `failed`, exit 1, not `killed`. The code that maps an exit to a state, installs the signal handler
  and ends the `exec` is `terminal_state`, `install_killed_handler` and `finish_after_exec` in SGK's
  `docs/gktk-remedial/scoped_pipeline_run.py` (`:256`, `:412`, `:429` at SGK `ef34ed0`).
  `runregistrykit` at `v0.1.0` defines none of the three: `git grep -nP "def
  (terminal_state|finish_after_exec|install_killed_handler)\(" v0.1.0` is empty, and `git grep -n
  'SIGTERM\|signal\.' v0.1.0 -- runregistrykit` finds nothing outside the tests. The package's
  `Run.finish(state, exit_code)` records what its caller reports and says so ("`killed` is something
  a caller reports, not something observed", `runregistrykit/__init__.py:424-`).
- **`[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]`** (SGK `a3-v2-readiness`
  board, opened 2026-10-03). The window between `ray.init` installing its `SIGTERM` handler and the
  driver's replacing it. Both ends are in SGK's driver (`install_killed_handler`, `:412`, installed
  after `ray.init` returns) and in Ray. The package is not involved.

Both stay on SGK's boards, untouched.

---

## 2. The five entries

### 2.1 `[04-runs-do-not-say-which-machine-produced-them]`

*SGK `run-registry` board, §3 (`prompts/run-registry/IMPLEMENTATION_STATE.md:61` at SGK `ef34ed0`).
Opened 2026-09-24 from a cross-machine question, not by a prompt. **A decided change, not a defect**
(user decision of 2026-09-24: "do it"). Not built.*

- **What is wanted, and why it is SGK's decision.** SGK decided, for SGK's runs, to add a `machine`
  field to the manifest, to make `list` refuse to interpret liveness for a record whose machine is
  not this one, and to transfer only terminal-state run directories between machines. Whether
  RunRegistryKit adopts it **for every client** is RunRegistryKit's own question: ChamPBH and
  StochasticInstantons may or may not run one registry across machines. SGK asks for the decision, not
  for an implementation it has already decided for the package.
- **The package's lines at `v0.1.0`.** The manifest is built in `begin` (`runregistrykit/__init__.py:604`);
  its fields are `run_id`, `created`, `purpose`, `campaign`, `prompt`, `argv`, `cwd`, `script`,
  `script_sha256`, `unit`, `expected_units`, `checkpoint`, `results`, `results_store_id`, `scope` and
  `heartbeat_means` (`:707-723`: `argv` at `:712`, `cwd` at `:713`), then `git_provenance`'s `git_head`
  and `git_dirty` (`:152`, `:724`) and `packages` (`:725`). **No `machine`, no `platform` and no
  hostname is recorded.** Liveness is `pid_alive` (`:258`, `os.kill(pid, 0)`, this machine's table) and
  `liveness` (`:294`), whose arguments are the status payload, the staleness window and `now`; it takes
  no machine.
- **Measurement.** `git grep -n -i 'machine\|platform\|node()' v0.1.0 -- runregistrykit` finds only
  an unrelated comment about path separators (`stores.py:240`) and a comment in a test
  (`tests/test_run_registry.py:169`). Read 2026-10-16.
- **Impact.** Two, and the second is worse.
  - *Provenance.* A second machine (`macstudio-tunnel`, an M1 Max) is to run SGK's A3 rebuild, and rows
    computed on different hardware can differ in their last bits. For a store whose purpose is to be a
    comparator, mixed provenance is a defect nothing records: `git_head` says what code ran, not what
    ran it.
  - *Liveness.* Once run directories are rsync'd between machines, `list` evaluates `kill -0 <pid>`
    against this machine's process table for a pid that belonged to another. A terminal-state record is
    harmless. A record still marked `running` gives a confident wrong answer: a false "dead", or, if
    the pid has been reused locally, a false "alive". That is `CLAUDE.md` run-registry rule 2's failure
    from the opposite direction.
- **Next step.** Decide whether the manifest records the machine for every client. If so: add `machine`
  (`platform.node()` and `platform.machine()`), make `list` report a record from another machine as
  *foreign* rather than guess at its liveness (reporting correctly is recording, so this stays within
  the registry's "it records; it does not supervise" boundary), and state the rule that only
  terminal-state directories are transferred. A manifest written before the field exists would have
  none: say what `list` does for it. None of SGK's 43 run directories is `running`, so SGK has nothing live to migrate (read 2026-10-16, `list`).

**Proposed `docs/OPEN_ISSUES.md` row.**

| `[04-runs-do-not-say-which-machine-produced-them]` | none | A manifest records `git_head`, `script_sha256`, `argv` and `cwd` but not the machine, so `list` evaluates `kill -0` for a copied record against the wrong process table. SGK's decided change; RunRegistryKit decides it for every client |

### 2.2 `[00-a-copy-carries-its-sources-present-tense-fields]`

*SGK `store-retirement` board, §3 (`prompts/store-retirement/IMPLEMENTATION_STATE.md:508`). Opened
2026-09-25 by SGK's audit of store retirement (`docs/store-retirement-audit.md` §2.12).*

- **The defect.** `copy_store` replaces seven fields of the source's sidecar (`store_id`, `datastore`,
  `name`, `purpose`, `created`, `copied_from`, `history`) and carries every other field **verbatim**,
  by design. Some of the carried fields make claims about the store they sit in. A copy of SGK's live
  A3 store carries three:
  - `backup`, whose `"retained": true` describes a backup *of the source*;
  - `restart.command`, whose `--database` is the **source's** path;
  - `run_history`, the source's runs.
- **The package's lines at `v0.1.0`.** `copy_store` is `runregistrykit/stores.py:1138`. Its docstring
  says: "Every other field, unknown ones included, is the source's, verbatim." The copy is
  `fields = _copy.deepcopy(reading.fields)` at `:1160`, followed by the seven assignments
  (`:1161-1172`: `store_id`, `datastore`, `name`, `purpose`, `created`, `copied_from`, then `history`),
  and `_write_new_sidecar` (`:1173`). `move_store` (`:1179`) keeps the sidecar's fields, which is right
  for a move (the store is the same store).
- **Measurement.** SGK's `store-fingerprint` log 05, Phase C, copied the live A3 store's sidecar and
  found all three fields in the copy (read in the audit's §2.12). Still present: the lines above.
- **Impact.** Nothing reads these fields, so no code misbehaves. A person who followed a copy's
  `restart` would resume the pipeline **into the source store**, which is the 2026-09-23 accident
  (SGK `2ebb7b6`, 54 rows written into the live A3 store) by another route.
- **Next step.** A decision about `copy_store`'s design: carry, drop, or flag such fields. "Unknown
  fields are carried" is deliberate, so the question is whether a field that names a path or a
  history is "unknown" in the sense that rule means. `store amend` is the remedy for a given copy
  (`python -m runregistrykit store amend`), so a first step that changes nothing in the code is a
  line in `copy`'s output naming the fields a copy carries that mention the source.

**Proposed row.**

| `[00-a-copy-carries-its-sources-present-tense-fields]` | none | `copy_store` carries unknown sidecar fields verbatim (`stores.py:1160`), so a copy can say it has a retained backup and a `restart.command` that resumes into the source store. Nothing reads them; a decision about copy's design |

### 2.3 `[01-cross-filesystem-move-advice-says-delete-by-hand]` — the relaying half

*SGK `store-retirement` board, §3 (`prompts/store-retirement/IMPLEMENTATION_STATE.md:525`). Opened
2026-09-25 by SGK's prompt 01; **narrowed 2026-10-12** by SGK's `dk-and-rpk-migration` 09.*

**Two halves; only the second is RunRegistryKit's.**

- **DatastoreKit's half, not asked for here.** `ShardedPool._failure_message` appends, on `EXDEV`,
  "copy it instead, and then delete the source by hand". In the installed `datastorekit` 0.4.0 that is
  `datastorekit/SQL/ShardedPool.py:3029` (the string), and `move_store`'s docstring says the same at
  `:2684`; the package has `ShardedPool.delete_store` (`:2704`). It is indexed in DatastoreKit's
  `docs/OPEN_ISSUES.md` §1.2, as `[01-cross-filesystem-move-advice-says-delete-by-hand]`, so it is
  DatastoreKit's to fix.
- **The half that stayed SGK's, and is now RunRegistryKit's: `store move` relays it for a registered
  store.**
  - *The package's lines at `v0.1.0`.* `move_store` (`runregistrykit/stores.py:1179`) calls
    `_store_files("move", ShardedPool.move_store, src, dst)` at `:1195`. `_store_files`
    (`:1119-1129`) catches any exception from the pool and re-raises `RuntimeError(f"{e}. No sidecar
    was written or moved: sidecar files at the destination [...]; at the source [...]")`
    (`:1125-1128`). The package's message reaches the user verbatim, with the sidecar listing
    appended. RunRegistryKit adds and shapes no advice.
  - *The defect.* A registered store's source is removed with `store retire`, never by hand (SGK's
    decision 6.1.1; RunRegistryKit's `retire` leaves the sidecar behind as a tombstone). Deleting the
    source by hand leaves a sidecar describing nothing, the broken case of SGK's audit §2.1, with no
    tombstone. The datastore package cannot know that a store is registered, so the place to say so is
    `_store_files` or `move_store`, where it is known.
- **Measurement.** Read from the code, 2026-10-16: the string at the installed `ShardedPool.py:3029`,
  and the relay at the lines above. Not run: it needs a cross-filesystem move that fails.
- **Impact.** Low, and only on a failed cross-filesystem move. Nothing acts on the advice
  automatically; a person who follows it creates the broken case. The test that asserted the text in
  SGK was deleted from SGK in `dk-and-rpk-migration` 04, so no SGK test constrains it, and none in
  RunRegistryKit asserts it (`git grep -n 'by hand' v0.1.0 -- runregistrykit` finds nothing about a deleted source).
- **Next step.** In `_store_files`, for `operation == "move"`, when the failure is the cross-filesystem
  one, say that the source is retired with `store retire` and not deleted by hand. Matching the
  package's message text for that is fragile (RunRegistryKit's own
  `[00-a-refusal-matches-datastorekit-message-text]` is the same hazard), so the cleaner route is to
  wait for DatastoreKit's reword and let this entry close with it. Either is RunRegistryKit's choice.

**Proposed row.**

| `[01-cross-filesystem-move-advice-says-delete-by-hand]` | none | `store move` relays DatastoreKit's "delete the source by hand" advice for a registered store (`stores.py:1119-1129`, `:1195`), against the rule that a registered store is retired. The advice text is DatastoreKit's own issue |

### 2.4 `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`

*SGK `a3-v2-readiness` board, §3 (`prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md:1375`). Opened
2026-10-03 by SGK's prompt 04, from the orchestrator's reading; **read, not measured**; does not block
A3 v2.*

- **The defect is the package's, though SGK's signal handler is what triggers it.** Every write of
  `status.json` goes through one fixed temporary name.
- **The package's lines at `v0.1.0`.** `write_json_atomic` is `runregistrykit/__init__.py:242-251`:
  `tmp = path + ".tmp"` (`:245`), `open(tmp, "w")`, `json.dump`, `flush`, `fsync`, then
  `os.replace(tmp, path)` (`:251`). `Run._update` (`:398-403`) calls it for `status.json`
  (`:402`), and `heartbeat` (`:405`) and `finish` (`:424`) both go through `_update`. So a heartbeat's
  write and a `finish`'s write share `status.json.tmp`.
- **The sequence.** If a driver's signal handler runs while a heartbeat is inside `with open(tmp,
  "w")`, before its `flush`, the handler's own `finish` truncates and rewrites the same file, renames
  it to `status.json`, and raises. Unwinding then closes the interrupted handle, whose buffered partial
  JSON is written at offset 0 of the same inode, now `status.json`.
- **Why it is not avoidable from SGK.** The handler that calls `finish` is SGK's
  (`install_killed_handler`, `docs/gktk-remedial/scoped_pipeline_run.py:412`), but nothing in SGK can
  avoid the shared inode: the temporary name is chosen inside the package.
- **Measurement.** Read, not measured. The window is the microseconds of one `json.dump`, once per
  heartbeat (at most every 30 s in SGK's driver). SGK has not reproduced it.
- **Impact.** SGK's `finish_after_exec` makes a second `finish`, which reads the file as `{}` (it is
  unparseable) and rewrites it valid, but without `pid`, `stage` or `units_*`. A `SIGKILL` between the
  two would leave `status.json` unparseable, and `list` would show the run as `unknown`.
- **Next step.** A unique temporary name per write in the same directory (for example `tempfile.mkstemp`
  with `dir=os.path.dirname(path)`), so that a nested write never shares an inode with the one it
  interrupts. It stays in the same directory, so `os.replace` stays atomic. A test that writes twice
  through a patched `json.dump` which writes the second time from inside the first is the
  reproduction. `write_json_atomic` is in `__all__` (`:96`), so a client may call it: say whether the
  name changes in the release notes.

**Proposed row.**

| `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` | none | `write_json_atomic` uses one fixed `<path>.tmp` (`__init__.py:245`), so a signal handler's `finish` during a heartbeat write shares its inode, and the interrupted handle's partial JSON can land in `status.json`. Read, not measured |

### 2.5 `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]`

*SGK `runregistry-extraction` board, §3 (`prompts/runregistry-extraction/IMPLEMENTATION_STATE.md`).
Opened 2026-10-15 in SGK's 06 review, by the orchestrator, at the user's request.*

- **The defect.** `python -m runregistrykit list` flushes `stdout` unguarded after printing the table.
  When the reader closes the pipe first, as `list | head -4` does, the flush raises `BrokenPipeError`:
  a traceback, then Python's "Exception ignored … BrokenPipeError" on exit.
- **The package's lines at `v0.1.0`.** `sys.stdout.flush()` at `runregistrykit/__main__.py:563`, the
  line after the summary `print` (`:562`); `main` is `:408`. Nothing in the file catches
  `BrokenPipeError`. Every other `print` in `main` is equally unguarded, so other commands (`store show`
  prints a long record) can probably do it too; only `list` was seen.
- **Measurement.** A race: 1 of 30 runs of `list | head -4`, on a 17 kB listing, at SGK `7af17bd`.
  It is inherited: SGK's frozen `RunRegistry/__main__.py:503` at `a45aefc` had the same flush, and
  RunRegistryKit imported it.
- **Impact.** Nothing is lost or written; what the reader saw is complete up to where it stopped. A
  shell with `pipefail`, or a script that treats stderr as failure, would read SGK's run-registry
  rule 1 (run `list` at the start of a session and before every launch) as failing. Cosmetic otherwise.
- **Next step.** Catch `BrokenPipeError` in `main` and point `stdout` at `os.devnull` before exiting
  (the usual remedy), for every
  command. A test: a reader that closes first, with the exit code and stderr asserted.

**Proposed row.**

| `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` | none | `list` flushes `stdout` unguarded (`__main__.py:563`), so a reader that closes the pipe first (`list \| head`) sometimes gets a `BrokenPipeError` traceback. Inherited from SGK's `RunRegistry/`; cosmetic, nothing lost |

---

## 3. What SGK asks for, and what it does not

- **Asked:** that RunRegistryKit read each entry against its own code, list the ones it accepts in its
  `docs/OPEN_ISSUES.md`, and tell SGK, so that SGK moves each entry to its board's resolved section as
  transferred. Where RunRegistryKit declines one (the `machine` field is the likeliest), say so, and
  SGK records the decision on its board.
- **Not asked:** a release. SGK's A3 v2 launch does not wait for any of these (SGK's board, §5.15).
- **No change to SGK** follows from handing them over. A fix reaches SGK by a pin move, in a commit of
  its own, as every package change does.
- **SGK's conventions bind this hand-over**: nothing in RunRegistryKit, DatastoreKit or RayPoolKit is
  changed by an SGK prompt; RunRegistryKit's records are RunRegistryKit's to write.
