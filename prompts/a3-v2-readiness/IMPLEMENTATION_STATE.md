# A3 v2 readiness campaign — implementation state

**Last updated:** 2026-10-03 · **Status: IN PROGRESS — 5 of 8 prompts landed.** Audit 00 has landed
([`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md)) and been reviewed; prompt 01
has landed (2026-10-02) and been reviewed; prompt 02 has landed (2026-10-03) and been reviewed;
prompt 03 has landed (2026-10-03) and been reviewed; prompt 04 has landed (2026-10-03). 01–06 are written: 02, 03 and 04 on 2026-10-02, against the audit and the user's decisions U1 and U2. 07
is held on 01–06 (README §2).

The campaign was opened on 2026-10-02, when the user asked what remained to fix before the A3 v2
LambdaCDM store is built, and decided that all seven defects of README §0 are fixed first, by one
campaign (D4). Four are on the `datastore-integrity` board and close there; one was assigned there
to `handover` and is reassigned here; one had no board and is taken over by this one; one is
opened here.

**Campaign:** [`README.md`](README.md) ·
**Code:** `config/defaults.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ShardedPool.py`,
`extract_*.py`, `docs/gktk-remedial/scoped_pipeline_run.py`, `docs/handover/quadsource_atol_sweep.py`,
`RunRegistry/stores.py` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.17

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-02 (README §6.1):**

| Decision | Decided |
|---|---|
| **D1** — A3 v2's pair | `(atol, rtol) = (1e-32, 1e-7)`, as recommended 2026-09-24; the science check is one rung tighter than the build's pair, so `1e-8` |
| **D2** — how the extractor gets the pair | `DEFAULT_QUADRATURE_RTOL` is set to `1e-7`; no flag |
| **D3** — how readers stop writing | a read-only pool, every file `mode=ro`, not per-site lookups |
| **D4** — the campaign | one campaign, `a3-v2-readiness`, owns all seven |

**Put to the user by audit 00 (2026-10-02), and taken by the user the same day, each as
recommended** (audit, closing section; README §6.2):

| Decision | Taken |
|---|---|
| **U1** — does 02 also put the prune at open of a replicated row through a record (`[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]`)? | **yes, in 02**: the pool prunes replicated classes under a `prune` record the check at open completes by re-running it (02's W3) |
| **U2** — what is done about `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`? | **record only**; no code change |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 00 | [Audit the readiness defects](00-audit-the-readiness-defects.md) | **R1**–**R3**, **S1**–**S4**, **V1**–**V2**, **M1**–**M2** | Opus | ✍️ yes, 2026-10-02 | ✅ 2026-10-02 | `cfd9791` | [`logs/00-…`](logs/00-audit-the-readiness-defects.md) |
| 01 | [Set the production quadrature `rtol`](01-set-the-production-quadrature-rtol.md) | **T1**–**T2** | Sonnet | ✍️ yes, 2026-10-02 | ✅ 2026-10-02 | `2a44f7e` | [`logs/01-…`](logs/01-set-the-production-quadrature-rtol.md) |
| 02 | [Write the version row through the recorded write](02-write-the-version-row-through-the-recorded-write.md) | **W1**–**W3** | Opus | ✍️ yes, 2026-10-02 | ✅ 2026-10-03 | `b647c0b` | [`logs/02-…`](logs/02-write-the-version-row-through-the-recorded-write.md) |
| 03 | [Open readers read-only](03-open-readers-read-only.md) | **O1**–**O4** | Opus | ✍️ yes, 2026-10-02; after 01, 02 | ✅ 2026-10-03 | `20977cb` | [`logs/03-…`](logs/03-open-readers-read-only.md) |
| 04 | [Record a `SIGTERM` as killed](04-record-a-sigterm-as-killed.md) | **K1**–**K4** | Opus | ✍️ yes, 2026-10-02; amended 2026-10-03 | ✅ 2026-10-03 | this commit | [`logs/04-…`](logs/04-record-a-sigterm-as-killed.md) |
| 05 | [Prune on resume](05-prune-on-resume.md) | **P1**–**P2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 06 | [Count only real references](06-count-only-real-references.md) | **F1**–**F2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 07 | Rehearse the A3 v2 launch | **L1**–**L3** | Opus | ⏸ held on 01–06 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-00.md`](orchestrator/prompt-00.md)
(2026-10-02, used), [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-02, used) and
[`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-02, used) and
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-03, used) and
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-03, ready). Writing 01's note found that prompt 01 gave the seam band's figure as 9.97e-07, which is `1e-6`
against `1e-7`. The sweep's §4 gives 8.26e-07 for `1e-7` against `1e-8`, and the prompt was
corrected in the same commit. It also gained a dated line on `levin-refactor`'s
`[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`, whose stale-comment list overlaps
T1, and a fifth record to name in §6.

Writing 02's note found two things prompt 02 did not know, and the prompt was amended in the same
commit:
- **W1 as written would write the primary on every open.** Every replicated `get` commits and
  deletes a record on the primary, even on a hit (audit R2). With a version `get` added in-process
  to the end of `ShardedPool.__init__` at `1c458d3`, 38 existing Datastore tests failed, each
  because `store.sqlite` changed on a reopen that should write nothing. W1 now reads the label's
  row first and writes only when it is absent, and breakage (vii) pins that.
- **The factory's prune swallows its own `SQLAlchemyError`.** It reports the error and returns
  normally, so a pool trusting its return could clear a `prune` record over shards that differ.
  §3 case 5 gained a case for it.

Writing 03's note, at `f1999b1`, found four places where 02's result or the code changes what
prompt 03 assumed. The prompt was amended in the same commit (each line dated 2026-10-03):
- **A `prune` record is completed before the comparison.** `_reconcile_replicated_tables` sends
  one to `_complete_interrupted_prune`, which deletes rows, ahead of any compare. A read-only
  refusal of a record must come before that branch (O1).
- **A missing table would fail a read-only open.** `_validate_on_startup` selects from the table
  of each of seven validating classes, and with no `_ensure_tables` a shard lacking one raises
  `no such table`. M1's measured case, `OneLoopIntegral`, validates nothing, so the audit did not
  see it. Such a class is now skipped and reported (O1).
- **`--models` is stricter than `scoped_pipeline_run.py`'s**, which drops an unknown label
  silently. The prompt's rule, that each unknown label is an error, now says that it governs (O4).
- **Breakage (v) is invisible to a checksum.** Journals are refused by presence first, and a
  read-write open of a shard with no journal writes nothing. Its test must observe the connection.

The note also records that `store_reader` imports both pool modules at module scope, so it can be
reused only by a function-scope import, and that `store_reader.py` stays off 03's file list.

Writing 04's note, at `74270dd`, found two things prompt 04 could not pass, each put to the user,
and three it had wrong. The prompt was amended in the same commit (each line dated 2026-10-03):
- **An existing test pins the drivers' post-`exec` lines**, the prompt's own fourth stop condition.
  `test_store_fingerprint.TestDrivers` counts each driver's `run.finish(` calls by `ast`: (3, 1)
  and (6, 2), split by whether they sit in a function named `terminal`. K2 moves `run_build`'s
  calls into `scoped_pipeline_run.py`, so the counts cannot hold. **The user named the test** under
  README §5 rule 7. Only its notion of a handler and its counts may change, and both assertions
  stay (§3 item 5).
- **A bare `ray.init` resolves to another cluster** (measured at 15:33). Another project's
  `ray start` cluster was running, and `/tmp/ray/ray_current_cluster` named it. Both drivers'
  `ray.init(num_cpus=…)` refused at once with `ValueError`. Without `num_cpus` they would have
  attached to it. `address="local"` started a fresh instance beside it and left nothing after
  `ray.shutdown()`. **The user put the fix in 04, as K4**, and let the probe run beside that cluster:
  the gate now excludes only Ray processes from this repository.
- **The handler remembers the signal first.** A print to stderr while the main thread is writing to
  it raises CPython's `RuntimeError: reentrant call`, and that error would have stopped the
  remember step and the raise.
- **The remembered signal decides whatever the exception's type.** A `KeyboardInterrupt` or
  `RuntimeError` after the handler fired would otherwise reach `except BaseException` and be
  recorded `failed`.
- **The unregistered `exec` has no `try` today.** It now goes through the same post-`exec`
  function, for the shutdown and the exit status.

It also leaves one thing to the agent: what is recorded when a signal arrives during a
not-signalled `finish(..., fingerprint=True)`. In that case `Run.finish` writes its own state over
`killed`. The agent records that window, and a second one in `status.json.tmp`, as observations.

**Orchestrator review of prompt 00 (2026-10-02).** Run against `cfd9791`, dispatched from
`5707851` with nothing `running` in this repository's registry, no other worktree and no Ray
process. **All nine checks in `orchestrator/prompt-00.md` §3 passed.** No §5 stop condition fired.
Three small remnants are fixed in the follow-up commit that records this review. The agent amended
its own commit once before reporting, to correct its message (first SHA `e5936d6`). That is inside
one prompt, so the commit boundary still holds.

The checks:
- **Dispatch.**
  - `HEAD` was `5707851`. The registry listed 12 runs: 10 finished and 2 `unknown` (`a3-pilot`,
    `a3-pilot-resume`), unchanged. The index held **104** rows.
  - Ray 2.43.0, Python 3.12.14, 10 CPUs. `RAY_ADDRESS` was unset, `ray status` found no instance,
    and `pgrep -lf 'gcs_server|raylet|ray::'` was empty.
  - The load average was 35.5 / 16.3 / 14.7 at 12:13.
  - The baselines matched README §7 exactly.
  - A read-only snapshot of `var/datastores/` and `var/runs/` held 69 files and 15 directories,
    with no `-journal`, `-wal` or `-shm` file.
- **Scope.**
  - 17 files, all on the note's list.
  - The diff over `*.py` outside `docs/a3-v2-readiness/` is empty.
  - The `datastore-integrity` board's numstat is `23 0`: lines added, none deleted.
- **Processes and `var/`.**
  - After the agent and after the review's own re-runs, `pgrep` was empty and `ray status` found
    no instance.
  - Every S output ends with its before/after lists and its kills.
  - The re-taken snapshot was byte-identical in all 69 files (SHA-256, size and mtime), with the
    same directories and no journal.
- **(A1) The marks.** Every finding carries a mark, and each inferred one names its source.
  - S2 states what it established. The abort is absl's 3 s `SIGALRM`, measured from the
    `sigaction` handlers, from `alarm(0)` returning 3, and from exit status 134. `alarm(0)` stops it,
    measured.
  - Two points stay inferred, and both say so: that Ray's Cython check does the `SystemExit(1)`
    conversion, and that absl's default option sets the 3 s.
  - Spot-checked as read: both drivers install `terminal` before `ray.init`, and Ray's
    `set_sigterm_handler` call in `ray.init` is unconditional.
- **(A2) The instruments.** The orchestrator re-ran `reader_writes_probe.py` and
  `readonly_open_probe.py`.
  - The outputs matched the committed `.txt` row for row: 563/563 and 125/125 lines. They differ
    only in stand-in serials, timestamps and wall time, as log observation 6 says.
  - Run 0's known writes are counted before any zero is reported. Neither probe needs Ray.
- **(A3) R1 is complete.** The orchestrator grepped `pool\.` itself over the six scripts,
  `config/model_list.py` and `extract_common.py`, together with every helper the pool is passed to
  (`resolve_run_selection`, `build_model_list`, `find_QuadSource_policy`). Every call is in R1's
  table, and the replicated column matches `config/sharding.py` `replicated_tables`.
- **S, re-run once.** S1, then S2, in the log's order, with `pgrep` empty before and after.
  - S1: all six variants caught the same exception type and code as the committed output.
  - S2: the last tick fell at 2.798–2.909 s and the runner saw the exits at 3.006–3.032 s, inside
    the recorded 2.78–3.06 s window. The load average was 10–17.
  - All three aborts in the re-run left a `gcs_server`, against 2 of 3 in the committed run. The
    probe killed its own (pids 71935, 72722 and 73536). This is consistent with S3's "can leave,
    not always".
- **S4.** 2.480 s, cold, on the 333.1 MiB store. HEAD's fingerprint did not refuse the format-1
  store: it reported the format mismatch (1 against 6) and its age's problems.
  - The time is inside the window by about 0.3 s, on a store without `GkSource` or
    `QuadSourceIntegral` rows. That A3 v2 will be slower is marked inferred.
  - Log 05's 6.66 s is quoted as an earlier figure, outside the window.
  - For 04: the fingerprint cannot sit inside the 3 s. With a handler installed after `ray.init`,
    no alarm is armed at all.
- **(A4) The closing section.** It gives facts for each of 02, 03 and 04, each pointing at a finding.
  - M1 has no "to be decided" row.
  - Two user decisions are put, U1 and U2, each with a recommendation. Both are recorded under
    *Decisions* above.
- **Records and suites.**
  - The board has 00's row, R1–M2 and §3.2's two issues.
  - Both `datastore-integrity` lines are dated, and their §1.17 hooks changed with them.
  - The index count is 106 = 104 + 2, by `grep -c`.
  - `black --check` is clean on the five probes.
  - All six suites, run serially by the orchestrator: AdaptiveLevin 32 OK, ComputeTargets 665 OK, CosmologyModels 39 OK,
    Datastore 551 OK, LiouvilleGreen 148 OK (1 skipped), RunRegistry 199 OK, 13:28–13:34 at a load
    average of 9–13. Every count is unchanged.

Remnants, fixed in the follow-up commit:
1. The audit's *Issues* section called `[10-a-sigterm-…]` **narrowed** where the board, the index
   and the log all say **widened**. It also said four classes insert on a miss where R1 says five.
   Both now agree with R1 and the board. R2 measured four, because the run's `store_tag` lookup is
   not reached on a miss.
2. This board's §1 commit column gave the subject, not the SHA.
3. README §7's suite command, `| tail -40`, can hide `AdaptiveLevin`'s verdict (log 00,
   observation 1). It now writes to a file and greps for the verdict.

**Orchestrator review of prompt 01 (2026-10-02).** Run against `2a44f7e`, which was dispatched
from `67e7e50`. At dispatch nothing was `running` in this repository's registry, and there was no
other worktree. **All eight checks in `orchestrator/prompt-01.md` §3 passed, with one exception
in check 5:** mutation (iii) is **not** caught by `test_main_plumbing`, though the prompt and the
note both said it would be. The agent found this, added a test that does catch it inside the
module the prompt allowed, and logged it as deviation 1, STRUCTURALLY REQUIRED. The review
accepts that classification and opens the test gap as an issue (§3.2). No §5 stop condition fired.

The checks:
- **Dispatch.**
  - `HEAD` was `67e7e50`, with the tree clean. The registry listed 12 runs: 10 finished and 2
    `unknown` (`a3-pilot`, `a3-pilot-resume`), unchanged. The index held **107** rows.
  - The baselines matched README §7 exactly, measured 14:49–14:56 at a load average of 12–30.
- **Scope.**
  - 9 files, all on the note's list.
  - The diff over `main.py`, `docs/handover/`, the existing tests and `var/` is empty.
  - Numstat: provenance `46 0`, `handover` `6 0`, `levin-refactor` `10 0`.
- **(B1) Only the value moved.**
  - `config/defaults.py`: of 17 top-level statements, the only `ast.dump` difference is
    `DEFAULT_QUADRATURE_RTOL = 1e-08` → `1e-07`.
  - `QuadSourceIntegral.py`: AST identical.
  - `git diff -U0` leaves the `atol` comment and line untouched.
- **(B2) The comment.** It sits above the constant. 1.9e-06 (§2), 2.84e-06 (§5), 8.26e-07 (§4,
  `1e-7` against `1e-8`) and 1.1e-02 (§0, §2.1) were each read against the sweep document. It
  names the old value, D1's `1e-8` check, and the document. "Converged" is §0 answer 3.
- **The T1 list.** The orchestrator's own run of the search gives the 12 lines of the note's
  table plus the other-tolerance hits. All are in the log, with the 12 more the unfiltered search
  returns. The two comments are corrected, and the history and the `atol` comment are left.
  - The `analytic_integral` sentence's `1e4` is now stated as true of `1e-25` only, which was the
    brief's added point.
- **(B4) The pins.** Each of (i)–(iii), extracted from the log, applies with `git apply`. Each
  fails exactly the test the log names, of 31 run with `test_main_plumbing`. `git apply -R`
  leaves `git status` clean after each.
- **`substitute_main_py`.** Four `replaced 1 occurrence` lines.
- **(B3) The records.**
  - The closed issue is in §4, and its index row is gone.
  - The provenance subsection is dated, cites the sweep and D1, and sits beneath the 2026-09-18
    entry unchanged.
  - The `handover` line keeps its issue open. The `levin-refactor` line and its hook say the three
    things.
  - The log's *Observations* carry the other-pair note and the five records, plus
    `reader_writes_probe.py`.
- **Suites.** Run serially by the orchestrator at `2a44f7e`; §5.3 below. `black --check` is clean
  on the three Python files.

Remnants, fixed in the follow-up commit that records this review:
1. The provenance subsection's `$z_{\rm response}$` had lost its `\r` to a literal carriage
   return. It was the only `\r` in the commit.
2. "This commit" in this board's §1 and §4 and in the `handover` and `levin-refactor` lines is now
   `2a44f7e`.
3. **The `handover` board's seam figure.** Its phase-2/3 table, and the
   `[a3-baseline-quadrature-tolerance-…]` index hook, gave 9.97e-07 as the seam band's `1e-7`
   deviation. That is sweep §4's `1e-6` against `1e-7`; the right figure is 8.26e-07. A dated
   correction is added under the table, and the hook is corrected. This predates 01 (log 01,
   observation 5).
4. **`[06a-readme-and-config-comments-state-the-superseded-quadrature-rtol-regime]`**
   (`tolerance-convergence`) named the `QuadSourceIntegral.py` comment 01 corrected. That entry is
   narrowed with a dated line, and its hook too. It stays open on two sites (log 01, observation 4).
5. **`[01-reader-tolerance-agreement-does-not-scan-object-read-batch]`** is opened in §3.2. The
   index count is now 107.

**The `atol` comment, after the review (2026-10-02).** The review offered the user a choice for
`config/defaults.py`'s `atol` comment: rewrite its "`rtol` was confirmed not to bind" or date it.
The user chose to rewrite it, since "`rtol` is now determinative". It is rewritten in its own
commit, comments only (`config/defaults.py`'s AST is identical, and both constants keep their
lines, `:173` and `:185`).
- The `DEFAULT_QUADRATURE_RTOL` comment above it no longer says the `atol` comment is left as
  written.
- That closes `levin-refactor`'s `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`,
  moved to that board's §4.
- It narrows `tolerance-convergence`'s `[06a-…]` to README §6.2's row alone.
- `TOLERANCE-PROVENANCE.md`'s update subsection gains a bullet. The index is now **106**.

Carried forward:
- 03 edits `extract_QuadSourceIntegral_data.py`. `test_quadrature_tolerance_pin.py`'s second class
  reads its `quad_atol`/`quad_rtol` bindings and its `object_read_batch` call, so 03's note must
  say so.
- 07's rehearsal at `(1e-32, 1e-7)` can use `main.py`'s own default for `rtol`. `RTOL_SITE` still
  matches once and is now an identity at `1e-7`.

Carried into the notes for 02–04:
- 04: `docs/handover/quadsource_atol_sweep.py` has **two** `_local_ray_init` wrappers, in
  `run_child` and `run_build`, and two `terminal` handlers, in `run_build` and `sweep`. 04 must say
  which paths the installer reaches. The charter names `run_build`; `run_child` and `sweep` are the
  sweep's own paths.
- 03: the audit recommends applying `--models` before any cosmology lookup. `build_model_list`
  (`config/model_list.py`) is outside README §1's scope list. A3 v2 holds both cosmology rows
  (R3), so the scripts can filter after `build_model_list` returns. Whether 03 may also touch
  `config/model_list.py` is settled when 03 is written.

**Orchestrator review of prompt 02 (2026-10-03).** Run against `b647c0b`, dispatched from
`56d1988`. At dispatch the tree was clean, there was no other worktree, and nothing was `running`
in this repository's registry. **All nine checks in `orchestrator/prompt-02.md` §3 passed.** No §5
stop condition fired, and the agent asked nothing. The remnants are records only, and are fixed in
the follow-up commit that records this review.

The checks:
- **Dispatch.**
  - `HEAD` was `56d1988`. The registry listed 12 runs: 10 finished and 2 `unknown` (`a3-pilot`,
    `a3-pilot-resume`), unchanged. The index held **106** rows.
  - The baselines matched §5.3 exactly, measured 23:53–00:07 at a load average of 75–386 on 10
    CPUs. The load was an unrelated Ray job in another of the user's projects, left alone.
    `test_wall_time_per_object` failed in the full ComputeTargets run, and passed alone.
  - The four pre-images were `979e0f7` (`ShardedPool.py`), `b78ced1` (`Datastore.py`), `b15fb2d`
    (`standin_pool.py`) and `abde385` (`test_reconcile_at_open.py`).
- **Scope (1).**
  - 10 files, all on the note's list.
  - The diff over `main.py`, `extract_*.py`, `docs/` other than the index, `var/` and every factory
    is empty. `version.py` is not touched.
  - `standin_pool.py`'s numstat is `40 0`. `test_reconcile_at_open.py` changes
    `TestPruningAfterRepair`'s docstring alone.
- **(C1) A clean open writes nothing (2).**
  - `Datastore.__init__` holds no `object_get("version"` and no `version_serial`.
  - `_insert`'s refusal comes before `with SerialLeaseManager`.
  - `TestNewStore` sees three `version` gets, each under one `("get", "version", controller)`
    record, and `set_version` on shards 0, 1 and 2. Faulted `"before"` on every shard, every shard
    file exists with an empty `version` table and the record is left.
  - The 38 checksum tests of the note's §0 pass unchanged in the Datastore suite. Breakage (vii)
    fails `TestCleanStoreUntouched`.
- **(C2) The version row's repair (3).**
  - `_plan_replicated_repair`'s diff is its docstring alone.
  - The log's *W2* states the timestamp-and-`store_id` fact. The `"after"` case asserts the record
    `("get", "version", controller, None)`. The one-replica case's controller is pinned by the
    stand-in's new `pin_controller`.
- **(C3) The prune (4).**
  - The new order in `__init__` is: `_check_shard_files`, the check at open,
    `replication_in_flight.create`, `_prune_replicated_tables`, then the broker. It matches the
    log's list.
  - `_commit_prune_record` commits on the primary before the first shard. Each shard is
    `_prune_shard_and_commit`'s own transaction. `_clear_in_flight_record` comes last.
  - A shard's prune counts as done only if the factory, asked again with `prune=False`, reports
    nothing. Rows may only be deleted, and only from the class and the tables that name it by
    foreign key.
  - `TestPruneFailsInTheFactory` uses a trigger that refuses the delete. The record stays, the
    open raises, and the next open completes the prune.
  - `TestRecordTableBeforePrune` rebuilds the record table with `controller_shard NOT NULL`. A
    pruning open then refuses with every file's SHA-256 unchanged, and opens without pruning.
  - The comment above `_reconcile_replicated_tables` names the recorded prune as the one
    exception.
  - Read for the review: `BackgroundModel` precedes every sharded class in `_factories`, and no
    foreign-key pragma is set. So the old actors also pruned it before any sharded class, and
    moving it into the pool changes no order.
- **The two kinds of record (5).**
  - A `prune` record returns to `_complete_interrupted_prune` before the licence is reached.
  - The licence still refuses any operation outside `get`/`store`/`validate`, which breakage (vi)
    loosens.
  - A `get` record of `BackgroundModel` reaches the licence and is repaired, not pruned
    (`TestRecordsStayApart`).
- **(C4) The pins (6).** Each of (i)–(vii), extracted from the log, applies with `git apply` at
  `b647c0b` and fails the test the log names. (i) and (ii) fail
  `TestNewStore.test_the_row_is_written_once_through_the_recorded_write`. (iii) fails
  `TestUninterruptedPrune.test_every_shard_identical_and_no_record_left`. (iv) fails
  `TestInterruptedPrune.test_faulted_after_each_shard`, with errors at `after=0` and `after=1`.
  (v) fails `TestNoActorDeletesAReplicatedRow.test_under_prune_unvalidated`. (vi) fails
  `TestTheLicenceAndTheVersionRow`'s test. (vii) fails
  `TestCleanStoreUntouched.test_opening_a_clean_store_writes_nothing`. `git apply -R` left
  `git status` clean after each.
- **The probe (7).** The orchestrator's run exits 0.
  - Case 2 crashes and reopens with `version serial 1`, and case 3 with `version serial 2`.
  - Cases 1 and 4 print `returned (no crash happened)`.
  - This is the log's account. The probe and its `.txt` are not in the diff.
- **(C5) The records (8).**
  - `[02-the-version-row-…]` is in the `datastore-integrity` board's §4, and
    `[00-the-prune-at-open-…]` in this board's §4. Each has a closing note naming the healing
    tests.
  - Both index rows are gone. `grep -c` gives **104**, which matches the header.
  - `[00-a-new-stores-first-open-…]` has one *Narrowed, not closed* line. `[00-a-drop-action-…]`
    is untouched. No issue was opened.
- **Suites (9).** Run serially by the orchestrator at `b647c0b`; §5.5 below. Datastore is 574 = 551
  + 23, and every other count is unchanged. The one failure is the `test_wall_time_per_object`
  flake. It fails the same way on the tree before 02, which does not touch `ComputeTargets/`.
  `black --check` is clean on the six Python files, as committed.

Remnants, fixed in the follow-up commit:
1. "This commit" stood where the SHA belongs: in this board's §1 row, in the closed prune entry's
   heading, closing note and *What it leaves*, and in §5.4. The `datastore-integrity` closing note
   gave the commit's subject. All now say `b647c0b`.
2. `[00-a-drop-action-…]` called itself the third replicated write a constructor makes, "beside
   the version row and the prune". After 02 it is the only one. A dated line now says so; the
   defect and its next step are unchanged. The agent rightly left the entry alone (log 02,
   observation 5).
3. The orchestrator notes list called `prompt-02.md` "ready" (now "used"), and the header did not
   say 02 was reviewed.

Not a remnant: log 02 observation 7, `replicated_write_fault_probe.py` stopping at its step 5. It
predates this campaign. `datastore-integrity` logs 08 and 09c both record it as "already failing at
`HEAD`". The probe is a record of the tree before that campaign's prompt 01.

Carried into 03's note:
- **The names.** The actor method is `set_version`, and the record's operation for a prune is
  `prune`. A `prune` record on the primary is a write in flight, which 03 refuses like any other.
- **Finding the row.** `ShardedPool._find_version_row` reads the label `mode=ro` from every shard
  and writes nothing. A read-only pool can call it and refuse on `None`, never reaching
  `_get_impl_replicated_table`. Without `set_version`, every versioned insert is already refused
  by `_insert`.
- **What else writes at open.** Under `read_only=True`, both `_prune_replicated_tables` and
  `_complete_interrupted_prune` are writes.
- **The old schema.** A store whose `replication_in_flight` predates `b647c0b` keeps
  `controller_shard NOT NULL`. A pruning open with a replicated row to prune then refuses, before
  any shard is touched. Every store under `var/` today is such a store; A3 v2 will not be.
- **The actor's signature.** `Datastore.__init__` now requires `replicated_tables`, and has lost
  `version_serial`.

**Orchestrator review of prompt 03 (2026-10-03).** Run against `20977cb`, dispatched from
`e4d6ccd`. At dispatch the tree was clean, there was no other worktree, and nothing was `running`
in this repository's registry. **All ten checks in `orchestrator/prompt-03.md` §3 passed.** No §5
stop condition fired, and the agent asked nothing. The remnants are records and one help string,
and are fixed in the follow-up commit that records this review.

The checks:
- **Dispatch.**
  - `HEAD` was `e4d6ccd`. The registry listed 12 runs: 10 finished and 2 `unknown` (`a3-pilot`,
    `a3-pilot-resume`), unchanged. The index held **104** rows.
  - The baselines matched §5.5 exactly, measured 09:35–09:45 at a load average of 15–17 on 10
    CPUs. No test failed. Part of the load was an unrelated Ray cluster in another of the user's
    projects, left alone.
  - The pre-images were `c1d30c4` (`ShardedPool.py`), `1604782` (`Datastore.py`), `25a88cd`
    (`replication.py`), `61bfc09` (`store_reader.py`), `bf58805` (`standin_pool.py`) and `d03b67f`
    (`extract_common.py`).
- **Scope (1).**
  - 16 files, all on the note's list.
  - The diff over `store_reader.py`, `config/`, `main.py`, `docs/` other than the index, `var/` and
    every factory is empty. `standin_pool.py` is not touched, and no existing test module is.
  - Both pool files and `replication.py` have numstat deletions of 0.
- **(D4) The read-write path, by reading (2).**
  - An `ast` comparison of every method against `e4d6ccd` finds that, in `ShardedPool`, only
    `__init__`, `_get_impl_replicated_table`, `object_store`, `object_validate` and
    `_assign_shard_keys` changed. In `Datastore`, only `__init__`, `_inserters_for`, `object_get`,
    `object_read_batch`, `object_store` and `object_validate` changed.
  - Each change is an early return to new code under `read_only`, a guard
    `if getattr(self, "_read_only", False):` before new code, or a call to `_read_only_refusal`,
    which returns at once on a read-write actor. Both new keywords default to `False`.
  - `_replicated_write`, `_plan_replicated_repair`, `_prune_replicated_tables`,
    `_complete_interrupted_prune`, `_find_version_row` and `_reconcile_replicated_tables` are
    unchanged.
- **(D1) Nothing written (3).**
  - `TestNothingWrittenOnAFullStore`'s first test runs QSI's whole sequence, imported from the
    probe. The read-write run is first: it changes `store.sqlite` alone, with 21
    `_replicated_write` entries.
  - The read-only run then leaves every file's SHA-256 unchanged, logs no `_replicated_write`, and
    issues no write statement. Every step's outcome equals the read-write run's.
  - Run by the orchestrator: the two new modules, `test_main_plumbing` and
    `test_quadrature_tolerance_pin`, 72 tests, OK.
- **(D2) At open (4).**
  - Every row of the log's M1 table names its function and its test. The table gains rows for 02's
    `_prune_replicated_tables`, `_complete_interrupted_prune`, version write and `set_version`.
  - A read-only pool never calls `_reconcile_replicated_tables`.
    `_check_replicated_tables_read_only` reads the record first, through the `mode=ro` engine, and
    raises `ReadOnlyWrite` naming it. It then compares on `mode=ro` engines, as the read-write
    function's no-record branch does. So a `prune` record is refused by name, and the record test
    patches both functions to fail if they are called.
  - `Datastore._validate_on_startup_read_only` skips and reports a validating class whose table is
    absent. `test_a_shard_lacking_a_table_is_reported_and_the_store_opens` drops `GkSource` (a
    validating class) and `OneLoopIntegral` in turn.
  - `_create_engine` issues no DDL, so the read-write engine it makes is never connected before it
    is replaced.
- **(D3) Every write stopped (5).**
  - `TestEachMissRaisesReadOnlyMiss` covers all five classes, each with no write statement and
    every file unchanged.
  - The actor's inserter map is rebuilt over every table, and so is each schema record's
    `insert`. The orchestrator checked in-process that every entry, the nested `*_tags` ones
    included, is `_refuse_insert`. The nested inserters are reached only from `store` paths, which
    are refused first.
  - `TestOtherWritesRaiseReadOnlyWrite` covers store, validate, a new shard key, a vectorized
    insert, `drop_actions` and `prune_unvalidated=True`, all before anything is opened, and the
    backstop.
  - The backstop matches `sqlite_errorcode == SQLITE_READONLY`, the code of "attempt to write a
    readonly database". `test_a_lookup_on_a_missing_table_fails_in_sqlite` sees `no such table`
    left as SQLAlchemy's `OperationalError`.
- **(D5) The readers (6).**
  - Each of the six has one `read_only=True`, no `prune_unvalidated`, one
    `add_models_argument(parser)`, one `select_models(...)`, one `read_models(model_list, ...)`
    and one `require_background(...)`. None still has the `RuntimeError` text or the
    `for model_data in model_list:` loop.
  - `require_background` stands at the old raise site. It raises `NoBackgroundModel`, which is not
    a `RuntimeError`. `read_models` catches it, prints the skip line, and raises `SystemExit`
    (status 1) when no model was read.
  - The deviation from "one helper" to two (deviation 5) is structurally required: the absence is
    known only inside `run_pipeline`.
- **(D6) The pins (7).** Each of (i)–(vii), extracted from the log, applies with `git apply` at
  `20977cb`. Each fails the test the log names, replayed in `bash`:
  - (i): `TestNothingWrittenOnAFullStore`, 4 errors;
  - (ii): `TestEachMissRaisesReadOnlyMiss`, 5 errors;
  - (iii): `test_the_backstop_a_flag_update_on_a_hit`;
  - (iv), in `store_reader._refuse_journals`: both empty-journal tests, 4 failures;
  - (v): `test_every_connection_is_opened_read_only`, which records every `sqlite3.connect` the pool
    module makes and every engine connection's URL, not a checksum;
  - (vi): `test_each_opens_its_pool_read_only`;
  - (vii): `TestTheSkip`, 3 errors.

  `git apply -R` left `git status` clean after each.
- **End to end (8).** Run in-process from the scratchpad on a store from `build_full_store`, and
  not committed.
  - QSI's sequence on a read-only pool left all four files unchanged. It stopped at
    `QCD_Cosmology`'s background, which is the probe's transcription of the script before 03.
  - With both `GkSourcePolicy` rows deleted on every shard (`delete_rows`), it raised
    `ReadOnlyMiss` naming `GkSourcePolicy`, and all four files were unchanged.
- **(D7) The records (9).**
  - Both issues are in the `datastore-integrity` board's §4, each with a closing note naming the
    tests. Both index rows are gone. `grep -c` gives **102**, which matches the header.
  - The log's *Observations* open no issue.
- **Suites (10).** Run serially by the orchestrator at `20977cb`, 10:33–10:44, at a load average
  of 6–80 (§5.7 below).
  - Datastore is 598 = 574 + 24, and ComputeTargets 686 = 669 + 17. Every other count is
    unchanged, and no test failed.
  - `black --check` is clean on the 12 Python files.

Remnants, fixed in the follow-up commit:
1. "This commit" stood where the SHA belongs: in this board's §1 row, the §3.1 closure note and
   §5.6; in both `datastore-integrity` closing notes; and in seven places in log 03. All now say
   `20977cb`.
2. Log 03 observation 1: the six scripts' `--database` help still said "read/write work items".
   It now says "read work items from the specified database cache, opened read-only".
   `test_reader_read_only`, `test_main_plumbing` and `test_quadrature_tolerance_pin` pass after the
   change.
3. The orchestrator notes list called `prompt-03.md` "ready" (now "used"), and the header did not
   say 03 was reviewed.

Not remnants:
- Log 03 observation 2. The test imports the probe, which the note left to the agent, and the log
  says why.
- Observations 3–5. They are the mode's stated limits:
  - the exceptions subclass `RuntimeError`;
  - journals are checked at open only;
  - a store a build has open is refused.

  07's note carries the last two.

Carried into 07's note:
- **The names and messages.** `ShardedPool(read_only=True)`; `ReadOnlyMiss` and `ReadOnlyWrite`,
  in `Datastore/replication.py` and importable from `Datastore.SQL.ShardedPool`. Their full
  messages are in log 03, *The new exceptions' messages*. Both end "Nothing was written".
- **The invocation.** Each reader takes `--models LABEL [LABEL …]`. On a LambdaCDM-only store:
  - with no `--models`, it reads LambdaCDM, prints
    `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it`, and exits 0;
  - with `--models LambdaCDM`, it reads LambdaCDM alone. That is the rehearsal's invocation;
  - an unknown label refuses before any model is read, naming `LambdaCDM, QCD_Cosmology`.
- **Journals.** A reader on a store with a journal beside it is refused at open, so run the
  readers only after the build has exited. Journals are not re-checked after open.
- **The label.** The readers open under `VERSION_LABEL = "2025.1.1"`, the build's. Any other
  label refuses with `ReadOnlyMiss("version", …)`, naming the labels present.

---

## 2. Items

| Item | Prompt | What | Status |
|---|---|---|---|
| **R1** | 00 | Every `pool.` call the six readers and `build_model_list` make: class, replicated or not, whether it inserts on a miss | ✅ audit R1: seven replicated classes `object_get`, five insert on a miss; eight sharded lookups, none writes |
| **R2** | 00 | The writes one reader's lookup sequence makes, counted on a stand-in, on a full store and on one missing a row | ✅ audit R2: QSI's whole sequence, 21 replicated gets = 42 primary commits and no row on a full store; a missing replicated row is inserted on every shard |
| **R3** | 00 | Whether A3 v2 holds a `QCD_Cosmology` row; each reader's path on a store without a model's `BackgroundModel` | ✅ audit R3: yes (read); each exits 1 at that model's `BackgroundModel` lookup |
| **S1** | 00 | A handler installed after `ray.init`: does it run during `ray.wait`, and what reaches the frame | ✅ audit S1: it runs, replaces absl's, arms no alarm; `SystemExit(143)` arrives as `SystemExit(1)` |
| **S2** | 00 | Ray's abort about 3 s after a `SIGTERM`: what it is, and whether `alarm(0)` stops it | ✅ audit S2: absl's 3 s `SIGALRM` → `SIGABRT`, 2.78–3.06 s; `alarm(0)` stops it |
| **S3** | 00 | What leaves a `gcs_server` behind | ✅ audit S3: any exit without `ray.shutdown()` (7 of 12); none through it (0 of 10) |
| **S4** | 00 | How long `finish(fingerprint=True)` takes on a real-sized store | ✅ audit S4: 2.48 s on the A3 baseline (333 MiB, partial), read-only, unchanged |
| **V1** | 00 | The version row's write sequence at open, and the crash window | ✅ audit V1: measured; two sibling windows opened |
| **V2** | 00 | The forms 02 can take, and how a read-only open finds the row | ✅ audit V2: form A recommended; found by label `mode=ro`, refuse when absent |
| **M1** | 00 | Every write at open, and what `read_only=True` does with each | ✅ audit M1: table, no row left to decide |
| **M2** | 00 | Whether every reader lookup can be served `mode=ro` | ✅ audit M2: yes, if replicated lookups skip `_replicated_write` |
| **T1** | 01 | `DEFAULT_QUADRATURE_RTOL = 1e-7`, its comment, every stale statement of its value | ✅ the constant and its 14-line comment; two stale comments in `QuadSourceIntegral.py` corrected (`MIN_SUBINTERVAL_LOG_WIDTH`'s, and `analytic_integral`'s "1e-25 and 1e-8"); every hit of the search is in log 01, 24 lines, the rest left as other tolerances or history; ASTs differ from the parents' only in the constant |
| **T2** | 01 | The provenance subsection, the `handover` board line, a value pin | ✅ subsection added (46 lines, no deletions); a dated line on each of the `handover` and `levin-refactor` entries, both issues still open; `test_quadrature_tolerance_pin.py`, four tests, three breakages each caught (log 01; (iii) needed a test of this prompt's own, log 01 deviation 1) |
| **W1** | 02 | The version row written once through `_replicated_write` | ✅ actors no longer touch it; `ShardedPool.__init__` reads the label `mode=ro` on every shard, writes it through `_get_impl_replicated_table` only when absent, then `set_version` on every actor; `_insert` refuses a versioned insert before it, ahead of the lease (`test_version_row_at_open`, 12 tests; breakages (i), (ii), (vii)) |
| **W2** | 02 | The check at open treats a missing version row like any other replicated row | ✅ the generic `get` repair admits it unchanged (no `timestamp` column); interrupted after the controller, before it, on one replica, and in a new store's first open, each healed; with no record a difference still refuses (log 02, *W2*) |
| **W3** | 02 | The prune at open of a replicated class, by the pool, under a `prune` record the check at open completes (U1) | ✅ `_prune_replicated_tables` before any actor, per-shard transactions checked by the factory's own rule, record cleared last; `_complete_interrupted_prune` re-runs it whatever `prune_unvalidated` is; actors report replicated classes only; `controller_shard` nullable (`test_prune_at_open`, 11 tests; breakages (iii)–(vi)) |
| **O1** | 03 | `ShardedPool(read_only=True)`, every file `mode=ro`, nothing written at open | ✅ `ShardedPool._open_read_only` replaces the whole constructor: `drop_actions`/`prune_unvalidated=True` refused first; `store_reader.open_read_only` (function-scope import) refuses a missing primary, creating nothing, and any journal by presence; every engine `mode=ro`; `_check_replicated_tables_read_only` refuses a record of any operation by name before anything else, then compares; no DDL, no prune, no repair; the version row found by `_find_version_row` and an absent label refused; no broker; actors `read_only=True`, validating report-only and skipping a class whose table the file lacks (`test_read_only_pool.TestAtOpen`, 10 tests; breakages (iii)–(v)) |
| **O2** | 03 | `ReadOnlyMiss` on a lookup that would have inserted; `ReadOnlyWrite` on any write | ✅ both in `Datastore/replication.py`, importable from the pool's module; a replicated get goes to one drawn shard with no record (`_get_impl_replicated_table_read_only`); every actor inserter refuses (`Datastore._refuse_insert`): `ReadOnlyMiss` for a replicated row, `ReadOnlyWrite` for any other; store, validate, a new shard key refused; SQLite's `SQLITE_READONLY` re-raised as `ReadOnlyWrite` (`_read_only_refusal`) (`TestNothingWrittenOnAFullStore`, `TestEachMissRaisesReadOnlyMiss`, `TestOtherWritesRaiseReadOnlyWrite`; breakages (i)–(iii)) |
| **O3** | 03 | The six readers open read-only | ✅ each constructs `ShardedPool(..., read_only=True)` and drops `prune_unvalidated=False`; no lookup changed (`test_reader_read_only.TestTheSixReaders`; breakage (vi)) |
| **O4** | 03 | `--models` on each reader; a model the store lacks is skipped with a message | ✅ `extract_common.add_models_argument`, `select_models` (an unknown label refuses, naming the known ones), `require_background` at the old `RuntimeError`'s site, `read_models` in place of the loop (skips with one line; exits 1 if no model is read); `config/model_list.py` unchanged (`test_reader_read_only`, 17 tests; breakage (vii)) |
| **K1** | 04 | One handler, installed after `ray.init` by both drivers, writing `killed` at once | ✅ `scoped_pipeline_run.install_killed_handler(run, latch)`, called from `main`'s and `run_build`'s `_local_ray_init` after the real `ray.init`, registered or not, for `SIGTERM` and `SIGINT`. Its handler (`make_killed_handler`'s `killed_handler`) remembers the signal in a `SignalLatch` first, writes `killed`/`128+signum` with no fingerprint, prints one line (never raising), raises `SystemExit(128+signum)`; any later call, or one after the pipeline's outcome is decided, prints and returns. Each driver's `terminal` kept before Ray. `run_child`/`sweep` unchanged (`TestTheHandler`, `TestInstall`, `TestTheDrivers`; breakages (i), (ii), (v)). On real Ray `killed`/143 was written 0.001–0.007 s after the signal in 23 of 27 runs and **1.003–1.006 s in 4**: Ray checks for signals once a second inside a wait (log 04; `[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`) |
| **K2** | 04 | The fingerprint best-effort after the state; Ray shut down; no `gcs_server` left | ✅ `finish_after_exec(run, exc, latch, ray.shutdown)` in each driver's one `except BaseException` and after the `try`. Signalled, whatever the exception: `alarm(0)`, `killed`/`128+signum` with the fingerprint, `ray.shutdown()`, `SystemExit(128+signum)`. Not signalled: the existing outcome exactly, with the fingerprint, then `ray.shutdown()`, then the driver's bare `raise`. A signal once the outcome is decided changes nothing (the prompt's choice). Probe: 27 of 27 runs `killed`/143, fingerprint present, exit 143, no abort, no `gcs_server` left (`TestSignalled`, `TestNotSignalled`; breakages (iii), (iv), (vii)) |
| **K3** | 04 | `terminal_state`'s docstring made true | ✅ rewritten against audit S1–S3; mapping unchanged; `TestTerminalState` untouched, its name recording the old belief (log 04, observation 6) |
| **K4** | 04 | Both drivers' real `ray.init` start a local instance (`address="local"`), not attach to another `ray start` cluster (added 2026-10-03, user) | ✅ both pass `address="local"`, nothing else in the call changed; each probe child's GCS on its own port beside ChamPBH's `127.0.0.1:6379` (`TestTheDrivers`; breakage (vi)). `run_child` shares the defect, recorded and unchanged |
| **P1** | 05 | `--no-prune-unvalidated` out of both invocations and both usage lines | ⬜ |
| **P2** | 05 | A test that keeps it out | ⬜ |
| **F1** | 06 | The stores root is not a target | ⬜ |
| **F2** | 06 | Tests: sequential retirements, a sidecar naming the root, the subdirectory shape | ⬜ |
| **L1** | 07 | A registered `--build` at `(1e-32, 1e-7)` into a throwaway store, `SIGTERM`ed and resumed | ⏸ |
| **L2** | 07 | The recorded states, and the fingerprint against an uninterrupted reference | ⏸ |
| **L3** | 07 | Every reader, read-only, against the resumed store | ⏸ |

---

## 3. Active and unresolved issues

### 3.1 Assigned to this campaign (2026-10-02)

Each stays on its owning board, which carries an **Assigned** line, and closes there.

| Issue | Owning board | Prompt |
|---|---|---|
| `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` | `datastore-integrity` §3.3; reassigned from `handover` | 05 |

**Closed by prompt 04 (2026-10-03, this commit), on its owning board's §4:**
`[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` (`datastore-integrity`
§4, with its closing note; log 04). Its index row is deleted.

**Closed by prompt 03 (2026-10-03, `20977cb`), on their owning board's §4:**
`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` and
`[10-extraction-scripts-exit-1-on-a-store-without-every-model]` (`datastore-integrity` §4, each
with its closing note; log 03). Both index rows are deleted.

### 3.2 Held on this board

- **[10-store-retire-reports-every-tombstone-as-a-reference]** *(seen 2026-10-02, when the user
  retired `datastore-integrity` prompt 10's two stores; indexed the same day in `424469e` with no
  board, and taken over by this board when it was created)*
  - **The defect.** `_find_references` (`RunRegistry/stores.py`) takes the store's directory,
    `primary.parent`, as a target beside the primary and the planned files. Every tombstone
    records `references.stores_root`, the directory its retirement searched. For a store directly
    in `var/datastores/` the two are the same directory, so each `store retire` reports every
    earlier tombstone there as a sidecar naming the store.
  - **Measured.** Retiring `datastore-integrity-10-reference` listed two tombstones, and
    `datastore-integrity-10-resumed` three: `handover-atol-sweep`, the backup, and
    `datastore-integrity-10-reference`, each by `$.retired.references.stores_root` and by nothing
    else (read 2026-10-02 from the resumed store's tombstone).
  - **Impact.** Report only: nothing is refused or rewritten. But the list grows by one with every
    retirement, and a real reference in it is harder to see.
  - **Next step.** Prompt 06: the stores root is not a target; a store alone in a subdirectory,
    the backup's shape, keeps its directory as one.
- **[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]** *(opened 2026-10-02 by
  audit 00, V1; does not block A3 v2)*
  - **The defect.** `ShardedPool.__init__` writes a new store's primary and its `shards` table
    first, and each shard file is created by its actor's constructor afterwards. Interrupted in
    between, the store's next open is refused by `_check_shard_files` for a shard that does not
    exist.
  - **Measured** (`version_window_probe.py`, cases 1 and 2): `shard #0 … which does not exist;
    shard #1 … which does not exist`, and `shard #1 … which does not exist`.
  - **Impact.** None to data: the store is empty. But a person must delete it, since `--build`
    refuses an existing path and `--build --resume` checks the store first.
  - **Next step.** User decision U2 (audit, closing section): recommended, record only; 07's
    rehearsal can confirm the refusal makes the remedy clear.
  - **Decided (2026-10-02):** record only (U2); no prompt changes code for it. 02 moves the version
    row after the shard files exist, which narrows the window without closing it, and its log says
    by how much. It stays open here.
  - **Narrowed, not closed (2026-10-03, prompt 02):** the shard files after the first no longer
    wait on the first actor's constructor, its version insert and a `ray.get`; every constructor is
    submitted at once, and the version row comes after the last file exists (log 02, observation 1).
- **[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]** *(opened 2026-10-02
  by the orchestrator, writing 02; read, not measured; does not block A3 v2)*
  - **The defect.** `_drop_actions` (`Datastore/SQL/Datastore.py`) maps `gk-source-policy-records`
    to `GkSourcePolicy`, a replicated table. Each actor drops it from its own shard in its
    constructor (`Datastore.__init__` → `_drop_actions`), and recreates it empty in
    `_ensure_tables`, outside the `replication_in_flight` record. It is the third replicated write
    a constructor makes, beside the version row and the prune.
  - **Since 02 (2026-10-03, `b647c0b`):** it is the only one. The version row and the prune of a
    replicated class are now the pool's, each under a record (log 02, observation 5). The defect
    and its remedy are unchanged.
  - **Impact (inferred, from the code).** `main.py --drop gk-source-policy-records`, interrupted
    between actors, would leave `GkSourcePolicy` rows on some shards and not others. The check at
    open would refuse that store, with no record to repair it from. A3 v2's launch passes no
    `--drop`, so A3 v2 does not reach this.
  - **Next step.** Not assigned. The user did not decide it with U1, so 02 does not fix it (02 §6).
    The remedy has U1's shape: the pool drops replicated tables before any actor exists, under a
    record. Measure it first on stand-in shards, as audit V1 case 4 did for the prune.
- **[01-reader-tolerance-agreement-does-not-scan-object-read-batch]** *(seen 2026-10-02 by prompt
  01's deliberate breakage (iii), log 01 deviation 1 and observation 3; opened by the
  orchestrator's review; does not block A3 v2)*
  - **The defect.** `ComputeTargets/tests/test_main_plumbing.py`'s `ReaderToleranceAgreementTestCase`
    takes each reader's targets from `lookup_constants`, which reads `object_get` and
    `object_get_vectorized` calls only. A reader that reads a tolerance-keyed target with
    `object_read_batch` is not compared with `main.py`. The module's own comment above
    `_own_nodes` records the same blind spot for one `main.py` site. It does not record it for the
    readers.
  - **Measured.** With `extract_QuadSourceIntegral_data.py`'s `rtol` lookup moved to
    `DEFAULT_GK_NUMERIC_REL_TOLERANCE`, `test_main_plumbing` still passes all 27 tests. That is log
    01's mutation (iii), replayed by the orchestrator. Prompt 01 and its orchestrator note had both
    said this test catches it.
  - **Impact.** None today. Of the six readers, only the `QuadSourceIntegral` extractor reads a
    tolerance-keyed target with `object_read_batch` (read, 2026-10-02). That read is now pinned by
    `test_quadrature_tolerance_pin.py`'s second class. A future reader of that shape would not be
    checked.
  - **Next step.** Not assigned. A prompt that may edit `test_main_plumbing.py` widens
    `lookup_constants` to `object_read_batch` calls with literal `atol`/`rtol` names. It must also
    handle `main.py`'s loop-variable site, or keep excluding it. Prompt 03 edits the extractor, so
    03 must keep `test_quadrature_tolerance_pin.py` passing.

- **[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]** *(opened 2026-10-03 by prompt 04;
  does not block A3 v2)*
  - **The defect.** A Python signal handler installed after `ray.init` runs, inside `ray.wait`,
    only when Ray next checks for signals, which is at the next whole second of the wait. So a
    `SIGTERM` is recorded `killed` from a few milliseconds to about a second after it arrives.
    Prompt 04's acceptance asked for "within a second of the signal". Audit S1's "within
    milliseconds" was measured with a timer that sent the signal 2.0 s into the wait, on that
    boundary, so it was the early side of it, not a bound.
  - **Measured** (`docs/a3-v2-readiness/sigterm_handler_probe.txt`, 27 runs).
    - `killed` was written 0.001–0.007 s after the signal in 23 runs and 1.003–1.006 s in 4, in
      both drivers' wrappers, at 1-minute loads of 4–26.
    - The probe's `phase` section sent one `SIGTERM` at each of ten offsets into one wait. The
      `SystemExit` reached the frame 0.900, 0.747, 0.499, 0.248, 0.101, 0.905, 0.504, 0.009,
      0.961 and 0.511 s after the signal, for offsets 0.10, 0.25, 0.50, 0.75, 0.90, 1.10, 1.50,
      2.00, 2.05 and 2.50 s.
    - That the period is Ray's `get_check_signal_interval_milliseconds` is inferred: the name is in
      `_raylet.so`'s strings, and its 1000 ms default is not read.
  - **Impact.** Small. Nothing aborts in that second, since no alarm is armed, and every run
    recorded `killed`/143 and exited 143. But the record can be up to ~1 s late, and the
    criterion as written fails in about one run in seven of the probe's.
  - **Next step.** A user decision: accept a bound of about one second plus the write, or have
    both drivers' real `ray.init` lower Ray's signal-check interval (a `_system_config` entry or
    the `RAY_get_check_signal_interval_milliseconds` variable). The second changes the call that K4
    kept to `address`.
- **[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]** *(opened 2026-10-03
  by prompt 04, from the orchestrator's reading; read, not measured; does not block A3 v2)*
  - **The defect.** `write_json_atomic` (`RunRegistry/__init__.py`) writes every update through
    one fixed `status.json.tmp`. If the driver's signal handler runs while a stdout heartbeat is
    inside that write's `with open(tmp, "w")`, before its `flush`, the handler's own `finish`:
    - truncates and rewrites the same file;
    - renames it to `status.json`;
    - raises.

    Unwinding then closes the interrupted handle, whose buffered partial JSON is written at
    offset 0 of the same inode, now `status.json`.
  - **Impact.** `finish_after_exec`'s second `finish` reads the file as `{}` and rewrites it valid,
    but without `pid`, `stage` or `units_*`. A `SIGKILL` between the two would leave it
    unparseable. The window is the microseconds of one `json.dump` every 30 s at most.
  - **Next step.** In `RunRegistry`: a unique temporary name per write (for example `mkstemp` in
    the same directory), so that a nested write never shares an inode with the one it interrupts.
    Not prompt 04's to change (its §6). Not assigned.
- **[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]** *(opened 2026-10-03 by prompt
  04; read, not measured; does not block A3 v2)*
  - **The defect.** `ray.init` installs Ray's `sys.exit(signum)` at its start (`worker.py`
    L1498–1502), and the driver's handler is installed only after it returns: about 2.3 s on the
    agent's machine (the orchestrator's measurement, 2026-10-03). A `SIGTERM` in that window goes
    to Ray's handler. The latch stays empty, so `finish_after_exec` takes the not-signalled path:
    `alarm(0)` (which cancels absl's alarm, if armed; breakage (i) of log 04 shows it doing so in
    a wait), then `terminal_state(exc.code)`. That is `killed`/15 if the `SystemExit` arrives as
    15, and `failed`/1 if Ray's signal check converts it to 1, with that exit status.
  - **Impact.** A run interrupted in its first seconds may read `failed`, and exits 15 or 1, not
    143. Which one happens is not established.
  - **Next step.** Measure it: a probe variant of `sigterm_handler_probe.py` that signals during
    `ray.init`. Not assigned.
- **[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]** *(opened 2026-10-03 by
  prompt 04; read; does not block A3 v2)*
  - **The defect.** `quadsource_atol_sweep.py`'s `run_child`, the sweep's per-pair child, was left
    unchanged by prompt 04 (K1, K4).
    - Its `_local_ray_init` passes no `address`. On a machine where `ray start` has left a cluster
      in `/tmp/ray/ray_current_cluster`, its `ray.init(num_cpus=…)` raises `ValueError` (the
      orchestrator's measurement for the same call, 2026-10-03).
    - It installs no handler, so Ray's governs it. A `SIGTERM` in `ray.wait` reaches it as
      `SystemExit(1)` with absl's 3 s alarm armed, and the parent `sweep` then raises its own
      `RuntimeError`.
    - `sweep`'s own `terminal` works: `sweep` never calls `ray.init`.
  - **Impact.** None on A3 v2, which is `--build`. A tolerance sweep on such a machine refuses to
    start, and an interrupted pair dies as audit S1–S3 describe.
  - **Next step.** Give `run_child`'s wrapper `address="local"` and `install_killed_handler(None,
    latch)`, and end its `exec` through `finish_after_exec`. Not assigned.

---

## 4. Resolved issues

- **[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]** *(opened
  2026-10-02 by audit 00, V1; does not block A3 v2; **closed 2026-10-03 by prompt 02**,
  `b647c0b`)*
  - **The defect.** With `prune_unvalidated=True` (`main.py`'s default, and A3 v2's resume once
    prompt 05 lands), each `Datastore` actor deletes an unvalidated `BackgroundModel` — a
    replicated class — with its values and tags from its own shard, in its own constructor
    (`_validate_on_startup` → `sqla_BackgroundModelFactory.validate_on_startup`), outside the
    `replication_in_flight` record. The same defect as
    `[02-the-version-row-is-written-outside-the-in-flight-record]`, for the other replicated write
    a constructor makes.
  - **Measured** (`docs/a3-v2-readiness/version_window_probe.py`, case 4, stand-in shards): every
    actor after the first killed during a pruning open left shard 2 with no `BackgroundModel` and
    shards 0 and 1 with `[(1, 0)]`; the next open was refused by the check at open with five
    differences (`BackgroundModel`, three `BackgroundModelValue`, one `BackgroundModel_tags`).
  - **Impact.** A store that needs a person. Reached only when a pruning open of a store holding an
    unvalidated `BackgroundModel` is interrupted between actors; for A3 v2, only if a launch is
    interrupted between the background's store and its validate and the resume is then interrupted
    in its first seconds.
  - **Next step.** User decision U1 (audit, closing section): recommended, in prompt 02, the pool
    prunes replicated classes itself, before any actor exists, under a `prune` record that the
    check at open repairs by running the idempotent prune again.
  - **Assigned (2026-10-02):** to prompt 02, as its W3. The user took U1 as recommended. The same
    defect as the version row's, for the other replicated write a constructor makes, so it is
    fixed in the same change to `ShardedPool.__init__` and the check at open.
  - **Closed 2026-10-03 by prompt 02**, `b647c0b`. Under `prune_unvalidated`, the pool prunes
    every replicated class whose factory prunes at startup (`BackgroundModel`, its tags and values),
    after the check at open and before the broker or any actor exists, under a `prune` record
    committed before the first shard and cleared last (`ShardedPool._prune_replicated_tables`).
    Each shard's prune is its own transaction, checked by the factory's own report and by what was
    deleted. The check at open completes an interrupted one by running it again on every shard
    (`_complete_interrupted_prune`). Actors report replicated classes only. Healed, on stand-in
    shards, in `Datastore/tests/test_prune_at_open.py`:
    `TestInterruptedPrune.test_faulted_after_each_shard` (after shard 0, 1 and 2),
    `test_faulted_before_the_first_shard`, and
    `TestPruneFailsInTheFactory.test_the_record_stays_and_the_next_open_completes_it`; no actor
    deletes a replicated row (`TestNoActorDeletesAReplicatedRow`). Log:
    [`logs/02-write-the-version-row-through-the-recorded-write.md`](logs/02-write-the-version-row-through-the-recorded-write.md).
  - **What it leaves.** A store whose `replication_in_flight` table was created before `b647c0b`
    keeps `controller_shard NOT NULL` and refuses a pruning open that has a replicated row to prune,
    before any shard is touched (log 02, *The record's schema*). The audit probe's case 4 no longer
    kills anything (log 02, observation 3).
- **[00-quadsource-extraction-reads-at-the-default-rtol]** *(opened 2026-10-02 by this campaign's
  plan; **closed 2026-10-02 by prompt 01**, `2a44f7e`)*
  - **The defect.** `extract_QuadSourceIntegral_data.py` looked up the tolerance rows its
    `QuadSourceIntegral` reads are keyed on at `DEFAULT_QUADRATURE_RTOL`, which was `1e-8`, while
    A3 v2 is built at `1e-7` (D1) by a substitution into `main.py` that `config.defaults` never
    learned of. The extractor would have matched none of A3 v2's rows.
  - **Closed by.** `config/defaults.py`: `DEFAULT_QUADRATURE_RTOL = 1e-7` (D2), with its comment and
    its provenance (`docs/TOLERANCE-PROVENANCE.md`, *Update 2026-10-02*). `main.py`, the build and
    the extractor now share the pair `(1e-32, 1e-7)`. Pinned by
    `ComputeTargets/tests/test_quadrature_tolerance_pin.py`, which also pins that the extractor binds
    its `quad_atol`/`quad_rtol` to the two constants. Log:
    [`logs/01-set-the-production-quadrature-rtol.md`](logs/01-set-the-production-quadrature-rtol.md).
  - **What it leaves.** A store built at any pair other than `(1e-32, 1e-7)` is unreadable by the
    extractor at the default; log 01, observation 1. The extractor's `object_get` on the replicated
    `tolerance` class still inserts a row on a miss, which is `[08-…]`'s (prompt 03).

---

## 5. Baselines

Inherited, README §7: AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39, Datastore 551,
LiouvilleGreen 148 (1 skipped), RunRegistry 199, at `f4f1896`. Each prompt adds a subsection here
with the counts after its commit.

### 5.1 After prompt 00 (2026-10-02)

Measured 13:12–13:19 at the tree of this commit (the audit adds no test), serially, each suite's
output written to a file and its verdict read from it (log 00, deviation 9):

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 665 OK |
| CosmologyModels | 39 OK |
| Datastore | 551 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Unchanged from README §7. Load average 10–27 during the run; no flake fired.

### 5.2 After prompt 01 (2026-10-02)

Measured 15:01–15:08 at the tree of `2a44f7e`, serially, each suite's output written to a file and
its verdict read from it. The first background pass of `ComputeTargets` stopped without a verdict
(log 01, observation 9); it was re-run alone:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 669 OK |
| CosmologyModels | 39 OK |
| Datastore | 551 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`ComputeTargets` is 665 + the four tests of `test_quadrature_tolerance_pin.py`; every other count is
unchanged from README §7. Load average 6–12 during the run, on a machine shared with unrelated work;
no flake fired. Index: **106 open**.

### 5.3 After prompt 01, re-measured by the orchestrator (2026-10-02)

Measured 16:01–16:07 by the orchestrator's review at `2a44f7e`'s code (the follow-up touches only
Markdown), serially, with README §7's command:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 669 OK |
| CosmologyModels | 39 OK |
| Datastore | 551 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

These agree with §5.2. The load average was 6–8, and no flake fired. Index: **107 open**, after the
review opened `[01-reader-tolerance-agreement-does-not-scan-object-read-batch]`.

### 5.4 After prompt 02 (2026-10-03)

Measured 00:35–00:46 at the tree of `b647c0b`, serially, with README §7's command, each suite's
output written to a file and its verdict read from it:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 669 OK |
| CosmologyModels | 39 OK |
| Datastore | 574 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Datastore is 551 + the 23 tests of `test_version_row_at_open.py` (12) and `test_prune_at_open.py`
(11); every other count is unchanged from §5.3. Load average 10–22, on a machine shared with an
unrelated Ray job; no flake fired. Index: **104 open**.

### 5.5 After prompt 02, re-measured by the orchestrator (2026-10-03)

Measured 00:59–01:11 by the orchestrator's review at `b647c0b`'s code (the follow-up touches only
Markdown), serially, with README §7's command:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 669: 668 pass, 1 wall-clock flake (below) |
| CosmologyModels | 39 OK |
| Datastore | 574 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

The counts agree with §5.4. The load average was 12–17.
- **The flake** is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`: cold 0.0996 s against
  its 0.06 s limit.
- **Run alone** at a load of 11–17 it failed again, twice.
- **Run on an export of `56d1988`**, the tree before 02, under the same load, it failed the same
  way (0.065 s and 0.081 s, against 0.082 s and 0.091 s at `b647c0b`).
- **Not 02's.** `git diff 56d1988 b647c0b -- ComputeTargets` is empty. This is the known wall-clock
  flake with its ~1× margin, not a regression.

Index: **104 open**.

### 5.6 After prompt 03 (2026-10-03)

Measured 10:14–10:24 at the tree of `20977cb`, serially, with README §7's command, each suite's
output written to a file and its verdict read from it:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 OK |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

ComputeTargets is 669 + the 17 tests of `test_reader_read_only.py`; Datastore is 574 + the 24 of
`test_read_only_pool.py`. Every other count is unchanged from §5.5. Load average 8–29 (1-minute) on
10 CPUs, shared with an unrelated Ray cluster; no flake fired. Index: **102 open**.

### 5.7 After prompt 03, re-measured by the orchestrator (2026-10-03)

Measured 10:33–10:44 at `20977cb`, serially, with README §7's command, at a 1-minute load average
of 6–80 on 10 CPUs. The load peaked during ComputeTargets.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 OK |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

The counts agree with §5.6. No test failed.

Index: **102 open**.

### 5.8 After prompt 04 (2026-10-03)

Measured 16:35–16:43 at the tree of this commit, serially, with README §7's command, each suite's
output written to a file and its verdict read from it:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 OK |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 225 OK |

RunRegistry is 199 + the 26 tests of `test_killed_handler.py`; `TestDrivers` changed in place (§3
item 5) and adds none. Every other count is unchanged from §5.7. Load average 5.9–17.5 (1-minute)
on 10 CPUs, beside another project's Ray cluster; no flake fired. Index: **105 open**.
