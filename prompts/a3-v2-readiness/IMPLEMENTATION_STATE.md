# A3 v2 readiness campaign — implementation state

**Last updated:** 2026-10-02 · **Status: IN PROGRESS — 2 of 8 prompts landed.** Audit 00 has landed
([`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md)) and been reviewed; prompt 01
has landed (2026-10-02) and been reviewed. 01–06
are written: 02, 03 and 04 on 2026-10-02, against the audit and the user's decisions U1 and U2. 07
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
| 02 | [Write the version row through the recorded write](02-write-the-version-row-through-the-recorded-write.md) | **W1**–**W3** | Opus | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 03 | [Open readers read-only](03-open-readers-read-only.md) | **O1**–**O4** | Opus | ✍️ yes, 2026-10-02; after 01, 02 | ⬜ | — | — |
| 04 | [Record a `SIGTERM` as killed](04-record-a-sigterm-as-killed.md) | **K1**–**K3** | Opus | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 05 | [Prune on resume](05-prune-on-resume.md) | **P1**–**P2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 06 | [Count only real references](06-count-only-real-references.md) | **F1**–**F2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 07 | Rehearse the A3 v2 launch | **L1**–**L3** | Opus | ⏸ held on 01–06 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-00.md`](orchestrator/prompt-00.md)
(2026-10-02, used), [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-02, used) and
[`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-02, ready). Writing 01's note found that prompt 01 gave the seam band's figure as 9.97e-07, which is `1e-6`
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
| **W1** | 02 | The version row written once through `_replicated_write` | ⬜ |
| **W2** | 02 | The check at open treats a missing version row like any other replicated row | ⬜ |
| **W3** | 02 | The prune at open of a replicated class, by the pool, under a `prune` record the check at open completes (U1) | ⬜ |
| **O1** | 03 | `ShardedPool(read_only=True)`, every file `mode=ro`, nothing written at open | ⬜ |
| **O2** | 03 | `ReadOnlyMiss` on a lookup that would have inserted; `ReadOnlyWrite` on any write | ⬜ |
| **O3** | 03 | The six readers open read-only | ⬜ |
| **O4** | 03 | `--models` on each reader; a model the store lacks is skipped with a message | ⬜ |
| **K1** | 04 | One handler, installed after `ray.init` by both drivers, writing `killed` at once | ⬜ |
| **K2** | 04 | The fingerprint best-effort after the state; Ray shut down; no `gcs_server` left | ⬜ |
| **K3** | 04 | `terminal_state`'s docstring made true | ⬜ |
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
| `[02-the-version-row-is-written-outside-the-in-flight-record]` | `datastore-integrity` §3.3 | 02 |
| `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` | `datastore-integrity` §3.3 | 03 |
| `[10-extraction-scripts-exit-1-on-a-store-without-every-model]` | `datastore-integrity` §3.3 | 03 |
| `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` | `datastore-integrity` §3.3 | 04 |
| `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` | `datastore-integrity` §3.3; reassigned from `handover` | 05 |
| `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]` | this board, §3.2 | 02 (U1) |

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
- **[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]** *(opened
  2026-10-02 by audit 00, V1; does not block A3 v2)*
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
- **[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]** *(opened 2026-10-02
  by the orchestrator, writing 02; read, not measured; does not block A3 v2)*
  - **The defect.** `_drop_actions` (`Datastore/SQL/Datastore.py`) maps `gk-source-policy-records`
    to `GkSourcePolicy`, a replicated table. Each actor drops it from its own shard in its
    constructor (`Datastore.__init__` → `_drop_actions`), and recreates it empty in
    `_ensure_tables`, outside the `replication_in_flight` record. It is the third replicated write
    a constructor makes, beside the version row and the prune.
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

---

## 4. Resolved issues

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
