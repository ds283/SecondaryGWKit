# Campaign — A3 v2 readiness

**Written:** 2026-10-02 on `handover-remedial` at `424469e`, by Claude Opus 5.5, from the open-issue
index and the user's decisions in §6. **Prompts 00, 01, 05 and 06 are written.** Their tasks are
fixed by §6 alone. **02, 03 and 04 were held** on what audit 00 measured (§2). They were written on
2026-10-02, after the audit's review, against its findings and the user's decisions U1 and U2
(§6.2). **07 is held** on 01–06.

## 0. Why this campaign exists

The A3 v2 LambdaCDM store is the first store meant to have archival value, and the comparator for
the hand-over change (`handover` E1, B2). `datastore-integrity` closed on 2026-10-02 having fixed
every defect it was opened for; its prompt 10 built a throwaway store end to end, and that run and
its review left a short list of things that would go wrong on A3 v2 itself. On 2026-10-02 the user
listed three, asked whether there were others, and decided that all seven below are fixed first:

1. **The launch resumes without pruning** (`[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`).
   `quadsource_atol_sweep.py --build` passes `--no-prune-unvalidated`, so a resume duplicates every
   row stored but not validated at the interruption. Measured: 45 `GkSource` rows twice.
2. **The extraction scripts exit 1 on a LambdaCDM-only store**
   (`[10-extraction-scripts-exit-1-on-a-store-without-every-model]`), which A3 v2's is.
3. **`store retire` reports every earlier tombstone as a reference**
   (`[10-store-retire-reports-every-tombstone-as-a-reference]`), and the list grows with each
   retirement.
4. **A `SIGTERM` is never recorded `killed`**
   (`[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`). `run_build` uses the
   same `terminal_state` and the same registration as `scoped_pipeline_run.py`, so an interrupted
   A3 v2 build reads as a real failure, or as a crashed run with a `gcs_server` left behind.
5. **A reader writes to the store it reads**
   (`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`). Recorded
   for the `GkSourcePolicy` lookup in two scripts. Read on 2026-10-02 it is wider: **every**
   `object_get` on a replicated class goes through `ShardedPool._replicated_write`, which commits an
   in-flight record on the primary even on a hit, and the six scripts also `object_get` the
   `tolerance` rows, the wavenumbers, the exit times and, through `build_model_list`, both
   cosmologies. Audit 00 measures how wide.
6. **The version row is written outside the in-flight record**
   (`[02-the-version-row-is-written-outside-the-in-flight-record]`). A crash in a new store's first
   open leaves a store that refuses to open.
7. **The extractor reads `QuadSourceIntegral` at the wrong tolerance**
   (`[00-quadsource-extraction-reads-at-the-default-rtol]`, opened by this plan).
   `extract_QuadSourceIntegral_data.py` looks its rows up at `DEFAULT_QUADRATURE_RTOL = 1e-8`. A3
   v2 is built at `1e-7` by a textual substitution into `main.py`, so the extractor finds none of
   its rows, and its `object_get` may insert a `1e-8` tolerance row into the store.

**0.1 Correctness is the only objective.** Work is sequenced by epistemic dependency:
1. The audit comes first, because three of the remedies rest on mechanisms that are inferred, not
   measured (Ray's signal handling, which reader calls write), and one issue is already wider than
   its record.
2. The tolerance default before the readers, so that the extractor's lookup pair is the build's
   before anything is tested against a real store.
3. The version row before the read-only pool, because a read-only open must *find* the version
   row and never write it, and how it is written decides how it is found.
4. The launch's own fixes (`SIGTERM`, pruning) and the registry's (tombstones) are independent.
5. Last, a rehearsal of the launch on a throwaway store, so that A3 v2 is not the first thing to
   exercise the result.

## 1. Scope

**In scope:**
- `config/defaults.py`: `DEFAULT_QUADRATURE_RTOL`, under D2 (prompt 01);
- `Datastore/SQL/Datastore.py` and `Datastore/SQL/ShardedPool.py`: the version row (02) and a
  read-only pool (03);
- the six `extract_*.py` scripts: read-only, a model filter (03);
- `docs/gktk-remedial/scoped_pipeline_run.py` and `docs/handover/quadsource_atol_sweep.py`: the
  signal handling (04) and the build's argv (05);
- `RunRegistry/stores.py`: `_find_references` (06);
- the seven issues in §0, and the boards that own them.

**Out of scope:**
- the A3 v2 launch itself, which is `handover`'s, and the user's to dispatch;
- every other open issue on the `datastore-integrity` board — none touches a build (its
  verification record, `docs/datastore-integrity-verification.md` §2.6, says why for each);
- `handover`'s `[03-gk-classification-is-diagnosed-and-ignored]`. A `fail` row would stop the build
  in `build_partition`; the A3 baseline passed that stage on the same grid, so it is not expected
  to recur. Not a readiness defect;
- `DEFAULT_LEVIN_MAX_DEPTH` and `DEFAULT_QUADRATURE_ATOL`, unchanged;
- every store under `var/` except prompt 07's own (§3).

## 2. Prompts

| # | Prompt | Covers | Status |
|---|---|---|---|
| 00 | [`00-audit-the-readiness-defects.md`](00-audit-the-readiness-defects.md) | Read-only. **R** every write a reader makes: each `object_get` in the six scripts and `build_model_list`, which inserts on a miss, which writes an in-flight record on a hit, measured on a stand-in pool. **S** the `SIGTERM` path: does a handler re-installed after `ray.init` survive `ray.wait`; does `signal.alarm(0)` cancel Ray's abort; how long `finish(fingerprint=True)` takes on a real-sized store. **V** the version-row window, and whether a read-only open can find the row. **M** what the read-only mode needs from `store_reader.py`, prompt 02's check at open, and SQLite's `mode=ro`. Writes `docs/a3-v2-readiness-audit.md`; changes no code. | **written**; landed 2026-10-02, `cfd9791` |
| 01 | [`01-set-the-production-quadrature-rtol.md`](01-set-the-production-quadrature-rtol.md) | `DEFAULT_QUADRATURE_RTOL = 1e-7`, with its provenance; the stale comments that state its value; `TOLERANCE-PROVENANCE.md`, additively. Closes `[00-quadsource-extraction-reads-at-the-default-rtol]`. | **written**; independent of 00; landed 2026-10-02, `2a44f7e` |
| 02 | [`02-write-the-version-row-through-the-recorded-write.md`](02-write-the-version-row-through-the-recorded-write.md) | Actors are built without inserting the version row; the row is then written once through `_replicated_write`; the check at open sees a missing version row like any other replicated row. Under U1, the prune at open of a replicated class moves to the pool, under a `prune` record the check at open completes. Closes `[02-the-version-row-…]` and `[00-the-prune-at-open-…]`. | **written** 2026-10-02; landed 2026-10-03, `b647c0b` |
| 03 | `03-open-readers-read-only.md` | A read-only pool (§4): every file `mode=ro`, no in-flight record, no prune, no drop, no repair; a lookup that misses refuses instead of inserting. The six scripts open it, and take `--models`, skipping with a message a model the store does not hold. Closes `[08-…]` and `[10-extraction-scripts-exit-1-…]`. | **written** 2026-10-02 ([`03-open-readers-read-only.md`](03-open-readers-read-only.md)); after 01 and 02; landed 2026-10-03, `20977cb` |
| 04 | `04-record-a-sigterm-as-killed.md` | One handler, shared by both drivers, installed after `ray.init`: `killed`/143 written at once without the fingerprint, then the fingerprint best-effort, then Ray shut down, no `gcs_server` left. `terminal_state`'s docstring made true. Closes `[10-a-sigterm-…]`. | **written** 2026-10-02 ([`04-record-a-sigterm-as-killed.md`](04-record-a-sigterm-as-killed.md)); landed 2026-10-03, `79ab21a` |
| 05 | [`05-prune-on-resume.md`](05-prune-on-resume.md) | `--no-prune-unvalidated` out of `run_build`, `run_child` and both drivers' usage lines, with a test pinning the build's argv. Closes `[10-a3-v2-launch-…]`. | **written**; independent; landed 2026-10-03, `dd7f272` |
| 06 | [`06-count-only-real-references.md`](06-count-only-real-references.md) | `_find_references` stops matching a store's directory when that directory is shared with other stores, so a tombstone's `references.stores_root` no longer names every later store. The backup shape (a store alone in its own directory) is still found. Closes `[10-store-retire-…]`. | **written**; independent |
| 07 | `07-rehearse-the-a3-v2-launch.md` | Remedial. A scoped `--build --register` at `(1e-32, 1e-7)` into a throwaway store, `SIGTERM`ed in a sharded stage and resumed with `--build --resume`; the recorded state, the fingerprint against an uninterrupted reference, and every extraction script read-only against it. Closes the campaign. | **held** on 01–06 |

**Why 02–04 and 07 were held** (02–04 released 2026-10-02). Their charters above are fixed and cannot drift to fit what 00
finds. What waits is their method: 02's form depends on whether a read-only open can find the
version row as written today, 03's on the list of writes 00 makes, and 04's on whether the
alarm can be cancelled at all. 07 is written against what 01–06 ship.

**Order.**
- 00 first. 01, 05 and 06 do not depend on it and may land before it or beside it, one at a time.
- 01 → 03, and 02 → 03.
- 04 and 05 both edit `quadsource_atol_sweep.py`'s `run_build`; never concurrently in one checkout.
- 07 last.

## 3. Datastores

One store lives under `var/datastores/`: the A3 baseline, `handover-A3-baseline-lambdacdm`. Its
sidecar says Keep; it is the only comparator until A3 v2 exists and is verified. **No prompt opens
it.** Audit 00 may read it `mode=ro` through `Datastore/store_reader.py` only, to time a fingerprint
on a real-sized store, and records that it did.

**Prompts 00–06 write nothing under `var/`.** Their tests build stores in temporary directories or
in memory. **Prompt 07 is the one exception**, and it writes only new stores of its own, under names
no other store, tombstone or run uses, and never A3 v2's.

**Before touching any store,** run `python -m RunRegistry list` and confirm that nothing is
`running`.

**A3 v2 is planned for the macstudio** (user, 2026-09-24), which keeps its own registry. A `list`
here does not see a job there, and an empty listing here is not evidence that nothing is running.

## 4. The interfaces between prompts

The **names** below are fixed here so that each prompt can be written against the one before it.
The internals are the implementing prompt's to design.

- **Prompt 01** ships `DEFAULT_QUADRATURE_RTOL = 1e-7`. From then on `main.py` unpatched, the build
  and the extractor agree on the pair.
- **Prompt 03** ships:
  - a keyword **`read_only=True`** on `ShardedPool`, passed to each `Datastore` actor;
  - the exception **`ReadOnlyMiss`**, raised when a lookup on a read-only pool matches no row,
    naming the class and the payload, where a read-write pool would have inserted;
  - the exception **`ReadOnlyWrite`**, raised by any store, validate, prune, drop or repair
    attempted on a read-only pool, before anything is written;
  - **`--models`** on each `extract_*.py`, with `scoped_pipeline_run.py --models`' meaning.
- **Prompt 04** ships one function, named by 04, that both drivers call from their `ray.init`
  wrappers to install the handler.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`
   (§1.17), plus the owning board of any issue it closes.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. A
   prompt may add tests beside them.
8. **Deliberate breakage.** Each prompt names mutations its tests must catch, and the log records
   each as a diff, exactly as applied, so the orchestrator can replay it with `git apply`.
   Mutations are never committed.
9. **No test needs Ray, and no test opens anything under `var/`.** Stand-in shards follow
   `docs/datastore-integrity-audit/replicated_write_fault_probe.py`. A probe that needs a real Ray
   instance (00's `SIGTERM` measurements, 04's verification) is a script under
   `docs/a3-v2-readiness/`, run by hand, its output committed beside it; never a test.
10. **No agent writes a store under `var/`**, except prompt 07's own new stores.
11. **`main.py` edits keep `quadsource_atol_sweep.py` working.** A prompt that edits `main.py` runs
    `substitute_main_py(filter_cases=False)` afterwards and shows each of its four substitutions
    still matches once.
12. **A long job is launched through the registry**, detached, and not polled (`CLAUDE.md`, the
    run registry). Only 07 launches one.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-02), and not reopened here

- **D1 — A3 v2's pair is `(atol, rtol) = (1e-32, 1e-7)`.** This is the pair recommended on
  2026-09-24 as the loosest converged in all three response-redshift bands
  (`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` §0, §7), accepted then, and recorded as the next
  step on the `handover` board. The user's 2026-09-23 remark that science outputs be re-run at
  `1e-9` meant **one rung tighter than the pair a store is built at** — `1e-8` was shipped when it
  was said — so for a store at `1e-7` the check is `1e-8`. The `handover` board records this beside
  the original wording.
- **D2 — the default is set, not a flag added.** `DEFAULT_QUADRATURE_RTOL` becomes `1e-7`, so that
  `main.py`, the build and the extractor share one constant and cannot drift. The `handover` board
  records `DEFAULT_QUADRATURE_*` as `levin-refactor`'s and `qsi-phase-groups`' to change; both are
  closed, and this is the user's decision.
- **D3 — readers stop writing by a read-only mode, not by per-site lookups.** The defect is a class
  of calls, and only a store opened `mode=ro` proves a reader did not write.
- **D4 — one campaign owns all seven**, under this name, including the three issues first assigned
  to `handover` or left without a board.

### 6.2 Decisions this campaign expects to need

Audit 00 put two (its closing section). The user took both on 2026-10-02, each as recommended:

- **U1 — the prune at open of a replicated row goes through a record, in 02.** Yes. The pool, not
  the actors, prunes replicated classes, before any actor exists, under a `prune` record that the
  check at open completes by re-running the idempotent prune.
  `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]` is assigned to 02.
- **U2 — a new store's first open, interrupted before every shard file exists.** Record only; no
  code change. `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` stays open on
  this campaign's board. The store is empty and a person deletes it.

None is open.

## 7. Baselines

Last measured at `f4f1896` by `datastore-integrity` prompt 10 (its log, *State handed to the next
prompt*); the two commits since touch only `docs/OPEN_ISSUES.md` and a board:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 665 OK |
| CosmologyModels | 39 OK |
| Datastore | 551 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Re-measure every suite at `HEAD`, clean, before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

The suites print banners, and `AdaptiveLevin`'s buffered stdout is flushed after unittest's summary, so
no `| tail -N` reliably shows the verdict (log 00, observation 1); read it from the file. Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`. `ComputeTargets`' `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake.
