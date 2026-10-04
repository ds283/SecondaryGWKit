# Log — prompt 07: rehearse the A3 v2 launch on a throwaway store

**Subject:** *Rehearse the A3 v2 launch on a throwaway store* · **Model:** Claude Opus 5.5 ·
**Date:** 2026-10-04 · **Base:** `dbd8a51`, clean (its code is `7bd202f`'s, 06a) · **Result:**
landed, with **acceptance 4 not met**.
- **The four launches.** A3 v2's launcher, `quadsource_atol_sweep.py --build --rehearsal
  --register`, built a reference store straight through. It built a second store that was killed
  twice by `SIGTERM`, resumed twice with `--build --resume`, and run to `done`.
- **The comparison.** The resumed store's fingerprint equals the reference's: digest `39809dca…`,
  all 21 classes.
- **Every launch ended as recorded**: `done`/0, `killed`/143, `killed`/143, `done`/0. Each driver's
  exit status is in its launch log, and each killed launch has its fingerprint.
- **The resume recomputed nothing it had stored**, pruned exactly the 35 unvalidated rows, and
  duplicated nothing. Both stores are sound.
- **Two §5 stops**, each put to the user through the orchestrator:
  - a killed build's Ray agents outliving the driver (§2.1.3 of the record). Decided: wait for
    them to exit, record it, continue.
  - `extract_tensor_source_data.py` exiting 1, a defect of the reader that predates this campaign
    (record §2.3.2). Decided: record it, and exercise the rest of the reader through a wrapper that
    sorts its draw.
- **The other readers.** The other five exit 0. No reader refused anything or changed any file of
  the store.
- **Issues.** Three opened on the board's §3.2. The index goes from 102 to 105.
- The commit is `a6c8f39`.

The record is [`docs/a3-v2-readiness-verification.md`](../../../docs/a3-v2-readiness-verification.md);
its small outputs are in [`docs/a3-v2-readiness/rehearsal/`](../../../docs/a3-v2-readiness/rehearsal/).

## The decisions, as the brief gave them

- **D7-1 — twice** (the user, 2026-10-04, as recommended).
  - `resumed-a` is interrupted in `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`.
  - `resumed-b`, with `--resume`, is interrupted in `CALCULATE QUADRATIC SOURCE INTEGRALS`.
  - `resumed-c`, with `--resume`, runs to `done`.
  - So four launches: `reference`, `resumed-a`, `-b` and `-c`.
- **D7-2 — keep** (the user, 2026-10-04, as recommended). Both stores stay in `var/datastores/`
  with their sidecars. `store retire` is never run here.
- **Taken during the prompt, by the user through the orchestrator (2026-10-04):**
  - on the orphaned Ray agents: wait for the orphan to exit on its own, and if it does, record it
    and continue. It did;
  - on the tensor-source reader: record it as an issue, with acceptance 4 not met for it, and
    exercise the rest of the reader through a sorted-order wrapper, labelled as not the shipped
    reader.

## What shipped

No production code, test or verification tool changed. Nothing tracked was modified before this
commit. Every launch ran a tree whose `HEAD` (`dbd8a51`) held all the code that ran, with `git
status --short` empty.

- **`docs/a3-v2-readiness-verification.md`**, new: every measurement, every command verbatim, the
  checklist for A3 v2's launch (§2.4) superseding `docs/datastore-integrity-verification.md` §2.6
  (left as it is), and the issues opened.
- **`docs/a3-v2-readiness/rehearsal/`**, new.
  - **Scripts**, each read-only on the stores:
    - `closed_store_state.py`: journals, `replication_in_flight`, `version` rows and
      `validated = 0` rows per file;
    - `prune_serials.py`: the 35 serials by shard, and distinct `GkSource` keys;
    - `store_soundness.py`: `integrity_check`, `foreign_key_check`, `version` and
      `read_inventory`;
    - `stage_counts_07.py`: `stage_counts.read`, labelled by `-07-`;
    - `run_readers.sh`: the readers, with SHA-256, mtime and journal checks;
    - `tensor_source_sorted_sample.py`: **not the shipped reader**; the unmodified tensor-source
      reader with its `random.sample` sorted.
  - **Outputs:**
    - `interrupter_dummy_test.txt`, `interrupter_build_dummy_test.txt`, `interrupter_firings.txt`;
    - `prelaunch_checks.txt`, `run_records.txt`, `store_sidecars.txt`;
    - `closed_store_state.txt`, `stage_counts.txt`, `prune_serials.txt`;
    - `census.txt`, `store_soundness.txt`, `fingerprint_comparison.txt`;
    - `guard.txt`, `readers.txt`, `reader_rows.txt`.
- **This log**, the board (`IMPLEMENTATION_STATE.md`: header, §1 row 07, L1–L3, three §3.2 entries,
  §5.16) and `docs/OPEN_ISSUES.md` (three rows in §1.17, the count and the date).

### Every run

| Slug | Run id | pid | State | Exit code | Driver exit status | Wall time | Load (1/5/15 min) before / after |
|---|---|---|---|---|---|---|---|
| `reference` | `a3-v2-readiness-07-reference-20261004T012909` | 27325 | `done` | 0 | 0 | 12 m 58 s | 13.10 13.27 15.59 / 20.57 43.94 39.60 |
| `resumed-a` | `a3-v2-readiness-07-resumed-a-20261004T014440` | 41825 | `killed` | 143 | 143 | 6 m 29 s | 20.57 43.94 39.60 / 22.18 44.48 43.95 |
| `resumed-b` | `a3-v2-readiness-07-resumed-b-20261004T015203` | 49456 | `killed` | 143 | 143 | 8 m 18 s | 19.04 42.28 43.17 / 123.24 110.98 76.89 |
| `resumed-c` | `a3-v2-readiness-07-resumed-c-20261004T020421` | 64172 | `done` | 0 | 0 | 4 m 54 s | 16.72 68.25 65.91 / 62.95 72.16 67.76 |

The pid is the Python process's, in `status.json`; the `sh -c` wrappers were 27322, 41820, 49451
and 64166. Wall time runs from the manifest's `created` to the final heartbeat. All four manifests
name `git_head` `dbd8a511…`, `git_dirty` false, and `script_sha256` `3d0e6ecd…`, and each `scope`
begins `REHEARSAL build, not A3 v2:`.

## Deviations from the prompt

1. **An interrupter launched at the wrong directory** — `UNINTENDED DRIFT`.
   - **What happened.** For `resumed-b`, the agent's loop waiting for the run directory used a zsh
     glob qualifier, `(N)`, that the shell did not honour, so it never matched. After 90 s it
     launched `interrupt_when.py` (pid 52113, 01:53:43) with an empty run directory.
   - **What it did.** It read the repository root, logged "the run's manifest names no --register
     slug; exiting without signalling", and exited. It signalled nothing.
   - **What it left.** It wrote `interrupter.log` into the repository root, an untracked file,
     created after `resumed-b`'s manifest (01:52:03), whose `git_dirty: false` stands. The file
     was moved to `var/a3-v2-readiness-07/stray-interrupter-resumed-b.log` and is reproduced in
     `interrupter_firings.txt`.
   - **The correct interrupter** started at 01:54:12, with the run in `APPLY GKSOURCE POLICIES`,
     before the stage it watched, so the interruption point is unaffected. Later waits used
     `find`.
2. **`--interval 5` on both interrupters** — `IMPLEMENTATION CHOICE`. The default is 15 s. The
   reference's `GkSource` stage took 62 s, so the agent chose `--hold 15` with a 5 s interval to
   put the signal 15–20 s into it. It landed 15 s in, with 102 of 291 rows stored.
3. **`stage_counts.py` could not be used as it stands** — `STRUCTURALLY REQUIRED`. Its `main`
   labels columns by splitting the run id on `-10-`, and raised `IndexError` on these runs. The
   prompt says to use the tools as they stand, and §7 forbids changing them. `stage_counts_07.py`
   imports its `read` unchanged and prints the same table. Opened as
   `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]` (observation 1).
4. **The first reader pass's mtime column was broken** — `UNINTENDED DRIFT`.
   - **The cause.** `run_readers.sh` as first written called `stat -f '%Sm %z %N'`. The GNU `stat`
     first on this machine's `PATH` reads `-f` as "file system status", so the column held
     free-block counts, which change.
   - **What it printed.** `STORE FILES CHANGED` after every one of the six readers. That is false:
     the same output's SHA-256 column is identical before and after each reader, and every file's
     mtime (02:04:28, 02:09:10 and 01:44:50) predates the pass (02:12:54).
   - **Since.** The script calls `/usr/bin/stat`. The wrapper run and the run without `--models`
     used the corrected script, and show SHA-256, mtime and size identical. The first pass's output
     is kept in `readers.txt` under a header saying this.
5. **The `QuadSourceIntegral` extractor does not read all 864 rows, and cannot as written** —
   `STRUCTURALLY REQUIRED`.
   - **Why.** It draws 2 of 3 response k and 3 of 6 (q, r) pairs at random, and reads the rows of
     those 6 triples.
   - **What it read.** Every stored row of each: 96 × 3 = 288. The other three triples drawn hold
     no rows, being outside the triangle filter.
   - **What this means.** Prompt 01's point holds, since the extractor at its default pair finds
     every row it asks for. "Its count … must equal the store's" cannot hold without changing or
     wrapping the script, which the prompt did not ask for. Recorded, not opened: the subsample is
     the script's design (record §2.3.3).
6. **`extract_tensor_source_data.py` exits 1: acceptance 4 is not met for this reader** —
   `STRUCTURALLY REQUIRED`, a §5 stop.
   - **The cause.** It draws its three k in random order and pairs them, so pairs with q > r ask
     for `QuadSource` rows the store does not hold (q ≤ r only). A labels-only miss returns an
     unpopulated object, and the remote plot dies on `values has not yet been populated`.
   - **It predates the campaign.** On 2026-10-01 the same script passed LambdaCDM on a 1-in-6
     draw.
   - **The user's decision:** record it and exercise the rest through
     `tensor_source_sorted_sample.py`, which is not the shipped reader. That wrapper exited 0, read
     all six `QuadSource` rows, refused nothing, and left the store byte-identical. Opened as
     `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]` (record §2.3.2).
7. **"No `duplicate` after each resumed launch" read through the fingerprint for `resumed-a` and
   `-b`** — `IMPLEMENTATION CHOICE`. `read_inventory` was run directly after `resumed-c`. For the two
   killed launches the evidence is each finish's fingerprint. It is built from `read_inventory`,
   and its `problems` tallies every problem kind by class: `{}` for both.
8. **Black reformatted three scripts after their outputs were taken** — `IMPLEMENTATION CHOICE`.
   `closed_store_state.py`, `prune_serials.py` and `store_soundness.py` were written, run, then
   formatted. The formatted versions were re-run read-only and gave output identical to the saved
   files, and both stores' hashes were unchanged.
9. **The pre-launch Ray listing went beyond §1.3's `pgrep`** — `IMPLEMENTATION CHOICE`. It also
   gives each process's parent and searches the whole process table for `SecondaryGWKit/venv`
   together with `ray`. It was this second search that found the runtime-env agent `pgrep`'s
   pattern does not match (observation 4).
10. **The closed-store state was also recorded after `reference`** — `IMPLEMENTATION CHOICE`. §1.5
    item 4 says "between launches"; the reference store's state is the baseline the others are read
    against.

## Verification performed

**Dummy tests (§3), at `HEAD`, before the first launch.**
- `interrupter_dummy_test.sh`, into a new scratch directory: all five cases as recorded.
  - A signalled, and the child exited 143.
  - B exited silently.
  - C refused with exit 2, with the new wording, and its child lived.
  - D held, then signalled.
  - E did not signal.
- `interrupter_build_dummy_test.sh`: A signalled the `quadsource_atol_sweep.py` child. B refused
  the `scoped_pipeline_run.py` child, exit 2. C, with no `argv[0]`, exited 2.
- No `sleep` child was left.

**Both real firings.**
- **`resumed-a`.** The banner was seen at 01:50:53. After the hold, `sent SIGTERM to pid 41825` at
  01:51:08. The handler's line reads `** SIGTERM at 2026-10-04 01:51:08 +0100: recorded killed,
  exit code 143; …`.
- **`resumed-b`.** At the 100 % line, `sent SIGTERM to pid 49456` at 02:00:06. The handler's line
  reads `** SIGTERM at 2026-10-04 02:00:06 +0100: …`.
- **The delay.** Both are whole-second stamps, equal within each pair, so the delay is bounded below
  1 s, not measured.
- **Neither killed launch** printed a traceback or Ray's `*** SIGTERM received` banner.

**§2.2's tables.** Lookup / compute / store per stage:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT, SOURCE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT, RESPONSE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| Tk NUMERICAL | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| Tk WKB | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| Gk NUMERICAL | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| Gk WKB | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | 224/224/224 | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | 864/393/343 (last progress) | 471/471/471 |

- **`GkSource`.** `resumed-a` stored 102 rows, 35 of them unvalidated. `resumed-b` pruned the 35 and
  computed 224 = 291 − 102 + 35.
- **`QuadSourceIntegral`.** `resumed-b` stored 393. Its last progress line has 471 computes in
  flight, and `resumed-c` computed 471: 393 + 471 = 864.
- **The 35 unvalidated serials.** Shard 0: 11–15, 22, 28–31, 33, 34. Shard 1: 214–219, 228.
  Shard 3: 522, 523, 528–531, 539–545, 547–549. After the next open, 32 are absent, and 547–549
  are reused by validated rows `resumed-b` stored at 01:52:33–40.
- **The integrity warnings.** `resumed-b` printed them: shard 1's seven in full, and 28 entries
  folded by Ray as `[repeated 28x across cluster]`.
- **The version row** is `(1, '2025.1.1')` on all eight shards.
- **The comparison.** `compare_stores.py` exits 0, with 0 differences, both at format 6, digest
  `39809dca…`, each equal to its sidecar.
  - Every class count equals `datastore-integrity` 10's but `tolerance`: 7 against 6. The
    quadrature `rtol` of `1e-7` no longer coincides with `DEFAULT_GK_NUMERIC_REL_TOLERANCE =
    1e-8`. Not a defect.
- **Integrity.** The census finds no divergence on either store. Its list is `replicated_tables`
  plus `BackgroundModel_tags`. `integrity_check` is `ok` and `foreign_key_check` is empty on all
  ten files. `read_inventory` reports no problem in either store.

**L1, besides the states.**
- **The manifests** are as §2.1 asks. Each `stdout.log` shows four `replaced 1 occurrence` lines,
  the rehearsal scope line, `--zend 250`, no `--no-prune-unvalidated`, and the local Ray
  bootstrap.
- **The guard** refused, exit 1, naming the five files, before Ray and the registry. The files'
  SHA-256 and the sidecar's were unchanged.
- **No Ray process of this repository** was left after `reference`, `resumed-a` or `resumed-c`.
  After `resumed-b`, the dashboard agent and runtime-env agent were left (observation 2).

**L3.**
- Six readers, `--ray-address local --models LambdaCDM`. Exits 0, 0, 0, 0, **1**
  (`extract_tensor_source_data.py`) and 0.
- `extract_Gk_data.py` without `--models` printed `>> SKIPPING MODEL QCD_Cosmology: …` and exited 0.
- The sorted-sample wrapper exited 0 with 6 `QuadSource` reads.
- None refused anything. The store's SHA-256 was identical across every run, and no journal
  appeared.
- `store fingerprint` afterwards matches the sidecar.

**Black.** `./venv/bin/python -m black --check docs/a3-v2-readiness/rehearsal/*.py`: 5 files would
be left unchanged.

**The six suites**, serially, on the staged tree, with README §7's command, beside the pre-dispatch
baselines (measured by the orchestrator at `68b5f4b`):

| Suite | Pre-dispatch baseline (`68b5f4b`) | This prompt, staged tree |
|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK |
| ComputeTargets | 686 OK | 686 run; 2 failures, `test_background_tau.TestBackgroundTau.test_build_cost` (0.620 s against 0.5 s) and `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.092 s against 0.06 s); the other 684 pass |
| CosmologyModels | 39 OK | 39 OK |
| Datastore | 598 OK | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) |
| RunRegistry | 281 OK | 281 OK |

- **When.** Measured 02:39–02:54, serially, each suite's output written to
  `var/a3-v2-readiness-07/suites/<suite>.txt` and its verdict grepped from it.
- **The load.** The one-minute load average was 129 at the start, 24 when ComputeTargets finished,
  and 12 at the end. Another of the user's sessions was running in a worktree.
- **The two failures are the dispatch brief's load-sensitive tests.** Each was re-run alone at a
  load of 13:
  - `test_build_cost` passed;
  - `test_wall_time_per_object` failed (0.132 s).
- **On an export of the parent commit `dbd8a51`**, with the venv linked: `test_wall_time_per_object`
  failed (0.084 s against 0.06 s) and `test_build_cost` passed.
- **Neither is this prompt's.** This commit changes nothing under `ComputeTargets/`, or any code a
  suite imports. Every count equals the baseline.
- **The markdown was written after the suites.** The log, the board, the index and the record were
  finished after the suites ran; the files the suites could import were the same.

## The deliberate-breakage record

None. The prompt names no mutations: it changes no code and adds no test (README §5 rule 8 applies
to prompts that change code).

## Observations not acted on

1. **`stage_counts.py` cannot label a run of any prompt but 10.** `main` takes each column's label
   from `run.name.split("-10-", 1)[1]`, which raises `IndexError` on any other run id. It is a
   verification tool of `datastore-integrity`'s, which §7 does not allow this prompt to change.
   Opened `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]`.
2. **A killed build's Ray agents outlive the driver** (record §2.1.3).
   - **What was seen.** After `resumed-b`, pid 49729 (`ray/dashboard/agent.py`) and 49730
     (`ray/_private/runtime_env/agent/main.py`), both from this checkout's venv, were orphaned to
     pid 1 in the driver's process group 49456. They were alive 40 s after the driver exited.
     49730 had gone by 02:01:30. 49729 was alive at 02:02:05 and gone by 02:02:36 (the
     orchestrator's bounded wait).
   - **The stop and the decision.** It is a §5 stop; the user decided to wait for it, and it
     exited on its own. Nothing was killed.
   - **Not seen elsewhere.** `reference` and `resumed-a` showed none at checks about 60 s and
     30–40 s after exit.
   - **What it means for A3 v2.** "No Ray process left" is not true at the moment the driver exits,
     so A3 v2's check after a kill must wait for these by pid. Opened
     `[07-a-killed-builds-ray-agents-outlive-the-driver]`.
3. **`killed` is recorded before the fingerprint.** The orchestrator's first read of `resumed-b`
   (02:00:06) saw `killed`/143 with no `fingerprint`, while the pid was alive. That is prompt 04's
   design. A wait that ends on the recorded state ends before the fingerprint and before `ray.shutdown()`;
   it must end on the pid. This is in the record's §2.4 checklist.
4. **`pgrep -f 'gcs_server|raylet|ray::'` misses a runtime-env agent.** It listed the dashboard
   agent 49729 only because that command line holds `--raylet-name=…`. 49730's command line names
   none of the three. The agent's whole-table search for `SecondaryGWKit/venv` and `ray` found
   both. This is part of the issue in observation 2.
5. **Another of the user's sessions ran in the worktree `SecondaryGWKit-wt-replicated-key`**
   throughout §2.3, and a search for `SecondaryGWKit` matches its processes. Worktrees share this
   checkout's venv by a symlink, so a Ray process started from one may show either path. The
   one-minute load reached 365. Not touched.
6. **`resumed-b`'s handler saw `SystemExit(1)`, `resumed-a`'s `SystemExit(143)`.** The second signal
   arrived in `ray.wait`, where Ray's signal check converts the handler's exit (audit S1). The latch
   records `killed`/143 and the exit is 143 either way. That is what prompt 04's latch is for. Not a
   defect.
7. **A labels-only `QuadSource` miss returns an unpopulated object silently.** It does not report
   absence, so the tensor-source reader's failure surfaces as `values has not yet been populated`
   and names neither q nor r. This is recorded inside
   `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`, not separately.
   The reader's draw order is the defect, and the silent miss is why it is hard to read.
8. **The readers draw random subsamples**, so two runs of the same reader read different objects.
   `extract_Gk_data.py` wrote 414 files with `--models` and 351 without. A read-back is a sample
   of the store, not all of it (deviation 5).
9. **Shard 2 of each store holds only replicated rows** (1 077 248 bytes). With three k, the
   sharded classes fall on shards 0, 1 and 3. No defect; A3 v2's eight k will spread further.
10. **The `QuadSourceIntegral` stage's first progress line came at about 5 minutes** in every launch
    that reached it, and read 100.00 % (2 work items). Only `resumed-b`'s interrupter could use it.
    Noted for A3 v2: its progress lines will be the 120 s notifications of a larger list.

## State handed to the next prompt

07 is the campaign's last prompt. For the orchestrator's review and the user:
- **The stores** stay in `var/datastores/`: `a3-v2-readiness-07-reference.sqlite` (store_id
  `b7ad6350…`) and `a3-v2-readiness-07-resumed.sqlite` (store_id `b880a3a3…`), each with four
  shards and a sidecar whose fingerprint is `39809dca…`. `store retire` is the user's (D7-2).
- **The runs** are under `var/runs/a3-v2-readiness-07-*`, with launch logs beside them.
  `var/a3-v2-readiness-07/` holds the pre-launch captures, the states, the check outputs, the
  readers' outputs and the agent's notes.
- **Issues opened** on the board's §3.2:
  - `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`, which **needs a
    fix before A3 v2's read-back**;
  - `[07-a-killed-builds-ray-agents-outlive-the-driver]`;
  - `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]`.
  - Index: **105 open** (102 + 3).
- **A3 v2's launch checklist** is the record's §2.4.
- **The SHA** of this commit is `a6c8f39`, filled in by the orchestrator here and on the board.
