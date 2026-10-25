# Log — py313-venv prompt 02, move the Mac Studio's venv to Python 3.13, and read A3 v2 under it

- **Subject:** [`02-move-the-mac-studio-venv-to-python-3.13.md`](../02-move-the-mac-studio-venv-to-python-3.13.md),
  board items M1, M2, R7 and A1. It ran in two dispatches, both on the Mac Studio (`mps026298`) in
  the main checkout, on `handover-remedial`.
  - **The first dispatch**, at `aac52bc`, 2026-10-25, 16:18–16:36 GMT. It made steps 1–6 and
    stopped at step 8: R7's first `reference` failed at `ray.init` with `InvalidAuthToken`. That
    became `[02-the-token-mode-setting-comes-after-import-ray-so-every-ray-start-fails]` (board
    §4), fixed on the laptop in `14ada2f`. The orchestrator released the lock at 16:36:46 on the
    user's ruling.
  - **The resume**, at the resume SHA **`c9026af`**, 2026-10-25, 16:44–17:11 GMT, under the
    prompt's §7. It made §7 steps 1–3, then steps 7–11.
- **Commit:** none on the Mac Studio (prompt §4). The orchestrator copies these records to the
  laptop and makes the one commit (prompt §6).
- **Result:** **complete. Acceptance 1–7 hold, 5a included.**
  - R7's two stores are both `39809dca…`. They equal each other, their sidecars, and R3's recorded
    `reference` and `resumed-c`.
  - A3 v2 reads under 3.13. All seven reader runs exit 0, every store file is byte-identical, and
    its fingerprint, computed under 3.13, is `0e32f24a…`.
  - The token mode works: all five token checks hold.
- **Model:** Opus 5.5.
- The first dispatch's own stop log is kept, as written, at `state/log_first_dispatch_as_written.md`.

## What shipped

Nothing tracked. Everything is under `var/`.
- **The venv** (first dispatch, steps 3–4):
  - Python **3.13.16**; `ray` 2.59.0, `SQLAlchemy` 2.0.55, SQLite 3.53.4.
  - Its freeze is identical to the laptop's last freeze (`diff` exit 0).
  - `pip check` is clean, and the provenance equals the laptop's 3.13 record.
  - The 3.12.15 venv was moved, not deleted, to
    `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025` (817 MB).
- **The token** (first dispatch, step 4): made by `ray get-auth-token --generate > /dev/null` before
  any Ray start, then `chmod 600`. It is `~/.ray/auth_token`, `-rw-------`, 64 bytes, and was never
  printed.
- **R7** (resume):
  - four runs, `py313-venv-02-{reference-20261025T164631, resumed-a-20261025T165114,
    resumed-b-20261025T165402, resumed-c-20261025T165549}`;
  - their four `-resume` launch logs;
  - two stores with sidecars, `var/datastores/py313-venv-02-{reference,resumed}.sqlite`
    (store_ids `c85d2c25…`, `55c9d142…`). Retiring them is the orchestrator's.
- **The first dispatch's failed run** stays as a record: `var/runs/py313-venv-02-reference-20261025T163202`
  (`failed`/1), with its launch log `var/runs/py313-venv-02-reference.launch.log`.
- **Reader output:**
  - `var/py313-venv-02/extract/`, A3 v2's, step 9 (891 MB);
  - `var/py313-venv-02/extract-r7/`, L3's on R7's resumed store.
- **Records:**
  - `out/` (every saved output; the resume's blocks under `== resume (c9026af)` headers in the files
    that existed, and new files for the rest);
  - `state/`, `scripts/` (the first dispatch's helpers, untouched, plus `*_resume` copies);
  - this log, `verification.md` and `issues.md`.

## Deviations from the prompt

**First dispatch** (as its stop log recorded them):
1. The waits were the orchestrator's: one check per launch, then the turn ended. —
   **STRUCTURALLY REQUIRED**.
2. Every lock command ran as a shell command of its own. The header was zsh's builtin `print -P`,
   and it records the temp dir as `<the temp dir>`. — **STRUCTURALLY REQUIRED**.
3. The dummy tests' work dir was under `var/py313-venv-02/work/`, as R4's was. —
   **IMPLEMENTATION CHOICE**.
4. Measurements beyond the prompt's — **IMPLEMENTATION CHOICE**:
   - the `ps -E` probe and the auth-mode import-order probe, neither of which starts Ray or prints
     the token;
   - the 3.12 freeze against the laptop's 3.12 freeze;
   - warning counts per root.
5. Two record slips, both corrected in place. Two `pip_02.txt` diff commands broke on a
   parenthesised comment and were re-run. The `ps -E` probe's first run read the wrong field and
   was re-run. — **UNINTENDED DRIFT**.
6. A second `ray-processes` call came during `reference`'s check, because the first came before
   the raylet was up. — **IMPLEMENTATION CHOICE**.

**Resume:**

7. **The waits were the orchestrator's** (the brief, and its note's §1). After each of the four
   launches I made one check and ended my turn. I reported the driver's Python pid, the `sh` pid,
   the run id, the interrupter's pid and the lock's owner file. The orchestrator waited on the pid
   in the background and resumed me with the final `status.json`. After each `SIGTERM`ed launch I
   listed Ray processes by name (`ps -Ao pid,ppid,command | grep -E "gcs_server|raylet"`). None was
   left, so there were no extra pids to wait on. I never waited on a run, never sleep-looped or
   polled, and never ran `pgrep -f`. Step 8's "one background wait on the driver's pid" was
   therefore the orchestrator's, not mine. — **STRUCTURALLY REQUIRED**: a subagent's turn is not
   resumed by its own background command.
8. **Each lock command was a shell command of its own** and the only process it started. Output went
   by `>> out/lock_02.txt 2>&1`, with `exit=$?` recorded by the builtin `echo` and the header by
   zsh's builtin `print -P`, which names the temp dir as `<the temp dir>`. There was no pipe, and no
   `--purpose` names the temp dir. — **STRUCTURALLY REQUIRED** (`CLAUDE.md`, the brief).
9. **The interrupter was started by a separate command, `scripts/start_interrupter_resume.sh`,
   right after the launch**, with no loop. The first dispatch's `launch_with_interrupter.sh` waited
   up to 20 s, `sleep 1` at a time, for the run directory. The brief forbids any sleep-loop. Both
   interrupters started 3 s after their launches, and both fired at their holds. —
   **STRUCTURALLY REQUIRED**.
10. **The helpers ran as `*_resume` copies** in `scripts/`:
    - the dispatch SHA is `c9026af`;
    - the launch logs are `-resume`;
    - every run-directory glob excludes the failed `…-reference-20261025T163202`, which the first
      dispatch's `head -1` would have picked;
    - `log_comparison_resume.py` also searches `InvalidAuthToken|AuthenticationError`.

    The originals are untouched. — **IMPLEMENTATION CHOICE**.
11. **I fast-forwarded with `git merge --ff-only c9026af`** before running §7 step 1's own command,
    `git fetch origin && git merge --ff-only origin/handover-remedial`. The prompt's command then
    said `Already up to date`, with `HEAD` = `origin/handover-remedial` = `c9026af`. The tree is the
    same either way. — **UNINTENDED DRIFT**, the agent's, recorded in `out/host_02.txt`.
12. **The first `stage_counts.py` call ran in zsh.** zsh did not word-split an unquoted `$RUNS`, so
    the call raised `FileNotFoundError`. I re-ran it through `bash` at once (rc 0) and overwrote its
    file. This is 01's deviation 4 again. — **UNINTENDED DRIFT**, the agent's. It read nothing
    twice that mattered and wrote nothing else.
13. **The interrupter dummy tests were not run again in the resume.** The first dispatch's runs
    (16:30:57–16:31:23, both rc 0, equal to R4's when normalised) stand: the venv is the same, and
    `git log aac52bc..c9026af` touches neither test script nor `interrupt_when.py`. —
    **IMPLEMENTATION CHOICE**.
14. **Where L3 and step 9 were written.** L3's readers, on R7's resumed store, wrote to
    `extract-r7/` and `out/readers_r7_02.txt`. That leaves `extract/` and `out/readers_02.txt` to
    step 9, which names them. — **IMPLEMENTATION CHOICE**.
15. **Measurements beyond the prompt's** — **IMPLEMENTATION CHOICE**:
    - the raylet's `RAY_AUTH_MODE` beside the `gcs_server`'s, with the same one-variable filter;
    - A3 v2's guard on R7's resumed store, as R4 and R6 ran it;
    - the reader-log scan against handover 10's (`out/reader_logs_scan_02.txt`);
    - `var/datastores/` before and after (`out/var_datastores_02.txt`);
    - L3's warning and token counts.

## Verification performed

**The checkout (acceptance 1).**
- The first dispatch was at `aac52bc` and the resume at `c9026af`, which is `origin/handover-remedial`.
- `git status --short` was empty at every pre-launch check, after the last launch, and at the end
  (`out/run_records_02.txt`).
- The Gate's `git diff --stat b90752d c9026af -- ObjectFactories config main.py` lists
  `ObjectFactories/tests/test_quadsource_parent_key.py`, `config/ray_local.py` and `main.py`, and
  nothing else.

**The venv (acceptance 2).**
- 3.13.16. The freeze is identical to the laptop's last freeze, and differs from this host's 3.12
  freeze in exactly `ray` and `SQLAlchemy`. This host's 3.12 freeze equals the laptop's 3.12 freeze.
- `pip check` is clean, and the old venv was moved (above).

**The roots (acceptance 3)**, serially, in the foreground (`out/roots_02.txt`):

| Root | 3.12 (aac52bc) | 3.13 (aac52bc) | 3.13 (c9026af, the resume) |
|---|---|---|---|
| AdaptiveLevin | 32 OK, 1 s | 32 OK, 1 s | — |
| ComputeTargets | 737 OK, 138 s | 737 OK, 146 s | — |
| CosmologyModels | 39 OK, 1 s | 39 OK, 3 s | — |
| LiouvilleGreen | 148 OK (skipped=1), 25 s | 148 OK (skipped=1), 26 s | — |
| ObjectFactories | 280 OK, 14 s | 280 OK, 15 s | — |
| RunDrivers | 124 OK (skipped=1), 6 s | 124 OK (skipped=1), 7 s | **127 OK (skipped=1)**, 9 s |
| tools | 49 OK, 10 s | 49 OK, 13 s | — |

- Every count is 01's under 3.13, but RunDrivers. That root is 124 at `aac52bc`, by
  `d833ee7`'s 4 tests, and **127 at `c9026af`**, by `14ada2f`'s `TestRaySeesTheMode` (3 tests).
- No known flake fired.
- `test_ray_auth_mode` on its own: 7 OK.
- Warnings by kind are equal under 3.12 and 3.13. ObjectFactories has 0 `ResourceWarning` here,
  where the laptop's 3.13 had 1, because `28f94c7` disposed of the test's engine.

**R7 (acceptance 4).**

| Slug | Run id | Driver (sh) | State | Exit | Wall | Interrupter |
|---|---|---|---|---|---|---|
| `reference` | `…-reference-20261025T164631` | 929 (922) | `done` | 0 | 3 m 32 s | — |
| `resumed-a` | `…-resumed-a-20261025T165114` | 3417 (3412) | `killed` | 143 | 1 m 50 s | 3468, hold 10 s, `SIGTERM` 16:53:01 |
| `resumed-b` | `…-resumed-b-20261025T165402` | 5005 (5000) | `killed` | 143 | 1 m 13 s | 5057, hold 27 s, `SIGTERM` 16:55:11 |
| `resumed-c` | `…-resumed-c-20261025T165549` | 6029 (6024) | `done` | 0 | 1 m 23 s | — |

- **The points**, chosen from R7's own `reference` by R3's and R4's rule:
  - `GkSource` took 22 s; 40 % is 8.8 s, under the 10 s floor, so the hold was 10 s (R4's own);
  - QSI took 79.8 s; a third is 26.6 s, so the hold was 27 s (R4 held 30 s).
- **Both stores are `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`**: format 6,
  `encoding_format` 1, 21 classes, problems none.
  - The `reference` store's fingerprint was written into its sidecar by `store fingerprint
    --write`.
  - `resumed-b` and `resumed-c` wrote their own (`fingerprint_sidecar` `written`).
  - `compare_stores.py`: `compare_fingerprints(reference, resumed): 0 difference(s)`, each with 0
    against its sidecar, verdict `equal`.
  - `compare_with_recorded.py` against R3's two run directories under `var/handover-06/recorded/`:
    0 differences each, verdict `equal`.
- **The resumed chain:**
  - `resumed-a` stored `GkSource` 166 (57 + 55 + 0 + 54), 49 of them unvalidated (16 + 17 + 0 + 16).
  - `resumed-b` pruned those 49 at open: `prune_check` gave "49 absent, 0 reused, 0 still
    unvalidated". It computed **174 = 291 − 166 + 49**, then stored QSI 332.
  - `resumed-c` computed **532 = 864 − 332**.
  - Every finished stage computed 0 on the next launch.
- **Integrity, read-only:** the census found no divergence, `store_soundness.py` gave `=> clean`,
  there is no journal and no in-flight row, and each shard has one version row. The store files
  were identical before and after the checks.
- **The manifests.** Every one has `host.python` 3.13.16 and the seven-distribution `packages`,
  equal as JSON to the record. Each has `git_head` `c9026af…`, **`git_dirty` false**, the launcher
  `5b984c75…`, and 20 keys as R3's. Every status's host id and boot are this machine's.
- **The lock saw the 2.59.0 instance.** During `reference`'s check, `ray-processes` exited 1 with
  18 lines, `gcs_server` 960 and `raylet` 1008 among them.
- **No Ray process outlived a driver:**
  - `reference`'s were gone by 16:50:32;
  - `resumed-a`'s were gone ≤ 31 s after its signal, and `resumed-b`'s ≤ 19 s after;
  - `resumed-c`'s were gone by 16:57:28.
  - `ray-processes` exited 0 after each.
- **L3, the readers on R7's resumed store** (`out/readers_r7_02.txt`):
  - all seven runs exit 0 with R3's lookups: 116, 348, 10, 3, 6, 24, and 116 with the
    `QCD_Cosmology` skip;
  - the store files are identical after each, and 0 Ray processes are left;
  - files written: 441, 2436, 180, 27, 27, 73 and 360. The counts differ from R3's and R6's only
    where a reader draws an unseeded subsample.
  - `store fingerprint` afterwards matches `resumed-c`'s `39809dca…`.
- **The guard** (`out/guard_02.txt`): exit 1 within a second, with the clean refusal. The store files
  and `var/runs/` (20 entries) are unchanged.
- **L4 against R4** (`out/log_comparison_02.txt`, `out/stage_counts_02.txt`,
  `out/policy_statistics_02_*.txt`):
  - the `reference` stage counts equal R4's;
  - the statistics block is byte-identical to R4's for `reference` and `resumed-c`, and equal when
    sorted for `resumed-b`: 291 instances; numeric 109; mixed 26 (25 complete, 1 minimal); WKB 156;
  - in all four runs the `** ` sequence and the stage banners equal R4's (13/13, 11/11);
  - the printed-line counts equal R4's in all four;
  - the stage times are within about 10 % of R4's (recorded, not compared).

**A3 v2 (acceptance 5).**
- §7 step 3 (16:45): A3 v2's five files and its sidecar are unchanged against step 6.
- **The readers** (`out/readers_02.txt`), against handover 10's in brackets:

  | Reader | Exit | Wall | Lookups (last queue) | Files |
  |---|---|---|---|---|
  | `extract_Gk_data` | 0 | 60 s (54) | 522 (522) | 1791 (1251) |
  | `extract_GkWKB_data` | 0 | 145 s (171) | 1392 (1392) | 9744 (9744) |
  | `extract_GkSource_data` | 0 | 41 s (34) | 45 (45) | 877 (847) |
  | `extract_TkWKB_data` | 0 | 23 s (19) | 8 (8) | 72 (72) |
  | `extract_tensor_source_data` | 0 | 25 s (22) | 36 (36) | 172 (172) |
  | `extract_QuadSourceIntegral_data` | 0 | 34 s (31) | 240 (240) | 691 (687) |
  | `extract_Gk_data`, `nomodels` | 0 | 58 s (58) | 522 (522) | 1458 (1980) |

  - The output totals 891 MB against handover 10's 879 MB.
  - The file counts differ only in the subsampling readers (`[10-the-readers-draw-their-work-lists-unseeded]`).
  - The store files are identical after each reader, and 0 Ray processes are left.
- **Step 10:**
  - `state/store_files_after.txt` equals `state/store_files_before.txt` (`diff` exit 0, and again
    after the fingerprint);
  - **`store fingerprint`, without `--write`, under 3.13 gives
    `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`**, problems none, "matches
    the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752d…";
  - no `gcs_server` or `raylet` is running.
- **No reader log holds the token-mode warning** (`out/reader_logs_scan_02.txt`), nor any
  `InvalidAuthToken` or `AuthenticationError`.

**Token mode (acceptance 5a)** (`out/token_02.txt`):
1. **No R7 log holds `Token authentication is enabled`.** That covers the four resumed launches'
   twelve logs (launch log, `stdout.log`, `stderr.log`), and every reader log of L3 and step 9.
   R6's all did.
2. **None holds `Generated new authentication token`.** That line appears only in step 4's own
   record of the generation, `out/token_02.txt`, which is not an R7 log.
3. **During `reference`'s one check, the `gcs_server`'s (960) exec-time environment holds
   `RAY_AUTH_MODE=token`.** I used `ps -E -ww -p 960 -o command= | tr ' ' '\n' | grep '^RAY_AUTH_MODE='`.
   The raylet (1008) shows the same.
4. **`~/.ray/auth_token` is still `-rw-------`, 64 bytes**, after the last launch and at the end.
5. **No R7 log holds `InvalidAuthToken` or `AuthenticationError`.** The twelve logs give 0 each.
   The first dispatch's failed run has 22 such lines; it is the record of the fixed defect, not one
   of R7's launches.

**The lock (acceptance 6)** (`out/lock_02.txt`):
- The first dispatch acquired it at 16:31:52, and the orchestrator released it at 16:36:46.
- The resume ran `status` free at 16:46:12 and `acquire` exit 0 at 16:46:19.
- It ran `set-pid` 929, 3417, 5005 and 6029, each exit 0.
- `release` exited 0 at 17:10:19. Afterwards `status` said `lock: free`, and `ray-processes`
  exited 0 with nothing listed. `~/.raypoolkit/` is empty, and `~/ScienceData/.rehearsal-lock`
  does not exist.

**The registry (step 11):** 10 runs, "10 finished; 0 with no host recorded". R7's four are among
them, with the first dispatch's `failed` record. Nothing is `running` or stale.

**`var/datastores/`:** the 8 files present before the resume are unchanged in SHA-256, size and
`mtime_ns`. The 12 new files are R7's two stores and sidecars. There is no journal.

**The records (acceptance 7):** `out/`, `state/`, this log, `verification.md` and `issues.md`.

## What is new in the logs

Against R4's same slug (raw logs, on this machine), **no line kind is new in any R7 log**
(`out/log_comparison_02.txt` §2). The count differences, each explained:
1. **The token-mode lines of R6 are gone.** There is no `Token authentication is enabled`, no
   `Generated new …` and no authentication error, against R6's warning in every launch log. This
   is `d833ee7` with `14ada2f`'s refresh: a mode set by name makes Ray log nothing. R4, under
   2.43.0, had none either.
2. **`SystemExit(1)` in `resumed-b`, where R4 had `SystemExit(143)`.** It depends on whether the
   signal lands inside `ray.wait` (R2 §3.2.2). The record is `killed`/143 either way, and
   `resumed-a`'s `SystemExit(1)` is R4's.
3. **`resumed-b`'s `stdout.log` and launch log have 16 more lines** (145 against 129, and 154
   against 138). They are the `INTEGRITY WARNING`'s per-row listing of unvalidated `GkSource` rows
   (`-- "<row>" … contains N z-sample values | expected=N`), 16 lines of each kind against R4's 8.
   The listing's length follows how many rows `resumed-a`'s kill left unvalidated (49 against
   R4's 43).
4. **One "R4-only kind" in `resumed-c`**, `ALL WORK ITEMS COMPLETE in time Nm Ns`. R7's QSI took
   60.4 s and printed as seconds, where R4's took 1 m 6.87 s. It is formatting.
5. **The `IntegrationWarning` is in `reference` and `resumed-b`**, as at R4 and R6.

The A3 v2 reader logs hold no new kind against handover 10's. `extract_GkSource_data` has 4
`MatplotlibDeprecationWarning`, as at 10. 10's 2 `RuntimeWarning` in `extract_GkWKB_data` are absent
here, which follows its unseeded subsample.

## Observations not acted on

- **The partial-store fingerprints** are `resumed-a` `f62d7710…` and `resumed-b` `a51f9657…`. They
  differ from R3's, R4's and R6's, as they must: each digests a store killed at a different point.
- **The load** was 5–11 during R7 and rose to 27–46 during the readers. The stage times were close
  to R4's.
- **`resumed-b` listed two `INTEGRITY WARNING`s for three shards with unvalidated rows** (0, 1 and 3).
  That is the count R3 and R6 had. It is not new, and I have not looked into it.
- **No issue is opened by the resume.** The one the first dispatch opened is closed (`issues.md`).

## State handed on

- **To the queued analysis jobs:** the Mac Studio's venv on 3.13.16, `ray` 2.59.0 and `SQLAlchemy`
  2.0.55, token mode on with the token at mode 600, and A3 v2 shown to read the same under it.
- **The venv** is 3.13.16. The 3.12 venv is at
  `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`, and retiring it is the user's.
- **Under `var/`:**
  - R7's four run directories and four `-resume` launch logs;
  - the first dispatch's failed run and its launch log;
  - the stores `py313-venv-02-{reference,resumed}.sqlite` with their sidecars, for the orchestrator
    to retire after the review;
  - `var/py313-venv-02/`.
- **The lock** is free, and no Ray process is left. Nothing is `running` or stale. `git status
  --short` is empty at `c9026af`.
