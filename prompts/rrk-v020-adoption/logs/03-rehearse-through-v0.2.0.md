# Log — rrk-v020-adoption prompt 03, rehearse through `v0.2.0` (R3), and the hard-kill route

- **Subject:** [`03-rehearse-through-v0.2.0.md`](../03-rehearse-through-v0.2.0.md), board items L1–L4 (R3), H1 and
  H2; gates R3 and HK.
- **Commit:** this commit (parent `aab4075`).
- **Result:** landed, with one stop.
  - **R3 holds.**
  - **The hard-kill route (H1) holds.** `[a3-v2-a-hard-killed-run-blocks-its-store]` closes.
  - **H2 does not hold as stated.** Ray's GCS server outlived the `SIGKILL`ed driver past ten
    minutes. The prompt's stop fired, the user ruled, and the orchestrator ended the process. It is
    opened as `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`.
  - Every other acceptance item passes.
- **Model:** Opus 5.5.
- **Date:** 2026-10-19 (dispatched 2026-10-18).

**What R3 shows:**
- **The four launches.** Both stores are `39809dca…`, format 6, **`encoding_format` 1**, all 21
  classes. They are equal to each other, to their sidecars, and to the fingerprints recorded by R3's
  own `reference` and `resumed-c` and by **R2's** `reference` and `resumed-c`. The pin to
  `runregistrykit` 0.2.0 is invisible to a build: the one log line that names it is the launcher's
  banner, as at R2. L1–L4 hold against R2.
- **The hard-kill pair**, on a third store:
  - `hardkill-a` was `SIGKILL`ed in the QSI stage, its status left `running`;
  - `list` marked it stale `!!`;
  - the resume was refused at `begin`, with nothing created and nothing changed;
  - `end` recorded it `lost`, verified, and a second `end` was refused;
  - the sidecar was made;
  - `hardkill-b` ended `done`, with its fingerprint `39809dca…` recorded and the sidecar `written`.
- **Ray after the `SIGKILL`.** Everything but the GCS server went within 20 s. The GCS server sat
  in its own shutdown until a `SIGKILL` at 00:10:05, by the user's ruling.

**Where it ran.** Dispatched at `aab4075`, whose code is `2818780`'s (02).
- The tree was clean, in the only worktree, at every launch.
- `list` showed 45 runs (43 finished, 2 `unknown`; "45 with no host recorded"), none `running` or
  stale.
- The lock was free, `ray-processes` listed nothing, and `RAY_ADDRESS` was unset.
- The venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`, `datastorekit` 0.4.0 and `raypoolkit`
  0.2.0, none editable.
- `this_host()` gave `id` `8d2b247824882486` and `boot` `376461AF-…`, 01's reading.

The record is [`docs/rrk-v020-adoption-verification.md`](../../../docs/rrk-v020-adoption-verification.md), and this
log points into it. The orchestrator's notes for this prompt are cited as "the note", by correction number.

## What shipped

- **`docs/rrk-v020-adoption-verification.md`** (new), with this content:
  - a header naming the document, the dispatch `HEAD`, and that 04 adds §3;
  - **§1, "R3 — through `runregistrykit` 0.2.0"**, §1.1–§1.9 in R2's §3 shape (§1.1.3 is the lock);
  - **§2, "The hard-kill route on the A3 v2 launcher"**: §2.1–§2.9, one per prompt step; §2.3,
    Ray after the kill; and §2.10, the issues.
- **`docs/rrk-v020-adoption/rehearsal/`**: 24 saved outputs named `*_03*`:
  - the run records and checks: `prelaunch_checks_03.txt`, `run_records_03.txt` (the six runs, and
    `list` after R3, counted by state), `var_datastores_03.txt`;
  - the lock: `lock_03.txt`;
  - the interrupter: `interrupter_dummy_test_03.txt`, `interrupter_build_dummy_test_03.txt`,
    `interrupter_firings_03.txt` (three logs, `hardkill-a`'s among them);
  - the stores: `store_sidecars_03.txt`, `closed_store_state_03.txt` (six states),
    `prune_check_03.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_03.txt`,
    `fingerprint_against_recorded_03.txt`, `census_03.txt`, `store_soundness_03.txt`, the last three
    with the `hardkill` store's runs appended;
  - the guard and L4: `guard_03.txt`, `stage_counts_03.txt` (six runs), and
    `policy_statistics_03_{reference,resumed-b,resumed-c,hardkill-a,hardkill-b}.txt`;
  - the readers: `readers_03.txt`, with the fingerprint taken after them;
  - `log_comparison_03.txt`: the searches, the line kinds against R2's paired slug, the `** `
    sequences, the status fields, the stage times and the reader logs;
  - **`hardkill_03.txt`**: the pair, every command and output in order, with the user's rulings
    and an excerpt of the GCS server's stack sample.
- **This log; the board** (header, the §1 row, L1–L4, H1, H2, the R3 and HK rows of §2, §3 and §4,
  `### 5.3`); **the `handover` board** (the entry moves from §3 to §4; the header's count);
  **`docs/OPEN_ISSUES.md`** (§1.22: the closed row leaves and the new row joins; the header).
- **Under `var/`, not committed:**
  - three stores with their sidecars;
  - six run directories and seven launch logs;
  - `var/rrk-v020-adoption-03/`, with the orchestrator's `gcs_server_99918_sample.txt`.

  No store is retired. No code, test or script changed; README, `prompts/INDEX.md` and
  `orchestrator/` are untouched.

## The runs

| Slug | Run id | Python pid (`sh`) | Interrupter | State | Exit code | Driver exit status | Wall time |
|---|---|---|---|---|---|---|---|
| `reference` | `rrk-v020-adoption-03-reference-20261018T232838` | 90795 (90789) | — | `done` | 0 | 0 | 3 m 52 s |
| `resumed-a` | `rrk-v020-adoption-03-resumed-a-20261018T233357` | 93122 (93116) | 93128, `--hold 10` | `killed` | 143 | 143 | 2 m 18 s |
| `resumed-b` | `rrk-v020-adoption-03-resumed-b-20261018T233734` | 94708 (94702) | 94723, `--hold 30` | `killed` | 143 | 143 | 1 m 19 s |
| `resumed-c` | `rrk-v020-adoption-03-resumed-c-20261018T233941` | 95847 (95843) | — | `done` | 0 | 0 | 1 m 46 s |
| `hardkill-a` | `rrk-v020-adoption-03-hardkill-a-20261018T234619` | 99903 (99897) | 99907, `--hold 30 --signal KILL` | `running` → `lost` by `end` | `null` | 137 | 2 m 25 s to the last heartbeat |
| (refused) | none: `begin` refused `rrk-v020-adoption-03-hardkill-b-20261019T001055` | (10117) | — | — | — | 1 | — |
| `hardkill-b` | `rrk-v020-adoption-03-hardkill-b-20261019T001145` | 10646 (10642) | — | `done` | 0 | 0 | 1 m 46 s |

**Fingerprints.** All are format 6, `encoding_format` 1, problems `{}`, 21 classes:
- `reference`, `resumed-c` and `hardkill-b`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `c60e0d19…`;
- `resumed-b`: `7e41cbe2…`;
- `hardkill-a` has none: no `finish` ran.

**Every manifest** (six) has:
- `git_head` `aab40752…` and `git_dirty` false;
- `packages` equal, as JSON, to the note's record;
- `script_sha256` `2a985303…`;
- twenty keys, R2's nineteen and `host`, whose `id` and `name` are `this_host()`'s.

Every status's `host` has this machine's `id` and `boot`.

**The interruptions** (from R3's own reference: `GkSource` 24.8 s, QSI 91.4 s):
- **`resumed-a`.** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10`.
  It became eligible at 23:36:03 and sent `SIGTERM` at 23:36:13. That left **168 `GkSource` rows,
  48 unvalidated**.
- **`resumed-b`.** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 30`. It became
  eligible at 23:38:20 and sent `SIGTERM` at 23:38:50, with 181 QSI rows stored.
- **`hardkill-a`.** `resumed-b`'s pattern and hold, with `--signal KILL`. It became eligible at
  23:48:45 and sent `SIGKILL` at 23:49:15. **238 QSI rows** were stored: 864 − 626, inferred from
  what `hardkill-b` computed.

**What each resume pruned and recomputed:**
- **`resumed-b`** pruned the 48 at open (`prune_check`: 48 absent, 0 reused, 0 unvalidated, run
  twice). It computed `GkSource` 171 = 291 − 168 + 48, `GkSourcePolicyData` 291, and 181 QSI before
  its signal.
- **`resumed-c`** computed QSI 683 = 864 − 181 and nothing else.
- **`hardkill-b`** pruned nothing and rolled back nothing. `hardkill-a` left no unvalidated row,
  since QSI rows carry no `validated` column and `GkSource` had completed, and it left no journal.
  `hardkill-b` computed QSI 626 and nothing else.

**What the `SIGKILL` left:**
- `status.json` said `running`, pid 99903, `exit_code` `null`, heartbeat 23:48:44, with this
  machine's `host`.
- There was no `status.json.*.tmp` and no `-journal`, `-wal` or `-shm`.
- The launch log ended `** driver exit status 137`, after the shell's `… 99903 Killed: 9 …` line.
- **One Ray process outlived it:** the GCS server, 99918, reparented to pid 1. Everything else of
  the run was gone within 20 s (by the orchestrator's reading at 23:49:35).

**The lock** ([`lock_03.txt`](../../../docs/rrk-v020-adoption/rehearsal/lock_03.txt); verification §1.1.3):
- `acquire` ran at 23:27:51, before `reference`;
- `set-pid` ran at all six launches that began a run (90795, 93122, 94708, 95847, 99903 and
  10646), and not at the refused launch;
- it was held through the readers, the guard and the pair. From 23:59:15 to 00:10:05 it was held
  with the GCS server alive, by correction 10;
- `release` ran at 00:14:55, exit 0, and afterwards `status` said `lock: free` and `ray-processes`
  listed none;
- every command exited 0 but one `status` call at 23:34:09, whose exit status is not recorded
  (deviation 17).

## Deviations from the prompt

1. **`runregistry-extraction` 08's corrections carried over** (the note, correction 1):
   - `prune_serials.py` was not run; `prune_check.py` ran on R3's own `closed_store_state.py`
     output;
   - the six scripts and `run_readers.sh` ran as they stand, the other tools from where R2 ran
     them, none copied;
   - the pre-launch Ray check was `ray-processes`, with the machine-wide `pgrep` recorded beside
     it;
   - `fingerprint_both_ways.py` and RunRegistryKit's measure were not run;
   - the guard ran after the readers, giving `guard_03.txt`.

   **IMPLEMENTATION CHOICE**, by the note.
2. **`packages` was compared as parsed JSON, not "in this order"**: every record is written with
   `sort_keys`. **STRUCTURALLY REQUIRED**, by the note (correction 1).
3. **`resumed-a`'s hold is 10 s**: 40 % of R3's own `GkSource` stage, 24.8 s, is 9.92 s, which
   rounds to the 10 s floor. It still left 168 rows stored and 48 unvalidated. **IMPLEMENTATION
   CHOICE**, by the note (correction 2).
4. **`resumed-b`'s and `hardkill-a`'s hold is 30 s**: a third of R3's QSI stage, 91.4 s, under
   250 s. `hardkill-a` used `resumed-b`'s pattern and hold exactly, with `--signal KILL`.
   **IMPLEMENTATION CHOICE**, by the note (correction 3).
5. **No progress line appeared, and this is not a finding**: no stage ran past 300 s.
   **IMPLEMENTATION CHOICE**, by the note (correction 4).
6. **The refused resume imports Ray but does not start it.** The prompt says `begin` "must raise
   before Ray is imported". `run_build` imports `ray` before the pre-check. The evidence recorded
   is the absence of `bootstrapping a local Ray instance` and an empty `ray-processes`.
   **STRUCTURALLY REQUIRED**, by the note (correction 6).
7. **Every file of the `hardkill` store was hashed, and journals were looked for, at each step**;
   the sidecar is counted once it exists. **IMPLEMENTATION CHOICE**, by the note (correction 7).
8. **`closed_store_state.py` ran on the `hardkill` store after the Ray wait and before the refused
   resume, and again after `hardkill-b`.** No journal was found, so no read failed. **IMPLEMENTATION
   CHOICE**, by the note (correction 8).
9. **The Ray wait after `hardkill-a` was the agent's**: one foreground `kill -0` loop until the
   interrupter's stamp plus 600 s (23:59:15). Its outcome is deviation 10. **IMPLEMENTATION
   CHOICE**, by the note (correction 10).
10. **The prompt's §2.3 stop fired, and the route continued on the user's rulings.**
    - The GCS server 99918 was alive at 23:59:15. The agent stopped, signalled nothing, ran no
      `ray stop`, kept the lock, and reported.
    - The user ruled "Claude sends SIGTERM". The orchestrator sent it at 00:06:13, with no effect.
    - The orchestrator took a read-only stack sample (main thread in `GcsServer::Stop()` →
      `GrpcServer::Shutdown()` → `grpc::Server::ShutdownInternal` → `gpr_cv_wait`).
    - The user then ruled "Claude sends SIGKILL", sent at 00:10:05, and the process went at once.
    - The agent resumed at §2.2/§2.4. H2 is recorded as not holding as the prompt states it, and
      the behaviour is opened as an issue.

    **STRUCTURALLY REQUIRED**, by the user's two rulings under README §6.1 U3, relayed by the
    orchestrator.
11. **The refused launch's check was the agent's** (a `kill -0` wait on its `sh`, bounded at
    120 s; it took 1 s), with no `set-pid` and no hand-off. **IMPLEMENTATION CHOICE**, by the note
    (correction 11).
12. **The manifest's `host` has no `boot`**, so the manifests were checked on `id` and `name` and
    the statuses on `id` and `boot`. **STRUCTURALLY REQUIRED**, by the note (correction 12).
13. **"Every key the status held before is kept" was checked with `end`'s four writes set aside**
    (`state`, `exit_code`, `ended`, `heartbeat`). `ended.evidence`'s `pid`, `status_readable`,
    `heartbeat` and `heartbeat_age` are recorded. **IMPLEMENTATION CHOICE**, by the note
    (correction 13).
14. **`docs/OPEN_ISSUES.md`: the count stays at 100**: the closed row leaves §1.22 and the opened
    row joins it.
    - `grep -c '^| \`\['` gives 100 before and after.
    - The header's "**eight**" for the `handover` board becomes seven, and its clause on the
      hard-kill issue now says that the issue closed here.
    - "across twenty-one campaigns" becomes **twenty-two**. Counting the distinct boards in the
      index's Board column gives 17 before and 18 after, since `rrk-v020-adoption` now holds a row
      of its own. The base figure of 21 could not be reproduced by any count of the index, so
      only the increment was applied.
    - "**Two** of these hold no open issue" is not moved by this commit.

    **IMPLEMENTATION CHOICE**, by the note (correction 14 as amended by the orchestrator after the
    stop).
15. **The verification document's header and shape** follow correction 15. §1 is R2's §3.1–§3.9,
    with the guard folded into §1.4. §2 has §2.1–§2.9, with Ray after the kill as §2.3 (where the
    prompt puts it) and the issues as §2.10. **IMPLEMENTATION CHOICE**, by the note
    (corrections 15, 16).
16. **The `handover` board's header** said "**Eight** issues are open in §3". The entry's move
    makes that false, so the count became seven and the sentence says where the eighth closed.
    Nothing else in the header moved. **IMPLEMENTATION CHOICE** (README §6.2 C4: a statement this
    commit makes false is corrected in it).
17. **One `status` call's record in `lock_03.txt` lacks its exit line.** The agent piped
    `lk.sh status` through `| head` for display, which closed the `tee` into the record early. The
    call's listing up to that point is in the record, with a note saying so; its exit status is
    not recorded. No later call was displayed that way. **UNINTENDED DRIFT.**
18. **Two capture slips, each corrected in the working record before anything depended on it.**
    - The first extraction of `reference`'s Ray pids used a `git grep --no-index` form that failed.
      It was re-run with a Python one-liner, and both attempts are in `prelaunch_checks_03.txt`.
    - The first `store fingerprint` after the readers was displayed through a filter, so its `rc`
      was the filter's. It was re-run whole, and both are in `readers_03.txt`.

    **UNINTENDED DRIFT.**
19. **The echoed `end` command in `hardkill_03.txt` abbreviates the `--reason`'s time**
    (`00:11:12`), where the text passed and recorded in `ended.reason` is `2026-10-19 00:11:12
    +0100`. A note in the record says so. **UNINTENDED DRIFT.**
20. **The purpose was passed to `sh -c` as `$0`**, as at R0–R2. So the shell's `Killed: 9` line in
    `hardkill-a`'s launch log begins with the purpose, and that line names `runregistrykit` and
    `datastorekit`. It is the only launch-log line besides the banner to name either.
    **IMPLEMENTATION CHOICE.**
21. **The statistics blocks of `hardkill-a` and `hardkill-b` were also taken**, and the stage
    counts were taken for all six runs. **IMPLEMENTATION CHOICE.**

## Verification performed

**The acceptance items** (prompt §4):
1. **L1–L3 hold.** Both four-launch stores are `39809dca…`, format 6, `encoding_format` 1. They
   are equal to each other, to their sidecars, and to R2's two recorded fingerprints
   (`compare_with_recorded.py`: 0 differences each, verdict `equal`, exit 0). ✅
2. **Every manifest's `packages`** names the three packages at their pins, none editable. Every
   `git_dirty` is false, and every manifest and status `host` is this machine's. ✅
3. **The hard-kill route holds** on the real launcher, step by step (verification §2.1–§2.9). ✅
4. **No Ray process outlived `hardkill-a` past ten minutes: not met.** The GCS server outlived it
   by 20 m 50 s, until a `SIGKILL` on the user's ruling. Each process's time to go is recorded
   (verification §2.3). ❌, recorded and opened (deviation 10).
5. **L4 against R2**: every count equal, every statistics block byte-identical to R2's same slug,
   the chain's identities hold; each log difference is explained. Nothing is opened from L4. ✅
6. **`list`** shows 51 runs, 49 finished and 2 `unknown`, nothing `running` or stale. By state:
   27 `done`, 18 `killed`, 3 `failed`, 1 `lost`, 2 `unknown`. ✅
7. **The lock** was taken before `reference`, set at each launch that began a run, held through the
   readers and the pair, and released. Afterwards `status` says free and `ray-processes` lists
   nothing. One `status` exit is unrecorded (deviation 17). ✅
8. **The seven roots** pass at 02's counts. ✅
9. **`docs/rrk-v020-adoption-verification.md`** is created with §1 and §2. ✅
10. **`[a3-v2-a-hard-killed-run-blocks-its-store]` closes**, on both boards and in the index. ✅

**L1–L4 against R2** (verification §1.2–§1.5):

| | R2 | R3 |
|---|---|---|
| States / exits | done 0, killed 143, killed 143, done 0 | the same |
| `SystemExit` in `resumed-a` / `resumed-b` | 1 / 143 | 1 / 1 |
| Ray processes left after each `SIGTERM` | none | none |
| Statistics counts | 291; 109; 26 (25 + 1); 156 | the same; each block byte-identical to R2's same slug |
| Reference stage counts | — | equal stage by stage |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 44 = 44 | **48 = 48**, all absent, none reused |
| `resumed-b` `GkSource` = 291 − stored + pruned | 170 = 291 − 165 + 44 | 171 = 291 − 168 + 48 |
| `resumed-c` QSI = 864 − stored by `resumed-b` | 567 = 864 − 297 | 683 = 864 − 181 |
| `** ` lines | 26, 23, 26, 26 | the same; sequences equal normalised; banners verbatim |
| Completion / `Queue summary` / no-compute lines | 10/10/1, 7/7/1, 9/9/7, 10/10/9 | the same |
| Lines naming `runregistrykit` | the banner only | the banner only |
| Raylet disk-space lines | 54 | **0** (volume at 93 %, under Ray's 95 %) |
| Reader lookups | 116, 348, 10, 3, 6, 24; nomodels 116 | the same |
| Reference build / `GkSource` / QSI | 4 m 33 s / 27.4 s / 119.8 s | 3 m 52 s / 24.8 s / 91.4 s |

**The hard-kill route** (H1; verification §2.1–§2.9):

| Step | Expected | Found |
|---|---|---|
| 2.2 the kill | status `running`, exit `null`, launch log 137, stage in QSI | ✅ all; no `.tmp`, no journal |
| 2.4 `list` | `stale`, `!!`, stderr names `end` | ✅ "1 stale", `!!`, the `end` sentence |
| 2.4 `store show` | no sidecar; the run that names the store | ✅ `absent`; `hardkill-a running stale by results` |
| 2.5 refused resume | `RuntimeError` at `begin`, naming the dead run, `stale`, `end`, "No run directory was created"; non-zero exit | ✅ exit 1; no Ray started; 50 directories before and after; every store file byte-identical |
| 2.6 `end` | exit 0, `lost`, verified | ✅ evidence `this`/`this`/`false`/`running`; `git_head` the dispatch `HEAD`; other keys kept; manifest unchanged |
| 2.6 second `end` | refused | ✅ exit 1, "already ended, as 'lost'"; `status.json` byte-identical |
| 2.7 sidecar | `store create` | ✅ `63ca0315…`; files unchanged |
| 2.8 `hardkill-b` | `done`/0, fingerprint with `encoding_format`, sidecar `written` | ✅ `39809dca…`; no `fingerprint_error` |
| 2.9 checks | `39809dca…` = `reference`; soundness, census, prune check | ✅ all; `prune_check` "nothing to check" (no unvalidated row existed) |

**Ray after the `SIGKILL`** (H2; verification §2.3):
- the driver went at 23:49:15;
- the 17 other processes of the run listed at 23:46:27, all but the GCS server, were gone by
  23:49:35;
- the GCS server 99918 was alive at 23:59:15, survived a `SIGTERM` at 00:06:13, and went at the
  `SIGKILL` at 00:10:05.

**The suites**, serially in the foreground, after the release:

| Root | Baseline (02) | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 9.82 |
| CosmologyModels | 39 OK | 39 OK | 9.82 |
| ObjectFactories | 279 OK | 279 OK | 9.19 |
| tools | 91 OK | 91 OK | 8.45 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 8.76 |
| RunDrivers | 103 OK | 103 OK | 16.78 |
| ComputeTargets | 700 OK | 700 OK | 15.84 |

No failure, so no known flake was re-run. No `.py` file was touched, so `black` was not run.

**`var/` outside R3's records.**
- The 21 files under `var/datastores/` have the same SHA-256, size and `mtime_ns` before and after.
- The 18 new files are R3's three stores and their sidecars, and there is no journal.
- Nothing else under `var/` is newer than the first capture.
- `git status --short` was empty at every launch, and through `hardkill-b`'s end.

## The deliberate-breakage record

None. This prompt adds no code, no test and no script (README §5 rule 8).

## Observations not acted on

1. **The GCS server's deadlock** is opened as `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`
   (board §3). Two things are recorded and not explained:
   - The sample shows `Stop()` entered from the GCS server's own `boost::asio` signal handler, in
     the second of the driver's kill. What signalled it (the dying raylet, Ray's process-group
     handling, or another path) was not established.
   - Whether every `SIGKILL` of a driver leaves the GCS server like this, or only some, is one
     observation of one.
2. **The prune at open is not exercised by the hard-kill pair.** `hardkill-a` was killed in QSI,
   after `GkSource` had completed and validated all its rows, and QSI rows have no `validated`
   column. So the hard kill left nothing to prune and no journal to roll back. A `SIGKILL` inside
   the `GkSource` stage, or inside a write, would test both. This prompt did not ask for one.
3. **`hardkill-a`'s 238 stored QSI rows are inferred** (864 − 626). No fingerprint or count was
   taken between the kill and `hardkill-b`: `closed_store_state.py` does not count that class, and
   `store fingerprint` without `--write` was not run on a store a stale run named (README §6.2 C8's
   reasoning applies).
4. **`SystemExit(1)` in both `SIGTERM`'d launches**, against R2's 1 and 143. This is R2's §3.2.2
   explanation, with where in the pool's loop each signal fell. The record is `killed`/143 either
   way.
5. **The index header's "twenty-one campaigns" and "Two of these hold no open issue" could not be
   reproduced by counting.** By the index's Board column, 17 distinct boards held rows at `aab4075`.
   Several listed boards hold none: `runregistry-extraction` says so of itself in §1.21. This commit
   applied only the increment its own change makes (deviation 14). A recount of those clauses is
   housekeeping for 04 or the orchestrator.
6. **R2's disk-space warning is gone** at 93 % used. A3 v2's stores are larger, and the threshold
   is 95 %.

## State handed to the next prompt

- **R3 holds** (board §2), resting on:
  - the dispatch `HEAD` `aab4075`, whose code is `2818780`'s;
  - this commit;
  - the runs `rrk-v020-adoption-03-reference-20261018T232838` and
    `rrk-v020-adoption-03-resumed-c-20261018T233941`, both `done`, both `39809dca…`, equal to R2's.
- **HK holds for the route** (`begin` refuses, `end` verifies, the resume reproduces `39809dca…` with
  its fingerprint recorded), resting on `hardkill-a` (`lost`) and
  `rrk-v020-adoption-03-hardkill-b-20261019T001145` (`done`).
  **H2 does not hold as stated:**
  - the GCS server deadlocks in shutdown after a `SIGKILL` of the driver;
  - after `end`, a person checks `ray-processes`, and may need to `SIGKILL` a GCS server by its pid,
    never with `ray stop`;
  - 04 states this among what A3 v2 inherits.
- **R3's three stores** are `var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`,
  each with a sidecar carrying `39809dca…` and none retired. The user retires them after 04's
  review, or not.
- **The lock is free** and no Ray process of this repository is running.
- **The venv is unchanged:** 03 installed nothing. `docs/OPEN_ISSUES.md` stays at 100: one closed,
  one opened.
- **The registry** holds 51 runs: 49 finished (27 `done`, 18 `killed`, 3 `failed`, 1 `lost`) and 2
  `unknown`. None is `running` or stale.
- **The roots:** AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91, LiouvilleGreen
  148 (1 skipped), RunDrivers 103, ComputeTargets 700.
