# Log — prompt 02, rehearse the start tree (R0)

**Prompt:** [`../02-rehearse-the-start-tree.md`](../02-rehearse-the-start-tree.md) ·
**Board items:** S1, L1–L4 (R0) · **Date:** 2026-10-11 · **Model:** Opus 5.5 ·
**Commit:** `7650102` · **Result:** landed. **R0 holds:** both stores are `39809dca…`, all 21
classes, equal to each other and to the fingerprints 07's reference and 09's rehearsal recorded.
L1–L4 recorded; every suite at its dispatch count; two issues opened, none closed.

Dispatched at `a5392fd` ("Write the orchestrator note for dk-and-rpk-migration prompt 02"), whose
code is `f54f1c5`'s. Tree clean, the only worktree; `RunRegistry list` 17 runs (15 finished, 2
`unknown`), none `running`; no Ray process at all; `RAY_ADDRESS` unset; neither `datastorekit` nor
`raypoolkit` in the venv. Every launch ran at `a5392fd` with `git status --short` empty. The record
is [`docs/dk-and-rpk-migration-verification.md`](../../../docs/dk-and-rpk-migration-verification.md)
§1; this log points into it rather than repeat it.

## What shipped

- **`docs/dk-and-rpk-migration-verification.md`**, new: its first paragraph names the campaign, the
  tree, the stores and the runs; **§1, "R0 — the start tree"**, holds every measurement, with every
  command verbatim.
- **`docs/dk-and-rpk-migration/rehearsal/`**, new.
  - **S1's three scripts**, each with its usage in its docstring, no prompt number inside, and every
    path an argument or derived from one:
    - `run_readers.sh <primary> <output dir> [nomodels <script>]`. The six readers with
      `--ray-address local --models LambdaCDM`, or one without `--models`; before and after each,
      each store file's SHA-256, mtime and size (BSD `/usr/bin/stat`) and any journal. The shards
      come from the primary's `shards` table. Exit 1 if a reader fails or a file changes.
    - `stage_counts.py <run dir>...`. 07's table through
      `docs/datastore-integrity-verification/stage_counts.read`, each column labelled by the slug
      recovered from the manifest's own `run_id`, less the `<campaign>-<prompt>-` prefix its
      `campaign` and `prompt` give (through `RunRegistry._slug`, as `run_id` builds it) and the
      timestamp.
    - `policy_statistics.py <run dir>`. The block from `** GkSourcePolicyData SUMMARY STATISTICS`
      up to, not including, the first later line that begins `** ` or `@@ ` after leading
      whitespace, or the end of the file; verbatim, nothing else. Exit 1 when the header appears 0
      or more than 1 times (checked on `resumed-a`, which has none, and on a two-header stand-in).
  - **Three more**, reusable in the same way, which 05 and 08 run too:
    - `prune_check.py <closed_store_state output> <primary>`: the serials a killed launch left
      unvalidated, read from `closed_store_state.py`'s saved output, each looked up again `mode=ro`.
      07's `prune_serials.py` hardcodes its own serials.
    - `compare_with_recorded.py <primary>... --against <run dir>...`: each store fingerprinted
      read-only, compared with its sidecar and with the fingerprint in each run's `status.json`.
    - `store_soundness.py <primary>...`: 07's, passing the registry to `read_inventory`, which 07's
      no longer does (deviation 3).
  - **Saved outputs**, each named `*_02*`: `run_records_02.txt`, `prelaunch_checks_02.txt`,
    `interrupter_dummy_test_02.txt`, `interrupter_build_dummy_test_02.txt`,
    `interrupter_firings_02.txt`, `store_sidecars_02.txt`, `closed_store_state_02.txt`,
    `prune_check_02.txt`, `fingerprint_comparison_02.txt`, `fingerprint_against_recorded_02.txt`,
    `census_02.txt`, `store_soundness_02.txt`, `guard_02.txt`, `stage_counts_02.txt`,
    `policy_statistics_02_reference.txt`, `readers_02.txt`.
- **This log, the board, `docs/OPEN_ISSUES.md`** (§1.20, 104 → 106), and **one line under README
  §4's "02 ships"** (deviation 5).
- **Under `var/`, not committed:** the two stores and their sidecars, four run directories and their
  launch logs, and `var/dk-and-rpk-migration-02/` (verification §1.9).

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-02-reference-20261011T143003` | 65761 (65752) | — | `done` | 0 | 0 | 8 m 0 s | 30.74 35.73 25.37 / 13.66 26.01 27.30 |
| `resumed-a` | `dk-and-rpk-migration-02-resumed-a-20261011T143917` | 71276 (71273) | 71349 | `killed` | 143 | 143 | 2 m 27 s | 9.17 23.05 26.14 / 6.82 16.56 22.84 |
| `resumed-b` | `dk-and-rpk-migration-02-resumed-b-20261011T144253` | 73797 (73794) | 73873 | `killed` | 143 | 143 | 2 m 8 s | 6.38 15.81 22.43 / 17.43 17.75 22.19 |
| `resumed-c` | `dk-and-rpk-migration-02-resumed-c-20261011T144549` | 75882 (75880) | — | `done` | 0 | 0 | 3 m 0 s | 15.51 17.26 21.88 / 22.54 22.44 23.09 |

Fingerprints: `reference` and `resumed-c` `39809dca…`; `resumed-a` `eaf590af…`; `resumed-b`
`fc02616e…`; all format 6, problems `{}`. Every manifest: `git_head` `a5392fd1…`, `git_dirty` false,
`packages` `{"datastorekit": null, "raypoolkit": null}`, `script_sha256` `d745783e…` in all four.
After each launch the agent made one check and ended its turn; the orchestrator's one wait per
launch ended `finished` each time, the heartbeat never stale.

**The interrupter.**
- **Dummy tests**, before the first launch, into new scratch directories: both exit 0, every
  verdict as 07's record (five cases; three `--build` cases), no child left.
- **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10`.
  Eligible 14:41:32, `sent SIGTERM` 14:41:42; handler line 14:41:42, delay under 1 s. 119 of 291
  `GkSource` rows stored, 43 unvalidated.
- **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 70`. Eligible
  14:43:43, `sent SIGTERM` 14:44:53; handler line 14:44:54, delay under 2 s (the stamps are whole
  seconds). 332 of 864 `QuadSourceIntegral` rows stored. `SystemExit(1)` from `ray.wait`, recorded
  `killed`/143, as at 07.

**The killed launches' Ray agents.** None to wait for. After `resumed-a` (14:42:28, ~44 s after
exit), and after `resumed-b` (14:45:16, ~15 s after exit), `pgrep -lf 'gcs_server|raylet|ray::'`,
the whole-table search for `SecondaryGWKit/venv` with `ray`, and a listing of the driver's process
group were all empty. 07's orphaned agents did not appear, so no wait by pid was run, and their
exit time is "none listed at the first check".

## Deviations from the prompt

1. **`resumed-a`'s point: the banner with `--hold 10`.** R0's reference `GkSource` stage took 32.2 s
   and printed no progress line, so the signal was put 10–15 s in, as 07 did from the banner. 07's
   `--hold 15` was sized for a 62 s stage. **IMPLEMENTATION CHOICE.**
2. **`resumed-b`'s point: the banner with `--hold 70`, the pattern not anchored at the end.** R0's
   reference QSI stage took 3 m 32.3 s and printed no progress line, so 07's trigger (the 100 %
   line) would never have fired. The point is about a third of the stage.
   - The agent first planned `'^CALCULATE QUADRATIC SOURCE INTEGRALS$'`, as the briefing suggested,
     arguing that no progress line could come before 300 s. The orchestrator corrected this before
     the launch: by `RayWorkPool`'s defaults a line can come after about 60 s, inside the hold, and
     a `$` pattern would then never signal. So the `$` was dropped.
   - The agent checked that the un-anchored pattern cannot fire between stages: `progress()` gives
     `None` for `…, all work items complete`.
   - Read afterwards, the first argument was right for this queue: `main.py` passes it
     `notify_min_time_interval=MIN_NOTIFY_INTERVAL` (300 s) and `notify_batch_size=2000`
     (`main.py:119`, `:4155-4156`), so no line can come before 300 s. The un-anchored pattern is
     correct either way.

   **IMPLEMENTATION CHOICE.**
3. **`store_soundness.py` is this campaign's copy, not 07's.** README §4.1's L2 names 07's
   `docs/a3-v2-readiness/rehearsal/store_soundness.py`. At this tree it prints the reference
   store's pragmas, then raises `TypeError: read_inventory() missing 1 required positional argument:
   'factories'`: since `19f07ee`, `read_inventory` takes the registry. 07's file is a record and was
   not changed. The copy passes `config.datastore.factories` as `fingerprint_store` does, and reads
   the shards from the `shards` table. Opened
   `[02-07s-store-soundness-calls-read-inventory-without-its-registry]`. **STRUCTURALLY REQUIRED**:
   L2 needs the check, and the prompt allows "any other script of yours".
4. **Two scripts beyond S1, and one extra comparison.** `prune_check.py`, which the briefing asked
   for, and `compare_with_recorded.py`, because `compare_stores.py` compares two stores and the
   prompt's comparison is against a fingerprint recorded in a `status.json`. Both stores were also
   compared with 09's rehearsal fingerprint, which the prompt does not ask for; it is the last
   rebuild that reproduced `39809dca`, and it costs one argument. **IMPLEMENTATION CHOICE.**
5. **README §4, one line under "02 ships".** The prompt's §6 does not list README, but 01 recorded
   its interface there and 05 and 08 read it; the briefing asked for the line. It says the scripts
   shipped as named and names the three more. **STRUCTURALLY REQUIRED.**
6. **The guard's after-check was completed against the 14:49 hashes.** The agent's first command
   compared the before and after hashes with `==` in zsh, which zsh rejects. The guard itself had
   run and refused. The comparison was then made with `shasum -c` against
   `var/dk-and-rpk-migration-02/state/after-resumed-c.sha256`, taken at 14:49, after `resumed-c`
   and before the guard (14:52:26). Between them only read-only checks ran, each of which found
   the same hashes. All 12 files, the two sidecars included, matched. **UNINTENDED DRIFT**, of the
   agent's tooling, not of the measurement; `guard_02.txt` holds both parts.
7. **`compare_with_recorded.py` was formatted by `black` after its first run.** It was run again
   afterwards, and its output is byte-identical to the saved one. **IMPLEMENTATION CHOICE.**

## Verification performed

**L1–L3** (verification §1.2–§1.4):
- **L1.** Every launch ended as recorded; each driver's exit status agrees; no Ray process of this
  repository left after any launch; every manifest as the prompt's §2 requires; every `stdout.log`
  as 07 §2.1 requires; the guard refused before Ray started, naming the five files, and changed
  nothing.
- **L2.**
  - Both stores are format 6, `39809dca…`, all 21 classes equal, each equal to its sidecar's.
    `compare_fingerprints` is empty between them and against 07's and 09's recorded fingerprints.
  - The 43 unvalidated `GkSource` serials are all absent after `resumed-b`'s open, none reused.
    11 + 32 = 43 in the folded warnings.
  - No stored row was recomputed, and nothing is duplicated.
  - Version row `(1, '2025.1.1')` on all 8 shards; census `no divergence` on both; `integrity_check`
    `ok`, `foreign_key_check` empty, and no inventory problem on all 10 files.
- **L3.** All six readers exit 0, and `extract_Gk_data.py` without `--models` exits 0 with the
  skip line. Nothing refused; every store file byte-identical; `store fingerprint` afterwards
  matches the sidecar.

**L4, the baseline** (verification §1.5):
- The reference's statistics block, verbatim:
  - 291 instances;
  - numeric 109 (109 complete);
  - mixed 26 (25 complete, 1 minimal);
  - WKB 156 (156 complete).
- **Against 09's: the same 14 lines, but `minimal` and `complete` under `mixed` are swapped.**
  Sorted, the blocks are equal. The order follows Ray's completion order: R0's own `resumed-b`
  prints 09's order. A finding, not a stop.
- Stage counts:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | 215/215/215 | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | 532/532/532 |

- Pruned and recomputed:

| Launch | Pruned at open | Recomputed |
|---|---|---|
| `resumed-b` | 43 `GkSource` (shards 0, 1, 3: 11, 10, 22) | `GkSource` 215 = 291 − 119 + 43; `GkSourcePolicyData` 291 (none stored before); `QuadSourceIntegral` 332 stored at the signal |
| `resumed-c` | nothing | `QuadSourceIntegral` 532 = 864 − 332; nothing else |

**The suites**, serially in the foreground after the last launch, README §7's command:

| Root | Dispatch baseline | Now | Load |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 26.7 |
| CosmologyModels | 39 OK | 39 OK | 26.7 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 26.7 |
| RunRegistry | 286 OK | 286 OK | 20.5 |
| ComputeTargets | 693, 1 load flake | 693 OK | 20.3 |
| Datastore | 739 = 163 + 185 + 200 + 191 OK | 739 = 163 + 185 + 200 + 191 OK | 6–12 |

`black --check` is clean on the five Python files added.

## The deliberate-breakage record

None. This prompt changes no production code and no test, and names no mutation. The scripts' own
failure paths were exercised on real inputs instead:
- `policy_statistics.py` exits 1 on `resumed-a` (no block) and on a two-header stand-in, and 2
  with no argument.
- `prune_check.py` reports "nothing to check" on the state after `resumed-b`.
- 07's `store_soundness.py` failing is deviation 3.

## Observations not acted on

1. **The briefing said 07's `store_soundness.py` "runs unchanged".** It does not (deviation 3).
   `closed_store_state.py`, `compare_stores.py` and `replicated_serial_census.py` do.
2. **07's `[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur**, at loads 7–17 and
   checks 15–44 s after exit. That is consistent with 07's inference that the agents linger under
   load, but it establishes nothing. The issue is `a3-v2-readiness`'s and stays open.
3. **The readers print `!! WARNING: number of shards read from database (=4) does not match
   specified number of shards (=10)`**, from their default `--shards`. 07's logs hold the same line.
   It is harmless, and no issue is opened.
4. **`script_sha256` is `d745783e…`, not 07's `3d0e6ecd…`**: `quadsource_atol_sweep.py` has changed
   since `dbd8a51`. Expected. The fingerprint is unchanged.
5. **`docs/OPEN_ISSUES.md`'s campaign count** went from "nineteen" to "twenty" by adding this
   campaign. The earlier figure was not recounted.
6. **The `GkSource` queue has the same notify settings** (`MIN_NOTIFY_INTERVAL`, 2000), so neither
   interrupted stage can print a progress line before 300 s. Interruption points in R1 and R2 come
   from the banner with `--hold`, unless a stage runs past 300 s.

## State handed to the next prompt

- **R0 holds.** Its stores are `var/datastores/dk-and-rpk-migration-02-{reference,resumed}.sqlite`,
  with sidecars carrying `39809dca…`, and are not retired. 04's read-only fingerprint and R1's
  comparison read them, or `compare_with_recorded.py --against
  var/runs/dk-and-rpk-migration-02-reference-20261011T143003`.
- **05 and 08 run** `run_readers.sh`, `stage_counts.py`, `policy_statistics.py`, `prune_check.py`,
  `compare_with_recorded.py` and this campaign's `store_soundness.py`, unchanged. The statistics
  block is compared for its counts: line for line, and sorted where only the order differs
  (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
- The venv is unchanged (no `pip install`). `docs/OPEN_ISSUES.md` is at 106.
