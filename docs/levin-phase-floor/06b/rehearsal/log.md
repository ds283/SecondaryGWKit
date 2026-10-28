## §4 — rehearsals R15 and R16 on the Mac Studio

**Prompt:** [`06b-roundoff-exit-premise.md`](../06b-roundoff-exit-premise.md) §4, under the
orchestrator's Part B dispatch (`92bc62f`) and its addendum B.5.
**Host:** mps026298, the Mac Studio.
- **R15** ran in the worktree `SecondaryGWKit-lpf06b-da5c24d`, detached at `da5c24d`. That is the
  code commit `4ddc3c2`'s parent.
- **R16**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06b-92bc62f`,
  detached at `92bc62f`. Its code is `4ddc3c2`'s: `git diff --stat 4ddc3c2 92bc62f` outside
  `docs/` and `prompts/` is empty.
- **The main checkout** stayed at `28ebaf0` and clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R15's launch;
2. R15's checks and R16's launch;
3. R16's checks, the comparison, the tabulation, the readers, the release and these records.

**Date:** 2026-10-28, 13:52–14:12 UTC.
**Commit:** none. The records are committed on the laptop, in 06b's records commit.
**Result:** COMPLETE WITH DEVIATIONS. No deviation stopped the work or changed a computation, and
no §7 stop condition fired.
- **The fingerprints.** R15 and R16 both ended `done`, exit 0, with fingerprint `1b332ffa…`.
  Each equals R8's class by class, and R16's `classes` object equals R15's.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` or the residual-memo counters;
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`:**
  - 82 of 864 rows carry a phase-limited flag in R15;
  - all 782 rows without one are bit-identical in every value field and every work counter;
  - the 82 are tabulated;
  - R16 carries **no** phase-limited flag on any row.
- **The readers** exited 0, and R16 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6b, "The round-off-limited exit".

### What shipped

Nothing in the repository: no code and no commit. Everything was written on the Mac Studio, in the
main checkout.
- **The records directory** is `var/levin-phase-floor-06b/rehearsal/`:
  - `out/` holds every command (`commands.log`) and every output;
  - `scripts/` holds:
    - 05's `classify_non_qsi.py`, unchanged;
    - 06's `run_readers_worktree.sh`, unchanged;
    - this agent's `tabulate_qsi_phase_limited.py` (§4.3's per-row tabulation), as run in
      attempt 2;
    - `rec.sh`, the recorder of read-only checks;
  - `state/` holds the lock transcript, the launch records, the store hashes, attempt 1's script
    and `progress.md`;
  - `extract/` is the readers' output, 180 MB, and is not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores:**
  - `var/datastores/levin-phase-floor-06b-r15.sqlite` and `-r16.sqlite`, each with 4 shards;
  - **they are kept** (§4.6).
- **The runs:**
  - `var/runs/levin-phase-floor-06b-r15-parent-20261028T135349/` and
    `…-r16-premise-20261028T135745/`;
  - the launch logs `var/runs/levin-phase-floor-06b-r15.launch.log` and `-r16.launch.log`.

### Predictions and verdicts

§4 states acceptance rules, not pre-registered predictions. Each is scored here as written.

| rule (§4, §6.2) | number | verdict |
|---|---|---|
| R15 `done` with `1b332ffa…` | `done`, exit 0, digest `1b332ffa948a3dbd…`, classes equal R8's | **holds** |
| R16 `done` with `1b332ffa…` | `done`, exit 0, digest `1b332ffa948a3dbd…`, classes equal R8's and R15's | **holds** |
| Outside `QuadSourceIntegral`, every value table identical | 7 value tables, 360,462 rows identical, 0 differ | **holds** |
| …every other difference of the four volatile kinds | 0 unclassified (`classify_non_qsi.py`, exit 0) | **holds** |
| …no row in one store only | 0 only-in-A, only-in-B or unmatched, in every class | **holds** |
| Inside, a row with no phase-limited flag in R15 is bit-identical in `total`, `total_abserr`, the analytic columns and every group record | 0 of 782 differ, in that set, in every value field and in every work counter | **holds** |
| Inside, every other row is tabulated | 82 rows, in `out/qsi_tabulation.txt` and `out/qsi_rows.csv` | **done** |
| The readers exit 0, and the store is byte-identical | 6 of 6 exit 0; 5 files identical by SHA-256, mtime and size | **holds** |
| The lock is released, and `ray-processes` lists nothing | released, exit 0; `status` free; `ray-processes` none | **holds** |

### Where the phase-limited flags are stored (§4.3)

The flags were established from the code at `92bc62f` (`ComputeTargets/QuadSourceIntegral.py`,
`ObjectFactories/QuadSourceIntegral.py`; the factory at `da5c24d` is the same) and from the stores.
They all live in the shard table `QuadSourceIntegral`.

| flag | column | encoding | where it is set |
|---|---|---|---|
| (a) `total_phase_limited` | `total_phase_limited` | SQLAlchemy `Boolean`, nullable. Stored as SQLite `integer` 0/1: `typeof` is `integer` on all 864 rows of both stores | `evaluate_QuadSource_integral`: the OR of every group's flag, over every Levin sub-interval |
| (b) each group record's `phase_limited` | `metadata`, a JSON string (`json.dumps`, read with `json.loads`), at `WKB_Levin.subintervals[i].groups[j].phase_limited` | JSON `true`/`false` | `phase_group_Levin_integral`: each phase group's `adaptive_levin_sincos` flag |
| (c) each analytic block's `phase_limited` | `metadata`, at `analytic.{0pt5,2pt5}.{J,Y}{1..4}_data.phase_limited` when that three-Bessel integral is all Levin, or at `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.phase_limited` when it is split between quad and Levin | JSON `true`/`false` | `_three_bessel_Levin`'s per-block record (via `_three_bessel_integrals` and `analytic_integral`) |

- **Two more stored keys** are ORs of (b): `WKB_Levin.phase_limited`, and
  `WKB_Levin.subintervals[i].phase_limited`. The script asserts that they are on every row, and
  that `total_phase_limited` equals the OR of (b). No assertion fired.
- **No other** metadata leaf named `phase_limited` exists in either store.
- **The ORs of (c),** `J_phase_limited` and `Y_phase_limited`, are computed in
  `_three_bessel_Levin` but **stored nowhere**. So (c) has to be read block by block.
- **"Phase-limited in R15"** is (a) or any (b) or any (c), on R15's row.

### Deviations from the prompt

1. **The lock transcript** is `state/lock.log`; 06's was `out/lock.log`. The brief puts lock
   transcripts in `state/`. `IMPLEMENTATION CHOICE`.
2. **Extra `ray-processes` looks:** one before `acquire`, one after each build, and one after the
   readers. They are read-only and not in §4.1's list. Each was a command of its own, appended by
   `>>`. `IMPLEMENTATION CHOICE`.
3. **The drivers ran with both `PYTHONDONTWRITEBYTECODE=1` and `-B`.** R13 used the variable
   alone; the brief asks for either. No effect on any value. `IMPLEMENTATION CHOICE`.
4. **`scripts/rec.sh`** records each read-only check: command, cwd, UTC start and end, output and
   exit code, in `out/NAME.txt`. It was used for checks only, never for a lock command or a
   launch. `IMPLEMENTATION CHOICE`.
5. **R16's purpose string** adds "QuadSourceIntegral alone is expected to differ, only in rows with
   a phase-limited flag in R15". That is §0.3 and §4.3's expectation, stated beside the brief's
   required content (host, tree SHA, `4ddc3c2`, `1b332ffa…`). `IMPLEMENTATION CHOICE`.
6. **The tabulation, attempt 1, failed.** It counted store-bookkeeping columns as values: `serial`,
   every `*_serial` (the row's own and its parents' serials, which each store assigns for itself)
   and `timestamp`. It reported 782 "value violations", all in those columns. Rows are paired by
   physical key, so these columns are not values. They are like `parents_digest` of
   `[05-store-value-diff-compares-volatile-columns]`, which `store_value_diff` itself never
   compares.
   - **The edit:** an `identity` kind for these columns, which are then not compared.
   - **The records:** the diff is `out/tabulate_qsi_phase_limited.attempt1_to_2.diff`. Attempt 1's
     script is `state/tabulate_qsi_phase_limited.attempt1.py`, and its output
     `out/qsi_tabulation.attempt1.txt`, `.stderr` and `out/qsi_rows.attempt1.csv`.
   - Attempt 2's flagged-row table is identical to attempt 1's. Only the verdict lines changed.

   `UNINTENDED DRIFT` (the script's first classification was wrong), corrected before any result
   was used.
7. **`run_readers.sh` cannot run from the code worktree as written.** It `cd`s to its own tree and
   calls `./venv/bin/python`, which the worktree lacks. 06's `run_readers_worktree.sh` was used,
   copied unchanged (`cmp` identical, `ec96cf09…`). Its diff against `run_readers.sh` equals 06's
   (`out/run_readers_worktree.diff`). `STRUCTURALLY REQUIRED`.
8. **One `rec.sh` check failed to parse.** Its first form, `readers_check`, used bash process
   substitution, which `sh -c` rejects (exit 2). Nothing ran. The second form overwrote its
   output file. The failure is recorded in `out/commands.log`. `UNINTENDED DRIFT`, with no effect.
9. **Two inline read-only looks were not saved:**
   - the tool JSON's `QuadSourceIntegral` keys and counts. `out/qsi_tabulation.txt`'s cross-check
     restates what was used;
   - the readers' file-name patterns against 06's, and the grep of `random.sample` in the readers.

   `IMPLEMENTATION CHOICE`.
10. **The scripts are left as run, not formatted by `black`.** `black --check` would reformat
    `tabulate_qsi_phase_limited.py` and `classify_non_qsi.py` (`out/black_check.txt`). They are
    records, byte-identical to the run, as 05's and 06's rehearsal scripts are. Committing them
    under `docs/levin-phase-floor/06b/rehearsal/scripts/` needs that path in `CLAUDE.md`'s
    formatting exemptions. `IMPLEMENTATION CHOICE`, for the laptop.

### Verification performed

**Gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`, `out/02-gate-launcher.txt`):
- **The trees:**
  - the main checkout is `handover-remedial` at `28ebaf0` and clean;
  - the two worktrees are at `da5c24d` and `92bc62f`, both clean, ignored files included;
  - `4ddc3c2^` is `da5c24d`;
  - `da5c24d..92bc62f` on code paths touches only `AdaptiveLevin/levin_quadrature.py` and the new
    test module.
- **The registry and the lock:** the registry held 25 runs, all finished. The lock was free, with
  no old-location lock.
- **The host:** `auth_token` is mode 600, and 109 GiB were free.
- **The tools by SHA-256:** the launcher is `5b984c75…` in all three trees. `store_value_diff.py`
  is `7997af85…` in both worktrees, and `run_readers.sh` is `7c63a5ba…`.

**The builds** (`out/r15_launch_check.txt`, `out/r15_end_check.txt`, `out/r16_launch_check.txt`,
`out/r16_end_check.txt`):
- **The launch check.** The manifests carry the right `git_head`, with `git_dirty` false. Each
  driver's cwd (`lsof`) is its worktree.
- **The end.** No Traceback in either launch log, `stdout.log` or `stderr.log`, and no
  `!! WARNING` in either launch log. Each driver exited 0.
- **The worktrees** were clean after each build, with nothing newer than the pre-R15 marker.

**The stores:**
- R15's five files are unchanged from the end of its build to the end of the session;
- both stores' ten files are identical before and after the comparison, and at the end;
- R16's five are identical before and after the readers;
- there is never a journal, `-wal` or `-shm` file after a build.

The hashes are in `state/r15_store_sha256_after_build.txt`, `state/store_sha256_before_diff.txt`,
`state/store_sha256_after_diff.txt`, `state/r16_store_sha256_{before,after}_reads.txt` and
`state/stores_sha256_final.txt`.

**The comparison** (`out/store_value_diff.json`, `51b9b94a…`, 40.3 MB; `out/classify_non_qsi.txt`;
`out/qsi_tabulation.txt`; `out/qsi_set_check.txt`):
- **The tool** exited 1 in 18 s. `QuadSourceIntegral` is the only class with a value-like
  difference.
- **The cross-checks.** The tabulation's pairing by $(k, q, r, z_{\rm resp})$ gives the tool's key
  set, and `total`'s `float.hex` agrees with the tool on all 864 rows: 0 mismatches. The metadata
  key sets are equal on every row.
- **Rows whose `total` moved** are exactly the 26 rows with a group flag (b) in R15.
- **Rows whose `analytic_rad` moved** are exactly the 70 rows with an analytic-block flag (c) in
  R15.

**Work counters.** These are treated as counters, not values: the columns `WKB_Levin_num_regions`,
`WKB_Levin_evaluations`, `WKB_Levin_simple_regions`, `WKB_Levin_SVD_errors`,
`WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`, `WKB_Levin_max_depth`,
`numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`, and the metadata leaves `regions`,
`simple_regions` and `evaluations`.
- **Why.** Each counts what the integrator did, not a number it computed. The addendum expects
  evaluation counts to move only where halves are now solved.
- **They were compared anyway:** 0 of the 782 unflagged rows differ in any of them.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- all six exit 0, each with the store identical before and after, no journal, and no Ray process
  left;
- 0 Tracebacks;
- each prints one `!! WARNING: number of shards read from database (=4) does not match specified
  number of shards (=10)`, as 06's did.

**The lock** (`state/lock.log`): ten commands, each a shell command of its own, every one exit 0:
1. look;
2. acquire;
3. set-pid 96498 (R15);
4. look;
5. set-pid 97360 (R16);
6. look;
7. look;
8. release;
9. `status` (free, no Ray processes);
10. look.

### Observations not acted on

1. **R16 carries no phase-limited flag at all.** On the rehearsal scope the premise check turns
   off every phase-limited acceptance: 48 group flags on 26 rows, and 143 analytic-block flags on
   70 rows, all go to 0.
   - **The test's problem against production.** §2 test 2 shows the exit still fires on a
     constructed problem with $G_1/G_0 < 3$. No production row here exercises that case.
   - **What it bears on:** §5's descriptive "how often the exit fired", and W3.
   - It is not scored here.
2. **In the 26 rows whose total moved:**
   - `WKB_Levin_max_depth` is unchanged in every one;
   - Levin regions rose from 530 to 612, and evaluations from 1314 to 1638;
   - `total_abserr` fell in 24, by factors of 0.242–0.985, and rose in 2, by 1.002 and 1.012;
   - 7 went from unconverged to converged, and 19 stayed unconverged;
   - $|\Delta{\rm total}|$ is at most 0.36 × `total_abserr`$_{R15}$ and at most 0.597 ×
     `total_abserr`$_{R16}$.
   - In the 70 analytic-flagged rows, $|\Delta$`analytic_rad`$|$ is at most 0.748 × R15's analytic
     `abserr` (median 0.058).
   - Honesty is §5 W2's question, not §4's. These numbers are descriptive.
3. **`_three_bessel_Levin` computes `J_phase_limited` and `Y_phase_limited` and stores neither.**
   A reader of `analytic_rad` has to walk the per-block metadata (flag (c)) to learn that an
   analytic block was accepted on its floor. This is not a defect of 06b's.
4. **The readers' output file counts differ from 06's run on R14:**
   - `extract_Gk_data`: 468 against 396;
   - `extract_GkSource_data`: 180 against 176;
   - `extract_QuadSourceIntegral_data`: 76 against 70.

   `extract_Gk_data.py`, `extract_GkSource_data.py` and `extract_QuadSourceIntegral_data.py` each
   draw their subsample with unseeded `random.sample`, so the counts vary from run to run. Not
   investigated further.

### State handed on

- **Stores, kept:**
  - `var/datastores/levin-phase-floor-06b-r15{,-shard000[0-3]}.sqlite`;
  - `var/datastores/levin-phase-floor-06b-r16{,-shard000[0-3]}.sqlite`;
  - the final hashes are in `state/stores_sha256_final.txt`.
- **Runs:** `levin-phase-floor-06b-r15-parent-20261028T135349` and
  `levin-phase-floor-06b-r16-premise-20261028T135745`, both `done`. The registry holds 27 runs,
  all finished.
- **The lock is free,** and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing. The census
  agent may launch.
- **The trees.** Both worktrees are clean and unwritten. The main checkout is at `28ebaf0` and
  clean.
- **For the laptop:**
  - copy `var/levin-phase-floor-06b/rehearsal/` back without `extract/` (180 MB);
  - `out/store_value_diff.json` is 40.3 MB, and the orchestrator decides whether to commit it;
  - add `docs/levin-phase-floor/06b/rehearsal/scripts/` to `CLAUDE.md`'s `black` exemptions
    (deviation 10).
