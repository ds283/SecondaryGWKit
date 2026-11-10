# levin-phase-floor 07 Part B — the orchestrator's §B.3 review (Mac Studio, 2026-11-10)

**Verdict: PASS.** All nine §B.3 checks pass. Two predictions fail, V4 and V5, and the review re-scores both
failures exactly. Under prompt §8 neither is a stop condition, and under §B.4 they go to the user with the numbers.
The rest of this file gives each check with the commands I ran and what they printed. Everything named under
`review/` is in `var/levin-phase-floor-07/review/`. Nothing was committed, and no git command that writes was run.

Note B.7 item 5 widens the review's run checks to R29 and checks R28 only for being unchanged; the review follows
that. The earlier records this review rests on are `review/stop1.md`, `review/hold_before_census.md`,
`review/stop1p/notes.md` (stop 1′), `review/stop2.md`, `review/stop3.md`, `review/stop4.md` and
`review/xcheck/notes.md`.

## 1. Scope — PASS

- **The trees.** `git -C <tree> --no-optional-locks rev-parse HEAD` and `status --short --ignored`:
  - `SecondaryGWKit-lpf07-91472db` is at `91472dbe`;
  - `-7460035` is at `74600350`;
  - `-e182bc7` is at `e182bc7a`.

  All three are detached, as dispatched, and print 0 lines.
  - The main checkout is `handover-remedial` at `28ebaf07`, unmoved, and `status --short` prints 0 lines.
  - Its `--ignored` listing equals the one I took before dispatch (`b3/main_ignored_before.txt` against
    `b3/main_ignored_now.txt`: no difference).
  - `worktree list` shows exactly the four trees.
- **New files.** `find var docs prompts -newer review/dispatch_marker` (the marker is 2026-11-08T00:04:05Z;
  `b3/newer_than_marker.txt`) lists only Part B's own paths:
  - `var/levin-phase-floor-07/{census,rehearsal,review}`;
  - the R27, R28 and R29 stores (15 files);
  - the 07 run directories `r27-parent`, `r28-guard`, `r29-guard-rule-a`, `p1-references`, `k0`, `census` and
    `references`, and the three `*.launch.log` files.

  Nothing in `docs/` or `prompts/` is newer than the marker. Outside `var/`, `venv/` and `.git/` nothing in the
  main checkout is newer either.
- **The protected files.** `shasum -a 256 -c checks/protected_before.sha256`, run from the main checkout root,
  gives **181/181 OK** (`checks/protected_at_review.txt`). Those 181 files are:
  - A3 v2 (6 files);
  - R23–R24, the 06g stores (10);
  - R25–R26, the 06m stores (10);
  - the scratch copy `a3v2_copy` (5);
  - 06m's census run directory (150).

  The census agent's own before/after hashes of the scratch copy and of 06m's census agree (`census/out/cmd/96`–`99b`).
- **Observation, not a failure.** The main checkout's `.git` directory changed its mtime at 2026-11-10T02:38:33Z,
  and `.git/worktrees/SecondaryGWKit-lpf07-e182bc7` changed its mtime at 02:39:09Z. Both fall inside the census
  agent's step 10, between its `out/cmd/99b` and `101`.
  - No file in `.git` has a later mtime. The four indexes, `HEAD` and the refs are unchanged: the newest index is
    e182bc7's, from 2026-11-08T12:34:30Z.
  - The agent's logged git commands all carry `--no-optional-locks`.
  - The cause is not identified. No git state changed.

## 2. The runs — PASS, with one note

`./venv/bin/python -m runregistrykit list` prints "69 run(s) … 69 finished". Each manifest and status:

| run | state / exit / units | `git_head` in manifest | `git_dirty` | tree SHA named |
|---|---|---|---|---|
| `levin-phase-floor-07-r27-parent-20261108T000634` | done 0 | `91472dbe` | false | yes, the head |
| `levin-phase-floor-07-r28-guard-20261108T001916` | done 0 | `74600350` | false | yes, the head |
| `levin-phase-floor-07-r29-guard-rule-a-20261108T123429` | done 0 | `e182bc7a` | false | yes, the head |
| `levin-phase-floor-07-p1-references-20261108T103132` | done 0, 2/2 | `28ebaf07` | false | `purpose`: captured at 91472dbe, run from 74600350 |
| `levin-phase-floor-07-k0-20261108T135240` | done 0, 7/7 | `28ebaf07` | false | `purpose`: 91472dbe (b), e182bc7a (c) |
| `levin-phase-floor-07-census-20261108T171137` | done 0, 18,560/18,560 | `28ebaf07` | false | `purpose`: e182bc7a |
| `levin-phase-floor-07-references-20261109T005328` | done 0, 74/74 | `28ebaf07` | false | `purpose`: e182bc7a |

- Every manifest has `campaign` `levin-phase-floor`, `prompt` `07` and its slug in the run id.
- **The note.** The four driver scripts live under the main checkout's `var/`, so the registry's `git_head` for
  those runs is the main checkout's `28ebaf07`, not the worktree the code was imported from.
  - The tree each ran from is named in the manifest's `purpose`.
  - Every census unit's record carries `tree_head` `e182bc7a`.
  - Every reading's header carries tree g `e182bc7a`, clean.

  The check asks for "the SHA of the tree it ran from". The manifest gives it, but in `purpose` rather than in
  `git_head`. I judge that to meet the check; the user may read it otherwise.
- **The references run.**
  - It ended at 2026-11-10T02:31:07Z, state `done`, exit 0, 74/74.
  - Its stdout and stderr are both empty.
  - `checkpoint.jsonl` has 74 lines and 74 distinct units, each with `data.exit` 0, git_head `28ebaf07`,
    git_dirty false and script `6227aea4…`. Its SHA-256 is `40a9d871…cb0`.
  - Its wall time was about 25.6 h, against a projection of 89,230 s and the 48 h backstop.

## 3. The fingerprints — PASS

`PYTHONPATH=. venv/bin/python -m runregistrykit store fingerprint <primary>`, run in tree g without `--write`
(`b3/fingerprint2_r2{7,8,9}.txt`), gives:
- R27: `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, exit 0;
- R28: the same, exit 0;
- R29: the same, exit 0.

**My own slip, with no effect.** My first attempt ran from the main checkout. Its code at `28ebaf07` does not
declare the `theta_residual` columns, so all three exited 1 with "nothing was read" (`b3/fingerprint_r2*.txt`).
The 15 store files hash the same before, between and after both attempts, and no `-wal`, `-shm` or journal file
appeared (`b3/stores_before_fp.sha256`, `b3/stores_after_fp*.sha256`, `b3/stores_sidefiles_*.txt`). Tree g's
`status --ignored` stayed empty.

## 4. The rehearsal values — PASS

- **`tools.store_value_diff`.** I re-ran it in tree g:
  `env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. venv/bin/python -B -m tools.store_value_diff R27 R29 --json`.
  - It exited 1, which means "differences found", with empty stderr.
  - Its output is **byte-identical** to the agent's `rehearsal/r29/out/store_value_diff_r27_r29.json`, SHA-256
    `c511b582…` in both.
- **Outside `QuadSourceIntegral`, every value table is identical:**

  | value table | rows identical | rows differing |
  |---|---|---|
  | BackgroundModelValue | 1,159 | 0 |
  | TkNumericValue | 1,478 | 0 |
  | TkWKBValue | 2,209 | 0 |
  | GkNumericValue | 35,874 | 0 |
  | GkWKBValue | 147,642 | 0 |
  | GkSourceValue | 169,653 | 0 |
  | QuadSourceValue | 2,447 | 0 |

  - The integration rows differ only in timing, `label` and `parents_digest`, plus GkWKB's 28 `rho_reuse`
    rows of `metadata` and `stage_1_RHS_evaluations`.
  - The agent's 05 `classify_non_qsi.py` leaves 0 unclassified.
- **Inside it, by my own reader.**
  - `review/scripts/qsi_guard_compare.py` (`907fafae…`, stop 1's reader), run on R27 and R29
    (`b3/qsi_guard_r27_r29.txt`), exits 0.
  - 864 rows in each store. Of these, **692 are untouched (every count 0) and all 692 are bit-identical** in every
    field but the counts.
  - The other 172 rows hold 44 group and 340 analytic records with a nonzero count. Of these:
    - 128 have `total` bit-identical;
    - 168 hold an analytic call that located a point;
    - `guarded_regions` sums to 4 (the four O9 blocks), and `guard_at_depth_max` to 0.
  - The largest |Δtotal| is 73,773× R27's `total_abserr`, at (10⁶)³, z = 2284.25.
  - 17 rows gain convergence and none loses it.
- **The tabulation is complete.**
  - My 172 rows are exactly the agent's 172 in `rehearsal/r29/out/qsi_rows_r27_r29.csv`.
  - For all 172, total and abserr are equal in hex in both stores (`b3/qsi_tabulation_match_and_spot.txt`).
- **The spot check.** I read the moved row (10⁶)³, z = 2284.251956909201, straight from each store's shard 1 with
  sqlite in read-only mode. R27 holds total `0x1.374cf5c5ecc48p-69` and R29 holds `0x1.393a67dec64bdp-69`, each
  equal to both tables.
- **S0 stands.** R27 is unchanged: its five files equal the stop-1 hashes, and its fingerprint is
  `1b332ffa…`. S0 was established by my own comparison at stop 1 (`stop1/vdiff_r26_r27.txt`,
  `stop1/qsi_full_r26_r27.txt`).
- **R28 is unchanged:** its five files equal `stop1/stores_before_fp.sha256`, and its fingerprint is `1b332ffa…`.
  It is not re-reviewed (B.7 item 5).

## 5. S1, S2, the readers and the lock — PASS

- **S1, S2 and the readers on R29** were checked by my own reads at stop 1′ (`stop1p/notes.md` items 5–8;
  `stop1p/replay_s1_s2_recheck.txt`):
  - **S1:** 12 of 12 honest, the worst 0.2176 (w11). Every call equals R28's.
  - **S2:** R29 at 0.02025 and 0.2632 of its error; R27 at 5.6e5 and 4.1e4.
  - **The readers:** 6 exit 0, 6 store-identical, 0 Tracebacks. R29's five files equal their pre-comparison
    hashes, and they still do now.
  - S1 and S2 on R28, and R28's readers, were checked at stop 1 in the same way (`hold_before_census.md`).
- **The lock transcripts.** Every command exits 0, and each driver's pid is recorded:
  - `rehearsal/state/lock.log`: `acquire`; `set-pid` 83008 (R27) and 83825 (R28); `release`; `status` free;
    `ray-processes` none.
  - `rehearsal/r29/state/lock.log`: `acquire`; `set-pid` 14528 (R29); `release`; free; none.
  - `census/state/lock_transcript.txt`, census: `acquire`; `set-pid` 28799; `release`; free.
  - The same file, references: `acquire`; `set-pid` 82448; `release`; free.
  - Its `status` lines print "Ray processes: no temp dir given", because `status` was run without
    `--ray-temp-dir`. The `ray-processes --ray-temp-dir /tmp/ray-sgk` checks were run as commands of their own
    (`census/out/cmd/41`, and the hand entry of 2026-11-10), exit 0.
- **My own commands,** each a shell command of its own (`b3/lock_review.txt`):
  - `rehearsal_lock status --ray-temp-dir /tmp/ray-sgk`: "lock: free (… does not exist)", exit 0;
  - `rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk`: "Ray processes of /tmp/ray-sgk: none", exit 0.

## 6. V0 — PASS

- **The saved outputs:**
  - (a) is `census/out/v0/v0a_imports.txt` (14:06:31Z);
  - (b) is the K0 comparison inside the `k0` run, 100 of 100 equal (`census/out/k0_results.json`);
  - (c) is `census/out/v0c/compare_det.json` (14:05:40Z).
- **Order:** the `k0` run ended at 14:05:41Z, before the census's `manifest.json` at 17:11:37Z.
- **My own re-runs at planned stop 2** (`stop2.md`, `stop2/`):
  - two K0 units at the parent against 06m's census, with 0 unclassified differences;
  - one stationary-point unit at g, run twice, deterministic.

## 7. The census verdicts, re-scored by my own readers — PASS (V1–V3 hold; V4 and V5 fail)

The census checkpoint is `03dfe786…` and 06m's is `ea5bc3ab…`. Each verdict below is mine, and each agrees with
the agent's.

- **V1 holds.**
  - **Reader:** `review/scripts/stop3_compare.py` (`a8c322c1…`), with its two supplements.
  - **Calls:** 289,482 in all. 284,514 locate nothing and guard nothing; **0 of them differ** from 06m's, at
    both rtols.
  - **Rows:** 16,104 units contain only such calls. All are equal whole, with `total` and `total_abserr` equal in
    hex.
  - **Agent:** 142,257 calls per rtol, and 8,052 such units per rtol.
- **V2 holds.**
  - **Reader:** `review/scripts/v235_rescore.py` (`7bfe605a…`; `b3/v235_rescore.txt`). It reads 1,945 region
    records, each checked by SHA.
  - **Population:** 182,066 scanned calls: 37,325 group calls at each rtol, and 107,416 analytic calls at 1e-7.
  - **Result: 0 Levin leaves hold a sign change.**
    - My grid test equals the scan's own grid listing, leaf for leaf, in every call.
    - The only grid-listed leaves are 168 Clenshaw–Curtis leaves, none a window and none guarded.
    - The scan's "ends" listings number 2,566, none of them Levin. The agent's classes of these are 1,514 + 526 +
      526 windows.
- **V3 holds.**
  - **0 violations** among 170,582 scored calls.
  - Of these, 2,954 have a sign change. Each has a located root within h, and every root lies within h of a
    sign change.
  - No point of kind `end` occurs.
  - **§10's set-aside:** 11,484 calls. 168 of them show a sign change, all analytic, and 14 of those are guarded
    (O9's).
  - The agent's "3,584 have a sign change or a located root" is the number of scored calls its summary kept: those
    with a sign change, any located point including minima, an unlocated minimum or a listed leaf. Its prose label
    is loose; the verdict does not move.
- **V4 FAILS.**
  - **Reader:** `review/scripts/v4_rescore.py` (`4349a916…`; `v4/rescore.txt`, `v4/rescore.json`). It uses
    exact Fractions: census doubles from hex, readings from their 30-digit `value_30`.
  - **All 74 readings are usable:**
    - the tool is `d37e633b…`;
    - each input pickle's SHA-256 matches its reading;
    - each is Route B;
    - every non-monotone segment is `direct`;
    - each `value_30` rounds to its `value_hex`.
  - **Every call's readings agree.** The largest spread is 0.0039 of the tolerance (v08). So all 33 calls are
    scored.
  - **The 24 calls:** 19 honest and 5 not:

    | call | unit | ratio for 07 | ratio for 06m |
    |---|---|---|---|
    | v08 | `g-1e-7-1-38834` | 3.16 | 3.0e4 |
    | v09 | `g-1e-8-3-34962` | 2.21 | 639 |
    | v12 | `g-1e-8-0-14299` | 3.08 | 9.1e3 |
    | v17 | `g-1e-7-0-1772` | 2.38 | 497 |
    | v22 | `g-1e-8-3-14096` | 2.53 | 1.67e4 |

  - **The other nine:** z00 (0.718), z12 (0.718) and Z3's seven (≤ 0.0021) are honest on 06g's readings.
  - **Against the agent:** the same verdicts, the same basis (four readings for 13 calls; FCC r1 and r2 for 11) and
    the same ratios for all 33, with a relative difference of 0.
  - **What the five failures share,** each confirmed from the checkpoint:
    - each converged at g, with 0 leaves at `depth_max` and a maximum depth of 6–14;
    - each located one root, and has one window stopped by §1.3's `iroff1` at 6, with 4–13 leaves still falling;
    - each is class (c) in the cross-check.
  - **The stationary-point calls before:** 06m is honest on 17 of the 33 (v00–v03, v06, v11, v16, v21, v23, z00
    and Z3's seven). On the other 16 it is 356–5.2e5× its abserr from the reference, with z12 at 1,681×.
- **V5 FAILS.**
  - **Reader:** `v235_rescore.py`, over 4,968 calls that locate a point or guard a leaf.
  - **23 violations,** all `G+q+r`: 10 at 1e-7 and 13 at 1e-8.
  - Their leaves at `depth_max` number 2–54. Three are converged. 0 units are over budget.
  - This is a finding, not a fault. It reopens G2's L2.
- **The descriptive items are present** in `census/log.md` and in `census/out/scores/`: depth, windows, the
  stationary-point calls before, totals in the 720, locations, convergence with 06m's flag-cost table beside it,
  §1.3's firings, and cost.

## 8. The predictions — PASS

- `census/state/predictions.md` has SHA-256 `92e22911…`, unchanged since stop 2.
- **Its mtime is 2026-11-08T14:10:10Z.** That is later than V0's outputs (14:05:40Z and 14:06:31Z) and earlier
  than the census `manifest.json` (17:11:37Z).
- **It quotes the prompt word for word,** which my reader confirms:
  - prompt §6's "Predictions, fixed before the run", V1–V5, as one exact substring of 2,596 characters;
  - prompt §10's last bullet on V3.
- It fixes V4's 24 calls in its part 3 and `state/v4_selection.json` (`15a4391b…`).
- **The harness diff against 06m's copy** (`diff -rq`; `b3/harness_diff_now.txt`):
  - Eight copied scripts were changed: `check_captures`, `prov`, `refs_common`, `run_06j`, `run_census`,
    `run_k0`, `worker` and `x1_capture`.
  - 21 scripts are new.
  - Every one of the 29 is named in `census/log.md`'s deviations (1, 8, 15, 18, 24, 28 and 30–35).

## 9. The records — PASS

- **Each `log.md` follows campaign README §5.1,** with all six headings:
  - `rehearsal/log.md` has deviations 1–15;
  - `rehearsal/r29/log.md` has 16–24;
  - `census/log.md` has 1–40.

  Every numbered deviation is classified (`STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`).
  In the census log there are 5, 20 and 13 of each, with deviations 10 and 37 classified on a wrapped line.
- **Each `verification.md` exists,** written as its part of `VERIFICATION.md` §7; the census's is §7.C, with
  `07/census/…` links.
- **Each `issues.md` exists.** The census's proposes four items and one observation, and decides none:
  - the `iroff1` stop leaving five converged calls outside their error;
  - V5 reopening G2's L2;
  - O9's 28 guarded calls, plus 154 unguarded calls with a skipped phase and a sign change;
  - the 786 class (c) calls;
  - the observation: lone-`G` minima.
- **The `KEPT_ON_MAC_STUDIO.txt` lists,** checked by `review/scripts/kept_check.py` (`3053c626…`;
  `b3/kept_check.txt`):
  - **Census:** 312 entries. Every one exists with its listed SHA-256 and size. None is missing: every pickle and
    every file over 2 MB under `census/` is listed, and so is every file of the census, `k0` and `references` run
    directories.
  - **Rehearsal:** 44 entries, all OK, none missing.
  - **R29:** 40 entries, all OK.
    - Its 12 "unlisted" files are symlinks under `rehearsal/r29/state/stores/r2{7,8}/` to the R27 and R28 store
      shards, which the rehearsal list and the stop-1 hashes hold.
    - They are aliases, not files of their own.

## The report

- **Fingerprints:** R27, R28 and R29 all `1b332ffa…`. S0 holds: R27 equals R26 in every value table.
- **R29 against R27:**
  - 864 rows: 692 untouched and bit-identical, 172 moved or located.
  - The largest change is 73,773× R27's `total_abserr`.
  - 17 rows went from unconverged to converged, and none the other way.
  - 4 guarded analytic blocks, O9's case.
  - Summed regions: R29 114,460, R28 227,883, R27 114,826.
- **S1:** 12 of 12 honest, the worst 0.218. **S2:** 0.020 and 0.263 at R29; 5.6e5 and 4.1e4 at R27.
- **V0** holds; **V1, V2 and V3** hold; **V4** fails (5 of the 24 are not honest, at 2.2–3.2× their abserr); **V5** fails (23 calls).
  The numbers are in check 7.
- **Depth (rows at 20):** 77 against 06m's 192 at 1e-7, and 83 against 213 at 1e-8. In the 720: 24 against 139,
  and 27 against 157.
- **Over budget:** 0 units.
- **Cost:**
  - census wall 22,708 s against 06m's 18,465 s;
  - group-call regions 952,275 / 1,065,288 against 939,882 / 1,053,755.
- **The analytic calls located** 1,514 roots per rtol (757 J4, 757 Y4). 14 per rtol were guard-only (O9), and none
  is unconverged.
- **The cross-check** (interim, `review/xcheck/notes.md`):
  - 843 calls disagree with 06m beyond the stated error;
  - (a) 31, (c) 786, (e) 26;
  - r_g ≤ 1 in 2,482 of 2,484 pairs.
