# 07b Part C — the Mac Studio orchestrator's review (§C.2), 2026-11-18

**Verdict: pass.** Every §C.2 check passes, and my own readers reproduce every scored figure the agents report.
- R31 holds S0′–S4.
- U0–U8 all hold. U4, U6 and U7 hold as §9.5 predicted.
- No §8 or §9.7 stop condition came up.

One descriptive figure of the census agent's is defined differently from what its wording implies (below, "Descriptive,
checked"). It is not scored, and I have not repaired it.

The records of each step are in this directory:
- `stop1/`, `stop2/` and `stop3/`, each with a `notes.md`;
- `final/`;
- `checks/` (the protected-file lists).

Every script is my own, written for this review and independent of the agents' scripts. Floats are compared by
`float.hex` or as exact Fractions. Every command used `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`
or `-I -B`.

## The user's words in Part C, quoted as relayed by the laptop

- **Stop 1:** "Yes, continue, and settle which store understates".
- **Stop 2:** "Yes, launch the census".
- **Stop 3:** "Yes, score U1–U7".

## 07's §B.3 checks 1–3 and 8–9, with R31

1. **Scope** (`final/state_at_review.txt`; `stop3/state_check.txt`). **Pass.**
   - The worktree is at `0a103b5a`, and `git status --short --ignored` is empty in it.
   - The main checkout is at `28ebaf07`, with `status --short` empty.
   - `find var -newer MARKER_dispatch` (marker touched at 19:14:22Z on 11-16) lists nothing outside §C.1's paths and
     `review/`.
   - The protected files are unchanged by hash at each stop and at the end:
     - 159 files: R30's store, 07a's census run directory and the scratch copy;
     - Part A's `measurement/`, 7,645 files;
     - R29, 5 files.
   - The lock directory does not exist, and the lock is free.
2. **The runs** (`final/runs_check.txt`). **Pass.**
   - `runregistrykit list` shows 86 runs, all finished.
   - R31, the R31 replay, the R31 references, k0 (11/11) and the census (18,560/18,560) are each `done`, exit 0.
   - Each manifest names campaign `levin-phase-floor` and prompt `07b`, with `git_dirty` false.
   - R31's manifest names `0a103b5a`. The others ran with cwd under the main checkout, so they name `28ebaf07`, as 07a's
     did.
3. **The fingerprint.** **Pass.**
   - R31 is `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`.
   - I re-ran it without `--write` at stop 1 and again here (`final/fingerprint_r31.txt`), from the worktree with
     `PYTHONPATH` set to it.
   - R31's five store files equal the rehearsal's KEPT list (`final/r31_files_check.txt`).
8. **The predictions.** **Pass.**
   - `census/state/predictions.md` is `0c6055ac…`, mtime 2026-11-16T21:03:31.28Z.
   - That is after every U0 output and before the census manifest (01:02:00Z on 11-17). It is unchanged at the end.
   - It quotes prompt §6's, §9.5's and §9.7's blocks verbatim (`stop2/predictions_check.txt`).
9. **The records.** **Pass.**
   - Each `log.md` has §5.1's six sections; the census log adds a descriptive section.
   - Every deviation is classified:
     - the rehearsal's 26;
     - the census's 23: 3 STRUCTURALLY REQUIRED, 18 IMPLEMENTATION CHOICE and 2 UNINTENDED DRIFT entries (items 14
       and 15).
   - `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt` exist for both.

## The rehearsal (stop 1 and after; `stop1/`, `stop2/notes_understates.md`)

**S0′ holds.**
- Outside `QuadSourceIntegral`, every value table of R31 equals R30's.
- The only differences are timing, `label`, `GkSource.parents_digest` (291 rows) and rho reuse (23 rows).
- My `tools.store_value_diff` JSON equals the agent's (`b7d508c1…`).

**Inside it,** of 864 rows:
- 569 moved;
- `total_converged` went 862 → 861, the one being (1e5, 1e6, 1e6) at z = 144627.7, G+q+r, with 2 → 143 regions;
- the largest move is 0.278 of abserr₃₀.

| check | result |
|---|---|
| S3 (exact) | holds: 0 rows beyond the bound, largest 0.128 |
| S1 | holds: 12/12 honest, worst 0.067 |
| S2 | holds: 0.0202 and 0.111 |
| S4 | holds: 4 blocks each locate their root, guard nothing, and are honest at 4.4e-5 to 6.2e-4 |
| replay | 452/452 bit-identical |

**The seven records the user asked about:** R30 understates in all seven, at 2.91–59.29 of its abserr. R31 is honest in
all seven, at 0.0014–0.298. All 14 calls are converged.

## U0–U8, re-scored from the two checkpoints by my own readers

Both census checkpoints were read through my own indices (`stop3/index07{b,a}.json`): 07b's at `396784bc…` and 07a's at
`461df2ec…`. U7's population was read from 07's checkpoint, `03dfe786…`. Every line was checked by its SHA-256, and every
region record against its `pickle_record_sha256`.

**U0 holds** (`stop2/`).
- My parent re-runs of `g-1e-7-1-35492` and `g-1e-8-0-12617` equal 07a's census in every value.
- My two runs of `g-1e-8-2-34906` at h, a unit where the edge rule acts, are identical.

**U1 holds** (`stop3/compare_07b_vs_07a.*`, sha `47de0084…`).
- 60,160 untouched calls: 29,758 at 1e-7 and 30,402 at 1e-8.
- All equal 07a's in value, abserr, `converged`, `num_regions` and `max_depth`, and in every other path of their call and
  region records.
- 5,688 rows are untouched in every call. All have `total` and `total_abserr` equal in hex.

**U2 holds** (`final/u_rescore.*`, sha `63de9138…`).
- 182,066 scanned calls.
- 0 returned Levin leaves hold a sign change by my grid test, and my grid sets equal the scan's in every call.
- 0 `ends` listings of Levin leaves.

**U3 holds.**
- 0 violations, both ways, on all 182,066 scanned calls; 3,122 of them have a sign change or a root.
- 784 skipped phases, all `zero_rate`, none with a sign change.
- No points of kind `end`.
- The agent scores 23,998 calls with 0 violations, by its own selection; both readings hold.

**U4 holds, as predicted** (`final/u4u6_v2.*`, sha of `u4u6_v2.py` in `final/sha256.txt`; `u4u6.*` is the same without
the four pair calls).
- 190 calls:
  - Part A's v00–v23 and cal_z12, R1 (60), R2 (60), R3 (24), the four pair calls, and item 3;
  - 06g's references for z00 and Z3's seven;
  - 07a's references for 1-34599 and 3-678 (8).
- 0 are converged and not honest.
- Unconverged: v04, R2-28 and R2-28~o. R2-28 is at 1.1071 of its abserr, as §9.5 predicted.
- The largest ratio among converged calls is 0.805 (R2-08~o).
- 6 calls' readings do not agree within min(abserr)/100: R2-27, R2-29 and R3a-0-33666-a13, each at both rtols. All six
  are honest at ≤ 0.035 either way. The agent reports them as not scored.
- v08, v09, v12, v17 and v22 are at 0.032, 0.0011, 0.044, 0.0042 and 0.0015 (07a: 2.2–3.2).

**U5 holds.**
- 0 Clenshaw–Curtis regions have a Levin half.
- 0 gated solves take the node-partition sum.
- `num_guarded_regions` is 0 in every call.
- **Descriptive:** 18,238 forced partitions in 4,416 calls (07a: 18,236).

**U6 holds, as predicted.**
- 24 of 24 are honest against S10, all converged, at 1.8e-5 to 0.828.
- The four dishonest at 07a (0-12977 #5, 1-2174 #2, 2-14345 #3, 2-34473 #3) are now at 0.049, 0.0013, 0.83 and 0.00014.

**U7 holds, as predicted.**
- 2,484 pairs, none unpaired.
- r_g ≤ 1 in all 2,484 at h (2,482 at 07a). The largest is 0.467 (3-34923).
- 3-14501 is at 0.00046 (07a: 1.385), and 3-35006 at 0.00057 (07a: 1.084).

**U8 holds** (`stop3/u8_check.txt`): 187 of 187 equal (h)'s replay records.

## Where the fix acted, and convergence against 07a's (descriptive, checked)

- **Classes** (equal to the agent's):

  | rtol | kind | untouched | range only | edge |
  |---|---|---|---|---|
  | 1e-7 | group | 16,274 | 20,711 | 340 |
  | 1e-7 | analytic | 13,484 | 93,932 | 0 |
  | 1e-8 | group | 16,651 | 20,340 | 334 |
  | 1e-8 | analytic | 13,751 | 93,665 | 0 |

- **Refusals:**
  - edge: 438 (1e-7) and 430 (1e-8), all in group calls;
  - range at 1e-7: 53,794 group and 351,789 analytic;
  - range at 1e-8: 53,162 group and 323,256 analytic.
- **Convergence:**
  - 198 group calls are newly unconverged: 97 at 1e-7 (95 range, 2 edge) and 101 at 1e-8 (99 range, 2 edge);
  - 5 group calls at 1e-8 are newly converged;
  - no analytic call changes;
  - rows' `total_converged`: 139 lost (68 / 71) and 4 gained.
- **abserr against 07a's,** h over 07a:
  - untouched calls are exactly 1;
  - range analytic: median 0.0026, maximum 4.57e5;
  - range group: median 0.48, maximum 250;
  - edge: median 0.043, maximum 7.1.
- **Values moved:**
  - by more than 07a's abserr: 2,672 group calls (1,397 / 1,275), equal to the agent's;
  - **by more than the sum of both abserrs: 326 group calls, 324 of them converged under both** (`final/moves.txt`).
    The largest is 0-14199 #1 G+q+r, at 75 times 07b's abserr.
  - **The agent's 605 (603 converged)** count calls that moved by more than each abserr separately (its
    `summarise_07b.py` line 247: `> abserr_07b` among those `> abserr_07a`).
  - Its `issues.md` item 2 reads them as calls where "one of the two declarations does not bound its error". That
    follows only for the 324, where the move exceeds the sum. A move beyond the sum is beyond each abserr, so the 326 are among the 605. For the other 279, both can be honest. Recorded
    here; not repaired.
- **Cost:**
  - census wall 45,016 s against 23,284 s (1.93×); the projection was 37,616 s;
  - summed unit wall 52,154 / 54,667 s against 24,952 / 28,513 s;
  - regions 6,039,639 against 2,909,134 (2.08×);
  - no unit over budget.

## Not re-checked by me

- The agent's edge-rule details: depths, median q, the 2,802 analytic-window tests, and "never on a minima window".
- The depth-20 leaf descent analysis for the 198 newly unconverged (192 with every depth-20 leaf below a refused leaf).
- The band and k breakdown, and the per-07a-class acted fractions.
