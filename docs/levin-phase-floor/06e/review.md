# levin-phase-floor 06e, Part B — the orchestrator's review (Mac Studio)

**Verdict: PASS. Part B is complete, every review check holds, and no §7 stop condition arose.**
Of the predictions:
- **Z0, Z1 and Z4 hold.**
- **Z2 and Z5 fail at 1e-8,** both on one call, `g-1e-8-0-33995` call 5.
- **Z3 fails on one counted reference,** sample 3, at 1.245 × abserr$_g$. 06c's code was also
  dishonest there, at 1.197 × abserr$_{06c}$.

These failures are scored results under §6 acceptance item 3 ("scored holds, fails or
undecided"), not stops. What follows from them is for the user.

Written 2026-10-31 on mps026298 by the orchestrator session that dispatched the two Part B agents.

**The trees:**
- parent `50277f3` (worktree `SecondaryGWKit-lpf06e-50277f3`);
- code / $g$ `6d81fad`, whose code is `f4d8e17` (worktree `SecondaryGWKit-lpf06e-6d81fad`);
- the main checkout, `handover-remedial` at `28ebaf0`.

**Where things are:**
- The briefs are `review/dispatch_*.md` (`review/dispatch_sha256.txt`).
- The preflight is `review/preflight/`.
- Every command of this review and its output is in `review/checks/`.
- My readers are `review/scripts/` (`orch_rescore.py`, `orch_z3.py`, `orch_z1.py`).
- My Z3 re-run is `review/z3_rerun/`.

Every float comparison below is exact, on the recorded doubles (`float.hex` / `Fraction`).

## 1. Scope

| Check | Result | Record |
|---|---|---|
| Main checkout | `28ebaf0`, `handover-remedial`, `git status --short` empty, never moved | `checks/scope_final.txt` |
| Worktrees | at `50277f3` and `6d81fad`; `status --short --ignored` empty (no `__pycache__`); nothing newer than the dispatch marker (07:42:52Z 2026-10-30) | `checks/scope_final.txt` |
| `var/` against the preflight listing (63,062 files) | 0 files gone or changed (size or mtime); 174 new, all in the R19/R20 stores and the 06e runs. Files newer than the marker outside `var/levin-phase-floor-06e/`, those stores and those runs: none | `checks/var_list_diff.txt`, `checks/var_newer_outside_allowed.txt` (empty), `checks/scope_final.txt` |
| A3 v2 | its five files and manifest have the same size and mtime as at preflight (`ls -l`, whitespace-normalised: GNU against BSD padding); no journal, `-wal` or `-shm` | `checks/a3v2_listing_compare.txt` |
| A3 v2's scratch copy | SHA-256 of all five files identical to the preflight's (= `copy_files_certified.txt`); listing identical; no journal | `checks/a3v2_copy_after.sha256` |
| 06c's census checkpoint | `b61cb034c1ee9521…` before and after | `checks/06c_checkpoint_after.sha256` |
| R19/R20 store files | identical to the rehearsal agent's final hashes, and unchanged by my comparison | `checks/sizes_and_store_hashes.txt`, `checks/stores_{before,after}_diff.sha256` |
| The `lpf06c-*` worktrees and earlier prompts' `var/` | untouched (covered by the `var/` listing; nothing was run from them) | — |
| Registry | 35 runs, all finished, none running or stale | `checks/registry_list.txt` |

## 2. The runs

| Run | State | Manifest |
|---|---|---|
| `levin-phase-floor-06e-r19-parent-20261030T074440` | `done`, exit 0 | campaign `levin-phase-floor`, prompt `06e`, `git_head` `50277f3`, `git_dirty` false, launcher `b0135218` |
| `levin-phase-floor-06e-r20-endnodes-20261030T075034` | `done`, exit 0 | `06e`, `git_head` `6d81fad`, `git_dirty` false, launcher `b0135218` |
| `levin-phase-floor-06e-census-20261030T081318` | `done`, exit 0, 18,560/18,560 | `06e`, `git_head` **`28ebaf0`**, `git_dirty` false, script `a6d9bac4` |
| `levin-phase-floor-06e-references-20261030T192730` | `done`, exit 0, 4/4, "s15-P64 exit 0" | `06e`, `git_head` **`28ebaf0`**, `git_dirty` false, script `77dcfdd9` |

**Reading — the manifests' `git_head`.** The census and references manifests record the main
checkout's `28ebaf0`, because their launchers ran from `var/levin-phase-floor-06e/census/`. 06c's
census manifest does the same (`28ebaf0`, cwd `var/levin-phase-floor-06c/census`). The tree the
units ran is recorded per unit:
- all 18,560 records carry `tree` `g`, `tree_head` `6d81fad`, `tree_status_short` empty, and
  `script_sha256` `a6d9bac4`;
- every record loaded its 102 own-tree modules with none foreign;
- each of the 15 key module files it lists is under the code worktree (`checks/census_record_provenance.txt`);
- each reference output's `tree` line reads `6d81fad … status lines 0`.

So provenance holds. The manifest field names the launcher's checkout, not the tree.

## 3. §4 — R19 and R20

- **Fingerprints, re-run myself.** `runregistrykit store fingerprint` without `--write`, from the
  code worktree, gives `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` for both
  R19 and R20, with problems none. That is R8's (`checks/fingerprint_r19.txt`, `fingerprint_r20.txt`).
  `config/` and `runregistrykit.toml` are the same at `28ebaf0` and `6d81fad`.
- **The value comparison, re-run myself.** `tools.store_value_diff --json`, from the code worktree,
  gives a JSON byte-identical to the agent's (`cdafa6fcb04c9070…`), exit 1 because
  `QuadSourceIntegral` differs.
  - 05's `classify_non_qsi.py` (`26690e61`, equal to 05's copy) on my JSON gives output identical
    to the agent's, with 0 unclassified (`checks/classify_non_qsi_orch.txt`).
  - Outside `QuadSourceIntegral` every value table is identical.
  - The other differences are labels, timings, `parents_digest`, and `rho_reuse` in 32
    `GkWKBIntegration` rows.
  - No row is in one store only.
- **Z1, re-scored by my own reader.** `review/scripts/orch_z1.py` uses plain `sqlite3`, `mode=ro`,
  no package code, matched by physical key (`checks/z1_rescore.txt`). **Z1 holds.**
  - 864 rows scored; 0 non-finite; 0 violations.
  - `total` is bit-identical in 361 rows; 503 moved.
  - Largest $|\Delta T|/(e_{19}+e_{20})$ = 0.00312 ($|\Delta T|/e_{19}$ = 0.00624), at
    $k=3.16\times10^5$, $q=10^5$, $r=3.16\times10^5$, $z=437173.17$. There T19 =
    `0x1.49a315251da54p-54` and T20 = `0x1.49a31528f01e6p-54`. The agent's CSV row has the same two
    hex totals (spot-check).
  - `total_converged` 863 → 864; `WKB_Levin_num_regions` 34,892 → 31,216; rows at depth ≥ 12:
    46 → 11. All three equal the agent's.
- **The readers:** six, every one exit 0 (`rehearsal/out/readers_check.txt`). R20's files are
  identical before and after.
- **The lock:**
  - `rehearsal/state/lock.log`: acquire, `set-pid` 98917 (R19), `set-pid` 99675 (R20), release, all
    exit 0.
  - My own `status` and `ray-processes`, each its own command appending by `>>`, at 10:59Z and
    11:38Z: free, none, exit 0 (`checks/lock_after.txt`, `ray_processes_after.txt`, `lock_final.txt`).

## 4. §5 — the census

- **Z0, my own checks** (`checks/z0_independent.txt`, `checks/z0_clause1_and_predictions.txt`).
  **Z0 holds.**
  - **Clause 1:** `git diff --stat e5764a6 50277f3 -- . ':!docs' ':!prompts'` shows only
    `CLAUDE.md`. The parent's 102 own-tree modules are all under the parent worktree.
  - **Clause 2, independently:** Z0's P3 (`p-1e-8-0-1971`, `tree_head` `50277f3`, clean) against
    06c's checkpoint unit `g-1e-8-0-1971`. 383 fields compared, 70 of them by `float.hex`, 0
    differing. `total` is `0x1.6ad248e2d3adap-66` and `total_abserr` `0x1.350e5d9cc9dc3p-92` in
    both.
  - **The copy's fingerprint** `0e32f24a…` was taken before and after the run by the agent
    (`census/out/cmd/15`, `16`, `42`). Its files are identical to the certification at my check.
- **The predictions file**, `census/state/predictions.md`:
  - its sha is `f27771cc…`;
  - its mtime, 08:12:47Z, falls between Z0's last output (08:10:02Z) and the census manifest
    (08:13:18Z);
  - its Part A is byte-identical to prompt 06e lines 345–375 at `6d81fad` (sha `42d8143f…`,
    `checks/predictions_partA_vs_prompt.txt`).
- **Z2, Z4 and Z5, re-scored by my own reader** (`review/scripts/orch_rescore.py`;
  `checks/z2_z4_z5_rescore.txt`) from the two checkpoints, with 06a's 1e-7 scans giving the 720.
  My counts equal the agent's on every item:

  | | 1e-7 | 1e-8 | Verdict |
  |---|---|---|---|
  | **Z2** calls outside the 720 at `depth_max` (06c: 84 / 133; inside the 720, $g$: 11 / 17) | 0 | **1**: `g-1e-8-0-33995` call 5 (G−q+r), depth 20, unconverged | **fails** |
  | **Z4** rows scored / non-finite / violations | 8,560 / 0 / 0; max ratio 0.0099 | 8,560 / 0 / 0; max ratio 0.042 | **holds** |
  | **Z5** 06c-unconverged calls outside the 720 / not converged under $g$ | 85 / 0 | 173 / **1** (the same call: 8,706 regions, depth 20) | **fails** |

- **Z3, re-scored by my own reader** (`review/scripts/orch_z3.py`, `checks/z3_rescore.txt`). It
  reads the 13 samples' reference files and both checkpoints. **Z3 fails.**
  - Six count and seven are dropped, the same as the agent's.
  - The violator is **sample 3** (`g-1e-7-3-34370` call 4, G−q−r, converged at depth 2), at
    1.24505 × abserr$_g$:
    - $v_g$ = `0x1.2549f248a5a85p-88`, abserr$_g$ = `0x1.cb647a7f43ea2p-112`;
    - the reference, S128(P64), is `0x1.2549f00caea6ep-88`, with scatter 0.0083 of min(abserr);
    - 06c is at 1.197.
  - The other counted samples are 2 (0.998), 8 (0.9995), 9 (0.998), 14 (0.998) and 15 (0.781).
  - For every sample run at $g$, the driver's replay equals $g$'s census call bit for bit.
- **Z3, my own reference at a different resolution pair.** Sample 2 (`g-1e-7-2-1627` call 3, counted)
  ran by `orch_ref.py` (`8a0d291d`) at $g$ on the agent's `inputs/s02.pkl` (`0100f98a…`). It used
  P = 96 with 80/112 nodes, a pair the agent did not use, six workers, nothing else computing:
  11:04:33–11:35:26Z, exit 0 (`review/z3_rerun/`, `checks/z3_rerun_s02.txt`).
  - S80 = `-0x1.82749f74cd6b9p-95` and S112 = `-0x1.82749f74cad11p-95`. Both are within
    2.3 × 10⁻⁶ min(abserr) of the agent's reference.
  - The range over all five resolutions is 5.0 × 10⁻⁶ min(abserr).
  - $|v_g - S112|/{\rm abserr}_g$ = 0.998476, against the agent's 0.998477.
  - The replay equals $g$'s census call.
  - The agent's reference for this sample is confirmed.
- **Over budget:** no unit at either rtol, and no `wall_s` over 300 s. The over-budget pass was
  not needed.
- **Cost** (my reader):

  | Unit wall | 06e | 06c |
  |---|---|---|
  | 1e-7 | 2.282 h | 3.208 h (×0.711) |
  | 1e-8 | 2.974 h | 5.775 h (×0.515) |

  - The largest unit took 6.0 s at 1e-7 and 105.1 s at 1e-8, against 06c's 425.6 s and 1,184.7 s.
  - The run took 08:13:18 → about 10:00Z, about 1 h 47 min, against 06c's 2 h 30 min.
  - Solves fell 7.26 M → 4.86 M at 1e-7 and 12.44 M → 6.21 M at 1e-8; regions fell 0.97 M → 0.63 M
    and 2.09 M → 0.80 M.
  - **The cost fell**, as the addendum expected.
- **Convergence and depth** (my reader):

  | | 1e-7, 06c | 1e-7, 06e | 1e-8, 06c | 1e-8, 06e |
  |---|---|---|---|---|
  | Calls at depth 20 | 109 | 11 | 153 | 18 |
  | Group calls unconverged | 15 | 9 | 49 | 15 |
  | Analytic calls unconverged | 89 | 0 | 151 | 0 |
  | Rows with `total_converged` false | 13 | 9 | 45 | 15 |
  | … of them outside the 720 | 9 | 0 | 33 | 1 |

  The row max-depth histograms are in `checks/z2_z4_z5_rescore.txt`; their mass moves from depths
  8–20 to 3–8.

## 5. The records

- **Both `log.md` files** carry every README §5.1 heading: the subject line, What shipped,
  Predictions and verdicts, Deviations, Verification performed, Observations not acted on, and
  State handed on. Every numbered deviation is classified:
  - **the rehearsal's 1–9:** 2 STRUCTURALLY REQUIRED, 4 IMPLEMENTATION CHOICE, 3 UNINTENDED DRIFT;
  - **the census's 1–24:** census deviation 11 is split between two classes, and deviation 16's
    class wraps a line.
- **`verification.md` and `issues.md`** exist in both directories.
- **Sizes:**

  | Directory | Size |
  |---|---|
  | census run directory (kept in full for 07) | 1.6 GB, 149 files |
  | references run | 12 KB, 5 files |
  | R19 / R20 run directories | 28 KB, 4 files each |
  | R19 / R20 stores | 74 MB / 75 MB |
  | `var/levin-phase-floor-06e/rehearsal/` | 221 MB, of which `extract/` (the readers' output) is 180 MB and is not to be copied back |
  | `census/` | 21 MB |
  | `review/` | 1.8 MB |

  Sources: `checks/sizes_and_store_hashes.txt`.

## 6. Readings for the user (recorded, not decided here)

1. **Rehearsal deviation 2 (STRUCTURALLY REQUIRED):** the readers start a local Ray, under the
   lock.
2. **Rehearsal deviation 4:** Z1's exclusion of "06a's 720" is vacuous at rehearsal scope. No R8-
   scope row is an A3 v2 row, so all 864 were scored.
3. **Census deviation 21:** "more than twelve over-budget units → stop before the pass" is the
   agent's own threshold, taken from 06c's Part B. It did not arise.
4. **Census deviation 13 / predictions Part B:** Z3's "resolutions agree" is read as the range of
   S96(P128), S96(P64) and S128(P64) ≤ min(abserr)/100, with S128(P64) as the reference. My
   re-score uses the same reading.
5. **The prompt's "06c's census had 109 and 153 such calls"** counts every depth-20 call,
   including the 720's. Outside the 720, 06c had 84 and 133.

## 7. Issues proposed by the agents (for the board; none opened here)

**The rehearsal** proposes none. It records three observations:
- the vacuous stationary-point exclusion at rehearsal scope;
- 06c's 12 equilateral Y1 rows, which stay where 06c put them (06e moves them ≤ 1.8 × 10⁻⁶ of the
  summed errors);
- `runregistrykit list | head` giving a `BrokenPipeError`: cosmetic, RunRegistryKit's own CLI,
  possibly to report upstream.

**The census** (`census/issues.md`) proposes six:
- **D1:** the Z2/Z5 call, `g-1e-8-0-33995` call 5. Its floor of 1.33e-33, 37× below 06c's, still
  exceeds atol 6.7e-34, and the loop chases it to depth 20. Suggested: amend
  `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
- **D2:** Z3's sample 3, dishonest under both codes. Suggested: narrow
  `[06b-moved-calls-declare-less-than-their-error]`.
- **D3:** $|v-{\rm ref}|/{\rm abserr}$ sits at 0.998–0.9995 on samples 2, 8, 9 and 14 in both
  codes, unexplained. Suggested: an issue for 07 or a follow-up.
- **D4:** Clenshaw–Curtis floors are 99.4 % the declared-phase proxy, with 06e's node term at
  0.06 %. Suggested: an issue.
- **D5:** `*-1-2174` call 1 converges at 1e-8 at depth 7 on depth20-tiling's pinned value, which
  is probably dishonest. Suggested: amend 07's stationary-point issue.
- **D6:** Y1's 12 equilateral rows are not A3 v2 rows, so the census cannot report them.

Whether `[06c-chebyshev-end-nodes-miss-the-span]` and `[06d-cc-node-term-is-pessimistic]` close
given these results is for the user.
