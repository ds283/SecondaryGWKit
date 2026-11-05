**Verdict: PASSES — all seven §B.3 checks hold; 06k's records are correct for the tree they were taken on. No route passes as a candidate remedy (every route fails C2; routes 1, 2 and 2 literal fail K1; route 3 fails L1 and H1; routes 2, 2 literal and 3 fail S1).**

# levin-phase-floor 06k — the orchestrator's review (§B.3), on the Mac Studio

Written by the Mac Studio orchestrator on mps026298, 2026-11-05, after the agent's final report. Tree: the worktree
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060` at the dispatch SHA `89370609`. Every check is a script
or command of mine under `review/checks/` (stop-1 checks) and `review/checks/b3/` (this review), each with its
output beside it. My own re-runs wrote only under `review/checks/b3/`.

## The verdicts the review confirms

| | route 1 | route 2 (as built) | route 2 literal | route 3 |
|---|---|---|---|---|
| **L1** | holds (0 of 17) | holds (0) | holds (0) | **fails** (c01 left, D = 1.7e-4 × η × request) |
| **L2** | holds (114/142, 80.3 %) | **fails** (57/142, 40.1 %) | **fails** (57/142, 40.1 %) | holds (142/142) |
| **C1** | holds (792/801, 818/830) | holds (801/801, 828/830) | holds (794/801, 821/830) | holds (801/801, 829/830) |
| **C2** | **fails** (7, all 1e-8) | **fails** (5, 1e-8) | **fails** (53: 25 at 1e-7, 28 at 1e-8) | **fails** (1, 1e-8) |
| **H1** | holds (worst v03 1.185 ≤ 1.203) | holds (v03 1.202) | holds (v03 1.200) | **fails** (c01 3.935, r56 1.492) |
| **S1** | holds (0) | **fails** (10 calls, worst 1.323) | **fails** (same 10) | **fails** (46 calls, worst 1.645) |
| **K1** | **fails** (89,171 of 229,338 ≥ 5.8) | **fails** (82,565 of 167,281) | **fails** (88,186 of 191,623) | not scored |

C1 counts are V8's lost calls outside the 720 at 1e-7, then 1e-8. H1 is scored on the 124 matched references (115
outside the 720). V1's 12 are unscored on the user's ruling *"(a) as recommended, please"* (log deviation 24). I
re-derived the C1, C2 and S1 counts and eight H1 ratios myself (checks 5 and 6). L1, L2 and K1 are the agent's,
re-checked at leaf level on five leaves (check 4).

## 1. Scope — holds

- **Outside 06k:** no file under `var/` is newer than `review/dispatch_marker` (2026-11-04T08:48:43Z) outside
  `var/levin-phase-floor-06k/` and the seven `var/runs/levin-phase-floor-06k-*` runs: 0 files, at the start of the
  review and again after my re-runs (`c1_newer_outside.txt`).
- **The scratch copy** `var/levin-phase-floor-06/wider/state/a3v2_copy/`: its 5 files are unchanged against
  `preflight/a3v2_copy_before.sha256`, and its listing is unchanged (`c1_a3v2_ls_after.txt`). The agent's after-check
  also finds it equal to its certification, with fingerprint `0e32f24a…`.
- **06g's census** checkpoint and its 16 region pickles: 17 of 17 match `preflight/census06g_regions_before.sha256`.
  The input `v8_calls.json` matches too.
- **The worktree** shows `git status --porcelain --ignored` empty at `89370609`, and the main checkout's status is
  empty (`c1_worktree.txt`).
- **No Ray:**
  - the lock is free;
  - `ray-processes` lists none (`c1_ray_processes_final.txt`);
  - `/tmp/ray-sgk` holds the same 97 sessions as before dispatch, with no new one;
  - every worker of my own re-runs reports `ray_initialized False` and no foreign modules.
- **Nothing new in `var/datastores/`:** its `ls -la` equals the pre-dispatch listing (`c1_datastores_final.txt`).
- **The registry:** the four censuses are `done` 18,560/18,560, `lgdraws` and `route2lit` are `done`, and `k0partL`
  is `failed`, which is recorded as log deviation 8. Nothing of 06k is `running` or stale.

## 2. The variants and K0 — holds

- **AST:** my own comparison (`checks/s1b_ast_compare.py`), re-run now, gives output identical to stop 1's
  (`c2_ast_compare.txt`).
  - The identity variant is the tree's module, byte for byte (`d89c959e…`).
  - Each route changes only `_adaptive_levin`, with its signature and decorators equal:
    - routes 1, 2 and 2 literal change only the nested `_check_leaf`;
    - route 3 changes `_push_candidate` and adds `_declaration_06k` and `_relevel_06k`.
  - The variant files' SHA-256 are unchanged since stop 1, and are the ones each census manifest records: route 1
    `23cd69e4…`, route 2 `017fae1c…`, route 2 literal `2fd10b78…`, route 3 `e139ec71…`.
- **K0 re-run myself:** two K0 units on which route 2 fired nowhere in K0, at different k and rtol: `g-1e-7-3-35986`
  (k3_201) and `g-1e-8-2-13642` (k2_2). I ran each under the identity variant and under route 2 (as built) with
  `scripts/run_k0.py` (`k0rerun/`).
  - Against 06g's census record, all 4 × 21 call records are equal except `elapsed` and `route_06k`.
  - The four region pickles are equal leaf by leaf, arrays by bytes, with timing keys and `route_06k` aside.
  - Route 2's firing count is 0 on both units (`c2_k0_rerun_compare.txt`).
- **Stop 1's K0 holds by my own reader** (`checks/s1_k0_*.txt`, `checks/s1b_k0_*.txt`):
  - identity: 2,110/2,110 calls and 100/100 region pickles equal;
  - every route equals 06g on every call where it never fired.

## 3. The samples and predictions — holds

- **`state/predictions_and_rules_20261104T0915Z.md`:**
  - It is read-only, with mtime 2026-11-04T08:59:36Z and SHA-256 `e315d74d…`.
  - It holds prompt 06k's §4 byte for byte, heading included.
  - It also holds K0's draw rule and seed (PCG64, `seed = 20261104·100 + 10s + t`).
- **The log's copy:** `log.md`'s "Predictions and verdicts" holds §4's body byte for byte; only the `## 4.` heading
  line is replaced by the log's own. The verdicts sit below the marker line (`c3_verbatim.txt`).
- **Order:** every `out/cmd` record before 08:59:36Z is a Gate check (08:54:49Z–08:56:05Z). The first command that
  produces a number of 06k's is `10_make_variants` (09:02:09Z). The first K0 unit ran at 09:07:05Z and the first
  census at 10:30:09Z.
- **The file's name** says "0915Z", which is later than when it was written. This is log deviation 1 (cosmetic).

## 4. Part L, re-checked — holds

I took three of 06j's slow leaves (L-a) and two 06h carrying leaves (L-b), chosen to cover each decision pattern
(`c4_partL_recheck.py`/`.txt`):

| Leaf | Call | Recomputed decision |
|---|---|---|
| s02 #3 | `g-1e-7-3-34738` #1 | routes 1 and 2 trust |
| s01 #1 | `g-1e-7-2-21980` #3 | route 1 only |
| s05 #1 | `g-1e-7-1-11233` #4 | route 2 only |
| c01 | `g-1e-7-2-33416` #6 | route 3 leaves it (L1's failure) |
| c07 | `g-1e-8-3-37323` #1 | Δ1/φP = 5.799, K1's yardstick; no route trusts or leaves it |

For each I recomputed the following from the solves' recorded numbers:
- ν1 = (φP + φL) + φR;
- Δ1c and Δ2c from the values corrected by −E, with a Levin solve's absent E taken as 0, as the variant does, and r_c;
- D, against η × the request, where the request is recomputed as max(atol, rtol |v_g|) from 06g's census value.

The results:
- **Decisions:** every decision of every route on all five leaves equals the probe's.
- **Exact numbers:** ν1, D and the request are bit-equal to the probe's.
- **Δ1c:** it agrees to 0 (c01, c07, s02) or within 1.5–3 × 10⁻³ relative (s01, s05). Halves and quarters are
  recorded as sums, so my Δ1c differs from the variant's correct-then-sum by rounding. The rounding moves no decision
  near its boundary: s01's r_c is 1.260 against 1.255, and s05's is 0.176 against 0.179.
- **Independent of the probe:**
  - all five leaves are among 06g's census `checked_leaves_06g` at the same span by hex;
  - their Δ1, Δ2, r, φP, V_P and E_P equal the census's by hex (E_P is NaN in the census for Levin solves, where the
    probe records none);
  - route 3's census fired on c01 itself, with D bit-equal to mine (9.810060e-41, D/(η·request) = 1.734 × 10⁻⁴).
- **What this check does not cover:** the children's E, φL and φR are taken from the probe's records, not re-solved
  by me.

## 5. Part C, re-checked — holds

- **My own reader** (`c5_counts.py`, `c5b_counts.py`) reads the raw checkpoints, not the agent's extracts:
  - 06g's and each route's checkpoint, 18,560 records each, all `done`;
  - the 720 from 06a's census scan by 06e/06h's rule, which gives 720 rows and agrees with every
    `stationary_point_row` flag in `v8_calls.json`.

  | | C1 1e-7 | C1 1e-8 | C2 lost outside the 720 | S1 calls | S1 units | S1 rows (shard-serial) | S1 worst |
  |---|---|---|---|---|---|---|---|
  | route 1 | 792/801 | 818/830 | 7 (1e-8) | 0 | 0 | 0 | — |
  | route 2 | 801/801 | 828/830 | 5 (1e-8) | 10 | 10 | 5 | 1.3234 |
  | route 2 literal | 794/801 | 821/830 | 53 (25 + 28) | 10 | 10 | 5 | 1.3234 |
  | route 3 | 801/801 | 829/830 | 1 (1e-8) | 46 | 44 | 22 | 1.6452 |

  Every count and worst ratio equals the agent's. Its "rows" in S1 (10 for route 2, 44 for route 3) are census units
  (an rtol and a row). As this campaign uses the word, a row is shard-serial, so S1 touches 5 rows under routes 2 and
  2 literal and 22 under route 3. This is a label only; no verdict changes. It is recorded under "Observations" below.
- **One recovered call re-run myself** (`recovered_rerun/`, `c5_recovered_compare.txt`): V8's lost call
  `g-1e-7-3-825` #0, unconverged under 06g and converged under route 1. I re-ran its whole unit under route 1 with
  `scripts/run_k0.py`.
  - The call equals route 1's census record bit for bit:
    - value `0x1.32cd3d792ae78p-97`;
    - abserr `0x1.1f25b6709afb5p-119`;
    - 61 regions against 06g's 85, max depth 13 against 20;
    - converged `True` against 06g's `False`, with 0 untrusted regions;
    - 15 firings, the firing records equal.
  - All 10 of the unit's call records equal the census's, `elapsed` aside, and its `result` equals the census's,
    `compute_time` aside.

## 6. The references — holds

I recomputed |v − ref|/e exactly, for two referenced calls per route. Each uses the reference's 30-digit `ref_30`
from its own source (06g's `scores.json` `rows`, 06j's `w_calls.csv`) and v and e from the raw census checkpoints
(`c6_refs.py`/`.txt`).

| Route | Call | |v_g − ref|/e_g | |v_route − ref|/e_route | H1 on the call |
|---|---|---|---|---|
| route 1 | v03 | 1.202676 | 1.185379 | holds |
| route 1 | z01 | 0.320263 | 0.320263 | holds |
| route 2 | v03 | 1.202676 | 1.202382 | holds |
| route 2 | r56 | 0.005909 | 0.005909 | holds |
| route 2 literal | v03 | 1.202676 | 1.200298 | holds |
| route 2 literal | c01 | 0.001382 | 0.001382 | holds |
| route 3 | c01 | 0.001382 | 3.934672 | **fails** |
| route 3 | r56 | 0.005909 | 1.491725 | **fails** |

All 16 ratios equal the agent's `h1_calls.csv` to 1 × 10⁻¹².

## 7. The records — holds

- **`log.md`'s structure** (`c7_records.txt`):
  - It has README §5.1's headings: subject, commit and result, What shipped, Predictions and verdicts, Deviations,
    Verification performed, Observations not acted on, State handed on. It also has a "The stages, in order".
  - There are 28 deviations, numbered 1–28, each with exactly one classification: 6 STRUCTURALLY REQUIRED,
    14 IMPLEMENTATION CHOICE and 8 UNINTENDED DRIFT.
  - Deviation 17 quotes both of the user's stop-1 rulings verbatim (whitespace-normalised), and deviation 24 quotes
    *"(a) as recommended, please"*.
- **`out/results.md` proposes no code** (`c7_results.txt`). It has no code fences or diffs. It says so in its
  provenance comment, its remedy section is headed "(no code is proposed)", and it ends "Which route, if any, becomes
  code is for the user."
- **Provenance** (`c7_provenance.txt`):
  - Each JSON and CSV under `out/partC/`, `out/k0/`, `out/partL_*` and `out/lg_draws/` carries a header with its
    script's SHA-256, except the four `out/partC/<route>.json`.
  - Those four carry their checkpoint's SHA-256 but no header. They are written by the same run as `summary.json`
    (`analyse_c.py`, `out/cmd/101`), whose header covers them. This is noted below.
  - `results.md` and `verification.md` carry provenance prose.
- **`KEPT_ON_MAC_STUDIO.txt`** (SHA-256 `963ea70b…`) lists 873 files with SHA-256 and size: 261 under 06k's directory
  (pickles and files over 2 MB) and every file of the seven run directories.
  - A seeded sample of 12 (seed 20261105) matches by SHA-256 and size (`c7_kept_sample.txt`).
  - The four census checkpoints, hashed in full, match it (`c7_census_checkpoints.sha256`).

## Stops and deviations of the orchestration

- **Planned stop 1**, after Gate K0 and Part L: stop 1's AST and K0 checks hold, and the census plan was accepted.
  - The user ruled on two questions, relayed verbatim:
    - Q1 (route 2): *"OK, so this isn't really a contradiction, it's just that the note said that trust would be
      preserved if the leaf had `r < 1/4`, and it might not be. We had better run that fourth census."*
    - Q2 (route 3): *"stick with what has been built."*
  - So a fourth variant (route 2 literal) was built, checked as at stop 1, and given a fourth census. These are log
    deviations 17–21.
- **Planned stop 2**, after each census: each was `done` 18,560/18,560, exit 0, stderr empty.

  | Route | Wall | Projection P |
  |---|---|---|
  | route 1 | 16,758 s | 16,183 s |
  | route 2 | 13,621 s | 14,332 s |
  | route 2 literal | 13,968 s | 15,037 s |
  | route 3 | 7,626 s | 11,866 s |

  The backstop reported once, for route 1 at 15:00:12Z (P_o = 16,261 s), and stopped nothing.
- **An unplanned §7 stop** ("a reference cannot be matched to its call"): V1's 12 references are R24 calls, not A3 v2
  census calls. It was relayed verbatim. The user's ruling was *"(a) as recommended, please"*: V1 is dropped from H1
  as an error in the prompt, and nothing more was computed (deviation 24, STRUCTURALLY REQUIRED).
- **My own deviations:** none beyond what this review records. My re-runs were short (under 70 s each, at most 4
  processes), Ray-free and not registered. They were checks of this review, not 06k's measurements, and they wrote
  only under `review/checks/b3/`.

## Observations, not acted on (for the laptop's §B.4)

1. **S1's "rows" in the agent's records are census units** (rtol × row): 10 for routes 2 and 2 literal, 44 for
   route 3. In shard-serial rows they are 5, 5 and 22. A wording fix for `results.md`, the log and `issues.md` when the
   records are committed; no number changes.
2. **The four `out/partC/<route>.json` carry no provenance header of their own.** Their provenance is
   `summary.json`'s header and `out/cmd/101`.
3. **Check 4 leaves the children's numbers unchecked.** The children's E, φL and φR come from the probe's records;
   only the parents' numbers were matched to 06g's census.

## Cost per census (from each run's manifest and status)

| | census wall | unit wall vs 06g | group-call regions vs 06g |
|---|---|---|---|
| 06g | 20,361 s | 1 | 1 |
| 06h | 9,193 s | 0.32 | 0.23 |
| route 1 | 16,758 s | 0.78 | 0.78 |
| route 2 | 13,621 s | 0.60 | 0.65 |
| route 2 literal | 13,968 s | 0.62 | 0.67 |
| route 3 | 7,626 s | 0.30 | 0.27 |

## Kept on the Mac Studio

Everything stays on the Mac Studio until the laptop's §B.4 copies the small records back:
- `var/levin-phase-floor-06k/`, 1.2 GB, review included;
- the seven runs `var/runs/levin-phase-floor-06k-*`: the four census directories at 4.2, 4.0, 4.6 and 3.4 GB, the
  other three under 32 KB.

The census checkpoints' SHA-256:
- route 1: `36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38`
- route 2: `8363ae821ce41cd6a9d79c47457d0fad1c5cd5faf2ebf2f0874d6133f720463e`
- route 2 literal: `7fb9c41e996ef03309b03e83b3ee72180f2fd5f9d3874009a6a1fcd12914c156`
- route 3: `89afd6da13fe7549d7bd714cf0109e17194ac8224b5c4524c04ef16729d81d0d`

The agent's records:

| File | SHA-256 |
|---|---|
| `log.md` | `f4eb1114fb28ddcad362b68caaff1a98a472a2fb6f7c902f69494c4f351a7d42` |
| `verification.md` | `5f7264e1bdf31c7234a7cf9acdf74a5d19521d5df5adca156a8e519d2bf7b43b` |
| `issues.md` | `fd301e76c0c2ede48d92da9410b08ddd35a7efb5bfaca21d09a4de5898d3d273` |
| `KEPT_ON_MAC_STUDIO.txt` | `963ea70bfd0354007aacfc7f7c528184f48d4c7c41093b149226e37ddd61c9e0` |
| `out/results.md` | `6845e1aa335558fb3fa1efa49efc0817f233207088bfcdd550513ddca49d015b` |
| `out/partC/summary.json` | `ef7bfe87a037b2d6fdbdfd375734d022a4bf22c5cd9460c7977999dcc7fbbb99` |
| `out/partC/h1.json` | `1ba4786bb899bc85385ef8b49422ccb40f65ccbb94364f043698ec36b9695f27` |
| `state/predictions_and_rules_20261104T0915Z.md` | `e315d74d6f5bd38e064f7ff98c110498a9bf12abad2fd4dbad1660bd53473951` |

Nothing was committed, and no `git` write was run. Nothing outside 06k's directory and runs was touched, and no store
was read except 06's scratch copy, read-only, by 06k's harness.
