# levin-phase-floor 06, Part B — the Mac Studio orchestrator's review (note §B.3)

Host mps026298, 2026-10-26, from 21:52 UTC (Gate) to about 23:45 UTC. Reviewer: the Mac Studio
orchestrator session, on Opus 5.5. Outputs of every command below are in `review/` beside this
file. The 1.09 GB re-run of the value comparison stayed in the session scratchpad, and its SHA-256
is in `review/c4_svd.json.sha256`.

**Verdict, after the user's ruling: PASS, all ten checks.** The user ruled on check 4, given here
verbatim: "Check 4 passes: the classifier's 185 UNCLASSIFIED all lie inside the five classes prompt
§5.4 expects to change, with 0 outside, so the review passes." The review's verdict as first
written is kept below.

**Verdict as first written: STOP, on check 4 alone, for the user's reading.** As written, check 4 requires that the
copied `classify_non_qsi.py` print no `UNCLASSIFIED`. It prints 185 and exits 1. All 185 are inside
the five classes the prompt expects to change. They are 182 `GkSourcePolicyData` rows whose
metadata key `Levin_z_dtheta_dlogz` moved, and the 3 value tables of `TkWKBIntegration`,
`GkWKBIntegration` and `GkSource`. There are 0 outside them. 05's script, unchanged, exempts only
`QuadSourceIntegral`. Every other clause of check 4 holds, and checks 1–3 and 5–10 pass. The
rehearsal agent read the count the same way, as its deviation 6 (IMPLEMENTATION CHOICE), and said
so in its report. Whether that reading stands is the user's call.

## The Gate (21:52 UTC), read-only

These held as the laptop stated them:
- the main checkout at `28ebaf0`, clean;
- the worktrees `-9f66622` and `-f5cee4c`, clean, at their SHAs (`dc9d28c^` = `9f66622`, and
  `dc9d28c..f5cee4c` touches only `prompts/`);
- the registry at 21 runs, all finished;
- the lock free, no Ray processes under `/tmp/ray-sgk`, and no `~/ScienceData/.rehearsal-lock`;
- no `var/levin-phase-floor-06/` and no `levin-phase-floor-06-*` store;
- `sample.json` at `73aa7876…`, and 05a's wider run directory at 122 files;
- `~/.ray/auth_token` at `-rw-------`, with 112 GiB free.

The load average was 19.9 / 9.8 / 9.1, from Spotlight, ChronoSync, OrbStack and fileproviderd, not
Python. That is above the laptop's "about 4–8". It was not gated, as at the addendum's 17:06 check.

Before dispatch, the baseline hashes of A3 v2's six files and of 05's and 05a's `var/` and run
directories (6,835 files) went to the scratchpad. The marker was touched at 21:53:13.92 UTC.

## The stops relayed during Part B

1. **The re-measurement agent, before U0** (about 22:05 UTC). `s` cannot open A3 v2, because
   `StoreSchemaMismatch` reports `theta_residual` missing in three value tables. It offered routes
   (A), (B) and (C). Before relaying, I checked that A3 v2 was unchanged, no run existed, no copy
   existed, and the trees were clean. The question went verbatim to the laptop session.
   - **The user's answer, verbatim:** "(B). We have to be practical and pragmatic. The repo rules
     are there to prevent cheating, not to get in the way of honest work."
   - Passed on verbatim, with the laptop's description of route (B).
2. **The re-measurement agent, after launch** (relayed 22:40 UTC). The question was which
   pure-WKB tag scores S1: A3 v2 stores no null `z_init` (`out/cmd/28`). At relay, 40 of 109 units
   had checkpointed, none an `s` unit on R1–R7, so no S1 number existed.
   - **The user's answer, verbatim:** "Clearly (I)."
   - It reached me at 22:44:56 UTC, when 49 units had checkpointed, none an `s` unit on R1–R7. It
     was passed on verbatim.
   - The laptop corrected check 9's "all 9 are null" to `z_init == z_source` on all 9 samples.

## Check 1 — Scope: PASS (`review/c1.txt`)

- **The trees.** The main checkout is at `28ebaf0`, the parent worktree at `9f66622` and the code
  worktree at `f5cee4c`. `git status --short` is empty in all three.
- **`find var -newer <marker>`** lists only these:
  - `var/levin-phase-floor-06/{rehearsal,wider}`. `wider/state/a3v2_copy/` is route (B)'s copy,
    which the user approved;
  - the ten `levin-phase-floor-06-r1[34]*` store files;
  - the three run directories;
  - the two launch logs.
- **The worktrees.** `find <worktree> -newer <marker>`, excluding `__pycache__` and `.git`, lists
  nothing in either.
- **The records outside scope.** A3 v2's six files (with the sidecar manifest) match the
  baseline's SHA-256s. So do all 6,835 files of 05's and 05a's `var/` and run directories.

## Check 2 — The runs: PASS (`review/c2.txt`)

`runregistrykit list` shows 24 runs, all finished. R13, R14 and `wider` are `done`, `wider` at
109/109.

| Run | `campaign` / `prompt` | `git_dirty` | `git_head` | Notes |
|---|---|---|---|---|
| R13 (`…-r13-parent-…`) | `levin-phase-floor` / `06` | false | `9f66622d…` (the parent) | script `5b984c75…` |
| R14 (`…-r14-residual-…`) | `levin-phase-floor` / `06` | false | `f5cee4c7…` (the dispatch SHA) | script `5b984c75…` |
| `wider` | `levin-phase-floor` / `06` | false | `28ebaf0`, the main checkout's | see below |

- No manifest has a separate slug field; each slug is in its run id.
- `wider`'s `git_head` is the main checkout's, where its scripts live, as 05a's was. Its purpose
  names both worktrees by path and SHA. Every checkpoint carries its own tree's `tree_head` and
  `tree_status_short` (check 7).

## Check 3 — The fingerprints: PASS (`review/c3_fp_r1[34].txt`)

- **The run records.** R13's and R14's `status.json` both record
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, with problems `{}`.
- **Re-taken by me**, without `--write`:
  ```
  cd <worktree> && PYTHONPATH=$PWD <main>/venv/bin/python -B -m runregistrykit store fingerprint --runs-root <main>/var/runs <store>
  ```
  R13 from the parent worktree and R14 from the code worktree each give `1b332ffa…`, problems
  none. No sidecar exists, so nothing is recorded.

## Check 4 — The values: PASS on the user's ruling (first written: STOP on its first clause as written; the rest PASS)

- **The comparison, re-run.** I ran
  `cd <code worktree> && PYTHONPATH=$PWD <main>/venv/bin/python -B -m tools.store_value_diff <R13> <R14> --json`
  into the scratchpad. It exits 1, as expected. `cmp` against the agent's
  `out/store_value_diff.json` finds them identical: SHA-256 `7e42eadc…` for both.
- **The classifier, run on mine** (`review/c4_classify.txt`). It is copied byte-identical to 05's
  (`26690e61…`; `tabulate_qsi.py` is `b7508351…`, also identical). It prints **"UNCLASSIFIED
  differences or structural problems outside QuadSourceIntegral: 185"** and exits 1:
  - `GkSourcePolicyData` metadata: 182 `UNCLASSIFIED`. I read the JSON for these, and every one
    is `json_keys {'Levin_z_dtheta_dlogz': 182}`;
  - `TkWKBValue` (2,162), `GkWKBValue` (146,251) and `GkSourceValue` (146,251) "not identical".

  **All 185 are inside the five classes.** As check 4 is written, the literal "no `UNCLASSIFIED`"
  is not met. This is the stop.
- **Outside the five classes.** Every value table is identical: `BackgroundModelValue` 1,159,
  `TkNumericValue` 1,478, `GkNumericValue` 35,874 and `QuadSourceValue` 2,447. Every other
  difference is `timing`, `label` or `rho_reuse`. No class has a row in one store only, an
  unmatched row or an unresolvable column.
- **`theta_residual`.** Columns present in one store only: `columns_only_in_b: ['theta_residual']`
  in `TkWKBValue`, `GkWKBValue` and `GkSourceValue`, and nothing else. None is in A only.
- **Spot-check by `sqlite3` `mode=ro`** (`review/c4_spot.txt`):
  - A changed `GkWKBValue` row, R13 serial 45125 against R14 serial 45927, at `div` −7459548:
    - `theta_mod_2pi`: `-0x1.5b3fd23b4506ep+1` → `-0x1.5b3fd23835f08p+1`;
    - `G_WKB`: −7637534950.897188 → −7637534974.699695;
    - R14's `theta_residual`: 1.5709023512059446.

    All of these equal the tabulation's first row.
  - R14's `theta_residual`, over the whole store:
    - `GkWKBValue`: 147,642 rows, none null, range [0, 1.5714];
    - `TkWKBValue`: 2,209 rows, none null, range [1.7001, 1.7643];
    - `GkSourceValue`: 169,653 rows; 22,011 are null, each exactly where `theta_mod_2pi` is null.

    0 values fall outside (−π − 2e-3, π + 2e-3).
- **Spot-check of `QuadSourceIntegral`** (`review/c4_qsi.txt`), on the row (k, q, r) = (1e5)³ at
  z_resp 250. It matches the tabulation:
  - `total`: `-0x1.1f665c7aeff31p-59` → `-0x1.1f665c9de3ef8p-59`;
  - `total_abserr`: 1.117e-25 → 8.69e-26;
  - regions: 16 → 15;
  - `numeric_quad` and `analytic_rad` are unchanged.
- **`theta_div_2pi`.** By my recount of `phase_classes_rows.csv`, it changes in no row of the three
  tables, so none needs testing against the 4ε|θ| bound.
  - The |Δ `theta_mod_2pi`| maxima are 272, 8.4 and 7.2 ε|θ|. 56 rows are above 4 (52 + 2 + 2);
    the agent places them at |θ| < 1 rad, near each object's anchor.
  - §5.4 sets no bound on Δ `theta_mod_2pi`; these are tabulated, as it asks.
- **`!! WARNING`.** It occurs 0 times in R14's launch log, `stdout.log` and `stderr.log`.

## Check 5 — The readers and the lock: PASS

- **The readers** (`rehearsal/out/readers.txt`): all six `exit 0`. Every reader records R14's
  files as identical before and after (SHA-256, mtime, size), with no journal. R14's five files
  hash the same now: primary `fff6cf7f…`, shards `1c0f3cf6…`, `866511b7…`, `dfaad949…`,
  `410bf34b…`.
- **The lock transcript** (`rehearsal/out/lock.log`):
  - `acquire` 0;
  - `set-pid 97740` 0 (R13's driver);
  - `set-pid 99938` 0 (R14's driver);
  - `ray-processes` three times, each 0 with none;
  - `release` 0, then `status` free.
- **Run by me**, each as a command of its own: `ray-processes --ray-temp-dir /tmp/ray-sgk` exits 0
  and lists nothing; `status --ray-temp-dir /tmp/ray-sgk` says "lock: free", with no Ray
  processes.

## Check 6 — U0 and the store: PASS

- **U0's saved output** (`wider/out/U0_a.txt` at 22:23:59, `wider/out/U0.txt` at 22:32:12), all
  before the launch manifest at 22:36:34:
  - **(a) holds:** the diff shows `CLAUDE.md` and the two comment-only files, whose ASTs equal
    `1fc349c`'s;
  - **(b) holds:** on `low-G+r-1` and `ctrl-low-1`, every field is shown with `==` and `float.hex`,
    16 doubles per row;
  - **(c) holds:** on both rows, 0 differences in `theta_div_2pi`, `theta_mod_2pi`, `G_WKB` and
    `T_WKB` over 241,146 G and 5,172 T values, and 0 coefficient mismatches. The only differences
    are the diagnostic `analytic_*` columns (W2);
  - **(d) holds:** every compared block is "identical".

  U0's comparisons use no `isclose`. The only `isclose` in `wider/scripts/` is in `draw_sample.py`,
  unchanged from 05a, where it matches tolerances.
- **My independent comparison** (`review/c6_t5_vs_05a.py`, `.txt`), on row `mid-G-2`, a depth-20
  row U0 did not use: `t5-1e-7` against 05a's `t-1e-7` is bit-identical in `total`
  `0x1.3a5d5e42eaae4p-81`, `total_abserr` `0x1.d39df375a33dbp-103`, and the value and abserr of
  all 5 calls.
- **The store.** A3 v2's six files hash as at baseline. I ran `runregistrykit store fingerprint`
  from the main checkout, without `--write` (`review/c6_fp_a3v2.txt`): `0e32f24a2dc2…d21d56`,
  problems none, "matches the fingerprint taken … at b90752d". Its files are unchanged after the
  fingerprint.
- **Journals.** There is no `-journal`, `-wal` or `-shm` file anywhere under `var/datastores` or
  `var/levin-phase-floor-06`.

## Check 7 — The trees, read: PASS

- **`PYTHONPATH`.** `run_wider.py:142` sets `PYTHONPATH` to the tree's worktree for each worker,
  and `launch.py:104` removes it for the driver.
- **The evidence in `stdout.log`.** In the worker-evidence lines (`stdout.log:3`, `:144`), every
  module is in its tree, with `all_in_tree: true`. That covers `ComputeTargets.*`,
  `WKB_phase_function`, `GkWKBIntegration`, `TkWKBIntegration`, `GkSource`, `StoreFixtures` and
  `main.py`:

  | | `t5` | `s` |
  |---|---|---|
  | path | `…-lpf06-9f66622/` | `…-lpf06-f5cee4c/` |
  | HEAD | `9f66622` | `f5cee4c` |
  | `status_short` | empty | empty |
  | `GkWKBValue_has_theta_residual` | false | true |
  | `producers_store_reduce_exactly` | false | true |
  | store it reads | A3 v2 | the certified copy |
- **The re-runs.** The `t5` units carry no re-run (`rerun` is null on all 39). The 70 `s` units
  carry it, with the re-run values against stored ones. For example, on `s-1e-7-R1`,
  `theta_mod_2pi` differs in 110,681 of 111,309 G values, and `theta_div_2pi` in 0.

## Check 8 — The sample and the predictions: PASS

- **The sample.** `state/sample.json` is
  `73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7`.
- **The predictions' timing.** `state/predictions.md` is at 22:36:09, later than U0's output
  (22:32:12) and earlier than `manifest.json` (22:36:34). Its first version,
  `predictions_as_first_written.md`, is at 22:32:53. The added section is deviation 17.
- **The wording.** Prompt §6's block, from "**Predictions, fixed before the run**" to §7, is a
  substring of both files, with 0 lines missing.
- **The harness.** Every file that differs from or is new against
  `docs/levin-phase-floor/05a/wider/scripts/` maps to a numbered deviation:

  | Files | Deviation |
  |---|---|
  | `prov.py` | 2 |
  | `rec.sh` | 3 |
  | `explore_store.py` | 5 |
  | `make_copy.py`, `certify_copy.py`, `fingerprint_copy.py`, `hash_store.py` | 8 |
  | `harness.py` | 9 |
  | `rebuild.py` | 10 |
  | `rerun.py` | 10, 11 |
  | `phase_probe06.py` | 12 |
  | `worker.py` | 13 |
  | `run_wider.py`, `launch.py` | 14 |
  | `u0.py`, `u0_clause_a.py` | 16 |
  | `analyse06.py`, `characterise_s2.py` | 19 |
  | `pure_wkb_zinit.py` | listed under "What shipped"; its output `out/cmd/28` is deviation 17's evidence |

  `convergence_kinds.py` is identical to `prompts/levin-phase-floor/measure/`'s, at `49c83a72…`.

## Check 9 — The verdicts, re-scored: PASS (`review/c9_rescore.py`, `.txt`)

- **The method.** My scratch reader recomputes every probe window's σ under both trees from the
  saved φ samples. It uses 04's `window_rms` (degree 4, 9 samples, in u = log1p z). 0 of 23,156
  windows differ from the saved σ.
- **The pure-WKB tags.** I re-derived them by `z_init == z_source` on all 9 samples, as the laptop
  corrected. They agree with the saved `pure_I` in every window. `z_init` is the same under both
  trees in every G window, and no window qualifies under (L).

| Rule | My numbers | Verdict | Agrees with the agent |
|---|---|---|---|
| S1, reading (I) of record | 6,700 pure-WKB G windows, true in 6,700. σ_s/F max 8.8e-14, median 5.7e-16 | **holds** | yes |
| S1, reading (L), reported | 0 windows | undecided | yes |
| S2 | 12,862 T windows; 10,747 true, 2,115 false | **fails** | yes |
| S3 | 39 rows; largest depth 11; no call with `n_depth_ge_18`. The two NULL controls make no Levin call | **holds** | yes |
| S4 | 34 of 39. False: `low-G-3` 10.80, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 (× `total_abserr`_t5) | **fails** | yes |
| S5 | 31 rows with both `s` units in budget; 28 true. False: `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31 | **fails** | yes |

- **S1's 3,594 numerically seeded windows:**
  - σ_s/F: median 2.1e-4, max 1.24e-3;
  - σ_t5/F: median 0.27, max 0.78;
  - Δθ RMS/F: median 2.8e-5, present for all 3,594.
- **Over budget:** 0. All 109 units are `done`.
- **The descriptive items** are present: `state/analysis.json` has `rows`, `capped`,
  `over_budget`, `worst_budget`, `policy_warnings` and `rerun`. `out/convergence_kinds_s.txt` was
  run under `s`, against 05's `p` capped set, as 05a ran it.

## Check 10 — The records: PASS

- **`rehearsal/log.md`** has README §5.1's six headings. It has 14 deviations, each with its class
  in its heading.
- **`wider/log.md`** has the six headings, with "Deviations" for "Deviations from the prompt", and
  adds a "Gate U0" section. It has 20 deviations, every one classified. Several carry more than one
  class, one for each part; for example, 8 is SR, UD, UD and IC.
- **The verification files.** Both `verification.md`s are written as subsections "6.x" of
  `VERIFICATION.md` §6. All 39 of their links are relative, in the `06/…` form.
- **The issues files.** Both `issues.md`s exist: `rehearsal/` has P1 and P2, and `wider/` has
  W1–W12.

## For the user and the laptop

- **Check 4's stop** needed the user's reading of the classifier's 185 `UNCLASSIFIED`. The user
  ruled that it passes (verbatim at the top of this file).
- **Route (B)'s scratch copy**, about 408 MB, is
  `var/levin-phase-floor-06/wider/state/a3v2_copy/`. It is to be deleted on the user's word after
  the records commit, and is not copied back.
- **The rehearsal stores** R13 and R14 are kept, and are the user's to keep or retire.
- **Bulk output not to copy back:**
  - `rehearsal/out/store_value_diff.json`, 1.09 GB;
  - `rehearsal/out/phase_classes_rows.csv`;
  - `rehearsal/extract/`;
  - the `wider` U0, smoke and dev pickles.
- **Issues proposed by the agents:**
  - **P1:** two `QuadSourceIntegral` rows at z_resp 250 moved 1.8e4× and 484× R13's
    `total_abserr`;
  - **P2:** Ray task output does not reach the launch log;
  - **W1–W12**, among them:
    - W3, the false premise of a null `z_init`;
    - W4, S2's rule where |θ_T| is small;
    - W5 and W6, the S4 and S5 rows;
    - W12, `store fingerprint` across schemas.
