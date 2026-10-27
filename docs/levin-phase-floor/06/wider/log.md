## §6 — the re-measurement (Mac Studio)

**Subject:** prompt 06 §6 on the Mac Studio (mps026298), Ray-free, no lock, by the re-measurement
agent. **Run:** `levin-phase-floor-06-wider-20261026T223634`, launched 22:36:34Z (sh 19695,
driver 19700). It ended `done`, 109/109, exit 0, at 23:23:16Z. **Commit:** none here; the orchestrator commits
the records in 06's records commit.

**Result:** U0 holds (all four clauses). **S1 holds** (reading (I), of record, the user's
ruling; 6,700 of 6,700 pure-WKB windows), **S2 fails** (2,115 of 12,862 T windows, all at
z ≥ 1.75e7, where `sigma_s ≈ sigma_t5`), **S3 holds** (39 of 39, depth ≤ 11), **S4 fails** (34 of
39), **S5 fails** (28 of 31). No unit went over budget. A3 v2 is byte-identical, fingerprint
`0e32f24a…`.

**Stop and the user's ruling.** `s` cannot open A3 v2 (`StoreSchemaMismatch`: its factories
declare `theta_residual`, which A3 v2's value tables lack; `out/cmd/06`). The user chose route (B)
(relayed 2026-10-26): an unregistered scratch copy of A3 v2 with the three nullable columns added,
all NULL, certified, opened read-only by `s`; the copy is kept until the records are committed and
deleted on the user's word.

### Gate U0 (before the launch): holds
- (a) `git diff --stat 1fc349c 9f66622 -- . ':!docs' ':!prompts'`: `CLAUDE.md`,
  `primitive_phase.py`, `test_g_phase_reanchor.py`; both files' ASTs without docstrings equal
  `1fc349c`'s (`out/U0_a.txt`). **Holds.**
- (b) t5@1e-7 on `low-G+r-1` and `ctrl-low-1` equals 05a's `t-1e-7-*` bit for bit: total, abserr,
  every call's value and abserr, the whole result summary, calls, `qs_check`, dispatcher, stored
  summary, region arrays (`out/U0.txt`). **Holds.**
- (c) the producers re-run at the parent on A3 v2 (both rows; 1,608 + 1,548 `GkWKBIntegration`,
  241,146 values; 4 `TkWKBIntegration`, 5,172 values): `theta_div_2pi`, `theta_mod_2pi`, `G_WKB`,
  `T_WKB` and every coefficient bit-identical to A3 v2. Beyond the clause: the re-assembled
  `GkSource` matches in every stored column but one `analytic_G_rad` (`0.0` vs `-0.0`), the
  recomputed `GkSourcePolicyData` equals the stored one, and each re-run unit's result, calls,
  regions are bit-identical to the stored-input t5 unit. Only the diagnostic `analytic_*_w` and
  some `analytic_G_rad` values differ (issues W2). **Holds.**
- (d) two in-process `s` runs per row, on the certified copy: result, calls, `qs_check`,
  dispatcher, regions, the re-run's record, the policy, the probe windows and samples, every
  `deltaTheta`: bit-identical. **Holds.**

### Predictions and verdicts

Prompt 06 §6's predictions, copied unchanged (as amended by §9 point 4 before dispatch).
`state/predictions.md` holds the same text, verified byte for byte, written at 22:32:53Z, after U0
held (22:32:12Z) and before the launch (22:36:34Z), with its readings and, added at 22:36:09Z
before the launch, the pure-WKB tag read two ways.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **S1 — $\varphi_G$'s noise goes.**
  - **The rule:** in every **pure-WKB** probe window of $G$ on R1–R7,
    $\sigma_s \le \max(10^{-3}\,\sigma_{t_5},\ 10^{-6}F)$.
  - **Holds** if true in every pure-WKB window. **Undecided** if there is none.
  - Numerically seeded windows are not scored: their $\sigma_s$ is reported beside the window RMS
    of $\Delta\theta$ alone (the probe, above).
  - 04 measured $\sigma$ at 0 at the median, quantised, and 0.46–0.78 $F$ at the maximum, for
    production's $\varphi_G$.
- **S2 — $\varphi_T$'s too.** S1's rule for every probe window of a $T$ factor. $T$ has one
  object per $k$, so its $\Delta\theta$ is a constant and every $T$ window is scored.
- **S3 — the cap does not return.** On every row (all 39), under $s$ at `1e-7`, no group has a
  region at depth ≥ 18. An over-budget unit counts as false.
- **S4 — the stored noise was within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_s - {\rm total}_{t_5}| \le$
    `total_abserr`$_{t_5}$, at `1e-7`.
  - A failure measures what the stored noise cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **S5 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $s$ units in budget,
    $|{\rm total}_s(10^{-7}) - {\rm total}_s(10^{-8})| \le$ `total_abserr`$_s(10^{-7})$.
  - 05a's T4, the same rule under $t_5$, held on 29 of 31.

**The pure-WKB tag, and the user's ruling.** A3 v2 stores a pure-WKB object's `z_init` as its
`z_source`, never null (9,552 of 9,552; `out/cmd/28`), so §6's "(`GkSource`'s `z_init` null)" tags
no window. Reading (I), `z_init == z_source` exactly, and reading (L), null, were fixed before the
launch. **The user ruled "Clearly (I)."** (relayed through the laptop session). The question was
relayed at 22:40 UTC, when no `s` unit on R1–R7 had checkpointed; when the ruling reached the
orchestrator at 22:44:56 UTC the checkpoint held 49 units, none an `s` unit on R1–R7. The ruling
was made with no S1 number in existence; this agent read nothing from the run between the launch
check and the ruling.

**Verdicts** (`out/cmd/39_analyse06.txt`; windows in `state/probe_windows.json`):

| | Number | Verdict |
|---|---|---|
| **U0 (a)–(d)** | see "Verification performed" | holds |
| **S1, reading (I), of record** | 6,700 pure-WKB G windows on R1–R7, all matched across trees, tag agreeing in t5, s and the re-run's parents; true in 6,700. `sigma_t5` is exactly 0 in every one; `sigma_s` ≤ 8.8e-14 F (median 5.7e-16 F), at most 8.8e-8 of the bound | **holds** |
| (S1, reading (L), unscored) | 0 windows | (undecided) |
| **S2** | 12,862 T windows (6,071 T_q, 6,791 T_r), all matched; true in 10,747, false in 2,115. Every false window is at z ≥ 1.75e7 (133 in [1e6, 1e8), 1,089 in [1e8, 1e10), 875 in [1e10, 1e12), 18 above), `\|theta_T\|` 0.05–1.5e3 rad; there `sigma_s/sigma_t5` = 0.952–1.04, `sigma` 1.7e-12–1.7e-10 rad = 5–1.5e7 `eps\|theta\|`, largest `sigma_s`/bound 102. The true windows: `sigma_s/sigma_t5` median 3.8e-5 | **fails** |
| **S3** | 39 of 39; largest depth 11 (11 also on the 11 rows capped under B1); no unit over budget | **holds** |
| **S4** | 34 of 39, none undecided; false on `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 (× `total_abserr_t5`; relative 4.0e-6, 2.5e-2, 5.4e-5, 5.7e-7, 1.7e-5) | **fails** |
| **S5** | 28 of 31, none excluded; false on `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31; true on all 11 capped under B1, largest 0.823 | **fails** |

**Numerically seeded G windows (reported, not scored):** 3,594 windows (all at z ≥ 4.2e10);
`sigma_s` 5.5e-7–1.2e-3 F (median 2.1e-4 F) against `sigma_t5` 0–0.78 F (median 0.27 F); the window
RMS of the re-run `deltaTheta` alone is 2.8e-5 F at the median (56 windows above 1e-3 rad, from
`deltaTheta` jumps that are not whole cycles); `sigma_s`/that RMS median 7.8. Per row in
`out/cmd/39`.

**Descriptive:**
- `|total_s - total_t5|`: relative median 1.3e-6, largest 2.5e-2 (`ctrl-low-1`); against
  `total_abserr_t5` median 0.29, largest 10.8 (`low-G-3`); bit-identical on `ctrl-NULL-1`,
  `ctrl-NULL-2`. Three rows have an infinite `total_abserr_t5` (`low-G+r-2`, `low-G-r-1`, `low-Gqr-3`).
- Regions: 901 → 742 (s@1e-7), 1,053 (s@1e-8); depth-20 rows 859 → 702; controls 42 → 40; capped
  under B1 426 → 336 (558 at 1e-8); R1–R7 250 → 200 (332). 48 of 219 group calls change regions,
  depth or convergence; the group sequence (regime, label) is the same in every row. Per row and
  group in `out/cmd/39`.
- `total_converged`: unconverged t5 21, s@1e-7 20 (`mid-G-r-3` converges), s@1e-8 19; controls all
  converged.
- `theta_deriv` (log) at the window centres: T relative change ≤ 4.6e-14; G pure-WKB windows
  median 1.4e-16, max 7.4e-8; G seeded windows median 4.3e-4, p90 0.07, max 4.9 (`out/cmd/43`).
- `GkSourcePolicyData`: `Levin_z` changes on `low-G+r-1` only (`0x1.56a9ce9670b63p+46` →
  `0x1.4edc7c892d90fp+46`); type, quality, crossover unchanged on all 39.
- `convergence_kinds.py` under s (SHA-256 `49c83a72…`, `out/convergence_kinds_s.txt`): 1e-7
  resolution 14, bound only 32, phase-limited 23 (capped rows 9, 23, 12); 1e-8 20, 25, 32 (16, 15,
  20). 05a's t: 18, 32, 17 and 21, 24, 27.
- The re-run values against A3 v2's (39 s@1e-7 units; 3,960,030 G, 73,015 T values): 17 G cycle
  changes, each a single cycle with the new remainder within `4 eps|theta|` of 0 or `-2 pi`; no T
  cycle change; re-run time 1,443 s over the 39 s@1e-7 units (in `rebuild_s`, outside the budgets).

### What shipped

Nothing is committed. Under `var/levin-phase-floor-06/wider/`:
- **`scripts/`**: 05a's harness, changed as deviations 8–16 say; new `rerun.py`,
  `phase_probe06.py`, `u0.py`, `u0_clause_a.py`, `make_copy.py`, `certify_copy.py`,
  `fingerprint_copy.py`, `pure_wkb_zinit.py`, `explore_store.py`, `analyse06.py`,
  `characterise_s2.py`; unchanged and unused: `probe.py`, `analyse.py`, `t0.py`, `l0.py`,
  `draw_sample.py`, `rescore_t1.py`, `t1_true_step_only.py`; `convergence_kinds.py` byte-identical
  to `prompts/levin-phase-floor/measure/`. Diffs against 05a's copy:
  `out/scripts_diff_against_05a_at_launch.txt` and `…_final.txt`.
- **`out/`**: `commands.log` and `cmd/00`–`43` (every command with its output), `U0_a.txt`,
  `U0.txt`, `convergence_kinds_s.txt`, the scripts diffs.
- **`state/`**: `sample.json` (05's, `73aa7876…`), `predictions.md`,
  `predictions_as_first_written.md`, the two `prompt_*.raw`, `U0/`, `smoke/`, `dev/`,
  `scripts_05a_reference/`, `store_files_{before,at_stop,prelaunch,after,after_fingerprint}.txt`,
  `copy_files_{certified,prelaunch,after,after_fingerprint}.txt`, `analysis.json`,
  `probe_windows.json` (every matched window under both trees, with its samples, `z_init`s and
  `deltaTheta`s, for re-scoring S1 and S2), and **`a3v2_copy/`, the scratch copy (408 MB), kept for
  the user to delete**.
- The run directory holds the checkpoints and the unit pickles (86 MB).

### Deviations
1. **Both trees run from worktrees.** STRUCTURALLY REQUIRED.
2. **`prov.py` reads git with `--no-optional-locks`.** IMPLEMENTATION CHOICE. `out/cmd/01` ran a
   plain `git status` before it: UNINTENDED DRIFT (no write seen; worktrees clean throughout).
3. **`rec.sh`'s comments name 06.** IMPLEMENTATION CHOICE.
4. **A compound command with an `rm` was refused by a safety check**, ran nothing, was re-issued
   without it. UNINTENDED DRIFT.
5. **`explore_store.py`**, read-only sizing (`out/cmd/05`). IMPLEMENTATION CHOICE.
6. **Development runs before U0** (`out/cmd/06`, `07`, `17`; `state/dev/`). IMPLEMENTATION CHOICE.
7. **`state/scripts_05a_reference/`**, 05a's harness unchanged, to diff against. IMPLEMENTATION CHOICE.
8. **Route (B): `s` reads a certified scratch copy of A3 v2.** STRUCTURALLY REQUIRED (the user's
   decision). `make_copy.py` (`out/cmd/10`): five files copied byte for byte (SHA-256 equal), then
   `ALTER TABLE … ADD COLUMN theta_residual FLOAT` on the three value tables of each shard.
   `certify_copy.py`: 3,411,578 rows, every one identical by bit pattern, schema identical but for
   the added column, `theta_residual` NULL in every row. Its first run (`out/cmd/11`) said NOT
   CERTIFIED because of the script's own check of the `CREATE` sql (SQLite inserts the column before
   the constraints): UNINTENDED DRIFT, fixed, re-run `out/cmd/12` CERTIFIED. Fingerprint: `python -m
   runregistrykit store fingerprint` loads the main checkout's fingerprinter, whose factories cannot
   read the copy (`out/cmd/13`); `fingerprint_copy.py` calls the `s` tree's same fingerprinter object
   (`out/cmd/14` failed on the `taken` argument's type, my error: UNINTENDED DRIFT; `out/cmd/15`): **`0e32f24a…`,
   problems none**. IMPLEMENTATION CHOICE. `hash_store.py` takes an optional directory.
9. **`harness.py`: four more in-process stand-ins** (`WKB_phase_function.remote` at both producer
   modules, `assemble_GkSource_values.remote`, `apply_GkSource_policy.remote`). STRUCTURALLY REQUIRED.
10. **`rebuild.py`: `rerun=`, `use_store_dir`, the solver map and redshifts in the context, the
    policy summary.** STRUCTURALLY REQUIRED.
11. **`rerun.py`, new: how the producers are re-run.** IMPLEMENTATION CHOICE where §6 leaves the
    method open: integration rows through the factory with `_do_not_populate`, cross-checked against
    the row read by `sqlite3` `mode=ro`; `z_sample` from the stored value rows' redshifts, descending;
    each mode's T rows on its own shard (from `shard_keys`); read-backs formed as the factories form
    them (stored serials, values descending, JSON metadata); `deltaTheta` captured by wrapping
    `atan2` at the producer module for one `store()` call; every re-run value compared with the
    stored one. The `GkSourcePolicyData` is recomputed from the re-run `GkSource` under `s`.
12. **`phase_probe06.py`, new; 05a's `probe.py` copied unchanged and unused.** STRUCTURALLY REQUIRED.
13. **`worker.py`:** trees; `rerun`; a pass-through `sys.stdout` counter of `!! WARNING
    (GkSourcePolicyData` lines; more module evidence; `build_partition`'s arguments kept for the
    probe. Trees/rerun/probe STRUCTURALLY REQUIRED; counter and evidence IMPLEMENTATION CHOICE.
14. **`run_wider.py`, `launch.py`:** trees t5/s, unit sets, `rerun` on every `s` unit, the probe on
    R1–R7 at 1e-7 only, prompt `06`. STRUCTURALLY REQUIRED.
15. **The producer re-run is outside the budget** (in `rebuild_s`, as the stored reads are for t5);
    the budget covers the integral's compute, as in 05/05a. IMPLEMENTATION CHOICE.
16. **U0 by new scripts `u0_clause_a.py` and `u0.py`** (from 05a's `t0.py`); U0 units carry the
    probe; clause (c) run on both gate rows and extended to the whole chain. IMPLEMENTATION CHOICE.
    The first t5 U0 run (`out/cmd/20`) errored in `rerun.py` (T rows read from k's shard, not the
    mode's); fixed, re-run (`out/cmd/21`) into the same ledger, the check reading each unit's last
    record. UNINTENDED DRIFT.
17. **`predictions.md` gained a section after it was written, before the launch:** the pure-WKB
    tag read two ways. A3 v2 stores a pure-WKB object's `z_init` as its `z_source`, never null
    (9,552 of 9,552; `out/cmd/28`), so the prompt's "(`z_init` null)" tags no window. Reading (I),
    `z_init == z_source`, is scored pending the user's ruling; (L) is reported. Found in the smoke
    test, after R7's G window RMS ranges had been seen (no S1 number). The first version is kept as
    `state/predictions_as_first_written.md`. IMPLEMENTATION CHOICE, for the user.
    - **The user's ruling (relayed through the laptop session):** "Clearly (I)." Reading (I),
      `z_init == z_source` exactly, is S1's verdict of record; reading (L) stays reported beside
      it, unscored. The ruling was made while no S1 number existed. The question was relayed at
      22:40 UTC, when no `s` unit on R1–R7 had checkpointed. When the ruling reached the
      orchestrator at 22:44:56 UTC, the checkpoint held 49 units, none of them an `s` unit on R1–R7
      (the orchestrator's account). This agent read nothing from the run between the launch check
      and the ruling.
18. **A driver smoke test** (`ctrl-NULL-1`, `R7`, all sets; `state/smoke/`) after the predictions,
    before the launch. IMPLEMENTATION CHOICE.

19. **The analysis is a new `analyse06.py`** (05a's `analyse.py` copied unchanged and unused), with
    a new read-only `characterise_s2.py` and an inline read-only summary of `theta_deriv` by window
    tag (`out/cmd/43`). IMPLEMENTATION CHOICE.
20. **`convergence_kinds.py` was run from the main checkout with `-I`** from this directory's
    unchanged copy, `--tree s --capped-run <05's run>`, as 05a ran it; `out/convergence_kinds_s.txt`
    drops the per-pickle hash lines (full output `out/cmd/41`). IMPLEMENTATION CHOICE.

Deviations 2 (first command), 4, 8 (the first certification and the first fingerprint call) and
16 (the first t5 U0 run) are the UNINTENDED DRIFT; none touched a store, a tree or a record outside `wider/` and the run directory.

### Verification performed
- **The trees.** t5 at `9f66622` and s at `f5cee4c`, both `git status --short --ignored` empty at
  the start, at the stop, before the launch, on every checkpoint record and at the end; the main
  checkout at `28ebaf0`, status empty, and no tree ran from it. Every record carries `run_wider.py`'s
  SHA-256 `962e12c7…` and its tree's `HEAD` and status; every worker recorded its modules' files,
  all from its own tree, the producers' exact reduction absent under t5 and present under s, and
  which store it read. Ray was never initialised and no lock was taken.
- **U0.** (a) `out/U0_a.txt`. (b)–(d) `out/U0.txt`: (b) 16 + 16 named doubles, 68-float result
  summaries, region arrays, all bit-identical to 05a's units; (c) the clause's four columns and
  every coefficient bit-identical on 241,146 G and 5,172 T values, and the whole chain reproduces
  the stored-input unit bit for bit; (d) both rows' two `s` runs identical in every compared field,
  8,438 and 8,218 probe samples and 1,608 and 1,548 `deltaTheta`. The scripts the run used are the
  ones U0 ran on (`out/cmd/31`).
- **The sample** is `cmp`-identical to 05's, SHA-256 `73aa7876…`, checked before any row ran.
- **The run.** `done`, 109/109 in the planned order, exit 0, problems none, no unit over budget,
  no error, no probe error, no `!! WARNING (GkSourcePolicyData` line (counted per unit, and 0 by
  `grep` of `stdout.log`). `stderr.log` holds 15 scipy `IntegrationWarning`s, as 05 and 05a.
- **The copy** (route B): certified before use (rows, schema, NULLs, fingerprint `0e32f24a…`);
  its five files identical in SHA-256, size and mtime from certification to the end; re-fingerprinted
  after the run, `0e32f24a…`.
- **A3 v2.** Its six files' SHA-256, size and mtime identical before, at the stop, before the
  launch, after the run and after the fingerprint; no journal, `-wal` or `-shm`. `store fingerprint`
  (no `--write`): `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none,
  "matches the fingerprint taken … at b90752d".
- **05's and 05a's records were only read**: 05a's harness is `diff`-identical in `var/` and
  `docs/`, and its run directory holds 122 files.

### Observations not acted on
All in `issues.md`: W1 (no post-06 SHA can read a pre-06 store; route B), W2 (A3 v2's analytic
columns predate `1c19998`), W3 (§6's pure-WKB premise), W4 (S2 badly posed where `|theta_T|` is
small), W5 (S4's five rows), W6 (S5's three rows, `mid-G-1` new), W7 (more phase-limited calls),
W8 (seeded windows), W9 (re-run against stored), W10 (`Levin_z` on one row), W11 (scipy
warnings), W12 (`store fingerprint` and worktrees).

### State handed on
- **Nothing is running.** The run is `done`; no lock was taken; nothing was written in either
  worktree or the main checkout outside `var/levin-phase-floor-06/wider/` and the run directory.
- **A3 v2 is unchanged**, fingerprint `0e32f24a…`.
- **The scratch copy `state/a3v2_copy/` is kept**, for the user to delete once the records are
  committed. It need not be copied into `docs/`.
- **For the orchestrator:** copy `out/`, `scripts/`, `state/` (without `a3v2_copy/`; the U0, smoke
  and dev pickles may be left out), `log.md`, `verification.md`, `issues.md` to
  `docs/levin-phase-floor/06/wider/`; append this section to `logs/06-producers-store-the-residual.md`;
  put `verification.md` in `VERIFICATION.md` §6. Its links assume `06/wider/`.
- **For the user:** S2 fails as written in the windows at high z where `phi_T`'s own structure, not
  storage noise, sets `sigma` (W4); S4 and S5 fail as findings (W5, W6).
