# levin-phase-floor 02 — phase-noise injection on the realistic fixture

**Subject:** prompt 02, run on the laptop (mps026088) in `/Users/ds283/Documents/Code/SecondaryGWKit`
from the dispatch SHA `3f75bbf`, with no Ray, no lock and no store. **It stopped at step 3**, and
the user decided on 2026-10-25 that it is superseded by prompt 03.

**Commit:** this one, the records commit for prompt 02, made on the laptop.

**Result:**
- **Step 2's bit-identity holds** on all eight cases: at δ = 0 the instrumented integral equals the
  unwrapped one in every result field and every per-group record, by `repr`.
- **Step 2's δ = 1e-4 check never ran.**
- **Step 3's probe was stopped.** Its first noised cell (`exact`/`together`/980, white noise,
  δ = 1e-5) ran about 69 minutes without finishing, against 0.4 s at δ = 0.
- The user judged the sweep infeasible and not informative. Noise wider than `rtol` that does not
  shrink with region width defeats the per-region relative test everywhere, so every Levin call
  bisects to the full depth-20 tree. The sweep was **never launched**.
- **N1–N4 are not scored.**
- The probe run is `levin-phase-floor-02-noise-probe-20261025T183809`: `killed`, exit 143, at 16 of
  19 units.

## Predictions and verdicts

The prompt's §4, copied unchanged:

## 4. Predictions, fixed before the sweep runs (README §5 rule 3)

Write these into the log's "Predictions and verdicts" section unchanged, and score them there.
**$\delta^*$** for a case is the smallest rung at which `max_depth` reaches
`DEFAULT_LEVIN_MAX_DEPTH`.

- **N1 — noise alone reaches the cap near `rtol`** (findings §2.6 item 2).
  - In arm A, for every case, the $\delta = 0$ cell stays below the cap.
  - $\delta^*$ exists and is at most $10\,\texttt{rtol} = 10^{-6}$.
  - From $\delta^*$ up, `num_regions` is non-decreasing in $\delta$, within 10 %.
  - **Holds** if all three are true in all eight cases. **Fails** if any is false in any
    `realistic` case.
- **N2 — the exact flavour's flatness is a property of its phases.** Two parts, each scored on its
  own:
  - at $\delta = 0$, each shape's `exact` `num_regions` at $x_{\rm resp} = 10^4$ is within a factor
    of 2 of its value at $980$, which is A2's flatness;
  - at $\delta = 10^{-6}$, the same ratio is at least 3.

  If both hold, noise in the phase is sufficient to produce the cost growth
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` records, and that issue is narrowed to
  say so. It is not closed: sufficiency is not identity.
- **N3 — any group is vulnerable.** In arm B, with white noise at $\delta = 10^{-5}$, **every** label
  reaches the cap when it alone is noised: slow and fast, sum and difference.
  - H-anchor needs this to hold for the fast groups. **It fails H-anchor**, as the account of A3
    v2's fast-group cost, if a fast group stays below the cap at $10^{-5}$.
  - The injection tests whether a group can be driven to the cap, not where A3 v2's noise comes
    from. So N3 cannot confirm either hypothesis. It can only fail H-anchor.
- **N4 — the declared error stays honest under noise.** In every cell of arms A and C with
  $\delta > 0$, $|{\rm total}(\delta) - {\rm total}(0)| \le$ `total_abserr`$(\delta)$.
  - **Holds** if true in every cell.
  - Otherwise report the cells where it fails and their ratio. Each such cell is a case where the
    stored error would understate the true one, which is the honesty layer 2 exists to restore.

Also report, as descriptive and unscored:
- white against staircase at each rung;
- the share of each cell's `elapsed` by group;
- `total_converged` and `total_phase_limited` against $\delta$.


### Verdicts

**The sweep was not run, so none of N1–N4 is scored.** `--report` encodes the scoring rules. They
were written and dry-run on synthetic ledgers before any data existed, but they were never applied
to a measurement.

| | Verdict | Where it now stands (user's decision, 2026-10-25) |
|---|---|---|
| **N1** | **not scored** | Answered by 03 on real A3 v2 rows. The jitter is at the floor in all 29 scored groups (σ_θ/F = 0.22–0.34), and the nested-pair discrepancy is flat over depths 10–20 at D/σ_θ ≈ 0.1. With the jitter removed it falls to round-off, by 3e6–2e8. |
| **N2** | **not scored** | Its original form, the exact flavour's growth under noise, is not tested. The user judged that this does not matter. |
| **N3** | **not scored** | Answered by 03: the floor is in every scored group, cancelling or not, and H-anchor holds in each. |
| **N4** | **not scored** | Moves to 04. The jitter-removed re-run gives a reference `total` for each real row, against which the stored `total_abserr` can be checked. The orchestrator adds this to 04's charter. |

## What shipped

- **[`docs/levin-phase-floor/noise_injection.py`](../../../docs/levin-phase-floor/noise_injection.py)**,
  SHA-256 `6ddaaeca85976b2d5f58af12d03ed703d02caa91784fd976d568da8f042656a7`. It is `black`-clean,
  and unchanged since the probe was launched. Its entry points:
  - `--probe` / `--probe-report`: steps 2–3 as a registered run, and their report from its ledger;
  - `--run` / `--report`: the sweep and its scoring, never launched;
  - `--worker` / `--probe-worker`: internal.

  Every record carries the following provenance:
  - the script's SHA-256;
  - the SHA-256 of the eight files the computation reads (`realistic_large_x.py`, three test
    modules, `phase_groups.py`, `QuadSourceIntegral.py`, `levin_quadrature.py`, `config/defaults.py`);
  - `git HEAD` and dirty state;
  - Python, numpy and scipy versions;
  - the load average before and after the cell, and wall and CPU time.
- **[`docs/levin-phase-floor/02/state/`](../../../docs/levin-phase-floor/02/state/)**: copies of the
  probe run's `checkpoint.jsonl` (16 units, SHA-256 `37e01a26…`), `manifest.json`, `status.json`,
  `stdout.log` and `stderr.log` (empty). The run directory stays in `var/runs/`.
- **[`docs/levin-phase-floor/02/out/probe_02.txt`](../../../docs/levin-phase-floor/02/out/probe_02.txt)**:
  `--probe-report` on the copied ledger.
- **`docs/levin-phase-floor/VERIFICATION.md` §2**.
- **This log and the board's 02 row and status line.** `docs/OPEN_ISSUES.md` and the `handover`
  board are unchanged: 02 narrows nothing.

## Deviations from the prompt

1. **The probe ran as a registered, detached run.** `STRUCTURALLY REQUIRED`.
   - The prompt registers only the sweep (step 4) and runs the probe directly (step 3).
   - A scratch timing of `exact`/`together`/980 at δ = 1e-5 was killed by its own 540 s alarm
     before finishing, so the probe was itself a long job.
   - `CLAUDE.md` ("Long-running jobs") and campaign README §5 rule 7 require such a job to be
     registered, launched detached and checkpointed. It must not be owned by a conversation.
   - So `--probe` uses the same launcher as `--run`: slug `noise-probe`, a `nohup` worker in a new
     session, its pid recorded with `Run.heartbeat(pid=…)`, and one ledger line per cell.
   - `--probe-report` reads that ledger alone.
2. **Step 2's bit-identity check was widened to all eight cases.** `IMPLEMENTATION CHOICE`.
   - The prompt asks for "a δ = 0 cell". Each case was run unwrapped and wrapped.
   - This also took every δ = 0 baseline before any noised cell, so the §5 stop on a baseline at
     the cap would fire before anything expensive ran.
3. **A per-group depth observer on `adaptive_levin_sincos`.** `IMPLEMENTATION CHOICE`.
   - It is a pass-through wrapper on the module attribute `ComputeTargets.QuadSourceIntegral.adaptive_levin_sincos`.
   - It records each Levin call's `max_depth`, regions, evaluations, `elapsed` and flags by group
     label. The stored metadata keeps only the maximum depth over all groups.
   - It was meant to score N3 on the noised label's own depth, and to read `theta_deriv` for arm B's
     slow/fast labelling, but only after the driver had returned.
   - The analytic oracle (`_three_bessel_Levin`) reaches the same attribute. Its calls are recorded
     as kind `analytic` and kept out of every per-group quantity.
   - The bit-identity of §2.1 covers this wrapper too.
   - It is allowed under README §5 rule 1.
4. **The white noise reads the blake2b bytes big-endian.** `IMPLEMENTATION CHOICE`. The prompt says
   "the first eight bytes … divided by 2^64", and does not give their byte order.
5. **The build lines of `realistic_large_x.run_cell` are repeated, not called.** `IMPLEMENTATION CHOICE`.
   - `run_cell` calls `case.run` at `REF_RTOL = 1e-12`, so it cannot be used as it stands.
   - `build_case` imports `scaled_shape` and `seed_fixtures` from that module, which is loaded from
     its file and not edited, and repeats the rest: `w_of_b(0.0)` and `Case(b=0.0, …)`, with the seam
     closed by not calling `open_the_seam`.
   - Its checkpoint `var/runs/realistic_large_x_cells.jsonl` was not written.
6. **`ComputeTargets.QuadSourceIntegral` is taken with `importlib.import_module`.** `STRUCTURALLY REQUIRED`.
   `ComputeTargets/__init__.py` re-exports a class named `QuadSourceIntegral`, which shadows the
   submodule, so `import ComputeTargets.QuadSourceIntegral as QSI` binds the class.
7. **The stop at step 3, and the supersession by 03.** `STRUCTURALLY REQUIRED`, by the user's
   decision of 2026-10-25.
   - The user stopped the probe by SIGTERM at 19:47:59Z, after 16 of 19 units, during its first
     noised cell.
   - The user then ruled that the sweep is not run in any form, because 03 answers the question
     on real rows.
   - So steps 4–5 did not happen, step 3's projection does not exist, the δ = 1e-4 half of step 2
     never ran, and acceptance items 1–3 cannot be met as written.
   - This is a decision about the prompt, not a choice made in carrying it out.
   - The user's earlier ruling, that a projection near 24 h must run on the Mac Studio, became moot.
8. **`VERIFICATION.md`'s placeholder line was replaced.** `IMPLEMENTATION CHOICE`. 01 had left
   "§2 is reserved for prompt 02, which has not landed." where §2 goes. §2 replaces that line and
   changes nothing else.

## Verification performed

- **The gate held at the start:**
  - `git status --short` was empty;
  - `runregistrykit list` showed nothing `running` or stale;
  - load average ~20 at 18:14.
- **Dry runs before launch.** `sweep()`, `probe_sweep()`, `--report` and `--probe-report` were run
  end to end on synthetic ledgers in the session scratchpad, built from one real cell. They found
  three defects, all fixed before the probe was launched:
  - the observer counted the analytic oracle's calls as phase groups;
  - the record label `tag` broke the bit-identity comparison;
  - a doubled bold mark in the output.

  An earlier smoke test showed that a wrapped and an unwrapped δ = 0 cell agree.
- **Step 2, re-checked from the copied checkpoint.** A field-by-field comparison of each unwrapped
  and wrapped δ = 0 record agreed exactly, by `repr`, on all eight cases:
  - `total`, `total_abserr`, `numeric_quad`, `WKB_Levin` and `analytic_rad`;
  - `max_depth`, `num_regions`, `num_simple_regions` and `evaluations`;
  - `total_converged` and `total_phase_limited`;
  - all 48 group records over 22 sub-intervals, with `elapsed` excluded.

  The script's own comparison in `probe_02.txt` agrees.
- **The ledger's provenance.** All 16 records carry the script hash `6ddaaeca…`, `git_head`
  `3f28791` and `git_dirty: true`. The committed script still hashes to `6ddaaeca…`, and
  `black --check` passes.
- **The copies.** `checkpoint.jsonl` in `02/state/` has the same SHA-256 as the run directory's.

## Observations not acted on

1. **The noise model could not discriminate at this tolerance.** This is the user's reason for
   stopping, recorded here because it is a property of the design and not of the run.
   - The injected error does not shrink with region width.
   - With δ > `rtol`, and with the per-region test relative to a group's own value, no region of a
     noised group can be accepted, so the driver bisects everywhere to the cap.
   - So arms A–C would mostly have measured the full depth-20 tree, and cost about 2^20 regions
     per Levin call, and not anything about δ.
   - 03 measured the real jitter's discrepancy directly instead (D/σ_θ ≈ 0.1, flat with depth).

   No issue is opened. Nothing in the code is defective. It is a lesson about the harness, and the
   question it was meant to answer is answered by 03.
2. **The probe's δ = 0 depths are not small for the realistic flavour** (6–8, against 0–2 for
   exact). At rtol 1e-7 the realistic fixtures' representation already makes the driver subdivide.
   A2 recorded the same thing at rtol 1e-12 (`REALISTIC-LARGE-X.md` Table 8). No issue: it is the
   fixture's known representation floor.
3. **`ComputeTargets/__init__.py` shadows the submodule `QuadSourceIntegral` with a class of the
   same name** (deviation 6). That is how the package exports it, and other harnesses wrap the
   attribute through `sys.modules` or `importlib` too. It is not a defect for this campaign, so no
   issue is opened.

## State handed on

- **The probe run** `levin-phase-floor-02-noise-probe-20261025T183809` is `killed`, exit 143, and is
  finished as far as the registry is concerned. Its directory stays in `var/runs/`, and its pid
  70545 is gone.
- **No sweep run exists.** No `noise` run was registered.
- **N4's question passes to 04's charter**, which the orchestrator amends. N1 and N3 are answered
  by 03. N2 is dropped.
- **The script stays in the tree** as the record the probe's ledger names. It must not be edited
  while that ledger is to be re-read: `--probe-report` discards records under another hash.
- **Nothing changes in `docs/OPEN_ISSUES.md` or on the `handover` board.**
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` and
  `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` are not narrowed by 02.
