# Source-remediation campaign verification and close-out

**Campaign:** [`prompts/source-remediation/README.md`](../prompts/source-remediation/README.md)
**Status board:** [`prompts/source-remediation/IMPLEMENTATION_STATE.md`](../prompts/source-remediation/IMPLEMENTATION_STATE.md)
**Source audit:** [`docs/spec-code-audit-2026-09.md`](spec-code-audit-2026-09.md) §4, §7, §8
**Prompt:** [`prompts/source-remediation/12-verification.md`](../prompts/source-remediation/12-verification.md)
**Date:** 2026-09-09
**Tree:** every measurement below was taken with the campaign's code at `95cc326` (prompt 11's
commit, the campaign tip). While this pass was running, three commits by the user landed on the
branch — which was also renamed `bessel-remedial-plan` → `transfer-remedial-plan` — and this
commit sits on top of them. They are `14fb9f6`, `5455bde` and `c0d8a83`, and
`git diff --name-only 95cc326 c0d8a83` returns **nothing outside `prompts/transfer-remedial/`**
(the sibling campaign's planning folder, formerly `prompts/bessel-remedial/`): no production code,
no `docs/`, nothing this document measures. The numbers below therefore describe the tree as
committed at `c0d8a83` as well as at `95cc326`.

---

## 1. What this document is

Prompts 01–11 landed A1–A7, B1–B11 and the `QuadSource`/`QuadSourceIntegral` refactor. Not one
of them touched a datastore or a Ray cluster: every log says so explicitly, and four of the
audit's own findings (§4.1–§4.4) were deferred here because they need stored data. This document
records the two layers prompt 12 asks for — the whole offline test tree and all 29 audit
reproduction scripts, then one live, scoped pipeline run — and states plainly what each one
settled and what it did not.

**Headline results.**

1. **The pipeline runs end to end again, and the source integral reproduces its own oracle where
   the oracle is valid.** A real `ShardedPool`/`RayWorkPool` run on a locally bootstrapped Ray
   instance completed every stage up to and including `GkSourcePolicyData`, and 1813 real
   `QuadSourceIntegral` work items were computed through the production `@ray.remote`
   `compute_QuadSource_integral` with the nine-key payload prompt 10 assembles. Against
   `analytic_rad`, `total` agrees to **1.2e-6 – 1.4e-4** relative wherever the response redshift
   is at most a few oscillations inside the horizon, and to **6.2e-5** on the rows that have no
   Liouville–Green hand-over inside their integration range. Before this campaign that
   comparison could not have been made at all: the `--quad-source-queue` stage crashed (A3) and
   `QuadSourceIntegral` read an oscillating source off a cubic spline (A2).
2. **A new run-blocking defect was found, and is not fixed here** (prompt 12's own instruction).
   `build_partition`'s region guard compares redshifts with an *absolute* tolerance of
   `DEFAULT_FLOAT_PRECISION = 1e-7`, which is meaningless at z ≳ 1e8. **1372 of 3185 (43 %)**
   production work items abort in it, and because `main.py`'s stage propagates the first failure,
   the `--quad-source-integral-queue` stage cannot complete at all. Board issue
   `[12-region-check-absolute-tolerance]`.
3. **Audit §4.1 is closed** (G is continuous at `crossover_z` to a median 7.4e-8, worst 4.0e-6),
   **§4.2 is closed in the affirmative** (the `minimal` band *is* reached — 1 of 462 policy rows —
   so A6 was a live defect, not a dead branch), **§4.3 is closed as unreachable in this
   configuration** (`has_WKB_violation` is set on 0 of 7 `TkWKBIntegration` and 0 of 5488
   `GkWKBIntegration` rows, so no policy decision is needed), and **§4.4 is closed** on a real
   QCD/GenericEOS background: every end-point relative error is now at or below the interior
   median (worst end/interior ratio 8.5, and ≤ 0.76 with the most accurate reference), against
   3.0e-01 and 3.8e-01 before prompt 03.
4. **Two further findings the user has to decide about.** The stored `total_abserr` is doing its
   job and it reports that `DEFAULT_QUADRATURE_ATOL = 1e-25` is *too loose* for this integral at
   production wavenumbers — 58 % of work items have a raw integral smaller than `atol`
   (`[12-atol-too-loose-for-the-source-integral]`); and sub-horizon, where every row carries a
   hand-over clamp gap, the residual against the oracle rises to a median 0.07–0.26, one to two
   orders above what prompt 08's fixtures predicted for `[08-handover-clamp-error]`
   (`[12-handover-clamp-error-in-production]`).

Nothing in production code was changed by this prompt.

---

## 2. Reconciliation

### 2.1 Git history

```
e9a43a2 (baseline: the audit documents, clean `main`)
0f50782 Exclude Lambda from the GenericEOS perturbation sound speed        (prompt 01)
3199c7b Fix WKB value, policy and label hygiene slips                      (prompt 02)
f8c75f5 Remove the grid-end bias in the background derivative splines      (prompt 03)
f2d973b Add orchestrator prompts for every source-remediation workstream   (campaign planning)
e3348e4 Add a two-region LG representation of T_k for source consumers     (prompt 05)
67aaa57 Record the hand-over and stored-phase accuracy issues found ...    (orchestrator review)
3df5604 Restrict QuadSource to the region where both T_k are numeric       (prompt 06)
f06f587 Add the phase-group decomposition of the source integrand          (prompt 07)
34d5cc0 Record the draft plan for the Bessel amplitude-phase rebuild       (unrelated campaign)
96f0d0d Require the closed-form tail in the Bessel rebuild plan            (unrelated campaign)
c4c4905 Set the high-order acceptance target to 1e-6                       (unrelated campaign)
39ed7fc Review Green function WKB phase accuracy                           (unrelated)
4afd531 Partition the source time integral and Levin-integrate its ...     (prompt 08)
ffc50ae Record b, an error bound and honest tolerances on ...              (prompt 09)
154126b Filter QuadSourceIntegral work items to triangle-closing triples   (prompt 04)
815217b Supply the transfer functions to the source integral stage         (prompt 10)
95cc326 Record the spec-code audit's three recommended annotations         (prompt 11)
14fb9f6 Plan the Bessel amplitude-phase rebuild as nine prompts            (unrelated campaign)
5455bde Rename the campaign from bessel-remedial to transfer-remedial      (unrelated campaign)
c0d8a83 Correct what source-remediation did to the transfer function       (unrelated campaign)
d242647 Verify the source-remediation campaign against a live scoped run   (prompt 12)
```

**One commit per prompt, in prompt order** (04 is serialised after 09, as its board row records —
workstream D was dispatched in parallel with C and the orchestrator rebased it). Seven commits
unrelated to this campaign are interleaved: six belonging to the sibling
`prompts/transfer-remedial/` campaign (`34d5cc0`, `96f0d0d`, `c4c4905`, `14fb9f6`, `5455bde`,
`c0d8a83` — the last three landed *during* this verification pass, and the folder and branch were
renamed from `bessel-remedial` at `5455bde`) and one `Review Green function WKB phase accuracy`
(`39ed7fc`), plus two of this campaign's own bookkeeping commits (`f2d973b` orchestrator prompts,
`67aaa57` the review that opened `docs/lg-phase-and-handover-followup-2026-09.md`). Nothing was
rewritten, rebased over, or squashed; every campaign commit named in the audit's §8 disposition
table exists with the subject recorded there (`git cat-file -e` on all ten, prompt 11's own
verification).

`git diff --name-only 95cc326 c0d8a83` touches nothing outside `prompts/transfer-remedial/`, so
none of the three commits that arrived mid-pass changed anything measured here.

The campaign's target branch was `main`; the work is on `transfer-remedial-plan` (renamed from
`bessel-remedial-plan` mid-pass), which is five campaign commits ahead of `main` (`main` sits at
prompt 07). This prompt did not switch, rebase or merge anything, and staged only its own files by
name.

### 2.2 Open issues on the board going into this prompt

Nine issues were open in `IMPLEMENTATION_STATE.md` §3. This pass touches them as follows; §7 has
the board updates.

| Issue | Status after this pass |
|---|---|
| `[01-genericeos-tz-spline-floor]` | confirmed live: the QCD background's derivative columns agree with an independent reference to ~1e-11 (ε) and ~4e-4 (w″), both above the 1.3e-9 T(z)-spline floor the issue names; no tolerance below it was used. Stays open as a standing note. |
| `[03-derivative-pad-clamp-on-coarse-grids]` | not exercised: the run used 100 samples per log10(1+z), where the clamp does not bind. Stays open. |
| `[05-numeric-region-is-now-the-accuracy-floor]` | measured live: `T_WKB` against the exact `analytic_T_rad` on real `TkWKBValue` rows is 7.1e-5 – 1.3e-3 of envelope (median, per mode), worst 5.8e-2 — one to two orders *worse* than the 6.1e-6 the offline fixture gave, at production x. Narrowed, stays open. |
| `[06-source-spline-residual-vs-handover]` | measured live: 1.4e-5 – 1.4e-4 of envelope between nodes at the bottom of the both-numeric region, i.e. better than the 4.5e-4 worst case the issue records. Narrowed, stays open. |
| `[07-lg-derivative-truncation-at-handover]` | not separated from the clamp error by this pass; see `[12-handover-clamp-error-in-production]`. Stays open. |
| `[07-phase-spline-chunking-precision]` | not exercised (no test of remainder composition here). Stays open. |
| `[08-handover-clamp-error]` | **confirmed and worse than predicted** in production; superseded in part by `[12-handover-clamp-error-in-production]`. |
| `[09-abserr-is-a-quadrature-bound]` | confirmed live, and sharpened: at pipeline tolerances the bound is not merely incomplete, it is *loose* — see `[12-atol-too-loose-for-the-source-integral]`. |
| `[09-WKB_quad-columns-are-vestigial]` | confirmed: `WKB_quad` is 0.0 on all 54 stored and all 1813 computed rows. Stays open. |
| `[10-levin-wholesale-cc-fallback]` | cost measured in production; see §5.8. Stays open. |
| `[10-classify-levin-keyerror]` | **not reached**: `apply_GkSource_policy` ran 462 + 21 times without a `KeyError`, so no shipped mode has `|dθ_G/dlog(1+z)|` below `Levin_threshold = 1.5` everywhere. Narrowed, stays open. |

---

## 3. Layer 1 — offline verification

### 3.1 Unit-test tree

From the repository root, `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <pkg>/tests -t .`:

| Package | Result |
|---|---|
| `CosmologyModels/tests` | **Ran 5 tests in 0.017 s — OK** |
| `AdaptiveLevin/tests` | **Ran 32 tests in 0.056 s — OK** |
| `ComputeTargets/tests` | **Ran 81 tests in 235.4 s — OK** (includes prompt 10's `test_main_plumbing`, 7 tests) |
| `ComputeTargets/tests/sympy_phase_groups.py` (run directly) | **"all residuals are identically zero"**, exit 0 |
| `LiouvilleGreen/tests` | **started, terminated before completing** (exit 143, SIGTERM, two tests in) — no result; out of this campaign's scope, README §5 item 8. See §6 item 5 |

118 unit tests, no failures, no skips. This is the whole tree the campaign wrote or touched.

### 3.2 The 29 audit reproduction scripts

All run from the repository root with `PYTHONPATH=. ./venv/bin/python docs/spec-code-audit/scripts/<name>.py`.
**28 of 29 exit 0; one exits non-zero, for a reason the campaign created deliberately.**

| Script | Expectation after the campaign | Observed |
|---|---|---|
| `GK_01_sympy_ode` | unchanged (sympy identity) | difference `0` on both forms |
| `GK_02_sympy_omega` | unchanged | four residuals `0`; `d_ln_omegaEff_dz == ω′/ω` `True` |
| `GK_03_numeric_analytic` | unchanged | same table as the audit |
| `GK_04_wkb_matching` | unchanged | worst `|G_WKB/G_exact − 1|` = 1.849e-03 |
| `GK_05_phase_reassembly` | unchanged | worst reassembly 2.220e-16; spline mod-2π 7.208e-13 |
| `GK_06_gksource_stitch` | unchanged | both branches same sign/normalisation, `sin_coeff > 0` |
| `GK_07_sympy_lnH` | unchanged | three residuals `0` |
| `GK_08_sympy_Tk_omega` | unchanged | code numerator − d(ω²)/dz `= 0`; pre-fix form non-zero (historical) |
| `GK_09_stage2_Q` | unchanged | `θ_Q/θ_direct − 1` ≤ 2.9e-12 |
| `QI_01_prefactor` | unchanged | `f_R11/f_R28 = (2b+3)²/(b+2)² = 1/c²` `True` |
| `QI_02_analytic_numeric` | slightly *better* — prompt 09 (B6) made `analytic_integral` honour the script's `rtol=1e-10, atol=1e-25` | runs; `code` vs spec R14 6.8e-09 … 3.8e-07 over its configurations, same 1e-8–1e-6 band the audit reports |
| **`QI_03_measure`** | **section 1 runs and agrees; sections 2–4 must fail** — they call `WKB_quad_integral`/`WKB_Levin_integral`, deleted by prompt 08 (log 08 deviation 4, observation 2) | section 1: `numeric_quad_integral` vs the R28 measure **4.267e-16**; then `AttributeError: module 'ComputeTargets.QuadSourceIntegral' has no attribute 'WKB_quad_integral'`, exit 1. **By design** |
| `QI_04_triangle` | unchanged; it *is* the A5 measurement, and prompt 04 reproduced it exactly | 5133 of 63750 (8.05 %); 3 nodes within ±0.05 of s=√3 |
| `QI_05_f_normalisation` | unchanged | `f_code/f_R28 = 1/c²` to ≤ 2.0e-13 |
| `QS_01_sympy_f` | unchanged | `spec f − code f = 0`; symmetry exact |
| `QS_02_deriv_and_f` | unchanged | `f(q,r) == f(r,q)` bit-identical |
| `QS_02b_deriv_mpmath` | unchanged | `compute_analytic_Tprime` is exactly `dT/dz` |
| `QS_02c_cancellation` | unchanged | catastrophic-cancellation table as recorded |
| **`QS_03_spline_error`** | **measures the A2 defect and must still show it** — it splines the analytic source on its own grid, and does not call `QuadSource` | reproduces the audit table (1.28 and 1.21 of envelope at x = 1e3, 1e4). **Pre-fix behaviour by design**; §5.7 measures the post-fix quantity on real rows |
| **`QS_04_coverage_and_jacobian`** | **part (a) must now raise** — prompt 06 changed `compute_quad_source` to read `Tk.values`, and the script's `MockTk` predates that protocol (log 06 observation 2) | part (a) prints `RAISED AttributeError: 'MockTk' object has no attribute 'values'` and continues; part (b) sympy Jacobian `difference = 0`; exit 0 |
| `QS_05_oracle_quality` | unchanged (a statement about the `analytic_source_w` oracle) | same text |
| `TK_01_Tk_ODE` | unchanged | `code RHS − spec RHS = 0` |
| `TK_02_omegaEff` | unchanged | four residuals `0` |
| `TK_03_numeric_vs_analytic` | unchanged | 5.6e-06 → 3.2e-09 under refinement |
| `TK_04_WKB_reconstruction` | unchanged | worst 4.363e-03 of envelope |
| **`TK_05_background_derivatives`** | **section (d) must still print the A1 discrepancy** — it recomputes the defective expression inline from a `LambdaCDM` object rather than calling `LambdaCDM_GenericEOS` (log 01 observation 1) | ratio 3.2138 at z=0, byte-identical to the audit. **Pre-fix behaviour by design**; §5.6 measures the fix on stored data |
| **`TK_06_spline_end_bias`** | **must still print the A7 table** — its docstring says it drives "the *same* code path", i.e. a copy of the pre-fix `make_interp_spline` stack, not `compute_background` | reproduces the audit's TK-5 table exactly (ε″ 3.005e-01 at the z=0.1 end). **Pre-fix behaviour by design**; §5.6 measures the fix |
| **`TK_07_omegaEff_spline_impact`** | same, one level down | ω_eff² 2.131e-07 and d ln ω_eff/dz 2.631e-04 at z=0.1, as recorded. **Pre-fix behaviour by design** |
| `TK_08_criterion_sign` | unchanged; it documents why B3's warning was dead code | `d ln ω_eff/dz` negative throughout |

Five scripts (`QS_03`, `QS_04`(a), `TK_05`(d), `TK_06`, `TK_07`) are *records of the pre-fix
state* that re-implement the defective code path inline, so they cannot show a fix; one (`QI_03`
sections 2–4) calls functions prompt 08 deleted. That is the complete list of scripts whose output
a post-campaign reader could misread, and §5.6/§5.7 give the corresponding post-fix measurements
on stored data.

---

## 4. Layer 2 — the live runs

### 4.1 Environment and exact commands

No Ray cluster was running. `ray.init(num_cpus=N, include_dashboard=False)` bootstraps a genuine
local instance in this environment (the backport campaign's verification found the same), and
`docs/source-remediation-verification/scoped_pipeline_run.py` patches `main.py`'s own
`ray.init(address=...)` call to do exactly that. The machine has 10 CPUs and 16 GB RAM; the
volume holding the datastores was 98 % full with 24–26 GB free throughout, and Ray printed its
`file_system_monitor` warning ("over 95 % full … object creation will fail if spilling is
required") every 10 s in every run. No spilling occurred and no run failed for space.

`main.py` cannot be imported (it parses `sys.argv`, opens a Ray connection and a `ShardedPool` at
module scope and then runs the pipeline), and its wavenumber sample is hardcoded at 50+50. The
driver therefore reads `main.py`'s source, replaces the two occurrences of the literal
`np.logspace(np.log10(1e5), np.log10(3e8), 50)` with a name bound in the execution globals, and
`exec`s the result; it also optionally filters `config.model_list.build_model_list`. Nothing else
is changed — every stage, tag, tolerance and queue parameter is `main.py`'s, and the driver prints
what it substituted.

**Run A — `LambdaCDM` (Planck 2018), the main run.** Datastore
`/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/c81575f0-880e-4187-b2ed-e78031b1eb8c/scratchpad/runA/verify-LambdaCDM.sqlite`
(a path that did not previously exist; nothing was overwritten, migrated or deleted, and the
driver refuses to start on an existing path unless `--allow-existing` is given).

```
PYTHONPATH=. ./venv/bin/python docs/source-remediation-verification/scoped_pipeline_run.py \
    --k-min 1e5 --k-max 1e7 --k-count 7 --cpus 10 --models LambdaCDM \
    -- --database <scratchpad>/runA/verify-LambdaCDM.sqlite \
       --job-name source-remediation-verify-LCDM --shards 4 \
       --zend 1e7 --source-samples-log10z 100
```

7 log-spaced wavenumbers over two decades (1e5 … 1e7 /Mpc) as both the source and the response
sample, production redshift density (100 per log10(1+z)), production response sparseness (12).
`z_exit(1e7) = 4.635e12`, so the grid runs from `z_exit_suph_e5 = 6.8788e14` down to `z_end = 1e7`
in **784 source redshifts** and **66 response redshifts**. Largest Bessel argument
`x = 4.635e5` (+7.5 % clearance). The triangle filter kept **3185 of 12740** (k,q,r,z) work items
(25.00 %), from **49 of 196** triangle-closing (k,q,r) triples — covering all three shapes prompt
08 tested (q≈r≈k: 7 triples; q≈r≫k: 9; q≪k≈r: 9).

`z_end = 1e7` rather than the production `0.1` was chosen deliberately and it is the one place
this run is not production-shaped. It keeps the entire run inside radiation domination
(ρ_m/ρ_r = 3403/(1+z) ≤ 3.4e-4), which is the regime in which `analytic_rad` — a *pure*-radiation
oracle — is a valid reference, as prompt 12 §2 item 1 asks. §6 item 1 states what that costs.

Stage timings (10 CPUs, one `LambdaCDM` model; total wall clock **4.5 minutes**):

| Stage | Items | Wall |
|---|---|---|
| horizon exit times (source + response) | 7 + 7 | 2.6 s |
| `BackgroundModel` | 1 | 0.85 s |
| Bessel phase splines (ν = 0.5, 2.5) | 2 | 0.10 s |
| `TkNumericIntegration` | 7 | 1.5 s |
| `TkWKBIntegration` | 7 | 0.56 s |
| `QuadSource` | 28 | 1.0 s |
| `GkNumericIntegration` | 3436 | 1 m 55 s |
| `GkWKBIntegration` | 5488 | 1 m 37 s |
| `GkSource` | 462 | 25.6 s |
| `GkSourcePolicyData` | 462 | 6.1 s |
| `QuadSourceIntegral` | 3185 scheduled | **aborted** — §5.1 |

Datastore size: **66 MB** (4 shards of 9–19 MB, plus a 32 kB primary file). Stored rows:
784 `BackgroundModelValue`, 7 + 7 `Tk*Integration`, 28 `QuadSource` / 11 952 `QuadSourceValue`,
3436 `GkNumericIntegration`, 5488 `GkWKBIntegration`, 462 `GkSource` / 182 028 `GkSourceValue`,
462 `GkSourcePolicyData`, 0 `QuadSourceIntegral`.

**Run B — `QCD_Cosmology` (a `LambdaCDM_GenericEOS`), background only.** For audit §4.4 and A1
the production redshift range matters and the compute queues do not, so this run has every queue
disabled and reaches `z = 0.1`:

```
PYTHONPATH=. ./venv/bin/python docs/source-remediation-verification/scoped_pipeline_run.py \
    --k-min 1e4 --k-max 1e5 --k-count 3 --cpus 2 --models QCD_Cosmology \
    -- --database <scratchpad>/runB/verify-QCD-background.sqlite \
       --job-name source-remediation-verify-QCD-bg --shards 2 \
       --zend 0.1 --source-samples-log10z 100 \
       --no-Tk-numeric-queue --no-Tk-WKB-queue --no-quad-source-queue \
       --no-Gk-numeric-queue --no-Gk-WKB-queue --no-Gk-source-queue \
       --no-quad-source-integral-queue
```

1401 redshifts over z ∈ [0.1, 1.0168e13] at 100 per log10(1+z) — the same shape as the grid the
audit's TK-5 table used (n = 1300, z ∈ [0.1, 1e12]). Background computed and stored in 1.08 s;
datastore 2.1 MB.

**Smoke run — the same driver at 10 samples per log10(1+z)**, 3 modes over 1e5 … 1e6, `zend = 1e6`,
which is the only run whose `--quad-source-integral-queue` stage completed (§5.1 explains why: its
seven response redshifts happen not to trip the guard). 54 `QuadSourceIntegral` rows were computed
**and stored**, and re-running the same driver against that datastore found all of them present
(§5.9).

**QuadSourceIntegral harness.** Because `main.py`'s stage aborts on the first failure, the
production work list was scheduled instead by
`docs/source-remediation-verification/run_quadsource_integrals.py`, which calls the real
`@ray.remote compute_QuadSource_integral` with the same nine-key payload
`main.py:build_QuadSourceIntegral_payload` builds, records failures instead of raising, and stores
nothing:

```
PYTHONPATH=. ./venv/bin/python docs/source-remediation-verification/run_quadsource_integrals.py \
    --database <scratchpad>/runA/verify-LambdaCDM.sqlite --shards 4 --cpus 10 \
    --zend 1e7 --samples-per-log10z 100 --z-stride 1 --include-blocked \
    --out <scratchpad>/runA/qsi_full.jsonl
```

All 3185 work items of run A's own stage: **1813 completed, 1372 failed**, every failure in the
region guard of §5.1. Wall clock 12 minutes.

### 4.2 Analysis scripts

Committed under `docs/source-remediation-verification/`; all read-only against the shard files
(`mode=ro`) except where they open a `ShardedPool` with `prune_unvalidated=False` and never call
`object_store`/`object_validate`:

| Script | What it produces |
|---|---|
| `scoped_pipeline_run.py` | the runs above |
| `run_quadsource_integrals.py` | the `QuadSourceIntegral` work list, as JSON lines (`--rtol`/`--atol` for the convergence study of §5.4) |
| `analyse_quadsource_integral.py` | §5.2, §5.3, §5.8 (reads either the stored table or the JSON lines) |
| `analyse_greens_and_source.py` | §5.5 (audit §4.1, §4.2, §4.3) and §5.7 |
| `analyse_background.py` | §5.6 (audit §4.4 and A1 on stored data) |

---

## 5. What Layer 2 measured

### 5.1 A new run-blocking defect: the region guard's absolute tolerance

`main.py`'s `--quad-source-integral-queue` stage aborted run A on its first work batch:

```
RuntimeError: compute_QuadSource_integral: sub-interval bottom z=4.8682e+14 is out-of-bounds
for the Gk numeric region (6.8788e+14, 4.8682e+14) [domain=5.4628e+14, 4.8682e+14]
```

The sub-interval bottom and the region bottom are *the same redshift*. `build_partition` carries
its breakpoints in `log(1+z)` and recovers `z_lo = exp(log_lo) - 1.0`
(`QuadSourceIntegral.py:440`), and `_check_region_covers` (`:561-570`) then compares that against
the region boundary with an **absolute** tolerance, `DEFAULT_FLOAT_PRECISION = 1e-7`:

```python
if z_min < region_min_z - DEFAULT_FLOAT_PRECISION:
    raise RuntimeError(...)
```

At z = 4.9e14 one unit in the last place is 0.06, so the log/exp round trip moves `z` by up to
~1e7 times the tolerance and the sign of that movement decides whether the guard fires.
Reproduction, needing no pipeline (`exp(log(1+z)) - 1 < z - 1e-7`) over run A's own grid:

| grid | redshifts | tripping the guard |
|---|---|---|
| response sample | 66 | **29 (44 %)**, smallest tripping z = 6.93e07, largest non-tripping z = 3.69e14 |
| source sample | 784 | **331 (42 %)** |
| smoke run's response sample | 7 | 0 — every one of its seven redshifts round-trips *upward*, which is why that run's stage completed |

Scheduled item by item, **1372 of 3185 (43 %)** production work items raise: 875 on the
`Gk numeric` region and 497 on the `Gk WKB` region, spread over 28 distinct response redshifts
from 6.93e07 to 4.87e14. No other failure mode occurred. The Green's-function region always ends
exactly at `z_response`, so the last sub-interval's bottom always coincides with a region
boundary and the guard is decided by rounding alone; the same applies to the top check (25 of the
66 response redshifts round-trip *upward* by more than 1e-7).

This is new code from prompt 08 (`build_partition` did not exist before) and it is the same class
of defect as audit B11, which prompt 09 closed by confirming that the *sub-interval width* guard
is in `log(1+z)`: the region-coverage guard next to it is not. It fails loudly, so no wrong number
is at risk, but **the stage cannot complete at production redshifts**. Board issue
`[12-region-check-absolute-tolerance]`; per prompt 12's instruction it is reported, not fixed.

### 5.2 `total` against `analytic_rad` (prompt 12 §2 item 1)

1813 rows, all with `b = 0.0`, 49 distinct (k,q,r), 37 distinct response redshifts from 1e7 to
3.69e14. Residuals are quoted relative to
`scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)`, the normalisation prompt 08 used, so
that a cancelling row is not flattered or punished; `x_resp = max(q,r) c_s a_0 η(z_response)` is
the accumulated phase of the faster transfer function at the response time, i.e. how far
sub-horizon the integral reaches.

| z_response | x_resp | rows | median residual | p90 | rows whose `total_abserr` < 1 % of `total` | median residual on those |
|---|---|---|---|---|---|---|
| 3.692e+14 | 0.0073 | 49 | **7.6e-05** | 8.1e-05 | 49 | 7.6e-05 |
| 7.024e+13 | 0.038 | 49 | **2.0e-06** | 3.0e-06 | 49 | 2.0e-06 |
| 2.323e+13 | 0.115 | 49 | **1.2e-06** | 2.4e-06 | 49 | 1.2e-06 |
| 1.462e+12 | 1.83 | 49 | **1.4e-04** | 1.5e-04 | 48 | 1.4e-04 |
| 2.781e+11 | 9.62 | 49 | **9.2e-05** | 8.5e-04 | 28 | 9.2e-05 |
| 7.634e+09 | 351 | 49 | 9.1e-02 | 8.2e-01 | 10 | 7.2e-02 |
| 1.452e+09 | 1.84e+03 | 49 | 1.0e-01 | 6.3e-01 | 19 | 9.8e-02 |
| 1.589e+08 | 1.68e+04 | 49 | 1.2e-01 | 1.2e+00 | 25 | 6.7e-02 |
| 3.023e+07 | 8.85e+04 | 49 | 1.9e-01 | 8.5e-01 | 27 | 1.5e-01 |
| 1.000e+07 | 2.68e+05 | 49 | 2.6e-01 | 7.2e-01 | 24 | 2.2e-01 |

Over all 1813 rows: median 4.8e-04, p25 1.9e-05, p75 0.15, p90 0.58, max 1.97 (the maximum of the
statistic when `total` and `analytic_rad` have opposite signs and similar magnitude). By shape:
q≈r≈k median 2.8e-04, q≈r≫k 2.3e-04, q≪k≈r 2.4e-02.

**Reading.** Where the response redshift is at most a few oscillations inside the horizon
(`x_resp ≲ 10`, the top five rows, 245 items) the agreement is **1.2e-06 – 1.4e-04**, i.e. at or
below the oracle's own phase/modulus spline floor (audit QI-1, 1e-8–1e-6 relative) plus the
board's representation floors. That is the end-to-end closure of A2 and A4: the numeric branch now
reproduces the fixed-w oracle through the hand-over, which the pre-campaign code could not have
done — it read the oscillating source off a cubic spline whose error reaches 100 % of the envelope
beyond ~95 cycles (A2/QS-5, and `QS_03_spline_error` still prints exactly that), and the
`--quad-source-queue` stage that feeds it crashed with an `IndexError` (A3).

No pre-campaign database with `analytic_rad` was available for the "before" number prompt 12 asks
about: the `test-qcd-db*.sqlite` files in the repository root are dated May 2025, and their
`QuadSourceIntegral` table has neither the four prompt-09 columns nor rows (they predate the
schema and the campaign). They were opened read-only and not modified. The pre-campaign figure
therefore stays as the audit's own measurement, quoted above.

Sub-horizon (`x_resp ≳ 350`) the residual rises to a median 0.07–0.26. §5.3 attributes it.

### 5.3 Where the sub-horizon residual comes from

Three candidate mechanisms, separated by measurement:

1. **The oracle's own fixed-w assumption.** `analytic_rad` is a *pure*-radiation solution, while
   the background has ρ_m/ρ_r = 3403/(1+z). The resulting deficit in the accumulated phase is
   ≈ ½ (3403/(1+z)) x_resp radians: 45 rad at z = 1e7, 5 rad at 3.0e7, 0.36 rad at 1.6e8,
   4.3e-3 rad at 1.45e9, 1.6e-4 rad at 7.6e9. So the oracle is decorrelated from any correct
   numeric answer for z ≲ 2e8, and this explains the bottom three rows of §5.2's table — but
   **not** z = 7.6e9 and 1.45e9, where the deficit is negligible and the residual is already 0.1.
2. **The hand-over clamp** (`[08-handover-clamp-error]`). Every row whose integration range
   reaches a transfer function's numeric→LG hand-over carries a clamp gap, because
   `main.py:695-697` truncates the WKB grid to the largest source-grid point at or below `z_init`.
   Measured gaps in `log(1+z)`: `Tq` on 1666 sub-intervals, `Tr` on 1798, median 1.2e-02, max
   2.2e-02 (≈ 1.0 mean grid step, against the allowance of 1.5); the source spline was clamped on
   108 sub-intervals. Splitting the z = 7.6e9 rows by their gap and by regime:

   | rows at z = 7.6e9 | n | median residual |
   |---|---|---|
   | no clamp gap at all (all-smooth integrand) | 4 | **6.2e-05** |
   | any clamp gap | 45 | 6.6e-02 … 4.6e-01 |
   | 0 oscillatory factors | 4 | 6.2e-05 |
   | 2 oscillatory factors | 23 | 7.8e-03 |
   | 3 oscillatory factors | 20 | 2.7e-01 |

   The residual grows monotonically with the number of oscillatory (i.e. clamped, LG-represented)
   factors, and the rows with no clamp gap agree with the oracle to 6e-5. The same ordering holds
   at z = 1.45e9 (2 factors 3.5e-03, 3 factors 1.4e-01). This is `[08-handover-clamp-error]`
   realised in production, **one to two orders of magnitude larger than the 5.1e-3 log 08's
   single-clamped-factor fixture predicted** — unsurprising in that two factors are clamped here,
   the range is far longer, and the offline fixtures never went beyond `x_resp = 980`. Board issue
   `[12-handover-clamp-error-in-production]`.
3. **The transfer function's own LG accuracy at production x.** On real `TkWKBValue` rows the
   stored `T_WKB` differs from the exact `analytic_T_rad` by a median 7.1e-05 – 1.3e-03 of
   envelope per mode (worst 5.8e-02) — one to two orders worse than the 6.1e-06 the offline
   fixture of log 05 gave, because the production phase spline covers x up to 2.7e5 with a few
   hundred samples and its fit error grows linearly with x
   (`docs/lg-phase-and-handover-followup-2026-09.md` §2). Since f is quadratic in T and DT, this
   floor enters `total` roughly doubled. It narrows
   `[05-numeric-region-is-now-the-accuracy-floor]`.

The clamp is the term this pass can attribute cleanly; separating (2) from (3) needs either an
independent high-accuracy quadrature of the same integrand at x ~ 1e5 (not feasible — that is why
Levin exists) or the extrapolation option of log 08 deviation 2. Recorded as such in the board
issue.

### 5.4 The stored error bound, and the tolerances

`total_abserr` (prompt 09, B8) behaves exactly as `[09-abserr-is-a-quadrature-bound]` says, and it
reveals something the offline fixtures could not:

- `total_abserr/|total|`: median 2.2e-04, p75 8.8e-02, p90 **4.3**; **276 of 1813 rows (15 %) have
  a bound larger than the value**, and `|total − analytic_rad| ≤ total_abserr` on 539 of 1813
  (30 %).
- The reason is absolute scale, not the integrator. `total` carries a factor (1+z_response), so
  the quantity the driver's `atol` is compared against is `|total|/(1+z_response)`: median
  **3.6e-26**, p25 9.8e-30, max 7.3e-21. **1044 of 1813 rows (58 %) have a raw integral smaller
  than `DEFAULT_QUADRATURE_ATOL = 1e-25`**, so for them the tolerance is satisfied before any work
  is done and the reported bound is meaningless as a relative statement.
- Confirmed by a convergence study. Tightening only `rtol` (1e-8 → 1e-11) on 159 work items across
  four response redshifts changes `total` **bit-identically nowhere at all** — every one of the 159
  values is unchanged, because `atol` binds. Tightening both (`atol` 1e-25 → 1e-32,
  `rtol` 1e-8 → 1e-10) on 98 items:

  | z_response | median \|Δtotal\|/\|total\| | tightened `total_abserr`/\|total\| | median residual vs oracle, pipeline → tightened |
  |---|---|---|---|
  | 8.409e+11 | 2.4e-08 (max 3.4e-06) | 5.8e-09 | 1.042e-05 → 1.038e-05 |
  | 3.043e+10 | 1.7e-04 (max 1.9) | 2.5e-08 | 1.53e-03 → 7.6e-04 |

  So at z ≈ 8e11 the pipeline tolerances are converged to ~1e-8 and the 1e-5 residual against the
  oracle is *not* quadrature error; at z ≈ 3e10 they are not converged, and tightening halves the
  residual, leaving 7.6e-04 — the representation floor of §5.3.

Board issue `[12-atol-too-loose-for-the-source-integral]`. `total_converged` is `False` on 93 of
1813 rows (5 %) and `total_phase_limited` on none; `WKB_quad` is 0.0 on every row, and no Levin
SVD error occurred anywhere.

### 5.5 The audit's four UNVERIFIED items

**§4.1 — continuity of G at `crossover_z`. CLOSED.** 62 `GkSourcePolicyData` rows came out type
`mixed`; 40 of them were sampled, their `GkSourceFunctions` rebuilt from the datastore, and
`numeric_Gk(crossover_z)` compared with `WKB_Gk(crossover_z)`. All 40 evaluated (no range error,
no missing branch). `|G_num − G_WKB| / max(|G_num|,|G_WKB|)`: min 8.0e-11, median **7.4e-08**,
p90 1.1e-06, **worst 4.0e-06** (k = 2.154e5/Mpc, z_response = 2.763e8, crossover_z = 1.2212e14;
numeric +1.758714e16 against WKB +1.758721e16). There is no step at the crossover: the residual is
the intrinsic WKB error the audit already quotes for the assembled `GkSource` (1e-5–1e-4 at 5–6.6
e-folds sub-horizon), and it is *smaller* here because the crossover sits well inside the overlap.

**§4.2 — reachability of A6. CLOSED, and the answer is yes.** Census of all 462 policy rows:

| type | quality | rows | |
|---|---|---|---|
| `numeric` | `complete` | 270 | 58.44 % |
| `WKB` | `complete` | 123 | 26.62 % |
| `mixed` | `complete` | 61 | 13.20 % |
| `mixed` | **`minimal`** | **1** | **0.22 %** |
| `fail` | `incomplete` | 7 | 1.52 % |

One shipped (k, z_response) falls through all nine earlier `CLASSIFICATION_BANDS` to the
`minimal` band, whose `"WKB_minimal"` test prompt 02 corrected from `numeric_clearance` to
`WKB_clearance`. So A6 was a live defect on 0.2 % of rows at these settings, not a dead branch,
and its `crossover_z`/`Levin_z` would have been chosen by the wrong clearance test. Separately,
**7 rows (1.5 %) are type `fail`, quality `incomplete`** — the policy could not classify them at
all; they are not among the (k, z_response) pairs the source integral reached here, and
`build_partition` would raise on them ("Green's function is smooth … but has no numeric
representation"). Recorded as an observation, not a new issue: `_classify_crossover` is outside
this campaign's remit.

**§4.3 — should a mode with `has_WKB_violation` be rejected? MOOT in this configuration.** The
flag is set on **0 of 7** `TkWKBIntegration` rows and **0 of 5488** `GkWKBIntegration` rows, so no
stored row is affected and no policy decision is forced. Report only, as the prompt requires, and
no comparison "just past the violation point" was possible. What *can* be reported is the quality
of the LG reconstruction on the rows that exist: `|T_WKB − analytic_T_rad|`, normalised by
`max(|analytic|, 1e-3 × envelope)`, per mode —

| k /Mpc | WKB samples | z range | median | worst (at z) |
|---|---|---|---|---|
| 1e+05 | 224 | [1e7, 1.707e9] | 1.31e-03 | 8.7e-03 (1.32e8) |
| 2.154e+05 | 257 | [1e7, 3.651e9] | 1.11e-03 | 1.2e-02 (8.95e8) |
| 4.642e+05 | 291 | [1e7, 7.994e9] | 6.20e-04 | 2.3e-02 (5.28e9) |
| 1e+06 | 324 | [1e7, 1.710e10] | 4.33e-04 | 6.5e-03 (5.16e9) |
| 2.154e+06 | 357 | [1e7, 3.659e10] | 2.66e-04 | 2.5e-02 (7.81e9) |
| 4.642e+06 | 390 | [1e7, 7.830e10] | 1.29e-04 | 7.3e-03 (6.08e10) |
| 1e+07 | 424 | [1e7, 1.714e11] | 7.08e-05 | 5.8e-02 (4.61e10) |

**§4.4 — the A7 end bias on a real GenericEOS run. CLOSED.** `QCD_Cosmology` supplies no analytic
derivative methods, so all five derivative columns of `BackgroundModelValue` come from prompt 03's
padded, refined, quintic spline stack. Compared against high-order central differences in
log(1+z) of the cosmology's own `Hubble`/`wPerturbations` (step h = 1e-4, and h = 3e-4 and 1e-3 to
expose the reference's own error), over 1401 stored redshifts spanning z ∈ [0.1, 1.0168e13]:

| column | low-z end (z = 0.1) | 2nd point | interior median | end/interior | high-z end |
|---|---|---|---|---|---|
| `d_lnH_dz` (= ε/(1+z)) | 7.2e-12 | 7.2e-12 | 6.4e-09 | **0.0011** | 9.8e-08 |
| `d2_lnH_dz2` | 2.5e-07 | 2.5e-07 | 5.6e-06 | **0.044** | 2.6e-05 |
| `d3_lnH_dz3` | 1.3e-03 | 1.3e-03 | 1.1e-03 | **1.14** | 1.1e-02 |
| `d_wPerturbations_dz` | 7.6e-12 | 7.6e-12 | 4.1e-08 | **0.00019** | 9.8e-08 |
| `d2_wPerturbations_dz2` | 3.7e-04 | 3.7e-04 | 4.3e-05 | **8.45** | 1.6e-05 |

Every ratio is inside prompt 03's 10× acceptance criterion, and the two worst columns are limited
by the *reference*, not by the data: with h = 3e-4, where the central differences are most
accurate, the same columns give 4.4e-07 (`d3_lnH_dz3`) and 2.8e-05 (`d2_wPerturbations_dz2`) at
the end, and every end/interior ratio falls to ≤ **0.76**. Against the audit's pre-fix numbers on
the same code path — ε″ 3.0e-01 and w″ 3.8e-01 at the z = 0.1 end — the improvement on a real
GenericEOS run is three to six orders of magnitude. ε itself (= (1+z) d lnH/dz) is 7.2e-12 at the
low-z end, 6.4e-09 in the interior and 9.8e-08 at the high-z end.

**A1 in stored data, as a bonus.** The stored `wPerturbations` column is **bit-identical**
(relative difference exactly 0.0 at every probe) to `w_eos(T) ρ_r / (ρ_m + ρ_r)`, the Λ-free form
prompt 01 installed, and differs from the pre-fix `w_eos(T) ρ_r / ρ_total` form by **1.66** (166 %)
at z = 0.1, 0.151 at z = 1.45 and 1.95e-04 at z = 21.4. The fix is in the data, not just the code.

### 5.6 The regime mix (prompt 12 §2 item 2)

3232 sub-intervals over the 1813 rows, classified against the README's phase-group table:

| oscillatory factors | sub-intervals | share | method | shapes in which it occurred |
|---|---|---|---|---|
| none (ordinary quadrature) | 1211 | 37.47 % | `quad` | all three |
| G only | 791 | 24.47 % | `Levin` | all three |
| T_q only | 0 | — | — | — |
| T_r only | 15 | 0.46 % | `Levin` | q≪k≈r, q≈r≈k |
| G and T_q | 0 | — | — | — |
| G and T_r | 377 | 11.66 % | `Levin` | q≪k≈r, q≈r≈k |
| **T_q and T_r, G smooth** | **93** | **2.88 %** | `Levin` | **q≈r≫k**, q≈r≈k |
| all three | 745 | 23.05 % | `Levin` | all three |

Six of the eight regimes occur. **The "G smooth, both T oscillatory" row is reached, and — as
prompt 12 §2 item 2 asks — it occurs for the q≈r≫k shape**, 93 times. The two empty rows are
`T_q only` and `G and T_q`, and they are *unreachable by construction*, not missing: `main.py`
schedules pairs with `itertools.combinations_with_replacement` over an ascending k array, so
q ≤ r always, and the larger mode always hands over first (log 08 deviation 11 predicted exactly
this). Sub-intervals per row: 914 rows with 1, 416 with 2, 446 with 3, 37 with 4. The Green's
function was type `numeric` on 1015 rows, `mixed` on 196 and `WKB` on 602. Prompt 09's `skipped`
record fired 509 times, always "hand-over within `MIN_SUBINTERVAL_LOG_WIDTH` of the sub-interval
above it" with `log_width = 0.0` — i.e. q = r, where the two hand-overs coincide exactly, which is
the case prompt 09 tested.

`metadata` JSON is 3.3 kB median, **6.1 kB maximum** — well over the
`String(DEFAULT_STRING_LENGTH = 256)` declared for the column
(`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`). SQLite does not enforce `VARCHAR` length,
so the 54 stored smoke rows round-trip intact (§5.9), but log 09 observation 5 asked for this to
be looked at and the answer is that the declared length is exceeded by a factor of 24: any backend
that enforces it would truncate or reject the partition record. Recorded as an observation.

### 5.7 The QuadSource spline residual near the hand-over (prompt 12 §2 item 7)

Real `QuadSource` rows, 12 of the 28 pairs, spline rebuilt exactly as `_create_functions` does
(cubic `make_interp_spline` in log(1+z) through the stored `source` values) and compared with the
exact radiation source recomputed from `compute_analytic_T`/`compute_analytic_Tprime` and
`source_function` on the stored background; the error is normalised by the largest
`|analytic_source_rad|` on the row, and the midpoints probed are the lowest quarter of the region,
i.e. nearest the hand-over:

| q, r /Mpc | stored z | region | at the nodes | at the midpoints |
|---|---|---|---|---|
| 1e5, 1e5 | 560 | [6.879e14, 1.746e9] | 9.6e-06 | 1.4e-05 |
| 1e5, 4.642e5 | 493 | [6.879e14, 8.181e9] | 2.7e-05 | 2.7e-05 |
| 1e5, 2.154e6 | 427 | [6.879e14, 3.745e10] | 8.9e-05 | 9.0e-05 |
| 1e5, 1e7 | 360 | [6.879e14, 1.754e11] | 1.3e-04 | **1.4e-04** |
| 1e6, 1e6 | 460 | [6.879e14, 1.750e10] | 8.3e-05 | 8.2e-05 |
| 2.154e6, 2.154e6 | 427 | [6.879e14, 3.745e10] | 9.2e-05 | 9.0e-05 |

Worst midpoint residual over the sampled pairs **1.4e-04** — *better* than the 4.5e-04 worst case
`[06-source-spline-residual-vs-handover]` records, and the midpoint value barely exceeds the node
value, which says the spline is not the limiting error inside the both-numeric region at this grid
density. Each row stores 360–560 of the grid's 784 redshifts, so prompt 06's truncation to the
both-numeric region is doing what it says on real data (a pre-06 row would have carried all 784).

### 5.8 Cost (prompt 12 §2 item 8)

Per `QuadSourceIntegral` work item, over the 1813 computed rows: `compute_time` median **0.095 s**
(p75 0.26, p90 0.47, max 1.43), of which the analytic oracle is median 0.071 s; total 315 s of
compute plus 146 s of oracle across the sweep, i.e. **12 minutes of wall clock on 10 cores for
1813 items**. Split by the richest regime in the row:

| oscillatory factors | phase groups | rows | median | max |
|---|---|---|---|---|
| 0 | 0 (plain quadrature) | 914 | 0.0038 s | 0.40 s |
| 1 | 1 | 13 | 0.14 s | 0.39 s |
| 2 | 2 | 141 | 0.17 s | 0.65 s |
| 3 | 4 | 745 | 0.28 s | 1.43 s |

Aggregated Levin counters per row: regions median 19 (p90 77, max 165), of which
Clenshaw–Curtis-fallback ("simple") regions median 13 (p90 66, max 156); integrand evaluations
median 43 (p90 169, max 343); `WKB_Levin_elapsed` median 0.092 s. So on production integrands the
driver still routes most regions to Clenshaw–Curtis, as prompt 08 §6 measured, and the
per-region overhead `[10-levin-wholesale-cc-fallback]` describes is real but modest in absolute
terms: 0.28 s for a four-group all-oscillatory row. Prompt 08's offline estimate for the same
shape was 0.1–0.9 s per integral on cheap fixtures, so production cost is at the low end of it —
the "2.65–3.56× slower than the retired direct quadrature" ratio is unchanged by anything here
and remains a performance item for a campaign allowed to edit `AdaptiveLevin/`.

Whole-pipeline cost, for sizing a bigger run: 4.5 minutes for 7 modes × 784 redshifts, of which
the two Green's-function stages are 85 %. `GkNumericIntegration` count scales as
(number of modes) × (decades from the grid top down to `z_exit_subh_e4`), which is why it produced
3436 items here.

### 5.9 Schema, persistence and resume

The prompt-09 schema change was exercised live for the first time. On the smoke datastore, all
**54 stored `QuadSourceIntegral` rows have non-null `b`, `total_abserr` and `total_converged`**,
and re-running the whole pipeline against that datastore found every object already present —
`0 lookup, 0 compute, 0 store` in every stage including `CALCULATE QUADRATIC SOURCE INTEGRALS`,
with row counts unchanged. So `register()`, `store()`, `build()` and `read_batch()` agree on the
four new columns, `available` is `True` on reload, and a stop/resume finds the work rather than
recomputing it.

`WKB_quad` is `0.0` on all 54 stored rows and all 1813 computed ones, confirming
`[09-WKB_quad-columns-are-vestigial]`.

---

## 6. What remains unverified, and what it would take

1. **The matter and Λ eras of the source integral.** Run A stops at z = 1e7 to keep
   `analytic_rad` valid (§4.1). Everything below matter–radiation equality — where `wPerturbations`
   departs from 1/3, where the audit's A7 measurement was originally taken, and where the
   response-redshift averaging question of the reconciliation document lives — was not exercised
   for the source integral. Cost of closing it: the same run with `--zend 0.1` is about 4 more
   decades of redshift, so ~1400 rather than 784 grid points, ~1.6× the Green's-function work
   (~10 minutes), plus a *second* oracle, because `analytic_rad` is not a valid reference there
   and the discrepancy would have to be reported as expected rather than measured against
   anything.
2. **A GenericEOS/QCD run of the full chain.** Run B is background-only. `QCD_Cosmology` differs
   from `LambdaCDM` in `wPerturbations`, `epsilon` and all five derivative columns, so it exercises
   prompts 01 and 03 (done, §5.5) but not their propagation into `TkWKBIntegration`'s
   ω_eff and LG friction, into `GkSource`, or into the source integral. Cost: run A's command with
   `--models QCD_Cosmology` — the same 4.5 minutes plus a slower background
   (121 ms against 14 ms per `compute_background`, log 03) and a slower `Hubble` on every
   integrand evaluation, so perhaps 2–3× run A. Not attempted here because the
   `QuadSourceIntegral` stage cannot complete until `[12-region-check-absolute-tolerance]` is
   fixed, and a second run would produce the same 43 % failure.
3. **A radiation-EOS `LambdaCDM_GenericEOS`**, which is what audit §4.4's "the radiation EOS makes
   them identical models" wording assumes. There is no such registered store class — only
   `LambdaCDM` and `QCD_Cosmology` have Datastore factories — so §5.5 uses an independent
   finite-difference reference on the QCD model instead. Closing it as the audit imagined would
   need a new factory, i.e. production code.
4. **The full 50+50 wavenumber grid.** 63 750 (k,q,r,z) triples before the triangle filter and
   5133 after, against 3185/49 here, on a redshift grid roughly twice as long (`z_exit(3e8)` is
   30× larger than `z_exit(1e7)`). Extrapolating §5.8: the Green's-function stages scale as
   modes × decades, so ~50/7 × ~1.3 ≈ 9× run A's 3.5 minutes for those, and the
   `QuadSourceIntegral` stage as triples × response redshifts ≈ 5133/49 × (130/66) ≈ 200× run A's
   12 minutes of harness time — order 1–2 days on 10 cores, and ~10–20 GB of datastore. That is a
   cluster job, not a verification pass.
5. **`LiouvilleGreen/tests`.** Launched here for information as the live runs were finishing and
   left unattended; it was still executing five hours later, having produced no output past its
   second test, and was **terminated during session clean-up** (exit 143, SIGTERM), so it produced
   no pass/fail result and none is claimed. Nothing here measured why it did not finish. The
   package is out of this campaign's scope (README §5 item 8), no campaign commit touched it, and
   the four suites the prompt does name are green (§3.1). Cost of closing it: run the suite alone
   on an idle machine — its own logs put it at "many minutes" (log 03's verification section).
6. **The 7 `fail`-quality Green's-function policies** (§5.5). They were not among the
   (k, z_response) pairs the source integral reached, so what `build_partition` does with one is
   still untested against real data. A targeted harness run over exactly those pairs would settle
   it in seconds — once the region guard is fixed.
7. **Separating the hand-over clamp error from the LG phase error** sub-horizon (§5.3). Needs
   either the first-order Taylor extension of log 08 deviation 2 (then re-run and compare) or an
   independent high-accuracy quadrature at x ~ 1e5, which is not feasible.
8. **`extract_*.py` against a populated datastore.** Not attempted: the scripts are out of scope
   (README §5 item 8), and `[09-WKB_quad-columns-are-vestigial]` already predicts what
   `extract_QuadSourceIntegral_data.py` will draw.

---

## 7. Deviations across the campaign

Collected from all twelve logs, so that a reader can judge the campaign without opening them.
"Class" is the log's own classification.

| Item | Prompt | What differed from the prompt | Class | One-line reason |
|---|---|---|---|---|
| Agreement tolerance 1e-8, not 1e-10 | 01 | regression test asserts 1e-8 | structurally required | the 500-point `T(z)` spline floors any GenericEOS comparison at ~1.3e-9 |
| Author's parenthetical kept | 01 | comment left verbatim | implementation choice | it is an author's open question about the convention, not a description of the code |
| Two extra test methods | 01 | 5 methods, not 3 | implementation choice | guards the premise (Ω_r match) and the converse (`wBackground` must change) |
| Commit SHA not embedded | 01, 03, 05–11 | log/board identify the commit by subject | implementation choice | a log inside its own commit cannot carry that commit's SHA; the backport campaign set the precedent |
| B9 left as-is | 02 | no code change, cross-reference comment only | implementation choice | the two θ-spline constructions solve different problems; judged deliberate — **a judgement, not a measurement**, and still unmeasured |
| Remedy is (a)+(b)+(c) | 03 | padding *and* quintic *and* refinement | implementation choice | neither padding nor a higher order alone meets the 10× criterion |
| `d_wPerturbations_dz` from the callable | 03 | `f_to_diff=` instead of `sample_to_diff=` | structurally required | its "previous level" is a production-grid sample array that cannot be padded |
| 10× criterion at the low-z end only | 03 | high-z end asserted in absolute error | structurally required | ε′, ε″, w′, w″ → 0 in radiation, so a relative error there is meaningless (the audit's own footnote) |
| Extra absolute-threshold test | 03 | second test case | implementation choice | the surviving residual is float64 round-off, so the ratio statistic is not stable |
| A class, not a namedtuple | 05 | `TkSourceFunctions` is a class | implementation choice | the prompt's own constructor signature is impossible for a namedtuple |
| `increasing=True` | 05 | opposite flag to `GkSourcePolicyData` | structurally required | θ_q increases with z where θ_G decreases; the flag is a sort order, not a formula |
| Region ends are the sampled ranges | 05 | `WKB_region[0] ≤ crossover_z ≤ numeric_region[1]` | structurally required | `main.py` stores no WKB sample at `z_init` and `phase_spline` cannot extrapolate — **this is the origin of the clamp error §5.3 measures** |
| `k` duck-typed, `z_exit` resolved | 05 | signature accepts either class | structurally required | `wavenumber` has no `z_exit` |
| Two WKB fixtures | 05 | "exact" and "LG" | structurally required | the exact envelope is not the LG amplitude; one fixture cannot carry both assertion sets |
| `sin_coeff` from the exact amplitude | 05 | not from the T,T′ matching | implementation choice | the matching belongs to `TkWKBIntegration` and is already verified exactly |
| Test thresholds relaxed | 05 | 1e-4/1e-3 at midpoints, 1e-5 on the production grid | structurally required | below the fit error of the grid the prompt itself prescribes |
| Extra `friction()` accessor | 05 | one extra public method | implementation choice | it is the only splined ingredient of the amplitude |
| `.z_sample` not consumed | 05 | reads `.values` only | implementation choice | in `mode="stop"` `z_sample` is not a description of what was sampled |
| `RuntimeError`, not `assert` | 05 | raised errors | implementation choice | the codebase's convention, and survives `python -O` |
| Region floor clamped to coverage | 06 | `max(crossover_q, crossover_r, Tq_z_min, Tr_z_min)` | structurally required | the hand-over can fall below a factor's last stored sample; without the clamp a benign case becomes run-blocking |
| `store()` truncates `z_sample` | 06 | task returns `z_store_ids` | implementation choice | otherwise `z_sample`, `values` and the spline range disagree, before and after a round trip |
| `numeric_region` derived, crossovers `None` | 06 | as the prompt anticipated | implementation choice | the crossovers are not persisted; the region is recoverable from the value rows |
| `numeric_crossover_z` a module function | 06 | not a method | implementation choice | needed by the Ray task, the tests and prompt 08 |
| Ten tests, not four | 06 | six extra | implementation choice | covers the deviations above, plus an A3 regression that replays the old loop |
| Oracle thresholds on exact stand-ins | 07 | 1e-10/1e-8 asserted with scipy-exact `m`, `ϑ` | structurally required | `bessel_phase` is itself accurate only to ~x·1e-8, and the LG closed forms carry O(x⁻⁴) truncation |
| Boundary threshold 5e-4/2e-3 | 07 | not 1e-6 | structurally required | the prompt's 1e-6 was never the right number for the `f` spline (4.5e-4) |
| Phase-composition assertion on exact remainders | 07 | `phase_spline` variant kept as a measurement | structurally required | `chunk_logstep=125` is geometric, so chunking stops protecting precision above ~7e4 rad |
| Extra public helpers | 07 | `levin_theta`, `evaluate_envelope`, the algebra functions | implementation choice | so the sympy script verifies the module's own code path rather than a re-transcription |
| Ingredient memoisation | 07 | `lru_cache(16384)` | implementation choice | the driver samples every group at the same nodes; without it a node costs 2×groups evaluations |
| Smooth `G` accepts either shape | 07 | callable or `.numeric_Gk` | implementation choice | the prompt's phrasing admits both |
| `G`-only integrand equal to rounding | 07 | 2.5e-16, not bitwise | implementation choice | one algebra path, verified by sympy, beats a second code path |
| Realistic oracle at the representation floor | 08 | parts asserted separately, not `total` at 1e-5 | structurally required | the floors are pointwise and the parts cancel by up to 5× |
| **Hand-over gap bridged by clamping** | 08 | `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5`, gap recorded | implementation choice (board-directed) | a strict assertion would block every production integral — **and §5.3 shows the clamp is now the dominant error** |
| Task body a plain function with a builder hook | 08 | `evaluate_QuadSource_integral` | implementation choice | an explicit seam beats patching a module attribute in tests |
| Two old integrators deleted | 08 | `WKB_quad_integral`, `WKB_Levin_integral` gone | implementation choice | dead code implementing the rejected scheme; costs `QI_03` sections 2–4 (§3.2) |
| Region guard in log(1+z) | 08 | `MIN_SUBINTERVAL_LOG_WIDTH` | structurally required | B11's fix in substance, since the old sites no longer exist |
| `metadata["WKB_Levin"]` always a dict | 08 | plus two new metadata keys | implementation choice | a fixed shape is simpler for prompt 09's consumer |
| `TkSourceFunctions` receives `q.k` | 08 | not the exit-time object | structurally required | `wavenumber_exit_time` has no `__float__` |
| `atol` distributed by log-width | 08 | matches the Levin driver's own scheme | implementation choice | keeps the partition transparent to the driver's contract — **and §5.4 shows `atol` is the binding tolerance in production** |
| `theta_deriv` passed | 08 | `LEVIN_USE_THETA_DERIV = True` | implementation choice | neutral on value, identical cost, up to 1e3× lower reported error floor |
| **§6 cost ratio above the 3× trigger** | 08 | 2.65–3.56× measured, not gated | measurement → user decision | resolved by the user in prompt 10: fix the driver's fallback choice instead |
| Test-design differences | 08 | exact seam, offset G phase, one-abscissa seam check | mixed | listed in log 08 deviation 11 |
| Small robustness edits | 08 | `get_z`, `getattr`, `_Gk_diagnostics` | implementation choice | so duck-typed stand-ins can drive the function |
| `b` guard numerical, in the task body | 09 | not an equality test in `compute()` | structurally required | `bessel_phase()` records no order, and in `compute()` the splines are still proxies |
| `WKB_quad` columns kept | 09 | not dropped | implementation choice | `extract_QuadSourceIntegral_data.py` reads them and is out of scope |
| `total_abserr` is a quadrature bound | 09 | test asserts what the number means | structurally required | the fixture floor is above `abserr` in every case — **§5.4 confirms this in production and finds `atol` too loose** |
| Naming `total_abserr` etc. | 09 | not a bare `abserr` | implementation choice | the dict already carries two other `abserr`s |
| Bottom-snap branch unreachable | 09 | test asserts the reachable case | unintended drift, kept | removing a defensive branch nobody asked about |
| Minimal region-guard stand-ins | 09 | new `_MinimalGk`/`_MinimalTk` | implementation choice | the prompt-08 `Case` machinery cannot reach z_response = 0 |
| Three files beyond the header's list | 10 | `QuadSourceIntegral.py`, `GkSourcePolicyData.py` comments | structurally required | the prompt's own §3 and §4 direct those edits |
| Dry test in `ComputeTargets/tests` | 10 | not a new root `tests/` | implementation choice | one discover root per package, and the contract under test is this package's |
| `ast` extraction, call site untested | 10 | function-level test only | implementation choice | `main.py` cannot be imported; the stage's call site is covered only by this live run |
| `black` reformatted prompt 04's docstring | 10 | two extra diff lines | implementation choice | so `black --check main.py` passes |
| Payload size measured per value | 10 | not a whole serialised payload | implementation choice | a real payload needs a populated datastore — which now exists (§4.1) |
| A2/A4 map to several commits | 11 | multi-commit rows in §8 | implementation choice | no single commit is "the" fix for either |
| B9 annotated in §8 | 11 | "evaluated, left as-is" | implementation choice | matches the board rather than the seven genuine fixes |
| **`zend = 1e7` in run A** | 12 | not the production 0.1 | implementation choice | keeps the run inside radiation domination, where `analytic_rad` is a valid oracle; §6 item 1 states the cost |
| **QSI work list scheduled by a harness** | 12 | not by `main.py`'s stage | structurally required | the stage aborts on the first of 1372 region-guard failures (§5.1); the compute path is the production one |
| GenericEOS run is background-only | 12 | not the full chain | structurally required | the source integral cannot complete until §5.1 is fixed |
| A7 reference is finite differences | 12 | not `LambdaCDM` analytic values | structurally required | no radiation-EOS `LambdaCDM_GenericEOS` is a registered store class |
| `rtol = 1e-11` sweep stopped early | 12 | 159 of 196 items | implementation choice | every one of the 159 was bit-identical to the pipeline value, so the answer was already unambiguous |

No deviation in any prompt changed a formula, a sign, a normalisation or the phase-group table.
Two were measurements that the orchestrator's stop rules referred to the user (prompt 08's cost
ratio and its clamp error); both were answered by the user and are recorded in the board's §4.

---

## 8. Summary for the status board

- **Layer 1: all green.** 118 unit tests in the three packages the campaign owns, plus the sympy
  phase-group script; 28 of 29 audit scripts exit 0, and the six output blocks a reader could
  misinterpret are enumerated in §3.2 with their post-fix counterparts in §5.6/§5.7.
- **Layer 2: one live run, and it found a run-blocker.** Every stage up to
  `GkSourcePolicyData` completed on a real Ray/`ShardedPool` run; 1813 real
  `QuadSourceIntegral` items were computed through the production task; and the
  `--quad-source-integral-queue` stage itself **cannot complete** because
  `_check_region_covers` compares redshifts with an absolute 1e-7 tolerance
  (`[12-region-check-absolute-tolerance]`, 43 % of work items).
- **Audit §4.1 closed** (7.4e-08 median, 4.0e-06 worst — no step at `crossover_z`).
  **§4.2 closed, affirmative** (1 of 462 rows reaches the `minimal` band, so A6 was live).
  **§4.3 closed as unreachable here** (0 of 7 and 0 of 5488 rows carry `has_WKB_violation`; report
  only, no policy change). **§4.4 closed** (every end/interior ratio inside prompt 03's 10×
  criterion, ≤ 0.76 with the most accurate reference, against 3.0e-01/3.8e-01 before the fix).
- **A2 and A4 close end to end where the oracle is valid**: `total` reproduces `analytic_rad` to
  1.2e-06 – 1.4e-04 for `x_resp ≲ 10` and to 6.2e-05 on rows with no hand-over inside the range.
- **Two new issues for the user**: `[12-atol-too-loose-for-the-source-integral]` (58 % of items
  have a raw integral below `DEFAULT_QUADRATURE_ATOL`) and
  `[12-handover-clamp-error-in-production]` (the clamp of `[08-handover-clamp-error]` costs a
  median 0.07–0.26 relative sub-horizon, one to two orders above the fixture prediction).
- **The prompt-09 schema round-trips** and a stop/resume finds every stored object, including
  `QuadSourceIntegral` rows with the four new columns.

The campaign's deliverables are verified; the source integral is **not yet fit for a production
sweep**, for the two reasons above, both of which are recorded on the board with reproductions.
