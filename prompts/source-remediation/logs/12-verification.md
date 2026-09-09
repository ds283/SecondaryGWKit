# Log 12 — Verification pass (audit §4; whole campaign)

**Prompt:** prompts/source-remediation/12-verification.md
**Commit:** *(this commit)* — "Verify the source-remediation campaign against a live scoped run"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Opus 5
**Date:** 2026-09-09
**Result:** COMPLETE WITH DEVIATIONS

Three things a reader should not miss. **(1) The `--quad-source-integral-queue` stage cannot
complete at production redshifts**: `build_partition`'s region guard compares redshifts with an
absolute tolerance of 1e-7, which is meaningless above z ≈ 1e8, and 1372 of 3185 (43 %) work items
abort in it. Per the prompt's own instruction this is reported, not fixed
(`[12-region-check-absolute-tolerance]`). **(2) All four audit §4 items are closed**, and §4.2
comes back *affirmative* — one of 462 policy rows reaches the `minimal` band, so A6 was a live
defect. **(3) `total` reproduces `analytic_rad` to 1.2e-06 – 1.4e-04 where the response redshift
is at most a few oscillations inside the horizon**, which is the end-to-end closure of A2 and A4;
sub-horizon the residual rises to a median 0.07–0.26 and tracks the hand-over clamp of
`[08-handover-clamp-error]`, an order or two above what its fixture predicted.

## What shipped

**New `docs/source-remediation-verification.md`** (the record the prompt asks for, in the format of
`docs/backport-modules-verification.md`): §2 reconciles the git history, §3 tabulates the whole
offline test tree and all 29 audit scripts (script → expectation → observed), §4 states the exact
commands, environment and datastore path of every live run, §5 gives the nine measurement blocks
(§5.1 the new defect, §5.2 `total` vs `analytic_rad`, §5.3 its attribution, §5.4 the error bound
and tolerances, §5.5 audit §4.1–§4.4, §5.6 the regime mix, §5.7 the QuadSource spline residual,
§5.8 cost, §5.9 schema and resume), §6 lists what remains unverified with a cost estimate, §7
collects every deviation from every one of the twelve logs into one table, and §8 is the board
summary.

**New `docs/source-remediation-verification/` (four scripts, all read-only w.r.t. the datastore):**

- `scoped_pipeline_run.py` — runs `main.py`'s own pipeline on a scoped wavenumber sample. It does
  not reimplement any stage: it patches `ray.init` to bootstrap a local instance, optionally
  filters `build_model_list`, textually substitutes the two hardcoded 50-point `np.logspace` k
  grids for a name in the exec globals, and `exec`s `main.py`'s source. Refuses to start on an
  existing datastore path unless `--allow-existing` (used only for the resume test).
- `run_quadsource_integrals.py` — schedules the production `QuadSourceIntegral` work list itself,
  calling the real `@ray.remote compute_QuadSource_integral` with the same nine-key payload
  `main.py:build_QuadSourceIntegral_payload` assembles, and recording failures as JSON lines
  instead of aborting. `--atol`/`--rtol` for the convergence study; stores nothing.
- `analyse_quadsource_integral.py` — `total` vs `analytic_rad`, the regime mix against the
  README §6 table, clamp gaps, `skipped` records, convergence flags and cost. Reads either the
  stored table or the JSON lines.
- `analyse_greens_and_source.py` — audit §4.1 (G continuity at `crossover_z`, through a real
  `ShardedPool`), §4.2 (the policy census), §4.3 (`has_WKB_violation`, and `T_WKB` against
  `analytic_T_rad` on real rows), and prompt 06 §3.4 (the source spline residual near the
  hand-over on real rows).
- `analyse_background.py` — audit §4.4 (the A7 end bias on a real QCD/GenericEOS background,
  against high-order central differences) and A1 as it appears in the stored `wPerturbations`
  column.

**Board:** row 12 set to ⚠️; items §4.1–§4.4 closed with pointers into the document; **Progress**
line updated; `[08-pipeline-non-runnable-until-10]`-style resolved entries left in §4 and three
issues narrowed in place; three new §3 issues opened (`[12-region-check-absolute-tolerance]`,
`[12-atol-too-loose-for-the-source-integral]`, `[12-handover-clamp-error-in-production]`) and a
campaign-completion note added.

**No production code was changed.** `git diff HEAD~1 --stat` touches only
`docs/source-remediation-verification.md`, `docs/source-remediation-verification/*.py`,
`prompts/source-remediation/logs/12-verification.md` and
`prompts/source-remediation/IMPLEMENTATION_STATE.md`.

## Deviations from the prompt

### 1. The QuadSourceIntegral work list was scheduled by a harness, not by `main.py`'s stage — STRUCTURALLY REQUIRED

§1 asks for "all queues enabled in pipeline order … → QuadSourceIntegral". They were, and the
stage aborted on its first batch with

```
RuntimeError: compute_QuadSource_integral: sub-interval bottom z=4.8682e+14 is out-of-bounds
for the Gk numeric region (6.8788e+14, 4.8682e+14) [domain=5.4628e+14, 4.8682e+14]
```

— the sub-interval bottom and the region bottom are the same redshift, and `_check_region_covers`
compares them with an absolute `DEFAULT_FLOAT_PRECISION = 1e-7` while `build_partition` recovers
`z = exp(log(1+z)) − 1`, whose round-trip error at z ≈ 5e14 is ~0.06. `RayWorkPool` propagates the
first failure, so the stage produced no rows at all.

Rather than lose the whole of §2 items 1, 2 and 8, the same work list is scheduled by
`run_quadsource_integrals.py`, which calls the production `@ray.remote` task with the payload
`main.py` builds and records failures instead of raising. **The compute path is the production
one**; only the scheduling is the harness's, and every failure is reported (1372 of 3185, all in
the same guard: 875 on the `Gk numeric` region and 497 on the `Gk WKB` region, over 28 distinct
response redshifts). The alternative — declaring Layer 2 blocked — would have left every §2
measurement unmade for a defect that affects 43 % of items rather than all of them.

The defect is *not fixed*, per the prompt's "open a §3 issue with the reproduction and stop".

### 2. `zend = 1e7` in the main run, not the production 0.1 — IMPLEMENTATION CHOICE

§1 allows "a shorter range if needed to keep the run under ~1–2 hours"; the reason here is
accuracy, not time. `analytic_rad` is a *pure*-radiation oracle, and the background carries
ρ_m/ρ_r = 3403/(1+z), which enters the transfer function's oscillation phase as a deficit of
≈ ½(3403/(1+z))·x radians. At z = 1e4 that is a 34 % error in the density ratio; at z = 1e7 it is
3.4e-4, and the resulting phase deficit is below 1 radian for every row above z ≈ 2e8. Keeping the
whole run inside radiation domination is therefore what makes §2 item 1's comparison mean
anything, and it is what the prompt itself asks for ("restrict the comparison to `z_response` deep
in radiation domination"). It also shortens the grid from ~1580 to 784 redshifts.

What it costs is stated in the document's §6 item 1: the matter and Λ eras of the source integral
are unexercised, and audit §4.4's low-z end is measured on run B (which does reach z = 0.1)
rather than on the main run. Alternatives considered: `zend = 0.1` with the matter-era discrepancy
"reported separately as expected" (the prompt's other option) — rejected because it would have
produced a table of numbers against an oracle known to be invalid over half its range, at 1.6× the
cost; and `zend = 1e4` — rejected because at that redshift matter is 34 % of the density and the
oracle is already useless.

### 3. The GenericEOS run is background-only — STRUCTURALLY REQUIRED

§1 asks for a `LambdaCDM_GenericEOS` run "if time allows". Two obstacles. First, the only
GenericEOS class with a Datastore factory is `QCD_Cosmology` (`config/sharding.py` registers
`LambdaCDM` and `QCD_Cosmology` and nothing else), so the "radiation EOS makes them identical
models" comparison audit §4.4 describes is not available without adding a production factory.
Second, a full chain run would abort in the same region guard as deviation 1, so it could not
produce a single source integral.

What is needed for audit §4.4 is a `BackgroundModel` on the production redshift range, and that is
computed before any queue. Run B therefore uses `QCD_Cosmology` with every queue disabled and
`--zend 0.1`, giving 1401 stored redshifts over z ∈ [0.1, 1.0168e13] — the same shape as the grid
the audit's TK-5 table used — in 1.08 s. The A7 reference is high-order central differences of the
cosmology's own `Hubble`/`wPerturbations` (deviation 4), and the same run also confirms A1 in
stored data.

### 4. The A7 reference is finite differences, not `LambdaCDM`'s analytic derivatives — STRUCTURALLY REQUIRED

§2 item 6 says to "compare … with the analytic `LambdaCDM` values at the same z (the radiation EOS
makes them identical models)". `QCD_Cosmology` is not a radiation EOS — its `w(T)` tracks
`4g_S/(3g) − 1` through the QCD transition and e⁺e⁻ annihilation — so `LambdaCDM` is not an oracle
for it at any redshift, and no radiation-EOS GenericEOS class is persistable (deviation 3).

Shipped instead: central differences in log(1+z) of the cosmology's own `Hubble` and
`wPerturbations` (4th-order stencils, converted to d/dz, d²/dz², d³/dz³ by the chain rule), which
is an independent implementation of the same derivative the spline stack estimates. Because the
reference has its own error, the measurement is repeated at three step sizes and both the
end-point errors and their sensitivity to the step are reported: at h = 1e-4 the worst end/interior
ratio is 8.45 (`d2_wPerturbations_dz2`), at h = 3e-4 — where the reference is most accurate — every
ratio is ≤ 0.76. Prompt 03's acceptance criterion is 10×, so the conclusion is the same either
way, and the *absolute* improvement (3.7e-04 against the audit's 3.8e-01 for w″) does not depend on
the reference's accuracy at all.

### 5. The `rtol = 1e-11` sweep was stopped after 159 of 196 items — IMPLEMENTATION CHOICE

Every one of the 159 completed items returned a **bit-identical** `total` to the pipeline
tolerances, because `atol` binds rather than `rtol` (§5.4 of the document). Continuing would have
consumed the machine for another 20 minutes to add 37 more bit-identical values. The question it
was asked to settle was then re-put as a joint `atol`+`rtol` study (1e-32 / 1e-10) on 98 items,
which is the measurement quoted.

### 6. The prompt's suggested "pre-campaign figure" could not be obtained — STRUCTURALLY REQUIRED

§2 item 1 says to quote a pre-campaign `analytic_rad` "if any pre-campaign database … is available
(`test-qcd-db*.sqlite` in the repository root may predate the schema change … do not spend long on
it)". They are dated May 2025 and their `QuadSourceIntegral` table has neither the prompt-09
columns nor any rows. They were opened read-only, nothing was written to them, and the
pre-campaign figure quoted in the document is the audit's own A2 measurement instead. Two minutes
were spent on this.

### 7. An `--allow-existing` flag was added to the driver after the fact — IMPLEMENTATION CHOICE

The driver's guard against touching an existing datastore (added because this prompt's own
instructions insist on it) also blocks the stop/resume test, which has to reopen a datastore it
just built. The flag permits that one case; it does not weaken anything, because the pipeline only
adds missing work and never migrates or rewrites a row, and the default is still to refuse.

### 8. `LiouvilleGreen/tests` was started but not waited on — IMPLEMENTATION CHOICE

§1 says "the full unit-test tree (`CosmologyModels/tests`, `ComputeTargets/tests`,
`AdaptiveLevin/tests`, plus any test module prompt 10 added)". Those four are green. The
`LiouvilleGreen` suite is not in the prompt's list, takes many minutes, and belongs to a package
README §5 item 8 places out of scope and no campaign commit touched. It was launched for
information as the live runs were finishing and left unattended; it was still executing five hours
later, having produced no output past its second test, and was **terminated during session
clean-up** (exit 143, SIGTERM), so it produced no result and the document claims none — §3.1 and
§6 item 5 say exactly that rather than implying a pass.

## Verification performed

All commands from the repository root with `PYTHONPATH=.` and `./venv/bin/python`. The document
carries the full tables; this section says what was run and quotes the headline numbers.

**Ran, passed.** `unittest discover` on `CosmologyModels/tests` (**5 tests, 0.017 s, OK**),
`AdaptiveLevin/tests` (**32 tests, 0.056 s, OK**), `ComputeTargets/tests` (**81 tests, 235.4 s,
OK** — includes prompt 10's 7-test `test_main_plumbing`), and
`ComputeTargets/tests/sympy_phase_groups.py` directly (**"all residuals are identically zero"**,
exit 0).

**Ran; 28 of 29 exit 0.** Every script in `docs/spec-code-audit/scripts/`. `QI_03_measure` exits 1
after its section 1 agrees to **4.267e-16**, because sections 2–4 call the two functions prompt 08
deleted — expected, and recorded in log 08 observation 2. Five further scripts still print
*pre-fix* numbers because they re-implement the defective code path inline rather than calling the
production one (`QS_03`, `QS_04` part (a), `TK_05` section (d), `TK_06`, `TK_07`); the document
§3.2 names each and points at the post-fix measurement that replaces it.

**Ran, live, on a locally bootstrapped Ray instance** (`ray.init(num_cpus=10,
include_dashboard=False)`; no cluster was running, and Ray warned every 10 s that the volume is
over 95 % full — 24–26 GB free — without ever needing to spill):

- **Run A**, `LambdaCDM` Planck2018, 7 log-spaced modes over 1e5 … 1e7 /Mpc as both source and
  response sample, 100 redshifts per log10(1+z), response sparseness 12, `--zend 1e7`, 4 shards.
  Fresh datastore at
  `/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/c81575f0-880e-4187-b2ed-e78031b1eb8c/scratchpad/runA/verify-LambdaCDM.sqlite`
  (a path that did not exist; nothing was overwritten, migrated or deleted). 784 source and 66
  response redshifts, grid 6.8788e14 → 1e7; largest Bessel x = 4.6345e5. Triangle filter kept
  **3185 of 12740** work items (25.00 %) from 49 of 196 triples. Every stage completed except
  `QuadSourceIntegral`: 3436 `GkNumericIntegration` in 1 m 55 s, 5488 `GkWKBIntegration` in
  1 m 37 s, 462 `GkSource` in 25.6 s, 462 `GkSourcePolicyData` in 6.1 s. **Total wall clock 4.5
  minutes; datastore 66 MB** (4 shards of 9–19 MB plus a 32 kB primary).
- **Run B**, `QCD_Cosmology`, background only, `--zend 0.1`, 100 per log10(1+z): 1401 redshifts
  over z ∈ [0.1, 1.0168e13], background computed in 1.08 s, datastore 2.1 MB.
- **Smoke run**, 3 modes, 10 per log10(1+z), `--zend 1e6`: the only run whose
  `--quad-source-integral-queue` stage completed (its seven response redshifts happen not to trip
  the guard), producing and **storing 54 `QuadSourceIntegral` rows**; 3.8 MB.
- **QuadSourceIntegral harness** over run A's own 3185 work items: **1813 completed, 1372 failed**,
  12 minutes.

**The new defect, reproduced without a pipeline.** `exp(log(1+z)) − 1 < z − 1e-7` holds for
**29 of run A's 66 response redshifts (44 %)** and 331 of its 784 source redshifts (42 %), the
smallest tripping value being z = 6.93e07; on the smoke run's seven response redshifts it holds for
none, which is why that stage completed. Scheduled item by item: **1372 of 3185 (43 %)** raise, 875
on the `Gk numeric` region and 497 on the `Gk WKB` region, over 28 distinct response redshifts from
6.93e07 to 4.87e14, and no other failure mode occurs.

**§2 item 1 — `total` vs `analytic_rad`, 1813 rows.** Normalised by
`max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)`: median 4.8e-04 overall, and by response
redshift — 7.6e-05 at z = 3.69e14 (x_resp 0.007), **2.0e-06** at 7.02e13, **1.2e-06** at 2.32e13,
1.4e-04 at 1.46e12, 9.2e-05 at 2.78e11 (x_resp 9.6), then 9.1e-02 at 7.63e09 (x_resp 351), 1.0e-01
at 1.45e09, 1.2e-01 at 1.59e08, 1.9e-01 at 3.02e07 and 2.6e-01 at 1e07 (x_resp 2.7e5).

**§2 item 1 — attribution of the sub-horizon residual.** Three mechanisms, separated by
measurement. (a) The oracle's own fixed-w phase deficit ≈ ½(3403/(1+z))·x_resp is 45 rad at
z = 1e7 and 0.36 rad at 1.6e8, so the oracle is decorrelated below z ≈ 2e8 — but it is 1.6e-4 rad
at z = 7.6e9, where the residual is already 0.1. (b) The hand-over clamp: at z = 7.6e9 the four
rows with **no** clamp gap agree to **6.2e-05**, while rows with a gap give 6.6e-02 – 4.6e-01, and
by regime 0/2/3 oscillatory factors give 6.2e-05 / 7.8e-03 / **2.7e-01**. Measured gaps:
`Tq` on 1666 sub-intervals and `Tr` on 1798, median 1.2e-02, max 2.2e-02 in log(1+z) (≈ 1.0 mean
grid step against the 1.5 allowance); the source spline was clamped on 108. (c) The LG accuracy of
the stored transfer function at production x: `T_WKB` vs `analytic_T_rad` on real `TkWKBValue`
rows is a median 7.1e-05 – 1.3e-03 of envelope per mode, worst 5.8e-02 — one to two orders worse
than the 6.1e-06 log 05's fixture gave.

**§2 item 1 — the error bound and the tolerances.** `total_abserr/|total|` median 2.2e-04,
p90 **4.3**; 276 of 1813 rows (15 %) have a bound larger than the value, and
`|total − analytic_rad| ≤ total_abserr` on 539 (30 %). The cause is absolute scale:
`|total|/(1+z_response)` — the quantity `atol` sees — has median **3.6e-26**, and **1044 of 1813
(58 %) are below `DEFAULT_QUADRATURE_ATOL = 1e-25`**. Confirmed by convergence study: tightening
`rtol` alone from 1e-8 to 1e-11 leaves **all 159** completed values bit-identical, while tightening
`atol` to 1e-32 with `rtol = 1e-10` moves `total` by a median 2.4e-08 at z = 8.4e11 (converged) but
1.7e-04 at z = 3.0e10 (not converged), where it also halves the residual against the oracle from
1.53e-03 to 7.6e-04. `total_converged` is `False` on 93 of 1813 (5 %), `total_phase_limited` on
none, `WKB_quad` is 0.0 on every row and no Levin SVD error occurred.

**§2 item 2 — regime mix.** 3232 sub-intervals: none 1211 (37.5 %), G only 791 (24.5 %), T_r only
15, G and T_r 377, **T_q and T_r with G smooth 93 (2.9 %), and it occurs for the q≈r≫k shape as
the prompt asks**, all three 745 (23.1 %). `T_q only` and `G and T_q` are empty and *unreachable by
construction* (`combinations_with_replacement` gives q ≤ r, so T_r always hands over first — log 08
deviation 11 predicted this). G type: numeric 1015, mixed 196, WKB 602. Prompt 09's `skipped`
record fired 509 times, always at `log_width = 0.0`, i.e. q = r.

**§2 item 3 — audit §4.1, ran, passed.** 62 `mixed` policies; 40 sampled, all 40 evaluated with no
range error. `|G_num − G_WKB| / max(...)` at `crossover_z`: min 8.0e-11, median **7.4e-08**,
p90 1.1e-06, **worst 4.0e-06** at k = 2.154e5/Mpc, z_response = 2.763e8 (numeric +1.758714e16
against WKB +1.758721e16). No step.

**§2 item 4 — audit §4.2, ran; the answer is yes.** 462 policy rows: numeric/complete 270,
WKB/complete 123, mixed/complete 61, **mixed/minimal 1**, fail/incomplete 7. So the `minimal` band
is reachable at shipped settings and A6 was a live defect on 0.2 % of rows. The 7 `fail` rows are
a separate observation (below).

**§2 item 5 — audit §4.3, ran.** `has_WKB_violation` is set on **0 of 7** `TkWKBIntegration` and
**0 of 5488** `GkWKBIntegration` rows, so no comparison past a violation point was possible and no
policy question arises in this configuration. Reported only.

**§2 item 6 — audit §4.4, ran on run B.** Stored derivative columns against central differences,
relative error at the z = 0.1 end / interior median / ratio: `d_lnH_dz` 7.2e-12 / 6.4e-09 /
**0.0011**; `d2_lnH_dz2` 2.5e-07 / 5.6e-06 / **0.044**; `d3_lnH_dz3` 1.3e-03 / 1.1e-03 / **1.14**;
`d_wPerturbations_dz` 7.6e-12 / 4.1e-08 / **1.9e-04**; `d2_wPerturbations_dz2` 3.7e-04 / 4.3e-05 /
**8.45**. All inside prompt 03's 10×; with the more accurate h = 3e-4 reference every ratio is
≤ 0.76 and the two worst columns fall to 4.4e-07 and 2.8e-05 at the end, showing the residual is
the reference's. Against the audit's pre-fix 3.0e-01 (ε″) and 3.8e-01 (w″) this is a three- to
six-order improvement on a real GenericEOS run. **A1 in stored data:** `wPerturbations` is
bit-identical (relative difference exactly 0.0) to the Λ-free form and differs from the pre-fix
form by 1.66 at z = 0.1.

**§2 item 7 — the source spline near the hand-over, ran.** 12 of 28 real `QuadSource` rows, spline
rebuilt as `_create_functions` does, compared with the exact radiation source at the nodes and at
log-midpoints in the lowest quarter of the region: node residual 9.6e-06 – 1.3e-04, midpoint
residual 1.4e-05 – **1.4e-04** of envelope — better than the 4.5e-04 worst case
`[06-source-spline-residual-vs-handover]` records. Each row stores 360–560 of the grid's 784
redshifts, so prompt 06's truncation is visible in real data.

**§2 item 8 — cost, measured.** Per work item: median 0.095 s (p90 0.47, max 1.43), plus a median
0.071 s for the analytic oracle; by richest regime 0.0038 s (0 oscillatory factors) / 0.14 s (1) /
0.17 s (2) / **0.28 s (3, four phase groups)**. Levin counters per row: regions median 19 (max
165), of which Clenshaw–Curtis-fallback regions median 13 (max 156); evaluations median 43 (max
343). The driver still routes most regions to Clenshaw–Curtis, as prompt 08 §6 found.

**Schema and resume, ran, passed.** All 54 stored `QuadSourceIntegral` rows have non-null `b`,
`total_abserr` and `total_converged`; re-running the entire pipeline against that datastore
reported `0 lookup, 0 compute, 0 store` in every stage, including
`CALCULATE QUADRATIC SOURCE INTEGRALS`, with row counts unchanged. `metadata` JSON is 3.3 kB
median and **6.1 kB max**, against the column's declared `String(256)`.

**Checked after the fact.** Three commits by the user landed on the branch while this pass was
running (`14fb9f6`, `5455bde`, `c0d8a83`), and the branch itself was renamed
`bessel-remedial-plan` → `transfer-remedial-plan`. `git diff --name-only 95cc326 c0d8a83` returns
nothing outside `prompts/transfer-remedial/` (the sibling campaign's planning folder, formerly
`prompts/bessel-remedial/`), so no production code, no `docs/` file and nothing measured here
changed; every number in the document holds for the tree at `c0d8a83` as well as at `95cc326`, and
the document's header and §2.1 record this. This commit sits on top of those three.

**Reasoned, not run.** The cost estimates in the document's §6 for the four things that remain
unverified (a `zend = 0.1` run, a full QCD chain, a radiation-EOS GenericEOS, and the 50+50 grid)
are extrapolations from the measured stage timings and item counts, and are labelled as such.

## Observations not acted on

1. **7 of 462 `GkSourcePolicyData` rows are type `fail`, quality `incomplete`** — the policy could
   not classify them at all. None of them was among the (k, z_response) pairs the source integral
   reached here, so what `build_partition` does with one is untested; it would raise ("Green's
   function is smooth on … but has no numeric representation"). `_classify_crossover` is outside
   this campaign's remit and the prompt forbids production changes, so this is recorded rather than
   investigated. A targeted harness run over exactly those pairs would settle it in seconds once
   `[12-region-check-absolute-tolerance]` is fixed.
2. **`metadata` exceeds its declared column length by 24×** (6.1 kB against `String(256)`). SQLite
   does not enforce `VARCHAR` length so nothing is lost today — the resume test proves the rows
   round-trip — but any backend that does would truncate or reject the partition record. This is
   log 09 observation 5 answered with a number.
3. **`T_q only` and `G and T_q` are unreachable by construction**, not merely unobserved: the pair
   grid is `combinations_with_replacement` over an ascending k array, so q ≤ r always and T_r hands
   over first. If those rows of the phase-group table are ever to be exercised, the pair grid would
   have to be changed (which belongs to whoever owns the (k,q,r) grid — audit §5 item 4 says that
   is `OneLoopIntegral`).
4. **`[10-classify-levin-keyerror]` was not reached.** `apply_GkSource_policy` ran 483 times
   without a `KeyError`, so no shipped mode has `|dθ_G/dlog(1+z)|` below `Levin_threshold = 1.5`
   everywhere in its WKB range. The latent bug is still there; it is now known not to fire at these
   settings.
5. **The `test-qcd-db*.sqlite` files in the repository root are pre-campaign and unusable** as a
   "before" reference: their `QuadSourceIntegral` table has none of the prompt-09 columns and no
   rows. They were opened read-only and left untouched, as the operational notes require.
6. **`docs/spec-code-audit/scripts/QI_03_measure.py` sections 2–4 and
   `QS_04_coverage_and_jacobian.py` part (a) remain broken**, and `TK_05` (d), `TK_06`, `TK_07`,
   `QS_03` still print pre-fix numbers. All six are the audit's frozen reproduction record
   (README §5 item 5); §3.2 of the document is the note a later reader needs, and no script was
   edited.
7. **The Ray `file_system_monitor` warning fired continuously** ("over 95 % full … object creation
   will fail if spilling is required", 24–26 GB free). No spilling occurred and no run failed for
   space, but a larger run on this machine would.

## State handed to the next campaign

1. **Two defects block a production sweep**, both on the board with reproductions:
   `[12-region-check-absolute-tolerance]` (the region guard, 43 % of work items, a one-line
   tolerance change in `_check_region_covers` — but a production change, so not made here) and
   `[12-atol-too-loose-for-the-source-integral]` (58 % of items have a raw integral below
   `DEFAULT_QUADRATURE_ATOL`). Neither can be fixed inside this campaign.
2. **`[12-handover-clamp-error-in-production]`** is the accuracy item: sub-horizon the residual
   against the oracle is a median 0.07–0.26 and it tracks the number of clamped LG factors, one to
   two orders above log 08's fixture prediction. The remedies are the three in
   `[08-handover-clamp-error]` — an LG sample at `z_init`, the overlap of
   `docs/lg-phase-and-handover-followup-2026-09.md` §1.4, or the first-order Taylor extension of
   log 08 deviation 2 — and this pass gives the production numbers needed to choose between them.
3. **The verification harness is reusable.** `scoped_pipeline_run.py` runs `main.py`'s pipeline at
   any wavenumber sample and model without editing production code;
   `run_quadsource_integrals.py` drives the production source-integral task over an arbitrary work
   list with arbitrary tolerances and records failures; the three `analyse_*.py` scripts take a
   datastore and print the tables of §5. A follow-up campaign should re-run all five after fixing
   (1) and compare against the numbers in `docs/source-remediation-verification.md`.
4. **What a production run costs, measured**: 4.5 minutes for 7 modes × 784 redshifts, 85 % of it
   the two Green's-function stages, 66 MB of datastore; the source integral is 0.095 s per work
   item (median). The document's §6 extrapolates to the 50+50 grid: order 1–2 days on 10 cores and
   10–20 GB.
5. **The audit's four UNVERIFIED items are closed** (§4.1, §4.2, §4.4) or answered as unreachable
   in this configuration (§4.3), with the numbers in `docs/source-remediation-verification.md`
   §5.5. `docs/spec-code-audit-2026-09.md` §8's "not yet run" row for §4.1–§4.4 is now superseded
   by that document; the audit itself was not edited (prompt 11 owns it).
