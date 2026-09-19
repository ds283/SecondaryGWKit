# Prompt 01 — The convergence harness, and one production grid

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board items **T1** and **T2**; README §2 **(b)** and **(h)**
**Narrows:** `[00-three-production-grid-reproductions]` (board §3)
**Depends on:** nothing. This is the first prompt of the campaign.
**Recommended model:** **Opus** — the production change is nil and the numerical content is
modest, but every measurement the campaign makes will be taken through what you build. A facility
that bakes in one sector's assumptions costs prompt 04 a rewrite; one that cannot express "one step
tighter" for an *integer order* as well as for a tolerance cannot serve prompt 04 at all.

**Files you may create or touch:**
`ComputeTargets/tests/convergence_reference.py` (**new**),
`ComputeTargets/tests/test_convergence_reference.py` (**new**),
`ComputeTargets/tests/wkb_reference.py` (**the grid helpers only** — do not touch the stand-in
models, the closed forms or the error measures),
`ComputeTargets/tests/test_source_grid.py` (to repoint it at the hoisted grid, nothing else),
`ComputeTargets/tests/test_background_segmentation.py` (to repoint it, nothing else),
`docs/gktk-remedial/tk_numeric_atol_sweep.py` (to repoint its import, nothing else),
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** any production file. **This prompt changes no production code at all** — not
`main.py`, not `config/defaults.py`, not anything under `ComputeTargets/` outside `tests/`. Not
`QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` or `AdaptiveLevin/` (README §0.4).
Not `ComputeTargets/tests/wkb_reference_data.json` — that fixture is prompt 04's, under §7 D5, and
it is not yours.

**Read first:** README §0.3 (including the re-anchor bullets), §2 (b), (f), (h), (i), §3.1 in full,
§5, §6.1; `RECONCILIATION.md` **§7.3 and §7.4**; board §3's
`[00-three-production-grid-reproductions]` entry **with its 2026-09-16 narrowing blockquote**;
`ComputeTargets/tests/wkb_reference.py` in full (523 lines — you are extending it and you must not
duplicate what it has); `ComputeTargets/tests/test_source_grid.py:96-165` (the lift block,
`_production_grid` and `_production_base_grid`) and `:185-203` (`_BrokenCosmology`, the stand-in);
`ComputeTargets/tests/test_main_plumbing.load_main_py_functions`;
`docs/gktk-remedial/tk_numeric_atol_sweep.py:24-64` (the method docstring), `:115-160` (the
constants), `:167-200` (`_Wavenumber`, `_KExit`, `_Proxy`), `:199-345` (`geometry`, `run`,
`sample_errors`, `summarise`) and `:1088-1160` (`run_gk`, `SECTORS`, `sector_errors`);
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §4 and §9.1.

---

## 1. Why this prompt exists first

`GkTk-remedial` prompt 17 needed a convergence facility and did not have one: its test is inline in
`sweep_model`, so the second sector could not reuse it and the campaign got one clean axis in one
sector out of eight. Worse, prompt 17 measured a reference that **had not converged** on `QCDModel`
at four wavenumbers and reported candidate errors through it anyway. That is the specific failure
this campaign exists to prevent, and §5 rule 5 is written from it.

So the campaign's first deliverable is the facility, and its second is the thing every measurement
taken through the facility has to name: **which grid it was taken on**. There are four
reproductions of "the production source grid" in the test tree and they are not the same grid.

**What changed at the re-anchor, and it changes your job.** The rebase opened
`[00-three-production-grid-reproductions]` believing there were three reproductions and that none
was version 2. There are four, and one of them — `test_source_grid.py:127` `_production_grid` — is
**already the complete version-2 construction**, already lifting both `main.py` helpers the way the
issue's "next step" prescribed, and already under the suite. It is private, and it lives in a test
module, so nothing else can reach it. **That is now the whole defect.** Your grid task is to
*hoist*, not to build. Do not write a second version-2 construction.

## 2. Part A — the convergence facility

Create `ComputeTargets/tests/convergence_reference.py`. It sits beside `wkb_reference.py` so that
it is importable by tests and by `docs/` scripts alike and is subject to the suite. **No Ray, no
datastore** (README §5, `GkTk-remedial` §5 rule 7): call remotes through their undecorated
`_function` and use `wkb_reference.py`'s stand-ins, as
`tk_numeric_atol_sweep.py:41-51` describes and `residual_convergence.py` does.

### 2.1 What it must provide

1. **A converged reference** for a given (target, model, $k$, geometry) at a reference accuracy
   setting the **caller supplies**. The facility does not own the reference setting — prompt 03
   and prompt 04 choose different ones and prompt 17's `(1e-18, 1e-12)` is not privileged.

2. **The drift statistic**, and the criterion **evaluated, not merely reported**: the same
   reference one step tighter, differenced in the caller's error measure, against
   `drift <= smallest_reported_difference / 10`. The return value must make the verdict
   available as a boolean *and* the numbers available for a table. A caller must not be able to
   obtain a drift figure without also obtaining whether it passed.

3. **"One step tighter" for both kinds of knob.** A decade for a tolerance pair; **one order** for
   a Gauss order. Prompt 04 audits four integer orders and has no tolerance to move, so a facility
   that can only multiply a float by 0.1 does not serve half the campaign. Express the step as a
   property of the knob, not of the call site.

4. **The anchor comparison**, against every closed form the tree already provides, wherever the
   model is constant-$w$ — README §3.1's table, all nine rows, not just the two solution oracles.
   The facility exposes them; the prompt that measures decides which apply. §5 rule 5 says a drift
   quoted without the oracle error beside it, where an oracle exists, is uncalibrated, and this is
   the mechanism that makes obeying it the easy path.

5. **The error measures**, by **reusing** `wkb_reference.py`'s `phase_error`,
   `difference_error` and `envelope_relative_error`. Do not restate their definitions. If one of
   them is the wrong shape for what you need, say so in the log as a deviation — do not write a
   fourth.

### 2.2 What to fold in, and what to leave alone

`tk_numeric_atol_sweep.py` already contains most of the machinery, written for one sector and one
campaign: `geometry`, `run`, `sample_errors`, `summarise`, the `_Wavenumber` / `_KExit` / `_Proxy`
stand-ins, and on the $G_k$ side `run_gk`, `SECTORS` and `sector_errors`. **Fold these into the
facility** so there is one implementation, and have the script import them back.

**The script's published figures must not move.** That is the acceptance in §5 below and it is the
whole construction check: if folding the machinery into a shared module changes what the script
prints, the fold changed a measurement, and a measurement that changes when you move its code was
not measuring what it claimed. `tk_numeric_atol_sweep.py`'s `main_break_points` and
`main_per_sector` entry points are `GkTk-remedial` prompts 18–20's and are **not** yours to
restructure; repoint their imports and leave their logic alone.

### 2.3 What it must not do

- It must not choose tolerances, recommend settings, or embed a target. §6.1's target rule belongs
  to prompts 03 and 04 and the user; the facility measures.
- It must not embed `(1e-18, 1e-12)` as a default. Prompt 17's reference setting is an input.
- It must not know about `config/defaults.py`'s values.

## 3. Part B — one production grid, hoisted

### 3.1 What is there now

| # | Site | Generation | What it is |
|---|---|---|---|
| 1 | `wkb_reference.py:152` `production_source_grid` | **v0** | bare `logspace`; docstring cites a `main.py:410-419` that has not been the grid code for two campaigns |
| 2 | `test_background_segmentation.py:90` `production_source_grid` | **v1** | `break_z` / `feature_z`, no `spacing` |
| 3 | `test_source_grid.py:127` `_production_grid` | **v2** | the full production construction — **hoist this one** |
| 4 | `test_source_grid.py:152` `_production_base_grid` | v0, deliberate and named | mirrors `main.py:944`'s own base-grid step, which the spacing profile is measured on |

`docs/gktk-remedial/tk_numeric_atol_sweep.py:215` imports #1, which is why every published
tolerance figure in this repository was scored on version 0.

### 3.2 What to do

Move #3 into `wkb_reference.py` — README §3.3's file table puts the grid helpers there — as a
**named, public, version-tagged** set alongside named v0 and v1 constructions. Then:

- repoint #1's callers, including `tk_numeric_atol_sweep.py:215`, at the named **v0** helper;
- repoint #2 at the named **v1** helper;
- repoint `test_source_grid.py` at the hoisted **v2**;
- leave #4 where it is, named as it is. It is not a stray: `main.py:944-963` has the same two-stage
  structure — a base grid, a spacing profile measured on it, then `populate_source_grid`.

**Keep versions 0 and 1 constructible and named.** Prompt 17's figures were taken on v0 and
`test_background_segmentation`'s assertions on v1; neither may be silently re-scored. A caller must
have to *say* which generation it wants — a bare `production_source_grid()` that silently means one
of them is the defect you are removing, so if you keep that name, it must fail loudly rather than
default.

**Cross-check the version against production, do not assert it.** `main.py` declares
`SOURCE_GRID_CONSTRUCTION_VERSION`; `test_source_grid.py:1030` already asserts it is 2. Your
hoisted v2 must be tagged with the value it reproduces, so that when production moves to version 3
the test tree says so instead of quietly disagreeing.

### 3.3 The trap the re-anchor found

`cosmology_feature_redshifts`, which the lift pulls out of `main.py`, **no longer computes** the
equality redshifts from `omega_m` / `omega_r` / `omega_cc`. Since
`prompts/background-solver-robustness` prompt 09 it asks the cosmology for
`z_matter_radiation_equality` and `z_matter_lambda_equality` and **raises `RuntimeError` if either
is missing, deliberately and with no fallback** (`RECONCILIATION.md` §7.4).

A stand-in that cannot answer both will hit that raise. `test_source_grid.py:185-203` already
carries one that can, and says in terms that it is deliberately not a `BaseCosmology`. **Reuse
it** — hoist it with the grid rather than writing a second. If you find yourself adding a fallback
to `cosmology_feature_redshifts`, or stubbing the two attributes with the closed form, stop: that
reinstates exactly the defect that campaign removed, and it is a production change besides.

## 4. Acceptance

README §3.1's acceptance is in two parts and both are required.

### 4.1 The construction check — reproduce prompt 17 on prompt 17's grid

Re-run `tk_numeric_atol_sweep.py`'s reference-convergence measurement through the new facility, on
the **named version-0 grid**, and reproduce `TK-NUMERIC-ATOL-SWEEP.md` §4's table:

| model | worst reference drift | at $k$ [1/Mpc] | median drift over the grid |
|---|---|---|---|
| `RadiationModel` | **4.21e-11** | 3e+08 | **1.92e-11** |
| `LambdaCDMModel` | **5.7e-11** | 1.561e+08 | **3.76e-11** |

These two rows are the reproduction target, **to the digits published**. They are single-segment
runs (§9.1, "max segments" = 1), so the numeric-ODE split that `GkTk-remedial` prompts 18–20 landed
does not touch them, and §9.1 re-published them unchanged after that split. If they do not
reproduce, the fold changed a measurement — stop, per §5.

**`QCDModel` is not a reproduction target and you must not treat it as one.** Its published figures
are 6.17e-06 (§4, unsplit) and 1.97e-07 (§9.1, split), and `qcd-background-audit` has since moved
the background twice more; the board quotes 7.08e-09 on the current tree. **Report what you
measure, beside all three published values, and say which tree each was taken on.** Do not tune
anything to land on any of them.

### 4.2 The tree check — the same statistic on version 2

Report the same statistic on the **version-2 production grid**, beside the version-0 figures, for
all three models. The log records the difference and does not explain it away. This is the first
figure in the campaign's record that carries its grid generation under §5 rule 6, and it is the
number prompts 03 and 04 will be measured against.

### 4.3 The mechanical checks

| Check | Threshold |
|---|---|
| Hoisted v2 against `_production_grid` before the move | **bit-identical** on both `QCD_Cosmology` and the `LambdaCDM_GenericEOS` stand-in — quote the sample counts (expect **1,996** on QCD, **1,778** on LambdaCDM) and the `redshift_grid_digest` of each |
| Named v0 against `wkb_reference.production_source_grid` before the move | **bit-identical**, and `tk_numeric_atol_sweep.py`'s §4 figures unchanged |
| Named v1 against `test_background_segmentation`'s construction before the move | **bit-identical**; that module's assertions pass unmodified |
| `ComputeTargets` suite | 452 → 452 + *n*, OK. Quote *n* and the new wall time |
| `CosmologyModels` suite | 39 → 39, OK |
| Production files in the diff | **zero** |
| `black --check` on every file you touched | clean |
| Runtime of the new test module | quote it. A `QCD_Cosmology(max_z=1e12)` costs 3,000 node solves, so build models in `setUpClass`; if the module exceeds ~30 s say why |

The suite is ~164 s at `bc6dc97`. If your additions take it past ~240 s, say so in the log and say
what dominates — this campaign will add to it five more times.

## 5. Stop conditions

Stop and report; do not repair, do not loosen, do not proceed.

- **§4.1's two rows do not reproduce.** The fold changed a measurement. Report both measured values
  beside both published ones and stop. Do not adjust the facility until it matches.
- **Any bit-identical check in §4.3 fails.** A hoist that changes the grid is not a hoist.
- **You need to change a production file to proceed** — including adding a fallback to
  `cosmology_feature_redshifts`, or touching `main.py` so it can be lifted more conveniently.
- **You find that `main.py`'s grid construction has moved again**, so that the hoisted v2 no longer
  reproduces what production builds. Say what moved; do not chase it.
- **You want to touch `wkb_reference_data.json`.** That is prompt 04's, under §7 D5.
- **A convergence test you write fails to converge and you are tempted to continue anyway.** That
  is the error prompt 17 made and the reason this campaign exists (README §4.3).

## 6. Deliverables

1. `ComputeTargets/tests/convergence_reference.py` and `test_convergence_reference.py`.
2. The grid helpers in `ComputeTargets/tests/wkb_reference.py`, three named generations, and the
   four call sites repointed.
3. `logs/01-convergence-harness-and-grid.md` on `GkTk-remedial` §5.1's template, with:
   - **Verification performed** carrying §4.1's reproduction table, §4.2's version-2 table for all
     three models side by side with version 0, and every §4.3 check with its measured value;
   - **State handed to the next prompt** giving: the facility's public API — every name and
     signature, including how a caller expresses a tolerance step and an order step; the names of
     the three grid generations and how a caller selects one; the sample count and
     `redshift_grid_digest` of the version-2 grid on each model; and the reference setting and
     drift figures you used, so prompt 03 does not re-derive them.
4. Board row 01, item rows **T1** and **T2**, and the narrowing of
   `[00-three-production-grid-reproductions]` in §3 — with `docs/OPEN_ISSUES.md` updated in the
   **same commit**, count and date corrected (`CLAUDE.md`).
5. One commit, README §5 rule 1.

**If you finish Part A and Part B is blocked, that is a `PARTIAL` and it is an acceptable
outcome** — say so in the log's Result line and hand the state over. Do not half-hoist the grid to
make the row go green.
