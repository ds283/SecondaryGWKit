# Log 03a — `TkNumericIntegration` and `wavenumber_exit_time`

**Prompt:** `prompts/tolerance-convergence/03a-tk-numeric-and-exit-time.md`
**Commit:** *"Sweep the Tk numeric and exit-time tolerances against their floors"* (SHA not
embedded, per the convention `prompts/background-solver-robustness` uses)
**Model:** Opus 5
**Date:** 2026-09-17
**Result:** COMPLETE WITH DEVIATIONS — both targets measured at full coverage and both
recommendations made; three §3 issues opened, one of which (the QCD grid's sample count) is a
finding about a file this prompt may not touch and one of which (the sporadic excursion) qualifies
the `TkNumericIntegration` recommendation rather than supporting it.

## What shipped

Two new files. **No production file is touched and no parameter moves** (README §5 rule 8), and
**no line of `ComputeTargets/tests/convergence_reference.py` changed** — prompt 03a §2.5 predicted
that T5 would need nothing there and that T6 might need a helper; it needed none either, because
`reference_drift` is generic in its `build` and `error_measure` and drives a root solve as happily
as an ODE solve.

- **`docs/tolerance-convergence/tk_numeric_exit_sweep.py`** (new, 1,600 lines). Regenerates every
  table of §§2–10 of the measurement document on stdout, with a progress log on stderr. Public
  entry point `main()`. Stages, in order: `build_subjects()` (three models, their version-2 grids
  at their own anchors, with the published anchor / sample-count / digest checked and a `raise` if
  any has moved); `sweep_subject()` (the `(atol, rtol)` matrix at all fifty wavenumbers under
  `BREAK_POINT_ALL`, plus the series-initial-data floor probe and, on the control, the exact-$T$
  oracle); `sweep_configuration()` (the production setting on the other three cells of the
  {grid generation} × {break-point policy} 2x2); `sweep_exit_time()` (the `(xtol, rtol)` matrix
  through `_solve_horizon_exit`, 50 $k$ × 3 offsets × 63 cells per model); `anchor_sensitivity()`
  (the version-2 grid rebuilt at eight perturbed anchors per model). Other new symbols:
  `Subject`, `CountingCosmology`, `hexit_solve`, `hexit_residual`, `exact_errors`,
  `series_initial_data`, `T_envelope_exact`, and the `report_*` emitters.
- **`docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md`** (new). §0 and §1 are written; §§2–10
  are the script's stdout verbatim from the `<!-- generated -->` comment down.
- Board (`IMPLEMENTATION_STATE.md`) — T5 and T6 filled in, the header and §1 row updated, three §3
  issues opened and `[02a-grid-digest-not-reproducible]` narrowed additively.
- `docs/OPEN_ISSUES.md` — three rows added, one row's text corrected, the count and date moved.

## Deviations from the prompt

### 1. The `rtol` axis carries two half-decades, and the reason is the opposite of §4.1's

**IMPLEMENTATION CHOICE.** §4.1 says "refine by half-decades around any crossing a decade is too
coarse to place". The axis shipped is `1e-6, 1e-7, 1e-8, 3e-9, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11` —
half-decades below the production setting and not around a crossing, because the first run (decades
only) showed there is no crossing to place: the maximum over the grid is **not monotone** in `rtol`
(§0.1 question 3). The half-decades were added to test whether the setting the rule selects is
stable under a finer sweep, and they changed the answer — with decades alone the rule selects
`1e-11`; with the half-decades it selects **`3e-11`**, which clears at 3.88e-8 while `1e-10` misses
at 1.31e-4 and `3e-10` misses at 5.76e-4. Alternatives considered: stopping at decades (would have
recommended a setting 3.3x tighter than necessary, which is the error README §6.1 rule 3 names);
going to third-decades (rejected — with a non-monotone maximum a finer grid measures the
phenomenon's density, not a threshold, and `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` is the
right place for that).

### 2. The `atol` off-axis points are one decade either side, not a subset of $k$

**IMPLEMENTATION CHOICE.** §4.1 permits "one decade either side of `1e-13` at two or three `rtol`
settings, at full $k$ coverage or a stated subset with its reason". Full $k$ coverage was taken at
four off-axis points — `atol` in {`1e-12`, `1e-14`} × `rtol` in {`1e-8`, `1e-10`} — because the
question `atol` has to answer in this sector turned out to be *which wavenumber excurses*, and a
subset of $k$ cannot answer it. The cost was 600 extra solves out of 6,300, which is not a reason
to sub-sample.

### 3. The break-point 2x2 is four cells, not two

**IMPLEMENTATION CHOICE.** §6 question 1 asks "separate the two changes if you can, and say so if
you cannot." They can be separated, so the production setting is taken on all four of
{version-0 per-$k$, version-2} × {`discontinuity`, `all`} rather than only on the production cell.
That is 450 extra solves and it turns "prompt 17's figures were taken under other conditions" into
two numbers instead of one difference. It also supplies the construction check: the
version-0/`discontinuity` cell reproduces prompt 17's published 3 and 13 wavenumbers above target
and its 8.64e-4 worst exactly on the two smooth models.

### 4. The exit-time error measure is the displacement in $u$, and the target is scored against the
guarantee rather than the achieved value

**IMPLEMENTATION CHOICE**, and the one a later reader is most likely to want to disagree with.
§2.4 requires the residual, the displacement from a converged re-solve and what the displacement
does to a grid, and leaves the *measure* open. Displacement in $u$ was chosen because it is
simultaneously the relative displacement in $1+z$ and therefore the relative displacement of the
whole lattice the anchor builds; all three quantities §2.4 lists are reported (§7.1, §7.2, §7.5).
The recommendation then applies README §6.1 rule 2 to Brent's **guarantee**
$x_{\rm tol} + r_{\rm tol}|u|$ rather than to the measured displacement, because the two disagree
about the only floor that exists: at the production setting the guarantee is 3.8e-7 and misses the
1e-7 row match while the achieved 7.86e-8 clears it. The argument for the guarantee is that this
parameter fixes a datastore key on machines other than this one; the argument against is that §6.1
rule 2 says "the row's own error measure, scored at the maximum over the whole production grid",
which is the measured column. **Both readings are tabulated in §7.6 and the user may take the other
one** — on the measured reading the answer is `unchanged`, with the floor dominating by 1.3x.

### 5. `docs/OPEN_ISSUES.md` §1.5's `[02a-grid-digest-not-reproducible]` row says prompt 03 owns the
prerequisite; it is 03a's

**STRUCTURALLY REQUIRED**, and already half-corrected in the tree. The issue was opened against
"**T6** / prompt 03" before §7 **D7** split the charter on 2026-09-17; the board's copy already says
03a in one place and prompt 03 in another (§3's "Next step, in two parts"). The index row was
corrected in this commit and the board's stale "prompt 03" reference with it. No measurement
changed.

## Verification performed

**Every figure below is on the version-2 source grid at each cosmology's own anchor, under
`BREAK_POINT_ALL` for the $T_k$ sector, with the reference's own drift beside it** (README §5 rules
5 and 6; board standing notes 1, 2 and 18).

**The three grids reproduce the record, and the script refuses to run if they do not.**
`RadiationModel` 2306 samples / `3bef2c06` at `z_init = 44523947729.772957` (prompt 03's control
anchor, reproduced to 15 digits); `LambdaCDMModel` 1778 / `60a3205a` at
`2.0636395964161516e+16`; `QCDModel` 2034 / `21ffc126` at `3.30033444460513e+16`. **No published
digest moved** (prompt 03a §9).

**References converged 50/50 on all three models**, worst drift 4.26e-11 / 5.23e-11 / 4.65e-09,
median 2.08e-11 / 3.82e-11 / 1.37e-09, smallest difference reported 3.96e-09 / 5.89e-09 / 6.83e-09.
60 of QCD's 650 cells did not stand ten times clear of their own wavenumber's drift and are
italicised in §8's table; **none of them is a maximum any conclusion rests on.** The three cells the
recommendation turns on are all resolved: at `rtol = 3e-11` the maxima are 2.70e-8 (Radiation, at
$k = 9.5585\times10^7$, whose own drift is 3.50e-11 — a factor of 771), 2.63e-8 (LambdaCDM,
$2.6256\times10^6$, 3.89e-11, ×676) and 3.88e-8 (QCD, $3.0455\times10^7$, 2.05e-09, ×18.9). The unresolved
cells are quiet wavenumbers on QCD, where the candidate error has fallen to the level of a reference
whose own drift is two orders above the other two models'.

**The radiation oracle calibrates the self-convergence.** The converged run with production initial
data is 2.53e-6 to 2.64e-6 from the exact $T$ at all fifty wavenumbers, and the series-initial-data
measure gives 2.53e-6 to 2.64e-6 on the same model — the two agree to three figures, as they did in
prompt 17 §8. For the root solve the oracle is exact: the reference re-solve is within **3.55e-15**
in $u$ of $1+z = k/(H_0e^N)$ at all 150 (k, offset) pairs, and the production setting's displacement
on the control is **exactly 0** at every one of them.

**The floor survives the grid and policy change.** 2.53e-6 / 2.52e-6 / 2.39e-6 minimum and
2.64e-6 / 2.64e-6 / 2.63e-6 maximum over $k$, against the inherited 2.52e-6 taken on the version-0
grid under `discontinuity` (`GkTk-remedial` prompt 17 §8). $x_i$ runs 0.003777 to 0.003981.

**Prompt 17's configuration reproduces.** Version-0 per-$k$ under `discontinuity` at the production
setting: Radiation 3 of 50 above 3e-6, worst 2.50e-4; LambdaCDM 13 of 50, worst **8.64e-4** — both
prompt 17's published figures. QCD gives 9 rather than 8 and is not a check, the background having
been replaced by `qcd-background-audit` prompts 04–06 since.

**The acceptance list, item by item.**

1. *Coverage.* 50 wavenumbers × 3 models × 13 `(atol, rtol)` cells for $T_k$, plus 3 extra
   configurations at the production setting; 50 × 3 × 3 offsets × 63 `(xtol, rtol)` cells for the
   root solve. Every table carries the grid generation, the anchor and the drift.
2. *`BREAK_POINT_ALL` used and stated*, and its difference from prompt 17's policy quantified: nil
   on the two models that declare no break points (identical evaluation counts, identical figures),
   and 9→6 / 7→4 wavenumbers above target on QCD for +0.82 % / +0.61 % of its evaluations.
3. *The `rtol` axis at full coverage*, nine settings, plus the four off-axis points. The tightest
   candidate `rtol` is `1e-11`, one decade above the reference's `1e-12` and three above SciPy's
   `2.22e-14` clamp; no step across the clamp is reported.
4. *The exit-time matrix through `_solve_horizon_exit`*, never through `object_get` or the
   datastore, at offsets 0, −5 and +4, with the binding term identified by measurement (0 of 150
   `xtol`-bound at the production `rtol`; `xtol` takes over below `rtol ≈ 2.6e-12`).
5. *Both floors addressed.* The $T_k$ initial condition re-confirmed with the inherited figure
   beside it; the exit-time bound established against the row match and declared **unestablished**
   against the digest in README §6.1 rule 6's words.
6. *`tk_numeric_exit_sweep.py` regenerates every table.* §§2–10 of the document are its stdout,
   unedited. No figure traces to `gk_numeric_sweep.py`.
7. *Prompts 01's and 03's figures unmoved.* `convergence_reference.py` is byte-identical, so
   nothing can have moved; `ComputeTargets/tests/test_convergence_reference.py` and
   `docs/gktk-remedial/tk_numeric_atol_sweep.py` were both run and are unchanged.
8. *Suites green* — see below.
9. *The five provenance fields* are in "State handed to the next prompt".

**Suites.** `ComputeTargets` **491 tests, OK, 187 s**; `CosmologyModels` **39 tests, OK**. 491 is
the count prompt 03's log records (the board header's note 16 says 484, which was prompt 01's
figure; prompt 02a added seven), and this prompt adds no test. **The wall-clock flake of note 16,
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, did not fire on this run.** The two
reproduction checks acceptance 7 asks for: `ComputeTargets.tests.test_convergence_reference` 32
tests OK, and `docs/gktk-remedial/tk_numeric_atol_sweep.py` emits its usual 294 lines with
`0.000864 @x=17.38`, 13 of 50 above target and 429,178 evaluations on `LambdaCDMModel` —
bit-for-bit the figures prompt 01 recorded, and the same three numbers this prompt's §3 reports for
the version-0 / `discontinuity` cell. `black --check` clean on both new files.

## Observations not acted on

1. **`TolerancePair.rtol_step_is_effective` applies `solve_ivp`'s clamp to a root solve.**
   `convergence_reference.SCIPY_RTOL_FLOOR = 2.22e-14` is `100 eps`, which is what
   `scipy.integrate` clamps at. `scipy.optimize.brentq` refuses anything below `4 eps = 8.88e-16`
   instead, so a `TolerancePair` driving `_solve_horizon_exit` is warned about at `rtol = 1e-15`
   when it should be warned about at `rtol < 8.9e-16`. The facility is not wrong for the sector it
   was written for; it is being used for one it was not. It cost nothing here — the exit-time
   reference tightening is `1e-15`, which draws a spurious note that is recorded rather than
   suppressed. `[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]`.
2. **The lifted `main.source_grid_spacing_profile` prints its guarded-node line to stdout**, so a
   script that emits markdown on stdout has grid-construction chatter above its first table — eleven
   lines here, one per version-2 QCD build. Harmless, and the document says where its body starts,
   but it means a `> FILE.md` redirection of this or any other version-2 script needs an editor
   afterwards. Not this prompt's file to change.
3. **The exit-time sector's cost is not in the record anywhere and is now measured**: 6,963 Hubble
   evaluations for 150 (k, offset) solves across all three models at the production setting,
   tens per solve. It is negligible beside anything else in the pipeline, which is why every setting
   in §7.6's axis costs within 4 % of every other.

## State handed to the next prompt

**For prompt 05 and for README §7 D1 — what the user is being asked to accept.** Two parameters,
both of them the `rtol` half of a pair whose `atol` half is settled or inert. **Accepting both
closes D1**, the `GkNumericIntegration` half having been accepted on 2026-09-17; prompt 05 may not
start until then.

**(a) `DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11`** — a *change* from the shared
`DEFAULT_REL_TOLERANCE = 1e-8`.

- **Value and what it keys:** `3e-11`; `TkNumericIntegration` alone, 50 objects per model, whose key
  is `(…, atol_serial, rtol_serial, break_point_kind)`.
- **What measurement chose it:** `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §4.1,
  README §6.1 rules 2 and 3 applied mechanically. Three models, all fifty production wavenumbers,
  **source-grid generation version 2 at each cosmology's own anchor**, `BREAK_POINT_ALL`,
  envelope-relative error, reference `(1e-18, 1e-12)` converged 50/50 with worst drift 4.26e-11 /
  5.23e-11 / 4.65e-09. Swept loose to tight over nine settings, `3e-11` is the first whose maximum
  over all three models — **3.88e-8** — is at or below the floor.
- **The floor it is competing against:** the $T = 1, T' = 0$ initial-condition truncation,
  **2.39e-6 to 2.64e-6** of the envelope, re-measured here on the version-2 grid under
  `BREAK_POINT_ALL` against the inherited 2.52e-6 (`GkTk-remedial` prompt 17 §8, version-0 grid,
  `discontinuity`). The recommended setting sits 62x under it; the production setting sits 141x
  *over* it at its worst wavenumber.
- **Cost, at the setting and one step either side:** right-hand-side evaluations for the whole
  sector, all three models (50 objects per model): `1e-10` **1,727,040** (+35.7 %), **`3e-11`
  1,774,977 (+39.4 %)**, `1e-11` 1,810,617 (+42.2 %), against 1,272,891 at the production `1e-8`.
  Per model at `3e-11`: +44.8 % / +37.2 % / +36.7 %.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 03a, this log, 2026-09-17.
- **The caveat that must travel with it:** the maximum is **not monotone** in `rtol`
  (`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`). One of the 150 runs sits above target at each
  of `1e-9`, `3e-10` and `1e-10`, and it is a different run each time. `3e-11` is the loosest
  setting that clears in this sweep; it is not a setting at which the excursion is shown to be
  impossible.

**(b) `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`** — **unchanged**, and not reopened (§7 D1, the
user, 2026-09-12). What this prompt adds to its provenance note is a measurement, not a value: in
this sector `atol` is **not** inert in prompt 03's sense. Across `1e-12 → 1e-14` at `rtol = 1e-8` it
moves the median of the per-$k$ maxima by at most 2.1x but the maximum by up to 205x, by changing
which wavenumber draws a bad step sequence. Its note should say that it is a step-selection knob
whose value was chosen by `GkTk-remedial` prompt 12 and confirmed by the user, and that no setting
of it removes the excursion — `rtol` does.

**(c) `DEFAULT_HEXIT_REL_TOLERANCE = 1e-9`** — a *change* from the shared `1e-8`.

- **Value and what it keys:** `1e-9`; `wavenumber_exit_time`, one object per wavenumber, 50 per
  model. **Its key is an inequality, not an equality**
  (`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`): the lookup returns the loosest
  stored row at least as tight as the request, so *tightening* to `1e-9` misses every existing row
  and recomputes, which is the behaviour prompt 05 wants, while a later loosening would silently
  reuse it.
- **What measurement chose it:** `TK-NUMERIC-AND-EXIT-TIME.md` §7.6. Three models, all fifty
  wavenumbers, three offsets (0, −5, +4), through `_solve_horizon_exit` and never the datastore;
  reference `(1e-300, 1e-14)` with worst drift 4.97e-14 / 7.82e-14 in $u$ and, on the control,
  3.55e-15 from the exact $1 + z = k/(H_0e^N)$. `1e-9` is the loosest `rtol` whose Brent guarantee
  $x_{\rm tol} + r_{\rm tol}|u|$ at the largest production $|u| = 38.04$ clears the floor: **3.81e-8
  against 1e-7**. Measured displacement at that setting is 6.66e-9 / 9.21e-9.
- **The floor it is competing against:** `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, the tolerance
  at which `Datastore/SQL/ObjectFactories/redshift.py:40` reuses an existing redshift row — the only
  threshold in the tree that says what displacement a grid can absorb. The production setting's
  guarantee, 3.8e-7, is 3.8x **over** it; its achieved displacement, 7.86e-8, is 1.3x under.
  Deviation 4 above records that the recommendation rests on the guarantee and that the user may
  read it the other way, in which case the answer is `unchanged`.
- **Cost:** 7,009 Hubble evaluations against 6,963 at the production setting, for all 150
  (k, offset) solves on all three models — **+0.7 %**.
- **Campaign, prompt, log, date:** as above.

**(d) `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10`** — **unchanged, inert and unchosen**, and README §1.2's
closing rule governs its note: *the provenance of this constant cannot be established from the
record*. It reaches `root_scalar` as `xtol` and binds at **0 of 150** (k, offset) pairs on each of
the three models at the production `rtol`, and at the recommended `rtol = 1e-9` it still binds
nowhere. What the note must also say is the coupling, because it is the one thing that makes the
constant matter: with `xtol = 1e-10` the pair **cannot pin the anchor better than 1e-10 relative**
however far `rtol` is tightened — `xtol` takes over below `rtol ≈ 2.6e-12` and the guarantee flatlines
at 1e-10 (§7.2, §7.6). Any design tolerance prompt 05 adopts below 1e-10 requires this constant to
move with `rtol`.

**For prompt 05 specifically, on `[02a-grid-digest-not-reproducible]`.** The prerequisite
measurement part (i) asked for is §7.5 and §7.6, and it carries two results the issue did not have.
*First*, its figures are confirmed independently and as a swept target rather than a review probe:
at $k = 3\times10^8$ and offset −5, which is the anchor itself, the production setting's
displacement from a converged re-solve is **3.62e-13** on LambdaCDM and **2.50e-9** on QCD — the
issue's 3.6e-13 and 2.5e-9. *Second*, the option the issue lists as the exact alternative — digest
the determining data, "the integer subdivision vector, whose ties are a measured 6e-3 clear" —
**does not survive measurement on QCD**: the version-2 grid's *sample count* there moves from 2034
to between 2013 and 2032 under relative anchor perturbations of 1e-14 and upwards, while
LambdaCDM's stays at 1778 and Radiation's at 2306
(`[03a-qcd-v2-grid-sample-count-is-not-reproducible]`). 1e-14 is six orders below the displacement
the production anchor solve achieves. A design tolerance applied to the row match and the digest
together therefore has to contend with a subdivision vector that is itself undetermined at the
anchor's precision, on the one cosmology that has declared break points.

**For prompt 04.** Nothing in this prompt's scope bears on **T7**. The three version-2 grids, their
anchors and their digests are unchanged and are the ones §7.5's perturbation table starts from; the
guarded-node census is prompt 02a's and is not re-taken here.
