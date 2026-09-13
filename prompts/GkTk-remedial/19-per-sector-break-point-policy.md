# Prompt 19 — Let each numeric integrator choose which declared break points it splits at

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[17-qcd-reference-not-converged]`
**Review sections:** §10.1 (the $G_k$ numeric run's accuracy and cost); §12.5. Also
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9 in full — prompt 18's measurement, which is the
whole evidential basis of this prompt.
**Design facts:** README §2 (h) — the numeric region is sound and its diagnostic is a live warning;
§2 (d) — the floors are not targets.
**Depends on:** 18 (which built the split and measured what this prompt decides), 02, 11, 12, 16, 17.
**Recommended model:** Opus — the same shared production integrator as prompt 18, and a policy that
now differs between the two sectors.
**Files you may touch:** `Quadrature/integrators/numeric_with_phase_cut.py`;
`ComputeTargets/TkNumericIntegration.py` and `ComputeTargets/GkNumericIntegration.py` (the call
sites, and *only* the break-point argument and its comment);
`ComputeTargets/tests/test_numeric_break_points.py`; extensions to
`docs/gktk-remedial/tk_numeric_atol_sweep.py` and a new additive section of
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`; plus the log, the status board and
`docs/OPEN_ISSUES.md`.
**Do not touch:** `config/defaults.py` — **no tolerance changes in this prompt**; `atol` stays at
`1e-10`, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` at `1e-13` and `rtol` at `1e-8`, whatever you measure.
`CosmologyModels/` — **the declaration is finished**; prompt 18 settled its shape and this prompt
changes only who asks for what. The `Datastore/` factories and every lookup key
(`[18-numeric-solver-not-in-lookup-key]` is still the user's). The cumulative-table quadrature
path, `BREAK_POINT_STANDOFF`'s value, the stop window and the $\sqrt{z_{e3}z_{e4}}$ limit,
`store()`'s $(B,\delta)$ algebra, the `GkWKBValue`/`TkWKBValue` schemas, the `GkSource` rectifier,
and everything README §5 rule 8 lists.

Read first: README §2 (d), (h), §5, §6; `IMPLEMENTATION_STATE.md`'s
`[17-qcd-reference-not-converged]` and `[18-numeric-solver-not-in-lookup-key]` entries in §3;
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9 (all of it, §9.1, §9.3 and §9.7 especially);
`logs/18-numeric-ode-break-points.md` — the "State handed to the next prompt" section.

---

## 1. What is wrong

Prompt 18 wired the cosmology's declaration to the ODE and split the numeric integration at the
declared **jumps**. That fixed the four wavenumbers prompt 17 could not measure through, by
347×–5764×, and took `QCDModel`'s $T_k$ reference from 23 wavenumbers above the convergence
criterion to **3** (worst $1.97\times10^{-7}$ against $\le3.4\times10^{-8}$). It did not close the
issue. The residue is the other kind of declared non-smoothness: the 404 interior knots of the
$T(z)$ spline, where quantities built from $T(z)$ are only $C^2$.

Prompt 18 measured what closing it would cost and, as its §4 required, asked rather than decided
(`TK-NUMERIC-ATOL-SWEEP.md` §9.7). Splitting at all 407 declared points takes the three offenders
to $6.89\times10^{-10}$, $4.65\times10^{-9}$ and $2.30\times10^{-9}$ — inside the criterion — at
**+218.8 %** ($T_k$) and **+154.8 %** ($G_k$) of the production right-hand-side evaluations.

**The reason that question has a cheap answer is that the two sectors do not share the problem.**
§9.1 measures $G_k$ converging at all 50 wavenumbers on all three models — worst
$8.41\times10^{-9}$ on QCD against the $3.4\times10^{-8}$ criterion — both before and after the
split. The $G_k$ sector never had the failure the knots would fix. And the sectors have very
different object counts (README §6): `TkNumericIntegration` is **one object per $k$, 50 per
model**, while `GkNumericIntegration` is one per $(k, z_{\rm source})$, **~65,000 per model**. So
paying +219 % on $T_k$ is ~34 s of compute once, and paying +155 % on $G_k$ is several core-hours
per model for an improvement to a quantity that already converges.

**The user's decision (2026-09-13):** the cosmology declares all its potential non-smoothness and
it is for each consumer to decide what to do with it. `numeric_with_phase_cut` is a reusable
integrator and must therefore let its caller **specify which declared break points to account
for**; the $T_k$ integrator asks for jumps *and* kinks, and the $G_k$ integrator asks for jumps
only. Each call site carries a note saying that the choice rests on measurement — necessary in one
sector, unnecessary in the other.

## 2. What to build

### 2.1 The parameter

Give `numeric_with_phase_cut` a keyword argument naming which kind of declared break point it
splits at, in the vocabulary prompt 18 established (`BREAK_POINT_ALL` /
`BREAK_POINT_DISCONTINUITY` in `CosmologyModels/GenericEOS/GenericEOS.py`, re-exported through
`ComputeTargets/BackgroundModel.py`). It is passed through to `declared_discontinuities_in_z`,
which currently hard-codes `BREAK_POINT_DISCONTINUITY` at
[`numeric_with_phase_cut.py:524`](../../Quadrature/integrators/numeric_with_phase_cut.py).

- **The default must reproduce today's behaviour exactly** — `BREAK_POINT_DISCONTINUITY`. Every
  other caller (the tests, the reproduction scripts under `docs/`, any future integrator) then
  keeps the numbers it has, bit for bit, and prompt 18's §9 figures stay valid for the sector that
  keeps them. If you think the parameter should instead be required, say so in the log and ask —
  do not make it required unilaterally.
- Nothing in `Quadrature/` or `ComputeTargets/` may name a temperature, a model or an equation of
  state. The generic property prompt 18 established is not weakened by this prompt: the caller
  chooses a *kind*, never a point.
- `declared_discontinuities_in_z`'s own default stays as it is, so
  `test_numeric_break_points.TestQCDReferenceConvergence.test_the_branch_crossing_is_inside_the_range`
  keeps its meaning and its expectation.

### 2.2 The two call sites, each with its reason

`TkNumericIntegration.py:374` and `GkNumericIntegration.py:341` both call
`numeric_with_phase_cut.remote(...)`. **Both must pass the argument explicitly** — the $G_k$ one
too, even though it is the default, because the point of this prompt is that the choice is a
decision each sector took on evidence, not something one sector inherited by omission.

Follow the shape of the `warn_unresolved_osc=False` comment already at both sites: a short comment
block giving the reason and pointing at the measurement. In substance:

- **$T_k$ asks for every declared break point.** Necessary, by measurement: with jumps only, 3 of
  the 50 production wavenumbers stay above the convergence criterion (worst
  $1.97\times10^{-7}$ against $3.4\times10^{-8}$); adding the $C^2$ knots takes them to
  $\le4.65\times10^{-9}$. It is affordable here and only here because this sector is one object per
  $k$, 50 per model — the +219 % in right-hand-side evaluations is ~34 s of compute for the whole
  sector on QCD.
- **$G_k$ asks for the jumps only.** Unnecessary, by measurement: this sector converges at all 50
  wavenumbers on all three models with the jumps alone (worst $8.41\times10^{-9}$ on QCD, against
  the same $3.4\times10^{-8}$), so the knots would buy nothing — and at ~65,000 objects per model
  the +155 % would be several core-hours per model.

Cite `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.1 and §9.7 from both comments. A later reader
must be able to see that the asymmetry is a measured decision, not an oversight, and must be able
to find the measurement that would have to be redone to revisit it.

### 2.3 What splitting at 404 points exposes that splitting at 1 did not

Prompt 18's segmented driver was exercised with **one** interior boundary in production
(`T_120_MEV` is the only declared jump inside any production numeric range). Asking for every
declared break point takes that to ~400 on the $T_k$ QCD range, and several things that could not
arise at one boundary can arise at four hundred. All of these are yours to handle and to test:

1. **Segments containing no requested sample.** With ~400 boundaries and ~100 samples, most
   segments contain none. The driver appends the segment's lower boundary to `t_eval` for
   every non-last segment and then checks `len(segment_t) == num_requested + 1`; confirm that
   holds when `num_requested` is 0, and that the assembled `t`/`y` still come out as exactly the
   requested `z_sample`, in order — the equality guard against the requested samples must still
   hold.
2. **Break points that nearly coincide, with each other or with a sample.** `BREAK_POINT_STANDOFF`
   displaces each boundary by $10^{-12}$ relative in $(1+z)$. With 404 knots, two boundaries
   closer than that — or a boundary that lands on or crosses a requested sample after the
   standoff — would give a zero-length or *inverted* segment. Establish whether the production
   knot spacing can do this; if it can, the driver must collapse or drop the offending boundary
   rather than emit a bad segment, and if it cannot, say what the minimum spacing actually is and
   guard it anyway. Prompt 18 §2.3 item 1's rule stands: do not round a break point onto a sample.
3. **`mode="stop"` with the event in a late segment.** $T_k$ runs in `"stop"` mode. The terminal
   event must still stop the *whole* integration, and `find_phase_extremum` must still find its
   extremum through the composite dense output, with ~400 dense segments rather than 2.
4. **The diagnostic and the accounting still aggregate** over ~400 segments, not report the last —
   `has_unresolved_osc`, `unresolved_z`, `unresolved_efolds_subh`, `RHS_evaluations`,
   `compute_steps`. Preserving the warning is a campaign stop condition (README §2 (h)).

If any of this forces a change to the driver beyond passing the parameter through, that is
expected — but it is a change to code both sectors run through, so say so plainly in the log and
re-run the $G_k$ regression of §3 item 2 to prove the sector that did *not* change its policy did
not change its numbers.

## 3. What must be measured

Extend `docs/gktk-remedial/tk_numeric_atol_sweep.py` and add a new top-level section to
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` — **additive**. §9 was correct for the tree it was
taken on; do not rewrite it.

1. **The acceptance test: the §2.1 convergence test on `QCDModel` $T_k$ at all 50 wavenumbers,
   with the new policy.** Prompt 18 measured the all-breaks split at **three** wavenumbers only
   (§9.7), so it is not yet known that splitting at the knots leaves the other 47 alone. §9.7 also
   records an all-breaks run making $k=4.972\times10^7$ *worse* (7.06e-06) before the standoff was
   added — boundary placement is not a settled matter at 404 boundaries the way it is at one.
   Report the drift per $k$, before and after, in the same envelope-relative measure.
2. **A $G_k$ regression, all three models, and it must be exact.** $G_k$'s policy is unchanged, so
   every $G_k$ figure in §9 must reproduce **bit for bit** — 1.94e-11, 2.1e-11 and 8.41e-09 worst
   over the grid, and the per-object evaluation count 13320 on QCD. Not "within a few per cent".
   If a $G_k$ number moves, the shared driver changed behaviour for a sector that did not ask it
   to: stop and report.
3. **A smooth-model regression.** `RadiationModel` and `LambdaCDMModel` declare nothing, so both
   sectors must reproduce exactly on both, whatever policy is requested — including prompt 17's
   two control figures (2.534e-6 at $k=10^6$; 2.56e-4 at $x=10.78$, $k=3\times10^8$) with RHS
   counts **7403 and 8483** unchanged. A model that declares nothing must take the single-call
   path under *either* policy; assert that, since it is now reachable two ways.
4. **The cost, in RHS evaluations** (§5 note 14 — counts are the measure), per object and summed
   over the grid, QCD only, $T_k$ before and after and $G_k$ before and after. **Then, additionally
   and secondarily, in seconds per object**, because the whole decision this prompt enacts turned
   on the sector's object count rather than its per-object cost, and the seconds are what make that
   legible: quote best-of-$N$ single-core wall time per object for each sector under its own
   policy, say what $N$ was, and state plainly that the counts are the reproducible measure and the
   seconds are for scoping only.
5. **How far the $T_k$ answer moves on QCD, at the production tolerances.** The policy change moves
   computed $T_k$ values on that model a second time, after prompt 18 already moved them. Say by
   how much, in the same envelope-relative measure as §9.4, and state the consequence for the
   datastore explicitly — it compounds with `[18-numeric-solver-not-in-lookup-key]`, which is still
   open and still the user's.

## 4. Verification and acceptance

- **The acceptance test.** With the new policy, prompt 17 §2.1's criterion holds on `QCDModel`
  $T_k$ at **all 50** wavenumbers: reference drift $\le3.4\times10^{-8}$ of the envelope. If it
  does not, **stop and say so with the measurement**; do not tighten a tolerance to reach it and do
  not move `BREAK_POINT_STANDOFF` to reach it without saying what it does to §9.6's figures and
  asking.
- **$G_k$ and both smooth models are bit-for-bit unchanged**, per §3 items 2 and 3. This is the
  check that the parameter really is a per-caller choice and not a change of behaviour for
  everyone.
- The suite passes: `discover -s ComputeTargets/tests -t .`, **320 before, plus your new cases**;
  none removed, none changed in expectation. In particular `test_background_tau.test_qcd_break_points`
  and `test_cumulative_table.py` are untouched and still pass — the quadrature path is not what
  this prompt changes — and `test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False`
  still passes.
- New tests must not need Ray or a datastore (README §5 rule 7): call the remote through its
  undecorated `_function`, use prompt 01's stand-ins. They must cover at least: the parameter
  selects the number of segments on a stand-in declaring both a jump and a kink; the default
  reproduces the jumps-only result exactly; a cosmology declaring nothing takes the single-call
  path under either policy; `t_eval` equality and `mode="stop"` survive a split with many empty
  segments (§2.3 items 1 and 3); and — read with `ast`, following
  `test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False` — that each
  production call site passes the kind its sector decided on.
- `black --check` clean on every file touched.
- `git diff HEAD~1 --stat` touches nothing outside the allowed list — in particular **not**
  `config/defaults.py`, **not** `CosmologyModels/`, **not** `Datastore/`.

## 5. Log and commit

Close `[17-qcd-reference-not-converged]` only if the acceptance test of §4 passes at all 50
wavenumbers; otherwise narrow it again with the measurement. Either way update
`docs/OPEN_ISSUES.md` in the same commit, including the `prompts/tolerance-convergence` §1
paragraph that currently records the three-wavenumber caveat. Update board row 19 and M22.

"State handed to the next prompt": whether prompt 13 may use a QCD datastore built before this
commit (prompt 18 already said no — say whether anything has changed and what now has to be
regenerated); the per-sector policy as it ended up, since it is now part of
`numeric_with_phase_cut`'s public signature; the cost in evaluations and in seconds; and whether
anything in §2.3 forced a change to the shared driver.

Commit subject, or something equally specific:
`Let each numeric sector choose which declared break points it splits at`.
