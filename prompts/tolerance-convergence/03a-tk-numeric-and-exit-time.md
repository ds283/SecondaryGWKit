# Prompt 03a — `TkNumericIntegration` and `wavenumber_exit_time`

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board items **T5** and **T6** · **Informs:** **D1**, the decoupled tolerance pairs
— it recommends, it does not decide, and **D1 does not close until this prompt reports**
**Depends on:** prompts 01, 02, 02a and 03. Prompt 01's `convergence_reference.py` is the only way
this prompt is allowed to measure; prompt 02's inventory is where both targets came from; prompt 02a
is why each model can be measured at its own anchor; prompt 03 settled the $G_k$ half of D1 and
established how the two tolerance axes behave in a sector where `atol` cannot bind.
**Recommended model:** **Opus** — two targets that fail in different ways, and the second has no
error measure until you choose one.

**Files you may create or touch:**
`docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` — new, the measurement document;
`docs/tolerance-convergence/tk_numeric_exit_sweep.py` — new, the script that regenerates every table
in it;
`ComputeTargets/tests/convergence_reference.py` — **additive only**, see §2.5;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py`; `main.py`; `CosmologyConcepts/wavenumber.py`; any file under
`ComputeTargets/` other than the additive block of §2.5; any `Datastore/` file;
`QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` or `AdaptiveLevin/` (README §0.4).
**Do not change a parameter** (README §5 rule 8) — this prompt reads constants and measures.

**Read first:** README §2 (b), (d), (e), (f), (g) and (i); §3.1's anchor table; §3.3a; **§6.1's
target rule in full**, and **rule 6 especially**, which may be this prompt's answer for one of its
two targets; §5 rules 5, 6 and 9; §1.2's five provenance fields; board items **T5**, **T6** and the
§3 issues `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`,
`[12-tk-numeric-atol-largest-k-excursion]` and `[02a-grid-digest-not-reproducible]`; the module
docstring of `ComputeTargets/tests/convergence_reference.py`, and in it `tk_geometry` (`:799`),
`tk_run` (`:910`) and `exact_z_exit` (`:477`); `_solve_horizon_exit`
(`CosmologyConcepts/wavenumber.py:919`).

**Do not read prompt 03's files before you have read this one and formed your own plan.** You will
need `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` for §4's comparison and you are expected to
use it — but it answers a different sector, and its conclusion is the one thing in this campaign
most likely to be borrowed rather than re-measured. See §4.3.

---

## 1. Why this prompt exists

**Two targets, and they are not alike.** The campaign splits them into one prompt because each is
small — 50 objects per model for `TkNumericIntegration`, one root solve per wavenumber for
`wavenumber_exit_time` — and because neither can move the compute budget the way prompt 03's sector
could. What they have in common is that the record's figures for both are **taken under conditions
that no longer hold**.

**`TkNumericIntegration` was measured under the wrong break-point policy, on a grid that no longer
exists.** `GkTk-remedial` prompt 17 swept it at all fifty wavenumbers on three models and that sweep
is the best measurement in the campaign's record — but it ran on the **version-0** grid and at
`numeric_with_phase_cut`'s module default, `BREAK_POINT_DISCONTINUITY`. **The production
`TkNumericIntegration` call site has passed `BREAK_POINT_ALL` since `GkTk-remedial` prompt 19.** So
the sector's own policy has never been swept, and the figures README §6.2 carries for this row
describe a configuration production does not run. What prompt 17 found under the old policy is
recorded in `[12-tk-numeric-atol-largest-k-excursion]`: 3 / 13 / 8 of 50 wavenumbers on
Radiation / LambdaCDM / QCD above README §6's 3e-6 of the envelope, worst **8.64e-4**. **The `atol`
half of that row is settled** — the user kept `1e-13` on 2026-09-12 and §7 D1 says no prompt here
revisits it — so what is open is the `rtol`, and whether the re-take under the right policy leaves
anything to retune at all.

**`wavenumber_exit_time` has never been measured, and it is not a value — it is a location.**
`xtol = 1e-10`, `rtol = 1e-8` in $u = \log(1+z)$ fix where every source grid begins and where every
horizon-relative cut sits. A misplaced $z_{\rm exit}$ does not make a number slightly wrong; it
moves the grid, and everything computed on the grid with it. That makes the error measure and the
floor genuinely open questions rather than inherited ones, and **README §6.1 rule 6 exists for
exactly this row**: where no floor can be established there is no target, and the note says so in
those words rather than inventing a justification.

**Part of T6 is already measured, by the orchestrator's review of prompt 02a, and you must confirm
rather than inherit it.** `[02a-grid-digest-not-reproducible]` records that Brent stops at
`xtol + rtol*|u|`, so at $u \approx 37.6$ the `rtol = 1e-8` term is **3.8e-7 relative** and the
`xtol = 1e-10` term never binds — which means README §6.2's row and prompt 02's inventory both name
the parameter that does nothing. It also records the shipped anchors as 3.6e-13 (LambdaCDM) and
2.5e-9 (QCD) from a converged re-solve, and `rtol = 1e-14` as a candidate that pins the anchor to
3.8e-13. **Those are measurements from a review, not from a swept target.** Re-take them as part of
your sweep and say whether they stand. If one does not, that is a finding and not an embarrassment.

## 2. What is measured, and on what

### 2.1 Common to both targets

- **Grid:** `SOURCE_GRID_V2`, and **name the anchor** — `PRODUCTION_Z_INIT_LAMBDACDM` for
  LambdaCDM, `PRODUCTION_Z_INIT_QCD` for QCD, both in `ComputeTargets/tests/wkb_reference.py:104`
  (prompt 02a; board standing note 18). A figure without its grid generation is not comparable
  (§5 rule 6) and, since 02a, one without its anchor is not either.
- **`RadiationModel` has no published anchor constant, and prompt 03 already chose one.** Its log
  deviation 6 built the control like the two production anchors and recorded it:
  `horizon_exit_z(model, 3e8/Mpc, -5) = 44523947729.772957`, **2306 samples**, digest
  **`3bef2c06`**. **Use that one.** A second radiation control, built the same way but landing
  elsewhere, would make your figures incomparable with prompt 03's for no gain; if you cannot
  reproduce that anchor and digest on this tree, stop and say so, because it means something under
  the grid construction has moved.
- **Models:** `RadiationModel`, `LambdaCDMModel`, `QCDModel`. **All fifty production wavenumbers**,
  `PRODUCTION_K_GRID_INV_MPC`, for both targets. §6.1 rule 2 forbids scoring a target at a
  representative $k$.
- **The radiation column is an oracle, not a self-convergence.** README §3.1's anchor table gives a
  closed form for both of this prompt's quantities on `RadiationModel`: `analytic_T` /
  `analytic_Tprime` for the transfer function, and `exact_z_exit` — $1 + z = k/(H_0 e^{N})$ — for
  the horizon crossing. Score the radiation column against them through `anchor_error`, and use it
  to calibrate what the self-convergence machinery reports on the other two models. A
  self-convergence drift quoted for a quantity in that table **without the oracle error beside it**
  has not been calibrated (§5 rule 5).

### 2.2 `TkNumericIntegration` — the sector, its geometry, its policy

- **Error measure:** envelope-relative, README §6's definition. Take it through
  `sector_errors("Tk", …)` and `sector_error_measure`; do not write your own.
- **Geometry:** `tk_geometry` — the **source** grid from five e-folds outside the horizon,
  truncated below at `0.85 z_e6`, with the `(z_e3, z_e6)` stop window. This is `main.py`'s
  `build_Tk_numeric_work` geometry (`main.py:1197-1211`) and you must not reconstruct it by hand.
- **Break-point policy: `BREAK_POINT_ALL`**, which is what the production call site passes. `tk_run`
  defaults to `BREAK_POINT_DISCONTINUITY` so that `TK-NUMERIC-ATOL-SWEEP.md` §9's entry point still
  reproduces §9; **you must pass the policy explicitly and say in the document that you did.** A
  reader coming from prompt 17 or from `tk_numeric_atol_sweep.py` will otherwise assume the default.
- **One object per $k$.** There is no second axis here: `tk_geometry` takes no source redshift.
  **Do not borrow prompt 03's `z_source` language.** In particular the words "bound" and "least
  favourable" belong to a check that failed in the $G_k$ sector
  (`[03-outermost-z-source-is-not-the-least-favourable]`) and have no counterpart here; fifty runs
  per model **are** the sector, and you should say so plainly rather than by analogy.

### 2.3 `wavenumber_exit_time` — through the solver, never through the store

- **Call `_solve_horizon_exit` directly**, with the cosmology, the wavenumber and the offset, at the
  `atol`/`rtol` you are sweeping. **Never go through `wavenumber_exit_time` or the datastore.** Its
  lookup filters `stored.log10_tol − requested <= DEFAULT_FLOAT_PRECISION` and orders descending, so
  it returns the **loosest row at least as tight as the request**: a loosened sweep point would
  silently reuse a tighter stored row and the object would report the stored pair, not the one you
  asked for (`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`). A sweep taken through the
  store measures the store.
- **Measure in $u = \log(1+z)$, which is where the root lives**, and report the $z$ figures
  separately and as derived. `_solve_horizon_exit` returns `exp(log_z_root) - 1.0`, and at the
  production anchors ($u \approx 37.6$, $z \sim 10^{16}$) the recovery of $z$ from $u$ is
  irreducibly lossy — `CLAUDE.md`'s redshift-arithmetic note gives the granularity. **State how much
  of the root's own precision survives that conversion.** If the conversion dominates the root
  tolerance over part of the production range, that is a result about the parameter and belongs in
  the document in those words.
- **The offsets production actually asks for.** `find_horizon_exit_time` solves at `offset_subh = 0`
  and then at each of the `subh_efolds` and `suph_efolds` lists. Sweep at least crossing itself,
  the outermost suph offset the source grid anchors on (`z_exit_suph_e5`) and the subh offset the
  $G_k$ numeric region stops at (`z_exit_subh_e4`), because those three are the ones a grid is
  built from.
- **`DEFAULT_HEXIT_TOLERANCE` is a post-hoc guard, not a tolerance you are sweeping.**
  `_solve_horizon_exit` raises if $|q_{\rm root}|$ exceeds it after the solve
  (`CosmologyConcepts/wavenumber.py:884` for the constant, `:1025` for the check). Record where each
  swept setting sits relative to that guard; do not change it.

### 2.4 What "accuracy" means for a location, and what bounds it

`TkNumericIntegration`'s error is an error in a value and README §6 defines it. **`wavenumber_exit_time`'s
is not**, and choosing its measure is part of this prompt's work rather than a preliminary to it.
Report at least:

- the residual $|q|$ at the returned root, which is what the code itself checks;
- the displacement in $u$ from a converged re-solve, and from `exact_z_exit` on the radiation
  control where truth is available;
- and **what that displacement does to a grid** — which is the only thing that makes it an accuracy
  rather than a curiosity. The source grid is built from the anchor, so the question §6.1 rule 6
  asks is what displacement the grid can absorb before something downstream can tell.

**You are not required to establish that bound, and you may not invent one.** If the record and the
tree do not fix what displacement is tolerable, say so in README §6.1 rule 6's words — *no floor
could be established, therefore no target* — and leave the parameter where it is with the
provenance note recording *unestablished*. That is a complete answer to T6 and the rule exists to
make it one. What you may not do is pick a number because a recommendation looks more finished than
a finding.

### 2.5 The facility, additively if at all

`tk_run` already takes `break_point_kind`, and `exact_z_exit` already exists, so **T5 should need no
change to `ComputeTargets/tests/convergence_reference.py` at all**. T6 may need a helper, since
nothing in the facility drives `_solve_horizon_exit`. If it does, you may **add** one. **No existing
line of that file may change**, and prompts 01's and 03's published figures must reproduce
bit-identically afterwards. If you cannot do it additively, stop and say so rather than editing.

## 3. The references, and the rule that no figure travels without its drift

**On `RadiationModel` you have the truth for both targets** (§2.1). Use it, and report the oracle
error beside every self-convergence figure.

**On `LambdaCDMModel` and `QCDModel`**, build the `Tk` reference with `converged_reference` and
obtain its drift through `reference_drift`, which will not return a number without the verdict
attached. `CRITERION_RATIO = 10.0`. For the exit-time sweep the "reference" is a converged re-solve
— `[02a-grid-digest-not-reproducible]` used `xtol=1e-300, rtol=1e-14`, which is what `_solve_T_z`
already uses — and the same rule applies to it: **no error is reported anywhere in this prompt's
document or log without its reference's drift beside it** (§5 rule 5), and no conclusion is drawn
from a signal that does not exceed it.

**Respect SciPy's clamp.** `rtol` is clamped from below at `100 eps = 2.220446049250313e-14`
(`SCIPY_RTOL_FLOOR`); `TolerancePair.rtol_step_is_effective` will tell you when a step is not really
applied. A step across the floor measures nothing and must not be reported. Note that this bites
*differently* for the two targets: for `TkNumericIntegration` it bounds the candidate axis and the
reference below it, as it did in the $G_k$ sector; for `_solve_horizon_exit` the relevant stopping
rule is Brent's `xtol + rtol*|u|` and the `xtol` term is a **separate** absolute floor in $u$ — work
out which term binds at each setting rather than assuming it is the same one throughout.

**If a reference fails to converge at any $(model, k)$, that is a stop.** Continuing past it is the
error `GkTk-remedial` prompt 17 made and the reason this campaign exists. Report which $(model, k)$
and what the drift was; do not drop the wavenumber and do not loosen the criterion.

## 4. The matrix, and what may and may not be carried over from prompt 03

### 4.1 `TkNumericIntegration`

**The anchor** is production: `atol = DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`,
`rtol = DEFAULT_REL_TOLERANCE = 1e-8`.

**The `rtol` axis, at full grid coverage**, is the measurement this prompt exists to take: hold
`atol = 1e-13`, move `rtol` by decades through the anchor, all fifty $k$, all three models, under
`BREAK_POINT_ALL`. Refine by half-decades around any crossing a decade is too coarse to place.

**The `atol` axis is not reopened, and a small number of off-axis points is still required.** D1
settled `atol = 1e-13`'s **value** and §7 D1 says no prompt here revisits it; measuring whether it
*binds* is a different question and the `rtol` recommendation is not separable without the answer.
So take enough off-axis points — one decade either side of `1e-13` at two or three `rtol` settings,
at full $k$ coverage or a stated subset with its reason — to say whether the two axes interact in
this sector. **Do not extend this into a full second axis** and do not recommend an `atol`.

**Note what is different here from the sector prompt 03 swept, and let the measurement say whether
it matters.** $T$ starts at 1 and decays; $G$ near the hand-over is $10^{12}$ to $10^{18}$ in
`Mpc_units`. README §2 (e)'s magnitude argument is therefore not the same argument in this sector,
and `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` was separated from the shared constant in the first place
because something bound. Measure it.

**Report, per $(model, k, atol, rtol)$:** the **maximum**, **second-largest** and **median**
envelope-relative error, **the location of the maximum** ($z$, and where it sits relative to the
hand-over and horizon crossing), and the **evaluation count**. Per $(model, atol, rtol)$: the
distribution over the fifty wavenumbers, not just its centre. Carry the count of wavenumbers above
README §6's **3e-6** of the envelope at each setting, because that is the statistic
`[12-tk-numeric-atol-largest-k-excursion]` is written in and the one a reader will compare.

### 4.2 `wavenumber_exit_time`

A matrix in `xtol` and `rtol` at all fifty wavenumbers and all three models, at the offsets §2.3
names. The solves are cheap, so **take both axes properly**; the reason the diagonal was a problem
in the $G_k$ sector was never runtime. Include settings that make the `xtol` term bind if any do,
since establishing that it never does over the production range is one of this prompt's answers and
an inference is not a measurement.

### 4.3 What may be carried over from prompt 03, and what may not

`GK-NUMERIC-SWEEP.md` is a measurement of a **different sector** under a **different break-point
policy** with a **different consumer**. You may and should cite it for comparison — §6 question 4
asks for exactly that — and you may reuse its method, its script's structure and its reference pair.

**You may not carry over its conclusions.** "`atol` is inert", "the axes separate cleanly at 10.1
per decade" and "the floor dominates so the answer is `unchanged`" are findings about
`GkNumericIntegration`, and each of the three has a specific reason not to transfer: the magnitude
argument (§4.1), the break-point policy, and a consumer that is a spline of $G$ rather than the
initial-condition truncation of $T$. **A figure in your document that traces to prompt 03's tables
rather than to your own script is a defect**, and §8's acceptance 6 is how it will be caught.

## 5. The floors

**`TkNumericIntegration`: the initial condition, 2.52e-6 of the envelope**, $k$-independent,
confirmed against the exact $T$ at all fifty wavenumbers (`GkTk-remedial` prompt 17 §8; README §2
(f)). That figure was taken on the version-0 grid under the other break-point policy. **Confirm it
on this tree** — it is an initial-condition truncation rather than a property of the grid, so it
ought to survive, and a floor that ought to survive and does not is the more interesting outcome.
Record both figures with their citations (§6.1 rule 5), and if they disagree say so plainly rather
than reconciling them by adjusting the method until they agree.

**`wavenumber_exit_time`: see §2.4.** Establish the bound or state in §6.1 rule 6's words that none
could be established.

**A measured accuracy below a floor you have just measured is a stop** (README §2 (f), §6.1
rule 5). It is an arithmetic error in the measurement and never a discovery.

## 6. The questions this prompt must answer

Answer each in the document, in these words, with the measurement beside it:

1. **What does `BREAK_POINT_ALL` change?** Prompt 17's figures for this sector were taken under
   `BREAK_POINT_DISCONTINUITY` on the version-0 grid. Re-taken under the production policy on the
   version-2 grid, do `[12-tk-numeric-atol-largest-k-excursion]`'s 3 / 13 / 8 wavenumbers above
   3e-6 and its worst 8.64e-4 stand, move, or disappear? Separate the two changes if you can, and
   say so if you cannot.
2. **Is the error in the $T_k$ sector set by `rtol`, and does `atol = 1e-13` bind?** §2 (d)'s prior,
   tested in the sector where §2 (e)'s magnitude argument does *not* obviously apply.
3. **Apply §6.1's target rule to `TkNumericIntegration`.** The target is the **loosest** `rtol`
   whose **maximum** error over the whole grid on all three models is at or below the floor — or it
   is **`unchanged`**, in that word, with the dominating factor recorded (rule 4). Tightest-is-safest
   is an error of the same kind as missing the floor (rule 3).
4. **What does it cost**, at the recommended setting and **one step either side**, in right-hand-side
   evaluations **times the sector's object count** (50 per model, and say so — this sector is three
   orders smaller than the one prompt 03 measured, which is why a decade here is not the decision a
   decade there would have been). Counts, never wall time (§2 (i)).
5. **What binds `_solve_horizon_exit`, and to what?** Which term of Brent's `xtol + rtol*|u|` is
   active at the production setting and over what part of the production range; what the residual
   and the displacement from a converged re-solve are; and how much of that survives the $u \to z$
   recovery.
6. **Apply §6.1's target rule to `wavenumber_exit_time`, or rule 6.** A pair, with the floor it
   clears — or the words of rule 6, that no floor could be established and therefore there is no
   target. Both are complete answers. **Neither may be a number chosen to look finished.**
7. **What would the decoupled pairs be**, with the evidence for each half? **Recommend; do not
   decide** — D1 is the user's, and accepting prompt 03's $G_k$ half on 2026-09-17 did not close it.

## 7. Out of scope, and where each piece went

- **`GkNumericIntegration`** — board item **T4**, prompt 03's, measured and **accepted by the user
  under D1 on 2026-09-17**. Do not re-measure it, do not re-recommend it, and do not reopen it
  because your sector behaves differently. §4.3 says what you may take from its document.
- **The consumer-spline floor near the hand-over** — measured by prompt 03 and **assigned out of
  this campaign** on the same decision, to the hand-over campaign
  (`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`,
  `docs/OPEN_ISSUES.md` §1.1). The source grid's spacing is not this prompt's business even where
  your measurement touches it.
- **The grid digest's reproducibility** — `[02a-grid-digest-not-reproducible]` part (ii) is
  **prompt 05's**: one design tolerance applied to both the redshift row match and the digest
  quantisation. You supply the prerequisite measurement (§2.3, §6 question 5) and stop there. **Do
  not change the digest, the quantisation, `SOURCE_GRID_MIN_SEPARATION` or
  `DEFAULT_REDSHIFT_RELATIVE_PRECISION`, and do not reorder the density guard's mask.**
- **Every parameter's value.** Prompt 05, after D1. You recommend with evidence and stop.
- **The source grid, the band, `RESIDUAL_WKB_REGION_MARGIN`, `_solve_T_z` and `main.py`'s anchors.**
  Prompt 02a's stops stand unchanged. If the sweep appears to need one of them moved, that is a
  finding for §3 of the board, not a change.
- **`find_phase_extremum`'s `xtol=1e-6, rtol=1e-4`** (`LiouvilleGreen/integration_tools.py:92`).
  It belongs to the hand-over campaign (`[11-stop-point-root-tolerance]`) and is not retuned here,
  even though it is the other unmeasured root solve in the tree.
- **The orders and `RESIDUAL_WKB_REGION_MARGIN`** — board item **T7**, prompt 04's.

## 8. Acceptance

1. **Coverage.** All fifty production wavenumbers, three models, both targets, at the production
   settings, each figure carrying its reference's drift verdict, its grid generation and its anchor.
2. **`BREAK_POINT_ALL` used and stated** for every `TkNumericIntegration` figure, with the
   difference from prompt 17's policy quantified rather than asserted (§6 question 1).
3. **The `rtol` axis at full grid coverage**, plus §4.1's off-axis points; the `rtol` floor
   respected and no step reported across it.
4. **The exit-time matrix taken through `_solve_horizon_exit`**, never through the datastore, at the
   three offsets §2.3 names, with the binding term of Brent's criterion identified by measurement.
5. **Both floors addressed**: the $T_k$ initial condition re-confirmed on this tree with the
   inherited figure beside it, and the exit-time bound either established or declared unestablished
   in §6.1 rule 6's words.
6. **`tk_numeric_exit_sweep.py` regenerates every table** in `TK-NUMERIC-AND-EXIT-TIME.md`. A table
   in the document that the script cannot reproduce is an unfinished prompt, and a figure traceable
   to prompt 03's script rather than yours is a defect (§4.3).
7. **Prompts 01's and 03's published figures are unmoved**, if you touched `convergence_reference.py`
   at all: `ComputeTargets/tests/test_convergence_reference.py` reads its previous count, and
   `docs/gktk-remedial/tk_numeric_atol_sweep.py` still reproduces.
8. **Suites green** at the board header's counts, allowing for `[computetargets-suite-flake]`'s
   wall-clock assertion.
9. **The five provenance fields** (§1.2, §5 rule 9) for everything you recommend, in the log's
   "State handed to the next prompt" — including, for a parameter you decline to set, the words
   rule 6 requires. A number shipped without them is an unfinished prompt.

## 9. Stop conditions

Stop and report rather than working around any of these:

- **A reported accuracy below a floor you have just measured** (§2 (f), §6.1 rule 5).
- **A reference does not converge** at any $(model, k)$, for either target — prompt 17's error, and
  not a wavenumber to drop.
- **A recommendation that would move a sector's total cost by more than a factor of two** (§4.3).
  Unlikely at 50 objects per model, and still the user's decision rather than yours.
- **A moved grid digest**, for either published grid, at any point and for any reason. A stop even
  if the new grid looks better.
- **You need to change a parameter, or edit a file outside the list above** — including any
  non-additive change to `convergence_reference.py`, and including anything in
  `CosmologyConcepts/wavenumber.py`.
- **You cannot state the grid generation *and* the anchor** for any figure you intend to publish.
- **The exit-time sweep cannot be taken without the datastore.** It can be; if you conclude
  otherwise, say so rather than going through the store (§2.3).

## 10. The log

`logs/03a-tk-numeric-and-exit-time.md`, on `GkTk-remedial` §5.1's template, classifying every
deviation. Beyond the template it must carry:

- the drift verdict tables, per model and per target, with the radiation oracle calibration beside
  the self-convergence figures;
- the $T_k$ initial-condition floor as re-confirmed on this tree, with the inherited figure beside
  it;
- the answers to §6's seven questions, each in one paragraph, with the number in it;
- what `BREAK_POINT_ALL` changed, stated as a comparison against prompt 17 rather than as a fresh
  measurement standing alone;
- and, under **"State handed to the next prompt"**, the five provenance fields of §1.2 for each
  recommended parameter — or rule 6's words for one that is declined — and, for **prompt 05** and
  **D1**, what the user is being asked to accept, in the form a decision can be taken from rather
  than a summary of the measurement. **Say explicitly that accepting it closes D1**, since the
  $G_k$ half was accepted on 2026-09-17 and these are the remaining halves.
