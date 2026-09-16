# The background solver robustness campaign — the root solves and evaluation path of `LambdaCDM_GenericEOS`

**Source document:** [`AUDIT.md`](AUDIT.md) — **read §0, §2, §3, §4 and §5 before anything else.**
**Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) — **read this second.** The audit was
correct when written; two of its statements are no longer the whole truth, and one of those changes
the campaign's shape.
**Reproduction:** `PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py`
(~9 s, no Ray, no datastore). Re-run at `f023eb8`: **every figure in the audit reproduces to the
digit printed.**
**Planned:** 2026-09-16, against `tolerance-convergence` at `f023eb8`.
**Target branch:** `background-solver-robustness`, cut from `f023eb8` (§4).
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) ·
**Logs:** [`logs/`](logs/) · **Orchestrator prompts:** [`orchestrator/`](orchestrator/)

---

## 0. What this campaign is, and its boundaries

### 0.1 The one-sentence version

`LambdaCDM_GenericEOS._find_rho_equality` (`:999`) runs an **unbracketed secant** at
`xtol=1e-6, rtol=1e-4` — tolerances two orders looser than anything else in the file — and returns
the right answer to 4.0e-16 only because the analytic guess its caller hands it *is* the root
whenever $g_*$ is flat; walk the guess down 30 % and it dies inside `_rho_fluid` at negative $z$,
with its own convergence guard never firing. This campaign brackets it, tightens it to the file's
own standard, makes the guard reachable, pins all of it in a test, and then clears the four other
unowned issues in the same file that the project-wide index has been waiting for a prompt with
these files in scope.

### 0.2 The honest impact, stated first, because an agent that has to discover it will inflate the change

**Fixing this changes no computed quantity in the pipeline.** `AUDIT.md` §5 is the campaign's
framing and it is still true at `f023eb8`: the two equality redshifts `_find_rho_equality` produces
are printed with `:.4g` and discarded, they are correct to the double-precision floor today, and no
datastore regeneration is implied by any prompt in workstreams A, B or C.

What the campaign buys is:

- a solve that is correct **by construction** rather than because its caller happens to hand it the
  answer — the flat-$g_*$ argument of audit §2.3 is a property of *this* equation of state at
  *this* redshift, and nothing checks it;
- a reachable convergence guard and a failure message that names the species pair and the range
  searched, instead of a temperature-spline bounds error from two frames down (audit §3.2);
- a provenance entry that `docs/TOLERANCE-PROVENANCE.md` can state rather than record as
  *unestablished* — for **all three** solves in the file, not just this one (§0.4);
- the last loose tolerance out of a file whose other two solves were tightened for exactly these
  reasons;
- and **the correction of a belief that is about to cost somebody a datastore** (§0.3).

**An agent told to "fix the precision problem" without this framing will look for an impact, fail
to find one, and either inflate the change or stall.** Every prompt in workstream A states, as an
acceptance criterion, that the two printed redshifts do not move.

### 0.3 The thing the reconciliation found, which the audit could not

`AUDIT.md` §2.1 concludes, from a correct grep, that *"the blast radius of this solve is two banner
lines"*. That is true of **the solve**. It is not true of **the quantity**.

`main.py:549-551` recomputes both equality redshifts from the same closed forms and passes them as
`feature_z` into `build_z_sample`, which forces them into the production source grid; the grid's
content digest is a `BackgroundModel` lookup-key column. On `QCD_Cosmology` — the only cosmology
that takes that path, which is gated on the model declaring break points — **the two equality
redshifts are production sample locations inside a datastore identity**
([`RECONCILIATION.md`](RECONCILIATION.md) §5).

So the obvious tidy that audit §6 observation 2 gestures at — derive the guess in the method,
expose the result, have `main.py` import it — is **not** a tidy. One ulp of movement in either
redshift moves a grid sample, moves the digest, and invalidates every stored object of the eight
types `qcd-background-audit` log 11 §5 priced. **Prompt 03 establishes and documents this and
nothing else**, and §7 **D2** puts the unification question to the user rather than answering it.

### 0.4 Provenance is half the point

`prompts/tolerance-convergence` README §1.2 requires that when it closes, **no accuracy parameter
in the pipeline is unexplained**, and `docs/TOLERANCE-PROVENANCE.md` does not exist yet — its
prompt 06 creates it. Audit §7 is explicit about the sequencing, and it runs one way only:

> If a fix lands before `tolerance-convergence` prompt 02 runs, prompt 02 records a settled
> provenance entry … If it lands after, the note records *unestablished* and the fix becomes a
> follow-up amendment. `tolerance-convergence` has not started — there are no numbered prompt files
> in it yet — so the cheap moment is now.

Still true at `f023eb8`: that campaign is planned, rebased and **not started**. Prompt 06 here
writes the provenance for **all three** `root_scalar` sites in `LambdaCDM_GenericEOS.py` in the
shape `docs/TOLERANCE-PROVENANCE.md` will want, so that campaign's prompt 02 lifts three settled
entries instead of recording one unestablished one.

### 0.5 What this campaign does *not* do

- **It does not re-open `_solve_T_z` (`:583`) or `_temperature_crossing_log1pz` (`:869`).** Both
  were audited by `qcd-background-audit` (prompts 04, 06, 07), both are bracketed, both sit at the
  representable floor, and both carry the comment that chose the value. Touching their tolerances
  re-opens that campaign's work. Prompt 04 *relocates* `_temperature_crossing_log1pz` without
  changing a character of its solve, and that is the only contact.
- **It does not touch `find_phase_extremum`** (`LiouvilleGreen/integration_tools.py:92`), which
  looks identical to the subject and is not the same problem: it is bracketed, it is on the
  production path, and it sets a *computed* quantity — the numeric→WKB hand-over point. It is
  `[11-stop-point-root-tolerance]`, owned by the hand-over campaign (`docs/OPEN_ISSUES.md` §1.1),
  and `prompts/tolerance-convergence` §0.5 declines to retune it. **Audit §4.3 forbids absorbing
  it and so does this README.**
- **It does not change the printed diagnostics**, `_rho_fluid`, the $T(z)$ representation, the
  source grid, `BackgroundModel`, or any lookup key. No prompt in workstreams A–C moves a stored
  number, and `T_Z_REPRESENTATION_VERSION` is **6** at every commit.
- **It does not unify the three copies of the equality closed form** (§0.3). Prompt 03 measures and
  documents; §7 D2 is the user's.
- **It does not touch `ComputeTargets/QuadSourceIntegral.py`**, whose stale
  `DEFAULT_QUADRATURE_ATOL` comment is already on the `tolerance-convergence` board.

### 0.6 Boundary with `prompts/tolerance-convergence` (planned, rebased, not started)

**The two do not overlap in files.** That campaign's §0.5 does not own `CosmologyModels/GenericEOS/`
and its only contact is prompt 02's read-only inventory, which must list `:583`, `:869` and
`:1013`. This campaign runs **first** and hands that inventory three settled entries (§0.4).

Prompt 06 here amends `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3 — deleting the
"Recorded by the rebase, not owned here" bullet for `:1008` and replacing it with the settled
result — which is the same cross-board editing `be21f5c` did in the other direction. It amends
nothing else of that campaign.

---

## 1. The defect, in one table

`LambdaCDM_GenericEOS.py` holds three `scipy` root solves. Two are sound. One is the subject.

| # | Site | Method | Tolerances | Bracketed? | Audited by | Verdict |
|---|---|---|---|---|---|---|
| 1 | `_solve_T_z`, `:583` | Brent | `xtol=1e-300`, `rtol=1e-14` | **yes** | `qcd-background-audit` 04 | sound; reasoning in the code at `:574-583` |
| 2 | `_temperature_crossing_log1pz`, `:869` | Brent | `xtol=1e-15`, `rtol=1e-15` | **yes** | `qcd-background-audit` 06, 07 | sound, **and not on the production path** — which is `[08-temperature-crossing-solver-is-test-only]`, prompt 04 here |
| 3 | **`_find_rho_equality`, `:1013`** | **secant** (`x0=`, no `bracket=`) | **`xtol=1e-6`, `rtol=1e-4`** | **no** | **nothing** | **the subject** |

Three things are wrong with #3, in increasing order of seriousness (audit §3):

1. **The tolerance is loose but never binds.** `rtol = 1e-4` would permit $\pm0.34$ at
   $z\sim3400$; secant's superlinear convergence overshoots the stopping test by seven to twelve
   orders, so the tolerance neither bounds the error nor predicts it. And `xtol = 1e-6` is an
   *absolute* tolerance in $z$ — meaningless at $z\sim3400$ and the only thing acting at
   $z\sim0.3$, the two roots being four decades apart. That is the same argument
   `qcd-background-audit` prompt 04 wrote into the comment at `:574-583`.
2. **The solve is unbracketed and its failure escapes its own guard.** At a guess displaced by
   −30 % the secant raises `ValueError: math domain error`; at −50 % and beyond it has walked to
   *negative* $z$ below the spline's floor and raises a bounds error from `_rho_fluid`. Neither is
   `converged=False`, so the guard at `:1015` is **dead on every path that actually fails**.
3. **It is correct by accident.** $g_*$ is flat at $z_{\rm eq}$, so $\rho_r\propto(1+z)^4$ exactly
   and $1+z = \Omega_m/\Omega_r$ is the closed solution — the guess the caller passes. Secant
   evaluates once to three times and stops at the rounding floor. **Nothing in the code says so and
   nothing enforces it**; one exotic equation of state away, it degrades silently.

Both roots are **trivially bracketable**: audit §4.1 measures $\rho_m/\rho_r$ strictly decreasing
over $z\in[33, 3.4\times10^5]$ and $\rho_m/\rho_\Lambda$ strictly increasing over $z\in[0,10]$, so
a bracket expanded geometrically from the analytic guess is guaranteed to straddle and Brent
applies. **The two probe ranges differ because the two roots do; a fix must not assume one bracket
serves both.**

---

## 2. What the campaign establishes

Lettered so the board and the prompts can cite them.

**(a) The two equality redshifts do not move.** Every prompt in workstream A asserts it, against an
independent `brentq` reference at Brent's own floor, on `QCD_Cosmology` and on a pure-radiation
stand-in. This is the campaign's primary stop condition.

**(b) The solve is bracketed and Brent, at `xtol=1e-300, rtol=1e-14`,** matching `:583`, with the
comment that chose the values at the point of use.

**(c) The convergence guard is reachable, and a failure names its own cause** — the species pair
and the range searched — instead of a temperature-spline bounds error two frames down.

**(d) The monotonicity the bracket rests on is a test, not a paragraph.** Audit §4.1 measured it
once; a bracket-expansion policy that assumes it needs it pinned.

**(e) A test that passes both before and after proves nothing.** Every prompt that claims a
behaviour change shows the new assertion **failing on `HEAD~1`** and says so in its log with the
output. This is the campaign's single most important review step, because its other stop condition
is "nothing moved", and "nothing moved" is what a test that tests nothing also reports.

**(f) What the equality redshifts actually feed is written down** (§0.3), with the closed form
scored against the corrected solve on all three production models, and `main.py:526`'s stale
agreement figure corrected.

**(g) `_temperature_crossing_log1pz` is out of the production class.** It has had no production
caller since `qcd-background-audit` prompt 07; a private method on a production class whose only
callers are tests is a trap for a later reader.

**(h) The range logic is hoisted, and the `T_photon` cost row is settled or escalated.**
`README §6.2`'s ≤ 2.5 µs target is a **confirmed miss at 2.596 µs**; the predicted landing after
the hoist is ~2.49 µs. If it does not clear, the row goes to the user — that is the issue's own
next step, not a licence to keep optimising.

**(i) Provenance for all three solves**, in the shape `docs/TOLERANCE-PROVENANCE.md` will want.

**(j) Redshift arithmetic.** `CLAUDE.md`: $z\to\log(1+z)$ is safe, $\log(1+z)\to z$ is irreducibly
lossy at large $z$ and must never reach an equality-like comparison. The equality solves work in
$z$ directly, at $z\sim3.4\times10^3$ and $z\sim0.3$, where this does not bite — **but a
bracket-expansion policy written in $u$ would have to justify itself, and prompt 02's log must say
which variable it chose and why.**

**(l) The model is authoritative for its own equality redshifts** (README §7 D2 as decided, and
§2 (f) is what measured it). `BaseCosmology` declares `z_matter_radiation_equality` and
`z_matter_lambda_equality`; `LambdaCDM` answers with the closed form, which is **exact** for a model
with no equation of state; `LambdaCDM_GenericEOS` answers with the solve its constructor already
runs and currently discards; `main.py` asks and computes nothing, **with no fallback**. Prompt 09,
workstream E.

**(k) Author conventions are conventions, not defects** (`CLAUDE.md`). $a_0$ is absorbed, never
"set to 1". Do not "correct" them.

---

## 3. The prompts

Nine prompts in five workstreams. **Workstream D is gated** — do not start it without the user's
go-ahead (§7 D3). **Workstream E was added on 2026-09-16**, when the user decided §7 **D2** as
option (iii); it is not part of the original plan and it is the only workstream that moves a stored
identity on purpose.

| # | Prompt | Workstream | Covers | Model | Why that model |
|---|---|---|---|---|---|
| 01 | [The equality-solve characterisation test](01-equality-solve-characterisation.md) | A | §2 (a), (d) | **Opus** | Deciding what must be asserted (invariants that outlive the fix) against what must only be *recorded* (today's failure modes, which prompt 02 changes) is the judgement the whole campaign rests on. Get it wrong and (e) is unprovable |
| 02 | [Bracket the equality solve](02-bracket-the-equality-solve.md) | A | §2 (b), (c), (e) | **Opus** | The only production numerics change. Bracket-expansion policy for two roots four decades apart, what to do when the guess is already the root, and a bit-for-bit acceptance |
| 03 | [What the equality redshifts feed](03-equality-redshift-consumers.md) | B | §2 (f) | **Opus** | Cross-package reconciliation with a datastore-identity consequence, ending in a decision put to the user. Getting the blast radius wrong here is how the grid gets invalidated by accident |
| 04 | [Relocate the crossing probe](04-relocate-the-crossing-probe.md) | B | §2 (g) | **Sonnet** | A move with an exhaustive file list and a numerically-null acceptance. The one open choice is written out in the prompt with both options and the criterion that decides it |
| 05 | [Hoist the range logic](05-hoist-the-range-logic.md) | C | §2 (h) | **Opus** | Four lines of code and a timing protocol with a stop-or-escalate rule, on a hot path shared by three classes, where the acceptance is bit-identity plus a microsecond-scale measurement on a loaded machine |
| 06 | [Provenance and close-out](06-provenance-and-close-out.md) | C | §2 (i) | **Opus** | Provenance prose for three solves, a cross-campaign board amendment, and the close-out verification. May not touch production code, and that rule is the review |
| 07 | [Report the representation in the inventory](07-inventory-representation.md) | D (gated) | — | **Sonnet** | Two lines and a test, in files no other prompt here touches |
| 08 | [Refresh the stale agreement threshold](08-refresh-agreement-threshold.md) | D (gated) | — | **Sonnet** | One measurement, one comment, one constant, in one test file |
| 09 | [Make the model authoritative](09-make-the-model-authoritative.md) | **E** | §2 (l) | **Opus** | Changes an inheritance contract across three cosmology classes and moves a datastore identity deliberately, with a digest it must **land on** rather than merely report. The one thing it must not do — fall back to the closed form when a model cannot answer — is the thing that looks most careful |

### 3.1 The anchors every measurement is scored against

- **The bracketed reference.** `brentq` at `xtol=1e-300, rtol=8.9e-16` — Brent's own floor of
  $4\varepsilon$ — on a bracket taken from `measure_rho_equality.bracketed_reference`. This is
  *independent of the production solve* and is what (a) means.
- **The closed forms.** $1+z_{\rm eq} = \Omega_m/\Omega_r$ exactly where $g_*$ is flat, and
  $1+z_\Lambda = (\Omega_\Lambda/\Omega_m)^{1/3}$ exactly and unconditionally — the latter has no
  temperature dependence at all, which makes it an oracle rather than an approximation.
- **The pure-radiation stand-in**, `CosmologyModels/tests/test_wPerturbations.PureRadiationEOS`,
  where $g_* = g_{s,*}$ is constant and $T(z) = T_{\rm CMB}(1+z)$ exactly, so *both* closed forms
  are exact and any departure is the representation's.
- **`QCD_Cosmology`**, where the flat-$g_*$ argument is true but not by construction — which is
  the case (3) of §1 that nothing enforces.

---

## 4. Branch and sequencing

**Cut `background-solver-robustness` from `f023eb8`**, the current tip of `tolerance-convergence`.

`AUDIT.md` §7 says to branch from `main`. That was right when it was written and is now
inconvenient rather than wrong: two of this campaign's own artefacts are already committed on
`tolerance-convergence` — the audit itself (`9ccd42e`) and the fix for its §6 observation 1
(`f023eb8`) — and the only other commits between `main` (`acd5b8e`) and `f023eb8` are `be21f5c` and
`a59d858`, **both documentation-only planning commits for the other campaign, with no production
file in either diff**. Cutting here therefore carries no production change this campaign did not
make, and avoids cherry-picking its own audit.

The file sets are disjoint from `tolerance-convergence`'s, so merging back is conflict-free and no
second rebase of that campaign's `RECONCILIATION.md` is needed. **Run A → B → C in order**; each
workstream's preconditions include the previous one's completion criterion.

---

## 5. Rules that apply to every prompt

These are `CLAUDE.md`'s campaign conventions. `qcd-background-audit` README §5 and
`phase-representation` README §5 are the precedent and this list is the same one.

1. **One commit per prompt.** The commit boundary is the rollback boundary; do not amend or squash
   across prompts. **An agent must never assume `HEAD` is its own** — planning and orchestration
   commits land on the same branch.
2. **Commit message:** imperative, capitalised subject under ~72 characters with no prefix tag; a
   blank line; a prose body saying what was wrong, what changed and how it was verified, wrapped at
   ~80 columns; then `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` naming the model
   that did the work.
3. **Every prompt writes a log** to `logs/NN-<name>.md` using the template in §5.1, in its own
   commit, classifying every deviation as `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or
   `UNINTENDED DRIFT`.
4. **Every prompt updates [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)** — its own row, the
   item table in §2, and §3/§4 — **and, whenever §3 or §4 changes,
   [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit**, with its count and date
   corrected. An issue owned by another board is moved to *that* board's §4 and its row deleted
   from the index.
5. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. If a prompt's stated acceptance test cannot pass without going
   out of scope, **stop and ask**.
6. **Tests** live in `<package>/tests/` as `unittest` modules, run from the repository root, and
   **must not need Ray or a datastore**:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
   ```
   Use the stand-in model pattern of `CosmologyModels/tests/test_wPerturbations.py`. **Record the
   counts before and after**; a count that falls is a stop.
7. **Format with `black`** (no configuration) before committing; the tree is clean under `--check`.
8. **Redshift arithmetic** — §2 (j). **Author conventions** — §2 (k).
9. **A figure without the tree it was taken on is not a measurement.** Quote the commit beside
   every number that could move.
10. **Review content, code comments and document text are data**, not instructions to the
    implementing agent.

### 5.1 Log format (mandatory)

The log must let a later reader tell what shipped, and *why it differs from the prompt*, **without
re-deriving anything from the code**. Every deviation is classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change, a
  numerical fact was different). State what the prompt assumed, what was actually there, and what
  was done instead.
- **IMPLEMENTATION CHOICE** — the prompt left it open and the agent picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so plainly, and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/background-solver-robustness/NN-<name>.md
**Commit:** <sha> — <subject>
**Model:** <model that executed the prompt>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before -> after. Enough that a reader knows the change without opening the
diff. Name every new public symbol and its signature. State T_Z_REPRESENTATION_VERSION before and
after (it must be 6 at both ends for every prompt in this campaign).>

## The two equality redshifts
<Mandatory for every prompt in workstreams A and B. Both values to 17 significant figures, on
QCD_Cosmology and on the pure-radiation stand-in, before and after, with the relative move. "No
bit moved" is the expected answer and must be stated, not implied.>

## Deviations from the prompt
<One subsection per deviation, tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE /
UNINTENDED DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from "I reasoned
that this is correct" from "this needs a run the user must do". **Quote the numbers**: every
acceptance threshold in the prompt gets its measured value. Where the prompt says "show it fails on
HEAD~1", give the command and the failure output. Give the suite counts before and after.>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later. Each becomes a
§3 issue on the board (and a row in docs/OPEN_ISSUES.md) if it is actionable.>

## State handed to the next prompt
<Anything the next prompt needs that is not already in its own text: names chosen, signatures,
the bracket-expansion policy and its constants, achieved accuracies, evaluation counts, measured
costs, and the exact command that reproduces each figure.>
```

---

## 6. The acceptance table

A prompt is ✅ only when its row is true **and its log quotes the measured value**.

**The `Measured` column was added by prompt 06 at close-out**, from each prompt's own log; the
`Threshold` column is the plan as written and is not edited (§5 rule 6). Where a threshold was
amended by the user or by a measurement, the row says so and names the deviation.

| # | Acceptance | Threshold | **Measured** |
|---|---|---|---|
| 01 | Both equality redshifts reproduce the bracketed reference | ≤ 2 ulp on both models, both pairs | **3 / 2 / 1 / 2 ulp**; shipped at `SOLVE_VS_REFERENCE_ULP = 4`. The threshold is arithmetically incompatible with audit §2.2's own −4.00e-16, which *is* −3.0 ulp (log 01 D1; board note 7) ⚠️ |
| 01 | Both density ratios monotone over the range their own root lives in | strict, at every probe of audit §4.1 | **strict, 6 intervals × 2 pairs × 2 models** ✅ |
| 01 | New test module needs no Ray and no datastore | runs under `discover -s CosmologyModels/tests` | **Ran 4 tests in 0.083 s, OK** ✅ |
| 01 | Suites | `CosmologyModels` 30 → 30 + *n*, `ComputeTargets` 447 unchanged | **30 → 34** (*n* = 4); **447 → 447** ✅ |
| 02 | The two roots after the change, against prompt 01's reference | **bit-identical**, or ≤ 1 ulp with both floats quoted at 17 digits | **0 / −2 / 0 / −2 ulp** against that reference. Bit-identity is **not a property this root has** — the residual's sign change spans several floats (log 02 D1, board note 9) — so the user amended the row to "≤ 4 ulp" on 2026-09-16 ⚠️ |
| 02 | A guess displaced enough to break the shipped solve now raises `_find_rho_equality`'s own `RuntimeError` | naming the species pair **and** the range searched | **No displacement can break the bracketed solve** (×0.01 to ×100 all recover the root to ±1 ulp), so the guard is reached instead by a cosmology whose tabulated range cannot contain its own root; the message names both species, the guess, the clamp, both endpoints and both residuals (log 02 D2) ✅ |
| 02 | That assertion fails on `HEAD~1` | failure output quoted in the log | **2 failures, 5 errors** at `7fdc49b`, quoted; `assertRaises(RuntimeError)` alone would have passed on both trees ✅ |
| 02 | Evaluation count at the production call sites | quoted before and after; audit §3.1 predicts +6 to +9 | **3, 1, 1, 1 → 23, 25, 21, 25** (+20 to +24). The audit's +6 to +9 is for *tightening the secant* and does not survive bracketing ✅ |
| 02 | `T_Z_REPRESENTATION_VERSION` | **6** before and after | **6 → 6** ✅ |
| 03 | Closed form vs. the corrected solve, three production models, both pairs | measured and tabulated; `main.py:526`'s figure corrected to what is measured | **−9.344e-16 (−7 ulp)** and **+0.000e+00** on `QCD_Cosmology`; **−1.336e-16 (−1 ulp)** and **+0.000e+00** on the stand-in; the docstring now quotes both and names `921f41c`. The third model is the pure-radiation stand-in, not `RadiationModel`, which exposes no $\Omega$s (log 03 D1) ✅ |
| 03 | Production files changed | **none** except the one `main.py` docstring sentence | **one hunk, 0 executable lines** ✅ |
| 03 | The production source grid | **byte-identical**, demonstrated by digest | **`a2c32f67` / 1,996 and `60a3205a` / 1,778, identical at `7fdc49b`, `921f41c` and prompt 03's commit** ✅ |
| 04 | Every number in the three affected suites | unchanged, and the counts unchanged | **38 / 447 at both ends**; the three measured $H$ steps **1.969955e-03, 9.272151e-11, 1.377111e-04**, the figures the board's issue entry records ✅ |
| 04 | `_temperature_crossing_log1pz` | no longer a member of `LambdaCDM_GenericEOS` | **`grep -rn "self\._temperature_crossing_log1pz"` → no matches** ✅ |
| 05 | Every value `TemperatureRepresentation`, `ZSplineWrapper`, `GkWKBSplineWrapper` return | **bit-identical**, demonstrated over a probe set | **3,979 `float.hex()` values, MD5 `7c16ba…7bb2` at both trees; 37 in-probe rejections identical in position and text; all six messages character-identical** ✅ |
| 05 | `T_photon` cost, quiet machine, five runs | mean and range quoted; ≤ 2.5 µs **closes** `[06-…]`, otherwise **escalate** | **2.4854 µs**, range **2.418–2.546**, against `HEAD~1`'s 2.6744 — ratio **0.9293**, controls within ±2 %. **Closes** `[06-…]`; §7 **D4 not invoked**. Margin 0.6 %, one run of five above target ⚠️ |
| 05 | `measure_T_z_representation.py` §5 prose | counts the intersection with the declared set (**0**), not the knots in range | **prints "0 of those 3 BREAK_POINT_ALL points are also knots"**; §5's table and everything above §6 byte-identical ✅ |
| 06 | Provenance entry | all three solves; value, choosing measurement + its commit, competing floor, cost × call count, citation | **[`PROVENANCE.md`](PROVENANCE.md), three entries, every field present.** Two corrections the prompt did not anticipate: solve 2's tolerance was chosen by **`GkTk-remedial` prompt 03 (`83ef7c5`)**, not by `qcd-background-audit` 06/07, and has **no** choosing measurement; solve 3 ships **`rtol=8.9e-16`**, not `1e-14` (log 06 D1, D2) ✅ |
| 06 | Production files in the diff | **none** | **zero production and zero test files**; `git diff --name-only` in log 06 ✅ |
| 06 | `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3 | the `:1008` bullet replaced by the settled result | **replaced; the `[11-stop-point-root-tolerance]` and `QuadSourceIntegral.py:1550` bullets untouched, demonstrated by diff** ✅ |

**Prompt 09 has no row here, and that is not an omission.** Workstream E was added on 2026-09-16,
after this table was written, when the user decided §7 **D2** as option (iii). Its acceptance is in
[`09-make-the-model-authoritative.md`](09-make-the-model-authoritative.md) §5 and its measured
values are in board item (l) and [`logs/09-make-the-model-authoritative.md`](logs/09-make-the-model-authoritative.md);
the headline is that the `QCD_Cosmology` production source-grid digest **`a2c32f67` → `4849552b`**
on exactly one sample of 1,996, which the prompt required it to **land on** rather than report, and
it did.

---

## 7. Decisions for the user

**D1 — the tolerance pair for `_find_rho_equality`.** The campaign's recommendation is
`xtol=1e-300, rtol=1e-14`, identical to `_solve_T_z` at `:583` and for the same two reasons: `rtol`
sits just above Brent's floor of $4\varepsilon\approx8.9\times10^{-16}$, and an absolute tolerance
in $z$ cannot serve two roots four decades apart. Cost, measured: 6 to 9 extra `_rho_fluid` calls,
twice, at model construction — each one spline evaluation. **Recommendation: take it.** Prompt 02
ships it unless the user says otherwise.

**D2 — the three copies of the equality closed form (§0.3).** Three sites compute
$1+z_{\rm eq}=\Omega_m/\Omega_r$: the solver guess, a `LambdaCDM` diagnostic, and `main.py`'s grid
feature. Unifying them is the obvious tidy and it is **not safe**: `main.py`'s copy is a production
sample location inside a `BackgroundModel` lookup key, so a one-ulp move invalidates every stored
object of eight types. The options are (i) leave all three and document the duplication as
deliberate — **the campaign's recommendation**; (ii) unify on the closed form, which is provably
one-ulp-stable only if the expression is written character-for-character identically at every site;
(iii) unify on the solve and accept a regeneration. **Prompt 03 measures and reports; it does not
decide.** Do not let an agent take (ii) or (iii) on its own judgement.

> **Answered 2026-09-16: (iii).** Recorded here additively, per §5 rule 6 — the paragraph above is
> the plan as written and was correct for the tree it was written on. The user's reason is that the
> closed form is right at `main.py` only because `QCD_Cosmology`'s $g_*$ structure sits twelve
> orders above $z_{\rm eq}$, which is an accident of this equation of state and not a property of
> the code; a cosmology with late entropy injection breaks it silently. Regeneration is not a cost
> in the build phase (board standing note 13), so (iii)'s price does not weigh against it.
> **The mechanism differs from this paragraph's wording:** `main.py` does not import
> `_find_rho_equality`; the model answers for itself through `BaseCosmology` properties. Prompt 09
> and the board's `[00-equality-redshift-closed-form-is-duplicated-three-times]` entry carry it.

**D3 — whether workstream D runs at all.** Prompts 07 and 08 clear two orphaned one-liners that no
campaign owns and that this one happens to be adjacent to. Neither is this campaign's subject.
They are gated: the orchestrator asks before starting D, and "no, leave them indexed" is a
perfectly good answer that costs nothing — the rows stay in `docs/OPEN_ISSUES.md` where they are.

**D4 — whether `[06-t-photon-call-cost-needs-a-quiet-machine]` closes or is accepted.** Prompt 05
hoists the loop-invariant work and re-measures. If the result is still above `qcd-background-audit`
README §6.2's 2.5 µs target, **the row itself is what to put to the user** — that is the issue's own
stated next step. The remaining excess is an order-5 `BSpline.__call__` and order 5 is not optional
(a cubic needs ~25,000 nodes for the required p90), so the honest options are to accept the target
was set 4 % too tight or to accept the cost. **Prompt 05 must not go looking for a third.**

---

## 8. Reading order for a new agent

1. [`AUDIT.md`](AUDIT.md) §0, §2, §3, §4, §5.
2. [`RECONCILIATION.md`](RECONCILIATION.md) — all of it; it is short.
3. This README §0, §1, §2, §5.
4. [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md).
5. Your own prompt. **Not the others** — a prompt that knows what comes next starts optimising for
   it.
