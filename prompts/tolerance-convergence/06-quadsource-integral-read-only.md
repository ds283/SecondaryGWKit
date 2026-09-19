# Prompt 06 — `QuadSourceIntegral`, read-only

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T11**. Opens nothing it can close itself: its output is a hand-off to
`prompts/levin-refactor` and `prompts/qsi-phase-groups` through `docs/OPEN_ISSUES.md` §1.2/§1.3.
**Depends on:** prompt 05a (`a9ad6fc`). Every other prompt in the campaign has landed.
**Recommended model:** **Opus** — the measurement is a sweep, but the design of it is the prompt.
What makes it succeed or fail is whether the two flavours of the fixture are used to *separate* the
quadrature error from the representation floor, rather than to measure one and report the other.

**This is the measurement half of what README §3.6 charters** (§7 **D11**, 2026-09-18). Prompt
**06a** takes the close-out document and `docs/TOLERANCE-PROVENANCE.md`, and is written against what
this prompt leaves. The split is the same one D7 and D9 made and for the same reason: §5 rule 1 makes
the commit the rollback boundary, and the campaign's named deliverable should not be reverted by a
failure in a sweep.

**It changes no production code, no schema and no number.** §5 rule 8's parameter freeze applies in
full: `config/defaults.py` is byte-identical at the end of this prompt. `DEFAULT_QUADRATURE_ATOL`
and `DEFAULT_QUADRATURE_RTOL` are **not this campaign's to move** (§0.4) and this prompt does not
recommend moving them — it *measures*, and reports to the campaigns that own them.

**Files you may create or touch:**
`docs/tolerance-convergence/quadsource_readonly.py` — new, the sweep;
`docs/tolerance-convergence/QUADSOURCE-READONLY.md` — new, its document;
`docs/tolerance-convergence/inventory.py` and `TOLERANCE-INVENTORY.md` — §4 below, and **§§1–4 of
that document are written and additive** (§5 rule 7);
`ComputeTargets/tests/convergence_reference.py` — **additively only, zero deletions**, and only if
§2 genuinely needs something that is not there;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
`AdaptiveLevin/` (README §0.4) — **not one line, not a comment, not a docstring**;
`ComputeTargets/tests/test_quadsource_integral.py`, which you **import from and do not edit**;
`config/defaults.py` (§5 rule 8); `main.py`; any factory or schema; the six `extract_*.py`;
`ComputeTargets/tests/test_main_plumbing.py`; the source grid, the band, `BREAK_POINT_KIND`,
`RESIDUAL_WKB_REGION_MARGIN` (README §0.5); and everything prompts 05, 05a and 05b own.

**Read first:** README §0.4 in full, §1.2, §2 (c), (e), (f), (i); §5 rules 5, 6, 7 and 8; §6.1 the
**whole** target rule, and especially rules 1, 4 and 5; §6.2's last row; §7 **D4**, which §0.4
settles as read-only and which §9 alone can put back to the user; board item **T11**;
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5.4 rows A, E and F — the rows this prompt is
about — and §6;
`prompts/source-remediation/IMPLEMENTATION_STATE.md`'s entries for
`[12-atol-too-loose-for-the-source-integral]`, `[12-handover-clamp-error-in-production]` and
`[09-abserr-is-a-quadrature-bound]`, **which you cite rather than restate**;
and `ComputeTargets/tests/test_quadsource_integral.py`'s module docstring, which describes the two
flavours this prompt is built on.

---

## 1. Why this prompt exists, and what the answer is likely to be

`QuadSourceIntegral` is the eighth keyed object type and the only one this campaign does not own.
It is also the only one whose tolerances were **already** decoupled and **already** chosen by
measurement — `DEFAULT_QUADRATURE_ATOL = 1e-32` and `DEFAULT_QUADRATURE_RTOL = 1e-8`, distributed
per sub-interval by log-width at `QuadSourceIntegral.py:818` and per phase group at `:1054`. README
§0.4 puts it out of scope for retuning and in scope for measurement, and §6.2's last row says
`read-only` in the Target column.

So the question this prompt answers is **not** "what should `quad_atol` be?". It is §6.1 rule 1's
question, asked of a sector nobody has asked it of: **what is the floor, and does the quadrature
error reach it?** The campaign has now twice found that the answer is a floor the parameter cannot
see — prompt 03's consumer spline dominating `GkNumericIntegration`'s solver error by ×631 to
×37,700, and prompt 04's double-precision accumulation dominating three Gauss orders by ×48 to
×3.03e5 — and in both cases §6.1 rule 4's `unchanged` was the result rather than the absence of one.

**The record already says which way this is likely to go, and says it three times.** You cite these;
you do not re-derive them and you do not re-run the live pipeline that produced them:

- `[09-abserr-is-a-quadrature-bound]` — `total_abserr` is the linear sum of the quadrature
  estimates and nothing else, and the true residual is up to **4.4e4×** larger. The column does not
  bound the error and `total_converged = False` is not a failure.
- `[12-handover-clamp-error-in-production]` — on production rows the hand-over clamp costs
  **6.6e-02 to 4.6e-01** against the analytic oracle where rows with no clamp gap agree to
  **6.2e-05**, and `T_WKB` itself differs from the exact `analytic_T_rad` by a median 7.1e-05–1.3e-03
  of envelope.
- `[12-atol-too-loose-for-the-source-integral]` — measured at `atol = 1e-25`, where 58 % of work
  items had a raw integral below the floor. **`config/defaults.py` now reads `1e-32`**, which is one
  of the three options that issue offered. §5 below is about what that means for the issue.

If the representation error is four to six orders above anything `atol` or `rtol` moves, then the
correct finding is that **the source integral's quadrature tolerance is not the limiting parameter
of the source integral**, and the hand-off says so. That is a result. An agent that reports a
tolerance recommendation without first establishing the floor has inverted §6.1 rule 1.

## 2. What is measured, and the one piece of experimental design that matters

**Offline, and that is a decision rather than a limitation** (user, 2026-09-18). The two live-run
drivers under `docs/source-remediation-verification/` need Ray and a populated datastore; every
sweep this campaign has run needs neither (`CLAUDE.md`, README §5), and the statistics a live run
would give are already in the record and cited in §1. **Do not attempt a datastore run**, and do not
treat its absence as a gap to apologise for — say in the log what the fixture is and is not.

**The vehicle already exists and you import it.** `ComputeTargets/tests/test_quadsource_integral.py`
builds `QuadSourceIntegral` offline on the exact constant-$w$ background, and exposes at module
level everything you need: `Case`, whose `run(atol=, rtol=)` calls the real
`evaluate_QuadSource_integral`; `SHAPES`; `X_RESP_VALUES`; `B_VALUES`; and the code's own oracle
`analytic_integral`, imported there from `QuadSourceIntegral.py`. **Import them. Do not edit that
module and do not copy it.** If something you need is local to a test method rather than to the
module, say so in the log and work around it in your own script rather than refactoring theirs.

**The design question, and it is the whole prompt.** `Case` has two flavours, and the module
docstring is explicit about what each carries:

- **exact** — the transfer functions, the Green's function and the source spline are all exact to
  rounding, so every ingredient is truth and the comparison against the oracle **tests the partition
  and the integrator alone**;
- **realistic** — `bessel_phase` amplitude and phase re-splined on the production 100-per-decade
  grid, a real `phase_spline` Green's function, and `f` a spline through exact samples as
  `QuadSource` stores it, which **carries the representation floors** logs 05, 06 and 07 of
  `source-remediation` measured.

That pair is exactly the instrument §6.1 rule 1 asks for, and nothing else in the tree is:

1. **Sweep `(atol, rtol)` on the *exact* flavour** to get the quadrature error, which is what the
   two constants actually control. Sweep loose to tight, both axes, not a diagonal — README §2 (d)
   is the record of what a diagonal costs — with the production pair inside the range and at least
   two steps either side of it on each axis, so that inertness is a measurement and not an
   assumption.
2. **Measure the floor on the *realistic* flavour**, at the production pair, as the residual against
   the same oracle. That is the representation error the production configuration carries into every
   stored `total`.
3. **Report the ratio.** The floor divided by the quadrature error, per case, as prompt 03 reported
   ×631–×37,700 and prompt 04 ×48.4–×3.03e5. §6.1 rule 4 is then applied, and if the answer is
   `unchanged` it is written in that word.

**A second oracle is available and should be used where it is cheap.** The module already compares
against `scipy.quad` of the exact integrand on a short range (`TestDirectQuadOracle`), which is
independent of `analytic_integral`. A quadrature error measured against the code's own closed form
is worth calibrating against one that shares none of its machinery, at least at one case.

**Rules 5 and 6 bind here as everywhere.** Every number carries its reference's own error — for the
exact flavour that is the oracle's rounding, which you must state rather than assume is zero — and
every number says which fixture flavour and which `(b, shape, x_resp)` it was taken at. This sector
has **no source grid**, so §5 rule 6 has nothing to bite on except the `PRODUCTION_SAMPLES_PER_LOG10Z`
sampling the realistic flavour uses; say so explicitly rather than leaving a reader to wonder which
generation a figure belongs to.

**Cost is counts, not wall time** (§2 (i)), and the object count belongs beside the per-object figure
(§2 (c)): the inventory gives `QuadSourceIntegral` as **1,275 × 50 × (response z) per model**, which
is the largest object count in the pipeline and the reason a per-item percentage here is meaningless
on its own. The Levin driver's counters are already reported through the result payload; use them.

## 3. The three things you must not conclude

- **Do not conclude from `total_abserr`.** It is a quadrature bound, and
  `[09-abserr-is-a-quadrature-bound]` measured it missing the true residual by up to 4.4e4×.
  Measure against an oracle, not against the code's own error estimate. If you report
  `total_abserr` at all, report it as the *declared* bound beside the *measured* residual, and say
  which is which.
- **Do not conclude that `rtol` binds without checking, or that it does not.** `source-remediation`
  log 12 found `1e-8 → 1e-11` **bit-identical on 159 live items** because `atol` bound at `1e-25`.
  `atol` is now `1e-32`, seven decades tighter, so that finding does **not** transfer and the
  question is open again. This is the single most likely error in this prompt: inheriting a
  conclusion measured at a constant that has since moved.
- **Do not conclude anything about the production `(k, q, r, z)` range from the fixture.** The 58 %
  figure, the regime mix and the clamp-gap census are live-run measurements; they are cited, not
  re-taken, and a fixture result must not be presented as superseding them.

## 4. The inventory is stale, and this prompt is where it is regenerated

`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5 is generated, and
`PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check` **reports it out of
date at `a9ad6fc`** — as it should, because prompts 05, 05a and 05b changed the constants, the lookup
predicates and which object types carry a pair at all. Prompt 06a assembles
`docs/TOLERANCE-PROVENANCE.md` from §5.4's rows and cannot do that against a table describing the
tree of 2026-09-17.

Regenerate it, and understand what regeneration does and does not do — §6 of that document says so
plainly. `--write` replaces the generated block; it will pick up the six constants prompt 05a
shipped, the changed values, and the key columns prompts 05 and 05b moved. It will **not** touch the
judgement columns — *Reaches*, *The real knob*, *Provenance*, *Owned by* are declared in
`inventory.py`'s `PARAMETERS` and a re-run reproduces them unchanged. Several are now **wrong**, and
correcting them is part of this prompt:

- `DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE` are listed as keying seven and eight object
  types. **After 05a they key none.** Their row must say what they are now — the float-comparison
  epsilon of `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` and a set of signature
  defaults — and the six new constants must appear with their owners and provenance.
- The four Gauss orders are listed as keying **none**. Prompt 05 put three of them in the
  `BackgroundModel` key and `RHO_GAUSS_ORDER` in both WKB keys, and prompt 05b made the recorded
  order the order the object was built at.
- `DEFAULT_QUADRATURE_ATOL`'s provenance cites the measurement but the value in the *issue index*
  still reads `1e-25`; see §5.

**§§1–4 of that document are written, not generated, and §5 rule 7 makes verification documents
additive.** Where the argument in §§1–4 has been overtaken — §1's headline and §2.3's paragraph about
what prompt 05 expects are the obvious candidates — **add a dated subsection recording what is now
true; do not rewrite a paragraph that was correct for the tree it was written on.** The
`inventory.py --check` gate passing is an acceptance condition (§8), and a hand-edit inside the
markers fails it.

## 5. The hand-off (board item **T11**)

§0.4's instruction is to **report, not act**. Two campaigns own these files and this campaign has no
standing to change them.

- **Write the hand-off into `docs/OPEN_ISSUES.md` §1.2 and §1.3**, as one-line index rows per
  `CLAUDE.md`'s rule, pointing at `QUADSOURCE-READONLY.md` for the measurement. If your finding is
  that the quadrature tolerance is not the limiting parameter, that is what the row says.
- **`[12-atol-too-loose-for-the-source-integral]`'s index row is stale**: it quotes
  `DEFAULT_QUADRATURE_ATOL = 1e-25` while the tree reads `1e-32`, which is one of the three options
  the issue itself offered. Determine what actually happened — `config/defaults.py`'s own comment
  above the constant is the primary record — and **narrow the row additively**, saying what the
  constant is now and what of the issue survives the change. Do **not** close it: whether 58 % of
  work items still meet their tolerance before doing any work is a *live-run* question and you have
  no live run. The `source-remediation` board entry is that campaign's; **add** to it rather than
  rewriting it (§5 rule 7), and say in your log that you did.
- **Any new issue you open against `QuadSourceIntegral` is theirs, not this campaign's** — it goes
  in `docs/OPEN_ISSUES.md` under §1.2 or §1.3 with the owning board named, and this campaign's board
  §3 records only that it was opened and where.

## 6. What this prompt does not do

- **It does not retune `quad_atol` or `quad_rtol`,** or recommend a value for either. §0.4, and
  §5 rule 8 independently. It may report *what the measurement implies* about them, phrased as a
  finding handed to the owning campaigns, and the difference between that and a recommendation is
  not cosmetic: the first is this campaign's to make and the second is not.
- **It does not touch `CHEBYSHEV_ORDER`, `DEFAULT_LEVIN_MAX_DEPTH`, `limit=100` or
  `BESSEL_ORDER_CHECK_TOL`** — inventory §5.4 rows E and F, all `levin-refactor`'s. It **records**
  them, including that `DEFAULT_LEVIN_MAX_DEPTH = 20` and `limit = 100` were **never chosen**, so
  that 06a has them.
- **It does not write `docs/TOLERANCE-PROVENANCE.md` or the close-out document.** Both are prompt
  06a's, and writing either here defeats the split (§7 D11). What it *must* do is leave 06a the five
  §1.2 fields for every parameter it touched, in its log's "State handed to the next prompt".
- **It does not re-open D4.** §7 D4 — whether `QuadSourceIntegral` should be retuned by this
  campaign rather than read — is the user's, and §9 says when to put it back to them.

## 7. The tests

The sweep is a `docs/` script and is not a test. But a measurement nobody can re-run is not a
measurement, so:

- **The script runs from the repository root with no arguments**, needs neither Ray nor a datastore,
  emits the document's tables on stdout and a progress log on stderr, and changes nothing — the
  pattern `docs/tolerance-convergence/gk_numeric_sweep.py` and `order_audit.py` set. Say the
  invocation in the document.
- **`inventory.py --check` must pass** after §4, and it is run as a test would be.
- **If you add anything to `ComputeTargets/tests/convergence_reference.py`, it is additive and it
  carries a test** in `test_convergence_reference.py`, and the existing count must not fall. Prefer
  not adding to it: the QSI sector's ingredients are `test_quadsource_integral.py`'s, not the
  harness's, and a sector added to the harness for one caller is a liability. If you do add, justify
  it in the log.

Neither the script nor any test may need Ray or a datastore (README §5, `CLAUDE.md`).

## 8. Acceptance

1. `docs/tolerance-convergence/QUADSOURCE-READONLY.md` exists and reports, for the production pair
   and the swept range: the quadrature error on the **exact** flavour, the representation floor on
   the **realistic** flavour, and the ratio between them, each with its reference's own error (§5
   rule 5) and each saying which flavour and which case it was taken at.
2. **Both axes swept, not a diagonal**, with the production pair interior to the range and at least
   two steps either side on each. Whether `rtol` binds at `atol = 1e-32` is answered on this tree
   and not inherited from log 12's measurement at `1e-25`.
3. §6.1's target rule is applied and its outcome stated in the rule's own words — including
   `unchanged` if that is what it gives, with the factor by which the floor dominates (rule 4).
4. The three findings of §1 are **cited**, not re-derived, and no fixture result is presented as
   superseding a live-run figure (§3).
5. `PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check` **passes**, the
   judgement columns of §4 are corrected, and §§1–4 of `TOLERANCE-INVENTORY.md` are **added to**
   rather than rewritten.
6. `docs/OPEN_ISSUES.md` §1.2/§1.3 carry the hand-off; `[12-atol-too-loose-for-the-source-integral]`
   is **narrowed, not closed**; anything new is filed to the owning board.
7. **No file under README §0.4 is in the diff**, `config/defaults.py` is byte-identical, and no
   production module is touched at all. `git diff --name-only HEAD~1 HEAD` contains nothing outside
   `docs/`, `prompts/tolerance-convergence/` and — only if §7 justified it —
   `ComputeTargets/tests/convergence_reference.py` with its test.
8. `ComputeTargets` **must not fall below 521**; `CosmologyModels` **39**. Both OK.
9. The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`. This
   prompt has no business moving them and the check is cheap.
10. `black --check` clean on every `.py` in the diff.
11. Board row 06, item **T11**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date corrected —
    all in the **same commit**.

## 9. Stop conditions

Stop and report rather than working around any of these:

- **The measurement says the quadrature tolerance *is* the limiting parameter** — that is, the
  quadrature error is at or above the representation floor at the production pair. That puts §7
  **D4** back to the user: §0.4 put `QuadSourceIntegral` out of scope on the premise that its
  tolerances were already chosen by measurement, and a finding that they are the binding constraint
  is exactly the circumstance D4 reserves for the user. Report the numbers; do not recommend a value.
- **You need to edit any file under §0.4** — `QuadSourceIntegral.py`, `QuadSource.py`,
  `phase_groups.py`, `AdaptiveLevin/` — to make the measurement work, or you need to edit
  `test_quadsource_integral.py`. Say what you needed and why.
- **The fixture cannot be driven at the production `atol = 1e-32`**, or its default of `1e-25` turns
  out to be load-bearing for the module's own tests in a way that makes the sweep unrepresentative.
- **`inventory.py --write` changes something you did not expect** — a lookup predicate, a keyed
  object type, or a constant that no prompt in this campaign moved. That is a report about the tree,
  not a merge conflict to resolve.
- **You conclude that `docs/TOLERANCE-PROVENANCE.md` cannot be assembled from the logs** for some
  parameter — say which and why, because that is prompt 06a's whole premise and it is better known
  now than then.
- **`ComputeTargets` falls at all.**

## 10. The log

`logs/06-quadsource-integral-read-only.md`, on the campaign's template, classifying every deviation.
Beyond the template:

- **The experimental design**: which flavour measured what, why, and what each one's own error is.
- **The floor**, stated as §6.1 rule 1 requires, and whether it was measured here or inherited with
  citation — and if inherited, from which document and taken on what.
- **The `rtol` question**, answered on this tree at `atol = 1e-32`, with the log 12 result named as
  the thing it supersedes or confirms.
- **The inventory**: what `--write` changed, which judgement columns you corrected and why each was
  wrong, and what you added to §§1–4 rather than rewrote.
- **The hand-off**: what went to §1.2 and §1.3, and what you did to
  `[12-atol-too-loose-for-the-source-integral]`.
- **"State handed to the next prompt"** — prompt 06a assembles the close-out document and
  `docs/TOLERANCE-PROVENANCE.md`. Give it the **five §1.2 fields** for every parameter in inventory
  §5.4 rows A, E and F that this prompt touched or measured, and — as important — say for each one
  you could **not** establish that the provenance cannot be established from the record, in those
  words (§1.2's closing rule). 06a assembles; it does not investigate.
