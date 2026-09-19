# Log 06 — `QuadSourceIntegral`, read-only

**Prompt:** `prompts/tolerance-convergence/06-quadsource-integral-read-only.md`
**Commit:** *(this prompt's own commit)* — "Measure the source integral against its own floor"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE, and the answer is `unchanged`.** The quadrature tolerance is **not** the
limiting parameter of the source integral: measured offline over the 18 cases of
`ComputeTargets/tests/test_quadsource_integral.py`, in both flavours and on both axes, the
representation floor the realistic fixture carries dominates the quadrature error at
`(atol, rtol) = (1e-32, 1e-8)` by **×3.46e+04 to ×2.15e+11** — the campaign's third `unchanged`
under README §6.1 rule 4 and the largest domination factor of the three by four orders. §7 **D4**
is therefore **not** reopened and this is a hand-back, not a stopping point (prompt §9, README
§4.1). `docs/tolerance-convergence/QUADSOURCE-READONLY.md` is the document and
`quadsource_readonly.py` regenerates everything from its §1 down.

**The `rtol` question, answered on this tree and answered the other way.** Prompt §3 named
"inheriting a conclusion measured at a constant that has since moved" as the single most likely
error here, and it was a live risk: `source-remediation` log 12's `1e-8 → 1e-11` **bit-identical on
159 live items** was measured at `atol = 1e-25`, where `atol` bound. At `1e-32`, `atol` is inert
across **twenty-eight decades** and binds only at `1e-16` and looser, and `rtol` is then the only
parameter that moves the answer — about a decade of error per decade of tolerance, and the same
three decades move `total` by a factor of **274**. The corners show the two never interact:
whichever term of `atol + rtol|value|` is larger binds, one at a time, and at production it is
`rtol`. **The regime has inverted since log 12 and nothing in the record said so.**

**Two things the prompt did not anticipate, and both changed the measurement rather than
decorating it.** (1) `analytic_rad`, the code's own oracle *and a stored column*, is computed
inside `evaluate_QuadSource_integral` **at the caller's own `atol` and `rtol`**, so a swept
`|total − analytic_rad|` compares two quantities that moved together; every oracle figure here is
therefore `analytic_rad` at a reference pair, held fixed, and the primary statistic is a
self-convergence one through `convergence_reference.reference_drift`. At production the oracle sits
up to **×1.25e+06** further from its converged value than `total` does. (2) Regenerating
`TOLERANCE-INVENTORY.md` §5.1 **dropped four of the campaign's own eight targets** until
`inventory.py`'s selection rule was widened to the Gauss-order columns prompt 05 introduced — §5
below, and deviation 2.

**Nothing in production moved.** No file under README §0.4 is in the diff — not a line, not a
comment — `config/defaults.py` is byte-identical, `test_quadsource_integral.py` is unedited,
nothing was added to `ComputeTargets/tests/convergence_reference.py`, and **zero production files
are in the diff**. `ComputeTargets` **521, OK**; `CosmologyModels` **39, OK**;
`inventory.py --check` passes; the three published grid digests are unmoved.

Board item **T11**. `[12-atol-too-loose-for-the-source-integral]` is **narrowed, not closed**. Two
issues are opened and **both are filed to the campaigns that own the files**, per prompt §5:
`[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]` on `levin-refactor`'s board and
`[06-analytic-rad-is-computed-at-the-callers-tolerance]` on `qsi-phase-groups`'.

---

## 1. The experimental design, and what each flavour's own error is

Prompt §2 says the design is the prompt, and this is it in one paragraph: the **exact** flavour
measures the quadrature error alone, because every ingredient is truth to rounding; the
**realistic** flavour at the *same* tolerance measures the representation floor alone, because the
quadrature error is then common to both runs and cancels out of the difference. A sweep on the
realistic flavour by itself measures the two convolved and can report neither, which is exactly
what §6.1 rule 1 forbids.

| what | flavour | reference | the reference's own error |
|---|---|---|---|
| quadrature error | **exact** | `total` at `(1e-45, 1e-12)`, per case | its drift against `(1e-46, 1e-13)`: **0 to 9.598e-14** of scale, and one ulp of `total` is 3.7e-17 to 2.2e-16 — the same rounding, from the same doubles |
| calibration of that reference | **exact** | `analytic_rad` at the same pair, held **fixed** | its own displacement is what §2.1 of the document measures; it is *not* zero and is why it is held fixed |
| second, independent oracle | **exact**, short range | `scipy.quad` of the exact integrand, `epsrel = 1e-12` | `quad`'s own declared error, **1.0e-13 to 7.0e-13** relative |
| the floor | **realistic** | the **exact** run of the same case at the same tolerance | the quadrature error, which cancels; residually the drift above |

**Normalisation.** `numeric_quad` and `WKB_Levin` cancel by up to ×5, so an error divided by
`|total|` is inflated by the cancellation. Everything is divided by the fixture's own
`scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)` taken from the reference run of the same
case, and the `|total|`-relative figure is reported beside it and labelled.

**Rule 6 has nothing to bite on and the document says so rather than leaving it silent.** This
sector has no source grid. The realistic flavour's only sampling parameter is
`PRODUCTION_SAMPLES_PER_LOG10Z = 100`; the exact flavour has none at all.

**Rule 5 at every use.** Where a case's measured difference at production is below the larger of
the reference's drift and one ulp, §8 of the document reports the *resolution* as the quadrature
figure and marks the row **bound**, so its ratio is a lower bound on the domination and not an
estimate of it. That happens on **3 of the 18** cases.

## 2. The floor (prompt §10, README §6.1 rule 1)

**Measured here, not inherited**, on the tree this campaign runs on: **2.523e-06 to 4.912e-04 of
scale** at the production pair over the 18 cases (2.523e-06 to 2.380e-03 of `|total|`). It is the
realistic flavour's representation error — the `QuadSource` spline of `f`, the Liouville–Green
closed forms at the hand-over, the re-splined `bessel_phase` amplitude and phase, and the hand-over
clamp — at `PRODUCTION_SAMPLES_PER_LOG10Z = 100`.

**It does not see the quadrature tolerance**, which is what makes §8's ratio a ratio of independent
quantities: across six decades of `rtol` it moves by at most a factor of **1.642**, and between
production and `rtol = 1e-11` — three decades over which the quadrature error falls by two orders —
by at most **17.14 %**.

**It is consistent with the record and supersedes nothing.**
`[06-source-spline-residual-vs-handover]` puts the `QuadSource` spline of `f` at 4.5e-4 of the local
envelope at the hand-over and `[07-lg-derivative-truncation-at-handover]` the LG closed forms at
7e-6 (`b=0`) to 1.4e-4 (`b=0.25`); the fixture's own module docstring quotes both. This is the same
floor re-measured through the same fixture, in this document's own normalisation.

## 3. The `rtol` question (prompt §3, second bullet; prompt §10)

Answered on this tree at `atol = 1e-32`, and it **supersedes** log 12's conclusion rather than
confirming it — while leaving log 12's *measurement* entirely intact.

| | log 12 (live, `atol = 1e-25`) | prompt 06 (fixture, `atol = 1e-32`) |
|---|---|---|
| which half binds | `atol` | **`rtol`** |
| `1e-8 → 1e-11` on `total` | **bit-identical**, 159 items | moves it by ×**274** |
| `atol` swept | not swept | inert over 28 decades; binds at `1e-16` and looser |
| consequence for the value | tighten `atol` (done: `1e-32`) | **`unchanged`** — the floor dominates by ×3.46e+04 to ×2.15e+11 |

The mechanism is arithmetic. `atol` is distributed per sub-interval by log-width
(`QuadSourceIntegral.py:818`) and per phase group by count (`:1054`), and `|total|` in these cases
is 9.3e-13 to 2.3e-08 — twenty decades above `1e-32`. That is README §2 (e)'s magnitude argument
appearing in the one sector where the constant had already been chosen by measurement: an absolute
tolerance is a statement about magnitudes, and at `1e-32` it is a statement about magnitudes this
sector does not have.

**What does not follow.** Nothing about the production `(k, q, r, z)` range: the 58 % census, the
regime mix and the clamp-gap counts are live-run measurements and are cited, never re-taken
(prompt §3, third bullet).

## 4. Cost, and `total_abserr`

Counts, not wall time (§2 (i)), against the object count **1,275 × 50 × (response z) per model**
(§2 (c)) — the largest in the pipeline. Summed over the 18 exact-flavour cases, `rtol` at
production costs 4,242 `quad` right-hand-side evaluations and 222 Levin evaluations; three decades
tighter is **+25.0 %** and three decades looser **−25.8 %**, while `atol` is free in both
directions below `1e-20`. The README §6.1 rule 3 ladder is §8.1 of the document: the loosest `rtol`
that clears the floor on **every** case is `1e-5`, and it clears it by only **×3.32**. That is the
one place where the comfortable summary — "the loose end of the axis is orders clear" — would have
been wrong, and the ladder exists so that it is not written.

`total_abserr` is reported **only** as the declared bound beside the measured residual, never
concluded from (prompt §3, first bullet): it overstates the measured quadrature error by ×3.19 to
×135 across the `rtol` axis here, which is the opposite sign to
`[09-abserr-is-a-quadrature-bound]`'s live 4.4e4× *understatement* against the true residual —
because the two are bounding different things, one the quadrature and the other the total. The
document says which is which. `total_converged = False` on 4 of 126 runs of the `rtol` axis, all at
`rtol` ∈ {1e-10, 1e-11}, and per that issue it is not a failure.

## 5. The inventory (prompt §4, prompt §10)

`inventory.py --check` failed at `a9ad6fc` as the prompt said it would, and passes now.

**What `--write` changed, and one of them was a stop condition worth reading twice.**

1. **The four Gauss orders became key columns and `GkSource` left the table.** §5.1's generated
   count went 12 → 11. On the *first* re-run it went 12 → **8**, because the selection rule
   (`inventory.py.ACCURACY_COLUMNS`) listed only `atol_serial`, `rtol_serial`, `log10_tol` and
   `Levin_threshold` — so `BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration` dropped out
   **not because they stopped being keyed on an accuracy parameter but because the parameter
   stopped being a tolerance**, which is the campaign's own result. Widening the set to the four
   order columns is deviation 2. `GkSource` leaves for the right reason: after prompt 05 it keys no
   accuracy parameter of any kind.
2. **`main.py` line drift** from 05a and 05b: the four Bessel literals move from `:1116-1128` to
   `:1197-1209`. Values unchanged at `1e-12`.
3. **Six constants appear in §5.2** — the five new names prompt 05a shipped plus
   `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, which predates it — once they were added to
   `imported_constants()`. §5.4 is now **44 rows**, not 39.

**The judgement columns a re-run does not touch, and why each was wrong.**

| row | was | now |
|---|---|---|
| `DEFAULT_ABS_TOLERANCE` | keys seven object types; reaches both | **keys none.** Reaches neither. What is left is seven bare float comparisons and a signature default production overrides; the provenance line now says in §1.2's own words that it cannot be established |
| `DEFAULT_REL_TOLERANCE` | keys eight | **keys none**; its three former consumers each have their own constant now |
| the five new `DEFAULT_*_TOLERANCE` rows | absent | added, each with the measurement that chose it, whether it is **chosen** or **inert and unchosen**, and the grid generation and anchor of that measurement |
| `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` | "the pair" | **a step-selection knob, not the level** — prompt 03a's re-characterisation, value unchanged |
| `DEFAULT_QUADRATURE_ATOL` / `_RTOL` | provenance: log 12 | log 12 **plus** this prompt's measurement, including that log 12's `rtol` finding does not transfer |
| the four Gauss orders | key **none**; "in no key" | key their targets, by equality, since prompt 05; provenance re-pointed from the 2026-09-10 block to `ORDER-AUDIT.md` and prompt 04b's regenerated one; 05b's construction-time correctness noted |
| `RESIDUAL_WKB_REGION_MARGIN` | "never chosen against a criterion" | still in no key, and now **by measurement** — bit-identical residual from margin 0.05 to 0.9, which is what let prompt 05 leave it out |
| `_solve_horizon_exit` `xtol`/`rtol` | "the shared pair", "never chosen" | its own pair since 05a, and **measured** by 03a — `xtol` binds at 0 of 150 pairs |

**§§1–4 are added to and not rewritten** (§5 rule 7). Four dated subsections: **§1.1** (what the
headline looks like after 04b, 05, 05b and 05a), **§2.5** (what prompts 05 and 05b did to §2.4's
"raise `TAU_GAUSS_ORDER` today and the pipeline serves the order-4 row" — it is no longer true, and
closing it was the point of prompt 05), **§3.5** (the four questions re-scored: §3.4's seventeen
parameters-in-no-key falls to twelve) and **§4.1** (the exclusion boundary, where the one argued
exclusion is now the whole of what `DEFAULT_ABS_TOLERANCE` does). No paragraph that was correct for
the tree it was written on has been altered.

## 6. The hand-off (prompt §5, board item **T11**)

- **`docs/OPEN_ISSUES.md` §1.2** gains `[06-analytic-rad-is-computed-at-the-callers-tolerance]`,
  board `qsi-phase-groups` — the campaign that owns the three-Bessel machinery `analytic_integral`
  is built from. That section read *(none open)*; the sentence explaining prompt 01's own closures
  is kept beneath the table rather than deleted.
- **§1.3** gains `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`, board
  `levin-refactor` — the campaign that owns `AdaptiveLevin/` and the per-sub-interval distribution.
- **Both issues are written in full on those boards' §3**, and the index rows are one line each
  pointing at `QUADSOURCE-READONLY.md`, per `CLAUDE.md`'s rule that the index never holds issue
  content. Deviation 1 records why this reaches two boards outside this campaign.
- **`[12-atol-too-loose-for-the-source-integral]` is narrowed, not closed.** The
  `source-remediation` board entry is **added to** (§5 rule 7) with a dated block, and the original
  paragraph is untouched. What the addition says: (i) the third of the three options the issue
  offered **was taken** — `DEFAULT_QUADRATURE_ATOL` has read `1e-32` since that campaign's own
  prompt 12, and `config/defaults.py:160-165` is the primary record, so the index's `1e-25` was
  seven decades stale; (ii) the regime has inverted, so the entry's own `rtol` bit-identity no
  longer transfers. **What is not closed:** whether 58 % of *production* work items still meet
  their tolerance before doing any work is a live-run question and this prompt had no live run. The
  revised next step is to re-take that census at `1e-32`.
- The index count moves **78 → 80**.

## 7. Verification performed

| check | result |
|---|---|
| `ComputeTargets` suite | **521 tests, OK** — prompt §8 requires "must not fall below 521" |
| `CosmologyModels` suite | **39 tests, OK** |
| `inventory.py --check` | **passes** (failed at `a9ad6fc`, as the prompt said) |
| the three published grid digests | unmoved — `3bef2c06`, `60a3205a`, `21ffc126`, asserted by `ComputeTargets/tests/test_source_grid.py` inside the suite above, and no production code is in the diff |
| `config/defaults.py` | **byte-identical** (`git diff --quiet` clean) |
| files under README §0.4 in the diff | **none** — `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/` untouched |
| `ComputeTargets/tests/test_quadsource_integral.py` | **unedited** |
| `ComputeTargets/tests/convergence_reference.py` | **unedited** — nothing was added to it (prompt §7's preference) |
| production files in the diff | **zero** |
| `black --check` | clean on `quadsource_readonly.py` and `inventory.py`, the two `.py` in the diff |
| the sweep, from the repository root, no arguments, no Ray, no datastore | runs in **~155 s** and emits the document's §§1–11 on stdout with a progress log on stderr |

## 8. Deviations from the prompt

1. **Two campaign boards outside this one are in the diff** — `prompts/levin-refactor/` and
   `prompts/qsi-phase-groups/` — plus `prompts/source-remediation/`. **`STRUCTURALLY REQUIRED`.**
   Prompt §8 acceptance 7 lists the permitted paths as `docs/`, `prompts/tolerance-convergence/`
   and (if justified) `convergence_reference.py`, but prompt §5 requires the
   `source-remediation` board entry to be **added to** and acceptance 6 requires anything new to be
   "filed to the owning board". `CLAUDE.md` forbids an issue existing only in the index. The three
   board edits are therefore the prompt's own instructions, and acceptance 7's path list is what is
   incomplete. Each edit is purely additive: one block appended to an existing entry on
   `source-remediation`, one new §3 entry on each of the other two, and no existing sentence
   altered on any of them.
2. **`inventory.py`'s `ACCURACY_COLUMNS` was widened by the four Gauss-order columns.**
   **`STRUCTURALLY REQUIRED`.** Without it the regenerated §5.1 drops `BackgroundModel`,
   `GkWKBIntegration` and `TkWKBIntegration` — three of this campaign's eight targets — for the
   wrong reason: their accuracy parameter is still in the key, it is just no longer a tolerance.
   Prompt §4 makes correcting the table's judgements part of this prompt and lists `inventory.py`
   among the files it may touch; leaving the rule unchanged would have published a keyed-object
   table that contradicts §2.5 of the same document. It is not the §9 stop condition
   (*"`--write` changes something you did not expect"*): what changed is what prompts 05 and 05b
   did, which §4 of the prompt names in advance, and the *rule* is this prompt's own code.
3. **The second oracle's construction is rebuilt in the sweep rather than imported.**
   **`IMPLEMENTATION CHOICE`, and the prompt authorises it in terms.** `scipy.quad` of the exact
   integrand is assembled inside `TestDirectQuadOracle.test_short_range`, a test method, not at
   module level; prompt §2 says to "say so in the log and work around it in your own script rather
   than refactoring theirs". Three lines are repeated: the `x_r = 60 → 4` short range, the `quad`
   call and the `(1 + z_resp)` weight. `Case.integrand_exact` and `z_at_tau` are imported.
4. **`analytic_rad` is held at a reference pair rather than read per run.**
   **`IMPLEMENTATION CHOICE`.** Prompt §2 item 2 says to measure the floor "as the residual against
   the same oracle", which reads naturally as the oracle of that run. It cannot be: the oracle is
   computed at the caller's tolerances (§1 above), so a swept residual against it measures the sum
   of two movements. Both are reported — the floor against the fixed oracle *and* against the exact
   twin at the same tolerance — and the difference between them is small, which is the check that
   the choice did not matter to the answer.
5. **One step of the `atol` ladder is four decades, not one.** **`IMPLEMENTATION CHOICE`.** Prompt
   §8 acceptance 2 requires the production pair interior with "at least two steps either side on
   each axis". On `atol` a decade at `1e-32` is not a step anyone would take, and a five-decade
   sweep would have found inertness and located nothing; the ladder runs `1e-12 … 1e-40`, two steps
   tighter and five looser, which is what locates the binding setting at `1e-16` instead of merely
   bounding it. `rtol`'s step is one decade, three either side.

## 9. Observations not acted on

- **The fixture builds its Bessel splines one order looser than production.**
  `Case.bessel_phase_data` calls `bessel_phase(..., atol=1e-25, rtol=5e-14)`, and those two keywords
  are the **deprecated pair that `bessel_phase` ignores** (`LiouvilleGreen/bessel_phase.py:932-945`,
  which raises a `DeprecationWarning` and maps them to nothing). The splines are therefore built at
  `DEFAULT_PHASE_ATOL = DEFAULT_AMPLITUDE_RTOL = 1e-11`, against `main.py:1200`'s `1e-12`. It is
  recorded in §11 of the document as the floor under `analytic_rad` here and **not repaired**:
  `test_quadsource_integral.py` is a module this prompt imports and may not edit. It does not
  affect any conclusion — the floor measured on the realistic flavour is four orders above anything
  the Bessel budget contributes — but a later prompt that tightens the fixture should know the
  keywords are inert.
- **No issue is opened for it**, because the owning module is a test fixture of
  `prompts/source-remediation` and the observation is about a call that has no effect; §5 of the
  prompt scopes new issues to `QuadSourceIntegral` itself.
- **`ComputeTargets/QuadSourceIntegral.py:1550`'s stale comment is now wrong twice over.** The
  board has recorded since prompt 02 that it still says "the pipeline supplies
  `DEFAULT_QUADRATURE_ATOL = 1e-25`" while the constant has read `1e-32` since
  `source-remediation` prompt 12. Confirmed still stale here, and the same paragraph also carries
  the `rtol` bit-identity that §3 above shows does not transfer at `1e-32`. The file is out of
  bounds under README §0.4 — not one line, not a comment — so the board bullet is annotated and
  nothing is edited. It is a comment in the file `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`
  is filed against, so whoever repairs that issue will be standing in front of it.
- **`GkSource` has left the keyed-object table for a reason the README's §2 (a) table does not yet
  reflect.** README §2 (a) still lists eight object types keyed on a tolerance pair and §0.1's
  summary sentence still says six share one. Prompt 02 recorded both as needing reconciliation and
  §3.6 assigns it to "prompt 06"; the split at §7 D11 leaves it ambiguous between 06 and 06a, and
  this prompt has recorded the correction in `TOLERANCE-INVENTORY.md` §1.1 and §3.5 rather than
  editing the README, which is where the campaign's own close-out (06a) can take it up with the
  rest of the reconciliation. Not a new issue; a note for 06a.

## 10. State handed to the next prompt

**Prompt 06a assembles; it does not investigate.** The coverage checklist is
`TOLERANCE-INVENTORY.md` §5.4, which this prompt leaves at **44 rows** (not the 39 README §3.6a
quotes from prompt 02's tree), plus the six constants prompt 05a shipped — all six of which are now
*in* those 44 rows, with their five fields, so the two sets no longer need adding together.

**The five §1.2 fields for every parameter this prompt touched or measured** (rows A, E and F of
§5.4):

**(a) `DEFAULT_QUADRATURE_ATOL = 1e-32`** — keys `QuadSourceIntegral`.
*Value and what it keys:* the absolute half of the source integral's pair, distributed per
sub-interval by log-width and per phase group by count.
*What measurement chose it:* `prompts/source-remediation` prompt 12, on a **live run**, raising it
from `1e-25` where 58 % of work items met their tolerance before doing any work; the measurement is
at `config/defaults.py:160-165`. **Re-measured read-only here**, offline, 18 fixture cases,
**no source grid in this sector**: `unchanged` under §6.1 rule 4.
*The competing floor:* the representation floor of the realistic flavour, **2.523e-06 to 4.912e-04
of scale**, dominating the quadrature error by **×3.46e+04 to ×2.15e+11**.
*Cost:* inert. Below `atol = 1e-20` the evaluation count does not move at all; the sector is
1,275 × 50 × (response z) objects per model.
*Campaign, prompt, log, date:* `prompts/source-remediation` prompt 12 (2026-09-09) for the value;
`prompts/tolerance-convergence` prompt 06, this log,
`docs/tolerance-convergence/QUADSOURCE-READONLY.md` (2026-09-18) for the confirmation.

**(b) `DEFAULT_QUADRATURE_RTOL = 1e-8`** — keys `QuadSourceIntegral`.
*Value and what it keys:* the relative half, passed unchanged to every sub-interval and every phase
group.
*What measurement chose it:* **nothing chose it.** `source-remediation` log 12 measured it *not
binding* at `atol = 1e-25` and left the value where it was. **That finding does not transfer to
this tree** and prompt 06 says so: at `1e-32` it is the binding half and `1e-8 → 1e-11` moves
`total` by ×274 on the fixture. So the *value* is inherited and unchosen, while the *decision to
leave it* is now measured.
*The competing floor:* as (a); at production the floor dominates by ×3.46e+04 to ×2.15e+11.
*Cost:* at production, 4,242 `quad` RHS + 222 Levin evaluations over the 18 fixture cases; one step
either side, `1e-9` is **+3.9 %** and `1e-7` **−5.7 %**; three decades tighter is +25.0 % and three
looser −25.8 %, on the largest object count in the pipeline.
*Campaign, prompt, log, date:* `prompts/tolerance-convergence` prompt 06, this log,
`QUADSOURCE-READONLY.md` §§5, 8.1 and 9 (2026-09-18). The README §6.1 rule 3 ladder is §8.1: the
loosest `rtol` clearing the floor on every case is `1e-5`, by ×3.32.

**(c) `CHEBYSHEV_ORDER = 24`** (§5.4 row E) — keys nothing; `QuadSourceIntegral` stores the
*achieved* `WKB_Levin_chebyshev_min_order` as a diagnostic and does not filter on it.
*Provenance:* `prompts/source-remediation`, a self-consistency sweep recorded at
`QuadSourceIntegral.py:60-70`, **with the code's own caveat that it "cannot rule out an
order-independent bias shared by every order tested"**. Not re-measured here; the fixture's runs all
report `chebyshev_min_order = 24`, i.e. no region needed less.
*Owner:* `prompts/levin-refactor` / `prompts/qsi-phase-groups`.

**(d) `DEFAULT_LEVIN_MAX_DEPTH = 20`** (row E) — keys nothing; production takes the default.
*Provenance:* **cannot be established from the record.** The comment at
`AdaptiveLevin/levin_quadrature.py:153` gives only the geometric reading ("1/2^20 is roughly 1E-6")
and no measurement. Observed here: the deepest bisection any fixture case reached was **8**, so the
cap is not binding on this instrument — which is an observation about the fixture and not a
provenance.
*Owner:* `prompts/levin-refactor`. 06a records it as unestablished **in those words**.

**(e) `limit = 100`** (row F, `Quadrature/simple_quadrature.py:92`) — keys nothing.
*Provenance:* **cannot be established from the record.** The literal carries no comment and no
campaign document mentions it. Not exercised to its cap here.
*Owner:* `prompts/levin-refactor` / `prompts/qsi-phase-groups`. 06a records it as unestablished.

**(f) `BESSEL_ORDER_CHECK_TOL = 1e-3`** (row F) — keys nothing; a guard against a wrong Bessel
order, not an accuracy request.
*Provenance:* argued, not swept, in the comment at `QuadSourceIntegral.py:79-92` — nine orders above
the reconstruction floor and two below the smallest defect it must catch. That *is* a provenance and
06a should not record it as unestablished.
*Owner:* `prompts/levin-refactor` / `prompts/qsi-phase-groups`.

**And one thing 06a must not assume.** `docs/TOLERANCE-PROVENANCE.md` **can** be assembled from the
logs for every row this prompt touched — none of the five §1.2 fields is missing for any of them,
provided (d) and (e) are written as *unestablished* rather than reconstructed from the geometric
readings in their comments. The rows most at risk elsewhere are the ones §5.4 already marks "never
chosen": `DEFAULT_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE`, `DEFAULT_HEXIT_TOLERANCE`,
`DEFAULT_LEVIN_THRESHOLD` and `find_phase_extremum`'s pair. For each of those the record contains a
*use* and no *choice*, and §1.2's closing rule is the only honest treatment.
