# Tolerance and convergence — the campaign close-out

**Campaign:** [`prompts/tolerance-convergence`](../prompts/tolerance-convergence/README.md) ·
**Board:** [`IMPLEMENTATION_STATE.md`](../prompts/tolerance-convergence/IMPLEMENTATION_STATE.md) ·
**Prompt:** [06a](../prompts/tolerance-convergence/06a-close-out-and-provenance.md) · **Board item:**
**T12** · **Companion:** [`TOLERANCE-PROVENANCE.md`](TOLERANCE-PROVENANCE.md), the per-parameter
record · **Date:** 2026-09-18

This is the campaign's narrative record: the eight object types keyed on an accuracy parameter
(README §2 (a)), the parameter each ended at, the evidence, and the floor each is now limited by. It
is shorter than the sum of its inputs — five campaign documents, eleven logs — because each of those
already holds its own tables; this document says what each prompt found and points at where the
tables are, rather than re-tabulating them.

---

## 1. Why the campaign existed

`GkTk-remedial` prompt 12 set one constant, `atol = 1e-13`, on one wavenumber of one model. Prompt
17 then measured the production grid and found the target missed at 3, 13 and 8 of 50 wavenumbers
on the three models, and the lever turned out to be `rtol` — a constant prompt 17 was not allowed to
touch. Nobody had checked the rest of the pipeline's accuracy knobs, and two campaigns had since run
that moved the background, the break-point set and the source grid underneath every figure that
existed. This campaign's job was to establish what every accuracy parameter in the pipeline really
is, measure each on its own terms, and decouple them so each quantity carries a parameter whose
provenance can be stated.

**The plan's own count of its subject was wrong, twice, and the campaign corrected itself both
times.** The 2026-09-12 plan counted five compute targets sharing two constants across four object
types. Prompt 02 measured the tree directly and found **eight** object types keyed on an accuracy
parameter — later **nine**, `OneLoopIntegral` being a ninth that computes nothing (§6 below) — and
that of the six sharing the pair, **two** used the value they were given, not the one the campaign's
own README §0.1 stated. Two targets the original plan never mentioned turned out to matter most:
`wavenumber_exit_time`, a live `root_scalar` fixing where every grid begins, and `BackgroundModel`,
whose three Gauss orders are the largest instance of the case where the knob is an integer order
rather than a tolerance at all.

---

## 2. The eight targets, and where each ended

### `wavenumber_exit_time` — **changed**

`rtol` moves from the shared `1e-8` to **`1e-9`**; `xtol` stays at `1e-10`, unchanged and
**coupled** — it floors the pair at ~1e-10 relative below `rtol ≈ 2.6e-12`, and its own provenance
cannot be established. The recommendation reads README §6.1's rule against Brent's stopping
*guarantee* ($x_{\rm tol}+r_{\rm tol}|u|$ at the largest production $|u|=38.04$), which clears the
only floor in the tree for this target — `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, the tolerance
at which a redshift row is reused — at **3.81e-08**. On the achieved-displacement reading of the same
measurement the answer would be `unchanged` (production's achieved 7.86e-08 already clears 1e-7 by
1.3×); the user accepted the guarantee reading, and that caveat ships with the value. Cost: +0.7 %
of 6,963 Hubble evaluations across 50 objects per model. Evidence:
[`TK-NUMERIC-AND-EXIT-TIME.md`](tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md) §7; board item
**T6**; prompt 03a.

### `GkNumericIntegration` — **`unchanged`**

`(atol, rtol) = (1e-10, 1e-8)` stays. This is the campaign's central measurement: ~29,000–58,000
objects per model, never swept before in either axis. `rtol` moves the solver's own error by
×13,300 over four decades and `atol` is inert to 1.2 %, confirming README §2 (d)'s prior in this
sector — but the consumer's own cubic spline of the numeric $G$
(`GkSourcePolicyData.py:654-680`) carries **1.6e-04 to 9.4e-03** of the envelope near the hand-over
against **2.6e-07** for the solver: **×631 to ×37,700**. Tightening the solver buys nothing while
that spline dominates by two to four and a half orders, so the target is `unchanged` under README
§6.1 rule 4 and the campaign's compute question in its largest sector settles at **zero**. Evidence:
[`GK-NUMERIC-SWEEP.md`](tolerance-convergence/GK-NUMERIC-SWEEP.md) §§5, 7, 9, 10; board item **T4**;
prompt 03.

### `TkNumericIntegration` — **changed**

`rtol` moves from the shared `1e-8` to **`3e-11`**; `atol` stays at `1e-13`, unchanged but
**re-characterised** — it is not the accuracy level in this sector, it is a step-selection knob that
determines *which* wavenumber draws a bad step sequence (moving the maximum by up to 205× across two
decades while moving the median by at most 2.1×). The re-measured initial-condition floor is
**2.39e-06 to 2.64e-06** of the envelope (against an inherited 2.52e-06, taken on a superseded grid
and break-point policy); `3e-11` is the loosest of nine swept settings whose maximum over all fifty
wavenumbers on all three models — **3.88e-08** — clears it, at **+39.4 %** of the sector's
right-hand-side evaluations across 50 objects per model. **The caveat that ships with the number**:
the maximum is not monotone in `rtol` (`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`, open), so
`3e-11` is the loosest setting that clears *in this sweep*, not a proven bound over the sector.
Evidence: [`TK-NUMERIC-AND-EXIT-TIME.md`](tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md) §4,
§4.1; board item **T5**; prompt 03a.

### `BackgroundModel` — **`unchanged`**

$N_\tau = N_{c_s\tau} = N_F = 4$, all three re-measured for the first time on the corrected
background and the 3-point break set (the evidence they previously rested on was generated
2026-09-10, against a $T(z)$ representation and a break-point set both since replaced). The
competing floor is double-precision accumulation over the grid — **2.16e-16 / 3.30e-16 / 8.01e-14**
relative for $\tau$/$c_s\tau$/$F$ — which order 4 dominates order 2's error by **×3.03e5 / ×1.99e5 /
×48.4**. `unchanged` under README §6.1 rule 4; the cost of the whole sector is one table per
(cosmology, grid). The three orders moved from a vestigial `(atol, rtol)` key column that reached no
solver to their own equality-keyed columns (`[20-wkb-gauss-orders-not-in-lookup-key]`, closed).
Evidence: [`ORDER-AUDIT.md`](tolerance-convergence/ORDER-AUDIT.md) §§3.1, 5; board item **T7**;
prompt 04.

### `GkWKBIntegration` and `TkWKBIntegration` — **`unchanged`**

$N_\rho = 4$ for both sectors, measured for the first time at **all fifty** production wavenumbers
on all three models in both sectors (300 cases), superseding `GkTk-remedial`'s own three-wavenumber
evidence, which this measurement finds was right and lucky by only ×1.5–×1.7. The competing floor
is the $\rho$ quadrature accumulation floor, **6.51e-17 rad**, dominated by order 4 by **×2.35e6** —
the largest dominating factor of the four orders. `unchanged` under README §6.1 rule 4.
`RESIDUAL_WKB_REGION_MARGIN = 0.5`, the fifth axis this sector carries, is **also `unchanged`**, but
under rule 6 rather than rule 4: it has no accuracy floor at all, because the residual a producer
reads is a fixed-redshift `delta` on the grid's own panels, so the margin can only make the anchor
unreachable, not less accurate. It stays out of the lookup key on that measured bit-identity (0.05
to 0.9, all three models, both sectors). $N_\rho$ moved into the lookup key on both WKB targets
(`[20-wkb-gauss-orders-not-in-lookup-key]`, closed), and the *recorded* order was then made the
order an object was actually built at, on both the compute and rehydration paths
(`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`, closed; §5 below).
Evidence: [`ORDER-AUDIT.md`](tolerance-convergence/ORDER-AUDIT.md) §§4, 5, 7.2; board item **T7**;
prompts 04, 05, 05b.

### `GkSource` — **drop**

`GkSource` assembles the numeric and WKB limbs; it integrates nothing, so it never used the
`(atol, rtol)` pair its schema carried and there is no order to put in its place. Prompt 05 dropped
the pair and added nothing. Board item **T9**; prompt 05.

### `QuadSourceIntegral` — **`unchanged`, read-only**

`(atol, rtol) = (1e-32, 1e-8)` stays, measured but not owned by this campaign (README §0.4 puts
`QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` and `AdaptiveLevin/` out of scope). The
representation floor, freshly measured on the *realistic* fixture flavour at **2.523e-06 to
4.912e-04** of scale, dominates the quadrature error at production by **×3.46e+04 to ×2.15e+11** —
the largest dominating factor of the campaign's three `unchanged` findings, measured with the
*exact*/*realistic* pair the fixture provides so that the quadrature error and the representation
floor are separated rather than convolved. **The regime has inverted since the value was chosen**:
`source-remediation` log 12 found `rtol` non-binding at `atol = 1e-25`; at the shipped `atol =
1e-32`, `atol` is inert over twenty-eight decades and `rtol` is now the only lever, moving `total`
by ×274 over three decades. Neither finding changes the value — the floor swamps both ends — but a
reader quoting the old bit-identity is quoting a superseded regime (standing note 26). **A second
supersession applies to the fixture's own oracle column**: `analytic_rad` is computed inside
`evaluate_QuadSource_integral` at *the caller's own* `atol`/`rtol` rather than at a fixed reference
pair (standing note 27), so a stored `analytic_rad` moves with its row's tolerance and is displaced
up to **×1.25e+06** further from its converged value than `total` is at production — any prior
reading of `|total − analytic_rad|` as a fixed-oracle residual is superseded by this. Two findings
handed to the campaigns that own the files rather than acted on:
`[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]` (`levin-refactor`, standing note 26) and
`[06-analytic-rad-is-computed-at-the-callers-tolerance]` (`qsi-phase-groups`, standing note 27).
Evidence:
[`QUADSOURCE-READONLY.md`](tolerance-convergence/QUADSOURCE-READONLY.md) §§4–9; board item **T11**;
prompt 06.

---

## 3. The three `unchanged` results are the campaign's main finding

Three of the eight targets came back `unchanged`, and that is not an absence of a result — it is the
result. Under README §6.1 rule 4, `unchanged` is what the rule gives when the measured error is
already below the competing floor, and each of the three is dominated by a different kind of floor:

| Target | Dominating factor | The floor |
|---|---|---|
| `GkNumericIntegration` | ×631–×37,700 | the consumer's own cubic spline of the numeric $G$ |
| `BackgroundModel` (three orders) | ×48.4–×3.03e5 | double-precision accumulation over the grid |
| Both WKB sectors' $N_\rho$ | ×2.35e6 | the $\rho$ quadrature accumulation floor |
| `QuadSourceIntegral` | ×3.46e+04–×2.15e+11 | the source integral's own representation floor |

Read together they say the same thing from four different sectors: **the campaign's central prior
— "the error is set by `rtol`" — held only in the one sector it was cleanly measured in
(`TkNumericIntegration`, 50 objects per model), and in every sector with a real object count the
answer was to spend nothing.** The two targets that did move — `TkNumericIntegration` and
`wavenumber_exit_time` — are both 50-object-per-model sectors, where a decade of `rtol` is free; the
~29,000–58,000-object sector that the original plan expected to be the main compute decision turned
out to be the sector where nothing should be spent at all. That reversal — measured, not assumed —
is this campaign's answer to the question it was opened to ask (README §7 D1).

---

## 4. What corrected the plan's own account of its subject

Two corrections to the campaign's own README, both left standing in the document rather than
rewritten (README §5 rule 7 — verification documents are additive):

- **§0.1's summary sentence — "of those six [live shared-pair targets] only one actually uses the
  value" — undercounts by one.** Prompt 02 found that **two** of the six use the value they are
  given: `GkNumericIntegration` and `wavenumber_exit_time`. `TOLERANCE-INVENTORY.md` §2 (a)'s table
  is right row by row; only the one-sentence summary was wrong.
- **There are nine keyed object types, not eight.** `Datastore/SQL/ObjectFactories/OneLoopIntegral.py`
  declares the same `atol_serial`/`rtol_serial` foreign keys as the other eight and filters on both
  in `build()`, but `main.py` never builds one — its `compute()` is a label-replacing stub, and the
  object count of the sector is zero. This is `[02-oneloopintegral-is-a-ninth-keyed-object-type]`,
  opened by prompt 02 and left **unassigned** — it is not obviously any of prompts 03–05's, and
  deciding whether to decouple a target that computes nothing before any row exists to invalidate,
  or to leave it as vestigial as `GkSource`'s pair was, is a decision this campaign did not make.

---

## 5. What this campaign built, beyond the eight targets

**One convergence facility** (`ComputeTargets/tests/convergence_reference.py`, prompt 01), covering
every target and calibrated against eleven constant-$w$ closed forms on `RadiationModel`, used by
every later measurement in this document. **One reproduction of the production source grid**
(`ComputeTargets/tests/wkb_reference.py`'s `source_grid(generation, ...)`, prompt 01), replacing four
disagreeing constructions with three named, bit-identical generations.

**The source grid was made buildable at every production anchor** (prompt 02a, board item **T13**):
the version-2 grid raised on `QCD_Cosmology` at QCD's own production anchor, so no QCD production
figure in this campaign's own record could have been taken at the cosmology's own anchor without
this fix. The guard marks a node where the Liouville–Green expansion does not exist as unusable
rather than raising, counts it, and refuses above a measured fraction of the band
(`SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`, 64.8× the worst measured band). Its acceptance was
bit-identity with the two previously published grid digests, and both stayed bit-identical: QCD at
its own anchor is **2034 samples / `21ffc126`**, LambdaCDM **1778 / `60a3205a`**, and the radiation
control **2306 / `3bef2c06`**. A follow-up measurement found the guarded-node census was not, as
first read, evidence of the source grid's density criterion reaching outside the horizon —
all 53 guarded evaluations turned out to be one node, inside the band, close to a declared crossing
(`[02a-stencil-reaches-across-a-declared-break-before-the-mask]`) — and the genuine over-reach
question (`[01-density-criterion-imposed-outside-the-wkb-region]`) was measured separately by prompt
04b §8, which is where the 13-e-fold figure quoted in §2 above comes from.

**The vestigial key columns were replaced with the orders** (prompts 05, 05b; README §7 D3,
`[20-wkb-gauss-orders-not-in-lookup-key]`, closed): `BackgroundModel` traded `atol_serial`/`rtol_serial`
for `tau_gauss_order`, `cs_tau_gauss_order` and `friction_F_gauss_order`; both WKB sectors traded
theirs for `rho_gauss_order`; `GkSource` lost the pair and gained nothing. The acceptance was not
that the columns exist but that the order is *filtered on*, not merely stored and read back —
`TOLERANCE-INVENTORY.md` §2.4 had measured that the old pair was joined into a label and never
compared. A second defect, opened by prompt 05's own agent, was that the *recorded* order and the
order an object's tables were actually built at could diverge — a residual table's `order=` keyword
defaulted at definition time, and a rehydrated `BackgroundModel` reassembled its cumulative tables at
the *current* module constant while its row's own order columns never reached the constructor.
Prompt 05b made the record faithful on both paths, computed and rehydrated, with `build()` still
filtering on the current module constant so that a row computed at another order is a different row
rather than a miss to repair.

**The convergence evidence for the four orders was taken back and re-written** (prompt 04b, board
item **T14**): `wkb_reference_data.json`'s `convergence` block, generated 2026-09-10 against a
$T(z)$ representation and a break-point set both since replaced, was regenerated against the tree
this campaign actually measured, and the two threshold tests that read it were rebuilt to bound the
production quantity against a measured accumulation floor rather than against the fixture's own
agreement with a converged reference — a comparison that had held only because both sides happened
to sit near $2\times10^{-14}$. `QCD_BREAK_POINT_ALIGNMENT_TOL` moved from `1.5e-04` to `3.0e-14`,
against offsets that are one to four ulp of $u$.

---

## 6. What is still open, and where it lives

This campaign closes with the answer to every question its own README posed, and it leaves the
following for other work — each is a `docs/OPEN_ISSUES.md` §1.5 entry with its own board record, not
a loose end this document invents a fix for:

- **`OneLoopIntegral`, the ninth keyed object type** (§4 above) — unassigned; the user decides
  whether it is decoupled, dropped or left as it is.
- **`TkNumericIntegration`'s `rtol = 3e-11` is a measured setting, not a proven bound**
  (`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`) — the sporadic excursion this campaign found is
  a property of the phenomenon, not of coverage, and a genuine bound would need the excursion's rate
  against `rtol` measured directly.
- **`wavenumber_exit_time`'s recommendation rests on Brent's guarantee, not the achieved
  displacement** — the user's acceptance is on the guarantee reading; the achieved reading gives
  `unchanged`, and both readings are recorded together in
  [`TOLERANCE-PROVENANCE.md`](TOLERANCE-PROVENANCE.md).
- **The QCD version-2 grid's sample count is not reproducible at the anchor solve's own precision**
  (`[03a-qcd-v2-grid-sample-count-is-not-reproducible]`) — a relative perturbation of `1e-14`, six
  orders below the achieved anchor displacement, moves the sample count between 2013 and 2032.
- **The numeric $G_k$ consumer spline is the dominant error near the hand-over**
  (`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`), assigned to the
  hand-over campaign on the user's D1 acceptance — this is the floor that makes
  `GkNumericIntegration` `unchanged`, and fixing it is a source-grid design question outside this
  campaign's boundary (README §0.5).
- **Two verification scripts still query `wavenumber_exit_time` under the pre-decoupling shared
  constants** (`[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`) —
  harmless today because the target's lookup is an inequality, but quietly serving a different
  stored pair than the script believes it asked for.
- **README §6.2's `QuadSourceIntegral` row, `config/defaults.py:163`'s comment and
  `QuadSourceIntegral.py:1550`'s comment all still state the superseded `atol = 1e-25` regime**
  (standing note 26) — this document and `TOLERANCE-PROVENANCE.md` record the current finding, but
  this prompt could not edit any of the three files under its own grant; a new §3 issue records it
  for whoever next touches them.
- **`DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE` are now pure float-comparison epsilons with
  no chosen value behind them** (`[02-shared-atol-doubles-as-a-float-comparison-epsilon]`) — no
  object type keys on either any more, so the hazard the issue was opened against did not fire, but
  the seven comparison sites still carry a value nobody chose for that purpose.
- **Two questions this campaign was expressly forbidden to answer**: whether
  `SOURCE_GRID_CONSUMER_TARGET_RAD`'s density criterion should apply over a horizon-limited band
  rather than `residual_node_range`'s full reachability band
  (`[01-density-criterion-imposed-outside-the-wkb-region]`, README §0.5), and whether
  `QuadSourceIntegral`'s decoupled pair should itself be retuned rather than merely measured
  (README §7 D4, not reopened by prompt 06's `unchanged` finding).

---

## 7. Numbers for the record

**Suites, at close:** `ComputeTargets` **521**, `CosmologyModels` **39**, both green throughout the
campaign's thirteen prompts (01, 02, 02a, 03, 03a, 04, 04b, 05, 05b, 05a, 06, plus this one).
**Published source-grid digests, unmoved since prompt 02a**: `3bef2c06` (Radiation, 2306 samples),
`60a3205a` (LambdaCDM, 1778 samples), `21ffc126` (QCD at its own anchor, 2034 samples).
**`config/defaults.py`**: byte-identical across every prompt except 05a, which shipped exactly the
six constants named in §2 above and left the two shared names in place, now used only as float
comparisons. **No prompt before 05a moved a parameter** (README §5 rule 8), and 05, 05b, 04b and 06a
changed no number at all.
