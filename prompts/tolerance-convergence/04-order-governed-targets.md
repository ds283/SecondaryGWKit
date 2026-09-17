# Prompt 04 — audit the order-governed targets

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board item **T7** · **Informs:** **D3**, what replaces the vestigial `atol`/`rtol`
key columns — it recommends, it does not decide
**Closes, if it can:** `[01-convergence-block-has-a-separate-generator]`, declined by two prompts of
another campaign on a scope argument this prompt is the first to be given the files for
**Depends on:** prompts 01, 02 and 02a. Prompt 01's `convergence_reference.py` supplies `GaussOrder`
and `radiation_anchors` and is the only way this prompt is allowed to measure; prompt 02's inventory
is where the vestigial columns were confirmed; prompt 02a is why each model can be measured at its
own anchor. **It does not depend on 03 or 03a** — no tolerance this prompt touches is a tolerance
those prompts swept.
**Recommended model:** **Opus** — the measurement is routine and the hazard is not: this prompt
rewrites a fixture that three test modules read, and only one of the three is inside its carve-out.

**Files you may create or touch:**
`docs/tolerance-convergence/ORDER-AUDIT.md` — new, the measurement document;
`docs/tolerance-convergence/order_audit.py` — new, the script that regenerates every table in it;
`docs/gktk-remedial/residual_convergence.py` — **the D5 carve-out**, §3.2;
`ComputeTargets/tests/wkb_reference_data.json` — **the D5 carve-out**, §2 and §3.4;
`ComputeTargets/tests/test_background_tau.py` — **the D5 carve-out**, §3.5;
plus this campaign's log, board, the `qcd-background-audit` board entry for
`[01-convergence-block-has-a-separate-generator]`, and `docs/OPEN_ISSUES.md`.

**`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` is `residual_convergence.py`'s other output and you
may not overwrite it.** It is `GkTk-remedial` prompt 02's published document and README §5 rule 6
makes verification documents additive. Write your re-run's tables to `ORDER-AUDIT.md` and have the
script emit the old document only under an explicit flag you do not pass, or to a path under
`docs/tolerance-convergence/`. If that cannot be arranged without editing the generator's output
path, editing the output path is inside the carve-out and the log says you did it.

**Do not touch:** `ComputeTargets/BackgroundModel.py` or `ComputeTargets/phase_residual.py` — the
four orders and `RESIDUAL_WKB_REGION_MARGIN` live there and **this prompt recommends, it does not
retune** (README §5 rule 8); `config/defaults.py`; `main.py`; any `Datastore/` file; any test module
other than `test_background_tau.py`; `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` or
`AdaptiveLevin/` (README §0.4). **The source grid is fixed** (README §0.5): you measure on it, you
do not move it, and a moved grid digest is a stop however good the reason.

**Read first:** README §2 (a), (b), (f) and (i); §3.1's anchor table **in full**, since four of its
rows are this prompt's oracles; §3.4; **§6.1's target rule**, and §7 in it — read §6.1 knowing that
its language is written for a tolerance and this prompt's knob is an integer (§7 below says what
that changes and what it does not); §5 rules 5, 6 and 8; §1.2's five provenance fields; **§7 D3 and
D5**; `RECONCILIATION.md` §5, which is the whole case for §3; board item **T7**; the issues
`[01-convergence-block-has-a-separate-generator]`, `[20-wkb-gauss-orders-not-in-lookup-key]` and
`[01-density-criterion-imposed-outside-the-wkb-region]`; the module docstring of
`ComputeTargets/tests/convergence_reference.py`, and in it `GaussOrder` (`:212`) and
`radiation_anchors` (`:534`); the docstring of `residual_convergence.py`; the docstring of
`LambdaCDM_GenericEOS.integration_break_points` (`:883`), **including the four paragraphs on why the
knots are no longer returned**; `ComputeTargets/phase_residual.residual_node_range` (`:241`) and the
comment block above `RESIDUAL_WKB_REGION_MARGIN` (`:238`).

---

## 1. Why this prompt exists

**Four of the eight keyed object types have no tolerance to converge, and nobody has ever audited
the knob they do have.** `BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration` carry
`atol`/`rtol` columns that are part of the lookup key and reach no solver (README §2 (a), confirmed
by prompt 02's inventory). What sets their accuracy is four integers — $N_\tau$, $N_{c_s\tau}$,
$N_F$, $N_\rho$, all **4** — and one float, `RESIDUAL_WKB_REGION_MARGIN = 0.5`, and not one of the
five is in any key, label or tag (`[20-wkb-gauss-orders-not-in-lookup-key]`). The campaign's subject
is "what does this parameter buy, measured", and these are the parameters it has not yet asked it
of.

**The evidence for all four orders is two representations out of date, and taking it back is this
prompt's first job.** `ComputeTargets/tests/wkb_reference_data.json`'s `convergence` block is where
the four orders are recorded and it is what every `*_GAUSS_ORDER` comment in the tree cites. It was
generated **2026-09-10**, and `RECONCILIATION.md` §5 lists what has changed under it since:

- the `T(z)` representation `qcd-background-audit` prompts 04–06 replaced — node solve `rtol` 1e-4 →
  1e-14, $T$-against-$u$ → a segmented entropy factor, max error **7.236e-04 → 6.807e-11**;
- an `integration_break_points` that returned the interpolant's ~404 knots, which prompt 07
  removed — **so the block's `decision.recommended_scheme`, `"branch+knots"`, names a split
  production can no longer perform**;
- the background derivative splines prompt 13 segmented.

Its visible residue is `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in
`ComputeTargets/tests/test_background_tau.py`, loosened from **1.4e-05** for no reason but the
block's age — the comment above it says so in those words and names the re-run as the fix.

**So the audit cannot be done against the record; the record has to be re-taken first.** That is
why this prompt is not read-only, and why it needed a user decision (README §7 **D5**, settled
**yes** on 2026-09-16) before it could be written at all.

**What the prompt is not.** It is not a retune. The four orders live in production modules and
README §5 rule 8 keeps them there until prompt 05. If the audit says 4 is right, the deliverable is
4 *with evidence*, which is the thing the tree has never had. If it says 4 is wrong, the deliverable
is a recommendation and a stop — see §2.4, because that case collides with a test.

---

## 2. The fixture, and the three readers that make this the dangerous half

**Read this section before you plan anything.** The measurement in §4 is routine. Writing its result
into `wkb_reference_data.json` is not, and the collision is structural rather than a matter of care.

### 2.1 What D5 actually granted

Exactly three files: `docs/gktk-remedial/residual_convergence.py`,
`ComputeTargets/tests/wkb_reference_data.json`, and
`ComputeTargets/tests/test_background_tau.py`. README §7 D5 says "exactly those three files and no
others; anything further is a stop under §4.3". That boundary is not a formality here, because —

### 2.2 — three test modules read the block, and two are outside the carve-out

| Reader | What it reads | In the carve-out? |
|---|---|---|
| `ComputeTargets/tests/test_background_tau.py:324` | `convergence.models.QCDModel.**branch+knots**.tau.json_vs_reference_max_rel`, as a **threshold**: asserts the production checkpoints are within `QCD_FLOOR_FACTOR = 3.0` of it | **yes** |
| `ComputeTargets/tests/test_background_tau.py:362` | `convergence.geometry.QCDModel` — branch boundaries and the $c_s^2$ transition, by interval index | **yes** |
| `ComputeTargets/tests/test_background_cs_tau_friction.py:606` | the **same threshold construction** on `cs_tau`: `assertLessEqual(worst_cs, 3.0 * floor)` | **NO** |
| `ComputeTargets/tests/test_background_cs_tau_friction.py:631` | `convergence.geometry.QCDModel`, same as above | **NO** |
| `ComputeTargets/tests/test_phase_residual.py:432` | `decision.N_rho`, asserted **equal to the production `RHO_GAUSS_ORDER`**, and `decision.rho_adaptive_fallback_required`, asserted false | **NO** |

Three consequences follow, and each is a way this prompt fails silently or stops badly.

**(i) A tighter floor is a tighter test, on a module you may not edit.** Both threshold sites divide
by nothing and multiply by 3: the assertion is *production error ≤ 3 × the block's recorded
reference floor*. The corrected background is four orders more accurate than the one the block was
taken on, so the regenerated floor will very likely be **smaller**, and `test_qcd_checkpoints` in
`test_background_cs_tau_friction.py` will then fail unless the production checkpoints improved by
the same factor. You may fix the `tau` one; you may not touch the `cs_tau` one. **If it fails, that
is a §11 stop** — report it with both numbers and what the old and new floors are, and do not
"solve" it by writing the old floor into the new block.

**(ii) The `branch+knots` key must survive as a key.** Two of the five sites index it by name. The
scheme is dead as a *production* scheme — `integration_break_points` has not returned knots since
`qcd-background-audit` prompt 07 — but the script can still build it, because it takes the knots off
the model's own `T(z)` spline (`residual_convergence.py:265-270`) and not off the cosmology's
contract. **Keep the scheme in the sweep and the key in the block** (§3.2 says in what role), so
that the block stays schema-compatible with its readers. Dropping the key is a two-module edit
outside the carve-out and therefore a stop.

**(iii) If a recommended order moves off 4, a test outside the carve-out breaks.**
`test_phase_residual.py:432` pins `RHO_GAUSS_ORDER == decision.N_rho`. Since rule 8 forbids you the
production constant, a block recording `N_rho = 6` fails that assertion immediately. §2.4 says what
to do.

### 2.3 The block is regenerated, not hand-edited, and only the block

Whatever you write into the JSON is written by `residual_convergence.py` running, not by you editing
the file. `git diff` on `ComputeTargets/tests/wkb_reference_data.json` must touch the
**`convergence`** key and nothing else: `schema_version`, `generated` at the top level, `models`,
`baselines`, `k_values`, `k_keys` and `rho_anchor_efolds_subh` are other prompts' fixtures and other
campaigns' evidence. Show that diff in the log.

### 2.4 The order that moves — write nothing, and ask

**Determine the four recommended orders before you write the JSON**, and if any of them is not 4,
**do not write it**. Keep the measurement in `ORDER-AUDIT.md`, leave the fixture as it stands, and
stop and ask the user. The reason is that the repair — changing the production constant, or
relaxing the test that pins it — is prompt 05's and outside this prompt's files either way, so
writing the block first leaves the tree red with no in-scope way to make it green. A prompt that
leaves the suite failing has broken README §5 rule 1's rollback boundary, whatever its measurement
was worth.

This is the one place where "regenerate, then look" is the wrong order of operations. Look, then
regenerate.

---

## 3. Taking the stale evidence back

### 3.1 What is re-run

`docs/gktk-remedial/residual_convergence.py`, on the corrected background, against the **3-point**
break set `integration_break_points` returns today on `QCD_Cosmology` (**2** under
`BREAK_POINT_DISCONTINUITY`; the docstring at `:883` gives both counts and says which is which).

### 3.2 The scheme sweep, which has to change

The script measures three build schemes side by side — `plain`, `branch`, `branch+knots` — and its
`decision.recommended_scheme` is `"branch+knots"`. **A scheme production cannot execute may not be
recommended.** Production splits at what `integration_break_points` returns, which is the break
temperatures and nothing else, so the recommendation must be `plain` or `branch`.

Keep all three in the sweep, and change what the third one is *for*:

- `plain` and `branch` are the candidates, and one of them is recommended;
- **`branch+knots` is retained as a control**, and the question it now answers is *what does
  splitting at the knots still buy?* Prompt 07 measured that there is nothing left at a knot for a
  panel edge to protect against — d1 to d4 agree across a knot to 1.3e-13 … 6.2e-08, the fifth jumps
  by 26 %, and the deepest derivative anything in the tree builds is the third. **Your measurement
  either confirms that at the quadrature level or contradicts it**, and the second would be a real
  finding about `qcd-background-audit` prompt 07 rather than about this campaign. Say which.

The block's `schemes` list keeps all three entries and `decision.recommended_scheme` names one of
the first two. If the recommendation changes from `branch+knots` to something else — which is the
expected outcome — the log says what the change costs in accuracy and the document tables both.

### 3.3 The alignment tolerance

`QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` was loosened from **1.4e-05** and its own comment says the
whole of the excess is the block's age rather than the representation: prompt 07 re-measured
1.418851e-04 at `T_120_MEV`, 1.728034e-05 at `T_LO` and 1.060594e-06 at `EOS_T_LO`, and attributed
it to the block's break points being root-found where prompt 07's are bisected, ~1e-14 apart in $u$.

**Put it back if it will go.** Re-measure the alignment against the regenerated block and set the
constant to the tightest value the measurement supports, with the three per-break figures in the
document and the arithmetic that chose the constant from them. **If it will not go back to 1.4e-05,
that is a finding about the representation and the log says so in those words** — not "left for a
later prompt", and not a constant chosen to be comfortably above whatever was measured. Say what it
is, what bounds it, and which of the three breaks binds.

### 3.4 What the regenerated block must carry that the old one did not

Every figure in it is a figure in this campaign, so README §5 rules 5 and 6 apply: **the reference's
own drift beside the number, and the grid generation beside it**. The old block records neither. Add
them to the payload — the generator writes the block, so this is a change to the generator, which is
inside the carve-out — and make `generated`, `campaign` and `generator` say that this campaign's
prompt 04 took it.

---

## 4. The order sweep

### 4.1 What is measured, and where the oracle is

Orders **2, 4, 6, 8, 12, 16** as the old sweep used, through `convergence_reference.GaussOrder`, for
which **one step is one order** and not a factor. Five integrands, and README §3.1 gives four of
them a closed form on `RadiationModel`:

| Knob | Integrand | Anchor on `RadiationModel` | Measure |
|---|---|---|---|
| $N_\tau$ | $1/H$ | `tau_delta`, an **interval** anchor | difference error, relative to the interval |
| $N_{c_s\tau}$ | $c_s/H$ | `cs_tau_delta`, interval | difference error |
| $N_F$ | $\frac{3}{2}(1 + c_s^2)$ | `friction_F_delta`, interval | difference error |
| $N_\rho$ | $C/(\omega + k/H)$ and $C_T/(\omega_T + k c_s/H)$ | **`rho_G` is identically zero** ($C = 0$ in exact radiation); `rho_T` has a closed form | phase error, radians |

**`rho_G` ≡ 0 is the best measurement in this prompt** and you should treat it as such: with no
reference to build, whatever the rule returns *is* the quadrature error, unpolluted by a reference's
own drift. Use it. `radiation_anchors` (`:534`) exposes all of these and refuses a model that is not
the exact control.

On `LambdaCDMModel` and `QCDModel` there is no oracle, so it is a converged-reference measurement
through `reference_drift`, and README §5 rule 5 binds: **no number travels without its reference's
drift beside it**, and a cell that does not stand `CRITERION_RATIO` clear of that drift is
**unresolved** and carries no conclusion.

### 4.2 Coverage

- $\tau$, $c_s\tau$ and $F$ are **$k$-independent**: one sweep per model over the production
  intervals, and say so rather than implying a $k$ sweep that would be the same figure fifty times.
- $\rho_G$ and $\rho_T$ are **$k$-dependent**, and the charter is explicit: **every production $k$**,
  all fifty, on all three models. `GkTk-remedial` prompt 02 measured three wavenumbers —
  $10^5$, $10^7$, $3\times10^8$ — and the question this prompt exists to answer is whether it was
  right at those three and lucky at the other forty-seven.
- All three models, each on the **version-2 grid at its own production anchor**
  (`wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM`, `PRODUCTION_Z_INIT_QCD`, and prompt 03's
  construction for the radiation control). Every table says which (README §2 (b), board note 18).

### 4.3 Cost

Integrand evaluations, at the recommended order and **one order either side** (§1.2), times the
object count of the sector the knob governs — and the three knobs do not govern the same count.
`BackgroundModel` is **one object per model**; the residual tables $N_\rho$ governs are per
`(model, k, sector)`. State each count and where you got it. **Never wall time** (README §2 (i)).

---

## 5. `RESIDUAL_WKB_REGION_MARGIN`

It is `0.5`, it is in no key, and the comment above it (`phase_residual.py:225-238`) argues it is
safe rather than measuring what it buys: the cut it makes lies above the highest node at which the
WKB validity criterion holds, production anchors sit three e-folds inside the horizon where the
ratio exceeds 0.99, and so a margin anywhere below ~1 would do. **Measure it.**

1. **What does the margin actually cut?** Node counts retained and discarded per `(model, k,
   sector)` at `0.5` and at a spread either side — the docstring says the transfer-function cut
   removes 250–620 of 1,732 nodes, on a grid generation that is no longer production's, so re-take
   it on the version-2 grid.
2. **Where does the answer stop depending on it?** The claim worth testing is that the phase residual
   is insensitive to the margin over some range. Show the range, or show that there is no such range.
3. **What happens as it approaches 1?** The comment says a margin below ~1 is safe and does not say
   what fails above. Find the value at which `residual_node_range` starts refusing, or where fewer
   than two nodes remain, on each model at the extreme $k$s.

**Then recommend, under §6.1 as §7 below adapts it, or invoke rule 6.** This is a plausible rule-6
row: if what bounds the margin is "the anchors the producer accepts", that is not an accuracy with a
floor, and saying so in rule 6's words is a complete answer. What is not a complete answer is
leaving `0.5` in place with the existing comment as its justification, which is the state this
prompt found.

**`[01-density-criterion-imposed-outside-the-wkb-region]` is the other half of this and is T7's.**
`main.source_grid_spacing_profile` imposes the fourth-derivative spline criterion over
`residual_node_range`'s band, which reaches 1.5–2.1 e-folds *outside* the horizon — about 5 e-folds
beyond the phase spline it protects — and 69 % of the samples version 2 adds on QCD lie above
horizon crossing for the smallest production $k$. That issue has been carrying the 69 % figure as
its only evidence since it was opened, and prompt 02a's guarded-node census was explicitly withdrawn
as a measurement of it. **Your margin sweep is the first thing in the campaign that can measure it:
how many of the band's nodes are outside the region the criterion is protecting, at each $k$, and
what would the grid look like if the band were the WKB region alone.** Recommend; do not change the
band, `main.py`, or the grid (README §0.5, and D6's carve-out was 02a's and is spent).

---

## 6. D3 — what replaces the vestigial key columns

`BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration` key on `atol`/`rtol` columns that
describe nothing, while the orders that set their accuracy are in no key
(`[20-wkb-gauss-orders-not-in-lookup-key]`). **The user's stated target for the campaign is that for
a Liouville–Green-type representation the key should carry an order, not a tolerance** (README
§7 D3). You recommend among three options, each with its schema-churn cost:

- **keep** — no migration; three permanently misleading column pairs; and an order change silently
  serves a stale row, which is the live defect;
- **drop** — a migration, and the defect half-closed: nothing misleading is left, but nothing
  distinguishes an order-4 row from an order-6 one either;
- **replace with the orders** — a migration, and the defect closed.

Say for each: how many tables, how many existing rows, whether the migration is expressible as a
column addition plus a backfill or needs a rebuild, and what breaks in the six `extract_*.py`
readers. `GkSource` is a fourth vestigial case and is **not** this question — it integrates nothing,
so there is no order to put there; note it and move on.

**Three targets, four orders, and they do not line up one-to-one.** $N_\tau$, $N_{c_s\tau}$ and
$N_F$ all belong to `BackgroundModel` while $N_\rho$ belongs to the two WKB targets, so "put the
order in the key" is a different shape for each. Say what the key actually becomes.

---

## 7. The target rule, for an integer knob

README §6.1 is written for a tolerance. Three of its six rules transfer unchanged and three need
reading:

- **Rule 1 — measure the floor first.** Unchanged. For these targets the floor is
  double-precision accumulation over the grid, and the radiation anchors measure it directly rather
  than by inheritance.
- **Rules 2 and 3 — the loosest setting that clears, not the tightest that works.** For an integer
  order, **loosest means lowest**. Sweep upwards from the lowest order and take the **first** that
  clears the floor at the **maximum** over the production grid — not the median, and not at a
  representative $k$.
- **Rule 4 — `unchanged` is a result.** If order 4 already clears, the answer is `unchanged` in that
  word, **with the factor by which the floor dominates beside it**. Given what the old block records
  this is the likely outcome for at least some of the four, and it is a finding: the tree has never
  had it measured on the current representation.
- **Rule 5 — an accuracy below the floor you just measured is an arithmetic error.** Unchanged, and
  campaign-wide.
- **Rule 6 — no floor, no target, in those words.** Most likely to apply to
  `RESIDUAL_WKB_REGION_MARGIN`.

One thing has no analogue in the tolerance rows and you must handle it: **a Gauss rule can get
worse as the order rises.** The old block shows it — `plain` `tau_errors_rel` runs 3.44e-07 at order
2, 4.46e-08 at 12, and back up to **1.03e-07** at 16. So "the first order that clears" is not
automatically a statement that every higher order also clears, and if your sweep shows a
non-monotone column, **say so and give the recommendation with that attached** rather than reporting
the first crossing as though the curve were monotone. Prompt 03a hit the same shape in a different
sector and its `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` is the precedent for how to record
it.

---

## 8. The questions this prompt must answer

Answer each in the document, in this order, each with a table behind it.

1. **What did the re-run change?** The regenerated block against the 2026-09-10 one, figure by
   figure for the four orders, with the two causes — corrected `T(z)`, and the knots leaving
   `integration_break_points` — separated if they can be separated and declared inseparable if they
   cannot.
2. **Are the four orders converged at 4 at every production $k$?** Or was `GkTk-remedial` prompt 02
   right at the three $k$ it measured and lucky at the rest? A per-$k$ answer for $N_\rho$, and a
   per-model answer for the other three.
3. **Which scheme does production actually need?** `plain` or `branch`, with what the split buys at
   the recommended order, and what `branch+knots` still buys now that nothing in production performs
   it.
4. **Does `QCD_BREAK_POINT_ALIGNMENT_TOL` go back to 1.4e-05?** Yes with the constant, or no with the
   reason and the binding break.
5. **What does `RESIDUAL_WKB_REGION_MARGIN` buy?** §5's three measurements, and a recommendation or
   rule 6's words.
6. **How much of `residual_node_range`'s band lies outside the region the density criterion is
   protecting?** §5's last paragraph — the measurement
   `[01-density-criterion-imposed-outside-the-wkb-region]` has been waiting for.
7. **What should replace the vestigial key columns?** §6's three options, costed, with a
   recommendation.
8. **What are the five provenance fields** (README §1.2) for each of $N_\tau$, $N_{c_s\tau}$, $N_F$,
   $N_\rho$ and `RESIDUAL_WKB_REGION_MARGIN`? Prompt 06 assembles the provenance note from these and
   may not re-derive them.

---

## 9. Out of scope, and where each piece went

- **Changing any of the five parameters.** Prompt 05's, on the user's D3 decision. You recommend.
- **The schema migration itself.** Prompt 05's (**T9**).
- **The source grid, the band `residual_node_range` returns, `main.py`.** README §0.5. D6's carve-out
  was prompt 02a's and is spent.
- **Tolerances.** Every tolerance in the pipeline is prompts 03, 03a and 06's; D1 closed on
  2026-09-17 and nothing here reopens it.
- **`QuadSourceIntegral` and everything in README §0.4.** Prompt 06's, read-only.
- **The other two campaigns' fixtures in `wkb_reference_data.json`.** §2.3: the `convergence` key and
  nothing else.

---

## 10. Acceptance

1. `docs/tolerance-convergence/ORDER-AUDIT.md` exists, `order_audit.py` regenerates **every** table
   in it, and the document names the version and anchor each figure was taken on.
2. Every §8 question is answered with a table, and each figure carries its reference's drift or is
   marked unresolved.
3. $N_\rho$ is measured at **all fifty** production wavenumbers on all three models.
4. The regenerated `convergence` block is written **by the generator**, carries drift and grid
   generation, and `git diff` on the JSON touches **no other top-level key** — or the block is
   deliberately not written, under §2.4, and the log says which.
5. `git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'` lists **at most**
   `ComputeTargets/tests/wkb_reference_data.json` and `ComputeTargets/tests/test_background_tau.py`.
   No other file under `ComputeTargets/`, and no production module at all.
6. Both suites green: `ComputeTargets` **at or above 491**, `CosmologyModels` **39**,
   `test_convergence_reference` **32**. A fixture rewrite that moves a count is a finding to report,
   not a number to accept.
7. `black --check` clean on every `.py` in the diff.
8. Log, board row 04, item **T7**, the `qcd-background-audit` entry for
   `[01-convergence-block-has-a-separate-generator]` and `docs/OPEN_ISSUES.md`, all in the **same
   commit**, with the index's count and date corrected.

---

## 11. Stop conditions

Stop, commit nothing further, and say so:

- **`test_background_cs_tau_friction.test_qcd_checkpoints` fails on the regenerated floor** (§2.2
  (i)). Report both floors and both errors. Do not edit that module and do not write the old floor
  into the new block.
- **Any recommended order is not 4** (§2.4). Write the measurement, not the fixture.
- **`test_phase_residual` or any other test outside the carve-out fails**, for any reason connected
  to the block.
- **A reference does not converge** at any (model, $k$, order), or an anchor disagrees with a
  converged reference by more than its drift. Prompt 17's error; not a wavenumber to drop.
- **An accuracy below a floor you have just measured** (§6.1 rule 5).
- **A moved source-grid digest**, at any point and for any reason.
- **The knot lattice cannot be reconstructed**, so `branch+knots` cannot be kept as a key (§2.2
  (ii)). Ask; do not drop the key.
- **You find yourself wanting to edit a production module, `config/defaults.py`, `main.py`, or a
  test module other than `test_background_tau.py`.**
- **`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` would be overwritten.** README §5 rule 6.

---

## 12. The log

`prompts/tolerance-convergence/logs/04-order-governed-targets.md`, to README §5.1's template.
Classify every deviation. The log must carry, beyond the template:

- the **diff of the `convergence` block** in summary — which fields moved, by how much, and which are
  new — or the statement that it was not written and why;
- the **five provenance fields** for each of the five parameters, in "State handed to the next
  prompt", because prompt 06 assembles from it and prompt 05 ships from it;
- what the **D3 recommendation** is and what it costs, stated so the user can decide from the log
  alone;
- every observation not acted on, as a §3 issue.
